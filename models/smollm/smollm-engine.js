/*
 * SmolLM2-135M-Instruct (q8) in pure JavaScript: model coordinator.
 *
 * Runs the same computation as onnx-community/SmolLM2-135M-Instruct-ONNX-MHA
 * onnx/model_quantized.onnx under onnxruntime-web (WASM, 1 thread), which is what
 * Transformers.js uses with { device: 'wasm', dtype: 'q8' }.
 *
 * Every float operation is rounded to float32 in the same order as the ONNX Runtime
 * CPU kernels (MLAS) compiled to WebAssembly SIMD, so the results match bit for bit:
 *   - DynamicQuantizeLinear + MatMulInteger (exact integer arithmetic)
 *   - (Skip)SimplifiedLayerNormalization, RotaryEmbedding
 *   - MultiHeadAttention: SGEMM accumulation order, MLAS softmax with its exp polynomial
 *   - Sigmoid through the MLAS rational approximation
 *   - lm_head through the dequantized, tied embedding table
 *
 * This file holds the parts that need the whole hidden state (residual stream, norms and
 * the per-tensor activation quantization). Everything else runs in shards
 * (smollm-shard.js), either in this thread or spread over Web Workers.
 *
 * ES5 + typed arrays only (no WebAssembly, no ES2015 syntax) for old Android browsers.
 */
(function (root) {
  'use strict';

  var ShardLib = (typeof module !== 'undefined' && module.exports) ? require('./smollm-shard.js') : root.SmolLMShard;

  var fround = Math.fround || (function () {
    var f = new Float32Array(1);
    return function (x) { f[0] = x; return f[0]; };
  })();

  var FLT_LOWEST = -3.4028234663852886e38;

  // std::nearbyint with the default round-half-to-even mode.
  function roundHalfEven(v) {
    var f = Math.floor(v);
    var d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }

  // RMS normalization of one row (ORT SimplifiedLayerNormalization, float).
  function rmsRow(x, xo, w, out, oo, n, eps) {
    var i, ms = 0, v;
    for (i = 0; i < n; i++) { v = x[xo + i]; ms = fround(ms + fround(v * v)); }
    ms = fround(Math.sqrt(fround(fround(ms / n) + eps)));
    for (i = 0; i < n; i++) out[oo + i] = fround(fround(x[xo + i] / ms) * w[i]);
  }

  // DynamicQuantizeLinear (uint8) over a whole tensor. Writes q - zero_point into qa
  // (Int32Array), the per-row sums of those values into rowSums, and returns the scale.
  function quantizeDynamic(x, len, qa, rowLen, rowSums) {
    var i, v, min = 3.4028234663852886e38, max = FLT_LOWEST;
    for (i = 0; i < len; i++) {
      v = x[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min > 0) min = 0;
    if (max < 0) max = 0;
    var scale = (max === min) ? 1 : fround(fround(max - min) / 255);
    var izp = fround(0 - fround(min / scale));
    if (izp < 0) izp = 0; else if (izp > 255) izp = 255;
    var zp = roundHalfEven(izp);
    var r, k, s, q, base;
    var rows = len / rowLen;
    for (r = 0; r < rows; r++) {
      s = 0; base = r * rowLen;
      for (k = 0; k < rowLen; k++) {
        q = roundHalfEven(fround(x[base + k] / scale)) + zp;
        if (q < 0) q = 0; else if (q > 255) q = 255;
        q -= zp;
        qa[base + k] = q;
        s += q;
      }
      rowSums[r] = s;
    }
    return scale;
  }

  // ---------------------------------------------------------------------------------------
  // Backends: where the shards live.

  // All weights in this thread, one shard; calls back synchronously.
  function LocalBackend(shard) {
    this.shard = shard;
    this.threads = 1;
  }
  LocalBackend.prototype.attn = function (l, qa, rowSums, rows, scale, pos0, hooks, cb) { cb(null, this.shard.attn(l, qa, rowSums, rows, scale, pos0, hooks)); };
  LocalBackend.prototype.o = function (l, qa, rowSums, rows, scale, cb) { cb(null, this.shard.o(l, qa, rowSums, rows, scale)); };
  LocalBackend.prototype.mlp = function (l, qa, rowSums, rows, scale, cb) { cb(null, this.shard.mlp(l, qa, rowSums, rows, scale)); };
  LocalBackend.prototype.down = function (l, qa, rowSums, rows, scale, cb) { cb(null, this.shard.down(l, qa, rowSums, rows, scale)); };
  LocalBackend.prototype.embed = function (ids, cb) { cb(null, this.shard.embed(ids).data); };
  LocalBackend.prototype.cand = function (h, blocked, seen, penalty, cb) { cb(null, this.shard.candidates(h, blocked, seen, penalty)); };
  LocalBackend.prototype.terminate = function () {};

  /*
   * Shards in workers. `ports` are objects { post(msg, transfer), onmessage: fn } wrapping a
   * Web Worker / MessagePort or a Node worker thread; `ranges[i]` are the ranges of worker i.
   */
  function PoolBackend(ports, ranges, cfg) {
    this.ports = ports;
    this.ranges = ranges;
    this.cfg = cfg;
    this.threads = ports.length;
    this.queues = [];
    this.failed = null;
    var self = this;
    for (var i = 0; i < ports.length; i++) {
      this.queues.push([]);
      (function (idx) {
        ports[idx].onmessage = function (msg) { self._reply(idx, msg); };
      })(i);
    }
  }

  PoolBackend.prototype._reply = function (idx, msg) {
    var cb = this.queues[idx].shift();
    if (msg && msg.op === 'error') {
      this.failed = new Error(msg.message || 'Worker error');
      if (cb) cb(this.failed);
      return;
    }
    if (cb) cb(null, msg);
  };

  // Send one message to every worker and call back with all replies (in worker order).
  PoolBackend.prototype._all = function (msg, cb) {
    if (this.failed) { cb(this.failed); return; }
    var n = this.ports.length, left = n, replies = new Array(n), done = false;
    for (var i = 0; i < n; i++) {
      (function (self, idx) {
        self.queues[idx].push(function (err, reply) {
          if (done) return;
          if (err) { done = true; cb(err); return; }
          replies[idx] = reply;
          if (--left === 0) { done = true; cb(null, replies); }
        });
        self.ports[idx].post(msg, []);
      })(this, i);
    }
  };

  // Gather column slices [rows x (n1 - n0)] from every worker into [rows x N].
  PoolBackend.prototype._gather = function (replies, key, rows, N) {
    var out = new Float32Array(rows * N);
    for (var w = 0; w < replies.length; w++) {
      var r = this.ranges[w][key], width = r[1] - r[0], src = replies[w].out;
      for (var row = 0; row < rows; row++) {
        out.set(src.subarray(row * width, (row + 1) * width), row * N + r[0]);
      }
    }
    return out;
  };

  // A matmul-style request: send the quantized activation, gather the output columns.
  PoolBackend.prototype._request = function (op, key, l, qa, rowSums, rows, scale, K, N, extra, cb) {
    var self = this;
    var msg = { op: op, l: l, qa: new Int32Array(qa.subarray(0, rows * K)), rowSums: new Float64Array(rowSums.subarray(0, rows)), rows: rows, scale: scale };
    for (var e in extra) msg[e] = extra[e];
    this._all(msg, function (err, rs) {
      if (err) return cb(err);
      cb(null, self._gather(rs, key, rows, N));
    });
  };

  PoolBackend.prototype.attn = function (l, qa, rowSums, rows, scale, pos0, hooks, cb) {
    var c = this.cfg;
    this._request('attn', 'q', l, qa, rowSums, rows, scale, c.hidden, c.hidden, { pos0: pos0 }, cb);
  };
  PoolBackend.prototype.o = function (l, qa, rowSums, rows, scale, cb) {
    this._request('o', 'o', l, qa, rowSums, rows, scale, this.cfg.hidden, this.cfg.hidden, {}, cb);
  };
  PoolBackend.prototype.mlp = function (l, qa, rowSums, rows, scale, cb) {
    this._request('mlp', 'gate', l, qa, rowSums, rows, scale, this.cfg.hidden, this.cfg.inter, {}, cb);
  };
  PoolBackend.prototype.down = function (l, qa, rowSums, rows, scale, cb) {
    this._request('down', 'down', l, qa, rowSums, rows, scale, this.cfg.inter, this.cfg.hidden, {}, cb);
  };

  PoolBackend.prototype.embed = function (ids, cb) {
    var H = this.cfg.hidden, list = Array.prototype.slice.call(ids);
    this._all({ op: 'embed', ids: list }, function (err, rs) {
      if (err) return cb(err);
      var out = new Float32Array(list.length * H);
      for (var w = 0; w < rs.length; w++) {
        for (var j = 0; j < rs[w].pos.length; j++) out.set(rs[w].data.subarray(j * H, (j + 1) * H), rs[w].pos[j] * H);
      }
      cb(null, out);
    });
  };

  PoolBackend.prototype.cand = function (h, blocked, seen, penalty, cb) {
    this._all({ op: 'cand', h: new Float32Array(h), blocked: blocked, seen: new Int32Array(seen), penalty: penalty }, function (err, rs) {
      if (err) return cb(err);
      var ids = [], vals = [];
      for (var w = 0; w < rs.length; w++) { ids = ids.concat(rs[w].ids); vals = vals.concat(rs[w].vals); }
      cb(null, { ids: ids, vals: vals });
    });
  };

  PoolBackend.prototype.terminate = function () {
    for (var i = 0; i < this.ports.length; i++) if (this.ports[i].terminate) this.ports[i].terminate();
  };

  // ---------------------------------------------------------------------------------------
  // Model coordinator.

  function Model(manifest, misc, backend) {
    var cfg = manifest.config;
    this.cfg = cfg;
    this.H = cfg.hidden; this.I = cfg.inter; this.L = cfg.layers;
    this.eps = fround(cfg.eps);
    this.maxPos = cfg.maxPos;
    this.backend = backend;
    function f32(t) { return new Float32Array(misc, t.offset, t.length); }
    this.finalNorm = f32(manifest.misc.finalNorm);
    this.norms = [];
    for (var l = 0; l < manifest.layers.length; l++) {
      this.norms.push({ input: f32(manifest.layers[l].inputNorm), post: f32(manifest.layers[l].postNorm) });
    }
    this.cacheLen = 0;
  }

  // Start a new sequence (the shards overwrite their KV caches from position 0).
  Model.prototype.resetCache = function () {
    this.cacheLen = 0;
  };

  Model.prototype._scratch = function (rows) {
    var s = this._buf;
    if (s && s.rows >= rows) return s;
    var H = this.H, I = this.I;
    s = {
      rows: rows,
      resid: new Float32Array(rows * H),
      norm: new Float32Array(rows * H),
      qa: new Int32Array(rows * I),
      rowSums: new Float64Array(rows)
    };
    this._buf = s;
    return s;
  };

  /*
   * Run the model on `tokens` at positions cacheLen.. (appending them to the KV caches).
   * Calls back with (err, hidden, blocked): the final-norm hidden state of the last token
   * and whether the lm_head should use the multi-row (prefill) kernel.
   * `hooks(name, array, length)` is optional and used by the tests (single thread only).
   */
  Model.prototype.forward = function (tokens, cb, hooks) {
    var self = this, rows = tokens.length, H = this.H, I = this.I;
    var pos0 = this.cacheLen, be = this.backend, s, resid, norm, proj = null;
    if (pos0 + rows > this.maxPos) { cb(new Error('Context too long (' + (pos0 + rows) + ' > ' + this.maxPos + ' tokens)')); return; }
    s = this._scratch(rows);
    resid = s.resid; norm = s.norm;

    function fail(e) { cb(e); }

    be.embed(tokens, function (err, emb) {
      if (err) return fail(err);
      resid.set(emb);
      if (hooks) hooks('embed', resid, rows * H);
      layer(0);
    });

    function layer(l) {
      if (l === self.L) return finish();
      var i, r, sc, nrm = self.norms[l];
      // input_layernorm (layer 0: plain RMS norm; later: residual add fused in SkipSimplifiedLayerNorm)
      if (l > 0) for (i = 0; i < rows * H; i++) resid[i] = fround(resid[i] + proj[i]);
      for (r = 0; r < rows; r++) rmsRow(resid, r * H, nrm.input, norm, r * H, H, self.eps);
      if (hooks) hooks('l' + l + '.inputNorm', norm, rows * H);
      sc = quantizeDynamic(norm, rows * H, s.qa, H, s.rowSums);
      be.attn(l, s.qa, s.rowSums, rows, sc, pos0, hooks, function (err, ctx) {
        if (err) return fail(err);
        if (hooks) hooks('l' + l + '.attn', ctx, rows * H);
        sc = quantizeDynamic(ctx, rows * H, s.qa, H, s.rowSums);
        be.o(l, s.qa, s.rowSums, rows, sc, function (err, o) {
          if (err) return fail(err);
          if (hooks) hooks('l' + l + '.oproj', o, rows * H);
          // post_attention_layernorm (SkipSimplifiedLayerNorm)
          for (i = 0; i < rows * H; i++) resid[i] = fround(resid[i] + o[i]);
          for (r = 0; r < rows; r++) rmsRow(resid, r * H, nrm.post, norm, r * H, H, self.eps);
          if (hooks) { hooks('l' + l + '.resid1', resid, rows * H); hooks('l' + l + '.postNorm', norm, rows * H); }
          sc = quantizeDynamic(norm, rows * H, s.qa, H, s.rowSums);
          be.mlp(l, s.qa, s.rowSums, rows, sc, function (err, act) {
            if (err) return fail(err);
            if (hooks) hooks('l' + l + '.mlpMul', act, rows * I);
            sc = quantizeDynamic(act, rows * I, s.qa, I, s.rowSums);
            be.down(l, s.qa, s.rowSums, rows, sc, function (err, d) {
              if (err) return fail(err);
              if (hooks) hooks('l' + l + '.down', d, rows * H);
              proj = d;
              layer(l + 1);
            });
          });
        });
      });
    }

    function finish() {
      // final norm (SkipSimplifiedLayerNorm) on the last row only; only it is needed.
      var last = (rows - 1) * H, i;
      for (i = 0; i < H; i++) resid[last + i] = fround(resid[last + i] + proj[last + i]);
      var fin = new Float32Array(H);
      rmsRow(resid, last, self.finalNorm, fin, 0, H, self.eps);
      if (hooks) hooks('finalNorm', fin, H);
      self.cacheLen = pos0 + rows;
      cb(null, fin, rows > 1);
    }
  };

  /*
   * Greedy token after the repetition penalty (penalized logits, first index of the max),
   * exactly as Transformers.js picks it from the ONNX Runtime logits.
   */
  Model.prototype.nextToken = function (hidden, blocked, seen, penalty, cb) {
    this.backend.cand(hidden, blocked, seen, penalty, function (err, c) {
      if (err) return cb(err);
      var best = -1, bv = -Infinity;
      for (var i = 0; i < c.ids.length; i++) {
        if (c.vals[i] > bv || (c.vals[i] === bv && c.ids[i] < best)) { bv = c.vals[i]; best = c.ids[i]; }
      }
      cb(null, best);
    });
  };

  // ---------------------------------------------------------------------------------------
  // Loading and sharding.

  // Split [0, n) into `parts` ranges with boundaries on multiples of `align`.
  function splitRange(n, parts, align) {
    var out = [], units = n / align, start = 0;
    for (var i = 0; i < parts; i++) {
      var end = Math.round(units * (i + 1) / parts) * align;
      out.push([start, end]);
      start = end;
    }
    return out;
  }

  // Work split for `parts` shards: query heads (with the key/value heads they read), output
  // rows of the other linear layers, and vocabulary rows of the lm_head.
  function shardRanges(cfg, parts) {
    if (parts > cfg.heads) throw new Error('At most ' + cfg.heads + ' threads are supported');
    var HD = cfg.headDim, group = cfg.heads / cfg.kvHeads;
    var heads = splitRange(cfg.heads, parts, 1);
    var o = splitRange(cfg.hidden, parts, 4), mlp = splitRange(cfg.inter, parts, 4);
    var down = splitRange(cfg.hidden, parts, 4), vocab = splitRange(cfg.vocab, parts, 4);
    var out = [];
    for (var i = 0; i < parts; i++) {
      var h0 = heads[i][0], h1 = heads[i][1];
      var kv0 = (h0 / group) | 0, kv1 = (((h1 - 1) / group) | 0) + 1;
      out.push({
        heads: [h0, h1], q: [h0 * HD, h1 * HD], k: [kv0 * HD, kv1 * HD], v: [kv0 * HD, kv1 * HD],
        o: o[i], gate: mlp[i], up: mlp[i], down: down[i], vocab: vocab[i]
      });
    }
    return out;
  }

  function shardSpec(manifest, ranges) {
    var layers = [], names = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
    for (var l = 0; l < manifest.layers.length; l++) {
      var m = {};
      for (var j = 0; j < names.length; j++) {
        var w = manifest.layers[l][names[j]];
        m[names[j]] = { scale: w.scale, zp: w.zp };
      }
      layers.push(m);
    }
    return { config: manifest.config, ranges: ranges, embed: { scale: manifest.embed.scale, zp: manifest.embed.zp }, layers: layers };
  }

  /*
   * Load the model.
   * opts.manifest: parsed manifest.json
   * opts.fetch(name, onProgress(loadedBytes), cb(err, ArrayBuffer)): reads one model file
   * opts.threads: number of compute workers (1 = everything in this thread)
   * opts.createPort(): returns a port { post(msg, transfer), onmessage, terminate() } to a
   *   new compute worker (needed when threads > 1)
   * onProgress(doneBytes, totalBytes, fileName)
   */
  function load(opts, onProgress, cb) {
    var manifest = opts.manifest, cfg = manifest.config, files = manifest.files;
    var threads = Math.max(1, opts.threads | 0), total = 0, done = 0, i, ranges;
    for (i = 0; i < files.length; i++) total += files[i].bytes;
    try { ranges = shardRanges(cfg, threads); } catch (e) { cb(e); return; }
    var local = null, ports = [], misc = null, failed = false;
    var names = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];

    function fail(e) {
      if (failed) return;
      failed = true;
      for (var p = 0; p < ports.length; p++) if (ports[p].terminate) ports[p].terminate();
      cb(e);
    }

    if (threads === 1) {
      local = new ShardLib.Shard(shardSpec(manifest, ranges[0]));
    } else {
      // Each worker answers 'init' and, after all weights, 'ready'.
      var acks = 0;
      var onAck = function (msg) {
        if (msg && msg.op === 'error') { fail(new Error(msg.message)); return; }
        acks++;
        if (acks === 2 * threads && !failed) cb(null, new Model(manifest, misc, new PoolBackend(ports, ranges, cfg)));
      };
      for (i = 0; i < threads; i++) {
        var port = opts.createPort();
        port.onmessage = onAck;
        ports.push(port);
        port.post({ op: 'init', spec: shardSpec(manifest, ranges[i]) }, []);
      }
    }

    // Hand the bytes of one file to the shards that need them.
    function distribute(fileIndex, buf) {
      var e = manifest.embed, w, l, j, r;
      if (fileIndex === manifest.misc.file) {
        misc = buf;
        var c = manifest.misc.cos, s = manifest.misc.sin;
        if (local) local.setRope(new Float32Array(buf, c.offset, c.length), new Float32Array(buf, s.offset, s.length));
        for (w = 0; w < ports.length; w++) {
          var cb1 = buf.slice(c.offset, c.offset + c.length * 4), sb = buf.slice(s.offset, s.offset + s.length * 4);
          ports[w].post({ op: 'rope', cos: cb1, sin: sb }, [cb1, sb]);
        }
        return;
      }
      var ei = e.files.indexOf(fileIndex);
      if (ei >= 0) {
        var row0 = ei * e.rowsPerFile, row1 = row0 + e.rowsPerFile;
        for (w = 0; w < threads; w++) {
          var vr = ranges[w].vocab, a = Math.max(row0, vr[0]), b = Math.min(row1, vr[1]);
          if (a >= b) continue;
          var off = (a - row0) * cfg.hidden, len = (b - a) * cfg.hidden;
          if (local) local.addEmbedChunk(a, b - a, new Uint8Array(buf, off, len));
          else {
            var copy = buf.slice(off, off + len);
            ports[w].post({ op: 'embedChunk', row: a, rows: b - a, buf: copy }, [copy]);
          }
        }
        return;
      }
      for (l = 0; l < manifest.layers.length; l++) {
        for (j = 0; j < names.length; j++) {
          var m = manifest.layers[l][names[j]];
          if (m.file !== fileIndex) continue;
          for (w = 0; w < threads; w++) {
            r = ranges[w][names[j]];
            var start = m.offset + r[0] * m.k, end = m.offset + r[1] * m.k;
            if (local) local.setWeight(l, names[j], new Uint8Array(buf, start, end - start));
            else {
              var part = buf.slice(start, end);
              ports[w].post({ op: 'weight', l: l, name: names[j], buf: part }, [part]);
            }
          }
        }
      }
    }

    var next = 0;
    function loadNext() {
      if (failed) return;
      if (next === files.length) {
        if (local) { cb(null, new Model(manifest, misc, new LocalBackend(local))); return; }
        for (var p = 0; p < ports.length; p++) ports[p].post({ op: 'ready' }, []);
        return;
      }
      var f = files[next], idx = next;
      opts.fetch(f.name, function (loaded) {
        if (onProgress) onProgress(done + loaded, total, f.name);
      }, function (err, buf) {
        if (err) return fail(err);
        if (buf.byteLength !== f.bytes) return fail(new Error(f.name + ': expected ' + f.bytes + ' bytes, got ' + buf.byteLength));
        done += f.bytes;
        if (onProgress) onProgress(done, total, f.name);
        try { distribute(idx, buf); } catch (e) { return fail(e); }
        next++;
        loadNext();
      });
    }

    loadNext();
  }

  var api = {
    Model: Model, load: load, LocalBackend: LocalBackend, PoolBackend: PoolBackend,
    shardRanges: shardRanges, shardSpec: shardSpec, fround: fround
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmolLMEngine = api;
})(typeof self !== 'undefined' ? self : this);
