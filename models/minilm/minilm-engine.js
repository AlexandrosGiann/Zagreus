/*
 * paraphrase-multilingual-MiniLM-L12-v2 (q8) in pure JavaScript: model coordinator.
 *
 * Runs the same computation as Xenova/paraphrase-multilingual-MiniLM-L12-v2
 * onnx/model_quantized.onnx under onnxruntime-web (WASM), which is what Transformers.js uses
 * with { device: 'wasm', dtype: 'q8' }, after ONNX Runtime's graph optimizations:
 *   - embeddings: Gather + DequantizeLinear (uint8), (word + token type) + position
 *   - LayerNormalization (float accumulation, ORT layer_norm_impl.cc)
 *   - DynamicQuantizeLinear of the activations (per tensor, the whole batch)
 *   - quantized linear layers, attention and BiasGelu in the shards (minilm-shard.js)
 * Every float operation is rounded to float32 in the same order as ONNX Runtime, so the last
 * hidden state is bit-for-bit identical. Mean pooling and L2 normalization follow
 * Transformers.js (feature-extraction pipeline, pooling 'mean', normalize true).
 *
 * This file holds the parts that need the whole hidden state (embeddings, residual stream,
 * layer norms, activation quantization). The shards run in this thread or in Web Workers.
 *
 * ES5 + typed arrays only (no WebAssembly, no ES2015 syntax) for old Android browsers.
 */
(function (root) {
  'use strict';

  var ShardLib = (typeof module !== 'undefined' && module.exports) ? require('./minilm-shard.js') : root.MiniLMShard;

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

  /*
   * DynamicQuantizeLinear (uint8) over a whole tensor, as ORT computes it (also inside
   * DynamicQuantizeMatMul). Writes q - zero_point into qa (Int32Array), the per-row sums of
   * those values into rowSums, and returns the scale.
   */
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

  // LayerNormalization of one row (ORT layer_norm_impl.cc, float).
  function layerNormRow(x, xo, g, b, out, n, eps) {
    var i, v, mean = 0, ms = 0;
    for (i = 0; i < n; i++) { v = x[xo + i]; mean = fround(mean + v); ms = fround(ms + fround(v * v)); }
    mean = fround(mean / n);
    var sd = fround(Math.sqrt(fround(fround(fround(ms / n) - fround(mean * mean)) + eps)));
    for (i = 0; i < n; i++) out[xo + i] = fround(fround(fround(fround(x[xo + i] - mean) / sd) * g[i]) + b[i]);
  }

  /*
   * Transformers.js mean_pooling: sum in double over the masked tokens, divided by their
   * count, stored as float32. hidden: [B * T * H], mask: B arrays of 0/1. Returns [B * H].
   */
  function meanPool(hidden, mask, B, T, H) {
    var out = new Float32Array(B * H), b, k, j, sum, count, attn;
    for (b = 0; b < B; b++) {
      for (k = 0; k < H; k++) {
        sum = 0; count = 0;
        for (j = 0; j < T; j++) {
          attn = mask[b][j];
          count += attn;
          sum += hidden[(b * T + j) * H + k] * attn;
        }
        out[b * H + k] = sum / count;
      }
    }
    return out;
  }

  // Transformers.js Tensor.normalize(2, -1): float32 running sum of squares, ** 0.5, division.
  function l2Normalize(x, B, H) {
    var out = new Float32Array(x.length), b, k, s;
    for (b = 0; b < B; b++) {
      s = 0;
      for (k = 0; k < H; k++) s = fround(s + x[b * H + k] * x[b * H + k]);
      s = fround(Math.sqrt(s));
      for (k = 0; k < H; k++) out[b * H + k] = x[b * H + k] / s;
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------
  // Backends: where the shards live.

  // One shard in this thread; calls back synchronously.
  function LocalBackend(shard) {
    this.shard = shard;
    this.threads = 1;
  }
  LocalBackend.prototype.attn = function (l, qa, rowSums, rows, scale, B, T, maskAdd, cb) {
    cb(null, this.shard.attn(l, qa, rowSums, rows, scale, B, T, maskAdd));
  };
  LocalBackend.prototype.linear = function (l, name, qa, rowSums, rows, scale, cb) {
    cb(null, this.shard.linear(l, name, qa, rowSums, rows, scale));
  };
  LocalBackend.prototype.terminate = function () {};

  /*
   * Shards in workers. `ports` are objects { post(msg, transfer), onmessage: fn } wrapping a
   * MessagePort to a Web Worker or a Node worker thread; `ranges[i]` are the ranges of worker i.
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

  // Gather column slices [rows x width] from every worker into [rows x N].
  PoolBackend.prototype._gather = function (replies, colRange, rows, N) {
    var out = new Float32Array(rows * N);
    for (var w = 0; w < replies.length; w++) {
      var r = colRange(this.ranges[w]), width = r[1] - r[0], src = replies[w].out;
      for (var row = 0; row < rows; row++) out.set(src.subarray(row * width, (row + 1) * width), row * N + r[0]);
    }
    return out;
  };

  // (typed arrays are copied to exactly their length: posting a view would clone its whole buffer)
  PoolBackend.prototype.attn = function (l, qa, rowSums, rows, scale, B, T, maskAdd, cb) {
    var self = this, H = this.cfg.hidden, HD = this.cfg.headDim;
    var msg = { op: 'attn', l: l, qa: new Int32Array(qa.subarray(0, rows * H)), rowSums: new Float64Array(rowSums.subarray(0, rows)), rows: rows, scale: scale, B: B, T: T, maskAdd: maskAdd };
    this._all(msg, function (err, rs) {
      if (err) return cb(err);
      cb(null, self._gather(rs, function (r) { return [r.heads[0] * HD, r.heads[1] * HD]; }, rows, H));
    });
  };

  PoolBackend.prototype.linear = function (l, name, qa, rowSums, rows, scale, cb) {
    var self = this, K = name === 'out' ? this.cfg.inter : this.cfg.hidden, N = name === 'inter' ? this.cfg.inter : this.cfg.hidden;
    var msg = { op: 'linear', l: l, name: name, qa: new Int32Array(qa.subarray(0, rows * K)), rowSums: new Float64Array(rowSums.subarray(0, rows)), rows: rows, scale: scale };
    this._all(msg, function (err, rs) {
      if (err) return cb(err);
      cb(null, self._gather(rs, function (r) { return r[name]; }, rows, N));
    });
  };

  PoolBackend.prototype.terminate = function () {
    for (var i = 0; i < this.ports.length; i++) if (this.ports[i].terminate) this.ports[i].terminate();
  };

  // ---------------------------------------------------------------------------------------
  // Model coordinator.

  /*
   * small: { wordFiles: [Uint8Array], pos, type (Uint8Array), lnW, lnB, ln: [{ ln1W, ln1B,
   * ln2W, ln2B }] } (Float32Array), the parts of the weights this thread keeps.
   */
  function Model(manifest, small, backend) {
    var cfg = manifest.config, e = manifest.embed;
    this.cfg = cfg;
    this.H = cfg.hidden; this.I = cfg.inter; this.L = cfg.layers;
    this.eps = fround(cfg.eps);
    this.maxPos = cfg.maxPos;
    this.backend = backend;
    this.threads = backend.threads;
    function table(scale, zp) {
      var t = new Float32Array(256), s = fround(scale);
      for (var i = 0; i < 256; i++) t[i] = fround((i - zp) * s);
      return t;
    }
    this.wordFiles = small.wordFiles;
    this.wordRows = e.word.rowsPerFile;
    this.wordTable = table(e.word.scale, e.word.zp);
    this.pos = small.pos; this.posTable = table(e.position.scale, e.position.zp);
    this.type = small.type; this.typeTable = table(e.tokenType.scale, e.tokenType.zp);
    this.lnW = small.lnW; this.lnB = small.lnB;
    this.ln = small.ln;
  }

  /*
   * ids: B arrays of token ids, all of length T (padded), mask: B arrays of 0/1.
   * Calls back with (err, hidden): the last hidden state, Float32Array [B * T * hidden].
   * hooks(name, array, length) is optional (tests).
   */
  Model.prototype.forward = function (ids, mask, cb, hooks) {
    var self = this, B = ids.length, T = ids[0].length, H = this.H, I = this.I;
    var rows = B * T, be = this.backend, b, t, i, r;
    if (T > this.maxPos) { cb(new Error('Sequence too long (' + T + ' > ' + this.maxPos + ' tokens)')); return; }
    var x = new Float32Array(rows * H), tmp;
    // embeddings: (word + token type 0) + position, then LayerNorm
    for (b = 0; b < B; b++) {
      for (t = 0; t < T; t++) {
        var id = ids[b][t], f = (id / this.wordRows) | 0, wo = (id - f * this.wordRows) * H;
        var wd = this.wordFiles[f], o = (b * T + t) * H, po = t * H;
        if (!wd || id < 0 || wo + H > wd.length) { cb(new Error('Token id out of range: ' + id)); return; }
        for (i = 0; i < H; i++) {
          x[o + i] = fround(fround(this.wordTable[wd[wo + i]] + this.typeTable[this.type[i]]) + this.posTable[this.pos[po + i]]);
        }
      }
    }
    for (r = 0; r < rows; r++) layerNormRow(x, r * H, this.lnW, this.lnB, x, H, this.eps);
    if (hooks) hooks('embed', x, rows * H);

    // attention mask term: (1 - mask) * float lowest
    var maskAdd = new Float32Array(B * T);
    for (b = 0; b < B; b++) for (t = 0; t < T; t++) maskAdd[b * T + t] = fround(fround(1 - mask[b][t]) * FLT_LOWEST);

    var qa = new Int32Array(rows * I), rowSums = new Float64Array(rows);

    function fail(e) { cb(e); }

    function residualNorm(add, g, bias) {
      for (var j = 0; j < rows * H; j++) add[j] = fround(add[j] + x[j]);
      for (var rr = 0; rr < rows; rr++) layerNormRow(add, rr * H, g, bias, x, H, self.eps);
    }

    function layer(l) {
      if (l === self.L) { cb(null, x); return; }
      var ln = self.ln[l];
      var sc = quantizeDynamic(x, rows * H, qa, H, rowSums);
      be.attn(l, qa, rowSums, rows, sc, B, T, maskAdd, function (err, ctx) {
        if (err) return fail(err);
        if (hooks) hooks('l' + l + '.ctx', ctx, rows * H);
        sc = quantizeDynamic(ctx, rows * H, qa, H, rowSums);
        be.linear(l, 'o', qa, rowSums, rows, sc, function (err, o) {
          if (err) return fail(err);
          residualNorm(o, ln.ln1W, ln.ln1B);
          if (hooks) hooks('l' + l + '.ln1', x, rows * H);
          sc = quantizeDynamic(x, rows * H, qa, H, rowSums);
          be.linear(l, 'inter', qa, rowSums, rows, sc, function (err, inter) {
            if (err) return fail(err);
            if (hooks) hooks('l' + l + '.gelu', inter, rows * I);
            sc = quantizeDynamic(inter, rows * I, qa, I, rowSums);
            be.linear(l, 'out', qa, rowSums, rows, sc, function (err, out) {
              if (err) return fail(err);
              residualNorm(out, ln.ln2W, ln.ln2B);
              if (hooks) hooks('l' + l + '.out', x, rows * H);
              layer(l + 1);
            });
          });
        });
      });
    }
    layer(0);
  };

  /*
   * Embeddings of a list of token id arrays (already with <s> ... </s>), run as one padded
   * batch like the Transformers.js pipeline does with an array of texts.
   * opts: { pad: pad id, pooling: 'mean' | 'none', normalize: bool }
   * Calls back with (err, { data: Float32Array, dims: [...], hidden: last hidden state }).
   */
  Model.prototype.embed = function (idLists, opts, cb) {
    var B = idLists.length, T = 0, i, j, H = this.H;
    for (i = 0; i < B; i++) if (idLists[i].length > T) T = idLists[i].length;
    var ids = [], mask = [];
    for (i = 0; i < B; i++) {
      var r = [], m = [];
      for (j = 0; j < T; j++) {
        r.push(j < idLists[i].length ? idLists[i][j] : opts.pad);
        m.push(j < idLists[i].length ? 1 : 0);
      }
      ids.push(r); mask.push(m);
    }
    this.forward(ids, mask, function (err, hidden) {
      if (err) return cb(err);
      if (opts.pooling !== 'mean') {
        if (opts.normalize) return cb(new Error('normalize needs pooling "mean"'));
        cb(null, { data: hidden, dims: [B, T, H], hidden: hidden });
        return;
      }
      var pooled = meanPool(hidden, mask, B, T, H);
      if (opts.normalize) pooled = l2Normalize(pooled, B, H);
      cb(null, { data: pooled, dims: [B, H], hidden: hidden });
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

  // Work split for `parts` shards: attention heads (q/k/v columns) and output columns.
  function shardRanges(cfg, parts) {
    if (parts > cfg.heads) throw new Error('At most ' + cfg.heads + ' threads are supported');
    var heads = splitRange(cfg.heads, parts, 1), o = splitRange(cfg.hidden, parts, 4);
    var inter = splitRange(cfg.inter, parts, 4), out = splitRange(cfg.hidden, parts, 4), res = [];
    for (var i = 0; i < parts; i++) {
      var hd = cfg.headDim;
      res.push({ heads: heads[i], q: [heads[i][0] * hd, heads[i][1] * hd], o: o[i], inter: inter[i], out: out[i] });
    }
    for (i = 0; i < parts; i++) { res[i].k = res[i].q; res[i].v = res[i].q; }
    return res;
  }

  var LINEARS = ['q', 'k', 'v', 'o', 'inter', 'out'];

  /*
   * Load the model.
   * opts.manifest: parsed manifest.json
   * opts.fetch(name, onProgress(loadedBytes), cb(err, ArrayBuffer)): reads one model file
   * opts.threads: number of compute workers (1 = everything in this thread)
   * opts.kernel: 'int', 'f64' or 'auto' (default: each shard times both, see minilm-shard.js)
   * opts.createPort(): returns a port { post(msg, transfer), onmessage, terminate() } to a
   *   new compute worker (needed when threads > 1)
   * onProgress(doneBytes, totalBytes, fileName)
   */
  function load(opts, onProgress, cb) {
    var manifest = opts.manifest, cfg = manifest.config, files = manifest.files, e = manifest.embed;
    var threads = Math.max(1, opts.threads | 0), total = 0, done = 0, i, ranges;
    for (i = 0; i < files.length; i++) total += files[i].bytes;
    try { ranges = shardRanges(cfg, threads); } catch (err) { cb(err); return; }
    var local = null, ports = [], failed = false, l;
    var small = { wordFiles: [], ln: [] };
    for (l = 0; l < cfg.layers; l++) small.ln.push({});

    function fail(err) {
      if (failed) return;
      failed = true;
      for (var p = 0; p < ports.length; p++) if (ports[p].terminate) ports[p].terminate();
      cb(err);
    }

    if (threads === 1) {
      local = new ShardLib.Shard({ config: cfg, ranges: ranges[0], kernel: opts.kernel });
    } else {
      // Each worker answers 'init' and, after all weights, 'ready'.
      var acks = 0, kernels = [];
      var onAck = function (msg) {
        if (msg && msg.op === 'error') { fail(new Error(msg.message)); return; }
        if (msg && msg.op === 'init') kernels.push(msg.kernel);
        acks++;
        if (acks === 2 * threads && !failed) {
          var mp = new Model(manifest, small, new PoolBackend(ports, ranges, cfg));
          mp.kernels = kernels;
          cb(null, mp);
        }
      };
      for (i = 0; i < threads; i++) {
        var port = opts.createPort();
        port.onmessage = onAck;
        ports.push(port);
        port.post({ op: 'init', spec: { config: cfg, ranges: ranges[i], kernel: opts.kernel } }, []);
      }
    }

    // copy of a float32 / uint8 tensor (so the file buffer can be freed)
    function f32(buf, t) { return new Float32Array(buf.slice(t.offset, t.offset + t.length * 4)); }
    function u8(buf, t) { return new Uint8Array(buf.slice(t.offset, t.offset + t.length)); }

    // Hand the bytes of one file to the shards that need them; keep the rest here.
    function distribute(fileIndex, buf) {
      var ei = e.word.files.indexOf(fileIndex), w, j, m, r, lin, kk;
      if (ei >= 0) { small.wordFiles[ei] = new Uint8Array(buf); return; }
      if (e.position.data.file === fileIndex) small.pos = u8(buf, e.position.data);
      if (e.tokenType.data.file === fileIndex) small.type = u8(buf, e.tokenType.data);
      if (e.lnW.file === fileIndex) small.lnW = f32(buf, e.lnW);
      if (e.lnB.file === fileIndex) small.lnB = f32(buf, e.lnB);
      for (l = 0; l < manifest.layers.length; l++) {
        m = manifest.layers[l];
        var names = ['ln1W', 'ln1B', 'ln2W', 'ln2B'];
        for (j = 0; j < names.length; j++) if (m[names[j]].file === fileIndex) small.ln[l][names[j]] = f32(buf, m[names[j]]);
        for (j = 0; j < LINEARS.length; j++) {
          lin = m[LINEARS[j]];
          if (lin.w.file !== fileIndex) continue;
          for (w = 0; w < threads; w++) {
            r = ranges[w][LINEARS[j]]; kk = lin.k;
            var wb = buf.slice(lin.w.offset + r[0] * kk, lin.w.offset + r[1] * kk);
            var sb = buf.slice(lin.scale.offset + r[0] * 4, lin.scale.offset + r[1] * 4);
            var zb = buf.slice(lin.zp.offset + r[0], lin.zp.offset + r[1]);
            var bb = buf.slice(lin.bias.offset + r[0] * 4, lin.bias.offset + r[1] * 4);
            if (local) {
              local.setLinear(l, LINEARS[j], { k: kk, n: r[1] - r[0], w: new Int8Array(wb), scale: new Float32Array(sb), zp: new Int8Array(zb), bias: new Float32Array(bb) });
            } else {
              ports[w].post({ op: 'linearWeights', l: l, name: LINEARS[j], k: kk, n: r[1] - r[0], w: wb, scale: sb, zp: zb, bias: bb }, [wb, sb, zb, bb]);
            }
          }
        }
      }
    }

    var next = 0;
    function loadNext() {
      if (failed) return;
      if (next === files.length) {
        if (local) {
          var ml = new Model(manifest, small, new LocalBackend(local));
          ml.kernels = [local.kernel];
          cb(null, ml);
          return;
        }
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
        try { distribute(idx, buf); } catch (err2) { return fail(err2); }
        next++;
        loadNext();
      });
    }

    loadNext();
  }

  var api = {
    Model: Model, load: load, meanPool: meanPool, l2Normalize: l2Normalize,
    LocalBackend: LocalBackend, PoolBackend: PoolBackend, shardRanges: shardRanges, fround: fround
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MiniLMEngine = api;
})(typeof self !== 'undefined' ? self : this);
