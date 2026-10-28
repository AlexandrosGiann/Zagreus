/*
 * Coordinator worker: loads the tokenizer and the model, and computes embeddings.
 * Messages from the page:
 *   { type: 'load', base: 'model/', threads: n, kernel: 'auto' }  (+ n MessagePorts to compute
 *     workers when n > 1)
 *   { type: 'embed', id, texts: [...], batch: bool, pooling: 'mean' | 'none', normalize: bool,
 *     hash: bool }  (hash: also return a hash of the last hidden state of every run)
 *     batch true: all texts as one padded batch (like extractor([a, b, ...]) in Transformers.js);
 *     batch false: one text at a time (like extractor(a), extractor(b), ...).
 * Messages to the page:
 *   { type: 'progress', stage, loaded, total }, { type: 'ready', threads, kernels, ms },
 *   { type: 'result', id, data: Float32Array, dims, ids: [[...]], hashes: [...], ms },
 *   { type: 'error', id, message }
 */
importScripts('minilm-nfkc-data.js', 'minilm-tokenizer.js', 'minilm-shard.js', 'minilm-engine.js');

var model = null, tokenizer = null, manifest = null, queue = [], busy = false;

function now() { return (self.performance && performance.now) ? performance.now() : Date.now(); }

function fetchFile(url, onProgress, cb) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', url, true);
  xhr.responseType = 'arraybuffer';
  xhr.onprogress = function (e) { if (onProgress) onProgress(e.loaded); };
  xhr.onload = function () {
    if (xhr.status !== 200 && xhr.status !== 0) { cb(new Error(url + ': HTTP ' + xhr.status)); return; }
    cb(null, xhr.response);
  };
  xhr.onerror = function () { cb(new Error(url + ': network error')); };
  xhr.send();
}

// UTF-8 bytes -> string (no TextDecoder in old browsers); the files here are valid UTF-8.
function utf8(buf) {
  var b = new Uint8Array(buf), n = b.length, i = 0, parts = [], chunk = [], c, cp;
  while (i < n) {
    c = b[i++];
    if (c < 0x80) cp = c;
    else if (c < 0xE0) cp = ((c & 0x1F) << 6) | (b[i++] & 0x3F);
    else if (c < 0xF0) { cp = ((c & 0x0F) << 12) | ((b[i] & 0x3F) << 6) | (b[i + 1] & 0x3F); i += 2; }
    else { cp = ((c & 0x07) << 18) | ((b[i] & 0x3F) << 12) | ((b[i + 1] & 0x3F) << 6) | (b[i + 2] & 0x3F); i += 3; }
    if (cp >= 0x10000) { cp -= 0x10000; chunk.push(0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)); }
    else chunk.push(cp);
    if (chunk.length >= 8192) { parts.push(String.fromCharCode.apply(null, chunk)); chunk = []; }
  }
  parts.push(String.fromCharCode.apply(null, chunk));
  return parts.join('');
}

// FNV-1a over the float32 bit patterns (same as tools/ref/ref_embed.mjs)
function fnv(data) {
  var u = new Uint32Array(data.buffer, data.byteOffset, data.length), h = 0x811c9dc5, i, s, x;
  for (i = 0; i < u.length; i++) {
    x = u[i];
    for (s = 0; s < 32; s += 8) {
      h ^= (x >>> s) & 255;
      // h * 16777619 mod 2^32 without Math.imul
      h = ((h & 0xffff) * 0x01000193 + ((((h >>> 16) * 0x0193) & 0xffff) << 16)) >>> 0;
    }
  }
  var hex = h.toString(16);
  while (hex.length < 8) hex = '0' + hex;
  return hex;
}

function fail(err, id) {
  self.postMessage({ type: 'error', id: id, message: String((err && err.message) || err) });
}

function load(data, ports) {
  var base = data.base || 'model/';
  var t0 = now();
  self.postMessage({ type: 'progress', stage: 'manifest' });
  fetchFile(base + 'manifest.json', null, function (err, buf) {
    if (err) return fail(err);
    try { manifest = JSON.parse(utf8(buf)); } catch (e) { return fail(e); }
    fetchFile(base + 'vocab.txt', function (loaded) {
      self.postMessage({ type: 'progress', stage: 'tokenizer', loaded: loaded, total: 2831346 + 2000016 });
    }, function (err, vbuf) {
      if (err) return fail(err);
      fetchFile(base + 'scores.bin', function (loaded) {
        self.postMessage({ type: 'progress', stage: 'tokenizer', loaded: 2831346 + loaded, total: 2831346 + 2000016 });
      }, function (err, sbuf) {
        if (err) return fail(err);
        try {
          tokenizer = new MiniLMTokenizer.Tokenizer(utf8(vbuf).split('\n'), new Float64Array(sbuf), manifest.tokenizer);
        } catch (e) { return fail(e); }
        vbuf = null; sbuf = null;
        var threads = ports.length > 1 ? ports.length : 1, portIndex = 0;
        MiniLMEngine.load({
          manifest: manifest,
          threads: threads,
          kernel: data.kernel || 'auto',
          fetch: function (name, progress, cb) { fetchFile(base + name, progress, cb); },
          createPort: function () {
            var p = ports[portIndex++];
            var wrapper = { post: function (msg, transfer) { p.postMessage(msg, transfer); }, onmessage: null, terminate: null };
            p.onmessage = function (e) { if (wrapper.onmessage) wrapper.onmessage(e.data); };
            if (p.start) p.start();
            return wrapper;
          }
        }, function (done, total, file) {
          self.postMessage({ type: 'progress', stage: 'model', loaded: done, total: total, file: file });
        }, function (err, m) {
          if (err) return fail(err);
          model = m;
          self.postMessage({ type: 'ready', threads: threads, kernels: m.kernels, ms: Math.round(now() - t0) });
        });
      });
    });
  });
}

function embed(req, done) {
  if (!model) { fail(new Error('Model not loaded'), req.id); done(); return; }
  var t0 = now(), lists = [], i, H = manifest.config.hidden;
  try { for (i = 0; i < req.texts.length; i++) lists.push(tokenizer.encode(String(req.texts[i]))); } catch (e) { fail(e, req.id); done(); return; }
  var tTok = now();
  var opts = { pad: manifest.tokenizer.pad, pooling: req.pooling || 'mean', normalize: !!req.normalize };
  var groups = req.batch ? [lists] : lists.map(function (l) { return [l]; });
  var outs = [], hashes = [], g = 0;
  function next() {
    if (g === groups.length) return finish();
    model.embed(groups[g], opts, function (err, r) {
      if (err) { fail(err, req.id); done(); return; }
      outs.push(r);
      if (req.hash) hashes.push(fnv(r.hidden));
      g++;
      setTimeout(next, 0);
    });
  }
  function finish() {
    var data, dims;
    if (outs.length === 1) { data = outs[0].data; dims = outs[0].dims; }
    else {
      var len = 0, off = 0;
      for (i = 0; i < outs.length; i++) len += outs[i].data.length;
      data = new Float32Array(len);
      for (i = 0; i < outs.length; i++) { data.set(outs[i].data, off); off += outs[i].data.length; }
      dims = opts.pooling === 'mean' ? [outs.length, H] : null;   // 'none': one tensor per text, see hashes
    }
    self.postMessage({ type: 'result', id: req.id, data: data, dims: dims, ids: lists, hashes: hashes, tokenizeMs: Math.round(tTok - t0), ms: Math.round(now() - t0) }, [data.buffer]);
    done();
  }
  next();
}

function pump() {
  if (busy || !queue.length) return;
  busy = true;
  embed(queue.shift(), function () { busy = false; pump(); });
}

self.onmessage = function (e) {
  var d = e.data || {};
  try {
    if (d.type === 'load') {
      // Zagreus: single-thread mode runs the shard here, so it gets the WebAssembly kernel too
      if (d.wasm && !(e.ports && e.ports.length > 1)) {
        try {
          importScripts('minilm-wasm.js');
          MiniLMWasm.install(MiniLMShard, MiniLMWasm.fetchSync(d.wasm));
          self.postMessage({ type: 'wasm', ok: true });
        } catch (err) { self.postMessage({ type: 'wasm', ok: false, error: String(err && err.message || err) }); }
      }
      load(d, e.ports || []);
    }
    else if (d.type === 'embed') { queue.push(d); pump(); }
  } catch (err) { fail(err, d.id); }
};
