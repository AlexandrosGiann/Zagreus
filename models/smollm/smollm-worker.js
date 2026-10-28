/*
 * Coordinator worker: loads the tokenizer and the model, and generates replies.
 * Messages from the page:
 *   { type: 'load', base: 'model/', threads: n }  (+ n MessagePorts to compute workers when n > 1)
 *   { type: 'generate', messages: [{ role, content }, ...] }
 *   { type: 'stop' }
 * Messages to the page: progress, ready, token, done, error (see index.html).
 */
importScripts('smollm-unicode.js', 'smollm-tokenizer.js', 'smollm-shard.js', 'smollm-engine.js', 'smollm-generate.js');

var model = null, tokenizer = null, current = null;

function now() { return (self.performance && performance.now) ? performance.now() : Date.now(); }

function fetchFile(url, type, onProgress, cb) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', url, true);
  xhr.responseType = type;
  xhr.onprogress = function (e) { if (onProgress) onProgress(e.loaded); };
  xhr.onload = function () {
    if (xhr.status !== 200 && xhr.status !== 0) { cb(new Error(url + ': HTTP ' + xhr.status)); return; }
    cb(null, xhr.response);
  };
  xhr.onerror = function () { cb(new Error(url + ': network error')); };
  xhr.send();
}

function fail(err) {
  self.postMessage({ type: 'error', message: String((err && err.message) || err) });
}

function load(data, ports) {
  var base = data.base || 'model/';
  var t0 = now();
  self.postMessage({ type: 'progress', stage: 'manifest' });
  fetchFile(base + 'manifest.json', 'text', null, function (err, text) {
    if (err) return fail(err);
    var manifest;
    try { manifest = JSON.parse(text); } catch (e) { return fail(e); }
    var tokBytes = 2053526;
    fetchFile(base + 'tokenizer.json', 'text', function (loaded) {
      self.postMessage({ type: 'progress', stage: 'tokenizer', loaded: loaded, total: tokBytes });
    }, function (err, tokText) {
      if (err) return fail(err);
      try { tokenizer = new SmolLMTokenizer.Tokenizer(JSON.parse(tokText)); } catch (e) { return fail(e); }
      tokText = null;
      var threads = ports.length > 1 ? ports.length : 1;
      var portIndex = 0;
      SmolLMEngine.load({
        manifest: manifest,
        threads: threads,
        fetch: function (name, progress, cb) { fetchFile(base + name, 'arraybuffer', progress, cb); },
        createPort: function () {
          var p = ports[portIndex++];
          var wrapper = {
            post: function (msg, transfer) { p.postMessage(msg, transfer); },
            onmessage: null,
            terminate: null
          };
          p.onmessage = function (e) { if (wrapper.onmessage) wrapper.onmessage(e.data); };
          if (p.start) p.start();
          return wrapper;
        }
      }, function (done, total, file) {
        self.postMessage({ type: 'progress', stage: 'model', loaded: done, total: total, file: file });
      }, function (err, m) {
        if (err) return fail(err);
        model = m;
        self.postMessage({ type: 'ready', threads: threads, ms: Math.round(now() - t0) });
      });
    });
  });
}

function generate(messages, maxNewTokens) {
  if (!model) return fail(new Error('Model not loaded'));
  var g;
  try { g = new SmolLMGenerate.Generation(model, tokenizer, messages, { maxNewTokens: maxNewTokens || 96 }); } catch (e) { return fail(e); }
  current = g;
  var t0 = now(), tFirst = 0;
  self.postMessage({ type: 'started', promptTokens: g.promptIds.length });
  g.run(function (id, gen) {
    if (!tFirst) tFirst = now();
    self.postMessage({ type: 'token', text: gen.text(), count: gen.generated.length, prefillMs: Math.round(tFirst - t0), ms: Math.round(now() - t0) });
  }, function (err) {
    current = null;
    if (err) return fail(err);
    var t1 = now();
    self.postMessage({
      type: 'done', text: g.finalText(), stopped: !!g.stopped,
      promptTokens: g.promptIds.length, tokens: g.generated.length,
      prefillMs: Math.round((tFirst || t1) - t0), ms: Math.round(t1 - t0), ids: g.generated
    });
  });
}

self.onmessage = function (e) {
  var d = e.data || {};
  try {
    if (d.type === 'load') {
      // Zagreus: single-thread mode runs the shard here, so it gets the WebAssembly kernel too
      if (d.wasm && !(e.ports && e.ports.length > 1)) {
        try {
          importScripts('smollm-wasm.js');
          SmolLMWasm.install(SmolLMShard, SmolLMWasm.fetchSync(d.wasm));
          self.postMessage({ type: 'wasm', ok: true });
        } catch (err) { self.postMessage({ type: 'wasm', ok: false, error: String(err && err.message || err) }); }
      }
      load(d, e.ports || []);
    }
    else if (d.type === 'generate') generate(d.messages, d.maxNewTokens);
    else if (d.type === 'stop') { if (current) current.stop(); }
  } catch (err) { fail(err); }
};
