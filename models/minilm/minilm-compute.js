/*
 * Compute worker: holds one shard of the model weights and runs its matmuls and attention
 * heads. The page creates it and hands it one end of a MessageChannel whose other end goes to
 * the coordinator worker (minilm-worker.js), so no nested workers are needed.
 */
importScripts('minilm-shard.js');

var state = {};

function serve(port) {
  port.onmessage = function (e) {
    var r;
    try {
      r = MiniLMShard.Shard.handle(state, e.data);
    } catch (err) {
      port.postMessage({ op: 'error', message: String((err && err.message) || err) });
      return;
    }
    if (r) port.postMessage(r[0], r[1]);
  };
  if (port.start) port.start();
}

// Zagreus: { type: 'port', wasm: url } switches the integer sums to WebAssembly SIMD
// (minilm-wasm.js; same embeddings, faster). Without `wasm`, or if it fails, the JS kernel stays.
self.onmessage = function (e) {
  if (e.data && e.data.type === 'port' && e.ports && e.ports[0]) {
    if (e.data.wasm) {
      try {
        importScripts('minilm-wasm.js');
        MiniLMWasm.install(MiniLMShard, MiniLMWasm.fetchSync(e.data.wasm));
        self.postMessage({ type: 'wasm', ok: true });
      } catch (err) { self.postMessage({ type: 'wasm', ok: false, error: String(err && err.message || err) }); }
    }
    serve(e.ports[0]);
  }
};
