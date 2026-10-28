/*
 * WebAssembly acceleration for the MiniLM pure-JS runtime (Zagreus addition).
 *
 * Plugs the `int_sums` kernel of zagreus-simd.wasm into minilm-shard.js (setSumsHook): the
 * exact integer part of every quantized linear layer. The float part stays in JS, so the
 * embeddings are bit-identical. Each weight matrix is copied into WebAssembly memory on first
 * use and its JS copy is released.
 *
 *   MiniLMWasm.install(MiniLMShard, wasmBytes)
 */
(function (root) {
  'use strict';

  function install(ShardLib, bytes) {
    var ex = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}).exports;
    if (!ex.int_sums) throw new Error('this wasm build has no int_sums');
    var mem = ex.memory, sc = { qa: 0, qaCap: 0, out: 0, outCap: 0 };
    function alloc(n) {
      var p = ex.zg_alloc(n);
      if (!p) throw new Error('WebAssembly memory exhausted');
      return p;
    }
    ShardLib.setSumsHook(function (qa, rows, K, lin, N, sums) {
      if (!lin._wp) {
        lin._wp = alloc(lin.w.length);
        new Int8Array(mem.buffer, lin._wp, lin.w.length).set(lin.w);
        lin.w = new Int8Array(0);
      }
      var nq = rows * K, no = rows * N;
      if (sc.qaCap < nq) { sc.qaCap = nq; sc.qa = alloc(nq * 2); }
      if (sc.outCap < no) { sc.outCap = no; sc.out = alloc(no * 8); }
      new Int16Array(mem.buffer, sc.qa, nq).set(qa.subarray(0, nq));
      ex.int_sums(sc.qa, rows, K, lin._wp, N, sc.out);
      sums.set(new Float64Array(mem.buffer, sc.out, no));
    });
    return true;
  }

  function fetchSync(url) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.responseType = 'arraybuffer';
    xhr.send(null);
    if (xhr.status !== 200 && xhr.status !== 0) throw new Error(url + ': HTTP ' + xhr.status);
    return xhr.response;
  }

  var api = { install: install, fetchSync: fetchSync };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MiniLMWasm = api;
})(typeof self !== 'undefined' ? self : this);
