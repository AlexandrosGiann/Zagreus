/*
 * WebAssembly acceleration for the SmolLM2 pure-JS runtime (Zagreus addition).
 *
 * Replaces Shard.prototype._mm, the q8 matmul behind q/k/v/o/gate/up/down, with the
 * `smol_matmul` kernel of zagreus-simd.wasm. The kernel computes the same exact integer sums
 * and the same two float32 roundings as matmulQ, so every logit and every token is unchanged.
 * Weights are moved into WebAssembly memory as they arrive (the JS copies are released).
 *
 *   SmolLMWasm.install(SmolLMShard, wasmBytes)   (synchronous; call before the weights arrive)
 */
(function (root) {
  'use strict';

  function install(ShardLib, bytes) {
    var mod = new WebAssembly.Module(bytes);
    var ex = new WebAssembly.Instance(mod, {}).exports;
    if (!ex.smol_matmul) throw new Error('this wasm build has no smol_matmul');
    var mem = ex.memory, scratch = { qa: 0, qaCap: 0, rs: 0, rsCap: 0, out: 0, outCap: 0 };

    function alloc(n) {
      var p = ex.zg_alloc(n);
      if (!p) throw new Error('WebAssembly memory exhausted');
      return p;
    }

    var P = ShardLib.Shard.prototype;
    var jsSetWeight = P.setWeight;

    P.setWeight = function (l, name, bytes) {
      var p = alloc(bytes.length);
      new Uint8Array(mem.buffer, p, bytes.length).set(bytes);
      if (!this._wasmW) this._wasmW = [];
      if (!this._wasmW[l]) this._wasmW[l] = {};
      this._wasmW[l][name] = p;
      // keep the JS array object (other code checks for it) but drop its bytes
      jsSetWeight.call(this, l, name, new Uint8Array(0));
    };

    P._mm = function (l, name, qa, rowSums, rows, K, actScale, out) {
      var r = this.ranges[name], m = this.layerMeta[l][name];
      var N = r[1] - r[0], nq = rows * K;
      if (scratch.qaCap < nq) { scratch.qaCap = nq; scratch.qa = alloc(nq * 2); }
      if (scratch.rsCap < rows) { scratch.rsCap = rows; scratch.rs = alloc(rows * 8); }
      if (scratch.outCap < rows * N) { scratch.outCap = rows * N; scratch.out = alloc(rows * N * 4); }
      var buf = mem.buffer;
      // qa is an Int32Array of zero-point-shifted uint8 values: it fits in int16
      new Int16Array(buf, scratch.qa, nq).set(qa.subarray(0, nq));
      new Float64Array(buf, scratch.rs, rows).set(rowSums.subarray(0, rows));
      var cs = Math.fround(actScale * Math.fround(m.scale));
      ex.smol_matmul(scratch.qa, scratch.rs, rows, K, this._wasmW[l][name], N, m.zp, cs, scratch.out);
      out.set(new Float32Array(mem.buffer, scratch.out, rows * N));
      return out;
    };
    return true;
  }

  // Fetches the wasm synchronously from inside a worker (sync XHR is allowed there).
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
  else root.SmolLMWasm = api;
})(typeof self !== 'undefined' ? self : this);
