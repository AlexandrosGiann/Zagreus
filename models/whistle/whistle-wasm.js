/*
 * WebAssembly acceleration for whistle.js.
 *
 * Replaces QMat.prototype.apply (the int8 x int8 matrix-vector product that takes about half of
 * the transcription time) with the `qmat_matvec` kernel of wasm/zagreus-*.wasm. The kernel does
 * the per-group dot products in exact integer arithmetic and accumulates in f64 in the same
 * order as the JS code, so transcripts and encoder outputs stay bit-identical.
 *
 * Usage (worker or Node, synchronous):
 *   WhistleWasm.install(Whistle, wasmBytes)   -> { simd: bool } or throws
 *   WhistleWasm.uninstall(Whistle)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.WhistleWasm = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var state = null;

  function views() {
    var buf = state.mem.buffer;
    if (state.buf !== buf) {
      state.buf = buf;
      state.i8 = new Int8Array(buf);
      state.f32 = new Float32Array(buf);
      state.f64 = new Float64Array(buf);
    }
  }

  function alloc(bytes) {
    var p = state.ex.zg_alloc(bytes);
    if (!p) throw new Error('WebAssembly memory exhausted');
    views();
    return p;
  }

  // Copies the expanded int8 weights and the group norms of one matrix into linear memory.
  function upload(m) {
    var n = m.out * m.inPad;
    var wp = alloc(n);
    var tmp = m.w;
    if (tmp) state.i8.set(tmp, wp);
    else for (var r = 0; r < m.out; r++) m.decodeRowInto(r, state.i8, wp + r * m.inPad);
    var np = alloc(m.nf.length * 4);
    state.f32.set(m.nf, np >> 2);
    m._wasm = { w: wp, nf: np, ver: state.ver };
    m.w = null; // the JS copy is no longer needed
  }

  function ensureScratch(qBytes, sCount, outCount) {
    var sc = state.scratch;
    if (sc.q < qBytes || sc.s < sCount || sc.out < outCount) {
      // scratch areas are allocated once at their largest size and reused
      sc.q = Math.max(sc.q, qBytes); sc.s = Math.max(sc.s, sCount); sc.out = Math.max(sc.out, outCount);
      sc.qp = alloc(sc.q);
      sc.sp = alloc(sc.s * 4);
      sc.op = alloc(sc.out * 8);
      state.lastQuant = null;
    }
  }

  function wasmApply(act, out, oOff, rowStart, rowCount) {
    rowStart = rowStart || 0;
    if (rowCount === undefined) rowCount = this.out - rowStart;
    if (!this._wasm || this._wasm.ver !== state.ver) upload(this);
    var G = this.G, nG = this.nG, qLen = nG * G;
    ensureScratch(qLen, nG, rowCount);
    var sc = state.scratch;
    views();
    // copy the activation only when it changed since the last call
    if (state.lastQuant !== act || state.lastVer !== act._zgVer || state.lastLen < qLen) {
      state.i8.set(act.q.subarray(0, qLen), sc.qp);
      state.f32.set(act.s.subarray(0, nG), sc.sp >> 2);
      state.lastQuant = act; state.lastVer = act._zgVer; state.lastLen = qLen;
    }
    state.ex.qmat_matvec(
      this._wasm.w + rowStart * this.inPad, this._wasm.nf + rowStart * nG * 4,
      sc.qp, sc.sp, sc.op, rowCount, this.inPad, G, nG);
    views();
    var f64 = state.f64, o = sc.op >> 3;
    for (var r = 0; r < rowCount; r++) out[oOff + r] = f64[o + r];
  }

  function install(Whistle, bytes) {
    var I = Whistle._internal;
    if (!I || !I.QMat || !I.Quantizer) throw new Error('whistle.js does not expose its kernels');
    if (typeof WebAssembly !== 'object') throw new Error('no WebAssembly');
    var mod = new WebAssembly.Module(bytes);
    var inst = new WebAssembly.Instance(mod, {});
    var ex = inst.exports;
    var prevVer = state ? state.ver : 0;
    state = {
      ex: ex, mem: ex.memory, buf: null, i8: null, f32: null, f64: null, ver: prevVer + 1,
      scratch: { q: 0, s: 0, out: 0, qp: 0, sp: 0, op: 0 }, lastQuant: null, lastVer: -1, lastLen: 0
    };
    views();
    var QP = I.QMat.prototype, ZP = I.Quantizer.prototype;
    if (!QP._jsApply) QP._jsApply = QP.apply;
    if (!ZP._jsSet) {
      ZP._jsSet = ZP.set;
      ZP.set = function (x, xOff, dim) {
        this._zgVer = (this._zgVer || 0) + 1;
        return this._jsSet(x, xOff, dim);
      };
    }
    QP.apply = wasmApply;
    return { simd: ex.zg_simd() === 1 };
  }

  function uninstall(Whistle) {
    var QP = Whistle._internal.QMat.prototype;
    if (QP._jsApply) QP.apply = QP._jsApply;
  }

  // A SIMD probe module: (func (result v128) i32.const 0 i32x4.splat ...)
  var SIMD_PROBE = [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11];

  function simdSupported() {
    try { return typeof WebAssembly === 'object' && WebAssembly.validate(new Uint8Array(SIMD_PROBE)); }
    catch (e) { return false; }
  }

  return { install: install, uninstall: uninstall, simdSupported: simdSupported };
});
