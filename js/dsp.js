// WebAssembly signal processing for the call mode and vector search.
// Loads wasm/zagreus-simd.wasm when the browser has WebAssembly SIMD, wasm/zagreus-base.wasm
// otherwise, and keeps pure-JS fallbacks for browsers without WebAssembly.

const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

export function wasmSupport() {
  if (typeof WebAssembly !== 'object') return { wasm: false, simd: false };
  let simd = false;
  try { simd = WebAssembly.validate(SIMD_PROBE); } catch (e) { simd = false; }
  return { wasm: true, simd };
}

const bytesCache = {};
// Resolves with { bytes: ArrayBuffer, simd: bool } (a fresh copy each time, so it can be transferred).
export async function fetchWasmBytes(preferSimd = true) {
  const sup = wasmSupport();
  if (!sup.wasm) throw new Error('WebAssembly is not available');
  const simd = preferSimd && sup.simd;
  const url = simd ? 'wasm/zagreus-simd.wasm' : 'wasm/zagreus-base.wasm';
  if (!bytesCache[url]) {
    bytesCache[url] = fetch(url).then((r) => {
      if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
      return r.arrayBuffer();
    });
  }
  const buf = await bytesCache[url];
  return { bytes: buf.slice(0), simd };
}

let dspPromise = null;
export function loadDsp() {
  if (!dspPromise) {
    dspPromise = (async () => {
      try {
        const { bytes, simd } = await fetchWasmBytes(true);
        const { instance } = await WebAssembly.instantiate(bytes, {});
        return new WasmDsp(instance.exports, simd);
      } catch (e) {
        console.warn('WebAssembly DSP unavailable, using JS:', e);
        return new JsDsp();
      }
    })();
  }
  return dspPromise;
}

class WasmDsp {
  constructor(ex, simd) {
    this.ex = ex;
    this.kind = simd ? 'wasm-simd' : 'wasm';
    this.inCap = 0; this.outCap = 0; this.inPtr = 0; this.outPtr = 0;
    this.vecCap = 0; this.qPtr = 0; this.matPtr = 0; this.scPtr = 0;
    this.matRows = 0; this.matDim = 0; this.matKey = null;
  }
  _f32() { return new Float32Array(this.ex.memory.buffer); }
  _ensureIO(n, m) {
    if (n > this.inCap) { this.inCap = n; this.inPtr = this.ex.zg_alloc(n * 4); }
    if (m > this.outCap) { this.outCap = m; this.outPtr = this.ex.zg_alloc(m * 4); }
  }
  resampleInit(inRate, outRate) { this.ex.rs_init(inRate, outRate); this.ratio = inRate / outRate; }
  // Float32Array at inRate -> Float32Array at outRate (streaming; keeps history between calls)
  resample(input) {
    const cap = Math.ceil(input.length / this.ratio) + 64;
    this._ensureIO(input.length, cap);
    this._f32().set(input, this.inPtr >> 2);
    const n = this.ex.rs_process(this.inPtr, input.length, this.outPtr, cap);
    return this._f32().slice(this.outPtr >> 2, (this.outPtr >> 2) + n);
  }
  vadInit(thresholdDb, endMs, frameMs) { this.ex.vad_init(thresholdDb, endMs, frameMs); }
  // 0 silence, 1 speech, 2 end of utterance
  vad(frame) {
    this._ensureIO(frame.length, 0);
    this._f32().set(frame, this.inPtr >> 2);
    return this.ex.vad_process(this.inPtr, frame.length);
  }
  levelDb() { return this.ex.vad_level_db(); }
  noiseDb() { return this.ex.vad_noise_db(); }
  // Cosine scores of `query` against rows of `matrix` (Float32Array rows*dim), both normalised.
  // The matrix is copied into WebAssembly memory once per `key`.
  scores(query, matrix, rows, dim, key) {
    if (!rows) return new Float32Array(0);
    if (this.matKey !== key || this.matRows !== rows || this.matDim !== dim) {
      if (rows * dim > this.vecCap) {
        this.vecCap = rows * dim;
        this.matPtr = this.ex.zg_alloc(rows * dim * 4);
        this.scPtr = this.ex.zg_alloc(rows * 4);
      }
      if (!this.qPtr || dim > (this.qDim || 0)) { this.qPtr = this.ex.zg_alloc(dim * 4); this.qDim = dim; }
      this._f32().set(matrix.subarray(0, rows * dim), this.matPtr >> 2);
      this.matKey = key; this.matRows = rows; this.matDim = dim;
    }
    const f = this._f32();
    f.set(query, this.qPtr >> 2);
    this.ex.cos_scores(this.qPtr, this.matPtr, rows, dim, this.scPtr);
    return this._f32().slice(this.scPtr >> 2, (this.scPtr >> 2) + rows);
  }
}

// Same interface in plain JavaScript (linear-interpolation resampler, energy VAD).
class JsDsp {
  constructor() { this.kind = 'js'; this.pos = 0; this.last = 0; }
  resampleInit(inRate, outRate) { this.ratio = inRate / outRate; this.pos = 0; this.last = 0; }
  resample(input) {
    const out = [];
    while (this.pos < input.length) {
      const i = Math.floor(this.pos), f = this.pos - i;
      const a = i === 0 ? (f === 0 ? input[0] : this.last) : input[i - 1];
      const b = input[i];
      out.push(i === 0 && f === 0 ? b : a + (b - a) * f);
      this.pos += this.ratio;
    }
    this.pos -= input.length;
    this.last = input[input.length - 1];
    return Float32Array.from(out);
  }
  vadInit(thresholdDb, endMs, frameMs) {
    this.th = thresholdDb; this.endFrames = Math.floor(endMs / frameMs); this.minSpeech = Math.floor(120 / frameMs) + 1;
    this.noise = -60; this.inSpeech = false; this.sp = 0; this.sil = 0; this.db = -100;
  }
  vad(frame) {
    let ss = 0; for (let i = 0; i < frame.length; i++) ss += frame[i] * frame[i];
    const db = 10 * Math.log10(ss / Math.max(1, frame.length) + 1e-12); this.db = db;
    const voiced = db > this.noise + this.th && db > -55;
    if (!this.inSpeech) {
      this.noise = db < this.noise ? 0.7 * this.noise + 0.3 * db : 0.995 * this.noise + 0.005 * db;
      if (voiced) { if (++this.sp >= this.minSpeech) { this.inSpeech = true; this.sil = 0; } return 1; }
      this.sp = 0; return 0;
    }
    if (voiced) { this.sil = 0; return 1; }
    if (++this.sil >= this.endFrames) { this.inSpeech = false; this.sp = 0; this.sil = 0; return 2; }
    return 1;
  }
  levelDb() { return this.db; }
  noiseDb() { return this.noise; }
  scores(query, matrix, rows, dim) {
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let s = 0; const o = r * dim;
      for (let k = 0; k < dim; k++) s += query[k] * matrix[o + k];
      out[r] = s;
    }
    return out;
  }
}
