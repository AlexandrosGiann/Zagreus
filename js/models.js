// Page-side clients for the five on-device models and the TTS engine.
// Every model runs in its own Web Worker; these classes turn the worker messages into
// promises and keep a small status object the UI renders (state, progress, detail).

import { fetchWasmBytes, wasmSupport } from './dsp.js';

const listeners = new Set();
export function onModelChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function emit(m) { for (const fn of listeners) { try { fn(m); } catch (e) { console.error(e); } } }

function cores() { return Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)); }

class ModelClient {
  constructor(key, label, sizeMB) {
    this.key = key; this.label = label; this.sizeMB = sizeMB;
    this.state = 'idle';      // idle | loading | ready | error
    this.progress = 0;        // 0..1 while loading
    this.detail = '';
    this.error = null;
    this._ready = null;
    this._seq = 0;
    this._pending = new Map();
  }
  set(patch) { Object.assign(this, patch); emit(this); }
  get ready() { return this.state === 'ready'; }
  load() {
    if (!this._ready) {
      this.set({ state: 'loading', progress: 0, error: null });
      this._ready = this._load().then(
        (v) => { this.set({ state: 'ready', progress: 1 }); return v; },
        (e) => { this._ready = null; this.set({ state: 'error', error: String(e && e.message || e) }); throw e; });
    }
    return this._ready;
  }
  _call(worker, msg, transfer) {
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, msg });
      worker.postMessage(Object.assign({ id }, msg), transfer || []);
    });
  }
  _settle(id, err, value) {
    const p = this._pending.get(id);
    if (!p) return null;
    this._pending.delete(id);
    if (err) p.reject(new Error(err)); else p.resolve(value);
    return p;
  }
}

// Spawns a coordinator worker plus n compute workers joined by MessageChannels
// (the layout used by the SmolLM2 and MiniLM pure-JS runtimes).
// wasm: URL of the WebAssembly kernels relative to the worker, or null; onWasm(ok) reports
// whether each worker installed them.
function spawnSharded(dir, coord, compute, n, wasm, onWasm) {
  const worker = new Worker(dir + coord);
  const ports = [], computeWorkers = [];
  if (n > 1 && typeof MessageChannel !== 'undefined') {
    for (let i = 0; i < n; i++) {
      const cw = new Worker(dir + compute);
      const ch = new MessageChannel();
      if (onWasm) cw.onmessage = (e) => { if (e.data && e.data.type === 'wasm') onWasm(e.data.ok, e.data.error); };
      cw.postMessage({ type: 'port', wasm }, [ch.port1]);
      ports.push(ch.port2);
      computeWorkers.push(cw);
    }
  }
  return { worker, ports, computeWorkers };
}

// The SIMD build is the one worth using inside the model workers (the scalar build is slower
// than the hand-tuned JS kernels).
let wantModelWasm = true;
export function setModelWasm(on) { wantModelWasm = on; }
function modelWasmUrl() { return wantModelWasm && wasmSupport().simd ? '../../wasm/zagreus-simd.wasm' : null; }

// ---------------------------------------------------------------- MiniLM (embeddings)

export class MiniLM extends ModelClient {
  constructor() { super('minilm', 'paraphrase-multilingual-MiniLM-L12-v2', 123); this.dim = 384; }
  _load() {
    return new Promise((resolve, reject) => {
      const n = Math.min(2, cores());
      const wasm = modelWasmUrl();
      this.kernel = { wasm: 0, js: 0 };
      const onWasm = (ok) => { this.kernel[ok ? 'wasm' : 'js']++; };
      const s = spawnSharded('models/minilm/', 'minilm-worker.js', 'minilm-compute.js', n, wasm, onWasm);
      this.worker = s.worker;
      this.worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === 'wasm') onWasm(d.ok);
        else if (d.type === 'progress') {
          if (d.stage === 'model' && d.total) this.set({ progress: 0.05 + 0.95 * d.loaded / d.total, detail: d.file || '' });
          else if (d.stage === 'tokenizer' && d.total) this.set({ progress: 0.05 * d.loaded / d.total, detail: 'tokenizer' });
        } else if (d.type === 'ready') {
          const k = this.kernel.wasm && !this.kernel.js ? 'WebAssembly SIMD' : 'pure JS';
          this.set({ detail: `${k} · ${d.threads} threads · ${(d.ms / 1000).toFixed(1)} s` });
          resolve();
        } else if (d.type === 'result') {
          this._settle(d.id, null, d);
        } else if (d.type === 'error') {
          if (!this._settle(d.id, d.message)) reject(new Error(d.message));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'MiniLM worker failed'));
      this.worker.postMessage({ type: 'load', base: 'model/', threads: s.ports.length || 1, kernel: 'auto', wasm }, s.ports);
    });
  }
  // Returns an array of Float32Array(384), mean-pooled and L2-normalised.
  async embed(texts) {
    await this.load();
    const list = Array.isArray(texts) ? texts : [texts];
    const r = await this._call(this.worker, { type: 'embed', texts: list, batch: false, pooling: 'mean', normalize: true });
    const out = [];
    for (let i = 0; i < list.length; i++) out.push(r.data.slice(i * this.dim, (i + 1) * this.dim));
    return out;
  }
}

// ---------------------------------------------------------------- SmolLM2 (chat)

export class SmolLM extends ModelClient {
  constructor() { super('smollm', 'SmolLM2-135M-Instruct', 139); this.current = null; }
  _load() {
    return new Promise((resolve, reject) => {
      const wasm = modelWasmUrl();
      this.kernel = { wasm: 0, js: 0 };
      const onWasm = (ok) => { this.kernel[ok ? 'wasm' : 'js']++; };
      const s = spawnSharded('models/smollm/', 'smollm-worker.js', 'smollm-compute.js', cores(), wasm, onWasm);
      this.worker = s.worker;
      this.worker.onmessage = (e) => {
        const d = e.data, cur = this.current;
        if (d.type === 'wasm') onWasm(d.ok);
        else if (d.type === 'progress') {
          if (d.stage === 'model' && d.total) this.set({ progress: 0.02 + 0.98 * d.loaded / d.total, detail: d.file || '' });
          else if (d.stage === 'tokenizer' && d.total) this.set({ progress: 0.02 * d.loaded / d.total, detail: 'tokenizer' });
        } else if (d.type === 'ready') {
          const k = this.kernel.wasm && !this.kernel.js ? 'WebAssembly SIMD' : 'pure JS';
          this.set({ detail: `${k} · ${d.threads} threads · ${(d.ms / 1000).toFixed(1)} s` });
          resolve();
        } else if (d.type === 'token' && cur) {
          cur.onToken && cur.onToken(d.text);
        } else if (d.type === 'done' && cur) {
          this.current = null;
          cur.resolve(d);
        } else if (d.type === 'error') {
          if (cur) { this.current = null; cur.reject(new Error(d.message)); }
          else reject(new Error(d.message));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'SmolLM worker failed'));
      this.worker.postMessage({ type: 'load', base: 'model/', threads: s.ports.length || 1, wasm }, s.ports);
    });
  }
  // messages: [{role, content}], onToken(textSoFar). Resolves with { text, tokens, ms, prefillMs }.
  async generate(messages, onToken, maxNewTokens) {
    await this.load();
    if (this.current) this.stop();
    return new Promise((resolve, reject) => {
      this.current = { resolve, reject, onToken };
      this.worker.postMessage({ type: 'generate', messages, maxNewTokens });
    });
  }
  stop() { if (this.worker) this.worker.postMessage({ type: 'stop' }); }
}

// ---------------------------------------------------------------- Whistle (speech to text)

export class WhistleSTT extends ModelClient {
  constructor() { super('whistle', 'Cactus Whistle (STT)', 17); this.accel = { on: false, simd: false }; this.wantAccel = true; }
  async _load() {
    const [cact, wasm] = await Promise.all([
      this._fetchCact(),
      fetchWasmBytes(true).catch(() => null)
    ]);
    // the SIMD build is the only one faster than the JS kernel; without SIMD stay on JS
    const useWasm = wasm && wasm.simd ? wasm.bytes : null;
    return new Promise((resolve, reject) => {
      this.worker = new Worker('models/whistle/whistle_worker.js');
      this.worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === 'loaded') {
          this.accel = d.accel;
          this.set({ detail: this._accelLabel() });
          if (!this.wantAccel && this.accel.on) this.setAccel(false);
          resolve();
        } else if (d.type === 'accel') {
          this.accel = d.accel;
          this.set({ detail: this._accelLabel() });
        } else if (d.type === 'partial') {
          const p = this._pending.get(d.id);
          if (p && p.msg.onPartial) p.msg.onPartial(d.text);
        } else if (d.type === 'result') {
          this._settle(d.id, null, d);
        } else if (d.type === 'error') {
          if (!this._settle(d.id, d.message)) reject(new Error(d.message));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'Whistle worker failed'));
      const transfer = [cact];
      this.worker.postMessage({ type: 'load', buffer: cact, wasm: useWasm }, transfer);
    });
  }
  _accelLabel() { return this.accel.on ? 'WebAssembly SIMD' : 'pure JS'; }
  _fetchCact() {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', 'models/whistle/whistle.cact', true);
      xhr.responseType = 'arraybuffer';
      xhr.onprogress = (e) => { if (e.total) this.set({ progress: e.loaded / e.total, detail: 'whistle.cact' }); };
      xhr.onload = () => (xhr.status === 200 || xhr.status === 0) ? resolve(xhr.response) : reject(new Error('whistle.cact: HTTP ' + xhr.status));
      xhr.onerror = () => reject(new Error('whistle.cact: network error'));
      xhr.send();
    });
  }
  setAccel(on) {
    this.wantAccel = on;
    if (this.worker && this.state === 'ready') this.worker.postMessage({ type: 'accel', on });
  }
  // pcm: Float32Array at 16 kHz. opts: { language, beams, onPartial }
  async transcribe(pcm, opts = {}) {
    await this.load();
    const copy = new Float32Array(pcm);
    const msg = { type: 'transcribe', pcm: copy, language: opts.language || null, beams: opts.beams || 5 };
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, msg: { onPartial: opts.onPartial } });
      this.worker.postMessage(Object.assign({ id }, msg), [copy.buffer]);
    });
  }
}

// ---------------------------------------------------------------- Needle 2 (tool calls)

export class Needle extends ModelClient {
  constructor() { super('needle', 'Needle 2 (tool calls)', 14); this.warm = {}; }
  async _load() {
    const spec = await loadToolSpec();
    return new Promise((resolve, reject) => {
      this.worker = new Worker('models/needle/needle-worker.js');
      this.worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === 'progress') {
          if (d.stage === 'model' && d.total) this.set({ progress: d.loaded / d.total, detail: 'needle2.cact' });
          else if (d.stage === 'warm') this.set({ detail: `prefix ${d.profile} ${d.loaded}/${d.total}` });
        } else if (d.type === 'ready') {
          this.set({ detail: 'warming tool prompts…' });
          resolve();
        } else if (d.type === 'warm') {
          this.warm[d.profile] = d;
          const n = Object.keys(this.warm).length, all = Object.keys(spec.profiles).length;
          this.set({ detail: n < all ? `tool prompts ${n}/${all}` : 'ready · cached tool prompts' });
        } else if (d.type === 'result') {
          this._settle(d.id, null, d);
        } else if (d.type === 'error') {
          if (!this._settle(d.id, d.message)) reject(new Error(d.message));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'Needle worker failed'));
      this.worker.postMessage({ type: 'load', profiles: spec.profiles, system: spec.system });
    });
  }
  async route(profile, text) {
    await this.load();
    return this._call(this.worker, { type: 'route', profile, text });
  }
}

let toolSpecPromise = null;
export function loadToolSpec() {
  if (!toolSpecPromise) {
    toolSpecPromise = new Promise((resolve, reject) => {
      if (self.ZagreusNeedleTools) return resolve(self.ZagreusNeedleTools);
      const s = document.createElement('script');
      s.src = 'js/needle-tools.js';
      s.onload = () => resolve(self.ZagreusNeedleTools);
      s.onerror = () => reject(new Error('needle-tools.js failed to load'));
      document.head.appendChild(s);
    });
  }
  return toolSpecPromise;
}

// ---------------------------------------------------------------- MobileCLIP (vision)

export class MobileClip extends ModelClient {
  constructor() { super('clip', 'MobileCLIP-S0 (vision)', 48); }
  _load() {
    return new Promise((resolve, reject) => {
      this.worker = new Worker('models/mobileclip/browser-fp32-opt-v2-worker-bundle.js');
      this.worker.onmessage = (e) => {
        const d = e.data, cur = this.current;
        if (d.type === 'load-progress') this.set({ progress: (d.percent || 0) / 100, detail: d.phase || '' });
        else if (d.type === 'ready') { this.set({ detail: `${d.labels} labels` }); resolve(); }
        else if (d.type === 'infer-progress' && cur) cur.onProgress && cur.onProgress(d.percent / 100);
        else if (d.type === 'result' && cur) { this.current = null; cur.resolve(d); }
        else if (d.type === 'error') {
          if (cur) { this.current = null; cur.reject(new Error(d.text)); }
          else reject(new Error(d.text));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'MobileCLIP worker failed'));
    });
  }
  // image: anything drawImage accepts. Resolves with { top: [{label, score}], inferenceMs }.
  async classify(image, onProgress) {
    await this.load();
    if (this.current) throw new Error('busy');
    const input = imageToTensor(image);
    return new Promise((resolve, reject) => {
      this.current = { resolve, reject, onProgress };
      this.worker.postMessage({ type: 'run', inputBuffer: input.buffer }, [input.buffer]);
    });
  }
}

// Centre square crop, 256x256, RGB / 255, NCHW float32, no mean/std (as the MobileCLIP runtime expects).
function imageToTensor(img) {
  const w = img.naturalWidth || img.videoWidth || img.width, h = img.naturalHeight || img.videoHeight || img.height;
  const side = Math.min(w, h), sx = (w - side) / 2, sy = (h - side) / 2;
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, sx, sy, side, side, 0, 0, 256, 256);
  const px = ctx.getImageData(0, 0, 256, 256).data, plane = 256 * 256;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = px[i * 4] / 255;
    out[plane + i] = px[i * 4 + 1] / 255;
    out[2 * plane + i] = px[i * 4 + 2] / 255;
  }
  return out;
}

// ---------------------------------------------------------------- meSpeak (text to speech)

export class MeSpeak extends ModelClient {
  constructor() { super('tts', 'meSpeak.js (TTS)', 6); }
  _load() {
    return new Promise((resolve, reject) => {
      this.worker = new Worker('vendor/mespeak/mespeak-worker.js');
      this.worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === 'ready') { this.voices = d.voices; this.set({ detail: d.voices.join(', ') }); resolve(); }
        else if (d.type === 'audio') this._settle(d.id, null, d);
        else if (d.type === 'error') { if (!this._settle(d.id, d.message)) reject(new Error(d.message)); }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'meSpeak worker failed'));
      this.worker.postMessage({ type: 'load', voices: ['voices/el.json', 'voices/en-us.json'] });
    });
  }
  // Resolves with { wav: ArrayBuffer, ms }.
  async synth(text, voice, opts = {}) {
    await this.load();
    return this._call(this.worker, { type: 'speak', text, voice, speed: opts.speed, pitch: opts.pitch, wordgap: opts.wordgap });
  }
}

export const models = {
  minilm: new MiniLM(),
  whistle: new WhistleSTT(),
  needle: new Needle(),
  smollm: new SmolLM(),
  clip: new MobileClip(),
  tts: new MeSpeak()
};
