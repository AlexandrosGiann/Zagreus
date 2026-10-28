// Microphone capture and speech playback.
//
// Mic: getUserMedia -> ScriptProcessor (works on old and new browsers) -> 16 kHz through the
// WebAssembly resampler -> 20 ms frames for the VAD. Speaker: meSpeak WAV bytes -> Web Audio,
// queued sentence by sentence so the first sentence plays while the next is synthesised.

import { loadDsp } from './dsp.js';
import { models } from './models.js';

let ctx = null;
export function audioContext() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC();
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

export const FRAME = 320; // 20 ms at 16 kHz

export class Mic {
  // onFrame(frame16k: Float32Array(320)) is called for every 20 ms frame.
  constructor({ onFrame, echoCancel = false } = {}) {
    this.onFrame = onFrame;
    this.echoCancel = echoCancel;
    this.stream = null; this.node = null; this.src = null;
    this.pending = new Float32Array(0);
    this.muted = false;
  }
  async start() {
    const ac = audioContext();               // created inside the user gesture
    this.dsp = await loadDsp();
    const base = { channelCount: 1 };
    const constraints = this.echoCancel
      ? Object.assign(base, { echoCancellation: true, noiseSuppression: true, autoGainControl: true })
      : Object.assign(base, { echoCancellation: false, noiseSuppression: false, autoGainControl: false });
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
    } catch (e) {
      if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) throw e;
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    if (ac.state === 'suspended') await ac.resume();
    this.dsp.resampleInit(ac.sampleRate, 16000);
    this.src = ac.createMediaStreamSource(this.stream);
    this.node = ac.createScriptProcessor(2048, 1, 1);
    this.node.onaudioprocess = (e) => this._process(e.inputBuffer.getChannelData(0));
    this.src.connect(this.node);
    this.node.connect(ac.destination); // ScriptProcessor only runs when connected; it outputs silence
    this.sampleRate = ac.sampleRate;
  }
  _process(input) {
    if (!this.dsp) return;
    const pcm = this.dsp.resample(input);
    let buf = pcm;
    if (this.pending.length) {
      buf = new Float32Array(this.pending.length + pcm.length);
      buf.set(this.pending); buf.set(pcm, this.pending.length);
    }
    let o = 0;
    for (; o + FRAME <= buf.length; o += FRAME) {
      const f = buf.subarray(o, o + FRAME);
      this.onFrame && this.onFrame(this.muted ? new Float32Array(FRAME) : f);
    }
    this.pending = buf.slice(o);
  }
  stop() {
    if (this.node) { this.node.onaudioprocess = null; try { this.node.disconnect(); } catch (e) {} }
    if (this.src) { try { this.src.disconnect(); } catch (e) {} }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.node = this.src = this.stream = null;
  }
}

// Push-to-talk recorder: collects 16 kHz audio between start() and stop().
export class Recorder {
  constructor(onLevel) { this.onLevel = onLevel; this.chunks = []; this.mic = null; }
  async start() {
    this.chunks = [];
    this.mic = new Mic({
      onFrame: (f) => {
        this.chunks.push(new Float32Array(f));
        if (this.onLevel) {
          let s = 0; for (let i = 0; i < f.length; i++) s += f[i] * f[i];
          this.onLevel(Math.sqrt(s / f.length));
        }
      }
    });
    await this.mic.start();
  }
  stop() {
    if (this.mic) this.mic.stop();
    this.mic = null;
    const n = this.chunks.reduce((a, c) => a + c.length, 0), out = new Float32Array(n);
    let o = 0; for (const c of this.chunks) { out.set(c, o); o += c.length; }
    this.chunks = [];
    return out;
  }
}

// ---------------------------------------------------------------- speech output

export const voiceFor = (text, lang) => {
  if (lang === 'el' || lang === 'en') return lang === 'el' ? 'el' : 'en/en-us';
  return /[Ͱ-Ͽἀ-῿]/.test(text) ? 'el' : 'en/en-us';
};

// Splits text into speakable sentences; strips markup and URLs.
export function sentences(text) {
  const clean = String(text || '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/[*_`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!clean) return [];
  const parts = clean.match(/[^.!?;·…\n]+[.!?;·…]*/g) || [clean];
  const out = [];
  for (const p of parts) {
    const s = p.trim();
    if (!s) continue;
    if (out.length && (s.length < 12 || out[out.length - 1].length < 12)) out[out.length - 1] += ' ' + s;
    else out.push(s);
  }
  return out;
}

export class Speaker {
  constructor(getSettings) {
    this.getSettings = getSettings;
    this.queue = [];          // promises of decoded AudioBuffers, in order
    this.playing = null;
    this.gen = 0;             // bumps on stop(), so stale synth results are dropped
    this.onState = null;      // (speaking: bool) => void
    this.onLevel = null;
    this.firstAudioMs = null;
  }
  get speaking() { return !!this.playing || this.queue.length > 0; }
  // Queue text; returns a promise that resolves when everything queued so far has played.
  say(text, lang) {
    const st = this.getSettings();
    const list = sentences(text);
    const gen = this.gen;
    for (const s of list) {
      const t0 = performance.now();
      const p = models.tts.synth(s, voiceFor(s, lang), { speed: st.voiceSpeed, pitch: st.voicePitch })
        .then(async (r) => {
          if (gen !== this.gen) return null;
          const buf = await audioContext().decodeAudioData(r.wav);
          return { buf, synthMs: performance.now() - t0 };
        })
        .catch((e) => { console.warn('TTS failed:', e); return null; });
      this.queue.push(p);
    }
    this._kick();
    return this.whenIdle();
  }
  whenIdle() {
    if (!this.speaking) return Promise.resolve();
    return new Promise((res) => { (this._idle || (this._idle = [])).push(res); });
  }
  async _kick() {
    if (this.playing || !this.queue.length) {
      if (!this.playing && !this.queue.length) this._settle();
      return;
    }
    const gen = this.gen;
    this.playing = this.queue.shift();
    if (this.onState) this.onState(true);
    const item = await this.playing;
    if (gen !== this.gen) return;
    if (!item) { this.playing = null; this._kick(); return; }
    const ac = audioContext();
    const src = ac.createBufferSource();
    src.buffer = item.buf;
    const analyser = ac.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser); analyser.connect(ac.destination);
    this.src = src;
    if (this.firstAudioMs === null) this.firstAudioMs = item.synthMs;
    const data = new Float32Array(analyser.fftSize);
    const meter = () => {
      if (this.src !== src) return;
      analyser.getFloatTimeDomainData(data);
      let s = 0; for (let i = 0; i < data.length; i++) s += data[i] * data[i];
      if (this.onLevel) this.onLevel(Math.sqrt(s / data.length));
      requestAnimationFrame(meter);
    };
    src.onended = () => {
      if (this.src === src) this.src = null;
      if (gen !== this.gen) return;
      this.playing = null;
      if (this.onLevel) this.onLevel(0);
      this._kick();
    };
    src.start();
    requestAnimationFrame(meter);
  }
  _settle() {
    if (this.onState) this.onState(false);
    const w = this._idle || []; this._idle = [];
    w.forEach((f) => f());
  }
  stop() {
    this.gen++;
    this.queue = [];
    if (this.src) { try { this.src.stop(); } catch (e) {} this.src = null; }
    this.playing = null;
    if (this.onLevel) this.onLevel(0);
    this._settle();
  }
}
