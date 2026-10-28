// Voice call with Zagreus: hands-free turn taking.
//
//   microphone ─▶ WebAssembly resampler (→16 kHz) ─▶ WebAssembly VAD ─▶ utterance
//   utterance ─▶ Whistle (WebAssembly SIMD matmuls, greedy) ─▶ brain ─▶ meSpeak, sentence by sentence
//
// While Zagreus speaks the VAD keeps running, so talking over it (barge-in) stops the voice and
// starts a new turn. Every stage is timed; the overlay shows the numbers.

import { Mic, FRAME } from './audio.js';
import { loadDsp } from './dsp.js';
import { models } from './models.js';

const PREROLL = 15;          // frames (300 ms) kept before speech onset
const MAX_FRAMES = 50 * 14;  // 14 s per utterance
const BARGE_FRAMES = 12;     // 240 ms of loud speech while Zagreus talks

export class CallSession {
  // host: { brain, speaker, settings(), onState(state), onTranscript(role, text, meta), onLevel(0..1),
  //         onMetrics(m), runAction(action) }
  constructor(host) {
    this.host = host;
    this.state = 'idle';
    this.frames = [];
    this.preroll = [];
    this.capturing = false;
    this.barge = 0;
    this.turn = 0;
    this.muted = false;
    this.metrics = {};
  }

  setState(s) { this.state = s; this.host.onState(s); }

  async start() {
    this.setState('connecting');
    this.dsp = await loadDsp();
    this.mic = new Mic({ echoCancel: true, onFrame: (f) => this.onFrame(f) });
    await this.mic.start();
    this.host.onMetrics({ dsp: this.dsp.kind });
    // models load in parallel; the call can show "connecting" until speech-to-text is ready
    const tts = models.tts.load();
    await models.whistle.load();
    await tts.catch(() => {});
    this.host.onMetrics({ dsp: this.dsp.kind, stt: models.whistle.accel.on ? 'wasm-simd' : 'js' });
    this.resetVad();
    this.startedAt = Date.now();
    const st = this.host.settings();
    const hello = st.uiLang === 'el' ? 'Γεια σου! Σε ακούω.' : 'Hi! I am listening.';
    this.host.onTranscript('bot', hello, {});
    this.setState('speaking');
    if (st.voice) await this.host.speaker.say(hello, st.uiLang);
    if (this.state === 'speaking') this.listen();
  }

  resetVad() {
    const st = this.host.settings();
    this.dsp.vadInit(st.vadThreshold || 10, st.vadEndMs || 700, 20);
  }

  listen() {
    this.resetVad();
    this.capturing = false;
    this.frames = [];
    this.barge = 0;
    this.setState('listening');
  }

  setMuted(m) { this.muted = m; if (this.mic) this.mic.muted = m; }

  onFrame(f) {
    if (this.state === 'idle' || this.state === 'ended') return;
    const v = this.dsp.vad(f);
    const db = this.dsp.levelDb();
    this.host.onLevel(Math.max(0, Math.min(1, (db + 60) / 50)));
    const copy = new Float32Array(f);
    if (this.state === 'speaking') {
      // barge-in: sustained speech clearly above the floor while Zagreus is talking
      if (v === 1 && db > this.dsp.noiseDb() + 18) this.barge++;
      else this.barge = Math.max(0, this.barge - 2);
      this.pushPreroll(copy);
      if (this.barge >= BARGE_FRAMES && this.host.settings().bargeIn) {
        this.host.speaker.stop();
        this.turn++;
        this.resetVad();
        this.setState('listening');
        this.capturing = true;
        this.frames = this.preroll.slice();
        this.barge = 0;
      }
      return;
    }
    if (this.state !== 'listening') return;
    if (!this.capturing) {
      this.pushPreroll(copy);
      if (v === 1) {
        this.capturing = true;
        this.frames = this.preroll.slice();
        this.host.onState('hearing'); // UI only: the session itself stays in 'listening'
      }
      return;
    }
    this.frames.push(copy);
    if (v === 2 || this.frames.length >= MAX_FRAMES) {
      const pcm = concat(this.frames);
      this.frames = [];
      this.capturing = false;
      this.process(pcm);
    }
  }

  pushPreroll(f) {
    this.preroll.push(f);
    if (this.preroll.length > PREROLL) this.preroll.shift();
  }

  async process(pcm) {
    const turn = ++this.turn;
    const st = this.host.settings();
    this.setState('transcribing');
    const t0 = performance.now();
    let res;
    try {
      res = await models.whistle.transcribe(pcm, { beams: st.callBeams || 1, language: st.sttLang === 'auto' ? null : st.sttLang });
    } catch (e) {
      console.warn(e);
      if (turn === this.turn) this.listen();
      return;
    }
    if (turn !== this.turn || this.state === 'ended') return;
    const sttMs = performance.now() - t0;
    const text = (res.text || '').trim();
    this.host.onMetrics({ sttMs: Math.round(sttMs), audioS: +(pcm.length / 16000).toFixed(1), rtf: +(sttMs / 1000 / (pcm.length / 16000)).toFixed(2), stt: res.accel ? 'wasm-simd' : 'js' });
    if (!text || res.silent || /^[\s.,!?-]*$/.test(text)) { this.listen(); return; }
    this.host.onTranscript('user', text, { lang: res.language });

    this.setState('thinking');
    const t1 = performance.now();
    let spokenUpTo = 0, firstAudio = null;
    const speaker = this.host.speaker;
    speaker.firstAudioMs = null;
    const speakNew = (full, final) => {
      // speak whole sentences as soon as they are complete
      const rest = full.slice(spokenUpTo);
      const m = final ? rest.length : lastSentenceEnd(rest);
      if (m > 0 && st.voice) {
        const chunk = rest.slice(0, m).trim();
        spokenUpTo += m;
        if (chunk) {
          if (firstAudio === null) firstAudio = performance.now();
          this.setState('speaking');
          speaker.say(chunk, st.voiceLang === 'auto' ? undefined : st.voiceLang);
        }
      }
    };
    const out = await this.host.brain.reply(text, { call: true, onPartial: (t) => { if (turn === this.turn) speakNew(t, false); } });
    if (turn !== this.turn || this.state === 'ended') return;
    const thinkMs = performance.now() - t1;
    this.host.onTranscript('bot', out.text, out);
    this.host.onMetrics({ thinkMs: Math.round(thinkMs), source: out.source });
    if (out.action) {
      if (out.action.name === 'end_call') {
        if (st.voice) { speakNew(out.text, true); await speaker.whenIdle(); }
        this.host.runAction(out.action);
        return;
      }
      if (out.action.name !== 'start_call') this.host.runAction(out.action);
    }
    speakNew(out.text, true);
    if (speaker.speaking) {
      this.setState('speaking');
      await speaker.whenIdle();
      if (speaker.firstAudioMs !== null) this.host.onMetrics({ ttsMs: Math.round(speaker.firstAudioMs) });
    }
    if (turn === this.turn && this.state !== 'ended') this.listen();
  }

  stop() {
    this.turn++;
    this.setState('ended');
    if (this.mic) this.mic.stop();
    this.mic = null;
    this.host.speaker.stop();
    models.smollm.stop();
  }
}

function concat(frames) {
  const out = new Float32Array(frames.length * FRAME);
  frames.forEach((f, i) => out.set(f, i * FRAME));
  return out;
}

// index just past the last complete sentence in s (0 if none yet)
function lastSentenceEnd(s) {
  let end = 0;
  const re = /[.!?;·…]+["')\]]?\s/g;
  let m;
  while ((m = re.exec(s))) end = m.index + m[0].length;
  return end;
}
