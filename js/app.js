// Zagreus — page controller: chat UI, settings, model panel, dictation, camera, timers, calls.

import { models, onModelChange, setModelWasm } from './models.js';
import { Brain, listNotes, clearNotes, humanDuration, REPLIES } from './brain.js';
import { Speaker, Recorder, audioContext } from './audio.js';
import { CallSession } from './call.js';
import { UI } from './i18n.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---------------------------------------------------------------- settings

const DEFAULTS = {
  uiLang: 'el', theme: 'dark', voice: true, autoLoad: true, useNeedle: true, bargeIn: true, wasm: true,
  voiceSpeed: 165, voicePitch: 45, voiceLang: 'auto', sttLang: 'en', callBeams: 1, vadThreshold: 10, vadEndMs: 700
};
const SKEY = 'zagreus_settings_v1';
let S = Object.assign({}, DEFAULTS);
try { Object.assign(S, JSON.parse(localStorage.getItem(SKEY) || '{}')); } catch (e) {}
const save = () => { try { localStorage.setItem(SKEY, JSON.stringify(S)); } catch (e) {} };
const t = () => UI[S.uiLang] || UI.el;

// ---------------------------------------------------------------- core objects

const speaker = new Speaker(() => S);
const brain = new Brain({ uiLang: () => S.uiLang, settings: () => S });
let call = null;

// ---------------------------------------------------------------- i18n + theme

function applyLang() {
  document.documentElement.lang = S.uiLang;
  const T = t();
  document.querySelectorAll('[data-i18n]').forEach((el) => { const v = T[el.dataset.i18n]; if (typeof v === 'string') el.textContent = v; });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => { const v = T[el.dataset.i18nTitle]; if (v) { el.title = v; el.setAttribute('aria-label', v); } });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => { el.placeholder = T[el.dataset.i18nPlaceholder]; });
  $('langBtn').textContent = S.uiLang === 'el' ? 'EN' : 'ΕΛ';
  const chips = $('suggestions');
  chips.innerHTML = '';
  T.suggestions.forEach((s) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'chip'; b.textContent = s;
    b.onclick = () => submit(s);
    chips.appendChild(b);
  });
  renderModels();
  renderNotes();
}

function applyTheme() {
  document.documentElement.dataset.theme = S.theme;
  document.querySelector('meta[name="theme-color"]').content = S.theme === 'dark' ? '#0f0c15' : '#f7f2ea';
}

// ---------------------------------------------------------------- toast & status

let toastTimer = null;
function toast(msg, ms = 3200) {
  const el = $('toast');
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}
function status(msg) { const el = $('statusLine'); el.textContent = msg || ''; el.hidden = !msg; }

// ---------------------------------------------------------------- model panel

const ORDER = ['minilm', 'tts', 'whistle', 'needle', 'smollm', 'clip'];
const CORE = ['minilm', 'tts', 'whistle', 'needle'];

function renderModels() {
  const T = t();
  const list = $('modelList');
  list.innerHTML = '';
  let ready = 0;
  const dots = $('modelDots');
  dots.innerHTML = '';
  for (const k of ORDER) {
    const m = models[k];
    if (m.ready) ready++;
    const dot = document.createElement('i');
    dot.className = m.state; dot.title = m.label;
    dots.appendChild(dot);
    const card = document.createElement('div');
    card.className = 'model ' + m.state;
    const btn = m.state === 'idle' || m.state === 'error'
      ? `<button class="btn small ghost" type="button" data-load="${k}">${esc(T.load)} · ${m.sizeMB} MB</button>`
      : `<span class="badge ${m.ready ? 'ivy' : 'gold'}">${esc(m.ready ? T.loaded : T.loading)}</span>`;
    card.innerHTML = `
      <div class="model-top"><div><div class="model-name">${esc(m.label)}</div><div class="model-role">${esc(T.roles[k])}</div></div>${btn}</div>
      ${m.state === 'loading' ? `<div class="progress"><i style="width:${Math.round(m.progress * 100)}%"></i></div>` : ''}
      <div class="model-detail">${esc(m.state === 'error' ? m.error : m.detail || (m.state === 'idle' ? T.idle : ''))}</div>`;
    list.appendChild(card);
  }
  $('modelsCount').textContent = T.modelsReady(ready, ORDER.length);
  list.querySelectorAll('[data-load]').forEach((b) => { b.onclick = () => loadModel(b.dataset.load); });
}

function loadModel(k) {
  const p = models[k].load();
  if (k === 'minilm') p.then(() => brain.prepareAnchors()).catch(() => {});
  return p.catch((e) => { console.warn(k, e); });
}

let renderQueued = false;
onModelChange(() => {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; renderModels(); });
});

// ---------------------------------------------------------------- chat rendering

const messagesEl = $('messages');
const transcript = []; // for export

function scrollDown() { const c = $('chat'); c.scrollTop = c.scrollHeight; }

function addUser(text, imgUrl) {
  $('hero').hidden = true;
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = (imgUrl ? `<img class="shot" src="${imgUrl}" alt="">` : '') + (text ? `<div class="bubble">${esc(text)}</div>` : '');
  messagesEl.appendChild(el);
  transcript.push({ role: 'user', text: text || '[image]', ts: Date.now() });
  scrollDown();
  return el;
}

function addBotPending() {
  $('hero').hidden = true;
  const el = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = '<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>';
  messagesEl.appendChild(el);
  scrollDown();
  return el;
}

function fillBot(el, out) {
  const T = t();
  let html = `<div class="bubble">${esc(out.text)}</div>`;
  if (out.reasoning) html += `<div class="think">⟡ ${esc(out.reasoning)}</div>`;
  for (const c of out.cards || []) html += renderCard(c);
  const meta = [];
  if (out.source) meta.push(`<span class="badge gold">${esc(out.source)}</span>`);
  if (out.via && out.via !== out.source && out.intent) meta.push(`<span class="badge">${esc(out.intent)} · ${esc(out.via)}${out.score ? ' ' + out.score.toFixed(2) : ''}</span>`);
  if (out.needleMs) meta.push(`<span class="badge ivy">Needle ${(out.needleMs / 1000).toFixed(1)} s</span>`);
  if (out.tps) meta.push(`<span class="badge">${out.tps.toFixed(1)} tok/s</span>`);
  if (out.ms != null) meta.push(`<span>${out.ms < 1000 ? out.ms + ' ms' : (out.ms / 1000).toFixed(1) + ' s'}</span>`);
  meta.push(`<button type="button" data-replay>▶ ${esc(T.replay)}</button>`);
  meta.push(`<button type="button" data-copy>${esc(T.copy)}</button>`);
  html += `<div class="meta">${meta.join('')}</div>`;
  el.innerHTML = html;
  el.querySelector('[data-replay]').onclick = () => { speaker.stop(); speaker.say(out.text, langOf(out)); };
  el.querySelector('[data-copy]').onclick = () => navigator.clipboard && navigator.clipboard.writeText(out.text);
  transcript.push({ role: 'zagreus', text: out.text, source: out.source, ts: Date.now() });
  scrollDown();
}

function renderCard(c) {
  if (c.type === 'wiki') {
    return `<div class="card wiki"><div><h4>${esc(c.title)}</h4>${c.extract ? `<p>${esc(c.extract)}</p>` : ''}${c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">Wikipedia →</a>` : ''}</div>${c.thumb ? `<img src="${esc(c.thumb)}" alt="" loading="lazy">` : ''}</div>`;
  }
  if (c.type === 'weather') {
    const days = c.days.map((d) => {
      const name = new Date(d.date + 'T12:00').toLocaleDateString(S.uiLang === 'el' ? 'el-GR' : 'en-US', { weekday: 'short' });
      return `<span>${esc(name)} ${d.min}°/${d.max}° · ☂ ${d.rain == null ? '–' : d.rain + '%'}</span>`;
    }).join('');
    return `<div class="card"><h4>${esc(c.place)}</h4><div class="wx-now"><span class="wx-temp">${c.temp}°</span><span>${esc(c.desc)} · ${c.wind} km/h</span></div><div class="wx-days">${days}</div></div>`;
  }
  if (c.type === 'labels') {
    const rows = c.top.map((x) => `<div class="label-row"><span>${esc(x.label)}</span><span class="bar"><i style="width:${Math.round(x.p * 100)}%"></i></span><span class="muted">${Math.round(x.p * 100)}%</span></div>`).join('');
    return `<div class="card"><div class="labels">${rows}</div></div>`;
  }
  return '';
}

const langOf = (out) => (S.voiceLang === 'auto' ? out.lang : S.voiceLang);

// ---------------------------------------------------------------- sending

let busy = false;

async function submit(text) {
  text = String(text || '').trim();
  if (!text || busy) return;
  busy = true;
  $('sendBtn').disabled = true;
  $('input').value = '';
  autosize();
  speaker.stop();
  addUser(text);
  const el = addBotPending();
  const T = t();
  const out = await brain.reply(text, {
    onStatus: (k) => status(T.status[k] || ''),
    onPartial: (p) => {
      status('');
      el.innerHTML = `<div class="bubble">${esc(p)}</div>`;
      scrollDown();
    }
  });
  status('');
  fillBot(el, out);
  if (out.action) runAction(out.action);
  if (S.voice && out.text && !(out.action && ['stop_speaking', 'start_call'].includes(out.action.name))) speaker.say(out.text, langOf(out));
  busy = false;
  $('sendBtn').disabled = false;
}

// ---------------------------------------------------------------- actions (from the brain or Needle)

function runAction(a) {
  const args = a.args || {};
  switch (a.name) {
    case 'set_theme': S.theme = args.mode === 'light' ? 'light' : 'dark'; save(); applyTheme(); break;
    case 'set_voice': S.voice = !!args.on; save(); syncSettings(); if (!S.voice) speaker.stop(); break;
    case 'stop_speaking': speaker.stop(); break;
    case 'set_language': S.uiLang = args.lang === 'en' ? 'en' : 'el'; save(); applyLang(); break;
    case 'start_call': if (!call) setTimeout(startCall, 300); break;
    case 'end_call': endCall(); break;
    case 'open_camera': openCamera(); break;
    case 'clear_chat': clearChat(); break;
    case 'set_timer': addTimer(args.seconds, args.label, args.done); break;
  }
}

function clearChat() {
  messagesEl.innerHTML = '';
  transcript.length = 0;
  brain.resetConversation();
  $('hero').hidden = false;
}

// ---------------------------------------------------------------- timers

const timers = [];
function addTimer(seconds, label, doneText) {
  const tm = { id: Date.now(), end: Date.now() + seconds * 1000, seconds, label, doneText };
  tm.handle = setTimeout(() => fireTimer(tm), seconds * 1000);
  timers.push(tm);
  renderTimers();
}
function fireTimer(tm) {
  const i = timers.indexOf(tm); if (i >= 0) timers.splice(i, 1);
  renderTimers();
  beep();
  const msg = tm.doneText + (tm.label ? ' ' + tm.label : '');
  toast('⏰ ' + msg, 8000);
  const el = addBotPending();
  fillBot(el, { text: msg, source: '⏰', lang: S.uiLang });
  if (S.voice) speaker.say(msg);
  if ('Notification' in window && Notification.permission === 'granted') new Notification('Ζαγρεύς', { body: msg, icon: 'assets/zagreus.svg' });
}
function renderTimers() {
  $('timersWrap').hidden = !timers.length;
  $('timersList').innerHTML = timers.map((tm) => {
    const left = Math.max(0, Math.round((tm.end - Date.now()) / 1000));
    return `<li><span>${esc(tm.label || humanDuration(tm.seconds, S.uiLang))}</span><time>${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}</time></li>`;
  }).join('');
}
setInterval(() => { if (timers.length && !$('drawer').hidden) renderTimers(); }, 1000);
function beep() {
  try {
    const ac = audioContext(), o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = 880; o.connect(g); g.connect(ac.destination);
    g.gain.setValueAtTime(0.0001, ac.currentTime);
    [0, 0.3, 0.6].forEach((d) => { g.gain.exponentialRampToValueAtTime(0.25, ac.currentTime + d + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + d + 0.2); });
    o.start(); o.stop(ac.currentTime + 0.9);
  } catch (e) {}
}

// ---------------------------------------------------------------- notes

function renderNotes() {
  const T = t(), notes = listNotes();
  $('notesList').innerHTML = notes.length
    ? notes.slice().reverse().map((n) => `<li><span>${esc(n.text)}</span><time>${new Date(n.ts).toLocaleDateString(S.uiLang === 'el' ? 'el-GR' : 'en-US')}</time></li>`).join('')
    : `<li class="muted">${esc(T.notesEmpty)}</li>`;
}

// ---------------------------------------------------------------- dictation (Whistle)

let recorder = null;
async function toggleMic() {
  const T = t(), btn = $('micBtn');
  if (location.protocol === 'file:') return toast(T.needsHttp);
  if (recorder) {
    const pcm = recorder.stop();
    recorder = null;
    btn.classList.remove('rec');
    btn.title = T.mic;
    if (pcm.length < 16000 * 0.3) return status('');
    status(T.transcribing);
    try {
      const r = await models.whistle.transcribe(pcm, {
        beams: 5, language: S.sttLang === 'auto' ? null : S.sttLang,
        onPartial: (p) => { $('input').value = p; autosize(); }
      });
      status('');
      const text = (r.text || '').trim();
      if (!text || r.silent) return toast(T.sttEmpty);
      $('input').value = text;
      autosize();
      submit(text);
    } catch (e) { status(''); toast(String(e.message || e)); }
    return;
  }
  try {
    speaker.stop();
    models.whistle.load().catch(() => {});
    recorder = new Recorder();
    await recorder.start();
    btn.classList.add('rec');
    btn.title = T.micStop;
    status(T.recording);
  } catch (e) {
    recorder = null;
    toast(e && e.name === 'NotAllowedError' ? T.micDenied : String(e.message || e));
  }
}

// ---------------------------------------------------------------- camera + vision (MobileCLIP)

let camStream = null;
async function openCamera() {
  $('camView').hidden = false;
  $('camNone').hidden = true;
  models.clip.load().catch(() => {});
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 720 }, height: { ideal: 720 } } });
    const v = $('camVideo'); v.srcObject = camStream; await v.play();
    $('camShoot').disabled = false;
  } catch (e) {
    $('camNone').hidden = false;
    $('camShoot').disabled = true;
  }
}
function closeCamera() {
  if (camStream) camStream.getTracks().forEach((tr) => tr.stop());
  camStream = null;
  $('camVideo').srcObject = null;
  $('camView').hidden = true;
}
function snapshot() {
  const v = $('camVideo');
  const c = document.createElement('canvas');
  const side = Math.min(v.videoWidth, v.videoHeight);
  c.width = c.height = 512;
  c.getContext('2d').drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, 512, 512);
  return c;
}

async function describeImage(imgSource, url) {
  const T = t(), lang = S.uiLang, R = REPLIES[lang];
  addUser('', url);
  const el = addBotPending();
  status(T.looking);
  const t0 = performance.now();
  try {
    const res = await models.clip.classify(imgSource, (p) => status(`${T.looking} ${Math.round(p * 100)}%`));
    status('');
    // turn the cosine scores of the top labels into a softmax distribution (CLIP logit scale 100)
    const top = res.top.slice(0, 5);
    const ex = top.map((x) => Math.exp(100 * (x.score - top[0].score)));
    const sum = ex.reduce((a, b) => a + b, 0);
    const dist = top.map((x, i) => ({ label: x.label, p: ex[i] / sum }));
    let text = R.imageSee(dist[0].label);
    const others = dist.slice(1, 3).filter((d) => d.p > 0.08).map((d) => d.label);
    if (others.length) text += ' ' + R.imageMaybe(others.join(', '));
    const out = { text, lang, source: 'MobileCLIP-S0', cards: [{ type: 'labels', top: dist }], ms: Math.round(performance.now() - t0) };
    fillBot(el, out);
    if (S.voice) speaker.say(text, langOf(out));
  } catch (e) {
    status('');
    fillBot(el, { text: String(e.message || e), source: 'MobileCLIP-S0', lang });
  }
}

// ---------------------------------------------------------------- call

function startCall() {
  const T = t();
  if (location.protocol === 'file:') return toast(T.needsHttp);
  if (call) return;
  speaker.stop();
  audioContext(); // unlock audio inside the click
  $('callView').hidden = false;
  $('callLog').innerHTML = '';
  $('callMetrics').innerHTML = '';
  $('muteBtn').classList.remove('on');
  const orb = $('callOrb');
  const metrics = {};
  let started = Date.now();
  const timerEl = $('callTimer');
  const tick = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    timerEl.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }, 500);
  speaker.onLevel = (l) => { if (call && call.state === 'speaking') orb.style.setProperty('--lvl', Math.min(1, l * 6).toFixed(3)); };
  call = new CallSession({
    brain, speaker,
    settings: () => S,
    onState: (s) => {
      $('callState').textContent = T.callStates[s] || s;
      orb.className = 'orb ' + (s === 'thinking' || s === 'transcribing' ? 'thinking' : s === 'listening' || s === 'hearing' ? 'listening' : '');
      if (s === 'listening') started = started || Date.now();
    },
    onLevel: (l) => { if (call && call.state !== 'speaking') orb.style.setProperty('--lvl', l.toFixed(3)); },
    onTranscript: (role, text, out) => {
      const d = document.createElement('div');
      d.className = role === 'user' ? 'u' : 'b';
      d.textContent = text;
      $('callLog').appendChild(d);
      $('callLog').scrollTop = $('callLog').scrollHeight;
      if (role === 'user') addUser(text); else { const el = addBotPending(); fillBot(el, Object.assign({ lang: S.uiLang }, out, { text })); }
    },
    onMetrics: (m) => { Object.assign(metrics, m); renderMetrics(metrics); },
    runAction: (a) => runAction(a)
  });
  call._tick = tick;
  call.start().catch((e) => {
    toast(e && e.name === 'NotAllowedError' ? T.micDenied : String(e.message || e));
    endCall();
  });
}

function renderMetrics(m) {
  const b = [];
  if (m.dsp) b.push(`VAD/resampler: ${m.dsp === 'wasm-simd' ? 'WASM SIMD' : m.dsp === 'wasm' ? 'WASM' : 'JS'}`);
  if (m.stt) b.push(`Whistle: ${m.stt === 'wasm-simd' ? 'WASM SIMD' : 'JS'}`);
  if (m.sttMs != null) b.push(`STT ${m.sttMs} ms / ${m.audioS} s (RTF ${m.rtf})`);
  if (m.thinkMs != null) b.push(`${m.source || ''} ${m.thinkMs} ms`);
  if (m.ttsMs != null) b.push(`TTS ${m.ttsMs} ms`);
  $('callMetrics').innerHTML = b.map((x) => `<span class="badge">${esc(x)}</span>`).join('');
}

function endCall() {
  if (!call) { $('callView').hidden = true; return; }
  clearInterval(call._tick);
  call.stop();
  call = null;
  speaker.onLevel = null;
  $('callView').hidden = true;
}

// ---------------------------------------------------------------- settings drawer

function syncSettings() {
  $('sVoice').checked = S.voice;
  $('sAutoLoad').checked = S.autoLoad;
  $('sNeedle').checked = S.useNeedle;
  $('sBarge').checked = S.bargeIn;
  $('sWasm').checked = S.wasm;
  $('sSpeed').value = S.voiceSpeed; $('sSpeedOut').textContent = S.voiceSpeed;
  $('sPitch').value = S.voicePitch; $('sPitchOut').textContent = S.voicePitch;
  $('sVoiceLang').value = S.voiceLang;
  $('sStt').value = S.sttLang;
}

function bindSettings() {
  const bool = (id, key, after) => { $(id).onchange = (e) => { S[key] = e.target.checked; save(); if (after) after(); }; };
  bool('sVoice', 'voice', () => { if (!S.voice) speaker.stop(); });
  bool('sAutoLoad', 'autoLoad');
  bool('sNeedle', 'useNeedle');
  bool('sBarge', 'bargeIn');
  bool('sWasm', 'wasm', () => { models.whistle.setAccel(S.wasm); setModelWasm(S.wasm); });
  $('sSpeed').oninput = (e) => { S.voiceSpeed = +e.target.value; $('sSpeedOut').textContent = S.voiceSpeed; save(); };
  $('sPitch').oninput = (e) => { S.voicePitch = +e.target.value; $('sPitchOut').textContent = S.voicePitch; save(); };
  $('sVoiceLang').onchange = (e) => { S.voiceLang = e.target.value; save(); };
  $('sStt').onchange = (e) => { S.sttLang = e.target.value; save(); };
  $('clearNotesBtn').onclick = () => { clearNotes(); renderNotes(); };
  $('clearChatBtn').onclick = () => { clearChat(); closeDrawer(); };
  $('exportBtn').onclick = exportChat;
  $('loadAllBtn').onclick = () => ORDER.forEach(loadModel);
  $('benchBtn').onclick = bench;
}

function openDrawer() { renderModels(); renderNotes(); renderTimers(); $('drawer').hidden = false; }
function closeDrawer() { $('drawer').hidden = true; }

// Transcribes the bundled 11 s English sample with the JS kernel and with the WebAssembly SIMD
// kernel (same worker, same model) and reports both times.
async function bench() {
  const T = t(), out = $('benchOut'), btn = $('benchBtn');
  btn.disabled = true; out.textContent = T.benchRun;
  try {
    await models.whistle.load();
    const buf = await (await fetch('assets/sample-en.wav')).arrayBuffer();
    const pcm = await decode16k(buf);
    const run = async (on) => {
      models.whistle.setAccel(on);
      const r = await models.whistle.transcribe(pcm, { beams: 1, language: 'en' });
      return { ms: Math.round(r.ms), text: r.text, accel: r.accel };
    };
    await run(true); // warm-up: uploads the weights into WebAssembly memory
    const js = await run(false);
    const w = await run(true);
    models.whistle.setAccel(S.wasm);
    out.textContent = w.accel ? T.benchRes(js.ms, w.ms, (js.ms / w.ms).toFixed(2)) : 'WebAssembly SIMD unavailable · JS ' + js.ms + ' ms';
    out.title = w.text;
  } catch (e) {
    out.textContent = String(e.message || e);
  }
  btn.disabled = false;
}

async function decode16k(buf) {
  const ac = new OfflineAudioContext(1, 16000, 16000);
  const audio = await ac.decodeAudioData(buf);
  if (audio.sampleRate === 16000) return audio.getChannelData(0).slice();
  const off = new OfflineAudioContext(1, Math.ceil(audio.duration * 16000), 16000);
  const src = off.createBufferSource(); src.buffer = audio; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}

function exportChat() {
  const lines = transcript.map((m) => `[${new Date(m.ts).toLocaleString()}] ${m.role === 'user' ? 'Εσύ' : 'Ζαγρεύς'}: ${m.text}`);
  const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'zagreus-' + new Date().toISOString().slice(0, 10) + '.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- composer

function autosize() {
  const ta = $('input');
  ta.style.height = 'auto';
  ta.style.height = Math.min(160, ta.scrollHeight) + 'px';
}

function bindUi() {
  $('composer').onsubmit = (e) => { e.preventDefault(); submit($('input').value); };
  $('input').oninput = autosize;
  $('input').onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit($('input').value); } };
  $('micBtn').onclick = toggleMic;
  $('callBtn').onclick = startCall;
  $('hangupBtn').onclick = endCall;
  $('muteBtn').onclick = () => {
    if (!call) return;
    const m = !call.muted; call.setMuted(m);
    $('muteBtn').classList.toggle('on', m);
    $('muteBtn').title = m ? t().unmute : t().mute;
  };
  $('imageBtn').onclick = openCamera;
  $('camCancel').onclick = closeCamera;
  $('camShoot').onclick = () => {
    const c = snapshot();
    const url = c.toDataURL('image/jpeg', 0.85);
    closeCamera();
    describeImage(c, url);
  };
  $('camFile').onchange = (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => { closeCamera(); describeImage(img, url); };
    img.src = url;
  };
  $('langBtn').onclick = () => { S.uiLang = S.uiLang === 'el' ? 'en' : 'el'; save(); applyLang(); };
  $('themeBtn').onclick = () => { S.theme = S.theme === 'dark' ? 'light' : 'dark'; save(); applyTheme(); };
  $('settingsBtn').onclick = openDrawer;
  $('modelsBtn').onclick = openDrawer;
  document.querySelectorAll('[data-close]').forEach((el) => { el.onclick = closeDrawer; });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!$('camView').hidden) closeCamera();
    else if (!$('drawer').hidden) closeDrawer();
  });
  // drag & drop an image onto the page
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f || !/^image\//.test(f.type)) return;
    const url = URL.createObjectURL(f), img = new Image();
    img.onload = () => describeImage(img, url);
    img.src = url;
  });
  // ask once for notifications when a timer is first set
  document.addEventListener('click', () => {
    if ('Notification' in window && Notification.permission === 'default' && timers.length) Notification.requestPermission();
  });
}

// ---------------------------------------------------------------- boot

applyTheme();
applyLang();
syncSettings();
bindSettings();
bindUi();
models.whistle.wantAccel = S.wasm;
setModelWasm(S.wasm);
if (location.protocol === 'file:') toast(t().needsHttp, 10000);
else if (S.autoLoad) {
  // the core models load one after another so the first answer is not starved of bandwidth
  (async () => { for (const k of CORE) await loadModel(k); })();
}

// for debugging from the console
window.zagreus = { models, brain, speaker, settings: S };
