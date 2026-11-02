/*
 * Zagreus vision worker: Orasis-2 image encoder (github.com/AlexandrosGiann/zagreus-orasis).
 *
 * The encoder gives a 1280-value unit vector. textmap.orh maps it into the 512-dimensional text space of
 * OpenCLIP ViT-B-16, where it is compared with the 1,253 words of bank.f16 by cosine similarity.
 * Runs on WebAssembly (SIMD build when the engine supports it) and falls back to the plain JS runtime.
 *
 * Messages in:  { type: 'load', wasm }  { type: 'run', rgb, w, h, k }
 * Messages out: { type: 'load-progress', percent, phase }  { type: 'ready', labels, kernel, dim }
 *               { type: 'result', top: [{ label, score }], inferenceMs, kernel }  { type: 'error', text }
 */
importScripts('orasis.js', 'head.js', 'orasis-wasm.js');

var encoder = null, kernel = 'js', textMap = null, bank = null, labels = null;

function get(url, type) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
    return type === 'json' ? r.json() : r.arrayBuffer();
  });
}

function progress(percent, phase) { self.postMessage({ type: 'load-progress', percent: percent, phase: phase }); }

async function load(useWasm) {
  progress(0, 'encoder');
  var model = await get('encoder.ors');
  progress(60, 'text bank');
  var parts = await Promise.all([get('textmap.orh'), get('bank.f16'), get('labels.json', 'json')]);
  textMap = OrasisHead.load(parts[0]);
  bank = Orasis.decodeF16(parts[1]);
  labels = parts[2];
  progress(85, 'runtime');
  encoder = null;
  if (useWasm && typeof WebAssembly === 'object') {
    var simd = OrasisWasm.simdSupported();
    try {
      encoder = await OrasisWasm.create(await get(simd ? 'orasis-simd.wasm' : 'orasis.wasm'), model);
      kernel = simd ? 'wasm-simd' : 'wasm';
    } catch (e) { encoder = null; }
  }
  if (!encoder) { encoder = new Orasis.Model(model); kernel = 'js'; }
  if (textMap.dim !== encoder.dim || bank.length !== labels.length * textMap.classes) throw new Error('model files do not match');
  progress(100, 'ready');
  self.postMessage({ type: 'ready', labels: labels.length, kernel: kernel, dim: encoder.dim });
}

// Unit image vector -> text space -> cosine against every word of the bank.
function words(z, k) {
  var s = OrasisHead.scores(textMap, z), n = 0, i;
  for (i = 0; i < s.length; i++) n += s[i] * s[i];
  n = Math.sqrt(n) || 1;
  var v = new Float32Array(s.length);
  for (i = 0; i < s.length; i++) v[i] = s[i] / n;
  return Orasis.topK(v, bank, k).map(function (r) { return { label: labels[r.index], score: r.score }; });
}

self.onmessage = async function (e) {
  var d = e.data;
  try {
    if (d.type === 'load') await load(d.wasm !== false);
    else if (d.type === 'run') {
      var t0 = performance.now();
      var input = Orasis.preprocess(new Uint8Array(d.rgb), d.w, d.h, encoder.size);
      var z = encoder.encode(input);
      var top = words(z, d.k || 10);
      self.postMessage({ type: 'result', top: top, inferenceMs: performance.now() - t0, kernel: kernel });
    }
  } catch (err) {
    self.postMessage({ type: 'error', text: String(err && err.message || err) });
  }
};
