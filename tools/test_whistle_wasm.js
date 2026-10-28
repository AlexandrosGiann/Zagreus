// Checks that the WebAssembly kernel gives bit-identical Whistle output and measures the speed-up.
// node tools/test_whistle_wasm.js [wav] [simd|base]
'use strict';
var fs = require('fs'), path = require('path');
var root = path.join(__dirname, '..');
var Whistle = require(path.join(root, 'models/whistle/whistle.js'));
var WW = require(path.join(root, 'models/whistle/whistle-wasm.js'));
var wav = process.argv[2] || path.join(root, 'assets/sample-en.wav');
var variant = process.argv[3] || 'simd';

function readWav(file) {
  var b = fs.readFileSync(file), off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    var id = b.toString('ascii', off, off + 4), size = b.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = { ch: b.readUInt16LE(off + 10), rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === 'data') data = b.subarray(off + 8, off + 8 + size);
    off += 8 + size + (size & 1);
  }
  if (!fmt || fmt.bits !== 16 || fmt.rate !== 16000) throw new Error('need 16 kHz 16-bit wav');
  var n = data.length / 2 / fmt.ch, pcm = new Float32Array(n);
  for (var i = 0; i < n; i++) pcm[i] = data.readInt16LE(i * 2 * fmt.ch) / 32768;
  return pcm;
}

var pcm = readWav(wav);
var cact = fs.readFileSync(path.join(root, 'models/whistle/whistle.cact'));
var buf = cact.buffer.slice(cact.byteOffset, cact.byteOffset + cact.length);

function run(label, opts) {
  var model = new Whistle.Model(buf.slice(0));
  model.transcribe(pcm.subarray(0, 16000), opts); // warm-up (uploads weights to wasm)
  var t0 = Date.now(), r = model.transcribe(pcm, opts), ms = Date.now() - t0;
  var e = model.embed(pcm);
  console.log(label.padEnd(10), String(ms).padStart(6) + ' ms', JSON.stringify(r.text));
  return { text: r.text, emb: new Uint32Array(new Float32Array(e.data).buffer), ms: ms };
}

var ok = true;
[{ beams: 1 }, { beams: 5 }].forEach(function (opts) {
  console.log('-- beams=' + opts.beams);
  WW.uninstall(Whistle);
  var a = run('js', opts);
  var info = WW.install(Whistle, fs.readFileSync(path.join(root, 'wasm/zagreus-' + variant + '.wasm')));
  var b = run('wasm' + (info.simd ? '+simd' : ''), opts);
  var same = a.text === b.text && a.emb.length === b.emb.length;
  for (var i = 0; same && i < a.emb.length; i++) if (a.emb[i] !== b.emb[i]) same = false;
  console.log('bit-identical:', same, ' speed-up: x' + (a.ms / b.ms).toFixed(2));
  ok = ok && same;
});
process.exit(ok ? 0 : 1);
