// MiniLM embeddings with the JS kernel and with the WebAssembly SIMD kernel (separate processes):
// checks bit-identical output and reports the speed-up.   node tools/test_minilm_wasm.js
'use strict';
var fs = require('fs'), path = require('path'), cp = require('child_process');
var root = path.join(__dirname, '..'), dir = path.join(root, 'models/minilm/model');
var mode = process.argv[2];
var TEXTS = ['Τι είναι η Ακρόπολη;', 'Who was Alexander the Great?', 'βάλε σκοτεινό θέμα', 'Remember that my keys are in the drawer',
  'Η φωτοσύνθεση είναι η διαδικασία με την οποία τα φυτά μετατρέπουν το φως σε χημική ενέργεια.'];

if (!mode) {
  var r = {};
  ['js', 'wasm'].forEach(function (m) {
    r[m] = JSON.parse(cp.execFileSync(process.execPath, [__filename, m], { encoding: 'utf8', maxBuffer: 1 << 26 }));
    console.log(m.padEnd(5), r[m].ms + ' ms for ' + TEXTS.length + ' texts x 3');
  });
  var same = r.js.hash === r.wasm.hash;
  console.log('bit-identical:', same, ' speed-up: x' + (r.js.ms / r.wasm.ms).toFixed(2));
  process.exit(same ? 0 : 1);
}

var vm = require('vm');
global.self = global;
vm.runInThisContext(fs.readFileSync(path.join(root, 'models/minilm/minilm-nfkc-data.js'), 'utf8'));
var Shard = require(path.join(root, 'models/minilm/minilm-shard.js'));
if (mode === 'wasm') require(path.join(root, 'models/minilm/minilm-wasm.js')).install(Shard, fs.readFileSync(path.join(root, 'wasm/zagreus-simd.wasm')));
var Engine = require(path.join(root, 'models/minilm/minilm-engine.js'));
var Tok = require(path.join(root, 'models/minilm/minilm-tokenizer.js'));
var manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8'));
var sb = fs.readFileSync(path.join(dir, 'scores.bin'));
var tok = new Tok.Tokenizer(fs.readFileSync(path.join(dir, 'vocab.txt'), 'utf8').split('\n'), new Float64Array(sb.buffer.slice(sb.byteOffset, sb.byteOffset + sb.length)), manifest.tokenizer);
Engine.load({ manifest: manifest, threads: 1, kernel: 'int', fetch: function (name, p, done) {
  var b = fs.readFileSync(path.join(dir, name)); done(null, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
} }, null, function (err, model) {
  if (err) throw err;
  var opts = { pad: manifest.tokenizer.pad, pooling: 'mean', normalize: true }, all = [], i = 0, t0, rounds = 0;
  function next() {
    if (i === TEXTS.length) {
      if (++rounds < 3) { i = 0; all = []; return next(); }
      var u = new Uint8Array(Float32Array.from([].concat.apply([], all)).buffer), h = 0;
      for (var j = 0; j < u.length; j++) h = (Math.imul(h, 31) + u[j]) | 0;
      process.stdout.write(JSON.stringify({ ms: Date.now() - t0, hash: h }));
      process.exit(0);
    }
    model.embed([tok.encode(TEXTS[i])], opts, function (e, r) { if (e) throw e; all.push(Array.from(r.data)); i++; next(); });
  }
  // warm-up
  model.embed([tok.encode('warm up')], opts, function () { t0 = Date.now(); next(); });
});
