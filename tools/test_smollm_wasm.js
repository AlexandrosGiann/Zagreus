// Generates replies with the pure-JS SmolLM2 kernel and with the WebAssembly SIMD kernel
// (single thread, separate processes) and checks that the token ids are identical.
// node tools/test_smollm_wasm.js            (runs both and compares)
// node tools/test_smollm_wasm.js js|wasm    (one run, prints JSON)
'use strict';
var fs = require('fs'), path = require('path'), cp = require('child_process');
var root = path.join(__dirname, '..'), dir = path.join(root, 'models/smollm/model');
var mode = process.argv[2];
var PROMPTS = ['Hello! Tell me about yourself.', 'Give me an idea for a quiet evening at home.', 'What is 12 times 7?'];

if (!mode) {
  var res = {};
  ['js', 'wasm'].forEach(function (m) {
    res[m] = JSON.parse(cp.execFileSync(process.execPath, [__filename, m], { encoding: 'utf8', maxBuffer: 1 << 24 }));
    console.log(m.padEnd(5), res[m].map(function (r) { return r.ms + ' ms'; }).join(', '));
  });
  var same = res.js.every(function (r, i) { return r.ids.join(',') === res.wasm[i].ids.join(','); });
  res.wasm.forEach(function (r, i) { console.log('  ' + JSON.stringify(PROMPTS[i]) + ' -> ' + JSON.stringify(r.text.slice(0, 80))); });
  var tj = res.js.reduce(function (a, r) { return a + r.ms; }, 0), tw = res.wasm.reduce(function (a, r) { return a + r.ms; }, 0);
  console.log('token-identical:', same, ' speed-up: x' + (tj / tw).toFixed(2));
  process.exit(same ? 0 : 1);
}

global.SmolLMShard = require(path.join(root, 'models/smollm/smollm-shard.js'));
if (mode === 'wasm') {
  require(path.join(root, 'models/smollm/smollm-wasm.js')).install(global.SmolLMShard, fs.readFileSync(path.join(root, 'wasm/zagreus-simd.wasm')));
}
var Engine = require(path.join(root, 'models/smollm/smollm-engine.js'));
var Tok = require(path.join(root, 'models/smollm/smollm-tokenizer.js'));
var Gen = require(path.join(root, 'models/smollm/smollm-generate.js'));
var manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8'));
var tok = new Tok.Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')));
Engine.load({
  manifest: manifest, threads: 1,
  fetch: function (name, progress, done) {
    var b = fs.readFileSync(path.join(dir, name));
    done(null, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  }
}, null, function (err, model) {
  if (err) throw err;
  var out = [], i = 0;
  (function next() {
    if (i === PROMPTS.length) { process.stdout.write(JSON.stringify(out)); process.exit(0); }
    var g = new Gen.Generation(model, tok, [{ role: 'user', content: PROMPTS[i] }], { maxNewTokens: 40 });
    var t0 = Date.now();
    g.run(null, function (e) {
      if (e) throw e;
      out.push({ ids: g.generated, text: g.finalText(), ms: Date.now() - t0 });
      i++; next();
    });
  })();
});
