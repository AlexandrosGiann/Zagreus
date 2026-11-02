// Checks the vendored Orasis-2 files: the encoder on JS, WASM and SIMD WASM against the upstream golden vector,
// and the text map and word bank used by models/orasis/orasis-worker.js.
// Usage: node tools/test_orasis.js
'use strict';
const fs = require('fs');
const path = require('path');
const dir = path.join(__dirname, '..', 'models', 'orasis');
const Orasis = require(path.join(dir, 'orasis.js'));
const OrasisWasm = require(path.join(dir, 'orasis-wasm.js'));
const Head = require(path.join(dir, 'head.js'));

const read = (f) => fs.readFileSync(path.join(dir, f));
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
let failed = 0;

function check(name, z, exp, ms) {
  let cos = 0, max = 0;
  for (let i = 0; i < z.length; i++) { cos += z[i] * exp[i]; max = Math.max(max, Math.abs(z[i] - exp[i])); }
  const ok = z.length === exp.length && cos > 0.99999 && max < 1e-3;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name} cos=${cos.toFixed(8)} maxAbs=${max.toExponential(2)} ${ms} ms`);
}

(async () => {
  const model = read('encoder.ors');
  const img = new Uint8Array(read('golden.u8'));
  const exp = new Float32Array(ab(read('golden.f32')));
  const js = new Orasis.Model(ab(model));
  js.encode(img);
  let t = Date.now();
  const z = js.encode(img);
  check('encoder js', z, exp, Date.now() - t);
  for (const f of ['orasis.wasm', 'orasis-simd.wasm']) {
    const w = await OrasisWasm.create(read(f), model);
    w.encode(img);
    t = Date.now();
    check(`encoder ${f}`, w.encode(img), exp, Date.now() - t);
  }

  const map = Head.load(ab(read('textmap.orh')));
  const bank = Orasis.decodeF16(ab(read('bank.f16')));
  const labels = JSON.parse(read('labels.json'));
  const shapes = map.dim === z.length && map.classes === 512 && bank.length === labels.length * 512;
  if (!shapes) failed++;
  console.log(`${shapes ? 'PASS' : 'FAIL'} text map ${map.classes}x${map.dim}, bank ${labels.length} words`);

  const s = Head.scores(map, z);
  const n = Math.hypot(...s);
  const top = Orasis.topK(Float32Array.from(s, (v) => v / n), bank, 3);
  const ok = top.length === 3 && top.every((r) => r.score > -1 && r.score < 1 && labels[r.index]);
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} golden image words: ${top.map((r) => `${labels[r.index]} ${r.score.toFixed(3)}`).join(', ')}`);
  process.exit(failed ? 1 : 0);
})();
