// Runs the Needle 2 router on commands per Zagreus tool profile.
// node tools/test_needle.js
'use strict';
var fs = require('fs'), path = require('path'), vm = require('vm');
var root = path.join(__dirname, '..');
['cact-full.js', 'tokenizer-full.js', 'engine-full.js', 'needle-router.js'].forEach(function (f) {
  vm.runInThisContext(fs.readFileSync(path.join(root, 'models/needle', f), 'utf8'), { filename: f });
});
var spec = require(path.join(root, 'js/needle-tools.js'));
var b = fs.readFileSync(path.join(root, 'models/needle/needle2.cact'));
var model = new NeedleCact.CactModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
var tok = new NeedleTokenizer.Tokenizer(model), eng = new NeedleFull.Engine(model);
var cases = {
  display: [['Turn on dark mode', 'theme_dark'], ['Switch to light mode', 'theme_light'], ['It is too bright', 'theme_dark'], ['Make the screen white again', 'theme_light'], ['Dark theme please', 'theme_dark']],
  voice: [['Mute your voice', 'voice_off'], ['Stop talking', 'stop_speaking'], ['Read your answers out loud', 'voice_on'], ['Speak Greek please', 'lang_el'], ['Talk in English', 'lang_en'], ['Be quiet', 'stop_speaking'], ['Turn the voice back on', 'voice_on']],
  call: [['Call me', 'call_start'], ['Hang up', 'call_end'], ['Let us talk on a call', 'call_start'], ['Bye, end the call', 'call_end']]
};
var ok = 0, n = 0;
Object.keys(cases).forEach(function (p) {
  var r = new NeedleRouter(eng, tok);
  r.setTools(spec.profiles[p].tools, spec.system, spec.profiles[p].candidates);
  var tw = Date.now(), pre = r.warm();
  console.log('== ' + p + ': prefix ' + pre + ' tokens, warm ' + (Date.now() - tw) + ' ms');
  cases[p].forEach(function (c) {
    var o = r.route(c[0]);
    n++; if (o.label === c[1]) ok++;
    console.log((o.label === c[1] ? ' ok ' : ' XX ') + JSON.stringify(c[0]).padEnd(32), String(o.label).padEnd(14), o.ms + 'ms', o.steps + ' steps', o.reasoning ? 'think=' + JSON.stringify(o.reasoning) : '');
  });
});
console.log(ok + '/' + n + ' correct');
