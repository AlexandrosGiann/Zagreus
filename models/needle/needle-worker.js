/*
 * Needle 2 worker for Zagreus: loads needle2.cact, warms one cached prompt prefix per tool
 * profile, and answers routing requests.
 *
 * From the page:
 *   { type: 'load', profiles: { name: { tools, candidates } }, system }
 *   { type: 'route', id, profile, text }
 * To the page:
 *   { type: 'progress', stage, loaded, total, profile }
 *   { type: 'ready', ms }                       (model loaded; profiles warm lazily)
 *   { type: 'warm', profile, tokens, ms }
 *   { type: 'result', id, label, call, reasoning, reason, confidence, plain, ms, steps }
 *   { type: 'error', id, message }
 */
importScripts('cact-full.js', 'tokenizer-full.js', 'engine-full.js', 'needle-router.js');

var engine = null, tok = null, routers = {}, specs = {}, system = '', queue = [], busy = false;

function post(m) { self.postMessage(m); }

function load(d) {
  var t0 = Date.now();
  specs = d.profiles || {};
  system = d.system || '';
  var xhr = new XMLHttpRequest();
  xhr.open('GET', 'needle2.cact', true);
  xhr.responseType = 'arraybuffer';
  xhr.onprogress = function (e) { post({ type: 'progress', stage: 'model', loaded: e.loaded, total: e.total || 13737807 }); };
  xhr.onerror = function () { post({ type: 'error', message: 'needle2.cact: network error' }); };
  xhr.onload = function () {
    try {
      if (xhr.status !== 200 && xhr.status !== 0) throw new Error('needle2.cact: HTTP ' + xhr.status);
      var model = new NeedleCact.CactModel(xhr.response);
      tok = new NeedleTokenizer.Tokenizer(model);
      engine = new NeedleFull.Engine(model);
      post({ type: 'ready', ms: Date.now() - t0 });
      // warm every profile in the background, one at a time, between requests
      Object.keys(specs).forEach(function (p) { queue.push({ type: 'warm', profile: p }); });
      pump();
    } catch (e) {
      post({ type: 'error', message: String(e && e.message || e) });
    }
  };
  xhr.send(null);
}

function router(p) {
  if (!routers[p]) {
    var s = specs[p];
    if (!s) throw new Error('unknown tool profile ' + p);
    var r = new NeedleRouter(engine, tok);
    r.setTools(s.tools, system, s.candidates);
    routers[p] = r;
  }
  return routers[p];
}

function handle(job) {
  if (job.type === 'warm') {
    var r = router(job.profile);
    if (r.prefix) return;
    var t0 = Date.now();
    var n = r.warm(function (stage, done, total) {
      post({ type: 'progress', stage: 'warm', profile: job.profile, loaded: done, total: total });
    });
    post({ type: 'warm', profile: job.profile, tokens: n, ms: Date.now() - t0 });
    return;
  }
  if (job.type === 'route') {
    var rr = router(job.profile);
    if (!rr.prefix) rr.warm();
    var o = rr.route(job.text);
    o.type = 'result';
    o.id = job.id;
    o.profile = job.profile;
    post(o);
  }
}

function pump() {
  if (busy || !queue.length || !engine) return;
  busy = true;
  // routing requests go before background warm-ups
  var i, k = 0;
  for (i = 0; i < queue.length; i++) if (queue[i].type === 'route') { k = i; break; }
  var job = queue.splice(k, 1)[0];
  try { handle(job); } catch (e) { post({ type: 'error', id: job.id, message: String(e && e.message || e) }); }
  busy = false;
  setTimeout(pump, 0);
}

self.onmessage = function (e) {
  var d = e.data || {};
  if (d.type === 'load') load(d);
  else if (d.type === 'route') { queue.push(d); pump(); }
};
