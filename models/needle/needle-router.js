/*
 * Needle 2 tool router for Zagreus.
 *
 * Same decoding as the Needle2_PureJS_release runner (index.html there), made synchronous so it
 * runs inside a worker: root choice between <tool_call>, <think> and plain text, an optional
 * reasoning span, then a finite-grammar decode of the tool call over a fixed set of candidate
 * JSON calls. The model picks every token from its own logits; nothing is forced.
 *
 *   var r = new NeedleRouter(engine, tokenizer);
 *   r.setTools(tools, systemPrompt, candidates);   // candidates: [{ label, json }]
 *   r.route('Turn on dark mode')  -> { call: {name, arguments} | null, label, reasoning, reason,
 *                                      confidence, plain, ms, tokens }
 */
(function (global) {
  'use strict';

  var BOS = 2, THINK = 6, THINK_END = 7, CALL = 10, CALL_END = 11, FIRST_PLAIN = 14;

  function NeedleRouter(engine, tok) {
    this.engine = engine;
    this.tok = tok;
    this.tools = [];
    this.system = '';
    this.candidates = [];
  }

  NeedleRouter.prototype.rawEncode = function (text) {
    var x = this.tok.encode(text);
    if (x && x.ids) x = x.ids;
    return Array.prototype.slice.call(x);
  };

  NeedleRouter.prototype.encodeNoBos = function (text) {
    var ids = this.rawEncode(text);
    if (ids.length && ids[0] === BOS) ids.shift();
    return ids;
  };

  NeedleRouter.prototype.bodyFor = function (json) {
    var a = this.encodeNoBos('<tool_call>' + json + '</tool_call>');
    if (a[0] !== CALL || a[a.length - 1] !== CALL_END) throw new Error('unexpected tool_call framing');
    return a.slice(1, -1);
  };

  NeedleRouter.prototype.setTools = function (tools, system, candidates) {
    this.tools = tools;
    this.toolsJson = JSON.stringify(tools);
    this.system = system;
    this.candidates = [];
    for (var i = 0; i < candidates.length; i++) {
      this.candidates.push({ label: candidates[i].label, json: candidates[i].json, ids: this.bodyFor(candidates[i].json) });
    }
    this.prefix = null;
  };

  // The system turn and the tool list are the same for every request. They are run through the
  // model once; the engine state after them (KV cache, engram rings, confidence pool) is kept
  // and restored before each request, so a request only costs its own ~15-20 tokens.
  NeedleRouter.prototype.warm = function (onProgress) {
    var head = this.prompt('\u0000').split('\u0000')[0];
    var ids = this.rawEncode(head);
    if (!ids.length || ids[0] !== BOS) ids.unshift(BOS);
    // the last token can merge with the user's text, so the cached prefix stops before it
    ids.pop();
    var eng = this.engine;
    eng.reset();
    eng.confidenceReset();
    for (var i = 0; i < ids.length; i++) {
      eng.forwardToken(ids[i], false);
      if (onProgress && (i % 8 === 0 || i === ids.length - 1)) onProgress('warm', i + 1, ids.length);
    }
    this.prefix = { ids: ids, state: snapshot(eng) };
    return ids.length;
  };

  function copyOf(a) { return a.slice ? a.slice() : new a.constructor(a); }

  function snapshot(e) {
    return {
      position: e.position,
      history: copyOf(e.history),
      caches: e.caches.map(function (c) { return [copyOf(c.k), copyOf(c.v), copyOf(c.ks), copyOf(c.vs), copyOf(c.slotPos)]; }),
      rings: e.engramRings.map(function (r) { return [copyOf(r.data), copyOf(r.scale), copyOf(r.pos)]; }),
      conf: [copyOf(e._confMax), copyOf(e._confDenom), copyOf(e._confSum), e._confCellCount, e._confActive]
    };
  }

  function restore(e, s) {
    e.position = s.position;
    e.history.set(s.history);
    s.caches.forEach(function (a, i) { var c = e.caches[i]; c.k.set(a[0]); c.v.set(a[1]); c.ks.set(a[2]); c.vs.set(a[3]); c.slotPos.set(a[4]); });
    s.rings.forEach(function (a, i) { var r = e.engramRings[i]; r.data.set(a[0]); r.scale.set(a[1]); r.pos.set(a[2]); });
    e._confMax.set(s.conf[0]); e._confDenom.set(s.conf[1]); e._confSum.set(s.conf[2]);
    e._confCellCount = s.conf[3]; e._confActive = s.conf[4];
  }

  NeedleRouter.prototype.prompt = function (userText) {
    return '<|im_start|>system\n' + this.system + '<|im_end|>\n' +
      '<|im_start|>user\n<tools>' + this.toolsJson + '</tools>\n' + userText + '<|im_end|>\n' +
      '<|im_start|>assistant\n';
  };

  function best(logits, allowed) {
    var id = -1, v = -Infinity;
    for (var i = 0; i < allowed.length; i++) {
      var x = Number(logits[allowed[i]]);
      if (x > v) { v = x; id = allowed[i]; }
    }
    return { id: id, logit: v };
  }

  function bestPlain(logits, extra) {
    var id = -1, v = -Infinity, i;
    for (i = FIRST_PLAIN; i < logits.length; i++) {
      var x = Number(logits[i]);
      if (x > v) { v = x; id = i; }
    }
    if (extra !== undefined && Number(logits[extra]) > v) return { id: extra, logit: Number(logits[extra]) };
    return { id: id, logit: v };
  }

  function sameStart(a, p) {
    for (var i = 0; i < p.length; i++) if (a[i] !== p[i]) return false;
    return true;
  }

  function normalize(parts) {
    if (!parts || !parts.length) return null;
    var t = parts.join('').replace(/<0x0A>/g, '').replace(/▁/g, ' ').trim();
    return t.length ? t : null;
  }

  // onProgress(stage, done, total) is optional.
  NeedleRouter.prototype.route = function (userText, onProgress) {
    var eng = this.engine, self = this, t0 = Date.now();
    var ids = this.rawEncode(this.prompt(userText));
    if (!ids.length || ids[0] !== BOS) ids.unshift(BOS);
    var res = null, i, start = 0, pre = this.prefix;
    if (pre && ids.length > pre.ids.length && sameStart(ids, pre.ids)) {
      restore(eng, pre.state);
      start = pre.ids.length;
    } else {
      eng.reset();
      eng.confidenceReset();
    }
    var steps = ids.length - start;
    for (i = start; i < ids.length; i++) {
      res = eng.forwardToken(ids[i], i === ids.length - 1);
      if (onProgress && (i % 8 === 0 || i === ids.length - 1)) onProgress('prefill', i + 1, ids.length);
    }
    var out = { call: null, label: null, reasoning: null, reason: null, confidence: null, plain: null, tokens: ids.length, ms: 0 };
    function finish() {
      var c = eng.confidenceValue();
      out.confidence = c === null || c === undefined ? null : c;
      out.ms = Date.now() - t0;
      out.steps = steps;
      return out;
    }
    var plain = bestPlain(res.logits);
    var root = [
      { id: CALL, logit: Number(res.logits[CALL]) },
      { id: THINK, logit: Number(res.logits[THINK]) },
      plain
    ].sort(function (a, b) { return b.logit - a.logit; })[0];

    if (root.id !== CALL && root.id !== THINK) {
      out.plain = this.tok.pieceText(root.id);
      return finish();
    }
    if (root.id === THINK) {
      if (onProgress) onProgress('think', 0, 32);
      res = eng.forwardToken(THINK, true); steps++;
      var think = [], closed = false;
      for (i = 0; i < 32; i++) {
        var c1 = bestPlain(res.logits, THINK_END);
        res = eng.forwardToken(c1.id, true); steps++;
        if (c1.id === THINK_END) { closed = true; break; }
        think.push(this.tok.pieceText(c1.id));
      }
      out.reasoning = normalize(think);
      if (!closed) return finish();
      var post = [], toCall = false;
      for (i = 0; i < 8; i++) {
        var c2 = bestPlain(res.logits, CALL);
        if (c2.id === CALL) { toCall = true; break; }
        post.push(this.tok.pieceText(c2.id));
        res = eng.forwardToken(c2.id, true); steps++;
      }
      out.reason = normalize(post);
      if (!toCall) return finish();
    }
    // finite-grammar tool call
    if (onProgress) onProgress('call', 0, 1);
    res = eng.forwardToken(CALL, true); steps++;
    var active = this.candidates.slice(), pos = 0;
    for (;;) {
      var allowed = [], j;
      for (j = 0; j < active.length; j++) {
        var next = pos < active[j].ids.length ? active[j].ids[pos] : CALL_END;
        if (allowed.indexOf(next) < 0) allowed.push(next);
      }
      if (!allowed.length) break;
      var ch = best(res.logits, allowed);
      if (ch.id === CALL_END) {
        var done = active.filter(function (c) { return pos >= c.ids.length; });
        if (done.length === 1) {
          out.label = done[0].label;
          out.call = JSON.parse(done[0].json);
        }
        res = eng.forwardToken(CALL_END, true); steps++;
        break;
      }
      active = active.filter(function (c) { return pos < c.ids.length && c.ids[pos] === ch.id; });
      if (!active.length) break;
      // a single survivor: the rest of its body is determined, no need to run the model on it
      if (active.length === 1) {
        out.label = active[0].label;
        out.call = JSON.parse(active[0].json);
        break;
      }
      res = eng.forwardToken(ch.id, true); steps++;
      pos++;
    }
    return finish();
  };

  global.NeedleRouter = NeedleRouter;
}(typeof self !== 'undefined' ? self : this));
