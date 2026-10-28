/*
 * Text generation with the settings of the reference page (worker.js there):
 *   pipeline('text-generation', ..., { dtype: 'q8', device: 'wasm' })
 *   generator(messages, { max_new_tokens: 96, do_sample: false, repetition_penalty: 1.12 })
 * i.e. greedy decoding with the Transformers.js RepetitionPenaltyLogitsProcessor,
 * stopping at <|im_end|> (eos_token_id 2) or after 96 new tokens.
 *
 * The generation runs one model call per step() so the caller can stop between tokens.
 */
(function (root) {
  'use strict';

  var EOS = 2;

  function isWS(c) {
    return (c >= 9 && c <= 13) || c === 32 || c === 0xA0 || c === 0x1680 ||
      (c >= 0x2000 && c <= 0x200A) || c === 0x2028 || c === 0x2029 || c === 0x202F ||
      c === 0x205F || c === 0x3000 || c === 0xFEFF;
  }

  // String.prototype.trim with the modern whitespace set, independent of the browser's engine.
  function trim(s) {
    var a = 0, b = s.length;
    while (a < b && isWS(s.charCodeAt(a))) a++;
    while (b > a && isWS(s.charCodeAt(b - 1))) b--;
    return s.slice(a, b);
  }

  function Generation(model, tokenizer, messages, opts) {
    opts = opts || {};
    this.model = model;
    this.tokenizer = tokenizer;
    this.maxNewTokens = opts.maxNewTokens || 96;
    this.penalty = opts.repetitionPenalty || 1.12;
    this.prompt = tokenizer.applyChatTemplate(messages);
    this.promptIds = tokenizer.encode(this.prompt);
    this.generated = [];
    // ids that get the repetition penalty: every id of the sequence so far, once
    this.seen = [];
    this.seenFlag = {};
    for (var i = 0; i < this.promptIds.length; i++) this._see(this.promptIds[i]);
    this.done = false;
  }

  Generation.prototype._see = function (id) {
    if (!this.seenFlag[id]) { this.seenFlag[id] = true; this.seen.push(id); }
  };

  // Runs the next model call (the first one processes the whole prompt) and calls back with
  // (err, tokenId). tokenId is -1 once generation has finished.
  Generation.prototype.step = function (cb) {
    if (this.done) { cb(null, -1); return; }
    var self = this, input;
    if (this.generated.length === 0) {
      this.model.resetCache();
      input = this.promptIds;
    } else {
      input = [this.generated[this.generated.length - 1]];
    }
    this.model.forward(input, function (err, hidden, blocked) {
      if (err) return cb(err);
      self.model.nextToken(hidden, blocked, self.seen, self.penalty, function (err, id) {
        if (err) return cb(err);
        self.generated.push(id);
        self._see(id);
        if (id === EOS || self.generated.length >= self.maxNewTokens) self.done = true;
        cb(null, id);
      });
    });
  };

  var defer = (typeof setImmediate === 'function') ? setImmediate : function (f) { setTimeout(f, 0); };

  // Run step() until done or stop() is called; onToken(id, gen) after each token; cb(err).
  // Each token starts from a fresh task, so the call stack stays short and messages
  // (such as a stop request) are handled between tokens.
  Generation.prototype.run = function (onToken, cb) {
    var self = this;
    function next() {
      if (self.stopped) return cb(null);
      self.step(function (err, id) {
        if (err) return cb(err);
        if (id < 0) return cb(null);
        if (onToken) onToken(id, self);
        if (self.done) return cb(null);
        defer(next);
      });
    }
    next();
  };

  Generation.prototype.stop = function () {
    this.stopped = true;
  };

  // Text so far (special tokens skipped).
  Generation.prototype.text = function () {
    return this.tokenizer.decode(this.generated, true);
  };

  // Final reply as the reference worker returns it: the streamed text, trimmed.
  Generation.prototype.finalText = function () {
    return trim(this.text());
  };

  var api = { Generation: Generation, trim: trim, EOS: EOS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmolLMGenerate = api;
})(typeof self !== 'undefined' ? self : this);
