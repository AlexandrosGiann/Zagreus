/*
 * Tokenizer of paraphrase-multilingual-MiniLM-L12-v2 (XLM-R SentencePiece Unigram), giving the
 * same ids as Transformers.js 3.8.1:
 *   - added tokens (<s> <pad> </s> <unk> <mask>) split out first, longest match; <mask> strips
 *     the whitespace before it (lstrip)
 *   - Precompiled normalizer as Transformers.js implements it: control characters removed,
 *     some spaces mapped to ' ', then NFKC (own implementation, Unicode 17, so it does not
 *     depend on String.prototype.normalize or the Unicode version of the browser)
 *   - WhitespaceSplit + Metaspace ('\u2581' prepended to every word)
 *   - Unigram Viterbi with the same tie-breaking, unknown pieces fused
 *   - <s> ... </s>, truncation to 512 tokens
 *
 * ES5 only (no Map, no String.prototype.normalize, no Array.from, no \p{..} regexes).
 */
(function (root) {
  'use strict';

  var NfkcData = (typeof module !== 'undefined' && module.exports) ? require('./minilm-nfkc-data.js') : root.MiniLMNfkcData;

  // ---------------------------------------------------------------------------------------
  // NFKC

  var decomp = null, cccMap = null, compose = null;

  function initNfkc() {
    if (decomp) return;
    var i, j, parts, items, cps;
    decomp = {}; cccMap = {}; compose = {};
    items = NfkcData.d.split(',');
    for (i = 0; i < items.length; i++) {
      parts = items[i].split(':');
      cps = parts[1].split(' ');
      for (j = 0; j < cps.length; j++) cps[j] = parseInt(cps[j], 36);
      decomp[parseInt(parts[0], 36)] = cps;
    }
    items = NfkcData.c.split(',');
    for (i = 0; i < items.length; i++) {
      parts = items[i].split(':');
      var start = parseInt(parts[0], 36), len = parseInt(parts[1], 36), cls = parseInt(parts[2], 36);
      for (j = 0; j <= len; j++) cccMap[start + j] = cls;
    }
    items = NfkcData.p.split(',');
    for (i = 0; i < items.length; i++) {
      parts = items[i].split(':');
      cps = parts[0].split(' ');
      compose[parseInt(cps[0], 36) * 0x110000 + parseInt(cps[1], 36)] = parseInt(parts[1], 36);
    }
  }

  var SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
  var LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

  // UTF-16 string -> code points (a lone surrogate stays one element, like Array.from)
  function codePoints(s) {
    var out = [], i = 0, n = s.length, c, d;
    while (i < n) {
      c = s.charCodeAt(i++);
      if (c >= 0xD800 && c <= 0xDBFF && i < n) {
        d = s.charCodeAt(i);
        if (d >= 0xDC00 && d <= 0xDFFF) { i++; c = (c - 0xD800) * 0x400 + (d - 0xDC00) + 0x10000; }
      }
      out.push(c);
    }
    return out;
  }

  function fromCodePoint(c) {
    if (c < 0x10000) return String.fromCharCode(c);
    c -= 0x10000;
    return String.fromCharCode(0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF));
  }

  function cpsToString(cps) {
    var s = '', i;
    for (i = 0; i < cps.length; i++) s += fromCodePoint(cps[i]);
    return s;
  }

  function ccc(c) { return cccMap[c] || 0; }

  function nfkc(s) {
    var i, ascii = true;
    for (i = 0; i < s.length; i++) if (s.charCodeAt(i) > 0x7F) { ascii = false; break; }
    if (ascii) return s;
    initNfkc();
    var cps = codePoints(s), d = [], c, k, m, sIndex, t;
    // full compatibility decomposition
    for (i = 0; i < cps.length; i++) {
      c = cps[i];
      sIndex = c - SBase;
      if (sIndex >= 0 && sIndex < SCount) {
        d.push(LBase + ((sIndex / NCount) | 0), VBase + (((sIndex % NCount) / TCount) | 0));
        t = sIndex % TCount;
        if (t) d.push(TBase + t);
        continue;
      }
      m = decomp[c];
      if (m) { for (k = 0; k < m.length; k++) d.push(m[k]); } else d.push(c);
    }
    // canonical ordering (stable sort of each run of non-starters by combining class)
    var cls = new Array(d.length);
    for (i = 0; i < d.length; i++) cls[i] = ccc(d[i]);
    for (i = 1; i < d.length; i++) {
      if (cls[i] === 0) continue;
      k = i;
      while (k > 0 && cls[k - 1] > cls[k]) {
        t = d[k]; d[k] = d[k - 1]; d[k - 1] = t;
        t = cls[k]; cls[k] = cls[k - 1]; cls[k - 1] = t;
        k--;
      }
    }
    // canonical composition
    var out = [], starter = -1, lastCls = -1, cc, comp, a, lIndex, vIndex;
    for (i = 0; i < d.length; i++) {
      c = d[i]; cc = cls[i];
      if (starter >= 0 && (lastCls === 0 || lastCls < cc)) {
        a = out[starter];
        comp = undefined;
        lIndex = a - LBase; vIndex = c - VBase;
        if (lastCls === 0 && lIndex >= 0 && lIndex < LCount && vIndex >= 0 && vIndex < VCount) {
          comp = SBase + (lIndex * VCount + vIndex) * TCount;
        } else if (lastCls === 0 && a - SBase >= 0 && a - SBase < SCount && (a - SBase) % TCount === 0 &&
                   c - TBase > 0 && c - TBase < TCount) {
          comp = a + (c - TBase);
        } else {
          comp = compose[a * 0x110000 + c];
        }
        if (comp !== undefined) { out[starter] = comp; continue; }
      }
      if (cc === 0) starter = out.length;
      out.push(c);
      lastCls = cc;
    }
    return cpsToString(out);
  }

  // ---------------------------------------------------------------------------------------
  // Tokenizer

  // Characters of JavaScript's \s (and String.prototype.trim) in current engines.
  function isJsSpace(c) {
    return c === 0x20 || (c >= 0x09 && c <= 0x0D) || c === 0xA0 || c === 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
      c === 0x2028 || c === 0x2029 || c === 0x202F || c === 0x205F || c === 0x3000 || c === 0xFEFF;
  }

  function trimEnd(s) {
    var e = s.length;
    while (e > 0 && isJsSpace(s.charCodeAt(e - 1))) e--;
    return s.substring(0, e);
  }

  function trimStart(s) {
    var b = 0;
    while (b < s.length && isJsSpace(s.charCodeAt(b))) b++;
    return s.substring(b);
  }

  var CONTROL_RE = /[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]/g;
  var SPACE_RE = /[\u0009\u000A\u000C\u000D\u00a0\u1680\u2000-\u200f\u2028\u2029\u202f\u205f\u2581\u3000\ufeff\ufffd]/g;

  // Transformers.js Precompiled normalizer (nmt_nfkc as it implements it)
  function normalize(text) {
    text = text.replace(CONTROL_RE, '').replace(SPACE_RE, ' ');
    if (text.indexOf('\uff5e') >= 0) {
      var parts = text.split('\uff5e');
      for (var i = 0; i < parts.length; i++) parts[i] = nfkc(parts[i]);
      return parts.join('\uff5e');
    }
    return nfkc(text);
  }

  // text.match(/\S+/g) || []
  function whitespaceSplit(s) {
    var out = [], i = 0, n = s.length, b;
    while (i < n) {
      while (i < n && isJsSpace(s.charCodeAt(i))) i++;
      if (i >= n) break;
      b = i;
      while (i < n && !isJsSpace(s.charCodeAt(i))) i++;
      out.push(s.substring(b, i));
    }
    return out;
  }

  /*
   * vocab: array of pieces (index = id), scores: Float64Array, cfg: manifest.tokenizer
   */
  function Tokenizer(vocab, scores, cfg) {
    this.cfg = cfg;
    this.vocab = vocab;
    this.unkId = cfg.unkId;
    this.replacement = cfg.replacement;
    var ids = {}, i, maxLen = 0, n, minScore = Infinity;
    for (i = 0; i < vocab.length; i++) {
      ids[' ' + vocab[i]] = i;       // prefix: keys like '__proto__' stay ordinary properties
      n = codePoints(vocab[i]).length;
      if (n > maxLen) maxLen = n;
      if (scores[i] < minScore) minScore = scores[i];
    }
    this.ids = ids;
    this.maxLen = maxLen;
    this.scores = new Float64Array(scores);
    this.unkScore = minScore - 10.0;
    this.scores[this.unkId] = this.unkScore;
    this.added = {};
    this.addedList = [];
    for (i = 0; i < cfg.added.length; i++) {
      this.added[' ' + cfg.added[i].content] = cfg.added[i];
      this.addedList.push(cfg.added[i].content);
    }
  }

  Tokenizer.prototype.idOf = function (piece) {
    var id = this.ids[' ' + piece];
    return id === undefined ? undefined : id;
  };

  // DictionarySplitter: split around the added tokens (longest match, UTF-16 units)
  Tokenizer.prototype._splitAdded = function (text) {
    var out = [], n = text.length, start = 0, i = 0, list = this.addedList, j, m, t;
    while (i < n) {
      m = null;
      for (j = 0; j < list.length; j++) {
        t = list[j];
        if (text.substr(i, t.length) === t && (m === null || t.length > m.length)) m = t;
      }
      if (m !== null) {
        if (i > start) out.push(text.substring(start, i));
        out.push(m);
        i += m.length;
        start = i;
      } else i++;
    }
    if (start < n) out.push(text.substring(start));
    return out;
  };

  // Unigram Viterbi over one pre-token; returns pieces (strings)
  Tokenizer.prototype._viterbi = function (word) {
    var chars = codePoints(word), len = chars.length, strs = new Array(len), i, p, e, k;
    for (i = 0; i < len; i++) strs[i] = fromCodePoint(chars[i]);
    // nodes: parallel arrays; begin[p] / end[p]: node indices in insertion order
    var nPos = [], nLen = [], nScore = [], nPrev = [], nBt = [];
    var begin = new Array(len + 1), end = new Array(len + 1);
    for (i = 0; i <= len; i++) { begin[i] = []; end[i] = []; }
    // node 0 = BOS (ends at 0), node 1 = EOS (begins at len)
    nPos.push(0, len); nLen.push(0, 0); nScore.push(0, 0); nPrev.push(-1, -1); nBt.push(0, 0);
    end[0].push(0);
    begin[len].push(1);
    var unk = this.unkId, piece, id, single, node;
    for (p = 0; p < len; p++) {
      single = false;
      piece = '';
      for (k = 1; k <= this.maxLen && p + k <= len; k++) {
        piece += strs[p + k - 1];
        id = this.ids[' ' + piece];
        if (id === undefined) continue;
        node = nPos.length;
        nPos.push(p); nLen.push(k); nScore.push(this.scores[id]); nPrev.push(-1); nBt.push(0);
        begin[p].push(node); end[p + k].push(node);
        if (k === 1) single = true;
      }
      if (!single) {
        node = nPos.length;
        nPos.push(p); nLen.push(1); nScore.push(this.unkScore); nPrev.push(-1); nBt.push(0);
        begin[p].push(node); end[p + 1].push(node);
      }
    }
    var r, b, best, bestScore, score, l, list;
    for (p = 0; p <= len; p++) {
      if (begin[p].length === 0) return [];
      for (b = 0; b < begin[p].length; b++) {
        r = begin[p][b];
        best = -1; bestScore = 0;
        list = end[p];
        for (e = 0; e < list.length; e++) {
          l = list[e];
          score = nBt[l] + nScore[r];
          if (best < 0 || score > bestScore) { best = l; bestScore = score; }
        }
        if (best < 0) return [];
        nPrev[r] = best;
        nBt[r] = bestScore;
      }
    }
    var out = [];
    node = nPrev[1];
    if (node < 0) return [];
    while (nPrev[node] >= 0) {
      out.push(strs.slice(nPos[node], nPos[node] + nLen[node]).join(''));
      node = nPrev[node];
    }
    out.reverse();
    return out;
  };

  Tokenizer.prototype._fuseUnk = function (arr) {
    var fused = [], i = 0, unk = this.unkId, self = this;
    function idOr(t) { var id = self.idOf(t); return id === undefined ? unk : id; }
    while (i < arr.length) {
      fused.push(arr[i]);
      if (idOr(arr[i]) !== unk) { ++i; continue; }
      while (++i < arr.length && idOr(arr[i]) === unk) {
        if (self.idOf(fused[fused.length - 1]) !== unk) fused[fused.length - 1] += arr[i];
      }
    }
    return fused;
  };

  // Token strings of one text (no <s> </s>)
  Tokenizer.prototype.tokenize = function (text) {
    var sections = this._splitAdded(text), i, a, tokens = [], x, words, pieces, w;
    for (i = 0; i < sections.length; i++) {
      a = this.added[' ' + sections[i]];
      if (a) {
        if (a.lstrip && i > 0) sections[i - 1] = trimEnd(sections[i - 1]);
        if (a.rstrip && i < sections.length - 1) sections[i + 1] = trimStart(sections[i + 1]);
      }
    }
    for (i = 0; i < sections.length; i++) {
      x = sections[i];
      if (x.length === 0) continue;
      if (this.added[' ' + x]) { tokens.push(x); continue; }
      x = normalize(x);
      if (x.length === 0) continue;
      words = whitespaceSplit(x);
      pieces = [];
      for (w = 0; w < words.length; w++) {
        x = words[w];
        if (x.substring(0, this.replacement.length) !== this.replacement) x = this.replacement + x;
        pieces = pieces.concat(this._viterbi(x));
      }
      tokens = tokens.concat(this._fuseUnk(pieces));
    }
    return tokens;
  };

  // Ids with <s> ... </s>, truncated to maxLength (Transformers.js cuts the end, </s> included)
  Tokenizer.prototype.encode = function (text, noTruncate) {
    var tokens = this.tokenize(text), ids = [this.cfg.bos], i, id;
    for (i = 0; i < tokens.length; i++) {
      id = this.idOf(tokens[i]);
      ids.push(id === undefined ? this.unkId : id);
    }
    ids.push(this.cfg.eos);
    if (!noTruncate && ids.length > this.cfg.maxLength) ids.length = this.cfg.maxLength;
    return ids;
  };

  var api = { Tokenizer: Tokenizer, nfkc: nfkc, normalize: normalize, codePoints: codePoints };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MiniLMTokenizer = api;
})(typeof self !== 'undefined' ? self : this);
