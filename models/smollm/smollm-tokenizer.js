/*
 * Byte-level BPE tokenizer for SmolLM2, written to give the same ids and text as the
 * Transformers.js 3.8.1 tokenizer built from tokenizer.json:
 *   added-token splitting -> Digits(individual) -> ByteLevel (GPT-2 regex) -> BPE
 * and ByteLevel decoding with WHATWG UTF-8 replacement.
 *
 * ES5 only: no Unicode property escapes, TextEncoder/TextDecoder, Map or Set.
 * Needs SmolLMUnicode (smollm-unicode.js).
 */
(function (root) {
  'use strict';

  var Uni = (typeof module !== 'undefined' && module.exports) ? require('./smollm-unicode.js') : root.SmolLMUnicode;

  function decodeRanges(s) {
    var parts = s.split(','), out = new Array(parts.length), prev = 0;
    for (var i = 0; i < parts.length; i++) { prev += parseInt(parts[i], 36); out[i] = prev; }
    return out;
  }
  var RANGES_L = decodeRanges(Uni.L), RANGES_N = decodeRanges(Uni.N);

  function inRanges(r, cp) {
    var lo = 0, hi = (r.length >> 1) - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (cp < r[2 * mid]) hi = mid - 1;
      else if (cp > r[2 * mid + 1]) lo = mid + 1;
      else return true;
    }
    return false;
  }

  // Latin-1 and ASCII fast paths keep the common case cheap.
  function isL(cp) {
    if (cp < 128) return (cp >= 65 && cp <= 90) || (cp >= 97 && cp <= 122);
    return inRanges(RANGES_L, cp);
  }
  function isN(cp) {
    if (cp < 128) return cp >= 48 && cp <= 57;
    return inRanges(RANGES_N, cp);
  }
  // JavaScript \s (WhiteSpace + LineTerminator).
  function isWS(cp) {
    return (cp >= 9 && cp <= 13) || cp === 32 || cp === 0xA0 || cp === 0x1680 ||
      (cp >= 0x2000 && cp <= 0x200A) || cp === 0x2028 || cp === 0x2029 || cp === 0x202F ||
      cp === 0x205F || cp === 0x3000 || cp === 0xFEFF;
  }

  // Code points of a string, the way a /u regex sees them (lone surrogates stay as is).
  function codePoints(str) {
    var out = [], i = 0, n = str.length, c, d;
    while (i < n) {
      c = str.charCodeAt(i);
      if (c >= 0xD800 && c <= 0xDBFF && i + 1 < n) {
        d = str.charCodeAt(i + 1);
        if (d >= 0xDC00 && d <= 0xDFFF) {
          out.push(((c - 0xD800) << 10) + (d - 0xDC00) + 0x10000);
          i += 2;
          continue;
        }
      }
      out.push(c);
      i++;
    }
    return out;
  }

  // UTF-8 bytes of code points (lone surrogates become U+FFFD, as TextEncoder does).
  function utf8Bytes(cps, start, end, out) {
    for (var i = start; i < end; i++) {
      var c = cps[i];
      if (c >= 0xD800 && c <= 0xDFFF) c = 0xFFFD;
      if (c < 0x80) out.push(c);
      else if (c < 0x800) out.push(0xC0 | (c >> 6), 0x80 | (c & 63));
      else if (c < 0x10000) out.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      else out.push(0xF0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    }
    return out;
  }

  function fromCodePoint(cp) {
    if (cp < 0x10000) return String.fromCharCode(cp);
    cp -= 0x10000;
    return String.fromCharCode(0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
  }

  // WHATWG UTF-8 decode with replacement (what TextDecoder('utf-8') does by default).
  function utf8Decode(bytes) {
    var out = '', i = 0, n = bytes.length;
    var cp = 0, need = 0, seen = 0, lower = 0x80, upper = 0xBF;
    while (i < n) {
      var b = bytes[i];
      if (need === 0) {
        i++;
        if (b <= 0x7F) { out += String.fromCharCode(b); continue; }
        if (b >= 0xC2 && b <= 0xDF) { need = 1; cp = b & 0x1F; }
        else if (b >= 0xE0 && b <= 0xEF) {
          if (b === 0xE0) lower = 0xA0;
          if (b === 0xED) upper = 0x9F;
          need = 2; cp = b & 0xF;
        } else if (b >= 0xF0 && b <= 0xF4) {
          if (b === 0xF0) lower = 0x90;
          if (b === 0xF4) upper = 0x8F;
          need = 3; cp = b & 0x7;
        } else { out += '�'; }
        continue;
      }
      if (b < lower || b > upper) {
        // Invalid continuation: emit U+FFFD and reprocess this byte.
        cp = need = seen = 0; lower = 0x80; upper = 0xBF;
        out += '�';
        continue;
      }
      lower = 0x80; upper = 0xBF;
      cp = (cp << 6) | (b & 0x3F);
      seen++; i++;
      if (seen === need) {
        out += fromCodePoint(cp);
        cp = need = seen = 0;
      }
    }
    if (need !== 0) out += '�';
    return out;
  }

  // GPT-2 bytes_to_unicode.
  function bytesToUnicode() {
    var bs = [], i;
    for (i = 33; i <= 126; i++) bs.push(i);
    for (i = 161; i <= 172; i++) bs.push(i);
    for (i = 174; i <= 255; i++) bs.push(i);
    var cs = bs.slice(), n = 0;
    for (i = 0; i < 256; i++) {
      if (bs.indexOf(i) < 0) { bs.push(i); cs.push(256 + n); n++; }
    }
    var enc = new Array(256), dec = Object.create(null);
    for (i = 0; i < bs.length; i++) { enc[bs[i]] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = bs[i]; }
    return { enc: enc, dec: dec };
  }

  /*
   * GPT-2 pre-tokenizer regex, applied to code points:
   *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
   * Returns [start, end) pairs in code point indices.
   */
  var CONTRACTIONS = [[115], [116], [114, 101], [118, 101], [109], [108, 108], [100]];
  function gpt2Split(c) {
    var out = [], n = c.length, i = 0, j, k, a, m, ok;
    while (i < n) {
      var cp = c[i];
      // 's 't 're 've 'm 'll 'd
      if (cp === 39) {
        m = -1;
        for (a = 0; a < CONTRACTIONS.length && m < 0; a++) {
          var t = CONTRACTIONS[a];
          ok = i + t.length < n || i + t.length === n;
          for (k = 0; ok && k < t.length; k++) if (i + 1 + k >= n || c[i + 1 + k] !== t[k]) ok = false;
          if (ok) m = i + 1 + t.length;
        }
        if (m > 0) { out.push(i, m); i = m; continue; }
      }
      var s = (cp === 32 && i + 1 < n) ? i + 1 : i;
      var first = c[s];
      if (s === i + 1 || !isWS(cp)) {
        if (isL(first)) {
          j = s + 1; while (j < n && isL(c[j])) j++;
          out.push(i, j); i = j; continue;
        }
        if (isN(first)) {
          j = s + 1; while (j < n && isN(c[j])) j++;
          out.push(i, j); i = j; continue;
        }
        if (!isWS(first)) {
          j = s + 1; while (j < n && !isWS(c[j]) && !isL(c[j]) && !isN(c[j])) j++;
          out.push(i, j); i = j; continue;
        }
      }
      // whitespace run
      j = i + 1; while (j < n && isWS(c[j])) j++;
      if (j < n && j - i >= 2) j--;   // \s+(?!\S) leaves the last space for the next word
      out.push(i, j); i = j;
    }
    return out;
  }

  function Tokenizer(json) {
    var model = json.model, i;
    // Maps without a prototype, so words such as "constructor" are ordinary keys.
    this.vocab = Object.create(null);   // token -> id
    this.idToToken = [];
    for (var tok in model.vocab) {
      if (Object.prototype.hasOwnProperty.call(model.vocab, tok)) {
        this.vocab[tok] = model.vocab[tok];
        this.idToToken[model.vocab[tok]] = tok;
      }
    }
    // merge ranks; a repeated merge keeps its last index, as in new Map(entries)
    this.ranks = Object.create(null);
    for (i = 0; i < model.merges.length; i++) {
      var mg = model.merges[i];
      var key = (typeof mg === 'string') ? mg.replace(' ', '\u0001') : mg[0] + '\u0001' + mg[1];
      this.ranks[key] = i;
    }
    this.added = Object.create(null);
    this.special = Object.create(null);
    this.addedTrie = Object.create(null);
    for (i = 0; i < json.added_tokens.length; i++) {
      var at = json.added_tokens[i];
      this.idToToken[at.id] = at.content;
      this.added[at.content] = at.id;
      if (at.special) this.special[at.content] = true;
      var node = this.addedTrie;
      for (var c = 0; c < at.content.length; c++) {
        var ch = at.content.charAt(c);
        if (!node[ch]) { node[ch] = Object.create(null); node[ch]._end = null; }
        node = node[ch];
      }
      node._end = at.content;
    }
    var b2u = bytesToUnicode();
    this.byteEnc = b2u.enc;
    this.byteDec = b2u.dec;
    this.cache = Object.create(null);
    this.cacheSize = 0;
  }

  // DictionarySplitter: longest added-token match, scanning UTF-16 code units.
  Tokenizer.prototype._splitAdded = function (text) {
    var out = [], n = text.length, start = 0, i = 0;
    while (i < n) {
      var node = this.addedTrie, match = null, j = i;
      while (j < n) {
        var ch = text.charAt(j);
        if (ch === '_end' || !Object.prototype.hasOwnProperty.call(node, ch)) break;
        node = node[ch];
        if (node._end) match = node._end;
        j++;
      }
      if (match) {
        if (i > start) out.push(text.slice(start, i));
        out.push(match);
        i += match.length;
        start = i;
      } else {
        i++;
      }
    }
    if (start < n) out.push(text.slice(start));
    return out;
  };

  Tokenizer.prototype._bpe = function (word) {
    var cached = this.cache[word];
    if (cached) return cached;
    var sym = [], i;
    // Array.from(token): split into code points
    var cps = codePoints(word);
    for (i = 0; i < cps.length; i++) sym.push(fromCodePoint(cps[i]));
    while (sym.length > 1) {
      var best = -1, bestRank = Infinity;
      for (i = 0; i < sym.length - 1; i++) {
        var r = this.ranks[sym[i] + '\u0001' + sym[i + 1]];
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      sym.splice(best, 2, sym[best] + sym[best + 1]);
    }
    if (word.length < 256) {
      if (this.cacheSize > 10000) { this.cache = Object.create(null); this.cacheSize = 0; }
      this.cache[word] = sym;
      this.cacheSize++;
    }
    return sym;
  };

  Tokenizer.prototype._encodeSection = function (text, ids) {
    var cps = codePoints(text), n = cps.length, p = 0;
    // Digits(individual_digits): [^\d]+|\d  (ASCII digits only)
    while (p < n) {
      var q = p;
      if (cps[p] >= 48 && cps[p] <= 57) q = p + 1;
      else { while (q < n && !(cps[q] >= 48 && cps[q] <= 57)) q++; }
      var piece = cps.slice(p, q);
      var spans = gpt2Split(piece);
      for (var s = 0; s < spans.length; s += 2) {
        var bytes = utf8Bytes(piece, spans[s], spans[s + 1], []);
        var word = '';
        for (var b = 0; b < bytes.length; b++) word += this.byteEnc[bytes[b]];
        var toks = this._bpe(word);
        for (var t = 0; t < toks.length; t++) {
          var id = this.vocab[toks[t]];
          if (id === undefined) throw new Error('Token not in vocabulary: ' + toks[t]);
          ids.push(id);
        }
      }
      p = q;
    }
  };

  Tokenizer.prototype.encode = function (text) {
    var sections = this._splitAdded(text), ids = [];
    for (var i = 0; i < sections.length; i++) {
      var sec = sections[i];
      if (sec.length === 0) continue;
      if (Object.prototype.hasOwnProperty.call(this.added, sec)) { ids.push(this.added[sec]); continue; }
      this._encodeSection(sec, ids);
    }
    return ids;
  };

  // decode(ids, { skip_special_tokens: true }) with clean_up_tokenization_spaces = false.
  Tokenizer.prototype.decode = function (ids, skipSpecial) {
    var out = '', bytes = [], i, j;
    for (i = 0; i < ids.length; i++) {
      var tok = this.idToToken[ids[i]];
      if (tok === undefined) continue;
      if (Object.prototype.hasOwnProperty.call(this.added, tok)) {
        if (skipSpecial && this.special[tok]) continue;
        if (bytes.length) { out += utf8Decode(bytes); bytes = []; }
        out += tok;
        continue;
      }
      for (j = 0; j < tok.length; j++) {
        var b = this.byteDec[tok.charAt(j)];
        if (b !== undefined) bytes.push(b);
      }
    }
    if (bytes.length) out += utf8Decode(bytes);
    return out;
  };

  // The model's chat template (tokenizer_config.json), rendered without a template engine.
  Tokenizer.prototype.applyChatTemplate = function (messages) {
    var s = '';
    for (var i = 0; i < messages.length; i++) {
      if (i === 0 && messages[0].role !== 'system') {
        s += '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n';
      }
      s += '<|im_start|>' + messages[i].role + '\n' + messages[i].content + '<|im_end|>\n';
    }
    return s + '<|im_start|>assistant\n';
  };

  var api = { Tokenizer: Tokenizer, utf8Decode: utf8Decode, _gpt2Split: gpt2Split, _codePoints: codePoints };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmolLMTokenizer = api;
})(typeof self !== 'undefined' ? self : this);
