/*
 * SmolLM2 q8 compute shard: holds a slice of the output rows of every linear layer and a
 * slice of the vocabulary, and runs the matmul-heavy parts of the model on it.
 *
 * One shard covering everything runs in the same thread when there is a single thread;
 * otherwise each Web Worker (smollm-compute.js) holds one shard.
 *
 * All results are bit-identical to ONNX Runtime (WASM) for the rows a shard owns.
 * ES5 + typed arrays only.
 */
(function (root) {
  'use strict';

  var fround = Math.fround || (function () {
    var f = new Float32Array(1);
    return function (x) { f[0] = x; return f[0]; };
  })();

  var U = 5.9604644775390625e-8;   // 2^-24, float32 unit roundoff

  // MLAS logistic constants (logistic.cpp, MlasLogisticConstants).
  var LG_LOWER = -18, LG_UPPER = 18;
  var LG_A9 = fround(4.37031012579801e-11);
  var LG_A7 = fround(1.15627324459942e-07);
  var LG_A5 = fround(6.08574864600143e-05);
  var LG_A3 = fround(8.51377133304701e-03);
  var LG_A1 = fround(2.48287947061529e-01);
  var LG_B10 = fround(6.10247389755681e-13);
  var LG_B8 = fround(5.76102136993427e-09);
  var LG_B6 = fround(6.29106785017040e-06);
  var LG_B4 = fround(1.70198817374094e-03);
  var LG_B2 = fround(1.16817656904453e-01);
  var LG_B0 = fround(9.93151921023180e-01);

  // MlasLogisticKernel for one value.
  function logistic(value) {
    var v = value < LG_LOWER ? LG_LOWER : value;
    v = v > LG_UPPER ? LG_UPPER : v;
    var v2 = fround(v * v);
    var p = fround(fround(v2 * LG_A9) + LG_A7);
    p = fround(fround(p * v2) + LG_A5);
    p = fround(fround(p * v2) + LG_A3);
    p = fround(fround(p * v2) + LG_A1);
    p = fround(p * v);
    var q = fround(fround(v2 * LG_B10) + LG_B8);
    q = fround(fround(q * v2) + LG_B6);
    q = fround(fround(q * v2) + LG_B4);
    q = fround(fround(q * v2) + LG_B2);
    q = fround(fround(q * v2) + LG_B0);
    return fround(fround(p / q) + 0.5);
  }

  /*
   * MatMulInteger + Cast + Mul(scale): out[r][n] for n in this shard's rows.
   * qa holds (q - zero_point) of the activation, rowSums its per-row sums,
   * w the weight rows [n][k] (uint8), zw the weight zero point.
   * Four output rows share each activation load.
   */
  function matmulQ(qa, rowSums, rows, K, w, N, zw, combinedScale, out) {
    var r, n, k, ao, w0, w1, w2, w3, s0, s1, s2, s3, a, zs, oo;
    var N4 = N - (N % 4);
    r = 0;
    if (N4 === N) {
      // Prefill: two rows at a time reuse every weight load (2 x 4 block).
      for (; r + 1 < rows; r += 2) matmulQ2x4(qa, rowSums, r, K, w, N, zw, combinedScale, out);
    }
    for (; r < rows; r++) {
      ao = r * K;
      zs = zw * rowSums[r];
      oo = r * N;
      for (n = 0; n < N4; n += 4) {
        w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
        s0 = 0; s1 = 0; s2 = 0; s3 = 0;
        for (k = 0; k < K; k++) {
          a = qa[ao + k];
          s0 = (s0 + a * w[w0 + k]) | 0;
          s1 = (s1 + a * w[w1 + k]) | 0;
          s2 = (s2 + a * w[w2 + k]) | 0;
          s3 = (s3 + a * w[w3 + k]) | 0;
        }
        out[oo + n] = fround(fround(s0 - zs) * combinedScale);
        out[oo + n + 1] = fround(fround(s1 - zs) * combinedScale);
        out[oo + n + 2] = fround(fround(s2 - zs) * combinedScale);
        out[oo + n + 3] = fround(fround(s3 - zs) * combinedScale);
      }
      for (; n < N; n++) {
        w0 = n * K; s0 = 0;
        for (k = 0; k < K; k++) s0 = (s0 + qa[ao + k] * w[w0 + k]) | 0;
        out[oo + n] = fround(fround(s0 - zs) * combinedScale);
      }
    }
  }

  // Rows r and r + 1 of matmulQ (N a multiple of 4).
  function matmulQ2x4(qa, rowSums, r, K, w, N, zw, cs, out) {
    var a0o = r * K, a1o = a0o + K, z0 = zw * rowSums[r], z1 = zw * rowSums[r + 1];
    var n, k, w0, w1, w2, w3, x0, x1, y, o;
    var s00, s01, s02, s03, s10, s11, s12, s13;
    for (n = 0; n < N; n += 4) {
      w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
      s00 = 0; s01 = 0; s02 = 0; s03 = 0; s10 = 0; s11 = 0; s12 = 0; s13 = 0;
      for (k = 0; k < K; k++) {
        x0 = qa[a0o + k]; x1 = qa[a1o + k];
        y = w[w0 + k]; s00 = (s00 + x0 * y) | 0; s10 = (s10 + x1 * y) | 0;
        y = w[w1 + k]; s01 = (s01 + x0 * y) | 0; s11 = (s11 + x1 * y) | 0;
        y = w[w2 + k]; s02 = (s02 + x0 * y) | 0; s12 = (s12 + x1 * y) | 0;
        y = w[w3 + k]; s03 = (s03 + x0 * y) | 0; s13 = (s13 + x1 * y) | 0;
      }
      o = r * N + n;
      out[o] = fround(fround(s00 - z0) * cs);
      out[o + 1] = fround(fround(s01 - z0) * cs);
      out[o + 2] = fround(fround(s02 - z0) * cs);
      out[o + 3] = fround(fround(s03 - z0) * cs);
      o += N;
      out[o] = fround(fround(s10 - z1) * cs);
      out[o + 1] = fround(fround(s11 - z1) * cs);
      out[o + 2] = fround(fround(s12 - z1) * cs);
      out[o + 3] = fround(fround(s13 - z1) * cs);
    }
  }

  // Bit casts between float32 and int32.
  var castF32 = new Float32Array(1);
  var castI32 = new Int32Array(castF32.buffer);

  var FLT_LOWEST = -3.4028234663852886e38;

  // MLAS exp constants (compute.cpp, MlasExpConstants).
  var EXP_LOWER_SUMEXP = fround(-88.3762626647949);
  var EXP_ROUNDING_BIAS = 12582912;
  var EXP_LOG2_RECIP = fround(1.44269504088896341);
  var EXP_LOG2_HIGH = fround(-6.93145752e-1);
  var EXP_LOG2_LOW = fround(-1.42860677e-6);
  var EXP_P0 = 0x1694000 / 0x1000000 / 1024;     // 0x1.694000p-10
  var EXP_P1 = 0x1125edc / 0x1000000 / 128;      // 0x1.125edcp-7
  var EXP_P2 = 0x1555b5a / 0x1000000 / 32;       // 0x1.555b5ap-5
  var EXP_P3 = 0x1555450 / 0x1000000 / 8;        // 0x1.555450p-3
  var EXP_P4 = 0x1fffff6 / 0x1000000 / 4;        // 0x1.fffff6p-2
  var EXP_P56 = 1;                               // 0x1.000000p+0
  var EXP_MAX_EXPONENT = 0x3F800000;

  // MlasComputeSumExpVector for one lane: exp(x - max) as MLAS computes it.
  function sumExp1(x, negMax) {
    var v = fround(x + negMax);
    if (!(v >= EXP_LOWER_SUMEXP)) v = EXP_LOWER_SUMEXP;
    var biased = fround(fround(v * EXP_LOG2_RECIP) + EXP_ROUNDING_BIAS);
    var m = fround(biased - EXP_ROUNDING_BIAS);
    v = fround(fround(m * EXP_LOG2_HIGH) + v);
    v = fround(fround(m * EXP_LOG2_LOW) + v);
    castF32[0] = biased;
    castI32[0] = (castI32[0] << 23) + EXP_MAX_EXPONENT;
    var scale = castF32[0];
    var p = EXP_P0;
    p = fround(fround(p * v) + EXP_P1);
    p = fround(fround(p * v) + EXP_P2);
    p = fround(fround(p * v) + EXP_P3);
    p = fround(fround(p * v) + EXP_P4);
    p = fround(fround(p * v) + EXP_P56);
    p = fround(fround(p * v) + EXP_P56);
    return fround(p * scale);
  }

  // MlasComputeSoftmax for one row, in place (float, not log, not smooth).
  function softmaxRow(x, off, d) {
    var i, max = FLT_LOWEST;
    for (i = 0; i < d; i++) if (x[off + i] > max) max = x[off + i];
    var negMax = -max;
    var acc = 0, n = d, p = off;
    if (n >= 4) {
      var a0 = 0, a1 = 0, a2 = 0, a3 = 0, e;
      while (n >= 8) {
        var e0 = sumExp1(x[p], negMax), e1 = sumExp1(x[p + 1], negMax);
        var e2 = sumExp1(x[p + 2], negMax), e3 = sumExp1(x[p + 3], negMax);
        var e4 = sumExp1(x[p + 4], negMax), e5 = sumExp1(x[p + 5], negMax);
        var e6 = sumExp1(x[p + 6], negMax), e7 = sumExp1(x[p + 7], negMax);
        a0 = fround(fround(a0 + e0) + e4);
        a1 = fround(fround(a1 + e1) + e5);
        a2 = fround(fround(a2 + e2) + e6);
        a3 = fround(fround(a3 + e3) + e7);
        x[p] = e0; x[p + 1] = e1; x[p + 2] = e2; x[p + 3] = e3;
        x[p + 4] = e4; x[p + 5] = e5; x[p + 6] = e6; x[p + 7] = e7;
        p += 8; n -= 8;
      }
      while (n >= 4) {
        e = sumExp1(x[p], negMax); a0 = fround(a0 + e); x[p] = e;
        e = sumExp1(x[p + 1], negMax); a1 = fround(a1 + e); x[p + 1] = e;
        e = sumExp1(x[p + 2], negMax); a2 = fround(a2 + e); x[p + 2] = e;
        e = sumExp1(x[p + 3], negMax); a3 = fround(a3 + e); x[p + 3] = e;
        p += 4; n -= 4;
      }
      acc = fround(fround(a0 + a2) + fround(a1 + a3));
    }
    while (n > 0) {
      var t = sumExp1(x[p], negMax);
      acc = fround(acc + t);
      x[p] = t;
      p++; n--;
    }
    var scale = fround(1 / acc);
    for (i = 0; i < d; i++) x[off + i] = fround(scale * x[off + i]);
  }

  /*
   * softmaxRow for a causal attention row of length d whose entries from n on are masked
   * (score = float lowest). MLAS gives those entries exp() = 0 exactly, so they add nothing
   * and are skipped; the unmasked ones are summed with the same lane / tail split as for
   * the full row of length d, which keeps every rounding identical.
   */
  function softmaxCausal(x, n, d) {
    var i, max = FLT_LOWEST;
    for (i = 0; i < n; i++) if (x[i] > max) max = x[i];
    var negMax = -max, acc = 0, e;
    var tail = d >= 4 ? d - (d % 4) : 0;
    if (d >= 4) {
      var a0 = 0, a1 = 0, a2 = 0, a3 = 0, end = n < tail ? n : tail;
      for (i = 0; i + 3 < end; i += 4) {
        e = sumExp1(x[i], negMax); a0 = fround(a0 + e); x[i] = e;
        e = sumExp1(x[i + 1], negMax); a1 = fround(a1 + e); x[i + 1] = e;
        e = sumExp1(x[i + 2], negMax); a2 = fround(a2 + e); x[i + 2] = e;
        e = sumExp1(x[i + 3], negMax); a3 = fround(a3 + e); x[i + 3] = e;
      }
      if (i < end) { e = sumExp1(x[i], negMax); a0 = fround(a0 + e); x[i] = e; i++; }
      if (i < end) { e = sumExp1(x[i], negMax); a1 = fround(a1 + e); x[i] = e; i++; }
      if (i < end) { e = sumExp1(x[i], negMax); a2 = fround(a2 + e); x[i] = e; i++; }
      acc = fround(fround(a0 + a2) + fround(a1 + a3));
    }
    for (i = tail; i < n; i++) {
      e = sumExp1(x[i], negMax);
      acc = fround(acc + e);
      x[i] = e;
    }
    var scale = fround(1 / acc);
    for (i = 0; i < n; i++) x[i] = fround(scale * x[i]);
  }

  // RotaryEmbedding (non-interleaved) for `heads` heads of 64 values starting at x[xo].
  function rope(x, xo, heads, cos, sin, pos) {
    var h, i, base, c, s, a, b;
    var co = pos * 32;
    for (h = 0; h < heads; h++) {
      base = xo + h * 64;
      for (i = 0; i < 32; i++) {
        c = cos[co + i]; s = sin[co + i];
        a = x[base + i]; b = x[base + i + 32];
        x[base + i] = fround(fround(a * c) - fround(b * s));
        x[base + i + 32] = fround(fround(b * c) + fround(a * s));
      }
    }
  }

  /*
   * A shard. `spec` = { config, ranges: { heads:[h0,h1], q:[n0,n1], k, v, o, gate, up, down,
   *   vocab:[v0,v1] }, embed: { scale, zp }, layers: [ { q: {scale, zp}, ... } ] }.
   * The q range covers query heads [h0, h1); k and v cover the key/value heads they use.
   * Weight slices are added with setWeight / addEmbedChunk, the rotary tables with setRope.
   */
  function Shard(spec) {
    this.cfg = spec.config;
    this.ranges = spec.ranges;
    this.layerMeta = spec.layers;
    this.layers = [];
    for (var l = 0; l < spec.layers.length; l++) this.layers.push({});
    this.embedScale = fround(spec.embed.scale);
    this.embedZp = spec.embed.zp;
    this.embedTable = new Float32Array(256);
    for (var i = 0; i < 256; i++) this.embedTable[i] = fround((i - this.embedZp) * this.embedScale);
    this.embedChunks = [];   // { row: first vocab id, rows: count, data: Uint8Array }
    var zpMax = Math.max(this.embedZp, 255 - this.embedZp);
    this.maxAbsW = zpMax * this.embedScale;
    this.kCache = [];
    this.vCache = [];
    this.cacheCap = 0;
  }

  // bytes: Uint8Array of rows [n0, n1) of matrix `name` in layer l, [n][k] layout.
  Shard.prototype.setWeight = function (l, name, bytes) {
    this.layers[l][name] = bytes;
  };

  Shard.prototype.addEmbedChunk = function (row, rows, data) {
    this.embedChunks.push({ row: row, rows: rows, data: data });
  };

  Shard.prototype.setRope = function (cos, sin) {
    this.cos = cos;
    this.sin = sin;
  };

  Shard.prototype._mm = function (l, name, qa, rowSums, rows, K, actScale, out) {
    var r = this.ranges[name], m = this.layerMeta[l][name];
    var N = r[1] - r[0];
    matmulQ(qa, rowSums, rows, K, this.layers[l][name], N, m.zp, fround(actScale * fround(m.scale)), out);
    return out;
  };

  function sliceLen(r) { return r[1] - r[0]; }

  Shard.prototype._ensureCache = function (need) {
    if (need <= this.cacheCap) return;
    var cap = Math.max(need, this.cacheCap ? this.cacheCap * 2 : 256);
    if (cap > this.cfg.maxPos) cap = this.cfg.maxPos;
    var stride = sliceLen(this.ranges.k);
    for (var l = 0; l < this.cfg.layers; l++) {
      var nk = new Float32Array(cap * stride), nv = new Float32Array(cap * stride);
      if (this.kCache[l]) { nk.set(this.kCache[l]); nv.set(this.vCache[l]); }
      this.kCache[l] = nk; this.vCache[l] = nv;
    }
    this.cacheCap = cap;
  };

  /*
   * Attention block for this shard's query heads: q/k/v projections, rotary embedding,
   * KV cache append at positions pos0.., and MultiHeadAttention. Earlier positions come from
   * the cache, so pos0 is also the number of cached tokens (0 starts a new sequence).
   * Returns the attention output [rows x (heads * 64)].
   */
  Shard.prototype.attn = function (l, qa, rowSums, rows, actScale, pos0, hooks) {
    var K = this.cfg.hidden, R = this.ranges, HD = this.cfg.headDim, group = this.cfg.heads / this.cfg.kvHeads;
    var qw = sliceLen(R.q), kw = sliceLen(R.k);
    var nh = qw / HD, nkv = kw / HD, h0 = R.heads[0], kv0 = R.k[0] / HD;
    var q = this._mm(l, 'q', qa, rowSums, rows, K, actScale, new Float32Array(rows * qw));
    var k = this._mm(l, 'k', qa, rowSums, rows, K, actScale, new Float32Array(rows * kw));
    var v = this._mm(l, 'v', qa, rowSums, rows, K, actScale, new Float32Array(rows * kw));
    if (hooks) { hooks('l' + l + '.qproj', q, rows * qw); hooks('l' + l + '.kproj', k, rows * kw); hooks('l' + l + '.vproj', v, rows * kw); }
    var T = pos0 + rows, r, i, j, d, h;
    this._ensureCache(T);
    var kc = this.kCache[l], vc = this.vCache[l];
    for (r = 0; r < rows; r++) {
      rope(q, r * qw, nh, this.cos, this.sin, pos0 + r);
      rope(k, r * kw, nkv, this.cos, this.sin, pos0 + r);
      var co = (pos0 + r) * kw;
      for (i = 0; i < kw; i++) { kc[co + i] = k[r * kw + i]; vc[co + i] = v[r * kw + i]; }
    }
    if (hooks) { hooks('l' + l + '.qrot', q, rows * qw); hooks('l' + l + '.krot', k, rows * kw); }

    var scores = this._scores;
    if (!scores || scores.length < T) scores = this._scores = new Float32Array(Math.max(T, 256));
    var ctx = new Float32Array(rows * qw);
    // probs x V: one SGEMM per head. Decode (1 row) uses the GEMV kernel; prefill uses the
    // blocked kernel, which splits K (= T) into blocks of 256 when T > 64.
    var kBlock = (rows > 1 && T > 64) ? 256 : T;
    var qo, ko, acc, kvo, n, out, b, bEnd, part;
    for (h = 0; h < nh; h++) {
      kvo = (((h0 + h) / group | 0) - kv0) * HD;
      for (r = 0; r < rows; r++) {
        // Keys after this position get the attention bias float lowest: their score is
        // float lowest whatever Q x K is, their probability exactly 0. They are skipped.
        n = pos0 + r + 1;
        qo = r * qw + h * HD;
        // Q x K^T * scale (+ attention bias 0)
        for (j = 0; j < n; j++) {
          ko = j * kw + kvo;
          acc = 0;
          for (d = 0; d < HD; d++) acc = fround(acc + fround(q[qo + d] * kc[ko + d]));
          scores[j] = fround(acc * 0.125);
        }
        softmaxCausal(scores, n, T);
        out = r * qw + h * HD;
        for (d = 0; d < HD; d++) {
          acc = 0;
          for (b = 0; b < n; b = bEnd) {
            bEnd = b + kBlock; if (bEnd > n) bEnd = n;
            part = 0;
            for (j = b; j < bEnd; j++) part = fround(part + fround(scores[j] * vc[j * kw + kvo + d]));
            acc = (b === 0) ? part : fround(part + acc);
          }
          ctx[out + d] = acc;
        }
      }
    }
    return ctx;
  };

  Shard.prototype.o = function (l, qa, rowSums, rows, actScale) {
    return this._mm(l, 'o', qa, rowSums, rows, this.cfg.hidden, actScale, new Float32Array(rows * sliceLen(this.ranges.o)));
  };

  // gate and up projections, then SiLU(gate) * up (gate and up share the same row range).
  Shard.prototype.mlp = function (l, qa, rowSums, rows, actScale) {
    var K = this.cfg.hidden, n = rows * sliceLen(this.ranges.gate);
    var g = this._mm(l, 'gate', qa, rowSums, rows, K, actScale, new Float32Array(n));
    var u = this._mm(l, 'up', qa, rowSums, rows, K, actScale, new Float32Array(n));
    for (var i = 0; i < n; i++) {
      var v = g[i];
      g[i] = fround(fround(v * logistic(v)) * u[i]);
    }
    return g;
  };

  Shard.prototype.down = function (l, qa, rowSums, rows, actScale) {
    return this._mm(l, 'down', qa, rowSums, rows, this.cfg.inter, actScale, new Float32Array(rows * sliceLen(this.ranges.down)));
  };

  // Dequantized embedding rows for the token ids this shard owns.
  // Returns { pos: [index in ids], data: Float32Array(count * hidden) }.
  Shard.prototype.embed = function (ids) {
    var H = this.cfg.hidden, pos = [], rows = [], i, c, ch;
    for (i = 0; i < ids.length; i++) {
      for (c = 0; c < this.embedChunks.length; c++) {
        ch = this.embedChunks[c];
        if (ids[i] >= ch.row && ids[i] < ch.row + ch.rows) { pos.push(i); rows.push([c, ids[i] - ch.row]); break; }
      }
    }
    var data = new Float32Array(pos.length * H), tbl = this.embedTable;
    for (i = 0; i < rows.length; i++) {
      var d = this.embedChunks[rows[i][0]].data, off = rows[i][1] * H;
      for (var k = 0; k < H; k++) data[i * H + k] = tbl[d[off + k]];
    }
    return { pos: pos, data: data };
  };

  /*
   * lm_head logit exactly as ONNX Runtime computes it: MatMul of the final hidden state with
   * the dequantized, transposed embedding. One row (decode) uses the MLAS GEMV kernel,
   * a plain sequential float32 sum; several rows (prefill) use the blocked SGEMM kernel,
   * which sums K in blocks of 128.
   */
  Shard.prototype._exactLogit = function (h, data, off, blocked) {
    var H = this.cfg.hidden, tbl = this.embedTable, kBlock = blocked ? 128 : H;
    var acc = 0, b, k, end, blk;
    for (b = 0; b < H; b += kBlock) {
      end = b + kBlock < H ? b + kBlock : H;
      blk = 0;
      for (k = b; k < end; k++) blk = fround(blk + fround(h[k] * tbl[data[off + k]]));
      acc = (b === 0) ? blk : fround(blk + acc);
    }
    return acc;
  };

  /*
   * Greedy candidates for the next token from this shard's vocabulary slice.
   * h: final hidden state (Float32Array), blocked: prefill flag, seen: ids that get the
   * repetition penalty, penalty: e.g. 1.12.
   *
   * Every logit is first estimated in double precision with a rigorous bound on the
   * difference to the float32 result; only tokens that can still be the maximum after the
   * penalty are evaluated exactly. Returns { ids: [...], vals: [...] } with exact,
   * penalized logits; any token not returned is strictly below the best returned one.
   */
  Shard.prototype.candidates = function (h, blocked, seen, penalty) {
    var H = this.cfg.hidden, i, k, c, v;
    var v0 = this.ranges.vocab[0], v1 = this.ranges.vocab[1];
    var flags = new Uint8Array(v1 - v0);
    for (i = 0; i < seen.length; i++) if (seen[i] >= v0 && seen[i] < v1) flags[seen[i] - v0] = 1;
    var hd = new Float64Array(H), sumH = 0, sumAbsH = 0;
    for (k = 0; k < H; k++) { hd[k] = h[k]; sumH += h[k]; sumAbsH += Math.abs(h[k]); }
    // |float32 sequential sum - exact| <= gamma_(K+2) * sum|h_k w_k|; plus slack for the
    // double-precision estimate and for underflow.
    var g = (H + 4) * U;
    var err = (g / (1 - g)) * this.maxAbsW * sumAbsH * 1.000001 + 1e-30;
    var scale = this.embedScale, zp = this.embedZp;
    var est = new Float64Array(v1 - v0), lo, hi, best = -Infinity;
    for (c = 0; c < this.embedChunks.length; c++) {
      var ch = this.embedChunks[c], d = ch.data, base = ch.row - v0;
      for (v = 0; v < ch.rows; v++) {
        var off = v * H, s0 = 0, s1 = 0;
        for (k = 0; k < H; k += 2) { s0 += hd[k] * d[off + k]; s1 += hd[k + 1] * d[off + k + 1]; }
        var e = scale * (s0 + s1 - zp * sumH);
        est[base + v] = e;
        lo = e - err;
        if (flags[base + v]) lo = lo < 0 ? fround(lo * penalty) : fround(lo / penalty);
        if (lo > best) best = lo;
      }
    }
    var ids = [], vals = [];
    for (c = 0; c < this.embedChunks.length; c++) {
      ch = this.embedChunks[c]; base = ch.row - v0;
      for (v = 0; v < ch.rows; v++) {
        hi = est[base + v] + err;
        if (flags[base + v]) hi = hi < 0 ? fround(hi * penalty) : fround(hi / penalty);
        if (hi < best) continue;
        var x = this._exactLogit(h, ch.data, v * H, blocked);
        if (flags[base + v]) x = x < 0 ? fround(x * penalty) : fround(x / penalty);
        ids.push(ch.row + v);
        vals.push(x);
      }
    }
    return { ids: ids, vals: vals };
  };

  // Exact logits of the whole slice (for tests).
  Shard.prototype.logits = function (h, blocked) {
    var v0 = this.ranges.vocab[0], out = new Float32Array(this.ranges.vocab[1] - v0), H = this.cfg.hidden;
    for (var c = 0; c < this.embedChunks.length; c++) {
      var ch = this.embedChunks[c];
      for (var v = 0; v < ch.rows; v++) out[ch.row - v0 + v] = this._exactLogit(h, ch.data, v * H, blocked);
    }
    return out;
  };

  // Message handler shared by the Web Worker and Node worker_threads wrappers.
  // Returns [reply, transferList].
  Shard.handle = function (state, msg) {
    var s = state.shard, r;
    switch (msg.op) {
      case 'init':
        state.shard = new Shard(msg.spec);
        return [{ op: 'init' }, []];
      case 'weight':
        s.setWeight(msg.l, msg.name, new Uint8Array(msg.buf));
        return null;
      case 'embedChunk':
        s.addEmbedChunk(msg.row, msg.rows, new Uint8Array(msg.buf));
        return null;
      case 'ready':
        return [{ op: 'ready' }, []];
      case 'rope':
        s.setRope(new Float32Array(msg.cos), new Float32Array(msg.sin));
        return null;
      case 'attn':
        r = s.attn(msg.l, msg.qa, msg.rowSums, msg.rows, msg.scale, msg.pos0);
        return [{ op: 'attn', out: r }, [r.buffer]];
      case 'o':
        r = s.o(msg.l, msg.qa, msg.rowSums, msg.rows, msg.scale);
        return [{ op: 'o', out: r }, [r.buffer]];
      case 'mlp':
        r = s.mlp(msg.l, msg.qa, msg.rowSums, msg.rows, msg.scale);
        return [{ op: 'mlp', out: r }, [r.buffer]];
      case 'down':
        r = s.down(msg.l, msg.qa, msg.rowSums, msg.rows, msg.scale);
        return [{ op: 'down', out: r }, [r.buffer]];
      case 'embed':
        r = s.embed(msg.ids);
        return [{ op: 'embed', pos: r.pos, data: r.data }, [r.data.buffer]];
      case 'cand':
        r = s.candidates(msg.h, msg.blocked, msg.seen, msg.penalty);
        return [{ op: 'cand', ids: r.ids, vals: r.vals }, []];
    }
    throw new Error('Unknown op ' + msg.op);
  };

  var api = { Shard: Shard, matmulQ: matmulQ, logistic: logistic, fround: fround, _softmaxRow: softmaxRow };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SmolLMShard = api;
})(typeof self !== 'undefined' ? self : this);
