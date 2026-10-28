/*
 * paraphrase-multilingual-MiniLM-L12-v2 q8 compute shard: holds a slice of the output columns
 * of every linear layer (and the attention heads that go with its q/k/v columns) and runs the
 * matmul-heavy parts of the model on it.
 *
 * One shard covering everything runs in the same thread when there is a single thread;
 * otherwise each Web Worker (minilm-compute.js) holds one shard. A shard computes whole
 * output values, so the results are bit-identical with any number of shards.
 *
 * The arithmetic follows ONNX Runtime (MLAS, WebAssembly SIMD build) exactly:
 *   - MatMulIntegerToFloat / DynamicQuantizeMatMul: exact integer sums, then
 *     float(sum) * float(activation scale * column scale), then + bias
 *   - FusedMatMul(Q, K^T, alpha) and probs x V: SGEMM, one float32 product and sum per step,
 *     K never split (K = 32, and K = sequence length <= 512 with N = 32 gives StrideK = 512)
 *   - Softmax: MlasComputeSoftmax with its exp polynomial and 4-lane sums
 *   - BiasGelu: MlasErfKernel (vector path) and 0.5 * x * (erf(x * sqrt(1/2)) + 1)
 *
 * ES5 + typed arrays only.
 */
(function (root) {
  'use strict';

  var fround = Math.fround || (function () {
    var f = new Float32Array(1);
    return function (x) { f[0] = x; return f[0]; };
  })();

  var FLT_LOWEST = -3.4028234663852886e38;

  /*
   * Integer part of a quantized linear layer on a column slice: sums[r][n] = sum_k qa[r][k] *
   * w[n][k]. qa holds q - zero_point of the activation, w: Int8Array [n][k] (n a multiple of 4).
   * Four rows x four columns per step share every load. Two variants give the same exact
   * integers: int32 arithmetic (qa: Int32Array) and double arithmetic (qa: Float64Array;
   * |sums| < 2^26, far below 2^53). Which one is faster depends on the CPU and the JavaScript
   * engine, so each shard times both (pickKernel).
   */
  function sumsInt(qa, rows, K, w, N, sums) {
    var n, r, k, a0, a1, a2, a3, w0, w1, w2, w3, x0, x1, x2, x3, y, o;
    var s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33;
    for (r = 0; r + 3 < rows; r += 4) {
      a0 = r * K; a1 = a0 + K; a2 = a1 + K; a3 = a2 + K;
      for (n = 0; n < N; n += 4) {
        w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
        s00 = 0; s01 = 0; s02 = 0; s03 = 0; s10 = 0; s11 = 0; s12 = 0; s13 = 0;
        s20 = 0; s21 = 0; s22 = 0; s23 = 0; s30 = 0; s31 = 0; s32 = 0; s33 = 0;
        for (k = 0; k < K; k++) {
          x0 = qa[a0 + k]; x1 = qa[a1 + k]; x2 = qa[a2 + k]; x3 = qa[a3 + k];
          y = w[w0 + k]; s00 = (s00 + x0 * y) | 0; s10 = (s10 + x1 * y) | 0; s20 = (s20 + x2 * y) | 0; s30 = (s30 + x3 * y) | 0;
          y = w[w1 + k]; s01 = (s01 + x0 * y) | 0; s11 = (s11 + x1 * y) | 0; s21 = (s21 + x2 * y) | 0; s31 = (s31 + x3 * y) | 0;
          y = w[w2 + k]; s02 = (s02 + x0 * y) | 0; s12 = (s12 + x1 * y) | 0; s22 = (s22 + x2 * y) | 0; s32 = (s32 + x3 * y) | 0;
          y = w[w3 + k]; s03 = (s03 + x0 * y) | 0; s13 = (s13 + x1 * y) | 0; s23 = (s23 + x2 * y) | 0; s33 = (s33 + x3 * y) | 0;
        }
        o = r * N + n;
        sums[o] = s00; sums[o + 1] = s01; sums[o + 2] = s02; sums[o + 3] = s03; o += N;
        sums[o] = s10; sums[o + 1] = s11; sums[o + 2] = s12; sums[o + 3] = s13; o += N;
        sums[o] = s20; sums[o + 1] = s21; sums[o + 2] = s22; sums[o + 3] = s23; o += N;
        sums[o] = s30; sums[o + 1] = s31; sums[o + 2] = s32; sums[o + 3] = s33;
      }
    }
    for (; r < rows; r++) {
      a0 = r * K;
      for (n = 0; n < N; n += 4) {
        w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
        s00 = 0; s01 = 0; s02 = 0; s03 = 0;
        for (k = 0; k < K; k++) {
          x0 = qa[a0 + k];
          s00 = (s00 + x0 * w[w0 + k]) | 0;
          s01 = (s01 + x0 * w[w1 + k]) | 0;
          s02 = (s02 + x0 * w[w2 + k]) | 0;
          s03 = (s03 + x0 * w[w3 + k]) | 0;
        }
        o = r * N + n;
        sums[o] = s00; sums[o + 1] = s01; sums[o + 2] = s02; sums[o + 3] = s03;
      }
    }
  }

  function sumsF64(qa, rows, K, w, N, sums) {
    var n, r, k, a0, a1, a2, a3, w0, w1, w2, w3, x0, x1, x2, x3, y, o;
    var s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33;
    for (r = 0; r + 3 < rows; r += 4) {
      a0 = r * K; a1 = a0 + K; a2 = a1 + K; a3 = a2 + K;
      for (n = 0; n < N; n += 4) {
        w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
        s00 = 0; s01 = 0; s02 = 0; s03 = 0; s10 = 0; s11 = 0; s12 = 0; s13 = 0;
        s20 = 0; s21 = 0; s22 = 0; s23 = 0; s30 = 0; s31 = 0; s32 = 0; s33 = 0;
        for (k = 0; k < K; k++) {
          x0 = qa[a0 + k]; x1 = qa[a1 + k]; x2 = qa[a2 + k]; x3 = qa[a3 + k];
          y = w[w0 + k]; s00 += x0 * y; s10 += x1 * y; s20 += x2 * y; s30 += x3 * y;
          y = w[w1 + k]; s01 += x0 * y; s11 += x1 * y; s21 += x2 * y; s31 += x3 * y;
          y = w[w2 + k]; s02 += x0 * y; s12 += x1 * y; s22 += x2 * y; s32 += x3 * y;
          y = w[w3 + k]; s03 += x0 * y; s13 += x1 * y; s23 += x2 * y; s33 += x3 * y;
        }
        o = r * N + n;
        sums[o] = s00; sums[o + 1] = s01; sums[o + 2] = s02; sums[o + 3] = s03; o += N;
        sums[o] = s10; sums[o + 1] = s11; sums[o + 2] = s12; sums[o + 3] = s13; o += N;
        sums[o] = s20; sums[o + 1] = s21; sums[o + 2] = s22; sums[o + 3] = s23; o += N;
        sums[o] = s30; sums[o + 1] = s31; sums[o + 2] = s32; sums[o + 3] = s33;
      }
    }
    for (; r < rows; r++) {
      a0 = r * K;
      for (n = 0; n < N; n += 4) {
        w0 = n * K; w1 = w0 + K; w2 = w1 + K; w3 = w2 + K;
        s00 = 0; s01 = 0; s02 = 0; s03 = 0;
        for (k = 0; k < K; k++) {
          x0 = qa[a0 + k];
          s00 += x0 * w[w0 + k];
          s01 += x0 * w[w1 + k];
          s02 += x0 * w[w2 + k];
          s03 += x0 * w[w3 + k];
        }
        o = r * N + n;
        sums[o] = s00; sums[o + 1] = s01; sums[o + 2] = s02; sums[o + 3] = s03;
      }
    }
  }

  // Time both kernels on this device (about 0.2 s) and return the faster one's name.
  function pickKernel() {
    var K = 384, N = 32, rows = 8, i, w = new Int8Array(N * K), qi = new Int32Array(rows * K);
    var qd = new Float64Array(rows * K), sums = new Float64Array(rows * N);
    for (i = 0; i < w.length; i++) w[i] = ((i * 7919) % 255) - 127;
    for (i = 0; i < qi.length; i++) { qi[i] = ((i * 104729) % 511) - 255; qd[i] = qi[i]; }
    function time(f, q) {
      var reps = 0, t0 = Date.now(), t;
      do { f(q, rows, K, w, N, sums); reps++; t = Date.now() - t0; } while (t < 25);
      return t / reps;
    }
    time(sumsInt, qi); time(sumsF64, qd);              // warm up / compile
    var a = Math.min(time(sumsInt, qi), time(sumsInt, qi));
    var b = Math.min(time(sumsF64, qd), time(sumsF64, qd));
    return b < a ? 'f64' : 'int';
  }

  /*
   * Quantized linear layer on a column slice: out[r][n] = bias[n] + float(sum_k qa[r][k] *
   * (w[n][k] - zw[n])) * float(actScale * wScale[n]), as MatMulIntegerToFloat /
   * DynamicQuantizeMatMul (MLAS scale processor) followed by Add(bias).
   * qa: Int32Array [rows x K] (q - zero_point), rowSums: their per-row sums,
   * lin: { k, n, w: Int8Array [n][k], scale: Float32Array, zp: Int8Array, bias: Float32Array }.
   */
  function linear(qa, rowSums, rows, lin, actScale, out, kernel) {
    var K = lin.k, N = lin.n, zw = lin.zp, sc = lin.scale, bias = lin.bias, sums = new Float64Array(rows * N);
    var mult = new Float32Array(N), n, r, o;
    for (n = 0; n < N; n++) mult[n] = fround(actScale * sc[n]);
    if (sumsHook) sumsHook(qa, rows, K, lin, N, sums);   // Zagreus: WebAssembly SIMD (minilm-wasm.js)
    else if (kernel === 'f64') sumsF64(new Float64Array(qa.subarray(0, rows * K)), rows, K, lin.w, N, sums);
    else sumsInt(qa, rows, K, lin.w, N, sums);
    for (r = 0; r < rows; r++) {
      o = r * N;
      for (n = 0; n < N; n++) out[o + n] = fround(bias[n] + fround(fround(sums[o + n] - zw[n] * rowSums[r]) * mult[n]));
    }
    return out;
  }

  // Bit casts between float32 and int32.
  var castF32 = new Float32Array(1);
  var castI32 = new Int32Array(castF32.buffer);

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

  // MLAS erf constants (erf.cpp, MlasErfConstants).
  var ERF_UPPER = fround(3.925), ERF_SPLIT = fround(0.921875);
  var ERF_S0 = fround(-5.99104969e-4), ERF_S1 = fround(4.99339588e-3), ERF_S2 = fround(-2.67667342e-2);
  var ERF_S3 = fround(1.12818025e-1), ERF_S4 = fround(-3.76124859e-1), ERF_S5 = fround(1.28379151e-1);
  var ERF_B0 = fround(1.72948930e-5), ERF_B1 = fround(-3.83208680e-4), ERF_B2 = fround(3.88393435e-3);
  var ERF_B3 = fround(-2.42545605e-2), ERF_B4 = fround(1.06777847e-1), ERF_B5 = fround(6.34846687e-1);
  var ERF_B6 = fround(1.28717512e-1);
  var ERF_EXP_LOWER = fround(-88.3762626647949), ERF_LOG2R = fround(1.44269504088896341);
  var ERF_LOG2_HI = fround(-6.93145752e-1), ERF_LOG2_LO = fround(-1.42860677e-6);
  var ERF_P0 = fround(1.38319808e-3), ERF_P1 = fround(8.37550033e-3), ERF_P2 = fround(4.16689515e-2);
  var ERF_P3 = fround(1.66664466e-1), ERF_P4 = fround(4.99999851e-1), ERF_C = 12582912;

  // MlasErfKernel, vector path (the GELU rows have 1536 values, a multiple of 4).
  function erf(x) {
    var neg = x < 0 || (x === 0 && 1 / x < 0);
    var a = neg ? -x : x;
    if (!(a <= ERF_UPPER)) a = ERF_UPPER;
    var r;
    if (a > ERF_SPLIT) {
      var rb = ERF_B0;
      rb = fround(fround(rb * a) + ERF_B1);
      rb = fround(fround(rb * a) + ERF_B2);
      rb = fround(fround(rb * a) + ERF_B3);
      rb = fround(fround(rb * a) + ERF_B4);
      rb = fround(fround(rb * a) + ERF_B5);
      rb = fround(fround(rb * a) + ERF_B6);
      rb = fround(fround(rb * a) + a);
      rb = -rb;
      if (rb < ERF_EXP_LOWER) rb = ERF_EXP_LOWER;
      var m = fround(fround(ERF_LOG2R * rb) + ERF_C);
      m = fround(m - ERF_C);
      var fx = fround(fround(m * ERF_LOG2_HI) + rb);
      fx = fround(fround(m * ERF_LOG2_LO) + fx);
      var y = ERF_P0;
      y = fround(fround(y * fx) + ERF_P1);
      y = fround(fround(y * fx) + ERF_P2);
      y = fround(fround(y * fx) + ERF_P3);
      y = fround(fround(y * fx) + ERF_P4);
      y = fround(fround(y * fx) + 1);
      y = fround(fround(y * fx) + 1);
      castI32[0] = ((m | 0) + 127) << 23;
      y = fround(y * castF32[0]);
      r = fround(1 - y);
    } else {
      var sq = fround(a * a);
      r = ERF_S0;
      r = fround(fround(r * sq) + ERF_S1);
      r = fround(fround(r * sq) + ERF_S2);
      r = fround(fround(r * sq) + ERF_S3);
      r = fround(fround(r * sq) + ERF_S4);
      r = fround(fround(r * sq) + ERF_S5);
      r = fround(fround(r * a) + a);
    }
    return neg ? -r : r;
  }

  var SQRT1_2 = fround(0.7071067811865476);

  // BiasGelu (erf variant) on values that already hold input + bias, in place.
  function gelu(x, len) {
    var i, v;
    for (i = 0; i < len; i++) {
      v = x[i];
      x[i] = fround(fround(v * 0.5) * fround(erf(fround(v * SQRT1_2)) + 1));
    }
  }

  /*
   * A shard. spec = { config, ranges: { heads: [h0, h1], o: [n0, n1], inter: [..], out: [..] },
   * kernel: 'int' | 'f64' | 'auto' }.
   * Weight slices come with setLinear(l, name, slice); name is q, k, v, o, inter or out and
   * slice = { k, n, w, scale, zp, bias } for the columns of this shard.
   */
  function Shard(spec) {
    this.cfg = spec.config;
    this.ranges = spec.ranges;
    this.alpha = fround(spec.config.alpha);
    this.kernel = (spec.kernel === 'int' || spec.kernel === 'f64') ? spec.kernel : pickKernel();
    this.layers = [];
    for (var l = 0; l < spec.config.layers; l++) this.layers.push({});
  }

  Shard.prototype.setLinear = function (l, name, slice) {
    this.layers[l][name] = slice;
  };

  /*
   * q/k/v projections and attention for this shard's heads, for a batch of B sequences of
   * T tokens. maskAdd: Float32Array [B * T] = (1 - attention mask) * float lowest.
   * Returns the attention output [rows x (heads * 32)] of these heads.
   */
  Shard.prototype.attn = function (l, qa, rowSums, rows, actScale, B, T, maskAdd) {
    var L = this.layers[l], HD = this.cfg.headDim;
    var W = L.q.n, nh = W / HD;
    var q = linear(qa, rowSums, rows, L.q, actScale, new Float32Array(rows * W), this.kernel);
    var k = linear(qa, rowSums, rows, L.k, actScale, new Float32Array(rows * W), this.kernel);
    var v = linear(qa, rowSums, rows, L.v, actScale, new Float32Array(rows * W), this.kernel);
    var ctx = new Float32Array(rows * W), scores = new Float32Array(T), vt = new Float32Array(T * HD);
    var alpha = this.alpha, b, h, i, j, d, base, qo, ko, acc;
    for (b = 0; b < B; b++) {
      base = b * T;
      for (h = 0; h < nh; h++) {
        // V of this head, transposed to [d][j] for a contiguous inner loop
        for (j = 0; j < T; j++) for (d = 0; d < HD; d++) vt[d * T + j] = v[(base + j) * W + h * HD + d];
        for (i = 0; i < T; i++) {
          qo = (base + i) * W + h * HD;
          for (j = 0; j < T; j++) {
            ko = (base + j) * W + h * HD;
            acc = 0;
            for (d = 0; d < HD; d++) acc = fround(acc + fround(q[qo + d] * k[ko + d]));
            scores[j] = fround(fround(acc * alpha) + maskAdd[base + j]);
          }
          softmaxRow(scores, 0, T);
          for (d = 0; d < HD; d++) {
            acc = 0;
            ko = d * T;
            for (j = 0; j < T; j++) acc = fround(acc + fround(scores[j] * vt[ko + j]));
            ctx[qo + d] = acc;
          }
        }
      }
    }
    return ctx;
  };

  Shard.prototype.linear = function (l, name, qa, rowSums, rows, actScale) {
    var lin = this.layers[l][name];
    var out = linear(qa, rowSums, rows, lin, actScale, new Float32Array(rows * lin.n), this.kernel);
    // the intermediate layer is followed by BiasGelu; linear() already added the bias
    if (name === 'inter') gelu(out, rows * lin.n);
    return out;
  };

  // Message handler shared by the Web Worker and Node worker_threads wrappers.
  // Returns [reply, transferList] or null.
  Shard.handle = function (state, msg) {
    var s = state.shard, r;
    switch (msg.op) {
      case 'init':
        state.shard = new Shard(msg.spec);
        return [{ op: 'init', kernel: state.shard.kernel }, []];
      case 'linearWeights':
        s.setLinear(msg.l, msg.name, {
          k: msg.k, n: msg.n, w: new Int8Array(msg.w), scale: new Float32Array(msg.scale),
          zp: new Int8Array(msg.zp), bias: new Float32Array(msg.bias)
        });
        return null;
      case 'ready':
        return [{ op: 'ready' }, []];
      case 'attn':
        r = s.attn(msg.l, msg.qa, msg.rowSums, msg.rows, msg.scale, msg.B, msg.T, msg.maskAdd);
        return [{ op: 'attn', out: r }, [r.buffer]];
      case 'linear':
        r = s.linear(msg.l, msg.name, msg.qa, msg.rowSums, msg.rows, msg.scale);
        return [{ op: 'linear', out: r }, [r.buffer]];
    }
    throw new Error('Unknown op ' + msg.op);
  };

  // Zagreus: an alternative implementation of the exact integer sums (same results).
  var sumsHook = null;
  function setSumsHook(f) { sumsHook = f; }

  var api = { Shard: Shard, linear: linear, pickKernel: pickKernel, erf: erf, softmaxRow: softmaxRow, fround: fround, setSumsHook: setSumsHook };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MiniLMShard = api;
})(typeof self !== 'undefined' ? self : this);
