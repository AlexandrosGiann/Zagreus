//! Zagreus WebAssembly kernels.
//!
//! Built twice (see build.sh): once with `+simd128` for current browsers and once without
//! for engines that have WebAssembly but no SIMD. JavaScript picks the build at run time and
//! falls back to the pure-JS code paths when WebAssembly is missing altogether.
//!
//! What lives here:
//!  * `qmat_matvec`: the int8 x int8 matrix-vector product of the Whistle speech model
//!    (Cactus Quants weights, Hadamard-rotated int8 activations). The per-group dot product is
//!    exact integer arithmetic and the float accumulation follows whistle.js term by term, so
//!    the output is bit-identical to the JavaScript kernel.
//!  * A streaming resampler and a voice-activity detector for the call mode.
//!  * Cosine top-k over a matrix of L2-normalised embeddings (memory / intent search).

#![no_std]

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    #[cfg(target_arch = "wasm32")]
    unreachable();
    #[cfg(not(target_arch = "wasm32"))]
    loop {}
}

// ------------------------------------------------------------------ memory

const PAGE: usize = 65536;
static mut HEAP_TOP: usize = 0;

extern "C" {
    static __heap_base: u8;
}

fn heap_start() -> usize {
    unsafe { &__heap_base as *const u8 as usize }
}

/// Bump allocation, 16-byte aligned. Grows linear memory when needed. Returns 0 on failure.
#[no_mangle]
pub extern "C" fn zg_alloc(bytes: usize) -> usize {
    unsafe {
        if HEAP_TOP == 0 {
            HEAP_TOP = (heap_start() + 15) & !15;
        }
        let p = (HEAP_TOP + 15) & !15;
        let end = p + bytes;
        let have = memory_size(0) * PAGE;
        if end > have {
            let need = (end - have + PAGE - 1) / PAGE;
            if memory_grow(0, need) == usize::MAX {
                return 0;
            }
        }
        HEAP_TOP = end;
        p
    }
}

/// Current allocation mark (to release scratch space later with `zg_reset`).
#[no_mangle]
pub extern "C" fn zg_mark() -> usize {
    unsafe {
        if HEAP_TOP == 0 {
            HEAP_TOP = (heap_start() + 15) & !15;
        }
        HEAP_TOP
    }
}

#[no_mangle]
pub extern "C" fn zg_reset(mark: usize) {
    unsafe {
        if mark >= heap_start() && mark <= HEAP_TOP {
            HEAP_TOP = mark;
        }
    }
}

#[no_mangle]
pub extern "C" fn zg_simd() -> i32 {
    if cfg!(target_feature = "simd128") { 1 } else { 0 }
}

// ------------------------------------------------------------------ Whistle int8 matvec

#[inline(always)]
unsafe fn dot_i8(w: *const i8, q: *const i8, n: usize) -> i32 {
    #[cfg(target_feature = "simd128")]
    {
        let mut acc = i32x4_splat(0);
        let mut k = 0usize;
        while k + 16 <= n {
            let wv = v128_load(w.add(k) as *const v128);
            let qv = v128_load(q.add(k) as *const v128);
            let wl = i16x8_extend_low_i8x16(wv);
            let wh = i16x8_extend_high_i8x16(wv);
            let ql = i16x8_extend_low_i8x16(qv);
            let qh = i16x8_extend_high_i8x16(qv);
            acc = i32x4_add(acc, i32x4_dot_i16x8(wl, ql));
            acc = i32x4_add(acc, i32x4_dot_i16x8(wh, qh));
            k += 16;
        }
        let mut d = i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc);
        while k < n {
            d += (*w.add(k) as i32) * (*q.add(k) as i32);
            k += 1;
        }
        d
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let mut d = 0i32;
        let mut k = 0usize;
        while k + 4 <= n {
            d += (*w.add(k) as i32) * (*q.add(k) as i32)
                + (*w.add(k + 1) as i32) * (*q.add(k + 1) as i32)
                + (*w.add(k + 2) as i32) * (*q.add(k + 2) as i32)
                + (*w.add(k + 3) as i32) * (*q.add(k + 3) as i32);
            k += 4;
        }
        while k < n {
            d += (*w.add(k) as i32) * (*q.add(k) as i32);
            k += 1;
        }
        d
    }
}

/// out[r] = sum_g dot(W[r, g], q[g]) * nf[r, g] * s[g]   for r in 0..rows
///
/// `w` points at row 0 of the slice (rows x in_pad int8), `nf` at its row 0 group norms
/// (rows x ng f32). Accumulation is in f64 in the same order as whistle.js
/// (`acc += d * nf[ni++] * s[g]`), so results match the JS kernel exactly.
#[no_mangle]
pub unsafe extern "C" fn qmat_matvec(
    w: *const i8,
    nf: *const f32,
    q: *const i8,
    s: *const f32,
    out: *mut f64,
    rows: usize,
    in_pad: usize,
    g: usize,
    ng: usize,
) {
    let mut wr = w;
    let mut ni = nf;
    for r in 0..rows {
        let mut acc = 0f64;
        for gi in 0..ng {
            let d = dot_i8(wr.add(gi * g), q.add(gi * g), g);
            acc += (d as f64) * (*ni as f64) * (*s.add(gi) as f64);
            ni = ni.add(1);
        }
        *out.add(r) = acc;
        wr = wr.add(in_pad);
    }
}

// ------------------------------------------------------------------ SmolLM2 q8 matmul

#[inline(always)]
unsafe fn dot_i16_u8(a: *const i16, w: *const u8, k: usize) -> i32 {
    #[cfg(target_feature = "simd128")]
    {
        let mut acc0 = i32x4_splat(0);
        let mut acc1 = i32x4_splat(0);
        let mut i = 0usize;
        while i + 16 <= k {
            let a0 = v128_load(a.add(i) as *const v128);
            let a1 = v128_load(a.add(i + 8) as *const v128);
            let w0 = u16x8_load_extend_u8x8(w.add(i));
            let w1 = u16x8_load_extend_u8x8(w.add(i + 8));
            acc0 = i32x4_add(acc0, i32x4_dot_i16x8(a0, w0));
            acc1 = i32x4_add(acc1, i32x4_dot_i16x8(a1, w1));
            i += 16;
        }
        let acc = i32x4_add(acc0, acc1);
        let mut s = i32x4_extract_lane::<0>(acc)
            .wrapping_add(i32x4_extract_lane::<1>(acc))
            .wrapping_add(i32x4_extract_lane::<2>(acc))
            .wrapping_add(i32x4_extract_lane::<3>(acc));
        while i < k {
            s = s.wrapping_add((*a.add(i) as i32) * (*w.add(i) as i32));
            i += 1;
        }
        s
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let mut s = 0i32;
        for i in 0..k {
            s = s.wrapping_add((*a.add(i) as i32) * (*w.add(i) as i32));
        }
        s
    }
}

/// The `matmulQ` of smollm-shard.js: out[r][n] = f32(f32(sum_k qa[r][k] * w[n][k] - zw * rowSums[r]) * cs)
/// with qa the zero-point-shifted activation (stored here as i16) and w the uint8 weight rows.
/// The integer sums are exact (as the JS `|0` sums are), so the output is bit-identical.
#[no_mangle]
pub unsafe extern "C" fn smol_matmul(
    qa: *const i16,
    row_sums: *const f64,
    rows: usize,
    k: usize,
    w: *const u8,
    n: usize,
    zw: f64,
    cs: f32,
    out: *mut f32,
) {
    for r in 0..rows {
        let a = qa.add(r * k);
        let zs = zw * *row_sums.add(r);
        let o = out.add(r * n);
        for j in 0..n {
            let s = dot_i16_u8(a, w.add(j * k), k);
            let d = ((s as f64) - zs) as f32;
            *o.add(j) = ((d as f64) * (cs as f64)) as f32;
        }
    }
}

// ------------------------------------------------------------------ MiniLM q8 integer sums

/// sums[r][n] = sum_k qa[r][k] * w[n][k]   (qa: i16 zero-point-shifted activation, w: int8 rows).
/// The integer part of minilm-shard.js `linear`; written as f64 like the JS Float64Array.
#[no_mangle]
pub unsafe extern "C" fn int_sums(qa: *const i16, rows: usize, k: usize, w: *const i8, n: usize, sums: *mut f64) {
    for r in 0..rows {
        let a = qa.add(r * k);
        for j in 0..n {
            let wr = w.add(j * k);
            #[cfg(target_feature = "simd128")]
            let s = {
                let mut acc0 = i32x4_splat(0);
                let mut acc1 = i32x4_splat(0);
                let mut i = 0usize;
                while i + 16 <= k {
                    acc0 = i32x4_add(acc0, i32x4_dot_i16x8(v128_load(a.add(i) as *const v128), i16x8_load_extend_i8x8(wr.add(i))));
                    acc1 = i32x4_add(acc1, i32x4_dot_i16x8(v128_load(a.add(i + 8) as *const v128), i16x8_load_extend_i8x8(wr.add(i + 8))));
                    i += 16;
                }
                let acc = i32x4_add(acc0, acc1);
                let mut s = i32x4_extract_lane::<0>(acc)
                    .wrapping_add(i32x4_extract_lane::<1>(acc))
                    .wrapping_add(i32x4_extract_lane::<2>(acc))
                    .wrapping_add(i32x4_extract_lane::<3>(acc));
                while i < k {
                    s = s.wrapping_add((*a.add(i) as i32) * (*wr.add(i) as i32));
                    i += 1;
                }
                s
            };
            #[cfg(not(target_feature = "simd128"))]
            let s = {
                let mut s = 0i32;
                for i in 0..k {
                    s = s.wrapping_add((*a.add(i) as i32) * (*wr.add(i) as i32));
                }
                s
            };
            *sums.add(r * n + j) = s as f64;
        }
    }
}

// ------------------------------------------------------------------ embeddings

/// scores[i] = dot(query, mat[i]) for i in 0..n; vectors are dim floats, L2-normalised.
#[no_mangle]
pub unsafe extern "C" fn cos_scores(query: *const f32, mat: *const f32, n: usize, dim: usize, scores: *mut f32) {
    for i in 0..n {
        let row = mat.add(i * dim);
        #[cfg(target_feature = "simd128")]
        let v = {
            let mut acc = f32x4_splat(0.0);
            let mut k = 0usize;
            while k + 4 <= dim {
                acc = f32x4_add(
                    acc,
                    f32x4_mul(v128_load(query.add(k) as *const v128), v128_load(row.add(k) as *const v128)),
                );
                k += 4;
            }
            let mut t = f32x4_extract_lane::<0>(acc)
                + f32x4_extract_lane::<1>(acc)
                + f32x4_extract_lane::<2>(acc)
                + f32x4_extract_lane::<3>(acc);
            while k < dim {
                t += *query.add(k) * *row.add(k);
                k += 1;
            }
            t
        };
        #[cfg(not(target_feature = "simd128"))]
        let v = {
            let mut t = 0f32;
            for k in 0..dim {
                t += *query.add(k) * *row.add(k);
            }
            t
        };
        *scores.add(i) = v;
    }
}

// ------------------------------------------------------------------ resampler

// Windowed-sinc (Blackman) fractional resampler with a small history, enough to bring
// 44.1/48 kHz microphone audio down to the 16 kHz the speech model expects.
const RS_TAPS: usize = 16; // taps on each side
const RS_HIST: usize = 4096;
const RS_PHASES: usize = 256; // fractional positions with precomputed filter coefficients
const RS_W: usize = 2 * RS_TAPS;

struct Resampler {
    ratio: f64, // in_rate / out_rate
    cutoff: f64,
    pos: f64, // position of the next output sample in input time, relative to hist[0]
    hist: [f32; RS_HIST],
    len: usize,
    // coef[p][j]: normalised windowed-sinc tap j (offset j - TAPS + 1) at fractional phase p
    coef: [f32; (RS_PHASES + 1) * RS_W],
}

static mut RS: Resampler = Resampler {
    ratio: 3.0,
    cutoff: 1.0 / 3.0,
    pos: 0.0,
    hist: [0.0; RS_HIST],
    len: 0,
    coef: [0.0; (RS_PHASES + 1) * RS_W],
};

#[no_mangle]
pub extern "C" fn rs_init(in_rate: f64, out_rate: f64) {
    unsafe {
        let rs = &mut *core::ptr::addr_of_mut!(RS);
        rs.ratio = in_rate / out_rate;
        rs.cutoff = if rs.ratio > 1.0 { 0.97 / rs.ratio } else { 0.97 };
        rs.pos = RS_TAPS as f64;
        for v in rs.hist.iter_mut() {
            *v = 0.0;
        }
        rs.len = RS_TAPS; // leading zeros so the first output has full left context
        let span = RS_TAPS as f64;
        for p in 0..=RS_PHASES {
            let frac = p as f64 / RS_PHASES as f64;
            let mut sum = 0f64;
            for j in 0..RS_W {
                let x = (j as f64 - span + 1.0) - frac;
                let wpos = (x + span) / (2.0 * span);
                let win = 0.42 - 0.5 * libm_cos(2.0 * core::f64::consts::PI * wpos)
                    + 0.08 * libm_cos(4.0 * core::f64::consts::PI * wpos);
                let c = sinc(x * rs.cutoff) * win;
                rs.coef[p * RS_W + j] = c as f32;
                sum += c;
            }
            for j in 0..RS_W {
                rs.coef[p * RS_W + j] = (rs.coef[p * RS_W + j] as f64 / sum) as f32;
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = core::f64::consts::PI * x;
        libm_sin(px) / px
    }
}

// Small sine/cosine (no std): range-reduced Taylor series, plenty for a filter window.
fn libm_sin(x: f64) -> f64 {
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut y = x - two_pi * floor(x / two_pi + 0.5);
    let pi = core::f64::consts::PI;
    if y > pi / 2.0 {
        y = pi - y;
    } else if y < -pi / 2.0 {
        y = -pi - y;
    }
    let y2 = y * y;
    y * (1.0 - y2 / 6.0 * (1.0 - y2 / 20.0 * (1.0 - y2 / 42.0 * (1.0 - y2 / 72.0 * (1.0 - y2 / 110.0)))))
}

fn libm_cos(x: f64) -> f64 {
    libm_sin(x + core::f64::consts::PI / 2.0)
}

fn floor(x: f64) -> f64 {
    let i = x as i64 as f64;
    if i > x { i - 1.0 } else { i }
}

/// Feeds `n` input samples, writes resampled output to `out` (capacity `cap`), returns the count.
#[no_mangle]
pub unsafe extern "C" fn rs_process(input: *const f32, n: usize, out: *mut f32, cap: usize) -> usize {
    let rs = &mut *core::ptr::addr_of_mut!(RS);
    let mut produced = 0usize;
    let mut consumed = 0usize;
    while consumed < n {
        // fill history
        let room = RS_HIST - rs.len;
        let take = if n - consumed < room { n - consumed } else { room };
        for i in 0..take {
            rs.hist[rs.len + i] = *input.add(consumed + i);
        }
        rs.len += take;
        consumed += take;
        // produce while the right context is available
        while rs.pos + (RS_TAPS as f64) < rs.len as f64 && produced < cap {
            let center = floor(rs.pos) as usize;
            let frac = rs.pos - center as f64;
            let p = (frac * RS_PHASES as f64 + 0.5) as usize;
            let c = &rs.coef[p * RS_W..p * RS_W + RS_W];
            let h = &rs.hist[center + 1 - RS_TAPS..center + 1 + RS_TAPS];
            let mut acc = 0f32;
            for j in 0..RS_W {
                acc += c[j] * h[j];
            }
            *out.add(produced) = acc;
            produced += 1;
            rs.pos += rs.ratio;
        }
        // drop consumed history, keep left context
        let keep_from = floor(rs.pos) as usize;
        let keep_from = if keep_from > RS_TAPS { keep_from - RS_TAPS } else { 0 };
        if keep_from > 0 {
            let remain = rs.len - keep_from;
            for i in 0..remain {
                rs.hist[i] = rs.hist[keep_from + i];
            }
            rs.len = remain;
            rs.pos -= keep_from as f64;
        }
        if produced >= cap {
            break;
        }
    }
    produced
}

// ------------------------------------------------------------------ voice activity

// Energy + zero-crossing VAD over 16 kHz frames with an adaptive noise floor and hangover.
// States returned by vad_process: 0 = silence, 1 = speech starting/continuing,
// 2 = end of utterance (enough trailing silence after speech).
struct Vad {
    noise_db: f32,
    speech_frames: u32,
    silence_frames: u32,
    in_speech: bool,
    start_db: f32,
    min_speech: u32,
    end_silence: u32,
    last_db: f32,
}

static mut VAD: Vad = Vad {
    noise_db: -60.0,
    speech_frames: 0,
    silence_frames: 0,
    in_speech: false,
    start_db: 9.0,
    min_speech: 6,
    end_silence: 28,
    last_db: -100.0,
};

/// threshold_db: how far above the noise floor counts as speech; end_ms: trailing silence that
/// ends an utterance; frame_ms: duration of each frame passed to vad_process.
#[no_mangle]
pub extern "C" fn vad_init(threshold_db: f32, end_ms: f32, frame_ms: f32) {
    unsafe {
        let v = &mut *core::ptr::addr_of_mut!(VAD);
        v.noise_db = -60.0;
        v.speech_frames = 0;
        v.silence_frames = 0;
        v.in_speech = false;
        v.start_db = threshold_db;
        v.end_silence = (end_ms / frame_ms) as u32;
        v.min_speech = (120.0 / frame_ms) as u32 + 1;
        v.last_db = -100.0;
    }
}

fn log10f(x: f32) -> f32 {
    // log10 via bit tricks + polynomial (accuracy ~1e-4, fine for a level meter)
    if x <= 0.0 {
        return -12.0;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 255) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1,2)
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln_m = 2.0 * t * (1.0 + t2 / 3.0 + t2 * t2 / 5.0 + t2 * t2 * t2 / 7.0);
    (ln_m + e as f32 * core::f32::consts::LN_2) * core::f32::consts::LOG10_E
}

#[no_mangle]
pub unsafe extern "C" fn vad_process(frame: *const f32, n: usize) -> i32 {
    let v = &mut *core::ptr::addr_of_mut!(VAD);
    let mut ss = 0f32;
    let mut zc = 0u32;
    let mut prev = 0f32;
    for i in 0..n {
        let x = *frame.add(i);
        ss += x * x;
        if i > 0 && ((x >= 0.0) != (prev >= 0.0)) {
            zc += 1;
        }
        prev = x;
    }
    let rms = if n > 0 { ss / n as f32 } else { 0.0 };
    let db = 10.0 * log10f(rms + 1e-12);
    v.last_db = db;
    let zcr = if n > 1 { zc as f32 / (n - 1) as f32 } else { 0.0 };
    // speech: loud enough above the floor and not pure hiss (very high zero-crossing rate)
    let loud = db > v.noise_db + v.start_db && db > -55.0;
    let voiced = loud && zcr < 0.45;
    if !v.in_speech {
        // track the noise floor only while nobody talks: fast down, slow up
        if db < v.noise_db {
            v.noise_db = 0.7 * v.noise_db + 0.3 * db;
        } else {
            v.noise_db = 0.995 * v.noise_db + 0.005 * db;
        }
        if voiced {
            v.speech_frames += 1;
            if v.speech_frames >= v.min_speech {
                v.in_speech = true;
                v.silence_frames = 0;
            }
            return 1;
        }
        v.speech_frames = 0;
        return 0;
    }
    if voiced {
        v.silence_frames = 0;
        v.speech_frames += 1;
        return 1;
    }
    v.silence_frames += 1;
    if v.silence_frames >= v.end_silence {
        v.in_speech = false;
        v.speech_frames = 0;
        v.silence_frames = 0;
        return 2;
    }
    1
}

#[no_mangle]
pub extern "C" fn vad_level_db() -> f32 {
    unsafe { (*core::ptr::addr_of!(VAD)).last_db }
}

#[no_mangle]
pub extern "C" fn vad_noise_db() -> f32 {
    unsafe { (*core::ptr::addr_of!(VAD)).noise_db }
}
