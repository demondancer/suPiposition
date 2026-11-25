//! Contract:
//! - Inputs: Unsigned limb slices, cached or fresh complex weight/twiddle tables (`FftTables`),
//!   transform configuration (`n_complex`, `coeff_bits` with `coeff_bits % 61 != 0`), and `Kernels`.
//! - Outputs: Whole products, squares, and fused two-product sums/differences (`Q2*T1 +- P1*T2`),
//!   returning `(limbs, check_word_w0, max_rounding_error)` and reusable `FftCachedFwd`.
//! - Data format: `2 * n_complex` interleaved `(re, im)` `f64` values sized to fit within a
//!   worker's L2 cache share.
//! - Work & bytes moved: 3 complex transforms of length `n_complex` for a fresh product, 2 for a
//!   square or cached-operand product, and 7 for a fused binary-splitting merge.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use crate::arena::{ScratchAlloc, TransformsAlloc};
use crate::limb::{LimbVec, trim_limbs};
use crate::refk::{FftCarryCfg, FftSplitCfg, Kernels, build_fft_tables};

type FftScratchTuple = (
    Vec<f64, ScratchAlloc>,
    Vec<f64, ScratchAlloc>,
    Vec<f64, ScratchAlloc>,
);

fn empty_fft_scratch_tuple() -> FftScratchTuple {
    (
        Vec::new_in(ScratchAlloc),
        Vec::new_in(ScratchAlloc),
        Vec::new_in(ScratchAlloc),
    )
}

thread_local! {
    static FFT_SCRATCH: RefCell<FftScratchTuple> = RefCell::new(empty_fft_scratch_tuple());
}

fn take_fft_fwd_buf(len: usize) -> Vec<f64, TransformsAlloc> {
    let mut v = Vec::with_capacity_in(len, TransformsAlloc);
    unsafe {
        v.set_len(len);
    }
    v
}

pub fn clear_tls_fwd_pool() {
    let mut scratch = empty_fft_scratch_tuple();
    let _ = FFT_SCRATCH.try_with(|cell| {
        scratch = std::mem::replace(&mut *cell.borrow_mut(), empty_fft_scratch_tuple());
    });
    drop(scratch);
    let mut mscratch = FftMergeScratch::new();
    let _ = FFT_MERGE_SCRATCH.try_with(|cell| {
        mscratch = std::mem::replace(&mut *cell.borrow_mut(), FftMergeScratch::new());
    });
    drop(mscratch);
}

/// Maximum permissible rounding error on any balanced digit of an FFT product (Section 5 & 11).
/// Any product exceeding `FFT_MAX_ERR_BOUND` is rejected and recomputed with the exact NTT.
pub const FFT_MAX_ERR_BOUND: f64 = 0.25;

pub struct FftTables {
    pub n_complex: usize,
    pub weights: Vec<f64>,
    pub inv_weights: Vec<f64>,
    pub twiddles: Vec<f64>,
    pub inv_twiddles: Vec<f64>,
}

static GLOBAL_FFT_TABLES: [OnceLock<FftTables>; 64] = [const { OnceLock::new() }; 64];

#[derive(Default)]
pub struct FftTableCache;

impl FftTableCache {
    #[inline]
    pub fn get(&self, n_complex: usize) -> &'static FftTables {
        debug_assert!(n_complex.is_power_of_two());
        let log_n = n_complex.trailing_zeros() as usize;
        GLOBAL_FFT_TABLES[log_n].get_or_init(|| {
            crate::arena::with_system_alloc(|| {
                let (weights, inv_weights, twiddles, inv_twiddles) = build_fft_tables(n_complex);
                FftTables {
                    n_complex,
                    weights,
                    inv_weights,
                    twiddles,
                    inv_twiddles,
                }
            })
        })
    }
}

/// Choose `(n_complex, coeff_bits)` for an `out_bits`-bit product (or `+1` bit for fused merge)
/// so that `coeff_bits % 61 != 0`, `2 * n_complex * coeff_bits >= out_bits + 32`, and expected
/// worst-case rounding error stays comfortably below `FFT_MAX_ERR_BOUND`.
pub fn choose_fft_params(
    out_bits: u64,
    fused_sum: bool,
    cache_share_bytes: u64,
) -> Option<(usize, u32)> {
    // Size the worker's active transform step (`16N` complex data + `16N` weights/twiddles = `32N`
    // bytes) within its L2 share (Section 6; read-only tables are also shared across SMT siblings).
    let max_bytes = cache_share_bytes.max(1024);
    let bytes_per_pt: u64 = 32;
    let mut n_complex = 16usize;
    while (n_complex as u64).saturating_mul(bytes_per_pt) <= max_bytes {
        let m_digits = (2 * n_complex) as u64;
        // Safe bits per balanced digit in IEEE-754 f64 right-angle convolution:
        // Balanced digits in `[-2^{b-1}, 2^{b-1}]` have variance `2^{2b}/12`; stepping `cap_b`
        // down by 1 bit per 4x growth in `n_complex` keeps `m_digits * 2^{2b-2} <= 2^52` and
        // 6-sigma rounding error well below `0.25` (checked on every call by `fft_carry` + `w0`).
        let log_m = m_digits.ilog2();
        let max_2b = 56u32.saturating_sub((log_m * 2) / 3 + u32::from(fused_sum));
        let cap_b = if n_complex <= 128 {
            22
        } else if n_complex <= 2048 {
            21
        } else if n_complex <= 4096 {
            if fused_sum { 20 } else { 21 }
        } else if n_complex <= 8192 {
            20
        } else if n_complex <= 16384 {
            if fused_sum { 19 } else { 20 }
        } else {
            19u32.saturating_sub((n_complex.ilog2() - 14) / 2)
        };
        let b = (max_2b / 2).min(cap_b);
        if b < 8 {
            break;
        }
        if m_digits.saturating_mul(b as u64) >= out_bits.saturating_add(4 * (b as u64)) {
            if n_complex <= 32 && b >= 16 && !(n_complex * (b as usize)).is_multiple_of(64) {
                let align_mask = if n_complex == 16 { !3u32 } else { !1u32 };
                let b_aligned = b & align_mask;
                if b_aligned >= 16
                    && m_digits.saturating_mul(b_aligned as u64)
                        >= out_bits.saturating_add(4 * (b_aligned as u64))
                {
                    return Some((n_complex, b_aligned));
                }
            }
            return Some((n_complex, b));
        }
        let Some(next_n) = n_complex.checked_shl(1) else {
            break;
        };
        n_complex = next_n;
    }
    None
}

#[derive(Clone)]
pub struct FftCachedFwd {
    pub n_complex: usize,
    pub coeff_bits: u32,
    pub data: Vec<f64, TransformsAlloc>,
}

pub fn fft_forward_cached(
    kernels: &Kernels,
    cache: &FftTableCache,
    limbs: &[u64],
    n_complex: usize,
    coeff_bits: u32,
) -> FftCachedFwd {
    debug_assert_ne!(coeff_bits % 61, 0);
    let tables = cache.get(n_complex);
    let mut buf = take_fft_fwd_buf(2 * n_complex);
    let scfg = FftSplitCfg {
        in_limbs: limbs.len() as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
    };
    if kernels.count_work {
        let stages = n_complex.ilog2() as u64;
        kernels
            .stats
            .fft_butterflies
            .fetch_add(((n_complex as u64) / 2) * stages, Ordering::Relaxed);
        kernels.stats.bytes_moved.fetch_add(
            ((limbs.len() * 8) + (n_complex * 16)) as u64,
            Ordering::Relaxed,
        );
    }
    let sub_n = kernels.fft_sub_n();
    unsafe {
        (kernels.active.fft_split)(
            buf.as_mut_ptr(),
            limbs.as_ptr(),
            tables.weights.as_ptr(),
            &scfg,
        );
        (kernels.active.fft_fwd)(
            buf.as_mut_ptr(),
            tables.twiddles.as_ptr(),
            n_complex as u64,
            sub_n,
        );
    }
    FftCachedFwd {
        n_complex,
        coeff_bits,
        data: buf,
    }
}

#[inline(always)]
pub const fn fft_direct_carry_limbs(n_complex: usize, coeff_bits: u32, out_limbs: usize) -> usize {
    let min_direct = ((n_complex * (coeff_bits as usize)) >> 6) + 4;
    if out_limbs >= min_direct {
        out_limbs
    } else {
        min_direct
    }
}

pub fn fft_mul(
    kernels: &Kernels,
    cache: &FftTableCache,
    a: &[u64],
    b: &[u64],
    n_complex: usize,
    coeff_bits: u32,
) -> Option<(LimbVec, u64, f64)> {
    debug_assert_ne!(coeff_bits % 61, 0);
    let out_limbs = a.len() + b.len() + 1;
    let carry_limbs = fft_direct_carry_limbs(n_complex, coeff_bits, out_limbs);
    let same = std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len();
    let tables = cache.get(n_complex);
    let mut out = crate::limb::take_limb_buf_uninit(carry_limbs);
    let mut ccfg = FftCarryCfg {
        out_limbs: carry_limbs as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    if kernels.count_work {
        let stages = n_complex.ilog2() as u64;
        let bf_one = ((n_complex as u64) / 2) * stages;
        let fwd_cnt = if same { 1u64 } else { 2u64 };
        kernels.stats.fft_butterflies.fetch_add(
            fwd_cnt * bf_one + bf_one + (n_complex as u64),
            Ordering::Relaxed,
        );
        let in_bytes = if same {
            a.len() * 8 + n_complex * 16
        } else {
            (a.len() + b.len()) * 8 + n_complex * 32
        };
        kernels
            .stats
            .bytes_moved
            .fetch_add(in_bytes as u64, Ordering::Relaxed);
    }
    FFT_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (fa, fb, _) = &mut *guard;
        if fa.len() < 2 * n_complex {
            fa.resize(2 * n_complex, 0.0);
        }
        if !same && fb.len() < 2 * n_complex {
            fb.resize(2 * n_complex, 0.0);
        }
        let scfg_a = FftSplitCfg {
            in_limbs: a.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let sub_n = kernels.fft_sub_n();
        unsafe {
            (kernels.active.fft_split)(
                fa.as_mut_ptr(),
                a.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_a,
            );
            (kernels.active.fft_fwd)(
                fa.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            let pb_ptr = if same {
                fa.as_ptr()
            } else {
                let scfg_b = FftSplitCfg {
                    in_limbs: b.len() as u64,
                    n_complex: n_complex as u64,
                    coeff_bits: coeff_bits as u64,
                };
                (kernels.active.fft_split)(
                    fb.as_mut_ptr(),
                    b.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_b,
                );
                (kernels.active.fft_fwd)(
                    fb.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );
                fb.as_ptr()
            };
            (kernels.active.fft_pmul)(fa.as_mut_ptr(), fa.as_ptr(), pb_ptr, n_complex as u64);
            (kernels.active.fft_inv)(
                fa.as_mut_ptr(),
                tables.inv_twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_carry)(
                out.as_mut_ptr(),
                fa.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg,
            );
            out.set_len(carry_limbs);
        }
    });
    if !(ccfg.max_err <= FFT_MAX_ERR_BOUND) {
        crate::limb::put_limb_buf(out);
        return None;
    }
    trim_limbs(&mut out);
    Some((out, ccfg.check_word, ccfg.max_err))
}

pub fn fft_pmuladd_cached(
    kernels: &Kernels,
    cache: &FftTableCache,
    a1: &FftCachedFwd,
    b1: &FftCachedFwd,
    a2: &FftCachedFwd,
    b2: &FftCachedFwd,
    sub_second: bool,
    out_limbs: usize,
) -> Option<(LimbVec, u64, f64)> {
    let n = a1.n_complex;
    debug_assert_eq!(b1.n_complex, n);
    debug_assert_eq!(a2.n_complex, n);
    debug_assert_eq!(b2.n_complex, n);

    let tables = cache.get(n);
    let carry_limbs = fft_direct_carry_limbs(n, a1.coeff_bits, out_limbs);
    let mut out = crate::limb::take_limb_buf_uninit(carry_limbs);
    let ptrs = [
        a1.data.as_ptr(),
        b1.data.as_ptr(),
        a2.data.as_ptr(),
        b2.data.as_ptr(),
    ];
    let mut ccfg = FftCarryCfg {
        out_limbs: carry_limbs as u64,
        n_complex: n as u64,
        coeff_bits: a1.coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    if kernels.count_work {
        let stages = n.ilog2() as u64;
        kernels
            .stats
            .fft_butterflies
            .fetch_add(((n as u64) / 2) * stages + 2 * (n as u64), Ordering::Relaxed);
    }
    FFT_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let fc = &mut guard.2;
        if fc.len() < 2 * n {
            fc.resize(2 * n, 0.0);
        }
        let sub_n = kernels.fft_sub_n();
        unsafe {
            (kernels.active.fft_pmuladd)(fc.as_mut_ptr(), &ptrs, n as u64, u64::from(sub_second));
            (kernels.active.fft_inv)(fc.as_mut_ptr(), tables.inv_twiddles.as_ptr(), n as u64, sub_n);
            (kernels.active.fft_carry)(
                out.as_mut_ptr(),
                fc.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg,
            );
            out.set_len(carry_limbs);
        }
    });
    if !(ccfg.max_err <= FFT_MAX_ERR_BOUND) {
        crate::limb::put_limb_buf(out);
        return None;
    }
    trim_limbs(&mut out);
    Some((out, ccfg.check_word, ccfg.max_err))
}

/// Zero-allocation fused `fft_split(a) + fft_fwd(a) + fft_pmul(a, fb) + fft_inv + fft_carry`
/// writing directly into `dst[..out_limbs]` using thread-local `FFT_SCRATCH`.
pub fn fft_mul_slice_with_cached_into(
    kernels: &Kernels,
    cache: &FftTableCache,
    dst: &mut [u64],
    a: &[u64],
    fb: &FftCachedFwd,
    out_limbs: usize,
) -> Option<(usize, u64, f64)> {
    debug_assert!(dst.len() >= out_limbs);
    let n = fb.n_complex;
    let coeff_bits = fb.coeff_bits;
    let want_direct = fft_direct_carry_limbs(n, coeff_bits, out_limbs);
    let carry_limbs = if dst.len() >= want_direct {
        want_direct
    } else {
        out_limbs
    };
    let tables = cache.get(n);
    let scfg = FftSplitCfg {
        in_limbs: a.len() as u64,
        n_complex: n as u64,
        coeff_bits: coeff_bits as u64,
    };
    let mut ccfg = FftCarryCfg {
        out_limbs: carry_limbs as u64,
        n_complex: n as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    if kernels.count_work {
        let stages = n.ilog2() as u64;
        let bf_one = ((n as u64) / 2) * stages;
        kernels
            .stats
            .fft_butterflies
            .fetch_add(2 * bf_one + (n as u64), Ordering::Relaxed);
        kernels
            .stats
            .bytes_moved
            .fetch_add(((a.len() * 8) + (n * 16)) as u64, Ordering::Relaxed);
    }
    FFT_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (fa, _, _) = &mut *guard;
        if fa.len() < 2 * n {
            fa.resize(2 * n, 0.0);
        }
        let sub_n = kernels.fft_sub_n();
        unsafe {
            (kernels.active.fft_split)(
                fa.as_mut_ptr(),
                a.as_ptr(),
                tables.weights.as_ptr(),
                &scfg,
            );
            (kernels.active.fft_fwd)(
                fa.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n as u64,
                sub_n,
            );
            (kernels.active.fft_pmul)(
                fa.as_mut_ptr(),
                fa.as_ptr(),
                fb.data.as_ptr(),
                n as u64,
            );
            (kernels.active.fft_inv)(fa.as_mut_ptr(), tables.inv_twiddles.as_ptr(), n as u64, sub_n);
            (kernels.active.fft_carry)(
                dst.as_mut_ptr(),
                fa.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg,
            );
        }
    });
    if !(ccfg.max_err <= FFT_MAX_ERR_BOUND) {
        return None;
    }
    let mut trimmed_len = carry_limbs;
    while trimmed_len > 0 && dst[trimmed_len - 1] == 0 {
        trimmed_len -= 1;
    }
    if trimmed_len > out_limbs {
        return None;
    }
    Some((trimmed_len, ccfg.check_word, ccfg.max_err))
}

struct FftMergeScratch {
    b0: Vec<f64, ScratchAlloc>,
    b1: Vec<f64, ScratchAlloc>,
    b2: Vec<f64, ScratchAlloc>,
    b3: Vec<f64, ScratchAlloc>,
    q2_sh: Vec<u64, ScratchAlloc>,
}

impl FftMergeScratch {
    fn new() -> Self {
        Self {
            b0: Vec::new_in(ScratchAlloc),
            b1: Vec::new_in(ScratchAlloc),
            b2: Vec::new_in(ScratchAlloc),
            b3: Vec::new_in(ScratchAlloc),
            q2_sh: Vec::new_in(ScratchAlloc),
        }
    }
}

thread_local! {
    static FFT_MERGE_SCRATCH: RefCell<FftMergeScratch> = RefCell::new(FftMergeScratch::new());
}

/// Zero-allocation L2-resident fused `(T, Q, P)` binary-splitting merge for `MulMethod::Fft`.
#[allow(clippy::too_many_arguments)]
pub fn fft_fused_merge_triple(
    kernels: &Kernels,
    cache: &FftTableCache,
    n_complex: usize,
    coeff_bits: u32,
    fft_err_bound: f64,
    cache_share_bytes: u64,
    q2_limbs: &[u64],
    q2_shift: u64,
    t1_limbs: &[u64],
    p1_limbs: &[u64],
    t2_limbs: &[u64],
    sub_second: bool,
    t_out_limbs: usize,
    q1_and_out: Option<(&[u64], usize)>,
    p2_and_out: Option<(&[u64], usize)>,
) -> Option<crate::ntt::FusedMergeRaw> {
    debug_assert_ne!(coeff_bits % 61, 0);
    let tables = cache.get(n_complex);
    let t_carry_limbs = fft_direct_carry_limbs(n_complex, coeff_bits, t_out_limbs);
    let q_carry_limbs = q1_and_out.map_or(0, |(_, ql)| fft_direct_carry_limbs(n_complex, coeff_bits, ql));
    let p_carry_limbs = p2_and_out.map_or(0, |(_, pl)| fft_direct_carry_limbs(n_complex, coeff_bits, pl));
    let mut t_out = crate::limb::take_limb_buf_uninit(t_carry_limbs);
    let mut q_out = q1_and_out.map(|_| crate::limb::take_limb_buf_uninit(q_carry_limbs));
    let mut p_out = p2_and_out.map(|_| crate::limb::take_limb_buf_uninit(p_carry_limbs));

    let mut ccfg_t = FftCarryCfg {
        out_limbs: t_carry_limbs as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    let mut ccfg_q = FftCarryCfg {
        out_limbs: q_carry_limbs as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    let mut ccfg_p = FftCarryCfg {
        out_limbs: p_carry_limbs as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };

    if kernels.count_work {
        let stages = n_complex.ilog2() as u64;
        let bf_one = ((n_complex as u64) / 2) * stages;
        let n_fwd = 4u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some());
        let n_inv = 1u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some());
        let pmul_eq =
            (2u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some()))
                * (n_complex as u64);
        kernels
            .stats
            .fft_butterflies
            .fetch_add((n_fwd + n_inv) * bf_one + pmul_eq, Ordering::Relaxed);
        let in_l = q2_limbs.len()
            + t1_limbs.len()
            + p1_limbs.len()
            + t2_limbs.len()
            + q1_and_out.map_or(0, |(s, _)| s.len())
            + p2_and_out.map_or(0, |(s, _)| s.len());
        kernels.stats.bytes_moved.fetch_add(
            ((in_l * 8) as u64) + n_fwd * ((n_complex * 16) as u64),
            Ordering::Relaxed,
        );
    }

    let ok = FFT_MERGE_SCRATCH.with(|cell| {
        let Ok(mut sc_guard) = cell.try_borrow_mut() else {
            return false;
        };
        let sc = &mut *sc_guard;
        let cap = 2 * n_complex;
        let use_4buf_fma =
            (n_complex as u64) * 64 <= (cache_share_bytes * 2).max(64 * 1024);
        if sc.b0.len() < cap {
            sc.b0.resize(cap, 0.0);
        }
        if sc.b1.len() < cap {
            sc.b1.resize(cap, 0.0);
        }
        if sc.b2.len() < cap {
            sc.b2.resize(cap, 0.0);
        }
        if use_4buf_fma && sc.b3.len() < cap {
            sc.b3.resize(cap, 0.0);
        }

        let q2_slice: &[u64] = if q2_shift == 0 {
            q2_limbs
        } else {
            let ws = (q2_shift >> 6) as usize;
            let bs = (q2_shift & 63) as u32;
            let sh_len = q2_limbs.len() + ws + usize::from(bs != 0);
            sc.q2_sh.clear();
            sc.q2_sh.resize(ws, 0);
            if bs == 0 {
                sc.q2_sh.extend_from_slice(q2_limbs);
            } else if !q2_limbs.is_empty() {
                let nq = q2_limbs.len();
                sc.q2_sh.reserve(nq + 1);
                unsafe {
                    let dst = sc.q2_sh.as_mut_ptr().add(ws);
                    let hi =
                        (kernels.active.mul_1)(dst, q2_limbs.as_ptr(), nq as u64, 1u64 << bs);
                    let mut len = ws + nq;
                    if hi != 0 {
                        *dst.add(nq) = hi;
                        len += 1;
                    }
                    sc.q2_sh.set_len(len);
                }
            }
            let _ = sh_len;
            sc.q2_sh.as_slice()
        };

        let scfg_q2 = FftSplitCfg {
            in_limbs: q2_slice.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let scfg_t1 = FftSplitCfg {
            in_limbs: t1_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let scfg_p1 = FftSplitCfg {
            in_limbs: p1_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let scfg_t2 = FftSplitCfg {
            in_limbs: t2_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };

        let sub_n = kernels.fft_sub_n();
        unsafe {
            // 1. `b0 = FFT(q2)`.
            (kernels.active.fft_split)(
                sc.b0.as_mut_ptr(),
                q2_slice.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_q2,
            );
            (kernels.active.fft_fwd)(
                sc.b0.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );

            // 2. Optional `Q = q1 * q2` using `b1` in-place while `b0` holds `FFT(q2)`.
            if let (Some((q1_l, _)), Some(q_vec)) = (q1_and_out, q_out.as_mut()) {
                let scfg_q1 = FftSplitCfg {
                    in_limbs: q1_l.len() as u64,
                    n_complex: n_complex as u64,
                    coeff_bits: coeff_bits as u64,
                };
                (kernels.active.fft_split)(
                    sc.b1.as_mut_ptr(),
                    q1_l.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_q1,
                );
                (kernels.active.fft_fwd)(
                    sc.b1.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );
                (kernels.active.fft_pmul)(
                    sc.b1.as_mut_ptr(),
                    sc.b1.as_ptr(),
                    sc.b0.as_ptr(),
                    n_complex as u64,
                );
                (kernels.active.fft_inv)(
                    sc.b1.as_mut_ptr(),
                    tables.inv_twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );
                (kernels.active.fft_carry)(
                    q_vec.as_mut_ptr(),
                    sc.b1.as_ptr(),
                    tables.inv_weights.as_ptr(),
                    &mut ccfg_q,
                );
                q_vec.set_len(q_carry_limbs);
                if !(ccfg_q.max_err <= fft_err_bound) {
                    return false;
                }
            }

            // 3. `b1 = FFT(t1)`. When 4 complex buffers fit in the worker's cache share
            // (`80 * N` bytes), defer multiplying `b0 * b1` so `fft_pmuladd` fuses `q2*t1 +/- p1*t2`!
            (kernels.active.fft_split)(
                sc.b1.as_mut_ptr(),
                t1_limbs.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_t1,
            );
            (kernels.active.fft_fwd)(
                sc.b1.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            if use_4buf_fma {
                // 4. `b2 = FFT(p1)`.
                (kernels.active.fft_split)(
                    sc.b2.as_mut_ptr(),
                    p1_limbs.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_p1,
                );
                (kernels.active.fft_fwd)(
                    sc.b2.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );

                // 5. Optional `P = p1 * p2` using `b3` while `b2` holds `FFT(p1)`.
                if let (Some((p2_l, _)), Some(p_vec)) = (p2_and_out, p_out.as_mut()) {
                    let scfg_p2 = FftSplitCfg {
                        in_limbs: p2_l.len() as u64,
                        n_complex: n_complex as u64,
                        coeff_bits: coeff_bits as u64,
                    };
                    (kernels.active.fft_split)(
                        sc.b3.as_mut_ptr(),
                        p2_l.as_ptr(),
                        tables.weights.as_ptr(),
                        &scfg_p2,
                    );
                    (kernels.active.fft_fwd)(
                        sc.b3.as_mut_ptr(),
                        tables.twiddles.as_ptr(),
                        n_complex as u64,
                        sub_n,
                    );
                    (kernels.active.fft_pmul)(
                        sc.b3.as_mut_ptr(),
                        sc.b2.as_ptr(),
                        sc.b3.as_ptr(),
                        n_complex as u64,
                    );
                    (kernels.active.fft_inv)(
                        sc.b3.as_mut_ptr(),
                        tables.inv_twiddles.as_ptr(),
                        n_complex as u64,
                        sub_n,
                    );
                    (kernels.active.fft_carry)(
                        p_vec.as_mut_ptr(),
                        sc.b3.as_ptr(),
                        tables.inv_weights.as_ptr(),
                        &mut ccfg_p,
                    );
                    p_vec.set_len(p_carry_limbs);
                    if !(ccfg_p.max_err <= fft_err_bound) {
                        return false;
                    }
                }

                // 6. `b3 = FFT(t2)`, then fused `b0 = b0 * b1 +/- b2 * b3` via `fft_pmuladd`!
                (kernels.active.fft_split)(
                    sc.b3.as_mut_ptr(),
                    t2_limbs.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_t2,
                );
                (kernels.active.fft_fwd)(
                    sc.b3.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );
                let ptrs = [
                    sc.b0.as_ptr(),
                    sc.b1.as_ptr(),
                    sc.b2.as_ptr(),
                    sc.b3.as_ptr(),
                ];
                (kernels.active.fft_pmuladd)(
                    sc.b0.as_mut_ptr(),
                    &ptrs,
                    n_complex as u64,
                    u64::from(sub_second),
                );
            } else {
                (kernels.active.fft_pmul)(
                    sc.b0.as_mut_ptr(),
                    sc.b0.as_ptr(),
                    sc.b1.as_ptr(),
                    n_complex as u64,
                );

                // 4. `b1 = FFT(p1)`.
                (kernels.active.fft_split)(
                    sc.b1.as_mut_ptr(),
                    p1_limbs.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_p1,
                );
                (kernels.active.fft_fwd)(
                    sc.b1.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );

                // 5. Optional `P = p1 * p2` using `b2` in-place while `b1` holds `FFT(p1)`.
                if let (Some((p2_l, _)), Some(p_vec)) = (p2_and_out, p_out.as_mut()) {
                    let scfg_p2 = FftSplitCfg {
                        in_limbs: p2_l.len() as u64,
                        n_complex: n_complex as u64,
                        coeff_bits: coeff_bits as u64,
                    };
                    (kernels.active.fft_split)(
                        sc.b2.as_mut_ptr(),
                        p2_l.as_ptr(),
                        tables.weights.as_ptr(),
                        &scfg_p2,
                    );
                    (kernels.active.fft_fwd)(
                        sc.b2.as_mut_ptr(),
                        tables.twiddles.as_ptr(),
                        n_complex as u64,
                        sub_n,
                    );
                    (kernels.active.fft_pmul)(
                        sc.b2.as_mut_ptr(),
                        sc.b1.as_ptr(),
                        sc.b2.as_ptr(),
                        n_complex as u64,
                    );
                    (kernels.active.fft_inv)(
                        sc.b2.as_mut_ptr(),
                        tables.inv_twiddles.as_ptr(),
                        n_complex as u64,
                        sub_n,
                    );
                    (kernels.active.fft_carry)(
                        p_vec.as_mut_ptr(),
                        sc.b2.as_ptr(),
                        tables.inv_weights.as_ptr(),
                        &mut ccfg_p,
                    );
                    p_vec.set_len(p_carry_limbs);
                    if !(ccfg_p.max_err <= fft_err_bound) {
                        return false;
                    }
                }

                // 6. `b2 = FFT(t2)`, `b2 = b1 * b2`, combine into `b0`.
                (kernels.active.fft_split)(
                    sc.b2.as_mut_ptr(),
                    t2_limbs.as_ptr(),
                    tables.weights.as_ptr(),
                    &scfg_t2,
                );
                (kernels.active.fft_fwd)(
                    sc.b2.as_mut_ptr(),
                    tables.twiddles.as_ptr(),
                    n_complex as u64,
                    sub_n,
                );
                (kernels.active.fft_pmul)(
                    sc.b2.as_mut_ptr(),
                    sc.b1.as_ptr(),
                    sc.b2.as_ptr(),
                    n_complex as u64,
                );
                if sub_second {
                    for (d, &s) in sc.b0[..cap].iter_mut().zip(&sc.b2[..cap]) {
                        *d -= s;
                    }
                } else {
                    for (d, &s) in sc.b0[..cap].iter_mut().zip(&sc.b2[..cap]) {
                        *d += s;
                    }
                }
            }
            (kernels.active.fft_inv)(
                sc.b0.as_mut_ptr(),
                tables.inv_twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_carry)(
                t_out.as_mut_ptr(),
                sc.b0.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg_t,
            );
            t_out.set_len(t_carry_limbs);
            if !(ccfg_t.max_err <= fft_err_bound) {
                return false;
            }
        }
        true
    });

    if !ok {
        crate::limb::put_limb_buf(t_out);
        if let Some(qv) = q_out {
            crate::limb::put_limb_buf(qv);
        }
        if let Some(pv) = p_out {
            crate::limb::put_limb_buf(pv);
        }
        return None;
    }

    trim_limbs(&mut t_out);
    let q = q_out.map(|mut qv| {
        trim_limbs(&mut qv);
        (qv, ccfg_q.check_word)
    });
    let p = p_out.map(|mut pv| {
        trim_limbs(&mut pv);
        (pv, ccfg_p.check_word)
    });
    Some(crate::ntt::FusedMergeRaw {
        t: (t_out, ccfg_t.check_word),
        q,
        p,
    })
}

/// Fused 5-FFT pair `(a1 * b, a2 * b)` sharing `fwd(b)` in thread-local `FFT_MERGE_SCRATCH` (`0` heap allocs).
#[allow(clippy::too_many_arguments)]
pub fn fft_mul_pair_shared_b(
    kernels: &Kernels,
    cache: &FftTableCache,
    n_complex: usize,
    coeff_bits: u32,
    fft_err_bound: f64,
    a1_limbs: &[u64],
    a1_out_limbs: usize,
    a2_limbs: &[u64],
    a2_out_limbs: usize,
    b_limbs: &[u64],
) -> Option<((LimbVec, u64), (LimbVec, u64))> {
    debug_assert_ne!(coeff_bits % 61, 0);
    let tables = cache.get(n_complex);
    let carry1 = fft_direct_carry_limbs(n_complex, coeff_bits, a1_out_limbs);
    let carry2 = fft_direct_carry_limbs(n_complex, coeff_bits, a2_out_limbs);
    let mut out1 = crate::limb::take_limb_buf_uninit(carry1);
    let mut out2 = crate::limb::take_limb_buf_uninit(carry2);

    let mut ccfg1 = FftCarryCfg {
        out_limbs: carry1 as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };
    let mut ccfg2 = FftCarryCfg {
        out_limbs: carry2 as u64,
        n_complex: n_complex as u64,
        coeff_bits: coeff_bits as u64,
        top_carry: 0,
        max_err: 0.0,
        check_word: 0,
    };

    if kernels.count_work {
        let stages = n_complex.ilog2() as u64;
        let bf_one = ((n_complex as u64) / 2) * stages;
        kernels
            .stats
            .fft_butterflies
            .fetch_add(5 * bf_one + 2 * (n_complex as u64), Ordering::Relaxed);
        let in_l = a1_limbs.len() + a2_limbs.len() + b_limbs.len();
        kernels.stats.bytes_moved.fetch_add(
            ((in_l * 8) as u64) + 3 * ((n_complex * 16) as u64),
            Ordering::Relaxed,
        );
    }

    let ok = FFT_MERGE_SCRATCH.with(|cell| {
        let Ok(mut sc_guard) = cell.try_borrow_mut() else {
            return false;
        };
        let sc = &mut *sc_guard;
        let cap = 2 * n_complex;
        if sc.b0.len() < cap {
            sc.b0.resize(cap, 0.0);
        }
        if sc.b1.len() < cap {
            sc.b1.resize(cap, 0.0);
        }

        let scfg_b = FftSplitCfg {
            in_limbs: b_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let scfg_a1 = FftSplitCfg {
            in_limbs: a1_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };
        let scfg_a2 = FftSplitCfg {
            in_limbs: a2_limbs.len() as u64,
            n_complex: n_complex as u64,
            coeff_bits: coeff_bits as u64,
        };

        let sub_n = kernels.fft_sub_n();
        unsafe {
            (kernels.active.fft_split)(
                sc.b0.as_mut_ptr(),
                b_limbs.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_b,
            );
            (kernels.active.fft_fwd)(
                sc.b0.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );

            (kernels.active.fft_split)(
                sc.b1.as_mut_ptr(),
                a1_limbs.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_a1,
            );
            (kernels.active.fft_fwd)(
                sc.b1.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_pmul)(
                sc.b1.as_mut_ptr(),
                sc.b1.as_ptr(),
                sc.b0.as_ptr(),
                n_complex as u64,
            );
            (kernels.active.fft_inv)(
                sc.b1.as_mut_ptr(),
                tables.inv_twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_carry)(
                out1.as_mut_ptr(),
                sc.b1.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg1,
            );
            out1.set_len(carry1);
            if !(ccfg1.max_err <= fft_err_bound) {
                return false;
            }

            (kernels.active.fft_split)(
                sc.b1.as_mut_ptr(),
                a2_limbs.as_ptr(),
                tables.weights.as_ptr(),
                &scfg_a2,
            );
            (kernels.active.fft_fwd)(
                sc.b1.as_mut_ptr(),
                tables.twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_pmul)(
                sc.b1.as_mut_ptr(),
                sc.b1.as_ptr(),
                sc.b0.as_ptr(),
                n_complex as u64,
            );
            (kernels.active.fft_inv)(
                sc.b1.as_mut_ptr(),
                tables.inv_twiddles.as_ptr(),
                n_complex as u64,
                sub_n,
            );
            (kernels.active.fft_carry)(
                out2.as_mut_ptr(),
                sc.b1.as_ptr(),
                tables.inv_weights.as_ptr(),
                &mut ccfg2,
            );
            out2.set_len(carry2);
            ccfg2.max_err <= fft_err_bound
        }
    });

    if !ok {
        crate::limb::put_limb_buf(out1);
        crate::limb::put_limb_buf(out2);
        return None;
    }

    trim_limbs(&mut out1);
    trim_limbs(&mut out2);
    Some(((out1, ccfg1.check_word), (out2, ccfg2.check_word)))
}
