//! Contract:
//! - Inputs: `Plan` (term count, balanced tree height `H`, `leaf_len`, `dividing_levels`, merge
//!   selector), `Shares`, `Calibration`, `MulCtx`, and optional resumed `SeriesResumeState` +
//!   checkpoint writer callback.
//! - Outputs: Root Chudnovsky pair `(Q, T)` over `(0, 2^H * leaf_len]` reduced by `R_{L_last}`,
//!   with `Q` held odd (`Q_val = Q_odd * 2^{Q.shift}`) and exact `DualCheck` words on both, plus
//!   atomic count of segmented-sieve operations.
//! - Data format: `BlockTriple { p, q, t }` of `BigNum`s; `p` omitted (`zero`) on the rightmost
//!   node of every level including the root; level factors `F_ell = R_{L_ell} / R_{L_prev}^{2^m}`
//!   as odd `BigNum`s.
//! - Work & bytes moved: Plain binary splitting inside L1-sized leaves; 7-transform fused merges or
//!   5+3-transform separate stripped-Q merges above leaves; inactive `BlockTriple`s parked in
//!   `Store` and moved on `Loader` beside `merge_pair` on out-of-core levels; segmented sieve +
//!   balanced factor tree + exact-checked Hensel division (`SequentialHensel` or cached-inverse
//!   `SingleProduct`, climbing Rung 1 repair and Rung 2 factor+inverse remake) once per dividing
//!   level.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::arena::{ScratchAlloc, SubtreeLifoScope};
use crate::check::{
    CorrectionOutcome, DualCheck, Rung, compute_limbs_all3, compute_limbs_w0, compute_limbs_w0_par,
    try_correct_single_limb, whole_value_w0,
};
use crate::limb::{
    BigNum, LimbVec, add_bignum_into_par, cmp_limbs, div_exact_hensel_seq, empty_limb_vec,
    inv_mod_u64, mul_bignum_u64, mul_limbs_basecase, mul_limbs_karatsuba, mul_limbs_karatsuba_into,
    put_limb_buf, shl_limbs, slice_to_limb_vec, take_limb_buf_uninit, trim_limbs,
};
use crate::mul::{CachedTransform, MulCtx, MulError, MulMethod};
use crate::plan::{
    Calibration, DivForm, DivLevelPlan, MergeForm, Plan, SplitFormOverride, choose_wrapped_ntt_spec,
};
use crate::refk::{
    Kernels, MOD_W0, PRIME_LOC1, PRIME_LOC2, fma_mul_mod, mod_pow_u64, mod61_mul, mod61_sub,
};
use crate::tier::Shares;

/// Constant odd part of `10939058860032000 = 2^15 * 333833583375`.
pub const CHUD_Q_ODD_CONST: u64 = 333_833_583_375;
pub const CHUD_A: u64 = 13_591_409;
pub const CHUD_B: u64 = 545_140_134;

#[derive(Clone, Debug)]
pub struct BlockTriple {
    pub p: BigNum,
    pub q: BigNum,
    pub t: BigNum,
}

#[derive(Clone, Debug, Default)]
pub struct SeriesResumeState {
    pub nodes: HashMap<(u32, u64), BlockTriple>,
}

pub trait SeriesCheckpointSink: Send + Sync {
    fn maybe_checkpoint_series(
        &self,
        completed_level: u32,
        live_nodes: &[(u32, u64, &BlockTriple)],
        work_sec_since_last: f64,
    ) -> Result<bool, MulError>;
}

/// Odd-only segmented Eratosthenes sieve up to `limit`, recording exact inner-loop operations in
/// `sieve_ops_counter` for Gate 4 deterministic accounting.
pub fn segmented_primes_up_to(limit: u64, sieve_ops_counter: &AtomicU64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let mut primes = vec![2u64];
    if limit == 2 {
        return primes;
    }

    let root = (limit as f64).sqrt() as usize + 1;
    let small_max = root.min(limit as usize);
    let small_odds = (small_max.saturating_sub(1)) / 2; // odd numbers `2*i + 3` for `i = 0..small_odds`
    let mut is_comp_small = vec![0u64; small_odds.div_ceil(64)];
    let mut ops = 0u64;

    let mut i = 0usize;
    while (2 * i + 3) * (2 * i + 3) <= small_max {
        if (is_comp_small[i >> 6] & (1u64 << (i & 63))) == 0 {
            let p = 2 * i + 3;
            let mut m = (p * p - 3) / 2;
            while m < small_odds {
                is_comp_small[m >> 6] |= 1u64 << (m & 63);
                m += p;
                ops += 1;
            }
        }
        i += 1;
    }

    let mut base_primes = Vec::new();
    for idx in 0..small_odds {
        if (is_comp_small[idx >> 6] & (1u64 << (idx & 63))) == 0 {
            let p = (2 * idx + 3) as u64;
            primes.push(p);
            base_primes.push(p);
        }
    }

    if (small_max as u64) >= limit {
        sieve_ops_counter.fetch_add(ops, Ordering::Relaxed);
        return primes;
    }

    // Segmented sieve over `(small_max ..= limit]` in 32 KiB L1-sized odd-bit segments.
    const SEG_WORDS: usize = 4096; // 32 KiB = 262,144 odd candidates = span of 524,288 integers
    const SEG_ODDS: u64 = (SEG_WORDS * 64) as u64;

    let mut seg_buf = vec![0u64; SEG_WORDS];
    let first_odd = ((small_max as u64) + 1) | 1;
    let last_odd = if (limit & 1) == 1 { limit } else { limit - 1 };

    let mut low_odd = first_odd;
    while low_odd <= last_odd {
        let count_odds = (((last_odd - low_odd) / 2) + 1).min(SEG_ODDS);
        let used_words = (count_odds as usize).div_ceil(64);
        seg_buf[..used_words].fill(0);
        let high_odd = low_odd + 2 * (count_odds - 1);

        for &p in &base_primes {
            let p2 = p * p;
            if p2 > high_odd {
                break;
            }
            let mut start = if p2 >= low_odd {
                p2
            } else {
                let rem = low_odd % p;
                let mut s = if rem == 0 { low_odd } else { low_odd + (p - rem) };
                if (s & 1) == 0 {
                    s += p;
                }
                s
            };
            let mut bit_idx = ((start - low_odd) >> 1) as usize;
            let step = p as usize;
            let lim_bits = count_odds as usize;
            while bit_idx < lim_bits {
                seg_buf[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
                bit_idx += step;
                ops += 1;
            }
            let _ = &mut start;
        }

        for idx in 0..(count_odds as usize) {
            if (seg_buf[idx >> 6] & (1u64 << (idx & 63))) == 0 {
                primes.push(low_odd + 2 * (idx as u64));
            }
        }
        low_odd = high_odd + 2;
    }

    sieve_ops_counter.fetch_add(ops, Ordering::Relaxed);
    primes
}

/// Compute `e_r(L) = v_r(L!) - floor(log_r(L))` for an odd prime `r` and block length `L` (Section 4).
#[inline]
pub fn prime_reduction_exp(l: u64, r: u64) -> u64 {
    if r > l || l == 0 {
        return 0;
    }
    let mut v_fact = 0u64;
    let mut cur = l / r;
    let mut floor_log = 0u64;
    while cur > 0 {
        v_fact += cur;
        floor_log += 1;
        cur /= r;
    }
    debug_assert!(v_fact >= floor_log);
    v_fact - floor_log
}

/// Evaluate single Chudnovsky term `k >= 1`:
/// - `p(k) = -(6k - 5)(2k - 1)(6k - 1)`
/// - `q(k) = 10939058860032000 * k^3 = (333833583375 * k_odd^3) * 2^{15 + 3*v_2(k)}`
/// - `t(k) = (13591409 + 545140134 * k) * p(k)`
pub fn single_term_triple(ctx: &MulCtx<'_>, k: u64) -> BlockTriple {
    debug_assert!(k >= 1);
    let six_k = (k as u128) * 6;
    let f1 = six_k - 5;
    let f2 = (k as u128) * 2 - 1;
    let f3 = six_k - 1;

    // `f1 * f2` fits in u128 for all `k < 2^61` (`12 * k^2 < 2^{126}`).
    let f12 = f1 * f2;
    let p_pos = mul_u128_by_u128(ctx, f12, f3, true); // neg = true

    // `q_odd = 333833583375 * k_odd^3`, `shift = 15 + 3 * v2(k)`.
    let v2 = k.trailing_zeros() as u64;
    let k_odd = (k >> v2) as u128;
    let k_odd_sq = k_odd * k_odd;
    let c_k = (CHUD_Q_ODD_CONST as u128) * k_odd;
    let mut q_bn = mul_u128_by_u128(ctx, k_odd_sq, c_k, false);
    q_bn.shift = 15 + 3 * v2;
    if !q_bn.limbs.is_empty() {
        let lw0 = compute_limbs_w0(ctx.kernels, &q_bn.limbs);
        q_bn.chk = DualCheck::new(whole_value_w0(lw0, false, q_bn.shift));
    }

    // `t(k) = (13591409 + 545140134 * k) * p(k)`
    let lin = (CHUD_A as u128) + (CHUD_B as u128) * (k as u128);
    let lin_bn = u128_to_bignum(lin, false);
    let t_limbs = mul_limbs_basecase(ctx.kernels, &p_pos.limbs, &lin_bn.limbs);
    let t_w0 = mod61_mul(p_pos.w0(), lin_bn.w0());
    let t_bn = BigNum {
        limbs: t_limbs,
        neg: true,
        shift: 0,
        chk: DualCheck::new(t_w0),
    };

    BlockTriple {
        p: p_pos,
        q: q_bn,
        t: t_bn,
    }
}

fn u128_to_bignum(val: u128, neg: bool) -> BigNum {
    if val == 0 {
        return BigNum::zero();
    }
    let lo = val as u64;
    let hi = (val >> 64) as u64;
    let limbs = if hi == 0 {
        crate::limb::slice_to_limb_vec(&[lo])
    } else {
        crate::limb::slice_to_limb_vec(&[lo, hi])
    };
    let lw0 = crate::refk::mod61_reduce(val);
    let w0 = whole_value_w0(lw0, neg, 0);
    BigNum {
        limbs,
        neg,
        shift: 0,
        chk: DualCheck::new(w0),
    }
}

fn mul_u128_by_u128(ctx: &MulCtx<'_>, a: u128, b: u128, neg: bool) -> BigNum {
    if a == 0 || b == 0 {
        return BigNum::zero();
    }
    if a <= (u64::MAX as u128) && b <= (u64::MAX as u128) {
        return u128_to_bignum(a * b, neg);
    }
    let ba = u128_to_bignum(a, false);
    let bb = u128_to_bignum(b, false);
    let limbs = mul_limbs_basecase(ctx.kernels, &ba.limbs, &bb.limbs);
    let lw0 = mod61_mul(ba.w0(), bb.w0());
    let w0 = whole_value_w0(lw0, neg, 0);
    BigNum {
        limbs,
        neg,
        shift: 0,
        chk: DualCheck::new(w0),
    }
}

std::thread_local! {
    static LEAF_POOL: std::cell::RefCell<Vec<u64, ScratchAlloc>> =
        std::cell::RefCell::new(Vec::new_in(ScratchAlloc));
}

pub fn clear_tls_scratch() {
    let _ = LEAF_POOL.try_with(|cell| {
        let old = std::mem::replace(&mut *cell.borrow_mut(), Vec::new_in(ScratchAlloc));
        drop(old);
    });
}

#[derive(Clone, Copy)]
struct SmallSpan {
    off: usize,
    len: usize,
    neg: bool,
    shift: u64,
}

impl SmallSpan {
    const ZERO: Self = Self {
        off: 0,
        len: 0,
        neg: false,
        shift: 0,
    };
}

#[derive(Clone, Copy)]
struct SmallTriple {
    p: SmallSpan,
    q: SmallSpan,
    t: SmallSpan,
}

#[inline]
fn u128_to_arr(v: u128) -> ([u64; 2], usize) {
    if v == 0 {
        ([0, 0], 0)
    } else {
        let lo = v as u64;
        let hi = (v >> 64) as u64;
        if hi == 0 {
            ([lo, 0], 1)
        } else {
            ([lo, hi], 2)
        }
    }
}

#[inline]
fn mul_u128_into_arr(kernels: &crate::refk::Kernels, a: u128, b: u128) -> ([u64; 4], usize) {
    if a == 0 || b == 0 {
        return ([0; 4], 0);
    }
    if a <= (u64::MAX as u128) && b <= (u64::MAX as u128) {
        let (arr2, l2) = u128_to_arr(a * b);
        return ([arr2[0], arr2[1], 0, 0], l2);
    }
    let (aa, la) = u128_to_arr(a);
    let (bb, lb) = u128_to_arr(b);
    let (long, nl, short, ns) = if la >= lb {
        (&aa, la, &bb, lb)
    } else {
        (&bb, lb, &aa, la)
    };
    let mut out = [0u64; 4];
    let lens = kernels.mul_base_lens(nl, ns);
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add((nl * ns) as u64, Ordering::Relaxed);
    }
    unsafe {
        (kernels.active.mul_base)(out.as_mut_ptr(), long.as_ptr(), short.as_ptr(), &lens);
    }
    let mut olen = nl + ns;
    while olen > 0 && out[olen - 1] == 0 {
        olen -= 1;
    }
    (out, olen)
}

#[inline]
fn push_slice<A: std::alloc::Allocator>(pool: &mut Vec<u64, A>, s: &[u64]) -> (usize, usize) {
    let mut len = s.len();
    while len > 0 && s[len - 1] == 0 {
        len -= 1;
    }
    let off = pool.len();
    if len > 0 {
        pool.extend_from_slice(&s[..len]);
    }
    (off, len)
}

#[inline]
fn mul_pool_spans<A: std::alloc::Allocator>(
    kernels: &crate::refk::Kernels,
    pool: &mut Vec<u64, A>,
    a_off: usize,
    a_len: usize,
    b_off: usize,
    b_len: usize,
    basecase_cutoff: usize,
) -> (usize, usize) {
    if a_len == 0 || b_len == 0 {
        return (pool.len(), 0);
    }
    if a_len.min(b_len) > basecase_cutoff {
        let out_off = pool.len();
        let cap = a_len + b_len + 1;
        if pool.capacity() < out_off + cap {
            pool.resize(out_off + cap, 0);
            pool.truncate(out_off);
        }
        let olen = unsafe {
            let base_ptr = pool.as_mut_ptr();
            let src_part = std::slice::from_raw_parts(base_ptr, out_off);
            let dst_part = std::slice::from_raw_parts_mut(base_ptr.add(out_off), cap);
            let olen = mul_limbs_karatsuba_into(
                kernels,
                dst_part,
                &src_part[a_off..(a_off + a_len)],
                &src_part[b_off..(b_off + b_len)],
                basecase_cutoff,
            );
            pool.set_len(out_off + olen);
            olen
        };
        return (out_off, olen);
    }
    let (l_off, nl, s_off, ns) = if a_len >= b_len {
        (a_off, a_len, b_off, b_len)
    } else {
        (b_off, b_len, a_off, a_len)
    };
    let out_off = pool.len();
    let cap = nl + ns;
    let use_scalar = if (kernels.bound_mask & (1 << 4)) != 0 {
        ns <= 2 && nl <= 4
    } else {
        ns <= 4 && nl <= 8
    };
    if use_scalar {
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add((nl * ns) as u64, Ordering::Relaxed);
        }
        pool.resize(out_off + cap, 0);
        for j in 0..ns {
            let bj = pool[s_off + j] as u128;
            if bj == 0 {
                continue;
            }
            let mut carry = 0u128;
            for i in 0..nl {
                let prod = (pool[l_off + i] as u128) * bj + (pool[out_off + j + i] as u128) + carry;
                pool[out_off + j + i] = prod as u64;
                carry = prod >> 64;
            }
            pool[out_off + j + nl] = carry as u64;
        }
    } else {
        pool.reserve(cap);
        unsafe {
            let base_ptr = pool.as_mut_ptr();
            let src_part = std::slice::from_raw_parts(base_ptr, out_off);
            let dst_part = std::slice::from_raw_parts_mut(base_ptr.add(out_off), cap);
            crate::limb::mul_base_dispatched(
                kernels,
                dst_part,
                &src_part[l_off..(l_off + nl)],
                &src_part[s_off..(s_off + ns)],
            );
            pool.set_len(out_off + cap);
        }
    }
    let mut olen = cap;
    while olen > 0 && pool[out_off + olen - 1] == 0 {
        olen -= 1;
    }
    pool.truncate(out_off + olen);
    (out_off, olen)
}

#[inline]
fn eval_pair_stack(
    start_k: u64,
    need_p: bool,
) -> ([u64; 4], usize, [u64; 6], usize, u64, [u64; 8], usize) {
    let k1 = start_k + 1;
    let k2 = start_k + 2;

    let six_k1 = k1 * 6;
    let six_k2 = k2 * 6;
    let (p1, p2) = if k2 <= 715_827_882 {
        (
            (((six_k1 - 5) * (six_k1 - 1)) as u128) * ((k1 * 2 - 1) as u128),
            (((six_k2 - 5) * (six_k2 - 1)) as u128) * ((k2 * 2 - 1) as u128),
        )
    } else {
        (
            ((six_k1 - 5) as u128 * (k1 * 2 - 1) as u128) * (six_k1 - 1) as u128,
            ((six_k2 - 5) as u128 * (k2 * 2 - 1) as u128) * (six_k2 - 1) as u128,
        )
    };
    let p1_0 = p1 as u64;
    let p1_1 = (p1 >> 64) as u64;
    let p2_0 = p2 as u64;
    let p2_1 = (p2 >> 64) as u64;

    let v2_1 = k1.trailing_zeros() as u64;
    let k1_odd_u64 = k1 >> v2_1;
    let k1_cb = if k1_odd_u64 < (1u64 << 32) {
        ((k1_odd_u64 * k1_odd_u64) as u128) * (k1_odd_u64 as u128)
    } else {
        let k1_odd = k1_odd_u64 as u128;
        (k1_odd * k1_odd) * k1_odd
    };
    let q1_lo = ((k1_cb as u64) as u128) * (CHUD_Q_ODD_CONST as u128);
    let q1_hi = ((k1_cb >> 64) as u128) * (CHUD_Q_ODD_CONST as u128) + (q1_lo >> 64);
    let q1 = [q1_lo as u64, q1_hi as u64, (q1_hi >> 64) as u64];

    let v2_2 = k2.trailing_zeros() as u64;
    let q2_shift = 15 + 3 * v2_2;
    let q_shift = 30 + 3 * (v2_1 + v2_2);
    let k2_odd_u64 = k2 >> v2_2;
    let k2_cb = if k2_odd_u64 < (1u64 << 32) {
        ((k2_odd_u64 * k2_odd_u64) as u128) * (k2_odd_u64 as u128)
    } else {
        let k2_odd = k2_odd_u64 as u128;
        (k2_odd * k2_odd) * k2_odd
    };
    let q2_lo = ((k2_cb as u64) as u128) * (CHUD_Q_ODD_CONST as u128);
    let q2_hi = ((k2_cb >> 64) as u128) * (CHUD_Q_ODD_CONST as u128) + (q2_lo >> 64);
    let q2 = [q2_lo as u64, q2_hi as u64, (q2_hi >> 64) as u64];

    let lin1 = (CHUD_A + CHUD_B * k1) as u128;
    let t1_lo = (p1_0 as u128) * lin1;
    let t1_hi = (p1_1 as u128) * lin1 + (t1_lo >> 64);
    let t1 = [t1_lo as u64, t1_hi as u64, (t1_hi >> 64) as u64];

    let lin2 = (CHUD_A + CHUD_B * k2) as u128;
    let t2_lo = (p2_0 as u128) * lin2;
    let t2_hi = (p2_1 as u128) * lin2 + (t2_lo >> 64);
    let t2 = [t2_lo as u64, t2_hi as u64, (t2_hi >> 64) as u64];

    let mut q_arr = [0u64; 6];
    for j in 0..3 {
        let bj = q2[j] as u128;
        if bj == 0 {
            continue;
        }
        let r0 = (q1[0] as u128) * bj + (q_arr[j] as u128);
        q_arr[j] = r0 as u64;
        let r1 = (q1[1] as u128) * bj + (q_arr[j + 1] as u128) + (r0 >> 64);
        q_arr[j + 1] = r1 as u64;
        let r2 = (q1[2] as u128) * bj + (q_arr[j + 2] as u128) + (r1 >> 64);
        q_arr[j + 2] = r2 as u64;
        q_arr[j + 3] = (r2 >> 64) as u64;
    }
    let mut q_len = 6;
    while q_len > 0 && q_arr[q_len - 1] == 0 {
        q_len -= 1;
    }

    let mut qt_raw = [0u64; 6];
    for j in 0..3 {
        let bj = t1[j] as u128;
        if bj == 0 {
            continue;
        }
        let r0 = (q2[0] as u128) * bj + (qt_raw[j] as u128);
        qt_raw[j] = r0 as u64;
        let r1 = (q2[1] as u128) * bj + (qt_raw[j + 1] as u128) + (r0 >> 64);
        qt_raw[j + 1] = r1 as u64;
        let r2 = (q2[2] as u128) * bj + (qt_raw[j + 2] as u128) + (r1 >> 64);
        qt_raw[j + 2] = r2 as u64;
        qt_raw[j + 3] = (r2 >> 64) as u64;
    }
    let ws = (q2_shift >> 6) as usize;
    let bs = (q2_shift & 63) as u32;
    let mut t_arr = [0u64; 8];
    if bs == 0 {
        t_arr[ws..(ws + 6)].copy_from_slice(&qt_raw);
    } else {
        let mut c = 0u64;
        for i in 0..6 {
            let w = qt_raw[i];
            t_arr[ws + i] = (w << bs) | c;
            c = w >> (64 - bs);
        }
        t_arr[ws + 6] = c;
    }

    let mut pt_arr = [0u64; 5];
    {
        let b0 = p1_0 as u128;
        let r0 = (t2[0] as u128) * b0;
        pt_arr[0] = r0 as u64;
        let r1 = (t2[1] as u128) * b0 + (r0 >> 64);
        pt_arr[1] = r1 as u64;
        let r2 = (t2[2] as u128) * b0 + (r1 >> 64);
        pt_arr[2] = r2 as u64;
        pt_arr[3] = (r2 >> 64) as u64;
        if p1_1 != 0 {
            let b1 = p1_1 as u128;
            let s0 = (t2[0] as u128) * b1 + (pt_arr[1] as u128);
            pt_arr[1] = s0 as u64;
            let s1 = (t2[1] as u128) * b1 + (pt_arr[2] as u128) + (s0 >> 64);
            pt_arr[2] = s1 as u64;
            let s2 = (t2[2] as u128) * b1 + (pt_arr[3] as u128) + (s1 >> 64);
            pt_arr[3] = s2 as u64;
            pt_arr[4] = (s2 >> 64) as u64;
        }
    }
    let mut borrow = false;
    for i in 0..8 {
        let sub_w = if i < 5 { pt_arr[i] } else { 0 };
        let (d1, b1) = t_arr[i].overflowing_sub(sub_w);
        let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
        t_arr[i] = d2;
        borrow = b1 | b2;
    }
    debug_assert!(!borrow);
    let mut t_len = 8;
    while t_len > 0 && t_arr[t_len - 1] == 0 {
        t_len -= 1;
    }

    let (p_arr, p_len) = if need_p {
        let pp0 = (p1_0 as u128) * (p2_0 as u128);
        let pp1 = (p1_1 as u128) * (p2_0 as u128) + (pp0 >> 64);
        let pp2 = (p1_0 as u128) * (p2_1 as u128) + (pp1 as u64 as u128);
        let pp3 = (p1_1 as u128) * (p2_1 as u128) + (pp1 >> 64) + (pp2 >> 64);
        let pa = [pp0 as u64, pp2 as u64, pp3 as u64, (pp3 >> 64) as u64];
        let mut pl = 4;
        while pl > 0 && pa[pl - 1] == 0 {
            pl -= 1;
        }
        (pa, pl)
    } else {
        ([0; 4], 0)
    };
    (p_arr, p_len, q_arr, q_len, q_shift, t_arr, t_len)
}

#[inline]
fn mul_stack_small(
    kernels: &crate::refk::Kernels,
    out: &mut [std::mem::MaybeUninit<u64>],
    a: &[u64],
    b: &[u64],
) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (long, nl, short, ns) = if a.len() >= b.len() {
        (a, a.len(), b, b.len())
    } else {
        (b, b.len(), a, a.len())
    };
    let use_scalar = if (kernels.bound_mask & (1 << 4)) != 0 {
        ns <= 2 && nl <= 4
    } else {
        (ns <= 6 && nl <= 9) || (ns <= 4 && nl <= 14)
    };
    if use_scalar {
        let mut buf = [0u64; 18];
        let s0 = short[0] as u128;
        let mut c0 = 0u128;
        for i in 0..nl {
            let p = (long[i] as u128) * s0 + c0;
            buf[i] = p as u64;
            c0 = p >> 64;
        }
        buf[nl] = c0 as u64;
        for j in 1..ns {
            let sj = short[j] as u128;
            if sj == 0 {
                continue;
            }
            let mut c = 0u128;
            for i in 0..nl {
                let p = (long[i] as u128) * sj + (buf[j + i] as u128) + c;
                buf[j + i] = p as u64;
                c = p >> 64;
            }
            buf[j + nl] = c as u64;
        }
        let mut olen = nl + ns;
        while olen > 0 && buf[olen - 1] == 0 {
            olen -= 1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), out.as_mut_ptr() as *mut u64, olen);
        }
        return olen;
    }
    let lens = kernels.mul_base_lens(nl, ns);
    unsafe {
        (kernels.active.mul_base)(
            out.as_mut_ptr() as *mut u64,
            long.as_ptr(),
            short.as_ptr(),
            &lens,
        );
        let mut olen = nl + ns;
        let out_ptr = out.as_ptr() as *const u64;
        while olen > 0 && *out_ptr.add(olen - 1) == 0 {
            olen -= 1;
        }
        olen
    }
}

#[inline]
fn eval_4_stack(
    kernels: &crate::refk::Kernels,
    start_k: u64,
    need_p: bool,
) -> (
    [std::mem::MaybeUninit<u64>; 8],
    usize,
    [std::mem::MaybeUninit<u64>; 12],
    usize,
    u64,
    [std::mem::MaybeUninit<u64>; 18],
    usize,
) {
    use std::mem::MaybeUninit;
    let (p1, pl1, q1, ql1, qs1, t1, tl1) = eval_pair_stack(start_k, true);
    let (p2, pl2, q2, ql2, qs2, t2, tl2) = eval_pair_stack(start_k + 2, need_p);
    let q_shift = qs1 + qs2;

    let mut q_arr = [MaybeUninit::uninit(); 12];
    let q_len = mul_stack_small(kernels, &mut q_arr, &q1[..ql1], &q2[..ql2]);

    let mut qt_raw = [MaybeUninit::uninit(); 14];
    let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, &q2[..ql2], &t1[..tl1]);
    let ws = (qs2 >> 6) as usize;
    let bs = (qs2 & 63) as u32;
    let mut t_arr = [MaybeUninit::uninit(); 18];
    let t_ptr = t_arr.as_mut_ptr() as *mut u64;
    let qt_ptr = qt_raw.as_ptr() as *const u64;
    let qt_len = unsafe {
        if qt_raw_len == 0 {
            0
        } else if bs == 0 {
            std::ptr::write_bytes(t_ptr, 0, ws);
            std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
            ws + qt_raw_len
        } else {
            std::ptr::write_bytes(t_ptr, 0, ws);
            let mut c = 0u64;
            for i in 0..qt_raw_len {
                let w = *qt_ptr.add(i);
                *t_ptr.add(ws + i) = (w << bs) | c;
                c = w >> (64 - bs);
            }
            *t_ptr.add(ws + qt_raw_len) = c;
            ws + qt_raw_len + 1
        }
    };

    let mut pt_arr = [MaybeUninit::uninit(); 12];
    let pt_len = mul_stack_small(kernels, &mut pt_arr, &p1[..pl1], &t2[..tl2]);
    let pt_ptr = pt_arr.as_ptr() as *const u64;
    let clear_end = (qt_len.max(pt_len) + 2).min(18);
    let t_len = unsafe {
        if qt_len < clear_end {
            std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
        }
        let mut carry = false;
        for i in 0..pt_len {
            let (s1, o1) = (*t_ptr.add(i)).overflowing_add(*pt_ptr.add(i));
            let (s2, o2) = s1.overflowing_add(u64::from(carry));
            *t_ptr.add(i) = s2;
            carry = o1 | o2;
        }
        let mut idx = pt_len;
        while carry {
            let (s, ov) = (*t_ptr.add(idx)).overflowing_add(1);
            *t_ptr.add(idx) = s;
            carry = ov;
            idx += 1;
        }
        let mut tl = qt_len.max(idx);
        while tl > 0 && *t_ptr.add(tl - 1) == 0 {
            tl -= 1;
        }
        tl
    };

    let mut p_arr = [MaybeUninit::uninit(); 8];
    let p_len = if need_p {
        mul_stack_small(kernels, &mut p_arr, &p1[..pl1], &p2[..pl2])
    } else {
        0
    };
    (p_arr, p_len, q_arr, q_len, q_shift, t_arr, t_len)
}

#[inline]
fn eval_3_stack(
    start_k: u64,
    need_p: bool,
) -> (
    [u64; 6],
    usize,
    [u64; 9],
    usize,
    u64,
    [u64; 14],
    usize,
) {
    let k1 = start_k + 1;
    let six_k1 = k1 * 6;
    let p1_u128 = if k1 <= 715_827_882 {
        (((six_k1 - 5) * (six_k1 - 1)) as u128) * ((k1 * 2 - 1) as u128)
    } else {
        (((six_k1 - 5) as u128) * ((k1 * 2 - 1) as u128)) * ((six_k1 - 1) as u128)
    };
    let p1_0 = p1_u128 as u64;
    let p1_1 = (p1_u128 >> 64) as u64;
    let p1_arr = [p1_0, p1_1];
    let pl1 = if p1_1 != 0 { 2 } else { usize::from(p1_0 != 0) };

    let v2_1 = k1.trailing_zeros() as u64;
    let qs1 = 15 + 3 * v2_1;
    let k1_odd_u64 = k1 >> v2_1;
    let k1_cb = if k1_odd_u64 < (1u64 << 32) {
        ((k1_odd_u64 * k1_odd_u64) as u128) * (k1_odd_u64 as u128)
    } else {
        let k1_odd = k1_odd_u64 as u128;
        (k1_odd * k1_odd) * k1_odd
    };
    let q1_lo = ((k1_cb as u64) as u128) * (CHUD_Q_ODD_CONST as u128);
    let q1_hi = ((k1_cb >> 64) as u128) * (CHUD_Q_ODD_CONST as u128) + (q1_lo >> 64);
    let q1_arr = [q1_lo as u64, q1_hi as u64, (q1_hi >> 64) as u64];
    let ql1 = if q1_arr[2] != 0 {
        3
    } else if q1_arr[1] != 0 {
        2
    } else {
        usize::from(q1_arr[0] != 0)
    };

    let lin1 = (CHUD_A + CHUD_B * k1) as u128;
    let t1_lo = (p1_0 as u128) * lin1;
    let t1_hi = (p1_1 as u128) * lin1 + (t1_lo >> 64);
    let t1_arr = [t1_lo as u64, t1_hi as u64, (t1_hi >> 64) as u64];
    let tl1 = if t1_arr[2] != 0 {
        3
    } else if t1_arr[1] != 0 {
        2
    } else {
        usize::from(t1_arr[0] != 0)
    };

    let (p2_arr, pl2, q2_arr, ql2, qs2, t2_arr, tl2) = eval_pair_stack(start_k + 1, need_p);
    let q_shift = qs1 + qs2;

    let mut q_buf = [0u64; 9];
    for i in 0..ql1 {
        let ai = q1_arr[i] as u128;
        if ai == 0 {
            continue;
        }
        let mut c = 0u128;
        for j in 0..ql2 {
            let prod = ai * (q2_arr[j] as u128) + (q_buf[i + j] as u128) + c;
            q_buf[i + j] = prod as u64;
            c = prod >> 64;
        }
        q_buf[i + ql2] = c as u64;
    }
    let mut q_len = ql1 + ql2;
    while q_len > 0 && q_buf[q_len - 1] == 0 {
        q_len -= 1;
    }

    let mut qt_raw = [0u64; 9];
    for i in 0..tl1 {
        let ai = t1_arr[i] as u128;
        if ai == 0 {
            continue;
        }
        let mut c = 0u128;
        for j in 0..ql2 {
            let prod = ai * (q2_arr[j] as u128) + (qt_raw[i + j] as u128) + c;
            qt_raw[i + j] = prod as u64;
            c = prod >> 64;
        }
        qt_raw[i + ql2] = c as u64;
    }
    let qt_raw_len = tl1 + ql2;
    let ws = (qs2 >> 6) as usize;
    let bs = (qs2 & 63) as u32;
    let mut t_arr = [0u64; 14];
    let qt_len = if bs == 0 {
        t_arr[ws..(ws + qt_raw_len)].copy_from_slice(&qt_raw[..qt_raw_len]);
        ws + qt_raw_len
    } else {
        let mut c = 0u64;
        for i in 0..qt_raw_len {
            let w = qt_raw[i];
            t_arr[ws + i] = (w << bs) | c;
            c = w >> (64 - bs);
        }
        t_arr[ws + qt_raw_len] = c;
        ws + qt_raw_len + 1
    };

    let mut pt_buf = [0u64; 10];
    for i in 0..pl1 {
        let ai = p1_arr[i] as u128;
        if ai == 0 {
            continue;
        }
        let mut c = 0u128;
        for j in 0..tl2 {
            let prod = ai * (t2_arr[j] as u128) + (pt_buf[i + j] as u128) + c;
            pt_buf[i + j] = prod as u64;
            c = prod >> 64;
        }
        pt_buf[i + tl2] = c as u64;
    }
    let pt_len = pl1 + tl2;
    let mut borrow = false;
    for i in 0..qt_len {
        let sub_w = if i < pt_len { pt_buf[i] } else { 0 };
        let (d1, b1) = t_arr[i].overflowing_sub(sub_w);
        let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
        t_arr[i] = d2;
        borrow = b1 | b2;
    }
    debug_assert!(!borrow);
    let mut t_len = qt_len;
    while t_len > 0 && t_arr[t_len - 1] == 0 {
        t_len -= 1;
    }

    let mut p_buf = [0u64; 6];
    let p_len = if need_p {
        for i in 0..pl1 {
            let ai = p1_arr[i] as u128;
            if ai == 0 {
                continue;
            }
            let mut c = 0u128;
            for j in 0..pl2 {
                let prod = ai * (p2_arr[j] as u128) + (p_buf[i + j] as u128) + c;
                p_buf[i + j] = prod as u64;
                c = prod >> 64;
            }
            p_buf[i + pl2] = c as u64;
        }
        let mut pl = pl1 + pl2;
        while pl > 0 && p_buf[pl - 1] == 0 {
            pl -= 1;
        }
        pl
    } else {
        0
    };
    (p_buf, p_len, q_buf, q_len, q_shift, t_arr, t_len)
}

#[inline]
fn eval_5_stack(
    kernels: &crate::refk::Kernels,
    start_k: u64,
    need_p: bool,
) -> (
    [std::mem::MaybeUninit<u64>; 10],
    usize,
    [std::mem::MaybeUninit<u64>; 15],
    usize,
    u64,
    [std::mem::MaybeUninit<u64>; 24],
    usize,
) {
    use std::mem::MaybeUninit;
    let (p1, pl1, q1, ql1, qs1, t1, tl1) = eval_pair_stack(start_k, true);
    let (p2, pl2, q2, ql2, qs2, t2, tl2) = eval_3_stack(start_k + 2, need_p);
    let q_shift = qs1 + qs2;

    let mut q_arr = [MaybeUninit::uninit(); 15];
    let q_len = mul_stack_small(kernels, &mut q_arr, &q1[..ql1], &q2[..ql2]);

    let mut qt_raw = [MaybeUninit::uninit(); 18];
    let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, &q2[..ql2], &t1[..tl1]);
    let ws = (qs2 >> 6) as usize;
    let bs = (qs2 & 63) as u32;
    let mut t_arr = [MaybeUninit::uninit(); 24];
    let t_ptr = t_arr.as_mut_ptr() as *mut u64;
    let qt_ptr = qt_raw.as_ptr() as *const u64;
    let qt_len = unsafe {
        if qt_raw_len == 0 {
            0
        } else if bs == 0 {
            std::ptr::write_bytes(t_ptr, 0, ws);
            std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
            ws + qt_raw_len
        } else {
            std::ptr::write_bytes(t_ptr, 0, ws);
            let mut c = 0u64;
            for i in 0..qt_raw_len {
                let w = *qt_ptr.add(i);
                *t_ptr.add(ws + i) = (w << bs) | c;
                c = w >> (64 - bs);
            }
            *t_ptr.add(ws + qt_raw_len) = c;
            ws + qt_raw_len + 1
        }
    };

    let mut pt_arr = [MaybeUninit::uninit(); 18];
    let pt_len = mul_stack_small(kernels, &mut pt_arr, &p1[..pl1], &t2[..tl2]);
    let pt_ptr = pt_arr.as_ptr() as *const u64;
    let clear_end = (qt_len.max(pt_len) + 2).min(24);
    let t_len = unsafe {
        if qt_len < clear_end {
            std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
        }
        let mut carry = false;
        for i in 0..pt_len {
            let (s1, o1) = (*t_ptr.add(i)).overflowing_add(*pt_ptr.add(i));
            let (s2, o2) = s1.overflowing_add(u64::from(carry));
            *t_ptr.add(i) = s2;
            carry = o1 | o2;
        }
        let mut idx = pt_len;
        while carry {
            let (s, ov) = (*t_ptr.add(idx)).overflowing_add(1);
            *t_ptr.add(idx) = s;
            carry = ov;
            idx += 1;
        }
        let mut tl = qt_len.max(idx);
        while tl > 0 && *t_ptr.add(tl - 1) == 0 {
            tl -= 1;
        }
        tl
    };

    let mut p_arr = [MaybeUninit::uninit(); 10];
    let p_len = if need_p {
        mul_stack_small(kernels, &mut p_arr, &p1[..pl1], &p2[..pl2])
    } else {
        0
    };
    (p_arr, p_len, q_arr, q_len, q_shift, t_arr, t_len)
}

fn eval_small_stack_into(
    kernels: &crate::refk::Kernels,
    start_k: u64,
    len: u64,
    need_p: bool,
    p_out: &mut [std::mem::MaybeUninit<u64>; 22],
    q_out: &mut [std::mem::MaybeUninit<u64>; 30],
    t_out: &mut [std::mem::MaybeUninit<u64>; 44],
) -> (usize, usize, u64, usize) {
    use std::mem::MaybeUninit;
    if len == 9 {
        let (p1, pl1, q1, ql1, qs1, t1, tl1) = eval_4_stack(kernels, start_k, true);
        let (p2, pl2, q2, ql2, qs2, t2, tl2) = eval_5_stack(kernels, start_k + 4, need_p);
        let q_shift = qs1 + qs2;
        let (p1s, q1s, t1s, p2s, q2s, t2s) = unsafe {
            (
                std::slice::from_raw_parts(p1.as_ptr() as *const u64, pl1),
                std::slice::from_raw_parts(q1.as_ptr() as *const u64, ql1),
                std::slice::from_raw_parts(t1.as_ptr() as *const u64, tl1),
                std::slice::from_raw_parts(p2.as_ptr() as *const u64, pl2),
                std::slice::from_raw_parts(q2.as_ptr() as *const u64, ql2),
                std::slice::from_raw_parts(t2.as_ptr() as *const u64, tl2),
            )
        };
        let q_len = mul_stack_small(kernels, q_out, q1s, q2s);
        let mut qt_raw = [MaybeUninit::uninit(); 38];
        let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, q2s, t1s);
        let ws = (qs2 >> 6) as usize;
        let bs = (qs2 & 63) as u32;
        let t_ptr = t_out.as_mut_ptr() as *mut u64;
        let qt_ptr = qt_raw.as_ptr() as *const u64;
        let qt_len = unsafe {
            if qt_raw_len == 0 {
                0
            } else if bs == 0 {
                std::ptr::write_bytes(t_ptr, 0, ws);
                std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
                ws + qt_raw_len
            } else {
                std::ptr::write_bytes(t_ptr, 0, ws);
                let mut c = 0u64;
                for i in 0..qt_raw_len {
                    let w = *qt_ptr.add(i);
                    *t_ptr.add(ws + i) = (w << bs) | c;
                    c = w >> (64 - bs);
                }
                *t_ptr.add(ws + qt_raw_len) = c;
                ws + qt_raw_len + 1
            }
        };
        let mut pt_arr = [MaybeUninit::uninit(); 34];
        let pt_len = mul_stack_small(kernels, &mut pt_arr, p1s, t2s);
        let pt_ptr = pt_arr.as_ptr() as *const u64;
        let clear_end = (qt_len.max(pt_len) + 2).min(44);
        let t_len = unsafe {
            if qt_len < clear_end {
                std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
            }
            let mut carry = false;
            for i in 0..pt_len {
                let (s1, o1) = (*t_ptr.add(i)).overflowing_add(*pt_ptr.add(i));
                let (s2, o2) = s1.overflowing_add(u64::from(carry));
                *t_ptr.add(i) = s2;
                carry = o1 | o2;
            }
            let mut idx = pt_len;
            while carry {
                let (s, ov) = (*t_ptr.add(idx)).overflowing_add(1);
                *t_ptr.add(idx) = s;
                carry = ov;
                idx += 1;
            }
            let mut tl = qt_len.max(idx);
            while tl > 0 && *t_ptr.add(tl - 1) == 0 {
                tl -= 1;
            }
            tl
        };
        let p_len = if need_p {
            mul_stack_small(kernels, p_out, p1s, p2s)
        } else {
            0
        };
        return (p_len, q_len, q_shift, t_len);
    }
    if len == 5 {
        let (pa, pl, qa, ql, qs, ta, tl) = eval_5_stack(kernels, start_k, need_p);
        unsafe {
            std::ptr::copy_nonoverlapping(pa.as_ptr(), p_out.as_mut_ptr(), pl);
            std::ptr::copy_nonoverlapping(qa.as_ptr(), q_out.as_mut_ptr(), ql);
            std::ptr::copy_nonoverlapping(ta.as_ptr(), t_out.as_mut_ptr(), tl);
        }
        return (pl, ql, qs, tl);
    }
    if len == 8 {
        let (p1, pl1, q1, ql1, qs1, t1, tl1) = eval_4_stack(kernels, start_k, true);
        let (p2, pl2, q2, ql2, qs2, t2, tl2) = eval_4_stack(kernels, start_k + 4, need_p);
        let q_shift = qs1 + qs2;
        let (p1s, q1s, t1s, p2s, q2s, t2s) = unsafe {
            (
                std::slice::from_raw_parts(p1.as_ptr() as *const u64, pl1),
                std::slice::from_raw_parts(q1.as_ptr() as *const u64, ql1),
                std::slice::from_raw_parts(t1.as_ptr() as *const u64, tl1),
                std::slice::from_raw_parts(p2.as_ptr() as *const u64, pl2),
                std::slice::from_raw_parts(q2.as_ptr() as *const u64, ql2),
                std::slice::from_raw_parts(t2.as_ptr() as *const u64, tl2),
            )
        };
        let q_len = mul_stack_small(kernels, q_out, q1s, q2s);
        let mut qt_raw = [MaybeUninit::uninit(); 38];
        let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, q2s, t1s);
        let ws = (qs2 >> 6) as usize;
        let bs = (qs2 & 63) as u32;
        let t_ptr = t_out.as_mut_ptr() as *mut u64;
        let qt_ptr = qt_raw.as_ptr() as *const u64;
        let qt_len = unsafe {
            if qt_raw_len == 0 {
                0
            } else if bs == 0 {
                std::ptr::write_bytes(t_ptr, 0, ws);
                std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
                ws + qt_raw_len
            } else {
                std::ptr::write_bytes(t_ptr, 0, ws);
                let mut c = 0u64;
                for i in 0..qt_raw_len {
                    let w = *qt_ptr.add(i);
                    *t_ptr.add(ws + i) = (w << bs) | c;
                    c = w >> (64 - bs);
                }
                *t_ptr.add(ws + qt_raw_len) = c;
                ws + qt_raw_len + 1
            }
        };
        let mut pt_arr = [MaybeUninit::uninit(); 34];
        let pt_len = mul_stack_small(kernels, &mut pt_arr, p1s, t2s);
        let pt_ptr = pt_arr.as_ptr() as *const u64;
        let clear_end = (qt_len.max(pt_len) + 2).min(44);
        let t_len = unsafe {
            if qt_len < clear_end {
                std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
            }
            let mut carry = false;
            for i in 0..pt_len {
                let (s1, o1) = (*t_ptr.add(i)).overflowing_add(*pt_ptr.add(i));
                let (s2, o2) = s1.overflowing_add(u64::from(carry));
                *t_ptr.add(i) = s2;
                carry = o1 | o2;
            }
            let mut idx = pt_len;
            while carry {
                let (s, ov) = (*t_ptr.add(idx)).overflowing_add(1);
                *t_ptr.add(idx) = s;
                carry = ov;
                idx += 1;
            }
            let mut tl = qt_len.max(idx);
            while tl > 0 && *t_ptr.add(tl - 1) == 0 {
                tl -= 1;
            }
            tl
        };
        let p_len = if need_p {
            mul_stack_small(kernels, p_out, p1s, p2s)
        } else {
            0
        };
        return (p_len, q_len, q_shift, t_len);
    }
    if len == 4 {
        let (pa, pl, qa, ql, qs, ta, tl) = eval_4_stack(kernels, start_k, need_p);
        unsafe {
            std::ptr::copy_nonoverlapping(pa.as_ptr(), p_out.as_mut_ptr(), pl);
            std::ptr::copy_nonoverlapping(qa.as_ptr(), q_out.as_mut_ptr(), ql);
            std::ptr::copy_nonoverlapping(ta.as_ptr(), t_out.as_mut_ptr(), tl);
        }
        return (pl, ql, qs, tl);
    }
    if len == 1 {
        let k = start_k + 1;
        let six_k = k * 6;
        let p_u128 = if k <= 715_827_882 {
            (((six_k - 5) * (six_k - 1)) as u128) * ((k * 2 - 1) as u128)
        } else {
            (((six_k - 5) as u128) * ((k * 2 - 1) as u128)) * ((six_k - 1) as u128)
        };
        let p0 = p_u128 as u64;
        let p1 = (p_u128 >> 64) as u64;
        let p_len = if need_p {
            p_out[0].write(p0);
            p_out[1].write(p1);
            if p1 != 0 {
                2
            } else {
                usize::from(p0 != 0)
            }
        } else {
            0
        };

        let v2 = k.trailing_zeros() as u64;
        let q_shift = 15 + 3 * v2;
        let k_odd_u64 = k >> v2;
        let k_odd_cb = if k_odd_u64 < (1u64 << 32) {
            ((k_odd_u64 * k_odd_u64) as u128) * (k_odd_u64 as u128)
        } else {
            let k_odd = k_odd_u64 as u128;
            (k_odd * k_odd) * k_odd
        };
        let q_lo = ((k_odd_cb as u64) as u128) * (CHUD_Q_ODD_CONST as u128);
        let q_hi = ((k_odd_cb >> 64) as u128) * (CHUD_Q_ODD_CONST as u128) + (q_lo >> 64);
        let q0 = q_lo as u64;
        let q1 = q_hi as u64;
        let q2 = (q_hi >> 64) as u64;
        q_out[0].write(q0);
        q_out[1].write(q1);
        q_out[2].write(q2);
        let q_len = if q2 != 0 {
            3
        } else if q1 != 0 {
            2
        } else {
            usize::from(q0 != 0)
        };

        let lin = (CHUD_A + CHUD_B * k) as u128;
        let t_lo = (p0 as u128) * lin;
        let t_hi = (p1 as u128) * lin + (t_lo >> 64);
        let tw0 = t_lo as u64;
        let tw1 = t_hi as u64;
        let tw2 = (t_hi >> 64) as u64;
        t_out[0].write(tw0);
        t_out[1].write(tw1);
        t_out[2].write(tw2);
        let t_len = if tw2 != 0 {
            3
        } else if tw1 != 0 {
            2
        } else {
            usize::from(tw0 != 0)
        };
        return (p_len, q_len, q_shift, t_len);
    }

    if len == 2 {
        let (pa, pl, qa, ql, qs, ta, tl) = eval_pair_stack(start_k, need_p);
        unsafe {
            std::ptr::copy_nonoverlapping(pa.as_ptr(), p_out.as_mut_ptr() as *mut u64, pl);
            std::ptr::copy_nonoverlapping(qa.as_ptr(), q_out.as_mut_ptr() as *mut u64, ql);
            std::ptr::copy_nonoverlapping(ta.as_ptr(), t_out.as_mut_ptr() as *mut u64, tl);
        }
        return (pl, ql, qs, tl);
    }

    if len == 3 {
        let (pa, pl, qa, ql, qs, ta, tl) = eval_3_stack(start_k, need_p);
        unsafe {
            std::ptr::copy_nonoverlapping(pa.as_ptr(), p_out.as_mut_ptr() as *mut u64, pl);
            std::ptr::copy_nonoverlapping(qa.as_ptr(), q_out.as_mut_ptr() as *mut u64, ql);
            std::ptr::copy_nonoverlapping(ta.as_ptr(), t_out.as_mut_ptr() as *mut u64, tl);
        }
        return (pl, ql, qs, tl);
    }

    let half = len / 2;
    let (mut p1, mut q1, mut t1) = (
        [MaybeUninit::uninit(); 22],
        [MaybeUninit::uninit(); 30],
        [MaybeUninit::uninit(); 44],
    );
    let (mut p2, mut q2, mut t2) = (
        [MaybeUninit::uninit(); 22],
        [MaybeUninit::uninit(); 30],
        [MaybeUninit::uninit(); 44],
    );
    let (pl1, ql1, qs1, tl1) =
        eval_small_stack_into(kernels, start_k, half, true, &mut p1, &mut q1, &mut t1);
    let (pl2, ql2, qs2, tl2) = eval_small_stack_into(
        kernels,
        start_k + half,
        len - half,
        need_p,
        &mut p2,
        &mut q2,
        &mut t2,
    );
    let q_shift = qs1 + qs2;
    let (p1s, q1s, t1s, p2s, q2s, t2s) = unsafe {
        (
            std::slice::from_raw_parts(p1.as_ptr() as *const u64, pl1),
            std::slice::from_raw_parts(q1.as_ptr() as *const u64, ql1),
            std::slice::from_raw_parts(t1.as_ptr() as *const u64, tl1),
            std::slice::from_raw_parts(p2.as_ptr() as *const u64, pl2),
            std::slice::from_raw_parts(q2.as_ptr() as *const u64, ql2),
            std::slice::from_raw_parts(t2.as_ptr() as *const u64, tl2),
        )
    };

    let q_len = mul_stack_small(kernels, q_out, q1s, q2s);

    let mut qt_raw = [MaybeUninit::uninit(); 38];
    let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, q2s, t1s);
    let ws = (qs2 >> 6) as usize;
    let bs = (qs2 & 63) as u32;
    let t_ptr = t_out.as_mut_ptr() as *mut u64;
    let qt_ptr = qt_raw.as_ptr() as *const u64;
    let qt_len = unsafe {
        if qt_raw_len == 0 {
            0
        } else if bs == 0 {
            std::ptr::write_bytes(t_ptr, 0, ws);
            std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
            ws + qt_raw_len
        } else {
            std::ptr::write_bytes(t_ptr, 0, ws);
            let mut c = 0u64;
            for i in 0..qt_raw_len {
                let w = *qt_ptr.add(i);
                *t_ptr.add(ws + i) = (w << bs) | c;
                c = w >> (64 - bs);
            }
            *t_ptr.add(ws + qt_raw_len) = c;
            ws + qt_raw_len + 1
        }
    };

    let mut pt_arr = [MaybeUninit::uninit(); 34];
    let pt_len = mul_stack_small(kernels, &mut pt_arr, p1s, t2s);
    let pt_ptr = pt_arr.as_ptr() as *const u64;
    let clear_end = (qt_len.max(pt_len) + 2).min(44);
    let t_len = unsafe {
        if qt_len < clear_end {
            std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
        }
        let mut tl = if (half & 1) == 0 {
            let mut carry = false;
            for i in 0..pt_len {
                let (s1, o1) = (*t_ptr.add(i)).overflowing_add(*pt_ptr.add(i));
                let (s2, o2) = s1.overflowing_add(u64::from(carry));
                *t_ptr.add(i) = s2;
                carry = o1 | o2;
            }
            let mut idx = pt_len;
            while carry {
                let (s, ov) = (*t_ptr.add(idx)).overflowing_add(1);
                *t_ptr.add(idx) = s;
                carry = ov;
                idx += 1;
            }
            qt_len.max(idx)
        } else {
            let mut borrow = false;
            for i in 0..qt_len {
                let sub_w = if i < pt_len { *pt_ptr.add(i) } else { 0 };
                let (d1, b1) = (*t_ptr.add(i)).overflowing_sub(sub_w);
                let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
                *t_ptr.add(i) = d2;
                borrow = b1 | b2;
            }
            debug_assert!(!borrow);
            qt_len
        };
        while tl > 0 && *t_ptr.add(tl - 1) == 0 {
            tl -= 1;
        }
        tl
    };

    let p_len = if need_p {
        mul_stack_small(kernels, p_out, p1s, p2s)
    } else {
        0
    };
    (p_len, q_len, q_shift, t_len)
}

fn leaf_rec_pool<A: std::alloc::Allocator>(
    ctx: &MulCtx<'_>,
    pool: &mut Vec<u64, A>,
    start_k: u64,
    len: u64,
    need_p: bool,
    basecase_cutoff: usize,
) -> SmallTriple {
    let mark = pool.len();
    if len == 1 {
        let k = start_k + 1;
        if k < (1u64 << 34) {
            let six_k = k * 6;
            let p_u128 = if k <= 715_827_882 {
                (((six_k - 5) * (six_k - 1)) as u128) * ((k * 2 - 1) as u128)
            } else {
                (((six_k - 5) as u128) * ((k * 2 - 1) as u128)) * ((six_k - 1) as u128)
            };
            let p0 = p_u128 as u64;
            let p1 = (p_u128 >> 64) as u64;
            let p_arr = [p0, p1];
            let p_len = if p1 != 0 { 2 } else { usize::from(p0 != 0) };

            let v2 = k.trailing_zeros() as u64;
            let q_shift = 15 + 3 * v2;
            let k_odd_u64 = k >> v2;
            let k_odd_cb = if k_odd_u64 < (1u64 << 32) {
                ((k_odd_u64 * k_odd_u64) as u128) * (k_odd_u64 as u128)
            } else {
                let k_odd = k_odd_u64 as u128;
                (k_odd * k_odd) * k_odd
            };
            let q_lo = ((k_odd_cb as u64) as u128) * (CHUD_Q_ODD_CONST as u128);
            let q_hi = ((k_odd_cb >> 64) as u128) * (CHUD_Q_ODD_CONST as u128) + (q_lo >> 64);
            let q_arr = [q_lo as u64, q_hi as u64, (q_hi >> 64) as u64];
            let q_len = if q_arr[2] != 0 {
                3
            } else if q_arr[1] != 0 {
                2
            } else {
                usize::from(q_arr[0] != 0)
            };

            let lin = (CHUD_A + CHUD_B * k) as u128;
            let t_lo = (p0 as u128) * lin;
            let t_hi = (p1 as u128) * lin + (t_lo >> 64);
            let t_arr = [t_lo as u64, t_hi as u64, (t_hi >> 64) as u64];
            let t_len = if t_arr[2] != 0 {
                3
            } else if t_arr[1] != 0 {
                2
            } else {
                usize::from(t_arr[0] != 0)
            };

            if ctx.kernels.count_work {
                ctx.kernels.stats.limb_ops.fetch_add(6, Ordering::Relaxed);
            }

            let p_span = if need_p && p_len > 0 {
                let (po, pl) = push_slice(pool, &p_arr[..p_len]);
                SmallSpan {
                    off: po,
                    len: pl,
                    neg: true,
                    shift: 0,
                }
            } else {
                SmallSpan::ZERO
            };
            let (qo, ql) = push_slice(pool, &q_arr[..q_len]);
            let (to, tl) = push_slice(pool, &t_arr[..t_len]);
            return SmallTriple {
                p: p_span,
                q: SmallSpan {
                    off: qo,
                    len: ql,
                    neg: false,
                    shift: q_shift,
                },
                t: SmallSpan {
                    off: to,
                    len: tl,
                    neg: tl > 0,
                    shift: 0,
                },
            };
        }

        let six_k = (k as u128) * 6;
        let f1 = six_k - 5;
        let f2 = (k as u128) * 2 - 1;
        let f3 = six_k - 1;
        let f12 = f1 * f2;
        let (p_arr, p_len) = mul_u128_into_arr(ctx.kernels, f12, f3);

        let v2 = k.trailing_zeros() as u64;
        let k_odd = (k >> v2) as u128;
        let k_odd_sq = k_odd * k_odd;
        let c_k = (CHUD_Q_ODD_CONST as u128) * k_odd;
        let (q_arr, q_len) = mul_u128_into_arr(ctx.kernels, k_odd_sq, c_k);
        let q_shift = 15 + 3 * v2;

        let lin = (CHUD_A as u128) + (CHUD_B as u128) * (k as u128);
        let (lin_arr, lin_len) = u128_to_arr(lin);
        let mut t_arr = [0u64; 6];
        let t_len = if p_len > 0 && lin_len > 0 {
            let (long_p, nl, short_p, ns) = if p_len >= lin_len {
                (p_arr.as_ptr(), p_len, lin_arr.as_ptr(), lin_len)
            } else {
                (lin_arr.as_ptr(), lin_len, p_arr.as_ptr(), p_len)
            };
            let lens = ctx.kernels.mul_base_lens(nl, ns);
            if ctx.kernels.count_work {
                ctx.kernels
                    .stats
                    .limb_ops
                    .fetch_add((nl * ns) as u64, Ordering::Relaxed);
            }
            unsafe {
                (ctx.kernels.active.mul_base)(t_arr.as_mut_ptr(), long_p, short_p, &lens);
            }
            let mut tl = nl + ns;
            while tl > 0 && t_arr[tl - 1] == 0 {
                tl -= 1;
            }
            tl
        } else {
            0
        };

        let p_span = if need_p && p_len > 0 {
            let (po, pl) = push_slice(pool, &p_arr[..p_len]);
            SmallSpan {
                off: po,
                len: pl,
                neg: true,
                shift: 0,
            }
        } else {
            SmallSpan::ZERO
        };
        let (qo, ql) = push_slice(pool, &q_arr[..q_len]);
        let (to, tl) = push_slice(pool, &t_arr[..t_len]);
        return SmallTriple {
            p: p_span,
            q: SmallSpan {
                off: qo,
                len: ql,
                neg: false,
                shift: q_shift,
            },
            t: SmallSpan {
                off: to,
                len: tl,
                neg: tl > 0,
                shift: 0,
            },
        };
    }

    if len == 2 && !ctx.kernels.count_work && start_k + 2 < (1u64 << 34) {
        let (p_arr, p_len, q_arr, q_len, q_shift, t_arr, t_len) =
            eval_pair_stack(start_k, need_p);
        let (qo, ql) = push_slice(pool, &q_arr[..q_len]);
        let (to, tl) = push_slice(pool, &t_arr[..t_len]);
        let p_span = if need_p && p_len > 0 {
            let (po, pl) = push_slice(pool, &p_arr[..p_len]);
            SmallSpan {
                off: po,
                len: pl,
                neg: false,
                shift: 0,
            }
        } else {
            SmallSpan::ZERO
        };
        return SmallTriple {
            p: p_span,
            q: SmallSpan {
                off: qo,
                len: ql,
                neg: false,
                shift: q_shift,
            },
            t: SmallSpan {
                off: to,
                len: tl,
                neg: tl > 0,
                shift: 0,
            },
        };
    }

    if len == 4 && !ctx.kernels.count_work && start_k + 4 < (1u64 << 34) {
        let (p_arr, p_len, q_arr, q_len, q_shift, t_arr, t_len) =
            eval_4_stack(ctx.kernels, start_k, need_p);
        let (qs, ts, ps) = unsafe {
            (
                std::slice::from_raw_parts(q_arr.as_ptr() as *const u64, q_len),
                std::slice::from_raw_parts(t_arr.as_ptr() as *const u64, t_len),
                std::slice::from_raw_parts(p_arr.as_ptr() as *const u64, p_len),
            )
        };
        let (qo, ql) = push_slice(pool, qs);
        let (to, tl) = push_slice(pool, ts);
        let p_span = if need_p && p_len > 0 {
            let (po, pl) = push_slice(pool, ps);
            SmallSpan {
                off: po,
                len: pl,
                neg: false,
                shift: 0,
            }
        } else {
            SmallSpan::ZERO
        };
        return SmallTriple {
            p: p_span,
            q: SmallSpan {
                off: qo,
                len: ql,
                neg: false,
                shift: q_shift,
            },
            t: SmallSpan {
                off: to,
                len: tl,
                neg: tl > 0,
                shift: 0,
            },
        };
    }

    if (11..=20).contains(&len)
        && !ctx.kernels.count_work
        && start_k + len < (1u64 << 34)
    {
        return eval_11_20_into_pool(ctx.kernels, pool, mark, start_k, len, need_p);
    }

    if (3..=10).contains(&len) && !ctx.kernels.count_work && start_k + len < (1u64 << 34) {
        use std::mem::MaybeUninit;
        let (mut p_arr, mut q_arr, mut t_arr) = (
            [MaybeUninit::uninit(); 22],
            [MaybeUninit::uninit(); 30],
            [MaybeUninit::uninit(); 44],
        );
        let (p_len, q_len, q_shift, t_len) = eval_small_stack_into(
            ctx.kernels,
            start_k,
            len,
            need_p,
            &mut p_arr,
            &mut q_arr,
            &mut t_arr,
        );
        let (qs, ts, ps) = unsafe {
            (
                std::slice::from_raw_parts(q_arr.as_ptr() as *const u64, q_len),
                std::slice::from_raw_parts(t_arr.as_ptr() as *const u64, t_len),
                std::slice::from_raw_parts(p_arr.as_ptr() as *const u64, p_len),
            )
        };
        let (qo, ql) = push_slice(pool, qs);
        let (to, tl) = push_slice(pool, ts);
        let p_span = if need_p && p_len > 0 {
            let (po, pl) = push_slice(pool, ps);
            SmallSpan {
                off: po,
                len: pl,
                neg: (len & 1) == 1,
                shift: 0,
            }
        } else {
            SmallSpan::ZERO
        };
        return SmallTriple {
            p: p_span,
            q: SmallSpan {
                off: qo,
                len: ql,
                neg: false,
                shift: q_shift,
            },
            t: SmallSpan {
                off: to,
                len: tl,
                neg: tl > 0,
                shift: 0,
            },
        };
    }

    let half = len / 2;
    let left = leaf_rec_pool(ctx, pool, start_k, half, true, basecase_cutoff);
    let right = leaf_rec_pool(ctx, pool, start_k + half, len - half, need_p, basecase_cutoff);

    let (q_off, q_len) = mul_pool_spans(
        ctx.kernels,
        pool,
        left.q.off,
        left.q.len,
        right.q.off,
        right.q.len,
        basecase_cutoff,
    );
    let q_shift = left.q.shift + right.q.shift;

    // qt1 = (right.q * left.t) << qt_shift, pre-padding `ws` zero limbs before `qt_raw` and
    // shifting in place via `mul_1` so `qt` lands directly after `q` with zero extra copies.
    let qt_shift = right.q.shift + left.t.shift;
    let (qt_off, qt_len) = if right.q.len == 0 || left.t.len == 0 {
        (pool.len(), 0)
    } else {
        let ws = (qt_shift >> 6) as usize;
        let bs = (qt_shift & 63) as u32;
        let out_o = pool.len();
        if ws > 0 {
            pool.resize(out_o + ws, 0);
        }
        let (qt_raw_off, qt_raw_len) = mul_pool_spans(
            ctx.kernels,
            pool,
            right.q.off,
            right.q.len,
            left.t.off,
            left.t.len,
            basecase_cutoff,
        );
        debug_assert_eq!(qt_raw_off, out_o + ws);
        if qt_raw_len == 0 {
            pool.truncate(out_o);
            (out_o, 0)
        } else if bs == 0 {
            (out_o, ws + qt_raw_len)
        } else {
            pool.reserve(1);
            let mut sl = ws + qt_raw_len;
            unsafe {
                let p = pool.as_mut_ptr().add(out_o + ws);
                let hi = (ctx.kernels.active.mul_1)(p, p, qt_raw_len as u64, 1u64 << bs);
                if hi != 0 {
                    *p.add(qt_raw_len) = hi;
                    sl += 1;
                }
                pool.set_len(out_o + sl);
            }
            while sl > 0 && pool[out_o + sl - 1] == 0 {
                sl -= 1;
            }
            pool.truncate(out_o + sl);
            (out_o, sl)
        }
    };
    let qt_neg = left.t.neg && qt_len > 0;

    // pt2 = left.p * right.t (shift == 0 inside plain leaf), placed immediately at `qt_off + qt_len`.
    let (pt_off, pt_len) = mul_pool_spans(
        ctx.kernels,
        pool,
        left.p.off,
        left.p.len,
        right.t.off,
        right.t.len,
        basecase_cutoff,
    );
    let pt_neg = (left.p.neg ^ right.t.neg) && pt_len > 0;

    // t = qt1 + pt2 accumulated in place at `qt_off` (`pt_off == qt_off + qt_len`), avoiding a
    // third temporary buffer and truncating `pt2` immediately before computing `p`.
    let (t_off, t_len, t_neg) = if qt_len == 0 {
        (pt_off, pt_len, pt_neg)
    } else if pt_len == 0 {
        pool.truncate(qt_off + qt_len);
        (qt_off, qt_len, qt_neg)
    } else if qt_neg == pt_neg {
        let nl = qt_len.max(pt_len);
        let ns = qt_len.min(pt_len);
        if ctx.kernels.count_work {
            ctx.kernels
                .stats
                .limb_ops
                .fetch_add(nl as u64, Ordering::Relaxed);
        }
        pool.reserve(1);
        unsafe {
            let bp = pool.as_mut_ptr();
            let mut carry = if ns <= 16 {
                let mut c = false;
                for i in 0..ns {
                    let (s1, o1) = (*bp.add(qt_off + i)).overflowing_add(*bp.add(pt_off + i));
                    let (s2, o2) = s1.overflowing_add(u64::from(c));
                    *bp.add(qt_off + i) = s2;
                    c = o1 | o2;
                }
                u64::from(c)
            } else {
                (ctx.kernels.active.add_n)(
                    bp.add(qt_off),
                    bp.add(qt_off),
                    bp.add(pt_off),
                    ns as u64,
                )
            };
            if pt_len > qt_len {
                for i in qt_len..pt_len {
                    let (s, ov) = (*bp.add(pt_off + i)).overflowing_add(carry);
                    *bp.add(qt_off + i) = s;
                    carry = ov as u64;
                }
            } else {
                for i in pt_len..qt_len {
                    if carry == 0 {
                        break;
                    }
                    let (s, ov) = (*bp.add(qt_off + i)).overflowing_add(carry);
                    *bp.add(qt_off + i) = s;
                    carry = ov as u64;
                }
            }
            let mut tl = nl;
            if carry != 0 {
                *bp.add(qt_off + nl) = carry;
                tl += 1;
            }
            pool.set_len(qt_off + tl);
            while tl > 0 && *bp.add(qt_off + tl - 1) == 0 {
                tl -= 1;
            }
            pool.truncate(qt_off + tl);
            (qt_off, tl, qt_neg && tl > 0)
        }
    } else {
        let ord = cmp_limbs(&pool[qt_off..(qt_off + qt_len)], &pool[pt_off..(pt_off + pt_len)]);
        match ord {
            std::cmp::Ordering::Equal => {
                pool.truncate(qt_off);
                (qt_off, 0, false)
            }
            std::cmp::Ordering::Greater => {
                if ctx.kernels.count_work {
                    ctx.kernels
                        .stats
                        .limb_ops
                        .fetch_add(qt_len as u64, Ordering::Relaxed);
                }
                unsafe {
                    let bp = pool.as_mut_ptr();
                    let mut borrow = if pt_len <= 16 {
                        let mut b = false;
                        for i in 0..pt_len {
                            let (d1, u1) =
                                (*bp.add(qt_off + i)).overflowing_sub(*bp.add(pt_off + i));
                            let (d2, u2) = d1.overflowing_sub(u64::from(b));
                            *bp.add(qt_off + i) = d2;
                            b = u1 | u2;
                        }
                        u64::from(b)
                    } else {
                        (ctx.kernels.active.sub_n)(
                            bp.add(qt_off),
                            bp.add(qt_off),
                            bp.add(pt_off),
                            pt_len as u64,
                        )
                    };
                    for i in pt_len..qt_len {
                        if borrow == 0 {
                            break;
                        }
                        let (d, un) = (*bp.add(qt_off + i)).overflowing_sub(borrow);
                        *bp.add(qt_off + i) = d;
                        borrow = un as u64;
                    }
                    let mut tl = qt_len;
                    while tl > 0 && *bp.add(qt_off + tl - 1) == 0 {
                        tl -= 1;
                    }
                    pool.truncate(qt_off + tl);
                    (qt_off, tl, qt_neg && tl > 0)
                }
            }
            std::cmp::Ordering::Less => {
                if ctx.kernels.count_work {
                    ctx.kernels
                        .stats
                        .limb_ops
                        .fetch_add(pt_len as u64, Ordering::Relaxed);
                }
                unsafe {
                    let bp = pool.as_mut_ptr();
                    let mut borrow = if qt_len <= 16 {
                        let mut b = false;
                        for i in 0..qt_len {
                            let (d1, u1) =
                                (*bp.add(pt_off + i)).overflowing_sub(*bp.add(qt_off + i));
                            let (d2, u2) = d1.overflowing_sub(u64::from(b));
                            *bp.add(qt_off + i) = d2;
                            b = u1 | u2;
                        }
                        u64::from(b)
                    } else {
                        (ctx.kernels.active.sub_n)(
                            bp.add(qt_off),
                            bp.add(pt_off),
                            bp.add(qt_off),
                            qt_len as u64,
                        )
                    };
                    for i in qt_len..pt_len {
                        let (d, un) = (*bp.add(pt_off + i)).overflowing_sub(borrow);
                        *bp.add(qt_off + i) = d;
                        borrow = un as u64;
                    }
                    let mut tl = pt_len;
                    while tl > 0 && *bp.add(qt_off + tl - 1) == 0 {
                        tl -= 1;
                    }
                    pool.truncate(qt_off + tl);
                    (qt_off, tl, pt_neg && tl > 0)
                }
            }
        }
    };

    let (p_off, p_len, p_neg) = if need_p {
        let (po, pl) = mul_pool_spans(
            ctx.kernels,
            pool,
            left.p.off,
            left.p.len,
            right.p.off,
            right.p.len,
            basecase_cutoff,
        );
        (po, pl, (left.p.neg ^ right.p.neg) && pl > 0)
    } else {
        (pool.len(), 0, false)
    };

    // Compact `(q, t, p)` down to `mark`: when contiguous (`t_off == q_off + q_len` and
    // `p_off == t_off + t_len`), coalesce into a single `copy_within` keeping `pool` hot in L1d!
    let final_q_off = mark;
    let final_t_off = final_q_off + q_len;
    let final_p_off = final_t_off + t_len;
    let total_out_len = q_len + t_len + p_len;
    if total_out_len > 0
        && (t_len == 0 || t_off == q_off + q_len)
        && (p_len == 0 || p_off == t_off + t_len)
    {
        let src_start = if q_len > 0 {
            q_off
        } else if t_len > 0 {
            t_off
        } else {
            p_off
        };
        if src_start != mark {
            pool.copy_within(src_start..(src_start + total_out_len), mark);
        }
    } else {
        if q_len > 0 && q_off != final_q_off {
            pool.copy_within(q_off..(q_off + q_len), final_q_off);
        }
        if t_len > 0 && t_off != final_t_off {
            pool.copy_within(t_off..(t_off + t_len), final_t_off);
        }
        if p_len > 0 && p_off != final_p_off {
            pool.copy_within(p_off..(p_off + p_len), final_p_off);
        }
    }
    pool.truncate(final_p_off + p_len);

    SmallTriple {
        p: SmallSpan {
            off: final_p_off,
            len: p_len,
            neg: p_neg,
            shift: 0,
        },
        q: SmallSpan {
            off: final_q_off,
            len: q_len,
            neg: false,
            shift: q_shift,
        },
        t: SmallSpan {
            off: final_t_off,
            len: t_len,
            neg: t_neg,
            shift: 0,
        },
    }
}

#[inline(never)]
fn eval_11_20_into_pool<A: std::alloc::Allocator>(
    kernels: &crate::refk::Kernels,
    pool: &mut Vec<u64, A>,
    mark: usize,
    start_k: u64,
    len: u64,
    need_p: bool,
) -> SmallTriple {
    use std::mem::MaybeUninit;
    let half = len / 2;
    let rlen = len - half;
    let (mut p1_arr, mut q1_arr, mut t1_arr) = (
        [MaybeUninit::uninit(); 22],
        [MaybeUninit::uninit(); 30],
        [MaybeUninit::uninit(); 44],
    );
    let (mut p2_arr, mut q2_arr, mut t2_arr) = (
        [MaybeUninit::uninit(); 22],
        [MaybeUninit::uninit(); 30],
        [MaybeUninit::uninit(); 44],
    );
    let (pl1, ql1, qs1, tl1) = eval_small_stack_into(
        kernels,
        start_k,
        half,
        true,
        &mut p1_arr,
        &mut q1_arr,
        &mut t1_arr,
    );
    let (pl2, ql2, qs2, tl2) = eval_small_stack_into(
        kernels,
        start_k + half,
        rlen,
        need_p,
        &mut p2_arr,
        &mut q2_arr,
        &mut t2_arr,
    );
    let q_shift = qs1 + qs2;
    let (p1s, q1s, t1s, p2s, q2s, t2s) = unsafe {
        (
            std::slice::from_raw_parts(p1_arr.as_ptr() as *const u64, pl1),
            std::slice::from_raw_parts(q1_arr.as_ptr() as *const u64, ql1),
            std::slice::from_raw_parts(t1_arr.as_ptr() as *const u64, tl1),
            std::slice::from_raw_parts(p2_arr.as_ptr() as *const u64, pl2),
            std::slice::from_raw_parts(q2_arr.as_ptr() as *const u64, ql2),
            std::slice::from_raw_parts(t2_arr.as_ptr() as *const u64, tl2),
        )
    };
    let ws = (qs2 >> 6) as usize;
    let bs = (qs2 & 63) as u32;
    let need_cap = ql1 + ql2 + tl1 + ql2 + ws + pl1 + tl2 + pl1 + pl2 + 8;
    pool.reserve(need_cap);
    let (final_q_off, q_len, final_t_off, t_len, final_p_off, p_len) = unsafe {
        let bp = pool.as_mut_ptr();
        let q_dst =
            std::slice::from_raw_parts_mut(bp.add(mark) as *mut MaybeUninit<u64>, ql1 + ql2);
        let ql = mul_stack_small(kernels, q_dst, q1s, q2s);

        let mut qt_raw = [MaybeUninit::uninit(); 74];
        let qt_raw_len = mul_stack_small(kernels, &mut qt_raw, q2s, t1s);
        let mut pt_arr = [MaybeUninit::uninit(); 66];
        let pt_len = mul_stack_small(kernels, &mut pt_arr, p1s, t2s);

        let t_off = mark + ql;
        let t_ptr = bp.add(t_off);
        let qt_ptr = qt_raw.as_ptr() as *const u64;
        let qt_len = if qt_raw_len == 0 {
            0
        } else if bs == 0 {
            std::ptr::write_bytes(t_ptr, 0, ws);
            std::ptr::copy_nonoverlapping(qt_ptr, t_ptr.add(ws), qt_raw_len);
            ws + qt_raw_len
        } else {
            std::ptr::write_bytes(t_ptr, 0, ws);
            let mut c = 0u64;
            for i in 0..qt_raw_len {
                let w = *qt_ptr.add(i);
                *t_ptr.add(ws + i) = (w << bs) | c;
                c = w >> (64 - bs);
            }
            *t_ptr.add(ws + qt_raw_len) = c;
            ws + qt_raw_len + 1
        };
        let pt_ptr = pt_arr.as_ptr() as *const u64;
        let tl = if (half & 1) == 0 {
            let clear_end = qt_len.max(pt_len) + 2;
            if qt_len < clear_end {
                std::ptr::write_bytes(t_ptr.add(qt_len), 0, clear_end - qt_len);
            }
            let mut cy = if pt_len == 0 {
                0u64
            } else {
                (kernels.active.add_n)(t_ptr, t_ptr, pt_ptr, pt_len as u64)
            };
            let mut idx = pt_len;
            while cy != 0 {
                let (s, ov) = (*t_ptr.add(idx)).overflowing_add(cy);
                *t_ptr.add(idx) = s;
                cy = u64::from(ov);
                idx += 1;
            }
            let mut len_t = qt_len.max(idx);
            while len_t > 0 && *t_ptr.add(len_t - 1) == 0 {
                len_t -= 1;
            }
            len_t
        } else {
            let mut br = if pt_len == 0 {
                0u64
            } else {
                (kernels.active.sub_n)(t_ptr, t_ptr, pt_ptr, pt_len as u64)
            };
            let mut idx = pt_len;
            while br != 0 {
                let (d, un) = (*t_ptr.add(idx)).overflowing_sub(br);
                *t_ptr.add(idx) = d;
                br = u64::from(un);
                idx += 1;
            }
            let mut len_t = qt_len;
            while len_t > 0 && *t_ptr.add(len_t - 1) == 0 {
                len_t -= 1;
            }
            len_t
        };

        let p_off = t_off + tl;
        let pl = if need_p {
            let p_dst =
                std::slice::from_raw_parts_mut(bp.add(p_off) as *mut MaybeUninit<u64>, pl1 + pl2);
            mul_stack_small(kernels, p_dst, p1s, p2s)
        } else {
            0
        };
        pool.set_len(p_off + pl);
        (mark, ql, t_off, tl, p_off, pl)
    };
    SmallTriple {
        p: if need_p && p_len > 0 {
            SmallSpan {
                off: final_p_off,
                len: p_len,
                neg: (len & 1) == 1,
                shift: 0,
            }
        } else {
            SmallSpan::ZERO
        },
        q: SmallSpan {
            off: final_q_off,
            len: q_len,
            neg: false,
            shift: q_shift,
        },
        t: SmallSpan {
            off: final_t_off,
            len: t_len,
            neg: t_len > 0,
            shift: 0,
        },
    }
}

#[inline]
fn span_to_bignum(pool: &[u64], sp: SmallSpan, kernels: &crate::refk::Kernels) -> BigNum {
    if sp.len == 0 {
        return BigNum::zero();
    }
    let mut v = crate::limb::take_limb_buf_uninit(sp.len);
    v.extend_from_slice(&pool[sp.off..(sp.off + sp.len)]);
    BigNum::from_limbs(v, sp.neg, sp.shift, kernels)
}

/// Plain binary splitting inside an L1-sized leaf `(start_k, start_k + len]`.
fn compute_leaf_plain(
    ctx: &MulCtx<'_>,
    start_k: u64,
    len: u64,
    need_p: bool,
    basecase_cutoff: usize,
) -> BlockTriple {
    debug_assert!(len >= 1);
    LEAF_POOL.with(|cell| {
        if let Ok(mut pool) = cell.try_borrow_mut() {
            pool.clear();
            if pool.capacity() < 8192 {
                pool.resize(8192, 0);
                pool.clear();
            }
            let st = leaf_rec_pool(ctx, &mut pool, start_k, len, need_p, basecase_cutoff);
            let p = span_to_bignum(&pool, st.p, ctx.kernels);
            let q = span_to_bignum(&pool, st.q, ctx.kernels);
            let t = span_to_bignum(&pool, st.t, ctx.kernels);
            pool.clear();
            BlockTriple { p, q, t }
        } else {
            let mut pool = vec![0u64; 8192];
            pool.clear();
            let st = leaf_rec_pool(ctx, &mut pool, start_k, len, need_p, basecase_cutoff);
            BlockTriple {
                p: span_to_bignum(&pool, st.p, ctx.kernels),
                q: span_to_bignum(&pool, st.q, ctx.kernels),
                t: span_to_bignum(&pool, st.t, ctx.kernels),
            }
        }
    })
}

/// Build the odd level factor `F_ell = R_{L_ell} / R_{L_prev}^{2^m}` from primes `<= L_ell` using
/// a balanced binary product tree (Section 4).
pub fn build_level_factor(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    primes: &[u64],
    l_cur: u64,
    l_prev: u64,
    span_m: u32,
) -> Result<BigNum, MulError> {
    let scale_prev = if l_prev == 0 { 0u64 } else { 1u64 << span_m };
    let mut word_leaves: Vec<u64> = Vec::new();
    let mut big_leaves: Vec<BigNum> = Vec::new();
    let mut cur_word: u64 = 1;

    for &r in primes {
        if r < 3 {
            continue;
        }
        if r > l_cur {
            break;
        }
        let e_cur = prime_reduction_exp(l_cur, r);
        let e_prev = if l_prev == 0 {
            0
        } else {
            prime_reduction_exp(l_prev, r)
        };
        debug_assert!(e_cur >= scale_prev * e_prev);
        let delta = e_cur - scale_prev * e_prev;
        if delta == 0 {
            continue;
        }
        let exp = if r == 3 { delta } else { 3 * delta };

        // Accumulate `r^exp` by packing into `cur_word` until u64 fills!
        let mut rem_e = exp;
        while rem_e > 0 {
            if let Some(prod) = cur_word.checked_mul(r) {
                cur_word = prod;
                rem_e -= 1;
            } else if cur_word > 1 {
                word_leaves.push(cur_word);
                cur_word = 1;
            } else {
                // `r` itself fits in u64; raise `r^rem_e` via BigNum squaring.
                big_leaves.push(pow_u64_bignum(ctx, plan, shares, cal, r, rem_e)?);
                rem_e = 0;
            }
        }
    }
    if cur_word > 1 {
        word_leaves.push(cur_word);
    }
    if word_leaves.is_empty() && big_leaves.is_empty() {
        return Ok(BigNum::from_u64(1));
    }

    // Coalesce packed 64-bit word leaves into ~64-word chunks in parallel with zero 1-limb Vec allocs!
    let active_w = shares.active_workers.max(1);
    let mut leaves: Vec<BigNum> = if word_leaves.len() >= 16 {
        let chunk_sz = 64usize;
        let num_chunks = word_leaves.len().div_ceil(chunk_sz);
        let mut chunked: Vec<Option<BigNum>> = (0..num_chunks).map(|_| None).collect();
        let chunked_ptr = chunked.as_mut_ptr() as usize;
        let words_slice = word_leaves.as_slice();
        let basecase_cut = cal.basecase_max_limbs;
        ctx.sched.par_for(num_chunks, 1, |c_idx| {
            let start = c_idx * chunk_sz;
            let end = (start + chunk_sz).min(words_slice.len());
            let sub = &words_slice[start..end];
            let mut local: Vec<BigNum> = Vec::with_capacity(sub.len().div_ceil(2));
            let mut wi = 0usize;
            while wi + 1 < sub.len() {
                let prod = (sub[wi] as u128) * (sub[wi + 1] as u128);
                let lo = prod as u64;
                let hi = (prod >> 64) as u64;
                let limbs = if hi == 0 { vec![lo] } else { vec![lo, hi] };
                let w0 = mod61_mul(
                    ((sub[wi] as u128) % (crate::refk::MOD_W0 as u128)) as u64,
                    ((sub[wi + 1] as u128) % (crate::refk::MOD_W0 as u128)) as u64,
                );
                local.push(BigNum::from_limbs_with_w0(limbs, false, 0, w0));
                wi += 2;
            }
            if wi < sub.len() {
                local.push(BigNum::from_u64(sub[wi]));
            }
            while local.len() > 1 {
                let mut nxt = Vec::with_capacity(local.len().div_ceil(2));
                let mut i = 0usize;
                while i + 1 < local.len() {
                    let a = &local[i];
                    let b = &local[i + 1];
                    let p_limbs = if a.limbs.len().min(b.limbs.len()) <= basecase_cut {
                        mul_limbs_basecase(ctx.kernels, &a.limbs, &b.limbs)
                    } else {
                        mul_limbs_karatsuba(ctx.kernels, &a.limbs, &b.limbs, basecase_cut)
                    };
                    let p_w0 = mod61_mul(a.w0(), b.w0());
                    nxt.push(BigNum::from_limbs_with_w0(p_limbs, false, 0, p_w0));
                    i += 2;
                }
                if i < local.len() {
                    nxt.push(std::mem::replace(&mut local[i], BigNum::zero()));
                }
                local = nxt;
            }
            unsafe {
                *(chunked_ptr as *mut Option<BigNum>).add(c_idx) = local.pop();
            }
        });
        chunked.into_iter().map(|o| o.unwrap()).collect()
    } else {
        word_leaves.into_iter().map(BigNum::from_u64).collect()
    };
    leaves.extend(big_leaves);

    // Balanced product tree over `leaves`.
    let mut step_id = 0xFA00_0000u64 ^ l_cur;
    while leaves.len() > 1 {
        let pairs = leaves.len() / 2;
        let cand_pw = if pairs >= active_w {
            1
        } else {
            (active_w / pairs.max(1)).max(1)
        };
        let sample_m = plan.select_mul_method(
            leaves[0].body_bits(),
            leaves[1].body_bits(),
            false,
            false,
            cand_pw,
            shares,
            cal,
        );
        let is_2d_ntt = matches!(sample_m, MulMethod::Ntt(s) if s.rows > 1);
        let par_pairs = if !matches!(sample_m, MulMethod::Ntt(s) if s.out_of_core) {
            pairs >= 2 && !is_2d_ntt
        } else {
            (pairs >= active_w && !is_2d_ntt)
                || (pairs >= 2 && !matches!(sample_m, MulMethod::Ntt(_)))
        };
        let pair_workers = if par_pairs {
            (active_w / pairs).max(1)
        } else {
            active_w
        };

        if par_pairs {
            let mut next_opts: Vec<Option<BigNum>> = (0..leaves.len().div_ceil(2)).map(|_| None).collect();
            if leaves.len() % 2 == 1 {
                let last_idx = leaves.len() - 1;
                next_opts[pairs] = Some(std::mem::replace(&mut leaves[last_idx], BigNum::zero()));
            }
            let leaves_ptr = leaves.as_mut_ptr() as usize;
            let next_ptr = next_opts.as_mut_ptr() as usize;
            let base_step = step_id;
            let tree_err: Mutex<Option<MulError>> = Mutex::new(None);
            let has_tree_err = AtomicBool::new(false);

            ctx.sched.par_for(pairs, 1, |p_idx| {
                if has_tree_err.load(Ordering::Relaxed) {
                    return;
                }
                let (mut a, mut b) = unsafe {
                    let lp = leaves_ptr as *mut BigNum;
                    (
                        std::mem::replace(&mut *lp.add(2 * p_idx), BigNum::zero()),
                        std::mem::replace(&mut *lp.add(2 * p_idx + 1), BigNum::zero()),
                    )
                };
                let m = plan.select_mul_method(
                    a.body_bits(),
                    b.body_bits(),
                    false,
                    false,
                    pair_workers,
                    shares,
                    cal,
                );
                match ctx.mul_whole(base_step.wrapping_add(p_idx as u64), &m, &mut a, &mut b) {
                    Ok(prod) => unsafe {
                        *(next_ptr as *mut Option<BigNum>).add(p_idx) = Some(prod);
                    },
                    Err(e) => {
                        has_tree_err.store(true, Ordering::Relaxed);
                        let mut g = tree_err.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                    }
                }
            });
            if let Some(e) = tree_err.into_inner().unwrap() {
                return Err(e);
            }
            step_id = step_id.wrapping_add(pairs as u64);
            leaves = next_opts.into_iter().map(|o| o.unwrap()).collect();
        } else {
            let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
            let mut idx = 0usize;
            while idx + 1 < leaves.len() {
                let mut a = std::mem::replace(&mut leaves[idx], BigNum::zero());
                let mut b = std::mem::replace(&mut leaves[idx + 1], BigNum::zero());
                let m = plan.select_mul_method(
                    a.body_bits(),
                    b.body_bits(),
                    false,
                    false,
                    pair_workers,
                    shares,
                    cal,
                );
                let p = ctx.mul_whole(step_id, &m, &mut a, &mut b)?;
                step_id = step_id.wrapping_add(1);
                next.push(p);
                idx += 2;
            }
            if idx < leaves.len() {
                next.push(std::mem::replace(&mut leaves[idx], BigNum::zero()));
            }
            leaves = next;
        }
    }

    Ok(leaves.pop().unwrap())
}

fn pow_u64_bignum(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    base_u64: u64,
    mut exp: u64,
) -> Result<BigNum, MulError> {
    let mut base = BigNum::from_u64(base_u64);
    let mut acc = BigNum::from_u64(1);
    let mut pid = 0xFA80_0000u64 ^ base_u64;
    while exp > 0 {
        if (exp & 1) == 1 {
            let m = plan.select_mul_method(
                acc.body_bits(),
                base.body_bits(),
                false,
                false,
                1,
                shares,
                cal,
            );
            acc = ctx.mul_whole(pid, &m, &mut acc, &mut base)?;
            pid = pid.wrapping_add(1);
        }
        exp >>= 1;
        if exp > 0 {
            let m = plan.select_mul_method(
                base.body_bits(),
                base.body_bits(),
                false,
                false,
                1,
                shares,
                cal,
            );
            base = ctx.sqr_whole(pid, &m, &mut base)?;
            pid = pid.wrapping_add(1);
        }
    }
    Ok(acc)
}

/// Compute the 2-adic modular inverse `V ≡ factor^{-1} (mod 2^{64 * target_limbs})` of an odd
/// `factor` by Newton doubling `V' ≡ V * (2 - factor * V) (mod 2^{64 * cur_limbs})` (Section 4).
pub fn build_hensel_inverse(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    factor: &BigNum,
    target_limbs: usize,
) -> Result<BigNum, MulError> {
    let n_tgt = target_limbs.max(1);
    debug_assert!(!factor.limbs.is_empty() && (factor.limbs[0] & 1) == 1);
    let v0 = inv_mod_u64(factor.limbs[0]);
    let mut v = BigNum::from_u64(v0);
    let mut prec = 1usize;
    let mut pid = 0xFB00_0000u64 ^ (n_tgt as u64);

    while prec < n_tgt {
        let next_prec = (prec * 2).min(n_tgt);
        let rem_prec = next_prec - prec;
        let f_trunc_len = factor.limbs.len().min(next_prec);
        let max_w = shares.active_workers.max(1);

        let m1 = plan.select_mul_method(
            (f_trunc_len as u64) * 64,
            v.body_bits(),
            false,
            false,
            shares.active_workers,
            shares,
            cal,
        );

        let mut fv_wrapped_opt: Option<LimbVec> = None;
        if ctx.fault_hook.is_none()
            && prec >= 2
            && next_prec >= 512
            && matches!(m1, MulMethod::Ntt(_))
        {
            let min_l = ((next_prec as u64) * 64) + 128;
            let total_bits = ((f_trunc_len + v.limbs.len()) as u64) * 64;
            if min_l + 192 < total_bits {
                let mut spec = choose_wrapped_ntt_spec(
                    min_l,
                    (f_trunc_len.max(v.limbs.len()) as u64) * 64,
                    shares.active_workers,
                    shares,
                    cal,
                    plan.ram_transform_budget,
                    plan.overrides.max_ntt_primes,
                );
                let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
                if l_bits >= min_l && l_bits < total_bits {
                    spec.wrap_cyclic = true;
                    if let Ok((w_limbs, _, _)) =
                        ctx.mul_wrapped(pid, spec, &factor.limbs[..f_trunc_len], &v.limbs)
                    {
                        fv_wrapped_opt = Some(w_limbs);
                    }
                }
            }
        }
        let fv_limbs = match fv_wrapped_opt {
            Some(w) => {
                pid = pid.wrapping_add(1);
                w
            }
            None => {
                let mut f_vec = take_limb_buf_uninit(f_trunc_len);
                f_vec.extend_from_slice(&factor.limbs[..f_trunc_len]);
                let mut f_low = BigNum::from_limbs_par(f_vec, false, 0, ctx.kernels, ctx.sched, max_w);
                let mut v_vec = take_limb_buf_uninit(v.limbs.len());
                v_vec.extend_from_slice(&v.limbs);
                let mut v_cur = BigNum::from_limbs_with_w0(v_vec, false, 0, v.w0());
                let fv_bn = ctx.mul_whole(pid, &m1, &mut f_low, &mut v_cur)?;
                put_limb_buf(f_low.into_limbs());
                put_limb_buf(v_cur.into_limbs());
                pid = pid.wrapping_add(1);
                fv_bn.into_limbs()
            }
        };

        // Since `factor * v ≡ 1 (mod 2^{64 * prec})`, limbs `0..prec` of `fv` are `[1, 0, ..., 0]`.
        // Extract `k = ((fv - 1) >> (64 * prec)) mod 2^{64 * rem_prec}` (`<= rem_prec` limbs)!
        if fv_limbs.len() <= prec {
            put_limb_buf(fv_limbs);
            prec = next_prec;
            continue;
        }
        let k_end = fv_limbs.len().min(next_prec);
        let mut k_limbs = take_limb_buf_uninit(k_end - prec);
        k_limbs.extend_from_slice(&fv_limbs[prec..k_end]);
        put_limb_buf(fv_limbs);
        trim_limbs(&mut k_limbs);
        if k_limbs.is_empty() {
            put_limb_buf(k_limbs);
            prec = next_prec;
            continue;
        }
        let mut k_bn = BigNum::from_limbs_par(k_limbs, false, 0, ctx.kernels, ctx.sched, max_w);

        // Half-length product `vk = (v_low * k) mod 2^{64 * rem_prec}`:
        let v_low_len = v.limbs.len().min(rem_prec);
        let mut v_low_vec = take_limb_buf_uninit(v_low_len);
        v_low_vec.extend_from_slice(&v.limbs[..v_low_len]);
        let mut v_low = BigNum::from_limbs_par(v_low_vec, false, 0, ctx.kernels, ctx.sched, max_w);
        let m2 = plan.select_mul_method(
            v_low.body_bits(),
            k_bn.body_bits(),
            false,
            false,
            shares.active_workers,
            shares,
            cal,
        );
        let vk = ctx.mul_whole(pid, &m2, &mut v_low, &mut k_bn)?;
        put_limb_buf(v_low.into_limbs());
        put_limb_buf(k_bn.into_limbs());
        pid = pid.wrapping_add(1);

        let vk_low_len = vk.limbs.len().min(rem_prec);
        if vk.limbs[..vk_low_len].iter().all(|&x| x == 0) {
            put_limb_buf(vk.into_limbs());
            prec = next_prec;
            continue;
        }

        // `v_next = (v - vk * 2^{64 * prec}) mod 2^{64 * next_prec}`:
        // Low `prec` limbs stay `v[0..prec]`, high `rem_prec` limbs are `2^{64 * rem_prec} - vk_low`!
        let mut v_limbs = v.into_limbs();
        v_limbs.resize(next_prec, 0);
        let mut borrow = 0u64;
        for i in 0..rem_prec {
            let x = vk.limbs.get(i).copied().unwrap_or(0);
            let (d1, b1) = 0u64.overflowing_sub(x);
            let (d2, b2) = d1.overflowing_sub(borrow);
            v_limbs[prec + i] = d2;
            borrow = u64::from(b1 | b2);
        }
        put_limb_buf(vk.into_limbs());
        trim_limbs(&mut v_limbs);
        let lw0 = compute_limbs_w0_par(
            ctx.kernels,
            ctx.sched,
            &v_limbs,
            max_w,
        );
        v = BigNum::from_limbs_with_w0(v_limbs, false, 0, lw0);
        prec = next_prec;
    }

    Ok(v)
}

pub struct PreparedLevelDivisor {
    pub div_plan: DivLevelPlan,
    pub factor: BigNum,
    pub factor_limbs_w0: u64,
    pub form: DivForm,
    pub max_quot_limbs: usize,
    pub inv_limbs: LimbVec,
    pub inv_limbs_w0: u64,
    pub inv_method: Option<MulMethod>,
    pub inv_cached: Option<CachedTransform>,
    pub q_max_quot_limbs: usize,
    pub q_inv_limbs: LimbVec,
    pub q_inv_limbs_w0: u64,
    pub q_inv_method: Option<MulMethod>,
    pub q_inv_cached: Option<CachedTransform>,
    pub p_max_quot_limbs: usize,
    pub p_inv_limbs: LimbVec,
    pub p_inv_limbs_w0: u64,
    pub p_inv_method: Option<MulMethod>,
    pub p_inv_cached: Option<CachedTransform>,
}

impl PreparedLevelDivisor {
    pub fn release_store(&mut self, store: Option<&crate::store::Store>) {
        if let Some(c) = &mut self.inv_cached {
            c.release_store(store);
        }
        self.inv_cached = None;
        if let Some(c) = &mut self.q_inv_cached {
            c.release_store(store);
        }
        self.q_inv_cached = None;
        if let Some(c) = &mut self.p_inv_cached {
            c.release_store(store);
        }
        self.p_inv_cached = None;
        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut self.inv_limbs));
        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut self.q_inv_limbs));
        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut self.p_inv_limbs));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_level_divisor(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    primes: &[u64],
    div_plan: &DivLevelPlan,
    sample_max_num_limbs: usize,
    sample_max_q_limbs: usize,
    sample_max_p_limbs: usize,
    div_par_workers: usize,
) -> Result<PreparedLevelDivisor, MulError> {
    prepare_level_divisor_ext(
        ctx,
        plan,
        shares,
        cal,
        primes,
        div_plan,
        sample_max_num_limbs,
        sample_max_q_limbs,
        sample_max_p_limbs,
        div_par_workers,
        div_par_workers,
        div_par_workers,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_level_divisor_ext(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    primes: &[u64],
    div_plan: &DivLevelPlan,
    sample_max_num_limbs: usize,
    sample_max_q_limbs: usize,
    sample_max_p_limbs: usize,
    t_par_workers: usize,
    q_par_workers: usize,
    p_par_workers: usize,
) -> Result<PreparedLevelDivisor, MulError> {
    let l_cur = div_plan.block_terms;
    let l_prev = if div_plan.prev_div_level == 0 {
        0
    } else {
        plan.leaf_len.saturating_mul(1u64 << div_plan.prev_div_level)
    };
    let span_m = div_plan.level - div_plan.prev_div_level;
    let factor = build_level_factor(ctx, plan, shares, cal, primes, l_cur, l_prev, span_m)?;
    let factor_limbs_w0 = compute_limbs_w0_par(
        ctx.kernels,
        ctx.sched,
        &factor.limbs,
        shares.active_workers.max(1),
    );
    if factor.limbs.len() == 1 && factor.limbs[0] == 1 {
        return Ok(PreparedLevelDivisor {
            div_plan: *div_plan,
            factor,
            factor_limbs_w0: 1,
            form: DivForm::SequentialHensel,
            max_quot_limbs: 1,
            inv_limbs: slice_to_limb_vec(&[1]),
            inv_limbs_w0: 1,
            inv_method: None,
            inv_cached: None,
            q_max_quot_limbs: 0,
            q_inv_limbs: empty_limb_vec(),
            q_inv_limbs_w0: 0,
            q_inv_method: None,
            q_inv_cached: None,
            p_max_quot_limbs: 0,
            p_inv_limbs: empty_limb_vec(),
            p_inv_limbs_w0: 0,
            p_inv_method: None,
            p_inv_cached: None,
        });
    }

    let max_quot_limbs = sample_max_num_limbs
        .saturating_sub(factor.limbs.len())
        .saturating_add(2)
        .max(2);

    if div_plan.form == DivForm::SequentialHensel {
        return Ok(PreparedLevelDivisor {
            div_plan: *div_plan,
            factor,
            factor_limbs_w0,
            form: DivForm::SequentialHensel,
            max_quot_limbs,
            inv_limbs: empty_limb_vec(),
            inv_limbs_w0: 0,
            inv_method: None,
            inv_cached: None,
            q_max_quot_limbs: 0,
            q_inv_limbs: empty_limb_vec(),
            q_inv_limbs_w0: 0,
            q_inv_method: None,
            q_inv_cached: None,
            p_max_quot_limbs: 0,
            p_inv_limbs: empty_limb_vec(),
            p_inv_limbs_w0: 0,
            p_inv_method: None,
            p_inv_cached: None,
        });
    }

    // SingleProduct form: build `factor^{-1} mod 2^{64 * max_quot_limbs}` and cache its transform
    // at `2 * max_quot_limbs` limbs (Section 4)!
    let inv_bn = build_hensel_inverse(ctx, plan, shares, cal, &factor, max_quot_limbs)?;
    let inv_limbs_w0 = inv_bn.limbs_w0();
    let div_nodes = 1u64 << plan.tree_height.saturating_sub(div_plan.level);
    let unpack_if_l3_fits = |m: &mut MulMethod| {
        if let MulMethod::Ntt(s) = m
            && !s.out_of_core
            && s.rows > 1
            && s.extra_sweeps == 0
            && shares.total_l3 > 0
        {
            let up_rb = if s.cols >= 512 { s.cols * 8 + 512 } else { s.cols * 8 };
            let up_grid = (s.k_primes as u64)
                .saturating_mul(s.rows as u64)
                .saturating_mul(up_rb as u64);
            let conc = (shares.active_workers / s.par_workers.max(1)).max(1) as u64;
            let shared_per_node = up_grid.saturating_add(up_grid / conc);
            let (up_num, up_den) = if conc <= 2 { (9, 10) } else { (7, 8) };
            if shares.fits_l3_domains(conc, shared_per_node, s.rows, up_num, up_den) {
                s.packed = false;
            }
        }
    };
    let prod_bits = (max_quot_limbs as u64) * 64;
    let mut method = plan.select_mul_method_amortized(
        prod_bits,
        prod_bits,
        false,
        false,
        t_par_workers.max(1),
        shares,
        cal,
        div_nodes,
    );
    unpack_if_l3_fits(&mut method);
    if !method.is_transform_checked()
        || matches!(method, MulMethod::SplitHalving { .. } | MulMethod::SplitOuter { .. })
    {
        if !method.is_transform_checked() {
            return Ok(PreparedLevelDivisor {
                div_plan: *div_plan,
                factor,
                factor_limbs_w0,
                form: DivForm::SingleProduct,
                max_quot_limbs,
                inv_limbs: inv_bn.into_limbs(),
                inv_limbs_w0,
                inv_method: Some(method),
                inv_cached: None,
                q_max_quot_limbs: 0,
                q_inv_limbs: empty_limb_vec(),
                q_inv_limbs_w0: 0,
                q_inv_method: None,
                q_inv_cached: None,
                p_max_quot_limbs: 0,
                p_inv_limbs: empty_limb_vec(),
                p_inv_limbs_w0: 0,
                p_inv_method: None,
                p_inv_cached: None,
            });
        }
        method = MulMethod::Ntt(ctx.fallback_ntt_spec(prod_bits * 2 + 64, false));
        unpack_if_l3_fits(&mut method);
    }

    // Also pre-cache `Q`'s and `P`'s shorter inverse transforms once per level whenever their
    // smaller bit-lengths step down to a smaller transform than `T`!
    let q_max_quot_limbs = if sample_max_q_limbs > factor.limbs.len() {
        sample_max_q_limbs
            .saturating_sub(factor.limbs.len())
            .saturating_add(2)
            .max(2)
    } else {
        0
    };
    let (q_inv_limbs, q_inv_limbs_w0, q_inv_method, q_inv_cached) =
        if q_max_quot_limbs >= 16 && q_max_quot_limbs < max_quot_limbs {
            let q_bits = (q_max_quot_limbs as u64) * 64;
            let mut qm = plan.select_mul_method_amortized(
                q_bits,
                q_bits,
                false,
                false,
                q_par_workers.max(1),
                shares,
                cal,
                div_nodes,
            );
            unpack_if_l3_fits(&mut qm);
            let mut q_inv = slice_to_limb_vec(&inv_bn.limbs[..inv_bn.limbs.len().min(q_max_quot_limbs)]);
            trim_limbs(&mut q_inv);
            let q_w0 = compute_limbs_w0_par(
                ctx.kernels,
                ctx.sched,
                &q_inv,
                shares.active_workers.max(1),
            );
            if qm.is_transform_checked()
                && !matches!(qm, MulMethod::SplitHalving { .. } | MulMethod::SplitOuter { .. })
                && qm != method
            {
                let mut build_qm = qm.clone();
                if let MulMethod::Ntt(ref mut s) = build_qm {
                    s.par_workers = shares.active_workers.max(1);
                }
                let mut qc = ctx.forward_cached_reusable(&build_qm, &q_inv)?;
                if let (CachedTransform::Ntt(nc), MulMethod::Ntt(orig_s)) = (&mut qc, &qm) {
                    nc.spec.par_workers = orig_s.par_workers;
                }
                (q_inv, q_w0, Some(qm), Some(qc))
            } else if !qm.is_transform_checked() {
                (q_inv, q_w0, Some(qm), None)
            } else {
                (empty_limb_vec(), 0, None, None)
            }
        } else {
            (empty_limb_vec(), 0, None, None)
        };

    let p_max_quot_limbs = if sample_max_p_limbs > factor.limbs.len() {
        sample_max_p_limbs
            .saturating_sub(factor.limbs.len())
            .saturating_add(2)
            .max(2)
    } else {
        0
    };
    let (p_inv_limbs, p_inv_limbs_w0, p_inv_method, p_inv_cached) =
        if p_max_quot_limbs >= 16 && p_max_quot_limbs < max_quot_limbs {
            let p_bits = (p_max_quot_limbs as u64) * 64;
            let mut pm = plan.select_mul_method_amortized(
                p_bits,
                p_bits,
                false,
                false,
                p_par_workers.max(1),
                shares,
                cal,
                div_nodes,
            );
            unpack_if_l3_fits(&mut pm);
            let mut p_inv = slice_to_limb_vec(&inv_bn.limbs[..inv_bn.limbs.len().min(p_max_quot_limbs)]);
            trim_limbs(&mut p_inv);
            let p_w0 = compute_limbs_w0_par(
                ctx.kernels,
                ctx.sched,
                &p_inv,
                shares.active_workers.max(1),
            );
            if pm.is_transform_checked()
                && !matches!(pm, MulMethod::SplitHalving { .. } | MulMethod::SplitOuter { .. })
                && pm != method
                && Some(&pm) != q_inv_method.as_ref()
            {
                let mut build_pm = pm.clone();
                if let MulMethod::Ntt(ref mut s) = build_pm {
                    s.par_workers = shares.active_workers.max(1);
                }
                let mut pc = ctx.forward_cached_reusable(&build_pm, &p_inv)?;
                if let (CachedTransform::Ntt(nc), MulMethod::Ntt(orig_s)) = (&mut pc, &pm) {
                    nc.spec.par_workers = orig_s.par_workers;
                }
                (p_inv, p_w0, Some(pm), Some(pc))
            } else if !pm.is_transform_checked() && Some(&pm) != q_inv_method.as_ref() {
                (p_inv, p_w0, Some(pm), None)
            } else {
                (empty_limb_vec(), 0, None, None)
            }
        } else {
            (empty_limb_vec(), 0, None, None)
        };

    let mut build_m = method.clone();
    if let MulMethod::Ntt(ref mut s) = build_m {
        s.par_workers = shares.active_workers.max(1);
    }
    let mut cached = ctx.forward_cached_reusable(&build_m, &inv_bn.limbs)?;
    if let (CachedTransform::Ntt(nc), MulMethod::Ntt(orig_s)) = (&mut cached, &method) {
        nc.spec.par_workers = orig_s.par_workers;
    }
    Ok(PreparedLevelDivisor {
        div_plan: *div_plan,
        factor,
        factor_limbs_w0,
        form: DivForm::SingleProduct,
        max_quot_limbs,
        inv_limbs: inv_bn.into_limbs(),
        inv_limbs_w0,
        inv_method: Some(method),
        inv_cached: Some(cached),
        q_max_quot_limbs,
        q_inv_limbs,
        q_inv_limbs_w0,
        q_inv_method,
        q_inv_cached,
        p_max_quot_limbs,
        p_inv_limbs,
        p_inv_limbs_w0,
        p_inv_method,
        p_inv_cached,
    })
}

/// Divide `num` exactly by `divisor.factor`, keeping `num.neg` and `num.shift` intact (since
/// `divisor.factor` is odd) and verifying `w0(factor) * w0(quot) == w0(num)` (Section 11).
fn divide_bignum_exact(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    par_workers: usize,
    point_id: u64,
    num: &mut BigNum,
    divisor: &PreparedLevelDivisor,
) -> Result<BigNum, MulError> {
    if num.is_zero() || (divisor.factor.limbs.len() == 1 && divisor.factor.limbs[0] == 1) {
        return Ok(num.clone());
    }

    let num_limbs_w0 = num.limbs_w0();
    let q_len = num
        .limbs
        .len()
        .saturating_sub(divisor.factor.limbs.len())
        .saturating_add(1)
        .max(1);

    let (mut q_limbs, cached_q_w0) = match divisor.form {
        DivForm::SequentialHensel => (
            div_exact_hensel_seq(ctx.kernels, &num.limbs, &divisor.factor.limbs),
            None,
        ),
        DivForm::SingleProduct => {
            let low_len = num.limbs.len().min(q_len);
            let num_low = &num.limbs[..low_len];
            let inv_len = divisor.inv_limbs.len().min(q_len);
            let inv_low = &divisor.inv_limbs[..inv_len];
            let mul_low_uncached =
                |m: &MulMethod, rhs_low: &[u64]| -> Result<LimbVec, MulError> {
                    if ctx.fault_hook.is_none() {
                        match m {
                            MulMethod::Basecase => {
                                let cap = num_low.len() + rhs_low.len() + 1;
                                let mut prod = crate::limb::take_limb_buf_zeroed(cap);
                                let plen = mul_limbs_karatsuba_into(
                                    ctx.kernels,
                                    &mut prod,
                                    num_low,
                                    rhs_low,
                                    num_low.len().max(rhs_low.len()).max(4),
                                );
                                prod.truncate(plen.min(q_len));
                                trim_limbs(&mut prod);
                                return Ok(prod);
                            }
                            MulMethod::Karatsuba { basecase_cutoff } => {
                                let cap = num_low.len() + rhs_low.len() + 1;
                                let mut prod = crate::limb::take_limb_buf_zeroed(cap);
                                let plen = mul_limbs_karatsuba_into(
                                    ctx.kernels,
                                    &mut prod,
                                    num_low,
                                    rhs_low,
                                    *basecase_cutoff,
                                );
                                prod.truncate(plen.min(q_len));
                                trim_limbs(&mut prod);
                                return Ok(prod);
                            }
                            _ => {}
                        }
                    }
                    let mut a = BigNum::from_limbs(num_low.to_vec(), false, 0, ctx.kernels);
                    let mut b = BigNum::from_limbs(rhs_low.to_vec(), false, 0, ctx.kernels);
                    let mut prod = ctx.mul_whole(point_id, m, &mut a, &mut b)?.into_limbs();
                    if prod.len() > q_len {
                        prod.truncate(q_len);
                    }
                    trim_limbs(&mut prod);
                    Ok(prod)
                };
            let mul_cached_quot = |m: &MulMethod,
                                   inv_slice: &[u64],
                                   inv_w0: u64,
                                   cached: &CachedTransform|
             -> Result<(LimbVec, Option<u64>), MulError> {
                // Section 11: only exact methods (`NTT`, limb) may make part of a product; an
                // `Fft` cached-inverse product is made whole, checked against `w0`, and sliced.
                if matches!(m, MulMethod::Fft { .. }) {
                    let full_out = num_low.len() + inv_slice.len() + 2;
                    let a_w0 = compute_limbs_w0(ctx.kernels, num_low);
                    let exp_w0 = mod61_mul(a_w0, inv_w0);
                    let (mut prod, _) = ctx.mul_limbs_with_cached(
                        point_id,
                        m,
                        num_low,
                        inv_slice,
                        cached,
                        full_out,
                        Some(exp_w0),
                    )?;
                    if prod.len() > q_len {
                        prod.truncate(q_len);
                    }
                    trim_limbs(&mut prod);
                    Ok((prod, None))
                } else {
                    let (mut prod, w0) = ctx.mul_limbs_with_cached(
                        point_id,
                        m,
                        num_low,
                        inv_slice,
                        cached,
                        q_len,
                        None,
                    )?;
                    trim_limbs(&mut prod);
                    Ok((prod, Some(w0)))
                }
            };
            if divisor.p_max_quot_limbs > 0
                && q_len <= divisor.p_max_quot_limbs
                && divisor.p_inv_method.is_some()
            {
                let pm = divisor.p_inv_method.as_ref().unwrap();
                if let Some(pc) = &divisor.p_inv_cached {
                    mul_cached_quot(pm, &divisor.p_inv_limbs, divisor.p_inv_limbs_w0, pc)?
                } else {
                    let p_inv_low = &divisor.p_inv_limbs[..divisor.p_inv_limbs.len().min(q_len)];
                    (mul_low_uncached(pm, p_inv_low)?, None)
                }
            } else if divisor.q_max_quot_limbs > 0
                && q_len <= divisor.q_max_quot_limbs
                && divisor.q_inv_method.is_some()
            {
                let qm = divisor.q_inv_method.as_ref().unwrap();
                if let Some(qc) = &divisor.q_inv_cached {
                    mul_cached_quot(qm, &divisor.q_inv_limbs, divisor.q_inv_limbs_w0, qc)?
                } else {
                    let q_inv_low = &divisor.q_inv_limbs[..divisor.q_inv_limbs.len().min(q_len)];
                    (mul_low_uncached(qm, q_inv_low)?, None)
                }
            } else if q_len * 2 <= divisor.max_quot_limbs && q_len >= 16 && !divisor.inv_limbs.is_empty() {
                let sub_bits = (q_len as u64) * 64;
                let sub_m = plan.select_mul_method(
                    sub_bits,
                    sub_bits,
                    false,
                    false,
                    par_workers.max(1),
                    shares,
                    cal,
                );
                (mul_low_uncached(&sub_m, inv_low)?, None)
            } else if q_len <= divisor.max_quot_limbs {
                if let (Some(m), Some(cached)) = (&divisor.inv_method, &divisor.inv_cached) {
                    mul_cached_quot(m, &divisor.inv_limbs, divisor.inv_limbs_w0, cached)?
                } else if let Some(m) = &divisor.inv_method {
                    (mul_low_uncached(m, inv_low)?, None)
                } else {
                    (
                        div_exact_hensel_seq(ctx.kernels, &num.limbs, &divisor.factor.limbs),
                        None,
                    )
                }
            } else {
                (
                    div_exact_hensel_seq(ctx.kernels, &num.limbs, &divisor.factor.limbs),
                    None,
                )
            }
        }
    };

    let q_lw0 = cached_q_w0
        .unwrap_or_else(|| compute_limbs_w0_par(ctx.kernels, ctx.sched, &q_limbs, par_workers.max(1)));
    if mod61_mul(divisor.factor_limbs_w0, q_lw0) != num_limbs_w0 {
        // Section 11: re-verify `num` at rest before climbing any rung!
        if !num.verify_at_rest(ctx.kernels, ctx.telemetry) {
            ctx.telemetry
                .record_failure(crate::par::current_core_id(), 0, None);
            return Err(MulError::NeedRollback(point_id));
        }
        let fresh_num_lw0 = num.limbs_w0();
        let recomputed_qw0 = compute_limbs_w0(ctx.kernels, &q_limbs);
        if mod61_mul(divisor.factor_limbs_w0, recomputed_qw0) == fresh_num_lw0 {
            ctx.telemetry
                .record_correction(crate::par::current_core_id(), 0, None);
            let neg = num.neg;
            let shift = num.shift;
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut num.limbs));
            return Ok(BigNum::from_limbs_with_w0(q_limbs, neg, shift, recomputed_qw0));
        }

        // Rung 1: attempt single-limb in-place quotient correction via `(w0, w1, w2)` when
        // `divisor.factor` still agrees with its own check word.
        let mut factor_ok = divisor.factor.chk.c0 == divisor.factor.chk.c1
            && compute_limbs_w0(ctx.kernels, &divisor.factor.limbs) == divisor.factor_limbs_w0;
        let rung1 = if factor_ok {
            ctx.ladder.claim_rung(point_id, Rung::Correct)
        } else {
            Rung::MakeAgain
        };
        if factor_ok && rung1 == Rung::Correct {
            let num3 = compute_limbs_all3(ctx.kernels, &num.limbs);
            let fac3 = compute_limbs_all3(ctx.kernels, &divisor.factor.limbs);
            if fac3[0] != 0 && fac3[1] != 0 && fac3[2] != 0 {
                let p1 = PRIME_LOC1 as f64;
                let p2 = PRIME_LOC2 as f64;
                let inv0 = mod_pow_u64(fac3[0], MOD_W0 - 2, MOD_W0);
                let inv1 = mod_pow_u64(fac3[1], PRIME_LOC1 - 2, PRIME_LOC1) as f64;
                let inv2 = mod_pow_u64(fac3[2], PRIME_LOC2 - 2, PRIME_LOC2) as f64;
                let exp3 = [
                    mod61_mul(num3[0], inv0),
                    fma_mul_mod(num3[1] as f64, inv1, p1, 1.0 / p1) as u64,
                    fma_mul_mod(num3[2] as f64, inv2, p2, 1.0 / p2) as u64,
                ];
                if q_limbs.len() < q_len {
                    q_limbs.resize(q_len, 0);
                }
                let act3 = compute_limbs_all3(ctx.kernels, &q_limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut q_limbs, act3, exp3)
                {
                    let page_addr = (&q_limbs[limb_index] as *const u64 as usize) & !4095usize;
                    ctx.telemetry
                        .record_correction(crate::par::current_core_id(), page_addr, None);
                    trim_limbs(&mut q_limbs);
                    let fixed_w0 = compute_limbs_w0(ctx.kernels, &q_limbs);
                    let neg = num.neg;
                    let shift = num.shift;
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut num.limbs));
                    return Ok(BigNum::from_limbs_with_w0(q_limbs, neg, shift, fixed_w0));
                }
            }
        }
        let _ = &mut factor_ok;

        // Rung 2 (Section 11: "where it fails, the factor and its inverse are made again first",
        // once with active kernels and then with reference kernels).
        for retry_step in 0..2u32 {
            let rung = ctx.ladder.claim_rung(point_id, Rung::MakeAgain);
            match rung {
                Rung::Correct | Rung::MakeAgain => {
                    ctx.telemetry
                        .record_retry(crate::par::current_core_id(), 0, None);
                    let ref_k_storage = (retry_step > 0).then(Kernels::reference_only);
                    let use_k = ref_k_storage.as_ref().unwrap_or(ctx.kernels);
                    let retry_ctx = MulCtx {
                        kernels: use_k,
                        fft_cache: ctx.fft_cache,
                        ntt_cache: ctx.ntt_cache,
                        sched: ctx.sched,
                        store: ctx.store,
                        ladder: ctx.ladder,
                        telemetry: ctx.telemetry,
                        fft_err_bound: ctx.fft_err_bound,
                        fault_hook: None,
                    };
                    let dummy_sieve = AtomicU64::new(0);
                    let dp = divisor.div_plan;
                    let fresh_primes = segmented_primes_up_to(dp.block_terms, &dummy_sieve);
                    let l_prev = if dp.prev_div_level == 0 {
                        0
                    } else {
                        plan.leaf_len.saturating_mul(1u64 << dp.prev_div_level)
                    };
                    let span_m = dp.level - dp.prev_div_level;
                    let fresh_factor = build_level_factor(
                        &retry_ctx,
                        plan,
                        shares,
                        cal,
                        &fresh_primes,
                        dp.block_terms,
                        l_prev,
                        span_m,
                    )?;
                    let fresh_fac_w0 = compute_limbs_w0(use_k, &fresh_factor.limbs);
                    let mut retry_q = if dp.form == DivForm::SingleProduct && q_len >= 4 {
                        let fresh_inv = build_hensel_inverse(
                            &retry_ctx,
                            plan,
                            shares,
                            cal,
                            &fresh_factor,
                            q_len.max(2),
                        )?;
                        let low_n = num.limbs.len().min(q_len);
                        let inv_n = fresh_inv.limbs.len().min(q_len);
                        let mut ba =
                            BigNum::from_limbs(num.limbs[..low_n].to_vec(), false, 0, use_k);
                        let mut bb =
                            BigNum::from_limbs(fresh_inv.limbs[..inv_n].to_vec(), false, 0, use_k);
                        let bits = (q_len as u64) * 64;
                        let m = MulMethod::Ntt(retry_ctx.fallback_ntt_spec(bits * 2 + 64, false));
                        let mut p = retry_ctx.mul_whole(point_id, &m, &mut ba, &mut bb)?.into_limbs();
                        if p.len() > q_len {
                            p.truncate(q_len);
                        }
                        trim_limbs(&mut p);
                        p
                    } else {
                        div_exact_hensel_seq(use_k, &num.limbs, &fresh_factor.limbs)
                    };
                    trim_limbs(&mut retry_q);
                    let retry_w0 = compute_limbs_w0(use_k, &retry_q);
                    if mod61_mul(fresh_fac_w0, retry_w0) == fresh_num_lw0 {
                        let neg = num.neg;
                        let shift = num.shift;
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut num.limbs));
                        return Ok(BigNum::from_limbs_with_w0(retry_q, neg, shift, retry_w0));
                    }
                }
                Rung::GoBack => {
                    ctx.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::NeedRollback(point_id));
                }
                Rung::Stop => {
                    ctx.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::FatalStop(
                        "Quotient check failed across all ladder rungs".to_string(),
                    ));
                }
            }
        }
        return Err(MulError::NeedRollback(point_id));
    }

    let _ = mod61_sub(0, 0);
    let neg = num.neg;
    let shift = num.shift;
    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut num.limbs));
    Ok(BigNum::from_limbs_with_w0(q_limbs, neg, shift, q_lw0))
}

/// Merge two adjacent `BlockTriple`s at `(level, node_idx)` using the plan's chosen merge form
/// (`MergeForm::Separate` vs `MergeForm::Fused`), skipping `P` on the rightmost node!
/// The transform method in which a pair of products `x * b` and `y * b` shares the forward
/// transform of `b` although the two would take different grids by themselves: the method of
/// the larger product, in which both are made with five transforms against three and three,
/// where that is the cheaper way; `None` where it is not, or the method is not a transform.
/// Such a pair was measured at 1.66 to 2.08 products of the larger (the price check, form 2),
/// not the five thirds its transforms count for.
fn shared_method_of_unequal_pair(
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    pw: usize,
    b_bits: u64,
    x_bits: u64,
    y_bits: u64,
) -> Option<MulMethod> {
    let big_bits = x_bits.max(y_bits);
    let small_bits = x_bits.min(y_bits);
    let m = plan.select_mul_method(big_bits, b_bits, false, false, pw, shares, cal);
    if !matches!(m, MulMethod::Ntt(_) | MulMethod::Fft { .. }) {
        return None;
    }
    let m_small = plan.select_mul_method(small_bits, b_bits, false, false, pw, shares, cal);
    let factor = crate::plan::pair_shared_cost_factor(&m, &m_small);
    let big = plan.product_worker_sec(big_bits, b_bits, false, pw, shares, cal);
    let small = plan.product_worker_sec(small_bits, b_bits, false, pw, shares, cal);
    (factor * big < big + small).then_some(m)
}

type DeferredProduct = (u64, BigNum, BigNum);

/// Before `P1 * T2` is made: ask that its buffer have the room for `T = Q2 * T1 + P1 * T2`.
/// The sum is then made where that product lies (`add_bignum_into_par`), in the limbs the node
/// has written last, and no third buffer of the size of `T` is brought in from memory for it.
fn ask_room_for_t(qt1: &BigNum, p1: &BigNum, t2: &BigNum) -> crate::limb::RoomAsked {
    let pt_limbs = p1.limbs.len() + t2.limbs.len();
    crate::limb::RoomAsked::new(pt_limbs, crate::limb::sum_room(qt1, pt_limbs + 2))
}

fn merge_pair(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    level: u32,
    node_idx: u64,
    is_rightmost: bool,
    par_workers: usize,
    left: BlockTriple,
    right: BlockTriple,
) -> Result<BlockTriple, MulError> {
    merge_pair_inner(
        ctx,
        plan,
        shares,
        cal,
        level,
        node_idx,
        is_rightmost,
        par_workers,
        None,
        false,
        None,
        left,
        right,
    )
    .map(|(tr, _)| tr)
}

#[inline]
fn same_transform_footprint(m1: &MulMethod, m2: &MulMethod) -> bool {
    match (m1, m2) {
        (MulMethod::Fft { n_complex: n1, .. }, MulMethod::Fft { n_complex: n2, .. }) => n1 == n2,
        (MulMethod::Ntt(s1), MulMethod::Ntt(s2)) => {
            s1.n == s2.n
                && s1.k_primes == s2.k_primes
                && s1.negacyclic == s2.negacyclic
                && s1.extra_sweeps == s2.extra_sweeps
                && s1.out_of_core == s2.out_of_core
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn merge_pair_inner(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    level: u32,
    node_idx: u64,
    is_rightmost: bool,
    par_workers: usize,
    rem_merges: Option<(&AtomicUsize, usize)>,
    defer_p: bool,
    ext_lp: Option<&BigNum>,
    mut left: BlockTriple,
    mut right: BlockTriple,
) -> Result<(BlockTriple, Option<DeferredProduct>), MulError> {
    let base_pid = 0x1000_0000u64 | ((level as u64) << 40) | (node_idx << 4);
    let pw = par_workers.max(1);
    let cur_pw = |base_pw: usize| -> usize {
        if let Some((rem_ctr, conc_cap)) = rem_merges {
            let live = rem_ctr.load(Ordering::Acquire).min(conc_cap).max(1);
            (shares.active_workers.max(1) / live).max(base_pw)
        } else {
            base_pw
        }
    };

    if pw == 1
        && ctx.fault_hook.is_none()
        && plan.overrides.force_merge_form.is_none()
        && plan.overrides.split_override == SplitFormOverride::Auto
        && left.t.shift == 0
        && right.t.shift == 0
        && left.p.shift == 0
        && right.p.shift == 0
        && !left.q.is_zero()
        && !right.q.is_zero()
        && !left.p.is_zero()
        && (is_rightmost || !right.p.is_zero())
    {
        let row_stream_cap = if cal.mul_base_rows_max < (cal.basecase_max_limbs as u64) {
            cal.mul_base_rows_max as usize
        } else {
            0
        };
        let quick_small_or_fft_method =
            |a: &BigNum, b: &BigNum, fft_opt: Option<(usize, u32)>| -> Option<MulMethod> {
                let min_l = a.limbs.len().min(b.limbs.len());
                let max_l = a.limbs.len().max(b.limbs.len());
                if max_l <= cal.basecase_max_limbs
                    || min_l <= row_stream_cap
                    || (min_l <= cal.basecase_max_limbs && max_l <= cal.basecase_max_limbs * 2)
                {
                    Some(MulMethod::Basecase)
                } else if max_l <= cal.karatsuba_max_limbs {
                    Some(MulMethod::Karatsuba {
                        basecase_cutoff: cal.basecase_max_limbs,
                    })
                } else {
                    fft_opt.map(|(n_complex, coeff_bits)| MulMethod::Fft {
                        n_complex,
                        coeff_bits,
                    })
                }
            };
        if left.t.limbs.len() <= cal.karatsuba_max_limbs
            || right.t.limbs.len() <= cal.karatsuba_max_limbs
        {
            let qt_fft = crate::fft::choose_fft_params(
                right.q.body_bits() + left.t.body_bits() + 64,
                false,
                shares.min_l2_share,
            );
            let qq_fft = crate::fft::choose_fft_params(
                left.q.body_bits() + right.q.body_bits() + 64,
                false,
                shares.min_l2_share,
            );
            let pt_fft = crate::fft::choose_fft_params(
                left.p.body_bits() + right.t.body_bits() + 64,
                false,
                shares.min_l2_share,
            );
            if let (Some(qt_m), Some(qq_m), Some(pt_m)) = (
                quick_small_or_fft_method(&right.q, &left.t, qt_fft),
                quick_small_or_fft_method(&left.q, &right.q, qq_fft),
                quick_small_or_fft_method(&left.p, &right.t, pt_fft),
            ) && !matches!(qt_m, MulMethod::Fft { .. })
                && !matches!(pt_m, MulMethod::Fft { .. })
            {
                let qt_bn = ctx.mul_whole(base_pid, &qt_m, &mut right.q, &mut left.t)?;
                let q_bn = ctx.mul_whole(base_pid + 1, &qq_m, &mut left.q, &mut right.q)?;
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
                let pt_bn = ctx.mul_whole(base_pid + 2, &pt_m, &mut left.p, &mut right.t)?;
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
                let p_bn = if is_rightmost {
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                    BigNum::zero()
                } else {
                    let pp_fft = crate::fft::choose_fft_params(
                        left.p.body_bits() + right.p.body_bits() + 64,
                        false,
                        shares.min_l2_share,
                    );
                    let pp_m = quick_small_or_fft_method(&left.p, &right.p, pp_fft)
                        .unwrap_or_else(|| {
                            plan.select_mul_method(
                                left.p.body_bits(),
                                right.p.body_bits(),
                                false,
                                false,
                                1,
                                shares,
                                cal,
                            )
                        });
                    ctx.mul_whole(base_pid + 3, &pp_m, &mut left.p, &mut right.p)?
                };
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                let t_bn = crate::limb::add_bignum(ctx.kernels, &qt_bn, &pt_bn);
                crate::limb::put_limb_buf(qt_bn.into_limbs());
                crate::limb::put_limb_buf(pt_bn.into_limbs());
                return Ok((
                    BlockTriple {
                        p: p_bn,
                        q: q_bn,
                        t: t_bn,
                    },
                    None,
                ));
            }
        }
        let fused_bits =
            (right.q.total_bits() + left.t.body_bits()).max(left.p.body_bits() + right.t.body_bits());
        let unshifted_bits = (right.q.body_bits() + left.t.body_bits())
            .max(left.p.body_bits() + right.t.body_bits());
        let nf_opt =
            crate::fft::choose_fft_params(fused_bits + 64, true, shares.min_l2_share);
        let ns_opt =
            crate::fft::choose_fft_params(unshifted_bits + 64, false, shares.min_l2_share);
        let p_out_bits = ((left.p.limbs.len() + right.p.limbs.len()) as u64) * 64;
        let np_opt = (!is_rightmost
            && left.p.limbs.len().min(right.p.limbs.len()) >= 24)
            .then(|| crate::fft::choose_fft_params(p_out_bits + 64, false, shares.min_l2_share))
            .flatten();
        let pt_max_b = right.t.body_bits().max(right.p.body_bits()) + left.p.body_bits();
        let n_pt_opt = (!is_rightmost)
            .then(|| crate::fft::choose_fft_params(pt_max_b + 64, false, shares.min_l2_share))
            .flatten();
        if let Some((nf, cbf)) = nf_opt
            && ns_opt.is_some_and(|(ns, _)| nf == ns)
            && (is_rightmost || n_pt_opt.is_none_or(|(n_pt, _)| n_pt >= nf))
        {
            let f_cap_bits = (2 * (nf as u64)) * (cbf as u64);
            let q_out_bits =
                ((left.q.limbs.len() + right.q.limbs.len()) as u64) * 64 + right.q.shift + 64;
            let q_odd_bits = left.q.body_bits() + right.q.body_bits();
            let nq_opt =
                crate::fft::choose_fft_params(q_odd_bits + 64, false, shares.min_l2_share);
            let fuse_q = q_out_bits <= f_cap_bits
                && left.q.limbs.len().max(right.q.limbs.len()) > cal.basecase_max_limbs
                && left.q.limbs.len().min(right.q.limbs.len())
                    >= (cal.basecase_max_limbs / 2).max(16)
                && nq_opt.is_some_and(|(nq, _)| nf <= nq);
            let fuse_p = !is_rightmost
                && p_out_bits <= f_cap_bits
                && left.p.limbs.len().max(right.p.limbs.len()) > cal.basecase_max_limbs
                && np_opt.is_some_and(|(np, _)| nf <= np);
            let fm = MulMethod::Fft {
                n_complex: nf,
                coeff_bits: cbf,
            };
            if let Some((t_bn, q_opt, p_opt)) = ctx.try_ntt_fused_merge_triple(
                &fm,
                1,
                plan.ram_transform_budget,
                &left.p,
                &left.q,
                &left.t,
                &right.p,
                &right.q,
                &right.t,
                fuse_q,
                fuse_p,
            ) {
                let q_bn = match q_opt {
                    Some(qb) => qb,
                    None => {
                        let qm = quick_small_or_fft_method(&left.q, &right.q, nq_opt)
                            .unwrap_or_else(|| {
                                plan.select_mul_method(
                                    left.q.body_bits(),
                                    right.q.body_bits(),
                                    false,
                                    false,
                                    1,
                                    shares,
                                    cal,
                                )
                            });
                        ctx.mul_whole(base_pid + 1, &qm, &mut left.q, &mut right.q)?
                    }
                };
                let p_bn = if is_rightmost {
                    BigNum::zero()
                } else {
                    match p_opt {
                        Some(pb) => pb,
                        None => {
                            let pm = quick_small_or_fft_method(&left.p, &right.p, np_opt)
                                .unwrap_or_else(|| {
                                    plan.select_mul_method(
                                        left.p.body_bits(),
                                        right.p.body_bits(),
                                        false,
                                        false,
                                        1,
                                        shares,
                                        cal,
                                    )
                                });
                            ctx.mul_whole(base_pid + 3, &pm, &mut left.p, &mut right.p)?
                        }
                    }
                };
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
                return Ok((
                    BlockTriple {
                        p: p_bn,
                        q: q_bn,
                        t: t_bn,
                    },
                    None,
                ));
            }
        } else if ns_opt.is_some() && !is_rightmost {
            let qt_max_b = left.t.body_bits().max(left.q.body_bits()) + right.q.body_bits();
            let pt_max_b = right.t.body_bits().max(right.p.body_bits()) + left.p.body_bits();
            if let (Some((n_qt, cb_qt)), Some((n_pt, cb_pt))) = (
                crate::fft::choose_fft_params(qt_max_b + 64, false, shares.min_l2_share),
                crate::fft::choose_fft_params(pt_max_b + 64, false, shares.min_l2_share),
            ) {
                let m_qt = MulMethod::Fft {
                    n_complex: n_qt,
                    coeff_bits: cb_qt,
                };
                let m_pt = MulMethod::Fft {
                    n_complex: n_pt,
                    coeff_bits: cb_pt,
                };
                if let Some((qt_bn, q_bn)) = ctx.try_mul_pair_shared_b(
                    &m_qt,
                    1,
                    plan.ram_transform_budget,
                    &left.t,
                    &left.q,
                    &right.q,
                ) {
                    let room_for_t = ask_room_for_t(&qt_bn, &left.p, &right.t);
                    if np_opt.is_some_and(|(np, _)| n_pt <= np) {
                        let pt_and_p = ctx.try_mul_pair_shared_b(
                            &m_pt,
                            1,
                            plan.ram_transform_budget,
                            &right.t,
                            &right.p,
                            &left.p,
                        );
                        drop(room_for_t);
                        if let Some((pt_bn, p_bn)) = pt_and_p {
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
                            let t_bn =
                                add_bignum_into_par(ctx.kernels, &qt_bn, pt_bn, ctx.sched, 1);
                            crate::limb::put_limb_buf(qt_bn.into_limbs());
                            return Ok((
                                BlockTriple {
                                    p: p_bn,
                                    q: q_bn,
                                    t: t_bn,
                                },
                                None,
                            ));
                        }
                    } else {
                        let pt_m = quick_small_or_fft_method(&left.p, &right.t, Some((n_pt, cb_pt)))
                            .unwrap_or(m_pt);
                        let pt_bn =
                            ctx.mul_whole(base_pid + 2, &pt_m, &mut left.p, &mut right.t)?;
                        let pm = quick_small_or_fft_method(&left.p, &right.p, np_opt)
                            .unwrap_or_else(|| {
                                plan.select_mul_method(
                                    left.p.body_bits(),
                                    right.p.body_bits(),
                                    false,
                                    false,
                                    1,
                                    shares,
                                    cal,
                                )
                            });
                        drop(room_for_t);
                        let p_bn =
                            ctx.mul_whole(base_pid + 3, &pm, &mut left.p, &mut right.p)?;
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
                        let t_bn = add_bignum_into_par(ctx.kernels, &qt_bn, pt_bn, ctx.sched, 1);
                        crate::limb::put_limb_buf(qt_bn.into_limbs());
                        return Ok((
                            BlockTriple {
                                p: p_bn,
                                q: q_bn,
                                t: t_bn,
                            },
                            None,
                        ));
                    }
                }
            }
        }
    }

    let p1_body_bits = ext_lp.map_or_else(|| left.p.body_bits(), BigNum::body_bits);
    // A merge whose two P were lent out for a stolen `P1 * P2` takes the separate form, the only
    // one that reads the lent `P1`: the lending was decided on that form, with the sizes of both
    // P. Here the right P is gone (it counts 0 bits), and a second choice made without it could
    // fall on the fused form, which would then multiply by the empty `P1` left in the node and
    // give a `T` without its `P1 * T2`, every check word in order.
    let form = if ext_lp.is_some() {
        MergeForm::Separate
    } else {
        plan.select_merge_form_node(
            left.q.body_bits(),
            right.q.body_bits(),
            right.q.shift,
            left.t.body_bits(),
            p1_body_bits,
            right.t.body_bits(),
            right.p.body_bits(),
            is_rightmost,
            pw,
            shares,
            cal,
        )
    };

    match form {
        MergeForm::Separate => {
            // Separate form (Section 4): `Q2` stays without its zero bits!
            if is_rightmost && !right.p.limbs.is_empty() {
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                right.p = BigNum::zero();
            }
            let can_cache_in_ram = |sm: &MulMethod| -> bool {
                match sm {
                    MulMethod::Ntt(s) if !s.out_of_core => {
                        let gb = (s.k_primes as u64)
                            .saturating_mul(s.rows as u64)
                            .saturating_mul(s.outer_row_bytes() as u64);
                        gb.saturating_mul(2) <= plan.ram_transform_budget
                    }
                    _ => true,
                }
            };
            let qt_method = plan.select_mul_method(
                right.q.body_bits(),
                left.t.body_bits(),
                false,
                false,
                pw,
                shares,
                cal,
            );
            let qq_method = plan.select_mul_method(
                left.q.body_bits(),
                right.q.body_bits(),
                false,
                false,
                pw,
                shares,
                cal,
            );

            let shared_q_method = if same_transform_footprint(&qt_method, &qq_method) {
                let m = if qt_method == qq_method {
                    qt_method.clone()
                } else {
                    plan.select_mul_method(
                        left.t.body_bits().max(left.q.body_bits()),
                        right.q.body_bits(),
                        false,
                        false,
                        pw,
                        shares,
                        cal,
                    )
                };
                if same_transform_footprint(&m, &qt_method) {
                    Some(m)
                } else {
                    None
                }
            } else {
                shared_method_of_unequal_pair(
                    plan,
                    shares,
                    cal,
                    pw,
                    right.q.body_bits(),
                    left.t.body_bits(),
                    left.q.body_bits(),
                )
            };

            let pack_pair_for_l3 = |sm: &MulMethod| -> MulMethod {
                if let MulMethod::Ntt(mut s) = *sm {
                    let est_conc = shares.active_workers.div_ceil(pw.max(1)).max(1);
                    let my_l3_cap = ctx.ntt_cache.l3_prime_cap_bytes() / est_conc;
                    let up_rb = if s.cols >= 512 {
                        s.cols * 8 + 512
                    } else {
                        s.cols * 8
                    };
                    let pair_unpacked = (2usize.saturating_mul(s.k_primes).saturating_add(1))
                        .saturating_mul(s.rows)
                        .saturating_mul(up_rb);
                    if !s.packed
                        && !s.out_of_core
                        && s.extra_sweeps == 0
                        && s.rows >= 32
                        && s.cols >= 256
                        && s.cols.is_multiple_of(256)
                        && s.tile_cols.is_multiple_of(8)
                        && pair_unpacked > my_l3_cap
                    {
                        s.packed = true;
                        return MulMethod::Ntt(s);
                    }
                }
                sm.clone()
            };

            let (q, qt1) = if let Some(ref sm_raw) = shared_q_method {
                let sm_packed = pack_pair_for_l3(sm_raw);
                let sm = &sm_packed;
                if plan.overrides.split_override == crate::plan::SplitFormOverride::Auto
                    && let Some((qt_bn, q_bn)) = ctx.try_mul_pair_shared_b(
                        sm,
                        pw,
                        plan.ram_transform_budget,
                        &left.t,
                        &left.q,
                        &right.q,
                    )
                {
                    (q_bn, qt_bn)
                } else if !right.q.is_zero()
                    && !left.t.is_zero()
                    && !left.q.is_zero()
                    && can_cache_in_ram(sm)
                {
                    let mut fq2 = ctx.forward_cached_reusable(sm, &right.q.limbs)?;
                    let q2_lw0 = right.q.limbs_w0();
                    let t1_lw0 = left.t.limbs_w0();
                    let exp_qt_lw0 = mod61_mul(q2_lw0, t1_lw0);
                    let qt_res = ctx.mul_limbs_with_cached(
                        base_pid,
                        sm,
                        &left.t.limbs,
                        &right.q.limbs,
                        &fq2,
                        left.t.limbs.len() + right.q.limbs.len() + 2,
                        Some(exp_qt_lw0),
                    );
                    let q1_lw0 = left.q.limbs_w0();
                    let exp_qq_lw0 = mod61_mul(q1_lw0, q2_lw0);
                    let qq_res = ctx.mul_limbs_with_cached(
                        base_pid + 1,
                        sm,
                        &left.q.limbs,
                        &right.q.limbs,
                        &fq2,
                        left.q.limbs.len() + right.q.limbs.len() + 2,
                        Some(exp_qq_lw0),
                    );
                    fq2.release_store(ctx.store);
                    let (qt_limbs, qt_lw0) = qt_res?;
                    let (qq_limbs, qq_lw0) = qq_res?;
                    let qt_bn = BigNum::from_limbs_with_w0(
                        qt_limbs,
                        left.t.neg,
                        right.q.shift + left.t.shift,
                        qt_lw0,
                    );
                    let q_bn = BigNum::from_limbs_with_w0(
                        qq_limbs,
                        false,
                        left.q.shift + right.q.shift,
                        qq_lw0,
                    );
                    (q_bn, qt_bn)
                } else {
                    let qt_bn = ctx.mul_whole(base_pid, &qt_method, &mut right.q, &mut left.t)?;
                    let q_bn = ctx.mul_whole(base_pid + 1, &qq_method, &mut left.q, &mut right.q)?;
                    (q_bn, qt_bn)
                }
            } else {
                let qt_bn = ctx.mul_whole(base_pid, &qt_method, &mut right.q, &mut left.t)?;
                let q_bn = ctx.mul_whole(base_pid + 1, &qq_method, &mut left.q, &mut right.q)?;
                (q_bn, qt_bn)
            };
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));

            let pw2 = cur_pw(pw);
            let pt_method = plan.select_mul_method(
                p1_body_bits,
                right.t.body_bits(),
                false,
                false,
                pw2,
                shares,
                cal,
            );
            let mut deferred_p: Option<DeferredProduct> = None;
            let room_for_t = ask_room_for_t(&qt1, ext_lp.unwrap_or(&left.p), &right.t);
            let (pt2, p) = if let Some(lp_ref) = ext_lp {
                let pt2 = ctx.mul_whole_immut(base_pid + 2, &pt_method, lp_ref, &right.t)?;
                (pt2, BigNum::zero())
            } else if is_rightmost {
                let pt2 = ctx.mul_whole(base_pid + 2, &pt_method, &mut left.p, &mut right.t)?;
                (pt2, BigNum::zero())
            } else {
                let pp_method = plan.select_mul_method(
                    left.p.body_bits(),
                    right.p.body_bits(),
                    false,
                    false,
                    pw2,
                    shares,
                    cal,
                );
                let shared_p_method = if same_transform_footprint(&pt_method, &pp_method) {
                    let m = if pt_method == pp_method {
                        pt_method.clone()
                    } else {
                        plan.select_mul_method(
                            right.t.body_bits().max(right.p.body_bits()),
                            left.p.body_bits(),
                            false,
                            false,
                            pw2,
                            shares,
                            cal,
                        )
                    };
                    if same_transform_footprint(&m, &pt_method) {
                        Some(m)
                    } else {
                        None
                    }
                } else {
                    shared_method_of_unequal_pair(
                        plan,
                        shares,
                        cal,
                        pw2,
                        left.p.body_bits(),
                        right.t.body_bits(),
                        right.p.body_bits(),
                    )
                };
                if let Some(ref sm_raw) = shared_p_method {
                    let sm_packed = pack_pair_for_l3(sm_raw);
                    let sm = &sm_packed;
                    if plan.overrides.split_override == crate::plan::SplitFormOverride::Auto
                        && let Some((pt_bn, p_bn)) = ctx.try_mul_pair_shared_b(
                            sm,
                            pw2,
                            plan.ram_transform_budget,
                            &right.t,
                            &right.p,
                            &left.p,
                        )
                    {
                        (pt_bn, p_bn)
                    } else if !left.p.is_zero()
                        && !right.t.is_zero()
                        && !right.p.is_zero()
                        && can_cache_in_ram(sm)
                    {
                        let mut fp1 = ctx.forward_cached_reusable(sm, &left.p.limbs)?;
                        let p1_lw0 = left.p.limbs_w0();
                        let t2_lw0 = right.t.limbs_w0();
                        let exp_pt_lw0 = mod61_mul(p1_lw0, t2_lw0);
                        let pt_res = ctx.mul_limbs_with_cached(
                            base_pid + 2,
                            sm,
                            &right.t.limbs,
                            &left.p.limbs,
                            &fp1,
                            right.t.limbs.len() + left.p.limbs.len() + 2,
                            Some(exp_pt_lw0),
                        );
                        let p2_lw0 = right.p.limbs_w0();
                        let exp_pp_lw0 = mod61_mul(p1_lw0, p2_lw0);
                        let pp_res = ctx.mul_limbs_with_cached(
                            base_pid + 3,
                            sm,
                            &right.p.limbs,
                            &left.p.limbs,
                            &fp1,
                            right.p.limbs.len() + left.p.limbs.len() + 2,
                            Some(exp_pp_lw0),
                        );
                        fp1.release_store(ctx.store);
                        let (pt_limbs, pt_lw0) = pt_res?;
                        let (pp_limbs, pp_lw0) = pp_res?;
                        let pt_bn = BigNum::from_limbs_with_w0(
                            pt_limbs,
                            left.p.neg ^ right.t.neg,
                            left.p.shift + right.t.shift,
                            pt_lw0,
                        );
                        let p_bn = BigNum::from_limbs_with_w0(
                            pp_limbs,
                            left.p.neg ^ right.p.neg,
                            left.p.shift + right.p.shift,
                            pp_lw0,
                        );
                        (pt_bn, p_bn)
                    } else if defer_p {
                        let pt_bn =
                            ctx.mul_whole(base_pid + 2, &pt_method, &mut left.p, &mut right.t)?;
                        deferred_p = Some((
                            base_pid + 3,
                            std::mem::replace(&mut left.p, BigNum::zero()),
                            std::mem::replace(&mut right.p, BigNum::zero()),
                        ));
                        (pt_bn, BigNum::zero())
                    } else {
                        let pt_bn =
                            ctx.mul_whole(base_pid + 2, &pt_method, &mut left.p, &mut right.t)?;
                        let pw3 = cur_pw(pw2);
                        let pp_m_eff = if pw3 > pw2 {
                            plan.select_mul_method(
                                left.p.body_bits(),
                                right.p.body_bits(),
                                false,
                                false,
                                pw3,
                                shares,
                                cal,
                            )
                        } else {
                            pp_method
                        };
                        let p_bn =
                            ctx.mul_whole(base_pid + 3, &pp_m_eff, &mut left.p, &mut right.p)?;
                        (pt_bn, p_bn)
                    }
                } else if defer_p {
                    let pt_bn =
                        ctx.mul_whole(base_pid + 2, &pt_method, &mut left.p, &mut right.t)?;
                    deferred_p = Some((
                        base_pid + 3,
                        std::mem::replace(&mut left.p, BigNum::zero()),
                        std::mem::replace(&mut right.p, BigNum::zero()),
                    ));
                    (pt_bn, BigNum::zero())
                } else {
                    let pt_bn =
                        ctx.mul_whole(base_pid + 2, &pt_method, &mut left.p, &mut right.t)?;
                    let pw3 = cur_pw(pw2);
                    let pp_m_eff = if pw3 > pw2 {
                        plan.select_mul_method(
                            left.p.body_bits(),
                            right.p.body_bits(),
                            false,
                            false,
                            pw3,
                            shares,
                            cal,
                        )
                    } else {
                        pp_method
                    };
                    let p_bn =
                        ctx.mul_whole(base_pid + 3, &pp_m_eff, &mut left.p, &mut right.p)?;
                    (pt_bn, p_bn)
                }
            };
            drop(room_for_t);
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
            let t = add_bignum_into_par(ctx.kernels, &qt1, pt2, ctx.sched, cur_pw(pw2));
            crate::limb::put_limb_buf(qt1.into_limbs());

            Ok((BlockTriple { p, q, t }, deferred_p))
        }
        MergeForm::Fused => {
            // Fused 7-transform merge: `T = Q2_full * T1 + P1 * T2` shares `fwd(Q2_full)`!
            let full_q2_bits = right.q.total_bits();
            let t1_bits = left.t.total_bits();
            let p1_bits = left.p.total_bits();
            let t2_bits = right.t.total_bits();
            let fused_bits = (full_q2_bits + t1_bits).max(p1_bits + t2_bits);
            let max_op_bits = full_q2_bits.max(t1_bits).max(p1_bits).max(t2_bits);
            let other_op_bits = fused_bits.saturating_sub(max_op_bits);
            let fused_method = plan.select_mul_method(
                max_op_bits,
                other_op_bits,
                true,
                false,
                pw,
                shares,
                cal,
            );

            // Decide upfront whether `fp1` pays to reuse for `P = P1 * P2` (2 transforms at
            // `fused_method` vs 3 transforms at `pp_method`) so `fused_linear_pair` only caches
            // `fp1`'s row pass when reused and releases it immediately otherwise!
            let qq_method = plan.select_mul_method(
                left.q.body_bits(),
                right.q.body_bits(),
                false,
                false,
                pw,
                shares,
                cal,
            );
            let can_reuse_fq2 = match (&fused_method, &qq_method) {
                (
                    MulMethod::Fft {
                        n_complex: nf,
                        coeff_bits: cbf,
                    },
                    MulMethod::Fft { n_complex: nq, .. },
                ) => {
                    let q_out_bits = ((left.q.limbs.len() + right.q.limbs.len()) as u64) * 64
                        + right.q.shift
                        + 64;
                    let f_cap_bits = (2 * (*nf as u64)) * (*cbf as u64);
                    !left.q.is_zero()
                        && !right.q.is_zero()
                        && q_out_bits <= f_cap_bits
                        && *nf <= *nq
                }
                (MulMethod::Ntt(sf), MulMethod::Ntt(sq)) => {
                    let q_out_bits = ((left.q.limbs.len() + right.q.limbs.len()) as u64) * 64
                        + right.q.shift
                        + 64;
                    let sf_cap_bits = (sf.n as u64) * (sf.coeff_bits as u64);
                    !left.q.is_zero()
                        && !right.q.is_zero()
                        && !sf.negacyclic
                        && !sf.wrap_cyclic
                        && q_out_bits <= sf_cap_bits
                        && 2 * sf.n * sf.k_primes <= 3 * sq.n * sq.k_primes
                }
                _ => false,
            };
            let pp_method_opt = if is_rightmost {
                None
            } else {
                Some(plan.select_mul_method(
                    left.p.body_bits(),
                    right.p.body_bits(),
                    false,
                    false,
                    pw,
                    shares,
                    cal,
                ))
            };
            let can_reuse_fp1 = match (&fused_method, &pp_method_opt) {
                (
                    MulMethod::Fft {
                        n_complex: nf,
                        coeff_bits: cbf,
                    },
                    Some(MulMethod::Fft { n_complex: np, .. }),
                ) => {
                    let p_out_bits = ((left.p.limbs.len() + right.p.limbs.len()) as u64) * 64;
                    let f_cap_bits = (2 * (*nf as u64)) * (*cbf as u64);
                    !left.p.is_zero()
                        && !right.p.is_zero()
                        && left.p.shift == 0
                        && right.p.shift == 0
                        && p_out_bits <= f_cap_bits
                        && *nf <= *np
                }
                (MulMethod::Ntt(sf), Some(MulMethod::Ntt(sp))) => {
                    let p_out_bits = ((left.p.limbs.len() + right.p.limbs.len()) as u64) * 64;
                    let sf_cap_bits = (sf.n as u64) * (sf.coeff_bits as u64);
                    let sf_grid = (sf.k_primes as u64)
                        .saturating_mul(sf.rows as u64)
                        .saturating_mul(sf.outer_row_bytes() as u64);
                    let sp_grid = (sp.k_primes as u64)
                        .saturating_mul(sp.rows as u64)
                        .saturating_mul(sp.outer_row_bytes() as u64);
                    let conc_m = (shares.active_workers / pw.max(1)).max(1) as u64;
                    let sf_l3 = shares.effective_l3_bytes(conc_m, sf.rows) * 7 / 8;
                    let sp_l3 = shares.effective_l3_bytes(conc_m, sp.rows) * 7 / 8;
                    let sf_pbp = (sf.rows * sf.cols * 16) as u64;
                    let sp_pbp = (sp.rows * sp.cols * 16) as u64;
                    let sf_spills_while_sp_fits =
                        (sf_grid * 3 > sf_l3 && sp_grid * 3 <= sp_l3)
                            || (sf_pbp > sf_l3 && sp_pbp <= sp_l3);
                    !left.p.is_zero()
                        && !right.p.is_zero()
                        && left.p.shift == 0
                        && right.p.shift == 0
                        && !sf.negacyclic
                        && !sf.wrap_cyclic
                        && p_out_bits <= sf_cap_bits
                        && !sf_spills_while_sp_fits
                        && 2 * sf.n * sf.k_primes <= 3 * sp.n * sp.k_primes
                }
                _ => false,
            };

            if !matches!(fused_method, MulMethod::Ntt(s) if s.out_of_core)
                && ctx.fault_hook.is_none()
            {
                let fuse_p = !is_rightmost && can_reuse_fp1;
                let is_root_node = is_rightmost
                    && ((left.t.limbs.len() + right.t.limbs.len()) as u64).saturating_mul(64)
                        >= plan.target_frac_bits;
                let live_root_bytes = ((left.p.limbs.len()
                    + left.q.limbs.len()
                    + left.t.limbs.len()
                    + right.p.limbs.len()
                    + right.q.limbs.len()
                    + right.t.limbs.len()) as u64)
                    .saturating_mul(8);
                let os_head = shares.total_ram / 16;
                let merge_tf_budget = if is_root_node && !plan.out_of_core {
                    shares
                        .total_ram
                        .saturating_sub(live_root_bytes + os_head)
                        .max(plan.ram_transform_budget)
                } else {
                    plan.ram_transform_budget
                };
                let mut fused_res = ctx.try_ntt_fused_merge_triple(
                    &fused_method,
                    pw,
                    merge_tf_budget,
                    &left.p,
                    &left.q,
                    &left.t,
                    &right.p,
                    &right.q,
                    &right.t,
                    can_reuse_fq2,
                    fuse_p,
                );
                if fused_res.is_none() && can_reuse_fq2 && fuse_p {
                    fused_res = ctx.try_ntt_fused_merge_triple(
                        &fused_method,
                        pw,
                        merge_tf_budget,
                        &left.p,
                        &left.q,
                        &left.t,
                        &right.p,
                        &right.q,
                        &right.t,
                        true,
                        false,
                    );
                }
                if fused_res.is_none() && (can_reuse_fq2 || fuse_p) {
                    fused_res = ctx.try_ntt_fused_merge_triple(
                        &fused_method,
                        pw,
                        merge_tf_budget,
                        &left.p,
                        &left.q,
                        &left.t,
                        &right.p,
                        &right.q,
                        &right.t,
                        false,
                        false,
                    );
                }
                if let Some((t_bn, q_opt, p_opt)) = fused_res {
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));
                    if is_rightmost {
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                    }
                    let q = match q_opt {
                        Some(qb) => {
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                            qb
                        }
                        None => {
                            let pw_q = cur_pw(pw);
                            let qm_eff = if pw_q > pw {
                                plan.select_mul_method(
                                    left.q.body_bits(),
                                    right.q.body_bits(),
                                    false,
                                    false,
                                    pw_q,
                                    shares,
                                    cal,
                                )
                            } else {
                                qq_method
                            };
                            let qb = ctx.mul_whole(base_pid + 1, &qm_eff, &mut left.q, &mut right.q)?;
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                            qb
                        }
                    };
                    let mut deferred_p: Option<DeferredProduct> = None;
                    let p = if is_rightmost {
                        BigNum::zero()
                    } else {
                        match p_opt {
                            Some(pb) => {
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                                pb
                            }
                            None if defer_p => {
                                deferred_p = Some((
                                    base_pid + 3,
                                    std::mem::replace(&mut left.p, BigNum::zero()),
                                    std::mem::replace(&mut right.p, BigNum::zero()),
                                ));
                                BigNum::zero()
                            }
                            None => {
                                let pw_p = cur_pw(pw);
                                let pp_method = if pw_p > pw {
                                    plan.select_mul_method(
                                        left.p.body_bits(),
                                        right.p.body_bits(),
                                        false,
                                        false,
                                        pw_p,
                                        shares,
                                        cal,
                                    )
                                } else {
                                    pp_method_opt.unwrap()
                                };
                                let pb = ctx.mul_whole(base_pid + 3, &pp_method, &mut left.p, &mut right.p)?;
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                                pb
                            }
                        }
                    };
                    return Ok((BlockTriple { p, q, t: t_bn }, deferred_p));
                }
            }

            let (t, maybe_fq2, held_fp1) = ctx.fused_linear_pair(
                base_pid,
                &fused_method,
                pw,
                &mut right.q,
                &mut left.t,
                &mut left.p,
                &mut right.t,
                can_reuse_fq2,
                can_reuse_fp1,
            )?;

            let q = if let Some((mut fq2, q2_full_limbs)) = maybe_fq2 {
                // `Q1_odd * Q2_full` has `(Q1_odd * Q2_odd) * 2^{right.q.shift}`, costing only 2
                // transforms (`fwd(Q1_odd)` + `inv`), then `strip_zero_bits_par` restores odd `Q`!
                let q1_lw0 = left.q.limbs_w0();
                let q2_lw0 = right.q.w0();
                let exp_lw0 = mod61_mul(q1_lw0, q2_lw0);
                let qq_res = ctx.mul_limbs_with_cached(
                    base_pid + 1,
                    &fused_method,
                    &left.q.limbs,
                    &q2_full_limbs,
                    &fq2,
                    left.q.limbs.len() + q2_full_limbs.len() + 2,
                    Some(exp_lw0),
                );
                fq2.release_store(ctx.store);
                crate::limb::put_limb_buf(q2_full_limbs);
                let (qq_limbs, qq_lw0) = qq_res?;
                let mut q_bn = BigNum::from_limbs_with_w0(qq_limbs, false, left.q.shift, qq_lw0);
                q_bn.strip_zero_bits_par(ctx.sched, pw);
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                q_bn
            } else {
                let qb = ctx.mul_whole(base_pid + 1, &qq_method, &mut left.q, &mut right.q)?;
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.q.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.q.limbs));
                qb
            };

            let mut deferred_p: Option<DeferredProduct> = None;
            let p = if is_rightmost {
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                BigNum::zero()
            } else if let Some(mut fp1) = held_fp1 {
                let p1_lw0 = left.p.limbs_w0();
                let p2_lw0 = right.p.limbs_w0();
                let exp_p_lw0 = mod61_mul(p2_lw0, p1_lw0);
                let pp_res = ctx.mul_limbs_with_cached(
                    base_pid + 3,
                    &fused_method,
                    &right.p.limbs,
                    &left.p.limbs,
                    &fp1,
                    left.p.limbs.len() + right.p.limbs.len() + 2,
                    Some(exp_p_lw0),
                );
                fp1.release_store(ctx.store);
                let (pp_limbs, pp_lw0) = pp_res?;
                let pb = BigNum::from_limbs_with_w0(pp_limbs, left.p.neg ^ right.p.neg, 0, pp_lw0);
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                pb
            } else if defer_p {
                deferred_p = Some((
                    base_pid + 3,
                    std::mem::replace(&mut left.p, BigNum::zero()),
                    std::mem::replace(&mut right.p, BigNum::zero()),
                ));
                BigNum::zero()
            } else {
                let pp_method = pp_method_opt.unwrap();
                let pb = ctx.mul_whole(base_pid + 3, &pp_method, &mut left.p, &mut right.p)?;
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.p.limbs));
                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.p.limbs));
                pb
            };
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut left.t.limbs));
            crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut right.t.limbs));

            Ok((BlockTriple { p, q, t }, deferred_p))
        }
    }
}

/// Run the full reduced binary-splitting series stage (Section 4), returning root `(Q, T)`.
#[allow(clippy::too_many_arguments)]
pub fn run_series(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    sieve_ops_counter: &AtomicU64,
    resume: Option<&SeriesResumeState>,
    ckpt_sink: Option<&dyn SeriesCheckpointSink>,
) -> Result<(BigNum, BigNum), MulError> {
    let h = plan.tree_height;
    if h == 0 {
        if let Some(st) = resume {
            if let Some(tr) = st.nodes.get(&(0, 0)) {
                return Ok((tr.q.clone(), tr.t.clone()));
            }
        }
        let tr = compute_leaf_plain(ctx, 0, plan.leaf_len, false, cal.basecase_max_limbs);
        return Ok((tr.q, tr.t));
    }

    // Segmented sieve up to maximum dividing block length `L_max` once upfront!
    let max_div_terms = plan
        .dividing_levels
        .iter()
        .map(|d| d.block_terms)
        .max()
        .unwrap_or(0);
    let do_trace = std::env::var_os("SUPI_TRACE").is_some();
    let t_sv0 = Instant::now();
    let primes = if max_div_terms >= 3 {
        segmented_primes_up_to(max_div_terms, sieve_ops_counter)
    } else {
        Vec::new()
    };
    if do_trace {
        eprintln!(
            "[TRACE SERIES] sieve={:.2}ms primes={}",
            t_sv0.elapsed().as_secs_f64() * 1e3,
            primes.len()
        );
    }

    // Determine subtree steal frontier `sub_level`:
    // Below the depth where `nodes_at_level > 16 * workers`, a subtree runs sequentially on one
    // worker and work stealing balances the pool (Section 10). Stop advancing `frontier_level`
    // in memory once the next merge level outgrows 1D L2 (`rows > 1`) so 16 sequential worker
    // subtrees never thrash shared L3 with concurrent 3-sweep 2D grids.
    let max_steal_nodes = ctx.sched.subtree_steal_max_nodes() as u64;
    let mut frontier_level = 0u32;
    while frontier_level < h && (1u64 << (h - frontier_level)) > max_steal_nodes {
        if !plan.out_of_core
            && plan.overrides.force_series_shape.is_none()
            && (1u64 << (h - frontier_level)) <= 32_768
        {
            let cand_lv = frontier_level + 1;
            let prev_d = plan
                .dividing_levels
                .iter()
                .filter(|d| d.level < cand_lv)
                .map(|d| d.level)
                .max()
                .unwrap_or(0);
            let est_bpt = if prev_d == 0 {
                55.0 + 3.0 * (plan.term_count.max(2) as f64).log2()
            } else {
                66.5 + 3.0 * ((h - prev_d) as f64)
            };
            let child_terms = plan.leaf_len.saturating_mul(1u64 << frontier_level);
            let est_child_bits = ((child_terms as f64) * est_bpt) as u64 + 128;
            let cand_m =
                plan.select_mul_method(est_child_bits, est_child_bits, true, false, 1, shares, cal);
            if let MulMethod::Ntt(s) = cand_m {
                let grid_4 = (s.k_primes as u64)
                    .saturating_mul(s.rows as u64)
                    .saturating_mul(s.outer_row_bytes() as u64)
                    .saturating_mul(4);
                let num_ch = (s.coeff_bits as u64).div_ceil(48).max(1);
                let ws_1d = ((s.k_primes as u64) + num_ch + 4)
                    .saturating_mul(s.n as u64)
                    .saturating_mul(8);
                let aw = shares.active_workers.max(1) as u64;
                let total_eff_l3 = shares.effective_l3_bytes(1, 1);
                if s.rows > 1
                    || !shares.fits_l3_domains(aw, ws_1d, 1, 7, 8)
                    || grid_4.saturating_mul(aw)
                        > plan.ram_transform_budget.min(total_eff_l3.saturating_mul(2))
                {
                    break;
                }
            }
        }
        frontier_level += 1;
    }
    // If resuming from a checkpoint with nodes below `frontier_level`, clamp `frontier_level` so
    // every saved node is picked up directly.
    if let Some(st) = resume {
        if let Some(&min_res_lv) = st.nodes.keys().map(|(l, _)| l).min() {
            frontier_level = frontier_level.min(min_res_lv);
        }
    }

    // Prebuild any dividing-level factors & cached inverses at or below `frontier_level` once so
    // all worker subtrees share them read-only without breadth-first DRAM materialization!
    let total_tree_terms = plan.leaf_len.saturating_mul(1u64 << h).max(2);
    let log2_total_terms = (total_tree_terms as f64).log2();
    let mut sub_divs: Vec<(u32, PreparedLevelDivisor)> = Vec::new();
    for div_plan in plan
        .dividing_levels
        .iter()
        .filter(|d| d.level <= frontier_level)
    {
        let t_sd0 = Instant::now();
        let l_cur = div_plan.block_terms.max(2);
        let unred_t_bits = ((l_cur as f64) * (53.30 + 3.0 * log2_total_terms)
            + 31.0
            + log2_total_terms
            + (l_cur as f64).log2())
        .ceil() as u64
            + 128;
        let unred_q_bits =
            ((l_cur as f64) * (38.5 + 3.0 * log2_total_terms)).ceil() as u64 + 128;
        let unred_p_bits =
            ((l_cur as f64) * (6.80 + 3.0 * log2_total_terms)).ceil() as u64 + 128;
        let mut shed_bits_before = 0u64;
        for (prev_lv, prev_div) in &sub_divs {
            let mult = 1u64 << (div_plan.level - *prev_lv);
            shed_bits_before = shed_bits_before.saturating_add(
                mult.saturating_mul(prev_div.factor.body_bits().saturating_sub(1)),
            );
        }
        let est_max_num_limbs =
            (unred_t_bits.saturating_sub(shed_bits_before).div_ceil(64) as usize) + 4;
        let est_max_q_limbs =
            (unred_q_bits.saturating_sub(shed_bits_before).div_ceil(64) as usize) + 4;
        let est_max_p_limbs =
            (unred_p_bits.saturating_sub(shed_bits_before).div_ceil(64) as usize) + 4;
        let divisor = prepare_level_divisor(
            ctx,
            plan,
            shares,
            cal,
            &primes,
            div_plan,
            est_max_num_limbs,
            est_max_q_limbs,
            est_max_p_limbs,
            1,
        )?;
        if do_trace {
            eprintln!(
                "[TRACE SERIES] prebuild sub_div lv={} form={:?} max_q={} -> {:.2}ms",
                div_plan.level,
                div_plan.form,
                divisor.max_quot_limbs,
                t_sd0.elapsed().as_secs_f64() * 1e3
            );
        }
        sub_divs.push((div_plan.level, divisor));
    }

    // Step 1: Evaluate all subtrees of height `frontier_level` (`1 << (h - frontier_level)` nodes)
    // in parallel across the worker pool!
    let frontier_nodes = (1usize) << (h - frontier_level);
    let mut cur_nodes: Vec<Option<BlockTriple>> = (0..frontier_nodes).map(|_| None).collect();

    // Populate any frontier nodes covered by `resume` (either exact level or covered by an ancestor
    // already in `resume`!).
    let is_covered_by_ancestor = |lv: u32, idx: u64| -> bool {
        let Some(st) = resume else {
            return false;
        };
        let mut c_lv = lv;
        let mut c_idx = idx;
        while c_lv <= h {
            if st.nodes.contains_key(&(c_lv, c_idx)) {
                return true;
            }
            c_lv += 1;
            c_idx >>= 1;
        }
        false
    };

    let err_slot: Mutex<Option<MulError>> = Mutex::new(None);
    let has_fr_err = AtomicBool::new(false);
    let cur_ptr = cur_nodes.as_mut_ptr() as usize;
    let sub_divs_slice = sub_divs.as_slice();

    let mut last_ckpt_instant = Instant::now();
    let t_fr0 = Instant::now();
    ctx.sched.par_for(frontier_nodes, 1, |step| {
        let lo = frontier_nodes - 1 - step;
        let idx = lo as u64;
        if let Some(st) = resume {
            if let Some(saved) = st.nodes.get(&(frontier_level, idx)) {
                unsafe {
                    *(cur_ptr as *mut Option<BlockTriple>).add(lo) = Some(saved.clone());
                }
                return;
            }
        }
        if is_covered_by_ancestor(frontier_level, idx) {
            return;
        }
        if has_fr_err.load(Ordering::Relaxed) {
            return;
        }
        let is_rightmost = idx + 1 == (frontier_nodes as u64);
        let active_w = shares.active_workers.max(1);
        let tail_split_depth = if frontier_level >= 4
            && lo < active_w.saturating_mul(2)
            && resume.is_none()
            && ctx.fault_hook.is_none()
        {
            if lo < active_w { 3 } else { 2 }
        } else {
            0
        };
        let sub_res = if tail_split_depth > 0 {
            eval_tail_subtree_split(
                ctx,
                plan,
                shares,
                cal,
                sub_divs_slice,
                frontier_level,
                idx,
                is_rightmost,
                tail_split_depth,
            )
        } else {
            eval_sequential_subtree(
                ctx,
                plan,
                shares,
                cal,
                sub_divs_slice,
                frontier_level,
                idx,
                is_rightmost,
            )
        };
        match sub_res {
            Ok(tr) => unsafe {
                *(cur_ptr as *mut Option<BlockTriple>).add(lo) = Some(tr);
            },
            Err(e) => {
                has_fr_err.store(true, Ordering::Relaxed);
                let mut g = err_slot.lock().unwrap();
                if g.is_none() {
                    *g = Some(e);
                }
            }
        }
    });
    for (_, mut d) in sub_divs {
        d.release_store(ctx.store);
    }
    ctx.sched.on_each_worker(|| {
        crate::fft::clear_tls_fwd_pool();
    });
    if do_trace {
        let lf = FR_LEAF_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
        let fm = FR_LOW_MERGE_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
        let hm = FR_HIGH_MERGE_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
        let dv = FR_DIV_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
        eprintln!(
            "[TRACE SERIES] frontier(lv={frontier_level}, nodes={frontier_nodes})={:.2}ms | cpu_sum: leaf={lf:.1}ms merge_lv<=6={fm:.1}ms merge_lv>=7={hm:.1}ms div={dv:.1}ms",
            t_fr0.elapsed().as_secs_f64() * 1e3
        );
    }
    if let Some(e) = err_slot.into_inner().unwrap() {
        return Err(e);
    }
    if frontier_level >= 1 {
        if let Some(sink) = ckpt_sink {
            let elapsed = last_ckpt_instant.elapsed().as_secs_f64();
            let mut live_refs = Vec::with_capacity(cur_nodes.len());
            for (n_idx, slot) in cur_nodes.iter().enumerate() {
                if let Some(tr) = slot {
                    live_refs.push((frontier_level, n_idx as u64, tr));
                }
            }
            if sink.maybe_checkpoint_series(frontier_level, &live_refs, elapsed)? {
                last_ckpt_instant = Instant::now();
            }
        }
    }

    // Step 2: Sweep upward level by level from `frontier_level + 1 ..= h`!
    let roomy_run = plan.roomy_run();
    let _ = max_steal_nodes;
    let active_w = shares.active_workers.max(1);
    let mut cur_parked: Vec<Option<ParkedTriple<'_>>> = (0..cur_nodes.len()).map(|_| None).collect();
    // The counts of nodes at once that are tried: those that share the workers out evenly and,
    // level by level, go into the level's nodes.
    let conc_cands = crate::tier::even_splits(active_w);
    for lv in (frontier_level + 1)..=h {
        if !plan.out_of_core && !roomy_run && (lv == 15 || lv == 18 || lv == 21) {
            if lv == 15 {
                ctx.sched.on_each_worker(|| {
                    crate::ntt::clear_tls_1d_fused();
                });
            }
            crate::arena::trim_active_arenas_above(
                ((shares.total_ram / 32).saturating_mul(31)) as usize,
            );
        }
        let nodes_at_lv = (1usize) << (h - lv);
        let mid_slot = (cur_nodes.len() / 2) & !1usize;
        let slot_stats = |idx: usize| -> Option<(u64, u64, u64, u64)> {
            if let Some(ref tr) = cur_nodes[idx] {
                Some((
                    tr.q.body_bits(),
                    tr.q.shift,
                    tr.t.body_bits(),
                    tr.p.body_bits(),
                ))
            } else {
                cur_parked[idx].as_ref().map(ParkedTriple::stats)
            }
        };
        let sample_bits = (mid_slot..cur_nodes.len())
            .chain(0..mid_slot)
            .find_map(slot_stats)
            .map(|(qb, _, tb, _)| tb.max(qb))
            .unwrap_or(64);
        // The sizes a representative merge near the middle of the level works on, to price a node.
        let node_sizes = {
            let mut it = (mid_slot..cur_nodes.len())
                .chain(0..mid_slot)
                .filter_map(slot_stats);
            match (it.next(), it.next()) {
                (Some((q1, _, t1, p1)), Some((q2, sh, t2, p2))) => {
                    Some((q1, q2, sh, t1, p1, t2, p2))
                }
                _ => None,
            }
        };
        let cand_pw = if nodes_at_lv >= active_w {
            1
        } else {
            (active_w / nodes_at_lv.max(1)).max(1)
        };
        let rep_mul_args = |pw_q: usize| -> (u64, u64, bool) {
            if let Some((q1, q2, sh, t1, p1, t2, p2)) = node_sizes {
                match plan.select_merge_form_node(
                    q1, q2, sh, t1, p1, t2, p2, false, pw_q, shares, cal,
                ) {
                    crate::plan::MergeForm::Separate => (q2.max(p1), t1.max(t2), false),
                    crate::plan::MergeForm::Fused => {
                        let q2_full = q2.saturating_add(sh);
                        let fused_bits = (q2_full + t1).max(p1 + t2);
                        let hi = q2_full.max(t1).max(p1).max(t2);
                        (hi, fused_bits.saturating_sub(hi), true)
                    }
                }
            } else {
                (sample_bits, sample_bits, true)
            }
        };
        let (rep_a0, rep_b0, rep_f0) = rep_mul_args(cand_pw);
        let sample_m = plan.select_mul_method(
            rep_a0,
            rep_b0,
            rep_f0,
            false,
            cand_pw,
            shares,
            cal,
        );
        let conc_merges = if !matches!(sample_m, MulMethod::Ntt(s) if s.out_of_core) {
            if let MulMethod::Ntt(s) = sample_m {
                let grid_1 = (s.k_primes as u64)
                    .saturating_mul(s.rows as u64)
                    .saturating_mul(s.outer_row_bytes() as u64)
                    .max(1);
                let grid_4 = grid_1.saturating_mul(4);
                let max_ram_conc = (plan.ram_transform_budget / grid_4) as usize;
                if s.rows == 1 {
                    nodes_at_lv.min(active_w).min(max_ram_conc.max(1)).max(1)
                } else {
                    // Fewer nodes at once leave a product more workers and the cache less
                    // crowded, more at once lose less between a product's workers: the count
                    // is the largest that the node's grids and the memory allow. A product
                    // should take no more workers than its grid has rows. With many workers and
                    // grids of few rows the counts that keep to that may have no room in the
                    // cache. The fewest nodes at once that keep to it are then run where their
                    // grids still lie within the caches they run on, crowded but not beyond
                    // them. Where even those do not, the nodes are larger than the cache in
                    // effect and take its rule: four workers each, as many at once as that
                    // gives. And near the top of the tree no count may keep to the rows at
                    // all: the cache alone then decides, the largest count with room, which
                    // also leaves a product the fewest workers. One node at once with every
                    // worker is the choice only where it is the best by these rules, not
                    // where nothing passed.
                    let mut by_rows_and_room: Option<usize> = None;
                    let mut by_rows: Option<(usize, bool)> = None;
                    let mut by_rows_four_each: Option<usize> = None;
                    let mut by_room: Option<usize> = None;
                    for &c in &conc_cands {
                        // (a count that does not go into the level's nodes would leave its last
                        // merges fewer than the count, each with a share of the workers only)
                        if c < 1 || c > nodes_at_lv || c > active_w || !nodes_at_lv.is_multiple_of(c)
                        {
                            continue;
                        }
                        let pw_c = (active_w / c).max(1);
                        let sc = if pw_c == cand_pw {
                            s
                        } else {
                            let (ra, rb, rf) = rep_mul_args(pw_c);
                            if let MulMethod::Ntt(sf) = plan.select_mul_method(
                                ra,
                                rb,
                                rf,
                                false,
                                pw_c,
                                shares,
                                cal,
                            ) {
                                sf
                            } else {
                                continue;
                            }
                        };
                        if sc.out_of_core {
                            continue;
                        }
                        let rows_ok = pw_c <= sc.rows.max(1);
                        let grid_c = (sc.k_primes as u64)
                            .saturating_mul(sc.rows as u64)
                            .saturating_mul(sc.outer_row_bytes() as u64)
                            .max(1);
                        // Several nodes at once hold their grids at once: only where that is at
                        // most half the room for transforms, and in a run that may need all the
                        // memory there is, a sixteenth of it (what a run holds near its end is
                        // more than what is live, by the space freed blocks leave in pieces, and
                        // two large nodes at once were measured to add 5 GB to it at 5e9 digits).
                        let room = if c > 1 {
                            if roomy_run {
                                plan.ram_transform_budget / 2
                            } else if !plan.out_of_core && h - lv >= 3 {
                                (plan.ram_transform_budget / 8).saturating_mul(5)
                            } else if !plan.out_of_core {
                                (plan.ram_transform_budget / 8).saturating_mul(3)
                            } else {
                                plan.ram_transform_budget / 4
                            }
                        } else {
                            plan.ram_transform_budget
                        };
                        if (c as u64).saturating_mul(grid_c.saturating_mul(4)) > room {
                            continue;
                        }
                        // By the room the node's transforms take in the last-level cache
                        // (`nodes_at_once_fit`): the largest count that fits is taken.
                        let has_room = crate::plan::nodes_at_once_fit(c, &sc, shares);
                        if rows_ok {
                            if by_rows.is_none() {
                                let grids = 3u64
                                    .saturating_mul(sc.k_primes as u64)
                                    .saturating_mul(sc.n as u64)
                                    .saturating_mul(8);
                                let within =
                                    shares.fits_l3_domains(c as u64, grids, sc.rows, 1, 1);
                                by_rows = Some((c, within));
                            }
                            if c <= (active_w / 4).max(1) {
                                by_rows_four_each = Some(c);
                            }
                            if has_room {
                                by_rows_and_room = Some(c);
                            }
                        }
                        if has_room {
                            by_room = Some(c);
                        }
                    }
                    let by_rows_alone = match by_rows {
                        Some((c, true)) => Some(c),
                        Some((c, false)) => by_rows_four_each.or(Some(c)),
                        None => None,
                    };
                    by_rows_and_room.or(by_rows_alone).or(by_room).unwrap_or(1)
                }
            } else {
                nodes_at_lv.min(active_w).max(1)
            }
        } else if nodes_at_lv >= 2 && !matches!(sample_m, MulMethod::Ntt(_)) {
            nodes_at_lv.min(active_w).max(1)
        } else {
            1
        };
        let par_merges = conc_merges >= 2;
        let prod_workers = if par_merges {
            (active_w / conc_merges).max(1)
        } else {
            active_w
        };

        let t_lv0 = Instant::now();
        let mut next_nodes: Vec<Option<BlockTriple>> = (0..nodes_at_lv).map(|_| None).collect();
        let mut next_parked: Vec<Option<ParkedTriple<'_>>> =
            (0..nodes_at_lv).map(|_| None).collect();

        if par_merges {
            unpark_all_triples(ctx, lv, &mut cur_nodes, &mut cur_parked, active_w)?;
            // When all nodes at this level run in one simultaneous wave (`nodes_at_lv == conc_merges`),
            // the rightmost node has smaller `Q2, T2` (after dividing levels) and skips `P1 * P2`,
            // so equal worker shares leave its threads idle behind the heavier sibling(s). Share
            // `active_w` in proportion to each node's non-deferred product work so they finish together.
            let node_pws: Vec<usize> =
                if nodes_at_lv == conc_merges && resume.is_none() && ctx.fault_hook.is_none() {
                    let mut sizes = vec![1u64; nodes_at_lv];
                    let mut caps = vec![1usize; nodes_at_lv];
                    for n_idx in 0..nodes_at_lv {
                        if is_covered_by_ancestor(lv, n_idx as u64) {
                            continue;
                        }
                        if let (Some(l), Some(r)) =
                            (&cur_nodes[2 * n_idx], &cur_nodes[2 * n_idx + 1])
                        {
                            let is_rm = n_idx + 1 == nodes_at_lv;
                            let def_p = !is_rm
                                && 2 * conc_merges <= active_w
                                && sample_bits >= 5_000_000;
                            let qt_b1 = r.q.body_bits();
                            let qt_b2 = l.t.body_bits().max(l.q.body_bits());
                            let w_q = (5.0 / 3.0)
                                * plan.product_worker_sec(
                                    qt_b1,
                                    qt_b2,
                                    false,
                                    prod_workers,
                                    shares,
                                    cal,
                                );
                            let w_pt = plan.product_worker_sec(
                                l.p.body_bits(),
                                r.t.body_bits(),
                                false,
                                prod_workers,
                                shares,
                                cal,
                            );
                            let w_pp = if !is_rm && !def_p {
                                plan.product_worker_sec(
                                    l.p.body_bits(),
                                    r.p.body_bits(),
                                    false,
                                    prod_workers,
                                    shares,
                                    cal,
                                )
                            } else {
                                0.0
                            };
                            sizes[n_idx] = ((w_q + w_pt + w_pp) * 1.0e9) as u64 + 1;
                            caps[n_idx] = match plan.select_mul_method(
                                qt_b1,
                                qt_b2,
                                false,
                                false,
                                active_w,
                                shares,
                                cal,
                            ) {
                                MulMethod::Ntt(s) if s.rows > 1 => s.rows.min(active_w),
                                _ => active_w,
                            };
                        }
                    }
                    let mut pw_vec = vec![1usize; nodes_at_lv];
                    let mut rem_w = active_w.saturating_sub(nodes_at_lv);
                    while rem_w > 0 {
                        let mut best: Option<usize> = None;
                        for i in 0..nodes_at_lv {
                            if pw_vec[i] >= caps[i].max(1) {
                                continue;
                            }
                            if best.is_none_or(|b| {
                                (sizes[i] as u128) * (pw_vec[b] as u128)
                                    > (sizes[b] as u128) * (pw_vec[i] as u128)
                            }) {
                                best = Some(i);
                            }
                        }
                        let Some(b) = best else { break };
                        pw_vec[b] += 1;
                        rem_w -= 1;
                    }
                    pw_vec
                } else {
                    Vec::new()
                };
            let node_pws_ref: &[usize] = &node_pws;
            let lpt_order = resume.is_none()
                && ctx.fault_hook.is_none()
                && crate::arena::active_numbers_live_bytes().saturating_mul(2)
                    <= plan.ram_numbers_budget;

            struct StealablePSlot {
                n_idx: usize,
                pid: u64,
                lp: std::cell::UnsafeCell<BigNum>,
                rp: std::cell::UnsafeCell<BigNum>,
                claimed: AtomicBool,
                refs: AtomicUsize,
                out: std::cell::UnsafeCell<Option<BigNum>>,
            }
            // SAFETY: `lp` and `rp` are initialized before `par_for_max` and accessed exclusively
            // as shared `&BigNum` until `refs` drops to 0 (`AcqRel`); `out` is written at most once
            // by the `claimed` CAS winner and read on the main thread after `par_for_max` joins.
            unsafe impl Sync for StealablePSlot {}

            let mut steal_slots: Vec<StealablePSlot> = Vec::new();
            let mut steal_idx_for_node: Vec<Option<usize>> = vec![None; nodes_at_lv];
            if 2 * conc_merges <= active_w
                && sample_bits >= 5_000_000
                && resume.is_none()
                && ctx.fault_hook.is_none()
            {
                let wave_start = nodes_at_lv.saturating_sub(conc_merges);
                for step in wave_start..nodes_at_lv {
                    let n_idx = if lpt_order {
                        step
                    } else {
                        nodes_at_lv - 1 - step
                    };
                    if n_idx + 1 >= nodes_at_lv || is_covered_by_ancestor(lv, n_idx as u64) {
                        continue;
                    }
                    let node_pw = node_pws_ref.get(n_idx).copied().unwrap_or(prod_workers).max(1);
                    let can_pre_steal = if let (Some(l), Some(r)) =
                        (&cur_nodes[2 * n_idx], &cur_nodes[2 * n_idx + 1])
                    {
                        let form = plan.select_merge_form_node(
                            l.q.body_bits(),
                            r.q.body_bits(),
                            r.q.shift,
                            l.t.body_bits(),
                            l.p.body_bits(),
                            r.t.body_bits(),
                            r.p.body_bits(),
                            false,
                            node_pw,
                            shares,
                            cal,
                        );
                        if matches!(form, MergeForm::Separate) {
                            let pt_m = plan.select_mul_method(
                                l.p.body_bits(),
                                r.t.body_bits(),
                                false,
                                false,
                                node_pw,
                                shares,
                                cal,
                            );
                            let pp_m = plan.select_mul_method(
                                l.p.body_bits(),
                                r.p.body_bits(),
                                false,
                                false,
                                node_pw,
                                shares,
                                cal,
                            );
                            !same_transform_footprint(&pt_m, &pp_m)
                                && shared_method_of_unequal_pair(
                                    plan,
                                    shares,
                                    cal,
                                    node_pw,
                                    l.p.body_bits(),
                                    r.t.body_bits(),
                                    r.p.body_bits(),
                                )
                                .is_none()
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if can_pre_steal {
                        let l = cur_nodes[2 * n_idx].as_mut().unwrap();
                        let lp = std::mem::replace(&mut l.p, BigNum::zero());
                        let r = cur_nodes[2 * n_idx + 1].as_mut().unwrap();
                        let rp = std::mem::replace(&mut r.p, BigNum::zero());
                        let _ = lp.w0();
                        let _ = rp.w0();
                        let base_pid =
                            0x1000_0000u64 | ((lv as u64) << 40) | ((n_idx as u64) << 4);
                        let si = steal_slots.len();
                        steal_slots.push(StealablePSlot {
                            n_idx,
                            pid: base_pid + 3,
                            lp: std::cell::UnsafeCell::new(lp),
                            rp: std::cell::UnsafeCell::new(rp),
                            claimed: AtomicBool::new(false),
                            refs: AtomicUsize::new(2),
                            out: std::cell::UnsafeCell::new(None),
                        });
                        steal_idx_for_node[n_idx] = Some(si);
                    }
                }
            }
            let release_steal_ref = |slot: &StealablePSlot| {
                if slot.refs.fetch_sub(1, Ordering::AcqRel) == 1 {
                    unsafe {
                        let lp = &mut *slot.lp.get();
                        let rp = &mut *slot.rp.get();
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut lp.limbs));
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut rp.limbs));
                    }
                }
            };

            let mut deferred_ps: Vec<Option<(usize, DeferredProduct)>> =
                (0..nodes_at_lv).map(|_| None).collect();
            let def_ptr = deferred_ps.as_mut_ptr() as usize;
            let next_ptr = next_nodes.as_mut_ptr() as usize;
            let prev_ptr = cur_nodes.as_mut_ptr() as usize;
            let lv_err: Mutex<Option<MulError>> = Mutex::new(None);
            let has_lv_err = AtomicBool::new(false);
            let live_merge_total = (0..nodes_at_lv)
                .filter(|&i| {
                    !resume.is_some_and(|st| st.nodes.contains_key(&(lv, i as u64)))
                        && !is_covered_by_ancestor(lv, i as u64)
                })
                .count();
            let rem_merges_ctr = AtomicUsize::new(live_merge_total);
            ctx.sched.par_for_max(nodes_at_lv, 1, conc_merges, |step| {
                let n_idx = if lpt_order {
                    step
                } else {
                    nodes_at_lv - 1 - step
                };
                let idx_u64 = n_idx as u64;
                if let Some(st) = resume {
                    if let Some(saved) = st.nodes.get(&(lv, idx_u64)) {
                        unsafe {
                            *(next_ptr as *mut Option<BlockTriple>).add(n_idx) =
                                Some(saved.clone());
                        }
                        return;
                    }
                }
                if is_covered_by_ancestor(lv, idx_u64) {
                    return;
                }
                if has_lv_err.load(Ordering::Relaxed) {
                    if let Some(si) = steal_idx_for_node[n_idx] {
                        release_steal_ref(&steal_slots[si]);
                    }
                    rem_merges_ctr.fetch_sub(1, Ordering::AcqRel);
                    return;
                }
                let (left, right) = unsafe {
                    let pp = prev_ptr as *mut Option<BlockTriple>;
                    (
                        (*pp.add(2 * n_idx)).take().expect("left child present"),
                        (*pp.add(2 * n_idx + 1))
                            .take()
                            .expect("right child present"),
                    )
                };
                let is_rightmost = n_idx + 1 == nodes_at_lv;
                let steal_si = steal_idx_for_node[n_idx];
                let ext_lp = steal_si.map(|si| unsafe { &*steal_slots[si].lp.get() });
                // P1 * P2 of the level's last merges is put off only where a merge has two workers
                // or more: with a worker a merge, the products put off would run a worker each
                // again and nothing would be gained.
                let defer_p = steal_si.is_none()
                    && !is_rightmost
                    && 2 * conc_merges <= active_w
                    && sample_bits >= 5_000_000
                    && step + conc_merges >= nodes_at_lv
                    && resume.is_none()
                    && ctx.fault_hook.is_none();
                let node_pw = node_pws_ref.get(n_idx).copied().unwrap_or(prod_workers);
                let merge_out = merge_pair_inner(
                    ctx,
                    plan,
                    shares,
                    cal,
                    lv,
                    idx_u64,
                    is_rightmost,
                    node_pw,
                    Some((&rem_merges_ctr, conc_merges)),
                    defer_p,
                    ext_lp,
                    left,
                    right,
                );
                if let Some(si) = steal_si {
                    release_steal_ref(&steal_slots[si]);
                }
                rem_merges_ctr.fetch_sub(1, Ordering::AcqRel);
                match merge_out {
                    Ok((tr, def_opt)) => unsafe {
                        *(next_ptr as *mut Option<BlockTriple>).add(n_idx) = Some(tr);
                        if let Some(dp) = def_opt {
                            *(def_ptr as *mut Option<(usize, DeferredProduct)>).add(n_idx) =
                                Some((n_idx, dp));
                        }
                    },
                    Err(e) => {
                        has_lv_err.store(true, Ordering::Relaxed);
                        let mut g = lv_err.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                        return;
                    }
                }
                for slot in &steal_slots {
                    if has_lv_err.load(Ordering::Relaxed) {
                        break;
                    }
                    if !slot.claimed.load(Ordering::Relaxed)
                        && slot
                            .claimed
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                            .is_ok()
                    {
                        let rem_m = rem_merges_ctr.load(Ordering::Acquire);
                        let steal_pw = if rem_m == 0 {
                            active_w
                        } else {
                            (active_w / (rem_m + 1).min(conc_merges).max(1)).max(node_pw)
                        };
                        let (lp_ref, rp_ref) = unsafe { (&*slot.lp.get(), &*slot.rp.get()) };
                        let pp_m = plan.select_mul_method(
                            lp_ref.body_bits(),
                            rp_ref.body_bits(),
                            false,
                            false,
                            steal_pw,
                            shares,
                            cal,
                        );
                        let res = ctx.mul_whole_immut(slot.pid, &pp_m, lp_ref, rp_ref);
                        release_steal_ref(slot);
                        match res {
                            Ok(p_bn) => unsafe {
                                *slot.out.get() = Some(p_bn);
                            },
                            Err(e) => {
                                has_lv_err.store(true, Ordering::Relaxed);
                                let mut g = lv_err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    }
                }
            });
            if let Some(e) = lv_err.into_inner().unwrap() {
                for slot in &steal_slots {
                    if !slot.claimed.load(Ordering::Acquire) {
                        release_steal_ref(slot);
                    }
                    if let Some(p_bn) = unsafe { (*slot.out.get()).take() } {
                        crate::limb::put_limb_buf(p_bn.into_limbs());
                    }
                }
                return Err(e);
            }
            let mut defs: Vec<(usize, DeferredProduct)> =
                deferred_ps.into_iter().flatten().collect();
            for slot in steal_slots {
                if slot.claimed.load(Ordering::Acquire) {
                    if let Some(p_bn) = unsafe { (*slot.out.get()).take() }
                        && let Some(ref mut tr) = next_nodes[slot.n_idx]
                    {
                        tr.p = p_bn;
                    }
                } else {
                    debug_assert_eq!(slot.refs.load(Ordering::Acquire), 1);
                    let lp = slot.lp.into_inner();
                    let rp = slot.rp.into_inner();
                    defs.push((slot.n_idx, (slot.pid, lp, rp)));
                }
            }
            let mut def_off = 0usize;
            while def_off < defs.len() {
                let rem = defs.len() - def_off;
                // as many at once as are left and as the merges ran, by an even share of the workers
                let batch = if rem == 1 {
                    1usize
                } else if rem <= conc_merges && active_w >= rem.saturating_mul(2) {
                    rem
                } else {
                    conc_cands
                        .iter()
                        .rev()
                        .copied()
                        .find(|&c| c <= rem && c <= conc_merges)
                        .unwrap_or(1)
                };
                let batch_pw = (active_w / batch).max(1);
                if batch == 1 {
                    let (n_idx, (pid, ref mut lp, ref mut rp)) = defs[def_off];
                    let pp_m = plan.select_mul_method(
                        lp.body_bits(),
                        rp.body_bits(),
                        false,
                        false,
                        batch_pw,
                        shares,
                        cal,
                    );
                    let p_bn = ctx.mul_whole(pid, &pp_m, lp, rp)?;
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut lp.limbs));
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut rp.limbs));
                    if let Some(ref mut tr) = next_nodes[n_idx] {
                        tr.p = p_bn;
                    }
                } else {
                    let slice_ptr = unsafe { defs.as_mut_ptr().add(def_off) } as usize;
                    let def_err: Mutex<Option<MulError>> = Mutex::new(None);
                    let has_def_err = AtomicBool::new(false);
                    ctx.sched.par_for_max(batch, 1, batch, |bi| {
                        if has_def_err.load(Ordering::Relaxed) {
                            return;
                        }
                        let bi_pw = (active_w / batch + usize::from(bi < (active_w % batch))).max(1);
                        let (n_idx, (pid, lp, rp)) = unsafe {
                            &mut *(slice_ptr as *mut (usize, DeferredProduct)).add(bi)
                        };
                        let pp_m = plan.select_mul_method(
                            lp.body_bits(),
                            rp.body_bits(),
                            false,
                            false,
                            bi_pw,
                            shares,
                            cal,
                        );
                        match ctx.mul_whole(*pid, &pp_m, lp, rp) {
                            Ok(p_bn) => {
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut lp.limbs));
                                crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut rp.limbs));
                                unsafe {
                                    if let Some(ref mut tr) =
                                        *(next_ptr as *mut Option<BlockTriple>).add(*n_idx)
                                    {
                                        tr.p = p_bn;
                                    }
                                }
                            }
                            Err(e) => {
                                has_def_err.store(true, Ordering::Relaxed);
                                let mut g = def_err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                            }
                        }
                    });
                    if let Some(e) = def_err.into_inner().unwrap() {
                        return Err(e);
                    }
                }
                def_off += batch;
            }
        } else if plan.out_of_core
            && ctx.store.is_some()
            && nodes_at_lv >= 2
            && (cur_parked.iter().any(Option::is_some)
                || crate::arena::active_numbers_live_bytes().saturating_mul(2)
                    > plan.ram_numbers_budget)
        {
            // Section 6 & 10: keep inactive `BlockTriple`s parked in `Store` and move them on
            // `supi-loader` (writing the previous parent and prefetching the next child pair)
            // while workers compute `merge_pair` on the active pair!
            let st = ctx.store.unwrap();
            let mut active_indices: Vec<usize> = Vec::with_capacity(nodes_at_lv);
            for n_idx in 0..nodes_at_lv {
                let idx_u64 = n_idx as u64;
                if let Some(res_st) = resume {
                    if let Some(saved) = res_st.nodes.get(&(lv, idx_u64)) {
                        next_nodes[n_idx] = Some(saved.clone());
                        continue;
                    }
                }
                if is_covered_by_ancestor(lv, idx_u64) {
                    continue;
                }
                active_indices.push(n_idx);
            }

            if active_indices.len() == 1 {
                let n_idx = active_indices[0];
                let pid_base = 0x1800_0000u64 | ((lv as u64) << 40) | ((n_idx as u64) << 2);
                let left = take_or_unpark_triple(
                    ctx,
                    pid_base,
                    2 * n_idx,
                    &mut cur_nodes,
                    &mut cur_parked,
                    prod_workers,
                )?;
                let right = take_or_unpark_triple(
                    ctx,
                    pid_base + 1,
                    2 * n_idx + 1,
                    &mut cur_nodes,
                    &mut cur_parked,
                    prod_workers,
                )?;
                next_nodes[n_idx] = Some(merge_pair(
                    ctx,
                    plan,
                    shares,
                    cal,
                    lv,
                    n_idx as u64,
                    n_idx + 1 == nodes_at_lv,
                    prod_workers,
                    left,
                    right,
                )?);
            } else if active_indices.len() >= 2 {
                let m0 = active_indices[0];
                let pid0 = 0x1800_0000u64 | ((lv as u64) << 40) | ((m0 as u64) << 2);
                let mut cur_left_opt = Some(take_or_unpark_triple(
                    ctx,
                    pid0,
                    2 * m0,
                    &mut cur_nodes,
                    &mut cur_parked,
                    prod_workers,
                )?);
                let mut cur_right_opt = Some(take_or_unpark_triple(
                    ctx,
                    pid0 + 1,
                    2 * m0 + 1,
                    &mut cur_nodes,
                    &mut cur_parked,
                    prod_workers,
                )?);

                let mut pending_writes: Vec<crate::mul::PreparedWriteJob> = Vec::new();
                for &m in &active_indices[1..] {
                    let pid_m = 0x1800_0000u64 | ((lv as u64) << 40) | ((m as u64) << 2);
                    for child_i in [2 * m, 2 * m + 1] {
                        if let Some(tr) = cur_nodes[child_i].take() {
                            let (pt, mut wjobs) =
                                prepare_park_triple(ctx, pid_m, tr, prod_workers)?;
                            cur_parked[child_i] = Some(pt);
                            pending_writes.append(&mut wjobs);
                        }
                    }
                }

                for (step_i, &n_idx) in active_indices.iter().enumerate() {
                    let has_next = step_i + 1 < active_indices.len();
                    let next_reads = if has_next {
                        let m_next = active_indices[step_i + 1];
                        let pid_n = 0x1800_0000u64 | ((lv as u64) << 40) | ((m_next as u64) << 2);
                        let rl = match cur_parked[2 * m_next].as_ref() {
                            Some(pt) => Some(pt.prepare_read_triple(pid_n)?),
                            None => None,
                        };
                        let rr = match cur_parked[2 * m_next + 1].as_ref() {
                            Some(pt) => Some(pt.prepare_read_triple(pid_n + 1)?),
                            None => None,
                        };
                        Some((rl, rr))
                    } else {
                        None
                    };
                    let writes_now = std::mem::take(&mut pending_writes);
                    let loader_h = (!writes_now.is_empty() || next_reads.is_some()).then(|| {
                        st.submit_loader_scoped(move || {
                            for wj in writes_now {
                                wj.execute(st)?;
                            }
                            if let Some((mut rl_opt, mut rr_opt)) = next_reads {
                                if let Some(ref mut arr) = rl_opt {
                                    for s in arr.iter_mut().flatten() {
                                        s.execute_read(st, ctx.telemetry)?;
                                    }
                                }
                                if let Some(ref mut arr) = rr_opt {
                                    for s in arr.iter_mut().flatten() {
                                        s.execute_read(st, ctx.telemetry)?;
                                    }
                                }
                                Ok::<_, crate::store::StoreError>(Some((rl_opt, rr_opt)))
                            } else {
                                Ok(None)
                            }
                        })
                    });

                    let merge_res = merge_pair(
                        ctx,
                        plan,
                        shares,
                        cal,
                        lv,
                        n_idx as u64,
                        n_idx + 1 == nodes_at_lv,
                        prod_workers,
                        cur_left_opt.take().unwrap(),
                        cur_right_opt.take().unwrap(),
                    );

                    let pid_io = 0x1800_0000u64 | ((lv as u64) << 40) | ((n_idx as u64) << 2);
                    let prefetched = if let Some(h) = loader_h {
                        match h.wait() {
                            Ok(v) => v,
                            Err(_) => return Err(MulError::NeedRollback(pid_io)),
                        }
                    } else {
                        None
                    };
                    let merged = merge_res?;

                    if has_next {
                        let m_next = active_indices[step_i + 1];
                        let pid_n = 0x1800_0000u64 | ((lv as u64) << 40) | ((m_next as u64) << 2);
                        let (rl_opt, rr_opt) = prefetched.unwrap_or((None, None));
                        cur_left_opt = Some(if let Some(pt) = cur_parked[2 * m_next].take() {
                            pt.finish_unpark_triple(
                                ctx,
                                rl_opt.unwrap(),
                                pid_n,
                                prod_workers,
                            )?
                        } else {
                            cur_nodes[2 * m_next].take().expect("left child present")
                        });
                        cur_right_opt = Some(if let Some(pt) = cur_parked[2 * m_next + 1].take() {
                            pt.finish_unpark_triple(
                                ctx,
                                rr_opt.unwrap(),
                                pid_n + 1,
                                prod_workers,
                            )?
                        } else {
                            cur_nodes[2 * m_next + 1]
                                .take()
                                .expect("right child present")
                        });
                    }

                    // Keep `(0, 1)` in RAM ready for `lv + 1`'s first merge, and park `n_idx >= 2`
                    // into `Store` beside the next merge whenever `has_next`!
                    if n_idx >= 2 && has_next {
                        let (pt, mut wjobs) =
                            prepare_park_triple(ctx, pid_io + 2, merged, prod_workers)?;
                        next_parked[n_idx] = Some(pt);
                        pending_writes.append(&mut wjobs);
                    } else {
                        next_nodes[n_idx] = Some(merged);
                    }
                }
            }
        } else {
            unpark_all_triples(ctx, lv, &mut cur_nodes, &mut cur_parked, prod_workers)?;
            for n_idx in 0..nodes_at_lv {
                let idx_u64 = n_idx as u64;
                if let Some(st) = resume {
                    if let Some(saved) = st.nodes.get(&(lv, idx_u64)) {
                        next_nodes[n_idx] = Some(saved.clone());
                        continue;
                    }
                }
                if is_covered_by_ancestor(lv, idx_u64) {
                    continue;
                }
                let left = cur_nodes[2 * n_idx].take().expect("left child present");
                let right = cur_nodes[2 * n_idx + 1]
                    .take()
                    .expect("right child present");
                let is_rightmost = n_idx + 1 == nodes_at_lv;
                next_nodes[n_idx] = Some(merge_pair(
                    ctx,
                    plan,
                    shares,
                    cal,
                    lv,
                    idx_u64,
                    is_rightmost,
                    prod_workers,
                    left,
                    right,
                )?);
            }
        }
        cur_nodes = next_nodes;
        cur_parked = next_parked;
        if do_trace {
            let trace_m = plan.select_mul_method(
                sample_bits,
                sample_bits,
                true,
                false,
                prod_workers,
                shares,
                cal,
            );
            eprintln!(
                "[TRACE SERIES] merge lv={lv}/{h} nodes={nodes_at_lv} conc={conc_merges} bits={sample_bits} pw={prod_workers} method={trace_m:?} -> {:.2}ms",
                t_lv0.elapsed().as_secs_f64() * 1e3
            );
            crate::ntt::dump_and_reset_ntt_trace(&format!("lv{lv}"));
            crate::refk::dump_and_reset_kernel_times(
                &format!("lv{lv}"),
                t_lv0.elapsed().as_secs_f64(),
                shares.active_workers,
            );
            crate::par::dump_and_reset_par_times(
                &format!("lv{lv}"),
                t_lv0.elapsed().as_secs_f64(),
                shares.active_workers,
            );
            crate::arena::trace_active_arenas(&format!("lv{lv}"));
        }

        // If `lv` is a dividing level, build `F_lv` and its inverse once and divide every live node
        // at level `lv` (that was not already loaded post-division from `resume`)!
        if let Some(div_plan) = plan.dividing_levels.iter().find(|d| d.level == lv) {
            unpark_all_triples(ctx, lv, &mut cur_nodes, &mut cur_parked, active_w)?;
            let t_dv0 = Instant::now();
            let max_num_limbs = cur_nodes
                .iter()
                .flatten()
                .map(|tr| tr.t.limbs.len().max(tr.q.limbs.len()).max(tr.p.limbs.len()))
                .max()
                .unwrap_or(0);
            let max_q_limbs = cur_nodes
                .iter()
                .flatten()
                .map(|tr| tr.q.limbs.len())
                .max()
                .unwrap_or(0);
            let max_p_limbs = cur_nodes
                .iter()
                .flatten()
                .map(|tr| tr.p.limbs.len())
                .max()
                .unwrap_or(0);
            if max_num_limbs > 0 {
                let est_q_bits = (max_num_limbs as u64)
                    .saturating_mul(64)
                    .saturating_sub(div_plan.est_factor_bits)
                    .max(64);
                let cand_div_pw = if nodes_at_lv >= active_w {
                    1
                } else {
                    (active_w / nodes_at_lv.max(1)).max(1)
                };
                let choose_div_conc = |comp_bits: u64| -> usize {
                    let div_reuse = if div_plan.form == DivForm::SingleProduct {
                        nodes_at_lv as u64
                    } else {
                        1
                    };
                    let est_div_m = plan.select_mul_method_amortized(
                        comp_bits,
                        comp_bits,
                        false,
                        false,
                        cand_div_pw,
                        shares,
                        cal,
                        div_reuse,
                    );
                    let div_is_2d = div_plan.form == DivForm::SingleProduct
                        && matches!(est_div_m, MulMethod::Ntt(s) if s.rows > 1);
                    if !matches!(est_div_m, MulMethod::Ntt(s) if s.out_of_core) {
                        if !div_is_2d {
                            nodes_at_lv.min(active_w).max(1)
                        } else if let MulMethod::Ntt(s) = est_div_m {
                            let mut chosen = 1usize;
                            let mut chosen_l3 = false;
                            let mut chosen_up_l3 = false;
                            // As for the merges: the count that makes a quotient cheapest in worker
                            // time, more at once only by a clear margin.
                            let mut best_div_cost = f64::INFINITY;
                            for &c in &conc_cands {
                                if c < 1
                                    || c > nodes_at_lv
                                    || c > active_w
                                    || !nodes_at_lv.is_multiple_of(c)
                                {
                                    continue;
                                }
                                let pw_c = (active_w / c).max(1);
                                let sc = if pw_c == cand_div_pw {
                                    s
                                } else if let MulMethod::Ntt(sf) = plan.select_mul_method_amortized(
                                    comp_bits,
                                    comp_bits,
                                    false,
                                    false,
                                    pw_c,
                                    shares,
                                    cal,
                                    div_reuse,
                                ) {
                                    sf
                                } else {
                                    continue;
                                };
                                if sc.out_of_core || pw_c > sc.rows.max(1) {
                                    continue;
                                }
                                let grid_1 = (sc.k_primes as u64)
                                    .saturating_mul(sc.rows as u64)
                                    .saturating_mul(sc.outer_row_bytes() as u64)
                                    .max(1);
                                let div_room = if c > 1 && !roomy_run {
                                    plan.ram_transform_budget / 2
                                } else {
                                    plan.ram_transform_budget
                                };
                                if (c as u64).saturating_mul(grid_1.saturating_mul(2)) > div_room {
                                    continue;
                                }
                                let cost = plan.product_worker_sec_amortized(
                                    comp_bits, comp_bits, false, pw_c, shares, cal, div_reuse,
                                );
                                let raw_io_b = comp_bits.div_ceil(8).saturating_mul(2);
                                let shared_per_node = grid_1
                                    .saturating_add(raw_io_b)
                                    .saturating_add(grid_1 / (c as u64).max(1));
                                let shared_l3_fits = div_plan.form == DivForm::SingleProduct
                                    && if c == 1 {
                                        let pbp_grid = grid_1 / (sc.k_primes.max(1) as u64);
                                        shares.fits_l3_domains(
                                            1,
                                            grid_1
                                                .saturating_add(pbp_grid.saturating_mul(2))
                                                .saturating_add(raw_io_b),
                                            sc.rows,
                                            9,
                                            10,
                                        )
                                    } else {
                                        sc.rows <= 128
                                            && sc.rows >= pw_c * 2
                                            && shares.fits_l3_domains(
                                                c as u64,
                                                shared_per_node,
                                                sc.rows,
                                                8,
                                                10,
                                            )
                                            && ((c as u64 + 1).saturating_mul(grid_1)
                                                <= plan.ram_transform_budget / 2)
                                    };
                                let up_rb = if sc.cols >= 512 {
                                    sc.cols * 8 + 512
                                } else {
                                    sc.cols * 8
                                };
                                let up_grid = (sc.k_primes as u64)
                                    .saturating_mul(sc.rows as u64)
                                    .saturating_mul(up_rb as u64);
                                let up_shared =
                                    up_grid.saturating_add(up_grid / (c as u64).max(1));
                                let (up_num, up_den) = if c <= 2 { (9, 10) } else { (7, 8) };
                                let up_l3_fits = div_plan.form == DivForm::SingleProduct
                                    && !sc.out_of_core
                                    && sc.rows > 1
                                    && sc.extra_sweeps == 0
                                    && shares.total_l3 > 0
                                    && shares.fits_l3_domains(
                                        c as u64,
                                        up_shared,
                                        sc.rows,
                                        up_num,
                                        up_den,
                                    );
                                let margin = if chosen_up_l3 && !up_l3_fits {
                                    0.88
                                } else if c >= 2 && (shared_l3_fits || up_l3_fits) {
                                    1.35
                                } else if !chosen_l3
                                    && c >= 2
                                    && c <= (active_w / 4).max(1)
                                    && sc.cols >= 4096
                                    && sc.rows <= active_w * 64
                                    && sc.rows >= pw_c * 4
                                {
                                    if sc.rows <= active_w * 4 {
                                        1.28
                                    } else if sc.rows < active_w * 16 {
                                        1.18
                                    } else {
                                        1.10
                                    }
                                } else {
                                    crate::plan::MORE_AT_ONCE_MARGIN
                                };
                                if cost < best_div_cost * margin {
                                    best_div_cost = cost.min(best_div_cost);
                                    chosen = c;
                                    chosen_l3 = shared_l3_fits;
                                    chosen_up_l3 = up_l3_fits;
                                }
                            }
                            chosen
                        } else {
                            1
                        }
                    } else if nodes_at_lv >= active_w && !div_is_2d {
                        nodes_at_lv.min(active_w).max(1)
                    } else if nodes_at_lv >= 2
                        && (div_plan.form == DivForm::SequentialHensel
                            || !matches!(est_div_m, MulMethod::Ntt(_)))
                    {
                        nodes_at_lv.min(active_w).max(1)
                    } else {
                        1
                    }
                };
                let conc_divs = choose_div_conc(est_q_bits);
                let div_par_workers = if conc_divs >= 2 {
                    (active_w / conc_divs).max(1)
                } else {
                    active_w
                };
                let est_qq_bits = (max_q_limbs as u64)
                    .saturating_mul(64)
                    .saturating_sub(div_plan.est_factor_bits)
                    .max(64);
                let req_conc_q = if max_q_limbs > 0 {
                    choose_div_conc(est_qq_bits)
                } else {
                    conc_divs
                };
                let req_pw_q = if req_conc_q >= 2 {
                    (active_w / req_conc_q).max(1)
                } else {
                    active_w
                };
                let est_pp_bits = (max_p_limbs as u64)
                    .saturating_mul(64)
                    .saturating_sub(div_plan.est_factor_bits)
                    .max(64);
                let req_conc_p = if max_p_limbs > 0 {
                    choose_div_conc(est_pp_bits)
                } else {
                    conc_divs
                };
                let req_pw_p = if req_conc_p >= 2 {
                    (active_w / req_conc_p).max(1)
                } else {
                    active_w
                };
                let mut divisor = prepare_level_divisor_ext(
                    ctx,
                    plan,
                    shares,
                    cal,
                    &primes,
                    div_plan,
                    max_num_limbs,
                    max_q_limbs,
                    max_p_limbs,
                    div_par_workers,
                    req_pw_q,
                    req_pw_p,
                )?;
                let eff_conc_q = if divisor.q_inv_method.is_some() {
                    req_conc_q
                } else {
                    conc_divs
                };
                let eff_pw_q = if eff_conc_q >= 2 {
                    (active_w / eff_conc_q).max(1)
                } else {
                    active_w
                };
                let p_reuses_q = divisor.p_inv_method.is_none()
                    && divisor.q_inv_method.is_some()
                    && divisor.p_max_quot_limbs > 0
                    && divisor.p_max_quot_limbs <= divisor.q_max_quot_limbs;
                let eff_conc_p = if divisor.p_inv_method.is_some() {
                    req_conc_p
                } else if p_reuses_q {
                    eff_conc_q
                } else {
                    conc_divs
                };
                let eff_pw_p = if eff_conc_p >= 2 {
                    (active_w / eff_conc_p).max(1)
                } else {
                    active_w
                };
                let prep_ms = t_dv0.elapsed().as_secs_f64() * 1e3;
                let t_dn0 = Instant::now();
                let par_divs = conc_divs >= 2;

                let div_err: Mutex<Option<MulError>> = Mutex::new(None);
                let has_div_err = AtomicBool::new(false);
                let cur_div_ptr = cur_nodes.as_mut_ptr() as usize;

                if ctx.fault_hook.is_some() {
                    let divide_one_node = |n_idx: usize| {
                        let idx_u64 = n_idx as u64;
                        if resume.is_some_and(|st| st.nodes.contains_key(&(lv, idx_u64))) {
                            return;
                        }
                        if has_div_err.load(Ordering::Relaxed) {
                            return;
                        }
                        let slot =
                            unsafe { &mut *(cur_div_ptr as *mut Option<BlockTriple>).add(n_idx) };
                        if let Some(tr) = slot {
                            let pid_base = 0x2000_0000u64 | ((lv as u64) << 40) | (idx_u64 << 2);
                            let is_rightmost = n_idx + 1 == nodes_at_lv;
                            let dq = divide_bignum_exact(
                                ctx,
                                plan,
                                shares,
                                cal,
                                div_par_workers,
                                pid_base,
                                &mut tr.q,
                                &divisor,
                            );
                            let dt = divide_bignum_exact(
                                ctx,
                                plan,
                                shares,
                                cal,
                                div_par_workers,
                                pid_base + 1,
                                &mut tr.t,
                                &divisor,
                            );
                            let dp = if is_rightmost {
                                Ok(BigNum::zero())
                            } else {
                                divide_bignum_exact(
                                    ctx,
                                    plan,
                                    shares,
                                    cal,
                                    div_par_workers,
                                    pid_base + 2,
                                    &mut tr.p,
                                    &divisor,
                                )
                            };
                            match (dp, dq, dt) {
                                (Ok(p), Ok(q), Ok(t)) => {
                                    *tr = BlockTriple { p, q, t };
                                }
                                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                                    has_div_err.store(true, Ordering::Relaxed);
                                    let mut g = div_err.lock().unwrap();
                                    if g.is_none() {
                                        *g = Some(e);
                                    }
                                }
                            }
                        }
                    };
                    if par_divs {
                        ctx.sched
                            .par_for_max(nodes_at_lv, 1, conc_divs, divide_one_node);
                    } else {
                        for n_idx in 0..nodes_at_lv {
                            divide_one_node(n_idx);
                            if has_div_err.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    }
                } else {
                    // Execute divisions component-by-component (`Q`, `P`, `T`) across both parallel
                    // and sequential modes so each cached inverse stays pinned in shared L3!
                    {
                        let div_ref = &divisor;
                        let div_q_one = |n_idx: usize| {
                            let idx_u64 = n_idx as u64;
                            if resume.is_some_and(|st| st.nodes.contains_key(&(lv, idx_u64)))
                                || has_div_err.load(Ordering::Relaxed)
                            {
                                return;
                            }
                            let slot = unsafe {
                                &mut *(cur_div_ptr as *mut Option<BlockTriple>).add(n_idx)
                            };
                            if let Some(tr) = slot {
                                let pid_base =
                                    0x2000_0000u64 | ((lv as u64) << 40) | (idx_u64 << 2);
                                match divide_bignum_exact(
                                    ctx,
                                    plan,
                                    shares,
                                    cal,
                                    eff_pw_q,
                                    pid_base,
                                    &mut tr.q,
                                    div_ref,
                                ) {
                                    Ok(q) => tr.q = q,
                                    Err(e) => {
                                        has_div_err.store(true, Ordering::Relaxed);
                                        let mut g = div_err.lock().unwrap();
                                        if g.is_none() {
                                            *g = Some(e);
                                        }
                                    }
                                }
                            }
                        };
                        if eff_conc_q >= 2 {
                            ctx.sched
                                .par_for_max(nodes_at_lv, 1, eff_conc_q, |step| {
                                    div_q_one(nodes_at_lv - 1 - step);
                                });
                        } else {
                            for n_idx in 0..nodes_at_lv {
                                div_q_one(n_idx);
                                if has_div_err.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    }
                    if !p_reuses_q {
                        if let Some(c) = &mut divisor.q_inv_cached {
                            c.release_store(ctx.store);
                        }
                        divisor.q_inv_cached = None;
                        divisor.q_inv_method = None;
                        divisor.q_max_quot_limbs = 0;
                        crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut divisor.q_inv_limbs));
                    }

                    if !has_div_err.load(Ordering::Relaxed) {
                        let div_ref = &divisor;
                        let div_p_one = |n_idx: usize| {
                            let idx_u64 = n_idx as u64;
                            if resume.is_some_and(|st| st.nodes.contains_key(&(lv, idx_u64)))
                                || has_div_err.load(Ordering::Relaxed)
                            {
                                return;
                            }
                            let slot = unsafe {
                                &mut *(cur_div_ptr as *mut Option<BlockTriple>).add(n_idx)
                            };
                            if let Some(tr) = slot {
                                let pid_base =
                                    0x2000_0000u64 | ((lv as u64) << 40) | (idx_u64 << 2);
                                let is_rightmost = n_idx + 1 == nodes_at_lv;
                                if is_rightmost {
                                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut tr.p.limbs));
                                    tr.p = BigNum::zero();
                                } else {
                                    match divide_bignum_exact(
                                        ctx,
                                        plan,
                                        shares,
                                        cal,
                                        eff_pw_p,
                                        pid_base + 2,
                                        &mut tr.p,
                                        div_ref,
                                    ) {
                                        Ok(p) => tr.p = p,
                                        Err(e) => {
                                            has_div_err.store(true, Ordering::Relaxed);
                                            let mut g = div_err.lock().unwrap();
                                            if g.is_none() {
                                                *g = Some(e);
                                            }
                                        }
                                    }
                                }
                            }
                        };
                        if eff_conc_p >= 2 && nodes_at_lv > 1 {
                            div_p_one(nodes_at_lv - 1);
                            ctx.sched
                                .par_for_max(nodes_at_lv - 1, 1, eff_conc_p, |step| {
                                    div_p_one(nodes_at_lv - 2 - step);
                                });
                        } else {
                            for n_idx in 0..nodes_at_lv {
                                div_p_one(n_idx);
                                if has_div_err.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(c) = &mut divisor.q_inv_cached {
                        c.release_store(ctx.store);
                    }
                    divisor.q_inv_cached = None;
                    divisor.q_inv_method = None;
                    divisor.q_max_quot_limbs = 0;
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut divisor.q_inv_limbs));
                    if let Some(c) = &mut divisor.p_inv_cached {
                        c.release_store(ctx.store);
                    }
                    divisor.p_inv_cached = None;
                    divisor.p_inv_method = None;
                    divisor.p_max_quot_limbs = 0;
                    crate::limb::put_limb_buf(crate::limb::take_limb_vec(&mut divisor.p_inv_limbs));

                    if !has_div_err.load(Ordering::Relaxed) {
                        let div_ref = &divisor;
                        let div_t_one = |n_idx: usize| {
                            let idx_u64 = n_idx as u64;
                            if resume.is_some_and(|st| st.nodes.contains_key(&(lv, idx_u64)))
                                || has_div_err.load(Ordering::Relaxed)
                            {
                                return;
                            }
                            let slot = unsafe {
                                &mut *(cur_div_ptr as *mut Option<BlockTriple>).add(n_idx)
                            };
                            if let Some(tr) = slot {
                                let pid_base =
                                    0x2000_0000u64 | ((lv as u64) << 40) | (idx_u64 << 2);
                                match divide_bignum_exact(
                                    ctx,
                                    plan,
                                    shares,
                                    cal,
                                    div_par_workers,
                                    pid_base + 1,
                                    &mut tr.t,
                                    div_ref,
                                ) {
                                    Ok(t) => tr.t = t,
                                    Err(e) => {
                                        has_div_err.store(true, Ordering::Relaxed);
                                        let mut g = div_err.lock().unwrap();
                                        if g.is_none() {
                                            *g = Some(e);
                                        }
                                    }
                                }
                            }
                        };
                        if par_divs {
                            ctx.sched
                                .par_for_max(nodes_at_lv, 1, conc_divs, |step| {
                                    div_t_one(nodes_at_lv - 1 - step);
                                });
                        } else {
                            for n_idx in 0..nodes_at_lv {
                                div_t_one(n_idx);
                                if has_div_err.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    }
                }
                divisor.release_store(ctx.store);
                if do_trace {
                    eprintln!(
                        "[TRACE SERIES] div lv={lv} form={:?} conc={conc_divs} pw={div_par_workers} prep={prep_ms:.2}ms nodes={:.2}ms -> {:.2}ms",
                        div_plan.form,
                        t_dn0.elapsed().as_secs_f64() * 1e3,
                        t_dv0.elapsed().as_secs_f64() * 1e3
                    );
                }
                if let Some(e) = div_err.into_inner().unwrap() {
                    return Err(e);
                }
            }
        }

        // Offer a checkpoint after completing level `lv` (Section 11)!
        if let Some(sink) = ckpt_sink {
            let elapsed = last_ckpt_instant.elapsed().as_secs_f64();
            if plan.checkpoint_mult <= 0.0
                || elapsed >= plan.checkpoint_mult * plan.est_checkpoint_write_sec
            {
                unpark_all_triples(ctx, lv, &mut cur_nodes, &mut cur_parked, active_w)?;
                let mut live_refs = Vec::with_capacity(cur_nodes.len());
                for (n_idx, slot) in cur_nodes.iter().enumerate() {
                    if let Some(tr) = slot {
                        live_refs.push((lv, n_idx as u64, tr));
                    }
                }
                if sink.maybe_checkpoint_series(lv, &live_refs, elapsed)? {
                    last_ckpt_instant = Instant::now();
                }
            }
        }
    }

    let _ = mul_bignum_u64;
    let _ = shl_limbs;
    unpark_all_triples(ctx, h, &mut cur_nodes, &mut cur_parked, active_w)?;
    let root = cur_nodes[0].take().expect("root triple computed");
    Ok((root.q, root.t))
}

enum MaybeParkedNum<'a> {
    Ram(BigNum),
    Parked(crate::mul::ParkedBigNum<'a>),
}

impl<'a> MaybeParkedNum<'a> {
    #[inline]
    fn body_bits(&self) -> u64 {
        match self {
            Self::Ram(bn) => bn.body_bits(),
            Self::Parked(pb) => pb.body_bits(),
        }
    }

    #[inline]
    fn shift(&self) -> u64 {
        match self {
            Self::Ram(bn) => bn.shift,
            Self::Parked(pb) => pb.shift,
        }
    }
}

struct ParkedTriple<'a> {
    p: MaybeParkedNum<'a>,
    q: MaybeParkedNum<'a>,
    t: MaybeParkedNum<'a>,
}

impl<'a> ParkedTriple<'a> {
    #[inline]
    fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.q.body_bits(),
            self.q.shift(),
            self.t.body_bits(),
            self.p.body_bits(),
        )
    }

    fn prepare_read_triple(
        &self,
        point_id: u64,
    ) -> Result<[Option<crate::mul::ParkedReadStage>; 3], MulError> {
        let rp = match &self.p {
            MaybeParkedNum::Parked(pb) => Some(pb.prepare_read(point_id)?),
            MaybeParkedNum::Ram(_) => None,
        };
        let rq = match &self.q {
            MaybeParkedNum::Parked(pb) => Some(pb.prepare_read(point_id + 1)?),
            MaybeParkedNum::Ram(_) => None,
        };
        let rt = match &self.t {
            MaybeParkedNum::Parked(pb) => Some(pb.prepare_read(point_id + 2)?),
            MaybeParkedNum::Ram(_) => None,
        };
        Ok([rp, rq, rt])
    }

    fn finish_unpark_triple(
        self,
        ctx: &MulCtx<'_>,
        stages: [Option<crate::mul::ParkedReadStage>; 3],
        point_id: u64,
        max_workers: usize,
    ) -> Result<BlockTriple, MulError> {
        let [sp, sq, st] = stages;
        let p = match self.p {
            MaybeParkedNum::Ram(bn) => bn,
            MaybeParkedNum::Parked(pb) => {
                pb.finish_unpark_bignum(ctx, sp.unwrap(), point_id, max_workers)?
            }
        };
        let q = match self.q {
            MaybeParkedNum::Ram(bn) => bn,
            MaybeParkedNum::Parked(pb) => {
                pb.finish_unpark_bignum(ctx, sq.unwrap(), point_id + 1, max_workers)?
            }
        };
        let t = match self.t {
            MaybeParkedNum::Ram(bn) => bn,
            MaybeParkedNum::Parked(pb) => {
                pb.finish_unpark_bignum(ctx, st.unwrap(), point_id + 2, max_workers)?
            }
        };
        Ok(BlockTriple { p, q, t })
    }

    fn unpark_triple_sync(
        self,
        ctx: &MulCtx<'_>,
        point_id: u64,
        max_workers: usize,
    ) -> Result<BlockTriple, MulError> {
        let mut stages = self.prepare_read_triple(point_id)?;
        if let Some(st) = ctx.store
            && stages.iter().any(Option::is_some)
        {
            let stages_ref = &mut stages;
            st.submit_loader_scoped(|| -> Result<(), crate::store::StoreError> {
                for s in stages_ref.iter_mut().flatten() {
                    s.execute_read(st, ctx.telemetry)?;
                }
                Ok(())
            })
            .wait()
            .map_err(|_| MulError::NeedRollback(point_id))?;
        }
        self.finish_unpark_triple(ctx, stages, point_id, max_workers)
    }
}

fn prepare_park_triple<'a>(
    ctx: &MulCtx<'a>,
    point_id: u64,
    mut tr: BlockTriple,
    max_workers: usize,
) -> Result<(ParkedTriple<'a>, Vec<crate::mul::PreparedWriteJob>), MulError> {
    let mut jobs = Vec::with_capacity(3);
    let p = match crate::mul::PreparedParkWrite::prepare_bignum(
        ctx,
        point_id,
        &mut tr.p,
        max_workers,
    )? {
        Some(prep) => {
            let (pb, wj) = prep.split_write_job();
            jobs.push(wj);
            MaybeParkedNum::Parked(pb)
        }
        None => MaybeParkedNum::Ram(tr.p),
    };
    let q = match crate::mul::PreparedParkWrite::prepare_bignum(
        ctx,
        point_id + 1,
        &mut tr.q,
        max_workers,
    )? {
        Some(prep) => {
            let (pb, wj) = prep.split_write_job();
            jobs.push(wj);
            MaybeParkedNum::Parked(pb)
        }
        None => MaybeParkedNum::Ram(tr.q),
    };
    let t = match crate::mul::PreparedParkWrite::prepare_bignum(
        ctx,
        point_id + 2,
        &mut tr.t,
        max_workers,
    )? {
        Some(prep) => {
            let (pb, wj) = prep.split_write_job();
            jobs.push(wj);
            MaybeParkedNum::Parked(pb)
        }
        None => MaybeParkedNum::Ram(tr.t),
    };
    Ok((ParkedTriple { p, q, t }, jobs))
}

fn take_or_unpark_triple<'a>(
    ctx: &MulCtx<'a>,
    point_id: u64,
    slot_idx: usize,
    cur_nodes: &mut [Option<BlockTriple>],
    cur_parked: &mut [Option<ParkedTriple<'a>>],
    max_workers: usize,
) -> Result<BlockTriple, MulError> {
    if let Some(tr) = cur_nodes[slot_idx].take() {
        return Ok(tr);
    }
    let pt = cur_parked[slot_idx]
        .take()
        .expect("series node triple present in RAM or Store");
    pt.unpark_triple_sync(ctx, point_id, max_workers)
}

fn unpark_all_triples<'a>(
    ctx: &MulCtx<'a>,
    lv: u32,
    cur_nodes: &mut [Option<BlockTriple>],
    cur_parked: &mut [Option<ParkedTriple<'a>>],
    max_workers: usize,
) -> Result<(), MulError> {
    for idx in 0..cur_parked.len() {
        if let Some(pt) = cur_parked[idx].take() {
            let pid = 0x1900_0000u64 | ((lv as u64) << 40) | ((idx as u64) << 2);
            cur_nodes[idx] = Some(pt.unpark_triple_sync(ctx, pid, max_workers)?);
        }
    }
    Ok(())
}

static FR_LEAF_NS: AtomicU64 = AtomicU64::new(0);
static FR_LOW_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static FR_HIGH_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static FR_DIV_NS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn series_trace_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("SUPI_TRACE").is_some())
}

#[derive(Default)]
struct SubtreeTraceAcc {
    leaf_ns: u64,
    low_merge_ns: u64,
    high_merge_ns: u64,
    div_ns: u64,
}

#[allow(clippy::too_many_arguments)]
fn eval_tail_subtree_split(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    sub_divs: &[(u32, PreparedLevelDivisor)],
    level: u32,
    node_idx: u64,
    is_rightmost: bool,
    split_depth: u32,
) -> Result<BlockTriple, MulError> {
    if split_depth == 0 || level <= 2 {
        return eval_sequential_subtree(
            ctx,
            plan,
            shares,
            cal,
            sub_divs,
            level,
            node_idx,
            is_rightmost,
        );
    }
    let (left_res, right_res) = ctx.sched.join(
        || {
            eval_tail_subtree_split(
                ctx,
                plan,
                shares,
                cal,
                sub_divs,
                level - 1,
                2 * node_idx,
                false,
                split_depth - 1,
            )
        },
        || {
            eval_tail_subtree_split(
                ctx,
                plan,
                shares,
                cal,
                sub_divs,
                level - 1,
                2 * node_idx + 1,
                is_rightmost,
                split_depth - 1,
            )
        },
    );
    let left = left_res?;
    let right = right_res?;
    let t0 = series_trace_enabled().then(Instant::now);
    let mut tr = merge_pair(
        ctx,
        plan,
        shares,
        cal,
        level,
        node_idx,
        is_rightmost,
        1,
        left,
        right,
    )?;
    if let Some(t) = t0 {
        let dt = t.elapsed().as_nanos() as u64;
        if level <= 6 {
            FR_LOW_MERGE_NS.fetch_add(dt, Ordering::Relaxed);
        } else {
            FR_HIGH_MERGE_NS.fetch_add(dt, Ordering::Relaxed);
        }
    }
    if let Some((_, divisor)) = sub_divs.iter().find(|(dlv, _)| *dlv == level) {
        let td0 = series_trace_enabled().then(Instant::now);
        let pid_base = 0x2000_0000u64 | ((level as u64) << 40) | (node_idx << 2);
        tr.q = divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base, &mut tr.q, divisor)?;
        tr.t = divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base + 1, &mut tr.t, divisor)?;
        tr.p = if is_rightmost {
            BigNum::zero()
        } else {
            divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base + 2, &mut tr.p, divisor)?
        };
        if let Some(td) = td0 {
            FR_DIV_NS.fetch_add(td.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    Ok(tr)
}

#[allow(clippy::too_many_arguments)]
fn eval_sequential_subtree(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    sub_divs: &[(u32, PreparedLevelDivisor)],
    level: u32,
    node_idx: u64,
    is_rightmost: bool,
) -> Result<BlockTriple, MulError> {
    // Section 10: a worker running a subtree alone keeps its subtree numbers in its own LIFO
    // region (`SubtreeLifoScope`), promoting only the escaping frontier `BlockTriple` into the
    // plan-assigned `Numbers` region above that depth.
    let _subtree_lifo = SubtreeLifoScope::enter();
    let escapes_to_upper = level < plan.tree_height;
    if !series_trace_enabled() {
        return eval_sequential_subtree_inner(
            ctx,
            plan,
            shares,
            cal,
            sub_divs,
            level,
            node_idx,
            is_rightmost,
            escapes_to_upper,
            None,
        );
    }
    let mut acc = SubtreeTraceAcc::default();
    let res = eval_sequential_subtree_inner(
        ctx,
        plan,
        shares,
        cal,
        sub_divs,
        level,
        node_idx,
        is_rightmost,
        escapes_to_upper,
        Some(&mut acc),
    );
    FR_LEAF_NS.fetch_add(acc.leaf_ns, Ordering::Relaxed);
    FR_LOW_MERGE_NS.fetch_add(acc.low_merge_ns, Ordering::Relaxed);
    FR_HIGH_MERGE_NS.fetch_add(acc.high_merge_ns, Ordering::Relaxed);
    FR_DIV_NS.fetch_add(acc.div_ns, Ordering::Relaxed);
    res
}

#[allow(clippy::too_many_arguments)]
fn eval_sequential_subtree_inner(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    sub_divs: &[(u32, PreparedLevelDivisor)],
    level: u32,
    node_idx: u64,
    is_rightmost: bool,
    is_frontier_top: bool,
    mut trace: Option<&mut SubtreeTraceAcc>,
) -> Result<BlockTriple, MulError> {
    let has_div_here = sub_divs.iter().any(|(dlv, _)| *dlv == level);
    let promote_merge = is_frontier_top && !has_div_here;
    if level == 0 {
        let _upper = promote_merge.then(SubtreeLifoScope::pause);
        let t0 = trace.as_ref().map(|_| Instant::now());
        let start_k = node_idx * plan.leaf_len;
        let r = compute_leaf_plain(
            ctx,
            start_k,
            plan.leaf_len,
            !is_rightmost,
            cal.basecase_max_limbs,
        );
        if let (Some(acc), Some(t)) = (trace.as_deref_mut(), t0) {
            acc.leaf_ns += t.elapsed().as_nanos() as u64;
        }
        return Ok(r);
    }
    let mut tr = if level <= 6
        && ctx.fault_hook.is_none()
        && plan.overrides.force_merge_form.is_none()
        && plan.overrides.split_override == SplitFormOverride::Auto
        && !sub_divs.iter().any(|(dlv, _)| *dlv < level)
        && {
            let sub_terms = plan.leaf_len.saturating_mul(1u64 << level);
            let end_k = node_idx.saturating_add(1).saturating_mul(sub_terms).max(2);
            let bpt = 54u64 + 3 * (64 - (end_k - 1).leading_zeros()) as u64;
            let est_child_limbs = ((sub_terms / 2) * bpt).div_ceil(64) + 2;
            est_child_limbs <= ((cal.karatsuba_max_limbs as u64) * 3 / 2).max(160)
        }
    {
        let _upper = promote_merge.then(SubtreeLifoScope::pause);
        let t0 = trace.as_ref().map(|_| Instant::now());
        let sub_terms = plan.leaf_len.saturating_mul(1u64 << level);
        let start_k = node_idx * sub_terms;
        let r = compute_leaf_plain(
            ctx,
            start_k,
            sub_terms,
            !is_rightmost,
            cal.basecase_max_limbs,
        );
        if let (Some(acc), Some(t)) = (trace.as_deref_mut(), t0) {
            acc.leaf_ns += t.elapsed().as_nanos() as u64;
        }
        r
    } else {
        let left = eval_sequential_subtree_inner(
            ctx,
            plan,
            shares,
            cal,
            sub_divs,
            level - 1,
            2 * node_idx,
            false,
            false,
            trace.as_deref_mut(),
        )?;
        let right = eval_sequential_subtree_inner(
            ctx,
            plan,
            shares,
            cal,
            sub_divs,
            level - 1,
            2 * node_idx + 1,
            is_rightmost,
            false,
            trace.as_deref_mut(),
        )?;
        let _upper = promote_merge.then(SubtreeLifoScope::pause);
        let t0 = trace.as_ref().map(|_| Instant::now());
        let r = merge_pair(
            ctx,
            plan,
            shares,
            cal,
            level,
            node_idx,
            is_rightmost,
            1,
            left,
            right,
        )?;
        if let (Some(acc), Some(t)) = (trace.as_deref_mut(), t0) {
            let dt = t.elapsed().as_nanos() as u64;
            if level <= 6 {
                acc.low_merge_ns += dt;
            } else {
                acc.high_merge_ns += dt;
            }
        }
        r
    };

    if let Some((_, divisor)) = sub_divs.iter().find(|(dlv, _)| *dlv == level) {
        let _upper = is_frontier_top.then(SubtreeLifoScope::pause);
        let t0 = trace.as_ref().map(|_| Instant::now());
        let pid_base = 0x2000_0000u64 | ((level as u64) << 40) | (node_idx << 2);
        tr.q = divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base, &mut tr.q, divisor)?;
        tr.t = divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base + 1, &mut tr.t, divisor)?;
        tr.p = if is_rightmost {
            BigNum::zero()
        } else {
            divide_bignum_exact(ctx, plan, shares, cal, 1, pid_base + 2, &mut tr.p, divisor)?
        };
        if let (Some(acc), Some(t)) = (trace.as_deref_mut(), t0) {
            acc.div_ns += t.elapsed().as_nanos() as u64;
        }
    }

    Ok(tr)
}
