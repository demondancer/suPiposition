//! Contract:
//! - Inputs: `BigNum` operands (64-bit limbs least-significant first in the `Numbers` arena,
//!   sign, zero-bit shift count, and dual `w0` check word), limb cutoffs, and `Kernels` dispatch.
//! - Outputs: Exact arbitrary-precision integer sums, differences, bit shifts, basecase products/
//!   squares, Karatsuba products/squares, 2-adic odd modular inverses, and sequential Hensel exact
//!   quotients, each carrying its derived dual `w0` check word.
//! - Data format: Canonical trimmed `Vec<u64, NumbersAlloc>` limbs (`0` has empty limbs,
//!   `neg = false`, `shift = 0`); non-zero `Q` keeps odd limbs with trailing zero bits in `shift`.
//! - Work & bytes moved: Basecase O(n*m), Karatsuba O(n^1.585), sequential Hensel O(n_q * n_f).

use std::alloc::Allocator;
use std::cmp::Ordering;
use std::sync::atomic::Ordering as AtomicOrd;

use crate::arena::{NumbersAlloc, ScratchAlloc};
use crate::check::{DualCheck, Telemetry, compute_limbs_w0, whole_value_w0};
use crate::refk::{Kernels, mod61_add, mod61_mul, mod61_pow2, mod61_reduce, mod61_sub};

/// Ordinary `Vec<u64>` backed by the `Numbers` standard-allocator arena (Section 10).
pub type LimbVec = Vec<u64, NumbersAlloc>;

#[inline]
pub fn empty_limb_vec() -> LimbVec {
    Vec::new_in(NumbersAlloc)
}

#[inline]
pub fn take_limb_vec(v: &mut LimbVec) -> LimbVec {
    std::mem::replace(v, empty_limb_vec())
}

#[inline]
pub fn slice_to_limb_vec(s: &[u64]) -> LimbVec {
    if s.is_empty() {
        return empty_limb_vec();
    }
    let mut out = take_limb_buf_uninit(s.len());
    out.extend_from_slice(s);
    out
}

pub trait IntoLimbVec {
    fn into_limb_vec(self) -> LimbVec;
}

impl IntoLimbVec for LimbVec {
    #[inline]
    fn into_limb_vec(self) -> LimbVec {
        self
    }
}

impl IntoLimbVec for Vec<u64> {
    #[inline]
    fn into_limb_vec(self) -> LimbVec {
        slice_to_limb_vec(&self)
    }
}

/// Below this many limbs no buffer is given more room than it was taken for.
const ROOM_ASKED_MIN_LEN: usize = 8_192;

thread_local! {
    /// What a `RoomAsked` holds for this thread: the smallest buffer it speaks of and the limbs
    /// that buffer is to have room for; `(0, 0)` when nothing is asked.
    static ROOM_ASKED: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// A caller's word to the `Numbers` arena helper: the next buffer of `from` limbs or more that
/// this thread takes is to have room for `room` limbs so `add_bignum_into_par` can run in place.
pub struct RoomAsked {
    prev: (usize, usize),
    _this_thread: std::marker::PhantomData<*const ()>,
}

impl RoomAsked {
    pub fn new(from: usize, room: usize) -> Self {
        let asked = if room >= ROOM_ASKED_MIN_LEN && from >= 1 && from <= room {
            (from, room)
        } else {
            (0, 0)
        };
        let prev = ROOM_ASKED.try_with(|c| c.replace(asked)).unwrap_or((0, 0));
        Self {
            prev,
            _this_thread: std::marker::PhantomData,
        }
    }
}

impl Drop for RoomAsked {
    fn drop(&mut self) {
        let _ = ROOM_ASKED.try_with(|c| c.set(self.prev));
    }
}

#[inline]
fn with_room_asked(min_cap: usize) -> usize {
    if min_cap < ROOM_ASKED_MIN_LEN / 2 {
        return min_cap;
    }
    ROOM_ASKED
        .try_with(|c| {
            let (from, room) = c.get();
            if from != 0 && min_cap >= from && min_cap <= room {
                c.set((0, 0));
                room
            } else {
                min_cap
            }
        })
        .unwrap_or(min_cap)
}

/// The limbs a buffer needs to hold `a + b` made in place, where `b` has at most `b_limbs`
/// limbs and no shift (`add_bignum_into_par`).
pub fn sum_room(a: &BigNum, b_limbs: usize) -> usize {
    Shifted::new(&a.limbs, a.shift).len().max(b_limbs) + 1
}

/// Acquire an uninitialized `LimbVec` (`len() == 0`, `capacity() >= min_cap`) from the `Numbers`
/// arena (using this worker's LIFO region in subtrees and plan-assigned regions above that depth).
#[inline]
pub fn take_limb_buf_uninit(min_cap: usize) -> LimbVec {
    let cap = with_room_asked(min_cap);
    crate::arena::alloc_numbers_u64_uninit(cap).0
}

/// Acquire an uninitialized `LimbVec` from `Numbers` whose backing pages are faulted across
/// workers when newly committed so strided parallel writers never contend on OS page-fault locks.
#[inline]
pub fn take_limb_buf_committed(
    min_cap: usize,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    let cap = with_room_asked(min_cap);
    let (mut v, needs_touch) = crate::arena::alloc_numbers_u64_uninit(cap);
    if needs_touch && cap >= 65_536 && max_workers > 1 {
        let base_ptr = v.as_mut_ptr() as usize;
        let num_pages = cap.div_ceil(512);
        let nw = max_workers.min(num_pages);
        let pages_per_w = num_pages.div_ceil(nw);
        sched.par_for_leaf_max(nw, 1, max_workers, |wi| {
            let p_start = wi * pages_per_w;
            let p_end = (p_start + pages_per_w).min(num_pages);
            for p in p_start..p_end {
                let off = p * 512;
                if off < cap {
                    unsafe {
                        (base_ptr as *mut u64).add(off).write(0);
                    }
                }
            }
        });
    }
    v
}

/// Acquire a zero-initialized `LimbVec` of length `len` from the `Numbers` arena.
#[inline]
pub fn take_limb_buf_zeroed(len: usize) -> LimbVec {
    crate::arena::alloc_numbers_u64_zeroed(len)
}

/// Release a `LimbVec` back to the `Numbers` arena.
#[inline]
pub fn put_limb_buf(v: LimbVec) {
    drop(v);
}

#[derive(Clone, Debug)]
pub struct BigNum {
    pub limbs: LimbVec,
    pub neg: bool,
    pub shift: u64,
    pub chk: DualCheck,
}

impl BigNum {
    pub fn zero() -> Self {
        Self {
            limbs: empty_limb_vec(),
            neg: false,
            shift: 0,
            chk: DualCheck::new(0),
        }
    }

    pub fn from_u64(v: u64) -> Self {
        if v == 0 {
            return Self::zero();
        }
        let w0 = mod61_reduce(v as u128);
        let mut limbs = take_limb_buf_uninit(1);
        limbs.push(v);
        Self {
            limbs,
            neg: false,
            shift: 0,
            chk: DualCheck::new(w0),
        }
    }

    pub fn from_i128(v: i128) -> Self {
        if v == 0 {
            return Self::zero();
        }
        let neg = v < 0;
        let mag = v.unsigned_abs();
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        let mut limbs = take_limb_buf_uninit(if hi == 0 { 1 } else { 2 });
        limbs.push(lo);
        if hi != 0 {
            limbs.push(hi);
        }
        let lw0 = mod61_reduce(mag);
        let w0 = whole_value_w0(lw0, neg, 0);
        Self {
            limbs,
            neg,
            shift: 0,
            chk: DualCheck::new(w0),
        }
    }

    #[inline]
    pub fn into_limbs(mut self) -> LimbVec {
        take_limb_vec(&mut self.limbs)
    }

    #[inline]
    pub fn clear_limbs(&mut self) {
        drop(take_limb_vec(&mut self.limbs));
        self.neg = false;
        self.shift = 0;
        self.chk = DualCheck::new(0);
    }

    pub fn from_limbs_with_w0(
        limbs_in: impl IntoLimbVec,
        neg: bool,
        shift: u64,
        limbs_w0: u64,
    ) -> Self {
        let mut limbs = limbs_in.into_limb_vec();
        trim_limbs(&mut limbs);
        if limbs.is_empty() {
            return Self::zero();
        }
        let w0 = whole_value_w0(limbs_w0, neg, shift);
        Self {
            limbs,
            neg,
            shift,
            chk: DualCheck::new(w0),
        }
    }

    pub fn from_limbs(
        limbs_in: impl IntoLimbVec,
        neg: bool,
        shift: u64,
        kernels: &Kernels,
    ) -> Self {
        let mut limbs = limbs_in.into_limb_vec();
        trim_limbs(&mut limbs);
        if limbs.is_empty() {
            return Self::zero();
        }
        let lw0 = compute_limbs_w0(kernels, &limbs);
        let w0 = whole_value_w0(lw0, neg, shift);
        Self {
            limbs,
            neg,
            shift,
            chk: DualCheck::new(w0),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Total bit length of the represented integer `|self| = limbs_val * 2^shift`.
    #[inline]
    pub fn total_bits(&self) -> u64 {
        let Some(&top) = self.limbs.last() else {
            return 0;
        };
        let limb_bits = ((self.limbs.len() - 1) as u64) * 64 + (64 - (top.leading_zeros() as u64));
        limb_bits + self.shift
    }

    /// Bit length of `limbs` alone (excluding `shift`).
    #[inline]
    pub fn body_bits(&self) -> u64 {
        let Some(&top) = self.limbs.last() else {
            return 0;
        };
        ((self.limbs.len() - 1) as u64) * 64 + (64 - (top.leading_zeros() as u64))
    }

    /// Authoritative `w0` check word, resolving a single corrupted check-word copy if needed.
    #[inline]
    pub fn w0(&self) -> u64 {
        debug_assert_eq!(self.chk.c0, self.chk.c1);
        self.chk.c0
    }

    /// Unsigned limb check word `w0(self.limbs)` recovered in `O(1)` from `self.w0()`.
    #[inline]
    pub fn limbs_w0(&self) -> u64 {
        if self.limbs.is_empty() {
            0
        } else {
            crate::check::limbs_w0_from_whole(self.w0(), self.neg, self.shift)
        }
    }

    /// Scrub this number at rest against its dual check word (Section 11).
    pub fn verify_at_rest(&mut self, kernels: &Kernels, telemetry: &Telemetry) -> bool {
        if self.limbs.is_empty() {
            return self.chk.verify_or_repair_at_rest(0, telemetry);
        }
        let lw0 = compute_limbs_w0(kernels, &self.limbs);
        let actual = whole_value_w0(lw0, self.neg, self.shift);
        self.chk.verify_or_repair_at_rest(actual, telemetry)
    }

    /// Parallelized scrub at rest for multi-megabyte numbers.
    pub fn verify_at_rest_par(
        &mut self,
        kernels: &Kernels,
        sched: &crate::par::Scheduler,
        max_workers: usize,
        telemetry: &Telemetry,
    ) -> bool {
        if self.limbs.is_empty() {
            return self.chk.verify_or_repair_at_rest(0, telemetry);
        }
        let lw0 = crate::check::compute_limbs_w0_par(kernels, sched, &self.limbs, max_workers);
        let actual = whole_value_w0(lw0, self.neg, self.shift);
        self.chk.verify_or_repair_at_rest(actual, telemetry)
    }

    pub fn from_limbs_par(
        limbs_in: impl IntoLimbVec,
        neg: bool,
        shift: u64,
        kernels: &Kernels,
        sched: &crate::par::Scheduler,
        max_workers: usize,
    ) -> Self {
        let mut limbs = limbs_in.into_limb_vec();
        trim_limbs(&mut limbs);
        if limbs.is_empty() {
            return Self::zero();
        }
        let lw0 = crate::check::compute_limbs_w0_par(kernels, sched, &limbs, max_workers);
        let w0 = whole_value_w0(lw0, neg, shift);
        Self {
            limbs,
            neg,
            shift,
            chk: DualCheck::new(w0),
        }
    }

    /// Materialize `shift` zero bits into `limbs` so `shift` becomes `0`.
    pub fn materialize_shift(&self, kernels: &Kernels) -> Self {
        if self.is_zero() || self.shift == 0 {
            return self.clone();
        }
        let mut res = shl_limbs(&self.limbs, self.shift);
        trim_limbs(&mut res);
        let _ = kernels;
        Self {
            limbs: res,
            neg: self.neg,
            shift: 0,
            chk: self.chk, // Whole value is unchanged, so w0 is identical!
        }
    }

    /// Parallelized `materialize_shift` for multi-megabyte operands.
    pub fn materialize_shift_par(
        &self,
        kernels: &Kernels,
        sched: &crate::par::Scheduler,
        max_workers: usize,
    ) -> Self {
        if self.is_zero() || self.shift == 0 {
            return self.clone();
        }
        let mut res = shl_limbs_par(&self.limbs, self.shift, sched, max_workers);
        trim_limbs(&mut res);
        let _ = kernels;
        Self {
            limbs: res,
            neg: self.neg,
            shift: 0,
            chk: self.chk,
        }
    }

    /// Strip trailing zero bits from `limbs` into `shift` in-place (used for Q in Section 4).
    pub fn strip_zero_bits(&mut self) {
        if self.limbs.is_empty() {
            self.shift = 0;
            self.neg = false;
            return;
        }
        let mut zero_limbs = 0usize;
        while zero_limbs < self.limbs.len() && self.limbs[zero_limbs] == 0 {
            zero_limbs += 1;
        }
        if zero_limbs == self.limbs.len() {
            self.clear_limbs();
            return;
        }
        let tz = self.limbs[zero_limbs].trailing_zeros() as u32;
        let total_tz = (zero_limbs as u64) * 64 + (tz as u64);
        if total_tz > 0 {
            let len = self.limbs.len();
            let rem_len = len - zero_limbs;
            if tz == 0 {
                self.limbs.copy_within(zero_limbs..len, 0);
            } else {
                for i in 0..rem_len {
                    let lo = self.limbs[zero_limbs + i] >> tz;
                    let hi = if zero_limbs + i + 1 < len {
                        self.limbs[zero_limbs + i + 1] << (64 - tz)
                    } else {
                        0
                    };
                    self.limbs[i] = lo | hi;
                }
            }
            self.limbs.truncate(rem_len);
            trim_limbs(&mut self.limbs);
            self.shift += total_tz;
        }
    }

    /// Parallelized `strip_zero_bits` for multi-megabyte operands, in-place for cache-sized buffers.
    pub fn strip_zero_bits_par(&mut self, sched: &crate::par::Scheduler, max_workers: usize) {
        if max_workers <= 1 || self.limbs.len() < 32_768 {
            self.strip_zero_bits();
            return;
        }
        if self.limbs.is_empty() {
            self.shift = 0;
            self.neg = false;
            return;
        }
        let mut zero_limbs = 0usize;
        while zero_limbs < self.limbs.len() && self.limbs[zero_limbs] == 0 {
            zero_limbs += 1;
        }
        if zero_limbs == self.limbs.len() {
            self.clear_limbs();
            return;
        }
        let tz = self.limbs[zero_limbs].trailing_zeros() as u64;
        let total_tz = (zero_limbs as u64) * 64 + tz;
        if total_tz > 0 {
            let new_limbs = shr_limbs_par(&self.limbs, total_tz, sched, max_workers);
            put_limb_buf(std::mem::replace(&mut self.limbs, new_limbs));
            trim_limbs(&mut self.limbs);
            self.shift += total_tz;
        }
    }
}

#[inline]
pub fn trim_limbs<A: Allocator>(limbs: &mut Vec<u64, A>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

pub fn shl_limbs(limbs: &[u64], shift_bits: u64) -> LimbVec {
    if limbs.is_empty() {
        return empty_limb_vec();
    }
    let word_shift = (shift_bits >> 6) as usize;
    let bit_shift = (shift_bits & 63) as u32;
    let out_len = limbs.len() + word_shift + usize::from(bit_shift != 0);
    let mut out = take_limb_buf_uninit(out_len);
    let out_ptr = out.as_mut_ptr();
    unsafe {
        if word_shift > 0 {
            std::ptr::write_bytes(out_ptr, 0, word_shift);
        }
        if bit_shift == 0 {
            std::ptr::copy_nonoverlapping(limbs.as_ptr(), out_ptr.add(word_shift), limbs.len());
        } else {
            let mut carry = 0u64;
            for (i, &w) in limbs.iter().enumerate() {
                out_ptr.add(word_shift + i).write((w << bit_shift) | carry);
                carry = w >> (64 - bit_shift);
            }
            out_ptr.add(word_shift + limbs.len()).write(carry);
        }
        out.set_len(out_len);
    }
    trim_limbs(&mut out);
    out
}

pub fn shl_limbs_par(
    limbs: &[u64],
    shift_bits: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    if limbs.is_empty() {
        return empty_limb_vec();
    }
    let word_shift = (shift_bits >> 6) as usize;
    let bit_shift = (shift_bits & 63) as u32;
    let out_len = limbs.len() + word_shift + usize::from(bit_shift != 0);
    if max_workers <= 1 || out_len < 8_192 {
        return shl_limbs(limbs, shift_bits);
    }
    let mut out = take_limb_buf_uninit(out_len);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = if out_len >= 65_536 { 16_384usize } else { 4_096usize };
    if word_shift > 0 {
        let z_chunks = word_shift.div_ceil(chunk);
        sched.par_for_leaf_max(z_chunks, 1, max_workers, |ci| {
            let s = ci * chunk;
            let e = (s + chunk).min(word_shift);
            unsafe {
                std::ptr::write_bytes((out_ptr as *mut u64).add(s), 0, e - s);
            }
        });
    }
    let n = limbs.len();
    let num_chunks = n.div_ceil(chunk);
    if bit_shift == 0 {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    limbs.as_ptr().add(start),
                    (out_ptr as *mut u64).add(word_shift + start),
                    end - start,
                );
            }
        });
    } else {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let mut prev = if start == 0 {
                0u64
            } else {
                limbs[start - 1] >> (64 - bit_shift)
            };
            for i in start..end {
                let w = limbs[i];
                let val = (w << bit_shift) | prev;
                prev = w >> (64 - bit_shift);
                unsafe {
                    (out_ptr as *mut u64).add(word_shift + i).write(val);
                }
            }
        });
        unsafe {
            (out_ptr as *mut u64)
                .add(word_shift + n)
                .write(limbs[n - 1] >> (64 - bit_shift));
        }
    }
    unsafe {
        out.set_len(out_len);
    }
    trim_limbs(&mut out);
    out
}

pub fn shr_limbs(limbs: &[u64], shift_bits: u64) -> LimbVec {
    let word_shift = (shift_bits >> 6) as usize;
    if word_shift >= limbs.len() {
        return empty_limb_vec();
    }
    let bit_shift = (shift_bits & 63) as u32;
    let rem_len = limbs.len() - word_shift;
    let mut out = take_limb_buf_uninit(rem_len);
    let out_ptr = out.as_mut_ptr();
    unsafe {
        if bit_shift == 0 {
            std::ptr::copy_nonoverlapping(limbs.as_ptr().add(word_shift), out_ptr, rem_len);
        } else {
            for i in 0..rem_len {
                let lo = limbs[word_shift + i] >> bit_shift;
                let hi = if word_shift + i + 1 < limbs.len() {
                    limbs[word_shift + i + 1] << (64 - bit_shift)
                } else {
                    0
                };
                out_ptr.add(i).write(lo | hi);
            }
        }
        out.set_len(rem_len);
    }
    trim_limbs(&mut out);
    out
}

pub fn shr_limbs_par(
    limbs: &[u64],
    shift_bits: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    let word_shift = (shift_bits >> 6) as usize;
    if word_shift >= limbs.len() {
        return empty_limb_vec();
    }
    let rem_len = limbs.len() - word_shift;
    if max_workers <= 1 || rem_len < 8_192 {
        return shr_limbs(limbs, shift_bits);
    }
    let bit_shift = (shift_bits & 63) as u32;
    let mut out = take_limb_buf_uninit(rem_len);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = if rem_len >= 65_536 { 16_384usize } else { 4_096usize };
    let num_chunks = rem_len.div_ceil(chunk);
    if bit_shift == 0 {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(rem_len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    limbs.as_ptr().add(word_shift + start),
                    (out_ptr as *mut u64).add(start),
                    end - start,
                );
            }
        });
    } else {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(rem_len);
            for i in start..end {
                let lo = limbs[word_shift + i] >> bit_shift;
                let hi = if word_shift + i + 1 < limbs.len() {
                    limbs[word_shift + i + 1] << (64 - bit_shift)
                } else {
                    0
                };
                unsafe {
                    (out_ptr as *mut u64).add(i).write(lo | hi);
                }
            }
        });
    }
    unsafe {
        out.set_len(rem_len);
    }
    trim_limbs(&mut out);
    out
}

pub fn cmp_limbs(a: &[u64], b: &[u64]) -> Ordering {
    let mut la = a.len();
    while la > 0 && a[la - 1] == 0 {
        la -= 1;
    }
    let mut lb = b.len();
    while lb > 0 && b[lb - 1] == 0 {
        lb -= 1;
    }
    if la != lb {
        return la.cmp(&lb);
    }
    for i in (0..la).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

pub fn add_limbs(kernels: &Kernels, a: &[u64], b: &[u64]) -> LimbVec {
    if a.is_empty() {
        return slice_to_limb_vec(b);
    }
    if b.is_empty() {
        return slice_to_limb_vec(a);
    }
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let n_short = short.len();
    let n_long = long.len();
    let mut out = take_limb_buf_uninit(n_long + 1);
    let out_ptr = out.as_mut_ptr();

    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, AtomicOrd::Relaxed);
    }

    unsafe {
        let mut carry = (kernels.active.add_n)(
            out_ptr,
            long.as_ptr(),
            short.as_ptr(),
            n_short as u64,
        );
        for i in n_short..n_long {
            let (s, ov) = long[i].overflowing_add(carry);
            out_ptr.add(i).write(s);
            carry = ov as u64;
        }
        out_ptr.add(n_long).write(carry);
        out.set_len(n_long + 1);
    }
    trim_limbs(&mut out);
    out
}

/// Subtract `b` from `a` assuming `a >= b`.
pub fn sub_limbs(kernels: &Kernels, a: &[u64], b: &[u64]) -> LimbVec {
    debug_assert!(cmp_limbs(a, b) != Ordering::Less);
    if b.is_empty() {
        return slice_to_limb_vec(a);
    }
    let n_short = b.len();
    let n_long = a.len();
    let mut out = take_limb_buf_uninit(n_long);
    let out_ptr = out.as_mut_ptr();

    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, AtomicOrd::Relaxed);
    }

    unsafe {
        let mut borrow = (kernels.active.sub_n)(
            out_ptr,
            a.as_ptr(),
            b.as_ptr(),
            n_short as u64,
        );
        for i in n_short..n_long {
            let (d, un) = a[i].overflowing_sub(borrow);
            out_ptr.add(i).write(d);
            borrow = un as u64;
        }
        debug_assert_eq!(borrow, 0);
        out.set_len(n_long);
    }
    trim_limbs(&mut out);
    out
}

pub fn add_bignum(kernels: &Kernels, a: &BigNum, b: &BigNum) -> BigNum {
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    let exp_w0 = mod61_add(a.w0(), b.w0());
    if a.shift != 0 || b.shift != 0 {
        let (x, y) = (Shifted::new(&a.limbs, a.shift), Shifted::new(&b.limbs, b.shift));
        let (limbs, neg) = if a.neg == b.neg {
            (add_sub_shifted_seq(kernels, x, y, false), a.neg)
        } else {
            match cmp_shifted(x, y) {
                Ordering::Equal => return BigNum::zero(),
                Ordering::Greater => (add_sub_shifted_seq(kernels, x, y, true), a.neg),
                Ordering::Less => (add_sub_shifted_seq(kernels, y, x, true), b.neg),
            }
        };
        if limbs.is_empty() {
            return BigNum::zero();
        }
        return BigNum {
            limbs,
            neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
    }
    if a.neg == b.neg {
        let limbs = add_limbs(kernels, &a.limbs, &b.limbs);
        let mut res = BigNum {
            limbs,
            neg: a.neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
        if res.limbs.is_empty() {
            res.neg = false;
            res.chk = DualCheck::new(0);
        }
        res
    } else {
        match cmp_limbs(&a.limbs, &b.limbs) {
            Ordering::Equal => BigNum::zero(),
            Ordering::Greater => BigNum {
                limbs: sub_limbs(kernels, &a.limbs, &b.limbs),
                neg: a.neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
            Ordering::Less => BigNum {
                limbs: sub_limbs(kernels, &b.limbs, &a.limbs),
                neg: b.neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
        }
    }
}

pub fn sub_bignum(kernels: &Kernels, a: &BigNum, b: &BigNum) -> BigNum {
    if b.is_zero() {
        return a.clone();
    }
    if a.is_zero() {
        let mut res = b.clone();
        res.neg = !res.neg;
        let bw0 = if res.w0() == 0 { 0 } else { crate::refk::MOD_W0 - res.w0() };
        res.chk = DualCheck::new(bw0);
        return res;
    }
    let bw0 = if b.w0() == 0 {
        0
    } else {
        crate::refk::MOD_W0 - b.w0()
    };
    let b_neg = !b.neg;
    let exp_w0 = mod61_add(a.w0(), bw0);
    if a.shift != 0 || b.shift != 0 {
        let (x, y) = (Shifted::new(&a.limbs, a.shift), Shifted::new(&b.limbs, b.shift));
        let (limbs, neg) = if a.neg == b_neg {
            (add_sub_shifted_seq(kernels, x, y, false), a.neg)
        } else {
            match cmp_shifted(x, y) {
                Ordering::Equal => return BigNum::zero(),
                Ordering::Greater => (add_sub_shifted_seq(kernels, x, y, true), a.neg),
                Ordering::Less => (add_sub_shifted_seq(kernels, y, x, true), b_neg),
            }
        };
        if limbs.is_empty() {
            return BigNum::zero();
        }
        return BigNum {
            limbs,
            neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
    }
    if a.neg == b_neg {
        let limbs = add_limbs(kernels, &a.limbs, &b.limbs);
        let mut res = BigNum {
            limbs,
            neg: a.neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
        if res.limbs.is_empty() {
            res.neg = false;
            res.chk = DualCheck::new(0);
        }
        res
    } else {
        match cmp_limbs(&a.limbs, &b.limbs) {
            Ordering::Equal => BigNum::zero(),
            Ordering::Greater => BigNum {
                limbs: sub_limbs(kernels, &a.limbs, &b.limbs),
                neg: a.neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
            Ordering::Less => BigNum {
                limbs: sub_limbs(kernels, &b.limbs, &a.limbs),
                neg: b_neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
        }
    }
}

pub fn mul_bignum_u64(kernels: &Kernels, a: &BigNum, m: u64) -> BigNum {
    if a.is_zero() || m == 0 {
        return BigNum::zero();
    }
    if m == 1 {
        return a.clone();
    }
    let n = a.limbs.len();
    let mut out = take_limb_buf_uninit(n + 1);
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, AtomicOrd::Relaxed);
    }
    unsafe {
        let hi = (kernels.active.mul_1)(out.as_mut_ptr(), a.limbs.as_ptr(), n as u64, m);
        out.as_mut_ptr().add(n).write(hi);
        out.set_len(n + 1);
    }
    trim_limbs(&mut out);
    let exp_w0 = mod61_mul(a.w0(), mod61_reduce(m as u128));
    BigNum {
        limbs: out,
        neg: a.neg,
        shift: a.shift,
        chk: DualCheck::new(exp_w0),
    }
}

pub fn mul_limbs_u64_par(
    kernels: &Kernels,
    a: &[u64],
    m: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    if a.is_empty() || m == 0 {
        return empty_limb_vec();
    }
    let n = a.len();
    if max_workers <= 1 || n < 32_768 {
        let mut out = take_limb_buf_uninit(n + 1);
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add(n as u64, AtomicOrd::Relaxed);
        }
        unsafe {
            let hi = (kernels.active.mul_1)(out.as_mut_ptr(), a.as_ptr(), n as u64, m);
            out.as_mut_ptr().add(n).write(hi);
            out.set_len(n + 1);
        }
        trim_limbs(&mut out);
        return out;
    }
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, AtomicOrd::Relaxed);
    }
    let mut out = take_limb_buf_uninit(n + 1);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = 16_384usize;
    let num_chunks = n.div_ceil(chunk);
    let mut carries = vec![0u64; num_chunks];
    let carries_ptr = carries.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let s = ci * chunk;
        let e = (s + chunk).min(n);
        let hi = unsafe {
            (kernels.active.mul_1)(
                (out_ptr as *mut u64).add(s),
                a.as_ptr().add(s),
                (e - s) as u64,
                m,
            )
        };
        unsafe {
            *(carries_ptr as *mut u64).add(ci) = hi;
        }
    });
    unsafe {
        (out_ptr as *mut u64).add(n).write(0);
        out.set_len(n + 1);
    }
    for (ci, &hi) in carries.iter().enumerate() {
        if hi == 0 {
            continue;
        }
        let mut pos = ((ci + 1) * chunk).min(n);
        let mut c = hi;
        while c != 0 {
            let (sum, ov) = out[pos].overflowing_add(c);
            out[pos] = sum;
            c = ov as u64;
            pos += 1;
        }
    }
    trim_limbs(&mut out);
    out
}

pub fn mul_bignum_u64_par(
    kernels: &Kernels,
    a: &BigNum,
    m: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> BigNum {
    if a.is_zero() || m == 0 {
        return BigNum::zero();
    }
    if m == 1 {
        return a.clone();
    }
    let out = mul_limbs_u64_par(kernels, &a.limbs, m, sched, max_workers);
    let exp_w0 = mod61_mul(a.w0(), mod61_reduce(m as u128));
    BigNum {
        limbs: out,
        neg: a.neg,
        shift: a.shift,
        chk: DualCheck::new(exp_w0),
    }
}

pub fn add_limbs_par(
    kernels: &Kernels,
    a: &[u64],
    b: &[u64],
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    if a.is_empty() {
        return slice_to_limb_vec(b);
    }
    if b.is_empty() {
        return slice_to_limb_vec(a);
    }
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let n_short = short.len();
    let n_long = long.len();
    if max_workers <= 1 || n_long < 8_192 {
        return add_limbs(kernels, a, b);
    }
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, AtomicOrd::Relaxed);
    }
    let mut out = take_limb_buf_uninit(n_long + 1);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = if n_long >= 65_536 { 16_384usize } else { 4_096usize };
    let num_chunks = n_long.div_ceil(chunk);
    let mut carries = vec![0u64; num_chunks];
    let carries_ptr = carries.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let s = ci * chunk;
        let e = (s + chunk).min(n_long);
        if s >= n_short {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    long.as_ptr().add(s),
                    (out_ptr as *mut u64).add(s),
                    e - s,
                );
            }
        } else if e <= n_short {
            let c = unsafe {
                (kernels.active.add_n)(
                    (out_ptr as *mut u64).add(s),
                    long.as_ptr().add(s),
                    short.as_ptr().add(s),
                    (e - s) as u64,
                )
            };
            unsafe {
                *(carries_ptr as *mut u64).add(ci) = c;
            }
        } else {
            let mut c = unsafe {
                (kernels.active.add_n)(
                    (out_ptr as *mut u64).add(s),
                    long.as_ptr().add(s),
                    short.as_ptr().add(s),
                    (n_short - s) as u64,
                )
            };
            for i in n_short..e {
                let (sum, ov) = long[i].overflowing_add(c);
                unsafe {
                    (out_ptr as *mut u64).add(i).write(sum);
                }
                c = ov as u64;
            }
            unsafe {
                *(carries_ptr as *mut u64).add(ci) = c;
            }
        }
    });
    unsafe {
        (out_ptr as *mut u64).add(n_long).write(0);
        out.set_len(n_long + 1);
    }
    for (ci, &c_in) in carries.iter().enumerate() {
        if c_in == 0 {
            continue;
        }
        let mut pos = ((ci + 1) * chunk).min(n_long);
        let mut c = c_in;
        while c != 0 {
            let (sum, ov) = out[pos].overflowing_add(c);
            out[pos] = sum;
            c = ov as u64;
            pos += 1;
        }
    }
    trim_limbs(&mut out);
    out
}

pub fn sub_limbs_par(
    kernels: &Kernels,
    a: &[u64],
    b: &[u64],
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    debug_assert!(cmp_limbs(a, b) != Ordering::Less);
    if b.is_empty() {
        return slice_to_limb_vec(a);
    }
    let n_short = b.len();
    let n_long = a.len();
    if max_workers <= 1 || n_long < 8_192 {
        return sub_limbs(kernels, a, b);
    }
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, AtomicOrd::Relaxed);
    }
    let mut out = take_limb_buf_uninit(n_long);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = if n_long >= 65_536 { 16_384usize } else { 4_096usize };
    let num_chunks = n_long.div_ceil(chunk);
    let mut borrows = vec![0u64; num_chunks];
    let borrows_ptr = borrows.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let s = ci * chunk;
        let e = (s + chunk).min(n_long);
        if s >= n_short {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.as_ptr().add(s),
                    (out_ptr as *mut u64).add(s),
                    e - s,
                );
            }
        } else if e <= n_short {
            let b_out = unsafe {
                (kernels.active.sub_n)(
                    (out_ptr as *mut u64).add(s),
                    a.as_ptr().add(s),
                    b.as_ptr().add(s),
                    (e - s) as u64,
                )
            };
            unsafe {
                *(borrows_ptr as *mut u64).add(ci) = b_out;
            }
        } else {
            let mut b_out = unsafe {
                (kernels.active.sub_n)(
                    (out_ptr as *mut u64).add(s),
                    a.as_ptr().add(s),
                    b.as_ptr().add(s),
                    (n_short - s) as u64,
                )
            };
            for i in n_short..e {
                let (d, un) = a[i].overflowing_sub(b_out);
                unsafe {
                    (out_ptr as *mut u64).add(i).write(d);
                }
                b_out = un as u64;
            }
            unsafe {
                *(borrows_ptr as *mut u64).add(ci) = b_out;
            }
        }
    });
    unsafe {
        out.set_len(n_long);
    }
    for (ci, &b_in) in borrows.iter().enumerate() {
        if b_in == 0 {
            continue;
        }
        let mut pos = (ci + 1) * chunk;
        let mut br = b_in;
        while br != 0 && pos < n_long {
            let (d, un) = out[pos].overflowing_sub(br);
            out[pos] = d;
            br = un as u64;
            pos += 1;
        }
        debug_assert_eq!(br, 0);
    }
    trim_limbs(&mut out);
    out
}

/// Limbs shifted left by a bit count, read limb by limb without being written out first.
#[derive(Clone, Copy)]
struct Shifted<'a> {
    limbs: &'a [u64],
    word_shift: usize,
    bit_shift: u32,
}

impl<'a> Shifted<'a> {
    fn new(limbs: &'a [u64], shift_bits: u64) -> Self {
        Self {
            limbs,
            word_shift: (shift_bits >> 6) as usize,
            bit_shift: (shift_bits & 63) as u32,
        }
    }

    /// Limb count of the shifted number; its top limb may be zero.
    fn len(&self) -> usize {
        if self.limbs.is_empty() {
            0
        } else {
            self.limbs.len() + self.word_shift + usize::from(self.bit_shift != 0)
        }
    }

    /// Bit length of the shifted number (`limbs` is trimmed).
    fn bit_len(&self) -> u64 {
        match self.limbs.last() {
            Some(&top) => {
                (self.limbs.len() as u64) * 64 - (top.leading_zeros() as u64)
                    + (self.word_shift as u64) * 64
                    + (self.bit_shift as u64)
            }
            None => 0,
        }
    }

    /// Limb `i` of the shifted number.
    #[inline]
    fn limb(&self, i: usize) -> u64 {
        if i < self.word_shift {
            return 0;
        }
        let j = i - self.word_shift;
        let n = self.limbs.len();
        let lo = if j < n { self.limbs[j] << self.bit_shift } else { 0 };
        let hi = if self.bit_shift != 0 && j >= 1 && j <= n {
            self.limbs[j - 1] >> (64 - self.bit_shift)
        } else {
            0
        };
        lo | hi
    }

    /// Limbs `[start, start + len)` of the shifted number: a pointer straight into `limbs` when
    /// they lie there unshifted, else they are written to `scratch` and that is returned.
    fn chunk(&self, start: usize, len: usize, scratch: &mut [u64]) -> *const u64 {
        if self.bit_shift == 0
            && start >= self.word_shift
            && start + len <= self.word_shift + self.limbs.len()
        {
            return unsafe { self.limbs.as_ptr().add(start - self.word_shift) };
        }
        let dst = &mut scratch[..len];
        let n = self.limbs.len();
        let up = self.bit_shift;
        // Zero limbs below the data, then the data from source limb `j` on.
        let below = self.word_shift.saturating_sub(start).min(len);
        dst[..below].fill(0);
        let mut d = below;
        if d < len {
            let mut j = start + d - self.word_shift;
            if up == 0 {
                let cnt = n.saturating_sub(j).min(len - d);
                if cnt > 0 {
                    dst[d..d + cnt].copy_from_slice(&self.limbs[j..j + cnt]);
                    d += cnt;
                }
            } else {
                let mut prev = if j >= 1 && j <= n {
                    self.limbs[j - 1] >> (64 - up)
                } else {
                    0
                };
                while d < len && j < n {
                    let w = self.limbs[j];
                    dst[d] = (w << up) | prev;
                    prev = w >> (64 - up);
                    d += 1;
                    j += 1;
                }
                if d < len && j == n {
                    // The limb just above the data holds the bits shifted out of its top limb.
                    dst[d] = prev;
                    d += 1;
                }
            }
        }
        dst[d..].fill(0);
        dst.as_ptr()
    }
}

fn add_sub_shifted_seq(
    kernels: &Kernels,
    x: Shifted<'_>,
    y: Shifted<'_>,
    subtract: bool,
) -> LimbVec {
    let n = x.len().max(y.len());
    if n == 0 {
        return empty_limb_vec();
    }
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, AtomicOrd::Relaxed);
    }
    let mut out = take_limb_buf_uninit(n + 1);
    let out_ptr = out.as_mut_ptr();
    const CHUNK: usize = 1024;
    let mut sc_x = [0u64; CHUNK];
    let mut sc_y = [0u64; CHUNK];
    let min_ws = x.word_shift.min(y.word_shift).min(n);
    if min_ws > 0 {
        unsafe {
            std::ptr::write_bytes(out_ptr, 0, min_ws);
        }
    }
    let mut s = min_ws;
    let mut c_in = 0u64;
    while s < n {
        if s >= y.len() && x.bit_shift == 0 && c_in == 0 {
            let rem_x = x.limbs.len().saturating_sub(s.saturating_sub(x.word_shift));
            let cp = rem_x.min(n - s);
            unsafe {
                if cp > 0 {
                    std::ptr::copy_nonoverlapping(
                        x.limbs.as_ptr().add(s - x.word_shift),
                        out_ptr.add(s),
                        cp,
                    );
                }
                if s + cp < n {
                    std::ptr::write_bytes(out_ptr.add(s + cp), 0, n - (s + cp));
                }
            }
            break;
        }
        let len = (s + CHUNK).min(n) - s;
        let px = x.chunk(s, len, &mut sc_x);
        let py = y.chunk(s, len, &mut sc_y);
        let dst = unsafe { out_ptr.add(s) };
        let c_step = unsafe {
            if subtract {
                (kernels.active.sub_n)(dst, px, py, len as u64)
            } else {
                (kernels.active.add_n)(dst, px, py, len as u64)
            }
        };
        if c_in != 0 {
            let mut k = 0usize;
            while c_in != 0 && k < len {
                unsafe {
                    let cur = *dst.add(k);
                    let (v, over) = if subtract {
                        cur.overflowing_sub(c_in)
                    } else {
                        cur.overflowing_add(c_in)
                    };
                    *dst.add(k) = v;
                    c_in = u64::from(over);
                }
                k += 1;
            }
        }
        c_in += c_step;
        s += len;
    }
    unsafe {
        out_ptr.add(n).write(if subtract { 0 } else { c_in });
        out.set_len(n + 1);
    }
    trim_limbs(&mut out);
    out
}

/// `x + y`, or `x - y` where `x >= y`, for two shifted numbers, in one parallel pass over both:
/// neither shifted number is written out first.
fn add_sub_shifted_par(
    kernels: &Kernels,
    x: Shifted<'_>,
    y: Shifted<'_>,
    subtract: bool,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    let n = x.len().max(y.len());
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, AtomicOrd::Relaxed);
    }
    let mut out = take_limb_buf_uninit(n + 1);
    let out_ptr = out.as_mut_ptr() as usize;
    const CHUNK: usize = 4_096;
    let num_chunks = n.div_ceil(CHUNK);
    let mut carries = vec![0u64; num_chunks];
    let carries_ptr = carries.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let mut sc_x = [0u64; CHUNK];
        let mut sc_y = [0u64; CHUNK];
        let s = ci * CHUNK;
        let len = (s + CHUNK).min(n) - s;
        let px = x.chunk(s, len, &mut sc_x);
        let py = y.chunk(s, len, &mut sc_y);
        let dst = unsafe { (out_ptr as *mut u64).add(s) };
        let c = unsafe {
            if subtract {
                (kernels.active.sub_n)(dst, px, py, len as u64)
            } else {
                (kernels.active.add_n)(dst, px, py, len as u64)
            }
        };
        unsafe { *(carries_ptr as *mut u64).add(ci) = c };
    });
    unsafe {
        (out_ptr as *mut u64).add(n).write(0);
        out.set_len(n + 1);
    }
    // A carry or borrow out of a chunk goes into the limbs above it; it seldom runs far.
    for (ci, &c_in) in carries.iter().enumerate() {
        let mut pos = ((ci + 1) * CHUNK).min(n);
        let mut c = c_in;
        while c != 0 && pos <= n {
            let (v, over) = if subtract {
                out[pos].overflowing_sub(c)
            } else {
                out[pos].overflowing_add(c)
            };
            out[pos] = v;
            c = over as u64;
            pos += 1;
        }
    }
    trim_limbs(&mut out);
    out
}

/// Compare two shifted numbers by value.
fn cmp_shifted(x: Shifted<'_>, y: Shifted<'_>) -> Ordering {
    let by_len = x.bit_len().cmp(&y.bit_len());
    if by_len != Ordering::Equal {
        return by_len;
    }
    for i in (0..x.len().max(y.len())).rev() {
        let by_limb = x.limb(i).cmp(&y.limb(i));
        if by_limb != Ordering::Equal {
            return by_limb;
        }
    }
    Ordering::Equal
}

pub fn add_bignum_par(
    kernels: &Kernels,
    a: &BigNum,
    b: &BigNum,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> BigNum {
    if max_workers <= 1 {
        return add_bignum(kernels, a, b);
    }
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    if a.shift != 0 || b.shift != 0 {
        let (x, y) = (Shifted::new(&a.limbs, a.shift), Shifted::new(&b.limbs, b.shift));
        if x.len().max(y.len()) < 8_192 {
            return add_bignum(kernels, a, b);
        }
        let exp_w0 = mod61_add(a.w0(), b.w0());
        let (limbs, neg) = if a.neg == b.neg {
            (add_sub_shifted_par(kernels, x, y, false, sched, max_workers), a.neg)
        } else {
            match cmp_shifted(x, y) {
                Ordering::Equal => return BigNum::zero(),
                Ordering::Greater => {
                    (add_sub_shifted_par(kernels, x, y, true, sched, max_workers), a.neg)
                }
                Ordering::Less => {
                    (add_sub_shifted_par(kernels, y, x, true, sched, max_workers), b.neg)
                }
            }
        };
        if limbs.is_empty() {
            return BigNum::zero();
        }
        return BigNum {
            limbs,
            neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
    }

    let exp_w0 = mod61_add(a.w0(), b.w0());
    if a.neg == b.neg {
        let limbs = add_limbs_par(kernels, &a.limbs, &b.limbs, sched, max_workers);
        let mut res = BigNum {
            limbs,
            neg: a.neg,
            shift: 0,
            chk: DualCheck::new(exp_w0),
        };
        if res.limbs.is_empty() {
            res.neg = false;
            res.chk = DualCheck::new(0);
        }
        res
    } else {
        match cmp_limbs(&a.limbs, &b.limbs) {
            Ordering::Equal => BigNum::zero(),
            Ordering::Greater => BigNum {
                limbs: sub_limbs_par(kernels, &a.limbs, &b.limbs, sched, max_workers),
                neg: a.neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
            Ordering::Less => BigNum {
                limbs: sub_limbs_par(kernels, &b.limbs, &a.limbs, sched, max_workers),
                neg: b.neg,
                shift: 0,
                chk: DualCheck::new(exp_w0),
            },
        }
    }
}

/// `a + b`, made in the buffer of `b` where that has the room for the sum and `b` has no shift.
///
/// A sum of two large numbers is bound by the memory it moves, and a fresh buffer for the result
/// is the dearest part of that: every line of it is brought in from memory before it is written.
/// The limbs of `b`, which the caller has only just made, are still in cache. Where the buffer
/// has no room, or the numbers are small, this is `add_bignum_par` and `b` goes back to its pool.
pub fn add_bignum_into_par(
    kernels: &Kernels,
    a: &BigNum,
    mut b: BigNum,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> BigNum {
    const CHUNK: usize = 4_096;
    let x = Shifted::new(&a.limbs, a.shift);
    let lb = b.limbs.len();
    let n = x.len().max(lb);
    if a.is_zero() || b.is_zero() || b.shift != 0 || n < 2 * CHUNK || b.limbs.capacity() < n + 1 {
        return add_bignum_par(kernels, a, &b, sched, max_workers);
    }
    // With unlike signs the larger of the two gives the sign and is the one subtracted from.
    let (subtract, b_is_larger, neg) = if a.neg == b.neg {
        (false, false, a.neg)
    } else {
        match cmp_shifted(x, Shifted::new(&b.limbs, 0)) {
            Ordering::Equal => return BigNum::zero(),
            Ordering::Greater => (true, false, a.neg),
            Ordering::Less => (true, true, b.neg),
        }
    };
    let exp_w0 = mod61_add(a.w0(), b.w0());
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, AtomicOrd::Relaxed);
    }
    let num_chunks = n.div_ceil(CHUNK);
    let mut carries = vec![0u64; num_chunks];
    let carries_ptr = carries.as_mut_ptr() as usize;
    let dst_ptr = b.limbs.as_mut_ptr() as usize;
    sched.par_for_init_max(
        num_chunks,
        1,
        max_workers,
        || vec![0u64; CHUNK],
        |scratch, ci| {
            let s = ci * CHUNK;
            let len = (s + CHUNK).min(n) - s;
            let dst = unsafe { (dst_ptr as *mut u64).add(s) };
            // Limbs of this chunk that `b` holds; above them it is zero and its buffer unwritten.
            let have = lb.saturating_sub(s).min(len);
            let c = unsafe {
                if have == 0 && !b_is_larger {
                    // Only `a` reaches up here: its limbs go straight into place.
                    let px = x.chunk(s, len, std::slice::from_raw_parts_mut(dst, len));
                    if !std::ptr::eq(px, dst as *const u64) {
                        std::ptr::copy_nonoverlapping(px, dst, len);
                    }
                    0
                } else {
                    if have < len {
                        std::ptr::write_bytes(dst.add(have), 0, len - have);
                    }
                    let px = x.chunk(s, len, scratch);
                    // The kernels read a limb of both operands before they write it, so the
                    // result may lie exactly on one of them (gate 1 holds them to that).
                    if !subtract {
                        (kernels.active.add_n)(dst, dst as *const u64, px, len as u64)
                    } else if b_is_larger {
                        (kernels.active.sub_n)(dst, dst as *const u64, px, len as u64)
                    } else {
                        (kernels.active.sub_n)(dst, px, dst as *const u64, len as u64)
                    }
                }
            };
            unsafe { *(carries_ptr as *mut u64).add(ci) = c };
        },
    );
    let mut out = take_limb_vec(&mut b.limbs);
    unsafe {
        (dst_ptr as *mut u64).add(n).write(0);
        out.set_len(n + 1);
    }
    // A carry or borrow out of a chunk goes into the limbs above it; it seldom runs far.
    for (ci, &c_in) in carries.iter().enumerate() {
        let mut pos = ((ci + 1) * CHUNK).min(n);
        let mut c = c_in;
        while c != 0 && pos <= n {
            let (v, over) = if subtract {
                out[pos].overflowing_sub(c)
            } else {
                out[pos].overflowing_add(c)
            };
            out[pos] = v;
            c = over as u64;
            pos += 1;
        }
    }
    trim_limbs(&mut out);
    if out.is_empty() {
        put_limb_buf(out);
        return BigNum::zero();
    }
    BigNum {
        limbs: out,
        neg,
        shift: 0,
        chk: DualCheck::new(exp_w0),
    }
}

/// `out[..nl + ns] = long * short` by the kernel `mul_base` (`nl >= ns > 0`), whatever the two
/// lengths: which way a body takes for them is the body's affair and the plan's (the crossover
/// among the kernel's lengths, `Kernels::mul_base_lens`), not this module's.
#[inline]
pub(crate) fn mul_base_dispatched(
    kernels: &Kernels,
    out: &mut [u64],
    long: &[u64],
    short: &[u64],
) {
    let nl = long.len();
    let ns = short.len();
    debug_assert!(nl >= ns && ns > 0 && out.len() >= nl + ns);
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add((nl * ns) as u64, AtomicOrd::Relaxed);
    }
    let lens = kernels.mul_base_lens(nl, ns);
    unsafe {
        (kernels.active.mul_base)(out.as_mut_ptr(), long.as_ptr(), short.as_ptr(), &lens);
    }
}

pub fn mul_limbs_basecase(kernels: &Kernels, a: &[u64], b: &[u64]) -> LimbVec {
    let la = trimmed_len(a);
    let lb = trimmed_len(b);
    if la == 0 || lb == 0 {
        return empty_limb_vec();
    }
    let a = &a[..la];
    let b = &b[..lb];
    if std::ptr::eq(a.as_ptr(), b.as_ptr()) && la == lb {
        return sqr_limbs_basecase(kernels, a);
    }
    let (long, short) = if la >= lb { (a, b) } else { (b, a) };
    let cap = long.len() + short.len();
    let mut out = take_limb_buf_uninit(cap);
    unsafe {
        out.set_len(cap);
    }
    mul_base_dispatched(kernels, &mut out, long, short);
    trim_limbs(&mut out);
    out
}

pub fn sqr_limbs_basecase(kernels: &Kernels, a: &[u64]) -> LimbVec {
    let n = trimmed_len(a);
    if n == 0 {
        return empty_limb_vec();
    }
    let a = &a[..n];
    let mut out = take_limb_buf_uninit(2 * n);
    unsafe {
        out.set_len(2 * n);
    }
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(((n * (n + 1)) / 2) as u64, AtomicOrd::Relaxed);
    }
    unsafe {
        (kernels.active.sqr_base)(
            out.as_mut_ptr(),
            a.as_ptr(),
            n as u64,
            kernels.sqr_rows_max(),
        );
    }
    trim_limbs(&mut out);
    out
}

std::thread_local! {
    static KARATSUBA_SCRATCH: std::cell::RefCell<Vec<u64, ScratchAlloc>> =
        std::cell::RefCell::new(Vec::new_in(ScratchAlloc));
}

pub fn clear_tls_scratch() {
    let _ = KARATSUBA_SCRATCH.try_with(|cell| {
        let old = std::mem::replace(&mut *cell.borrow_mut(), Vec::new_in(ScratchAlloc));
        drop(old);
    });
}

#[inline]
fn trimmed_len(s: &[u64]) -> usize {
    let mut n = s.len();
    while n > 0 && s[n - 1] == 0 {
        n -= 1;
    }
    n
}

#[inline]
fn add_slices_into(kernels: &Kernels, dst: &mut [u64], a: &[u64], b: &[u64]) -> usize {
    let la = trimmed_len(a);
    let lb = trimmed_len(b);
    if la == 0 {
        dst[..lb].copy_from_slice(&b[..lb]);
        return lb;
    }
    if lb == 0 {
        dst[..la].copy_from_slice(&a[..la]);
        return la;
    }
    let (long, short) = if la >= lb {
        (&a[..la], &b[..lb])
    } else {
        (&b[..lb], &a[..la])
    };
    let nl = long.len();
    let ns = short.len();
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(nl as u64, AtomicOrd::Relaxed);
    }
    let mut carry = unsafe {
        (kernels.active.add_n)(dst.as_mut_ptr(), long.as_ptr(), short.as_ptr(), ns as u64)
    };
    for i in ns..nl {
        let (s, ov) = long[i].overflowing_add(carry);
        dst[i] = s;
        carry = ov as u64;
    }
    if carry != 0 {
        dst[nl] = carry;
        nl + 1
    } else {
        trimmed_len(&dst[..nl])
    }
}

#[inline]
fn sub_in_place_slice(kernels: &Kernels, dst: &mut [u64], sub: &[u64]) {
    let ns = trimmed_len(sub);
    if ns == 0 {
        return;
    }
    debug_assert!(dst.len() >= ns);
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(dst.len() as u64, AtomicOrd::Relaxed);
    }
    let mut borrow = unsafe {
        (kernels.active.sub_n)(dst.as_mut_ptr(), dst.as_ptr(), sub.as_ptr(), ns as u64)
    };
    let mut i = ns;
    while borrow != 0 && i < dst.len() {
        let (d, un) = dst[i].overflowing_sub(borrow);
        dst[i] = d;
        borrow = un as u64;
        i += 1;
    }
    debug_assert_eq!(borrow, 0);
}

fn karatsuba_into(
    kernels: &Kernels,
    out: &mut [u64],
    a: &[u64],
    b: &[u64],
    scratch: &mut [u64],
    cutoff: usize,
) {
    let la = trimmed_len(a);
    let lb = trimmed_len(b);
    if la == 0 || lb == 0 {
        out.fill(0);
        return;
    }
    let a = &a[..la];
    let b = &b[..lb];
    if la < lb {
        karatsuba_into(kernels, out, b, a, scratch, cutoff);
        return;
    }
    if lb <= cutoff || (lb == cutoff + 1 && la <= cutoff + 1 && b[lb - 1] <= 1) {
        mul_base_dispatched(kernels, &mut out[..(la + lb)], a, b);
        out[(la + lb)..].fill(0);
        return;
    }
    if la >= 2 * lb {
        out.fill(0);
        let prod_cap = 2 * lb + 2;
        let (prod_buf, sub_scratch) = scratch.split_at_mut(prod_cap);
        let mut offset = 0usize;
        while offset < la {
            let end = (offset + lb).min(la);
            let clen = end - offset;
            karatsuba_into(
                kernels,
                &mut prod_buf[..(clen + lb + 1)],
                &a[offset..end],
                b,
                sub_scratch,
                cutoff,
            );
            let plen = trimmed_len(&prod_buf[..(clen + lb + 1)]);
            add_into_at(kernels, out, offset, &prod_buf[..plen]);
            offset = end;
        }
        return;
    }

    let half = la.div_ceil(2);
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half.min(lb));

    // Compute z0 = a0 * b0 and z2 = a1 * b1 directly into disjoint halves of `out`!
    let z0_cap = a0.len() + b0.len();
    karatsuba_into(kernels, &mut out[..z0_cap], a0, b0, scratch, cutoff);
    let z2_cap = a1.len() + b1.len();
    if !a1.is_empty() && !b1.is_empty() {
        out[z0_cap..(2 * half)].fill(0);
        karatsuba_into(
            kernels,
            &mut out[(2 * half)..(2 * half + z2_cap)],
            a1,
            b1,
            scratch,
            cutoff,
        );
        out[(2 * half + z2_cap)..].fill(0);
    } else {
        out[z0_cap..].fill(0);
    }

    let sa_cap = half + 1;
    let z1_cap = 2 * sa_cap + 1;
    let (sa_buf, rem1) = scratch.split_at_mut(sa_cap);
    let (sb_buf, rem2) = rem1.split_at_mut(sa_cap);
    let (z1_buf, sub_scratch) = rem2.split_at_mut(z1_cap);

    let lsa = add_slices_into(kernels, sa_buf, a0, a1);
    let lsb = add_slices_into(kernels, sb_buf, b0, b1);
    let z1_len_cap = lsa + lsb + 1;
    karatsuba_into(
        kernels,
        &mut z1_buf[..z1_len_cap],
        &sa_buf[..lsa],
        &sb_buf[..lsb],
        sub_scratch,
        cutoff,
    );
    sub_in_place_slice(kernels, &mut z1_buf[..z1_len_cap], &out[..z0_cap]);
    if !a1.is_empty() && !b1.is_empty() {
        sub_in_place_slice(
            kernels,
            &mut z1_buf[..z1_len_cap],
            &out[(2 * half)..(2 * half + z2_cap)],
        );
    }
    let z1_len = trimmed_len(&z1_buf[..z1_len_cap]);
    add_into_at(kernels, out, half, &z1_buf[..z1_len]);
}

fn sqr_karatsuba_into(
    kernels: &Kernels,
    out: &mut [u64],
    a: &[u64],
    scratch: &mut [u64],
    cutoff: usize,
) {
    let la = trimmed_len(a);
    if la == 0 {
        out.fill(0);
        return;
    }
    let a = &a[..la];
    if la <= cutoff {
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add(((la * (la + 1)) / 2) as u64, AtomicOrd::Relaxed);
        }
        unsafe {
            (kernels.active.sqr_base)(
                out.as_mut_ptr(),
                a.as_ptr(),
                la as u64,
                kernels.sqr_rows_max(),
            );
        }
        out[(2 * la)..].fill(0);
        return;
    }
    let half = la.div_ceil(2);
    let (a0, a1) = a.split_at(half);

    let z0_cap = 2 * a0.len();
    sqr_karatsuba_into(kernels, &mut out[..z0_cap], a0, scratch, cutoff);
    let z2_cap = 2 * a1.len();
    if !a1.is_empty() {
        out[z0_cap..(2 * half)].fill(0);
        sqr_karatsuba_into(
            kernels,
            &mut out[(2 * half)..(2 * half + z2_cap)],
            a1,
            scratch,
            cutoff,
        );
        out[(2 * half + z2_cap)..].fill(0);
    } else {
        out[z0_cap..].fill(0);
    }

    let sa_cap = half + 1;
    let z1_cap = 2 * sa_cap + 1;
    let (sa_buf, rem1) = scratch.split_at_mut(sa_cap);
    let (z1_buf, sub_scratch) = rem1.split_at_mut(z1_cap);

    let lsa = add_slices_into(kernels, sa_buf, a0, a1);
    let z1_len_cap = 2 * lsa + 1;
    sqr_karatsuba_into(
        kernels,
        &mut z1_buf[..z1_len_cap],
        &sa_buf[..lsa],
        sub_scratch,
        cutoff,
    );
    sub_in_place_slice(kernels, &mut z1_buf[..z1_len_cap], &out[..z0_cap]);
    if !a1.is_empty() {
        sub_in_place_slice(
            kernels,
            &mut z1_buf[..z1_len_cap],
            &out[(2 * half)..(2 * half + z2_cap)],
        );
    }
    let z1_len = trimmed_len(&z1_buf[..z1_len_cap]);
    add_into_at(kernels, out, half, &z1_buf[..z1_len]);
}

/// Portable Karatsuba multiplication directly into `out[..(la + lb + 1)]`, returning trimmed len.
pub fn mul_limbs_karatsuba_into(
    kernels: &Kernels,
    out: &mut [u64],
    a: &[u64],
    b: &[u64],
    basecase_cutoff: usize,
) -> usize {
    let la = trimmed_len(a);
    let lb = trimmed_len(b);
    if la == 0 || lb == 0 {
        return 0;
    }
    let a = &a[..la];
    let b = &b[..lb];
    let cutoff = basecase_cutoff.max(4);
    let cap = la + lb + 1;
    debug_assert!(out.len() >= cap);
    if la.min(lb) <= cutoff {
        let (long, short) = if la >= lb { (a, b) } else { (b, a) };
        mul_base_dispatched(kernels, &mut out[..(la + lb)], long, short);
        out[(la + lb)..cap].fill(0);
        return trimmed_len(&out[..(la + lb)]);
    }
    let max_l = la.max(lb);
    let eff_cutoff = if max_l <= 2 * cutoff.max(1) && max_l <= 128 {
        cutoff.max(1)
    } else {
        cutoff
    };
    let need_scratch = 8 * max_l + 128;
    KARATSUBA_SCRATCH.with(|cell| {
        if let Ok(mut guard) = cell.try_borrow_mut() {
            if guard.len() < need_scratch {
                guard.resize(need_scratch, 0);
            }
            karatsuba_into(
                kernels,
                &mut out[..cap],
                a,
                b,
                &mut guard[..need_scratch],
                eff_cutoff,
            );
        } else {
            let mut tmp = vec![0u64; need_scratch];
            karatsuba_into(kernels, &mut out[..cap], a, b, &mut tmp, eff_cutoff);
        }
    });
    trimmed_len(&out[..cap])
}

/// Portable Karatsuba multiplication down to `basecase_cutoff` limbs (Section 5 & Section 7).
pub fn mul_limbs_karatsuba(
    kernels: &Kernels,
    a: &[u64],
    b: &[u64],
    basecase_cutoff: usize,
) -> LimbVec {
    let la = trimmed_len(a);
    let lb = trimmed_len(b);
    if la == 0 || lb == 0 {
        return empty_limb_vec();
    }
    let cap = la + lb + 1;
    let mut out = take_limb_buf_uninit(cap);
    unsafe {
        out.set_len(cap);
    }
    let len = mul_limbs_karatsuba_into(kernels, &mut out, &a[..la], &b[..lb], basecase_cutoff);
    out.truncate(len);
    out
}

pub fn sqr_limbs_karatsuba(kernels: &Kernels, a: &[u64], basecase_cutoff: usize) -> LimbVec {
    let la = trimmed_len(a);
    if la == 0 {
        return empty_limb_vec();
    }
    let a = &a[..la];
    let cutoff = basecase_cutoff.max(4);
    if la <= cutoff {
        return sqr_limbs_basecase(kernels, a);
    }
    let need_scratch = 8 * la + 128;
    let cap = 2 * la + 1;
    let mut out = take_limb_buf_uninit(cap);
    unsafe {
        out.set_len(cap);
    }
    KARATSUBA_SCRATCH.with(|cell| {
        if let Ok(mut guard) = cell.try_borrow_mut() {
            if guard.len() < need_scratch {
                guard.resize(need_scratch, 0);
            }
            sqr_karatsuba_into(kernels, &mut out, a, &mut guard[..need_scratch], cutoff);
        } else {
            let mut tmp = vec![0u64; need_scratch];
            sqr_karatsuba_into(kernels, &mut out, a, &mut tmp, cutoff);
        }
    });
    trim_limbs(&mut out);
    out
}

#[inline]
pub fn add_into_at(kernels: &Kernels, dst: &mut [u64], offset: usize, addend: &[u64]) {
    if addend.is_empty() {
        return;
    }
    let n = addend.len().min(dst.len().saturating_sub(offset));
    let mut carry = unsafe {
        (kernels.active.add_n)(
            dst.as_mut_ptr().add(offset),
            dst.as_ptr().add(offset),
            addend.as_ptr(),
            n as u64,
        )
    };
    let mut idx = offset + n;
    while carry > 0 && idx < dst.len() {
        let (s, ov) = dst[idx].overflowing_add(carry);
        dst[idx] = s;
        carry = ov as u64;
        idx += 1;
    }
}

/// Modular inverse of an odd `u64` modulo `2^64` via 5 Newton steps.
#[inline]
pub const fn inv_mod_u64(odd: u64) -> u64 {
    debug_assert!((odd & 1) == 1);
    let mut x = odd; // odd * odd ≡ 1 (mod 8) -> 3 bits
    x = x.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(x))); // 6 bits
    x = x.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(x))); // 12 bits
    x = x.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(x))); // 24 bits
    x = x.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(x))); // 48 bits
    x.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(x))) // 64 bits
}

/// Sequential low-end exact Hensel division of `num` by odd positive `factor` (Section 4).
/// Negates `num[..q_len]` modulo `2^{64 * q_len}` and cancels low limbs via single-pass `addmul_1`.
pub fn div_exact_hensel_seq(kernels: &Kernels, num: &[u64], factor: &[u64]) -> LimbVec {
    if num.is_empty() {
        return empty_limb_vec();
    }
    debug_assert!(!factor.is_empty() && (factor[0] & 1) == 1);
    if factor.len() == 1 && factor[0] == 1 {
        let mut out = take_limb_buf_uninit(num.len());
        out.extend_from_slice(num);
        return out;
    }
    debug_assert!(num.len() >= factor.len());
    let f_len = factor.len();
    let q_len = num.len() - f_len + 1;
    let neg_inv0 = inv_mod_u64(factor[0]).wrapping_neg();

    let mut q = take_limb_buf_uninit(q_len);
    unsafe {
        q.set_len(q_len);
    }

    let mut rem_stack = [0u64; 256];
    let mut rem_heap: Vec<u64, ScratchAlloc> = if q_len > 256 {
        let mut v = Vec::with_capacity_in(q_len, ScratchAlloc);
        unsafe {
            v.set_len(q_len);
        }
        v
    } else {
        Vec::new_in(ScratchAlloc)
    };
    let rem: &mut [u64] = if q_len <= 256 {
        &mut rem_stack[..q_len]
    } else {
        &mut rem_heap[..q_len]
    };

    let mut borrow = 0u64;
    for i in 0..q_len {
        let (d1, b1) = 0u64.overflowing_sub(num[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        rem[i] = d2;
        borrow = u64::from(b1) + u64::from(b2);
    }

    if kernels.count_work {
        let ops = (0..q_len).map(|i| f_len.min(q_len - i) as u64).sum::<u64>();
        kernels.stats.limb_ops.fetch_add(ops, AtomicOrd::Relaxed);
    }

    if q_len >= 32 && f_len >= 16 {
        const BLK: usize = 16;
        let mut tri = [0u64; BLK * 2];
        let prod_cap = f_len + BLK;
        let mut prod_stack = [0u64; 512];
        let mut prod_heap: Vec<u64, ScratchAlloc> = if prod_cap > 512 {
            let mut v = Vec::with_capacity_in(prod_cap, ScratchAlloc);
            unsafe {
                v.set_len(prod_cap);
            }
            v
        } else {
            Vec::new_in(ScratchAlloc)
        };
        let prod_buf: &mut [u64] = if prod_cap <= 512 {
            &mut prod_stack[..prod_cap]
        } else {
            &mut prod_heap[..prod_cap]
        };

        let mut i = 0usize;
        while i < q_len {
            let b = BLK.min(q_len - i);
            if rem[i..i + b].iter().all(|&w| w == 0) {
                q[i..i + b].fill(0);
                i += b;
                continue;
            }
            tri[..b].copy_from_slice(&rem[i..i + b]);
            tri[b..2 * b].fill(0);
            for j in 0..b {
                let rj = tri[j];
                if rj == 0 {
                    q[i + j] = 0;
                    continue;
                }
                let qj = rj.wrapping_mul(neg_inv0);
                q[i + j] = qj;
                let cy = unsafe {
                    (kernels.active.addmul_1)(
                        tri.as_mut_ptr().add(j),
                        factor.as_ptr(),
                        b as u64,
                        qj,
                    )
                };
                tri[j + b] = cy;
            }

            let rem_after = q_len - (i + b);
            if rem_after > 0 {
                let add_tri = b.min(rem_after);
                let mut cy = unsafe {
                    (kernels.active.add_n)(
                        rem.as_mut_ptr().add(i + b),
                        rem.as_ptr().add(i + b),
                        tri.as_ptr().add(b),
                        add_tri as u64,
                    )
                };
                let mut k = i + b + add_tri;
                while cy > 0 && k < q_len {
                    let (s, c) = rem[k].overflowing_add(cy);
                    rem[k] = s;
                    cy = u64::from(c);
                    k += 1;
                }

                let max_f = f_len.min(q_len - i);
                if max_f > b {
                    let rem_f = max_f - b;
                    let lens = kernels.mul_base_lens(rem_f, b);
                    unsafe {
                        (kernels.active.mul_base)(
                            prod_buf.as_mut_ptr(),
                            factor.as_ptr().add(b),
                            q.as_ptr().add(i),
                            &lens,
                        );
                    }
                    let add_prod = (rem_f + b).min(rem_after);
                    let mut cy2 = unsafe {
                        (kernels.active.add_n)(
                            rem.as_mut_ptr().add(i + b),
                            rem.as_ptr().add(i + b),
                            prod_buf.as_ptr(),
                            add_prod as u64,
                        )
                    };
                    let mut k2 = i + b + add_prod;
                    while cy2 > 0 && k2 < q_len {
                        let (s, c) = rem[k2].overflowing_add(cy2);
                        rem[k2] = s;
                        cy2 = u64::from(c);
                        k2 += 1;
                    }
                }
            }
            i += b;
        }
        drop(prod_heap);
    } else {
        for i in 0..q_len {
            let r_i = rem[i];
            if r_i == 0 {
                q[i] = 0;
                continue;
            }
            let qi = r_i.wrapping_mul(neg_inv0);
            q[i] = qi;
            let max_f = f_len.min(q_len - i);
            let mut cy = unsafe {
                (kernels.active.addmul_1)(
                    rem.as_mut_ptr().add(i),
                    factor.as_ptr(),
                    max_f as u64,
                    qi,
                )
            };
            let mut k = i + max_f;
            while cy > 0 && k < q_len {
                let (s, c) = rem[k].overflowing_add(cy);
                rem[k] = s;
                cy = u64::from(c);
                k += 1;
            }
        }
    }

    drop(rem_heap);
    trim_limbs(&mut q);
    let _ = mod61_sub(0, 0);
    let _ = mod61_pow2(0);
    q
}

#[cfg(test)]
mod shifted_add_tests {
    use super::*;

    /// The one-pass shifted add and subtract of large numbers gives the same value and check
    /// word as the serial routine, for both signs, any shift, and operands of unequal length.
    #[test]
    fn one_pass_shifted_add_matches_the_serial_sum() {
        let kernels = Kernels::reference_only();
        let prof = crate::tier::Profile::synthetic(
            4,
            8,
            32 * 1024,
            512 * 1024,
            8 * 1024 * 1024,
            512 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = crate::tier::Shares::from_profile(&prof);
        let sched = crate::par::Scheduler::new(&shares, None);
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let shifts = [0u64, 1, 63, 64, 65, 16_384 * 64, 16_384 * 64 + 7, 40_000 * 64 - 1];
        for round in 0..48u64 {
            let la = 30_000 + (next() % 12_000) as usize;
            let lb = if round % 3 == 0 { la } else { 2_000 + (next() % 45_000) as usize };
            let mut al: Vec<u64> = (0..la).map(|_| next()).collect();
            let mut bl: Vec<u64> = (0..lb).map(|_| next()).collect();
            if round % 5 == 0 {
                // Long runs of ones and zeros make carries and borrows travel across chunks.
                al.iter_mut().skip(100).for_each(|w| *w = u64::MAX);
                bl.iter_mut().skip(50).take(20_000).for_each(|w| *w = 0);
            }
            *al.last_mut().unwrap() |= 1;
            *bl.last_mut().unwrap() |= 1;
            let sa = shifts[(round % 8) as usize];
            let sb = shifts[((round / 8 + round) % 8) as usize];
            let a = BigNum::from_limbs(al, round % 2 == 1, sa, &kernels);
            let b = BigNum::from_limbs(bl, round % 4 >= 2, sb, &kernels);
            let want = add_bignum(&kernels, &a, &b);
            let got = add_bignum_par(&kernels, &a, &b, &sched, 8);
            let want_m = want.materialize_shift_par(&kernels, &sched, 1);
            assert_eq!(got.shift, 0);
            assert_eq!(got.limbs, want_m.limbs, "round {round}: sa={sa} sb={sb}");
            assert_eq!(got.neg, want.neg, "round {round}");
            assert_eq!(got.w0(), want.w0(), "round {round}");
            assert_eq!(compute_limbs_w0(&kernels, &got.limbs), got.limbs_w0(), "round {round}");
        }
        // Equal magnitudes of opposite sign cancel.
        let l: Vec<u64> = (0..40_000).map(|_| next() | 1).collect();
        let a = BigNum::from_limbs(l.clone(), false, 129, &kernels);
        let b = BigNum::from_limbs(l, true, 129, &kernels);
        assert!(add_bignum_par(&kernels, &a, &b, &sched, 8).is_zero());
    }

    /// The sum made in the buffer of its second operand is the sum made in a buffer of its own,
    /// limbs, sign and check word, for both signs, any shift of the first operand and operands
    /// of unequal length; it is made in that buffer where there is room, and without the room
    /// it comes out the same the other way.
    #[test]
    fn sum_made_in_place_matches_the_sum() {
        let kernels = Kernels::reference_only();
        let prof = crate::tier::Profile::synthetic(
            4,
            8,
            32 * 1024,
            512 * 1024,
            8 * 1024 * 1024,
            512 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = crate::tier::Shares::from_profile(&prof);
        let sched = crate::par::Scheduler::new(&shares, None);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let check = |a: &BigNum, b0: &BigNum, workers: usize, tag: &str| {
            let want = add_bignum(&kernels, a, b0);
            let both = !a.is_zero() && !b0.is_zero();
            assert!(want.shift == 0 || !both, "{tag}");
            let n = Shifted::new(&a.limbs, a.shift).len().max(b0.limbs.len());
            for with_room in [true, false] {
                let mut b = b0.clone();
                if with_room {
                    let room = sum_room(a, b.limbs.len());
                    assert_eq!(room, n + 1, "{tag}");
                    b.limbs.reserve_exact(room - b.limbs.len());
                } else {
                    b.limbs.shrink_to_fit();
                }
                let buf = b.limbs.as_ptr();
                let got = add_bignum_into_par(&kernels, a, b, &sched, workers);
                assert_eq!(got.shift, want.shift, "{tag}");
                assert_eq!(got.limbs, want.limbs, "{tag} room={with_room}");
                assert_eq!(got.neg, want.neg, "{tag} room={with_room}");
                assert_eq!(got.w0(), want.w0(), "{tag} room={with_room}");
                assert_eq!(compute_limbs_w0(&kernels, &got.limbs), got.limbs_w0(), "{tag}");
                if with_room && both && b0.shift == 0 && n >= 8_192 && !got.is_zero() {
                    assert!(std::ptr::eq(got.limbs.as_ptr(), buf), "{tag}: not made in place");
                }
            }
        };
        let shifts = [0u64, 1, 63, 64, 65, 9_000 * 64, 9_000 * 64 + 7, 25_000 * 64 - 1];
        for round in 0..64u64 {
            let la = 20_000 + (next() % 12_000) as usize;
            let lb = match round % 4 {
                0 => la,
                // Short: whole chunks of the sum lie above it.
                1 => 300 + (next() % 4_000) as usize,
                2 => 9_000 + (next() % 45_000) as usize,
                // Long: it reaches above the first operand, shift and all.
                _ => la + 26_000 + (next() % 9_000) as usize,
            };
            let mut al: Vec<u64> = (0..la).map(|_| next()).collect();
            let mut bl: Vec<u64> = (0..lb).map(|_| next()).collect();
            match round % 5 {
                // Long runs of ones and zeros make carries and borrows travel across chunks.
                0 => {
                    al.iter_mut().skip(100).for_each(|w| *w = u64::MAX);
                    bl.iter_mut().skip(50).take(20_000).for_each(|w| *w = 0);
                }
                // All ones on both sides: with like signs the sum is a limb longer than either.
                1 => {
                    al.iter_mut().for_each(|w| *w = u64::MAX);
                    bl.iter_mut().for_each(|w| *w = u64::MAX);
                }
                _ => {}
            }
            *al.last_mut().unwrap() |= 1;
            *bl.last_mut().unwrap() |= 1;
            let sa = shifts[(round % 8) as usize];
            let a = BigNum::from_limbs(al, round % 2 == 1, sa, &kernels);
            let b = BigNum::from_limbs(bl, (round / 2) % 2 == 1, 0, &kernels);
            let workers = if round % 7 == 3 { 1 } else { 8 };
            check(&a, &b, workers, &format!("round {round}: la={la} lb={lb} sa={sa}"));
        }
        // The second operand is the larger and yet a limb shorter than the shifted first, whose
        // top limb is empty: the last chunk holds that limb alone.
        let mut al: Vec<u64> = (0..8_192).map(|_| next()).collect();
        *al.last_mut().unwrap() = 1;
        let mut bl: Vec<u64> = (0..8_192).map(|_| next()).collect();
        *bl.last_mut().unwrap() = u64::MAX;
        let a = BigNum::from_limbs(al, true, 1, &kernels);
        let b = BigNum::from_limbs(bl, false, 0, &kernels);
        check(&a, &b, 8, "larger and shorter");
        // A second operand with a shift of its own goes the other way.
        let al: Vec<u64> = (0..30_000).map(|_| next() | 1).collect();
        let bl: Vec<u64> = (0..30_000).map(|_| next() | 1).collect();
        let a = BigNum::from_limbs(al, false, 77, &kernels);
        let b = BigNum::from_limbs(bl, true, 130, &kernels);
        check(&a, &b, 8, "shifted second operand");
        // Equal magnitudes of opposite sign cancel, and the buffer is not handed on.
        let l: Vec<u64> = (0..40_000).map(|_| next() | 1).collect();
        let a = BigNum::from_limbs(l.clone(), false, 0, &kernels);
        let mut b = BigNum::from_limbs(l, true, 0, &kernels);
        b.limbs.reserve_exact(8);
        assert!(add_bignum_into_par(&kernels, &a, b, &sched, 8).is_zero());
        // Nothing to add.
        let l: Vec<u64> = (0..40_000).map(|_| next() | 1).collect();
        let a = BigNum::from_limbs(l, true, 65, &kernels);
        check(&a, &BigNum::zero(), 8, "zero second");
        check(&BigNum::zero(), &a, 8, "zero first");
    }

    /// A word for room speaks of the buffers from its `from` up to its `room`, gives the first
    /// of them that room and no second, and ends with its value; small rooms are not asked.
    #[test]
    fn room_asked_holds_for_one_buffer() {
        let word = RoomAsked::new(20_000, 40_000);
        assert_eq!(with_room_asked(19_999), 19_999);
        assert_eq!(with_room_asked(40_001), 40_001);
        assert_eq!(with_room_asked(20_000), 40_000);
        assert_eq!(with_room_asked(20_000), 20_000);
        drop(word);

        let outer = RoomAsked::new(20_000, 40_000);
        {
            let _inner = RoomAsked::new(100_000, 200_000);
            assert_eq!(with_room_asked(30_000), 30_000);
            assert_eq!(with_room_asked(150_000), 200_000);
        }
        assert_eq!(with_room_asked(30_000), 40_000);
        drop(outer);
        assert_eq!(with_room_asked(30_000), 30_000);

        let unused = RoomAsked::new(20_000, 40_000);
        drop(unused);
        assert_eq!(with_room_asked(30_000), 30_000);

        let small = RoomAsked::new(100, 5_000);
        assert_eq!(with_room_asked(4_500), 4_500);
        drop(small);
        let backwards = RoomAsked::new(50_000, 40_000);
        assert_eq!(with_room_asked(45_000), 45_000);
        drop(backwards);

        // A buffer taken under the word has the room, by whichever way the pool serves it.
        let word = RoomAsked::new(20_000, 40_000);
        let v = take_limb_buf_uninit(25_000);
        assert!(v.capacity() >= 40_000);
        let w = take_limb_buf_zeroed(25_000);
        assert_eq!(w.len(), 25_000);
        drop(word);
        put_limb_buf(v);
        put_limb_buf(w);
    }
}
