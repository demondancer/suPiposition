//! Contract:
//! - Inputs: Fixed-point fractional limbs of a number in `[0, 1)` (`frac_bn` representing
//!   `frac * 2^{-64 * in_frac_limbs}`, e.g. `pi - 3`), `int_part` (`3` for pi), `Plan` (`digits`,
//!   `OutputFormat`), `Shares`, `Calibration`, `MulCtx`, random-access `DigitSink`, and optional
//!   `RadixResumeState` + checkpoint callback.
//! - Outputs: Verified decimal digits written to `DigitSink`, overall `Dec Hash` check word
//!   (`floor((int_part + frac) * 10^digits) mod (2^61 - 1)`), and `digits_in_doubt: bool` when the
//!   dropped extra leaf digits are all `0`s or all `9`s.
//! - Data format: Binary tree of fractions `X * 2^{-64 * K(d)}` (`K(d)` carries `ceil(d * log2(10))`
//!   bits plus at least one 64-bit guard limb); every split has `left_leaves = 2^k` so `10^{d_L}`
//!   and its forward transform are built by squaring and cached once per level. From level 6 on
//!   (`d_L = 19 * 2^lv` a whole number of limbs) the product by `10^{d_L} = 5^{d_L} * 2^{d_L}` is
//!   made as one by `5^{d_L}` of the node's fraction without its top `d_L` bits, which only add
//!   the whole number `top * 5^{d_L}` to it (`Pow10Entry`): a wrapped transform 15% shorter and
//!   powers 30% shorter to square.
//! - Work & bytes moved: One exact wrapped (`mod 2^L - 1` NTT) or whole product per internal node
//!   (`O(M(D) log D)` total), parking `x_right` into `Store` on `Loader` while converting
//!   `left_leaves` when out-of-core, shifting every left-half cut fraction by `+-2^{-(b_L + 2)}`
//!   toward the middle of its digits, vectorized `to_digits` at 19-digit leaves, and parent
//!   check-word verification on every node.

use std::alloc::Allocator;
use std::cell::{RefCell, UnsafeCell};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::arena::{ScratchAlloc, SubtreeLifoScope};
use crate::check::{
    Ladder, Rung, compute_limbs_w0, compute_limbs_w0_par, wrapped_product_expected_w0,
};
use crate::limb::{
    BigNum, LimbVec, add_limbs, cmp_limbs, shl_limbs_par, shr_limbs_par, sub_limbs, take_limb_vec,
    trim_limbs,
};
use crate::mul::{CachedTransform, MulCtx, MulError, MulMethod};
use crate::ntt::{NttCachedFwd, NttSpec, ntt_forward_reusable};
use crate::plan::{Calibration, LOG2_10, OutputFormat, Plan, choose_wrapped_ntt_spec};
use crate::refk::{MOD_W0, POW10_19_MOD_W0, mod61_add, mod61_mul, mod61_pow2, mod61_reduce};
use crate::tier::Shares;

pub const LEAF_DIGITS: usize = 19;
pub const POW10_19: u64 = 10_000_000_000_000_000_000;
pub const PACKED19_MAGIC: u64 = 0x3931_4B50_4950_5553; // "SUPIPK19"
const BATCH_MAX_LEAVES: u64 = 16384;
/// The longest fraction, in limbs, that the batch path takes (a subtree of `BATCH_MAX_LEAVES`
/// leaves with its guard limbs); the checks on a batch's parts are held against it.
const BATCH_MAX_K_LIMBS: usize = 16896;
/// The first level whose nodes are multiplied by `5^d` in place of `10^d` (see `Pow10Entry`):
/// from level 6 on, `d = 19 * 2^lv` is a whole number of 64-bit limbs.
const STRIP_MIN_LV: usize = 6;
const _: () = assert!(STRIP_MIN_LV >= 6 && (LEAF_DIGITS << STRIP_MIN_LV).is_multiple_of(64));
/// The fewest nodes a level has for its cached multiplier to be turned (see
/// `Pow10Entry::wrapped_turn`). Turned, the multiplier has its zeros below it, where the
/// forward transform does not pass over them as it does over zeros above, so the level's
/// transform costs more, in proportion to the size of the level's nodes; what the turn saves, one
/// move of every right child, is the same at every level. Measured at 5e9 digits (conversion
/// alone, first round): the level of two nodes turned as well, 60 to 100 ms more than with it
/// left unturned; the level of four nodes left unturned too, no better there and 20 ms worse at
/// 1e9 digits.
const TURN_MIN_NODES: u64 = 3;
/// How many splits in a row may leave a buffer of the frontier where it is. A node's left
/// child is the top limbs of the node's own fraction: instead of being moved to the start of
/// the buffer at every split (a pass over half the node, which in the top waves is a pass
/// through memory), it is passed on as that part of the buffer, and only every third split
/// moves what is left, an eighth, and gives the rest of the buffer back. Until then a buffer
/// holds up to four times what is still in use of it.
const VIEW_LEVELS: usize = 2;
static RADIX_BATCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Trace runs: time the frontier's splits spent in all, and of that inside their products
/// (summed over the splits, which may run several at once).
static RADIX_SPLIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RADIX_SPLIT_PROD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Random-access output writer so leaves write directly at their own byte offset (Section 11).
pub trait DigitSink: Send + Sync {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String>;
    fn flush(&self) -> Result<(), String>;
}

/// In-memory `DigitSink` for unit and Gate 1–3 tests.
pub struct MemoryDigitSink {
    pub buf: Mutex<Vec<u8>>,
    pub fail_writes: std::sync::atomic::AtomicBool,
}

impl MemoryDigitSink {
    pub fn new(size: usize) -> Self {
        Self {
            buf: Mutex::new(vec![0u8; size]),
            fail_writes: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf.into_inner().unwrap()
    }
}

impl DigitSink for MemoryDigitSink {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        if self.fail_writes.load(Ordering::Relaxed) {
            return Err("Simulated output file write failure".to_string());
        }
        let mut g = self.buf.lock().unwrap();
        let off = file_offset as usize;
        if off + bytes.len() > g.len() {
            return Err(format!(
                "Write past preallocated output size ({} > {})",
                off + bytes.len(),
                g.len()
            ));
        }
        g[off..(off + bytes.len())].copy_from_slice(bytes);
        Ok(())
    }

    fn flush(&self) -> Result<(), String> {
        if self.fail_writes.load(Ordering::Relaxed) {
            return Err("Simulated output file flush failure".to_string());
        }
        Ok(())
    }
}

/// Optional fault-injection hook for Gate 2 digit-corruption testing (Section 11 & 15).
pub trait LeafFaultHook: Send + Sync {
    fn on_leaf_word(&self, leaf_idx: u64, attempt: u32, word_19: &mut u64, emitted_w0: &mut u64);
}

#[derive(Clone, Debug)]
pub struct PendingFraction {
    pub start_leaf: u64,
    pub leaf_count: u64,
    pub frac_limbs: Vec<u64>,
    pub frac_k_limbs: usize,
    pub required_w0: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct RadixResumeState {
    pub fractions: Vec<PendingFraction>,
    pub completed_w0: HashMap<(u64, u64), u64>,
}

pub trait RadixCheckpointSink: Send + Sync {
    fn maybe_checkpoint_radix(
        &self,
        pending: &[PendingFraction],
        completed_w0: &[(u64, u64, u64)],
        work_sec_since_last: f64,
    ) -> Result<bool, MulError>;
}

#[derive(Clone, Debug)]
pub struct ConversionResult {
    pub dec_hash_w0: u64,
    pub fractional_w0: u64,
    pub digits_in_doubt: bool,
}

/// Number of 64-bit fraction limbs `K(d)` kept for a node covering `d` decimal digits:
/// `ceil(d * log2(10) / 64) + 2` (at least one full 64-bit guard limb + margin below `10^{-d}`).
#[inline]
pub fn frac_limbs_for_digits(digits: u64) -> usize {
    let b = ((digits as f64) * LOG2_10).ceil() as u64;
    (b.div_ceil(64) as usize) + 2
}

#[inline]
fn bit_ceil_pow10(digits: u64) -> u64 {
    ((digits as f64) * LOG2_10).ceil() as u64
}

/// What the nodes of one level are multiplied by, and its transforms. A node holds a fraction
/// `x`, and its right child is the fraction of `x * 10^d`, `d` the digits of its left child.
/// Since `10^d = 5^d * 2^d`, the top `d` bits of `x` only add a whole number to that product:
/// with `x * 2^d = top + f` (`top` those bits as an integer, `f` the rest as a fraction),
/// `x * 10^d = top * 5^d + f * 5^d`. So from `STRIP_MIN_LV` on the multiplier kept here is `5^d`,
/// and a node multiplies only its limbs below the top `strip_limbs = d / 64`: operands 15% and
/// 30% shorter than with `10^d`, and a wrapped transform 15% shorter. The left child's digits
/// are then `top * 5^d` plus the integer part of that product. Below `STRIP_MIN_LV`,
/// `strip_limbs` is zero and the multiplier is `10^d`.
struct Pow10Entry {
    mult: BigNum,
    /// The multiplier modulo `2^61 - 1`, from its exponent and not from its limbs.
    mult_w0: u64,
    /// `10^d` modulo `2^61 - 1`, for combining left and right check words on balanced splits.
    pow10_w0: u64,
    /// Bits of the multiplier, as the level's transforms were sized for.
    mult_bits: u64,
    strip_limbs: usize,
    /// Low zero limbs `d / 64` of `10^d` when `strip_limbs == 0`.
    zero_lo_limbs: usize,
    whole_method: MulMethod,
    whole_cached: Option<CachedTransform>,
    wrapped_l_bits: Option<u64>,
    wrapped_s_pad: u64,
    /// Limbs the cached wrapped transform's multiplier is turned up by, beyond `wrapped_s_pad`:
    /// those of the right child of the level's full node. Modulo `2^L - 1` a product by
    /// `2^(64 * t)` is the product turned up `t` limbs around its `L` bits, so the right
    /// child's fraction, the top limbs of the plain product, comes out as the lowest limbs of
    /// this one and stays where it was written. Zero where the transform was made unturned.
    wrapped_turn: usize,
    wrapped_spec: Option<NttSpec>,
    wrapped_cached: Option<NttCachedFwd>,
}

/// The check words of the two parts that a node's own product takes its fraction in: of the
/// limbs below `cut` (those multiplied) and of the limbs from `cut` up (those stripped).
#[derive(Clone, Copy)]
struct PartWords {
    cut: usize,
    lo_w0: u64,
    hi_w0: u64,
}

/// Each cut with the next one: the two ends of every piece.
fn cut_pairs(cuts: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    cuts.iter().zip(cuts.iter().skip(1)).map(|(&a, &b)| (a, b))
}

/// The check words of a buffer taken piece by piece: of its limbs `[cuts[i], cuts[i + 1])` for
/// each `i`. The cuts are the places where the node the buffer was made for, and the left
/// children that will be passed on as the top of it (see `VIEW_LEVELS`), are divided by their
/// own products (`chain_cuts`): one pass over the buffer gives each of them its check words.
struct SegWords {
    cuts: Vec<usize>,
    w0s: Vec<u64>,
}

impl SegWords {
    /// The words of `limbs` between the consecutive places of `cuts`, which ascend and end at
    /// `limbs.len()`.
    fn compute(
        kernels: &crate::refk::Kernels,
        sched: &crate::par::Scheduler,
        limbs: &[u64],
        cuts: Vec<usize>,
        par_workers: usize,
    ) -> Self {
        debug_assert!(cut_pairs(&cuts).all(|(a, b)| a < b) && cuts.last() == Some(&limbs.len()));
        let w0s = cut_pairs(&cuts)
            .map(|(a, b)| compute_limbs_w0_par(kernels, sched, &limbs[a..b], par_workers))
            .collect();
        Self { cuts, w0s }
    }

    /// The check word of the limbs `[from, to)`, where both are cuts.
    fn range(&self, from: usize, to: usize) -> Option<u64> {
        let a = self.cuts.binary_search(&from).ok()?;
        let b = self.cuts.binary_search(&to).ok()?;
        if a > b {
            return None;
        }
        let mut w = 0u64;
        for i in (a..b).rev() {
            let len = (self.cuts[i + 1] - self.cuts[i]) as u64;
            w = mod61_add(mod61_mul(w, mod61_pow2(len * 64)), self.w0s[i]);
        }
        Some(w)
    }

    /// The two check words of the fraction that is the limbs `[off, off + k)`, cut `cut` limbs
    /// from its start.
    fn parts(&self, off: usize, cut: usize, k: usize) -> Option<PartWords> {
        Some(PartWords {
            cut,
            lo_w0: self.range(off, off + cut)?,
            hi_w0: self.range(off + cut, off + k)?,
        })
    }

    /// Limb `idx` went from `old` to `new`. False where the limb is not one of these pieces.
    fn limb_changed(&mut self, idx: usize, old: u64, new: u64) -> bool {
        if self.cuts.is_empty() || idx < self.cuts[0] || idx >= self.cuts[self.cuts.len() - 1] {
            return false;
        }
        let i = match self.cuts.binary_search(&idx) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let place = mod61_pow2(((idx - self.cuts[i]) as u64) * 64);
        let was = mod61_mul(mod61_reduce(old as u128), place);
        let is = mod61_mul(mod61_reduce(new as u128), place);
        self.w0s[i] = mod61_add(crate::refk::mod61_sub(self.w0s[i], was), is);
        true
    }

    /// The limbs from `from` up were all set to zero. False where `from` is not a cut.
    fn zeroed_from(&mut self, from: usize) -> bool {
        match self.cuts.binary_search(&from) {
            Ok(a) => {
                for w in self.w0s.iter_mut().skip(a) {
                    *w = 0;
                }
                true
            }
            Err(_) => false,
        }
    }
}

/// What `apply_guard_eta_tracked` did to a fraction.
enum EtaChange {
    /// The limbs from `first` up changed, `n` of them: `old[..n]` are the values they had.
    Limbs { first: usize, old: [u64; 4], n: usize },
    /// More limbs changed than are told here.
    Many,
    /// The fraction was below the guard and is all zero now.
    Zeroed,
}

/// `limbs * 2^shift_bits` modulo `2^l_bits - 1`: the limbs shifted up, and what passes bit
/// `l_bits` brought around to bit 0. `l_bits` is a whole number of limbs and not less than the
/// bits of `limbs`.
fn shl_limbs_cyclic(
    limbs: &[u64],
    shift_bits: u64,
    l_bits: u64,
    sched: &crate::par::Scheduler,
    par_workers: usize,
) -> LimbVec {
    debug_assert!(l_bits.is_multiple_of(64) && shift_bits < l_bits);
    let kl = (l_bits / 64) as usize;
    let mut v = shl_limbs_par(limbs, shift_bits, sched, par_workers);
    while v.len() > kl {
        let around: Vec<u64> = v[kl..].to_vec();
        v.truncate(kl);
        let mut carry = false;
        for (i, w) in v.iter_mut().enumerate() {
            if i >= around.len() && !carry {
                break;
            }
            let add = around.get(i).copied().unwrap_or(0);
            let (s1, o1) = w.overflowing_add(add);
            let (s2, o2) = s1.overflowing_add(u64::from(carry));
            *w = s2;
            carry = o1 | o2;
        }
        if carry {
            // once more around: the sum passed bit `l_bits` again
            v.push(1);
        }
    }
    v
}

impl Pow10Entry {
    fn release_store(&mut self, store: Option<&crate::store::Store>) {
        if let Some(c) = &mut self.whole_cached {
            c.release_store(store);
        }
        if let Some(w) = &mut self.wrapped_cached {
            w.release_store(store);
        }
    }
}

/// Convert `int_part + frac_bn * 2^{-64 * in_frac_limbs}` to `plan.digits` decimal digits!
#[allow(clippy::too_many_arguments)]
pub fn run_conversion(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    int_part: u64,
    frac_bn: &BigNum,
    in_frac_limbs: usize,
    sink: &dyn DigitSink,
    leaf_fault_hook: Option<&dyn LeafFaultHook>,
    resume: Option<&RadixResumeState>,
    ckpt_sink: Option<&dyn RadixCheckpointSink>,
) -> Result<ConversionResult, MulError> {
    let num_leaves = if plan.digits == 0 {
        1u64
    } else {
        plan.digits.div_ceil(LEAF_DIGITS as u64) + 1
    };
    let root_k = frac_limbs_for_digits(num_leaves * (LEAF_DIGITS as u64));
    let root_frac_limbs = resize_fraction_limbs(&frac_bn.limbs, in_frac_limbs, root_k);
    run_conversion_inner(
        ctx,
        plan,
        shares,
        cal,
        int_part,
        root_frac_limbs,
        0,
        sink,
        leaf_fault_hook,
        resume,
        ckpt_sink,
    )
}

/// Convert `int_part + frac_bn * 2^{-64 * in_frac_limbs}` consuming `frac_bn` in-place without cloning!
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_conversion_owned(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    int_part: u64,
    frac_bn: BigNum,
    in_frac_limbs: usize,
    sink: &dyn DigitSink,
    leaf_fault_hook: Option<&dyn LeafFaultHook>,
    resume: Option<&RadixResumeState>,
    ckpt_sink: Option<&dyn RadixCheckpointSink>,
) -> Result<ConversionResult, MulError> {
    let num_leaves = if plan.digits == 0 {
        1u64
    } else {
        plan.digits.div_ceil(LEAF_DIGITS as u64) + 1
    };
    let root_k = frac_limbs_for_digits(num_leaves * (LEAF_DIGITS as u64));
    let mut raw_limbs = frac_bn.into_limbs();
    let (root_frac_limbs, root_off) =
        if in_frac_limbs >= root_k && raw_limbs.len() <= in_frac_limbs {
            raw_limbs.resize(in_frac_limbs, 0);
            (raw_limbs, in_frac_limbs - root_k)
        } else {
            (
                resize_fraction_limbs_owned(raw_limbs, in_frac_limbs, root_k),
                0,
            )
        };
    run_conversion_inner(
        ctx,
        plan,
        shares,
        cal,
        int_part,
        root_frac_limbs,
        root_off,
        sink,
        leaf_fault_hook,
        resume,
        ckpt_sink,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_conversion_inner(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    int_part: u64,
    root_frac_limbs: LimbVec,
    root_off: usize,
    sink: &dyn DigitSink,
    leaf_fault_hook: Option<&dyn LeafFaultHook>,
    resume: Option<&RadixResumeState>,
    ckpt_sink: Option<&dyn RadixCheckpointSink>,
) -> Result<ConversionResult, MulError> {
    // Write output file prefix (`"3."` or `"3"` for Text, or 16-byte `SUPIPK19` header for Packed19)
    // on fresh runs (`resume.is_none()`); resumed runs keep already-flushed bytes untouched!
    if resume.is_none() {
        match plan.format {
            OutputFormat::Text => {
                debug_assert!(int_part < 10);
                let lead = b'0' + (int_part as u8);
                let header: &[u8] = if plan.digits == 0 {
                    &[lead]
                } else {
                    &[lead, b'.']
                };
                sink.write_at(0, header).map_err(MulError::FatalStop)?;
            }
            OutputFormat::Packed19 => {
                let mut hdr = [0u8; 16];
                hdr[0..8].copy_from_slice(&PACKED19_MAGIC.to_le_bytes());
                hdr[8..16].copy_from_slice(&plan.digits.to_le_bytes());
                sink.write_at(0, &hdr).map_err(MulError::FatalStop)?;
            }
        }
    }

    // Even when `plan.digits == 0`, convert 1 leaf (19 guard digits) to detect digits in doubt!
    let kept_full_leaves = plan.digits / (LEAF_DIGITS as u64);
    let rem_digits = (plan.digits % (LEAF_DIGITS as u64)) as usize;
    let num_leaves = if plan.digits == 0 {
        1u64
    } else {
        plan.digits.div_ceil(LEAF_DIGITS as u64) + 1
    };

    let total_tree_digits = num_leaves * (LEAF_DIGITS as u64);
    let root_k = frac_limbs_for_digits(total_tree_digits);

    if num_leaves == 1 {
        let mut guard_tail = Mutex::new(vec![0u8; LEAF_DIGITS]);
        let w0 = eval_leaf(
            ctx,
            plan,
            0,
            &root_frac_limbs[root_off..],
            root_k,
            kept_full_leaves,
            rem_digits,
            sink,
            &mut guard_tail,
            leaf_fault_hook,
            0,
        )?;
        crate::limb::put_limb_buf(root_frac_limbs);
        sink.flush().map_err(MulError::FatalStop)?;
        let tail = guard_tail.into_inner().unwrap();
        let doubt = is_all_zeros_or_nines(&tail[..LEAF_DIGITS]);
        let dec_hash = mod61_reduce(int_part as u128);
        return Ok(ConversionResult {
            dec_hash_w0: dec_hash,
            fractional_w0: w0,
            digits_in_doubt: doubt,
        });
    }

    // Build per-level powers of ten `10^{19 * 2^lv}` for `lv = 0 ..= max_lv` by squaring once!
    let do_trace = std::env::var_os("SUPI_TRACE").is_some();
    let t_p10 = Instant::now();
    let max_lv = (num_leaves - 1).ilog2() as usize;
    let skip_low_cached = leaf_fault_hook.is_none() && resume.is_none() && num_leaves > BATCH_MAX_LEAVES;
    let mut pow10_levels =
        build_pow10_table(ctx, plan, shares, cal, max_lv, num_leaves, skip_low_cached)?;
    if do_trace {
        eprintln!(
            "[TRACE CONV] pow10_table(max_lv={max_lv})={:.2}ms",
            t_p10.elapsed().as_secs_f64() * 1e3
        );
    }

    let guard_bytes_len = LEAF_DIGITS + if rem_digits > 0 { LEAF_DIGITS - rem_digits } else { 0 };
    let guard_tail: Mutex<Vec<u8>> = Mutex::new(vec![0u8; guard_bytes_len]);

    let resume_completed: HashMap<(u64, u64), u64> =
        resume.map(|r| r.completed_w0.clone()).unwrap_or_default();
    let spine_completed: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(
        resume_completed
            .iter()
            .map(|(&(s, c), &w)| (s, c, w))
            .collect(),
    );
    let resume_fracs: HashMap<(u64, u64), PendingFraction> = resume
        .map(|r| {
            r.fractions
                .iter()
                .map(|f| ((f.start_leaf, f.leaf_count), f.clone()))
                .collect()
        })
        .unwrap_or_default();

    let last_ckpt = Mutex::new(Instant::now());
    let t_cn0 = Instant::now();
    let ram_fits_conv_numbers =
        (root_k as u64).saturating_mul(20) <= plan.ram_numbers_budget;
    let can_use_fast_frontier = (!plan.out_of_core || ram_fits_conv_numbers)
        && ctx.fault_hook.is_none()
        && leaf_fault_hook.is_none()
        && resume_completed.is_empty()
        && resume_fracs.is_empty()
        && num_leaves > BATCH_MAX_LEAVES
        && (ckpt_sink.is_none()
            || (plan.checkpoint_mult > 0.0
                && (num_leaves as f64) * 3.0e-8
                    < plan.checkpoint_mult * plan.est_checkpoint_write_sec));
    let mut root_frac_limbs = root_frac_limbs;
    let conv_res = if can_use_fast_frontier {
        match convert_frontier_fast(
            ctx,
            plan,
            shares,
            cal,
            &mut pow10_levels,
            num_leaves,
            &mut root_frac_limbs,
            root_off,
            root_k,
            kept_full_leaves,
            rem_digits,
            sink,
            &guard_tail,
            &spine_completed,
            &last_ckpt,
        ) {
            Ok(Some(w0)) => Ok(w0),
            Ok(None) => convert_node(
                ctx,
                plan,
                shares,
                cal,
                &pow10_levels,
                0,
                num_leaves,
                &root_frac_limbs[root_off..],
                root_k,
                None,
                kept_full_leaves,
                rem_digits,
                num_leaves,
                sink,
                &guard_tail,
                leaf_fault_hook,
                &resume_completed,
                &spine_completed,
                &resume_fracs,
                ckpt_sink,
                &last_ckpt,
                1,
                1,
            ),
            Err(e) => Err(e),
        }
    } else {
        convert_node(
            ctx,
            plan,
            shares,
            cal,
            &pow10_levels,
            0,
            num_leaves,
            &root_frac_limbs[root_off..],
            root_k,
            None,
            kept_full_leaves,
            rem_digits,
            num_leaves,
            sink,
            &guard_tail,
            leaf_fault_hook,
            &resume_completed,
            &spine_completed,
            &resume_fracs,
            ckpt_sink,
            &last_ckpt,
            0,
            1,
        )
    };
    if do_trace {
        let b_ms = RADIX_BATCH_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
        eprintln!(
            "[TRACE CONV] convert_node={:.2}ms | batch_cpu_sum={b_ms:.1}ms",
            t_cn0.elapsed().as_secs_f64() * 1e3
        );
    }

    for entry in &mut pow10_levels {
        entry.release_store(ctx.store);
    }
    crate::limb::put_limb_buf(root_frac_limbs);
    let tree_w0 = conv_res?;
    let t_fl0 = Instant::now();
    sink.flush().map_err(MulError::FatalStop)?;
    if do_trace {
        eprintln!(
            "[TRACE CONV] sink.flush={:.2}ms",
            t_fl0.elapsed().as_secs_f64() * 1e3
        );
    }

    let tail = guard_tail.into_inner().unwrap();
    let check_len = tail.len().min(LEAF_DIGITS);
    let digits_in_doubt = check_len > 0 && is_all_zeros_or_nines(&tail[..check_len]);

    // Compute exact `fractional_w0` and `dec_hash_w0 = floor((int_part + frac) * 10^digits) mod (2^61 - 1)`!
    // Wait: `tree_w0` is the check word of all `num_leaves * 19` converted digits (`kept * 10^{dropped} + tail_val`).
    // Subtract `tail_val mod W0` and divide by `10^{dropped} mod W0`!
    let dropped_digits = (total_tree_digits - plan.digits) as usize;
    let mut tail_w0 = 0u64;
    for &ch in &tail[..dropped_digits] {
        let d = (ch.saturating_sub(b'0')) as u64;
        tail_w0 = mod61_add(mod61_mul(tail_w0, 10), d);
    }
    let pow10_drop = pow10_mod_w0(dropped_digits as u64);
    let inv_pow10_drop = crate::refk::mod_pow_u64(pow10_drop, MOD_W0 - 2, MOD_W0);
    let diff_w0 = crate::refk::mod61_sub(tree_w0, tail_w0);
    let fractional_w0 = mod61_mul(diff_w0, inv_pow10_drop);
    let int_scaled = mod61_mul(mod61_reduce(int_part as u128), pow10_mod_w0(plan.digits));
    let dec_hash_w0 = mod61_add(int_scaled, fractional_w0);

    Ok(ConversionResult {
        dec_hash_w0,
        fractional_w0,
        digits_in_doubt,
    })
}

fn pow10_mod_w0(exp: u64) -> u64 {
    pow_mod_w0(10, exp)
}

/// `base^exp` modulo `2^61 - 1`.
fn pow_mod_w0(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = mod61_mul(acc, base);
        }
        exp >>= 1;
        if exp > 0 {
            base = mod61_mul(base, base);
        }
    }
    acc
}

fn is_all_zeros_or_nines(bytes: &[u8]) -> bool {
    !bytes.is_empty() && (bytes.iter().all(|&b| b == b'0') || bytes.iter().all(|&b| b == b'9'))
}

fn resize_fraction_limbs(limbs: &[u64], in_k: usize, out_k: usize) -> LimbVec {
    let mut full = crate::limb::take_limb_buf_uninit(in_k.max(out_k));
    if out_k <= in_k {
        let drop_low = in_k - out_k;
        let copy_start = drop_low.min(limbs.len());
        let copy_end = in_k.min(limbs.len());
        if copy_end > copy_start {
            full.extend_from_slice(&limbs[copy_start..copy_end]);
        }
        full.resize(out_k, 0);
        full
    } else {
        let pad_low = out_k - in_k;
        full.resize(pad_low, 0);
        let copy_end = in_k.min(limbs.len());
        full.extend_from_slice(&limbs[..copy_end]);
        full.resize(out_k, 0);
        full
    }
}

fn resize_fraction_limbs_owned(mut limbs: LimbVec, in_k: usize, out_k: usize) -> LimbVec {
    limbs.resize(in_k, 0);
    if out_k == in_k {
        limbs
    } else if out_k < in_k {
        let drop_low = in_k - out_k;
        limbs.copy_within(drop_low..in_k, 0);
        limbs.truncate(out_k);
        limbs
    } else {
        let pad_low = out_k - in_k;
        limbs.resize(out_k, 0);
        limbs.copy_within(0..in_k, pad_low);
        limbs[..pad_low].fill(0);
        limbs
    }
}

fn build_pow10_table(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    max_lv: usize,
    num_leaves: u64,
    skip_low_cached: bool,
) -> Result<Vec<Pow10Entry>, MulError> {
    let mut entries: Vec<Pow10Entry> = Vec::with_capacity(max_lv + 1);
    let mut cur = BigNum::from_u64(POW10_19);
    let mut cur_mod_w0 = POW10_19_MOD_W0;
    let mut cur_pow10_w0 = POW10_19_MOD_W0;
    let mut cached_ram_bytes: u64 = 0;
    let mut cached_store_bytes: u64 = 0;
    let cache_ram_cap: u64 = plan.ram_transform_budget / 2;
    let cache_store_cap: u64 = shares.total_vram.saturating_add(shares.total_ssd) / 3;

    enum DeferredShift {
        None,
        Linear(u64),
        Cyclic { shift: u64, l_bits: u64 },
    }
    struct DeferredWrapNtt {
        lv: usize,
        spec: crate::ntt::NttSpec,
        shift: DeferredShift,
    }
    struct DeferredWhole {
        lv: usize,
        method: MulMethod,
    }
    let defer_small_caches = ctx.sched.num_workers() > 1;
    let mut def_wntt: Vec<DeferredWrapNtt> = Vec::new();
    let mut def_whole: Vec<DeferredWhole> = Vec::new();

    let build_res: Result<(), MulError> = (|| {
    for lv in 0..=max_lv {
        let left_leaves = 1u64 << lv;
        let d_left = left_leaves * (LEAF_DIGITS as u64);
        // From `STRIP_MIN_LV` on the multiplier is `5^d`: `10^d` without its `d` low bits, which
        // are zero and a whole number of limbs. The squarings go on from it.
        if lv == STRIP_MIN_LV {
            let zero_limbs = (d_left / 64) as usize;
            debug_assert!(d_left.is_multiple_of(64) && cur.shift == 0);
            debug_assert!(cur.limbs[..zero_limbs].iter().all(|&w| w == 0));
            let mut odd = crate::limb::take_limb_buf_uninit(cur.limbs.len() - zero_limbs);
            odd.extend_from_slice(&cur.limbs[zero_limbs..]);
            let ten = std::mem::replace(
                &mut cur,
                BigNum::from_limbs_par(
                    odd,
                    false,
                    0,
                    ctx.kernels,
                    ctx.sched,
                    shares.active_workers.max(1),
                ),
            );
            crate::limb::put_limb_buf(ten.into_limbs());
            cur_mod_w0 = pow_mod_w0(5, d_left);
        }
        let strip_limbs = if lv >= STRIP_MIN_LV {
            (d_left / 64) as usize
        } else {
            0
        };
        // Max parent size at level `lv` has at most `2 * left_leaves` leaves; `k_parent` counts
        // the limbs of its fraction that are multiplied (those below the stripped ones):
        let max_parent_leaves = (2 * left_leaves).min(num_leaves);
        let max_parent_d = max_parent_leaves * (LEAF_DIGITS as u64);
        let k_parent = frac_limbs_for_digits(max_parent_d) - strip_limbs;
        let k_right = frac_limbs_for_digits((max_parent_leaves - left_leaves).max(1) * (LEAF_DIGITS as u64));

        let x_bits = (k_parent as u64) * 64;
        let p_bits = cur.body_bits();
        let b_left = if strip_limbs > 0 {
            p_bits
        } else {
            bit_ceil_pow10(d_left)
        };
        let min_wrap_l = ((k_parent as u64) * 64).max(b_left + (k_right as u64) * 64 + 128);
        let full_prod_bits = x_bits + p_bits;
        let est_nodes = ((num_leaves + (1u64 << lv) - 1) >> (lv + 1)).max(1);
        let cand_pw = if !plan.out_of_core && est_nodes < (shares.active_workers as u64) {
            (shares.active_workers / (est_nodes as usize).max(1)).max(1)
        } else if plan.out_of_core && est_nodes == 1 {
            shares.active_workers
        } else {
            1
        };
        let probe_m = plan.select_mul_method(x_bits, p_bits, false, false, cand_pw, shares, cal);
        let will_wrap = left_leaves >= 64
            && min_wrap_l + 256 < full_prod_bits
            && matches!(probe_m, MulMethod::Ntt(_));
        let unpack_pow10_probe = |s: &mut NttSpec| {
            if !plan.out_of_core
                && !s.out_of_core
                && s.rows > 1
                && s.extra_sweeps == 0
                && shares.total_l3 > 0
            {
                let up_rb = if s.cols >= 512 { s.cols * 8 + 512 } else { s.cols * 8 };
                let up_grid = (s.k_primes as u64)
                    .saturating_mul(s.rows as u64)
                    .saturating_mul(up_rb as u64)
                    .max(1);
                let raw_node_b = 2u64.saturating_mul(k_parent as u64).saturating_mul(8);
                let up_node_set = up_grid.saturating_add(raw_node_b).max(1);
                let aw = shares.active_workers.max(1);
                let l3_90 = (shares.effective_l3_bytes(est_nodes.max(1), s.rows) * 9) / 10;
                let l3_tot = (shares.effective_l3_bytes(est_nodes.max(1), s.rows) * 7) / 8;
                if up_grid.saturating_mul(2) <= l3_90 {
                    let by_grid_2 = (l3_90.saturating_sub(up_grid) / up_grid).clamp(1, 2);
                    let by_l3_up = (l3_tot.saturating_sub(up_grid) / up_node_set)
                        .max(by_grid_2)
                        .max(1);
                    let cap_c = (by_l3_up as usize).max(1).min(aw).min(est_nodes as usize);
                    let mut est_c = 1usize;
                    for c in crate::tier::even_splits(aw).into_iter().rev() {
                        let pw_c = (aw / c).max(1);
                        if c >= 1 && c <= cap_c && (pw_c <= s.rows.max(1) || c == 1) {
                            est_c = c;
                            break;
                        }
                    }
                    let (shared_up, num, den) = if est_c <= 2 {
                        (up_grid.saturating_add(up_grid / (est_c as u64).max(1)), 9, 10)
                    } else {
                        (up_node_set.saturating_add(up_grid / (est_c as u64).max(1)), 7, 8)
                    };
                    if shares.fits_l3_domains(est_c as u64, shared_up, s.rows, num, den) {
                        s.packed = false;
                    }
                }
            }
        };
        let unpack_pow10_spec = |s: &mut NttSpec| {
            if !plan.out_of_core
                && !s.out_of_core
                && s.rows > 1
                && s.extra_sweeps == 0
                && shares.total_l3 > 0
            {
                let up_rb = if s.cols >= 512 { s.cols * 8 + 512 } else { s.cols * 8 };
                let up_grid = (s.k_primes as u64)
                    .saturating_mul(s.rows as u64)
                    .saturating_mul(up_rb as u64);
                let raw_node_b = 2u64.saturating_mul(k_parent as u64).saturating_mul(8);
                let up_node_set = up_grid.saturating_add(raw_node_b);
                let conc = (shares.active_workers / s.par_workers.max(1))
                    .max(1)
                    .min(est_nodes.max(1) as usize) as u64;
                let (shared_up, num, den) = if conc <= 2 {
                    (up_grid.saturating_add(up_grid / conc.max(1)), 9, 10)
                } else {
                    (up_node_set.saturating_add(up_grid / conc.max(1)), 7, 8)
                };
                if shares.fits_l3_domains(conc, shared_up, s.rows, num, den) {
                    s.packed = false;
                }
            }
        };
        let mut probe_ntt_spec = if will_wrap {
            Some(crate::plan::choose_wrapped_ntt_spec_amortized(
                min_wrap_l,
                x_bits.max(p_bits),
                cand_pw,
                shares,
                cal,
                plan.ram_transform_budget,
                plan.overrides.max_ntt_primes,
                est_nodes.max(1),
            ))
        } else if let MulMethod::Ntt(s) = probe_m {
            Some(s)
        } else {
            None
        };
        if let Some(ref mut s) = probe_ntt_spec {
            unpack_pow10_probe(s);
        }
        let par_w = if probe_ntt_spec.is_some_and(|s| s.out_of_core) {
            cand_pw
        } else {
            match probe_ntt_spec {
                Some(s) if s.rows > 1 => {
                    let grid_1 = (s.k_primes as u64)
                        .saturating_mul(s.rows as u64)
                        .saturating_mul(s.outer_row_bytes() as u64)
                        .max(1);
                    let up_rb = if s.cols >= 512 { s.cols * 8 + 512 } else { s.cols * 8 };
                    let prime_b = (s.rows as u64).saturating_mul(up_rb as u64).max(1);
                    let aw = shares.active_workers.max(1);
                    let l3_90 = shares.effective_l3_bytes(est_nodes.max(1), s.rows) * 9 / 10;
                    let l3_tot = shares.effective_l3_bytes(est_nodes.max(1), s.rows) * 7 / 8;
                    let est_node_set = grid_1.saturating_add(2 * (k_parent as u64) * 8).max(1);
                    let by_grid_2 = if !s.packed && grid_1.saturating_mul(2) <= l3_90 {
                        (l3_90.saturating_sub(grid_1) / grid_1).clamp(1, 2)
                    } else {
                        0
                    };
                    let by_l3_whole = (l3_tot.saturating_sub(grid_1) / est_node_set)
                        .max(by_grid_2)
                        .max(1);
                    let ram_half = (plan.ram_transform_budget / 2) / est_node_set;
                    let spills_l3 = s.packed || grid_1.saturating_mul(2) > l3_90;
                    let by_l3 = if est_nodes >= 2 && s.packed && s.rows >= 64 && ram_half >= 2 {
                        by_l3_whole.max(4.min(ram_half))
                    } else if est_nodes >= 2 && spills_l3 && prime_b <= l3_tot {
                        by_l3_whole
                            .max((l3_tot / (prime_b.saturating_mul(5) / 4).max(1)).clamp(2, 4))
                    } else {
                        by_l3_whole
                    };
                    let by_ram =
                        plan.ram_transform_budget.saturating_sub(grid_1) / grid_1;
                    let cap_c = (by_l3.min(by_ram) as usize)
                        .max(1)
                        .min(aw)
                        .min(est_nodes as usize);
                    let mut est_c = 1usize;
                    for c in crate::tier::even_splits(aw).into_iter().rev() {
                        let pw_c = (aw / c).max(1);
                        if c >= 1 && c <= cap_c && (pw_c <= s.rows.max(1) || c == 1) {
                            est_c = c;
                            break;
                        }
                    }
                    (aw / est_c.max(1)).max(1)
                }
                _ => cand_pw,
            }
        };

        let mut whole_method = if par_w == cand_pw {
            probe_m
        } else {
            plan.select_mul_method(x_bits, p_bits, false, false, par_w, shares, cal)
        };
        if let MulMethod::Ntt(ref mut s) = whole_method {
            unpack_pow10_spec(s);
        }
        if left_leaves >= 64
            && max_parent_leaves <= BATCH_MAX_LEAVES
            && matches!(plan.overrides.split_override, crate::plan::SplitFormOverride::Auto)
            && let Some((n_complex, coeff_bits)) = crate::fft::choose_fft_params(
                full_prod_bits + 128,
                false,
                shares.min_l2_share,
            )
        {
            whole_method = MulMethod::Fft {
                n_complex,
                coeff_bits,
            };
        }
        let handled_by_batch = max_parent_leaves <= 1024
            || (max_parent_leaves <= BATCH_MAX_LEAVES
                && matches!(whole_method, MulMethod::Fft { .. }));
        let need_cached = !(skip_low_cached
            && handled_by_batch
            && !matches!(whole_method, MulMethod::Fft { .. }));

        // Only exact methods (`NTT`, limb) may wrap a product (Section 5 & Section 11):
        // Wrap length `L >= max(64 * k_parent, b_left + 64 * k_right + 128)` saves ~1/3 of transform!
        let mut wrapped_turn = 0usize;
        let (wrapped_l_bits, wrapped_s_pad, wrapped_spec, wrapped_cached) = if need_cached
            && matches!(whole_method, MulMethod::Ntt(_))
            && left_leaves >= 64
            && min_wrap_l + 256 < full_prod_bits
        {
            let mut spec = if par_w == cand_pw {
                probe_ntt_spec.unwrap()
            } else {
                crate::plan::choose_wrapped_ntt_spec_amortized(
                    min_wrap_l,
                    x_bits.max(p_bits),
                    par_w,
                    shares,
                    cal,
                    plan.ram_transform_budget,
                    plan.overrides.max_ntt_primes,
                    est_nodes.max(1),
                )
            };
            unpack_pow10_spec(&mut spec);
            let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
            if l_bits >= min_wrap_l && l_bits + 128 < full_prod_bits {
                spec.wrap_cyclic = true;
                let entry_bytes = (spec.k_primes as u64) * (spec.rows as u64) * (spec.outer_row_bytes() as u64);
                let worth_two_node_cache = if spec.rows > 1 {
                    if est_nodes >= 3 {
                        true
                    } else if est_nodes == 2 && num_leaves > 3 * left_leaves {
                        let r1_leaves = num_leaves - 2 * left_leaves;
                        let r1_k_p =
                            frac_limbs_for_digits(r1_leaves * (LEAF_DIGITS as u64)) - strip_limbs;
                        let r1_k_r = frac_limbs_for_digits((r1_leaves - left_leaves).max(1) * (LEAF_DIGITS as u64));
                        let r1_x_bits = (r1_k_p as u64) * 64;
                        let r1_min_l = r1_x_bits.max(b_left + (r1_k_r as u64) * 64 + 128);
                        let r1_prefers_tight = r1_min_l + 4096 < l_bits
                            && r1_min_l.saturating_mul(5) <= l_bits.saturating_mul(3)
                            && r1_min_l + 256 < r1_x_bits + p_bits;
                        !r1_prefers_tight
                    } else {
                        false
                    }
                } else {
                    est_nodes >= 2
                };
                let cache_fits = if spec.out_of_core {
                    cached_store_bytes.saturating_add(entry_bytes) <= cache_store_cap
                } else {
                    cached_ram_bytes.saturating_add(entry_bytes) <= cache_ram_cap
                };
                if worth_two_node_cache && cache_fits {
                    let s_pad_std = l_bits.saturating_sub((k_parent as u64) * 64);
                    let can_preshift = s_pad_std > 0 && cur.body_bits().saturating_add(s_pad_std) <= l_bits;
                    let turn_bits = (k_right as u64) * 64;
                    let can_turn = est_nodes >= TURN_MIN_NODES
                        && l_bits.is_multiple_of(64)
                        && s_pad_std + turn_bits < l_bits
                        && turn_bits + b_left + 128 <= l_bits;
                    if defer_small_caches && !spec.out_of_core && spec.rows == 1 {
                        let (shift, used_s_pad) = if can_turn {
                            wrapped_turn = k_right;
                            (
                                DeferredShift::Cyclic {
                                    shift: s_pad_std + turn_bits,
                                    l_bits,
                                },
                                s_pad_std,
                            )
                        } else if can_preshift {
                            (DeferredShift::Linear(s_pad_std), s_pad_std)
                        } else {
                            (DeferredShift::None, 0)
                        };
                        cached_ram_bytes = cached_ram_bytes.saturating_add(entry_bytes);
                        def_wntt.push(DeferredWrapNtt { lv, spec, shift });
                        (Some(l_bits), used_s_pad, Some(spec), None)
                    } else {
                        let mut build_spec = spec;
                        build_spec.par_workers = shares.active_workers.max(1);
                        let (mut cached, used_s_pad) = if can_turn {
                            // The multiplier goes in turned up by the right child's limbs as well
                            // (see `Pow10Entry::wrapped_turn`).
                            let cur_sh = shl_limbs_cyclic(
                                &cur.limbs,
                                s_pad_std + turn_bits,
                                l_bits,
                                ctx.sched,
                                shares.active_workers.max(1),
                            );
                            let c_res = ntt_forward_reusable(
                                ctx.kernels,
                                ctx.ntt_cache,
                                ctx.sched,
                                ctx.store,
                                &cur_sh,
                                build_spec,
                            );
                            crate::limb::put_limb_buf(cur_sh);
                            wrapped_turn = k_right;
                            (
                                c_res.map_err(|_| MulError::NeedRollback(0x4E00_0000 | (lv as u64)))?,
                                s_pad_std,
                            )
                        } else if can_preshift {
                            let cur_sh = shl_limbs_par(
                                &cur.limbs,
                                s_pad_std,
                                ctx.sched,
                                shares.active_workers.max(1),
                            );
                            let c_res = ntt_forward_reusable(
                                ctx.kernels,
                                ctx.ntt_cache,
                                ctx.sched,
                                ctx.store,
                                &cur_sh,
                                build_spec,
                            );
                            crate::limb::put_limb_buf(cur_sh);
                            (
                                c_res.map_err(|_| MulError::NeedRollback(0x4E00_0000 | (lv as u64)))?,
                                s_pad_std,
                            )
                        } else {
                            let c_res = ntt_forward_reusable(
                                ctx.kernels,
                                ctx.ntt_cache,
                                ctx.sched,
                                ctx.store,
                                &cur.limbs,
                                build_spec,
                            );
                            (
                                c_res.map_err(|_| MulError::NeedRollback(0x4E00_0000 | (lv as u64)))?,
                                0,
                            )
                        };
                        cached.spec.par_workers = spec.par_workers;
                        if spec.out_of_core {
                            cached_store_bytes = cached_store_bytes.saturating_add(entry_bytes);
                        } else {
                            cached_ram_bytes = cached_ram_bytes.saturating_add(entry_bytes);
                        }
                        (Some(l_bits), used_s_pad, Some(spec), Some(cached))
                    }
                } else {
                    (Some(l_bits), 0, Some(spec), None)
                }
            } else {
                (None, 0, None, None)
            }
        } else {
            (None, 0, None, None)
        };

        let whole_worth_cache = match &whole_method {
            MulMethod::Fft { .. } => true,
            MulMethod::Ntt(s) if s.rows > 1 => est_nodes >= 3,
            _ => est_nodes >= 2,
        };
        let whole_cached = if need_cached
            && wrapped_spec.is_none()
            && whole_worth_cache
            && (max_parent_leaves == 2 * left_leaves
                || matches!(whole_method, MulMethod::Fft { .. }))
            && matches!(whole_method, MulMethod::Fft { .. } | MulMethod::Ntt(_))
        {
            let (entry_bytes, entry_in_store) = match &whole_method {
                MulMethod::Fft { n_complex, .. } => ((*n_complex as u64) * 16, false),
                MulMethod::Ntt(s) => (
                    (s.k_primes as u64) * (s.rows as u64) * (s.outer_row_bytes() as u64),
                    s.out_of_core,
                ),
                _ => (0, false),
            };
            let whole_fits = if entry_in_store {
                cached_store_bytes.saturating_add(entry_bytes) <= cache_store_cap
            } else {
                cached_ram_bytes.saturating_add(entry_bytes) <= cache_ram_cap
            };
            if whole_fits {
                if defer_small_caches
                    && !entry_in_store
                    && !matches!(whole_method, MulMethod::Ntt(s) if s.rows > 1)
                {
                    cached_ram_bytes = cached_ram_bytes.saturating_add(entry_bytes);
                    def_whole.push(DeferredWhole {
                        lv,
                        method: whole_method.clone(),
                    });
                    None
                } else {
                    let mut build_method = whole_method.clone();
                    if let MulMethod::Ntt(ref mut s) = build_method {
                        s.par_workers = shares.active_workers.max(1);
                    }
                    let mut c = ctx.forward_cached_reusable(&build_method, &cur.limbs)?;
                    if let (CachedTransform::Ntt(nc), MulMethod::Ntt(orig_s)) = (&mut c, &whole_method) {
                        nc.spec.par_workers = orig_s.par_workers;
                    }
                    if entry_in_store {
                        cached_store_bytes = cached_store_bytes.saturating_add(entry_bytes);
                    } else {
                        cached_ram_bytes = cached_ram_bytes.saturating_add(entry_bytes);
                    }
                    Some(c)
                }
            } else {
                None
            }
        } else {
            None
        };
        let zero_lo_limbs = if strip_limbs > 0 {
            0
        } else {
            (d_left / 64) as usize
        };
        debug_assert!(cur.limbs[..zero_lo_limbs].iter().all(|&w| w == 0));

        // The entry takes the power itself, below, once the next level's has been made from it:
        // a copy would be one worker's pass over all of it while the others wait.
        entries.push(Pow10Entry {
            mult: BigNum::zero(),
            mult_w0: cur_mod_w0,
            pow10_w0: cur_pow10_w0,
            mult_bits: b_left,
            strip_limbs,
            zero_lo_limbs,
            whole_method,
            whole_cached,
            wrapped_l_bits,
            wrapped_s_pad,
            wrapped_turn,
            wrapped_spec,
            wrapped_cached,
        });

        if lv < max_lv {
            let sq_m = plan.select_mul_method(
                cur.body_bits(),
                cur.body_bits(),
                false,
                false,
                shares.active_workers,
                shares,
                cal,
            );
            let next = ctx.sqr_whole(0x4F00_0000 | (lv as u64), &sq_m, &mut cur)?;
            entries[lv].mult = std::mem::replace(&mut cur, next);
            cur_mod_w0 = mod61_mul(cur_mod_w0, cur_mod_w0);
            cur_pow10_w0 = mod61_mul(cur_pow10_w0, cur_pow10_w0);
        } else {
            entries[lv].mult = std::mem::replace(&mut cur, BigNum::zero());
        }
    }

    let n_wn = def_wntt.len();
    let n_wh = def_whole.len();
    let total_def = n_wn + n_wh;
    if total_def > 0 {
        let mut out_wn: Vec<Option<Result<crate::ntt::NttCachedFwd, MulError>>> =
            (0..n_wn).map(|_| None).collect();
        let mut out_wh: Vec<Option<Result<CachedTransform, MulError>>> =
            (0..n_wh).map(|_| None).collect();
        let wn_ptr = out_wn.as_mut_ptr() as usize;
        let wh_ptr = out_wh.as_mut_ptr() as usize;
        let entries_ref = entries.as_slice();
        ctx.sched.par_for(total_def, 1, |ti| {
            if ti < n_wn {
                let idx = ti;
                let d = &def_wntt[idx];
                let mut build_spec = d.spec;
                build_spec.par_workers = 1;
                let limbs_src = &entries_ref[d.lv].mult.limbs;
                let c_res = match d.shift {
                    DeferredShift::Cyclic { shift, l_bits } => {
                        let cur_sh = shl_limbs_cyclic(limbs_src, shift, l_bits, ctx.sched, 1);
                        let r = ntt_forward_reusable(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            &cur_sh,
                            build_spec,
                        );
                        crate::limb::put_limb_buf(cur_sh);
                        r
                    }
                    DeferredShift::Linear(s_pad) => {
                        let cur_sh = shl_limbs_par(limbs_src, s_pad, ctx.sched, 1);
                        let r = ntt_forward_reusable(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            &cur_sh,
                            build_spec,
                        );
                        crate::limb::put_limb_buf(cur_sh);
                        r
                    }
                    DeferredShift::None => ntt_forward_reusable(
                        ctx.kernels,
                        ctx.ntt_cache,
                        ctx.sched,
                        ctx.store,
                        limbs_src,
                        build_spec,
                    ),
                };
                let res = c_res
                    .map_err(|_| MulError::NeedRollback(0x4E10_0000 | (d.lv as u64)))
                    .map(|mut c| {
                        c.spec.par_workers = d.spec.par_workers;
                        c
                    });
                unsafe {
                    *(wn_ptr as *mut Option<Result<crate::ntt::NttCachedFwd, MulError>>).add(idx) =
                        Some(res);
                }
            } else {
                let idx = ti - n_wn;
                let d = &def_whole[idx];
                let mut build_method = d.method.clone();
                if let MulMethod::Ntt(ref mut s) = build_method {
                    s.par_workers = 1;
                }
                let res = ctx
                    .forward_cached_reusable(&build_method, &entries_ref[d.lv].mult.limbs)
                    .map(|mut c| {
                        if let (CachedTransform::Ntt(nc), MulMethod::Ntt(orig_s)) =
                            (&mut c, &d.method)
                        {
                            nc.spec.par_workers = orig_s.par_workers;
                        }
                        c
                    });
                unsafe {
                    *(wh_ptr as *mut Option<Result<CachedTransform, MulError>>).add(idx) =
                        Some(res);
                }
            }
        });
        for (d, slot) in def_wntt.iter().zip(out_wn.into_iter()) {
            entries[d.lv].wrapped_cached = Some(slot.unwrap()?);
        }
        for (d, slot) in def_whole.iter().zip(out_wh.into_iter()) {
            entries[d.lv].whole_cached = Some(slot.unwrap()?);
        }
    }
    Ok(())
    })();
    if let Err(e) = build_res {
        for ent in &mut entries {
            ent.release_store(ctx.store);
        }
        return Err(e);
    }

    Ok(entries)
}

/// How a node's product ties the digits of its left child to it (Section 11): the check word
/// those digits must have, or the terms of the wrapped product that they complete.
#[derive(Clone, Copy)]
enum NodeCheck {
    Whole {
        expected_left_w0: u64,
    },
    Wrapped {
        wrapped_w0: u64,
        x_shifted_w0: u64,
        l_bits: u64,
        all_zeros_kept: bool,
        /// What the node's stripped top limbs add to the left child's digits, as a check word.
        top_term_w0: u64,
    },
}

impl NodeCheck {
    /// Whether digits with check word `left_w0` are those of the left child of the node this
    /// product was made for, `mult_w0` being the check word of the level's multiplier.
    fn holds(&self, left_w0: u64, mult_w0: u64) -> bool {
        match *self {
            NodeCheck::Whole { expected_left_w0 } => left_w0 == expected_left_w0,
            NodeCheck::Wrapped {
                wrapped_w0,
                x_shifted_w0,
                l_bits,
                all_zeros_kept,
                top_term_w0,
            } => {
                let folded_w0 = crate::refk::mod61_sub(left_w0, top_term_w0);
                let req_whole_w0 = mod61_mul(x_shifted_w0, mult_w0);
                wrapped_product_expected_w0(wrapped_w0, folded_w0, l_bits, false) == req_whole_w0
                    || (all_zeros_kept
                        && wrapped_product_expected_w0(wrapped_w0, folded_w0, l_bits, true)
                            == req_whole_w0)
            }
        }
    }
}

/// The product of one node: the fraction of its right child (`k_right` limbs), whether that is
/// below one half, and the check that the left child's digits must pass. The node's fraction is
/// `use_limbs` (`use_k` limbs); of it the limbs below the level's stripped ones are multiplied
/// by the level's multiplier (see `Pow10Entry`). The product is taken modulo `2^L - 1` where the
/// level has a wrapped transform and the bits between the wrapped-around integer part and the
/// right child's fraction show that no carry ran through them; otherwise, and always with
/// `force_whole`, it is the whole product (with the level's cached transform where `use_k` is
/// at most `max_cached_k` and not `force_whole`).
///
/// `pre_w0` holds the node's check words where they came with its buffer; without them they
/// are taken from its limbs. A product against a turned multiplier (see
/// `Pow10Entry::wrapped_turn`) has to take the right child's check words for its own check, and
/// returns them piece by piece between the places of `child_cuts` (ascending, from zero to
/// `k_right`; see `chain_cuts`), for the child's product and those of its left children.
#[allow(clippy::too_many_arguments)]
fn node_right_fraction(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    p_entry: &Pow10Entry,
    point_id: u64,
    use_limbs: &[u64],
    use_k: usize,
    k_right: usize,
    max_cached_k: usize,
    node_par_w: usize,
    force_whole: bool,
    pre_w0: Option<PartWords>,
    child_cuts: &[usize],
) -> Result<(LimbVec, bool, NodeCheck, Option<SegWords>), MulError> {
    let have = use_limbs.len().min(use_k);
    let f_k = use_k - p_entry.strip_limbs;
    let f_limbs = &use_limbs[..have.min(f_k)];
    let top_limbs = &use_limbs[have.min(f_k)..have];
    let b_mult = p_entry.mult_bits;
    // The check words of the two parts, where the node came with them cut at this place.
    let known = pre_w0.filter(|p| have == use_k && p.cut == f_k);
    let top_term_w0 = if top_limbs.is_empty() {
        0
    } else {
        let top_w0 = match known {
            Some(p) => p.hi_w0,
            None => compute_limbs_w0_par(ctx.kernels, ctx.sched, top_limbs, node_par_w),
        };
        mod61_mul(top_w0, p_entry.mult_w0)
    };
    let f_w0_of = || match known {
        Some(p) => p.lo_w0,
        None => compute_limbs_w0_par(ctx.kernels, ctx.sched, f_limbs, node_par_w),
    };

    let mut node_prod: Option<(LimbVec, bool, NodeCheck, Option<SegWords>)> = None;
    if !force_whole && let Some(l_bits) = p_entry.wrapped_l_bits {
        let x_bits = (f_k as u64) * 64;
        if l_bits >= x_bits {
            let s_pad = l_bits - x_bits;
            let min_node_l = x_bits.max(b_mult + (k_right as u64) * 64 + 128);
            let p_bits = p_entry.mult.body_bits();
            let prefer_tight_over_cache = min_node_l + 4096 < l_bits
                && min_node_l.saturating_mul(5) <= l_bits.saturating_mul(3)
                && min_node_l + 256 < x_bits + p_bits;
            let mut active_l_bits = l_bits;
            // Limbs the product comes turned up by (see `Pow10Entry::wrapped_turn`).
            let mut turn = 0usize;
            let t_prod = Instant::now();
            let w_res = if let Some(w_cached) = p_entry
                .wrapped_cached
                .as_ref()
                .filter(|_| s_pad >= p_entry.wrapped_s_pad && !prefer_tight_over_cache)
            {
                turn = p_entry.wrapped_turn;
                let rem_pad = s_pad - p_entry.wrapped_s_pad;
                if rem_pad == 0 {
                    ctx.mul_wrapped_with_cached(point_id, f_limbs, w_cached, node_par_w)
                } else {
                    let x_rem = shl_limbs_par(f_limbs, rem_pad, ctx.sched, node_par_w);
                    let res = ctx.mul_wrapped_with_cached(point_id, &x_rem, w_cached, node_par_w);
                    crate::limb::put_limb_buf(x_rem);
                    res
                }
            } else if let Some(mut w_spec) = p_entry.wrapped_spec {
                if min_node_l + 4096 < l_bits
                    && min_node_l.saturating_mul(4) <= l_bits.saturating_mul(3)
                    && min_node_l + 256 < x_bits + p_bits
                {
                    let tight_spec = choose_wrapped_ntt_spec(
                        min_node_l,
                        x_bits.max(p_bits),
                        node_par_w,
                        shares,
                        cal,
                        plan.ram_transform_budget,
                        plan.overrides.max_ntt_primes,
                    );
                    let tight_l = (tight_spec.n as u64) * (tight_spec.coeff_bits as u64);
                    if tight_l >= min_node_l && tight_l < l_bits {
                        w_spec = tight_spec;
                        w_spec.wrap_cyclic = true;
                        active_l_bits = tight_l;
                    }
                }
                let use_s_pad = active_l_bits - x_bits;
                w_spec.par_workers = node_par_w;
                // The shorter operand takes the shift that puts the binary point on bit `L`.
                // Here the multiplier is not turned as a cached one is: turned, its zeros lie
                // below it instead of above, where its transform does not pass over them, and
                // that costs more than moving the right child does (measured on the root of
                // 2.5e9 digits: the product 50 to 70 ms longer, the move 15 ms).
                if p_entry.mult.limbs.len() < f_limbs.len() {
                    let p_shifted =
                        shl_limbs_par(&p_entry.mult.limbs, use_s_pad, ctx.sched, node_par_w);
                    let r = ctx.mul_wrapped(point_id, w_spec, f_limbs, &p_shifted);
                    crate::limb::put_limb_buf(p_shifted);
                    r
                } else {
                    let x_shifted = shl_limbs_par(f_limbs, use_s_pad, ctx.sched, node_par_w);
                    let r = ctx.mul_wrapped(point_id, w_spec, &x_shifted, &p_entry.mult.limbs);
                    crate::limb::put_limb_buf(x_shifted);
                    r
                }
            } else {
                Err(MulError::NeedRollback(point_id))
            };
            RADIX_SPLIT_PROD_NS.fetch_add(t_prod.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Ok((w_limbs, w_w0, _)) = w_res {
                // The integer part came around onto the low `b_mult` bits. A carry out of them
                // that reached the right child's fraction would have left every bit in between
                // zero.
                let r_start_bit = active_l_bits - (k_right as u64) * 64;
                if turn == 0 {
                    if r_start_bit >= b_mult + 32
                        && has_nonzero_in_bit_range(&w_limbs, b_mult + 1, r_start_bit)
                    {
                        let x_right = if (r_start_bit & 63) == 0 && w_limbs.capacity() >= k_right {
                            let word_shift = (r_start_bit >> 6) as usize;
                            let copied = w_limbs.len().saturating_sub(word_shift).min(k_right);
                            shift_limbs_down_in_place_par(
                                w_limbs,
                                word_shift,
                                copied,
                                k_right,
                                ctx.sched,
                                node_par_w,
                            )
                        } else {
                            let mut v = shr_limbs_par(&w_limbs, r_start_bit, ctx.sched, node_par_w);
                            v.resize(k_right, 0);
                            crate::limb::put_limb_buf(w_limbs);
                            v
                        };
                        let right_under_half = (x_right[k_right - 1] >> 63) == 0;
                        node_prod = Some((
                            x_right,
                            right_under_half,
                            NodeCheck::Wrapped {
                                wrapped_w0: w_w0,
                                x_shifted_w0: mod61_mul(f_w0_of(), mod61_pow2(active_l_bits - x_bits)),
                                l_bits: active_l_bits,
                                all_zeros_kept: false,
                                top_term_w0,
                            },
                            None,
                        ));
                    } else {
                        crate::limb::put_limb_buf(w_limbs);
                    }
                } else {
                    // A turned product: its limbs `[turn - k_right, turn)` are the right
                    // child's fraction and the integer part lies from limb `turn` up, so the
                    // bits between the two run from above the integer part to bit `L` and on
                    // from bit 0 to the right child.
                    let below = turn.saturating_sub(k_right);
                    let gap_from = (turn as u64) * 64 + b_mult + 1;
                    let gap_nonzero = has_nonzero_in_bit_range(&w_limbs, gap_from, active_l_bits)
                        || has_nonzero_in_bit_range(&w_limbs, 0, (below as u64) * 64);
                    if turn >= k_right && r_start_bit >= b_mult + 32 && gap_nonzero {
                        let lim = |i: usize| i.min(w_limbs.len());
                        let w0_of = |from: usize, to: usize| {
                            if lim(from) < lim(to) {
                                compute_limbs_w0_par(
                                    ctx.kernels,
                                    ctx.sched,
                                    &w_limbs[lim(from)..lim(to)],
                                    node_par_w,
                                )
                            } else {
                                0
                            }
                        };
                        // The right child's check words, piece by piece between the places
                        // where it and the left children passed on from it will be divided.
                        let whole_child = [0usize, k_right];
                        let cuts: &[usize] = if child_cuts.first() == Some(&0)
                            && child_cuts.last() == Some(&k_right)
                            && cut_pairs(child_cuts).all(|(a, b)| a < b)
                        {
                            child_cuts
                        } else {
                            &whole_child
                        };
                        let below_w0 = w0_of(0, below);
                        let piece_w0s: Vec<u64> = cut_pairs(cuts)
                            .map(|(a, b)| w0_of(below + a, below + b))
                            .collect();
                        let mut right_w0 = 0u64;
                        for (i, &pw0) in piece_w0s.iter().enumerate().rev() {
                            let len = (cuts[i + 1] - cuts[i]) as u64;
                            right_w0 = mod61_add(mod61_mul(right_w0, mod61_pow2(len * 64)), pw0);
                        }
                        // What the turn took from the top of the plain product is this one's
                        // lowest `turn` limbs `c`: plain product * `2^(64 * turn)` = this one +
                        // `c * (2^L - 1)`. That gives the plain product's check word, which is
                        // what the check below is stated in.
                        let around_w0 = mod61_add(
                            below_w0,
                            mod61_mul(mod61_pow2((below as u64) * 64), right_w0),
                        );
                        let right_seg = SegWords {
                            cuts: cuts.to_vec(),
                            w0s: piece_w0s,
                        };
                        let wrap_w0 = crate::refk::mod61_sub(mod61_pow2(active_l_bits), 1);
                        let turn_bits = (turn as u64) * 64;
                        let plain_w0 = mod61_mul(
                            mod61_add(w_w0, mod61_mul(around_w0, wrap_w0)),
                            mod61_pow2(61 - turn_bits % 61),
                        );
                        let copied = lim(turn) - lim(below);
                        let x_right = shift_limbs_down_in_place_par(
                            w_limbs,
                            below,
                            copied,
                            k_right,
                            ctx.sched,
                            node_par_w,
                        );
                        let right_under_half = (x_right[k_right - 1] >> 63) == 0;
                        node_prod = Some((
                            x_right,
                            right_under_half,
                            NodeCheck::Wrapped {
                                wrapped_w0: plain_w0,
                                x_shifted_w0: mod61_mul(f_w0_of(), mod61_pow2(active_l_bits - x_bits)),
                                l_bits: active_l_bits,
                                all_zeros_kept: false,
                                top_term_w0,
                            },
                            Some(right_seg),
                        ));
                    } else {
                        crate::limb::put_limb_buf(w_limbs);
                    }
                }
            }
        }
    }
    if let Some(v) = node_prod {
        return Ok(v);
    }

    // The whole product: its limbs from `f_k` up are the integer part.
    let x_w0 = f_w0_of();
    let exp_prod_w0 = mod61_mul(x_w0, p_entry.mult.w0());
    let prod_limbs = if !force_whole && use_k <= max_cached_k && p_entry.whole_cached.is_some() {
        let mut m_adj = p_entry.whole_method.clone();
        if let MulMethod::Ntt(ref mut s) = m_adj {
            s.par_workers = node_par_w;
        }
        ctx.mul_limbs_with_cached(
            point_id,
            &m_adj,
            f_limbs,
            &p_entry.mult.limbs,
            p_entry.whole_cached.as_ref().unwrap(),
            f_k + p_entry.mult.limbs.len() + 2,
            Some(exp_prod_w0),
        )?
        .0
    } else {
        let mut vx = crate::limb::take_limb_buf_uninit(f_limbs.len());
        vx.extend_from_slice(f_limbs);
        let mut bx = BigNum::from_limbs_par(vx, false, 0, ctx.kernels, ctx.sched, node_par_w);
        let mut bp = p_entry.mult.clone();
        let m = plan.select_mul_method(
            (f_k as u64) * 64,
            bp.body_bits(),
            false,
            false,
            node_par_w,
            shares,
            cal,
        );
        let res_bn = ctx.mul_whole(point_id, &m, &mut bx, &mut bp)?;
        crate::limb::put_limb_buf(bx.into_limbs());
        crate::limb::put_limb_buf(bp.into_limbs());
        res_bn.into_limbs()
    };

    let mut padded = prod_limbs;
    if padded.len() < f_k {
        padded.resize(f_k, 0);
    }
    let int_w0 = if padded.len() > f_k {
        compute_limbs_w0_par(ctx.kernels, ctx.sched, &padded[f_k..], node_par_w)
    } else {
        0
    };
    let mut xr = shr_limbs_par(
        &padded[..f_k],
        ((f_k - k_right) as u64) * 64,
        ctx.sched,
        node_par_w,
    );
    crate::limb::put_limb_buf(padded);
    xr.resize(k_right, 0);
    let under_half = (xr[k_right - 1] >> 63) == 0;
    Ok((
        xr,
        under_half,
        NodeCheck::Whole {
            expected_left_w0: mod61_add(int_w0, top_term_w0),
        },
        None,
    ))
}

/// Where a node's own product will cut its fraction of `k_limbs` limbs: below the limbs its
/// level strips. A node that is not split by a product (a single leaf) is not cut.
fn node_cut(pow10_levels: &[Pow10Entry], leaf_count: u64, k_limbs: usize) -> usize {
    if leaf_count < 2 {
        return k_limbs;
    }
    let lv = (leaf_count - 1).ilog2() as usize;
    k_limbs.saturating_sub(pow10_levels[lv].strip_limbs)
}

/// The places where the fraction of a node of `leaf_count` leaves, the limbs
/// `[base, base + k_limbs)` of its buffer, will be divided: by the node's own product
/// (`node_cut`) and, for `levels` of the left children that are passed on as the top of it,
/// at their lower ends and by their own products. Ascending, from `base` to `base + k_limbs`.
fn chain_cuts(
    pow10_levels: &[Pow10Entry],
    leaf_count: u64,
    k_limbs: usize,
    base: usize,
    levels: usize,
) -> Vec<usize> {
    let top = base + k_limbs;
    let mut cuts = vec![base, top];
    let (mut lc, mut k) = (leaf_count, k_limbs);
    for depth in 0..=levels {
        if lc < 2 {
            break;
        }
        let off = top - k;
        if depth > 0 {
            cuts.push(off);
        }
        cuts.push(off + node_cut(pow10_levels, lc, k));
        let left = 1u64 << (lc - 1).ilog2();
        let k_left = frac_limbs_for_digits(left * (LEAF_DIGITS as u64));
        if k_left > k {
            break;
        }
        lc = left;
        k = k_left;
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Workers for nodes that run at once, in proportion to their sizes: each has one, and the
/// others go one at a time to the node with the most left to do for each worker it has, none
/// beyond its cap.
fn workers_by_size(workers: usize, sizes: &[u64], caps: &[usize]) -> Vec<usize> {
    let mut pw = vec![1usize; sizes.len()];
    let mut left = workers.saturating_sub(sizes.len());
    while left > 0 {
        let mut best: Option<usize> = None;
        for i in 0..sizes.len() {
            if pw[i] >= caps[i].max(1) {
                continue;
            }
            let more_to_do = |b: usize| {
                (sizes[i] as u128) * (pw[b] as u128) > (sizes[b] as u128) * (pw[i] as u128)
            };
            if best.is_none_or(more_to_do) {
                best = Some(i);
            }
        }
        let Some(b) = best else { break };
        pw[b] += 1;
        left -= 1;
    }
    pw
}

struct RootBorrow {
    undo: Mutex<Vec<(usize, u64)>>,
    parked: Mutex<Option<LimbVec>>,
}

#[inline]
fn release_frontier_buf(buf: LimbVec, is_root_buf: bool, root_borrow: &RootBorrow) {
    if buf.is_empty() {
        return;
    }
    if is_root_buf {
        *root_borrow.parked.lock().unwrap() = Some(buf);
    } else {
        crate::limb::put_limb_buf(buf);
    }
}

#[inline]
fn can_track_root_eta(slice: &[u64], k_left: usize, b_left: u64, move_up: bool) -> bool {
    if slice.len() < k_left || (k_left as u64) * 64 <= b_left + 4 {
        return false;
    }
    let local_eta_bit = (k_left as u64) * 64 - (b_left + 2);
    let eta_word = (local_eta_bit / 64) as usize;
    if eta_word >= k_left {
        return false;
    }
    let eta_mask = 1u64 << (local_eta_bit & 63);
    if move_up {
        if slice[eta_word] <= u64::MAX - eta_mask {
            return true;
        }
        (1..4).any(|j| eta_word + j < k_left && slice[eta_word + j] != u64::MAX)
    } else {
        if slice[eta_word] >= eta_mask {
            return true;
        }
        (1..4).any(|j| eta_word + j < k_left && slice[eta_word + j] != 0)
    }
}

struct FrontierNode {
    start_leaf: u64,
    leaf_count: u64,
    /// The buffer the fraction lies in: it is the limbs `[frac_off, frac_off + k_limbs)`, which
    /// end with the buffer (a fraction whose top limbs are zero may come in a shorter one, at
    /// `frac_off` zero).
    frac_limbs: LimbVec,
    frac_off: usize,
    k_limbs: usize,
    /// How many splits in a row have passed this buffer on without moving it (`VIEW_LEVELS`).
    views: usize,
    /// The buffer's check words where they were taken (see `SegWords`).
    seg: Option<SegWords>,
    /// True when `frac_limbs` is the owned root buffer (tracked via `RootBorrow` for Rung 2 retry).
    is_root_buf: bool,
}

struct SplitRecord {
    parent_slot: usize,
    left_slot: usize,
    right_slot: usize,
    d_right: u64,
    lv: usize,
    point_id: u64,
    check: NodeCheck,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn split_frontier_one(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    pow10_levels: &[Pow10Entry],
    start_leaf: u64,
    leaf_count: u64,
    use_limbs: LimbVec,
    use_off: usize,
    use_k: usize,
    views: usize,
    seg: Option<SegWords>,
    is_root_buf: bool,
    root_borrow: &RootBorrow,
    node_par_w: usize,
    rem_splits_ctr: Option<(&AtomicUsize, usize)>,
) -> Result<(FrontierNode, FrontierNode, u64, usize, u64, NodeCheck), MulError> {
    let cur_pw = || -> usize {
        if let Some((ctr, max_conc)) = rem_splits_ctr {
            let rem = ctr.load(Ordering::Relaxed).max(1).min(max_conc.max(1));
            let aw = shares.active_workers.max(1);
            if rem == 1 {
                aw
            } else {
                (aw / rem).max(node_par_w)
            }
        } else {
            node_par_w
        }
    };
    let lv = (leaf_count - 1).ilog2() as usize;
    let left_leaves = 1u64 << lv;
    let right_leaves = leaf_count - left_leaves;
    let d_left = left_leaves * (LEAF_DIGITS as u64);
    let d_right = right_leaves * (LEAF_DIGITS as u64);
    let k_left = frac_limbs_for_digits(d_left);
    let k_right = frac_limbs_for_digits(d_right);
    let b_left = bit_ceil_pow10(d_left);
    let p_entry = &pow10_levels[lv];
    let point_id = 0x5000_0000u64 | ((lv as u64) << 44) | (start_leaf << 4);

    let t_split = Instant::now();
    // A fraction that ends with its buffer can pass its top on as the left child where it lies.
    let whole = use_limbs.len() == use_off + use_k;
    debug_assert!(whole || use_off == 0);
    let may_view = whole && !plan.out_of_core && use_k >= k_left;
    // The node's check words: those that came with its buffer, or else taken now in one pass,
    // piece by piece for the left children that will be passed on from it as well.
    let mut seg = seg;
    let own_cut = node_cut(pow10_levels, leaf_count, use_k);
    let mut own_w0 = seg.as_ref().and_then(|sw| sw.parts(use_off, own_cut, use_k));
    if own_w0.is_none() && whole {
        let levels = if may_view { VIEW_LEVELS.saturating_sub(views) } else { 0 };
        let cuts = chain_cuts(pow10_levels, leaf_count, use_k, use_off, levels);
        let sw = SegWords::compute(ctx.kernels, ctx.sched, &use_limbs, cuts, cur_pw());
        own_w0 = sw.parts(use_off, own_cut, use_k);
        seg = Some(sw);
    }
    let right_cuts = chain_cuts(
        pow10_levels,
        right_leaves,
        k_right,
        0,
        if plan.out_of_core { 0 } else { VIEW_LEVELS },
    );
    let max_cached_k = frac_limbs_for_digits(2 * left_leaves * (LEAF_DIGITS as u64));
    let right_res = node_right_fraction(
        ctx,
        plan,
        shares,
        cal,
        p_entry,
        point_id,
        &use_limbs[use_off.min(use_limbs.len())..],
        use_k,
        k_right,
        max_cached_k,
        cur_pw(),
        false,
        own_w0,
        &right_cuts,
    );
    let (x_right, right_under_half, check_info, right_seg) = match right_res {
        Ok(v) => v,
        Err(e) => {
            release_frontier_buf(use_limbs, is_root_buf, root_borrow);
            return Err(e);
        }
    };

    // The left child is the top `k_left` limbs of the node's fraction, adjusted by half a guard.
    let can_view_here = may_view
        && views < VIEW_LEVELS
        && (!is_root_buf
            || can_track_root_eta(
                &use_limbs[(use_off + (use_k - k_left))..(use_off + use_k)],
                k_left,
                b_left,
                right_under_half,
            ));
    let post_pw = cur_pw();
    let (left_limbs, left_off, left_views, left_seg, left_is_root) = if can_view_here {
        // Passed on where it lies. The guard changes a limb or two at its lower end, and the
        // check words of the piece they are in follow.
        let mut buf = use_limbs;
        let off = use_off + (use_k - k_left);
        let change =
            apply_guard_eta_tracked(&mut buf[off..off + k_left], k_left, b_left, right_under_half);
        if is_root_buf {
            if let EtaChange::Limbs { first, old, n } = change {
                let mut u = root_borrow.undo.lock().unwrap();
                for j in 0..n {
                    u.push((off + first + j, old[j]));
                }
            }
        }
        let seg = seg.and_then(|mut sw| match change {
            EtaChange::Limbs { first, old, n } => (0..n)
                .all(|j| sw.limb_changed(off + first + j, old[j], buf[off + first + j]))
                .then_some(sw),
            EtaChange::Zeroed => sw.zeroed_from(off).then_some(sw),
            EtaChange::Many => None,
        });
        (buf, off, views + 1, seg, is_root_buf)
    } else if is_root_buf {
        let buf = make_guard_adjusted_left_fraction(
            ctx.kernels,
            &use_limbs[use_off.min(use_limbs.len())..],
            use_k,
            k_left,
            b_left,
            right_under_half,
            ctx.sched,
            post_pw,
        );
        release_frontier_buf(use_limbs, true, root_borrow);
        (buf, 0, 0, None, false)
    } else {
        // Moved to the start of the buffer, and the rest of the buffer given back.
        let buf = make_guard_adjusted_left_fraction_in_place(
            ctx.kernels,
            use_limbs,
            use_off + use_k,
            k_left,
            b_left,
            right_under_half,
            ctx.sched,
            post_pw,
        );
        (buf, 0, 0, None, false)
    };
    RADIX_SPLIT_NS.fetch_add(t_split.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok((
        FrontierNode {
            start_leaf,
            leaf_count: left_leaves,
            frac_limbs: left_limbs,
            frac_off: left_off,
            k_limbs: k_left,
            views: left_views,
            seg: left_seg,
            is_root_buf: left_is_root,
        },
        FrontierNode {
            start_leaf: start_leaf + left_leaves,
            leaf_count: right_leaves,
            frac_limbs: x_right,
            frac_off: 0,
            k_limbs: k_right,
            views: 0,
            seg: right_seg,
            is_root_buf: false,
        },
        d_right,
        lv,
        point_id,
        check_info,
    ))
}

/// Breadth-first top-frontier expansion followed by a single LPT-ordered `par_for` over
/// independent 1-worker subtrees, eliminating depth-first barrier stalls on unbalanced trees!
#[allow(clippy::too_many_arguments)]
fn convert_frontier_fast(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    pow10_levels: &mut [Pow10Entry],
    num_leaves: u64,
    root_frac_limbs: &mut LimbVec,
    root_off: usize,
    root_k: usize,
    kept_full_leaves: u64,
    rem_digits: usize,
    sink: &dyn DigitSink,
    guard_tail: &Mutex<Vec<u8>>,
    spine_completed: &Mutex<Vec<(u64, u64, u64)>>,
    last_ckpt: &Mutex<Instant>,
) -> Result<Option<u64>, MulError> {
    let steal_cap = (shares.active_workers.max(1) * 16).max(4);
    let root_borrow = RootBorrow {
        undo: Mutex::new(Vec::with_capacity(8)),
        parked: Mutex::new(None),
    };
    let reclaim_all_nodes = |nodes: &mut [FrontierNode], out_root: &mut LimbVec| {
        let mut found = root_borrow.parked.lock().unwrap().take();
        for nd in nodes.iter_mut() {
            if nd.is_root_buf && found.is_none() && !nd.frac_limbs.is_empty() {
                found = Some(take_limb_vec(&mut nd.frac_limbs));
            } else if !nd.frac_limbs.is_empty() {
                crate::limb::put_limb_buf(take_limb_vec(&mut nd.frac_limbs));
            }
        }
        if let Some(mut rb) = found {
            let undo = root_borrow.undo.lock().unwrap();
            for &(idx, old_v) in undo.iter().rev() {
                rb[idx] = old_v;
            }
            *out_root = rb;
        }
    };

    let mut nodes: Vec<FrontierNode> = Vec::with_capacity(steal_cap * 4);
    nodes.push(FrontierNode {
        start_leaf: 0,
        leaf_count: num_leaves,
        frac_limbs: take_limb_vec(root_frac_limbs),
        frac_off: root_off,
        k_limbs: root_k,
        views: 0,
        seg: None,
        is_root_buf: true,
    });
    let mut records: Vec<SplitRecord> = Vec::with_capacity(steal_cap * 2);
    let mut active_slots: Vec<usize> = vec![0];

    fn entry_2d_footprint(leaf_cnt: u64, p10: &[Pow10Entry]) -> (u64, u64, usize, bool, bool) {
        if leaf_cnt <= 1 {
            return (0, 0, 0, false, false);
        }
        let lv = (leaf_cnt - 1).ilog2() as usize;
        let e = &p10[lv];
        let spec_opt = e.wrapped_spec.or(match e.whole_method {
            MulMethod::Ntt(ws) => Some(ws),
            _ => None,
        });
        if let Some(ws) = spec_opt
            && ws.rows > 1
        {
            let gb = (ws.k_primes as u64)
                .saturating_mul(ws.rows as u64)
                .saturating_mul(ws.outer_row_bytes() as u64);
            let up_rb = if ws.cols >= 512 {
                ws.cols * 8 + 512
            } else {
                ws.cols * 8
            };
            let prime_b = (ws.rows as u64).saturating_mul(up_rb as u64);
            (gb, prime_b, ws.rows, ws.out_of_core, ws.packed)
        } else {
            (0, 0, 0, false, false)
        }
    }
    let do_trace = std::env::var_os("SUPI_TRACE").is_some();
    let t_bfs0 = Instant::now();
    loop {
        let mut split_slots: Vec<usize> = Vec::new();
        let mut next_active: Vec<usize> = Vec::with_capacity(active_slots.len() * 2);
        let need_more_width = active_slots.len() < steal_cap;

        let active_w_u64 = shares.active_workers.max(1) as u64;
        for &slot in &active_slots {
            let lc = nodes[slot].leaf_count;
            let (gb, _, rw, ws_ooc, _) = entry_2d_footprint(lc, pow10_levels);
            let exceeds_l3 = if ws_ooc {
                true
            } else {
                let par_nodes = active_w_u64.min(active_slots.len().max(1) as u64);
                let l3_cap = shares.effective_l3_bytes(par_nodes, rw.max(1)) * 7 / 8;
                let node_set = gb.saturating_add(2 * (nodes[slot].k_limbs as u64) * 8);
                let tot_set = gb.saturating_add(par_nodes.saturating_mul(node_set));
                tot_set > l3_cap || tot_set > plan.ram_transform_budget
            };
            if lc > BATCH_MAX_LEAVES && (need_more_width || exceeds_l3) {
                split_slots.push(slot);
            } else {
                next_active.push(slot);
            }
        }
        if split_slots.is_empty() {
            break;
        }
        split_slots.sort_by_key(|&s| std::cmp::Reverse(nodes[s].leaf_count));

        let wave_t0 = std::env::var_os("SUPI_TRACE")
            .is_some()
            .then(std::time::Instant::now);
        let wave_splits = split_slots.len();
        let wave_max_lc = split_slots
            .iter()
            .map(|&s| nodes[s].leaf_count)
            .max()
            .unwrap_or(0);
        let any_2d = split_slots
            .iter()
            .any(|&s| entry_2d_footprint(nodes[s].leaf_count, pow10_levels).0 > 0);
        let any_ooc_2d = split_slots
            .iter()
            .any(|&s| entry_2d_footprint(nodes[s].leaf_count, pow10_levels).3);
        let conc_splits = if split_slots.len() <= 1 || shares.active_workers <= 1 || any_ooc_2d {
            1usize
        } else if !any_2d {
            split_slots.len().min(shares.active_workers.max(1))
        } else {
            let mut max_grid_1 = 0u64;
            let mut max_prime_b = 0u64;
            let mut max_node_set = 0u64;
            let mut top_2d_rows = 1usize;
            let mut any_packed = false;
            for &sl in &split_slots {
                let (g, pb, rw, _, pk) = entry_2d_footprint(nodes[sl].leaf_count, pow10_levels);
                if g > 0 {
                    if g > max_grid_1 || (g == max_grid_1 && rw > top_2d_rows) {
                        top_2d_rows = rw.max(1);
                    }
                    max_grid_1 = max_grid_1.max(g);
                    max_prime_b = max_prime_b.max(pb);
                    any_packed |= pk;
                    // what one node's product touches: its grid, its fraction and its result
                    max_node_set =
                        max_node_set.max(g + 2 * (nodes[sl].k_limbs as u64) * 8);
                }
            }
            let aw = shares.active_workers.max(1);
            if max_grid_1 == 0 {
                split_slots.len().min(aw)
            } else {
                // As many nodes at once as fit in the last-level cache together, next to the
                // level's cached transform of the power of ten that they all read. More than
                // that and every sweep of every node runs from memory: measured at 1e9 digits,
                // 16 nodes of 2^18 coefficients at once took 108 ms a wave, 4 at once 82 ms.
                let eff_l3_raw = shares.effective_l3_bytes(
                    split_slots.len().min(aw).max(1) as u64,
                    top_2d_rows,
                );
                let l3_90 = (eff_l3_raw * 9) / 10;
                let l3_tot = (eff_l3_raw * 7) / 8;
                let by_grid_2 = if !any_packed
                    && max_grid_1.saturating_add(max_prime_b) <= l3_90
                {
                    (l3_90.saturating_sub(max_prime_b) / max_grid_1.max(1)).clamp(1, 2)
                } else {
                    0
                };
                let by_l3 = (l3_tot.saturating_sub(max_grid_1) / max_node_set.max(1))
                    .max(by_grid_2)
                    .max(1);
                let by_ram =
                    plan.ram_transform_budget.saturating_sub(max_grid_1) / max_node_set.max(1);
                let spills_l3 = any_packed || max_grid_1.saturating_mul(2) > l3_90;
                let eff_l3 = if split_slots.len() >= 2
                    && spills_l3
                    && max_node_set.saturating_mul(2) <= plan.ram_transform_budget / 2
                {
                    let pbp_c = if top_2d_rows >= 64 {
                        4
                    } else if max_prime_b <= l3_tot {
                        (l3_tot / (max_prime_b.saturating_mul(5) / 4).max(1)).clamp(2, 4)
                    } else {
                        1
                    };
                    let ram_half = (plan.ram_transform_budget / 2) / max_node_set.max(1);
                    by_l3.max(pbp_c.min(ram_half))
                } else {
                    by_l3
                };
                let cap_c = (eff_l3.min(by_ram) as usize).max(1).min(aw);
                let mut chosen = 1usize;
                for c in crate::tier::even_splits(aw).into_iter().rev() {
                    let pw_c = (aw / c).max(1);
                    if c >= 1
                        && c <= aw
                        && c <= cap_c
                        && c <= split_slots.len()
                        && (pw_c <= top_2d_rows || c == 1)
                    {
                        chosen = c;
                        break;
                    }
                }
                if chosen == 1
                    && split_slots.len() < aw
                    && split_slots.len() <= cap_c
                    && (aw / split_slots.len().max(1)).max(1) <= top_2d_rows
                {
                    split_slots.len().max(1)
                } else {
                    chosen
                }
            }
        };

        if conc_splits <= 1 {
            let pw = shares.active_workers.max(1);
            let p10_ref: &[Pow10Entry] = &*pow10_levels;
            for slot in split_slots {
                let parent_limbs = take_limb_vec(&mut nodes[slot].frac_limbs);
                let parent_seg = nodes[slot].seg.take();
                let parent_is_root = nodes[slot].is_root_buf;
                nodes[slot].is_root_buf = false;
                let res = split_frontier_one(
                    ctx,
                    plan,
                    shares,
                    cal,
                    p10_ref,
                    nodes[slot].start_leaf,
                    nodes[slot].leaf_count,
                    parent_limbs,
                    nodes[slot].frac_off,
                    nodes[slot].k_limbs,
                    nodes[slot].views,
                    parent_seg,
                    parent_is_root,
                    &root_borrow,
                    pw,
                    None,
                );
                let (l_node, r_node, d_right, lv, point_id, check) = match res {
                    Ok(v) => v,
                    Err(e) => {
                        reclaim_all_nodes(&mut nodes, root_frac_limbs);
                        return Err(e);
                    }
                };
                let l_slot = nodes.len();
                nodes.push(l_node);
                let r_slot = nodes.len();
                nodes.push(r_node);
                next_active.push(l_slot);
                next_active.push(r_slot);
                records.push(SplitRecord {
                    parent_slot: slot,
                    left_slot: l_slot,
                    right_slot: r_slot,
                    d_right,
                    lv,
                    point_id,
                    check,
                });
            }
        } else {
            let aw = shares.active_workers.max(1);
            let mut wave_out: Vec<
                Option<
                    Result<
                        (FrontierNode, FrontierNode, u64, usize, u64, NodeCheck),
                        MulError,
                    >,
                >,
            > = (0..split_slots.len()).map(|_| None).collect();
            // What the workers of a small wave are shared out by: each node's size, and the
            // most workers its product has rows for (one where it is not a grid).
            let wave_sizes: Vec<u64> = split_slots.iter().map(|&s| nodes[s].leaf_count).collect();
            let wave_caps: Vec<usize> = wave_sizes
                .iter()
                .map(|&lc| entry_2d_footprint(lc, pow10_levels).2.max(1))
                .collect();
            let nodes_ptr = nodes.as_mut_ptr() as usize;
            let out_ptr = wave_out.as_mut_ptr() as usize;
            let p10_ref: &[Pow10Entry] = &*pow10_levels;
            let rb_ref: &RootBorrow = &root_borrow;
            let run_split = |idx: usize, pw: usize, rem_ctr: Option<(&AtomicUsize, usize)>| {
                let slot = split_slots[idx];
                let (start_leaf, leaf_count, frac_off, k_limbs, views, parent_seg, parent_is_root, parent_limbs) = unsafe {
                    let nd = &mut *(nodes_ptr as *mut FrontierNode).add(slot);
                    let is_rb = nd.is_root_buf;
                    nd.is_root_buf = false;
                    (
                        nd.start_leaf,
                        nd.leaf_count,
                        nd.frac_off,
                        nd.k_limbs,
                        nd.views,
                        nd.seg.take(),
                        is_rb,
                        take_limb_vec(&mut nd.frac_limbs),
                    )
                };
                let r = split_frontier_one(
                    ctx,
                    plan,
                    shares,
                    cal,
                    p10_ref,
                    start_leaf,
                    leaf_count,
                    parent_limbs,
                    frac_off,
                    k_limbs,
                    views,
                    parent_seg,
                    parent_is_root,
                    rb_ref,
                    pw,
                    rem_ctr,
                );
                if let Some((ctr, _)) = rem_ctr {
                    ctr.fetch_sub(1, Ordering::Relaxed);
                }
                unsafe {
                    *(out_ptr
                        as *mut Option<
                            Result<
                                (FrontierNode, FrontierNode, u64, usize, u64, NodeCheck),
                                MulError,
                            >,
                        >)
                        .add(idx) = Some(r);
                }
            };

            if split_slots.len() <= 2 * conc_splits {
                // A wave of few nodes. With equal shares, a node that is done leaves its
                // workers with nothing to do until the largest one ends (measured at 1e9
                // digits: a sixth of the thread time of the wave of two nodes, a fifth of the
                // wave of four). So the nodes run in rounds of `conc_splits`, largest first,
                // each with workers in proportion to its size, and a round ends together.
                let mut start = 0usize;
                while start < split_slots.len() {
                    let end = (start + conc_splits).min(split_slots.len());
                    let pws = workers_by_size(aw, &wave_sizes[start..end], &wave_caps[start..end]);
                    if end - start == 1 {
                        run_split(start, pws[0], None);
                    } else {
                        let round_rem = AtomicUsize::new(end - start);
                        let rem_ref = &round_rem;
                        let max_c = end - start;
                        ctx.sched.par_for_max(max_c, 1, max_c, |ri| {
                            run_split(start + ri, pws[ri], Some((rem_ref, max_c)));
                        });
                    }
                    start = end;
                }
            } else {
                let pw_full = (aw / conc_splits).max(1);
                let wave_rem = AtomicUsize::new(split_slots.len());
                let rem_ref = &wave_rem;
                ctx.sched
                    .par_for_max(split_slots.len(), 1, conc_splits, |idx| {
                        run_split(idx, pw_full, Some((rem_ref, conc_splits)));
                    });
            }

            for (idx, slot) in split_slots.into_iter().enumerate() {
                let (l_node, r_node, d_right, lv, point_id, check) =
                    match wave_out[idx].take().unwrap() {
                        Ok(v) => v,
                        Err(e) => {
                            for o in &mut wave_out[(idx + 1)..] {
                                if let Some(Ok((ln, rn, ..))) = o.take() {
                                    release_frontier_buf(ln.frac_limbs, ln.is_root_buf, &root_borrow);
                                    release_frontier_buf(rn.frac_limbs, rn.is_root_buf, &root_borrow);
                                }
                            }
                            reclaim_all_nodes(&mut nodes, root_frac_limbs);
                            return Err(e);
                        }
                    };
                let l_slot = nodes.len();
                nodes.push(l_node);
                let r_slot = nodes.len();
                nodes.push(r_node);
                next_active.push(l_slot);
                next_active.push(r_slot);
                records.push(SplitRecord {
                    parent_slot: slot,
                    left_slot: l_slot,
                    right_slot: r_slot,
                    d_right,
                    lv,
                    point_id,
                    check,
                });
            }
        }
        if let Some(wt) = wave_t0 {
            let wm = wt.elapsed().as_secs_f64() * 1e3;
            let split_ms = RADIX_SPLIT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
            let prod_ms = RADIX_SPLIT_PROD_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
            if wm >= 15.0 {
                eprintln!(
                    "[TRACE CONV WAVE] splits={wave_splits} conc={conc_splits} max_lc={wave_max_lc} -> {wm:.2}ms | in the splits {split_ms:.1}ms, of that products {prod_ms:.1}ms"
                );
            }
        }
        active_slots = next_active;

        // Immediately recycle cached transforms above the highest remaining active `lv` back to `buf_pool`!
        let max_rem_lv = active_slots
            .iter()
            .map(|&s| {
                let lc = nodes[s].leaf_count;
                if lc > 1 { (lc - 1).ilog2() as usize } else { 0 }
            })
            .max()
            .unwrap_or(0);
        for entry in pow10_levels.iter_mut().skip(max_rem_lv + 1) {
            if let Some(mut c) = entry.wrapped_cached.take() {
                c.release_store(ctx.store);
            }
            if let Some(mut c) = entry.whole_cached.take() {
                c.release_store(ctx.store);
            }
        }
    }

    // Sort subtrees largest-first (`leaf_count` descending) to prevent unequal-size tail
    // stragglers, breaking equal-size ties by ascending file offset (`start_leaf`).
    active_slots.sort_by(|&a, &b| {
        nodes[b]
            .leaf_count
            .cmp(&nodes[a].leaf_count)
            .then_with(|| nodes[a].start_leaf.cmp(&nodes[b].start_leaf))
    });
    let bfs_ms = t_bfs0.elapsed().as_secs_f64() * 1e3;
    let t_sub0 = Instant::now();

    let mut slot_w0: Vec<u64> = vec![0u64; nodes.len()];
    let slot_w0_ptr = slot_w0.as_mut_ptr() as usize;
    let nodes_ptr = nodes.as_mut_ptr() as usize;
    let stop_sub = std::sync::atomic::AtomicBool::new(false);
    let retry_sub = std::sync::atomic::AtomicBool::new(false);
    let first_err: Mutex<Option<MulError>> = Mutex::new(None);
    let seq_par_nodes = (ctx.sched.subtree_steal_max_nodes() as u64)
        .max(steal_cap as u64)
        .max(1);
    let p10_sub_ref: &[Pow10Entry] = &*pow10_levels;
    let rb_sub_ref: &RootBorrow = &root_borrow;

    ctx.sched.par_for(active_slots.len(), 1, |idx| {
        if stop_sub.load(Ordering::Relaxed) {
            return;
        }
        let slot = active_slots[idx];
        let nd_owned = unsafe {
            let p = (nodes_ptr as *mut FrontierNode).add(slot);
            let is_rb = (*p).is_root_buf;
            (*p).is_root_buf = false;
            FrontierNode {
                start_leaf: (*p).start_leaf,
                leaf_count: (*p).leaf_count,
                frac_limbs: take_limb_vec(&mut (*p).frac_limbs),
                frac_off: (*p).frac_off,
                k_limbs: (*p).k_limbs,
                views: (*p).views,
                seg: (*p).seg.take(),
                is_root_buf: is_rb,
            }
        };
        let _subtree_lifo = SubtreeLifoScope::enter();
        let res = if nd_owned.leaf_count > 1 && nd_owned.leaf_count <= 1_048_576 {
            let (base_off, max_bytes) = match plan.format {
                OutputFormat::Text => (
                    2 + nd_owned.start_leaf * (LEAF_DIGITS as u64),
                    (nd_owned.leaf_count as usize) * LEAF_DIGITS,
                ),
                OutputFormat::Packed19 => (
                    16 + nd_owned.start_leaf * 8,
                    (nd_owned.leaf_count as usize) * 8,
                ),
            };
            let coal = SubtreeCoalesceSink::new(base_off, max_bytes);
            let r = eval_frontier_subtree_owned(
                ctx,
                plan,
                shares,
                cal,
                p10_sub_ref,
                nd_owned,
                rb_sub_ref,
                kept_full_leaves,
                rem_digits,
                num_leaves,
                &coal,
                guard_tail,
                spine_completed,
                last_ckpt,
                seq_par_nodes,
            );
            match r {
                Ok(Some(w)) => coal.flush_to(sink).map(|()| Some(w)).map_err(MulError::FatalStop),
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            eval_frontier_subtree_owned(
                ctx,
                plan,
                shares,
                cal,
                p10_sub_ref,
                nd_owned,
                rb_sub_ref,
                kept_full_leaves,
                rem_digits,
                num_leaves,
                sink,
                guard_tail,
                spine_completed,
                last_ckpt,
                seq_par_nodes,
            )
        };

        match res {
            Ok(Some(w0)) => unsafe {
                *(slot_w0_ptr as *mut u64).add(slot) = w0;
            },
            Ok(None) => {
                retry_sub.store(true, Ordering::Relaxed);
                stop_sub.store(true, Ordering::Relaxed);
            }
            Err(e) => {
                stop_sub.store(true, Ordering::Relaxed);
                let mut g = first_err.lock().unwrap();
                if g.is_none() {
                    *g = Some(e);
                }
            }
        }
    });

    let sub_ms = t_sub0.elapsed().as_secs_f64() * 1e3;
    let t_ver0 = Instant::now();

    if let Some(e) = first_err.into_inner().unwrap() {
        reclaim_all_nodes(&mut nodes, root_frac_limbs);
        return Err(e);
    }
    if retry_sub.load(Ordering::Relaxed) {
        reclaim_all_nodes(&mut nodes, root_frac_limbs);
        return Ok(None);
    }

    for nd in &mut nodes {
        if !nd.frac_limbs.is_empty() {
            release_frontier_buf(take_limb_vec(&mut nd.frac_limbs), nd.is_root_buf, &root_borrow);
            nd.is_root_buf = false;
        }
    }

    // Bottom-up `MOD_W0` verification across every split record (Section 11)!
    for rec in records.iter().rev() {
        let left_w0 = slot_w0[rec.left_slot];
        let right_w0 = slot_w0[rec.right_slot];
        let p_entry = &pow10_levels[rec.lv];
        let check_ok = rec.check.holds(left_w0, p_entry.mult_w0);
        if !check_ok {
            ctx.telemetry
                .record_retry(crate::par::current_core_id(), 0, None);
            let rung = ctx.ladder.claim_rung(rec.point_id, Rung::MakeAgain);
            match rung {
                Rung::Correct | Rung::MakeAgain => {
                    reclaim_all_nodes(&mut nodes, root_frac_limbs);
                    return Ok(None);
                }
                Rung::GoBack => {
                    reclaim_all_nodes(&mut nodes, root_frac_limbs);
                    return Err(MulError::NeedRollback(rec.point_id));
                }
                Rung::Stop => {
                    reclaim_all_nodes(&mut nodes, root_frac_limbs);
                    return Err(MulError::FatalStop(
                        "Radix conversion node check failed across all ladder rungs".to_string(),
                    ));
                }
            }
        }
        let pow10_right_w0 = pow10_mod_w0(rec.d_right);
        slot_w0[rec.parent_slot] = mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0);
    }
    if let Some(rb) = root_borrow.parked.lock().unwrap().take() {
        crate::limb::put_limb_buf(rb);
    }
    if do_trace {
        eprintln!(
            "[TRACE RADIX] frontier_fast slots={} bfs={bfs_ms:.2}ms subtrees={sub_ms:.2}ms verify={:.2}ms",
            active_slots.len(),
            t_ver0.elapsed().as_secs_f64() * 1e3
        );
    }

    Ok(Some(slot_w0[0]))
}

/// Evaluate a 1-worker subtree from `convert_frontier_fast` while owning `FrontierNode`,
/// preserving `VIEW_LEVELS` zero-copy left-child views and `SegWords` check-word tracking all the
/// way down to `BATCH_MAX_LEAVES`.
#[allow(clippy::too_many_arguments)]
fn eval_frontier_subtree_owned(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    pow10_levels: &[Pow10Entry],
    nd: FrontierNode,
    root_borrow: &RootBorrow,
    kept_full_leaves: u64,
    rem_digits: usize,
    total_leaves: u64,
    sink: &dyn DigitSink,
    guard_tail: &Mutex<Vec<u8>>,
    spine_completed: &Mutex<Vec<(u64, u64, u64)>>,
    last_ckpt: &Mutex<Instant>,
    seq_par_nodes: u64,
) -> Result<Option<u64>, MulError> {
    if nd.leaf_count == 1 {
        let use_slice = &nd.frac_limbs[nd.frac_off.min(nd.frac_limbs.len())..];
        let res = eval_leaf(
            ctx,
            plan,
            nd.start_leaf,
            use_slice,
            nd.k_limbs,
            kept_full_leaves,
            rem_digits,
            sink,
            guard_tail,
            None,
            0,
        );
        release_frontier_buf(nd.frac_limbs, nd.is_root_buf, root_borrow);
        return res.map(Some);
    }

    if nd.leaf_count <= BATCH_MAX_LEAVES {
        let use_slice = &nd.frac_limbs[nd.frac_off.min(nd.frac_limbs.len())..];
        if ctx.fault_hook.is_none() {
            let t_b0 = Instant::now();
            let b_res = convert_batch_subtree(
                ctx,
                plan,
                cal,
                pow10_levels,
                nd.start_leaf,
                nd.leaf_count,
                use_slice,
                nd.k_limbs,
                kept_full_leaves,
                rem_digits,
                total_leaves,
                sink,
                guard_tail,
            );
            RADIX_BATCH_NS.fetch_add(t_b0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            match b_res {
                Ok(Some(batch_w0)) => {
                    release_frontier_buf(nd.frac_limbs, nd.is_root_buf, root_borrow);
                    return Ok(Some(batch_w0));
                }
                Ok(None) => {}
                Err(e) => {
                    release_frontier_buf(nd.frac_limbs, nd.is_root_buf, root_borrow);
                    return Err(e);
                }
            }
        }
        let empty_completed: HashMap<(u64, u64), u64> = HashMap::new();
        let empty_fracs: HashMap<(u64, u64), PendingFraction> = HashMap::new();
        let nd_w0 = nd.seg.as_ref().and_then(|sw| {
            sw.parts(
                nd.frac_off,
                node_cut(pow10_levels, nd.leaf_count, nd.k_limbs),
                nd.k_limbs,
            )
        });
        let fb = convert_node(
            ctx,
            plan,
            shares,
            cal,
            pow10_levels,
            nd.start_leaf,
            nd.leaf_count,
            use_slice,
            nd.k_limbs,
            nd_w0,
            kept_full_leaves,
            rem_digits,
            total_leaves,
            sink,
            guard_tail,
            None,
            &empty_completed,
            spine_completed,
            &empty_fracs,
            None,
            last_ckpt,
            0,
            seq_par_nodes,
        );
        release_frontier_buf(nd.frac_limbs, nd.is_root_buf, root_borrow);
        return fb.map(Some);
    }

    let (left_nd, right_nd, d_right, lv, point_id, check_info) = split_frontier_one(
        ctx,
        plan,
        shares,
        cal,
        pow10_levels,
        nd.start_leaf,
        nd.leaf_count,
        nd.frac_limbs,
        nd.frac_off,
        nd.k_limbs,
        nd.views,
        nd.seg,
        nd.is_root_buf,
        root_borrow,
        1,
        None,
    )?;

    let left_res = eval_frontier_subtree_owned(
        ctx,
        plan,
        shares,
        cal,
        pow10_levels,
        left_nd,
        root_borrow,
        kept_full_leaves,
        rem_digits,
        total_leaves,
        sink,
        guard_tail,
        spine_completed,
        last_ckpt,
        seq_par_nodes,
    );
    let left_w0 = match left_res {
        Ok(Some(w)) => w,
        Ok(None) => {
            release_frontier_buf(right_nd.frac_limbs, right_nd.is_root_buf, root_borrow);
            return Ok(None);
        }
        Err(e) => {
            release_frontier_buf(right_nd.frac_limbs, right_nd.is_root_buf, root_borrow);
            return Err(e);
        }
    };

    let p_entry = &pow10_levels[lv];
    if !check_info.holds(left_w0, p_entry.mult_w0) {
        release_frontier_buf(right_nd.frac_limbs, right_nd.is_root_buf, root_borrow);
        ctx.telemetry
            .record_retry(crate::par::current_core_id(), 0, None);
        let rung = ctx.ladder.claim_rung(point_id, Rung::MakeAgain);
        return match rung {
            Rung::Correct | Rung::MakeAgain => Ok(None),
            Rung::GoBack => Err(MulError::NeedRollback(point_id)),
            Rung::Stop => Err(MulError::FatalStop(
                "Radix conversion node check failed across all ladder rungs".to_string(),
            )),
        };
    }

    let right_w0 = match eval_frontier_subtree_owned(
        ctx,
        plan,
        shares,
        cal,
        pow10_levels,
        right_nd,
        root_borrow,
        kept_full_leaves,
        rem_digits,
        total_leaves,
        sink,
        guard_tail,
        spine_completed,
        last_ckpt,
        seq_par_nodes,
    )? {
        Some(w) => w,
        None => return Ok(None),
    };
    let pow10_right_w0 = pow10_mod_w0(d_right);
    Ok(Some(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0)))
}

#[allow(clippy::too_many_arguments)]
fn convert_node(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    pow10_levels: &[Pow10Entry],
    start_leaf: u64,
    leaf_count: u64,
    frac_limbs: &[u64],
    k_limbs: usize,
    part_w0: Option<PartWords>,
    kept_full_leaves: u64,
    rem_digits: usize,
    total_leaves: u64,
    sink: &dyn DigitSink,
    guard_tail: &Mutex<Vec<u8>>,
    leaf_fault_hook: Option<&dyn LeafFaultHook>,
    resume_completed: &HashMap<(u64, u64), u64>,
    spine_completed: &Mutex<Vec<(u64, u64, u64)>>,
    resume_fracs: &HashMap<(u64, u64), PendingFraction>,
    ckpt_sink: Option<&dyn RadixCheckpointSink>,
    last_ckpt: &Mutex<Instant>,
    attempt: u32,
    active_par_nodes: u64,
) -> Result<u64, MulError> {
    if !resume_completed.is_empty() {
        if let Some(&done_w0) = resume_completed.get(&(start_leaf, leaf_count)) {
            return Ok(done_w0);
        }
    }

    // A saved fraction comes without the check words of the one it stands in for.
    let (use_limbs, use_k, part_w0) = if !resume_fracs.is_empty() {
        if let Some(saved) = resume_fracs.get(&(start_leaf, leaf_count)) {
            (saved.frac_limbs.as_slice(), saved.frac_k_limbs, None)
        } else {
            (frac_limbs, k_limbs, part_w0)
        }
    } else {
        (frac_limbs, k_limbs, part_w0)
    };

    if leaf_count == 1 {
        return eval_leaf(
            ctx,
            plan,
            start_leaf,
            use_limbs,
            use_k,
            kept_full_leaves,
            rem_digits,
            sink,
            guard_tail,
            leaf_fault_hook,
            attempt,
        );
    }

    // Fast allocation-free batched subtree conversion + SIMD `to_digits` + single `write_at`
    // for subtrees up to `BATCH_MAX_LEAVES` (`<= 4,864` digits) when not checkpointing/faulting!
    if leaf_count <= BATCH_MAX_LEAVES
        && attempt == 0
        && (ckpt_sink.is_none() || plan.checkpoint_mult > 0.0)
        && leaf_fault_hook.is_none()
        && ctx.fault_hook.is_none()
        && resume_completed.is_empty()
        && resume_fracs.is_empty()
    {
        let t_b0 = Instant::now();
        let b_res = convert_batch_subtree(
            ctx,
            plan,
            cal,
            pow10_levels,
            start_leaf,
            leaf_count,
            use_limbs,
            use_k,
            kept_full_leaves,
            rem_digits,
            total_leaves,
            sink,
            guard_tail,
        )?;
        RADIX_BATCH_NS.fetch_add(t_b0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(batch_w0) = b_res {
            return Ok(batch_w0);
        }
    }

    let lv = (leaf_count - 1).ilog2() as usize;
    let left_leaves = 1u64 << lv;
    let right_leaves = leaf_count - left_leaves;
    let d_left = left_leaves * (LEAF_DIGITS as u64);
    let d_right = right_leaves * (LEAF_DIGITS as u64);

    // If this spine node's left subtree was already converted, verified, and checkpointed in a
    // prior run, skip straight to the remaining right subtree so resumed runs write only the
    // digits their checkpoint still covers (Section 11)!
    if !resume_completed.is_empty() && !resume_fracs.contains_key(&(start_leaf, leaf_count)) {
        if let Some(&left_w0) = resume_completed.get(&(start_leaf, left_leaves)) {
            let right_w0 = convert_node(
                ctx,
                plan,
                shares,
                cal,
                pow10_levels,
                start_leaf + left_leaves,
                right_leaves,
                &[],
                0,
                None,
                kept_full_leaves,
                rem_digits,
                total_leaves,
                sink,
                guard_tail,
                leaf_fault_hook,
                resume_completed,
                spine_completed,
                resume_fracs,
                ckpt_sink,
                last_ckpt,
                attempt,
                active_par_nodes,
            )?;
            let pow10_right_w0 = pow10_mod_w0(d_right);
            return Ok(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0));
        }
    }

    let k_left = frac_limbs_for_digits(d_left);
    let k_right = frac_limbs_for_digits(d_right);
    let b_left = bit_ceil_pow10(d_left);
    let p_entry = &pow10_levels[lv];
    let point_id = 0x5000_0000u64 | ((lv as u64) << 44) | (start_leaf << 4);

    let mut cur_attempt = attempt;
    let mut force_whole = attempt > 0;

    loop {
        let est_nodes = ((total_leaves + (1u64 << lv) - 1) >> (lv + 1)).max(1);
        let eff_nodes = est_nodes.min(active_par_nodes.max(1));
        let node_par_w = if !plan.out_of_core && eff_nodes == 1 {
            shares.active_workers.max(1)
        } else if !plan.out_of_core && eff_nodes < (shares.active_workers as u64) {
            (shares.active_workers / (eff_nodes as usize)).max(1)
        } else if plan.out_of_core && eff_nodes == 1 {
            shares.active_workers.max(1)
        } else {
            1
        };

        // The right half's fraction, and what ties the left half's digits to this node's
        // product. After a failed check the product is made again whole.
        let max_cached_k =
            frac_limbs_for_digits((2 * left_leaves).min(total_leaves) * (LEAF_DIGITS as u64));
        let (x_right, right_under_half, check_info, right_parts) = node_right_fraction(
            ctx,
            plan,
            shares,
            cal,
            p_entry,
            point_id,
            use_limbs,
            use_k,
            k_right,
            max_cached_k,
            node_par_w,
            force_whole,
            part_w0,
            &chain_cuts(pow10_levels, right_leaves, k_right, 0, 0),
        )?;
        let right_parts = right_parts.and_then(|sw| {
            sw.parts(0, node_cut(pow10_levels, right_leaves, k_right), k_right)
        });

        // Cut `use_limbs` (`use_k` limbs) to `k_left` limbs and adjust by half a guard `+-2^{-(b_left + 2)}`
        // (`+` when `right_under_half`, `-` otherwise, Section 3)!
        let x_left = make_guard_adjusted_left_fraction(
            ctx.kernels,
            use_limbs,
            use_k,
            k_left,
            b_left,
            right_under_half,
            ctx.sched,
            node_par_w,
        );

        // On the checkpointed right-spine (`ckpt_sink.is_some()`), serialize `left_leaves` before
        // `right_leaves` only when a checkpoint is actually forced (`checkpoint_mult <= 0`) or
        // due within this subtree; otherwise fall through to full parallel fork-join!
        let use_spine_ckpt = if ckpt_sink.is_some() {
            if plan.checkpoint_mult <= 0.0 || !resume_completed.is_empty() || !resume_fracs.is_empty() {
                true
            } else {
                let elapsed = last_ckpt.lock().unwrap().elapsed().as_secs_f64();
                let est_sub_sec = (leaf_count as f64) * 3.0e-8;
                elapsed + est_sub_sec >= plan.checkpoint_mult * plan.est_checkpoint_write_sec
            }
        } else {
            false
        };
        if let Some(sink_ckpt) = ckpt_sink.filter(|_| use_spine_ckpt) {
            // Section 6 & 10: park `x_right` in `Store` on `supi-loader` only when `numbers` RAM
            // headroom requires it!
            let need_park_xr = crate::arena::active_numbers_live_bytes()
                .saturating_add((x_right.len() as u64).saturating_mul(8))
                > plan.ram_numbers_budget / 2;
            let (mut xr_ram_opt, mut xr_parked_opt, xr_write_h) = if plan.out_of_core
                && need_park_xr
                && let Some(st) = ctx.store
                && x_right.len() >= 64
            {
                match crate::mul::PreparedParkWrite::prepare_raw_limbs(
                    ctx,
                    point_id ^ 0x5555,
                    x_right,
                    node_par_w,
                )? {
                    Ok(prep) => {
                        let (parked, wjob) = prep.split_write_job();
                        let h = st.submit_loader_scoped(move || wjob.execute(st));
                        (None, Some(parked), Some(h))
                    }
                    Err(orig) => (Some(orig), None, None),
                }
            } else {
                (Some(x_right), None, None)
            };

            let left_res = convert_node(
                ctx,
                plan,
                shares,
                cal,
                pow10_levels,
                start_leaf,
                left_leaves,
                &x_left,
                k_left,
                None,
                kept_full_leaves,
                rem_digits,
                total_leaves,
                sink,
                guard_tail,
                leaf_fault_hook,
                resume_completed,
                spine_completed,
                resume_fracs,
                None,
                last_ckpt,
                cur_attempt,
                active_par_nodes,
            );
            crate::limb::put_limb_buf(x_left);
            if let Some(wh) = xr_write_h {
                if wh.wait().is_err() {
                    if let Some(xr) = xr_ram_opt.take() {
                        crate::limb::put_limb_buf(xr);
                    }
                    return Err(MulError::NeedRollback(point_id));
                }
            }
            let left_w0 = match left_res {
                Ok(v) => v,
                Err(e) => {
                    if let Some(xr) = xr_ram_opt.take() {
                        crate::limb::put_limb_buf(xr);
                    }
                    return Err(e);
                }
            };

            let check_ok = check_info.holds(left_w0, p_entry.mult_w0);

            if !check_ok {
                if let Some(xr) = xr_ram_opt.take() {
                    crate::limb::put_limb_buf(xr);
                }
                drop(xr_parked_opt);
                ctx.telemetry
                    .record_retry(crate::par::current_core_id(), 0, None);
                let rung = ctx.ladder.claim_rung(point_id, Rung::MakeAgain);
                match rung {
                    Rung::Correct | Rung::MakeAgain => {
                        cur_attempt += 1;
                        force_whole = true;
                        continue;
                    }
                    Rung::GoBack => return Err(MulError::NeedRollback(point_id)),
                    Rung::Stop => {
                        return Err(MulError::FatalStop(
                            "Radix conversion node check failed across all ladder rungs"
                                .to_string(),
                        ));
                    }
                }
            }

            let x_right = if let Some(parked) = xr_parked_opt.take() {
                let st = ctx.store.unwrap();
                let rstage = parked.prepare_read(point_id)?;
                let rstage = st
                    .submit_loader_scoped(move || rstage.execute_read_owned(st, ctx.telemetry))
                    .wait()
                    .map_err(|_| MulError::NeedRollback(point_id))?;
                parked.finish_unpark_raw_limbs(ctx, rstage, point_id, node_par_w)?
            } else {
                xr_ram_opt.take().unwrap()
            };

            let done_vec: Vec<(u64, u64, u64)> = {
                let mut g = spine_completed.lock().unwrap();
                if let Some(existing) = g.iter_mut().find(|(s, c, _)| *s == start_leaf && *c == left_leaves) {
                    existing.2 = left_w0;
                } else {
                    g.push((start_leaf, left_leaves, left_w0));
                }
                g.clone()
            };

            let elapsed = last_ckpt.lock().unwrap().elapsed().as_secs_f64();
            if elapsed >= plan.checkpoint_mult * plan.est_checkpoint_write_sec {
                let pending = [PendingFraction {
                    start_leaf: start_leaf + left_leaves,
                    leaf_count: right_leaves,
                    frac_limbs: x_right.to_vec(),
                    frac_k_limbs: k_right,
                    required_w0: None,
                }];
                if let Err(e) = sink_ckpt
                    .maybe_checkpoint_radix(&pending, &done_vec, elapsed)
                    .map(|wrote| {
                        if wrote {
                            *last_ckpt.lock().unwrap() = Instant::now();
                        }
                    })
                {
                    crate::limb::put_limb_buf(x_right);
                    return Err(e);
                }
            }

            let right_res = convert_node(
                ctx,
                plan,
                shares,
                cal,
                pow10_levels,
                start_leaf + left_leaves,
                right_leaves,
                &x_right,
                k_right,
                right_parts,
                kept_full_leaves,
                rem_digits,
                total_leaves,
                sink,
                guard_tail,
                leaf_fault_hook,
                resume_completed,
                spine_completed,
                resume_fracs,
                ckpt_sink,
                last_ckpt,
                cur_attempt,
                active_par_nodes,
            );
            crate::limb::put_limb_buf(x_right);
            let right_w0 = right_res?;
            let pow10_right_w0 = pow10_mod_w0(d_right);
            return Ok(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0));
        }

        // Convert left and right halves: keep 2D NTT levels (`rows > 1` or out-of-core) serialized
        // across nodes so all workers share one transform and `x_right` can be parked in `Store`
        // beside `left_leaves`, then fork-join parallel subtrees once children enter the 1D/L2 regime!
        let steal_cap = ctx.sched.subtree_steal_max_nodes() as u64;
        let child_is_2d = {
            let l_lv = if left_leaves >= 2 {
                Some((left_leaves - 1).ilog2() as usize)
            } else {
                None
            };
            l_lv.is_some_and(|clv| {
                let ce = &pow10_levels[clv];
                ce.wrapped_spec
                    .is_some_and(|s| s.par_workers > 1 || s.out_of_core)
                    || matches!(
                        ce.whole_method,
                        MulMethod::Ntt(s) if s.par_workers > 1 || s.out_of_core
                    )
            })
        };
        let ram_allows_par = !plan.out_of_core
            && (k_left as u64).saturating_mul(64 * 7) <= plan.ram_transform_budget;
        let par_split =
            leaf_count >= 4 && !child_is_2d && active_par_nodes < steal_cap && ram_allows_par;
        let child_par_nodes = if par_split {
            active_par_nodes.saturating_mul(2)
        } else {
            active_par_nodes
        };
        let run_child = |c_start: u64,
                         c_leaves: u64,
                         c_frac: &[u64],
                         c_k: usize,
                         c_parts: Option<PartWords>|
         -> Result<u64, MulError> {
            let child_is_seq_root = par_split
                && child_par_nodes >= steal_cap
                && c_leaves > BATCH_MAX_LEAVES
                && c_leaves <= 262_144
                && cur_attempt == 0
                && leaf_fault_hook.is_none()
                && resume_completed.is_empty()
                && resume_fracs.is_empty();
            if child_is_seq_root {
                let _subtree_lifo = SubtreeLifoScope::enter();
                let (base_off, max_bytes) = match plan.format {
                    OutputFormat::Text => (
                        2 + c_start * (LEAF_DIGITS as u64),
                        (c_leaves as usize) * LEAF_DIGITS,
                    ),
                    OutputFormat::Packed19 => (16 + c_start * 8, (c_leaves as usize) * 8),
                };
                let coal = SubtreeCoalesceSink::new(base_off, max_bytes);
                let w0 = convert_node(
                    ctx,
                    plan,
                    shares,
                    cal,
                    pow10_levels,
                    c_start,
                    c_leaves,
                    c_frac,
                    c_k,
                    c_parts,
                    kept_full_leaves,
                    rem_digits,
                    total_leaves,
                    &coal,
                    guard_tail,
                    leaf_fault_hook,
                    resume_completed,
                    spine_completed,
                    resume_fracs,
                    None,
                    last_ckpt,
                    cur_attempt,
                    child_par_nodes,
                )?;
                coal.flush_to(sink).map_err(MulError::FatalStop)?;
                Ok(w0)
            } else {
                convert_node(
                    ctx,
                    plan,
                    shares,
                    cal,
                    pow10_levels,
                    c_start,
                    c_leaves,
                    c_frac,
                    c_k,
                    c_parts,
                    kept_full_leaves,
                    rem_digits,
                    total_leaves,
                    sink,
                    guard_tail,
                    leaf_fault_hook,
                    resume_completed,
                    spine_completed,
                    resume_fracs,
                    None,
                    last_ckpt,
                    cur_attempt,
                    child_par_nodes,
                )
            }
        };
        if par_split {
            let (res_l, res_r) = ctx.sched.join(
                || run_child(start_leaf, left_leaves, &x_left, k_left, None),
                || run_child(start_leaf + left_leaves, right_leaves, &x_right, k_right, right_parts),
            );
            crate::limb::put_limb_buf(x_left);
            crate::limb::put_limb_buf(x_right);
            let left_w0 = res_l?;
            let right_w0 = res_r?;
            if check_info.holds(left_w0, p_entry.mult_w0) {
                let pow10_right_w0 = pow10_mod_w0(d_right);
                return Ok(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0));
            }
        } else if plan.out_of_core
            && crate::arena::active_numbers_live_bytes()
                .saturating_add((x_right.len() as u64).saturating_mul(8))
                > plan.ram_numbers_budget / 2
            && let Some(st) = ctx.store
            && x_right.len() >= 64
        {
            let (mut xr_ram_opt, mut xr_parked_opt, xr_write_h) =
                match crate::mul::PreparedParkWrite::prepare_raw_limbs(
                    ctx,
                    point_id ^ 0x5555,
                    x_right,
                    node_par_w,
                )? {
                    Ok(prep) => {
                        let (parked, wjob) = prep.split_write_job();
                        let h = st.submit_loader_scoped(move || wjob.execute(st));
                        (None, Some(parked), Some(h))
                    }
                    Err(orig) => (Some(orig), None, None),
                };
            let res_l = run_child(start_leaf, left_leaves, &x_left, k_left, None);
            crate::limb::put_limb_buf(x_left);
            if let Some(wh) = xr_write_h {
                if wh.wait().is_err() {
                    if let Some(xr) = xr_ram_opt.take() {
                        crate::limb::put_limb_buf(xr);
                    }
                    return Err(MulError::NeedRollback(point_id));
                }
            }
            let left_w0 = match res_l {
                Ok(v) => v,
                Err(e) => {
                    if let Some(xr) = xr_ram_opt.take() {
                        crate::limb::put_limb_buf(xr);
                    }
                    return Err(e);
                }
            };
            if check_info.holds(left_w0, p_entry.mult_w0) {
                let xr_restored = if let Some(parked) = xr_parked_opt.take() {
                    let rstage = parked.prepare_read(point_id)?;
                    let rstage = st
                        .submit_loader_scoped(move || rstage.execute_read_owned(st, ctx.telemetry))
                        .wait()
                        .map_err(|_| MulError::NeedRollback(point_id))?;
                    parked.finish_unpark_raw_limbs(ctx, rstage, point_id, node_par_w)?
                } else {
                    xr_ram_opt.take().unwrap()
                };
                let res_r = run_child(
                    start_leaf + left_leaves,
                    right_leaves,
                    &xr_restored,
                    k_right,
                    right_parts,
                );
                crate::limb::put_limb_buf(xr_restored);
                let right_w0 = res_r?;
                let pow10_right_w0 = pow10_mod_w0(d_right);
                return Ok(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0));
            }
            if let Some(xr) = xr_ram_opt.take() {
                crate::limb::put_limb_buf(xr);
            }
            drop(xr_parked_opt);
        } else {
            let res_l = run_child(start_leaf, left_leaves, &x_left, k_left, None);
            let res_r = run_child(start_leaf + left_leaves, right_leaves, &x_right, k_right, right_parts);
            crate::limb::put_limb_buf(x_left);
            crate::limb::put_limb_buf(x_right);
            let left_w0 = res_l?;
            let right_w0 = res_r?;
            if check_info.holds(left_w0, p_entry.mult_w0) {
                let pow10_right_w0 = pow10_mod_w0(d_right);
                return Ok(mod61_add(mod61_mul(left_w0, pow10_right_w0), right_w0));
            }
        }

        // Check failed! Climb Rung 2 (`MakeAgain` from kept fraction `use_limbs`), then Rung 3/4!
        ctx.telemetry
            .record_retry(crate::par::current_core_id(), 0, None);
        let rung = ctx.ladder.claim_rung(point_id, Rung::MakeAgain);
        match rung {
            Rung::Correct | Rung::MakeAgain => {
                cur_attempt += 1;
                force_whole = true;
                continue;
            }
            Rung::GoBack => return Err(MulError::NeedRollback(point_id)),
            Rung::Stop => {
                return Err(MulError::FatalStop(
                    "Radix conversion node check failed across all ladder rungs".to_string(),
                ));
            }
        }
    }
}

thread_local! {
    static RADIX_TLS_COALESCE_BUF: RefCell<Vec<u8, ScratchAlloc>> =
        RefCell::new(Vec::new_in(ScratchAlloc));
}

pub fn clear_tls_scratch() {
    let _ = RADIX_TLS_COALESCE_BUF.try_with(|cell| {
        let old = std::mem::replace(&mut *cell.borrow_mut(), Vec::new_in(ScratchAlloc));
        drop(old);
    });
    let _ = RADIX_BATCH_TLS.try_with(|cell| {
        let old = cell.borrow_mut().take();
        drop(old);
    });
}

/// Single-worker sequential subtree coalescing buffer (`par_split == false` throughout).
struct SubtreeCoalesceSink {
    base_off: u64,
    busy: std::sync::atomic::AtomicBool,
    state: UnsafeCell<Vec<u8, ScratchAlloc>>,
}

// SAFETY: `SubtreeCoalesceSink` is only instantiated for single-worker sequential subtrees
// (`active_par_nodes >= steal_cap`, where `par_split == false`), and `busy` enforces exclusive
// access at runtime across both `write_at` and `flush_to`.
unsafe impl Sync for SubtreeCoalesceSink {}

impl SubtreeCoalesceSink {
    fn new(base_off: u64, cap_bytes: usize) -> Self {
        let mut buf = RADIX_TLS_COALESCE_BUF
            .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), Vec::new_in(ScratchAlloc)));
        buf.clear();
        if buf.capacity() < cap_bytes {
            buf.reserve(cap_bytes - buf.capacity());
        }
        Self {
            base_off,
            busy: std::sync::atomic::AtomicBool::new(false),
            state: UnsafeCell::new(buf),
        }
    }

    fn flush_to(&self, parent: &dyn DigitSink) -> Result<(), String> {
        assert!(
            !self.busy.swap(true, Ordering::Acquire),
            "concurrent access to SubtreeCoalesceSink::flush_to"
        );
        let res = {
            let g = unsafe { &*self.state.get() };
            if !g.is_empty() {
                parent.write_at(self.base_off, g)
            } else {
                Ok(())
            }
        };
        self.busy.store(false, Ordering::Release);
        res
    }
}

impl Drop for SubtreeCoalesceSink {
    fn drop(&mut self) {
        let mut buf = std::mem::replace(self.state.get_mut(), Vec::new_in(ScratchAlloc));
        buf.clear();
        if buf.capacity() <= 8 * 1024 * 1024 {
            RADIX_TLS_COALESCE_BUF.with(|cell| {
                let mut g = cell.borrow_mut();
                if buf.capacity() > g.capacity() {
                    *g = buf;
                }
            });
        }
    }
}

impl DigitSink for SubtreeCoalesceSink {
    #[inline]
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let rel = file_offset
            .checked_sub(self.base_off)
            .ok_or_else(|| "Subtree write below base offset".to_string())? as usize;
        let end = rel + bytes.len();
        assert!(
            !self.busy.swap(true, Ordering::Acquire),
            "concurrent access to SubtreeCoalesceSink::write_at"
        );
        let g = unsafe { &mut *self.state.get() };
        if rel == g.len() {
            g.extend_from_slice(bytes);
        } else {
            if end > g.len() {
                g.resize(end, 0);
            }
            g[rel..end].copy_from_slice(bytes);
        }
        self.busy.store(false, Ordering::Release);
        Ok(())
    }

    fn flush(&self) -> Result<(), String> {
        Ok(())
    }
}

fn has_nonzero_in_bit_range(limbs: &[u64], start_bit: u64, end_bit: u64) -> bool {
    if start_bit >= end_bit {
        return false;
    }
    let total_bits = (limbs.len() as u64) * 64;
    if start_bit >= total_bits {
        return false;
    }
    let end = end_bit.min(total_bits);
    let first_w = (start_bit / 64) as usize;
    let last_w = ((end - 1) / 64) as usize;
    if first_w == last_w {
        let lo = start_bit & 63;
        let width = end - start_bit;
        let mask = if width >= 64 {
            u64::MAX
        } else {
            ((1u64 << width) - 1) << lo
        };
        return (limbs[first_w] & mask) != 0;
    }
    let end_rem = end & 63;
    let last_mask = if end_rem == 0 {
        u64::MAX
    } else {
        (1u64 << end_rem) - 1
    };
    if (limbs[last_w] & last_mask) != 0 {
        return true;
    }
    for &w in limbs[(first_w + 1)..last_w].iter().rev() {
        if w != 0 {
            return true;
        }
    }
    let first_mask = u64::MAX << (start_bit & 63);
    (limbs[first_w] & first_mask) != 0
}

/// Cut `frac_limbs` (`in_k` limbs, representing `F = X * 2^{-64 * in_k}`) to `k_left` limbs after
/// shifting `F` by `+-2^{-(b_left + 2)}` (Section 3: half a guard toward the middle of its digits).
#[allow(clippy::too_many_arguments)]
fn make_guard_adjusted_left_fraction(
    _kernels: &crate::refk::Kernels,
    frac_limbs: &[u64],
    in_k: usize,
    k_left: usize,
    b_left: u64,
    move_up: bool,
    sched: &crate::par::Scheduler,
    par_workers: usize,
) -> LimbVec {
    let total_left_bits = (k_left as u64) * 64;
    debug_assert!(total_left_bits > b_left + 4);

    // Slice or pad only the top `k_left` limbs once (`eta_bit` always lies inside the top `k_left`
    // limbs because `64 * k_left > b_left + 4`, so low dropped limbs never carry or borrow!).
    let mut out = crate::limb::take_limb_buf_committed(k_left, sched, par_workers);
    let copy_in = frac_limbs.len().min(in_k);
    if k_left >= 65_536 && par_workers > 1 {
        let out_ptr = out.as_mut_ptr() as usize;
        let src_ptr = frac_limbs.as_ptr() as usize;
        let nw = par_workers.min(k_left.div_ceil(4096)).max(1);
        let chunk = k_left.div_ceil(nw);
        sched.par_for_leaf_max(nw, 1, par_workers, |wi| {
            let d0 = wi * chunk;
            let d1 = (d0 + chunk).min(k_left);
            if d0 >= d1 {
                return;
            }
            unsafe {
                let dst = (out_ptr as *mut u64).add(d0);
                if in_k >= k_left {
                    let drop_low = in_k - k_left;
                    let copied = copy_in.saturating_sub(drop_low);
                    let c0 = d0.min(copied);
                    let c1 = d1.min(copied);
                    if c1 > c0 {
                        std::ptr::copy_nonoverlapping(
                            (src_ptr as *const u64).add(drop_low + c0),
                            (out_ptr as *mut u64).add(c0),
                            c1 - c0,
                        );
                    }
                    if d1 > c1 {
                        std::ptr::write_bytes((out_ptr as *mut u64).add(c1), 0, d1 - c1);
                    }
                } else {
                    let pad_low = k_left - in_k;
                    let mid_end = pad_low + copy_in;
                    for pos in d0..d1 {
                        let v = if pos >= pad_low && pos < mid_end {
                            *(src_ptr as *const u64).add(pos - pad_low)
                        } else {
                            0
                        };
                        *dst.add(pos - d0) = v;
                    }
                }
            }
        });
        unsafe {
            out.set_len(k_left);
        }
    } else {
        unsafe {
            if in_k >= k_left {
                let drop_low = in_k - k_left;
                let copied = copy_in.saturating_sub(drop_low);
                if copied > 0 {
                    std::ptr::copy_nonoverlapping(
                        frac_limbs.as_ptr().add(drop_low),
                        out.as_mut_ptr(),
                        copied,
                    );
                }
                if copied < k_left {
                    std::ptr::write_bytes(out.as_mut_ptr().add(copied), 0, k_left - copied);
                }
            } else {
                let pad_low = k_left - in_k;
                std::ptr::write_bytes(out.as_mut_ptr(), 0, pad_low);
                if copy_in > 0 {
                    std::ptr::copy_nonoverlapping(
                        frac_limbs.as_ptr(),
                        out.as_mut_ptr().add(pad_low),
                        copy_in,
                    );
                }
                let rem = k_left - (pad_low + copy_in);
                if rem > 0 {
                    std::ptr::write_bytes(out.as_mut_ptr().add(pad_low + copy_in), 0, rem);
                }
            }
            out.set_len(k_left);
        }
    }

    apply_guard_eta(&mut out, k_left, b_left, move_up);
    let _ = (
        resize_fraction_limbs,
        trim_limbs_keep_len::<crate::arena::NumbersAlloc>,
        add_limbs,
        sub_limbs,
        cmp_limbs,
    );
    out
}

fn shift_limbs_down_in_place_par<A: Allocator>(
    mut buf: Vec<u64, A>,
    drop_low: usize,
    copied: usize,
    out_len: usize,
    sched: &crate::par::Scheduler,
    par_workers: usize,
) -> Vec<u64, A> {
    if copied > 0 && drop_low > 0 {
        if par_workers >= 2 && copied >= 16_384 {
            let ptr = buf.as_mut_ptr() as usize;
            let mut pos = 0usize;
            while pos < copied {
                let band = (copied - pos).min(drop_low);
                if band >= 16_384 {
                    let nw = par_workers.min(band.div_ceil(4096)).max(1);
                    let chunk = band.div_ceil(nw);
                    sched.par_for_leaf_max(nw, 1, par_workers, |wi| {
                        let d0 = wi * chunk;
                        let d1 = (d0 + chunk).min(band);
                        if d1 > d0 {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    (ptr as *const u64).add(pos + drop_low + d0),
                                    (ptr as *mut u64).add(pos + d0),
                                    d1 - d0,
                                );
                            }
                        }
                    });
                } else {
                    buf.copy_within((pos + drop_low)..(pos + drop_low + band), pos);
                }
                pos += band;
            }
        } else {
            buf.copy_within(drop_low..(drop_low + copied), 0);
        }
    }
    if buf.len() >= out_len {
        if copied < out_len {
            buf[copied..out_len].fill(0);
        }
        buf.truncate(out_len);
    } else {
        buf.truncate(copied);
        buf.resize(out_len, 0);
    }
    if out_len * 8 >= crate::arena::REGION_MIN_BLOCK
        && buf.capacity().saturating_sub(out_len) * 8 >= crate::arena::REGION_MIN_BLOCK
    {
        buf.shrink_to_fit();
    }
    buf
}

#[allow(clippy::too_many_arguments)]
fn make_guard_adjusted_left_fraction_in_place(
    kernels: &crate::refk::Kernels,
    frac_limbs: LimbVec,
    in_k: usize,
    k_left: usize,
    b_left: u64,
    move_up: bool,
    sched: &crate::par::Scheduler,
    par_workers: usize,
) -> LimbVec {
    if in_k < k_left || frac_limbs.capacity() < k_left {
        let out = make_guard_adjusted_left_fraction(
            kernels,
            &frac_limbs,
            in_k,
            k_left,
            b_left,
            move_up,
            sched,
            par_workers,
        );
        crate::limb::put_limb_buf(frac_limbs);
        return out;
    }
    let copy_in = frac_limbs.len().min(in_k);
    let drop_low = in_k - k_left;
    let copied = copy_in.saturating_sub(drop_low);
    let mut frac_limbs = shift_limbs_down_in_place_par(
        frac_limbs,
        drop_low,
        copied,
        k_left,
        sched,
        par_workers,
    );
    apply_guard_eta(&mut frac_limbs, k_left, b_left, move_up);
    frac_limbs
}

fn apply_guard_eta(out: &mut [u64], k_left: usize, b_left: u64, move_up: bool) {
    let total_left_bits = (k_left as u64) * 64;
    let local_eta_bit = total_left_bits - (b_left + 2);
    let eta_word = (local_eta_bit / 64) as usize;
    let eta_mask = 1u64 << (local_eta_bit & 63);

    if move_up {
        let (s, mut carry) = out[eta_word].overflowing_add(eta_mask);
        out[eta_word] = s;
        let mut idx = eta_word + 1;
        while carry && idx < k_left {
            let (ns, nc) = out[idx].overflowing_add(1);
            out[idx] = ns;
            carry = nc;
            idx += 1;
        }
    } else {
        let mut ge_eta = out[eta_word] >= eta_mask;
        if !ge_eta {
            for &w in &out[(eta_word + 1)..k_left] {
                if w != 0 {
                    ge_eta = true;
                    break;
                }
            }
        }
        if ge_eta {
            let (d, mut brw) = out[eta_word].overflowing_sub(eta_mask);
            out[eta_word] = d;
            let mut idx = eta_word + 1;
            while brw && idx < k_left {
                let (nd, nb) = out[idx].overflowing_sub(1);
                out[idx] = nd;
                brw = nb;
                idx += 1;
            }
        } else {
            out.fill(0);
        }
    }
}

/// `apply_guard_eta`, telling what it changed, for a fraction whose check words are kept.
fn apply_guard_eta_tracked(out: &mut [u64], k_left: usize, b_left: u64, move_up: bool) -> EtaChange {
    let total_left_bits = (k_left as u64) * 64;
    let local_eta_bit = total_left_bits - (b_left + 2);
    let eta_word = (local_eta_bit / 64) as usize;
    let eta_mask = 1u64 << (local_eta_bit & 63);
    let mut old = [0u64; 4];
    let mut n = 0usize;
    let mut note = |v: u64| {
        if n < old.len() {
            old[n] = v;
        }
        n += 1;
    };

    if move_up {
        note(out[eta_word]);
        let (s, mut carry) = out[eta_word].overflowing_add(eta_mask);
        out[eta_word] = s;
        let mut idx = eta_word + 1;
        while carry && idx < k_left {
            note(out[idx]);
            let (ns, nc) = out[idx].overflowing_add(1);
            out[idx] = ns;
            carry = nc;
            idx += 1;
        }
    } else {
        let mut ge_eta = out[eta_word] >= eta_mask;
        if !ge_eta {
            for &w in &out[(eta_word + 1)..k_left] {
                if w != 0 {
                    ge_eta = true;
                    break;
                }
            }
        }
        if !ge_eta {
            out.fill(0);
            return EtaChange::Zeroed;
        }
        note(out[eta_word]);
        let (d, mut brw) = out[eta_word].overflowing_sub(eta_mask);
        out[eta_word] = d;
        let mut idx = eta_word + 1;
        while brw && idx < k_left {
            note(out[idx]);
            let (nd, nb) = out[idx].overflowing_sub(1);
            out[idx] = nd;
            brw = nb;
            idx += 1;
        }
    }
    if n > old.len() {
        EtaChange::Many
    } else {
        EtaChange::Limbs { first: eta_word, old, n }
    }
}

#[inline]
fn trim_limbs_keep_len<A: Allocator>(v: &mut Vec<u64, A>, len: usize) {
    v.resize(len, 0);
}

#[allow(clippy::too_many_arguments)]
fn eval_leaf(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    leaf_idx: u64,
    frac_limbs: &[u64],
    k_limbs: usize,
    kept_full_leaves: u64,
    rem_digits: usize,
    sink: &dyn DigitSink,
    guard_tail: &Mutex<Vec<u8>>,
    leaf_fault_hook: Option<&dyn LeafFaultHook>,
    attempt: u32,
) -> Result<u64, MulError> {
    let mut padded = frac_limbs.to_vec();
    padded.resize(k_limbs, 0);

    let mut prod = vec![0u64; k_limbs + 1];
    let hi = unsafe {
        (ctx.kernels.active.mul_1)(
            prod.as_mut_ptr(),
            padded.as_ptr(),
            k_limbs as u64,
            POW10_19,
        )
    };
    prod[k_limbs] = hi;
    let mut word_19 = prod[k_limbs].min(POW10_19 - 1);

    // Format via `to_digits` kernel!
    let mut ascii_buf = [0u8; LEAF_DIGITS];
    let mut packed_buf = [0u8; 8];

    if ctx.kernels.count_work {
        ctx.kernels.stats.digit_words.fetch_add(1, Ordering::Relaxed);
    }

    let mut emitted_w0 = unsafe {
        (ctx.kernels.active.to_digits)(ascii_buf.as_mut_ptr(), &word_19, 1, 1)
    };
    if plan.format == OutputFormat::Packed19 {
        let _ = unsafe {
            (ctx.kernels.active.to_digits)(packed_buf.as_mut_ptr(), &word_19, 1, 0)
        };
    }

    if let Some(hook) = leaf_fault_hook {
        let orig = word_19;
        hook.on_leaf_word(leaf_idx, attempt, &mut word_19, &mut emitted_w0);
        if word_19 != orig {
            // Re-render corrupted word so `ascii_buf` reflects the injected fault.
            emitted_w0 = unsafe {
                (ctx.kernels.active.to_digits)(ascii_buf.as_mut_ptr(), &word_19, 1, 1)
            };
        }
    }

    // Write kept portion to `DigitSink` at this leaf's exact offset (Section 11), and capture
    // dropped guard digits into `guard_tail`!
    if plan.digits > 0 {
        if leaf_idx < kept_full_leaves {
            match plan.format {
                OutputFormat::Text => {
                    let off = 2 + leaf_idx * (LEAF_DIGITS as u64);
                    sink.write_at(off, &ascii_buf)
                        .map_err(MulError::FatalStop)?;
                }
                OutputFormat::Packed19 => {
                    let off = 16 + leaf_idx * 8;
                    sink.write_at(off, &packed_buf)
                        .map_err(MulError::FatalStop)?;
                }
            }
        } else if rem_digits > 0 && leaf_idx == kept_full_leaves {
            match plan.format {
                OutputFormat::Text => {
                    let off = 2 + leaf_idx * (LEAF_DIGITS as u64);
                    sink.write_at(off, &ascii_buf[..rem_digits])
                        .map_err(MulError::FatalStop)?;
                }
                OutputFormat::Packed19 => {
                    let mut div = 1u64;
                    for _ in 0..(LEAF_DIGITS - rem_digits) {
                        div *= 10;
                    }
                    let partial_word = word_19 / div;
                    let off = 16 + leaf_idx * 8;
                    sink.write_at(off, &partial_word.to_le_bytes())
                        .map_err(MulError::FatalStop)?;
                }
            }
            let mut g = guard_tail.lock().unwrap();
            let tail_part = &ascii_buf[rem_digits..LEAF_DIGITS];
            g[..tail_part.len()].copy_from_slice(tail_part);
        } else {
            // Extra leaf (`leaf_idx == num_leaves - 1`):
            let mut g = guard_tail.lock().unwrap();
            let dst_off = if rem_digits > 0 {
                LEAF_DIGITS - rem_digits
            } else {
                0
            };
            let copy_len = g.len().saturating_sub(dst_off).min(LEAF_DIGITS);
            if copy_len > 0 {
                g[dst_off..(dst_off + copy_len)].copy_from_slice(&ascii_buf[..copy_len]);
            }
        }
    } else {
        let mut g = guard_tail.lock().unwrap();
        g[..LEAF_DIGITS].copy_from_slice(&ascii_buf);
    }

    let _ = (
        Ladder::default,
        mod61_pow2,
        trim_limbs::<crate::arena::NumbersAlloc>,
    );
    Ok(emitted_w0)
}

#[inline(always)]
fn adjust_left_in_place(out: &mut [u64], k_left: usize, b_left: u64, move_up: bool) {
    // (a batch's fraction has at most `BATCH_MAX_K_LIMBS` limbs, and its left part no more)
    debug_assert!(
        out.len() == k_left && k_left <= BATCH_MAX_K_LIMBS && (k_left as u64) * 64 > b_left + 4
    );
    let total_left_bits = (k_left as u64) * 64;
    let local_eta_bit = total_left_bits - (b_left + 2);
    let eta_word = (local_eta_bit / 64) as usize;
    let eta_mask = 1u64 << (local_eta_bit & 63);

    if move_up {
        let (s, mut carry) = out[eta_word].overflowing_add(eta_mask);
        out[eta_word] = s;
        let mut idx = eta_word + 1;
        while carry && idx < k_left {
            let (ns, nc) = out[idx].overflowing_add(1);
            out[idx] = ns;
            carry = nc;
            idx += 1;
        }
    } else {
        let mut ge_eta = out[eta_word] >= eta_mask;
        if !ge_eta {
            for &w in &out[(eta_word + 1)..k_left] {
                if w != 0 {
                    ge_eta = true;
                    break;
                }
            }
        }
        if ge_eta {
            let (d, mut brw) = out[eta_word].overflowing_sub(eta_mask);
            out[eta_word] = d;
            let mut idx = eta_word + 1;
            while brw && idx < k_left {
                let (nd, nb) = out[idx].overflowing_sub(1);
                out[idx] = nd;
                brw = nb;
                idx += 1;
            }
        } else {
            out.fill(0);
        }
    }
}

#[inline(always)]
fn make_guard_adjusted_left_stack(
    frac_limbs: &[u64],
    in_k: usize,
    k_left: usize,
    b_left: u64,
    move_up: bool,
    out_ptr: *mut u64,
) {
    debug_assert!(k_left <= BATCH_MAX_K_LIMBS && (k_left as u64) * 64 > b_left + 4);
    let copy_in = frac_limbs.len().min(in_k);
    let out: &mut [u64] = unsafe {
        if in_k >= k_left {
            let drop_low = in_k - k_left;
            let copied = copy_in.saturating_sub(drop_low);
            if copied > 0 {
                std::ptr::copy_nonoverlapping(frac_limbs.as_ptr().add(drop_low), out_ptr, copied);
            }
            if copied < k_left {
                std::ptr::write_bytes(out_ptr.add(copied), 0, k_left - copied);
            }
        } else {
            let pad_low = k_left - in_k;
            std::ptr::write_bytes(out_ptr, 0, pad_low);
            if copy_in > 0 {
                std::ptr::copy_nonoverlapping(
                    frac_limbs.as_ptr(),
                    out_ptr.add(pad_low),
                    copy_in,
                );
            }
            let rem = k_left - (pad_low + copy_in);
            if rem > 0 {
                std::ptr::write_bytes(out_ptr.add(pad_low + copy_in), 0, rem);
            }
        }
        std::slice::from_raw_parts_mut(out_ptr, k_left)
    };
    adjust_left_in_place(out, k_left, b_left, move_up);
}

struct RadixBatchScratch {
    padded_root: Vec<u64, ScratchAlloc>,
    words_out: Vec<u64, ScratchAlloc>,
    ascii_buf: Vec<u8, ScratchAlloc>,
    packed_buf: Vec<u8, ScratchAlloc>,
    work_u64: Vec<u64, ScratchAlloc>,
}

impl RadixBatchScratch {
    fn new() -> Self {
        let mut padded_root = Vec::with_capacity_in(17408, ScratchAlloc);
        padded_root.resize(17408, 0);
        let mut words_out = Vec::with_capacity_in(16384, ScratchAlloc);
        words_out.resize(16384, 0);
        let mut ascii_buf = Vec::with_capacity_in(16384 * LEAF_DIGITS, ScratchAlloc);
        ascii_buf.resize(16384 * LEAF_DIGITS, 0);
        let mut packed_buf = Vec::with_capacity_in(16384 * 8, ScratchAlloc);
        packed_buf.resize(16384 * 8, 0);
        let mut work_u64 = Vec::with_capacity_in(196608, ScratchAlloc);
        work_u64.resize(196608, 0);
        Self {
            padded_root,
            words_out,
            ascii_buf,
            packed_buf,
            work_u64,
        }
    }
}

thread_local! {
    static RADIX_BATCH_TLS: std::cell::RefCell<Option<RadixBatchScratch>> =
        const { std::cell::RefCell::new(None) };
}

struct BatchScratchLease(Option<RadixBatchScratch>);

impl BatchScratchLease {
    #[inline]
    fn acquire() -> Self {
        let sc = RADIX_BATCH_TLS
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_else(RadixBatchScratch::new);
        Self(Some(sc))
    }

    #[inline]
    fn get_mut(&mut self) -> &mut RadixBatchScratch {
        self.0.as_mut().unwrap()
    }
}

impl Drop for BatchScratchLease {
    #[inline]
    fn drop(&mut self) {
        if let Some(sc) = self.0.take() {
            RADIX_BATCH_TLS.with(|cell| {
                *cell.borrow_mut() = Some(sc);
            });
        }
    }
}

#[inline(always)]
fn red64_w0(v: u64) -> u64 {
    let r = (v & MOD_W0) + (v >> 61);
    if r >= MOD_W0 { r - MOD_W0 } else { r }
}

#[inline(always)]
fn eval_leaf2_fast(
    ctx: &MulCtx<'_>,
    rel_leaf: usize,
    frac_limbs: &[u64],
    words_out: &mut [u64],
    covered_by_ancestor: bool,
) -> Option<u64> {
    if frac_limbs.len() >= 4 {
        let f0 = frac_limbs[0];
        let f1 = frac_limbs[1];
        let f2 = frac_limbs[2];
        let f3 = frac_limbs[3];
        if ctx.kernels.count_work {
            ctx.kernels.stats.limb_ops.fetch_add(4, Ordering::Relaxed);
        }
        const P19: u128 = POW10_19 as u128;
        let m0 = (f0 as u128) * P19;
        let m1 = (f1 as u128) * P19 + (m0 >> 64);
        let m2 = (f2 as u128) * P19 + (m1 >> 64);
        let m3 = (f3 as u128) * P19 + (m2 >> 64);
        let p1 = m1 as u64;
        let p2 = m2 as u64;
        let p3 = m3 as u64;
        let p4 = (m3 >> 64) as u64;

        const ETA2: u64 = 1u64 << 62;
        let (xl0, xl1, xl2) = if (p3 >> 63) == 0 {
            let (s1, c1) = f2.overflowing_add(ETA2);
            (f1, s1, f3.wrapping_add(u64::from(c1)))
        } else if f2 >= ETA2 || f3 != 0 {
            let (d1, b1) = f2.overflowing_sub(ETA2);
            (f1, d1, f3.wrapping_sub(u64::from(b1)))
        } else {
            (0, 0, 0)
        };

        let l0 = (xl0 as u128) * P19;
        let l1 = (xl1 as u128) * P19 + (l0 >> 64);
        let l2 = (xl2 as u128) * P19 + (l1 >> 64);
        let left_word = ((l2 >> 64) as u64).min(POW10_19 - 1);
        let left_w0 = red64_w0(left_word);
        if !covered_by_ancestor && left_w0 != red64_w0(p4) {
            return None;
        }

        let r0 = (p1 as u128) * P19;
        let r1 = (p2 as u128) * P19 + (r0 >> 64);
        let r2 = (p3 as u128) * P19 + (r1 >> 64);
        let right_word = ((r2 >> 64) as u64).min(POW10_19 - 1);
        let right_w0 = red64_w0(right_word);
        words_out[rel_leaf] = left_word;
        words_out[rel_leaf + 1] = right_word;
        return Some(mod61_reduce(
            (left_w0 as u128) * (POW10_19_MOD_W0 as u128) + (right_w0 as u128),
        ));
    }

    let copy_n = frac_limbs.len().min(4);
    if ctx.kernels.count_work && copy_n > 0 {
        ctx.kernels
            .stats
            .limb_ops
            .fetch_add(copy_n as u64, Ordering::Relaxed);
    }
    let mut prod_s = [0u64; 5];
    let mut carry = 0u64;
    for (i, &w) in frac_limbs[..copy_n].iter().enumerate() {
        let p = (w as u128) * (POW10_19 as u128) + (carry as u128);
        prod_s[i] = p as u64;
        carry = (p >> 64) as u64;
    }
    prod_s[copy_n] = carry;

    let under_half = (prod_s[3] >> 63) == 0;
    let mut xl_s = [0u64; 3];
    make_guard_adjusted_left_stack(frac_limbs, 4, 3, 64, under_half, xl_s.as_mut_ptr());

    let mut hi_l = 0u64;
    for &w in &xl_s {
        let p = (w as u128) * (POW10_19 as u128) + (hi_l as u128);
        hi_l = (p >> 64) as u64;
    }
    let left_word = hi_l.min(POW10_19 - 1);
    let left_w0 = red64_w0(left_word);
    if !covered_by_ancestor {
        let exp_left_w0 = red64_w0(prod_s[4]);
        if left_w0 != exp_left_w0 {
            return None;
        }
    }

    let mut hi_r = 0u64;
    for &w in &prod_s[1..4] {
        let p = (w as u128) * (POW10_19 as u128) + (hi_r as u128);
        hi_r = (p >> 64) as u64;
    }
    let right_word = hi_r.min(POW10_19 - 1);
    let right_w0 = red64_w0(right_word);
    words_out[rel_leaf] = left_word;
    words_out[rel_leaf + 1] = right_word;
    Some(mod61_reduce(
        (left_w0 as u128) * (POW10_19_MOD_W0 as u128) + (right_w0 as u128),
    ))
}

#[inline(always)]
fn eval_leaf4_fast(
    ctx: &MulCtx<'_>,
    rel_leaf: usize,
    frac_limbs: &[u64],
    words_out: &mut [u64],
    covered_by_ancestor: bool,
) -> Option<u64> {
    const P38_FULL: u128 = (POW10_19 as u128) * (POW10_19 as u128);
    const P38_LO: u128 = (P38_FULL as u64) as u128;
    const P38_HI: u128 = P38_FULL >> 64;
    const P38_MOD_W0: u64 =
        (((POW10_19_MOD_W0 as u128) * (POW10_19_MOD_W0 as u128)) % (MOD_W0 as u128)) as u64;

    if frac_limbs.len() >= 6 {
        let f0 = frac_limbs[0];
        let f1 = frac_limbs[1];
        let f2 = frac_limbs[2];
        let f3 = frac_limbs[3];
        let f4 = frac_limbs[4];
        let f5 = frac_limbs[5];
        if ctx.kernels.count_work {
            ctx.kernels.stats.limb_ops.fetch_add(12, Ordering::Relaxed);
        }

        let a0 = (f0 as u128) * P38_LO;
        let a1 = (f1 as u128) * P38_LO + (a0 >> 64);
        let a2 = (f2 as u128) * P38_LO + (a1 >> 64);
        let a3 = (f3 as u128) * P38_LO + (a2 >> 64);
        let a4 = (f4 as u128) * P38_LO + (a3 >> 64);
        let a5 = (f5 as u128) * P38_LO + (a4 >> 64);

        let b1 = (f0 as u128) * P38_HI + ((a1 as u64) as u128);
        let b2 = (f1 as u128) * P38_HI + ((a2 as u64) as u128) + (b1 >> 64);
        let b3 = (f2 as u128) * P38_HI + ((a3 as u64) as u128) + (b2 >> 64);
        let b4 = (f3 as u128) * P38_HI + ((a4 as u64) as u128) + (b3 >> 64);
        let b5 = (f4 as u128) * P38_HI + ((a5 as u64) as u128) + (b4 >> 64);

        let p2 = b2 as u64;
        let p3 = b3 as u64;
        let p4 = b4 as u64;
        let p5 = b5 as u64;

        const ETA4: u64 = 1u64 << 63;
        let xl_s = if (p5 >> 63) == 0 {
            let (s1, c1) = f3.overflowing_add(ETA4);
            let (s2, c2) = f4.overflowing_add(u64::from(c1));
            [f2, s1, s2, f5.wrapping_add(u64::from(c2))]
        } else if f3 >= ETA4 || (f4 | f5) != 0 {
            let (d1, bw1) = f3.overflowing_sub(ETA4);
            let (d2, bw2) = f4.overflowing_sub(u64::from(bw1));
            [f2, d1, d2, f5.wrapping_sub(u64::from(bw2))]
        } else {
            [0, 0, 0, 0]
        };

        let left2_w0 = eval_leaf2_fast(ctx, rel_leaf, &xl_s, words_out, true)?;
        if !covered_by_ancestor {
            let b6 = (f5 as u128) * P38_HI + (a5 >> 64) + (b5 >> 64);
            let exp_left_w0 = mod61_add(
                red64_w0(b6 as u64),
                mod61_mul(red64_w0((b6 >> 64) as u64), 8),
            );
            if left2_w0 != exp_left_w0 {
                return None;
            }
        }
        let xr_s = [p2, p3, p4, p5];
        let right2_w0 = eval_leaf2_fast(ctx, rel_leaf + 2, &xr_s, words_out, covered_by_ancestor)?;
        return Some(mod61_reduce(
            (left2_w0 as u128) * (P38_MOD_W0 as u128) + (right2_w0 as u128),
        ));
    }

    let mut la = frac_limbs.len().min(6);
    while la > 0 && frac_limbs[la - 1] == 0 {
        la -= 1;
    }
    let mut prod_s = [0u64; 8];
    if la > 0 {
        if ctx.kernels.count_work {
            ctx.kernels
                .stats
                .limb_ops
                .fetch_add((la * 2) as u64, Ordering::Relaxed);
        }
        let mut c0 = 0u128;
        for (i, &w) in frac_limbs[..la].iter().enumerate() {
            let p = (w as u128) * P38_LO + c0;
            prod_s[i] = p as u64;
            c0 = p >> 64;
        }
        prod_s[la] = c0 as u64;
        let mut c1 = 0u128;
        for (i, &w) in frac_limbs[..la].iter().enumerate() {
            let p = (w as u128) * P38_HI + (prod_s[i + 1] as u128) + c1;
            prod_s[i + 1] = p as u64;
            c1 = p >> 64;
        }
        prod_s[la + 1] = c1 as u64;
    }
    let under_half = (prod_s[5] >> 63) == 0;
    let mut xl_s = [0u64; 4];
    make_guard_adjusted_left_stack(frac_limbs, 6, 4, 127, under_half, xl_s.as_mut_ptr());
    let left2_w0 = eval_leaf2_fast(ctx, rel_leaf, &xl_s, words_out, true)?;
    if !covered_by_ancestor {
        let exp_left_w0 = mod61_add(
            red64_w0(prod_s[6]),
            mod61_mul(red64_w0(prod_s[7]), 8),
        );
        if left2_w0 != exp_left_w0 {
            return None;
        }
    }
    let right2_w0 = eval_leaf2_fast(ctx, rel_leaf + 2, &prod_s[2..6], words_out, covered_by_ancestor)?;
    Some(mod61_reduce(
        (left2_w0 as u128) * (P38_MOD_W0 as u128) + (right2_w0 as u128),
    ))
}

const fn pow10_76_hi3() -> [u64; 3] {
    let p38 = (POW10_19 as u128) * (POW10_19 as u128);
    let a0 = (p38 as u64) as u128;
    let a1 = p38 >> 64;
    let p00 = a0 * a0;
    let p01 = a0 * a1 + (p00 >> 64);
    let p10 = a1 * a0 + ((p01 as u64) as u128);
    let p11 = a1 * a1 + (p01 >> 64) + (p10 >> 64);
    assert!((p00 as u64) == 0 && (p11 >> 64) != 0);
    [p10 as u64, p11 as u64, (p11 >> 64) as u64]
}

#[inline(always)]
fn eval_leaf8_fast(
    ctx: &MulCtx<'_>,
    rel_leaf: usize,
    frac_limbs: &[u64],
    words_out: &mut [u64],
    covered_by_ancestor: bool,
) -> Option<u64> {
    let mut f_pad = [0u64; 10];
    let f: &[u64; 10] = if frac_limbs.len() >= 10 {
        frac_limbs[..10].try_into().unwrap()
    } else {
        let copy_n = frac_limbs.len();
        f_pad[..copy_n].copy_from_slice(&frac_limbs[..copy_n]);
        &f_pad
    };

    const P76_HI3: [u64; 3] = pow10_76_hi3();
    const M1: u128 = P76_HI3[0] as u128;
    const M2: u128 = P76_HI3[1] as u128;
    const M3: u128 = P76_HI3[2] as u128;
    const P38_MOD_W0: u64 =
        (((POW10_19_MOD_W0 as u128) * (POW10_19_MOD_W0 as u128)) % (MOD_W0 as u128)) as u64;
    const P76_MOD_W0: u64 =
        (((P38_MOD_W0 as u128) * (P38_MOD_W0 as u128)) % (MOD_W0 as u128)) as u64;

    if ctx.kernels.count_work {
        let ops = if covered_by_ancestor { 24 } else { 30 };
        ctx.kernels.stats.limb_ops.fetch_add(ops, Ordering::Relaxed);
    }

    let t1 = (f[0] as u128) * M1;
    let t2 = (f[1] as u128) * M1 + (t1 >> 64);
    let t3 = (f[2] as u128) * M1 + (t2 >> 64);
    let t4 = (f[3] as u128) * M1 + (t3 >> 64);
    let t5 = (f[4] as u128) * M1 + (t4 >> 64);
    let t6 = (f[5] as u128) * M1 + (t5 >> 64);
    let t7 = (f[6] as u128) * M1 + (t6 >> 64);
    let t8 = (f[7] as u128) * M1 + (t7 >> 64);
    let t9 = (f[8] as u128) * M1 + (t8 >> 64);

    let u2 = (f[0] as u128) * M2 + ((t2 as u64) as u128);
    let u3 = (f[1] as u128) * M2 + ((t3 as u64) as u128) + (u2 >> 64);
    let u4 = (f[2] as u128) * M2 + ((t4 as u64) as u128) + (u3 >> 64);
    let u5 = (f[3] as u128) * M2 + ((t5 as u64) as u128) + (u4 >> 64);
    let u6 = (f[4] as u128) * M2 + ((t6 as u64) as u128) + (u5 >> 64);
    let u7 = (f[5] as u128) * M2 + ((t7 as u64) as u128) + (u6 >> 64);
    let u8 = (f[6] as u128) * M2 + ((t8 as u64) as u128) + (u7 >> 64);
    let u9 = (f[7] as u128) * M2 + ((t9 as u64) as u128) + (u8 >> 64);

    let v3 = (f[0] as u128) * M3 + ((u3 as u64) as u128);
    let v4 = (f[1] as u128) * M3 + ((u4 as u64) as u128) + (v3 >> 64);
    let v5 = (f[2] as u128) * M3 + ((u5 as u64) as u128) + (v4 >> 64);
    let v6 = (f[3] as u128) * M3 + ((u6 as u64) as u128) + (v5 >> 64);
    let v7 = (f[4] as u128) * M3 + ((u7 as u64) as u128) + (v6 >> 64);
    let v8 = (f[5] as u128) * M3 + ((u8 as u64) as u128) + (v7 >> 64);
    let v9 = (f[6] as u128) * M3 + ((u9 as u64) as u128) + (v8 >> 64);

    let p9 = v9 as u64;
    let xl_s = if (p9 >> 63) == 0 {
        let (s2, c2) = f[6].overflowing_add(2);
        let (s3, c3) = f[7].overflowing_add(u64::from(c2));
        let (s4, c4) = f[8].overflowing_add(u64::from(c3));
        [f[4], f[5], s2, s3, s4, f[9].wrapping_add(u64::from(c4))]
    } else if f[6] >= 2 || (f[7] | f[8] | f[9]) != 0 {
        let (d2, b2) = f[6].overflowing_sub(2);
        let (d3, b3) = f[7].overflowing_sub(u64::from(b2));
        let (d4, b4) = f[8].overflowing_sub(u64::from(b3));
        [f[4], f[5], d2, d3, d4, f[9].wrapping_sub(u64::from(b4))]
    } else {
        [0; 6]
    };

    let left4_w0 = eval_leaf4_fast(ctx, rel_leaf, &xl_s, words_out, true)?;
    if !covered_by_ancestor {
        let t10 = (f[9] as u128) * M1 + (t9 >> 64);
        let u10 = (f[8] as u128) * M2 + ((t10 as u64) as u128) + (u9 >> 64);
        let u11 = (f[9] as u128) * M2 + (t10 >> 64) + (u10 >> 64);
        let v10 = (f[7] as u128) * M3 + ((u10 as u64) as u128) + (v9 >> 64);
        let v11 = (f[8] as u128) * M3 + ((u11 as u64) as u128) + (v10 >> 64);
        let v12 = (f[9] as u128) * M3 + (u11 >> 64) + (v11 >> 64);
        let hi_limbs = [v10 as u64, v11 as u64, v12 as u64, (v12 >> 64) as u64];
        if left4_w0 != compute_limbs_w0(ctx.kernels, &hi_limbs) {
            return None;
        }
    }

    let xr_s = [
        v4 as u64,
        v5 as u64,
        v6 as u64,
        v7 as u64,
        v8 as u64,
        p9,
    ];
    let right4_w0 = eval_leaf4_fast(ctx, rel_leaf + 4, &xr_s, words_out, covered_by_ancestor)?;
    Some(mod61_reduce(
        (left4_w0 as u128) * (P76_MOD_W0 as u128) + (right4_w0 as u128),
    ))
}

#[allow(clippy::too_many_arguments)]
fn eval_batch_rec(
    ctx: &MulCtx<'_>,
    cal: &Calibration,
    pow10_levels: &[Pow10Entry],
    rel_leaf: usize,
    leaf_count: u64,
    frac_limbs: &mut [u64],
    k_limbs: usize,
    words_out: &mut [u64],
    work: &mut [u64],
    covered_by_ancestor: bool,
) -> Option<u64> {
    if leaf_count == 1 {
        let copy_n = frac_limbs.len().min(k_limbs);
        let mut hi = 0u64;
        for &w in &frac_limbs[..copy_n] {
            let p = (w as u128) * (POW10_19 as u128) + (hi as u128);
            hi = (p >> 64) as u64;
        }
        if copy_n < k_limbs {
            hi = 0;
        }
        let word_19 = hi.min(POW10_19 - 1);
        words_out[rel_leaf] = word_19;
        return Some(red64_w0(word_19));
    }

    if leaf_count == 2 && k_limbs == 4 {
        return eval_leaf2_fast(ctx, rel_leaf, frac_limbs, words_out, covered_by_ancestor);
    }

    if leaf_count == 4 && k_limbs == 6 {
        return eval_leaf4_fast(ctx, rel_leaf, frac_limbs, words_out, covered_by_ancestor);
    }

    if leaf_count == 8 && k_limbs == 10 {
        return eval_leaf8_fast(ctx, rel_leaf, frac_limbs, words_out, covered_by_ancestor);
    }

    let lv = (leaf_count - 1).ilog2() as usize;
    let left_leaves = 1u64 << lv;
    let right_leaves = leaf_count - left_leaves;
    let d_left = left_leaves * (LEAF_DIGITS as u64);
    let d_right = right_leaves * (LEAF_DIGITS as u64);
    let k_left = frac_limbs_for_digits(d_left);
    let k_right = frac_limbs_for_digits(d_right);
    let b_left = bit_ceil_pow10(d_left);
    let p_entry = &pow10_levels[lv];
    let strip_limbs = p_entry.strip_limbs;
    let f_k = k_limbs - strip_limbs;
    let have = frac_limbs.len().min(k_limbs);
    let f_limbs = &frac_limbs[..have.min(f_k)];
    let top_limbs = &frac_limbs[have.min(f_k)..have];
    let p_limbs = p_entry.mult.limbs.as_slice();

    let mut la = f_limbs.len();
    while la > 0 && f_limbs[la - 1] == 0 {
        la -= 1;
    }
    let mut lb = p_limbs.len();
    while lb > 0 && p_limbs[lb - 1] == 0 {
        lb -= 1;
    }

    let fft_direct_need = match &p_entry.whole_cached {
        Some(CachedTransform::Fft(fb)) => {
            crate::fft::fft_direct_carry_limbs(fb.n_complex, fb.coeff_bits, 0)
        }
        _ => 0,
    };
    let prod_need = (la + lb + 2).max(f_k + 1).max(fft_direct_need);
    let in_place_xl = frac_limbs.len() == k_limbs && k_limbs >= k_left;
    let step_need = prod_need + if in_place_xl { 0 } else { k_left };
    if work.len() < step_need {
        return None;
    }
    let (step_buf, sub_work) = work.split_at_mut(step_need);
    let (prod_s, xl_fallback) = step_buf.split_at_mut(prod_need);

    let p_ptr = prod_s.as_mut_ptr();
    let mut prod_len = 0usize;
    let mut used_fft = false;
    let xr_start = f_k - k_right;

    if la > 0 && lb > 0 {
        let min_l = la.min(lb);
        let max_l = la.max(lb);
        let full_base_fast = cal.basecase_max_limbs >= 48;
        let base_cap = if full_base_fast {
            cal.basecase_max_limbs.min(64)
        } else {
            36
        };
        if min_l <= base_cap && (max_l <= 72 || min_l <= 32) {
            let z_lo = p_entry.zero_lo_limbs.min(lb);
            let lb_nz = lb - z_lo;
            if covered_by_ancestor && !full_base_fast {
                // Below a transform node: compute only the middle parallelogram `guard_lo <= i + j < f_k`
                // needed for `xr_s` (`prod_s[xr_start..f_k]`), letting the transform ancestor above check
                // all emitted digits (DESIGN.md Section 1 & Section 8).
                let guard_lo = xr_start.saturating_sub(2);
                prod_s[guard_lo..=f_k].fill(0);
                let mut ops_done = 0usize;
                for (j_rel, &bj_u64) in p_limbs[z_lo..lb].iter().enumerate() {
                    if bj_u64 == 0 {
                        continue;
                    }
                    let j = z_lo + j_rel;
                    let i_lo = guard_lo.saturating_sub(j);
                    let i_hi = f_k.saturating_sub(j).min(la);
                    if i_lo >= i_hi {
                        continue;
                    }
                    let len = i_hi - i_lo;
                    let dst_off = i_lo + j;
                    ops_done += len;
                    let carry = if len <= 12 {
                        let bj = bj_u64 as u128;
                        let mut c = 0u128;
                        for t in 0..len {
                            let p = (f_limbs[i_lo + t] as u128) * bj
                                + (prod_s[dst_off + t] as u128)
                                + c;
                            prod_s[dst_off + t] = p as u64;
                            c = p >> 64;
                        }
                        c as u64
                    } else {
                        unsafe {
                            (ctx.kernels.active.addmul_1)(
                                p_ptr.add(dst_off),
                                f_limbs.as_ptr().add(i_lo),
                                len as u64,
                                bj_u64,
                            )
                        }
                    };
                    let mut c_pos = dst_off + len;
                    let mut c_val = carry;
                    while c_val != 0 && c_pos < f_k {
                        let (ns, nc) = prod_s[c_pos].overflowing_add(c_val);
                        prod_s[c_pos] = ns;
                        c_val = u64::from(nc);
                        c_pos += 1;
                    }
                }
                if ctx.kernels.count_work && ops_done > 0 {
                    ctx.kernels
                        .stats
                        .limb_ops
                        .fetch_add(ops_done as u64, Ordering::Relaxed);
                }
                // If limb `xr_start - 1` is within 256 ulps of carry-out, recompute full basecase so
                // `xr_s` is 100% bit-for-bit identical on every input.
                if guard_lo > z_lo && prod_s[xr_start - 1] >= u64::MAX - 256 {
                    let (long, short) = if la >= lb_nz {
                        (&f_limbs[..la], &p_limbs[z_lo..lb])
                    } else {
                        (&p_limbs[z_lo..lb], &f_limbs[..la])
                    };
                    let lens = ctx.kernels.mul_base_lens(long.len(), short.len());
                    unsafe {
                        if z_lo > 0 {
                            std::ptr::write_bytes(p_ptr, 0, z_lo.min(f_k));
                        }
                        (ctx.kernels.active.mul_base)(
                            p_ptr.add(z_lo),
                            long.as_ptr(),
                            short.as_ptr(),
                            &lens,
                        );
                        if z_lo + la + lb_nz < f_k {
                            std::ptr::write_bytes(
                                p_ptr.add(z_lo + la + lb_nz),
                                0,
                                f_k - (z_lo + la + lb_nz),
                            );
                        }
                    }
                }
            } else {
                let mut la_eff = if covered_by_ancestor {
                    la.min(f_k.saturating_sub(z_lo))
                } else {
                    la
                };
                while la_eff > 0 && f_limbs[la_eff - 1] == 0 {
                    la_eff -= 1;
                }
                if la_eff == 0 || lb_nz == 0 {
                    prod_s[..f_k].fill(0);
                } else {
                    let a_sub = &f_limbs[..la_eff];
                    let b_sub = &p_limbs[z_lo..lb];
                    let (long, short) = if la_eff >= lb_nz {
                        (a_sub, b_sub)
                    } else {
                        (b_sub, a_sub)
                    };
                    if ctx.kernels.count_work {
                        ctx.kernels
                            .stats
                            .limb_ops
                            .fetch_add((la_eff * lb_nz) as u64, Ordering::Relaxed);
                    }
                    if short.len() <= 4 && long.len() <= 9 {
                        let clear_len = (z_lo + la_eff + lb_nz).max(f_k);
                        prod_s[..clear_len].fill(0);
                        for (j, &bj_u64) in short.iter().enumerate() {
                            if bj_u64 == 0 {
                                continue;
                            }
                            let bj = bj_u64 as u128;
                            let mut carry = 0u128;
                            for (i, &ai_u64) in long.iter().enumerate() {
                                let prod = (ai_u64 as u128) * bj
                                    + (prod_s[z_lo + j + i] as u128)
                                    + carry;
                                prod_s[z_lo + j + i] = prod as u64;
                                carry = prod >> 64;
                            }
                            prod_s[z_lo + j + long.len()] = carry as u64;
                        }
                    } else {
                        let lens = ctx.kernels.mul_base_lens(long.len(), short.len());
                        unsafe {
                            if !covered_by_ancestor {
                                if z_lo > 0 {
                                    std::ptr::write_bytes(p_ptr, 0, z_lo);
                                }
                            } else if z_lo > xr_start {
                                std::ptr::write_bytes(
                                    p_ptr.add(xr_start),
                                    0,
                                    z_lo.min(f_k) - xr_start,
                                );
                            }
                            (ctx.kernels.active.mul_base)(
                                p_ptr.add(z_lo),
                                long.as_ptr(),
                                short.as_ptr(),
                                &lens,
                            );
                            let end_written = z_lo + la_eff + lb_nz;
                            if end_written < f_k {
                                std::ptr::write_bytes(
                                    p_ptr.add(end_written),
                                    0,
                                    f_k - end_written,
                                );
                            }
                        }
                    }
                    if !covered_by_ancestor {
                        prod_len = z_lo + la_eff + lb_nz;
                        while prod_len > 0 && prod_s[prod_len - 1] == 0 {
                            prod_len -= 1;
                        }
                    }
                }
            }
        } else if let Some(CachedTransform::Fft(fb)) = p_entry.whole_cached.as_ref().filter(|c| {
            matches!(
                c,
                CachedTransform::Fft(fb)
                    if ((la + lb + 1) as u64) * 64 + 32
                        <= (2 * fb.n_complex as u64) * (fb.coeff_bits as u64)
            )
        }) {
            let out_l = (la + lb + 1).max(f_k);
            let Some((trimmed_l, prod_w0, max_err)) = crate::fft::fft_mul_slice_with_cached_into(
                ctx.kernels,
                ctx.fft_cache,
                prod_s,
                &f_limbs[..la],
                fb,
                out_l,
            ) else {
                return None;
            };
            if !(max_err <= ctx.fft_err_bound) {
                return None;
            }
            let a_w0 = compute_limbs_w0(ctx.kernels, &f_limbs[..la]);
            if prod_w0 != mod61_mul(a_w0, p_entry.mult_w0) {
                return None;
            }
            prod_len = trimmed_l;
            used_fft = true;
        } else {
            let need_prod = (la + lb + 1).max(f_k);
            prod_s[..need_prod].fill(0);
            prod_len = crate::limb::mul_limbs_karatsuba_into(
                ctx.kernels,
                &mut prod_s[..(la + lb + 1)],
                &f_limbs[..la],
                &p_limbs[..lb],
                cal.basecase_max_limbs,
            );
        }
    } else {
        prod_s[..f_k].fill(0);
    }

    let check_left = used_fft || !covered_by_ancestor;
    let exp_left_w0 = if check_left {
        let top_term_w0 = if top_limbs.is_empty() {
            0
        } else {
            mod61_mul(compute_limbs_w0(ctx.kernels, top_limbs), p_entry.mult_w0)
        };
        let i_left_slice = if prod_len > f_k {
            &prod_s[f_k..prod_len]
        } else {
            &[]
        };
        let int_w0 = compute_limbs_w0(ctx.kernels, i_left_slice);
        mod61_add(int_w0, top_term_w0)
    } else {
        0
    };

    let under_half = (prod_s[f_k - 1] >> 63) == 0;
    let xl_slice: &mut [u64] = if in_place_xl {
        let drop_low = k_limbs - k_left;
        let s = &mut frac_limbs[drop_low..k_limbs];
        adjust_left_in_place(s, k_left, b_left, under_half);
        s
    } else {
        make_guard_adjusted_left_stack(
            frac_limbs,
            k_limbs,
            k_left,
            b_left,
            under_half,
            xl_fallback.as_mut_ptr(),
        );
        xl_fallback
    };

    let left_covered = covered_by_ancestor || used_fft;
    let left_w0 = eval_batch_rec(
        ctx,
        cal,
        pow10_levels,
        rel_leaf,
        left_leaves,
        xl_slice,
        k_left,
        words_out,
        sub_work,
        left_covered,
    )?;
    if check_left && left_w0 != exp_left_w0 {
        return None;
    }

    let right_w0 = eval_batch_rec(
        ctx,
        cal,
        pow10_levels,
        rel_leaf + (left_leaves as usize),
        right_leaves,
        &mut prod_s[xr_start..f_k],
        k_right,
        words_out,
        sub_work,
        covered_by_ancestor,
    )?;
    let right_pow10_w0 = if right_leaves == left_leaves {
        p_entry.pow10_w0
    } else {
        pow10_mod_w0(d_right)
    };
    Some(mod61_add(mod61_mul(left_w0, right_pow10_w0), right_w0))
}

#[allow(clippy::too_many_arguments)]
fn convert_batch_subtree(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    cal: &Calibration,
    pow10_levels: &[Pow10Entry],
    start_leaf: u64,
    leaf_count: u64,
    frac_limbs: &[u64],
    k_limbs: usize,
    kept_full_leaves: u64,
    rem_digits: usize,
    total_leaves: u64,
    sink: &dyn DigitSink,
    guard_tail: &Mutex<Vec<u8>>,
) -> Result<Option<u64>, MulError> {
    if k_limbs > BATCH_MAX_K_LIMBS || leaf_count > BATCH_MAX_LEAVES {
        return Ok(None);
    }
    if leaf_count > 1024
        && !matches!(
            pow10_levels[(leaf_count - 1).ilog2() as usize].whole_cached,
            Some(CachedTransform::Fft(_))
        )
    {
        return Ok(None);
    }
    let mut lease = BatchScratchLease::acquire();
    let sc = lease.get_mut();
    let copy_n = frac_limbs.len().min(k_limbs);
    sc.padded_root[..copy_n].copy_from_slice(&frac_limbs[..copy_n]);
    if copy_n < k_limbs {
        sc.padded_root[copy_n..k_limbs].fill(0);
    }

    let lc_usize = leaf_count as usize;
    let covered_by_ancestor = start_leaf + leaf_count < total_leaves
        && matches!(
            plan.overrides.split_override,
            crate::plan::SplitFormOverride::Auto
        );
    let Some(tree_w0) = eval_batch_rec(
        ctx,
        cal,
        pow10_levels,
        0,
        leaf_count,
        &mut sc.padded_root[..k_limbs],
        k_limbs,
        &mut sc.words_out[..lc_usize],
        &mut sc.work_u64,
        covered_by_ancestor,
    ) else {
        return Ok(None);
    };

    let emitted_w0 = unsafe {
        (ctx.kernels.active.to_digits)(
            sc.ascii_buf.as_mut_ptr(),
            sc.words_out.as_ptr(),
            leaf_count,
            1,
        )
    };
    if emitted_w0 != tree_w0 {
        return Ok(None);
    }

    if ctx.kernels.count_work {
        ctx.kernels
            .stats
            .digit_words
            .fetch_add(leaf_count, Ordering::Relaxed);
    }

    let end_leaf = start_leaf + leaf_count;
    if plan.digits > 0 {
        let full_end = end_leaf.min(kept_full_leaves);
        if start_leaf < full_end {
            let full_cnt = (full_end - start_leaf) as usize;
            match plan.format {
                OutputFormat::Text => {
                    let off = 2 + start_leaf * (LEAF_DIGITS as u64);
                    sink.write_at(off, &sc.ascii_buf[..(full_cnt * LEAF_DIGITS)])
                        .map_err(MulError::FatalStop)?;
                }
                OutputFormat::Packed19 => {
                    let _ = unsafe {
                        (ctx.kernels.active.to_digits)(
                            sc.packed_buf.as_mut_ptr(),
                            sc.words_out.as_ptr(),
                            full_cnt as u64,
                            0,
                        )
                    };
                    let off = 16 + start_leaf * 8;
                    sink.write_at(off, &sc.packed_buf[..(full_cnt * 8)])
                        .map_err(MulError::FatalStop)?;
                }
            }
        }

        if rem_digits > 0 && start_leaf <= kept_full_leaves && kept_full_leaves < end_leaf {
            let rel = (kept_full_leaves - start_leaf) as usize;
            let leaf_ascii = &sc.ascii_buf[(rel * LEAF_DIGITS)..((rel + 1) * LEAF_DIGITS)];
            match plan.format {
                OutputFormat::Text => {
                    let off = 2 + kept_full_leaves * (LEAF_DIGITS as u64);
                    sink.write_at(off, &leaf_ascii[..rem_digits])
                        .map_err(MulError::FatalStop)?;
                }
                OutputFormat::Packed19 => {
                    let mut div = 1u64;
                    for _ in 0..(LEAF_DIGITS - rem_digits) {
                        div *= 10;
                    }
                    let partial_word = sc.words_out[rel] / div;
                    let off = 16 + kept_full_leaves * 8;
                    sink.write_at(off, &partial_word.to_le_bytes())
                        .map_err(MulError::FatalStop)?;
                }
            }
            let mut g = guard_tail.lock().unwrap();
            let tail_part = &leaf_ascii[rem_digits..LEAF_DIGITS];
            g[..tail_part.len()].copy_from_slice(tail_part);
        }

        let extra_leaf = total_leaves - 1;
        if start_leaf <= extra_leaf && extra_leaf < end_leaf {
            let rel = (extra_leaf - start_leaf) as usize;
            let leaf_ascii = &sc.ascii_buf[(rel * LEAF_DIGITS)..((rel + 1) * LEAF_DIGITS)];
            let mut g = guard_tail.lock().unwrap();
            let dst_off = if rem_digits > 0 {
                LEAF_DIGITS - rem_digits
            } else {
                0
            };
            let copy_len = g.len().saturating_sub(dst_off).min(LEAF_DIGITS);
            if copy_len > 0 {
                g[dst_off..(dst_off + copy_len)].copy_from_slice(&leaf_ascii[..copy_len]);
            }
        }
    } else {
        let mut g = guard_tail.lock().unwrap();
        g[..LEAF_DIGITS].copy_from_slice(&sc.ascii_buf[..LEAF_DIGITS]);
    }

    Ok(Some(emitted_w0))
}

#[cfg(test)]
mod wrapped_tests {
    use super::*;

    fn mul_small(v: &mut Vec<u64>, m: u64) {
        let mut carry = 0u128;
        for w in v.iter_mut() {
            let p = (*w as u128) * (m as u128) + carry;
            *w = p as u64;
            carry = p >> 64;
        }
        if carry != 0 {
            v.push(carry as u64);
        }
    }

    fn mul_school(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len() + b.len()];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in b.iter().enumerate() {
                let p = (x as u128) * (y as u128) + (out[i + j] as u128) + carry;
                out[i + j] = p as u64;
                carry = p >> 64;
            }
            out[i + b.len()] = carry as u64;
        }
        out
    }

    /// Section 5 & Section 11: only exact methods (`NTT`, limb) may wrap a product or make part
    /// of one; floating-point `Fft` entries in `Pow10Entry` are always whole products.
    #[test]
    fn conversion_never_wraps_inexact_fft_transforms() {
        let kernels = crate::refk::Kernels::reference_only();
        let prof = crate::tier::Profile::synthetic(
            2,
            2,
            32 * 1024,
            512 * 1024,
            4 * 1024 * 1024,
            128 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = Shares::from_profile(&prof);
        let cal = Calibration::synthetic(3.0e9, 20.0e9, true, Vec::new());
        let sched = crate::par::Scheduler::new(&shares, None);
        let fft_cache = crate::fft::FftTableCache::default();
        let ntt_cache = crate::ntt::NttTableCache::default();
        let ladder = Ladder::default();
        let tel = crate::check::Telemetry::default();
        let ctx = MulCtx::new(&kernels, &fft_cache, &ntt_cache, &sched, None, &ladder, &tel);

        let digits = 400_000u64;
        let plan = Plan::build(
            digits,
            OutputFormat::Text,
            &shares,
            &cal,
            crate::plan::PlanOverrides::default(),
        )
        .unwrap();
        let num_leaves = digits.div_ceil(LEAF_DIGITS as u64) + 1;
        let max_lv = (num_leaves - 1).ilog2() as usize;
        let table = build_pow10_table(&ctx, &plan, &shares, &cal, max_lv, num_leaves, false).unwrap();

        let mut saw_whole_fft = false;
        let mut saw_wrapped_ntt = false;
        for entry in &table {
            if let Some(CachedTransform::Fft(fb)) = &entry.whole_cached {
                saw_whole_fft = true;
                // Whole FFT capacity must cover the entire `mult * frac` product, never wrapped.
                let min_bits = entry.mult.body_bits() * 2;
                assert!((2 * fb.n_complex as u64) * (fb.coeff_bits as u64) >= min_bits);
            }
            if entry.wrapped_l_bits.is_some() {
                saw_wrapped_ntt = true;
                assert!(
                    entry.wrapped_spec.is_some_and(|s| s.wrap_cyclic),
                    "Wrapped conversion level must use exact cyclic NTT"
                );
            }
        }
        assert!(saw_whole_fft, "Expected whole cached FFT on L2-sized batch levels");
        assert!(saw_wrapped_ntt, "Expected exact wrapped NTT above batch levels");
    }

    /// Above the batches a node is multiplied by `5^d`, without its top `d` bits. Whatever way
    /// the product is made (wrapped against the level's cached transform, wrapped on its own,
    /// or whole), the right child's limbs are those of the fraction of `x * 10^d`, and the
    /// check passes for the whole number on top of `x * 10^d` and for no other.
    #[test]
    fn stripped_node_product_gives_the_split_by_the_power_of_ten() {
        let kernels = crate::refk::Kernels::reference_only();
        let prof = crate::tier::Profile::synthetic(
            2,
            2,
            32 * 1024,
            512 * 1024,
            4 * 1024 * 1024,
            128 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = Shares::from_profile(&prof);
        let cal = Calibration::synthetic(3.0e9, 20.0e9, true, Vec::new());
        let sched = crate::par::Scheduler::new(&shares, None);
        let fft_cache = crate::fft::FftTableCache::default();
        let ntt_cache = crate::ntt::NttTableCache::default();
        let ladder = Ladder::default();
        let tel = crate::check::Telemetry::default();
        let ctx = MulCtx::new(&kernels, &fft_cache, &ntt_cache, &sched, None, &ladder, &tel);

        let digits = (10 * BATCH_MAX_LEAVES) * (LEAF_DIGITS as u64);
        let plan = Plan::build(
            digits,
            OutputFormat::Text,
            &shares,
            &cal,
            crate::plan::PlanOverrides::default(),
        )
        .unwrap();
        let num_leaves = digits.div_ceil(LEAF_DIGITS as u64) + 1;
        let max_lv = (num_leaves - 1).ilog2() as usize;
        assert!(max_lv > STRIP_MIN_LV);
        let table = build_pow10_table(&ctx, &plan, &shares, &cal, max_lv, num_leaves, false).unwrap();

        let mut seed = 0x0BAD_5EED_1234_5677u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut wrapped_taken = 0usize;
        let mut turned_taken = 0usize;
        let mut turned_from_cache = 0usize;
        let mut turned_odd_from_cache = 0usize;
        let ntt_lv = BATCH_MAX_LEAVES.ilog2() as usize;
        assert!(max_lv > ntt_lv && ntt_lv >= STRIP_MIN_LV);
        for (lv, extra_leaves) in [
            (ntt_lv, 1u64 << ntt_lv),
            (ntt_lv, 601),
            (ntt_lv + 1, 1u64 << (ntt_lv + 1)),
            (ntt_lv + 1, 77),
            // nearly full: these stay with the cached transform of the full node
            (ntt_lv, (BATCH_MAX_LEAVES * 15) / 16),
            (ntt_lv + 1, (BATCH_MAX_LEAVES * 27) / 16),
        ] {
            let left_leaves = 1u64 << lv;
            let d_left = left_leaves * (LEAF_DIGITS as u64);
            let d_right = extra_leaves * (LEAF_DIGITS as u64);
            let k = frac_limbs_for_digits(d_left + d_right);
            let k_right = frac_limbs_for_digits(d_right);
            let p_entry = &table[lv];
            assert_eq!(p_entry.strip_limbs as u64 * 64, d_left);
            let mut p10 = vec![1u64];
            for _ in 0..left_leaves {
                mul_small(&mut p10, POW10_19);
            }
            // the entry holds the power of ten without its low zero limbs
            assert!(p10[..p_entry.strip_limbs].iter().all(|&w| w == 0));
            assert_eq!(&p10[p_entry.strip_limbs..], p_entry.mult.limbs.as_slice());
            assert_eq!(p_entry.mult_w0, compute_limbs_w0(&kernels, &p_entry.mult.limbs));

            let max_cached_k = frac_limbs_for_digits(2 * d_left);
            let point_id = 0x5000_0000u64 | ((lv as u64) << 44);
            for case in 0..5 {
                let x: Vec<u64> = match case {
                    // no bits below the stripped ones: the product is nothing but the top term
                    2 => (0..k)
                        .map(|i| if i >= k - p_entry.strip_limbs { next() } else { 0 })
                        .collect(),
                    3 => vec![u64::MAX; k],
                    4 => vec![0u64; k],
                    _ => (0..k).map(|_| next()).collect(),
                };
                let whole = mul_school(&x, &p10);
                let left_w0 = compute_limbs_w0(&kernels, &whole[k..]);
                // the check words of the node as the product that made it would hand them on,
                // and a place where the right child might be cut by a product of its own
                let cut_x = k - p_entry.strip_limbs;
                let words = PartWords {
                    cut: cut_x,
                    lo_w0: compute_limbs_w0(&kernels, &x[..cut_x]),
                    hi_w0: compute_limbs_w0(&kernels, &x[cut_x..]),
                };
                for (force_whole, par_w, pre, child_cuts) in [
                    (false, 1usize, None, vec![0, k_right]),
                    (false, 1, Some(words), vec![0, k_right / 3, k_right]),
                    (false, 2, None, vec![0, 1, k_right / 2, k_right - 1, k_right]),
                    (false, 2, Some(words), chain_cuts(&table, extra_leaves, k_right, 0, VIEW_LEVELS)),
                    (true, 1, None, vec![0, k_right]),
                    (true, 2, Some(words), vec![0, k_right / 2, k_right]),
                    // cuts that are not the child's: its check words then come as one piece
                    (false, 2, Some(words), vec![1, k_right]),
                ] {
                    let (x_right, under_half, check, parts) = node_right_fraction(
                        &ctx,
                        &plan,
                        &shares,
                        &cal,
                        p_entry,
                        point_id,
                        &x,
                        k,
                        k_right,
                        max_cached_k,
                        par_w,
                        force_whole,
                        pre,
                        &child_cuts,
                    )
                    .unwrap();
                    let what = format!(
                        "level {lv}, {extra_leaves} leaves right, case {case}, whole {force_whole}, {par_w} worker(s), words {}",
                        pre.is_some()
                    );
                    assert_eq!(x_right.as_slice(), &whole[k - k_right..k], "right fraction, {what}");
                    assert_eq!(under_half, (whole[k - 1] >> 63) == 0, "{what}");
                    assert!(check.holds(left_w0, p_entry.mult_w0), "check, {what}");
                    assert!(!check.holds(mod61_add(left_w0, 1), p_entry.mult_w0), "{what}");
                    if matches!(check, NodeCheck::Wrapped { .. }) {
                        assert!(!force_whole);
                        wrapped_taken += 1;
                    } else {
                        assert!(force_whole || case >= 2, "an ordinary fraction takes the wrapped product, {what}");
                    }
                    // a turned product hands on the check words of the right child
                    if let Some(sw) = parts {
                        assert!(matches!(check, NodeCheck::Wrapped { .. }), "{what}");
                        let want_cuts = if child_cuts[0] == 0 { child_cuts.clone() } else { vec![0, k_right] };
                        assert_eq!(sw.cuts, want_cuts, "{what}");
                        for (i, (a, b)) in cut_pairs(&sw.cuts).enumerate() {
                            assert_eq!(sw.w0s[i], compute_limbs_w0(&kernels, &x_right[a..b]), "{what}");
                        }
                        let any_cut = sw.cuts[sw.cuts.len() / 2];
                        let pw = sw.parts(0, any_cut, k_right).unwrap();
                        assert_eq!(pw.lo_w0, compute_limbs_w0(&kernels, &x_right[..any_cut]), "{what}");
                        assert_eq!(pw.hi_w0, compute_limbs_w0(&kernels, &x_right[any_cut..]), "{what}");
                        turned_taken += 1;
                        let from_cache = p_entry.wrapped_cached.is_some()
                            && p_entry.wrapped_turn > 0
                            && matches!(check, NodeCheck::Wrapped { l_bits, .. } if Some(l_bits) == p_entry.wrapped_l_bits);
                        if from_cache {
                            turned_from_cache += 1;
                            // a node smaller than the full one: its right child lies above
                            // the bottom of the turned product
                            if k_right < p_entry.wrapped_turn {
                                turned_odd_from_cache += 1;
                            }
                        }
                    }
                    // Words that are not those of the node must not pass for it: the word of
                    // the stripped limbs in every product, the word of the multiplied limbs
                    // where the product is the wrapped one, which takes it as handed on.
                    if let (Some(w), false) = (pre, force_whole) {
                        let wrong_hi = PartWords {
                            hi_w0: mod61_add(w.hi_w0, 1),
                            ..w
                        };
                        let wrong_lo = PartWords {
                            lo_w0: mod61_add(w.lo_w0, 1),
                            ..w
                        };
                        for (wrong, applies) in [
                            (wrong_hi, true),
                            (wrong_lo, matches!(check, NodeCheck::Wrapped { .. })),
                        ] {
                            if !applies {
                                continue;
                            }
                            let res = node_right_fraction(
                                &ctx,
                                &plan,
                                &shares,
                                &cal,
                                p_entry,
                                point_id,
                                &x,
                                k,
                                k_right,
                                max_cached_k,
                                par_w,
                                false,
                                Some(wrong),
                                &child_cuts,
                            );
                            assert!(
                                !matches!(&res, Ok((_, _, c, _)) if c.holds(left_w0, p_entry.mult_w0)),
                                "wrong words, {what}"
                            );
                        }
                    }
                }
            }
        }
        assert!(wrapped_taken >= 8);
        assert!(turned_taken >= 8, "turned products taken: {turned_taken}");
        assert!(turned_from_cache >= 2, "turned products from a cached transform: {turned_from_cache}");
        assert!(
            turned_odd_from_cache >= 2,
            "turned products of smaller nodes from a cached transform: {turned_odd_from_cache}"
        );
    }

    fn test_shares_and_sched(cores: u64) -> (Shares, crate::par::Scheduler) {
        let prof = crate::tier::Profile::synthetic(
            cores,
            cores,
            32 * 1024,
            512 * 1024,
            4 * 1024 * 1024,
            256 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = Shares::from_profile(&prof);
        let sched = crate::par::Scheduler::new(&shares, None);
        (shares, sched)
    }

    /// The check words taken piece by piece give the word of every range between two cuts, and
    /// follow a changed limb and a zeroed top.
    #[test]
    fn segment_words_give_ranges_and_follow_changes() {
        let kernels = crate::refk::Kernels::reference_only();
        let (_shares, sched) = test_shares_and_sched(2);
        let mut seed = 0x0123_4567_89AB_CDEFu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &len in &[1usize, 2, 17, 64, 300] {
            let mut limbs: Vec<u64> = (0..len).map(|_| next()).collect();
            let mut cuts = vec![0usize, len];
            for _ in 0..5 {
                cuts.push((next() as usize) % (len + 1));
            }
            cuts.sort_unstable();
            cuts.dedup();
            let mut sw = SegWords::compute(&kernels, &sched, &limbs, cuts.clone(), 2);
            let all_ranges = |sw: &SegWords, limbs: &[u64]| {
                for (i, &a) in cuts.iter().enumerate() {
                    for &b in &cuts[i..] {
                        assert_eq!(sw.range(a, b), Some(compute_limbs_w0(&kernels, &limbs[a..b])), "{a}..{b} of {len}");
                    }
                }
            };
            all_ranges(&sw, &limbs);
            // a place that is not a cut has no word
            if let Some(not_cut) = (0..=len).find(|c| !cuts.contains(c)) {
                assert_eq!(sw.range(0, not_cut), None);
                assert_eq!(sw.range(not_cut, len), None);
                assert!(sw.parts(0, not_cut, len).is_none());
            }
            // limbs change one at a time
            for _ in 0..20 {
                let idx = (next() as usize) % len;
                let old = limbs[idx];
                limbs[idx] = if next() & 1 == 0 { next() } else { old.wrapping_add(1) };
                assert!(sw.limb_changed(idx, old, limbs[idx]));
                all_ranges(&sw, &limbs);
            }
            assert!(!sw.limb_changed(len, 0, 1));
            // the top from a cut up is set to zero
            let from = cuts[cuts.len() / 2];
            limbs[from..].fill(0);
            assert!(sw.zeroed_from(from));
            all_ranges(&sw, &limbs);
        }
    }

    /// The guard adjustment that tells what it changed does what the plain one does, and what
    /// it tells is enough to keep the check words.
    #[test]
    fn tracked_guard_matches_the_plain_one() {
        let kernels = crate::refk::Kernels::reference_only();
        let (_shares, sched) = test_shares_and_sched(2);
        let mut seed = 0x0F1E_2D3C_4B5A_6978u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut seen = [0usize; 3];
        for leaves in [1u64, 2, 3, 64, 500] {
            let d_left = leaves * (LEAF_DIGITS as u64);
            let k_left = frac_limbs_for_digits(d_left);
            let b_left = bit_ceil_pow10(d_left);
            let eta_word = ((k_left as u64 * 64 - (b_left + 2)) / 64) as usize;
            for case in 0..60 {
                let mut x: Vec<u64> = (0..k_left).map(|_| next()).collect();
                match case % 6 {
                    // a run of ones above the guard bit: the carry runs through it
                    1 => x[eta_word..(eta_word + 1 + case % 7).min(k_left)].fill(u64::MAX),
                    // a run of zeros above it: the borrow does
                    2 => x[eta_word..(eta_word + 1 + case % 7).min(k_left)].fill(0),
                    // nothing from the guard bit up
                    3 => x[eta_word..].fill(0),
                    // all ones, all zeros
                    4 => x.fill(u64::MAX),
                    5 => x.fill(0),
                    _ => {}
                }
                for move_up in [false, true] {
                    let mut plain = x.clone();
                    apply_guard_eta(&mut plain, k_left, b_left, move_up);
                    let mut tracked = x.clone();
                    // the fraction as the top of a longer buffer, with its check words
                    let below = (case % 4) as usize;
                    let mut buf: Vec<u64> = (0..below).map(|_| next()).collect();
                    buf.extend_from_slice(&x);
                    let mut cuts = vec![0usize, below, below + k_left / 2, buf.len()];
                    cuts.sort_unstable();
                    cuts.dedup();
                    let mut sw = Some(SegWords::compute(&kernels, &sched, &buf, cuts.clone(), 1));
                    let change = apply_guard_eta_tracked(&mut tracked, k_left, b_left, move_up);
                    assert_eq!(tracked, plain, "{leaves} leaves, case {case}, up {move_up}");
                    buf[below..].copy_from_slice(&tracked);
                    match change {
                        EtaChange::Limbs { first, old, n } => {
                            seen[0] += 1;
                            assert_eq!(first, eta_word);
                            assert!((1..=4).contains(&n));
                            for j in 0..n {
                                assert_eq!(old[j], x[first + j]);
                                assert!(sw.as_mut().unwrap().limb_changed(below + first + j, old[j], tracked[first + j]));
                            }
                            // nothing else changed
                            for i in 0..k_left {
                                if i < first || i >= first + n {
                                    assert_eq!(tracked[i], x[i]);
                                }
                            }
                        }
                        EtaChange::Many => {
                            seen[1] += 1;
                            sw = None;
                        }
                        EtaChange::Zeroed => {
                            seen[2] += 1;
                            assert!(tracked.iter().all(|&w| w == 0));
                            assert!(sw.as_mut().unwrap().zeroed_from(below));
                        }
                    }
                    if let Some(sw) = &sw {
                        for (i, &a) in cuts.iter().enumerate() {
                            for &b in &cuts[i..] {
                                assert_eq!(sw.range(a, b), Some(compute_limbs_w0(&kernels, &buf[a..b])));
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&c| c > 0), "kinds of change seen: {seen:?}");
    }

    /// The frontier's way through a conversion, with left children passed on where they lie
    /// and moved every third split, writes the digits that the plain way node by node writes.
    #[test]
    fn frontier_with_left_children_in_place_writes_the_same_digits() {
        struct NoFault;
        impl LeafFaultHook for NoFault {
            fn on_leaf_word(&self, _leaf_idx: u64, _attempt: u32, _word_19: &mut u64, _emitted_w0: &mut u64) {}
        }
        let kernels = crate::refk::Kernels::reference_only();
        let (shares, sched) = test_shares_and_sched(1);
        let cal = Calibration::synthetic(3.0e9, 20.0e9, true, Vec::new());
        let fft_cache = crate::fft::FftTableCache::default();
        let ntt_cache = crate::ntt::NttTableCache::default();
        let ladder = Ladder::default();
        let tel = crate::check::Telemetry::default();
        let ctx = MulCtx::new(&kernels, &fft_cache, &ntt_cache, &sched, None, &ladder, &tel);

        // Enough leaves for four splits down the left side: two passed on, one moved, one more
        // passed on (see `VIEW_LEVELS`).
        let digits = (8 * BATCH_MAX_LEAVES + 1000) * (LEAF_DIGITS as u64);
        let plan = Plan::build(
            digits,
            OutputFormat::Text,
            &shares,
            &cal,
            crate::plan::PlanOverrides::default(),
        )
        .unwrap();
        assert!(!plan.out_of_core);
        let num_leaves = digits.div_ceil(LEAF_DIGITS as u64) + 1;
        assert!(num_leaves > 8 * BATCH_MAX_LEAVES);
        let k = frac_limbs_for_digits(num_leaves * (LEAF_DIGITS as u64));
        let mut seed = 0x5EED_0FA5_7FAC_E001u64;
        let limbs: Vec<u64> = (0..k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
            .collect();
        let frac = BigNum::from_limbs_par(limbs, false, 0, &kernels, &sched, 1);

        let run = |hook: Option<&dyn LeafFaultHook>| {
            let sink = MemoryDigitSink::new(plan.output_file_bytes as usize);
            let res = run_conversion(&ctx, &plan, &shares, &cal, 3, &frac, k, &sink, hook, None, None).unwrap();
            (sink.into_bytes(), res.dec_hash_w0, res.digits_in_doubt)
        };
        let retries_before = tel.retries.load(Ordering::Relaxed);
        let fast = run(None);
        assert_eq!(tel.retries.load(Ordering::Relaxed), retries_before, "a check of the frontier failed");
        // a hook on the leaves keeps the conversion off the frontier and off the batches
        let plain = run(Some(&NoFault));
        assert!(!fast.2 && !plain.2);
        assert_eq!(fast.1, plain.1);
        assert!(fast.0 == plain.0, "digits differ");
        assert_eq!(&fast.0[..2], b"3.");
    }

    /// Nodes that run at once take workers by their sizes, so that they end together.
    #[test]
    fn workers_go_by_size() {
        assert_eq!(workers_by_size(16, &[100, 57], &[64, 64]), vec![10, 6]);
        assert_eq!(workers_by_size(16, &[1, 1], &[64, 64]), vec![8, 8]);
        assert_eq!(workers_by_size(16, &[100, 14], &[64, 64]), vec![14, 2]);
        assert_eq!(workers_by_size(16, &[100, 14], &[4, 64]), vec![4, 12]);
        assert_eq!(workers_by_size(16, &[100, 14], &[4, 1]), vec![4, 1]);
        assert_eq!(workers_by_size(2, &[5, 5, 5], &[9, 9, 9]), vec![1, 1, 1]);
        assert_eq!(workers_by_size(16, &[7], &[64]), vec![16]);
        assert_eq!(workers_by_size(16, &[8, 8, 8, 1], &[64, 64, 64, 64]), vec![5, 5, 5, 1]);
        assert!(workers_by_size(16, &[], &[]).is_empty());
    }

    /// A shift modulo `2^L - 1` is the shift with what passes bit `L` brought around.
    #[test]
    fn cyclic_shift_brings_the_top_around() {
        let prof = crate::tier::Profile::synthetic(
            2,
            2,
            32 * 1024,
            512 * 1024,
            4 * 1024 * 1024,
            128 * 1024 * 1024,
            &[],
            &[],
        );
        let shares = Shares::from_profile(&prof);
        let sched = crate::par::Scheduler::new(&shares, None);
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for kl in [1usize, 2, 3, 7, 40] {
            let l_bits = (kl as u64) * 64;
            for len in 1..=kl {
                for _ in 0..40 {
                    let x: Vec<u64> = (0..len).map(|_| next()).collect();
                    let shift = next() % l_bits;
                    let got = shl_limbs_cyclic(&x, shift, l_bits, &sched, 1);
                    assert!(got.len() <= kl);
                    // bit `i` of `x` lands on bit `(i + shift) mod L`; no two bits of `x` land
                    // on the same place, so nothing carries
                    let mut want = vec![0u64; kl];
                    for i in 0..(len as u64) * 64 {
                        if (x[(i / 64) as usize] >> (i % 64)) & 1 == 1 {
                            let j = (i + shift) % l_bits;
                            want[(j / 64) as usize] |= 1u64 << (j % 64);
                        }
                    }
                    let mut got_p = got.clone();
                    got_p.resize(kl, 0);
                    assert_eq!(got_p, want, "{len} limb(s) by {shift} around {l_bits}");
                }
            }
        }
    }
}
