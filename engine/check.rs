//! Contract:
//! - Inputs: Spans of 64-bit limbs, expected check words `(w0, w1, w2)`, `Kernels` reference, and
//!   `CheckpointRecord` metadata / binary frame slices.
//! - Outputs: Dual-copy check words (`DualCheck`) for numbers in memory, 3-word check triples for
//!   stored pieces, in-place Rung-1 single-limb error correction, 4-rung `Ladder` decisions with
//!   per-point rollback tracking, fault/scrub `Telemetry` per `(core, page, device)`, `add_n`/
//!   `sub_n` parity piece construction/reconstruction, and `CheckpointRecord` frame codec.
//! - Data format: `w0` modulo `2^61 - 1`; `w1` modulo `159*2^42 + 1`; `w2` modulo `115*2^42 + 1`;
//!   checkpoint records as versioned binary frames with embedded `w0` integrity word.
//! - Work & bytes moved: O(n) single pass for check-word evaluation and single-limb fault location;
//!   zero extra passes on the happy path when kernels emit `w0` directly.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::refk::{
    Kernels, MOD_W0, PRIME_LOC1, PRIME_LOC2, fma_mul_mod, mod_pow_u64, mod61_add, mod61_mul,
    mod61_pow2, mod61_reduce, mod61_sub,
};

pub const RECORD_MAGIC: u64 = 0x5355_5049_5245_4331; // "SUPIREC1"
pub const RECORD_FORMAT_VERSION: u32 = 1;
pub const RECORD_SLOT_BYTES: u64 = 65536;

/// Dual-copy first check word carried by every number in memory (Section 11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualCheck {
    pub c0: u64,
    pub c1: u64,
}

impl DualCheck {
    #[inline]
    pub const fn new(w0: u64) -> Self {
        Self { c0: w0, c1: w0 }
    }

    /// Resolve the authoritative `w0` of a number in memory against its recomputed limb `w0`,
    /// repairing a single corrupted check-word copy in place (Rung 1, Section 11).
    pub fn verify_or_repair_at_rest(&mut self, actual_w0: u64, telemetry: &Telemetry) -> bool {
        if self.c0 == actual_w0 && self.c1 == actual_w0 {
            return true;
        }
        let page_addr = (self as *const Self as usize) & !4095usize;
        if self.c0 == actual_w0 && self.c1 != actual_w0 {
            self.c1 = actual_w0;
            telemetry.record_correction(crate::par::current_core_id(), page_addr, None);
            return true;
        }
        if self.c1 == actual_w0 && self.c0 != actual_w0 {
            self.c0 = actual_w0;
            telemetry.record_correction(crate::par::current_core_id(), page_addr, None);
            return true;
        }
        false
    }
}

/// Compute whole-value `w0` modulo `2^61 - 1` from unsigned limb check word, sign, and bit shift.
#[inline]
pub fn whole_value_w0(limbs_w0: u64, neg: bool, shift: u64) -> u64 {
    if limbs_w0 == 0 {
        return 0;
    }
    let shifted = mod61_mul(limbs_w0, mod61_pow2(shift));
    if neg && shifted != 0 {
        MOD_W0 - shifted
    } else {
        shifted
    }
}

/// Recover unsigned limb check word `limbs_w0` modulo `2^61 - 1` in `O(1)` from `whole_w0`, sign,
/// and `shift` (inverse of `whole_value_w0` since `2^61 == 1 (mod 2^61 - 1)`).
#[inline]
pub fn limbs_w0_from_whole(whole_w0: u64, neg: bool, shift: u64) -> u64 {
    let w = whole_w0 % MOD_W0;
    if w == 0 {
        return 0;
    }
    let unsigned_shifted = if neg { MOD_W0 - w } else { w };
    let inv_exp = (61 - (shift % 61)) % 61;
    mod61_mul(unsigned_shifted, mod61_pow2(inv_exp))
}

/// Compute `w0` of a span of limbs through the active kernel table.
#[inline]
pub fn compute_limbs_w0(kernels: &Kernels, limbs: &[u64]) -> u64 {
    if kernels.count_work {
        kernels
            .stats
            .check_limbs
            .fetch_add(limbs.len() as u64, Ordering::Relaxed);
    }
    if limbs.len() <= 4 {
        let acc: u128 = match limbs.len() {
            0 => return 0,
            1 => limbs[0] as u128,
            2 => (limbs[0] as u128) + ((limbs[1] as u128) << 3),
            3 => (limbs[0] as u128) + ((limbs[1] as u128) << 3) + ((limbs[2] as u128) << 6),
            _ => {
                (limbs[0] as u128)
                    + ((limbs[1] as u128) << 3)
                    + ((limbs[2] as u128) << 6)
                    + ((limbs[3] as u128) << 9)
            }
        };
        return mod61_reduce(acc);
    }
    if limbs.len() <= 16 {
        let mut w0 = 0u64;
        for &limb in limbs.iter().rev() {
            let l_mod = mod61_reduce(limb as u128);
            let shifted = mod61_reduce((w0 as u128) << 3);
            w0 = mod61_add(shifted, l_mod);
        }
        return w0;
    }
    let mut out = [0u64; 3];
    unsafe { (kernels.active.check_n)(limbs.as_ptr(), limbs.len() as u64, &mut out, 0) }
}

/// Compute `w0` of a large span of limbs in parallel across `max_workers` workers.
pub fn compute_limbs_w0_par(
    kernels: &Kernels,
    sched: &crate::par::Scheduler,
    limbs: &[u64],
    max_workers: usize,
) -> u64 {
    if max_workers <= 1 || limbs.len() < 65_536 {
        return compute_limbs_w0(kernels, limbs);
    }
    let chunk = 32_768usize;
    let num_chunks = limbs.len().div_ceil(chunk);
    let mut chunk_w0s = vec![0u64; num_chunks];
    let cw_ptr = chunk_w0s.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(limbs.len());
        let w = compute_limbs_w0(kernels, &limbs[start..end]);
        unsafe {
            *(cw_ptr as *mut u64).add(ci) = w;
        }
    });
    let step_shift = mod61_pow2((chunk as u64) * 3);
    let mut shift = 1u64;
    let mut total_w0 = 0u64;
    for &cw in &chunk_w0s {
        total_w0 = mod61_add(total_w0, mod61_mul(cw, shift));
        shift = mod61_mul(shift, step_shift);
    }
    total_w0
}

/// Compute all three check words `(w0, w1, w2)` of a span of limbs.
#[inline]
pub fn compute_limbs_all3(kernels: &Kernels, limbs: &[u64]) -> [u64; 3] {
    if kernels.count_work {
        kernels
            .stats
            .check_limbs
            .fetch_add((limbs.len() as u64) * 3, Ordering::Relaxed);
    }
    let mut out = [0u64; 3];
    unsafe {
        (kernels.active.check_n)(limbs.as_ptr(), limbs.len() as u64, &mut out, 1);
    }
    out
}

/// Expected whole-product check word from a wrapped transform of bit-length `L` (Section 11).
#[inline]
pub fn wrapped_product_expected_w0(
    wrapped_w0: u64,
    folded_w0: u64,
    transform_bits_l: u64,
    carry_left_all_zeros: bool,
) -> u64 {
    let two_l_minus_1 = mod61_sub(mod61_pow2(transform_bits_l), 1);
    let folded_adj = if carry_left_all_zeros {
        mod61_add(folded_w0, 1)
    } else {
        folded_w0
    };
    mod61_add(wrapped_w0, mod61_mul(two_l_minus_1, folded_adj))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionOutcome {
    Clean,
    Corrected { limb_index: usize },
    BeyondOneLimb,
}

/// Rung 1 single-limb error locator and corrector using `(w0, w1, w2)` (Section 11).
/// Locates `j` and signed error `e` with `|e| < 2^64` via CRT on `(P_1, P_2)` and disambiguates
/// any false pair via `w0` (`2^61 - 1`). Repairs in place iff a unique single-limb fix matches.
pub fn try_correct_single_limb(
    limbs: &mut [u64],
    actual: [u64; 3],
    expected: [u64; 3],
) -> CorrectionOutcome {
    if actual == expected {
        return CorrectionOutcome::Clean;
    }
    if limbs.is_empty()
        || actual[0] >= MOD_W0
        || expected[0] >= MOD_W0
        || expected[1] >= PRIME_LOC1
        || expected[2] >= PRIME_LOC2
    {
        return CorrectionOutcome::BeyondOneLimb;
    }

    let d0 = mod61_sub(actual[0], expected[0]);
    let d1 = (actual[1] + PRIME_LOC1 - (expected[1] % PRIME_LOC1)) % PRIME_LOC1;
    let d2 = (actual[2] + PRIME_LOC2 - (expected[2] % PRIME_LOC2)) % PRIME_LOC2;

    let two64_p1 = ((1u128 << 64) % (PRIME_LOC1 as u128)) as u64;
    let two64_p2 = ((1u128 << 64) % (PRIME_LOC2 as u128)) as u64;
    let inv_step1 = mod_pow_u64(two64_p1, PRIME_LOC1 - 2, PRIME_LOC1) as f64;
    let inv_step2 = mod_pow_u64(two64_p2, PRIME_LOC2 - 2, PRIME_LOC2) as f64;

    let p1_f = PRIME_LOC1 as f64;
    let inv_p1_f = 1.0 / p1_f;
    let p2_f = PRIME_LOC2 as f64;
    let inv_p2_f = 1.0 / p2_f;

    let p1_mod_p2 = PRIME_LOC1 % PRIME_LOC2;
    let inv_p1_mod_p2 = mod_pow_u64(p1_mod_p2, PRIME_LOC2 - 2, PRIME_LOC2) as f64;

    // Max k for |e| < 2^64 with P_1 = 159*2^42 + 1 (~6.993e14): floor(2^64 / P_1) = 26,379.
    const MAX_K: u64 = 26_382;

    let mut alpha1 = d1 as f64;
    let mut alpha2 = d2 as f64;
    let mut found: Option<(usize, u64)> = None;

    for (j, &cur_limb) in limbs.iter().enumerate() {
        let a1 = alpha1 as u64;
        let a2 = alpha2 as u64;
        let w0_weight = 1u64 << ((3 * j) % 61);

        // Candidate +e in (0, 2^64): u'_j = u_j + e => fixed = u'_j - e
        let a1_m2 = a1 % PRIME_LOC2;
        let diff_pos = if a2 >= a1_m2 {
            a2 - a1_m2
        } else {
            a2 + PRIME_LOC2 - a1_m2
        };
        let k_pos = fma_mul_mod(diff_pos as f64, inv_p1_mod_p2, p2_f, inv_p2_f) as u64;
        if k_pos <= MAX_K {
            let e128 = (a1 as u128) + (k_pos as u128) * (PRIME_LOC1 as u128);
            if e128 > 0 && e128 <= (u64::MAX as u128) {
                let e64 = e128 as u64;
                if let Some(fixed) = cur_limb.checked_sub(e64) {
                    if mod61_mul(mod61_reduce(e128), w0_weight) == d0 {
                        if found.is_some() {
                            return CorrectionOutcome::BeyondOneLimb;
                        }
                        found = Some((j, fixed));
                    }
                }
            }
        }

        // Candidate -m where m in (0, 2^64): u'_j = u_j - m => fixed = u'_j + m
        let b1 = if a1 == 0 { 0 } else { PRIME_LOC1 - a1 };
        let b2 = if a2 == 0 { 0 } else { PRIME_LOC2 - a2 };
        let b1_m2 = b1 % PRIME_LOC2;
        let diff_neg = if b2 >= b1_m2 {
            b2 - b1_m2
        } else {
            b2 + PRIME_LOC2 - b1_m2
        };
        let k_neg = fma_mul_mod(diff_neg as f64, inv_p1_mod_p2, p2_f, inv_p2_f) as u64;
        if k_neg <= MAX_K {
            let m128 = (b1 as u128) + (k_neg as u128) * (PRIME_LOC1 as u128);
            if m128 > 0 && m128 <= (u64::MAX as u128) {
                let m64 = m128 as u64;
                if let Some(fixed) = cur_limb.checked_add(m64) {
                    let neg_delta0 = mod61_sub(0, mod61_mul(mod61_reduce(m128), w0_weight));
                    if neg_delta0 == d0 {
                        if let Some((prev_j, prev_f)) = found {
                            if prev_j != j || prev_f != fixed {
                                return CorrectionOutcome::BeyondOneLimb;
                            }
                        } else {
                            found = Some((j, fixed));
                        }
                    }
                }
            }
        }

        alpha1 = fma_mul_mod(alpha1, inv_step1, p1_f, inv_p1_f);
        alpha2 = fma_mul_mod(alpha2, inv_step2, p2_f, inv_p2_f);
    }

    match found {
        Some((idx, fixed_val)) => {
            limbs[idx] = fixed_val;
            CorrectionOutcome::Corrected { limb_index: idx }
        }
        None => CorrectionOutcome::BeyondOneLimb,
    }
}

/// The four rungs of the correctness ladder (Section 11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    Correct = 1,
    MakeAgain = 2,
    GoBack = 3,
    Stop = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryMode {
    SameMethod,
    ForceNtt,
    ForceReference,
}

/// Ladder state tracking consecutive failures at a given operation/point so that any rung failing
/// twice at the same point escalates to the next rung, and repeated rollbacks stop cleanly.
pub struct Ladder {
    point_counts: Mutex<HashMap<(u64, u8), u32>>,
    goback_points: Mutex<HashMap<u64, u32>>,
    pub total_gobacks: AtomicU64,
    pub max_gobacks: u64,
}

impl Default for Ladder {
    fn default() -> Self {
        Self::new(8)
    }
}

impl Ladder {
    pub fn new(max_gobacks: u64) -> Self {
        Self {
            point_counts: Mutex::new(HashMap::new()),
            goback_points: Mutex::new(HashMap::new()),
            total_gobacks: AtomicU64::new(0),
            max_gobacks,
        }
    }

    /// Record an attempt on `rung` at `point_id`; if `rung` has already failed twice at `point_id`,
    /// escalates to the next rung (Section 11: "a rung that has failed twice at the same thing gives
    /// way to the next"). Rollback accounting (`goback_points`, `total_gobacks`) is recorded by
    /// `record_goback` once device-exclusion checks complete.
    pub fn claim_rung(&self, point_id: u64, mut desired: Rung) -> Rung {
        let mut map = self.point_counts.lock().unwrap();
        while (desired as u8) < (Rung::GoBack as u8) {
            let key = (point_id, desired as u8);
            let cnt = map.entry(key).or_insert(0);
            if *cnt < 2 {
                *cnt += 1;
                return desired;
            }
            desired = match desired {
                Rung::Correct => Rung::MakeAgain,
                Rung::MakeAgain => Rung::GoBack,
                Rung::GoBack | Rung::Stop => Rung::Stop,
            };
        }
        desired
    }

    /// Clear per-point Rung-1 and Rung-2 attempt counts when rolling back to an earlier checkpoint,
    /// while preserving `goback_points` and `total_gobacks` across the run.
    pub fn reset_below_goback(&self) {
        self.point_counts.lock().unwrap().clear();
    }

    /// Record a Rung-3 rollback at `point_id` when no device is being excluded.
    /// Section 11: "Going back a second time at the same point stops at the fourth rung."
    pub fn record_goback(&self, point_id: u64) -> Rung {
        let mut gb = self.goback_points.lock().unwrap();
        let p_cnt = gb.entry(point_id).or_insert(0);
        *p_cnt += 1;
        let tot = self.total_gobacks.fetch_add(1, Ordering::Relaxed) + 1;
        if *p_cnt >= 2 || tot > self.max_gobacks {
            Rung::Stop
        } else {
            Rung::GoBack
        }
    }
}

#[derive(Default)]
pub struct Telemetry {
    pub corrections: AtomicU64,
    pub retries: AtomicU64,
    pub scrubs: AtomicU64,
    pub core_faults: Mutex<HashMap<u64, u64>>,
    pub page_faults: Mutex<HashMap<usize, u64>>,
    pub device_faults: Mutex<HashMap<String, u64>>,
    pub stopped_devices: Mutex<HashSet<String>>,
    pub rollback_suspect_devices: Mutex<HashSet<String>>,
    pub device_gobacks: Mutex<HashMap<String, u32>>,
}

impl Telemetry {
    pub fn record_scrub(&self) {
        self.scrubs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_correction(&self, core_id: u64, addr: usize, device: Option<&str>) {
        self.corrections.fetch_add(1, Ordering::Relaxed);
        self.note_origin(core_id, addr, device);
    }

    pub fn record_failure(&self, core_id: u64, addr: usize, device: Option<&str>) -> u64 {
        self.retries.fetch_add(1, Ordering::Relaxed);
        if let Some(dev) = device {
            self.rollback_suspect_devices
                .lock()
                .unwrap()
                .insert(dev.to_string());
        }
        self.note_origin(core_id, addr, device)
    }

    pub fn record_retry(&self, core_id: u64, addr: usize, device: Option<&str>) {
        let _ = self.record_failure(core_id, addr, device);
    }

    pub fn mark_device_stopped(&self, dev: &str) {
        self.stopped_devices.lock().unwrap().insert(dev.to_string());
        self.rollback_suspect_devices
            .lock()
            .unwrap()
            .insert(dev.to_string());
    }

    pub fn is_device_stopped(&self, dev: &str) -> bool {
        self.stopped_devices.lock().unwrap().contains(dev)
    }

    pub fn take_rollback_device_gobacks(&self) -> Vec<(String, u32)> {
        let suspects: Vec<String> = self
            .rollback_suspect_devices
            .lock()
            .unwrap()
            .drain()
            .collect();
        if suspects.is_empty() {
            return Vec::new();
        }
        let mut gb = self.device_gobacks.lock().unwrap();
        let mut out = Vec::with_capacity(suspects.len());
        for dev in suspects {
            let cnt = gb.entry(dev.clone()).or_insert(0);
            *cnt += 1;
            out.push((dev, *cnt));
        }
        out
    }

    pub fn device_goback_count(&self, dev: &str) -> u32 {
        self.device_gobacks
            .lock()
            .unwrap()
            .get(dev)
            .copied()
            .unwrap_or(0)
    }

    fn note_origin(&self, core_id: u64, addr: usize, device: Option<&str>) -> u64 {
        if let Some(dev) = device {
            let mut dm = self.device_faults.lock().unwrap();
            let c = dm.entry(dev.to_string()).or_insert(0);
            *c += 1;
            return *c;
        }
        *self.core_faults.lock().unwrap().entry(core_id).or_insert(0) += 1;
        if addr != 0 {
            let page = addr >> 12;
            *self.page_faults.lock().unwrap().entry(page).or_insert(0) += 1;
        }
        0
    }

    pub fn summary_line(&self) -> String {
        let mut cores: Vec<(u64, u64)> = self
            .core_faults
            .lock()
            .unwrap()
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        cores.sort_unstable_by_key(|&(k, _)| k);
        let cores_str = cores
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",");

        let mut pages: Vec<(usize, u64)> = self
            .page_faults
            .lock()
            .unwrap()
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        pages.sort_unstable_by_key(|&(k, _)| k);
        let pages_str = pages
            .iter()
            .map(|(k, v)| format!("{k:#x}:{v}"))
            .collect::<Vec<_>>()
            .join(",");

        let mut devs: Vec<(String, u64)> = self
            .device_faults
            .lock()
            .unwrap()
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        devs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let devs_str = devs
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "scrubs={} corrections={} retries={} cores=[{cores_str}] pages=[{pages_str}] devices=[{devs_str}]",
            self.scrubs.load(Ordering::Relaxed),
            self.corrections.load(Ordering::Relaxed),
            self.retries.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// Checkpoint Parity Rounds & Dual-Slot Record Serialization (Section 11)
// ============================================================================

/// Compute limb-wise parity piece `sum = (p_0 + p_1 + ... + p_{k-1}) mod 2^{64*L}` via `add_n`
/// with carry out dropped (Section 11).
pub fn build_parity_piece(kernels: &Kernels, data_pieces: &[&[u64]], out_sum: &mut [u64]) {
    let len = out_sum.len();
    out_sum.fill(0);
    for piece in data_pieces {
        debug_assert_eq!(piece.len(), len);
        unsafe {
            (kernels.active.add_n)(
                out_sum.as_mut_ptr(),
                out_sum.as_ptr(),
                piece.as_ptr(),
                len as u64,
            );
        }
    }
}

/// Reconstruct one missing piece `missing_idx` in a round of `m >= 2` pieces (where `m - 1` is the
/// parity sum piece): uses `add_n` if `missing_idx == m - 1`, else `sub_n` with borrow dropped.
pub fn reconstruct_missing_piece(
    kernels: &Kernels,
    pieces: &[Option<&[u64]>],
    missing_idx: usize,
    recovered: &mut [u64],
) -> bool {
    let m = pieces.len();
    if m < 2 || missing_idx >= m {
        return false;
    }
    let len = recovered.len();
    if missing_idx == m - 1 {
        recovered.fill(0);
        for i in 0..(m - 1) {
            let Some(p) = pieces[i] else { return false };
            if p.len() != len {
                return false;
            }
            unsafe {
                (kernels.active.add_n)(
                    recovered.as_mut_ptr(),
                    recovered.as_ptr(),
                    p.as_ptr(),
                    len as u64,
                );
            }
        }
        true
    } else {
        let Some(sum_piece) = pieces[m - 1] else {
            return false;
        };
        if sum_piece.len() != len {
            return false;
        }
        recovered.copy_from_slice(sum_piece);
        for i in 0..(m - 1) {
            if i == missing_idx {
                continue;
            }
            let Some(p) = pieces[i] else { return false };
            if p.len() != len {
                return false;
            }
            unsafe {
                (kernels.active.sub_n)(
                    recovered.as_mut_ptr(),
                    recovered.as_ptr(),
                    p.as_ptr(),
                    len as u64,
                );
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageRecord {
    Series {
        term_count: u64,
        leaf_len: u64,
        dividing_mask: u64,
        completed_leaves: u64,
    },
    Finish {
        precision_limbs: u64,
        step_index: u32,
    },
    Convert {
        leaf_digits: u64,
        completed_leaves: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedNumberMeta {
    pub tag: u64,
    pub neg: bool,
    pub shift: u64,
    pub limb_len: u64,
    pub w0: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointRecord {
    pub format_version: u32,
    pub serial: u64,
    pub ssd_member_id: String,
    pub digits: u64,
    pub output_path: String,
    pub format_text: bool,
    pub piece_limbs: u64,
    pub stripe_width: u32,
    pub data_offset_bytes: u64,
    pub excluded_devices: Vec<String>,
    pub stage: StageRecord,
    pub numbers: Vec<SavedNumberMeta>,
}

impl CheckpointRecord {
    pub fn encode(&self, kernels: &Kernels) -> Vec<u8> {
        let mut payload = Vec::with_capacity(512);
        let push_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let push_u64 = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
        let push_str = |buf: &mut Vec<u8>, s: &str| {
            let b = s.as_bytes();
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        };

        push_u64(&mut payload, RECORD_MAGIC);
        push_u32(&mut payload, self.format_version);
        push_u32(&mut payload, u32::from(self.format_text));
        push_u64(&mut payload, self.serial);
        push_str(&mut payload, &self.ssd_member_id);
        push_u64(&mut payload, self.digits);
        push_str(&mut payload, &self.output_path);
        push_u64(&mut payload, self.piece_limbs);
        push_u32(&mut payload, self.stripe_width);
        push_u32(&mut payload, self.excluded_devices.len() as u32);
        push_u64(&mut payload, self.data_offset_bytes);
        for d in &self.excluded_devices {
            push_str(&mut payload, d);
        }

        match &self.stage {
            StageRecord::Series {
                term_count,
                leaf_len,
                dividing_mask,
                completed_leaves,
            } => {
                push_u32(&mut payload, 1);
                push_u32(&mut payload, 0);
                push_u64(&mut payload, *term_count);
                push_u64(&mut payload, *leaf_len);
                push_u64(&mut payload, *dividing_mask);
                push_u64(&mut payload, *completed_leaves);
            }
            StageRecord::Finish {
                precision_limbs,
                step_index,
            } => {
                push_u32(&mut payload, 2);
                push_u32(&mut payload, *step_index);
                push_u64(&mut payload, *precision_limbs);
            }
            StageRecord::Convert {
                leaf_digits,
                completed_leaves,
            } => {
                push_u32(&mut payload, 3);
                push_u32(&mut payload, 0);
                push_u64(&mut payload, *leaf_digits);
                push_u64(&mut payload, *completed_leaves);
            }
        }

        push_u32(&mut payload, self.numbers.len() as u32);
        push_u32(&mut payload, 0);
        for n in &self.numbers {
            push_u64(&mut payload, n.tag);
            push_u64(&mut payload, u64::from(n.neg));
            push_u64(&mut payload, n.shift);
            push_u64(&mut payload, n.limb_len);
            push_u64(&mut payload, n.w0);
        }

        while payload.len() % 8 != 0 {
            payload.push(0);
        }
        let limb_len = payload.len() / 8;
        let mut limbs = Vec::with_capacity(limb_len);
        for chunk in payload.chunks_exact(8) {
            limbs.push(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rec_w0 = compute_limbs_w0(kernels, &limbs);

        let mut frame = Vec::with_capacity(16 + payload.len());
        frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        frame.extend_from_slice(&rec_w0.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    pub fn decode(frame: &[u8], kernels: &Kernels) -> Result<Self, RecordError> {
        if frame.len() < 24 {
            return Err(RecordError::Corrupt);
        }
        let payload_len = u64::from_le_bytes(frame[0..8].try_into().unwrap()) as usize;
        let expected_w0 = u64::from_le_bytes(frame[8..16].try_into().unwrap());
        if payload_len == 0 || payload_len % 8 != 0 || 16 + payload_len > frame.len() {
            return Err(RecordError::EmptyOrTruncated);
        }
        let payload = &frame[16..(16 + payload_len)];
        let magic = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        if magic == 0 {
            return Err(RecordError::EmptyOrTruncated);
        }
        if magic != RECORD_MAGIC {
            return Err(RecordError::Corrupt);
        }

        let mut limbs = Vec::with_capacity(payload_len / 8);
        for chunk in payload.chunks_exact(8) {
            limbs.push(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let actual_w0 = compute_limbs_w0(kernels, &limbs);
        if actual_w0 != expected_w0 {
            return Err(RecordError::Corrupt);
        }

        let mut pos = 8usize;
        let read_u32 = |p: &mut usize| -> Result<u32, RecordError> {
            if *p + 4 > payload.len() {
                return Err(RecordError::Corrupt);
            }
            let v = u32::from_le_bytes(payload[*p..(*p + 4)].try_into().unwrap());
            *p += 4;
            Ok(v)
        };
        let read_u64 = |p: &mut usize| -> Result<u64, RecordError> {
            if *p + 8 > payload.len() {
                return Err(RecordError::Corrupt);
            }
            let v = u64::from_le_bytes(payload[*p..(*p + 8)].try_into().unwrap());
            *p += 8;
            Ok(v)
        };
        let read_str = |p: &mut usize| -> Result<String, RecordError> {
            let len = read_u32(p)? as usize;
            if *p + len > payload.len() {
                return Err(RecordError::Corrupt);
            }
            let s = std::str::from_utf8(&payload[*p..(*p + len)])
                .map_err(|_| RecordError::Corrupt)?
                .to_string();
            *p += len;
            Ok(s)
        };

        let format_version = read_u32(&mut pos)?;
        if format_version != RECORD_FORMAT_VERSION {
            return Err(RecordError::WrongVersion(format_version));
        }
        let format_text = read_u32(&mut pos)? != 0;
        let serial = read_u64(&mut pos)?;
        let ssd_member_id = read_str(&mut pos)?;
        let digits = read_u64(&mut pos)?;
        let output_path = read_str(&mut pos)?;
        let piece_limbs = read_u64(&mut pos)?;
        let stripe_width = read_u32(&mut pos)?;
        let excl_len = read_u32(&mut pos)? as usize;
        let data_offset_bytes = read_u64(&mut pos)?;
        let mut excluded_devices = Vec::with_capacity(excl_len);
        for _ in 0..excl_len {
            excluded_devices.push(read_str(&mut pos)?);
        }

        let stage_tag = read_u32(&mut pos)?;
        let stage_aux = read_u32(&mut pos)?;
        let stage = match stage_tag {
            1 => StageRecord::Series {
                term_count: read_u64(&mut pos)?,
                leaf_len: read_u64(&mut pos)?,
                dividing_mask: read_u64(&mut pos)?,
                completed_leaves: read_u64(&mut pos)?,
            },
            2 => StageRecord::Finish {
                precision_limbs: read_u64(&mut pos)?,
                step_index: stage_aux,
            },
            3 => StageRecord::Convert {
                leaf_digits: read_u64(&mut pos)?,
                completed_leaves: read_u64(&mut pos)?,
            },
            _ => return Err(RecordError::Corrupt),
        };

        let num_cnt = read_u32(&mut pos)? as usize;
        let _pad = read_u32(&mut pos)?;
        let mut numbers = Vec::with_capacity(num_cnt);
        for _ in 0..num_cnt {
            numbers.push(SavedNumberMeta {
                tag: read_u64(&mut pos)?,
                neg: read_u64(&mut pos)? != 0,
                shift: read_u64(&mut pos)?,
                limb_len: read_u64(&mut pos)?,
                w0: read_u64(&mut pos)?,
            });
        }

        Ok(Self {
            format_version,
            serial,
            ssd_member_id,
            digits,
            output_path,
            format_text,
            piece_limbs,
            stripe_width,
            data_offset_bytes,
            excluded_devices,
            stage,
            numbers,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordError {
    EmptyOrTruncated,
    WrongVersion(u32),
    Corrupt,
}
