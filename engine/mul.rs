//! Contract:
//! - Inputs: `BigNum` or limb-slice operands, `MulMethod` chosen by `Plan` (`Basecase`,
//!   `Karatsuba`, `Fft`, `Ntt`, `SplitHalving`, or `SplitOuter` modulo `2^m + 1`), runtime context
//!   (`Kernels`, transform caches, `Scheduler`, optional `Store`, `Ladder`, `Telemetry`), and
//!   operation `point_id`.
//! - Outputs: Verified whole products, squares, cached forward transforms, fused two-product
//!   merges (`Q2*T1 +- P1*T2`), and wrapped products (`mod 2^L - 1`), climbing the 4-rung
//!   correctness ladder on any check-word or rounding failure.
//! - Data format: `BigNum` carries trimmed 64-bit limbs, sign, zero-bit `shift`, and `DualCheck`;
//!   transform coefficients and outer-split parts never use multiples of 61 bits.
//! - Work & bytes moved: Dispatches the exact method and grid shape specified by `MulMethod`, plus
//!   `ParkedBigNum` staging to stream inactive numbers to and from `Store` extents on `Loader`
//!   beside worker compute with 3-word piece checks and Rung-1 repair; happy-path product check
//!   cost is a single `O(1)` `w0` comparison against kernel-emitted check words.

use crate::check::{
    CorrectionOutcome, DualCheck, Ladder, Rung, Telemetry, compute_limbs_all3, compute_limbs_w0,
    compute_limbs_w0_par, try_correct_single_limb, whole_value_w0,
};
use crate::fft::{
    FFT_MAX_ERR_BOUND, FftCachedFwd, FftTableCache, fft_forward_cached, fft_fused_merge_triple,
    fft_mul, fft_mul_pair_shared_b, fft_mul_slice_with_cached_into, fft_pmuladd_cached,
};
use crate::limb::{
    BigNum, LimbVec, add_bignum_par, add_into_at, add_limbs, cmp_limbs, empty_limb_vec,
    mul_limbs_basecase, mul_limbs_karatsuba, shl_limbs, shr_limbs, slice_to_limb_vec,
    sqr_limbs_basecase, sqr_limbs_karatsuba, sub_limbs, take_limb_buf_zeroed, trim_limbs,
};
use crate::ntt::{
    NttCachedFwd, NttOperand, NttSpec, NttTableCache, largest_primes_for_two_adic,
    ntt_combine_and_inverse, ntt_forward, ntt_forward_reusable, ntt_fused_merge_eligible,
    ntt_fused_merge_triple, ntt_mul_pair_shared_b, ntt_mul_slices, ntt_mul_with_cached_op,
};
use crate::par::Scheduler;
use crate::refk::{
    Kernels, PRIME_LOC1, PRIME_LOC2, fma_add_mod, fma_mul_mod, fma_sub_mod, mod61_add, mod61_mul,
    mod61_sub,
};
use crate::store::Store;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MulMethod {
    Basecase,
    Karatsuba {
        basecase_cutoff: usize,
    },
    Fft {
        n_complex: usize,
        coeff_bits: u32,
    },
    Ntt(NttSpec),
    SplitHalving {
        levels: u32,
        inner: Box<MulMethod>,
    },
    SplitOuter {
        k_parts: usize,
        part_bits_w: u64,
        inner_ntt: NttSpec,
    },
}

impl MulMethod {
    #[inline]
    pub fn is_transform_checked(&self) -> bool {
        !matches!(self, MulMethod::Basecase | MulMethod::Karatsuba { .. })
    }
}

pub enum CachedTransform {
    Fft(FftCachedFwd),
    Ntt(NttCachedFwd),
}

impl CachedTransform {
    pub fn release_store(&mut self, store: Option<&Store>) {
        if let CachedTransform::Ntt(n) = self {
            n.release_store(store);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MulError {
    NeedRollback(u64),
    FatalStop(String),
}

/// Optional fault-injection hook used by Gate 2 tests to exercise every rung of Section 11.
pub trait MulFaultHook: Send + Sync {
    fn on_product_limbs(
        &self,
        point_id: u64,
        attempt: u32,
        limbs: &mut LimbVec,
        emitted_w0: &mut u64,
        fft_err: &mut f64,
    );
}

pub struct MulCtx<'a> {
    pub kernels: &'a Kernels,
    pub fft_cache: &'a FftTableCache,
    pub ntt_cache: &'a NttTableCache,
    pub sched: &'a Scheduler,
    pub store: Option<&'a Store>,
    pub ladder: &'a Ladder,
    pub telemetry: &'a Telemetry,
    pub fft_err_bound: f64,
    pub fault_hook: Option<&'a dyn MulFaultHook>,
}

impl<'a> MulCtx<'a> {
    pub fn new(
        kernels: &'a Kernels,
        fft_cache: &'a FftTableCache,
        ntt_cache: &'a NttTableCache,
        sched: &'a Scheduler,
        store: Option<&'a Store>,
        ladder: &'a Ladder,
        telemetry: &'a Telemetry,
    ) -> Self {
        Self {
            kernels,
            fft_cache,
            ntt_cache,
            sched,
            store,
            ladder,
            telemetry,
            fft_err_bound: FFT_MAX_ERR_BOUND,
            fault_hook: None,
        }
    }

    /// Construct a guaranteed-exact fallback `NttSpec` for Rung-2 recovery when an FFT product
    /// exceeds its rounding bound or fails its check word.
    pub fn fallback_ntt_spec(&self, out_bits: u64, extra_sum_bit: bool) -> NttSpec {
        let needed_bits = out_bits.saturating_add(128);
        let mut n = 16usize;
        loop {
            let log_n = n.ilog2();
            let avail = largest_primes_for_two_adic(log_n.max(40), 18);
            if avail.len() >= 3 {
                let mut b = (needed_bits.div_ceil(n as u64) as u32).max(8);
                if b % 61 == 0 {
                    b += 1;
                }
                if b <= 400 {
                    // Pick smallest k_primes in 3..=avail.len() whose product carries sum safely.
                    let req_prime_bits =
                        (2 * b) as f64 + (n as f64).log2() + if extra_sum_bit { 4.0 } else { 3.0 };
                    let mut acc_bits = 0.0f64;
                    for (idx, &p) in avail.iter().rev().enumerate() {
                        acc_bits += (p as f64).log2();
                        let k = idx + 1;
                        if k >= 3 && acc_bits >= req_prime_bits {
                            return NttSpec {
                                n,
                                rows: 1,
                                cols: n,
                                tile_cols: n,
                                extra_sweeps: 0,
                                k_primes: k,
                                coeff_bits: b,
                                packed: false,
                                use_nt: false,
                                negacyclic: false,
                                wrap_cyclic: false,
                                out_of_core: false,
                                par_workers: 1,
                            };
                        }
                    }
                }
            }
            n <<= 1;
            debug_assert!(n <= (1usize << 42));
        }
    }

    /// Multiply two `BigNum` values as a whole product, stripping zero-bit shifts automatically
    /// and enforcing Section 11's check-word and recovery ladder.
    pub fn mul_whole(
        &self,
        point_id: u64,
        method: &MulMethod,
        a: &mut BigNum,
        b: &mut BigNum,
    ) -> Result<BigNum, MulError> {
        if a.is_zero() || b.is_zero() {
            return Ok(BigNum::zero());
        }
        let out_neg = a.neg ^ b.neg;
        let out_shift = a.shift + b.shift;

        // Limb methods pass on a derived check word to be verified at the first transform consumer
        // (Section 11: "The limb methods pass on a derived check word...").
        if !method.is_transform_checked() {
            let same = std::ptr::eq(a.limbs.as_ptr(), b.limbs.as_ptr()) && a.limbs.len() == b.limbs.len();
            let limbs = match method {
                MulMethod::Basecase => {
                    if same {
                        sqr_limbs_basecase(self.kernels, &a.limbs)
                    } else {
                        mul_limbs_basecase(self.kernels, &a.limbs, &b.limbs)
                    }
                }
                MulMethod::Karatsuba { basecase_cutoff } => {
                    if same {
                        sqr_limbs_karatsuba(self.kernels, &a.limbs, *basecase_cutoff)
                    } else {
                        mul_limbs_karatsuba(self.kernels, &a.limbs, &b.limbs, *basecase_cutoff)
                    }
                }
                _ => unreachable!(),
            };
            if limbs.is_empty() {
                return Ok(BigNum::zero());
            }
            let exp_whole_w0 = mod61_mul(a.w0(), b.w0());
            return Ok(BigNum {
                limbs,
                neg: out_neg,
                shift: out_shift,
                chk: DualCheck::new(exp_whole_w0),
            });
        }

        let out_bits = a.body_bits() + b.body_bits() + 64;
        let mut cur_method = method.clone();
        let mut ref_kernels: Option<Kernels> = None;
        let mut use_ref = false;
        let mut attempt = 0u32;

        loop {
            let k_ref: &Kernels = if use_ref {
                ref_kernels.get_or_insert_with(Kernels::reference_only)
            } else {
                self.kernels
            };
            let raw_res = self.run_raw_whole(k_ref, &cur_method, &a.limbs, &b.limbs);

            let (mut prod_limbs, mut emitted_w0, mut fft_err, store_failed) = match raw_res {
                Ok(t) => (t.0, t.1, t.2, false),
                Err(_) => (crate::limb::empty_limb_vec(), 0, 0.0, true),
            };

            if let Some(hook) = self.fault_hook {
                if !store_failed {
                    hook.on_product_limbs(
                        point_id,
                        attempt,
                        &mut prod_limbs,
                        &mut emitted_w0,
                        &mut fft_err,
                    );
                }
            }
            attempt += 1;

            if store_failed {
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Store failure exceeded rollback ladder".to_string(),
                    )),
                };
            }

            // Rounding bound check for FFT products (Section 11: over bound -> Rung 2 as NTT).
            if matches!(cur_method, MulMethod::Fft { .. }) && !(fft_err <= self.fft_err_bound) {
                self.telemetry
                    .record_retry(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::MakeAgain);
                match rung {
                    Rung::Correct | Rung::MakeAgain => {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                        continue;
                    }
                    Rung::GoBack => return Err(MulError::NeedRollback(point_id)),
                    Rung::Stop => {
                        return Err(MulError::FatalStop(
                            "Repeated FFT/NTT retry exhaustion".to_string(),
                        ));
                    }
                }
            }

            let cand_whole_w0 = whole_value_w0(emitted_w0, out_neg, out_shift);
            let exp_whole_w0 = mod61_mul(a.w0(), b.w0());

            if cand_whole_w0 == exp_whole_w0 {
                trim_limbs(&mut prod_limbs);
                if prod_limbs.is_empty() {
                    return Ok(BigNum::zero());
                }
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: out_neg,
                    shift: out_shift,
                    chk: DualCheck::new(exp_whole_w0),
                });
            }
            if std::env::var_os("SUPI_TRACE").is_some() {
                eprintln!(
                    "[MISMATCH mul_whole] pid=0x{:x} method={:?} use_ref={} la={} lb={} cand_w0={} exp_w0={}",
                    point_id, cur_method, use_ref, a.limbs.len(), b.limbs.len(), cand_whole_w0, exp_whole_w0
                );
            }

            // Before climbing any rung, re-verify sources and recompute product w0 from limbs
            // (Section 11: "Before a rung is climbed, the check words of the thing and of what
            // made it are computed again from their limbs").
            if !a.verify_at_rest(k_ref, self.telemetry) || !b.verify_at_rest(k_ref, self.telemetry) {
                self.telemetry
                    .record_failure(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Source number corrupted at rest beyond rollback limit".to_string(),
                    )),
                };
            }

            let recomputed_lw0 = compute_limbs_w0(k_ref, &prod_limbs);
            let recomputed_whole_w0 = whole_value_w0(recomputed_lw0, out_neg, out_shift);
            let fresh_exp_whole_w0 = mod61_mul(a.w0(), b.w0());
            if recomputed_whole_w0 == fresh_exp_whole_w0 {
                // Only the emitted check word had a transient discrepancy; the limbs stand!
                self.telemetry
                    .record_correction(crate::par::current_core_id(), 0, None);
                trim_limbs(&mut prod_limbs);
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: out_neg,
                    shift: out_shift,
                    chk: DualCheck::new(fresh_exp_whole_w0),
                });
            }

            // Rung 1: Compute locating check words (w1, w2) of A, B, and C, and attempt single-limb
            // in-place repair!
            let rung1 = self.ladder.claim_rung(point_id, Rung::Correct);
            if rung1 == Rung::Correct {
                let a3 = compute_limbs_all3(k_ref, &a.limbs);
                let b3 = compute_limbs_all3(k_ref, &b.limbs);
                let p1 = PRIME_LOC1 as f64;
                let p2 = PRIME_LOC2 as f64;
                let exp3 = [
                    mod61_mul(a3[0], b3[0]),
                    fma_mul_mod(a3[1] as f64, b3[1] as f64, p1, 1.0 / p1) as u64,
                    fma_mul_mod(a3[2] as f64, b3[2] as f64, p2, 1.0 / p2) as u64,
                ];
                let act3 = compute_limbs_all3(k_ref, &prod_limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut prod_limbs, act3, exp3)
                {
                    let page_addr = (&prod_limbs[limb_index] as *const u64 as usize) & !4095usize;
                    self.telemetry.record_correction(
                        crate::par::current_core_id(),
                        page_addr,
                        None,
                    );
                    trim_limbs(&mut prod_limbs);
                    return Ok(BigNum {
                        limbs: prod_limbs,
                        neg: out_neg,
                        shift: out_shift,
                        chk: DualCheck::new(fresh_exp_whole_w0),
                    });
                }
            }

            // Rung 2: Make again (as NTT if it was FFT, then with reference kernels).
            let next_rung = if rung1 == Rung::Correct {
                self.ladder.claim_rung(point_id, Rung::MakeAgain)
            } else {
                rung1
            };
            match next_rung {
                Rung::Correct | Rung::MakeAgain => {
                    self.telemetry
                        .record_retry(crate::par::current_core_id(), 0, None);
                    if matches!(cur_method, MulMethod::Fft { .. }) {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                    } else if !use_ref {
                        use_ref = true;
                    }
                    continue;
                }
                Rung::GoBack => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::NeedRollback(point_id));
                }
                Rung::Stop => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::FatalStop(
                        "Product check failed across all ladder rungs".to_string(),
                    ));
                }
            }
        }
    }

    pub fn mul_whole_immut(
        &self,
        point_id: u64,
        method: &MulMethod,
        a: &BigNum,
        b: &BigNum,
    ) -> Result<BigNum, MulError> {
        if a.is_zero() || b.is_zero() {
            return Ok(BigNum::zero());
        }
        let out_neg = a.neg ^ b.neg;
        let out_shift = a.shift + b.shift;

        if !method.is_transform_checked() {
            let same = std::ptr::eq(a.limbs.as_ptr(), b.limbs.as_ptr())
                && a.limbs.len() == b.limbs.len();
            let limbs = match method {
                MulMethod::Basecase => {
                    if same {
                        sqr_limbs_basecase(self.kernels, &a.limbs)
                    } else {
                        mul_limbs_basecase(self.kernels, &a.limbs, &b.limbs)
                    }
                }
                MulMethod::Karatsuba { basecase_cutoff } => {
                    if same {
                        sqr_limbs_karatsuba(self.kernels, &a.limbs, *basecase_cutoff)
                    } else {
                        mul_limbs_karatsuba(self.kernels, &a.limbs, &b.limbs, *basecase_cutoff)
                    }
                }
                _ => unreachable!(),
            };
            if limbs.is_empty() {
                return Ok(BigNum::zero());
            }
            let exp_whole_w0 = mod61_mul(a.w0(), b.w0());
            return Ok(BigNum {
                limbs,
                neg: out_neg,
                shift: out_shift,
                chk: DualCheck::new(exp_whole_w0),
            });
        }

        let out_bits = a.body_bits() + b.body_bits() + 64;
        let mut cur_method = method.clone();
        let mut ref_kernels: Option<Kernels> = None;
        let mut use_ref = false;
        let mut attempt = 0u32;

        loop {
            let k_ref: &Kernels = if use_ref {
                ref_kernels.get_or_insert_with(Kernels::reference_only)
            } else {
                self.kernels
            };
            let raw_res = self.run_raw_whole(k_ref, &cur_method, &a.limbs, &b.limbs);

            let (mut prod_limbs, mut emitted_w0, mut fft_err, store_failed) = match raw_res {
                Ok(t) => (t.0, t.1, t.2, false),
                Err(_) => (crate::limb::empty_limb_vec(), 0, 0.0, true),
            };

            if let Some(hook) = self.fault_hook {
                if !store_failed {
                    hook.on_product_limbs(
                        point_id,
                        attempt,
                        &mut prod_limbs,
                        &mut emitted_w0,
                        &mut fft_err,
                    );
                }
            }
            attempt += 1;

            if store_failed {
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Store failure exceeded rollback ladder".to_string(),
                    )),
                };
            }

            if matches!(cur_method, MulMethod::Fft { .. }) && !(fft_err <= self.fft_err_bound) {
                self.telemetry
                    .record_retry(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::MakeAgain);
                match rung {
                    Rung::Correct | Rung::MakeAgain => {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                        continue;
                    }
                    Rung::GoBack => return Err(MulError::NeedRollback(point_id)),
                    Rung::Stop => {
                        return Err(MulError::FatalStop(
                            "Repeated FFT/NTT retry exhaustion".to_string(),
                        ));
                    }
                }
            }

            let cand_whole_w0 = whole_value_w0(emitted_w0, out_neg, out_shift);
            let exp_whole_w0 = mod61_mul(a.w0(), b.w0());

            if cand_whole_w0 == exp_whole_w0 {
                trim_limbs(&mut prod_limbs);
                if prod_limbs.is_empty() {
                    return Ok(BigNum::zero());
                }
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: out_neg,
                    shift: out_shift,
                    chk: DualCheck::new(exp_whole_w0),
                });
            }

            let actual_aw0 = whole_value_w0(compute_limbs_w0(k_ref, &a.limbs), a.neg, a.shift);
            let actual_bw0 = whole_value_w0(compute_limbs_w0(k_ref, &b.limbs), b.neg, b.shift);
            if actual_aw0 != a.w0() || actual_bw0 != b.w0() {
                self.telemetry
                    .record_failure(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Source number corrupted at rest beyond rollback limit".to_string(),
                    )),
                };
            }

            let recomputed_lw0 = compute_limbs_w0(k_ref, &prod_limbs);
            let recomputed_whole_w0 = whole_value_w0(recomputed_lw0, out_neg, out_shift);
            let fresh_exp_whole_w0 = mod61_mul(actual_aw0, actual_bw0);
            if recomputed_whole_w0 == fresh_exp_whole_w0 {
                self.telemetry
                    .record_correction(crate::par::current_core_id(), 0, None);
                trim_limbs(&mut prod_limbs);
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: out_neg,
                    shift: out_shift,
                    chk: DualCheck::new(fresh_exp_whole_w0),
                });
            }

            let rung1 = self.ladder.claim_rung(point_id, Rung::Correct);
            if rung1 == Rung::Correct {
                let a3 = compute_limbs_all3(k_ref, &a.limbs);
                let b3 = compute_limbs_all3(k_ref, &b.limbs);
                let p1 = PRIME_LOC1 as f64;
                let p2 = PRIME_LOC2 as f64;
                let exp3 = [
                    mod61_mul(a3[0], b3[0]),
                    fma_mul_mod(a3[1] as f64, b3[1] as f64, p1, 1.0 / p1) as u64,
                    fma_mul_mod(a3[2] as f64, b3[2] as f64, p2, 1.0 / p2) as u64,
                ];
                let act3 = compute_limbs_all3(k_ref, &prod_limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut prod_limbs, act3, exp3)
                {
                    let page_addr = (&prod_limbs[limb_index] as *const u64 as usize) & !4095usize;
                    self.telemetry.record_correction(
                        crate::par::current_core_id(),
                        page_addr,
                        None,
                    );
                    trim_limbs(&mut prod_limbs);
                    return Ok(BigNum {
                        limbs: prod_limbs,
                        neg: out_neg,
                        shift: out_shift,
                        chk: DualCheck::new(fresh_exp_whole_w0),
                    });
                }
            }

            let next_rung = if rung1 == Rung::Correct {
                self.ladder.claim_rung(point_id, Rung::MakeAgain)
            } else {
                rung1
            };
            match next_rung {
                Rung::Correct | Rung::MakeAgain => {
                    self.telemetry
                        .record_retry(crate::par::current_core_id(), 0, None);
                    if matches!(cur_method, MulMethod::Fft { .. }) {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                    } else if !use_ref {
                        use_ref = true;
                    }
                    continue;
                }
                Rung::GoBack => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::NeedRollback(point_id));
                }
                Rung::Stop => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::FatalStop(
                        "Product check failed across all ladder rungs".to_string(),
                    ));
                }
            }
        }
    }

    pub fn sqr_whole(
        &self,
        point_id: u64,
        method: &MulMethod,
        a: &mut BigNum,
    ) -> Result<BigNum, MulError> {
        if a.is_zero() {
            return Ok(BigNum::zero());
        }
        let out_shift = 2 * a.shift;

        if !method.is_transform_checked() {
            let limbs = match method {
                MulMethod::Basecase => sqr_limbs_basecase(self.kernels, &a.limbs),
                MulMethod::Karatsuba { basecase_cutoff } => {
                    sqr_limbs_karatsuba(self.kernels, &a.limbs, *basecase_cutoff)
                }
                _ => unreachable!(),
            };
            if limbs.is_empty() {
                return Ok(BigNum::zero());
            }
            let aw0 = a.w0();
            let exp_whole_w0 = mod61_mul(aw0, aw0);
            return Ok(BigNum {
                limbs,
                neg: false,
                shift: out_shift,
                chk: DualCheck::new(exp_whole_w0),
            });
        }

        let out_bits = 2 * a.body_bits() + 64;
        let mut cur_method = method.clone();
        let mut ref_kernels: Option<Kernels> = None;
        let mut use_ref = false;
        let mut attempt = 0u32;

        loop {
            let k_ref: &Kernels = if use_ref {
                ref_kernels.get_or_insert_with(Kernels::reference_only)
            } else {
                self.kernels
            };
            // Passing `(&a.limbs, &a.limbs)` gives `same == true` inside `run_raw_whole`,
            // running a single forward transform instead of two!
            let raw_res = self.run_raw_whole(k_ref, &cur_method, &a.limbs, &a.limbs);

            let (mut prod_limbs, mut emitted_w0, mut fft_err, store_failed) = match raw_res {
                Ok(t) => (t.0, t.1, t.2, false),
                Err(_) => (crate::limb::empty_limb_vec(), 0, 0.0, true),
            };

            if let Some(hook) = self.fault_hook {
                if !store_failed {
                    hook.on_product_limbs(
                        point_id,
                        attempt,
                        &mut prod_limbs,
                        &mut emitted_w0,
                        &mut fft_err,
                    );
                }
            }
            attempt += 1;

            if store_failed {
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Store failure exceeded rollback ladder".to_string(),
                    )),
                };
            }

            if matches!(cur_method, MulMethod::Fft { .. }) && !(fft_err <= self.fft_err_bound) {
                self.telemetry
                    .record_retry(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::MakeAgain);
                match rung {
                    Rung::Correct | Rung::MakeAgain => {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                        continue;
                    }
                    Rung::GoBack => return Err(MulError::NeedRollback(point_id)),
                    Rung::Stop => {
                        return Err(MulError::FatalStop(
                            "Repeated FFT/NTT retry exhaustion".to_string(),
                        ));
                    }
                }
            }

            let cand_whole_w0 = whole_value_w0(emitted_w0, false, out_shift);
            let aw0 = a.w0();
            let exp_whole_w0 = mod61_mul(aw0, aw0);

            if cand_whole_w0 == exp_whole_w0 {
                trim_limbs(&mut prod_limbs);
                if prod_limbs.is_empty() {
                    return Ok(BigNum::zero());
                }
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: false,
                    shift: out_shift,
                    chk: DualCheck::new(exp_whole_w0),
                });
            }

            if !a.verify_at_rest(k_ref, self.telemetry) {
                self.telemetry
                    .record_failure(crate::par::current_core_id(), 0, None);
                let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                return match rung {
                    Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                    _ => Err(MulError::FatalStop(
                        "Source number corrupted at rest beyond rollback limit".to_string(),
                    )),
                };
            }

            let recomputed_lw0 = compute_limbs_w0(k_ref, &prod_limbs);
            let recomputed_whole_w0 = whole_value_w0(recomputed_lw0, false, out_shift);
            let fresh_aw0 = a.w0();
            let fresh_exp_whole_w0 = mod61_mul(fresh_aw0, fresh_aw0);
            if recomputed_whole_w0 == fresh_exp_whole_w0 {
                self.telemetry
                    .record_correction(crate::par::current_core_id(), 0, None);
                trim_limbs(&mut prod_limbs);
                return Ok(BigNum {
                    limbs: prod_limbs,
                    neg: false,
                    shift: out_shift,
                    chk: DualCheck::new(fresh_exp_whole_w0),
                });
            }

            let rung1 = self.ladder.claim_rung(point_id, Rung::Correct);
            if rung1 == Rung::Correct {
                let a3 = compute_limbs_all3(k_ref, &a.limbs);
                let p1 = PRIME_LOC1 as f64;
                let p2 = PRIME_LOC2 as f64;
                let exp3 = [
                    mod61_mul(a3[0], a3[0]),
                    fma_mul_mod(a3[1] as f64, a3[1] as f64, p1, 1.0 / p1) as u64,
                    fma_mul_mod(a3[2] as f64, a3[2] as f64, p2, 1.0 / p2) as u64,
                ];
                let act3 = compute_limbs_all3(k_ref, &prod_limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut prod_limbs, act3, exp3)
                {
                    let page_addr = (&prod_limbs[limb_index] as *const u64 as usize) & !4095usize;
                    self.telemetry.record_correction(
                        crate::par::current_core_id(),
                        page_addr,
                        None,
                    );
                    trim_limbs(&mut prod_limbs);
                    return Ok(BigNum {
                        limbs: prod_limbs,
                        neg: false,
                        shift: out_shift,
                        chk: DualCheck::new(fresh_exp_whole_w0),
                    });
                }
            }

            let next_rung = if rung1 == Rung::Correct {
                self.ladder.claim_rung(point_id, Rung::MakeAgain)
            } else {
                rung1
            };
            match next_rung {
                Rung::Correct | Rung::MakeAgain => {
                    self.telemetry
                        .record_retry(crate::par::current_core_id(), 0, None);
                    if matches!(cur_method, MulMethod::Fft { .. }) {
                        cur_method = MulMethod::Ntt(self.fallback_ntt_spec(out_bits, false));
                    } else if !use_ref {
                        use_ref = true;
                    }
                    continue;
                }
                Rung::GoBack => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::NeedRollback(point_id));
                }
                Rung::Stop => {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::FatalStop(
                        "Square check failed across all ladder rungs".to_string(),
                    ));
                }
            }
        }
    }

    /// Build a forward transform for `a.limbs` (single-consumer fast path).
    pub fn forward_cached(
        &self,
        method: &MulMethod,
        limbs: &[u64],
    ) -> Result<CachedTransform, MulError> {
        self.forward_cached_with_kernels(self.kernels, method, limbs, false)
    }

    /// Build a reusable forward transform for `a.limbs` shared across multiple products (completing
    /// the row pass upfront when `rows > 1` so consumers skip redundant row transforms).
    pub fn forward_cached_reusable(
        &self,
        method: &MulMethod,
        limbs: &[u64],
    ) -> Result<CachedTransform, MulError> {
        self.forward_cached_with_kernels(self.kernels, method, limbs, true)
    }

    fn forward_cached_with_kernels(
        &self,
        kernels: &Kernels,
        method: &MulMethod,
        limbs: &[u64],
        reusable: bool,
    ) -> Result<CachedTransform, MulError> {
        match method {
            MulMethod::Fft {
                n_complex,
                coeff_bits,
            } => Ok(CachedTransform::Fft(fft_forward_cached(
                kernels,
                self.fft_cache,
                limbs,
                *n_complex,
                *coeff_bits,
            ))),
            MulMethod::Ntt(spec) => {
                let res = if reusable {
                    ntt_forward_reusable(
                        kernels,
                        self.ntt_cache,
                        self.sched,
                        self.store,
                        limbs,
                        *spec,
                    )
                } else {
                    ntt_forward(
                        kernels,
                        self.ntt_cache,
                        self.sched,
                        self.store,
                        limbs,
                        *spec,
                    )
                };
                res.map(CachedTransform::Ntt)
                    .map_err(|_| MulError::NeedRollback(0xF000_0000_0000_0001))
            }
            _ => unreachable!("forward_cached only applies to FFT or NTT"),
        }
    }

    /// Multiply `a_limbs` with pre-transformed `b_cached`, checking whole result when `!wrap_cyclic`.
    pub fn mul_limbs_with_cached(
        &self,
        point_id: u64,
        method: &MulMethod,
        a_limbs: &[u64],
        b_limbs_for_retry: &[u64],
        b_cached: &CachedTransform,
        out_limbs: usize,
        expected_limbs_w0: Option<u64>,
    ) -> Result<(LimbVec, u64), MulError> {
        // Fast path using existing `b_cached`.
        let mut r1_cand: Option<LimbVec> = None;
        match (method, b_cached) {
            (
                MulMethod::Fft {
                    n_complex,
                    coeff_bits,
                },
                CachedTransform::Fft(fb),
            ) => {
                debug_assert_eq!(*n_complex, fb.n_complex);
                debug_assert_eq!(*coeff_bits, fb.coeff_bits);
                let mut out = crate::limb::take_limb_buf_uninit(out_limbs);
                unsafe {
                    out.set_len(out_limbs);
                }
                if let Some((trimmed_len, mut w0, mut err)) = fft_mul_slice_with_cached_into(
                    self.kernels,
                    self.fft_cache,
                    &mut out,
                    a_limbs,
                    fb,
                    out_limbs,
                ) {
                    out.truncate(trimmed_len);
                    if let Some(hook) = self.fault_hook {
                        hook.on_product_limbs(point_id, 0, &mut out, &mut w0, &mut err);
                    }
                    if err <= self.fft_err_bound {
                        if expected_limbs_w0.is_none_or(|exp| exp == w0) {
                            return Ok((out, w0));
                        }
                        r1_cand = Some(out);
                    } else {
                        crate::limb::put_limb_buf(out);
                    }
                } else {
                    crate::limb::put_limb_buf(out);
                }
            }
            (MulMethod::Ntt(spec), CachedTransform::Ntt(fb)) => {
                match ntt_mul_with_cached_op(
                    self.kernels,
                    self.ntt_cache,
                    self.sched,
                    self.store,
                    self.telemetry,
                    a_limbs,
                    NttOperand::Shared(fb),
                    *spec,
                    out_limbs,
                ) {
                    Ok((mut out, mut w0)) => {
                        let mut dummy_err = 0.0;
                        if let Some(hook) = self.fault_hook {
                            hook.on_product_limbs(point_id, 0, &mut out, &mut w0, &mut dummy_err);
                        }
                        if expected_limbs_w0.is_none_or(|exp| exp == w0) {
                            return Ok((out, w0));
                        }
                        if !spec.wrap_cyclic {
                            r1_cand = Some(out);
                        }
                    }
                    Err(_) => {
                        let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
                        return match rung {
                            Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                            _ => Err(MulError::FatalStop(
                                "Store failure exceeded rollback ladder".to_string(),
                            )),
                        };
                    }
                }
            }
            _ => {}
        }

        // Rung 1 single-limb repair when `mul_limbs_with_cached` is making a whole product!
        if let (Some(mut cand_limbs), Some(exp_w0)) = (r1_cand, expected_limbs_w0) {
            let r1 = self.ladder.claim_rung(point_id, Rung::Correct);
            if r1 == Rung::Correct {
                let a3 = compute_limbs_all3(self.kernels, a_limbs);
                let b3 = compute_limbs_all3(self.kernels, b_limbs_for_retry);
                if mod61_mul(a3[0], b3[0]) != exp_w0 {
                    self.telemetry
                        .record_failure(crate::par::current_core_id(), 0, None);
                    return Err(MulError::NeedRollback(point_id));
                }
                let p1 = PRIME_LOC1 as f64;
                let p2 = PRIME_LOC2 as f64;
                let exp3 = [
                    mod61_mul(a3[0], b3[0]),
                    fma_mul_mod(a3[1] as f64, b3[1] as f64, p1, 1.0 / p1) as u64,
                    fma_mul_mod(a3[2] as f64, b3[2] as f64, p2, 1.0 / p2) as u64,
                ];
                let act3 = compute_limbs_all3(self.kernels, &cand_limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut cand_limbs, act3, exp3)
                {
                    let page_addr = (&cand_limbs[limb_index] as *const u64 as usize) & !4095usize;
                    self.telemetry
                        .record_correction(crate::par::current_core_id(), page_addr, None);
                    trim_limbs(&mut cand_limbs);
                    return Ok((cand_limbs, exp_w0));
                }
            }
        }

        // Rung 2: Rebuild cached transform from `b_limbs_for_retry` and recompute (Section 11:
        // "with any cached transform it uses made again too").
        self.telemetry
            .record_retry(crate::par::current_core_id(), 0, None);
        let rung = self.ladder.claim_rung(point_id, Rung::MakeAgain);
        if rung >= Rung::GoBack {
            return if rung == Rung::GoBack {
                Err(MulError::NeedRollback(point_id))
            } else {
                Err(MulError::FatalStop(
                    "Cached product failed beyond ladder".to_string(),
                ))
            };
        }

        let mut ba = BigNum::from_limbs(a_limbs.to_vec(), false, 0, self.kernels);
        let mut bb = BigNum::from_limbs(b_limbs_for_retry.to_vec(), false, 0, self.kernels);
        let fallback_method = match method {
            MulMethod::Fft { .. } => {
                let bits = ((a_limbs.len() + b_limbs_for_retry.len()) as u64) * 64 + 64;
                MulMethod::Ntt(self.fallback_ntt_spec(bits, false))
            }
            other => other.clone(),
        };
        let whole = self.mul_whole(point_id, &fallback_method, &mut ba, &mut bb)?;
        let mut limbs = whole.into_limbs();
        if limbs.len() > out_limbs {
            limbs.truncate(out_limbs);
        }
        let w0 = compute_limbs_w0(self.kernels, &limbs);
        trim_limbs(&mut limbs);
        Ok((limbs, w0))
    }

    /// Fused binary-splitting merge for `T = q2 * t1 + p1 * t2` (7 transforms instead of 9,
    /// Section 4), also returning cached forward transforms of `(q2, p1)` (plus `q2_full.limbs`)
    /// so `Q = q1 * q2` shares `fwd(q2)` without re-shifting `q2`!
    #[allow(clippy::too_many_arguments)]
    pub fn fused_linear_pair(
        &self,
        point_id: u64,
        method: &MulMethod,
        par_workers: usize,
        q2: &mut BigNum,
        t1: &mut BigNum,
        p1: &mut BigNum,
        t2: &mut BigNum,
        reuse_q2: bool,
        reuse_p1: bool,
    ) -> Result<
        (
            BigNum,
            Option<(CachedTransform, LimbVec)>,
            Option<CachedTransform>,
        ),
        MulError,
    > {
        let pw = par_workers.max(1);
        // Materialize shifts only when non-zero; borrow `t1`, `p1`, `t2` directly when `shift == 0`!
        let q2_full = q2.materialize_shift_par(self.kernels, self.sched, pw);
        let t1_shifted = if !t1.is_zero() && t1.shift != 0 {
            Some(t1.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let p1_shifted = if !p1.is_zero() && p1.shift != 0 {
            Some(p1.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let t2_shifted = if !t2.is_zero() && t2.shift != 0 {
            Some(t2.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let t1_ref: &BigNum = t1_shifted.as_ref().unwrap_or(&*t1);
        let p1_ref: &BigNum = p1_shifted.as_ref().unwrap_or(&*p1);
        let t2_ref: &BigNum = t2_shifted.as_ref().unwrap_or(&*t2);

        if q2_full.is_zero() && p1_ref.is_zero() {
            return Ok((BigNum::zero(), None, None));
        }
        if q2_full.is_zero() || t1_ref.is_zero() {
            let mut a = p1_shifted.unwrap_or_else(|| p1.clone());
            let mut b = t2_shifted.unwrap_or_else(|| t2.clone());
            let res = self.mul_whole(point_id, method, &mut a, &mut b)?;
            return Ok((res, None, None));
        }
        if p1_ref.is_zero() || t2_ref.is_zero() {
            let mut a = q2_full;
            let mut b = t1_shifted.unwrap_or_else(|| t1.clone());
            let res = self.mul_whole(point_id, method, &mut a, &mut b)?;
            return Ok((res, None, None));
        }

        let sign1 = q2_full.neg ^ t1_ref.neg;
        let sign2 = p1_ref.neg ^ t2_ref.neg;
        let sub_second = sign1 != sign2;

        // When subtracting, ensure `|q2*t1| >= |p1*t2|` by swapping order if needed so the unsigned
        // magnitude emitted by `fft_carry` / `ntt_crt` is non-negative!
        let est1 = q2_full.body_bits() + t1_ref.body_bits();
        let est2 = p1_ref.body_bits() + t2_ref.body_bits();
        let swap_terms = sub_second && est2 + 2 >= est1;
        if swap_terms || !matches!(method, MulMethod::Fft { .. } | MulMethod::Ntt(_)) {
            // Safe exact two-product path when magnitudes are close on subtraction or non-cacheable.
            let mut a1 = q2_full;
            let mut b1 = t1_shifted.unwrap_or_else(|| t1.clone());
            let mut a2 = p1_shifted.unwrap_or_else(|| p1.clone());
            let mut b2 = t2_shifted.unwrap_or_else(|| t2.clone());
            let prod1 = self.mul_whole(point_id, method, &mut a1, &mut b1)?;
            let prod2 = self.mul_whole(point_id.wrapping_add(1), method, &mut a2, &mut b2)?;
            return Ok((
                add_bignum_par(self.kernels, &prod1, &prod2, self.sched, pw),
                None,
                None,
            ));
        }

        let out_limbs = (q2_full.limbs.len() + t1_ref.limbs.len())
            .max(p1_ref.limbs.len() + t2_ref.limbs.len())
            + 2;
        let exp_whole_w0 = mod61_add(
            mod61_mul(q2_full.w0(), t1_ref.w0()),
            mod61_mul(p1_ref.w0(), t2_ref.w0()),
        );

        let lazy_ntt_reuse = matches!(
            method,
            MulMethod::Ntt(s)
                if s.rows > 1 && s.cols >= 64 && s.extra_sweeps == 0 && !s.out_of_core
        );
        let mut fq2 = if reuse_q2 && !lazy_ntt_reuse {
            self.forward_cached_reusable(method, &q2_full.limbs)?
        } else {
            self.forward_cached(method, &q2_full.limbs)?
        };
        let mut ft1 = match self.forward_cached(method, &t1_ref.limbs) {
            Ok(v) => v,
            Err(e) => {
                fq2.release_store(self.store);
                return Err(e);
            }
        };
        let mut fp1 = match if reuse_p1 && !lazy_ntt_reuse {
            self.forward_cached_reusable(method, &p1_ref.limbs)
        } else {
            self.forward_cached(method, &p1_ref.limbs)
        } {
            Ok(v) => v,
            Err(e) => {
                fq2.release_store(self.store);
                ft1.release_store(self.store);
                return Err(e);
            }
        };
        let mut ft2 = match self.forward_cached(method, &t2_ref.limbs) {
            Ok(v) => v,
            Err(e) => {
                fq2.release_store(self.store);
                ft1.release_store(self.store);
                fp1.release_store(self.store);
                return Err(e);
            }
        };

        let mut fused_store_failed = false;
        let fused_attempt = match (&mut fq2, &mut ft1, &mut fp1, &mut ft2) {
            (
                CachedTransform::Fft(a1),
                CachedTransform::Fft(b1),
                CachedTransform::Fft(a2),
                CachedTransform::Fft(b2),
            ) => fft_pmuladd_cached(
                self.kernels,
                self.fft_cache,
                a1,
                b1,
                a2,
                b2,
                sub_second,
                out_limbs,
            )
            .and_then(|(l, w0, err)| {
                if err <= self.fft_err_bound {
                    Some((l, w0))
                } else {
                    None
                }
            }),
            (
                CachedTransform::Ntt(a1),
                CachedTransform::Ntt(b1),
                CachedTransform::Ntt(a2),
                CachedTransform::Ntt(b2),
            ) => {
                let op_a1 = if reuse_q2 {
                    NttOperand::ReuseInPlace(a1)
                } else {
                    NttOperand::Owned(a1)
                };
                let op_a2 = if reuse_p1 {
                    NttOperand::ReuseInPlace(a2)
                } else {
                    NttOperand::Owned(a2)
                };
                match ntt_combine_and_inverse(
                    self.kernels,
                    self.ntt_cache,
                    self.sched,
                    self.store,
                    self.telemetry,
                    b1,
                    Some(op_a1),
                    Some((op_a2, NttOperand::Owned(b2), sub_second)),
                    out_limbs,
                ) {
                    Ok(v) => Some(v),
                    Err(_) => {
                        fused_store_failed = true;
                        None
                    }
                }
            }
            _ => None,
        };

        // Release temporary transforms `ft1`, `ft2` from store if out-of-core.
        ft1.release_store(self.store);
        ft2.release_store(self.store);
        if fused_store_failed {
            fq2.release_store(self.store);
            fp1.release_store(self.store);
            let rung = self.ladder.claim_rung(point_id, Rung::GoBack);
            return match rung {
                Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                _ => Err(MulError::FatalStop(
                    "Store failure exceeded rollback ladder".to_string(),
                )),
            };
        }

        if let Some((mut limbs, mut lw0)) = fused_attempt {
            let mut dummy_err = 0.0;
            if let Some(hook) = self.fault_hook {
                hook.on_product_limbs(point_id, 0, &mut limbs, &mut lw0, &mut dummy_err);
            }
            let cand_w0 = whole_value_w0(lw0, sign1, 0);
            if cand_w0 == exp_whole_w0 {
                let res = BigNum {
                    limbs,
                    neg: sign1,
                    shift: 0,
                    chk: DualCheck::new(exp_whole_w0),
                };
                let ret_fq2 = if reuse_q2 {
                    Some((fq2, q2_full.into_limbs()))
                } else {
                    fq2.release_store(self.store);
                    crate::limb::put_limb_buf(q2_full.into_limbs());
                    None
                };
                let ret_fp1 = if reuse_p1 {
                    Some(fp1)
                } else {
                    fp1.release_store(self.store);
                    None
                };
                return Ok((res, ret_fq2, ret_fp1));
            }

            // Rung 1 single-limb repair on fused whole result!
            let r1 = self.ladder.claim_rung(point_id, Rung::Correct);
            if r1 == Rung::Correct {
                let q2_3 = compute_limbs_all3(self.kernels, &q2_full.limbs);
                let t1_3 = compute_limbs_all3(self.kernels, &t1_ref.limbs);
                let p1_3 = compute_limbs_all3(self.kernels, &p1_ref.limbs);
                let t2_3 = compute_limbs_all3(self.kernels, &t2_ref.limbs);
                let p1_f = PRIME_LOC1 as f64;
                let p2_f = PRIME_LOC2 as f64;
                let term1_w0 = mod61_mul(q2_3[0], t1_3[0]);
                let term2_w0 = mod61_mul(p1_3[0], t2_3[0]);
                let exp_l0 = if sub_second {
                    mod61_sub(term1_w0, term2_w0)
                } else {
                    mod61_add(term1_w0, term2_w0)
                };
                let term1_1 = fma_mul_mod(q2_3[1] as f64, t1_3[1] as f64, p1_f, 1.0 / p1_f);
                let term2_1 = fma_mul_mod(p1_3[1] as f64, t2_3[1] as f64, p1_f, 1.0 / p1_f);
                let exp_l1 = if sub_second {
                    fma_sub_mod(term1_1, term2_1, p1_f) as u64
                } else {
                    fma_add_mod(term1_1, term2_1, p1_f) as u64
                };
                let term1_2 = fma_mul_mod(q2_3[2] as f64, t1_3[2] as f64, p2_f, 1.0 / p2_f);
                let term2_2 = fma_mul_mod(p1_3[2] as f64, t2_3[2] as f64, p2_f, 1.0 / p2_f);
                let exp_l2 = if sub_second {
                    fma_sub_mod(term1_2, term2_2, p2_f) as u64
                } else {
                    fma_add_mod(term1_2, term2_2, p2_f) as u64
                };
                let act3 = compute_limbs_all3(self.kernels, &limbs);
                if let CorrectionOutcome::Corrected { limb_index } =
                    try_correct_single_limb(&mut limbs, act3, [exp_l0, exp_l1, exp_l2])
                {
                    let page_addr = (&limbs[limb_index] as *const u64 as usize) & !4095usize;
                    self.telemetry
                        .record_correction(crate::par::current_core_id(), page_addr, None);
                    trim_limbs(&mut limbs);
                    let res = BigNum {
                        limbs,
                        neg: sign1,
                        shift: 0,
                        chk: DualCheck::new(exp_whole_w0),
                    };
                    let ret_fq2 = if reuse_q2 {
                        Some((fq2, q2_full.into_limbs()))
                    } else {
                        fq2.release_store(self.store);
                        crate::limb::put_limb_buf(q2_full.into_limbs());
                        None
                    };
                    let ret_fp1 = if reuse_p1 {
                        Some(fp1)
                    } else {
                        fp1.release_store(self.store);
                        None
                    };
                    return Ok((res, ret_fq2, ret_fp1));
                }
            }
        }

        // Release `fq2` and `fp1` and fall back via Rung 2 (`mul_whole` handles NTT/ref fallback).
        fq2.release_store(self.store);
        fp1.release_store(self.store);
        self.telemetry
            .record_retry(crate::par::current_core_id(), 0, None);
        let fallback_method = match method {
            MulMethod::Fft { .. } => {
                MulMethod::Ntt(self.fallback_ntt_spec((out_limbs as u64) * 64, true))
            }
            other => other.clone(),
        };
        let mut a1 = q2_full;
        let mut b1 = t1_shifted.unwrap_or_else(|| t1.clone());
        let mut a2 = p1_shifted.unwrap_or_else(|| p1.clone());
        let mut b2 = t2_shifted.unwrap_or_else(|| t2.clone());
        let prod1 = self.mul_whole(point_id, &fallback_method, &mut a1, &mut b1)?;
        let prod2 = self.mul_whole(point_id.wrapping_add(1), &fallback_method, &mut a2, &mut b2)?;
        Ok((
            add_bignum_par(self.kernels, &prod1, &prod2, self.sched, pw),
            None,
            None,
        ))
    }

    fn run_raw_whole(
        &self,
        kernels: &Kernels,
        method: &MulMethod,
        a: &[u64],
        b: &[u64],
    ) -> Result<(LimbVec, u64, f64), crate::store::StoreError> {
        let same = std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len();
        match method {
            MulMethod::Basecase => {
                let limbs = if same {
                    sqr_limbs_basecase(kernels, a)
                } else {
                    mul_limbs_basecase(kernels, a, b)
                };
                let w0 = compute_limbs_w0(kernels, &limbs);
                Ok((limbs, w0, 0.0))
            }
            MulMethod::Karatsuba { basecase_cutoff } => {
                let limbs = if same {
                    sqr_limbs_karatsuba(kernels, a, *basecase_cutoff)
                } else {
                    mul_limbs_karatsuba(kernels, a, b, *basecase_cutoff)
                };
                let w0 = compute_limbs_w0(kernels, &limbs);
                Ok((limbs, w0, 0.0))
            }
            MulMethod::Fft {
                n_complex,
                coeff_bits,
            } => {
                debug_assert_ne!(coeff_bits % 61, 0);
                match fft_mul(kernels, self.fft_cache, a, b, *n_complex, *coeff_bits) {
                    Some((limbs, w0, err)) => Ok((limbs, w0, err)),
                    None => Ok((empty_limb_vec(), 0, 1.0)), // err = 1.0 triggers Rung-2 NTT fallback
                }
            }
            MulMethod::Ntt(spec) => {
                debug_assert_ne!(spec.coeff_bits % 61, 0);
                let out_limbs = a.len() + b.len() + 2;
                let (limbs, w0) = ntt_mul_slices(
                    kernels,
                    self.ntt_cache,
                    self.sched,
                    self.store,
                    self.telemetry,
                    a,
                    b,
                    *spec,
                    out_limbs,
                )?;
                Ok((limbs, w0, 0.0))
            }
            MulMethod::SplitHalving { levels, inner } => {
                let limbs = self.run_split_halving(kernels, *levels, inner, a, b)?;
                let w0 = compute_limbs_w0(kernels, &limbs);
                Ok((limbs, w0, 0.0))
            }
            MulMethod::SplitOuter {
                k_parts,
                part_bits_w,
                inner_ntt,
            } => {
                let limbs =
                    self.run_split_outer(kernels, *k_parts, *part_bits_w, *inner_ntt, a, b)?;
                let w0 = compute_limbs_w0(kernels, &limbs);
                Ok((limbs, w0, 0.0))
            }
        }
    }

    /// Split Form 1: Halving (Karatsuba-style splitting over `inner`, `levels` deep, Section 5).
    fn run_split_halving(
        &self,
        kernels: &Kernels,
        levels: u32,
        inner: &MulMethod,
        a: &[u64],
        b: &[u64],
    ) -> Result<LimbVec, crate::store::StoreError> {
        let mut la = a.len();
        while la > 0 && a[la - 1] == 0 {
            la -= 1;
        }
        let mut lb = b.len();
        while lb > 0 && b[lb - 1] == 0 {
            lb -= 1;
        }
        if la == 0 || lb == 0 {
            return Ok(empty_limb_vec());
        }
        let a = &a[..la];
        let b = &b[..lb];
        if levels == 0 || la.max(lb) <= 4 {
            let (limbs, _, _) = self.run_raw_whole(kernels, inner, a, b)?;
            return Ok(limbs);
        }

        let max_len = la.max(lb);
        let half = max_len.div_ceil(2);
        let (a0, a1) = a.split_at(half.min(la));
        let (b0, b1) = b.split_at(half.min(lb));

        let z0 = self.run_split_halving(kernels, levels - 1, inner, a0, b0)?;
        let z2 = self.run_split_halving(kernels, levels - 1, inner, a1, b1)?;
        let sa = add_limbs(kernels, a0, a1);
        let sb = add_limbs(kernels, b0, b1);
        let mut z1 = self.run_split_halving(kernels, levels - 1, inner, &sa, &sb)?;
        z1 = sub_limbs(kernels, &z1, &z0);
        z1 = sub_limbs(kernels, &z1, &z2);

        let mut out = take_limb_buf_zeroed(la + lb + 2);
        out[..z0.len()].copy_from_slice(&z0);
        add_into_at(kernels, &mut out, half, &z1);
        add_into_at(kernels, &mut out, 2 * half, &z2);
        trim_limbs(&mut out);
        Ok(out)
    }

    /// Split Form 2: Outer transform over `Z / (2^m + 1) Z` with `k` parts and odd-root (`negacyclic`)
    /// inner NTT products of `m`-bit numbers (Section 5).
    fn run_split_outer(
        &self,
        kernels: &Kernels,
        k_parts: usize,
        part_bits_w: u64,
        mut inner_ntt: NttSpec,
        a: &[u64],
        b: &[u64],
    ) -> Result<LimbVec, crate::store::StoreError> {
        debug_assert!(k_parts >= 4 && k_parts.is_power_of_two());
        debug_assert_ne!(part_bits_w % 61, 0);
        inner_ntt.negacyclic = true;
        let m_bits = (inner_ntt.n as u64) * (inner_ntt.coeff_bits as u64);
        debug_assert_ne!(m_bits % 61, 0);
        debug_assert_eq!((2 * m_bits) % (k_parts as u64), 0);

        let same = std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len();
        let mut poly_a = cut_into_parts(a, k_parts, part_bits_w);
        fermat_outer_fwd(kernels, &mut poly_a, m_bits);

        let poly_b = if same {
            poly_a.clone()
        } else {
            let mut pb = cut_into_parts(b, k_parts, part_bits_w);
            fermat_outer_fwd(kernels, &mut pb, m_bits);
            pb
        };

        let modulus = make_fermat_modulus(m_bits);
        let two_m = make_pow2_limbs(m_bits);
        let out_inner_limbs = (m_bits.div_ceil(64) as usize) + 2;

        let mut poly_c: Vec<LimbVec> = (0..k_parts).map(|_| empty_limb_vec()).collect();
        for i in 0..k_parts {
            let xa = &poly_a[i];
            let xb = &poly_b[i];
            if xa.is_empty() || xb.is_empty() {
                poly_c[i] = empty_limb_vec();
            } else if cmp_limbs(xa, &two_m) == std::cmp::Ordering::Equal {
                // xa == 2^m ≡ -1 (mod 2^m + 1)
                poly_c[i] = fermat_neg(kernels, xb, &modulus);
            } else if cmp_limbs(xb, &two_m) == std::cmp::Ordering::Equal {
                poly_c[i] = fermat_neg(kernels, xa, &modulus);
            } else {
                let mut fa =
                    ntt_forward(kernels, self.ntt_cache, self.sched, self.store, xa, inner_ntt)?;
                if same {
                    let res = ntt_combine_and_inverse(
                        kernels,
                        self.ntt_cache,
                        self.sched,
                        self.store,
                        self.telemetry,
                        &mut fa,
                        None,
                        None,
                        out_inner_limbs,
                    );
                    fa.release_store(self.store);
                    poly_c[i] = res?.0;
                } else {
                    let mut fb = ntt_forward(
                        kernels,
                        self.ntt_cache,
                        self.sched,
                        self.store,
                        xb,
                        inner_ntt,
                    )?;
                    let res = ntt_combine_and_inverse(
                        kernels,
                        self.ntt_cache,
                        self.sched,
                        self.store,
                        self.telemetry,
                        &mut fa,
                        Some(NttOperand::Owned(&mut fb)),
                        None,
                        out_inner_limbs,
                    );
                    fa.release_store(self.store);
                    fb.release_store(self.store);
                    poly_c[i] = res?.0;
                }
            }
        }

        fermat_outer_inv(kernels, &mut poly_c, m_bits);

        // Reassemble `sum_{j=0}^{k-1} poly_c[j] * 2^{j * part_bits_w}`!
        let total_out_limbs = a.len() + b.len() + 4;
        let mut out = take_limb_buf_zeroed(total_out_limbs);
        for (j, coeff) in poly_c.iter().enumerate() {
            if coeff.is_empty() {
                continue;
            }
            let bit_off = (j as u64) * part_bits_w;
            let word_off = (bit_off >> 6) as usize;
            let bit_rem = bit_off & 63;
            if bit_rem == 0 {
                add_into_at(kernels, &mut out, word_off, coeff);
            } else {
                let shifted = shl_limbs(coeff, bit_rem);
                add_into_at(kernels, &mut out, word_off, &shifted);
            }
        }
        trim_limbs(&mut out);
        Ok(out)
    }
}

fn cut_into_parts(limbs: &[u64], k_parts: usize, part_bits_w: u64) -> Vec<LimbVec> {
    let total_bits = (limbs.len() as u64) * 64;
    let mut parts = Vec::with_capacity(k_parts);
    for j in 0..k_parts {
        let start = (j as u64) * part_bits_w;
        if start >= total_bits {
            parts.push(empty_limb_vec());
            continue;
        }
        let mut s = shr_limbs(limbs, start);
        let top_word = (part_bits_w / 64) as usize;
        let top_bit = (part_bits_w & 63) as u32;
        if top_word < s.len() {
            if top_bit == 0 {
                s.truncate(top_word);
            } else {
                s[top_word] &= (1u64 << top_bit) - 1;
                s.truncate(top_word + 1);
            }
        }
        trim_limbs(&mut s);
        parts.push(s);
    }
    parts
}

fn make_pow2_limbs(m_bits: u64) -> LimbVec {
    let mut v = take_limb_buf_zeroed((m_bits / 64 + 1) as usize);
    v[(m_bits / 64) as usize] = 1u64 << (m_bits & 63);
    v
}

fn make_fermat_modulus(m_bits: u64) -> LimbVec {
    let mut v = make_pow2_limbs(m_bits);
    v[0] |= 1;
    v
}

#[inline]
fn fermat_add(kernels: &Kernels, a: &[u64], b: &[u64], modulus: &[u64]) -> LimbVec {
    let mut sum = add_limbs(kernels, a, b);
    if cmp_limbs(&sum, modulus) != std::cmp::Ordering::Less {
        sum = sub_limbs(kernels, &sum, modulus);
    }
    sum
}

#[inline]
fn fermat_sub(kernels: &Kernels, a: &[u64], b: &[u64], modulus: &[u64]) -> LimbVec {
    if cmp_limbs(a, b) != std::cmp::Ordering::Less {
        sub_limbs(kernels, a, b)
    } else {
        let diff = sub_limbs(kernels, b, a);
        sub_limbs(kernels, modulus, &diff)
    }
}

#[inline]
fn fermat_neg(kernels: &Kernels, a: &[u64], modulus: &[u64]) -> LimbVec {
    if a.is_empty() {
        empty_limb_vec()
    } else {
        sub_limbs(kernels, modulus, a)
    }
}

/// Multiply `x in [0, 2^m]` by `2^shift_mod_2m` modulo `2^m + 1` using shifts and additions only.
fn fermat_mul_pow2(
    kernels: &Kernels,
    x: &[u64],
    shift_bits: u64,
    m_bits: u64,
    two_m: &[u64],
    modulus: &[u64],
) -> LimbVec {
    if x.is_empty() {
        return empty_limb_vec();
    }
    let s = shift_bits % (2 * m_bits);
    if s == 0 {
        return slice_to_limb_vec(x);
    }
    let neg_half = s >= m_bits;
    let r = s % m_bits;

    // Handle `x == 2^m ≡ -1 (mod 2^m + 1)`.
    if cmp_limbs(x, two_m) == std::cmp::Ordering::Equal {
        let pow_r = make_pow2_limbs(r);
        // -1 * (-1)^{neg_half} * 2^r = (-1)^{1 + neg_half} * 2^r
        return if neg_half {
            pow_r
        } else {
            sub_limbs(kernels, modulus, &pow_r)
        };
    }

    if r == 0 {
        return fermat_neg(kernels, x, modulus);
    }

    // Now `x < 2^m` and `0 < r < m`: `x * 2^r = lo + hi * 2^m ≡ lo - hi (mod 2^m + 1)`.
    let mut lo = shl_limbs(x, r);
    let top_word = (m_bits / 64) as usize;
    let top_bit = (m_bits & 63) as u32;
    if top_word < lo.len() {
        if top_bit == 0 {
            lo.truncate(top_word);
        } else {
            lo[top_word] &= (1u64 << top_bit) - 1;
            lo.truncate(top_word + 1);
        }
    }
    trim_limbs(&mut lo);
    let hi = shr_limbs(x, m_bits - r);

    if !neg_half {
        fermat_sub(kernels, &lo, &hi, modulus)
    } else {
        fermat_sub(kernels, &hi, &lo, modulus)
    }
}

/// Forward outer transform of length `k` over `Z / (2^m + 1) Z` using Cooley-Tukey DIF with
/// bit-reversed root shifts `2^{br(j) * step}` (zero permutations needed!).
fn fermat_outer_fwd(kernels: &Kernels, poly: &mut [LimbVec], m_bits: u64) {
    let k = poly.len();
    let g = (2 * m_bits) / (k as u64);
    let two_m = make_pow2_limbs(m_bits);
    let modulus = make_fermat_modulus(m_bits);

    let mut b_blocks = 1usize;
    let mut half = k >> 1;
    while half >= 1 {
        let s_bits = b_blocks.ilog2();
        let step = k / (2 * b_blocks);
        for j in 0..b_blocks {
            let br_j = if s_bits == 0 {
                0usize
            } else {
                j.reverse_bits() >> (usize::BITS - s_bits)
            };
            let shift = ((br_j * step) as u64) * g;
            let base = j * 2 * half;
            for i in 0..half {
                let u = poly[base + i].clone();
                let v = fermat_mul_pow2(
                    kernels,
                    &poly[base + half + i],
                    shift,
                    m_bits,
                    &two_m,
                    &modulus,
                );
                poly[base + i] = fermat_add(kernels, &u, &v, &modulus);
                poly[base + half + i] = fermat_sub(kernels, &u, &v, &modulus);
            }
        }
        b_blocks <<= 1;
        half >>= 1;
    }
}

/// Inverse outer transform of length `k` over `Z / (2^m + 1) Z` (Gentleman-Sande DIT + `k^{-1}`).
fn fermat_outer_inv(kernels: &Kernels, poly: &mut [LimbVec], m_bits: u64) {
    let k = poly.len();
    let g = (2 * m_bits) / (k as u64);
    let two_m = make_pow2_limbs(m_bits);
    let modulus = make_fermat_modulus(m_bits);

    let mut half = 1usize;
    while half < k {
        let b_blocks = k / (2 * half);
        let s_bits = b_blocks.ilog2();
        let step = k / (2 * b_blocks);
        for j in 0..b_blocks {
            let br_j = if s_bits == 0 {
                0usize
            } else {
                j.reverse_bits() >> (usize::BITS - s_bits)
            };
            let fwd_shift = (((br_j * step) as u64) * g) % (2 * m_bits);
            let inv_shift = if fwd_shift == 0 {
                0
            } else {
                2 * m_bits - fwd_shift
            };
            let base = j * 2 * half;
            for i in 0..half {
                let u = poly[base + i].clone();
                let v = poly[base + half + i].clone();
                poly[base + i] = fermat_add(kernels, &u, &v, &modulus);
                let diff = fermat_sub(kernels, &u, &v, &modulus);
                poly[base + half + i] =
                    fermat_mul_pow2(kernels, &diff, inv_shift, m_bits, &two_m, &modulus);
            }
        }
        half <<= 1;
    }

    // Scale by `k^{-1} ≡ 2^{2m - log2(k)} (mod 2^m + 1)`.
    let log_k = k.ilog2() as u64;
    let inv_k_shift = 2 * m_bits - log_k;
    for coeff in poly.iter_mut() {
        *coeff = fermat_mul_pow2(kernels, coeff, inv_k_shift, m_bits, &two_m, &modulus);
    }
}

impl<'a> MulCtx<'a> {
    /// Execute an exact wrapped NTT product modulo `2^L - 1`, where `L = spec.n * spec.coeff_bits`
    /// (Section 11: "Only the methods that section 5 marks exact may wrap a product or make part
    /// of one"). Returns `(wrapped_limbs_untrimmed_to_L, raw_w0, L_bits)`.
    pub fn mul_wrapped(
        &self,
        point_id: u64,
        mut spec: NttSpec,
        a_limbs: &[u64],
        b_limbs: &[u64],
    ) -> Result<(LimbVec, u64, u64), MulError> {
        debug_assert!(!spec.negacyclic);
        debug_assert_ne!(spec.coeff_bits % 61, 0);
        spec.wrap_cyclic = true;
        let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
        let out_limbs = (l_bits.div_ceil(64) as usize) + 1;

        let (mut limbs, mut w0) = ntt_mul_slices(
            self.kernels,
            self.ntt_cache,
            self.sched,
            self.store,
            self.telemetry,
            a_limbs,
            b_limbs,
            spec,
            out_limbs,
        )
        .map_err(|_| MulError::NeedRollback(point_id))?;
        let mut dummy_err = 0.0;
        if let Some(hook) = self.fault_hook {
            hook.on_product_limbs(point_id, 0, &mut limbs, &mut w0, &mut dummy_err);
        }
        Ok((limbs, w0, l_bits))
    }

    /// Execute an exact wrapped NTT product modulo `2^L - 1` against a cached forward transform.
    pub fn mul_wrapped_with_cached(
        &self,
        point_id: u64,
        a_limbs: &[u64],
        b_cached: &NttCachedFwd,
        par_workers: usize,
    ) -> Result<(LimbVec, u64, u64), MulError> {
        let mut spec = b_cached.spec;
        spec.par_workers = par_workers.max(1);
        spec.wrap_cyclic = true;
        debug_assert!(!spec.negacyclic);
        debug_assert_ne!(spec.coeff_bits % 61, 0);
        let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
        let out_limbs = (l_bits.div_ceil(64) as usize) + 1;

        let (mut limbs, mut w0) = ntt_mul_with_cached_op(
            self.kernels,
            self.ntt_cache,
            self.sched,
            self.store,
            self.telemetry,
            a_limbs,
            NttOperand::Shared(b_cached),
            spec,
            out_limbs,
        )
        .map_err(|_| MulError::NeedRollback(point_id))?;
        let mut dummy_err = 0.0;
        if let Some(hook) = self.fault_hook {
            hook.on_product_limbs(point_id, 0, &mut limbs, &mut w0, &mut dummy_err);
        }
        Ok((limbs, w0, l_bits))
    }

    /// Execute an exact wrapped NTT product modulo `2^L - 1` against a cached forward transform,
    /// completing `b_cached`'s row pass lazily in-place during Sweep 2 when not yet transformed.
    pub fn mul_wrapped_with_cached_reuse(
        &self,
        point_id: u64,
        a_limbs: &[u64],
        b_cached: &mut NttCachedFwd,
        par_workers: usize,
    ) -> Result<(LimbVec, u64, u64), MulError> {
        let mut spec = b_cached.spec;
        spec.par_workers = par_workers.max(1);
        spec.wrap_cyclic = true;
        debug_assert!(!spec.negacyclic);
        debug_assert_ne!(spec.coeff_bits % 61, 0);
        let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
        let out_limbs = (l_bits.div_ceil(64) as usize) + 1;

        let (mut limbs, mut w0) = ntt_mul_with_cached_op(
            self.kernels,
            self.ntt_cache,
            self.sched,
            self.store,
            self.telemetry,
            a_limbs,
            NttOperand::ReuseInPlace(b_cached),
            spec,
            out_limbs,
        )
        .map_err(|_| MulError::NeedRollback(point_id))?;
        let mut dummy_err = 0.0;
        if let Some(hook) = self.fault_hook {
            hook.on_product_limbs(point_id, 0, &mut limbs, &mut w0, &mut dummy_err);
        }
        Ok((limbs, w0, l_bits))
    }

    /// Fast L2/L3/DRAM-resident fused `(T, Q, P)` binary-splitting merge for `Fft` and `Ntt`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_ntt_fused_merge_triple(
        &self,
        method: &MulMethod,
        par_workers: usize,
        ram_transform_budget: u64,
        left_p: &BigNum,
        left_q: &BigNum,
        left_t: &BigNum,
        right_p: &BigNum,
        right_q: &BigNum,
        right_t: &BigNum,
        fuse_q: bool,
        fuse_p: bool,
    ) -> Option<(BigNum, Option<BigNum>, Option<BigNum>)> {
        if self.fault_hook.is_some() {
            return None;
        }
        if let MulMethod::Fft {
            n_complex,
            coeff_bits,
        } = *method
        {
            if left_t.shift != 0
                || left_p.shift != 0
                || right_t.shift != 0
                || right_q.is_zero()
                || left_t.is_zero()
                || left_p.is_zero()
                || right_t.is_zero()
            {
                return None;
            }
            let sign1 = right_q.neg ^ left_t.neg;
            let sign2 = left_p.neg ^ right_t.neg;
            let sub_second = sign1 != sign2;
            let est1 = right_q.total_bits() + left_t.body_bits();
            let est2 = left_p.body_bits() + right_t.body_bits();
            if sub_second && est2 + 2 >= est1 {
                return None;
            }
            let q2_full_len = right_q.limbs.len()
                + ((right_q.shift >> 6) as usize)
                + usize::from((right_q.shift & 63) != 0);
            let t_out_limbs = (q2_full_len + left_t.limbs.len())
                .max(left_p.limbs.len() + right_t.limbs.len())
                + 2;
            let q1_and_out = if fuse_q {
                Some((
                    left_q.limbs.as_slice(),
                    left_q.limbs.len() + q2_full_len + 2,
                ))
            } else {
                None
            };
            let p2_and_out = if fuse_p {
                Some((
                    right_p.limbs.as_slice(),
                    left_p.limbs.len() + right_p.limbs.len() + 2,
                ))
            } else {
                None
            };
            let raw = fft_fused_merge_triple(
                self.kernels,
                self.fft_cache,
                n_complex,
                coeff_bits,
                self.fft_err_bound,
                self.sched.min_l2_share(),
                &right_q.limbs,
                right_q.shift,
                &left_t.limbs,
                &left_p.limbs,
                &right_t.limbs,
                sub_second,
                t_out_limbs,
                q1_and_out,
                p2_and_out,
            )?;
            let q2_w0 = right_q.w0();
            let exp_t_w0 = mod61_add(
                mod61_mul(q2_w0, left_t.w0()),
                mod61_mul(left_p.w0(), right_t.w0()),
            );
            if whole_value_w0(raw.t.1, sign1, 0) != exp_t_w0 {
                crate::limb::put_limb_buf(raw.t.0);
                if let Some((qv, _)) = raw.q {
                    crate::limb::put_limb_buf(qv);
                }
                if let Some((pv, _)) = raw.p {
                    crate::limb::put_limb_buf(pv);
                }
                return None;
            }
            let q_opt = if let Some((qq_limbs, qq_lw0)) = raw.q {
                let exp_q_lw0 = mod61_mul(left_q.limbs_w0(), q2_w0);
                if qq_lw0 != exp_q_lw0 {
                    crate::limb::put_limb_buf(raw.t.0);
                    crate::limb::put_limb_buf(qq_limbs);
                    if let Some((pv, _)) = raw.p {
                        crate::limb::put_limb_buf(pv);
                    }
                    return None;
                }
                let mut q_bn = BigNum::from_limbs_with_w0(qq_limbs, false, left_q.shift, qq_lw0);
                q_bn.strip_zero_bits();
                Some(q_bn)
            } else {
                None
            };
            let p_opt = if let Some((pp_limbs, pp_lw0)) = raw.p {
                let exp_p_lw0 = mod61_mul(right_p.limbs_w0(), left_p.limbs_w0());
                if pp_lw0 != exp_p_lw0 {
                    crate::limb::put_limb_buf(raw.t.0);
                    crate::limb::put_limb_buf(pp_limbs);
                    if let Some(qb) = q_opt {
                        crate::limb::put_limb_buf(qb.into_limbs());
                    }
                    return None;
                }
                Some(BigNum::from_limbs_with_w0(
                    pp_limbs,
                    left_p.neg ^ right_p.neg,
                    0,
                    pp_lw0,
                ))
            } else {
                None
            };
            let t_bn = BigNum {
                limbs: raw.t.0,
                neg: sign1,
                shift: 0,
                chk: DualCheck::new(exp_t_w0),
            };
            return Some((t_bn, q_opt, p_opt));
        }

        let MulMethod::Ntt(mut spec) = *method else {
            return None;
        };
        spec.par_workers = par_workers.max(1);
        let n_out_grids = 1usize + usize::from(fuse_q) + usize::from(fuse_p);
        if !ntt_fused_merge_eligible(
            &spec,
            ram_transform_budget,
            self.ntt_cache.l3_prime_cap_bytes(),
            self.sched.num_workers(),
            n_out_grids,
        ) {
            return None;
        }
        let pw = spec.par_workers;
        let q2_full = right_q.materialize_shift_par(self.kernels, self.sched, pw);
        let t1_sh = if !left_t.is_zero() && left_t.shift != 0 {
            Some(left_t.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let p1_sh = if !left_p.is_zero() && left_p.shift != 0 {
            Some(left_p.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let t2_sh = if !right_t.is_zero() && right_t.shift != 0 {
            Some(right_t.materialize_shift_par(self.kernels, self.sched, pw))
        } else {
            None
        };
        let t1_ref: &BigNum = t1_sh.as_ref().unwrap_or(left_t);
        let p1_ref: &BigNum = p1_sh.as_ref().unwrap_or(left_p);
        let t2_ref: &BigNum = t2_sh.as_ref().unwrap_or(right_t);

        if q2_full.is_zero() || t1_ref.is_zero() || p1_ref.is_zero() || t2_ref.is_zero() {
            crate::limb::put_limb_buf(q2_full.into_limbs());
            return None;
        }
        let sign1 = q2_full.neg ^ t1_ref.neg;
        let sign2 = p1_ref.neg ^ t2_ref.neg;
        let sub_second = sign1 != sign2;
        let est1 = q2_full.body_bits() + t1_ref.body_bits();
        let est2 = p1_ref.body_bits() + t2_ref.body_bits();
        if sub_second && est2 + 2 >= est1 {
            crate::limb::put_limb_buf(q2_full.into_limbs());
            return None;
        }

        let t_out_limbs = (q2_full.limbs.len() + t1_ref.limbs.len())
            .max(p1_ref.limbs.len() + t2_ref.limbs.len())
            + 2;
        let q1_and_out = if fuse_q {
            Some((
                left_q.limbs.as_slice(),
                left_q.limbs.len() + q2_full.limbs.len() + 2,
            ))
        } else {
            None
        };
        let p2_and_out = if fuse_p {
            Some((
                right_p.limbs.as_slice(),
                left_p.limbs.len() + right_p.limbs.len() + 2,
            ))
        } else {
            None
        };

        let raw_opt = ntt_fused_merge_triple(
            self.kernels,
            self.ntt_cache,
            self.sched,
            spec,
            ram_transform_budget,
            &q2_full.limbs,
            &t1_ref.limbs,
            &p1_ref.limbs,
            &t2_ref.limbs,
            sub_second,
            t_out_limbs,
            q1_and_out,
            p2_and_out,
        );
        let q2_w0 = q2_full.w0();
        crate::limb::put_limb_buf(q2_full.into_limbs());
        let raw = raw_opt?;

        let exp_t_w0 = mod61_add(
            mod61_mul(q2_w0, t1_ref.w0()),
            mod61_mul(p1_ref.w0(), t2_ref.w0()),
        );
        if whole_value_w0(raw.t.1, sign1, 0) != exp_t_w0 {
            return None;
        }
        let t_bn = BigNum {
            limbs: raw.t.0,
            neg: sign1,
            shift: 0,
            chk: DualCheck::new(exp_t_w0),
        };

        let q_opt = if let Some((qq_limbs, qq_lw0)) = raw.q {
            let exp_q_lw0 = mod61_mul(left_q.limbs_w0(), q2_w0);
            if qq_lw0 != exp_q_lw0 {
                return None;
            }
            let mut q_bn = BigNum::from_limbs_with_w0(qq_limbs, false, left_q.shift, qq_lw0);
            q_bn.strip_zero_bits_par(self.sched, pw);
            Some(q_bn)
        } else {
            None
        };

        let p_opt = if let Some((pp_limbs, pp_lw0)) = raw.p {
            let exp_p_lw0 = mod61_mul(right_p.limbs_w0(), left_p.limbs_w0());
            if pp_lw0 != exp_p_lw0 {
                return None;
            }
            Some(BigNum::from_limbs_with_w0(
                pp_limbs,
                left_p.neg ^ right_p.neg,
                0,
                pp_lw0,
            ))
        } else {
            None
        };

        Some((t_bn, q_opt, p_opt))
    }

    /// Fast 5-transform `(a1 * b, a2 * b)` sharing `fwd(b)` inside L2/L3 without allocating cached grids.
    pub fn try_mul_pair_shared_b(
        &self,
        method: &MulMethod,
        par_workers: usize,
        ram_transform_budget: u64,
        a1: &BigNum,
        a2: &BigNum,
        b: &BigNum,
    ) -> Option<(BigNum, BigNum)> {
        if self.fault_hook.is_some() || a1.is_zero() || a2.is_zero() || b.is_zero() {
            return None;
        }
        let out1_limbs = a1.limbs.len() + b.limbs.len() + 2;
        let out2_limbs = a2.limbs.len() + b.limbs.len() + 2;
        let ((l1, lw0_1), (l2, lw0_2)) = match *method {
            MulMethod::Fft {
                n_complex,
                coeff_bits,
            } => fft_mul_pair_shared_b(
                self.kernels,
                self.fft_cache,
                n_complex,
                coeff_bits,
                self.fft_err_bound,
                &a1.limbs,
                out1_limbs,
                &a2.limbs,
                out2_limbs,
                &b.limbs,
            )?,
            MulMethod::Ntt(mut spec) => {
                spec.par_workers = par_workers.max(1);
                ntt_mul_pair_shared_b(
                    self.kernels,
                    self.ntt_cache,
                    self.sched,
                    spec,
                    ram_transform_budget,
                    &a1.limbs,
                    out1_limbs,
                    &a2.limbs,
                    out2_limbs,
                    &b.limbs,
                )?
            }
            _ => return None,
        };

        let b_lw0 = b.limbs_w0();
        let exp1 = mod61_mul(a1.limbs_w0(), b_lw0);
        let exp2 = mod61_mul(a2.limbs_w0(), b_lw0);
        if lw0_1 != exp1 || lw0_2 != exp2 {
            crate::limb::put_limb_buf(l1);
            crate::limb::put_limb_buf(l2);
            return None;
        }
        let bn1 = BigNum::from_limbs_with_w0(l1, a1.neg ^ b.neg, a1.shift + b.shift, lw0_1);
        let bn2 = BigNum::from_limbs_with_w0(l2, a2.neg ^ b.neg, a2.shift + b.shift, lw0_2);
        Some((bn1, bn2))
    }

    /// Zero-copy slice version of `try_mul_pair_shared_b` (`(a1 * b, a2 * b)` sharing `fwd(b)`).
    pub fn try_mul_slices_pair_shared_b(
        &self,
        method: &MulMethod,
        par_workers: usize,
        ram_transform_budget: u64,
        a1_limbs: &[u64],
        a2_limbs: &[u64],
        b_limbs: &[u64],
    ) -> Option<(LimbVec, LimbVec)> {
        self.try_mul_slices_pair_shared_b_with_w0(
            method,
            par_workers,
            ram_transform_budget,
            a1_limbs,
            None,
            a2_limbs,
            b_limbs,
            None,
        )
    }

    /// Zero-copy slice `(a1 * b, a2 * b)` with optional precomputed `(a1_w0, b_w0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_mul_slices_pair_shared_b_with_w0(
        &self,
        method: &MulMethod,
        par_workers: usize,
        ram_transform_budget: u64,
        a1_limbs: &[u64],
        a1_w0_opt: Option<u64>,
        a2_limbs: &[u64],
        b_limbs: &[u64],
        b_w0_opt: Option<u64>,
    ) -> Option<(LimbVec, LimbVec)> {
        if self.fault_hook.is_some()
            || a1_limbs.is_empty()
            || a2_limbs.is_empty()
            || b_limbs.is_empty()
        {
            return None;
        }
        let pw = par_workers.max(1);
        let out1_limbs = a1_limbs.len() + b_limbs.len() + 2;
        let out2_limbs = a2_limbs.len() + b_limbs.len() + 2;
        let ((mut l1, lw0_1), (mut l2, lw0_2)) = match *method {
            MulMethod::Fft {
                n_complex,
                coeff_bits,
            } => fft_mul_pair_shared_b(
                self.kernels,
                self.fft_cache,
                n_complex,
                coeff_bits,
                self.fft_err_bound,
                a1_limbs,
                out1_limbs,
                a2_limbs,
                out2_limbs,
                b_limbs,
            )?,
            MulMethod::Ntt(mut spec) => {
                spec.par_workers = pw;
                ntt_mul_pair_shared_b(
                    self.kernels,
                    self.ntt_cache,
                    self.sched,
                    spec,
                    ram_transform_budget,
                    a1_limbs,
                    out1_limbs,
                    a2_limbs,
                    out2_limbs,
                    b_limbs,
                )?
            }
            _ => return None,
        };

        let b_lw0 = b_w0_opt
            .unwrap_or_else(|| compute_limbs_w0_par(self.kernels, self.sched, b_limbs, pw));
        let a1_lw0 = a1_w0_opt
            .unwrap_or_else(|| compute_limbs_w0_par(self.kernels, self.sched, a1_limbs, pw));
        let exp1 = mod61_mul(a1_lw0, b_lw0);
        let exp2 = mod61_mul(
            compute_limbs_w0_par(self.kernels, self.sched, a2_limbs, pw),
            b_lw0,
        );
        if lw0_1 != exp1 || lw0_2 != exp2 {
            crate::limb::put_limb_buf(l1);
            crate::limb::put_limb_buf(l2);
            return None;
        }
        trim_limbs(&mut l1);
        trim_limbs(&mut l2);
        Some((l1, l2))
    }

    /// Zero-copy slice whole product (`a_limbs * b_limbs`) streaming in L2/L3 without cached grids.
    pub fn try_mul_slices(
        &self,
        method: &MulMethod,
        par_workers: usize,
        a_limbs: &[u64],
        b_limbs: &[u64],
    ) -> Option<LimbVec> {
        self.try_mul_slices_with_w0(method, par_workers, a_limbs, None, b_limbs, None)
    }

    /// Zero-copy slice whole product (`a_limbs * b_limbs`) with optional precomputed `(a_w0, b_w0)`.
    pub fn try_mul_slices_with_w0(
        &self,
        method: &MulMethod,
        par_workers: usize,
        a_limbs: &[u64],
        a_w0_opt: Option<u64>,
        b_limbs: &[u64],
        b_w0_opt: Option<u64>,
    ) -> Option<LimbVec> {
        if self.fault_hook.is_some()
            || a_limbs.is_empty()
            || b_limbs.is_empty()
            || matches!(method, MulMethod::Ntt(s) if s.out_of_core)
        {
            return None;
        }
        let pw = par_workers.max(1);
        let mut m = method.clone();
        if let MulMethod::Ntt(ref mut s) = m {
            s.par_workers = pw;
        }
        let (mut prod_limbs, emitted_w0, fft_err) =
            self.run_raw_whole(self.kernels, &m, a_limbs, b_limbs).ok()?;
        if matches!(m, MulMethod::Fft { .. }) && !(fft_err <= self.fft_err_bound) {
            crate::limb::put_limb_buf(prod_limbs);
            return None;
        }
        let a_w0 =
            a_w0_opt.unwrap_or_else(|| compute_limbs_w0_par(self.kernels, self.sched, a_limbs, pw));
        let b_w0 =
            b_w0_opt.unwrap_or_else(|| compute_limbs_w0_par(self.kernels, self.sched, b_limbs, pw));
        let exp_w0 = mod61_mul(a_w0, b_w0);
        if emitted_w0 != exp_w0 {
            crate::limb::put_limb_buf(prod_limbs);
            return None;
        }
        trim_limbs(&mut prod_limbs);
        Some(prod_limbs)
    }
}

/// RAII handle for a `BigNum` (or raw `LimbVec` fraction) parked in an `ExtentDevice` of `Store`
/// until needed (`DESIGN.md` §6 & §10). Automatically frees its extent and decrements
/// `store.parked_live_bytes()` on `Drop` across both normal unparking and Rung-3 rollbacks.
pub struct ParkedBigNum<'a> {
    store: &'a Store,
    ext: Option<crate::store::ExtentHandle>,
    payload_bytes: u64,
    pub limb_len: usize,
    pub top_limb_bits: u32,
    pub neg: bool,
    pub shift: u64,
    pub chk: DualCheck,
}

impl<'a> Drop for ParkedBigNum<'a> {
    fn drop(&mut self) {
        if let Some(h) = self.ext.take() {
            self.store.free(h);
            self.store.note_parked_free(self.payload_bytes);
        }
    }
}

pub struct PreparedParkWrite<'a> {
    pub parked: ParkedBigNum<'a>,
    plan: Option<crate::store::SlotBatchPlan>,
    slot_buf: Option<crate::store::AlignedSlotBuf>,
}

impl<'a> PreparedParkWrite<'a> {
    /// Pack `bn` + 3 piece check words into an `AlignedSlotBuf` and immediately return `bn`'s
    /// `LimbVec` to `NumbersAlloc` so `execute_write` can stream the slots to `Store` on the
    /// `Loader` thread beside compute work (`Ok(None)` iff `ctx.store.is_none()` or `OutOfSpace`).
    pub fn prepare_bignum(
        ctx: &MulCtx<'a>,
        point_id: u64,
        bn: &mut BigNum,
        max_workers: usize,
    ) -> Result<Option<Self>, MulError> {
        let Some(st) = ctx.store else {
            return Ok(None);
        };
        if bn.is_zero() {
            return Ok(Some(Self {
                parked: ParkedBigNum {
                    store: st,
                    ext: None,
                    payload_bytes: 0,
                    limb_len: 0,
                    top_limb_bits: 0,
                    neg: false,
                    shift: 0,
                    chk: bn.chk,
                },
                plan: None,
                slot_buf: None,
            }));
        }
        if !bn.verify_at_rest_par(ctx.kernels, ctx.sched, max_workers, ctx.telemetry) {
            ctx.telemetry
                .record_failure(crate::par::current_core_id(), 0, None);
            return Err(MulError::NeedRollback(point_id));
        }
        let limb_len = bn.limbs.len();
        let top_bits = 64 - bn.limbs[limb_len - 1].leading_zeros();
        let neg = bn.neg;
        let shift = bn.shift;
        let chk = bn.chk;

        let Some((ext, payload_bytes, plan, slot_buf)) =
            pack_limbs_for_park(ctx, st, point_id, &bn.limbs, max_workers)?
        else {
            return Ok(None);
        };
        let old_limbs = crate::limb::take_limb_vec(&mut bn.limbs);
        crate::limb::put_limb_buf(old_limbs);
        *bn = BigNum::zero();

        Ok(Some(Self {
            parked: ParkedBigNum {
                store: st,
                ext: Some(ext),
                payload_bytes,
                limb_len,
                top_limb_bits: top_bits,
                neg,
                shift,
                chk,
            },
            plan: Some(plan),
            slot_buf: Some(slot_buf),
        }))
    }

    /// Variant for raw fixed-point `LimbVec` (preserving exact `limbs.len()` including any top
    /// zero limbs, e.g. `x_right` in `radix`). Consumes and frees `limbs` to `NumbersAlloc` iff
    /// parking succeeds; returns `Err(limbs)` unchanged if `ctx.store.is_none()` or `OutOfSpace`.
    pub fn prepare_raw_limbs(
        ctx: &MulCtx<'a>,
        point_id: u64,
        limbs: LimbVec,
        max_workers: usize,
    ) -> Result<Result<Self, LimbVec>, MulError> {
        let Some(st) = ctx.store else {
            return Ok(Err(limbs));
        };
        let limb_len = limbs.len();
        if limb_len == 0 {
            crate::limb::put_limb_buf(limbs);
            return Ok(Ok(Self {
                parked: ParkedBigNum {
                    store: st,
                    ext: None,
                    payload_bytes: 0,
                    limb_len: 0,
                    top_limb_bits: 0,
                    neg: false,
                    shift: 0,
                    chk: DualCheck::new(0),
                },
                plan: None,
                slot_buf: None,
            }));
        }
        let w0 = compute_limbs_w0_par(ctx.kernels, ctx.sched, &limbs, max_workers);
        let top_bits = 64 - limbs[limb_len - 1].leading_zeros();
        let Some((ext, payload_bytes, plan, slot_buf)) =
            pack_limbs_for_park(ctx, st, point_id, &limbs, max_workers)?
        else {
            return Ok(Err(limbs));
        };
        crate::limb::put_limb_buf(limbs);
        Ok(Ok(Self {
            parked: ParkedBigNum {
                store: st,
                ext: Some(ext),
                payload_bytes,
                limb_len,
                top_limb_bits: top_bits,
                neg: false,
                shift: 0,
                chk: DualCheck::new(w0),
            },
            plan: Some(plan),
            slot_buf: Some(slot_buf),
        }))
    }

    /// Split into `(ParkedBigNum, PreparedWriteJob)` so the caller can store `ParkedBigNum` in its
    /// tree/state while moving `PreparedWriteJob` onto the `Loader` thread (`submit_loader_scoped`).
    pub fn split_write_job(mut self) -> (ParkedBigNum<'a>, PreparedWriteJob) {
        let job = PreparedWriteJob {
            plan: self.plan.take(),
            slot_buf: self.slot_buf.take(),
        };
        (self.parked, job)
    }
}

pub struct PreparedWriteJob {
    plan: Option<crate::store::SlotBatchPlan>,
    slot_buf: Option<crate::store::AlignedSlotBuf>,
}

impl PreparedWriteJob {
    pub fn execute(mut self, st: &Store) -> Result<(), crate::store::StoreError> {
        if let (Some(plan), Some(sbuf)) = (self.plan.take(), self.slot_buf.take()) {
            st.devices[plan.extent.device_idx].write_planned_slots(&plan, &sbuf, &st.bytes_moved)?;
        }
        Ok(())
    }
}

pub struct ParkedReadStage {
    plan: Option<crate::store::SlotBatchPlan>,
    slot_buf: Option<crate::store::AlignedSlotBuf>,
}

impl ParkedReadStage {
    pub fn execute_read(
        &mut self,
        st: &Store,
        telemetry: &Telemetry,
    ) -> Result<(), crate::store::StoreError> {
        if let (Some(plan), Some(sbuf)) = (self.plan.as_ref(), self.slot_buf.as_mut()) {
            st.devices[plan.extent.device_idx].read_planned_slots(
                plan,
                sbuf,
                telemetry,
                &st.bytes_moved,
            )?;
        }
        Ok(())
    }
    pub fn execute_read_owned(
        mut self,
        st: &Store,
        telemetry: &Telemetry,
    ) -> Result<Self, crate::store::StoreError> {
        self.execute_read(st, telemetry)?;
        Ok(self)
    }
}

impl<'a> ParkedBigNum<'a> {
    #[inline]
    pub fn body_bits(&self) -> u64 {
        if self.limb_len == 0 {
            0
        } else {
            ((self.limb_len - 1) as u64) * 64 + (self.top_limb_bits as u64)
        }
    }

    #[inline]
    pub fn extent_handle(&self) -> Option<crate::store::ExtentHandle> {
        self.ext
    }

    pub fn prepare_read(&self, point_id: u64) -> Result<ParkedReadStage, MulError> {
        let Some(ext) = self.ext else {
            return Ok(ParkedReadStage {
                plan: None,
                slot_buf: None,
            });
        };
        let dev = &self.store.devices[ext.device_idx];
        let range = [crate::store::ExtentRange {
            piece_start: 0,
            piece_count: ext.num_pieces,
            host_limb_offset: 0,
        }];
        let plan = dev
            .plan_slot_batch(&ext, &range)
            .map_err(|_| MulError::NeedRollback(point_id))?;
        let slot_buf = crate::store::AlignedSlotBuf::new_uninit(plan.required_buf_bytes);
        Ok(ParkedReadStage {
            plan: Some(plan),
            slot_buf: Some(slot_buf),
        })
    }

    pub fn finish_unpark_bignum(
        mut self,
        ctx: &MulCtx<'_>,
        stage: ParkedReadStage,
        point_id: u64,
        max_workers: usize,
    ) -> Result<BigNum, MulError> {
        let neg = self.neg;
        let shift = self.shift;
        let chk = self.chk;
        let limbs = self.finish_unpark_raw_inner(ctx, stage, point_id, max_workers)?;
        if limbs.is_empty() {
            return Ok(BigNum::zero());
        }
        let mut bn = BigNum {
            limbs,
            neg,
            shift,
            chk,
        };
        if !bn.verify_at_rest_par(ctx.kernels, ctx.sched, max_workers, ctx.telemetry) {
            crate::limb::put_limb_buf(bn.into_limbs());
            ctx.telemetry
                .record_failure(crate::par::current_core_id(), 0, None);
            return Err(MulError::NeedRollback(point_id));
        }
        Ok(bn)
    }

    pub fn finish_unpark_raw_limbs(
        mut self,
        ctx: &MulCtx<'_>,
        stage: ParkedReadStage,
        point_id: u64,
        max_workers: usize,
    ) -> Result<LimbVec, MulError> {
        let mut chk = self.chk;
        let limbs = self.finish_unpark_raw_inner(ctx, stage, point_id, max_workers)?;
        if !limbs.is_empty() {
            let actual_w0 = compute_limbs_w0_par(ctx.kernels, ctx.sched, &limbs, max_workers);
            if !chk.verify_or_repair_at_rest(actual_w0, ctx.telemetry) {
                crate::limb::put_limb_buf(limbs);
                ctx.telemetry
                    .record_failure(crate::par::current_core_id(), 0, None);
                return Err(MulError::NeedRollback(point_id));
            }
        }
        Ok(limbs)
    }

    pub fn unpark_bignum_sync(
        self,
        ctx: &MulCtx<'_>,
        point_id: u64,
        max_workers: usize,
    ) -> Result<BigNum, MulError> {
        let st = self.store;
        let mut stage = self.prepare_read(point_id)?;
        if stage.plan.is_some() {
            let stage_ref = &mut stage;
            st.submit_loader_scoped(|| stage_ref.execute_read(st, ctx.telemetry))
                .wait()
                .map_err(|_| MulError::NeedRollback(point_id))?;
        }
        self.finish_unpark_bignum(ctx, stage, point_id, max_workers)
    }

    fn finish_unpark_raw_inner(
        &mut self,
        ctx: &MulCtx<'_>,
        mut stage: ParkedReadStage,
        point_id: u64,
        max_workers: usize,
    ) -> Result<LimbVec, MulError> {
        let Some(ext) = self.ext.take() else {
            return Ok(empty_limb_vec());
        };
        let st = self.store;
        let payload_bytes = self.payload_bytes;
        let limb_len = self.limb_len;
        let (Some(plan), Some(mut slot_buf)) = (stage.plan.take(), stage.slot_buf.take()) else {
            st.free(ext);
            st.note_parked_free(payload_bytes);
            return Err(MulError::NeedRollback(point_id));
        };

        let dev = &st.devices[plan.extent.device_idx];
        let slot = plan.piece_slot_bytes;
        let p_limbs = plan.piece_limbs;
        let mut out = crate::limb::take_limb_buf_uninit(limb_len);
        unsafe {
            out.set_len(limb_len);
        }

        let scrub_res: Result<(), crate::store::StoreError> =
            if max_workers > 1 && plan.pieces.len() >= 4 {
                let base_addr = slot_buf.as_mut_ptr() as usize;
                let dst_addr = out.as_mut_ptr() as usize;
                let err_flag = std::sync::atomic::AtomicBool::new(false);
                let first_err: std::sync::Mutex<Option<crate::store::StoreError>> =
                    std::sync::Mutex::new(None);
                ctx.sched.par_for_max(plan.pieces.len(), 1, max_workers, |idx| {
                    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    let pe = &plan.pieces[idx];
                    let slot_mut = unsafe {
                        std::slice::from_raw_parts_mut((base_addr + pe.buf_byte_off) as *mut u8, slot)
                    };
                    if let Err(e) = dev.scrub_or_repair_slot_in_place(
                        &ext,
                        pe.piece_id,
                        slot_mut,
                        ctx.kernels,
                        ctx.telemetry,
                        &st.bytes_moved,
                        false,
                    ) {
                        err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        let mut g = first_err.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                        return;
                    }
                    let copy_l = limb_len.saturating_sub(pe.host_limb_offset).min(p_limbs);
                    if copy_l > 0 {
                        unsafe {
                            copy_slot_bytes_to_u64(
                                slot_mut.as_ptr(),
                                (dst_addr as *mut u64).add(pe.host_limb_offset),
                                copy_l,
                            );
                        }
                    }
                });
                match first_err.into_inner().unwrap() {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            } else {
                let raw_bytes = unsafe { slot_buf.as_import_slice_mut() };
                let mut res = Ok(());
                for pe in &plan.pieces {
                    let slot_mut = &mut raw_bytes[pe.buf_byte_off..(pe.buf_byte_off + slot)];
                    if let Err(e) = dev.scrub_or_repair_slot_in_place(
                        &ext,
                        pe.piece_id,
                        slot_mut,
                        ctx.kernels,
                        ctx.telemetry,
                        &st.bytes_moved,
                        false,
                    ) {
                        res = Err(e);
                        break;
                    }
                    let copy_l = limb_len.saturating_sub(pe.host_limb_offset).min(p_limbs);
                    if copy_l > 0 {
                        unsafe {
                            copy_slot_bytes_to_u64(
                                slot_mut.as_ptr(),
                                out.as_mut_ptr().add(pe.host_limb_offset),
                                copy_l,
                            );
                        }
                    }
                }
                res
            };

        st.free(ext);
        st.note_parked_free(payload_bytes);
        drop(slot_buf);

        if scrub_res.is_err() {
            crate::limb::put_limb_buf(out);
            let rung = ctx.ladder.claim_rung(point_id, Rung::GoBack);
            return match rung {
                Rung::GoBack => Err(MulError::NeedRollback(point_id)),
                _ => Err(MulError::FatalStop(
                    "Parked number piece failed beyond rollback ladder".to_string(),
                )),
            };
        }
        Ok(out)
    }
}

fn pack_limbs_for_park(
    ctx: &MulCtx<'_>,
    st: &Store,
    point_id: u64,
    src_limbs: &[u64],
    max_workers: usize,
) -> Result<
    Option<(
        crate::store::ExtentHandle,
        u64,
        crate::store::SlotBatchPlan,
        crate::store::AlignedSlotBuf,
    )>,
    MulError,
> {
    let payload_bytes = (src_limbs.len() * 8) as u64;
    let ext = match st.allocate(payload_bytes) {
        Ok(h) => h,
        Err(crate::store::StoreError::OutOfSpace) => return Ok(None),
        Err(_) => return Err(MulError::NeedRollback(point_id)),
    };
    let dev = &st.devices[ext.device_idx];
    let range = [crate::store::ExtentRange {
        piece_start: 0,
        piece_count: ext.num_pieces,
        host_limb_offset: 0,
    }];
    let plan = match dev.plan_slot_batch(&ext, &range) {
        Ok(p) => p,
        Err(_) => {
            st.free(ext);
            return Err(MulError::NeedRollback(point_id));
        }
    };
    st.note_parked_alloc(payload_bytes);

    let mut slot_buf = crate::store::AlignedSlotBuf::new_uninit(plan.required_buf_bytes);
    let slot = plan.piece_slot_bytes;
    let p_limbs = plan.piece_limbs;
    let base_addr = slot_buf.as_mut_ptr() as usize;

    let pack_one = |pe: &crate::store::SlotPieceEntry| {
        let avail = src_limbs.len().saturating_sub(pe.host_limb_offset).min(p_limbs);
        let dst_slot =
            unsafe { std::slice::from_raw_parts_mut((base_addr + pe.buf_byte_off) as *mut u8, slot) };
        unsafe {
            if avail > 0 {
                copy_u64_to_slot_bytes(
                    src_limbs.as_ptr().add(pe.host_limb_offset),
                    dst_slot.as_mut_ptr(),
                    avail,
                );
            }
            if avail < p_limbs {
                std::ptr::write_bytes(dst_slot.as_mut_ptr().add(avail * 8), 0, (p_limbs - avail) * 8);
            }
        }
        dev.compute_slot_check_in_place(dst_slot, ctx.kernels, false);
    };

    if max_workers > 1 && plan.pieces.len() >= 4 {
        ctx.sched.par_for_max(plan.pieces.len(), 1, max_workers, |idx| {
            pack_one(&plan.pieces[idx]);
        });
    } else {
        for pe in &plan.pieces {
            pack_one(pe);
        }
    }

    Ok(Some((ext, payload_bytes, plan, slot_buf)))
}

#[inline]
unsafe fn copy_u64_to_slot_bytes(src: *const u64, dst_bytes: *mut u8, count: usize) {
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst_bytes, count * 8);
    }
    #[cfg(not(target_endian = "little"))]
    unsafe {
        for i in 0..count {
            let w = *src.add(i);
            std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), dst_bytes.add(i * 8), 8);
        }
    }
}

#[inline]
unsafe fn copy_slot_bytes_to_u64(src_bytes: *const u8, dst: *mut u64, count: usize) {
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(src_bytes, dst as *mut u8, count * 8);
    }
    #[cfg(not(target_endian = "little"))]
    unsafe {
        for i in 0..count {
            let b = std::slice::from_raw_parts(src_bytes.add(i * 8), 8);
            *dst.add(i) = u64::from_le_bytes(b.try_into().unwrap());
        }
    }
}
