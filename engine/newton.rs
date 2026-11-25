//! Contract:
//! - Inputs: Root Chudnovsky `(Q, T)` from `splitting`, `Plan` (`target_frac_bits`), `Shares`,
//!   `Calibration`, `MulCtx`, and optional `NewtonResumeState` + checkpoint callback.
//! - Outputs: Fixed-point fractional limbs of `pi - 3` (`pi_frac` of `out_frac_limbs` 64-bit limbs
//!   representing `floor((pi - 3) * 2^{64 * out_frac_limbs})`) verified to have integer part `3`.
//! - Data format: `A` normalized to `[0.5, 1.0) * 2^{64 * n_a}`; step iterates `Z_m` and `U_m` as
//!   `m`-limb unsigned fixed-point fractions in `(0, 1) * 2^{64 * m}`.
//! - Work & bytes moved: One coupled Newton iteration `z = 1 / (sqrt(10005) * A)` (`u = A*z`,
//!   `rho = 1 - 10005*u^2`, `z += z*rho/2`, `u += u*rho/2`), wrapping `A*z` 1-2 limbs above step
//!   length and `10005*u^2` at `1.5x` step length when transform-sized, sharing `fwd(z)` and
//!   `fwd(rho)`, parking `q_root` into `Store` on `Loader` beside the iterations and prefetching it
//!   beside the final step, followed by one whole product `4270934400 * Q_odd * z` (`~24` pi-length
//!   transforms total).

use std::alloc::Allocator;
use std::time::Instant;

use crate::check::{
    DualCheck, compute_limbs_w0, compute_limbs_w0_par, whole_value_w0,
    wrapped_product_expected_w0,
};
use crate::limb::{
    BigNum, LimbVec, add_bignum_par, add_limbs_par, cmp_limbs, empty_limb_vec, mul_bignum_u64_par,
    mul_limbs_u64_par, shl_limbs, shl_limbs_par, shr_limbs, shr_limbs_par, slice_to_limb_vec,
    sub_limbs, sub_limbs_par, take_limb_vec, trim_limbs,
};
use crate::mul::{CachedTransform, MulCtx, MulError, MulMethod};
use crate::ntt::{NttCachedFwd, NttOperand, ntt_combine_and_inverse};
use crate::plan::{Calibration, Plan, choose_wrapped_ntt_spec};
use crate::refk::{mod61_add, mod61_mul, mod61_pow2, mod61_reduce, mod61_sub};
use crate::splitting::CHUD_A;
use crate::tier::Shares;

pub const FINISH_C_MUL: u64 = 4_270_934_400; // 426880 * 10005
pub const FINISH_D_SQR: u64 = 10_005;

#[derive(Clone, Debug)]
pub struct NewtonResumeState {
    pub step_limbs: usize,
    pub acc_bits: u64,
    pub z: BigNum,
    pub u: BigNum,
}

pub trait NewtonCheckpointSink: Send + Sync {
    fn should_checkpoint(&self, _work_sec_since_last: f64) -> bool {
        true
    }

    fn maybe_checkpoint_newton(
        &self,
        state: &NewtonResumeState,
        q_root: &BigNum,
        t_root: &BigNum,
        work_sec_since_last: f64,
    ) -> Result<bool, MulError>;
}

/// A number of the finish that gains limbs at its low end from step to step.
///
/// Its buffer is taken once, at the length the number ends with, and the number lies at the top
/// of it: a step writes its new limbs below the old ones, which stay where they are. A number
/// extended the usual way, at the low end of a buffer of its own, is moved whole at every step,
/// which over the steps is twice its final length read and four times written.
struct GrowsDown {
    buf: LimbVec,
    len: usize,
}

impl GrowsDown {
    fn new(full_len: usize, limbs: LimbVec) -> Self {
        let full_len = full_len.max(limbs.len());
        let len = limbs.len();
        if len == full_len {
            return Self { buf: limbs, len };
        }
        let mut buf = crate::limb::take_limb_buf_uninit(full_len);
        unsafe {
            buf.set_len(full_len);
        }
        buf[(full_len - len)..].copy_from_slice(&limbs);
        crate::limb::put_limb_buf(limbs);
        Self { buf, len }
    }

    fn limbs(&self) -> &[u64] {
        &self.buf[(self.buf.len() - self.len)..]
    }

    /// The number with room below it up to `new_len` limbs, as one slice. The limbs below the
    /// old ones are not yet written: whoever asks writes them all before anything reads them.
    fn grown(&mut self, new_len: usize) -> &mut [u64] {
        assert!(new_len >= self.len && new_len <= self.buf.len());
        self.len = new_len;
        let at = self.buf.len() - new_len;
        &mut self.buf[at..]
    }

    fn into_vec(self) -> LimbVec {
        if self.len == self.buf.len() {
            return self.buf;
        }
        let limbs = slice_to_limb_vec(self.limbs());
        crate::limb::put_limb_buf(self.buf);
        limbs
    }
}

/// Run the coupled single-iteration finish (Section 3) to produce fixed-point `pi`.
#[allow(clippy::too_many_arguments)]
pub fn run_finish(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    q_root: &mut BigNum,
    t_root: &mut BigNum,
    resume: Option<&NewtonResumeState>,
    ckpt_sink: Option<&dyn NewtonCheckpointSink>,
) -> Result<(u64, BigNum, usize), MulError> {
    let do_trace = std::env::var_os("SUPI_TRACE").is_some();
    let t_f0 = Instant::now();
    let max_w = shares.active_workers.max(1);
    // Q and T are scrubbed against their check words here, where they are used without a
    // product to check them. The pass that makes A of both in one go reads every limb of
    // them, so it takes their words on the way and they are held to them right after it;
    // the other ways of making A have them read for it first.
    let scrub = |q_root: &mut BigNum, t_root: &mut BigNum| -> bool {
        q_root.verify_at_rest_par(ctx.kernels, ctx.sched, max_w, ctx.telemetry)
            && t_root.verify_at_rest_par(ctx.kernels, ctx.sched, max_w, ctx.telemetry)
    };

    // 1. Form `A = 13591409 * Q + T > 0` (`mul_1` on odd Q before materializing zero shift!).
    let can_drop_t_early = ckpt_sink.is_none()
        || (plan.checkpoint_mult > 0.0
            && (plan.target_frac_bits as f64) * 3.0e-9
                < plan.checkpoint_mult * plan.est_checkpoint_write_sec);
    let a_norm = if !q_root.neg
        && !q_root.is_zero()
        && t_root.neg
        && t_root.shift == 0
        && q_root.limbs.len() >= 4096
    {
        if let Some((a_vec, s_n, q_limbs_w0, t_limbs_w0)) = try_build_a_norm_fused_par(
            ctx.kernels,
            &q_root.limbs,
            q_root.shift,
            &mut t_root.limbs,
            can_drop_t_early,
            CHUD_A,
            ctx.sched,
            max_w,
        ) {
            let q_actual = whole_value_w0(q_limbs_w0, q_root.neg, q_root.shift);
            let t_actual = whole_value_w0(t_limbs_w0, t_root.neg, t_root.shift);
            if !q_root.chk.verify_or_repair_at_rest(q_actual, ctx.telemetry)
                || !t_root.chk.verify_or_repair_at_rest(t_actual, ctx.telemetry)
            {
                crate::limb::put_limb_buf(a_vec);
                return Err(MulError::NeedRollback(0x3F00_0001));
            }
            (a_vec, s_n)
        } else {
            if !scrub(q_root, t_root) {
                return Err(MulError::NeedRollback(0x3F00_0002));
            }
            let mut a_vec = mul_u64_shl_limbs_par(
                ctx.kernels,
                &q_root.limbs,
                CHUD_A,
                q_root.shift,
                ctx.sched,
                max_w,
            );
            sub_limbs_in_place_par(ctx.kernels, &mut a_vec, &t_root.limbs, ctx.sched, max_w);
            if can_drop_t_early {
                crate::limb::put_limb_buf(take_limb_vec(&mut t_root.limbs));
            }
            let top_limb = *a_vec.last().unwrap();
            let s_n = top_limb.leading_zeros() as u64;
            shl_sublimb_in_place_par(&mut a_vec, s_n, ctx.sched, max_w);
            (a_vec, s_n)
        }
    } else {
        if !scrub(q_root, t_root) {
            return Err(MulError::NeedRollback(0x3F00_0003));
        }
        let aq = mul_bignum_u64_par(ctx.kernels, q_root, CHUD_A, ctx.sched, max_w);
        let a_bn = add_bignum_par(ctx.kernels, &aq, t_root, ctx.sched, max_w);
        crate::limb::put_limb_buf(aq.into_limbs());
        if can_drop_t_early {
            crate::limb::put_limb_buf(take_limb_vec(&mut t_root.limbs));
        }
        debug_assert!(!a_bn.neg && !a_bn.is_zero());
        let top_limb = *a_bn.limbs.last().unwrap();
        let s_n = top_limb.leading_zeros() as u64;
        let mut a_vec = a_bn.into_limbs();
        shl_sublimb_in_place_par(&mut a_vec, s_n, ctx.sched, max_w);
        (a_vec, s_n)
    };
    let (a_norm, s_norm) = a_norm;
    let setup_ms = t_f0.elapsed().as_secs_f64() * 1e3;
    let n_a = a_norm.len();
    debug_assert!(n_a >= 1 && (a_norm[n_a - 1] >> 63) == 1);

    let out_frac_limbs = (plan.target_frac_bits.div_ceil(64) as usize) + 2;
    let final_m = out_frac_limbs + 2;
    let target_acc_bits = (final_m as u64) * 64 - 32;

    // 2. Seed `z` and `u` in double precision (`m = 1` limb = 64 fractional bits, ~46 bits accurate)
    //    or restore from `NewtonResumeState`!
    let (mut cur_m, mut acc_bits, z_limbs, mut u_limbs) = if let Some(st) = resume {
        let mut z_chk = st.z.clone();
        let mut u_chk = st.u.clone();
        if !z_chk.verify_at_rest(ctx.kernels, ctx.telemetry)
            || !u_chk.verify_at_rest(ctx.kernels, ctx.telemetry)
        {
            return Err(MulError::NeedRollback(0x3F00_0004));
        }
        let mut zv = z_chk.into_limbs();
        zv.resize(st.step_limbs, 0);
        let mut uv = u_chk.into_limbs();
        uv.resize(st.step_limbs, 0);
        (st.step_limbs, st.acc_bits, zv, uv)
    } else {
        let hi = a_norm[n_a - 1] as f64;
        let lo = if n_a >= 2 { a_norm[n_a - 2] as f64 } else { 0.0 };
        let a_tilde_f64 = hi * 2.0f64.powi(-64) + lo * 2.0f64.powi(-128);
        let inv_sqrt_d = 1.0 / (FINISH_D_SQR as f64).sqrt();
        let z0_f64 = inv_sqrt_d / a_tilde_f64;
        let scale64 = 2.0f64.powi(64);
        let z0_u64 = (z0_f64 * scale64).round() as u64;
        let u0_u64 = (inv_sqrt_d * scale64).round() as u64;
        (
            1usize,
            44u64,
            slice_to_limb_vec(&[z0_u64]),
            slice_to_limb_vec(&[u0_u64]),
        )
    };

    // The check words of z and of u go with them from step to step: a step's corrections give
    // the words of what they add, and the next step takes its sources' words from here, not
    // from another pass over the limbs. (A resumed run takes them from the limbs once.) The
    // word of the slice of A grows with the slice, from the limbs each step adds.
    let (mut z_w0, mut u_w0): (Option<u64>, Option<u64>) = if resume.is_some() {
        (None, None)
    } else {
        (
            Some(compute_limbs_w0(ctx.kernels, &z_limbs)),
            Some(compute_limbs_w0(ctx.kernels, &u_limbs)),
        )
    };
    // z grows downward in one buffer of its final length (`GrowsDown`).
    let mut z = GrowsDown::new(final_m, z_limbs);
    let mut a_top: Option<(usize, u64)> = None;

    // Build schedule of step precisions `(step_m, next_acc_bits)` backward from `target_acc_bits`
    // so the top iterations land on exact 50%, 25%, 12.5%, ... of `target_acc_bits`!
    let mut target_accs: Vec<u64> = Vec::new();
    let mut cur_t = target_acc_bits;
    while cur_t > acc_bits {
        target_accs.push(cur_t);
        cur_t = (cur_t + 4).div_ceil(2);
    }
    target_accs.reverse();
    let mut schedule: Vec<(usize, u64)> = Vec::with_capacity(target_accs.len());
    let mut prev_m = cur_m;
    for &next_acc in &target_accs {
        let step_m = ((next_acc + 64).div_ceil(64) as usize)
            .min(final_m)
            .max((prev_m + 1).min(final_m));
        schedule.push((step_m, next_acc));
        prev_m = step_m;
    }

    let mut last_ckpt = Instant::now();
    let num_steps = schedule.len();

    // Section 6 & 10: in out-of-core runs, park `q_root` (and `t_root` when kept for checkpoints)
    // into `Store` on `supi-loader` beside the Newton iterations, and prefetch `q_root` back
    // beside the final iteration (`is_last`) right before `4270934400 * Q * z`!
    let mut q_parked: Option<crate::mul::ParkedBigNum<'_>> = None;
    let mut q_park_write_h: Option<
        crate::store::LoaderScopedHandle<'_, Result<(), crate::store::StoreError>>,
    > = None;
    let mut q_prefetch_h: Option<
        crate::store::LoaderScopedHandle<
            '_,
            Result<crate::mul::ParkedReadStage, crate::store::StoreError>,
        >,
    > = None;
    if plan.out_of_core
        && let Some(st) = ctx.store
        && !q_root.is_zero()
        && num_steps >= 1
    {
        if let Some(prep) =
            crate::mul::PreparedParkWrite::prepare_bignum(ctx, 0x3F00_0010, q_root, max_w)?
        {
            let (parked, wjob) = prep.split_write_job();
            q_parked = Some(parked);
            q_park_write_h = Some(st.submit_loader_scoped(move || wjob.execute(st)));
        }
    }

    for (step_idx, &(step_m, next_acc)) in schedule.iter().enumerate() {
        let t_step0 = Instant::now();
        let is_last = step_idx + 1 == num_steps;
        let pid_base = 0x3000_0000u64 | ((step_m as u64) << 16);

        if is_last {
            if let Some(wh) = q_park_write_h.take() {
                wh.wait().map_err(|_| MulError::NeedRollback(0x3F00_0011))?;
            }
            if let (Some(st), Some(qp)) = (ctx.store, q_parked.as_ref()) {
                let rstage = qp.prepare_read(0x3F00_0012)?;
                q_prefetch_h = Some(
                    st.submit_loader_scoped(move || rstage.execute_read_owned(st, ctx.telemetry)),
                );
            }
        }

        // Slice top `k_a` limbs of `a_norm` (`tilde_A_slice = a_slice * 2^{-64 * k_a}`).
        let k_a = (step_m + 2).min(n_a);
        let a_slice = &a_norm[(n_a - k_a)..n_a];
        let a_from = n_a - k_a;
        let a_slice_w0 = match a_top {
            Some((from, w0)) if a_from <= from => add_block_w0_at(
                compute_limbs_w0_par(ctx.kernels, ctx.sched, &a_norm[a_from..from], max_w),
                w0,
                from - a_from,
            ),
            _ => compute_limbs_w0_par(ctx.kernels, ctx.sched, a_slice, max_w),
        };
        a_top = Some((a_from, a_slice_w0));

        // --------------------------------------------------------------------
        // Step A: Compute `U_raw = floor(tilde_A_slice * tilde_z * 2^{64 * step_m})` (`step_m` limbs)
        // using wrapped NTT (`m + 6` limbs) when `step_m` is large and overlap >= 24 bits,
        // optionally retaining `fwd(z_limbs)` for Step C when within transform RAM budget!
        // --------------------------------------------------------------------
        let prod_total_limbs = k_a + cur_m;
        let shift_down_limbs = prod_total_limbs - step_m; // we want `P_AZ >> (64 * shift_down_limbs)`
        let (u_step, mut cached_z_opt, cached_z_w0, u_step_w0) = compute_az_step(
            ctx,
            plan,
            shares,
            cal,
            pid_base,
            a_slice,
            a_slice_w0,
            z.limbs(),
            z_w0,
            &u_limbs,
            u_w0,
            cur_m,
            step_m,
            acc_bits,
            shift_down_limbs,
            is_last,
        )?;
        crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));

        // --------------------------------------------------------------------
        // Step B: Compute `rho = 1 - 10005 * u_step^2` using wrapped NTT at ~1.5x step length!
        // `u_step` represents `U * 2^{-64 * step_m}`, so `10005 * U^2` is near `2^{128 * step_m}`
        // within `< 2^{128 * step_m - acc_bits + 4}`!
        // --------------------------------------------------------------------
        let rho_res = compute_rho_step(
            ctx,
            plan,
            shares,
            cal,
            pid_base + 16,
            &u_step,
            u_step_w0,
            step_m,
            acc_bits,
        );
        let (rho_neg, rho_limbs, rho_frac_limbs, cached_rho_w0) = match rho_res {
            Ok(v) => v,
            Err(e) => {
                if let Some(mut fz) = cached_z_opt {
                    fz.release_store(ctx.store);
                }
                return Err(e);
            }
        };
        let mut u_step = u_step;
        let will_ckpt_last =
            ckpt_sink.is_some_and(|s| s.should_checkpoint(last_ckpt.elapsed().as_secs_f64() + 15.0));
        if is_last && !will_ckpt_last {
            crate::limb::put_limb_buf(take_limb_vec(&mut u_step));
        }

        // --------------------------------------------------------------------
        // Step C: Update `z <- z + z * rho / 2` and (if `!is_last`) `u <- u + u * rho / 2`,
        // sharing `fwd(z)` from Step A and `fwd(rho)` across both products (Section 3)!
        // --------------------------------------------------------------------
        // z with room for the limbs this step adds, `step_m - cur_m` of them below the old
        // ones. The correction writes them; until then only the old limbs are read.
        let room = step_m - cur_m;
        let z_ext: &mut [u64] = z.grown(step_m);

        let base_corr_limbs = ((step_m * 64).saturating_sub((acc_bits as usize).saturating_sub(16)))
            .div_ceil(64)
            .clamp(2, step_m);
        let corr_limbs = if cached_z_opt.is_some() && is_last && cur_m <= step_m {
            cur_m
        } else {
            base_corr_limbs
        };

        // A small step has the room zeroed first and its correction applied the general way.
        // Otherwise only the limbs the correction's product reads below the old z are zeroed,
        // one as a rule; the correction then writes the whole room, those limbs too.
        let room_zeroed = room < 32_768;
        if room_zeroed {
            z_ext[..room].fill(0);
        } else {
            let reach_below = corr_limbs.saturating_sub(cur_m).min(room);
            z_ext[(room - reach_below)..room].fill(0);
        }

        // The words of the corrections to z and to u, from the passes that apply them.
        let z_w0_step = cached_z_w0.or(z_w0);
        let mut dz_w0: Option<u64> = rho_limbs.is_empty().then_some(0);
        let mut du_w0: Option<u64> = rho_limbs.is_empty().then_some(0);

        if !rho_limbs.is_empty() {
            let corr_method = plan.select_mul_method(
                (corr_limbs as u64) * 64,
                (rho_limbs.len() as u64) * 64,
                false,
                false,
                shares.active_workers,
                shares,
                cal,
            );
            let r_shift_bits = 1u64 + 64u64 * ((corr_limbs + rho_frac_limbs - step_m) as u64);

            // Fast path: reuse `f_z` from Step A whenever `corr_limbs == cur_m` (`z_top == z_limbs`)
            // and the exact whole product fits strictly inside `f_z`'s modulus `< 2^L` without
            // widening the NTT transform size!
            let mut handled_with_fz = false;
            if let Some(mut fz) = cached_z_opt.take() {
                let l_fz = (fz.spec.n as u64) * (fz.spec.coeff_bits as u64);
                let need_whole_bits = ((corr_limbs + rho_limbs.len() + 2) as u64) * 64;
                let size_ok = match corr_method {
                    MulMethod::Ntt(cs) => {
                        let fz_work = fz.spec.n.saturating_mul(fz.spec.k_primes);
                        let cs_work = cs.n.saturating_mul(cs.k_primes);
                        if is_last {
                            2 * fz_work <= 3 * cs_work
                        } else {
                            4 * fz_work <= 5 * cs_work
                        }
                    }
                    _ => false,
                };
                if corr_limbs == cur_m && need_whole_bits <= l_fz && size_ok {
                    let rho_w0 = cached_rho_w0.unwrap_or_else(|| {
                        compute_limbs_w0_par(ctx.kernels, ctx.sched, &rho_limbs, max_w)
                    });
                    let z_w0 = cached_z_w0.unwrap_or_else(|| {
                        compute_limbs_w0_par(ctx.kernels, ctx.sched, &z_ext[room..], max_w)
                    });
                    let exp_zr_w0 = mod61_mul(z_w0, rho_w0);
                    let out_cap = corr_limbs + rho_limbs.len() + 2;
                    if is_last {
                        let zr_wrap =
                            ctx.mul_wrapped_with_cached(pid_base + 32, &rho_limbs, &fz, max_w);
                        fz.release_store(ctx.store);
                        let (mut zr_prod, zr_w0, _) = zr_wrap?;
                        trim_limbs(&mut zr_prod);
                        if zr_w0 == exp_zr_w0 {
                            dz_w0 = apply_shifted_signed_delta_below(
                                ctx.kernels,
                                ctx.sched,
                                max_w,
                                z_ext,
                                room,
                                room_zeroed,
                                &zr_prod,
                                r_shift_bits,
                                rho_neg,
                            );
                            crate::limb::put_limb_buf(zr_prod);
                            handled_with_fz = true;
                        } else {
                            crate::limb::put_limb_buf(zr_prod);
                        }
                    } else {
                        fz.spec.wrap_cyclic = false;
                        let f_rho_res = if (fz.spec.n as u64) * 8
                            > shares.effective_l3_bytes(1, fz.spec.rows) / 2
                        {
                            crate::ntt::ntt_forward(
                                ctx.kernels,
                                ctx.ntt_cache,
                                ctx.sched,
                                ctx.store,
                                &rho_limbs,
                                fz.spec,
                            )
                        } else {
                            crate::ntt::ntt_forward_reusable(
                                ctx.kernels,
                                ctx.ntt_cache,
                                ctx.sched,
                                ctx.store,
                                &rho_limbs,
                                fz.spec,
                            )
                        };
                        let mut f_rho = match f_rho_res {
                            Ok(v) => v,
                            Err(_) => {
                                fz.release_store(ctx.store);
                                return Err(MulError::NeedRollback(pid_base + 33));
                            }
                        };
                        let zr_comb = ntt_combine_and_inverse(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            ctx.telemetry,
                            &mut fz,
                            Some(NttOperand::ReuseInPlace(&mut f_rho)),
                            None,
                            out_cap,
                        );
                        fz.release_store(ctx.store);
                        let (mut zr_prod, zr_w0) = match zr_comb {
                            Ok(v) => v,
                            Err(_) => {
                                f_rho.release_store(ctx.store);
                                return Err(MulError::NeedRollback(pid_base + 33));
                            }
                        };
                        let u_top = &u_step[(step_m - corr_limbs)..step_m];
                        let exp_ur_w0 = mod61_mul(
                            compute_limbs_w0_par(ctx.kernels, ctx.sched, u_top, max_w),
                            rho_w0,
                        );
                        let mut fr_spec = f_rho.spec;
                        fr_spec.par_workers = max_w.max(1);
                        let ur_res = crate::ntt::ntt_mul_with_cached_op(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            ctx.telemetry,
                            u_top,
                            NttOperand::Owned(&mut f_rho),
                            fr_spec,
                            out_cap,
                        );
                        f_rho.release_store(ctx.store);
                        let (mut ur_prod, mut ur_w0) = match ur_res {
                            Ok(v) => v,
                            Err(_) => {
                                crate::limb::put_limb_buf(zr_prod);
                                return Err(MulError::NeedRollback(pid_base + 33));
                            }
                        };
                        let mut dummy_err = 0.0;
                        if let Some(hook) = ctx.fault_hook {
                            hook.on_product_limbs(
                                pid_base + 33,
                                0,
                                &mut ur_prod,
                                &mut ur_w0,
                                &mut dummy_err,
                            );
                        }
                        trim_limbs(&mut zr_prod);
                        trim_limbs(&mut ur_prod);
                        if zr_w0 == exp_zr_w0 && ur_w0 == exp_ur_w0 {
                            dz_w0 = apply_shifted_signed_delta_below(
                                ctx.kernels,
                                ctx.sched,
                                max_w,
                                z_ext,
                                room,
                                room_zeroed,
                                &zr_prod,
                                r_shift_bits,
                                rho_neg,
                            );
                            du_w0 = apply_shifted_signed_delta_in_place_par(
                                ctx.kernels,
                                ctx.sched,
                                max_w,
                                &mut u_step,
                                &ur_prod,
                                r_shift_bits,
                                rho_neg,
                                step_m,
                            );
                            crate::limb::put_limb_buf(zr_prod);
                            crate::limb::put_limb_buf(ur_prod);
                            handled_with_fz = true;
                        } else {
                            crate::limb::put_limb_buf(zr_prod);
                            crate::limb::put_limb_buf(ur_prod);
                        }
                    }
                } else {
                    fz.release_store(ctx.store);
                }
            }

            if handled_with_fz {
                crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
                u_limbs = u_step;
            } else {
                let mut handled_pair_shared = false;
                if !plan.out_of_core
                    && ctx.fault_hook.is_none()
                    && matches!(corr_method, MulMethod::Fft { .. } | MulMethod::Ntt(_))
                {
                    let z_top_s = &z_ext[(step_m - corr_limbs)..step_m];
                    let known_z_w0 = (corr_limbs == cur_m).then_some(z_w0_step).flatten();
                    if !is_last {
                        let u_top = &u_step[(step_m - corr_limbs)..step_m];
                        if let Some((zr_prod, ur_prod)) = ctx.try_mul_slices_pair_shared_b_with_w0(
                            &corr_method,
                            max_w,
                            plan.ram_transform_budget,
                            z_top_s,
                            known_z_w0,
                            u_top,
                            &rho_limbs,
                            cached_rho_w0,
                        ) {
                            dz_w0 = apply_shifted_signed_delta_below(
                                ctx.kernels,
                                ctx.sched,
                                max_w,
                                z_ext,
                                room,
                                room_zeroed,
                                &zr_prod,
                                r_shift_bits,
                                rho_neg,
                            );
                            du_w0 = apply_shifted_signed_delta_in_place_par(
                                ctx.kernels,
                                ctx.sched,
                                max_w,
                                &mut u_step,
                                &ur_prod,
                                r_shift_bits,
                                rho_neg,
                                step_m,
                            );
                            crate::limb::put_limb_buf(zr_prod);
                            crate::limb::put_limb_buf(ur_prod);
                            handled_pair_shared = true;
                        }
                    } else if let Some(zr_prod) = ctx.try_mul_slices_with_w0(
                        &corr_method,
                        max_w,
                        z_top_s,
                        known_z_w0,
                        &rho_limbs,
                        cached_rho_w0,
                    ) {
                        dz_w0 = apply_shifted_signed_delta_below(
                            ctx.kernels,
                            ctx.sched,
                            max_w,
                            z_ext,
                            room,
                            room_zeroed,
                            &zr_prod,
                            r_shift_bits,
                            rho_neg,
                        );
                        crate::limb::put_limb_buf(zr_prod);
                        handled_pair_shared = true;
                    }
                }
                if handled_pair_shared {
                    crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
                    u_limbs = u_step;
                } else {
                    let z_top = &z_ext[(step_m - corr_limbs)..step_m];
                    if matches!(corr_method, MulMethod::Fft { .. } | MulMethod::Ntt(_)) {
                // Share `fwd(rho)` across `z * rho` and (when `!is_last`) `u * rho`, avoiding slice clones!
                let mut f_rho = if !is_last {
                    ctx.forward_cached_reusable(&corr_method, &rho_limbs)?
                } else {
                    ctx.forward_cached(&corr_method, &rho_limbs)?
                };
                let rho_w0 = cached_rho_w0.unwrap_or_else(|| {
                    compute_limbs_w0_par(ctx.kernels, ctx.sched, &rho_limbs, max_w)
                });

                let z_top_w0 = if corr_limbs == cur_m {
                    cached_z_w0.unwrap_or_else(|| {
                        compute_limbs_w0_par(ctx.kernels, ctx.sched, z_top, max_w)
                    })
                } else {
                    compute_limbs_w0_par(ctx.kernels, ctx.sched, z_top, max_w)
                };
                let exp_zr_w0 = mod61_mul(z_top_w0, rho_w0);
                let zr_res = ctx.mul_limbs_with_cached(
                    pid_base + 32,
                    &corr_method,
                    z_top,
                    &rho_limbs,
                    &f_rho,
                    corr_limbs + rho_limbs.len() + 2,
                    Some(exp_zr_w0),
                );

                if !is_last {
                    let u_top = &u_step[(step_m - corr_limbs)..step_m];
                    let exp_ur_w0 = mod61_mul(
                        compute_limbs_w0_par(ctx.kernels, ctx.sched, u_top, max_w),
                        rho_w0,
                    );
                    let ur_res = ctx.mul_limbs_with_cached(
                        pid_base + 33,
                        &corr_method,
                        u_top,
                        &rho_limbs,
                        &f_rho,
                        corr_limbs + rho_limbs.len() + 2,
                        Some(exp_ur_w0),
                    );
                    f_rho.release_store(ctx.store);

                    let (zr_prod, _) = zr_res?;
                    let (ur_prod, _) = ur_res?;
                    dz_w0 = apply_shifted_signed_delta_below(
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                        z_ext,
                        room,
                        room_zeroed,
                        &zr_prod,
                        r_shift_bits,
                        rho_neg,
                    );
                    du_w0 = apply_shifted_signed_delta_in_place_par(
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                        &mut u_step,
                        &ur_prod,
                        r_shift_bits,
                        rho_neg,
                        step_m,
                    );
                    crate::limb::put_limb_buf(zr_prod);
                    crate::limb::put_limb_buf(ur_prod);
                    crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
                    u_limbs = u_step;
                } else {
                    f_rho.release_store(ctx.store);
                    let (zr_prod, _) = zr_res?;
                    dz_w0 = apply_shifted_signed_delta_below(
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                        z_ext,
                        room,
                        room_zeroed,
                        &zr_prod,
                        r_shift_bits,
                        rho_neg,
                    );
                    crate::limb::put_limb_buf(zr_prod);
                    crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
                    u_limbs = u_step;
                }
            } else {
                let mut bz = BigNum::from_limbs(z_top.to_vec(), false, 0, ctx.kernels);
                let mut br = BigNum::from_limbs(rho_limbs.clone(), false, 0, ctx.kernels);
                let zr_bn = ctx.mul_whole(pid_base + 32, &corr_method, &mut bz, &mut br)?;
                let z_delta = shr_limbs(&zr_bn.limbs, r_shift_bits);
                if !room_zeroed {
                    z_ext[..room].fill(0);
                }
                let mut z_whole = z_ext.to_vec();
                apply_signed_delta_in_place_par(
                    ctx.kernels,
                    ctx.sched,
                    max_w,
                    &mut z_whole,
                    &z_delta,
                    rho_neg,
                    step_m,
                );
                z_ext.copy_from_slice(&z_whole);

                if !is_last {
                    let u_top = &u_step[(step_m - corr_limbs)..step_m];
                    let mut bu = BigNum::from_limbs(u_top.to_vec(), false, 0, ctx.kernels);
                    let ur_bn = ctx.mul_whole(pid_base + 33, &corr_method, &mut bu, &mut br)?;
                    let u_delta = shr_limbs(&ur_bn.limbs, r_shift_bits);
                    apply_signed_delta_in_place_par(
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                        &mut u_step,
                        &u_delta,
                        rho_neg,
                        step_m,
                    );
                }
                crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
                u_limbs = u_step;
                    }
                }
            }
        } else {
            if let Some(mut fz) = cached_z_opt.take() {
                fz.release_store(ctx.store);
            }
            if !room_zeroed {
                z_ext[..room].fill(0);
            }
            crate::limb::put_limb_buf(take_limb_vec(&mut u_limbs));
            u_limbs = u_step;
        }
        crate::limb::put_limb_buf(rho_limbs);

        // The new z is the old one moved up by the limbs the step adds, plus or minus its
        // correction; the new u is the step's u plus or minus its own.
        let with_delta = |w0: Option<u64>, delta_w0: Option<u64>| -> Option<u64> {
            let (w0, delta_w0) = (w0?, delta_w0?);
            Some(if rho_neg { mod61_sub(w0, delta_w0) } else { mod61_add(w0, delta_w0) })
        };
        z_w0 = with_delta(
            z_w0_step.map(|w0| mod61_mul(w0, mod61_pow2(((step_m - cur_m) as u64) * 64))),
            dz_w0,
        );
        u_w0 = if is_last { None } else { with_delta(u_step_w0, du_w0) };
        cur_m = step_m;
        acc_bits = next_acc;
        if do_trace && step_m >= 65_536 {
            eprintln!(
                "[TRACE FINISH STEP] idx={step_idx}/{num_steps} step_m={step_m} is_last={is_last} -> {:.2}ms",
                t_step0.elapsed().as_secs_f64() * 1e3
            );
        }

        // A checkpoint of the finish carries T. Where T was let go above, on the plan's word
        // that no checkpoint falls in the finish, none is offered: the clock may say otherwise
        // on a loaded or slower machine, and T's limbs are gone.
        if let Some(sink) = ckpt_sink.filter(|_| !can_drop_t_early) {
            let elapsed = last_ckpt.elapsed().as_secs_f64();
            if sink.should_checkpoint(elapsed) {
                if let Some(wh) = q_park_write_h.take() {
                    wh.wait().map_err(|_| MulError::NeedRollback(0x3F00_0011))?;
                }
                if let Some(qp) = q_parked.take() {
                    if let Some(rh) = q_prefetch_h.take() {
                        let rstage = rh.wait().map_err(|_| MulError::NeedRollback(0x3F00_0012))?;
                        *q_root = qp.finish_unpark_bignum(ctx, rstage, 0x3F00_0012, max_w)?;
                    } else {
                        *q_root = qp.unpark_bignum_sync(ctx, 0x3F00_0012, max_w)?;
                    }
                }
                let st = NewtonResumeState {
                    step_limbs: cur_m,
                    acc_bits,
                    z: BigNum::from_limbs_par(
                        z.limbs().to_vec(),
                        false,
                        0,
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                    ),
                    u: BigNum::from_limbs_par(
                        u_limbs.clone(),
                        false,
                        0,
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                    ),
                };
                if sink.maybe_checkpoint_newton(&st, q_root, t_root, elapsed)? {
                    last_ckpt = Instant::now();
                }
                if !is_last
                    && plan.out_of_core
                    && let Some(store_ref) = ctx.store
                    && !q_root.is_zero()
                {
                    if let Some(prep) = crate::mul::PreparedParkWrite::prepare_bignum(
                        ctx,
                        0x3F00_0010,
                        q_root,
                        max_w,
                    )? {
                        let (parked, wjob) = prep.split_write_job();
                        q_parked = Some(parked);
                        q_park_write_h =
                            Some(store_ref.submit_loader_scoped(move || wjob.execute(store_ref)));
                    }
                }
            }
        }
    }

    let t_fin_prod0 = Instant::now();
    let steps_ms = t_f0.elapsed().as_secs_f64() * 1e3 - setup_ms;
    crate::limb::put_limb_buf(a_norm);
    crate::limb::put_limb_buf(u_limbs);
    crate::limb::put_limb_buf(take_limb_vec(&mut t_root.limbs));

    if let Some(wh) = q_park_write_h.take() {
        wh.wait().map_err(|_| MulError::NeedRollback(0x3F00_0011))?;
    }
    if let Some(qp) = q_parked.take() {
        if let Some(rh) = q_prefetch_h.take() {
            let rstage = rh.wait().map_err(|_| MulError::NeedRollback(0x3F00_0012))?;
            *q_root = qp.finish_unpark_bignum(ctx, rstage, 0x3F00_0012, max_w)?;
        } else {
            *q_root = qp.unpark_bignum_sync(ctx, 0x3F00_0012, max_w)?;
        }
    }

    // 3. Final whole product: `pi = 4270934400 * Q * z` (Section 3 & 11)!
    // Exact binary point position: `pi = (4270934400 * Q_odd * z_final) * 2^{-frac_bits_total}`!
    let q_shift = q_root.shift;
    let frac_bits_total = ((n_a + cur_m) as u64) * 64 - q_shift - s_norm;
    let out_frac_bits = (out_frac_limbs as u64) * 64;
    debug_assert!(frac_bits_total >= out_frac_bits);
    let drop_low_bits = frac_bits_total - out_frac_bits;

    let z_limbs = z.into_vec();
    let mut z_final = match z_w0 {
        Some(w0) => BigNum::from_limbs_with_w0(z_limbs, false, 0, w0),
        None => BigNum::from_limbs_par(z_limbs, false, 0, ctx.kernels, ctx.sched, max_w),
    };

    // Slice `q_root` to its top `cur_m + 6` limbs BEFORE multiplying by `FINISH_C_MUL` when
    // `q_root` carries unreduced low limbs far below `drop_low_bits` ("The product by Q, three
    // transforms at twice the length", Section 3), verifying two guard conditions:
    //  (1) `cq_sub.limbs[1] != u64::MAX` (so omitted `q_root[..cut_q - 2] * C` cannot carry into `cut_q`)
    //  (2) `p_top.limbs[guard_word] != u64::MAX` (so omitted `cq_odd[..cut_q] * z_final` cannot carry into `out_frac_bits`)!
    let min_keep_bits = ((cur_m as u64) + 4) * 64;
    let mut scaled_opt: Option<(LimbVec, u64, u64)> = None;
    if ctx.fault_hook.is_none() && q_root.limbs.len() >= 512 && drop_low_bits > min_keep_bits {
        let cut_cq_limbs = ((drop_low_bits - min_keep_bits) / 64) as usize;
        if cut_cq_limbs >= 66 && cut_cq_limbs + 4 < q_root.limbs.len() {
            let p0 = (q_root.limbs[cut_cq_limbs - 2] as u128) * (FINISH_C_MUL as u128);
            let p1 =
                (q_root.limbs[cut_cq_limbs - 1] as u128) * (FINISH_C_MUL as u128) + (p0 >> 64);
            if (p1 as u64) != u64::MAX {
                let carry_in = (p1 >> 64) as u64;
                let rem_drop_bits = drop_low_bits - (cut_cq_limbs as u64) * 64;
                let guard_word = ((rem_drop_bits / 64) as usize).saturating_sub(1);
                let mut cq_top = mul_slice_u64_with_carry_par(
                    ctx.kernels,
                    ctx.sched,
                    max_w,
                    &q_root.limbs[cut_cq_limbs..],
                    FINISH_C_MUL,
                    carry_in,
                );
                let top_method = plan.select_mul_method(
                    cq_top.body_bits(),
                    z_final.body_bits(),
                    false,
                    false,
                    shares.active_workers,
                    shares,
                    cal,
                );
                let p_top = ctx.mul_whole(0x3FFF_0000, &top_method, &mut cq_top, &mut z_final)?;
                crate::limb::put_limb_buf(cq_top.into_limbs());
                if p_top.limbs.get(guard_word).copied().unwrap_or(0) != u64::MAX {
                    crate::limb::put_limb_buf(take_limb_vec(&mut q_root.limbs));
                    crate::limb::put_limb_buf(take_limb_vec(&mut z_final.limbs));
                    let s = shr_truncate_and_w0_par(
                        ctx.kernels,
                        &p_top.limbs,
                        rem_drop_bits,
                        out_frac_limbs,
                        ctx.sched,
                        max_w,
                    );
                    crate::limb::put_limb_buf(p_top.into_limbs());
                    scaled_opt = Some(s);
                } else {
                    crate::limb::put_limb_buf(p_top.into_limbs());
                }
            }
        }
    }

    let (scaled, int_part, frac_w0) = match scaled_opt {
        Some(s) => s,
        None => {
            let mut cq_odd =
                mul_bignum_u64_par(ctx.kernels, q_root, FINISH_C_MUL, ctx.sched, max_w);
            crate::limb::put_limb_buf(take_limb_vec(&mut q_root.limbs));
            cq_odd.shift = 0;
            cq_odd.chk = DualCheck::new(compute_limbs_w0_par(
                ctx.kernels,
                ctx.sched,
                &cq_odd.limbs,
                max_w,
            ));
            let final_method = plan.select_mul_method(
                cq_odd.body_bits(),
                z_final.body_bits(),
                false,
                false,
                shares.active_workers,
                shares,
                cal,
            );
            let p_final = ctx.mul_whole(0x3FFF_0000, &final_method, &mut cq_odd, &mut z_final)?;
            crate::limb::put_limb_buf(cq_odd.into_limbs());
            crate::limb::put_limb_buf(z_final.into_limbs());
            let s = shr_truncate_and_w0_par(
                ctx.kernels,
                &p_final.limbs,
                drop_low_bits,
                out_frac_limbs,
                ctx.sched,
                max_w,
            );
            crate::limb::put_limb_buf(p_final.into_limbs());
            s
        }
    };
    if int_part != 3 {
        crate::limb::put_limb_buf(scaled);
        return Err(MulError::FatalStop(format!(
            "Finish integrity check failed: expected integer part 3, got {int_part}"
        )));
    }
    let frac_bn = BigNum {
        limbs: scaled,
        neg: false,
        shift: 0,
        chk: DualCheck::new(frac_w0),
    };
    if do_trace {
        eprintln!(
            "[TRACE FINISH] setup_A={setup_ms:.2}ms steps={steps_ms:.2}ms final_cq_z={:.2}ms",
            t_fin_prod0.elapsed().as_secs_f64() * 1e3
        );
    }
    let _ = CachedTransform::release_store;
    Ok((int_part, frac_bn, out_frac_limbs))
}

fn shr_truncate_and_w0_par(
    kernels: &crate::refk::Kernels,
    src: &[u64],
    shift_bits: u64,
    out_frac_limbs: usize,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> (LimbVec, u64, u64) {
    let int_part = extract_u64_at_bit(src, shift_bits + (out_frac_limbs as u64) * 64);
    if out_frac_limbs == 0 {
        return (empty_limb_vec(), int_part, 0);
    }
    let ws = (shift_bits >> 6) as usize;
    let bs = (shift_bits & 63) as u32;
    let src_len = src.len();
    let mut out = crate::limb::take_limb_buf_uninit(out_frac_limbs);
    let chunk = 16_384usize;
    let num_chunks = out_frac_limbs.div_ceil(chunk);
    if max_workers <= 1 || num_chunks <= 1 {
        for i in 0..out_frac_limbs {
            let si = ws + i;
            let v = if si >= src_len {
                0
            } else if bs == 0 {
                src[si]
            } else {
                let lo = src[si] >> bs;
                let hi = if si + 1 < src_len {
                    src[si + 1] << (64 - bs)
                } else {
                    0
                };
                lo | hi
            };
            out.push(v);
        }
        let w0 = unsafe {
            let mut tmp = [0u64; 3];
            (kernels.active.check_n)(out.as_ptr(), out.len() as u64, &mut tmp, 0)
        };
        trim_limbs(&mut out);
        return (out, int_part, w0);
    }
    let mut chunk_w0s = vec![0u64; num_chunks];
    let out_ptr = out.as_mut_ptr() as usize;
    let w0_ptr = chunk_w0s.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(out_frac_limbs);
        let dst = unsafe { (out_ptr as *mut u64).add(start) };
        if bs == 0 {
            for i in start..end {
                let si = ws + i;
                let v = if si < src_len { src[si] } else { 0 };
                unsafe { dst.add(i - start).write(v) };
            }
        } else {
            for i in start..end {
                let si = ws + i;
                let v = if si >= src_len {
                    0
                } else {
                    let lo = src[si] >> bs;
                    let hi = if si + 1 < src_len {
                        src[si + 1] << (64 - bs)
                    } else {
                        0
                    };
                    lo | hi
                };
                unsafe { dst.add(i - start).write(v) };
            }
        }
        let cw0 = unsafe {
            let mut tmp = [0u64; 3];
            (kernels.active.check_n)(dst as *const u64, (end - start) as u64, &mut tmp, 0)
        };
        unsafe { *(w0_ptr as *mut u64).add(ci) = cw0 };
    });
    unsafe { out.set_len(out_frac_limbs) };
    let step_shift = mod61_pow2((chunk as u64) * 3);
    let mut shift_acc = 1u64;
    let mut total_w0 = 0u64;
    for &cw in &chunk_w0s {
        total_w0 = mod61_add(total_w0, mod61_mul(cw, shift_acc));
        shift_acc = mod61_mul(shift_acc, step_shift);
    }
    trim_limbs(&mut out);
    (out, int_part, total_w0)
}

fn sub_limbs_in_place_par<A: Allocator>(
    kernels: &crate::refk::Kernels,
    a: &mut Vec<u64, A>,
    b: &[u64],
    sched: &crate::par::Scheduler,
    max_workers: usize,
) {
    let mut n_short = b.len();
    while n_short > 0 && b[n_short - 1] == 0 {
        n_short -= 1;
    }
    if n_short == 0 {
        trim_limbs(a);
        return;
    }
    let n_long = a.len();
    debug_assert!(n_long >= n_short);
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let chunk = 16_384usize;
    let num_chunks = n_short.div_ceil(chunk);
    if max_workers <= 1 || num_chunks <= 1 {
        let mut br = unsafe {
            (kernels.active.sub_n)(a.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n_short as u64)
        };
        let mut pos = n_short;
        while br != 0 && pos < n_long {
            let (d, un) = a[pos].overflowing_sub(br);
            a[pos] = d;
            br = u64::from(un);
            pos += 1;
        }
        debug_assert_eq!(br, 0);
        trim_limbs(a);
        return;
    }
    let mut borrows = vec![0u64; num_chunks];
    let a_ptr = a.as_mut_ptr() as usize;
    let borrows_ptr = borrows.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let s = ci * chunk;
        let e = (s + chunk).min(n_short);
        let b_out = unsafe {
            (kernels.active.sub_n)(
                (a_ptr as *mut u64).add(s),
                (a_ptr as *const u64).add(s),
                b.as_ptr().add(s),
                (e - s) as u64,
            )
        };
        unsafe {
            *(borrows_ptr as *mut u64).add(ci) = b_out;
        }
    });
    for (ci, &b_in) in borrows.iter().enumerate() {
        if b_in == 0 {
            continue;
        }
        let mut pos = ((ci + 1) * chunk).min(n_short);
        let mut br = b_in;
        while br != 0 && pos < n_long {
            let (d, un) = a[pos].overflowing_sub(br);
            a[pos] = d;
            br = u64::from(un);
            pos += 1;
        }
        debug_assert_eq!(br, 0);
    }
    trim_limbs(a);
}

fn shl_sublimb_in_place_par(
    limbs: &mut [u64],
    shift_bits: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) {
    let bs = (shift_bits & 63) as u32;
    if bs == 0 || limbs.is_empty() {
        return;
    }
    let n = limbs.len();
    let chunk = 16_384usize;
    let num_chunks = n.div_ceil(chunk);
    if max_workers <= 1 || num_chunks <= 1 {
        let mut prev = 0u64;
        for w in limbs.iter_mut() {
            let cur = *w;
            *w = (cur << bs) | prev;
            prev = cur >> (64 - bs);
        }
        debug_assert_eq!(prev, 0);
        return;
    }
    let prev_words: Vec<u64> = (0..num_chunks)
        .map(|ci| {
            if ci == 0 {
                0u64
            } else {
                limbs[ci * chunk - 1] >> (64 - bs)
            }
        })
        .collect();
    let ptr = limbs.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(n);
        let mut prev = prev_words[ci];
        for i in start..end {
            unsafe {
                let p = (ptr as *mut u64).add(i);
                let cur = *p;
                *p = (cur << bs) | prev;
                prev = cur >> (64 - bs);
            }
        }
    });
}

/// A step's correction applied to a number that has just been given `room` limbs below it
/// (`GrowsDown::grown`): `dst[room..]` holds the number and the result is
/// `number * 2^(64 room) +/- (prod >> r_shift_bits)`.
///
/// The correction is about as long as the room, so nearly all the new limbs are the
/// correction's own, or their complement for a correction taken away, and they are written
/// straight from the product: the room is not filled with zeros that are then read back and
/// added to. What reaches above the room, with the carry or borrow, is settled in the old
/// limbs, a few of them. With `room_zeroed` the room already holds zeros and the general pass
/// is used, as it is for the cases this one leaves alone. Returns the check word of the
/// correction on the terms of `apply_shifted_signed_delta_in_place_par`.
#[allow(clippy::too_many_arguments)]
fn apply_shifted_signed_delta_below(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    dst: &mut [u64],
    room: usize,
    room_zeroed: bool,
    prod: &[u64],
    r_shift_bits: u64,
    delta_neg: bool,
) -> Option<u64> {
    const CHUNK: usize = 16_384;
    let n = dst.len();
    let ws = (r_shift_bits >> 6) as usize;
    let bs = (r_shift_bits & 63) as u32;
    let plen = prod.len();
    let shr_at = |idx: usize| -> u64 {
        let p_i = ws + idx;
        let lo = prod[p_i] >> bs;
        let hi = if p_i + 1 < plen {
            prod[p_i + 1] << (64 - bs)
        } else {
            0
        };
        lo | hi
    };
    let mut n_d = plen.saturating_sub(ws);
    if bs != 0 {
        while n_d > 0 && shr_at(n_d - 1) == 0 {
            n_d -= 1;
        }
    }
    // The general pass, on a copy: for the room already zeroed, a correction of no length, of
    // nearly the number's, or reaching far above the room, a shift of whole limbs, a number
    // with nothing in its top limb.
    if room_zeroed
        || bs == 0
        || max_workers <= 1
        || room < 2 * CHUNK
        || n_d == 0
        || n_d + 2 > n
        || n_d > room + 4_096
        || dst[n - 1] == 0
    {
        if !room_zeroed {
            dst[..room].fill(0);
        }
        let mut whole = crate::limb::slice_to_limb_vec(dst);
        let delta_w0 = apply_shifted_signed_delta_in_place_par(
            kernels,
            sched,
            max_workers,
            &mut whole,
            prod,
            r_shift_bits,
            delta_neg,
            n,
        );
        dst.copy_from_slice(&whole);
        crate::limb::put_limb_buf(whole);
        return delta_w0;
    }

    // The correction's limbs that fall in the room, written as they are or complemented.
    let low = room.min(n_d);
    let num_chunks = low.div_ceil(CHUNK);
    let mut chunk_w0s = vec![0u64; num_chunks];
    let w0s_ptr = chunk_w0s.as_mut_ptr() as usize;
    let dst_ptr = dst.as_mut_ptr() as usize;
    let flip = if delta_neg { u64::MAX } else { 0 };
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let c_start = ci * CHUNK;
        let c_end = (c_start + CHUNK).min(low);
        let mut blk = [0u64; 256];
        let mut cw0 = 0u64;
        let mut at = c_start;
        while at < c_end {
            let len = (c_end - at).min(256);
            for (t, slot) in blk.iter_mut().enumerate().take(len) {
                *slot = shr_at(at + t);
            }
            cw0 = add_block_w0_at(cw0, compute_limbs_w0(kernels, &blk[..len]), at - c_start);
            unsafe {
                let out = (dst_ptr as *mut u64).add(at);
                for (t, &v) in blk.iter().enumerate().take(len) {
                    out.add(t).write(v ^ flip);
                }
            }
            at += len;
        }
        unsafe {
            *(w0s_ptr as *mut u64).add(ci) = cw0;
        }
    });
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let mut delta_w0 = 0u64;
    for (ci, &cw0) in chunk_w0s.iter().enumerate() {
        delta_w0 = add_block_w0_at(delta_w0, cw0, ci * CHUNK);
    }
    // Room above the correction's end: zeros, or their complement.
    dst[low..room].fill(flip);

    // The rest in the old limbs. Taken away, the correction's low part is `2^(64 room)` less
    // itself in the room, the complement plus one, and one is borrowed above unless that part
    // was zero.
    let mut c = 0u64;
    if delta_neg {
        let mut one = 1u64;
        for x in dst[..room].iter_mut() {
            let (v, over) = x.overflowing_add(one);
            *x = v;
            one = u64::from(over);
            if one == 0 {
                break;
            }
        }
        c = 1 - one;
    }
    for i in room..n_d {
        let d = shr_at(i);
        delta_w0 = add_block_w0_at(delta_w0, mod61_reduce(d as u128), i);
        let (v1, o1) = if delta_neg { dst[i].overflowing_sub(d) } else { dst[i].overflowing_add(d) };
        let (v2, o2) = if delta_neg { v1.overflowing_sub(c) } else { v1.overflowing_add(c) };
        dst[i] = v2;
        c = u64::from(o1) + u64::from(o2);
    }
    let mut i = n_d.max(room);
    while c != 0 && i < n {
        let (v, over) = if delta_neg { dst[i].overflowing_sub(c) } else { dst[i].overflowing_add(c) };
        dst[i] = v;
        c = u64::from(over);
        i += 1;
    }
    // A carry or a borrow out of the top: the result is not the sum or the difference.
    (c == 0).then_some(delta_w0)
}

/// `base +/- (prod >> r_shift_bits)`, in place, cut to `keep_limbs`.
///
/// Returns the check word of the correction it applied, `prod >> r_shift_bits`, where it took
/// it: the word of the result is then the word `base` had plus or minus that one, and neither
/// number has to be read again for its word. `None` where it did not, or where the result is
/// not that sum or difference (a carry past `keep_limbs`, a correction not below `base`).
#[allow(clippy::too_many_arguments)]
fn apply_shifted_signed_delta_in_place_par<A: Allocator>(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    base: &mut Vec<u64, A>,
    prod: &[u64],
    r_shift_bits: u64,
    delta_neg: bool,
    keep_limbs: usize,
) -> Option<u64> {
    let ws = (r_shift_bits >> 6) as usize;
    let bs = (r_shift_bits & 63) as u32;
    if prod.len() <= ws {
        base.resize(keep_limbs, 0);
        return Some(0);
    }
    if bs == 0 {
        apply_signed_delta_in_place_par(
            kernels,
            sched,
            max_workers,
            base,
            &prod[ws..],
            delta_neg,
            keep_limbs,
        );
        return None;
    }
    let plen = prod.len();
    let shr_at = |idx: usize| -> u64 {
        let p_i = ws + idx;
        let lo = prod[p_i] >> bs;
        let hi = if p_i + 1 < plen {
            prod[p_i + 1] << (64 - bs)
        } else {
            0
        };
        lo | hi
    };
    let mut n_d = plen - ws;
    while n_d > 0 && shr_at(n_d - 1) == 0 {
        n_d -= 1;
    }
    if n_d == 0 {
        base.resize(keep_limbs, 0);
        return Some(0);
    }
    // The word of a block of the correction at limb `at` of it, added to `acc`.
    let add_block_w0 = |acc: u64, blk: &[u64], at: usize| -> u64 {
        mod61_add(acc, mod61_mul(compute_limbs_w0(kernels, blk), mod61_pow2((at as u64) * 3)))
    };

    if !delta_neg {
        if base.len() < n_d {
            base.resize(n_d, 0);
        }
        let n_long = base.len();
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add(n_long as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let chunk = 16_384usize;
        let num_chunks = n_d.div_ceil(chunk);
        let mut delta_w0 = 0u64;
        if max_workers <= 1 || num_chunks <= 1 {
            let mut blk = [0u64; 256];
            let mut cy = 0u64;
            let mut s = 0usize;
            while s < n_d {
                let len = (n_d - s).min(256);
                for (t, slot) in blk.iter_mut().enumerate().take(len) {
                    *slot = shr_at(s + t);
                }
                delta_w0 = add_block_w0(delta_w0, &blk[..len], s);
                let mut c_blk = unsafe {
                    (kernels.active.add_n)(
                        base.as_mut_ptr().add(s),
                        base.as_ptr().add(s),
                        blk.as_ptr(),
                        len as u64,
                    )
                };
                if cy != 0 {
                    let mut p = s;
                    while cy != 0 && p < s + len {
                        let (sum, ov) = base[p].overflowing_add(cy);
                        base[p] = sum;
                        cy = u64::from(ov);
                        p += 1;
                    }
                    c_blk += cy;
                }
                cy = c_blk;
                s += len;
            }
            let mut pos = n_d;
            while cy != 0 && pos < base.len() {
                let (sum, ov) = base[pos].overflowing_add(cy);
                base[pos] = sum;
                cy = u64::from(ov);
                pos += 1;
            }
            if cy != 0 {
                base.push(cy);
            }
        } else {
            let mut carries = vec![0u64; num_chunks];
            let mut chunk_w0s = vec![0u64; num_chunks];
            let base_ptr = base.as_mut_ptr() as usize;
            let carries_ptr = carries.as_mut_ptr() as usize;
            let w0s_ptr = chunk_w0s.as_mut_ptr() as usize;
            sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
                let c_start = ci * chunk;
                let c_end = (c_start + chunk).min(n_d);
                let mut blk = [0u64; 256];
                let mut cy = 0u64;
                let mut cw0 = 0u64;
                let mut s = c_start;
                while s < c_end {
                    let len = (c_end - s).min(256);
                    for (t, slot) in blk.iter_mut().enumerate().take(len) {
                        *slot = shr_at(s + t);
                    }
                    cw0 = add_block_w0(cw0, &blk[..len], s - c_start);
                    let mut c_blk = unsafe {
                        (kernels.active.add_n)(
                            (base_ptr as *mut u64).add(s),
                            (base_ptr as *const u64).add(s),
                            blk.as_ptr(),
                            len as u64,
                        )
                    };
                    if cy != 0 {
                        let mut p = s;
                        while cy != 0 && p < s + len {
                            unsafe {
                                let bp = (base_ptr as *mut u64).add(p);
                                let (sum, ov) = (*bp).overflowing_add(cy);
                                *bp = sum;
                                cy = u64::from(ov);
                            }
                            p += 1;
                        }
                        c_blk += cy;
                    }
                    cy = c_blk;
                    s += len;
                }
                unsafe {
                    *(carries_ptr as *mut u64).add(ci) = cy;
                    *(w0s_ptr as *mut u64).add(ci) = cw0;
                }
            });
            for (ci, &cw0) in chunk_w0s.iter().enumerate() {
                delta_w0 = add_block_w0_at(delta_w0, cw0, ci * chunk);
            }
            for (ci, &c_in) in carries.iter().enumerate() {
                if c_in == 0 {
                    continue;
                }
                let mut pos = ((ci + 1) * chunk).min(n_d);
                let mut cy = c_in;
                while cy != 0 && pos < base.len() {
                    let (sum, ov) = base[pos].overflowing_add(cy);
                    base[pos] = sum;
                    cy = u64::from(ov);
                    pos += 1;
                }
                if cy != 0 {
                    base.push(cy);
                }
            }
        }
        let whole = base.len() <= keep_limbs;
        base.resize(keep_limbs, 0);
        return whole.then_some(delta_w0);
    }

    // `delta_neg == true`: check `base > delta` from the top limb down without allocating `delta`.
    let mut b_len = base.len();
    while b_len > 0 && base[b_len - 1] == 0 {
        b_len -= 1;
    }
    let greater = if b_len != n_d {
        b_len > n_d
    } else {
        let mut ord = std::cmp::Ordering::Equal;
        for i in (0..n_d).rev() {
            let bw = base[i];
            let dw = shr_at(i);
            if bw != dw {
                ord = bw.cmp(&dw);
                break;
            }
        }
        ord == std::cmp::Ordering::Greater
    };
    if !greater {
        base.clear();
        base.resize(keep_limbs, 0);
        return None;
    }

    let n_long = base.len();
    let mut delta_w0 = 0u64;
    if kernels.count_work {
        kernels
            .stats
            .limb_ops
            .fetch_add(n_long as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let chunk = 16_384usize;
    let num_chunks = n_d.div_ceil(chunk);
    if max_workers <= 1 || num_chunks <= 1 {
        let mut blk = [0u64; 256];
        let mut br = 0u64;
        let mut s = 0usize;
        while s < n_d {
            let len = (n_d - s).min(256);
            for (t, slot) in blk.iter_mut().enumerate().take(len) {
                *slot = shr_at(s + t);
            }
            delta_w0 = add_block_w0(delta_w0, &blk[..len], s);
            let mut b_blk = unsafe {
                (kernels.active.sub_n)(
                    base.as_mut_ptr().add(s),
                    base.as_ptr().add(s),
                    blk.as_ptr(),
                    len as u64,
                )
            };
            if br != 0 {
                let mut p = s;
                while br != 0 && p < s + len {
                    let (d, un) = base[p].overflowing_sub(br);
                    base[p] = d;
                    br = u64::from(un);
                    p += 1;
                }
                b_blk += br;
            }
            br = b_blk;
            s += len;
        }
        let mut pos = n_d;
        while br != 0 && pos < n_long {
            let (d, un) = base[pos].overflowing_sub(br);
            base[pos] = d;
            br = u64::from(un);
            pos += 1;
        }
        debug_assert_eq!(br, 0);
    } else {
        let mut borrows = vec![0u64; num_chunks];
        let mut chunk_w0s = vec![0u64; num_chunks];
        let base_ptr = base.as_mut_ptr() as usize;
        let borrows_ptr = borrows.as_mut_ptr() as usize;
        let w0s_ptr = chunk_w0s.as_mut_ptr() as usize;
        sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
            let c_start = ci * chunk;
            let c_end = (c_start + chunk).min(n_d);
            let mut blk = [0u64; 256];
            let mut br = 0u64;
            let mut cw0 = 0u64;
            let mut s = c_start;
            while s < c_end {
                let len = (c_end - s).min(256);
                for (t, slot) in blk.iter_mut().enumerate().take(len) {
                    *slot = shr_at(s + t);
                }
                cw0 = add_block_w0(cw0, &blk[..len], s - c_start);
                let mut b_blk = unsafe {
                    (kernels.active.sub_n)(
                        (base_ptr as *mut u64).add(s),
                        (base_ptr as *const u64).add(s),
                        blk.as_ptr(),
                        len as u64,
                    )
                };
                if br != 0 {
                    let mut p = s;
                    while br != 0 && p < s + len {
                        unsafe {
                            let bp = (base_ptr as *mut u64).add(p);
                            let (d, un) = (*bp).overflowing_sub(br);
                            *bp = d;
                            br = u64::from(un);
                        }
                        p += 1;
                    }
                    b_blk += br;
                }
                br = b_blk;
                s += len;
            }
            unsafe {
                *(borrows_ptr as *mut u64).add(ci) = br;
                *(w0s_ptr as *mut u64).add(ci) = cw0;
            }
        });
        for (ci, &cw0) in chunk_w0s.iter().enumerate() {
            delta_w0 = add_block_w0_at(delta_w0, cw0, ci * chunk);
        }
        for (ci, &b_in) in borrows.iter().enumerate() {
            if b_in == 0 {
                continue;
            }
            let mut pos = ((ci + 1) * chunk).min(n_d);
            let mut br = b_in;
            while br != 0 && pos < n_long {
                let (d, un) = base[pos].overflowing_sub(br);
                base[pos] = d;
                br = u64::from(un);
                pos += 1;
            }
            debug_assert_eq!(br, 0);
        }
    }
    base.resize(keep_limbs, 0);
    (n_long <= keep_limbs).then_some(delta_w0)
}

/// `acc` plus the check word `w0` of a piece that begins at limb `at`.
#[inline]
fn add_block_w0_at(acc: u64, w0: u64, at: usize) -> u64 {
    mod61_add(acc, mod61_mul(w0, mod61_pow2((at as u64) * 3)))
}

fn apply_signed_delta_in_place_par<A: Allocator>(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    base: &mut Vec<u64, A>,
    delta: &[u64],
    delta_neg: bool,
    keep_limbs: usize,
) {
    let mut n_d = delta.len();
    while n_d > 0 && delta[n_d - 1] == 0 {
        n_d -= 1;
    }
    if n_d == 0 {
        base.resize(keep_limbs, 0);
        return;
    }
    let delta = &delta[..n_d];
    if !delta_neg {
        if base.len() < n_d {
            base.resize(n_d, 0);
        }
        let n_long = base.len();
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add(n_long as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let chunk = 16_384usize;
        let num_chunks = n_d.div_ceil(chunk);
        if max_workers <= 1 || num_chunks <= 1 {
            let mut cy = unsafe {
                (kernels.active.add_n)(base.as_mut_ptr(), base.as_ptr(), delta.as_ptr(), n_d as u64)
            };
            let mut pos = n_d;
            while cy != 0 && pos < base.len() {
                let (s, ov) = base[pos].overflowing_add(cy);
                base[pos] = s;
                cy = u64::from(ov);
                pos += 1;
            }
            if cy != 0 {
                base.push(cy);
            }
        } else {
            let mut carries = vec![0u64; num_chunks];
            let base_ptr = base.as_mut_ptr() as usize;
            let carries_ptr = carries.as_mut_ptr() as usize;
            sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
                let s = ci * chunk;
                let e = (s + chunk).min(n_d);
                let c_out = unsafe {
                    (kernels.active.add_n)(
                        (base_ptr as *mut u64).add(s),
                        (base_ptr as *const u64).add(s),
                        delta.as_ptr().add(s),
                        (e - s) as u64,
                    )
                };
                unsafe {
                    *(carries_ptr as *mut u64).add(ci) = c_out;
                }
            });
            for (ci, &c_in) in carries.iter().enumerate() {
                if c_in == 0 {
                    continue;
                }
                let mut pos = ((ci + 1) * chunk).min(n_d);
                let mut cy = c_in;
                while cy != 0 && pos < base.len() {
                    let (s, ov) = base[pos].overflowing_add(cy);
                    base[pos] = s;
                    cy = u64::from(ov);
                    pos += 1;
                }
                if cy != 0 {
                    base.push(cy);
                }
            }
        }
        base.resize(keep_limbs, 0);
    } else if cmp_limbs(base, delta) == std::cmp::Ordering::Greater {
        sub_limbs_in_place_par(kernels, base, delta, sched, max_workers);
        base.resize(keep_limbs, 0);
    } else {
        base.clear();
        base.resize(keep_limbs, 0);
    }
}

/// Compute `u_step = floor((a_slice * z_prev) >> (64 * shift_down_limbs))` (`step_m` limbs),
/// using wrapped NTT (`~step_m + 6` limbs) when large and overlap with `u_prev` >= 24 bits,
/// returning `(u_step, Some(fwd(z_prev)))` when `fwd(z_prev)` can be reused by Step C!
#[allow(clippy::too_many_arguments)]
fn compute_az_step(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    point_id: u64,
    a_slice: &[u64],
    a_slice_w0: u64,
    z_prev: &[u64],
    z_prev_w0: Option<u64>,
    u_prev: &[u64],
    u_prev_w0: Option<u64>,
    cur_m: usize,
    step_m: usize,
    acc_bits: u64,
    shift_down_limbs: usize,
    is_last: bool,
) -> Result<(LimbVec, Option<NttCachedFwd>, Option<u64>, Option<u64>), MulError> {
    let total_prod_limbs = a_slice.len() + z_prev.len();
    let whole_m = plan.select_mul_method(
        (a_slice.len() as u64) * 64,
        (z_prev.len() as u64) * 64,
        false,
        false,
        shares.active_workers,
        shares,
        cal,
    );
    let can_wrap = matches!(whole_m, MulMethod::Ntt(_))
        && step_m >= 512
        && acc_bits >= 160
        && total_prod_limbs > step_m + 6;

    if can_wrap {
        // Choose wrapped NTT length `L >= 64 * (step_m + 8)` such that `64 * total_prod_limbs - L <= acc_bits - 24`!
        let min_l_from_acc = ((total_prod_limbs as u64) * 64).saturating_sub(acc_bits - 24);
        let min_l = ((step_m as u64 + 8) * 64).max(min_l_from_acc);
        if min_l + 192 < (total_prod_limbs as u64) * 64 {
            let mut spec = choose_wrapped_ntt_spec(
                min_l,
                (a_slice.len().max(z_prev.len()) as u64) * 64,
                shares.active_workers,
                shares,
                cal,
                plan.ram_transform_budget,
                plan.overrides.max_ntt_primes,
            );
            let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
            if l_bits >= min_l && l_bits < (total_prod_limbs as u64) * 64 {
                spec.wrap_cyclic = true;
                let grid_bytes = (spec.k_primes as u64)
                    .saturating_mul(spec.rows as u64)
                    .saturating_mul(spec.outer_row_bytes() as u64);
                let est_rho_limbs = step_m.saturating_sub((acc_bits as usize) / 64).max(2);
                let est_corr_bits = ((cur_m + est_rho_limbs + 2) as u64) * 64;
                let step_c_would_reuse = if est_corr_bits <= l_bits {
                    match plan.select_mul_method(
                        (cur_m as u64) * 64,
                        (est_rho_limbs as u64) * 64,
                        false,
                        false,
                        shares.active_workers,
                        shares,
                        cal,
                    ) {
                        MulMethod::Ntt(cs) => {
                            let fz_w = spec.n.saturating_mul(spec.k_primes);
                            let cs_w = cs.n.saturating_mul(cs.k_primes);
                            if is_last {
                                2 * fz_w <= 3 * cs_w
                            } else {
                                4 * fz_w <= 5 * cs_w
                            }
                        }
                        _ => false,
                    }
                } else {
                    false
                };
                let single_p_bytes =
                    (spec.rows as u64).saturating_mul(spec.outer_row_bytes() as u64);
                let keep_fz_need = if is_last {
                    grid_bytes.saturating_mul(2).saturating_add(single_p_bytes)
                } else {
                    grid_bytes.saturating_mul(3).saturating_add(single_p_bytes)
                };
                let keep_fz = is_last
                    && step_c_would_reuse
                    && !plan.out_of_core
                    && !spec.out_of_core
                    && ctx.fault_hook.is_none()
                    && keep_fz_need <= plan.ram_transform_budget;

                if keep_fz {
                    let fz_res = if (spec.n as u64) * 8
                        > shares.effective_l3_bytes(1, spec.rows) / 2
                    {
                        crate::ntt::ntt_forward(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            z_prev,
                            spec,
                        )
                    } else {
                        crate::ntt::ntt_forward_reusable(
                            ctx.kernels,
                            ctx.ntt_cache,
                            ctx.sched,
                            ctx.store,
                            z_prev,
                            spec,
                        )
                    };
                    if let Ok(mut fz) = fz_res {
                        let w_res = ctx.mul_wrapped_with_cached_reuse(
                            point_id,
                            a_slice,
                            &mut fz,
                            shares.active_workers.max(1),
                        );
                        if let Ok((mut wrapped_limbs, w_w0, _)) = w_res {
                            let z_w0 = z_prev_w0.unwrap_or_else(|| {
                                compute_limbs_w0_par(
                                    ctx.kernels,
                                    ctx.sched,
                                    z_prev,
                                    shares.active_workers,
                                )
                            });
                            let ext_opt = extract_az_from_wrapped(
                                ctx.kernels,
                                ctx.sched,
                                shares.active_workers.max(1),
                                &mut wrapped_limbs,
                                w_w0,
                                l_bits,
                                u_prev,
                                u_prev_w0,
                                ((total_prod_limbs - cur_m) as u64) * 64,
                                shift_down_limbs,
                                step_m,
                                mod61_mul(a_slice_w0, z_w0),
                            );
                            crate::limb::put_limb_buf(wrapped_limbs);
                            if let Some((u_out, u_w0)) = ext_opt {
                                return Ok((u_out, Some(fz), Some(z_w0), u_w0));
                            }
                        }
                        fz.release_store(ctx.store);
                    }
                } else if let Ok((mut wrapped_limbs, w_w0, _)) =
                    ctx.mul_wrapped(point_id, spec, a_slice, z_prev)
                {
                    let z_w0 = z_prev_w0.unwrap_or_else(|| {
                        compute_limbs_w0_par(
                            ctx.kernels,
                            ctx.sched,
                            z_prev,
                            shares.active_workers,
                        )
                    });
                    let ext_opt = extract_az_from_wrapped(
                        ctx.kernels,
                        ctx.sched,
                        shares.active_workers.max(1),
                        &mut wrapped_limbs,
                        w_w0,
                        l_bits,
                        u_prev,
                        u_prev_w0,
                        ((total_prod_limbs - cur_m) as u64) * 64,
                        shift_down_limbs,
                        step_m,
                        mod61_mul(a_slice_w0, z_w0),
                    );
                    crate::limb::put_limb_buf(wrapped_limbs);
                    if let Some((u_out, u_w0)) = ext_opt {
                        return Ok((u_out, None, Some(z_w0), u_w0));
                    }
                }
            }
        }
    }

    // Whole product path (used for small steps and as exact Rung-2 fallback):
    let mut ba = BigNum::from_limbs(a_slice.to_vec(), false, 0, ctx.kernels);
    let mut bz = BigNum::from_limbs(z_prev.to_vec(), false, 0, ctx.kernels);
    let paz = ctx.mul_whole(point_id, &whole_m, &mut ba, &mut bz)?;
    let mut u_out = if paz.limbs.len() > shift_down_limbs {
        crate::limb::slice_to_limb_vec(&paz.limbs[shift_down_limbs..])
    } else {
        crate::limb::empty_limb_vec()
    };
    crate::limb::put_limb_buf(paz.into_limbs());
    u_out.resize(step_m, 0);
    Ok((u_out, None, None, None))
}

#[inline]
fn bit_len_limbs(limbs: &[u64]) -> u64 {
    for (idx, &w) in limbs.iter().enumerate().rev() {
        if w != 0 {
            return (idx as u64) * 64 + (64 - w.leading_zeros() as u64);
        }
    }
    0
}

#[inline]
fn extract_u64_at_bit(limbs: &[u64], bit_pos: u64) -> u64 {
    let w_idx = (bit_pos >> 6) as usize;
    if w_idx >= limbs.len() {
        return 0;
    }
    let b_off = (bit_pos & 63) as u32;
    if b_off == 0 {
        limbs[w_idx]
    } else {
        let lo = limbs[w_idx] >> b_off;
        let hi = if w_idx + 1 < limbs.len() {
            limbs[w_idx + 1] << (64 - b_off)
        } else {
            0
        };
        lo | hi
    }
}

/// `u = A * z` cut to the step from its wrapped product: `az_from_wrapped_unmoved` where that
/// takes the case, which also gives the check word of `u`, else `extract_az_from_wrapped_by_parts`.
#[allow(clippy::too_many_arguments)]
fn extract_az_from_wrapped<A: Allocator>(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    wrapped_limbs: &mut Vec<u64, A>,
    wrapped_w0: u64,
    l_bits: u64,
    u_prev: &[u64],
    u_prev_w0: Option<u64>,
    approx_shift_bits: u64,
    shift_down_limbs: usize,
    step_m: usize,
    expected_whole_w0: u64,
) -> Option<(LimbVec, Option<u64>)> {
    if let Some((u_out, u_w0)) = az_from_wrapped_unmoved(
        kernels,
        sched,
        max_workers,
        wrapped_limbs,
        wrapped_w0,
        l_bits,
        u_prev,
        u_prev_w0,
        approx_shift_bits,
        shift_down_limbs,
        step_m,
        expected_whole_w0,
    ) {
        return Some((u_out, Some(u_w0)));
    }
    extract_az_from_wrapped_by_parts(
        kernels,
        sched,
        max_workers,
        wrapped_limbs,
        wrapped_w0,
        l_bits,
        u_prev,
        approx_shift_bits,
        shift_down_limbs,
        step_m,
        expected_whole_w0,
    )
    .map(|u_out| (u_out, None))
}

/// The usual case of `extract_az_from_wrapped_by_parts` with nothing moved that need not be,
/// and with the check word of the `u` it writes.
///
/// The folded part `p_hi` is the limbs of `u_prev` from one limb up (the transform's length and
/// the shift of `u_prev` are both whole limbs), so it is read where it lies and not copied out
/// first. The unfolded part is `w - p_hi`, of which only the limbs from `shift_down_limbs` up
/// are kept, and `p_hi` ends below those: they are the limbs of `w` itself unless a borrow runs
/// through every limb between, which is left to the long way. So `w` is not subtracted from at
/// all. That leaves one pass that writes `u`: the top of `w` below, `p_hi` above, its check
/// word taken a chunk at a time while the chunk is in cache. `None` for every other case, with
/// nothing changed.
#[allow(clippy::too_many_arguments)]
fn az_from_wrapped_unmoved(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    wrapped: &[u64],
    wrapped_w0: u64,
    l_bits: u64,
    u_prev: &[u64],
    u_prev_w0: Option<u64>,
    approx_shift_bits: u64,
    shift_down_limbs: usize,
    step_m: usize,
    expected_whole_w0: u64,
) -> Option<(LimbVec, u64)> {
    const CHUNK: usize = 16_384;
    if l_bits % 64 != 0
        || l_bits < approx_shift_bits + 64
        || (l_bits - approx_shift_bits) % 64 != 0
    {
        return None;
    }
    let l_limbs = (l_bits / 64) as usize;
    let ws = ((l_bits - approx_shift_bits) / 64) as usize;
    if ws >= u_prev.len()
        || wrapped.len() > l_limbs
        || wrapped.len() <= shift_down_limbs
        || shift_down_limbs + 1 >= l_limbs
    {
        return None;
    }
    let mut s_len = u_prev.len() - ws;
    while s_len > 0 && u_prev[ws + s_len - 1] == 0 {
        s_len -= 1;
    }
    if s_len == 0 || (wrapped_w0 == 0 && wrapped.iter().all(|&x| x == 0)) {
        return None;
    }
    let s_hi = &u_prev[ws..(ws + s_len)];
    // Its word: what is left of the word of `u_prev` without the few limbs below it, moved
    // down by them (`2^61 = 1` modulo the word's modulus), or a pass over it.
    let s_hi_w0 = match u_prev_w0 {
        Some(u_w0) => mod61_mul(
            mod61_sub(u_w0, compute_limbs_w0(kernels, &u_prev[..ws])),
            mod61_pow2(61 - ((ws as u64) * 64) % 61),
        ),
        None => compute_limbs_w0_par(kernels, sched, s_hi, max_workers),
    };

    // The folded part is what `u_prev` says, or one more or less: the check word tells which.
    let folded_is = |phi_w0: u64| -> bool {
        wrapped_product_expected_w0(wrapped_w0, phi_w0, l_bits, false) == expected_whole_w0
    };
    let delta_hi: i128 = if folded_is(s_hi_w0) {
        0
    } else if folded_is(mod61_sub(s_hi_w0, 1)) {
        -1
    } else if folded_is(mod61_add(s_hi_w0, 1)) {
        1
    } else {
        return None;
    };
    let owned_phi = match delta_hi {
        -1 => Some(sub_limbs_par(kernels, s_hi, &[1], sched, max_workers)),
        1 => Some(add_limbs_par(kernels, s_hi, &[1], sched, max_workers)),
        _ => None,
    };
    let give_back = |owned: Option<LimbVec>| {
        if let Some(buf) = owned {
            crate::limb::put_limb_buf(buf);
        }
    };
    let p_hi: &[u64] = owned_phi.as_deref().unwrap_or(s_hi);
    let mut plen = p_hi.len();
    while plen > 0 && p_hi[plen - 1] == 0 {
        plen -= 1;
    }
    let w_off = l_limbs - shift_down_limbs;
    let lo_len = wrapped.len() - shift_down_limbs;
    // `p_hi` has to end below the part kept, `w - p_hi` must not be negative, and a borrow out
    // of the limbs `p_hi` covers has to die out before the part kept.
    let usual = plen > 0
        && plen <= shift_down_limbs
        && w_off + plen <= step_m
        && (cmp_limbs(&wrapped[..plen], &p_hi[..plen]) != std::cmp::Ordering::Less
            || wrapped[plen..shift_down_limbs].iter().any(|&x| x != 0));
    // What `u_prev` holds below the folded part and what `w` has on top lie close together.
    let top_p_lo = wrapped.get(l_limbs - 1).copied().unwrap_or(0);
    let top_approx_lo = if ws >= 1 { u_prev[ws - 1] } else { 0 };
    let signed_top_diff = (delta_hi << 64) + (top_p_lo as i128) - (top_approx_lo as i128);
    if !usual || ws == 0 || signed_top_diff.abs() >= (1i128 << 52) {
        give_back(owned_phi);
        return None;
    }

    let mut u_out = crate::limb::take_limb_buf_uninit(step_m);
    let out_ptr = u_out.as_mut_ptr() as usize;
    let num_chunks = step_m.div_ceil(CHUNK);
    let mut chunk_w0s = vec![0u64; num_chunks];
    let w0_ptr = chunk_w0s.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * CHUNK;
        let end = (start + CHUNK).min(step_m);
        unsafe {
            let dst = (out_ptr as *mut u64).add(start);
            // In turn: limbs of `w`, zeros up to where `p_hi` begins, `p_hi`, zeros above it.
            let mut at = start;
            let from_w = end.min(lo_len);
            if at < from_w {
                std::ptr::copy_nonoverlapping(
                    wrapped.as_ptr().add(shift_down_limbs + at),
                    dst.add(at - start),
                    from_w - at,
                );
                at = from_w;
            }
            let gap_end = end.min(w_off);
            if at < gap_end {
                std::ptr::write_bytes(dst.add(at - start), 0, gap_end - at);
                at = gap_end;
            }
            let from_hi = end.min(w_off + plen);
            if at < from_hi {
                std::ptr::copy_nonoverlapping(
                    p_hi.as_ptr().add(at - w_off),
                    dst.add(at - start),
                    from_hi - at,
                );
                at = from_hi;
            }
            if at < end {
                std::ptr::write_bytes(dst.add(at - start), 0, end - at);
            }
            let mut tmp = [0u64; 3];
            let cw0 = (kernels.active.check_n)(dst as *const u64, (end - start) as u64, &mut tmp, 0);
            *(w0_ptr as *mut u64).add(ci) = cw0;
        }
    });
    unsafe { u_out.set_len(step_m) };
    give_back(owned_phi);
    let step_shift = mod61_pow2((CHUNK as u64) * 3);
    let mut shift_acc = 1u64;
    let mut u_w0 = 0u64;
    for &cw0 in &chunk_w0s {
        u_w0 = mod61_add(u_w0, mod61_mul(cw0, shift_acc));
        shift_acc = mod61_mul(shift_acc, step_shift);
    }
    Some((u_out, u_w0))
}

/// Reconstruct `u_out = ((P_lo + P_hi * 2^L) >> (64 * shift_down_limbs))[0..step_m]` directly
/// from `W ≡ P (mod 2^L - 1)` and `approx = u_prev << approx_shift_bits` without allocating
/// full-length `1.5 * step_m` temporaries, verifying `w0` and proximity `< 2^{L - 12}` (Section 11).
#[allow(clippy::too_many_arguments)]
fn extract_az_from_wrapped_by_parts<A: Allocator>(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    wrapped_limbs: &mut Vec<u64, A>,
    wrapped_w0: u64,
    l_bits: u64,
    u_prev: &[u64],
    approx_shift_bits: u64,
    shift_down_limbs: usize,
    step_m: usize,
    expected_whole_w0: u64,
) -> Option<LimbVec> {
    let s_hi = if l_bits >= approx_shift_bits {
        shr_limbs_par(u_prev, l_bits - approx_shift_bits, sched, max_workers)
    } else {
        shl_limbs_par(u_prev, approx_shift_bits - l_bits, sched, max_workers)
    };

    let top_bit = l_bits.saturating_sub(64);
    let top_approx_lo = if top_bit >= approx_shift_bits {
        extract_u64_at_bit(u_prev, top_bit - approx_shift_bits)
    } else if top_bit + 64 <= approx_shift_bits {
        0
    } else {
        u_prev.first().copied().unwrap_or(0) << (approx_shift_bits - top_bit)
    };

    let s_hi_w0 = compute_limbs_w0_par(kernels, sched, &s_hi, max_workers);
    let mut cand_w0s: [(i128, u64); 3] = [(0, s_hi_w0), (1, mod61_add(s_hi_w0, 1)), (0, 0)];
    let num_cands = if !s_hi.is_empty() {
        cand_w0s[1] = (-1, mod61_sub(s_hi_w0, 1));
        cand_w0s[2] = (1, mod61_add(s_hi_w0, 1));
        3usize
    } else {
        2usize
    };

    let all_wrap_zero = wrapped_w0 == 0 && wrapped_limbs.iter().all(|&x| x == 0);

    for &(delta_hi, phi_w0) in &cand_w0s[..num_cands] {
        for carry_zeros in [false, true] {
            if wrapped_product_expected_w0(wrapped_w0, phi_w0, l_bits, carry_zeros)
                == expected_whole_w0
            {
                let owned_phi = if delta_hi == -1 {
                    Some(sub_limbs_par(kernels, &s_hi, &[1], sched, max_workers))
                } else if delta_hi == 1 {
                    Some(add_limbs_par(kernels, &s_hi, &[1], sched, max_workers))
                } else {
                    None
                };
                let p_hi: &[u64] = owned_phi.as_deref().unwrap_or(&s_hi);
                let orig_wrap_len = wrapped_limbs.len();
                let mut mutated_wrap = false;
                let mut p_lo_owned: Option<LimbVec> = None;
                if carry_zeros && all_wrap_zero {
                    p_lo_owned = Some(complement_2l_minus_1(p_hi, l_bits, sched, max_workers));
                } else if cmp_limbs(wrapped_limbs, p_hi) != std::cmp::Ordering::Less {
                    sub_limbs_in_place_par(kernels, wrapped_limbs, p_hi, sched, max_workers);
                    mutated_wrap = true;
                } else {
                    let d = sub_limbs_par(kernels, p_hi, wrapped_limbs, sched, max_workers);
                    let res = complement_2l_minus_1(&d, l_bits, sched, max_workers);
                    crate::limb::put_limb_buf(d);
                    p_lo_owned = Some(res);
                }
                let p_lo_ref: &[u64] = p_lo_owned.as_deref().unwrap_or(wrapped_limbs.as_slice());

                // Proximity check `|(delta_hi * 2^L + p_lo - approx_lo)| < 2^{L - 12}` in O(1):
                let top_p_lo = extract_u64_at_bit(p_lo_ref, top_bit);
                let signed_top_diff =
                    (delta_hi << 64) + (top_p_lo as i128) - (top_approx_lo as i128);
                if signed_top_diff.abs() < (1i128 << 52) {
                    // Since `0 <= p_lo < 2^L`, `p_lo` and `p_hi << l_bits` have disjoint bits!
                    let shift_down_bits = (shift_down_limbs as u64) * 64;
                    // One parallel pass writes `(p_lo >> shift_down_bits) | (p_hi << rel_bit)`,
                    // cut to `step_m` limbs.
                    let hi_above = l_bits >= shift_down_bits;
                    let rel_bit = l_bits.saturating_sub(shift_down_bits);
                    let w_off = (rel_bit >> 6) as usize;
                    let b_off = (rel_bit & 63) as u32;
                    let lo_len = p_lo_ref.len().saturating_sub(shift_down_limbs);
                    let mut u_out = crate::limb::take_limb_buf_uninit(step_m);
                    let out_ptr = u_out.as_mut_ptr() as usize;
                    let chunk = 16_384usize;
                    sched.par_for_leaf_max(step_m.div_ceil(chunk), 1, max_workers, |ci| {
                        let start = ci * chunk;
                        let end = (start + chunk).min(step_m);
                        for i in start..end {
                            let mut w = if i < lo_len {
                                p_lo_ref[i + shift_down_limbs]
                            } else {
                                0
                            };
                            if hi_above && i >= w_off {
                                let j = i - w_off;
                                if j < p_hi.len() {
                                    w |= p_hi[j] << b_off;
                                }
                                if b_off != 0 && j >= 1 && j <= p_hi.len() {
                                    w |= p_hi[j - 1] >> (64 - b_off);
                                }
                            }
                            unsafe { (out_ptr as *mut u64).add(i).write(w) };
                        }
                    });
                    unsafe { u_out.set_len(step_m) };
                    if let Some(buf) = p_lo_owned {
                        crate::limb::put_limb_buf(buf);
                    }
                    if !hi_above {
                        let hi_shr = shr_limbs(p_hi, shift_down_bits - l_bits);
                        let n = hi_shr.len().min(step_m);
                        for i in 0..n {
                            u_out[i] |= hi_shr[i];
                        }
                    }
                    if let Some(buf) = owned_phi {
                        crate::limb::put_limb_buf(buf);
                    }
                    crate::limb::put_limb_buf(s_hi);
                    let _ = shl_limbs;
                    let _ = compute_limbs_w0;
                    return Some(u_out);
                }
                if mutated_wrap {
                    apply_signed_delta_in_place_par(
                        kernels,
                        sched,
                        max_workers,
                        wrapped_limbs,
                        p_hi,
                        false,
                        orig_wrap_len,
                    );
                    trim_limbs(wrapped_limbs);
                }
                if let Some(buf) = p_lo_owned {
                    crate::limb::put_limb_buf(buf);
                }
                if let Some(buf) = owned_phi {
                    crate::limb::put_limb_buf(buf);
                }
            }
        }
    }
    crate::limb::put_limb_buf(s_hi);
    None
}

/// How far down `exact_mul_carry_into` looks for the limb that settles a carry.
const CARRY_SEARCH_LIMBS: usize = 4_096;

/// The carry into limb `t` of `w * factor`, that is `floor(w[..t] * factor / 2^(64 t))`, found
/// from the limbs below `t` that decide it. The carry out of a limb depends on the carry into it
/// only where the limb's product comes out within `factor` of all ones or of zero, so one limb
/// settles it nearly always; a stretch of the product that is all ones or all zeros leaves it
/// open down to the stretch's end. `None` where `CARRY_SEARCH_LIMBS` limbs do not settle it.
fn exact_mul_carry_into(w: &[u64], t: usize, factor: u64) -> Option<u64> {
    if t == 0 || factor == 0 {
        return Some(0);
    }
    let limb = |i: usize| -> u64 { w.get(i).copied().unwrap_or(0) };
    let lowest = t.saturating_sub(CARRY_SEARCH_LIMBS);
    let mut j = t - 1;
    let mut c = loop {
        let p = (limb(j) as u128) * (factor as u128);
        let least = (p >> 64) as u64;
        if j == 0 {
            break least;
        }
        let most = ((p + (factor as u128 - 1)) >> 64) as u64;
        if least == most {
            break least;
        }
        if j == lowest {
            return None;
        }
        j -= 1;
    };
    for i in (j + 1)..t {
        c = (((limb(i) as u128) * (factor as u128) + (c as u128)) >> 64) as u64;
    }
    Some(c)
}

/// The carry out of the top of `w * factor`, `w` of `l_limbs` limbs, where the top of the
/// product is one long stretch of ones or of zeros, and whether it is zeros.
///
/// A wrapped square of the finish has such a top wherever its transform is longer than the step
/// needs, for thousands to millions of limbs, and inside the stretch no limb settles the carry
/// into the next. Its lowest limb does: the carry into it is settled from below, and the limb
/// then comes out as all ones or as zero. The stretch is found by halving, on the test that a
/// limb's own product lies within `factor` of all ones or of zero. A limb outside the stretch
/// passes that once in 10^15, except the limb the number below the stretch ends in, which
/// passes whenever what it holds is below `factor`; so the limbs are followed upward from the
/// one found until one comes out as all ones or as zero. With the kind of stretch known, the
/// top limb gives the carry. `None` where the top is no such stretch above `lowest`. The caller
/// checks the carry against the one its pass ends with.
fn carry_out_of_quiet_top(
    w: &[u64],
    l_limbs: usize,
    lowest: usize,
    factor: u64,
) -> Option<(u64, bool)> {
    if w.len() != l_limbs || l_limbs < 2 || lowest + 1 >= l_limbs {
        return None;
    }
    let quiet = |i: usize| -> bool {
        let lo = ((w[i] as u128) * (factor as u128)) as u64;
        lo == 0 || lo >= 0u64.wrapping_sub(factor)
    };
    if !quiet(l_limbs - 1) || quiet(lowest) {
        return None;
    }
    let (mut below, mut bottom) = (lowest, l_limbs - 1);
    while bottom - below > 1 {
        let mid = below + (bottom - below) / 2;
        if quiet(mid) {
            bottom = mid;
        } else {
            below = mid;
        }
    }
    let mut c = exact_mul_carry_into(w, bottom, factor)?;
    let mut i = bottom;
    let zeros = loop {
        let p = (w[i] as u128) * (factor as u128) + (c as u128);
        match p as u64 {
            u64::MAX => break false,
            0 => break true,
            _ => {}
        }
        c = (p >> 64) as u64;
        i += 1;
        if i >= l_limbs || i > bottom + 8 {
            return None;
        }
    };
    let top = (w[l_limbs - 1] as u128) * (factor as u128);
    if zeros {
        Some((((top + (factor as u128 - 1)) >> 64) as u64, true))
    } else {
        Some(((top >> 64) as u64, false))
    }
}

/// What `rho_from_wrapped_square` made of the wrapped square it was given.
enum RhoFromSquare {
    /// The residual: its sign, its limbs, in the square's buffer, and their check word.
    Done(bool, LimbVec, u64),
    /// Not a case it takes. The square is as it was.
    NotTaken(LimbVec),
    /// A case it does not take showed only once the square was written over.
    Spoiled,
}

/// `rho = 1 - factor * u^2` from the wrapped square `w = u^2 mod (2^L - 1)`, in one pass over the
/// third of `w` that `rho` comes from.
///
/// `factor * w mod (2^L - 1)` is the wrapped form of `factor * u^2`, and `rho` is its limbs from
/// `step_m` up, or their complement. Multiplying all of `w` and then taking that part moves three
/// times what the part alone does. Nothing below `step_m` is needed but two carries, and those
/// are settled by a few limbs each (`exact_mul_carry_into`): the one into `step_m`, and the one
/// out of the top, which folds in at the bottom and dies out there. The check word of the whole
/// follows from the word the product's kernel returned for `w`, as it does for `A * z`:
/// `factor * w - q (2^L - 1)`, `q` the carry out of the top. It says, before anything is
/// written, which of the two folded parts the whole has and so whether the limbs are kept or
/// complemented. The limbs kept get their own word from this pass. Any case but the usual one
/// is left to the two passes of before.
#[allow(clippy::too_many_arguments)]
fn rho_from_wrapped_square(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    w: LimbVec,
    w_w0: u64,
    exp_whole_w0: u64,
    factor: u64,
    step_m: usize,
    l_bits: u64,
    full_bits_1: u64,
) -> RhoFromSquare {
    const CHUNK: usize = 8_192;
    let l_limbs = (l_bits / 64) as usize;
    let s_bits = (step_m as u64) * 64;
    if max_workers <= 1
        || l_bits % 64 != 0
        || l_bits >= full_bits_1
        || factor < 2
        || factor >= (1u64 << 32)
        || w.len() > l_limbs
        || w.len() <= step_m + 8
    {
        return RhoFromSquare::NotTaken(w);
    }
    let rem_bits = full_bits_1 - l_bits;
    let out_len = l_limbs - step_m;
    if rem_bits >= s_bits
        || l_bits <= s_bits + 32
        || out_len < 32_768
        || out_len > step_m
        || (rem_bits / 64) as usize + 2 > step_m
    {
        return RhoFromSquare::NotTaken(w);
    }

    // The carry out of the top of `factor * w`, and the limb below the part kept, which must not
    // be zero: the low part is then not below `2^rem`, the one case of it that changes `rho`.
    let (q, zeros_on_top) = match exact_mul_carry_into(&w, l_limbs, factor) {
        Some(q) => (q, None),
        None => match carry_out_of_quiet_top(&w, l_limbs, step_m, factor) {
            Some((q, zeros)) => (q, Some(zeros)),
            None => return RhoFromSquare::NotTaken(w),
        },
    };
    let Some(c_below) = exact_mul_carry_into(&w, step_m - 1, factor) else {
        return RhoFromSquare::NotTaken(w);
    };
    let p_below = (w[step_m - 1] as u128) * (factor as u128) + (c_below as u128);
    if p_below as u64 == 0 {
        return RhoFromSquare::NotTaken(w);
    }
    let carry_into_kept = (p_below >> 64) as u64;
    // `q` folds in at the bottom. It has to die out within eight limbs, and they must not all
    // come out as ones (the whole would then be `2^L - 1`, which stands for zero).
    let mut c = q as u128;
    let mut carry_mul = 0u128;
    let mut all_ones = true;
    for &wl in &w[..8] {
        let p = (wl as u128) * (factor as u128) + carry_mul;
        carry_mul = p >> 64;
        let s = ((p as u64) as u128) + c;
        c = s >> 64;
        all_ones &= (s as u64) == u64::MAX;
    }
    if c != 0 || all_ones {
        return RhoFromSquare::NotTaken(w);
    }

    // The check word of `factor * w mod (2^L - 1)`, and by it which of the two folded parts the
    // whole has: all ones, `rho >= 0`, or a one and zeros, `rho < 0`.
    let whole_w0 = mod61_sub(
        mod61_mul(mod61_reduce(factor as u128), w_w0),
        mod61_mul(mod61_reduce(q as u128), mod61_sub(mod61_pow2(l_bits), 1)),
    );
    let pow2_rem_w0 = mod61_pow2(rem_bits);
    let folded_is = |phi_w0: u64| -> bool {
        wrapped_product_expected_w0(whole_w0, phi_w0, l_bits, false) == exp_whole_w0
    };
    let neg = if folded_is(mod61_sub(pow2_rem_w0, 1)) {
        false
    } else if folded_is(pow2_rem_w0) {
        true
    } else {
        return RhoFromSquare::NotTaken(w);
    };
    // A top of zeros goes with a one and zeros folded, a top of ones with all ones.
    if zeros_on_top.is_some_and(|zeros| zeros != neg) {
        return RhoFromSquare::NotTaken(w);
    }

    // One pass: the part kept, times `factor`, complemented where `rho >= 0`, written to the
    // bottom of the buffer, with its check word taken while a chunk is in cache. The chunks read
    // from `step_m` up and write below `out_len <= step_m`. A chunk is made without the carry
    // into it; the carries out are put right afterwards.
    let num_chunks = out_len.div_ceil(CHUNK);
    let mut buf = w;
    let w_len = buf.len();
    let ptr = buf.as_mut_ptr() as usize;
    let mut chunk_w0s = vec![0u64; num_chunks];
    let w0_ptr = chunk_w0s.as_mut_ptr() as usize;
    let mut carries_out = vec![0u64; num_chunks];
    let co_ptr = carries_out.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * CHUNK;
        let len = (start + CHUNK).min(out_len) - start;
        let src_at = step_m + start;
        let have = w_len.saturating_sub(src_at).min(len);
        unsafe {
            let dst = (ptr as *mut u64).add(start);
            let mut top = 0u64;
            if have > 0 {
                top = (kernels.active.mul_1)(
                    dst,
                    (ptr as *const u64).add(src_at),
                    have as u64,
                    factor,
                );
            }
            if have < len {
                dst.add(have).write(top);
                if have + 1 < len {
                    std::ptr::write_bytes(dst.add(have + 1), 0, len - have - 1);
                }
                top = 0;
            }
            *(co_ptr as *mut u64).add(ci) = top;
            let out = std::slice::from_raw_parts_mut(dst, len);
            if !neg {
                for x in out.iter_mut() {
                    *x = !*x;
                }
            }
            let mut tmp = [0u64; 3];
            let cw0 = (kernels.active.check_n)(dst as *const u64, len as u64, &mut tmp, 0);
            *(w0_ptr as *mut u64).add(ci) = cw0;
        }
        if kernels.count_work {
            kernels
                .stats
                .limb_ops
                .fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        }
    });
    buf.truncate(out_len);
    let step_shift = mod61_pow2((CHUNK as u64) * 3);
    let mut shift_acc = 1u64;
    let mut rho_w0 = 0u64;
    for &cw0 in &chunk_w0s {
        rho_w0 = mod61_add(rho_w0, mod61_mul(cw0, shift_acc));
        shift_acc = mod61_mul(shift_acc, step_shift);
    }

    // The carry into the part kept and the carry out of every chunk, each added where it
    // belongs: to the limbs as they are, or, complemented, taken from them. It seldom runs far.
    // What leaves the top is the carry that was folded in at the bottom.
    let mut top_carry = carries_out[num_chunks - 1];
    for fix in 0..num_chunks {
        let (mut idx, mut c) = if fix == 0 {
            (0usize, carry_into_kept)
        } else {
            (fix * CHUNK, carries_out[fix - 1])
        };
        while c != 0 && idx < out_len {
            let old = buf[idx];
            let (v, over) = if neg { old.overflowing_add(c) } else { old.overflowing_sub(c) };
            w0_delta_limb(&mut rho_w0, idx, old, v);
            buf[idx] = v;
            c = u64::from(over);
            idx += 1;
        }
        top_carry += c;
    }
    // `rho` lies within the step's precision of zero: its top thirteen bits are empty.
    if top_carry != q || buf[out_len - 1] >> 51 != 0 {
        crate::limb::put_limb_buf(buf);
        return RhoFromSquare::Spoiled;
    }
    trim_limbs(&mut buf);
    let rho_neg = neg && !buf.is_empty();
    RhoFromSquare::Done(rho_neg, buf, rho_w0)
}

/// Compute `rho = 1 - 10005 * u^2` using wrapped squaring at `~1.5x` step length when transform-sized
/// (Section 3 & 11: folded part is all ones or a one and zeros), returning `(rho_neg, rho_limbs, rho_frac_limbs)`.
#[allow(clippy::too_many_arguments)]
fn compute_rho_step(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
    point_id: u64,
    u_step: &[u64],
    u_step_w0: Option<u64>,
    step_m: usize,
    acc_bits: u64,
) -> Result<(bool, LimbVec, usize, Option<u64>), MulError> {
    let full_bits_1 = (2 * step_m) as u64 * 64; // `1` is at `2^{128 * step_m}`
    let u_bits_est = ((u_step.len().max(1)) as u64) * 64;
    let whole_sq_m = plan.select_mul_method(
        u_bits_est,
        u_bits_est,
        false,
        false,
        shares.active_workers,
        shares,
        cal,
    );
    let can_wrap = matches!(whole_sq_m, MulMethod::Ntt(_)) && step_m >= 512 && acc_bits >= 160;

    if can_wrap {
        // `|2^{128*step_m} - 10005*U^2| < 2^{128*step_m - acc_bits + 8}`.
        // Choose `L >= 128*step_m - acc_bits + 32` (~1.5x step length!).
        let min_l = full_bits_1.saturating_sub(acc_bits - 32);
        if min_l + 192 < full_bits_1 {
            let spec = choose_wrapped_ntt_spec(
                min_l,
                (step_m as u64) * 64 + 16,
                shares.active_workers,
                shares,
                cal,
                plan.ram_transform_budget,
                plan.overrides.max_ntt_primes,
            );
            let l_bits = (spec.n as u64) * (spec.coeff_bits as u64);
            if l_bits >= min_l && l_bits < full_bits_1 {
                if let Ok((w_sq, w_sq_w0, _)) = ctx.mul_wrapped(point_id, spec, u_step, u_step) {
                    let max_w = shares.active_workers.max(1);
                    let u_w0 = u_step_w0.unwrap_or_else(|| {
                        compute_limbs_w0_par(ctx.kernels, ctx.sched, u_step, max_w)
                    });
                    let exp_whole_w0 = mod61_mul(
                        mod61_reduce(FINISH_D_SQR as u128),
                        mod61_mul(u_w0, u_w0),
                    );
                    // The usual case takes one pass over the part of the square that is kept.
                    // Under a fault hook the square goes the long way, which reads every limb.
                    let mut w_du2 = if ctx.fault_hook.is_some() {
                        w_sq
                    } else {
                        match rho_from_wrapped_square(
                            ctx.kernels,
                            ctx.sched,
                            max_w,
                            w_sq,
                            w_sq_w0,
                            exp_whole_w0,
                            FINISH_D_SQR,
                            step_m,
                            l_bits,
                            full_bits_1,
                        ) {
                            RhoFromSquare::Done(rho_neg, rho_out, rho_w0) => {
                                return Ok((rho_neg, rho_out, step_m, Some(rho_w0)));
                            }
                            RhoFromSquare::NotTaken(w) => w,
                            RhoFromSquare::Spoiled => {
                                ctx.mul_wrapped(point_id, spec, u_step, u_step)?.0
                            }
                        }
                    };
                    let w_du2_w0 = mul_u64_mod_2l_minus_1_in_place(
                        ctx.kernels,
                        ctx.sched,
                        max_w,
                        &mut w_du2,
                        FINISH_D_SQR,
                        l_bits,
                    );

                    let rem_bits = full_bits_1 - l_bits;
                    let s_bits = (step_m as u64) * 64;
                    let pow2_rem_w0 = mod61_pow2(rem_bits);
                    let all_ones_w0 = mod61_sub(pow2_rem_w0, 1);
                    let all_w_zero = w_du2_w0 == 0 && w_du2.iter().all(|&x| x == 0);

                    for (is_neg_branch, phi_w0) in [(false, all_ones_w0), (true, pow2_rem_w0)] {
                        for carry_z in [false, true] {
                            if wrapped_product_expected_w0(w_du2_w0, phi_w0, l_bits, carry_z)
                                == exp_whole_w0
                            {
                                if rem_bits < s_bits && l_bits > s_bits + 32 {
                                    if let Some((rho_out, rho_w0_opt)) = extract_rho_hi_fast(
                                        ctx.kernels,
                                        &mut w_du2,
                                        step_m,
                                        l_bits,
                                        rem_bits,
                                        is_neg_branch,
                                        carry_z && all_w_zero,
                                        ctx.sched,
                                        max_w,
                                    ) {
                                        crate::limb::put_limb_buf(w_du2);
                                        let rho_neg = is_neg_branch && !rho_out.is_empty();
                                        return Ok((rho_neg, rho_out, step_m, rho_w0_opt));
                                    }
                                    continue;
                                }

                                let p_hi = if is_neg_branch {
                                    make_pow2(rem_bits)
                                } else {
                                    make_2l_minus_1(rem_bits)
                                };
                                let p_lo = if carry_z && all_w_zero {
                                    complement_2l_minus_1(&p_hi, l_bits, ctx.sched, max_w)
                                } else if cmp_limbs(&w_du2, &p_hi) != std::cmp::Ordering::Less {
                                    sub_limbs_par(ctx.kernels, &w_du2, &p_hi, ctx.sched, max_w)
                                } else {
                                    let d = sub_limbs_par(
                                        ctx.kernels,
                                        &p_hi,
                                        &w_du2,
                                        ctx.sched,
                                        max_w,
                                    );
                                    let res = complement_2l_minus_1(&d, l_bits, ctx.sched, max_w);
                                    crate::limb::put_limb_buf(d);
                                    res
                                };
                                crate::limb::put_limb_buf(p_hi);
                                let diff = if !is_neg_branch {
                                    let comp =
                                        complement_2l_minus_1(&p_lo, l_bits, ctx.sched, max_w);
                                    crate::limb::put_limb_buf(p_lo);
                                    let res =
                                        add_limbs_par(ctx.kernels, &comp, &[1], ctx.sched, max_w);
                                    crate::limb::put_limb_buf(comp);
                                    res
                                } else {
                                    p_lo
                                };
                                let dbits = bit_len_limbs(&diff);
                                if dbits + 12 < l_bits {
                                    let mut rho_out =
                                        shr_limbs_par(&diff, s_bits, ctx.sched, max_w);
                                    crate::limb::put_limb_buf(diff);
                                    crate::limb::put_limb_buf(w_du2);
                                    trim_limbs(&mut rho_out);
                                    let rho_neg = is_neg_branch && !rho_out.is_empty();
                                    return Ok((rho_neg, rho_out, step_m, None));
                                }
                                crate::limb::put_limb_buf(diff);
                            }
                        }
                    }
                    crate::limb::put_limb_buf(w_du2);
                }
            }
        }
    }

    let max_w = shares.active_workers.max(1);
    let mut bu = BigNum::from_limbs_par(u_step.to_vec(), false, 0, ctx.kernels, ctx.sched, max_w);
    let sq_m = plan.select_mul_method(
        bu.body_bits(),
        bu.body_bits(),
        false,
        false,
        shares.active_workers,
        shares,
        cal,
    );
    let u2 = ctx.sqr_whole(point_id, &sq_m, &mut bu)?;
    let d_u2 = mul_bignum_u64_par(ctx.kernels, &u2, FINISH_D_SQR, ctx.sched, max_w).into_limbs();

    // `rho = (2^{128 * step_m} - d_u2) * 2^{-128 * step_m}`.
    let one_limbs = make_pow2(full_bits_1);
    let (rho_neg, diff_full) = if cmp_limbs(&one_limbs, &d_u2) != std::cmp::Ordering::Less {
        (false, sub_limbs_par(ctx.kernels, &one_limbs, &d_u2, ctx.sched, max_w))
    } else {
        (true, sub_limbs_par(ctx.kernels, &d_u2, &one_limbs, ctx.sched, max_w))
    };

    // Keep `rho` scaled at `rho_frac_limbs = step_m` (`shr` by `64 * step_m` bits)!
    let mut rho_out = if diff_full.len() > step_m {
        shr_limbs_par(&diff_full, (step_m as u64) * 64, ctx.sched, max_w)
    } else {
        crate::limb::empty_limb_vec()
    };
    crate::limb::put_limb_buf(d_u2);
    crate::limb::put_limb_buf(one_limbs);
    crate::limb::put_limb_buf(diff_full);
    trim_limbs(&mut rho_out);
    let _ = mod61_sub(0, 0);
    let _ = sub_limbs;
    Ok((rho_neg && !rho_out.is_empty(), rho_out, step_m, None))
}

fn make_pow2(bits: u64) -> LimbVec {
    let mut v = crate::limb::take_limb_buf_zeroed((bits / 64 + 1) as usize);
    v[(bits / 64) as usize] = 1u64 << (bits & 63);
    v
}

fn make_2l_minus_1(l_bits: u64) -> LimbVec {
    if l_bits == 0 {
        return empty_limb_vec();
    }
    let full_words = (l_bits / 64) as usize;
    let rem_bits = (l_bits & 63) as u32;
    let total = full_words + usize::from(rem_bits != 0);
    let mut v = crate::limb::take_limb_buf_uninit(total);
    unsafe {
        std::ptr::write_bytes(v.as_mut_ptr(), 0xFF, full_words);
        if rem_bits != 0 {
            v.as_mut_ptr().add(full_words).write((1u64 << rem_bits) - 1);
        }
        v.set_len(total);
    }
    v
}

/// Compute `(2^L - 1) - x` in parallel via bitwise complement without allocating a `2^L - 1` buffer!
fn complement_2l_minus_1(
    x: &[u64],
    l_bits: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    if l_bits == 0 {
        return empty_limb_vec();
    }
    let full_words = (l_bits / 64) as usize;
    let rem_bits = (l_bits & 63) as u32;
    let out_len = full_words + usize::from(rem_bits != 0);
    if max_workers <= 1 || out_len < 32_768 {
        let mut out = crate::limb::take_limb_buf_uninit(out_len);
        for i in 0..full_words {
            out.push(!x.get(i).copied().unwrap_or(0));
        }
        if rem_bits != 0 {
            let mask = (1u64 << rem_bits) - 1;
            out.push(mask & !x.get(full_words).copied().unwrap_or(0));
        }
        trim_limbs(&mut out);
        return out;
    }
    let mut out = crate::limb::take_limb_buf_uninit(out_len);
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = 16_384usize;
    let num_chunks = full_words.div_ceil(chunk);
    if num_chunks > 0 {
        sched.par_for_max(num_chunks, 1, max_workers, |ci| {
            let s = ci * chunk;
            let e = (s + chunk).min(full_words);
            for i in s..e {
                let v = !x.get(i).copied().unwrap_or(0);
                unsafe {
                    (out_ptr as *mut u64).add(i).write(v);
                }
            }
        });
    }
    if rem_bits != 0 {
        let mask = (1u64 << rem_bits) - 1;
        let top = mask & !x.get(full_words).copied().unwrap_or(0);
        unsafe {
            (out_ptr as *mut u64).add(full_words).write(top);
        }
    }
    unsafe {
        out.set_len(out_len);
    }
    trim_limbs(&mut out);
    out
}

#[inline]
fn is_geq_2l_minus_1(limbs: &[u64], l_bits: u64) -> bool {
    let full_words = (l_bits / 64) as usize;
    let rem_bits = (l_bits & 63) as u32;
    let need_len = full_words + usize::from(rem_bits != 0);
    if limbs.len() < need_len {
        return false;
    }
    for &w in &limbs[need_len..] {
        if w != 0 {
            return true;
        }
    }
    if rem_bits != 0 {
        let top_max = (1u64 << rem_bits) - 1;
        if limbs[full_words] > top_max {
            return true;
        }
        if limbs[full_words] < top_max {
            return false;
        }
    }
    limbs[..full_words].iter().all(|&w| w == u64::MAX)
}

fn reduce_limbs_mod_2l_minus_1(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    limbs: &[u64],
    l_bits: u64,
) -> LimbVec {
    let top_w = (l_bits / 64) as usize;
    let top_b = (l_bits & 63) as u32;
    let max_chunk_w = if top_b == 0 { top_w } else { top_w + 1 };

    let copy_n = limbs.len().min(max_chunk_w);
    let mut acc = shr_limbs_par(&limbs[..copy_n], 0, sched, max_workers);
    if top_b != 0 && top_w < acc.len() {
        acc[top_w] &= (1u64 << top_b) - 1;
    }
    trim_limbs(&mut acc);
    while is_geq_2l_minus_1(&acc, l_bits) {
        let modulus = make_2l_minus_1(l_bits);
        acc = sub_limbs_par(kernels, &acc, &modulus, sched, max_workers);
    }

    let total_bits = (limbs.len() as u64) * 64;
    let mut pos = l_bits;
    while pos < total_bits {
        let start_w = (pos / 64) as usize;
        let rem_slice = &limbs[start_w..limbs.len().min(start_w + max_chunk_w + 1)];
        let mut chunk = shr_limbs_par(rem_slice, pos & 63, sched, max_workers);
        if top_w < chunk.len() {
            if top_b == 0 {
                chunk.truncate(top_w);
            } else {
                chunk[top_w] &= (1u64 << top_b) - 1;
                chunk.truncate(top_w + 1);
            }
        }
        trim_limbs(&mut chunk);
        acc = add_limbs_par(kernels, &acc, &chunk, sched, max_workers);
        while is_geq_2l_minus_1(&acc, l_bits) {
            let modulus = make_2l_minus_1(l_bits);
            acc = sub_limbs_par(kernels, &acc, &modulus, sched, max_workers);
        }
        pos += l_bits;
    }
    acc
}

#[inline(always)]
fn w0_delta_limb(w0: &mut u64, idx: usize, old_val: u64, new_val: u64) {
    if old_val == new_val {
        return;
    }
    let shift = mod61_pow2((idx as u64) * 3);
    let old_m = mod61_reduce(old_val as u128);
    let new_m = mod61_reduce(new_val as u128);
    let delta = mod61_sub(new_m, old_m);
    *w0 = mod61_add(*w0, mod61_mul(delta, shift));
}

fn mul_u64_mod_2l_minus_1_in_place(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    buf: &mut LimbVec,
    factor: u64,
    l_bits: u64,
) -> u64 {
    let top_w = (l_bits / 64) as usize;
    let top_b = (l_bits & 63) as u32;
    let l_limbs = top_w + usize::from(top_b != 0);
    if l_bits < 128 || buf.len() > l_limbs || l_limbs < 3 {
        let raw = mul_limbs_u64_par(kernels, buf, factor, sched, max_workers);
        let red = reduce_limbs_mod_2l_minus_1(kernels, sched, max_workers, &raw, l_bits);
        crate::limb::put_limb_buf(raw);
        let old = std::mem::replace(buf, red);
        crate::limb::put_limb_buf(old);
        return compute_limbs_w0_par(kernels, sched, buf, max_workers);
    }

    if buf.len() < l_limbs {
        buf.resize(l_limbs, 0);
    }
    let chunk = 8_192usize;
    let num_chunks = l_limbs.div_ceil(chunk);
    let mut chunk_carries = vec![0u64; num_chunks];
    let mut chunk_w0s = vec![0u64; num_chunks];
    let buf_ptr = buf.as_mut_ptr() as usize;
    let cc_ptr = chunk_carries.as_mut_ptr() as usize;
    let cw_ptr = chunk_w0s.as_mut_ptr() as usize;

    sched.par_for_max(num_chunks, 1, max_workers, |ci| {
        let s = ci * chunk;
        let e = (s + chunk).min(l_limbs);
        let len = e - s;
        unsafe {
            let p = (buf_ptr as *mut u64).add(s);
            let c = (kernels.active.mul_1)(p, p, len as u64, factor);
            if kernels.count_work {
                kernels
                    .stats
                    .limb_ops
                    .fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let w = compute_limbs_w0(kernels, std::slice::from_raw_parts(p, len));
            *(cc_ptr as *mut u64).add(ci) = c;
            *(cw_ptr as *mut u64).add(ci) = w;
        }
    });

    let mut total_w0 = 0u64;
    for (ci, &cw) in chunk_w0s.iter().enumerate() {
        let shift = mod61_pow2(((ci * chunk) as u64) * 3);
        total_w0 = mod61_add(total_w0, mod61_mul(cw, shift));
    }

    let mut carry_past_end = chunk_carries[num_chunks - 1] as u128;
    for ci in 0..(num_chunks - 1) {
        let mut c = chunk_carries[ci];
        let mut idx = (ci + 1) * chunk;
        while c > 0 && idx < l_limbs {
            let old = buf[idx];
            let (s, ov) = old.overflowing_add(c);
            w0_delta_limb(&mut total_w0, idx, old, s);
            buf[idx] = s;
            c = u64::from(ov);
            idx += 1;
        }
        carry_past_end += c as u128;
    }

    let mut top_excess = if top_b == 0 {
        carry_past_end
    } else {
        let old_top = buf[top_w];
        let masked = old_top & ((1u64 << top_b) - 1);
        w0_delta_limb(&mut total_w0, top_w, old_top, masked);
        buf[top_w] = masked;
        ((old_top >> top_b) as u128) + (carry_past_end << (64 - top_b))
    };

    while top_excess > 0 {
        let add0 = top_excess as u64;
        let add1 = (top_excess >> 64) as u64;
        let mut c = 0u64;
        for (idx, addv) in [(0usize, add0), (1usize, add1)] {
            if addv == 0 && c == 0 {
                continue;
            }
            let old = buf[idx];
            let (s1, k1) = old.overflowing_add(addv);
            let (s2, k2) = s1.overflowing_add(c);
            w0_delta_limb(&mut total_w0, idx, old, s2);
            buf[idx] = s2;
            c = u64::from(k1) + u64::from(k2);
        }
        let mut idx = 2usize;
        while c > 0 && idx < l_limbs {
            let old = buf[idx];
            let (s, ov) = old.overflowing_add(c);
            w0_delta_limb(&mut total_w0, idx, old, s);
            buf[idx] = s;
            c = u64::from(ov);
            idx += 1;
        }
        if top_b == 0 {
            top_excess = c as u128;
        } else {
            let old_top = buf[top_w];
            let masked = old_top & ((1u64 << top_b) - 1);
            w0_delta_limb(&mut total_w0, top_w, old_top, masked);
            buf[top_w] = masked;
            top_excess = ((old_top >> top_b) as u128) + ((c as u128) << (64 - top_b));
        }
    }

    if is_geq_2l_minus_1(buf, l_bits) {
        buf.clear();
        total_w0 = 0;
    } else {
        trim_limbs(buf);
    }

    debug_assert_eq!(
        total_w0,
        unsafe {
            let mut tmp = [0u64; 3];
            (kernels.active.check_n)(buf.as_ptr(), buf.len() as u64, &mut tmp, 0)
        }
    );
    total_w0
}

#[allow(clippy::too_many_arguments)]
fn extract_rho_hi_fast(
    kernels: &crate::refk::Kernels,
    w_du2: &mut LimbVec,
    step_m: usize,
    l_bits: u64,
    rem_bits: u64,
    is_neg_branch: bool,
    zero_carry_case: bool,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> Option<(LimbVec, Option<u64>)> {
    let s_bits = (step_m as u64) * 64;
    let hi_bits_len = l_bits - s_bits;
    let r_word = (rem_bits / 64) as usize;
    let r_bit = (rem_bits & 63) as u32;

    let w_hi_nonzero = w_du2.len() > step_m;
    let mut w_lo_lt_pow2_r = true;
    let check_end = step_m.min(w_du2.len());
    if check_end > r_word + 1 {
        for &w in w_du2[(r_word + 1)..check_end].iter().rev() {
            if w != 0 {
                w_lo_lt_pow2_r = false;
                break;
            }
        }
    }
    if w_lo_lt_pow2_r && r_word < w_du2.len() && (w_du2[r_word] >> r_bit) != 0 {
        w_lo_lt_pow2_r = false;
    }

    if !is_neg_branch {
        let w_lt_phi = if w_hi_nonzero || !w_lo_lt_pow2_r {
            false
        } else {
            let r_mask = if r_bit == 0 { 0u64 } else { (1u64 << r_bit) - 1 };
            let r_val = w_du2.get(r_word).copied().unwrap_or(0);
            r_val != r_mask
                || w_du2.len() < r_word
                || w_du2[..r_word].iter().any(|&w| w != u64::MAX)
        };
        if zero_carry_case || w_lt_phi {
            if rem_bits + 13 < l_bits {
                return Some((empty_limb_vec(), Some(0)));
            }
            return None;
        }
        let full_words = (hi_bits_len / 64) as usize;
        let rem_b = (hi_bits_len & 63) as u32;
        let out_len = full_words + usize::from(rem_b != 0);
        if max_workers > 1
            && out_len >= 32_768
            && out_len <= step_m
            && hi_bits_len >= 64
            && w_du2.capacity() > out_len + 1
        {
            let chk_bit = hi_bits_len - 13;
            let cw = (chk_bit / 64) as usize;
            let cb = (chk_bit & 63) as u32;
            let hi_src = if w_du2.len() > step_m {
                &w_du2[step_m..]
            } else {
                &[]
            };
            let mut top_zero = true;
            for idx in cw..out_len {
                let sv = hi_src.get(idx).copied().unwrap_or(0);
                let cv = if idx == full_words && rem_b != 0 {
                    ((1u64 << rem_b) - 1) & !sv
                } else {
                    !sv
                };
                let mv = if idx == cw { cv >> cb } else { cv };
                if mv != 0 {
                    top_zero = false;
                    break;
                }
            }
            if top_zero {
                let mut rho_out = take_limb_vec(w_du2);
                let orig_len = rho_out.len();
                if orig_len < out_len {
                    rho_out.resize(out_len, 0);
                }
                let chunk = 16_384usize;
                let num_chunks = out_len.div_ceil(chunk);
                let mut chunk_w0s = vec![0u64; num_chunks];
                let ptr = rho_out.as_mut_ptr() as usize;
                let w0_ptr = chunk_w0s.as_mut_ptr() as usize;
                let top_mask = if rem_b == 0 {
                    u64::MAX
                } else {
                    (1u64 << rem_b) - 1
                };
                sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
                    let start = ci * chunk;
                    let end = (start + chunk).min(out_len);
                    let dst = unsafe { (ptr as *mut u64).add(start) };
                    for i in start..end {
                        let si = step_m + i;
                        let sv = if si < orig_len {
                            unsafe { *(ptr as *const u64).add(si) }
                        } else {
                            0
                        };
                        let cv = if i == full_words && rem_b != 0 {
                            top_mask & !sv
                        } else {
                            !sv
                        };
                        unsafe { dst.add(i - start).write(cv) };
                    }
                    let cw0 = unsafe {
                        let mut tmp = [0u64; 3];
                        (kernels.active.check_n)(
                            dst as *const u64,
                            (end - start) as u64,
                            &mut tmp,
                            0,
                        )
                    };
                    unsafe { *(w0_ptr as *mut u64).add(ci) = cw0 };
                });
                rho_out.truncate(out_len);
                let step_shift = mod61_pow2((chunk as u64) * 3);
                let mut shift_acc = 1u64;
                let mut total_w0 = 0u64;
                for &cw0 in &chunk_w0s {
                    total_w0 = mod61_add(total_w0, mod61_mul(cw0, shift_acc));
                    shift_acc = mod61_mul(shift_acc, step_shift);
                }
                if w_lo_lt_pow2_r {
                    let mut c = 1u64;
                    for (idx, slot) in rho_out.iter_mut().enumerate() {
                        let old = *slot;
                        let (s, ov) = old.overflowing_add(c);
                        w0_delta_limb(&mut total_w0, idx, old, s);
                        *slot = s;
                        c = u64::from(ov);
                        if c == 0 {
                            break;
                        }
                    }
                    if c > 0 {
                        let idx = rho_out.len();
                        w0_delta_limb(&mut total_w0, idx, 0, c);
                        rho_out.push(c);
                    }
                }
                trim_limbs(&mut rho_out);
                return Some((rho_out, Some(total_w0)));
            }
        }
        let hi_src = if w_du2.len() > step_m {
            &w_du2[step_m..]
        } else {
            &[]
        };
        let mut rho_out = complement_2l_minus_1(hi_src, hi_bits_len, sched, max_workers);
        if w_lo_lt_pow2_r {
            let mut c = 1u64;
            for slot in &mut rho_out {
                let (s, ov) = slot.overflowing_add(c);
                *slot = s;
                c = u64::from(ov);
                if c == 0 {
                    break;
                }
            }
            if c > 0 {
                rho_out.push(c);
            }
        }
        trim_limbs(&mut rho_out);
        let dbits = if rho_out.is_empty() {
            s_bits
        } else {
            s_bits + bit_len_limbs(&rho_out)
        };
        if dbits + 12 < l_bits {
            Some((rho_out, None))
        } else {
            crate::limb::put_limb_buf(rho_out);
            None
        }
    } else {
        if zero_carry_case || (!w_hi_nonzero && w_lo_lt_pow2_r) {
            return None;
        }
        let hi_len = w_du2.len().saturating_sub(step_m);
        if max_workers > 1 && hi_len >= 32_768 && hi_len <= step_m && hi_bits_len >= 64 {
            let chk_bit = hi_bits_len - 13;
            let cw = (chk_bit / 64) as usize;
            let cb = (chk_bit & 63) as u32;
            let mut top_zero = true;
            if hi_len > cw {
                for idx in cw..hi_len {
                    let sv = w_du2[step_m + idx];
                    let mv = if idx == cw { sv >> cb } else { sv };
                    if mv != 0 {
                        top_zero = false;
                        break;
                    }
                }
            }
            if top_zero {
                let mut rho_out = take_limb_vec(w_du2);
                let chunk = 16_384usize;
                let num_chunks = hi_len.div_ceil(chunk);
                let mut chunk_w0s = vec![0u64; num_chunks];
                let ptr = rho_out.as_mut_ptr() as usize;
                let w0_ptr = chunk_w0s.as_mut_ptr() as usize;
                sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
                    let start = ci * chunk;
                    let end = (start + chunk).min(hi_len);
                    let dst = unsafe { (ptr as *mut u64).add(start) };
                    let src = unsafe { (ptr as *const u64).add(step_m + start) };
                    unsafe { std::ptr::copy_nonoverlapping(src, dst, end - start) };
                    let cw0 = unsafe {
                        let mut tmp = [0u64; 3];
                        (kernels.active.check_n)(
                            dst as *const u64,
                            (end - start) as u64,
                            &mut tmp,
                            0,
                        )
                    };
                    unsafe { *(w0_ptr as *mut u64).add(ci) = cw0 };
                });
                rho_out.truncate(hi_len);
                let step_shift = mod61_pow2((chunk as u64) * 3);
                let mut shift_acc = 1u64;
                let mut total_w0 = 0u64;
                for &cw0 in &chunk_w0s {
                    total_w0 = mod61_add(total_w0, mod61_mul(cw0, shift_acc));
                    shift_acc = mod61_mul(shift_acc, step_shift);
                }
                if w_lo_lt_pow2_r {
                    let mut b = 1u64;
                    for (idx, slot) in rho_out.iter_mut().enumerate() {
                        let old = *slot;
                        let (d, un) = old.overflowing_sub(b);
                        w0_delta_limb(&mut total_w0, idx, old, d);
                        *slot = d;
                        b = u64::from(un);
                        if b == 0 {
                            break;
                        }
                    }
                }
                trim_limbs(&mut rho_out);
                return Some((rho_out, Some(total_w0)));
            }
        }
        let mut rho_out = if w_du2.len() > step_m {
            shr_limbs_par(&w_du2[step_m..], 0, sched, max_workers)
        } else {
            empty_limb_vec()
        };
        if w_lo_lt_pow2_r {
            let mut b = 1u64;
            for slot in &mut rho_out {
                let (d, un) = slot.overflowing_sub(b);
                *slot = d;
                b = u64::from(un);
                if b == 0 {
                    break;
                }
            }
        }
        trim_limbs(&mut rho_out);
        let dbits = if rho_out.is_empty() {
            s_bits
        } else {
            s_bits + bit_len_limbs(&rho_out)
        };
        if dbits + 12 < l_bits {
            Some((rho_out, None))
        } else {
            crate::limb::put_limb_buf(rho_out);
            None
        }
    }
}

fn mul_u64_shl_limbs_par(
    kernels: &crate::refk::Kernels,
    src: &[u64],
    factor: u64,
    shift: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> LimbVec {
    if src.is_empty() || factor == 0 {
        return empty_limb_vec();
    }
    let n = src.len();
    let ws = (shift / 64) as usize;
    let bs = (shift & 63) as u32;
    let cap = ws + n + 2;
    let mut out = crate::limb::take_limb_buf_committed(cap, sched, max_workers);
    let out_ptr = out.as_mut_ptr() as usize;
    let src_ptr = src.as_ptr() as usize;

    if ws > 0 {
        let z_chunk = 16_384usize;
        let z_cnt = ws.div_ceil(z_chunk);
        if max_workers <= 1 || z_cnt <= 1 {
            unsafe {
                std::ptr::write_bytes(out_ptr as *mut u64, 0, ws);
            }
        } else {
            sched.par_for_leaf_max(z_cnt, 1, max_workers, |ci| {
                let s = ci * z_chunk;
                let e = (s + z_chunk).min(ws);
                unsafe {
                    std::ptr::write_bytes((out_ptr as *mut u64).add(s), 0, e - s);
                }
            });
        }
    }

    let chunk = 4_096usize;
    let num_chunks = n.div_ceil(chunk);
    let mut bnd_lo = vec![0u64; num_chunks];
    let mut bnd_hi = vec![0u64; num_chunks];
    let blo_ptr = bnd_lo.as_mut_ptr() as usize;
    let bhi_ptr = bnd_hi.as_mut_ptr() as usize;

    let work_chunk = |ci: usize| {
        let s = ci * chunk;
        let e = (s + chunk).min(n);
        let len = e - s;
        unsafe {
            let sp = (src_ptr as *const u64).add(s);
            let dp = (out_ptr as *mut u64).add(ws + s);
            if kernels.count_work {
                kernels
                    .stats
                    .limb_ops
                    .fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
            }
            if bs == 0 {
                let c = (kernels.active.mul_1)(dp, sp, len as u64, factor);
                *(blo_ptr as *mut u64).add(ci) = c;
            } else {
                let mut tmp = [0u64; 4_096];
                let c = (kernels.active.mul_1)(tmp.as_mut_ptr(), sp, len as u64, factor);
                let rsh = 64 - bs;
                let mut prev = 0u64;
                for i in 0..len {
                    let w = tmp[i];
                    dp.add(i).write((w << bs) | prev);
                    prev = w >> rsh;
                }
                *(blo_ptr as *mut u64).add(ci) = (c << bs) | prev;
                *(bhi_ptr as *mut u64).add(ci) = c >> rsh;
            }
        }
    };

    if max_workers <= 1 || num_chunks <= 1 {
        for ci in 0..num_chunks {
            work_chunk(ci);
        }
    } else {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, work_chunk);
    }

    unsafe {
        (out_ptr as *mut u64).add(ws + n).write(0);
        (out_ptr as *mut u64).add(ws + n + 1).write(0);
        out.set_len(cap);
    }

    for ci in 0..num_chunks {
        let e = ((ci + 1) * chunk).min(n);
        let base_idx = ws + e;
        let lo = bnd_lo[ci];
        let hi = bnd_hi[ci];
        if lo == 0 && hi == 0 {
            continue;
        }
        let (s0, k0) = out[base_idx].overflowing_add(lo);
        out[base_idx] = s0;
        let (s1, k1) = out[base_idx + 1].overflowing_add(hi);
        let (s2, k2) = s1.overflowing_add(u64::from(k0));
        out[base_idx + 1] = s2;
        let mut carry = u64::from(k1) + u64::from(k2);
        let mut idx = base_idx + 2;
        while carry > 0 && idx < cap {
            let (sv, kv) = out[idx].overflowing_add(carry);
            out[idx] = sv;
            carry = u64::from(kv);
            idx += 1;
        }
    }

    trim_limbs(&mut out);
    out
}

fn mul_slice_u64_with_carry_par(
    kernels: &crate::refk::Kernels,
    sched: &crate::par::Scheduler,
    max_workers: usize,
    src: &[u64],
    factor: u64,
    carry_in: u64,
) -> BigNum {
    let n = src.len();
    if n == 0 {
        return BigNum::from_u64(carry_in);
    }
    let cap = n + 1;
    let mut out = crate::limb::take_limb_buf_committed(cap, sched, max_workers);
    let out_ptr = out.as_mut_ptr() as usize;
    let src_ptr = src.as_ptr() as usize;
    let chunk = 8_192usize;
    let num_chunks = n.div_ceil(chunk);
    let mut chunk_carries = vec![0u64; num_chunks];
    let mut chunk_w0s = vec![0u64; num_chunks];
    let cc_ptr = chunk_carries.as_mut_ptr() as usize;
    let cw_ptr = chunk_w0s.as_mut_ptr() as usize;

    let run_chunk = |ci: usize| {
        let s = ci * chunk;
        let e = (s + chunk).min(n);
        let len = e - s;
        unsafe {
            let sp = (src_ptr as *const u64).add(s);
            let dp = (out_ptr as *mut u64).add(s);
            let c = (kernels.active.mul_1)(dp, sp, len as u64, factor);
            if kernels.count_work {
                kernels
                    .stats
                    .limb_ops
                    .fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let w = compute_limbs_w0(kernels, std::slice::from_raw_parts(dp, len));
            *(cc_ptr as *mut u64).add(ci) = c;
            *(cw_ptr as *mut u64).add(ci) = w;
        }
    };

    if max_workers <= 1 || num_chunks <= 1 {
        for ci in 0..num_chunks {
            run_chunk(ci);
        }
    } else {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, run_chunk);
    }

    unsafe {
        (out_ptr as *mut u64).add(n).write(0);
        out.set_len(cap);
    }

    let mut total_w0 = 0u64;
    for (ci, &cw) in chunk_w0s.iter().enumerate() {
        let shift = mod61_pow2(((ci * chunk) as u64) * 3);
        total_w0 = mod61_add(total_w0, mod61_mul(cw, shift));
    }

    if carry_in > 0 {
        let mut c = carry_in;
        let mut idx = 0usize;
        while c > 0 && idx < cap {
            let old = out[idx];
            let (sv, ov) = old.overflowing_add(c);
            w0_delta_limb(&mut total_w0, idx, old, sv);
            out[idx] = sv;
            c = u64::from(ov);
            idx += 1;
        }
    }

    for ci in 0..num_chunks {
        let mut c = chunk_carries[ci];
        let mut idx = ((ci + 1) * chunk).min(n);
        while c > 0 && idx < cap {
            let old = out[idx];
            let (sv, ov) = old.overflowing_add(c);
            w0_delta_limb(&mut total_w0, idx, old, sv);
            out[idx] = sv;
            c = u64::from(ov);
            idx += 1;
        }
    }

    trim_limbs(&mut out);
    BigNum::from_limbs_with_w0(out, false, 0, total_w0)
}

#[allow(clippy::too_many_arguments)]
fn try_build_a_norm_fused_par(
    kernels: &crate::refk::Kernels,
    q_limbs: &[u64],
    q_shift: u64,
    t_limbs: &mut LimbVec,
    can_drop_t: bool,
    factor: u64,
    sched: &crate::par::Scheduler,
    max_workers: usize,
) -> Option<(LimbVec, u64, u64, u64)> {
    let q_len = q_limbs.len();
    let old_t_len = t_limbs.len();
    if q_len < 4096 || old_t_len == 0 || factor == 0 || factor >= (1u64 << 32) {
        return None;
    }
    let ws0 = (q_shift / 64) as usize;
    let bs0 = (q_shift & 63) as u32;
    let v_cap = (ws0 + q_len + 2).max(old_t_len + 1);
    if v_cap < 16 {
        return None;
    }
    let w_start = v_cap - 8;

    let eval_top = |c_seed: u64, br_seed: u64| -> Option<(usize, u64)> {
        let q_start = w_start.saturating_sub(ws0).saturating_sub(4);
        let span = (q_len + 2).saturating_sub(q_start);
        if span > 16 {
            return None;
        }
        let mut c = if q_start == 0 { 0u128 } else { c_seed as u128 };
        let mut raw_prod = [0u64; 16];
        for (off, slot) in raw_prod.iter_mut().enumerate().take(span) {
            let qi = q_start + off;
            let qw = if qi < q_len { q_limbs[qi] } else { 0 };
            let p = (qw as u128) * (factor as u128) + c;
            *slot = p as u64;
            c = p >> 64;
        }
        let v_at = |idx: usize| -> u64 {
            if idx < ws0 {
                return 0;
            }
            let qi = idx - ws0;
            let get_raw = |k: usize| -> u64 {
                if k < q_start || k - q_start >= span {
                    0
                } else {
                    raw_prod[k - q_start]
                }
            };
            if bs0 == 0 {
                get_raw(qi)
            } else if qi == 0 {
                get_raw(0) << bs0
            } else {
                (get_raw(qi) << bs0) | (get_raw(qi - 1) >> (64 - bs0))
            }
        };
        let mut br = br_seed;
        let mut diff = [0u64; 8];
        for (k, d_slot) in diff.iter_mut().enumerate() {
            let idx = w_start + k;
            let vw = v_at(idx);
            let uw = if idx < old_t_len { t_limbs[idx] } else { 0 };
            let (d1, b1) = vw.overflowing_sub(uw);
            let (d2, b2) = d1.overflowing_sub(br);
            *d_slot = d2;
            br = u64::from(b1) + u64::from(b2);
        }
        if br != 0 {
            return None;
        }
        for k in (0..8).rev() {
            if diff[k] != 0 {
                return Some((w_start + k + 1, diff[k].leading_zeros() as u64));
            }
        }
        None
    };

    let (a_len, s_n) = match (eval_top(0, 1), eval_top(factor, 0)) {
        (Some((l0, s0)), Some((l1, s1))) if l0 == l1 && s0 == s1 && l0 >= 1 => (l0, s0),
        _ => return None,
    };

    let tot_q_shift = q_shift + s_n;
    let ws = (tot_q_shift / 64) as usize;
    let bs_q = (tot_q_shift & 63) as u32;
    let bs_t = s_n as u32;
    let cap = (ws + q_len + 2).max(old_t_len + 2).max(a_len + 2);

    let chunk = 4_096usize;
    let num_chunks = cap.div_ceil(chunk);
    let t_prev: Vec<u64> = (0..num_chunks)
        .map(|ci| {
            let s = ci * chunk;
            if bs_t == 0 || s == 0 || s - 1 >= old_t_len {
                0u64
            } else {
                t_limbs[s - 1] >> (64 - bs_t)
            }
        })
        .collect();

    let reused_t = can_drop_t && t_limbs.capacity() >= cap;
    let mut out = if reused_t {
        let mut tb = take_limb_vec(t_limbs);
        tb.resize(cap, 0);
        tb
    } else {
        let mut b = crate::limb::take_limb_buf_committed(cap, sched, max_workers);
        unsafe {
            b.set_len(cap);
        }
        b
    };

    let out_ptr = out.as_mut_ptr() as usize;
    let t_src_ptr = if reused_t {
        out_ptr
    } else {
        t_limbs.as_ptr() as usize
    };
    let q_ptr = q_limbs.as_ptr() as usize;
    let mut deltas = vec![0i128; num_chunks];
    let deltas_ptr = deltas.as_mut_ptr() as usize;
    // The check words of the limbs of T and of Q that each chunk reads.
    let mut src_w0s = vec![[0u64; 2]; num_chunks];
    let src_w0s_ptr = src_w0s.as_mut_ptr() as usize;

    if kernels.count_work {
        kernels.stats.limb_ops.fetch_add(
            (q_len + cap * 2) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let work_chunk = |ci: usize| {
        let s = ci * chunk;
        let e = (s + chunk).min(cap);
        let len = e - s;
        unsafe {
            let mut u_loc = [0u64; 4_096];
            let copy_t = old_t_len.saturating_sub(s).min(len);
            // The word of these limbs of T, before the chunk's result is written over them
            // (where A is made in T's buffer).
            let t_w0 = if copy_t > 0 {
                compute_limbs_w0(
                    kernels,
                    std::slice::from_raw_parts((t_src_ptr as *const u64).add(s), copy_t),
                )
            } else {
                0
            };
            let mut q_w0 = 0u64;
            if bs_t == 0 {
                if copy_t > 0 {
                    std::ptr::copy_nonoverlapping(
                        (t_src_ptr as *const u64).add(s),
                        u_loc.as_mut_ptr(),
                        copy_t,
                    );
                }
            } else {
                let rsh_t = 64 - bs_t;
                let mut prev = t_prev[ci];
                for (i, u_slot) in u_loc.iter_mut().enumerate().take(copy_t) {
                    let tw = *(t_src_ptr as *const u64).add(s + i);
                    *u_slot = (tw << bs_t) | prev;
                    prev = tw >> rsh_t;
                }
                if copy_t < len {
                    u_loc[copy_t] = prev;
                }
            }

            let mut v_loc = [0u64; 4_096];
            let mut v_out_lo = 0u64;
            let mut v_out_hi = 0u64;
            let i_start = s.max(ws);
            if i_start < e && i_start - ws < q_len {
                let q_s = i_start - ws;
                let q_e = (e - ws).min(q_len);
                let q_cnt = q_e - q_s;
                let off = i_start - s;
                q_w0 = compute_limbs_w0(
                    kernels,
                    std::slice::from_raw_parts((q_ptr as *const u64).add(q_s), q_cnt),
                );
                if bs_q == 0 {
                    let c = (kernels.active.mul_1)(
                        v_loc.as_mut_ptr().add(off),
                        (q_ptr as *const u64).add(q_s),
                        q_cnt as u64,
                        factor,
                    );
                    if off + q_cnt < len {
                        v_loc[off + q_cnt] = c;
                    } else {
                        v_out_lo = c;
                    }
                } else {
                    let mut raw = [0u64; 4_096];
                    let c = (kernels.active.mul_1)(
                        raw.as_mut_ptr(),
                        (q_ptr as *const u64).add(q_s),
                        q_cnt as u64,
                        factor,
                    );
                    let rsh_q = 64 - bs_q;
                    let mut prev = 0u64;
                    for (k, &rw) in raw.iter().enumerate().take(q_cnt) {
                        v_loc[off + k] = (rw << bs_q) | prev;
                        prev = rw >> rsh_q;
                    }
                    let c_lo = (c << bs_q) | prev;
                    let c_hi = c >> rsh_q;
                    if off + q_cnt + 1 < len {
                        v_loc[off + q_cnt] = c_lo;
                        v_loc[off + q_cnt + 1] = c_hi;
                    } else if off + q_cnt < len {
                        v_loc[off + q_cnt] = c_lo;
                        v_out_lo = c_hi;
                    } else {
                        v_out_lo = c_lo;
                        v_out_hi = c_hi;
                    }
                }
            }

            let br = (kernels.active.sub_n)(
                (out_ptr as *mut u64).add(s),
                v_loc.as_ptr(),
                u_loc.as_ptr(),
                len as u64,
            );
            *(deltas_ptr as *mut i128).add(ci) =
                (v_out_lo as i128) + ((v_out_hi as i128) << 64) - (br as i128);
            *(src_w0s_ptr as *mut [u64; 2]).add(ci) = [t_w0, q_w0];
        }
    };

    if max_workers <= 1 || num_chunks <= 1 {
        for ci in 0..num_chunks {
            work_chunk(ci);
        }
    } else {
        sched.par_for_leaf_max(num_chunks, 1, max_workers, work_chunk);
    }

    if can_drop_t && !reused_t {
        crate::limb::put_limb_buf(take_limb_vec(t_limbs));
    }

    for (ci, &d) in deltas.iter().enumerate() {
        if d == 0 {
            continue;
        }
        let e = ((ci + 1) * chunk).min(cap);
        if e >= cap {
            continue;
        }
        if d > 0 {
            let ud = d as u128;
            let lo = ud as u64;
            let hi = (ud >> 64) as u64;
            let (s0, k0) = out[e].overflowing_add(lo);
            out[e] = s0;
            let mut carry = hi + u64::from(k0);
            let mut idx = e + 1;
            while carry > 0 && idx < cap {
                let (sv, kv) = out[idx].overflowing_add(carry);
                out[idx] = sv;
                carry = u64::from(kv);
                idx += 1;
            }
        } else {
            debug_assert_eq!(d, -1);
            let mut br = 1u64;
            let mut idx = e;
            while br > 0 && idx < cap {
                let (dv, bv) = out[idx].overflowing_sub(br);
                out[idx] = dv;
                br = u64::from(bv);
                idx += 1;
            }
        }
    }

    // The words of T and of Q as read: a chunk's limbs of T begin at the chunk's own limb, its
    // limbs of Q at that limb less the shift of Q.
    let (mut t_limbs_w0, mut q_limbs_w0) = (0u64, 0u64);
    for (ci, w0s) in src_w0s.iter().enumerate() {
        let at = ci * chunk;
        t_limbs_w0 = add_block_w0_at(t_limbs_w0, w0s[0], at);
        q_limbs_w0 = add_block_w0_at(q_limbs_w0, w0s[1], at.max(ws) - ws);
    }

    debug_assert!(out[a_len..cap].iter().all(|&w| w == 0));
    out.truncate(a_len);
    debug_assert_eq!(out[a_len - 1] >> 63, 1);
    Some((out, s_n, q_limbs_w0, t_limbs_w0))
}

#[cfg(test)]
mod finish_pass_tests {
    use super::*;
    use crate::refk::Kernels;

    fn scheduler() -> crate::par::Scheduler {
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
        crate::par::Scheduler::new(&shares, None)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The carry into limb `t` of `w * factor` by the whole chain from the bottom.
    fn carry_by_chain(w: &[u64], t: usize, factor: u64) -> u64 {
        let mut c = 0u128;
        for i in 0..t {
            let wl = w.get(i).copied().unwrap_or(0);
            c = ((wl as u128) * (factor as u128) + c) >> 64;
        }
        c as u64
    }

    /// The carry into a limb is found from the limbs just below it, also where those leave it
    /// open for a while, and is given up where `CARRY_SEARCH_LIMBS` limbs leave it open.
    #[test]
    fn carry_into_a_limb_is_settled_from_below() {
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        let factor = FINISH_D_SQR;
        let mut w: Vec<u64> = (0..6_000).map(|_| rng.next()).collect();
        for t in [0usize, 1, 2, 3, 64, 65, 500, 5_999, 6_000, 6_005] {
            assert_eq!(
                exact_mul_carry_into(&w, t, factor),
                Some(carry_by_chain(&w, t, factor)),
                "t={t}"
            );
        }
        // A limb whose product ends in all ones passes the carry into it straight on.
        let mut inv = factor;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(factor.wrapping_mul(inv)));
        }
        let open = u64::MAX.wrapping_mul(inv);
        assert_eq!(open.wrapping_mul(factor), u64::MAX);
        let t = 5_000usize;
        for run in [1usize, 2, 10, 63, 64, 65, CARRY_SEARCH_LIMBS - 1] {
            for below in [0u64, 1, u64::MAX, rng.next()] {
                w[t - run - 1] = below;
                for x in &mut w[(t - run)..t] {
                    *x = open;
                }
                assert_eq!(
                    exact_mul_carry_into(&w, t, factor),
                    Some(carry_by_chain(&w, t, factor)),
                    "run={run} below={below}"
                );
            }
        }
        for x in &mut w[(t - CARRY_SEARCH_LIMBS - 4)..t] {
            *x = open;
        }
        assert_eq!(exact_mul_carry_into(&w, t, factor), None);
        // At the bottom the chain ends by itself.
        let low = vec![open; 40];
        assert_eq!(
            exact_mul_carry_into(&low, 40, factor),
            Some(carry_by_chain(&low, 40, factor))
        );
    }

    /// `x mod m` for a small `m`.
    fn rem_small(x: &[u64], m: u64) -> u64 {
        let mut r = 0u128;
        for &l in x.iter().rev() {
            r = ((r << 64) | (l as u128)) % (m as u128);
        }
        r as u64
    }

    /// `x / m`, which must be exact.
    fn div_small_exact(x: &[u64], m: u64) -> LimbVec {
        let mut out = crate::limb::take_limb_buf_zeroed(x.len());
        let mut r = 0u128;
        for i in (0..x.len()).rev() {
            let cur = (r << 64) | (x[i] as u128);
            out[i] = (cur / (m as u128)) as u64;
            r = cur % (m as u128);
        }
        assert_eq!(r, 0);
        out
    }

    /// The two passes of before: `factor * w mod (2^L - 1)` in place, then the part kept.
    fn rho_by_two_passes(
        kernels: &Kernels,
        sched: &crate::par::Scheduler,
        mut w: LimbVec,
        exp_whole_w0: u64,
        step_m: usize,
        l_bits: u64,
        full_bits_1: u64,
    ) -> Option<(bool, LimbVec, u64)> {
        let w0 = mul_u64_mod_2l_minus_1_in_place(kernels, sched, 8, &mut w, FINISH_D_SQR, l_bits);
        let rem_bits = full_bits_1 - l_bits;
        let pow2_rem_w0 = mod61_pow2(rem_bits);
        let all_zero = w0 == 0 && w.iter().all(|&x| x == 0);
        for (neg, phi) in [(false, mod61_sub(pow2_rem_w0, 1)), (true, pow2_rem_w0)] {
            for carry_z in [false, true] {
                if wrapped_product_expected_w0(w0, phi, l_bits, carry_z) == exp_whole_w0 {
                    if let Some((rho, rho_w0)) = extract_rho_hi_fast(
                        kernels,
                        &mut w,
                        step_m,
                        l_bits,
                        rem_bits,
                        neg,
                        carry_z && all_zero,
                        sched,
                        8,
                    ) {
                        let rho_w0 = rho_w0.unwrap_or_else(|| compute_limbs_w0(kernels, &rho));
                        return Some((neg && !rho.is_empty(), rho, rho_w0));
                    }
                }
            }
        }
        None
    }

    /// A wrapped square as a step of the finish has it: `w` with `10005 w = 2^(128 m) -/+ delta`
    /// modulo `2^L - 1` for a `delta` of `delta_bits` bits, and the check word of that whole.
    /// `fold` adds a multiple of `(2^L - 1) / 15` to `w`, which leaves the product modulo
    /// `2^L - 1` as it is and gives its top a carry to fold.
    fn square_for(
        kernels: &Kernels,
        rng: &mut Rng,
        step_m: usize,
        l_limbs: usize,
        delta_bits: u64,
        neg: bool,
        fold: u64,
    ) -> (LimbVec, u64, u64) {
        let m = FINISH_D_SQR;
        let l_bits = (l_limbs as u64) * 64;
        let full_bits_1 = (2 * step_m as u64) * 64;
        let rem_bits = full_bits_1 - l_bits;
        assert!(delta_bits > rem_bits + 64 && delta_bits < l_bits);
        let d_limbs = delta_bits.div_ceil(64) as usize;
        let mut delta: Vec<u64> = (0..d_limbs).map(|_| rng.next()).collect();
        let top_bits = (delta_bits - 1) % 64 + 1;
        delta[d_limbs - 1] &= u64::MAX >> (64 - top_bits);
        delta[d_limbs - 1] |= 1u64 << (top_bits - 1);
        delta[0] = (delta[0] & 0x0000_FFFF_FFFF_0000) | 0x0000_0001_0000_0000;
        // The wrapped whole for this delta, then delta moved so that 10005 divides it.
        let wrapped_of = |delta: &[u64]| -> Vec<u64> {
            let mut r = vec![0u64; l_limbs];
            r[..delta.len()].copy_from_slice(delta);
            let (rw, rb) = ((rem_bits / 64) as usize, rem_bits % 64);
            if neg {
                // delta + 2^rem
                let mut c = 1u64 << rb;
                for x in &mut r[rw..] {
                    let (v, over) = x.overflowing_add(c);
                    *x = v;
                    c = u64::from(over);
                    if c == 0 {
                        break;
                    }
                }
            } else {
                // 2^L - 1 - (delta - 2^rem)
                let mut b = 1u64 << rb;
                for x in &mut r[rw..] {
                    let (v, under) = x.overflowing_sub(b);
                    *x = v;
                    b = u64::from(under);
                    if b == 0 {
                        break;
                    }
                }
                for x in &mut r {
                    *x = !*x;
                }
            }
            r
        };
        let r0 = rem_small(&wrapped_of(&delta), m);
        if neg {
            delta[0] -= r0;
        } else {
            delta[0] += r0;
        }
        let wrapped = wrapped_of(&delta);
        assert_eq!(rem_small(&wrapped, m), 0);
        let mut w = div_small_exact(&wrapped, m);
        let unit = 0x1111_1111_1111_1111u64 * fold;
        let mut c = 0u64;
        for x in &mut w {
            let (v1, o1) = x.overflowing_add(unit);
            let (v2, o2) = v1.overflowing_add(c);
            *x = v2;
            c = u64::from(o1) + u64::from(o2);
        }
        assert_eq!(c, 0);
        trim_limbs(&mut w);
        let delta_w0 = compute_limbs_w0(kernels, &delta);
        let one_w0 = mod61_pow2(full_bits_1);
        let whole_w0 = if neg {
            mod61_add(one_w0, delta_w0)
        } else {
            mod61_sub(one_w0, delta_w0)
        };
        let w_w0 = compute_limbs_w0(kernels, &w);
        (w, w_w0, whole_w0)
    }

    /// The residual taken in one pass from the kept third of the wrapped square is the one the
    /// two passes over the whole square give: sign, limbs and check word, for both signs, with
    /// and without a carry folded in at the bottom, and with a top of the product that is all
    /// ones or all zeros for a few limbs, a few thousand, or most of the part kept. A residual
    /// that ends below the part kept is left to the two passes.
    #[test]
    fn rho_in_one_pass_matches_the_two_passes() {
        let kernels = Kernels::reference_only();
        let sched = scheduler();
        let mut rng = Rng(0x1319_8A2E_0370_7344);
        let step_m = 70_000usize;
        let l_limbs = 106_000usize;
        let l_bits = (l_limbs as u64) * 64;
        let full_bits_1 = (2 * step_m as u64) * 64;
        let rem_bits = full_bits_1 - l_bits;
        let mut taken = 0;
        let mut left = 0;
        for &neg in &[false, true] {
            for &fold in &[0u64, 1, 7, 14] {
                for &delta_bits in &[
                    l_bits - 20,
                    l_bits - 14,
                    l_bits - 3_000,
                    l_bits - 200_000,
                    l_bits - 64 * (CARRY_SEARCH_LIMBS as u64) - 77,
                    l_bits - 64 * 20_000,
                    // The residual ends in a limb of eight bits, or of one.
                    l_bits - 64 * 20_000 - 56,
                    l_bits - 64 * 9_000 - 63,
                    (step_m as u64) * 64 + 1_000,
                    rem_bits + 6_000,
                    rem_bits + 200,
                ] {
                    let small = delta_bits < (step_m as u64) * 64;
                    let tag = format!("neg={neg} fold={fold} delta_bits={delta_bits}");
                    let (w, w_w0, whole_w0) =
                        square_for(&kernels, &mut rng, step_m, l_limbs, delta_bits, neg, fold);
                    let want = rho_by_two_passes(
                        &kernels,
                        &sched,
                        w.clone(),
                        whole_w0,
                        step_m,
                        l_bits,
                        full_bits_1,
                    )
                    .unwrap_or_else(|| panic!("{tag}: the two passes gave nothing"));
                    match rho_from_wrapped_square(
                        &kernels,
                        &sched,
                        8,
                        w.clone(),
                        w_w0,
                        whole_w0,
                        FINISH_D_SQR,
                        step_m,
                        l_bits,
                        full_bits_1,
                    ) {
                        RhoFromSquare::Done(rho_neg, rho, rho_w0) => {
                            assert!(!small, "{tag}");
                            assert_eq!(rho_neg, want.0, "{tag}");
                            assert_eq!(rho, want.1, "{tag}");
                            assert_eq!(rho_w0, want.2, "{tag}");
                            assert_eq!(rho_w0, compute_limbs_w0(&kernels, &rho), "{tag}");
                            taken += 1;
                        }
                        RhoFromSquare::NotTaken(back) => {
                            assert!(small, "{tag}: not taken");
                            assert_eq!(back, w, "{tag}");
                            left += 1;
                        }
                        RhoFromSquare::Spoiled => panic!("{tag}: spoiled"),
                    }
                }
            }
        }
        assert_eq!((taken, left), (72, 16));

        // A residual too large for the step is refused by both.
        for &neg in &[false, true] {
            let (w, w_w0, whole_w0) =
                square_for(&kernels, &mut rng, step_m, l_limbs, l_bits - 5, neg, 3);
            assert!(
                rho_by_two_passes(&kernels, &sched, w.clone(), whole_w0, step_m, l_bits, full_bits_1)
                    .is_none()
            );
            assert!(matches!(
                rho_from_wrapped_square(
                    &kernels,
                    &sched,
                    8,
                    w,
                    w_w0,
                    whole_w0,
                    FINISH_D_SQR,
                    step_m,
                    l_bits,
                    full_bits_1,
                ),
                RhoFromSquare::Spoiled
            ));
        }

        // A wrong word for the square, or for the whole, is not taken, and the square stays.
        let (w, w_w0, whole_w0) =
            square_for(&kernels, &mut rng, step_m, l_limbs, l_bits - 40, false, 5);
        for (sq_w0, exp_w0) in [(mod61_add(w_w0, 1), whole_w0), (w_w0, mod61_add(whole_w0, 1))] {
            match rho_from_wrapped_square(
                &kernels,
                &sched,
                8,
                w.clone(),
                sq_w0,
                exp_w0,
                FINISH_D_SQR,
                step_m,
                l_bits,
                full_bits_1,
            ) {
                RhoFromSquare::NotTaken(back) => assert_eq!(back, w),
                _ => panic!("a wrong check word was taken"),
            }
        }
        // One worker, or a square that does not reach the part kept: left to the two passes.
        assert!(matches!(
            rho_from_wrapped_square(
                &kernels,
                &sched,
                1,
                w.clone(),
                w_w0,
                whole_w0,
                FINISH_D_SQR,
                step_m,
                l_bits,
                full_bits_1,
            ),
            RhoFromSquare::NotTaken(_)
        ));
        assert!(matches!(
            rho_from_wrapped_square(
                &kernels,
                &sched,
                8,
                slice_to_limb_vec(&w[..step_m]),
                w_w0,
                whole_w0,
                FINISH_D_SQR,
                step_m,
                l_bits,
                full_bits_1,
            ),
            RhoFromSquare::NotTaken(_)
        ));
    }

    /// A square shorter than the transform (its top limbs zero) gives the same residual by both
    /// ways: the chunks above its top hold only the carry.
    #[test]
    fn rho_in_one_pass_of_a_short_square() {
        let kernels = Kernels::reference_only();
        let sched = scheduler();
        let mut rng = Rng(0xA409_3822_299F_31D0);
        let step_m = 70_000usize;
        let l_limbs = 106_000usize;
        let l_bits = (l_limbs as u64) * 64;
        let full_bits_1 = (2 * step_m as u64) * 64;
        let pow2_rem_w0 = mod61_pow2(full_bits_1 - l_bits);
        for &short_by in &[1usize, 2_000, 8_192, 20_000, 30_001] {
            let w_raw: Vec<u64> = (0..(l_limbs - short_by)).map(|_| rng.next() | 1).collect();
            let w = slice_to_limb_vec(&w_raw);
            let w_w0 = compute_limbs_w0(&kernels, &w);
            // No carry leaves the top, so the wrapped multiple is the multiple itself, and the
            // whole it stands for is taken to be the one with a one and zeros folded.
            let multiple_w0 = mod61_mul(mod61_reduce(FINISH_D_SQR as u128), w_w0);
            let whole_w0 = wrapped_product_expected_w0(multiple_w0, pow2_rem_w0, l_bits, false);
            let want =
                rho_by_two_passes(&kernels, &sched, w.clone(), whole_w0, step_m, l_bits, full_bits_1)
                    .expect("the two passes gave nothing");
            match rho_from_wrapped_square(
                &kernels,
                &sched,
                8,
                w,
                w_w0,
                whole_w0,
                FINISH_D_SQR,
                step_m,
                l_bits,
                full_bits_1,
            ) {
                RhoFromSquare::Done(rho_neg, rho, rho_w0) => {
                    assert!(rho_neg);
                    assert_eq!(rho_neg, want.0);
                    assert_eq!(rho, want.1, "short_by={short_by}");
                    assert_eq!(rho_w0, want.2);
                }
                _ => panic!("short_by={short_by}: not taken"),
            }
        }
    }

    /// A whole product `p` of `total` limbs as a step has it, its wrapped form, and the `u` of
    /// the step before for it: the limbs of `p` from `k_a` up, moved by `nudge` in the limb just
    /// below the folded part (a nudge across zero makes the folded part one more or one less
    /// than what `u_prev` says).
    #[allow(clippy::too_many_arguments)]
    fn az_case(
        kernels: &Kernels,
        sched: &crate::par::Scheduler,
        rng: &mut Rng,
        total: usize,
        k_a: usize,
        l_limbs: usize,
        edge_limb: Option<u64>,
        nudge: i64,
        shape: impl Fn(&mut Vec<u64>),
    ) -> (LimbVec, u64, Vec<u64>, u64) {
        let l_bits = (l_limbs as u64) * 64;
        let mut p: Vec<u64> = (0..total).map(|_| rng.next() | 1).collect();
        if let Some(v) = edge_limb {
            p[l_limbs - 1] = v;
        }
        shape(&mut p);
        let whole_w0 = compute_limbs_w0(kernels, &p);
        let wrapped = reduce_limbs_mod_2l_minus_1(kernels, sched, 8, &p, l_bits);
        let wrapped_w0 = compute_limbs_w0(kernels, &wrapped);
        // u_prev = (p + nudge * 2^(64 (l_limbs - 1))) >> (64 k_a)
        let mut moved = p.clone();
        let mut c = nudge.unsigned_abs();
        for x in &mut moved[(l_limbs - 1)..] {
            let (v, over) = if nudge >= 0 { x.overflowing_add(c) } else { x.overflowing_sub(c) };
            *x = v;
            c = u64::from(over);
            if c == 0 {
                break;
            }
        }
        assert_eq!(c, 0);
        let u_prev = moved[k_a..].to_vec();
        (wrapped, wrapped_w0, u_prev, whole_w0)
    }

    /// The `u` of a step read out of the wrapped product without moving the old `u` or
    /// subtracting from the product is the one the long way gives, with its check word: where
    /// the folded part is what the old `u` says, one more, or one less, with and without a
    /// borrow below the part kept. A borrow that would run into the part kept, and a product
    /// smaller than its folded part, go the long way and come out the same.
    #[test]
    fn u_from_the_wrapped_product_without_moving_it() {
        let kernels = Kernels::reference_only();
        let sched = scheduler();
        let mut rng = Rng(0x082E_FA98_EC4E_6C89);
        let cur_m = 30_000usize;
        let step_m = 59_990usize;
        let k_a = step_m + 2;
        let total = k_a + cur_m;
        let shift_down = total - step_m;
        let approx_shift_bits = (k_a as u64) * 64;
        let mut short = 0;
        let mut long = 0;
        for &l_limbs in &[step_m + 8, step_m + 70, step_m + 5_000] {
            let l_bits = (l_limbs as u64) * 64;
            let plen = total - l_limbs;
            for case in 0..10 {
                let tag = format!("l_limbs={l_limbs} case={case}");
                // Cases 0-2: the folded part is what u_prev says. 3-4: one less, one more.
                // 5: a borrow below the part kept that dies out there. 6: one that does not.
                // 7: the product is zero from the folded part's length up to the transform's,
                // so that nothing of the wrapped product is kept. 8, 9: as 0 and 3 with the old
                // u a limb too long (a zero on top).
                let (edge, nudge) = match case {
                    3 | 9 => (Some(u64::MAX - 5), 10i64),
                    4 => (Some(3), -10i64),
                    _ => (None, 0i64),
                };
                let (wrapped, wrapped_w0, mut u_prev, whole_w0) = az_case(
                    &kernels,
                    &sched,
                    &mut rng,
                    total,
                    k_a,
                    l_limbs,
                    edge,
                    nudge,
                    |p: &mut Vec<u64>| match case {
                        // The low limbs of the product and the folded part add to a carry (all
                        // ones meet a part that is not zero), which the kept part's lowest
                        // limbs do or do not let through.
                        5 => {
                            for x in &mut p[..plen] {
                                *x = u64::MAX;
                            }
                        }
                        6 => {
                            for x in &mut p[..shift_down] {
                                *x = u64::MAX;
                            }
                        }
                        7 => {
                            for x in &mut p[plen..l_limbs] {
                                *x = 0;
                            }
                            p[0] = 0;
                        }
                        _ => {}
                    },
                );
                if case >= 8 {
                    u_prev.push(0);
                }
                // The word of the old u is handed in, or left for the pass to take.
                let u_prev_w0 = (case % 2 == 1).then(|| compute_limbs_w0(&kernels, &u_prev));
                let by_parts = |w: &[u64]| {
                    let mut w = w.to_vec();
                    extract_az_from_wrapped_by_parts(
                        &kernels,
                        &sched,
                        8,
                        &mut w,
                        wrapped_w0,
                        l_bits,
                        &u_prev,
                        approx_shift_bits,
                        shift_down,
                        step_m,
                        whole_w0,
                    )
                };
                let want = by_parts(&wrapped).unwrap_or_else(|| panic!("{tag}: the long way gave nothing"));
                let unmoved = az_from_wrapped_unmoved(
                    &kernels,
                    &sched,
                    8,
                    &wrapped,
                    wrapped_w0,
                    l_bits,
                    &u_prev,
                    u_prev_w0,
                    approx_shift_bits,
                    shift_down,
                    step_m,
                    whole_w0,
                );
                match unmoved {
                    Some((u, u_w0)) => {
                        assert!(case != 6 && case != 7, "{tag}: taken");
                        assert_eq!(u, want, "{tag}");
                        assert_eq!(u_w0, compute_limbs_w0(&kernels, &u), "{tag}");
                        short += 1;
                    }
                    None => {
                        assert!(case == 6 || case == 7, "{tag}: not taken");
                        long += 1;
                    }
                }
                // The entry point gives the same by whichever way it goes.
                let mut w = wrapped.clone();
                let (u, u_w0) = extract_az_from_wrapped(
                    &kernels,
                    &sched,
                    8,
                    &mut w,
                    wrapped_w0,
                    l_bits,
                    &u_prev,
                    u_prev_w0,
                    approx_shift_bits,
                    shift_down,
                    step_m,
                    whole_w0,
                )
                .unwrap_or_else(|| panic!("{tag}: nothing"));
                assert_eq!(u, want, "{tag}");
                assert_eq!(u_w0.is_some(), case != 6 && case != 7, "{tag}");
                // A wrong word for the product is taken by neither.
                assert!(az_from_wrapped_unmoved(
                    &kernels,
                    &sched,
                    8,
                    &wrapped,
                    mod61_add(wrapped_w0, 3),
                    l_bits,
                    &u_prev,
                    u_prev_w0,
                    approx_shift_bits,
                    shift_down,
                    step_m,
                    whole_w0,
                )
                .is_none());
            }
        }
        assert_eq!((short, long), (24, 6));
    }

    /// `(prod >> shift)` cut to its own length, the correction the passes apply.
    fn shifted(prod: &[u64], shift_bits: u64) -> Vec<u64> {
        let mut d = shr_limbs(prod, shift_bits);
        trim_limbs(&mut d);
        d.to_vec()
    }

    /// The correction pass gives the word of the correction it applied, and a number that grows
    /// downward takes its correction in the room below it to the same limbs as the same number
    /// extended with zeros and corrected the general way: added and taken away, a correction
    /// shorter than the room, as long, and a few limbs longer, a low part of it that is zero,
    /// and old limbs that pass a carry or a borrow on.
    #[test]
    fn correction_below_matches_the_general_pass() {
        let kernels = Kernels::reference_only();
        let sched = scheduler();
        let mut rng = Rng(0x4528_21E6_38D0_1377);
        let old_len = 40_000usize;
        let room = 40_000usize;
        let n = old_len + room;
        let mut fast = 0;
        for case in 0..24u64 {
            let neg = case % 2 == 1;
            // The product's length decides the correction's: its limbs above the shift.
            let shift_limbs = 25_000usize;
            let r_shift_bits =
                (shift_limbs as u64) * 64 + [1u64, 1, 17, 63][(case / 2 % 4) as usize];
            let delta_len = match case / 8 {
                0 => room + 2,
                1 => room,
                _ => room - 9_000,
            };
            let mut prod: Vec<u64> = (0..(shift_limbs + delta_len)).map(|_| rng.next()).collect();
            *prod.last_mut().unwrap() |= 1 << 63;
            let mut old: Vec<u64> = (0..old_len).map(|_| rng.next() | 1).collect();
            match case % 6 {
                // Old limbs that pass a carry on (ones) or a borrow (zeros), for a stretch.
                2 => old[..300].fill(u64::MAX),
                3 => old[..300].fill(0),
                // The correction's part in the room is zero: nothing is borrowed above.
                4 | 5 => {
                    let cut = (shift_limbs + room + 1).min(prod.len());
                    prod[..cut].fill(0);
                }
                _ => {}
            }
            let tag = format!("case={case} neg={neg} delta_len={delta_len}");
            let delta = shifted(&prod, r_shift_bits);
            let delta_w0 = compute_limbs_w0(&kernels, &delta);

            // The general pass on the number extended with zeros.
            let mut want = vec![0u64; room];
            want.extend_from_slice(&old);
            let want_w0 = apply_shifted_signed_delta_in_place_par(
                &kernels,
                &sched,
                8,
                &mut want,
                &prod,
                r_shift_bits,
                neg,
                n,
            );
            assert_eq!(want.len(), n, "{tag}");
            assert_eq!(want_w0, Some(delta_w0), "{tag}");
            // And with one worker.
            let mut one = vec![0u64; room];
            one.extend_from_slice(&old);
            let one_w0 = apply_shifted_signed_delta_in_place_par(
                &kernels,
                &sched,
                1,
                &mut one,
                &prod,
                r_shift_bits,
                neg,
                n,
            );
            assert_eq!((one_w0, &one), (Some(delta_w0), &want), "{tag}");
            // The result is the number moved up, plus or minus the correction, by its word.
            let old_w0 =
                mod61_mul(compute_limbs_w0(&kernels, &old), mod61_pow2((room as u64) * 64));
            let sum_w0 = if neg {
                mod61_sub(old_w0, delta_w0)
            } else {
                mod61_add(old_w0, delta_w0)
            };
            assert_eq!(compute_limbs_w0(&kernels, &want), sum_w0, "{tag}");

            // The same in the room of a number that grows downward: the room is not written.
            for room_zeroed in [false, true] {
                let mut z = GrowsDown::new(n, crate::limb::slice_to_limb_vec(&old));
                assert_eq!(z.limbs(), &old[..]);
                let dst = z.grown(n);
                if room_zeroed {
                    dst[..room].fill(0);
                } else {
                    dst[..room].fill(0xDEAD_BEEF_DEAD_BEEF);
                    fast += 1;
                }
                let got_w0 = apply_shifted_signed_delta_below(
                    &kernels,
                    &sched,
                    8,
                    dst,
                    room,
                    room_zeroed,
                    &prod,
                    r_shift_bits,
                    neg,
                );
                assert_eq!(got_w0, Some(delta_w0), "{tag} zeroed={room_zeroed}");
                assert_eq!(z.limbs(), &want[..], "{tag} zeroed={room_zeroed}");
                assert_eq!(z.into_vec(), want, "{tag}");
            }
        }
        assert_eq!(fast, 24);

        // A number that has not reached its full length is handed on as a copy of its limbs.
        let z = GrowsDown::new(100, crate::limb::slice_to_limb_vec(&[7u64, 8, 9]));
        assert_eq!(z.limbs(), &[7u64, 8, 9]);
        assert_eq!(z.into_vec(), vec![7u64, 8, 9]);
        // Small room, and a correction of nearly the number's length: the general pass.
        let old: Vec<u64> = (0..3_000).map(|_| rng.next() | 1).collect();
        let prod: Vec<u64> = (0..2_500).map(|_| rng.next() | 1).collect();
        for neg in [false, true] {
            let mut want = vec![0u64; 2_000];
            want.extend_from_slice(&old);
            let want_w0 = apply_shifted_signed_delta_in_place_par(
                &kernels, &sched, 8, &mut want, &prod, 65, neg, 5_000,
            );
            let mut z = GrowsDown::new(5_000, crate::limb::slice_to_limb_vec(&old));
            let dst = z.grown(5_000);
            dst[..2_000].fill(0x1234);
            let got_w0 = apply_shifted_signed_delta_below(
                &kernels, &sched, 8, dst, 2_000, false, &prod, 65, neg,
            );
            assert_eq!((got_w0, z.limbs()), (want_w0, &want[..]));
        }
    }

    /// The pass that makes `A` of `Q` and `T` gives, with `A`, the check words of the limbs of
    /// both as it read them, whether it writes `A` into the buffer of `T` or into one of its
    /// own, with one worker and with several; and a limb of either changed beforehand shows in
    /// the word.
    #[test]
    fn making_a_gives_the_words_of_q_and_t() {
        let kernels = Kernels::reference_only();
        let sched = scheduler();
        let mut rng = Rng(0xBE54_66CF_34E9_0C6C);
        let q_len = 20_000usize;
        for &(q_shift, in_t, workers) in &[
            (64u64 * 100 + 5, false, 8usize),
            (64 * 100 + 5, true, 8),
            (64 * 37, true, 8),
            (0, false, 8),
            (64 * 9 + 63, true, 1),
            (1, false, 1),
        ] {
            let tag = format!("q_shift={q_shift} in_t={in_t} workers={workers}");
            let q: Vec<u64> = (0..q_len).map(|_| rng.next() | 1).collect();
            let t_len = q_len + (q_shift / 64) as usize - 3;
            let t: Vec<u64> = (0..t_len).map(|_| rng.next() | 1).collect();

            let mut want = mul_u64_shl_limbs_par(&kernels, &q, CHUD_A, q_shift, &sched, 8);
            sub_limbs_in_place_par(&kernels, &mut want, &t, &sched, 8);
            trim_limbs(&mut want);
            let want_s = want.last().unwrap().leading_zeros() as u64;
            shl_sublimb_in_place_par(&mut want, want_s, &sched, 8);

            let mut t_buf = crate::limb::take_limb_buf_uninit(t_len + 64);
            t_buf.extend_from_slice(&t);
            let (a, s_n, q_w0, t_w0) = try_build_a_norm_fused_par(
                &kernels, &q, q_shift, &mut t_buf, in_t, CHUD_A, &sched, workers,
            )
            .unwrap_or_else(|| panic!("{tag}: not made"));
            assert_eq!((&a, s_n), (&want, want_s), "{tag}");
            assert_eq!(q_w0, compute_limbs_w0(&kernels, &q), "{tag}");
            assert_eq!(t_w0, compute_limbs_w0(&kernels, &t), "{tag}");
            assert_eq!(t_buf.is_empty(), in_t, "{tag}");

            // One limb of Q, then one of T, changed: the words are those of the changed limbs.
            for (which, at) in
                [(0usize, 0usize), (0, q_len - 1), (0, 7_777), (1, 0), (1, t_len - 1), (1, 4_096)]
            {
                let (mut q2, mut t2) = (q.clone(), t.clone());
                if which == 0 {
                    q2[at] ^= 1 << 17;
                } else {
                    t2[at] ^= 1 << 40;
                }
                let mut t_buf = crate::limb::take_limb_buf_uninit(t_len + 64);
                t_buf.extend_from_slice(&t2);
                let (_, _, q_w0b, t_w0b) = try_build_a_norm_fused_par(
                    &kernels, &q2, q_shift, &mut t_buf, in_t, CHUD_A, &sched, workers,
                )
                .unwrap_or_else(|| panic!("{tag}: not made"));
                assert_eq!(q_w0b, compute_limbs_w0(&kernels, &q2), "{tag} which={which} at={at}");
                assert_eq!(t_w0b, compute_limbs_w0(&kernels, &t2), "{tag} which={which} at={at}");
                assert_eq!(q_w0b != q_w0, which == 0, "{tag}");
                assert_eq!(t_w0b != t_w0, which == 1, "{tag}");
            }
        }
    }
}
