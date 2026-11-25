//! Contract:
//! - Inputs: Command-line target (`digits`, `OutputFormat`), `Profile` + `Shares` derived by `tier`,
//!   startup, persisted-static, or supplied `Calibration` figures, and `PlanOverrides` (carrying
//!   resumed stage geometry and dividing-level masks or Gate-2 test overrides).
//! - Outputs: Deterministic `Plan` specifying Chudnovsky term count, leaf length, shortest-path
//!   dividing levels and division forms, merge forms, Newton step precisions, radix conversion
//!   tree, product `MulMethod`s (never using `61*k`-bit coefficients or outer parts, shrinking
//!   `rows` + adding sweeps when a store tile is smaller than `rows * min_store_range_bytes`, and
//!   pricing column-sweep `check_n` passes), tier room commitments, checkpoint schedule, capacity
//!   verification, and noise-free planned time.
//! - Data format: All capacities, counts, and bit sizes in 64-bit integers; times in `f64` seconds;
//!   rates in bytes/sec or ops/sec; persisted static calibration check-worded modulo `2^61 - 1`.
//! - Work & bytes moved: Startup micro-benchmarks measure non-temporal hints, per-core `ntt_fwd`
//!   speeds, all 24 kernel throughputs, crossovers, and sieve ops (persisted and reused while
//!   cores, threads, caches, build, and bound kernels are unchanged), plus startup DRAM bandwidth,
//!   sweep-step time + overlap alpha, and concurrent Store idle/batch/beside-sweep figures;
//!   planning itself is pure deterministic DP in milliseconds.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::check::compute_limbs_w0;
use crate::fft::{FftTableCache, choose_fft_params, fft_mul};
use crate::limb::{mul_limbs_basecase, mul_limbs_karatsuba};
use crate::mul::MulMethod;
use crate::ntt::{
    NttSpec, largest_primes_desc_slice, largest_primes_for_two_adic,
    largest_primes_prefix_bits_slice,
};
use crate::par::Scheduler;
use crate::refk::{
    FftCarryCfg, FftSplitCfg, Kernels, MoveCfg, NUM_KERNELS, NttCrtCfg, NttCrtSegs,
    NttSplitCfg, NttVecCfg, PRIME_LOC1, build_fft_tables, build_ntt_tables,
};
use crate::store::Store;
use crate::tier::{Profile, Shares};

pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_477;
pub const LOG2_10: f64 = std::f64::consts::LOG2_10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Packed19,
}

impl OutputFormat {
    /// Exact output file size in bytes for `digits` fractional decimal digits of pi.
    /// - `Text`: `"3."` (2 bytes) + `digits` ASCII bytes (`0` digits -> `"3"` = 1 byte).
    /// - `Packed19`: 16-byte header (`magic: u64 = "SUPIPK19"`, `digits: u64`) + `8 * ceil(digits / 19)`
    ///   little-endian 19-digit `u64` words.
    pub fn file_bytes(self, digits: u64) -> Option<u64> {
        match self {
            OutputFormat::Text => {
                if digits == 0 {
                    Some(1)
                } else {
                    digits.checked_add(2)
                }
            }
            OutputFormat::Packed19 => {
                let words = digits.div_ceil(19);
                words.checked_mul(8)?.checked_add(16)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivForm {
    SequentialHensel,
    SingleProduct,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeForm {
    Separate,
    Fused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitFormOverride {
    Auto,
    ForceHalving(u32),
    ForceOuter(usize),
}

#[derive(Clone, Debug)]
pub struct PlanOverrides {
    pub max_ntt_primes: usize,
    pub force_dividing_levels: Option<Vec<(u32, DivForm)>>,
    pub force_dividing_mask: Option<u64>,
    pub force_merge_form: Option<MergeForm>,
    pub split_override: SplitFormOverride,
    pub checkpoint_mult: f64,
    pub force_series_shape: Option<(u64, u64)>,
    pub force_conv_leaf_digits: Option<usize>,
}

impl Default for PlanOverrides {
    fn default() -> Self {
        Self {
            max_ntt_primes: 18,
            force_dividing_levels: None,
            force_dividing_mask: None,
            force_merge_form: None,
            split_override: SplitFormOverride::Auto,
            checkpoint_mult: 10.0,
            force_series_shape: None,
            force_conv_leaf_digits: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceCalibration {
    pub name: String,
    pub idle_rate_bps: f64,
    pub beside_sweep_rate_bps: f64,
    pub sweep_step_beside_sec: f64,
    pub min_batch_range_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Calibration {
    pub core_ntt_bf_per_sec: Vec<(u64, f64)>,
    pub kernel_rates: [f64; NUM_KERNELS],
    pub ntt_bf_per_sec_per_worker: f64,
    pub fft_bf_per_sec_per_worker: f64,
    pub crt_ops_per_sec_per_worker: f64,
    pub basecase_limbs_sq_per_sec: f64,
    pub sieve_ops_per_sec: f64,
    pub basecase_max_limbs: usize,
    pub karatsuba_max_limbs: usize,
    /// The crossover inside the kernel `mul_base`: the longest shorter operand for which rows
    /// of limbs are its faster way (`u64::MAX`: always). It goes to the kernel with every call.
    pub mul_base_rows_max: u64,
    /// Symmetric crossover inside `sqr_base` (`u64::MAX`: always scalar rows).
    pub sqr_base_rows_max: u64,
    pub ram_bw_bps: f64,
    pub sweep_step_sec: f64,
    pub overlap_alpha: f64,
    pub nt_effective: bool,
    pub nt_ratio: f64,
    pub reused_static: bool,
    pub devices: Vec<DeviceCalibration>,
}

const STATIC_CAL_MAGIC: u64 = 0x5355_5049_4341_4C32; // "SUPICAL2"

struct PersistedStaticCal {
    core_ntt_bf_per_sec: Vec<(u64, f64)>,
    kernel_rates: [f64; NUM_KERNELS],
    ntt_bf_per_sec_per_worker: f64,
    fft_bf_per_sec_per_worker: f64,
    crt_ops_per_sec_per_worker: f64,
    basecase_limbs_sq_per_sec: f64,
    sieve_ops_per_sec: f64,
    basecase_max_limbs: usize,
    karatsuba_max_limbs: usize,
    mul_base_rows_max: u64,
    sqr_base_rows_max: u64,
    nt_effective: bool,
    nt_ratio: f64,
}

fn encode_static_cal_key(profile: &Profile, build_id: u64, bound_mask: u32) -> Vec<u64> {
    let mut w = Vec::with_capacity(64);
    w.push(STATIC_CAL_MAGIC);
    w.push(build_id);
    w.push(bound_mask as u64);
    w.push(profile.cores);
    w.push(profile.threads);
    w.push(u64::from(profile.os_places_threads));
    w.push(profile.core_map.len() as u64);
    for ct in &profile.core_map {
        w.push(ct.core_id);
        w.push(ct.threads.len() as u64);
        w.extend_from_slice(&ct.threads);
    }
    for lvl in [&profile.l1, &profile.l2, &profile.l3] {
        w.push(lvl.len() as u64);
        for ce in lvl {
            w.push(ce.size);
            w.push(ce.threads.len() as u64);
            w.extend_from_slice(&ce.threads);
        }
    }
    w
}

fn try_load_static_cal(
    path: &Path,
    profile: &Profile,
    build_id: u64,
    kernels: &Kernels,
) -> Option<PersistedStaticCal> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 24 || !bytes.len().is_multiple_of(8) {
        return None;
    }
    let words: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (&saved_w0, body) = words.split_last()?;
    if compute_limbs_w0(kernels, body) != saved_w0 {
        return None;
    }
    let key = encode_static_cal_key(profile, build_id, kernels.bound_mask);
    if !body.starts_with(&key) {
        return None;
    }
    let mut pos = key.len();
    let core_len = (*body.get(pos)?) as usize;
    pos += 1;
    if core_len > 65_536 || pos + core_len * 2 + NUM_KERNELS + 11 != body.len() {
        return None;
    }
    let mut core_ntt = Vec::with_capacity(core_len);
    for _ in 0..core_len {
        let cid = body[pos];
        let rate = f64::from_bits(body[pos + 1]);
        if !rate.is_finite() || rate <= 0.0 {
            return None;
        }
        core_ntt.push((cid, rate));
        pos += 2;
    }
    let mut kernel_rates = [0.0f64; NUM_KERNELS];
    for slot in &mut kernel_rates {
        let r = f64::from_bits(body[pos]);
        if !r.is_finite() || r <= 0.0 {
            return None;
        }
        *slot = r;
        pos += 1;
    }
    let read_pos_f64 = |idx: usize| -> Option<f64> {
        let v = f64::from_bits(body[idx]);
        (v.is_finite() && v > 0.0).then_some(v)
    };
    let ntt_bf_per_sec_per_worker = read_pos_f64(pos)?;
    let fft_bf_per_sec_per_worker = read_pos_f64(pos + 1)?;
    let crt_ops_per_sec_per_worker = read_pos_f64(pos + 2)?;
    let basecase_limbs_sq_per_sec = read_pos_f64(pos + 3)?;
    let sieve_ops_per_sec = read_pos_f64(pos + 4)?;
    let basecase_max_limbs = (body[pos + 5] as usize).clamp(4, 4096);
    let karatsuba_max_limbs = (body[pos + 6] as usize).max(basecase_max_limbs + 1);
    let mul_base_rows_max = body[pos + 7];
    let sqr_base_rows_max = body[pos + 8];
    let nt_effective = body[pos + 9] != 0;
    let nt_ratio = read_pos_f64(pos + 10)?;
    Some(PersistedStaticCal {
        core_ntt_bf_per_sec: core_ntt,
        kernel_rates,
        ntt_bf_per_sec_per_worker,
        fft_bf_per_sec_per_worker,
        crt_ops_per_sec_per_worker,
        basecase_limbs_sq_per_sec,
        sieve_ops_per_sec,
        basecase_max_limbs,
        karatsuba_max_limbs,
        mul_base_rows_max,
        sqr_base_rows_max,
        nt_effective,
        nt_ratio,
    })
}

fn save_static_cal(
    path: &Path,
    profile: &Profile,
    build_id: u64,
    kernels: &Kernels,
    sc: &PersistedStaticCal,
) {
    let mut words = encode_static_cal_key(profile, build_id, kernels.bound_mask);
    words.push(sc.core_ntt_bf_per_sec.len() as u64);
    for &(cid, rate) in &sc.core_ntt_bf_per_sec {
        words.push(cid);
        words.push(rate.to_bits());
    }
    for &kr in &sc.kernel_rates {
        words.push(kr.to_bits());
    }
    words.push(sc.ntt_bf_per_sec_per_worker.to_bits());
    words.push(sc.fft_bf_per_sec_per_worker.to_bits());
    words.push(sc.crt_ops_per_sec_per_worker.to_bits());
    words.push(sc.basecase_limbs_sq_per_sec.to_bits());
    words.push(sc.sieve_ops_per_sec.to_bits());
    words.push(sc.basecase_max_limbs as u64);
    words.push(sc.karatsuba_max_limbs as u64);
    words.push(sc.mul_base_rows_max);
    words.push(sc.sqr_base_rows_max);
    words.push(u64::from(sc.nt_effective));
    words.push(sc.nt_ratio.to_bits());
    let w0 = compute_limbs_w0(kernels, &words);
    words.push(w0);

    let mut bytes = Vec::with_capacity(words.len() * 8);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp_path = path.with_extension("tmp");
    if std::fs::write(&tmp_path, &bytes).is_ok() {
        if std::fs::rename(&tmp_path, path).is_err() {
            let _ = std::fs::write(path, &bytes);
            let _ = std::fs::remove_file(&tmp_path);
        }
    } else {
        let _ = std::fs::write(path, &bytes);
    }
}

/// Time all 24 kernels in `KERNEL_NAMES` order across active workers on L1-resident working sets.
fn measure_all_24_kernels(
    kernels: &Kernels,
    sched: &Scheduler,
    nt_effective: bool,
    ntt_bf_rate: f64,
    fft_bf_rate: f64,
) -> [f64; NUM_KERNELS] {
    let nw = sched.num_workers().max(1);
    let reps = if cfg!(debug_assertions) { 8usize } else { 256usize };
    let rows_max = kernels.mul_base_rows_max.load(Ordering::Relaxed);
    let sqr_rows_max = kernels.sqr_rows_max();
    let fft_sub_n = kernels.fft_sub_n();

    let p_u64 = PRIME_LOC1;
    let pr_fwd = kernels.ntt_prime(p_u64, 1);
    let pr_inv = kernels.ntt_prime(p_u64, (p_u64 + 1) / 2);
    let n_ntt = 256usize;
    let (_tw_ntt, itw_ntt) = build_ntt_tables(p_u64, n_ntt, false);
    let (tw_col, itw_col) = build_ntt_tables(p_u64, 16, false);
    let n_fft = 256usize;
    let (f_w, f_iw, _f_tw, f_itw) = build_fft_tables(n_fft);
    let crt_ps = largest_primes_for_two_adic(42, 4);
    let crt_plan = crate::ntt::CrtPlan::build(crt_ps);

    let mut per_worker_rates: Vec<[f64; NUM_KERNELS]> = vec![[1.0e6; NUM_KERNELS]; nw];
    let out_ptr = per_worker_rates.as_mut_ptr() as usize;

    sched.on_each_worker(|| {
        let wi = crate::par::current_worker_idx().min(nw - 1);
        let mut r = [1.0e6f64; NUM_KERNELS];

        // 0..6: Limb kernels (`add_n`, `sub_n`, `mul_1`, `addmul_1`, `mul_base`, `sqr_base`).
        let limb_len = 64usize;
        let a_u64 = [0x9E37_79B9_7F4A_7C15u64; 64];
        let b_u64 = [0xD1B5_4A32_D192_ED03u64; 64];
        let mut d_u64 = [0u64; 128];

        let t0 = Instant::now();
        for _ in 0..reps {
            let c = unsafe {
                (kernels.active.add_n)(d_u64.as_mut_ptr(), a_u64.as_ptr(), b_u64.as_ptr(), limb_len as u64)
            };
            std::hint::black_box(c);
        }
        r[0] = ((limb_len * reps) as f64 / t0.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t1 = Instant::now();
        for _ in 0..reps {
            let c = unsafe {
                (kernels.active.sub_n)(d_u64.as_mut_ptr(), a_u64.as_ptr(), b_u64.as_ptr(), limb_len as u64)
            };
            std::hint::black_box(c);
        }
        r[1] = ((limb_len * reps) as f64 / t1.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t2 = Instant::now();
        for _ in 0..reps {
            let c = unsafe {
                (kernels.active.mul_1)(d_u64.as_mut_ptr(), a_u64.as_ptr(), limb_len as u64, 0xBF58_476D_1CE4_E5B9)
            };
            std::hint::black_box(c);
        }
        r[2] = ((limb_len * reps) as f64 / t2.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t3 = Instant::now();
        for _ in 0..reps {
            let c = unsafe {
                (kernels.active.addmul_1)(d_u64.as_mut_ptr(), a_u64.as_ptr(), limb_len as u64, 0x94D0_49BB_1331_11EB)
            };
            std::hint::black_box(c);
        }
        r[3] = ((limb_len * reps) as f64 / t3.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let mb_n = 16usize;
        let lens = [mb_n as u64, mb_n as u64, rows_max];
        let t4 = Instant::now();
        for _ in 0..reps {
            unsafe {
                (kernels.active.mul_base)(d_u64.as_mut_ptr(), a_u64.as_ptr(), b_u64.as_ptr(), &lens);
            }
            std::hint::black_box(d_u64[0]);
        }
        r[4] = ((mb_n * mb_n * reps) as f64 / t4.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t5 = Instant::now();
        for _ in 0..reps {
            unsafe {
                (kernels.active.sqr_base)(
                    d_u64.as_mut_ptr(),
                    a_u64.as_ptr(),
                    mb_n as u64,
                    sqr_rows_max,
                );
            }
            std::hint::black_box(d_u64[0]);
        }
        r[5] = ((mb_n * mb_n * reps) as f64 / t5.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        // 6..12: FFT kernels (`fft_split`, `fft_fwd`, `fft_inv`, `fft_pmul`, `fft_pmuladd`, `fft_carry`).
        let mut fc_a = vec![0.25f64; 2 * n_fft];
        let mut fc_b = vec![0.25f64; 2 * n_fft];
        let mut fc_o = vec![0.0f64; 2 * n_fft];
        let f_in_limbs = vec![0x1234_5678_9ABC_DEF0u64; (2 * n_fft * 18).div_ceil(64)];
        let mut f_out_limbs = vec![0u64; f_in_limbs.len() + 4];
        let fscfg = FftSplitCfg {
            in_limbs: f_in_limbs.len() as u64,
            n_complex: n_fft as u64,
            coeff_bits: 18,
        };
        let f_reps = (reps / 4).max(2);

        let t6 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.fft_split)(fc_a.as_mut_ptr(), f_in_limbs.as_ptr(), f_w.as_ptr(), &fscfg);
            }
            std::hint::black_box(fc_a[0]);
        }
        r[6] = ((n_fft * f_reps) as f64 / t6.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        r[7] = fft_bf_rate;

        let t8 = Instant::now();
        for _ in 0..f_reps {
            fc_b.fill(0.25);
            unsafe {
                (kernels.active.fft_inv)(
                    fc_b.as_mut_ptr(),
                    f_itw.as_ptr(),
                    n_fft as u64,
                    fft_sub_n,
                );
            }
            std::hint::black_box(fc_b[0]);
        }
        let bf_f = ((n_fft / 2) * (n_fft.ilog2() as usize) * f_reps) as f64;
        r[8] = (bf_f / t8.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t9 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.fft_pmul)(fc_o.as_mut_ptr(), fc_a.as_ptr(), fc_b.as_ptr(), n_fft as u64);
            }
            std::hint::black_box(fc_o[0]);
        }
        r[9] = ((n_fft * f_reps) as f64 / t9.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let f_ptrs = [fc_a.as_ptr(), fc_b.as_ptr(), fc_a.as_ptr(), fc_b.as_ptr()];
        let t10 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.fft_pmuladd)(fc_o.as_mut_ptr(), &f_ptrs, n_fft as u64, 0);
            }
            std::hint::black_box(fc_o[0]);
        }
        r[10] = ((n_fft * f_reps) as f64 / t10.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        fc_o.fill(1.0);
        let t11 = Instant::now();
        for _ in 0..f_reps {
            let mut fccfg = FftCarryCfg {
                out_limbs: f_out_limbs.len() as u64,
                n_complex: n_fft as u64,
                coeff_bits: 18,
                top_carry: 0,
                max_err: 0.0,
                check_word: 0,
            };
            unsafe {
                (kernels.active.fft_carry)(f_out_limbs.as_mut_ptr(), fc_o.as_ptr(), f_iw.as_ptr(), &mut fccfg);
            }
            std::hint::black_box(f_out_limbs[0]);
        }
        r[11] = ((n_fft * f_reps) as f64 / t11.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        // 12..20: NTT kernels (`ntt_split`, `ntt_fwd`, `ntt_inv`, `ntt_fwd_cols`, `ntt_inv_cols`, `ntt_pmul`, `ntt_pmuladd`, `ntt_crt`).
        let mut nr_a = vec![1.0f64; n_ntt];
        let nr_b = vec![2.0f64; n_ntt];
        let mut nr_o = vec![0.0f64; n_ntt];
        let nscfg = NttSplitCfg {
            in_limbs: f_in_limbs.len() as u64,
            start_coeff: 0,
            count: n_ntt as u64,
            coeff_bits: 36,
            col_stride: 0,
            cols_per_row: 0,
            p: pr_fwd.p,
            inv_p: pr_fwd.inv_p,
        };
        let t12 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.ntt_split)(nr_a.as_mut_ptr(), f_in_limbs.as_ptr(), &nscfg);
            }
            std::hint::black_box(nr_a[0]);
        }
        r[12] = ((n_ntt * f_reps) as f64 / t12.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        r[13] = ntt_bf_rate;

        let t14 = Instant::now();
        for _ in 0..f_reps {
            nr_a.fill(1.0);
            unsafe {
                (kernels.active.ntt_inv)(nr_a.as_mut_ptr(), itw_ntt.as_ptr(), n_ntt as u64, &pr_inv);
            }
            std::hint::black_box(nr_a[0]);
        }
        let bf_n = ((n_ntt / 2) * (n_ntt.ilog2() as usize) * f_reps) as f64;
        r[14] = (bf_n / t14.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let mut col_scratch = vec![0.0f64; kernels.ntt_cols_scratch_f64s(16)];
        let col_cfg = kernels.ntt_cols_cfg(16, 16, &mut col_scratch);
        let bf_col = (8 * 4 * 16 * f_reps) as f64;
        let t15 = Instant::now();
        for _ in 0..f_reps {
            nr_a.fill(1.0);
            unsafe {
                (kernels.active.ntt_fwd_cols)(nr_a.as_mut_ptr(), tw_col.as_ptr(), &col_cfg, &pr_fwd);
            }
            std::hint::black_box(nr_a[0]);
        }
        r[15] = (bf_col / t15.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t16 = Instant::now();
        for _ in 0..f_reps {
            nr_a.fill(1.0);
            unsafe {
                (kernels.active.ntt_inv_cols)(
                    nr_a.as_mut_ptr(),
                    itw_col.as_ptr(),
                    &col_cfg,
                    &pr_inv,
                );
            }
            std::hint::black_box(nr_a[0]);
        }
        r[16] = (bf_col / t16.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let nvcfg = NttVecCfg {
            n: n_ntt as u64,
            p: pr_fwd.p,
            inv_p: pr_fwd.inv_p,
            sub_second: 0,
        };
        let t17 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.ntt_pmul)(nr_o.as_mut_ptr(), nr_a.as_ptr(), nr_b.as_ptr(), &nvcfg);
            }
            std::hint::black_box(nr_o[0]);
        }
        r[17] = ((n_ntt * f_reps) as f64 / t17.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let n_ptrs = [nr_a.as_ptr(), nr_b.as_ptr(), nr_a.as_ptr(), nr_b.as_ptr()];
        let t18 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.ntt_pmuladd)(nr_o.as_mut_ptr(), &n_ptrs, &nvcfg);
            }
            std::hint::black_box(nr_o[0]);
        }
        r[18] = ((n_ntt * f_reps) as f64 / t18.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let crt_ptrs = [nr_a.as_ptr(), nr_b.as_ptr(), nr_a.as_ptr(), nr_b.as_ptr()];
        let mut crt_out = vec![0u64; (n_ntt * 80).div_ceil(64) + 8];
        let t19 = Instant::now();
        for _ in 0..f_reps {
            let mut ccfg = NttCrtCfg {
                k_primes: 4,
                count: n_ntt as u64,
                coeff_bits: 80,
                out_limbs: crt_out.len() as u64,
                signed: 0,
                wrap_cyclic: 0,
                primes: crt_plan.primes_f64.as_ptr(),
                inv_primes: crt_plan.inv_primes_f64.as_ptr(),
                garner_inv: crt_plan.garner_inv.as_ptr(),
                prefix_prods: crt_plan.prefix_prods.as_ptr(),
                k_words: crt_plan.k_words as u64,
                carry_limbs: std::ptr::null_mut(),
                segs: NttCrtSegs::ONE,
            };
            let w = unsafe {
                (kernels.active.ntt_crt)(crt_out.as_mut_ptr(), crt_ptrs.as_ptr(), &mut ccfg)
            };
            std::hint::black_box(w);
        }
        r[19] = ((n_ntt * f_reps) as f64 / t19.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        // 20..24: Movement, digits, and check-word kernels (`gather`, `scatter`, `to_digits`, `check_n`).
        let mv_cfg = MoveCfg {
            rows: 8,
            cols: 32,
            outer_stride_bytes: 32 * 8,
            packed: 0,
            use_nt: u64::from(nt_effective),
        };
        let mut mv_outer = vec![0u8; 8 * 32 * 8];
        let mut mv_tile = vec![1.0f64; 8 * 32];
        let t20 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.gather)(mv_tile.as_mut_ptr(), mv_outer.as_ptr(), &mv_cfg);
            }
            std::hint::black_box(mv_tile[0]);
        }
        r[20] = ((8 * 32 * 8 * f_reps) as f64 / t20.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let t21 = Instant::now();
        for _ in 0..f_reps {
            unsafe {
                (kernels.active.scatter)(mv_outer.as_mut_ptr(), mv_tile.as_ptr(), &mv_cfg);
            }
            std::hint::black_box(mv_outer[0]);
        }
        r[21] = ((8 * 32 * 8 * f_reps) as f64 / t21.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let dig_in = [1_234_567_890_123_456_789u64; 32];
        let mut dig_out = [0u8; 32 * 19];
        let t22 = Instant::now();
        for _ in 0..f_reps {
            let w = unsafe {
                (kernels.active.to_digits)(dig_out.as_mut_ptr(), dig_in.as_ptr(), 32, 1)
            };
            std::hint::black_box(w);
        }
        r[22] = ((32 * f_reps) as f64 / t22.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        let mut chk3 = [0u64; 3];
        let t23 = Instant::now();
        for _ in 0..reps {
            let w = unsafe {
                (kernels.active.check_n)(a_u64.as_ptr(), limb_len as u64, &mut chk3, 1)
            };
            std::hint::black_box(w);
        }
        r[23] = ((limb_len * reps) as f64 / t23.elapsed().as_secs_f64().max(1e-9)).max(1e6);

        unsafe {
            *(out_ptr as *mut [f64; NUM_KERNELS]).add(wi) = r;
        }
    });

    let mut med = [1.0e6f64; NUM_KERNELS];
    let mut col = Vec::with_capacity(nw);
    for (k, slot) in med.iter_mut().enumerate() {
        col.clear();
        for wr in &per_worker_rates {
            col.push(wr[k]);
        }
        col.sort_by(|a, b| a.total_cmp(b));
        *slot = col[col.len() / 2].max(1e6);
    }
    med
}

/// Measure startup dynamic figures (`ram_bw_bps`, `sweep_step_sec`, `overlap_alpha`, and optional
/// `nt_effective` + `Store` idle/batch/beside-sweep figures) on every run (Section 10).
fn measure_dynamic_startup(
    kernels: &Kernels,
    shares: &Shares,
    sched: &Scheduler,
    store: Option<&mut Store>,
    known_nt: Option<(bool, f64)>,
) -> (bool, f64, f64, f64, f64, Vec<DeviceCalibration>) {
    let nw = sched.num_workers().max(1);
    let nt_cols = 4096usize;
    let nt_tile_cols = 64usize;
    let target_total_bytes = shares
        .total_l3
        .saturating_mul(6)
        .max((nw as u64).saturating_mul(shares.min_l2_share.saturating_mul(8)))
        .min((shares.total_ram / 8).max(shares.min_l2_share.saturating_mul(4))) as usize;
    let nt_rows = ((target_total_bytes / nw) / (nt_cols * 8)).max(8);
    let cfg_ord = MoveCfg {
        rows: nt_rows as u64,
        cols: nt_tile_cols as u64,
        outer_stride_bytes: (nt_cols * 8) as u64,
        packed: 0,
        use_nt: 0,
    };
    let cfg_nt = MoveCfg {
        use_nt: 1,
        ..cfg_ord
    };

    // Fault per-worker DRAM buffers on each worker's own thread/NUMA node before timing.
    let mut w_bufs: Vec<(Vec<f64>, Vec<f64>)> = (0..nw).map(|_| (Vec::new(), Vec::new())).collect();
    let w_bufs_ptr = w_bufs.as_mut_ptr() as usize;
    sched.on_each_worker(|| {
        let wi = crate::par::current_worker_idx().min(nw - 1);
        let slot = unsafe { &mut *(w_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
        slot.0 = vec![1.0f64; nt_rows * nt_tile_cols];
        slot.1 = vec![0.0f64; nt_rows * nt_cols];
    });

    let num_nt_tiles = nt_cols / nt_tile_cols;
    let (nt_effective, nt_ratio) = if let Some((nt_eff, ratio)) = known_nt {
        (nt_eff, ratio)
    } else {
        let nt_probe_trials = if cfg!(debug_assertions) { 1usize } else { 3usize };
        let mut sec_ord = f64::INFINITY;
        let mut sec_nt = f64::INFINITY;
        for _ in 0..nt_probe_trials {
            let t0 = Instant::now();
            sched.par_for(nw, 1, |wi| {
                let slot = unsafe { &mut *(w_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
                for ct in 0..num_nt_tiles {
                    unsafe {
                        (kernels.active.scatter)(
                            slot.1.as_mut_ptr().add(ct * nt_tile_cols) as *mut u8,
                            slot.0.as_ptr(),
                            &cfg_ord,
                        );
                    }
                }
            });
            sec_ord = sec_ord.min(t0.elapsed().as_secs_f64().max(1e-9));

            let t1 = Instant::now();
            sched.par_for(nw, 1, |wi| {
                let slot = unsafe { &mut *(w_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
                for ct in 0..num_nt_tiles {
                    unsafe {
                        (kernels.active.scatter)(
                            slot.1.as_mut_ptr().add(ct * nt_tile_cols) as *mut u8,
                            slot.0.as_ptr(),
                            &cfg_nt,
                        );
                    }
                }
            });
            sec_nt = sec_nt.min(t1.elapsed().as_secs_f64().max(1e-9));
            if sec_ord / sec_nt >= 0.96 {
                break;
            }
        }
        let ratio = sec_ord / sec_nt;
        (ratio >= 0.96, ratio)
    };

    // All-worker DRAM `gather` + `scatter` bandwidth on disjoint halves of pre-faulted DRAM buffers.
    let half_rows = (nt_rows / 2).max(4);
    let cfg_bw = MoveCfg {
        rows: half_rows as u64,
        ..(if nt_effective { cfg_nt } else { cfg_ord })
    };
    let bw_trials = if cfg!(debug_assertions) { 1usize } else { 3usize };
    let mut bw_elapsed = f64::INFINITY;
    for _ in 0..bw_trials {
        let t_bw = Instant::now();
        sched.par_for(nw, 1, |wi| {
            let slot = unsafe { &mut *(w_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
            for ct in 0..num_nt_tiles {
                unsafe {
                    let src_p = slot.1.as_ptr().add(ct * nt_tile_cols);
                    let dst_p = slot.1.as_mut_ptr().add(half_rows * nt_cols + ct * nt_tile_cols);
                    (kernels.active.gather)(slot.0.as_mut_ptr(), src_p as *const u8, &cfg_bw);
                    (kernels.active.scatter)(dst_p as *mut u8, slot.0.as_ptr(), &cfg_bw);
                }
            }
            std::hint::black_box(slot.0[0]);
        });
        bw_elapsed = bw_elapsed.min(t_bw.elapsed().as_secs_f64());
    }
    bw_elapsed = bw_elapsed.max(1e-9);
    let ram_bw_bps = ((nw * half_rows * nt_cols * 8 * 2) as f64 / bw_elapsed).max(8.0e9);
    drop(w_bufs);

    // All-worker sweep step (`gather` + `ntt_fwd` + `scatter`) AND isolated move vs compute
    // steps so `overlap_alpha` is measured directly rather than hardcoded.
    let p_u64 = largest_primes_for_two_adic(42, 1)[0];
    let pr = kernels.ntt_prime(p_u64, 1);
    let (tw, _) = build_ntt_tables(p_u64, 1024, false);
    let tile_len = 2048usize;
    let sweeps_per_w = if cfg!(debug_assertions) { 8usize } else { 24usize };
    let sw_trials = if cfg!(debug_assertions) { 1usize } else { 3usize };
    let mut sw_bufs: Vec<(Vec<f64>, Vec<f64>)> = (0..nw)
        .map(|_| (vec![1.0f64; tile_len], vec![0.0f64; tile_len]))
        .collect();
    let sw_bufs_ptr = sw_bufs.as_mut_ptr() as usize;
    let mc = MoveCfg {
        rows: 1,
        cols: tile_len as u64,
        outer_stride_bytes: (tile_len * 8) as u64,
        packed: 0,
        use_nt: u64::from(nt_effective),
    };
    let mut move_elapsed = f64::INFINITY;
    let mut comp_elapsed = f64::INFINITY;
    let mut sw_elapsed = f64::INFINITY;
    for trial in 0..=sw_trials {
        let t_mv = Instant::now();
        sched.par_for(nw, 1, |wi| {
            let (src, dst) = unsafe { &mut *(sw_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
            for _ in 0..sweeps_per_w {
                unsafe {
                    (kernels.active.gather)(dst.as_mut_ptr(), src.as_ptr() as *const u8, &mc);
                    (kernels.active.scatter)(src.as_mut_ptr() as *mut u8, dst.as_ptr(), &mc);
                }
            }
            std::hint::black_box(src[0]);
        });
        let el_mv = t_mv.elapsed().as_secs_f64();

        let t_cp = Instant::now();
        sched.par_for(nw, 1, |wi| {
            let (_, dst) = unsafe { &mut *(sw_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
            dst.fill(1.0);
            for _ in 0..sweeps_per_w {
                unsafe {
                    (kernels.active.ntt_fwd)(dst.as_mut_ptr(), tw.as_ptr(), 1024, &pr);
                }
            }
            std::hint::black_box(dst[0]);
        });
        let el_cp = t_cp.elapsed().as_secs_f64();

        let t_sw = Instant::now();
        sched.par_for(nw, 1, |wi| {
            let (src, dst) = unsafe { &mut *(sw_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi) };
            src.fill(1.0);
            for _ in 0..sweeps_per_w {
                unsafe {
                    (kernels.active.gather)(dst.as_mut_ptr(), src.as_ptr() as *const u8, &mc);
                    (kernels.active.ntt_fwd)(dst.as_mut_ptr(), tw.as_ptr(), 1024, &pr);
                    (kernels.active.scatter)(src.as_mut_ptr() as *mut u8, dst.as_ptr(), &mc);
                }
            }
            std::hint::black_box(src[0]);
        });
        let el_sw = t_sw.elapsed().as_secs_f64();
        if trial > 0 {
            move_elapsed = move_elapsed.min(el_mv);
            comp_elapsed = comp_elapsed.min(el_cp);
            sw_elapsed = sw_elapsed.min(el_sw);
        }
    }
    sw_elapsed = sw_elapsed.max(1e-9);
    let sweep_step_sec = sw_elapsed / (sweeps_per_w as f64);
    let move_step_sec = move_elapsed.max(1e-9) / (sweeps_per_w as f64);
    let comp_step_sec = comp_elapsed.max(1e-9) / (sweeps_per_w as f64);
    let hi_step = comp_step_sec.max(move_step_sec);
    let lo_step = comp_step_sec.min(move_step_sec);
    let overlap_alpha = if lo_step > 1e-12 {
        ((sweep_step_sec - hi_step) / lo_step).clamp(0.05, 1.0)
    } else {
        0.35
    };

    if std::env::var_os("SUPI_TRACE").is_some() {
        eprintln!(
            "[TRACE CAL] target_mb={} nt_effective={} ram_bw={:.2} GB/s sweep_us={:.2} alpha={:.3}",
            target_total_bytes / (1024 * 1024),
            nt_effective,
            ram_bw_bps * 1e-9,
            sweep_step_sec * 1e6,
            overlap_alpha,
        );
    }

    // Store device calibration (idle rate, smallest batch range size, and concurrent beside-sweep
    // rate + beside-sweep step time) when the run may use the store (Section 10).
    let mut dev_cals = Vec::new();
    if let Some(st) = store {
        let telemetry = crate::check::Telemetry::default();
        for (d_idx, dev) in st.devices.iter_mut().enumerate() {
            let min_probe = if !cfg!(debug_assertions) && dev.capacity() >= 512 * 1024 * 1024 {
                64 * 1024 * 1024
            } else if !cfg!(debug_assertions) && dev.capacity() >= 256 * 1024 * 1024 {
                16 * 1024 * 1024
            } else {
                0
            };
            let probe_bytes = ((dev.piece_limbs * 8) * dev.stripe_width().max(1) * 8)
                .max(min_probe)
                .min(dev.capacity() as usize);
            if probe_bytes >= dev.piece_limbs * 8
                && let Ok(ext) = dev.allocate(d_idx, probe_bytes as u64)
            {
                let range = [crate::store::ExtentRange {
                    piece_start: 0,
                    piece_count: ext.num_pieces,
                    host_limb_offset: 0,
                }];
                if let Ok(full_plan) = dev.plan_slot_batch(&ext, &range) {
                    let mut slot_buf =
                        crate::store::AlignedSlotBuf::new_uninit(full_plan.required_buf_bytes);
                    unsafe {
                        slot_buf.as_import_slice_mut()[..full_plan.required_buf_bytes].fill(0xA5);
                    }
                    if min_probe > 0 {
                        let warm_cnt = AtomicU64::new(0);
                        let warm_r = [crate::store::ExtentRange {
                            piece_start: 0,
                            piece_count: (dev.stripe_width().max(1) as u64).min(ext.num_pieces),
                            host_limb_offset: 0,
                        }];
                        if let Ok(warm_plan) = dev.plan_slot_batch(&ext, &warm_r) {
                            let _ = dev.write_planned_slots(&warm_plan, &slot_buf, &warm_cnt);
                        }
                    }

                    // 1. Idle DMA transfer rate with workers idle (matches `Loader` pure-DMA path).
                    let idle_cnt = AtomicU64::new(0);
                    let ti = Instant::now();
                    let ok_w = dev
                        .write_planned_slots(&full_plan, &slot_buf, &idle_cnt)
                        .is_ok();
                    let ok_r = dev
                        .read_planned_slots(&full_plan, &mut slot_buf, &telemetry, &idle_cnt)
                        .is_ok();
                    let idle_sec = ti.elapsed().as_secs_f64().max(1e-9);
                    if ok_w && ok_r {
                        let moved = idle_cnt.load(Ordering::Relaxed) as f64;
                        dev.idle_rate_bps = (moved / idle_sec).max(1.0e7);

                        // 2. Smallest batch range size reaching >= 85% of `idle_rate_bps`.
                        let sample_pieces = ext.num_pieces.min(64).max(1);
                        let mut chosen_batch_pieces = sample_pieces;
                        for cand_p in [1u64, 2, 4, 8, 16, 32] {
                            if cand_p > sample_pieces {
                                break;
                            }
                            let mut sub_ranges = Vec::new();
                            let mut p0 = 0u64;
                            while p0 < sample_pieces {
                                let cnt = (sample_pieces - p0).min(cand_p);
                                sub_ranges.push(crate::store::ExtentRange {
                                    piece_start: p0,
                                    piece_count: cnt,
                                    host_limb_offset: (p0 as usize) * dev.piece_limbs,
                                });
                                p0 += cnt;
                            }
                            if let Ok(sub_plan) = dev.plan_slot_batch(&ext, &sub_ranges) {
                                let batch_cnt = AtomicU64::new(0);
                                let tb = Instant::now();
                                let ok_b = dev
                                    .read_planned_slots(
                                        &sub_plan,
                                        &mut slot_buf,
                                        &telemetry,
                                        &batch_cnt,
                                    )
                                    .is_ok();
                                let b_sec = tb.elapsed().as_secs_f64().max(1e-9);
                                if ok_b {
                                    let b_rate =
                                        (batch_cnt.load(Ordering::Relaxed) as f64) / b_sec;
                                    chosen_batch_pieces = cand_p;
                                    if b_rate >= dev.idle_rate_bps * 0.85 {
                                        break;
                                    }
                                }
                            }
                        }
                        dev.min_batch_range_bytes =
                            chosen_batch_pieces.saturating_mul(dev.piece_slot_bytes);

                        // 3. Concurrent beside-sweep DMA rate & beside-sweep step time:
                        //    run device `write_planned_slots` + `read_planned_slots` on a scope
                        //    thread while all workers repeat the sweep step.
                        let transfer_started = AtomicBool::new(false);
                        let transfer_done = AtomicBool::new(false);
                        let beside_cnt = AtomicU64::new(0);
                        let mut beside_io_sec = idle_sec;
                        let mut beside_io_ok = false;
                        let mut beside_sw_sec = 0.0f64;
                        let mut beside_sw_steps = 0usize;
                        let dev_ref: &crate::store::ExtentDevice = &*dev;
                        let plan_ref = &full_plan;
                        let slot_mut = &mut slot_buf;
                        let tel_ref = &telemetry;

                        std::thread::scope(|s| {
                            let io_handle = s.spawn(|| {
                                transfer_started.store(true, Ordering::Release);
                                let t_io = Instant::now();
                                let w_ok = dev_ref
                                    .write_planned_slots(plan_ref, slot_mut, &beside_cnt)
                                    .is_ok();
                                let r_ok = if w_ok {
                                    dev_ref
                                        .read_planned_slots(
                                            plan_ref,
                                            slot_mut,
                                            tel_ref,
                                            &beside_cnt,
                                        )
                                        .is_ok()
                                } else {
                                    false
                                };
                                let sec = t_io.elapsed().as_secs_f64().max(1e-9);
                                transfer_done.store(true, Ordering::Release);
                                (w_ok && r_ok, sec)
                            });

                            while !transfer_started.load(Ordering::Acquire) {
                                std::hint::spin_loop();
                            }
                            loop {
                                let t_step = Instant::now();
                                sched.par_for(nw, 1, |wi| {
                                    let (src, dst) = unsafe {
                                        &mut *(sw_bufs_ptr as *mut (Vec<f64>, Vec<f64>)).add(wi)
                                    };
                                    src.fill(1.0);
                                    for _ in 0..sweeps_per_w {
                                        unsafe {
                                            (kernels.active.gather)(
                                                dst.as_mut_ptr(),
                                                src.as_ptr() as *const u8,
                                                &mc,
                                            );
                                            (kernels.active.ntt_fwd)(
                                                dst.as_mut_ptr(),
                                                tw.as_ptr(),
                                                1024,
                                                &pr,
                                            );
                                            (kernels.active.scatter)(
                                                src.as_mut_ptr() as *mut u8,
                                                dst.as_ptr(),
                                                &mc,
                                            );
                                        }
                                    }
                                    std::hint::black_box(src[0]);
                                });
                                beside_sw_sec += t_step.elapsed().as_secs_f64();
                                beside_sw_steps += sweeps_per_w;
                                if transfer_done.load(Ordering::Acquire) {
                                    break;
                                }
                            }

                            if let Ok((ok, sec)) = io_handle.join() {
                                beside_io_ok = ok;
                                beside_io_sec = sec;
                            }
                        });

                        if beside_io_ok {
                            let b_moved = beside_cnt.load(Ordering::Relaxed) as f64;
                            dev.beside_sweep_rate_bps = (b_moved / beside_io_sec).max(1.0e7);
                            dev.sweep_step_beside_sec = if beside_sw_steps > 0 {
                                (beside_sw_sec / (beside_sw_steps as f64)).max(sweep_step_sec)
                            } else {
                                sweep_step_sec
                            };
                        } else {
                            dev.beside_sweep_rate_bps = dev.idle_rate_bps;
                            dev.sweep_step_beside_sec = sweep_step_sec;
                        }
                    }
                }
                dev.free(ext);
            }
        }
        st.devices.sort_by(|a, b| {
            b.beside_sweep_rate_bps
                .total_cmp(&a.beside_sweep_rate_bps)
        });
        for dev in &st.devices {
            if std::env::var_os("SUPI_TRACE").is_some() {
                eprintln!(
                    "[TRACE DEV] {} cap={:.2}GB idle={:.3}GB/s beside={:.3}GB/s step_beside={:.2}us min_batch={}KB",
                    dev.name,
                    (dev.capacity() as f64) / 1.073741824e9,
                    dev.idle_rate_bps * 1e-9,
                    dev.beside_sweep_rate_bps * 1e-9,
                    dev.sweep_step_beside_sec * 1e6,
                    dev.min_batch_range_bytes / 1024,
                );
            }
            dev_cals.push(DeviceCalibration {
                name: dev.name.clone(),
                idle_rate_bps: dev.idle_rate_bps,
                beside_sweep_rate_bps: dev.beside_sweep_rate_bps,
                sweep_step_beside_sec: dev.sweep_step_beside_sec.max(sweep_step_sec),
                min_batch_range_bytes: dev.min_batch_range_bytes,
            });
        }
    }
    drop(sw_bufs);
    (
        nt_effective,
        nt_ratio,
        ram_bw_bps,
        sweep_step_sec,
        overlap_alpha,
        dev_cals,
    )
}

impl Calibration {
    /// Deterministic synthetic calibration for `--profile` / `--calibration` planning and tests.
    pub fn synthetic(
        ntt_bf_per_sec_per_worker: f64,
        ram_bw_bps: f64,
        nt_effective: bool,
        devices: Vec<DeviceCalibration>,
    ) -> Self {
        let bf = ntt_bf_per_sec_per_worker.max(1.0);
        Self {
            core_ntt_bf_per_sec: Vec::new(),
            kernel_rates: [bf; NUM_KERNELS],
            ntt_bf_per_sec_per_worker,
            fft_bf_per_sec_per_worker: ntt_bf_per_sec_per_worker * 1.65,
            crt_ops_per_sec_per_worker: ntt_bf_per_sec_per_worker * 0.9,
            basecase_limbs_sq_per_sec: ntt_bf_per_sec_per_worker * 1.4,
            sieve_ops_per_sec: 2.5e9,
            basecase_max_limbs: 32,
            karatsuba_max_limbs: 128,
            mul_base_rows_max: u64::MAX,
            sqr_base_rows_max: u64::MAX,
            ram_bw_bps,
            sweep_step_sec: 1.0e-5,
            overlap_alpha: 0.35,
            nt_effective,
            nt_ratio: if nt_effective { 1.15 } else { 0.85 },
            reused_static: false,
            devices,
        }
    }

    /// Measure startup calibration figures on the live host without static cache persistence.
    pub fn measure(
        kernels: &Kernels,
        shares: &Shares,
        sched: &Scheduler,
        store: Option<&mut Store>,
    ) -> Self {
        let dummy_prof = Profile::synthetic(
            shares.cores,
            shares.threads,
            shares.min_l1_share,
            shares.min_l2_share,
            shares.total_l3,
            shares.total_ram,
            &[],
            &[],
        );
        Self::measure_with_cache(kernels, &dummy_prof, shares, sched, store, None, 0)
    }

    /// Measure startup calibration figures on the live host (Section 10), reusing persisted static
    /// kernel/core/crossover/NT figures when `(cores, threads, core_map, caches, build_id, bound_mask)`
    /// match, while always measuring `ram_bw_bps`, `sweep_step_sec`, `overlap_alpha`, and `store`
    /// figures (when `store` is present) at startup.
    pub fn measure_with_cache(
        kernels: &Kernels,
        profile: &Profile,
        shares: &Shares,
        sched: &Scheduler,
        store: Option<&mut Store>,
        cache_path: Option<&Path>,
        build_id: u64,
    ) -> Self {
        kernels.configure_from_shares(shares);
        if let Some(path) = cache_path
            && let Some(sc) = try_load_static_cal(path, profile, build_id, kernels)
        {
            kernels
                .mul_base_rows_max
                .store(sc.mul_base_rows_max, Ordering::Relaxed);
            kernels
                .sqr_base_rows_max
                .store(sc.sqr_base_rows_max, Ordering::Relaxed);
            let (nt_effective, nt_ratio, ram_bw_bps, sweep_step_sec, overlap_alpha, dev_cals) =
                measure_dynamic_startup(
                    kernels,
                    shares,
                    sched,
                    store,
                    Some((sc.nt_effective, sc.nt_ratio)),
                );
            sched.on_each_worker(|| {
                crate::limb::clear_tls_scratch();
                crate::fft::clear_tls_fwd_pool();
                crate::ntt::clear_tls_bufs();
            });
            return Self {
                core_ntt_bf_per_sec: sc.core_ntt_bf_per_sec,
                kernel_rates: sc.kernel_rates,
                ntt_bf_per_sec_per_worker: sc.ntt_bf_per_sec_per_worker,
                fft_bf_per_sec_per_worker: sc.fft_bf_per_sec_per_worker,
                crt_ops_per_sec_per_worker: sc.crt_ops_per_sec_per_worker,
                basecase_limbs_sq_per_sec: sc.basecase_limbs_sq_per_sec,
                sieve_ops_per_sec: sc.sieve_ops_per_sec,
                basecase_max_limbs: sc.basecase_max_limbs,
                karatsuba_max_limbs: sc.karatsuba_max_limbs,
                mul_base_rows_max: sc.mul_base_rows_max,
                sqr_base_rows_max: sc.sqr_base_rows_max,
                ram_bw_bps,
                sweep_step_sec,
                overlap_alpha,
                nt_effective,
                nt_ratio,
                reused_static: true,
                devices: dev_cals,
            };
        }

        // 1. Dynamic startup probe + non-temporal store/prefetch effectiveness.
        let (nt_effective, nt_ratio, ram_bw_bps, sweep_step_sec, overlap_alpha, dev_cals) =
            measure_dynamic_startup(kernels, shares, sched, store, None);

        // 2. Concurrent per-worker `ntt_fwd` and `fft_fwd` throughput, recording each worker's
        //    individual `ntt_fwd` speed on its pinned core (`core_ntt_bf_per_sec`).
        let nw = sched.num_workers().max(1);
        let median_sec = |times: &mut [f64]| -> f64 {
            times.sort_by(|a, b| a.total_cmp(b));
            times[times.len() / 2].max(1e-9)
        };
        let sync_workers = |ready: &std::sync::atomic::AtomicUsize| {
            if nw > 1 {
                ready.fetch_add(1, Ordering::AcqRel);
                let t_wait = Instant::now();
                while ready.load(Ordering::Acquire) < nw {
                    if t_wait.elapsed() > std::time::Duration::from_millis(5) {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        };
        let mut w_times = vec![0.0f64; nw];
        let w_times_ptr = w_times.as_mut_ptr() as usize;
        let mut w_best_ntt = vec![f64::INFINITY; nw];

        let p_u64 = largest_primes_for_two_adic(42, 1)[0];
        let pr = kernels.ntt_prime(p_u64, 1);
        let n_bench = 1024usize;
        let (tw, _) = build_ntt_tables(p_u64, n_bench, false);
        let iters = if cfg!(debug_assertions) { 32usize } else { 3072usize };
        let mut ntt_bufs: Vec<Vec<f64>> = (0..nw).map(|_| vec![1.0f64; n_bench]).collect();
        let ntt_bufs_ptr = ntt_bufs.as_mut_ptr() as usize;
        let ntt_trials = if cfg!(debug_assertions) { 1usize } else { 3usize };
        let mut sec_ntt = f64::INFINITY;
        for trial in 0..=ntt_trials {
            let ready = std::sync::atomic::AtomicUsize::new(0);
            sched.on_each_worker(|| {
                let wi = crate::par::current_worker_idx().min(nw - 1);
                let ntt_buf = unsafe { &mut *(ntt_bufs_ptr as *mut Vec<f64>).add(wi) };
                ntt_buf.fill(1.0);
                sync_workers(&ready);
                let tn = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        (kernels.active.ntt_fwd)(
                            ntt_buf.as_mut_ptr(),
                            tw.as_ptr(),
                            n_bench as u64,
                            &pr,
                        );
                    }
                }
                let el = tn.elapsed().as_secs_f64();
                std::hint::black_box(ntt_buf[0]);
                unsafe {
                    *(w_times_ptr as *mut f64).add(wi) = el;
                }
            });
            if trial > 0 {
                for (wi, &t) in w_times.iter().enumerate() {
                    w_best_ntt[wi] = w_best_ntt[wi].min(t.max(1e-9));
                }
                let mut tmp = w_times.clone();
                sec_ntt = sec_ntt.min(median_sec(&mut tmp));
            }
        }
        sec_ntt = sec_ntt.max(1e-9);
        drop(ntt_bufs);
        let bf_ntt = ((n_bench / 2) * (n_bench.ilog2() as usize) * iters) as f64;
        let ntt_bf_rate = (bf_ntt / sec_ntt).max(1e6);

        let mut core_ntt: Vec<(u64, f64)> = Vec::new();
        for (wi, w) in shares.workers.iter().enumerate() {
            if w.smt_slot == 0 && !core_ntt.iter().any(|&(cid, _)| cid == w.core_id) {
                let rate = if wi < nw && w_best_ntt[wi].is_finite() {
                    (bf_ntt / w_best_ntt[wi]).max(1e6)
                } else {
                    ntt_bf_rate
                };
                core_ntt.push((w.core_id, rate));
            }
        }

        let (_, _, f_tw, _) = build_fft_tables(n_bench);
        let fft_sub_n = kernels.fft_sub_n();
        let fft_outer = if cfg!(debug_assertions) { 1usize } else { 24usize };
        let fft_inner = 32usize;
        let fft_iters = fft_outer * fft_inner;
        let mut fft_bufs: Vec<Vec<f64>> = (0..nw).map(|_| vec![0.5f64; 2 * n_bench]).collect();
        let fft_bufs_ptr = fft_bufs.as_mut_ptr() as usize;
        let mut sec_fft = f64::INFINITY;
        for trial in 0..=ntt_trials {
            let ready = std::sync::atomic::AtomicUsize::new(0);
            sched.on_each_worker(|| {
                let wi = crate::par::current_worker_idx().min(nw - 1);
                let fft_buf = unsafe { &mut *(fft_bufs_ptr as *mut Vec<f64>).add(wi) };
                sync_workers(&ready);
                let tf = Instant::now();
                for _ in 0..fft_outer {
                    fft_buf.fill(0.5);
                    for _ in 0..fft_inner {
                        unsafe {
                            (kernels.active.fft_fwd)(
                                fft_buf.as_mut_ptr(),
                                f_tw.as_ptr(),
                                n_bench as u64,
                                fft_sub_n,
                            );
                        }
                    }
                }
                let el = tf.elapsed().as_secs_f64();
                std::hint::black_box(fft_buf[0]);
                unsafe {
                    *(w_times_ptr as *mut f64).add(wi) = el;
                }
            });
            if trial > 0 {
                sec_fft = sec_fft.min(median_sec(&mut w_times));
            }
        }
        sec_fft = sec_fft.max(1e-9);
        drop(fft_bufs);
        let bf_fft = ((n_bench / 2) * (n_bench.ilog2() as usize) * fft_iters) as f64;
        let fft_bf_rate = (bf_fft / sec_fft).max(1e6);

        // 3. `mul_base` and `sqr_base` internal row-vs-vector crossovers.
        let (mul_base_rows_max, sqr_base_rows_max) = {
            let reps_mb = if cfg!(debug_assertions) { 8usize } else { 192usize };
            let a = [0x9E37_79B9_7F4A_7C15u64; 32];
            let b = [0xD1B5_4A32_D192_ED03u64; 32];
            let mut mb_rows_max = u64::MAX;
            {
                let mut time_mul_way = |n: usize, rows_max: u64| -> f64 {
                    let lens = [n as u64, n as u64, rows_max];
                    let mut best = f64::INFINITY;
                    for _ in 0..ntt_trials {
                        sched.par_for(nw, 1, |wi| {
                            let mut out = [0u64; 64];
                            let t0 = Instant::now();
                            for _ in 0..reps_mb {
                                unsafe {
                                    (kernels.active.mul_base)(
                                        out.as_mut_ptr(),
                                        a.as_ptr(),
                                        b.as_ptr(),
                                        &lens,
                                    );
                                }
                                std::hint::black_box(&mut out);
                            }
                            unsafe {
                                *(w_times_ptr as *mut f64).add(wi) = t0.elapsed().as_secs_f64();
                            }
                        });
                        best = best.min(median_sec(&mut w_times));
                    }
                    best
                };
                for n in [32usize, 28, 24, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9] {
                    if time_mul_way(n, 0) < time_mul_way(n, u64::MAX) * 0.97 {
                        mb_rows_max = (n - 1) as u64;
                    } else {
                        break;
                    }
                }
            }
            let mut sb_rows_max = u64::MAX;
            {
                let mut time_sqr_way = |n: usize, rows_max: u64| -> f64 {
                    let mut best = f64::INFINITY;
                    for _ in 0..ntt_trials {
                        sched.par_for(nw, 1, |wi| {
                            let mut out = [0u64; 64];
                            let t0 = Instant::now();
                            for _ in 0..reps_mb {
                                unsafe {
                                    (kernels.active.sqr_base)(
                                        out.as_mut_ptr(),
                                        a.as_ptr(),
                                        n as u64,
                                        rows_max,
                                    );
                                }
                                std::hint::black_box(&mut out);
                            }
                            unsafe {
                                *(w_times_ptr as *mut f64).add(wi) = t0.elapsed().as_secs_f64();
                            }
                        });
                        best = best.min(median_sec(&mut w_times));
                    }
                    best
                };
                for n in [32usize, 28, 24, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9] {
                    if time_sqr_way(n, 0) < time_sqr_way(n, u64::MAX) * 0.97 {
                        sb_rows_max = (n - 1) as u64;
                    } else {
                        break;
                    }
                }
            }
            (mb_rows_max, sb_rows_max)
        };
        kernels
            .mul_base_rows_max
            .store(mul_base_rows_max, Ordering::Relaxed);
        kernels
            .sqr_base_rows_max
            .store(sqr_base_rows_max, Ordering::Relaxed);

        // 4. Measure all 24 kernels across workers (`kernel_rates`), deriving real
        //    `basecase_limbs_sq_per_sec` from measured `mul_base` and `sqr_base`.
        let kernel_rates =
            measure_all_24_kernels(kernels, sched, nt_effective, ntt_bf_rate, fft_bf_rate);
        let basecase_limbs_sq_per_sec = (0.5 * (kernel_rates[4] + kernel_rates[5])).max(1e6);

        // 5. Basecase -> Karatsuba and Karatsuba -> FFT crossovers across all active workers.
        //    Evaluate all candidates up to 64 limbs without breaking on an early dip between
        //    scalar `rows_max` and vector `mul_base`.
        let mut basecase_max_limbs = 16usize;
        let reps = if cfg!(debug_assertions) { 24usize } else { 128usize };
        for cand in [16usize, 24, 32, 36, 40, 44, 48, 56, 64] {
            let kara_cut = if cand <= 16 {
                8
            } else {
                basecase_max_limbs.min(cand - 1).max(8)
            };
            let c_reps = if cfg!(debug_assertions) {
                reps
            } else {
                ((reps * 1024) / (cand * cand).max(1)).max(reps)
            };
            let a = vec![0x1234_5678_9ABC_DEF1u64; cand];
            let b = vec![0xFEDC_BA98_7654_3211u64; cand];
            sched.par_for(nw, 1, |_| {
                let r0 = mul_limbs_basecase(kernels, &a, &b);
                crate::limb::put_limb_buf(r0);
                let r1 = mul_limbs_karatsuba(kernels, &a, &b, kara_cut);
                crate::limb::put_limb_buf(r1);
            });
            let mut sb = f64::INFINITY;
            let mut sk = f64::INFINITY;
            for _ in 0..ntt_trials {
                sched.par_for(nw, 1, |wi| {
                    let tb = Instant::now();
                    for _ in 0..c_reps {
                        let r = mul_limbs_basecase(kernels, &a, &b);
                        crate::limb::put_limb_buf(r);
                    }
                    unsafe {
                        *(w_times_ptr as *mut f64).add(wi) = tb.elapsed().as_secs_f64();
                    }
                });
                sb = sb.min(median_sec(&mut w_times));
                sched.par_for(nw, 1, |wi| {
                    let tk = Instant::now();
                    for _ in 0..c_reps {
                        let r = mul_limbs_karatsuba(kernels, &a, &b, kara_cut);
                        crate::limb::put_limb_buf(r);
                    }
                    unsafe {
                        *(w_times_ptr as *mut f64).add(wi) = tk.elapsed().as_secs_f64();
                    }
                });
                sk = sk.min(median_sec(&mut w_times));
            }
            if !(sk < sb * 0.95) {
                basecase_max_limbs = cand;
            }
        }

        let fft_bench_cache = FftTableCache::default();
        let mut karatsuba_max_limbs = basecase_max_limbs.max(48);
        let mut fft_wins = 0usize;
        let reps_k = if cfg!(debug_assertions) { 12usize } else { 128usize };
        for cand in [48usize, 64, 80, 96, 112, 128, 144, 160, 192, 256] {
            if cand <= basecase_max_limbs {
                continue;
            }
            let out_bits_c = ((cand as u64) * 128) + 64;
            let Some((n_c, cb_c)) = choose_fft_params(out_bits_c, false, shares.min_l2_share)
            else {
                fft_wins = 0;
                karatsuba_max_limbs = cand;
                continue;
            };
            let _ = fft_bench_cache.get(n_c);
            let a = vec![0x9E37_79B9_7F4A_7C15u64; cand];
            let b = vec![0xBF58_476D_1CE4_E5B9u64; cand];
            sched.par_for(nw, 1, |_| {
                let rk = mul_limbs_karatsuba(kernels, &a, &b, basecase_max_limbs);
                crate::limb::put_limb_buf(rk);
                if let Some((rf, _, _)) = fft_mul(kernels, &fft_bench_cache, &a, &b, n_c, cb_c) {
                    crate::limb::put_limb_buf(rf);
                }
            });
            let mut sk_c = f64::INFINITY;
            let mut sf_c = f64::INFINITY;
            let ok_fft = AtomicBool::new(true);
            for _ in 0..ntt_trials {
                sched.par_for(nw, 1, |wi| {
                    let tk_c = Instant::now();
                    for _ in 0..reps_k {
                        let r = mul_limbs_karatsuba(kernels, &a, &b, basecase_max_limbs);
                        crate::limb::put_limb_buf(r);
                    }
                    unsafe {
                        *(w_times_ptr as *mut f64).add(wi) = tk_c.elapsed().as_secs_f64();
                    }
                });
                sk_c = sk_c.min(median_sec(&mut w_times));
                sched.par_for(nw, 1, |wi| {
                    let tf_c = Instant::now();
                    for _ in 0..reps_k {
                        if let Some((r, _, _)) =
                            fft_mul(kernels, &fft_bench_cache, &a, &b, n_c, cb_c)
                        {
                            crate::limb::put_limb_buf(r);
                        } else {
                            ok_fft.store(false, Ordering::Relaxed);
                            break;
                        }
                    }
                    unsafe {
                        *(w_times_ptr as *mut f64).add(wi) = tf_c.elapsed().as_secs_f64();
                    }
                });
                sf_c = sf_c.min(median_sec(&mut w_times));
            }
            if ok_fft.load(Ordering::Relaxed) && sf_c < sk_c * 0.95 {
                fft_wins += 1;
                if fft_wins >= 2 {
                    break;
                }
            } else {
                fft_wins = 0;
                karatsuba_max_limbs = cand;
            }
        }

        // 6. Sieve operation rate & planar `ntt_split` + `ntt_crt` throughput.
        let mut sieve_bits = vec![0u64; 2048];
        let ts = Instant::now();
        let mut sieve_ops = 0u64;
        for p in [3usize, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let mut idx = p * p;
            while idx < sieve_bits.len() * 64 {
                sieve_bits[idx >> 6] |= 1u64 << (idx & 63);
                idx += 2 * p;
                sieve_ops += 1;
            }
        }
        std::hint::black_box(&sieve_bits);
        let sieve_ops_per_sec =
            ((sieve_ops as f64) / ts.elapsed().as_secs_f64().max(1e-9)).max(1e8);

        let crt_ps = largest_primes_for_two_adic(42, 4);
        let crt_plan = crate::ntt::CrtPlan::build(crt_ps);
        let crt_n = 1024usize;
        let crt_b = 80u64;
        let crt_limbs_in = vec![0x5A5A_1234_9876_FEDCu64; (crt_n * (crt_b as usize)).div_ceil(64)];
        let crt_iters = if cfg!(debug_assertions) { 8usize } else { 48usize };
        let mut crt_w_bufs: Vec<(Vec<f64>, [Vec<f64>; 4], Vec<u64>)> = (0..nw)
            .map(|_| {
                (
                    vec![0.0f64; 2 * crate::refk::ntt_planar_stride(crt_n)],
                    std::array::from_fn(|_| vec![1.0f64; crt_n]),
                    vec![0u64; (crt_n * (crt_b as usize)).div_ceil(64) + 4],
                )
            })
            .collect();
        let crt_w_ptr = crt_w_bufs.as_mut_ptr() as usize;
        let mut sec_crt = f64::INFINITY;
        for trial in 0..=ntt_trials {
            let ready = std::sync::atomic::AtomicUsize::new(0);
            sched.on_each_worker(|| {
                let wi = crate::par::current_worker_idx().min(nw - 1);
                let (planes_buf, crt_res, crt_out) = unsafe {
                    &mut *(crt_w_ptr as *mut (Vec<f64>, [Vec<f64>; 4], Vec<u64>)).add(wi)
                };
                let ext_cfg = NttSplitCfg {
                    in_limbs: crt_limbs_in.len() as u64,
                    start_coeff: 0,
                    count: crt_n as u64,
                    coeff_bits: crt_b,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: 0.0,
                    inv_p: 0.0,
                };
                sync_workers(&ready);
                let tc = Instant::now();
                for _ in 0..crt_iters {
                    unsafe {
                        (kernels.active.ntt_split)(
                            planes_buf.as_mut_ptr(),
                            crt_limbs_in.as_ptr(),
                            &ext_cfg,
                        );
                    }
                    for pi in 0..4 {
                        let red_cfg = NttSplitCfg {
                            in_limbs: u64::MAX,
                            start_coeff: 0,
                            count: crt_n as u64,
                            coeff_bits: crt_b,
                            col_stride: 0,
                            cols_per_row: 0,
                            p: crt_plan.primes_f64[pi],
                            inv_p: crt_plan.inv_primes_f64[pi],
                        };
                        unsafe {
                            (kernels.active.ntt_split)(
                                crt_res[pi].as_mut_ptr(),
                                planes_buf.as_ptr() as *const u64,
                                &red_cfg,
                            );
                        }
                    }
                    let ptrs: [*const f64; 4] = [
                        crt_res[0].as_ptr(),
                        crt_res[1].as_ptr(),
                        crt_res[2].as_ptr(),
                        crt_res[3].as_ptr(),
                    ];
                    let mut ccfg = NttCrtCfg {
                        k_primes: 4,
                        count: crt_n as u64,
                        coeff_bits: crt_b,
                        out_limbs: crt_out.len() as u64,
                        signed: 0,
                        wrap_cyclic: 0,
                        primes: crt_plan.primes_f64.as_ptr(),
                        inv_primes: crt_plan.inv_primes_f64.as_ptr(),
                        garner_inv: crt_plan.garner_inv.as_ptr(),
                        prefix_prods: crt_plan.prefix_prods.as_ptr(),
                        k_words: crt_plan.k_words as u64,
                        segs: NttCrtSegs::ONE,
                        carry_limbs: std::ptr::null_mut(),
                    };
                    unsafe {
                        let _ =
                            (kernels.active.ntt_crt)(crt_out.as_mut_ptr(), ptrs.as_ptr(), &mut ccfg);
                    }
                }
                let el = tc.elapsed().as_secs_f64();
                std::hint::black_box(crt_out[0]);
                unsafe {
                    *(w_times_ptr as *mut f64).add(wi) = el;
                }
            });
            if trial > 0 {
                sec_crt = sec_crt.min(median_sec(&mut w_times));
            }
        }
        sec_crt = sec_crt.max(1e-9);
        drop(crt_w_bufs);
        let crt_rate = (((crt_iters * crt_n) as f64) * 16.82 / sec_crt).max(1e6);

        if std::env::var_os("SUPI_TRACE").is_some() {
            eprintln!(
                "[TRACE CAL2] ntt_bf={:.3}G/s fft_bf={:.3}G/s crt={:.3}G/s base_sq={:.3}G/s base_max={} kara_max={} rows_max={}",
                ntt_bf_rate * 1e-9,
                fft_bf_rate * 1e-9,
                crt_rate * 1e-9,
                basecase_limbs_sq_per_sec * 1e-9,
                basecase_max_limbs,
                karatsuba_max_limbs,
                mul_base_rows_max,
            );
        }

        if let Some(path) = cache_path {
            let sc = PersistedStaticCal {
                core_ntt_bf_per_sec: core_ntt.clone(),
                kernel_rates,
                ntt_bf_per_sec_per_worker: ntt_bf_rate,
                fft_bf_per_sec_per_worker: fft_bf_rate,
                crt_ops_per_sec_per_worker: crt_rate,
                basecase_limbs_sq_per_sec,
                sieve_ops_per_sec,
                basecase_max_limbs,
                karatsuba_max_limbs,
                mul_base_rows_max,
                sqr_base_rows_max,
                nt_effective,
                nt_ratio,
            };
            save_static_cal(path, profile, build_id, kernels, &sc);
        }

        sched.on_each_worker(|| {
            crate::limb::clear_tls_scratch();
            crate::fft::clear_tls_fwd_pool();
            crate::ntt::clear_tls_bufs();
        });

        Self {
            core_ntt_bf_per_sec: core_ntt,
            kernel_rates,
            ntt_bf_per_sec_per_worker: ntt_bf_rate,
            fft_bf_per_sec_per_worker: fft_bf_rate,
            crt_ops_per_sec_per_worker: crt_rate,
            basecase_limbs_sq_per_sec,
            sieve_ops_per_sec,
            basecase_max_limbs,
            karatsuba_max_limbs,
            mul_base_rows_max,
            sqr_base_rows_max,
            ram_bw_bps,
            sweep_step_sec,
            overlap_alpha,
            nt_effective,
            nt_ratio,
            reused_static: false,
            devices: dev_cals,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaceBound {
    ComputeBound,
    TrafficBound,
}

#[derive(Clone, Copy, Debug)]
pub struct ProductPrice {
    pub compute_sec: f64,
    pub traffic_sec: f64,
    pub total_sec: f64,
    pub bytes_moved: u64,
    pub butterflies_equiv: f64,
    pub bound: PaceBound,
}

impl ProductPrice {
    pub fn combine(compute_sec: f64, traffic_sec: f64, alpha: f64, bytes_moved: u64, bf: f64) -> Self {
        let hi = compute_sec.max(traffic_sec);
        let sum = compute_sec + traffic_sec;
        let total_sec = hi + alpha.clamp(0.0, 1.0) * (sum - hi);
        let bound = if compute_sec >= traffic_sec {
            PaceBound::ComputeBound
        } else {
            PaceBound::TrafficBound
        };
        Self {
            compute_sec,
            traffic_sec,
            total_sec,
            bytes_moved,
            butterflies_equiv: bf,
            bound,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DivLevelPlan {
    pub level: u32,
    pub prev_div_level: u32,
    pub block_terms: u64,
    pub form: DivForm,
    pub est_factor_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    InsufficientCapacity {
        needed_bytes: u64,
        available_bytes: u64,
        detail: String,
    },
    OutputFileError(String),
}

static NEXT_PLAN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct Plan {
    pub plan_id: u64,
    pub digits: u64,
    pub format: OutputFormat,
    pub output_file_bytes: u64,
    pub conv_leaf_digits: usize,
    pub total_digits_converted: u64,
    pub target_frac_bits: u64,
    pub term_count: u64,
    pub tree_height: u32,
    pub leaf_len: u64,
    pub dividing_levels: Vec<DivLevelPlan>,
    pub newton_steps_bits: Vec<u64>,
    pub out_of_core: bool,
    pub ram_transform_budget: u64,
    pub ram_numbers_budget: u64,
    pub ram_scratch_per_worker: u64,
    /// The most the run's numbers take at once, by the planner's own sizes: the five inputs and
    /// the three products of the root merge, or the numbers of the finish where those are more.
    pub numbers_peak_bytes: u64,
    /// What the workers' scratch may ask for beyond `ram_scratch_per_worker` a worker
    /// (`arena::scratch_demand_bytes`).
    pub scratch_beyond_plan_bytes: u64,
    pub checkpoint_enabled: bool,
    pub checkpoint_mult: f64,
    pub est_checkpoint_write_sec: f64,
    pub planned_compute_sec: f64,
    pub planned_traffic_sec: f64,
    pub planned_total_sec: f64,
    pub planned_bytes_moved: u64,
    pub planned_butterflies: f64,
    pub pace_bound: PaceBound,
    pub overrides: PlanOverrides,
}

impl Plan {
    /// The plan of a run. What a plan commits of the memory never adds up to more than the
    /// profile has: the numbers get three eighths of it and the transforms the rest, and where a
    /// first plan finds that the numbers at their most (with the room of the workers' scratch)
    /// need more than their share, the run is planned again with that much set aside for them,
    /// so the transforms get what is left. A run that goes to the store keeps the shares.
    pub fn build(
        digits: u64,
        format: OutputFormat,
        shares: &Shares,
        cal: &Calibration,
        overrides: PlanOverrides,
    ) -> Result<Self, PlanError> {
        let mut plan =
            Self::build_with_numbers_floor(digits, format, shares, cal, overrides.clone(), 0)?;
        // (planned again, a run may divide at other levels and so hold other numbers: a few
        // rounds settle it)
        for _ in 0..3 {
            let need = plan
                .numbers_peak_bytes
                .saturating_add(plan.scratch_beyond_plan_bytes);
            if plan.out_of_core || need <= plan.ram_numbers_budget {
                break;
            }
            plan = Self::build_with_numbers_floor(
                digits,
                format,
                shares,
                cal,
                overrides.clone(),
                need,
            )?;
        }
        Ok(plan)
    }

    /// The least memory, to within a hundredth, with which this run is planned in memory on
    /// the other resources of `shares`: what a job has to be given. `None` where no amount of
    /// memory admits it. The planner admits a run with more memory wherever it admits it with
    /// less, so the least is found by halving.
    pub fn least_memory_admitted(
        digits: u64,
        format: OutputFormat,
        shares: &Shares,
        cal: &Calibration,
        overrides: &PlanOverrides,
    ) -> Option<u64> {
        let admits = |ram: u64| -> bool {
            let mut s = shares.clone();
            s.total_ram = ram;
            matches!(
                Self::build(digits, format, &s, cal, overrides.clone()),
                Ok(p) if !p.out_of_core
            )
        };
        let mut hi = shares.total_ram.max(1 << 20);
        let mut doublings = 0u32;
        while !admits(hi) {
            if hi >= (1u64 << 62) || doublings >= 48 {
                return None;
            }
            hi = hi.saturating_mul(2);
            doublings += 1;
        }
        let mut lo = 0u64;
        while hi - lo > hi / 128 {
            let mid = lo + (hi - lo) / 2;
            if admits(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    fn build_with_numbers_floor(
        digits: u64,
        format: OutputFormat,
        shares: &Shares,
        cal: &Calibration,
        overrides: PlanOverrides,
        numbers_floor: u64,
    ) -> Result<Self, PlanError> {
        let output_file_bytes = format.file_bytes(digits).ok_or_else(|| {
            PlanError::InsufficientCapacity {
                needed_bytes: u64::MAX,
                available_bytes: shares.total_ssd,
                detail: "Output file size overflows 64-bit byte count".to_string(),
            }
        })?;

        // Conversion leaf digits: multiple of 19 so `Packed19` leaf boundaries align on 64-bit words!
        let conv_leaf_digits = overrides.force_conv_leaf_digits.unwrap_or(38).max(19);
        let total_digits_converted = digits.saturating_add(conv_leaf_digits as u64);
        let target_frac_bits =
            ((total_digits_converted as f64) * LOG2_10).ceil() as u64 + 128;

        let (term_count, tree_height, leaf_len) =
            if let Some((forced_tc, forced_ll)) = overrides.force_series_shape {
                let tc = forced_tc.max(2);
                let ll = forced_ll.max(1);
                let mut h = 0u32;
                while h < 62 && ll.saturating_mul(1u64 << h) < tc {
                    h += 1;
                }
                (tc, h, ll)
            } else {
                let tc = if total_digits_converted == 0 {
                    1
                } else {
                    (((total_digits_converted as f64) / DIGITS_PER_TERM).ceil() as u64)
                        .saturating_add(2)
                        .max(2)
                };

                // Choose smallest power of two `2^H` such that a leaf's working set fits `min_l1_share`
                // (Section 4: "two half-triples and their merge, fits the smallest L1 share among the workers").
                let l1_cap = shares.min_l1_share.max(4096);
                let est_bits_per_leaf_term =
                    (56.0 + 3.0 * (tc as f64).max(2.0).log2()) as u64;
                // Working set of two half-triples (P, Q_odd, T) + merged (P, Q_odd, T) ~ 6 * terms * bits / 8 bytes.
                let max_terms_l1 =
                    ((l1_cap * 8) / (6 * est_bits_per_leaf_term.max(64))).clamp(4, 32);
                let mut h = 0u32;
                while h < 62 && tc.div_ceil(1u64 << h) > max_terms_l1 {
                    h += 1;
                }
                let ll = tc.div_ceil(1u64 << h).max(1);
                (tc, h, ll)
            };

        // Commit RAM across the 3 arenas (Section 6 & 10):
        // Keep 15/16 of physical RAM for the three arenas (`1/16` OS/stack/table headroom).
        let usable_ram = shares.total_ram.saturating_sub(shares.total_ram / 16);
        let nw_u64 = shares.worker_count.max(1);
        let max_scratch_pw = usable_ram / 4 / nw_u64;
        let scratch_per_worker = shares
            .min_l3_share
            .max(shares.min_l2_share.saturating_mul(2))
            .min(max_scratch_pw);
        let total_scratch = scratch_per_worker.saturating_mul(nw_u64);
        let rem_ram = usable_ram.saturating_sub(total_scratch);
        // The numbers get three eighths of what is left, or what a first plan found them to need
        // at their most where that is more (`numbers_floor`); the transforms get the rest.
        let ram_numbers_share = (rem_ram / 8) * 3 + ((rem_ram % 8) * 3) / 8;
        let ram_numbers_budget = ram_numbers_share.max(numbers_floor).min(rem_ram);
        let ram_transform_budget = rem_ram.saturating_sub(ram_numbers_budget);

        // Shortest-path DP over levels `0..=tree_height` to choose dividing levels (Section 4 & 10).
        let dividing_levels = if let Some(forced) = &overrides.force_dividing_levels {
            let mut out = Vec::new();
            let mut prev = 0u32;
            for &(lv, form) in forced {
                if lv >= 1 && lv < tree_height && lv > prev {
                    let m = lv - prev;
                    let block_terms = leaf_len.saturating_mul(1u64 << lv);
                    out.push(DivLevelPlan {
                        level: lv,
                        prev_div_level: prev,
                        block_terms,
                        form,
                        est_factor_bits: 3 * (m as u64) * block_terms,
                    });
                    prev = lv;
                }
            }
            out
        } else {
            plan_dividing_levels_dp(
                tree_height,
                leaf_len,
                term_count,
                shares,
                cal,
                ram_transform_budget,
                &overrides,
            )
        };

        // Newton step bit-precisions doubling down from `target_frac_bits` to <= 50 bits (Section 3).
        let mut newton_steps_bits = Vec::new();
        let mut cur_p = target_frac_bits;
        while cur_p > 50 {
            newton_steps_bits.push(cur_p);
            cur_p = cur_p.div_ceil(2) + 8;
            if cur_p <= 56 {
                break;
            }
        }
        newton_steps_bits.reverse();

        // Price full run (series + finish + conversion) & peak capacity check!
        let mut total_comp = 0.0f64;
        let mut total_traf = 0.0f64;
        let mut total_time = 0.0f64;
        let mut total_bytes = 0u64;
        let mut total_bf = 0.0f64;
        let mut uses_store = false;

        let mut last_div_at = vec![0u32; (tree_height as usize) + 1];
        let mut cur_d = 0u32;
        for lv in 1..=tree_height {
            last_div_at[lv as usize] = cur_d;
            if dividing_levels.iter().any(|d| d.level == lv) {
                cur_d = lv;
            }
        }

        let plain_bpt =
            53.28102 + 3.0 * ((term_count as f64) / std::f64::consts::E).max(1.0).log2();
        let shed_bpt_tbl = compute_level_shed_bpt(tree_height, leaf_len);
        // (the sizes of the root merge's inputs, for the count of what the numbers take)
        let mut top_qt_bits = 0u64;
        let mut top_p_bits = 0u64;
        let mut top_qt_moved = 0u64;
        for lv in 1..=tree_height {
            let nodes = 1u64 << (tree_height - lv);
            let child_terms = leaf_len.saturating_mul(1u64 << (lv - 1));
            let last_d = last_div_at[lv as usize];
            let reduced_bpt = if last_d == 0 {
                plain_bpt
            } else {
                (plain_bpt - shed_bpt_tbl[last_d as usize]).max(6.2)
            };
            let qt_bits = ((child_terms as f64) * reduced_bpt).ceil() as u64 + 64;
            let p_bits = qt_bits.saturating_sub(child_terms * 14).max(64);
            if lv == tree_height {
                top_qt_bits = qt_bits;
                top_p_bits = p_bits;
            }
            let par_w = if nodes < nw_u64 {
                ((nw_u64 / nodes.max(1)).max(1)).min(usize::MAX as u64) as usize
            } else {
                1
            };
            let (m_qt, mp_qt) = choose_product_method_and_price(
                qt_bits,
                qt_bits,
                true,
                false,
                par_w,
                shares,
                cal,
                ram_transform_budget,
                &overrides,
            );
            let (m_check, mp_p) = choose_product_method_and_price(
                p_bits,
                p_bits,
                false,
                false,
                par_w,
                shares,
                cal,
                ram_transform_budget,
                &overrides,
            );
            if matches!(m_check, MulMethod::Ntt(s) if s.out_of_core)
                || matches!(m_qt, MulMethod::Ntt(s) if s.out_of_core)
            {
                uses_store = true;
            }
            if lv == tree_height {
                top_qt_moved = mp_qt.bytes_moved;
            }
            // Fused merge ~ 7/3 of a single product + 1 P-product (skipping P on rightmost node).
            let conc_nodes = if nodes >= nw_u64 {
                nw_u64 as f64
            } else {
                (nodes as f64).max(1.0)
            };
            let scale_qt = (nodes as f64) * (7.0 / 3.0) / conc_nodes;
            let scale_p = ((nodes - 1) as f64) / conc_nodes;
            total_comp += mp_qt.compute_sec * scale_qt + mp_p.compute_sec * scale_p;
            total_traf += mp_qt.traffic_sec * scale_qt + mp_p.traffic_sec * scale_p;
            total_time += mp_qt.total_sec * scale_qt + mp_p.total_sec * scale_p;
            total_bytes = total_bytes.saturating_add(
                ((mp_qt.bytes_moved as f64) * (nodes as f64) * 2.33) as u64,
            );
            total_bf += mp_qt.butterflies_equiv * (nodes as f64) * (7.0 / 3.0)
                + mp_p.butterflies_equiv * ((nodes - 1) as f64);
        }

        // Check peak memory/storage capacity (Section 10: "A run that the system cannot hold... does not start").
        // Root Q + T + finish working numbers need ~ 5 * root_bits / 8 bytes across RAM + VRAM + SSD.
        let root_bpt = if cur_d == 0 {
            plain_bpt
        } else {
            (62.0 + 3.0 * (tree_height.saturating_sub(cur_d) as f64)).min(plain_bpt)
        };
        let root_num_bytes = (((term_count as f64) * root_bpt / 8.0) as u64).max(1024);
        let finish_bytes = (target_frac_bits / 8).max(1024);
        let min_hold_bytes = root_num_bytes
            .saturating_mul(3)
            .saturating_add(finish_bytes.saturating_mul(3));
        // What the numbers take at once at their most: at the root merge its five inputs and its
        // three products (`Q1 Q2`, `Q2 T1`, `P1 T2`), in the finish `min_hold_bytes`.
        let root_merge_bytes = top_p_bits
            .saturating_mul(2)
            .saturating_add(top_qt_bits.saturating_mul(9))
            / 8;
        let numbers_peak_bytes = root_merge_bytes.max(min_hold_bytes);
        let total_mem_and_store = usable_ram
            .saturating_add(shares.total_vram)
            .saturating_add(shares.total_ssd.saturating_sub(output_file_bytes.min(shares.total_ssd)));
        if min_hold_bytes > total_mem_and_store {
            return Err(PlanError::InsufficientCapacity {
                needed_bytes: min_hold_bytes,
                available_bytes: total_mem_and_store,
                detail: format!(
                    "Run for {digits} digits needs at least {min_hold_bytes} bytes across RAM+VRAM+SSD, but profile holds {total_mem_and_store} bytes"
                ),
            });
        }

        // Price finish (~7% of series) and conversion (~8% of series) from actual target_frac_bits.
        let fin_par_w = nw_u64.min(usize::MAX as u64) as usize;
        let (m_fin, fin_p) = choose_product_method_and_price(
            target_frac_bits,
            target_frac_bits,
            false,
            false,
            fin_par_w,
            shares,
            cal,
            ram_transform_budget,
            &overrides,
        );
        // Section 3: 24 transforms of pi's length (= 8 full 3-transform products) + conversion (~O(log D) levels).
        let conv_levels = (total_digits_converted.div_ceil(conv_leaf_digits as u64))
            .next_power_of_two()
            .ilog2()
            .max(1) as f64;
        let finish_plus_conv_mult = 8.0 + 0.75 * conv_levels;
        total_comp += fin_p.compute_sec * finish_plus_conv_mult;
        total_traf += fin_p.traffic_sec * finish_plus_conv_mult;
        total_time += fin_p.total_sec * finish_plus_conv_mult;
        total_bytes = total_bytes
            .saturating_add(((fin_p.bytes_moved as f64) * finish_plus_conv_mult) as u64);
        total_bf += fin_p.butterflies_equiv * finish_plus_conv_mult;
        if matches!(m_fin, MulMethod::Ntt(s) if s.out_of_core) {
            uses_store = true;
        }

        // What the workers' scratch may ask for, of which `total_scratch` is counted above: eight
        // cache shares a worker (`arena::scratch_demand_bytes`), and never more than the run's
        // largest product moves through memory, since scratch holds pieces of the product being
        // made. A small run on a machine of large caches asks for little.
        let largest_operand_bytes = top_qt_bits.max(target_frac_bits) / 8;
        let product_moves = top_qt_moved
            .max(fin_p.bytes_moved)
            .max(largest_operand_bytes.saturating_mul(8));
        let scratch_reserve_total: u64 = if shares.workers.is_empty() {
            crate::arena::scratch_demand_bytes(shares.min_l2_share.max(scratch_per_worker))
                .min(product_moves)
                .saturating_mul(nw_u64)
        } else {
            shares
                .workers
                .iter()
                .map(|w| {
                    crate::arena::scratch_demand_bytes(w.l2_share.max(scratch_per_worker))
                        .min(product_moves)
                })
                .fold(0u64, u64::saturating_add)
        };
        let scratch_beyond_plan_bytes = scratch_reserve_total.saturating_sub(total_scratch);
        // With that scratch, the numbers at their most are what has to be in memory beside the
        // transforms.
        let beside_transforms = numbers_peak_bytes.saturating_add(scratch_reserve_total);
        if beside_transforms > usable_ram {
            uses_store = true;
        }

        // Without a store, a run has to fit in memory (section 6): its numbers at their most, the
        // workers' scratch, and the transforms of its products. Where they do not, it does not
        // start (section 10), instead of starting and then passing the memory it was given.
        let store_bytes = total_mem_and_store.saturating_sub(usable_ram);
        if uses_store && store_bytes == 0 {
            let left = usable_ram.saturating_sub(beside_transforms);
            return Err(PlanError::InsufficientCapacity {
                needed_bytes: beside_transforms.max(usable_ram.saturating_add(1)),
                available_bytes: usable_ram,
                detail: format!(
                    "Run for {digits} digits does not fit in memory, and there is no store: its numbers take {numbers_peak_bytes} bytes at their most and the workers' scratch {scratch_reserve_total}, which leaves {left} of the {usable_ram} usable bytes for the transforms of its products"
                ),
            });
        }

        let checkpoint_enabled = shares.total_ssd > 0;
        let ssd_rate = if shares.total_vram > 0 && cal.devices.len() >= 2 {
            cal.devices
                .iter()
                .skip(1)
                .map(|d| d.idle_rate_bps)
                .fold(0.0f64, f64::max)
                .max(5.0e8)
        } else {
            cal.devices
                .iter()
                .map(|d| d.idle_rate_bps)
                .fold(0.0f64, f64::max)
                .max(5.0e8)
        };
        let est_ckpt_bytes = (root_num_bytes * 3).max(65536) as f64;
        // Write + read-back verification time of a checkpoint (plus record sync floor):
        let est_checkpoint_write_sec = (2.0 * est_ckpt_bytes) / ssd_rate + 0.02;

        let pace_bound = if total_comp >= total_traf {
            PaceBound::ComputeBound
        } else {
            PaceBound::TrafficBound
        };

        Ok(Self {
            plan_id: NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed),
            digits,
            format,
            output_file_bytes,
            conv_leaf_digits,
            total_digits_converted,
            target_frac_bits,
            term_count,
            tree_height,
            leaf_len,
            dividing_levels,
            newton_steps_bits,
            out_of_core: uses_store,
            ram_transform_budget,
            ram_numbers_budget,
            ram_scratch_per_worker: scratch_per_worker,
            numbers_peak_bytes,
            scratch_beyond_plan_bytes,
            checkpoint_enabled,
            checkpoint_mult: overrides.checkpoint_mult,
            est_checkpoint_write_sec,
            planned_compute_sec: total_comp,
            planned_traffic_sec: total_traf,
            planned_total_sec: total_time,
            planned_bytes_moved: total_bytes,
            planned_butterflies: total_bf,
            pace_bound,
            overrides,
        })
    }

    /// What the merge of two nodes costs in worker time, in the cheaper of its two forms, when
    /// a node has `parallel_workers` workers and the other workers run merges like it.
    #[allow(clippy::too_many_arguments)]
    pub fn merge_node_worker_sec(
        &self,
        q1_odd_bits: u64,
        q2_odd_bits: u64,
        q2_shift_bits: u64,
        t1_bits: u64,
        p1_bits: u64,
        t2_bits: u64,
        p2_bits: u64,
        is_rightmost: bool,
        parallel_workers: usize,
        shares: &Shares,
        cal: &Calibration,
    ) -> f64 {
        let pw = parallel_workers.max(1);
        let (sep, fused) = merge_form_secs(
            q1_odd_bits,
            q2_odd_bits,
            q2_shift_bits,
            t1_bits,
            p1_bits,
            t2_bits,
            p2_bits,
            is_rightmost,
            pw,
            shares,
            cal,
            self.ram_transform_budget,
            &self.overrides,
        );
        sep.min(fused) * (pw as f64)
    }

    /// True for a run whose whole need (about 24 times its result) is below the room set aside
    /// for transforms: it has memory to spare for several large nodes at once and for a work
    /// grid a prime in its largest products. A run that may need all the memory there is holds,
    /// near its end, more than what is live, by the space freed blocks leave in pieces.
    pub fn roomy_run(&self) -> bool {
        (self.target_frac_bits / 8).saturating_mul(24) <= self.ram_transform_budget
    }

    /// What a product of `(a_bits, b_bits)` costs in worker time (its time times its workers)
    /// when it has `parallel_workers` workers and the other workers run products like it.
    pub fn product_worker_sec(
        &self,
        a_bits: u64,
        b_bits: u64,
        fused_sum: bool,
        parallel_workers: usize,
        shares: &Shares,
        cal: &Calibration,
    ) -> f64 {
        self.product_worker_sec_amortized(
            a_bits,
            b_bits,
            fused_sum,
            parallel_workers,
            shares,
            cal,
            1,
        )
    }

    /// As `product_worker_sec`, when one operand's forward transform is reused across
    /// `reuse_count` products (`reuse_count <= 1` prices a standalone 3-transform product).
    #[allow(clippy::too_many_arguments)]
    pub fn product_worker_sec_amortized(
        &self,
        a_bits: u64,
        b_bits: u64,
        fused_sum: bool,
        parallel_workers: usize,
        shares: &Shares,
        cal: &Calibration,
        reuse_count: u64,
    ) -> f64 {
        let pw = parallel_workers.max(1);
        let base = choose_product_method_and_price(
            a_bits,
            b_bits,
            fused_sum,
            false,
            pw,
            shares,
            cal,
            self.ram_transform_budget,
            &self.overrides,
        );
        if reuse_count < 2
            || fused_sum
            || !matches!(base.0, MulMethod::Ntt(_))
            || !matches!(self.overrides.split_override, SplitFormOverride::Auto)
        {
            return base.1.total_sec * (pw as f64);
        }
        let out_bits = a_bits.saturating_add(b_bits).saturating_add(64);
        if let Some((_, price)) = search_best_ntt_spec_uncached(
            out_bits,
            false,
            false,
            16,
            pw as f64,
            shares,
            cal,
            self.ram_transform_budget,
            self.overrides.max_ntt_primes.clamp(1, 18),
            reuse_count,
            None,
        ) {
            price.total_sec * (pw as f64)
        } else {
            base.1.total_sec * (pw as f64)
        }
    }

    /// Select the multiplication method for a runtime product of `(a_bits, b_bits)`.
    pub fn select_mul_method(
        &self,
        a_bits: u64,
        b_bits: u64,
        fused_sum: bool,
        exact_only: bool,
        parallel_workers: usize,
        shares: &Shares,
        cal: &Calibration,
    ) -> MulMethod {
        #[derive(Clone)]
        struct MulMemoSlot {
            k0: u64,
            k1: u64,
            k2: u64,
            k3: u64,
            val: MulMethod,
        }
        thread_local! {
            static MUL_MEMO: std::cell::RefCell<Vec<MulMemoSlot>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let a_q = if a_bits >= 2048 {
            (a_bits + 63) & !63
        } else {
            a_bits
        };
        let b_q = if b_bits >= 2048 {
            (b_bits + 63) & !63
        } else {
            b_bits
        };
        let (lo_q, hi_q) = if a_q <= b_q { (a_q, b_q) } else { (b_q, a_q) };
        let k0 = self.plan_id ^ shares.worker_count.rotate_left(40) ^ shares.min_l2_share;
        let k1 = lo_q;
        let k2 = hi_q;
        let k3 = ((parallel_workers as u64) << 2)
            | (u64::from(fused_sum) << 1)
            | u64::from(exact_only);
        let h = (k0
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(k1.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add(k2.wrapping_mul(0x94D0_49BB_1331_11EB))
            .wrapping_add(k3.wrapping_mul(0xD6E8_FEB8_6659_FD93))
            >> 54) as usize;
        if let Some(hit) = MUL_MEMO.with(|cell| {
            let g = cell.borrow();
            if g.len() == 1024 {
                let s = &g[h];
                if s.k0 == k0 && s.k1 == k1 && s.k2 == k2 && s.k3 == k3 {
                    return Some(s.val.clone());
                }
            }
            None
        }) {
            return hit;
        }
        let m = choose_product_method_and_price(
            lo_q,
            hi_q,
            fused_sum,
            exact_only,
            parallel_workers,
            shares,
            cal,
            self.ram_transform_budget,
            &self.overrides,
        )
        .0;
        MUL_MEMO.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() != 1024 {
                g.resize(
                    1024,
                    MulMemoSlot {
                        k0: 0,
                        k1: 0,
                        k2: 0,
                        k3: 0,
                        val: MulMethod::Basecase,
                    },
                );
            }
            g[h] = MulMemoSlot {
                k0,
                k1,
                k2,
                k3,
                val: m.clone(),
            };
        });
        m
    }

    /// Select the multiplication method when one operand's forward transform is cached and reused
    /// across `reuse_count` products (`reuse_count <= 1` delegates to `select_mul_method`).
    #[allow(clippy::too_many_arguments)]
    pub fn select_mul_method_amortized(
        &self,
        a_bits: u64,
        b_bits: u64,
        fused_sum: bool,
        exact_only: bool,
        parallel_workers: usize,
        shares: &Shares,
        cal: &Calibration,
        reuse_count: u64,
    ) -> MulMethod {
        let base = self.select_mul_method(
            a_bits,
            b_bits,
            fused_sum,
            exact_only,
            parallel_workers,
            shares,
            cal,
        );
        if reuse_count < 2
            || fused_sum
            || !matches!(base, MulMethod::Ntt(_))
            || !matches!(self.overrides.split_override, SplitFormOverride::Auto)
        {
            return base;
        }
        let out_bits = a_bits.saturating_add(b_bits).saturating_add(64);
        if let Some((spec, _)) = search_best_ntt_spec_uncached(
            out_bits,
            false,
            false,
            16,
            parallel_workers.max(1) as f64,
            shares,
            cal,
            self.ram_transform_budget,
            self.overrides.max_ntt_primes.clamp(1, 18),
            reuse_count,
            None,
        ) {
            MulMethod::Ntt(spec)
        } else {
            base
        }
    }

    /// Choose `MergeForm::Separate` vs `MergeForm::Fused` for a binary-splitting node (Section 4).
    #[allow(clippy::too_many_arguments)]
    pub fn select_merge_form(
        &self,
        q2_odd_bits: u64,
        q2_shift_bits: u64,
        t1_bits: u64,
        p1_bits: u64,
        t2_bits: u64,
        par_workers: usize,
        shares: &Shares,
        cal: &Calibration,
    ) -> MergeForm {
        self.select_merge_form_node(
            q2_odd_bits,
            q2_odd_bits,
            q2_shift_bits,
            t1_bits,
            p1_bits,
            t2_bits,
            p1_bits,
            false,
            par_workers,
            shares,
            cal,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select_merge_form_node(
        &self,
        q1_odd_bits: u64,
        q2_odd_bits: u64,
        q2_shift_bits: u64,
        t1_bits: u64,
        p1_bits: u64,
        t2_bits: u64,
        p2_bits: u64,
        is_rightmost: bool,
        par_workers: usize,
        shares: &Shares,
        cal: &Calibration,
    ) -> MergeForm {
        if let Some(forced) = self.overrides.force_merge_form {
            return forced;
        }
        // Limb-sized merges (Basecase / Karatsuba) have no transform to share, so keeping Q
        // stripped of zero bits (`Separate`) avoids materializing `q2_shift_bits` and shrinks Q!
        let max_op_limbs = q1_odd_bits
            .max(q2_odd_bits)
            .max(t1_bits)
            .max(p1_bits)
            .max(t2_bits)
            .max(if is_rightmost { 0 } else { p2_bits })
            .div_ceil(64);
        let sep_max_limbs = ((cal.karatsuba_max_limbs as u64) * 3 / 2).max(64);
        if max_op_limbs <= sep_max_limbs {
            return MergeForm::Separate;
        }
        if q2_shift_bits < 64 {
            return MergeForm::Fused;
        }
        #[derive(Clone, Copy)]
        struct MergeMemoSlot {
            k0: u64,
            k1: u64,
            k2: u64,
            val: MergeForm,
        }
        thread_local! {
            static MERGE_MEMO: std::cell::RefCell<Vec<MergeMemoSlot>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let max_any_bits = (max_op_limbs * 64).max(q2_shift_bits);
        let sh_step = max_any_bits.ilog2().saturating_sub(7).max(6).min(40);
        let bucket = |b: u64| -> u64 {
            if b == 0 {
                0
            } else {
                (b + ((1u64 << sh_step) - 1)) >> sh_step
            }
        };
        let q1_q = bucket(q1_odd_bits);
        let q2_q = bucket(q2_odd_bits);
        let sh_q = bucket(q2_shift_bits);
        let t1_q = bucket(t1_bits);
        let p1_q = bucket(p1_bits);
        let t2_q = bucket(t2_bits);
        let p2_q = if is_rightmost { 0 } else { bucket(p2_bits) };

        let q1_b = q1_q << sh_step;
        let q2_b = q2_q << sh_step;
        let sh_b = sh_q << sh_step;
        let t1_b = t1_q << sh_step;
        let p1_b = p1_q << sh_step;
        let t2_b = t2_q << sh_step;
        let p2_b = p2_q << sh_step;

        let k0 = self.plan_id
            ^ ((par_workers as u64) << 44)
            ^ (shares.min_l2_share << 8)
            ^ ((sh_step as u64) << 56)
            ^ (u64::from(is_rightmost) << 62);
        let k1 = (q1_q << 48) | (q2_q << 32) | (t1_q << 16) | sh_q;
        let k2 = (p1_q << 32) | (t2_q << 16) | p2_q;
        let h = (k0
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(k1.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add(k2.wrapping_mul(0x94D0_49BB_1331_11EB))
            >> 52) as usize;
        if let Some(hit) = MERGE_MEMO.with(|cell| {
            let g = cell.borrow();
            if g.len() == 4096 {
                let s = &g[h];
                if s.k0 == k0 && s.k1 == k1 && s.k2 == k2 {
                    return Some(s.val);
                }
            }
            None
        }) {
            return hit;
        }
        let (sep_sec, fused_sec) = merge_form_secs(
            q1_b,
            q2_b,
            sh_b,
            t1_b,
            p1_b,
            t2_b,
            p2_b,
            is_rightmost,
            par_workers,
            shares,
            cal,
            self.ram_transform_budget,
            &self.overrides,
        );
        let res = if sep_sec <= fused_sec {
            MergeForm::Separate
        } else {
            MergeForm::Fused
        };
        MERGE_MEMO.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() != 4096 {
                g.resize(
                    4096,
                    MergeMemoSlot {
                        k0: 0,
                        k1: 0,
                        k2: 0,
                        val: MergeForm::Separate,
                    },
                );
            }
            g[h] = MergeMemoSlot {
                k0,
                k1,
                k2,
                val: res,
            };
        });
        res
    }
}

/// What the merge of two nodes costs in its two forms (Section 4), in seconds of a node that
/// has `par_workers` workers while the other workers run merges like it: the separate form
/// (the pairs `Q2*T1`, `Q1*Q2` and `P1*T2`, `P1*P2`, each sharing a transform where that pays,
/// and the addition) and the fused form (`T` in one pass, `Q` and `P` reusing its transforms
/// where they fit). `p2_bits` is ignored for the rightmost node, which makes no `P`.
#[allow(clippy::too_many_arguments)]
fn merge_form_secs(
    q1_b: u64,
    q2_b: u64,
    sh_b: u64,
    t1_b: u64,
    p1_b: u64,
    t2_b: u64,
    p2_b: u64,
    is_rightmost: bool,
    par_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    overrides: &PlanOverrides,
) -> (f64, f64) {
    let same_grid = |a: &MulMethod, b: &MulMethod| -> bool {
        match (a, b) {
            (MulMethod::Fft { n_complex: n1, .. }, MulMethod::Fft { n_complex: n2, .. }) => {
                n1 == n2
            }
            (MulMethod::Ntt(s1), MulMethod::Ntt(s2)) => {
                s1.n == s2.n && s1.k_primes == s2.k_primes
            }
            _ => false,
        }
    };

    let (qt_m, qt_mp) = choose_product_method_and_price(
        q2_b,
        t1_b,
        false,
        false,
        par_workers,
        shares,
        cal,
        ram_transform_budget,
        overrides,
    );
    let (qq_m, qq_mp) = choose_product_method_and_price(
        q1_b,
        q2_b,
        false,
        false,
        par_workers,
        shares,
        cal,
        ram_transform_budget,
        overrides,
    );
    let (pt_m, pt_mp) = choose_product_method_and_price(
        p1_b,
        t2_b,
        false,
        false,
        par_workers,
        shares,
        cal,
        ram_transform_budget,
        overrides,
    );
    let pp_opt = if is_rightmost {
        None
    } else {
        Some(choose_product_method_and_price(
            p1_b,
            p2_b,
            false,
            false,
            par_workers,
            shares,
            cal,
            ram_transform_budget,
            overrides,
        ))
    };

    let full_q2 = q2_b + sh_b;
    let fused_bits = (full_q2 + t1_b).max(p1_b + t2_b);
    let max_op = full_q2.max(t1_b).max(p1_b).max(t2_b);
    let (_, fused_mp) = choose_product_method_and_price(
        max_op,
        fused_bits.saturating_sub(max_op),
        true,
        false,
        par_workers,
        shares,
        cal,
        ram_transform_budget,
        overrides,
    );

    // A pair shares the transform of its common operand in the grid of the larger product
    // (five transforms there) wherever that beats three and three.
    let q_sep_sec = if same_grid(&qt_m, &qq_m) {
        (5.0 / 3.0) * qt_mp.total_sec.max(qq_mp.total_sec)
    } else {
        let (big_q_m, small_q_m) = if qt_mp.total_sec >= qq_mp.total_sec {
            (&qt_m, &qq_m)
        } else {
            (&qq_m, &qt_m)
        };
        let k_q = pair_shared_cost_factor(big_q_m, small_q_m);
        (qt_mp.total_sec + qq_mp.total_sec)
            .min(k_q * qt_mp.total_sec.max(qq_mp.total_sec))
    };
    let p_sep_sec = if let Some((ref pp_m, ref pp_mp)) = pp_opt {
        if same_grid(&pt_m, pp_m) {
            (5.0 / 3.0) * pt_mp.total_sec.max(pp_mp.total_sec)
        } else {
            let (big_p_m, small_p_m) = if pt_mp.total_sec >= pp_mp.total_sec {
                (&pt_m, pp_m)
            } else {
                (pp_m, &pt_m)
            };
            let k_p = pair_shared_cost_factor(big_p_m, small_p_m);
            (pt_mp.total_sec + pp_mp.total_sec)
                .min(k_p * pt_mp.total_sec.max(pp_mp.total_sec))
        }
    } else {
        pt_mp.total_sec
    };
    let add_bytes = ((3 * fused_bits) / 8) as f64;
    let add_sec = add_bytes / cal.ram_bw_bps.max(1e9);
    let sep_sec = q_sep_sec + p_sep_sec + add_sec;

    let shift_bytes = ((2 * full_q2) / 8) as f64;
    let shift_sec = shift_bytes / cal.ram_bw_bps.max(1e9);
    let strip_bytes = (((q1_b + full_q2) * 2) / 8) as f64;
    let strip_sec = strip_bytes / cal.ram_bw_bps.max(1e9);
    let q_fused_sec = if q1_b + full_q2 <= fused_bits {
        (0.68 * fused_mp.total_sec + strip_sec).min(qq_mp.total_sec)
    } else {
        qq_mp.total_sec
    };
    let p_fused_sec = if let Some((_, ref pp_mp)) = pp_opt {
        if p1_b + p2_b <= fused_bits {
            (0.68 * fused_mp.total_sec).min(pp_mp.total_sec)
        } else {
            pp_mp.total_sec
        }
    } else {
        0.0
    };
    let fused_sec = (5.0 / 3.0) * fused_mp.total_sec + shift_sec + q_fused_sec + p_fused_sec;

    (sep_sec, fused_sec)
}

/// What a level of merges costs in a run as a multiple of the sum of its products' prices, by
/// the bits of a node's T. Measured on runs of 1e9, 2.5e9 and 5e9 digits, where the three agree
/// to a few percent: nodes of 8 to 64 megabits cost a third more than their products' prices
/// (several workers on small grids, and the merge's own passes), nodes of a quarter gigabit as
/// much, and the largest a fifth less. Only the choice of dividing levels uses it: it compares
/// levels of different sizes, which the other choices never do.
fn level_cost_ratio(t_bits: u64) -> f64 {
    // (log2 of megabits, ratio)
    const PTS: [(f64, f64); 18] = [
        (-3.0, 0.78),
        (-2.0, 0.80),
        (-1.0, 0.84),
        (0.0, 0.92),
        (1.0, 0.93),
        (2.0, 1.2),
        (3.0, 1.36),
        (4.0, 1.42),
        (5.0, 1.38),
        (6.0, 1.33),
        (7.0, 1.07),
        (8.0, 1.0),
        (9.0, 0.98),
        (10.0, 0.95),
        (11.0, 0.86),
        (12.0, 0.83),
        (13.0, 0.8),
        (14.0, 0.78),
    ];
    let x = ((t_bits.max(1) as f64) * 1e-6).log2();
    if x <= PTS[0].0 {
        return PTS[0].1;
    }
    for i in 1..PTS.len() {
        let (lo, hi) = (PTS[i - 1], PTS[i]);
        if x <= hi.0 {
            let f = (x - lo.0) / (hi.0 - lo.0);
            return lo.1 + f * (hi.1 - lo.1);
        }
    }
    PTS[PTS.len() - 1].1
}

/// Shortest-path DP over tree levels `0..=H` finding the time-optimal set of dividing levels
/// and each level's division form (`SequentialHensel` vs `SingleProduct`, Section 4 & 10).
/// Exact number-theoretic bits shed per term `log2(R_{L_d}) / L_d` for each level `d in 0..=h`.
fn compute_level_shed_bpt(tree_height: u32, leaf_len: u64) -> Vec<f64> {
    let h = tree_height as usize;
    let mut out = vec![0.0f64; h + 1];
    if h == 0 || leaf_len == 0 {
        return out;
    }

    let mut comp = [0u64; 256]; // 16,384 odd numbers -> primes in 3..=32,767
    let mut i = 0usize;
    while (2 * i + 3) * (2 * i + 3) < 32_768 {
        if (comp[i >> 6] & (1u64 << (i & 63))) == 0 {
            let p = 2 * i + 3;
            let mut m = (p * p - 3) / 2;
            while m < 16_384 {
                comp[m >> 6] |= 1u64 << (m & 63);
                m += p;
            }
        }
        i += 1;
    }
    let mut primes = Vec::with_capacity(3512);
    for idx in 0..16_384usize {
        if (comp[idx >> 6] & (1u64 << (idx & 63))) == 0 {
            let r = (2 * idx + 3) as u64;
            primes.push((r, (r as f64).log2()));
        }
    }

    for (d, slot) in out.iter_mut().enumerate().skip(1) {
        let l = leaf_len.saturating_mul(1u64 << d);
        if l < 3 {
            continue;
        }
        let mut total_bits = 0.0f64;
        for &(r, log2_r) in &primes {
            if r > l {
                break;
            }
            let mut v_fact = 0u64;
            let mut cur = l / r;
            let mut floor_log = 0u64;
            while cur > 0 {
                v_fact += cur;
                floor_log += 1;
                cur /= r;
            }
            let e = v_fact - floor_log;
            if e > 0 {
                let mult = if r == 3 { 1.0 } else { 3.0 };
                total_bits += mult * (e as f64) * log2_r;
            }
        }

        if l > 65_536 {
            let m_max = l / 32_768;
            if m_max >= 2 {
                let mut tail_nat = if m_max <= 512 {
                    let mut s = 0.0f64;
                    for m in 2..=m_max {
                        s += ((l as f64) / (m as f64) - 32_768.0).max(0.0);
                    }
                    s
                } else {
                    let mf = m_max as f64;
                    let h_m_minus_1 = mf.ln() + 0.5772156649015329 - 1.0 + 0.5 / mf
                        - 1.0 / (12.0 * mf * mf);
                    ((l as f64) * h_m_minus_1 - 32_768.0 * ((m_max - 1) as f64)).max(0.0)
                };
                if l >= 32_768 * 32_768 {
                    let q_max = (l / (32_768 * 32_768)).min(512);
                    for q in 1..=q_max {
                        tail_nat += (((l as f64) / (q as f64)).sqrt() - 32_768.0).max(0.0);
                    }
                }
                total_bits += 3.0 * tail_nat * std::f64::consts::LOG2_E;
            }
        }
        *slot = total_bits / (l as f64);
    }
    out
}

fn plan_dividing_levels_dp(
    tree_height: u32,
    leaf_len: u64,
    term_count: u64,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    overrides: &PlanOverrides,
) -> Vec<DivLevelPlan> {
    let h = tree_height as usize;
    if h <= 2 {
        return Vec::new();
    }

    let plain_bpt = 53.28102 + 3.0 * ((term_count as f64) / std::f64::consts::E).max(1.0).log2();
    if plain_bpt <= 61.0 {
        return Vec::new();
    }

    // State `d in 0..h`: last dividing level was `d` (`0` = undivided since leaves).
    // Transition at level `lv in 1..=h`:
    //   1. Merge at level `lv` using operand sizes governed by `d` (which was decided at `lv - 1`).
    //   2. If `lv < h`, optionally divide at `lv` (transitioning to state `lv`), paying factor build
    //      + sieve + per-node Hensel division cost!
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; h]; h + 1];
    let mut parent: Vec<Vec<Option<(usize, Option<DivForm>)>>> = vec![vec![None; h]; h + 1];
    dp[0][0] = 0.0;

    let shed_bpt_tbl = compute_level_shed_bpt(tree_height, leaf_len);
    let bpt_after_div = |d: usize| -> f64 {
        if d == 0 {
            plain_bpt
        } else {
            (plain_bpt - shed_bpt_tbl[d]).max(6.2)
        }
    };
    let active_w = shares.active_workers.max(1);
    // The counts of nodes at once that are tried: those that share the workers out evenly and,
    // level by level, go into the level's nodes.
    let conc_cands = crate::tier::even_splits(active_w);
    // `scale` products at this price, `at_once` of them at a time (the price is that of a
    // product while the other workers run products like it).
    let level_wall_sec = |mp: &ProductPrice, scale: f64, at_once: f64| -> f64 {
        (mp.total_sec * scale) / at_once.max(1.0)
    };

    // How many nodes of a level run at once, for these estimates: all the workers allow where a
    // node's product is one worker's (a one-dimensional transform or a smaller method), else the
    // count that makes a product cheapest in worker time by these prices. The series itself
    // takes its merges by the room their grids have in the last-level cache
    // (`nodes_at_once_fit`), which was measured faster at every size; the estimates were left on
    // the prices, because the dividing levels chosen from them were measured the better ones
    // (with the series' counts here, 10^8 digits took 2.5% longer and 2.5 * 10^9 0.9%).
    let level_conc = |a_bits: u64,
                      b_bits: u64,
                      fused: bool,
                      is_sp_div: bool,
                      nodes: u64,
                      worker_sec_at: &dyn Fn(usize) -> f64|
     -> usize {
        let cap = (nodes.min(active_w as u64) as usize).max(1);
        let cand_pw = (active_w / cap).max(1);
        let (m, _) = choose_product_method_and_price(
            a_bits,
            b_bits,
            fused,
            false,
            cand_pw,
            shares,
            cal,
            ram_transform_budget,
            overrides,
        );
        if !matches!(m, MulMethod::Ntt(ref sp) if sp.rows > 1) {
            return cap;
        }
        let mut best = (f64::INFINITY, 1usize);
        for &c in &conc_cands {
            if c < 1 || c > cap || !nodes.is_multiple_of(c as u64) {
                continue;
            }
            let pw_c = (active_w / c).max(1);
            let (mc, _) = choose_product_method_and_price(
                a_bits,
                b_bits,
                fused,
                false,
                pw_c,
                shares,
                cal,
                ram_transform_budget,
                overrides,
            );
            let mut margin = MORE_AT_ONCE_MARGIN;
            if let MulMethod::Ntt(sp) = mc {
                if pw_c > sp.rows.max(1) {
                    continue;
                }
                let grid_1 = (sp.k_primes as u64)
                    .saturating_mul(sp.rows as u64)
                    .saturating_mul(sp.outer_row_bytes() as u64)
                    .max(1);
                if is_sp_div {
                    let div_room = if c > 1 {
                        ram_transform_budget / 2
                    } else {
                        ram_transform_budget
                    };
                    if (c as u64).saturating_mul(grid_1.saturating_mul(2)) > div_room {
                        continue;
                    }
                    let u_grid_1 = (sp.k_primes as u64)
                        .saturating_mul(sp.n as u64)
                        .saturating_mul(8);
                    let raw_io_b = a_bits.div_ceil(8).saturating_mul(2);
                    let shared_per_node = u_grid_1
                        .saturating_add(raw_io_b)
                        .saturating_add(u_grid_1 / (c as u64).max(1));
                    let shared_l3_fits = c >= 2
                        && sp.rows <= 128
                        && sp.rows >= pw_c * 2
                        && shares.fits_l3_domains(c as u64, shared_per_node, sp.rows, 6, 10)
                        && ((c as u64 + 1).saturating_mul(grid_1) <= ram_transform_budget / 2);
                    if shared_l3_fits {
                        margin = 1.15;
                    } else if (2..=4).contains(&c)
                        && sp.packed
                        && sp.cols >= 8192
                        && sp.rows <= active_w.saturating_mul(16)
                        && sp.rows >= pw_c * 8
                    {
                        margin = 1.05;
                    }
                } else {
                    let room = if c > 1 {
                        (ram_transform_budget / 8).saturating_mul(3)
                    } else {
                        ram_transform_budget
                    };
                    if (c as u64).saturating_mul(grid_1.saturating_mul(4)) > room {
                        continue;
                    }
                    let pbp_per_node = 2u64
                        .saturating_mul(sp.rows as u64)
                        .saturating_mul(sp.cols as u64)
                        .saturating_mul(8);
                    let pbp_l3_fits =
                        shares.fits_l3_domains(c as u64, pbp_per_node, sp.rows, 1, 1);
                    if (2..=4).contains(&c)
                        && sp.packed
                        && sp.cols >= 8192
                        && sp.rows <= active_w.saturating_mul(16)
                        && sp.rows >= pw_c * 8
                    {
                        margin = if pbp_l3_fits { 1.12 } else { 1.05 };
                    }
                }
            }
            let cost = worker_sec_at(pw_c);
            if cost < best.0 * margin {
                best = (cost.min(best.0), c);
            }
        }
        best.1
    };

    // `SUPI_TRACE_PLAN`: print every estimate the choice is made from, and the choice.
    let dp_trace = std::env::var_os("SUPI_TRACE_PLAN").is_some();
    for lv in 1..=h {
        let nodes = 1u64 << (h - lv);
        let child_terms = leaf_len.saturating_mul(1u64 << (lv - 1));
        let block_terms = child_terms * 2;

        for prev_d in 0..lv {
            let cur_cost = dp[lv - 1][prev_d];
            if !cur_cost.is_finite() {
                continue;
            }
            let t_bpt = bpt_after_div(prev_d);
            let q_bpt = (t_bpt - 18.0).max(16.0);
            let p_bpt = (t_bpt - 47.0).max(8.0);
            let t_bits = ((child_terms as f64) * t_bpt).ceil() as u64 + 64;
            let q_odd_bits = ((child_terms as f64) * q_bpt).ceil() as u64 + 64;
            let p_bits = ((child_terms as f64) * p_bpt).ceil() as u64 + 64;

            // The level as the series runs it: as many nodes at once as make a node cheapest
            // in worker time, every node in the cheaper of its two forms (the prices carry what
            // the crowding of the cache costs), the rightmost one without its P, and the whole
            // weighed by what levels of this size of node were measured to cost against these
            // prices.
            let q_shift_bits = t_bits.saturating_sub(q_odd_bits);
            let node_sec = |pw: usize, rightmost: bool| -> f64 {
                let (sep, fused) = merge_form_secs(
                    q_odd_bits,
                    q_odd_bits,
                    q_shift_bits,
                    t_bits,
                    p_bits,
                    t_bits,
                    p_bits,
                    rightmost,
                    pw,
                    shares,
                    cal,
                    ram_transform_budget,
                    overrides,
                );
                sep.min(fused)
            };
            let conc_nodes = level_conc(
                t_bits,
                t_bits,
                true,
                false,
                nodes,
                &|pw| node_sec(pw, false) * (pw as f64),
            );
            let par_w = (active_w / conc_nodes.max(1)).max(1);
            let frontier_spill_mult =
                if nodes > (active_w as u64).saturating_mul(16) && nodes <= 32_768 && par_w == 1 {
                    let est_fr_bpt = if prev_d == 0 {
                        55.0 + 3.0 * (term_count.max(2) as f64).log2()
                    } else {
                        66.5 + 3.0 * ((h - prev_d) as f64)
                    };
                    let est_fr_bits = ((child_terms as f64) * est_fr_bpt) as u64 + 128;
                    let (m_fr, _) = choose_product_method_and_price(
                        est_fr_bits,
                        est_fr_bits,
                        true,
                        false,
                        1,
                        shares,
                        cal,
                        ram_transform_budget,
                        overrides,
                    );
                    let spills_fr = if let MulMethod::Ntt(s) = m_fr {
                        let grid_4 = (s.k_primes as u64)
                            .saturating_mul(s.rows as u64)
                            .saturating_mul(s.outer_row_bytes() as u64)
                            .saturating_mul(4);
                        let num_ch = (s.coeff_bits as u64).div_ceil(48).max(1);
                        let ws_1d = ((s.k_primes as u64) + num_ch + 4)
                            .saturating_mul(s.n as u64)
                            .saturating_mul(8);
                        let aw = (active_w.max(1)) as u64;
                        let total_eff_l3 = shares.effective_l3_bytes(1, 1);
                        s.rows > 1
                            || !shares.fits_l3_domains(aw, ws_1d, 1, 7, 8)
                            || grid_4.saturating_mul(aw)
                                > ram_transform_budget.min(total_eff_l3.saturating_mul(2))
                    } else {
                        false
                    };
                    if spills_fr { 1.45 } else { 1.0 }
                } else {
                    1.0
                };
            let merge_level_sec = (node_sec(par_w, false) * ((nodes - 1) as f64)
                + node_sec(par_w, true))
                / (conc_nodes as f64)
                * level_cost_ratio(t_bits)
                * frontier_spill_mult;
            if dp_trace {
                eprintln!(
                    "[TRACE DP] merge lv={lv} after_div={prev_d} nodes={nodes} conc={conc_nodes} t_mbit={:.1} est={:.1}ms",
                    (t_bits as f64) * 1e-6,
                    merge_level_sec * 1e3
                );
            }
            let after_merge = cur_cost + merge_level_sec;
            let mask_forces_div = overrides
                .force_dividing_mask
                .map(|m| lv < 63 && (m & (1u64 << lv)) != 0);

            // Choice A: Do NOT divide at `lv` -> keep `prev_d` (unless resumed mask requires dividing here).
            if mask_forces_div != Some(true) && after_merge < dp[lv][prev_d] {
                dp[lv][prev_d] = after_merge;
                parent[lv][prev_d] = Some((prev_d, None));
            }

            // Choice B: Divide at `lv` (either chosen freely by DP or required by resumed `force_dividing_mask`,
            // re-pricing `SequentialHensel` vs `SingleProduct` under current `Shares` & `Calibration`).
            let min_div_block = if h >= 12 { 256 } else { 128 };
            let allow_div = match mask_forces_div {
                Some(req) => req && lv < h,
                None => {
                    lv < h
                        && lv >= 4
                        && block_terms >= min_div_block
                        && (prev_d == 0 || lv >= prev_d + 4)
                }
            };
            if allow_div {
                let quot_t_bpt = bpt_after_div(lv);
                let mut shed_bpt = (t_bpt - quot_t_bpt).max(0.0);
                if shed_bpt <= 1.0 {
                    if mask_forces_div == Some(true) {
                        shed_bpt = 3.0;
                    } else {
                        continue;
                    }
                }
                let factor_bits =
                    (((block_terms as f64) * shed_bpt).ceil() as u64).clamp(64, t_bits);
                let quot_q_bpt = (quot_t_bpt - 18.0).max(16.0);
                let quot_p_bpt = (quot_t_bpt - 47.0).max(8.0);
                let quot_t_bits = ((block_terms as f64) * quot_t_bpt).ceil() as u64 + 64;
                let quot_q_bits = ((block_terms as f64) * quot_q_bpt).ceil() as u64 + 64;
                let quot_p_bits = ((block_terms as f64) * quot_p_bpt).ceil() as u64 + 64;

                // Price SequentialHensel (`O(q_limbs * f_limbs)` across Q, T, and P):
                let qt_limbs = quot_t_bits.div_ceil(64) as f64;
                let qq_limbs = quot_q_bits.div_ceil(64) as f64;
                let qp_limbs = quot_p_bits.div_ceil(64) as f64;
                let f_limbs = factor_bits.div_ceil(64) as f64;
                let hensel_ops = |ql: f64, fl: f64| -> f64 {
                    let eff_f = fl.min(ql);
                    (ql * eff_f - 0.5 * eff_f * (eff_f - 1.0)).max(ql)
                };
                let seq_ops_per_node =
                    hensel_ops(qt_limbs, f_limbs) + hensel_ops(qq_limbs, f_limbs) + hensel_ops(qp_limbs, f_limbs);
                // Measured in runs, the sequential form costs 2 to 2.7 times this count of limb
                // products at the basecase rate (4.7 s against 1.7 at 5e9 digits): 0.26 of the
                // rate, where the count alone would have 0.65.
                let seq_level_sec = (seq_ops_per_node * (nodes as f64))
                    / ((cal.basecase_limbs_sq_per_sec * 0.26).max(1e6) * (active_w as f64));

                // Price SingleProduct at the concurrency the series gives it: as many nodes at
                // once as make a quotient cheapest in worker time.
                let div_conc = level_conc(quot_t_bits, quot_t_bits, false, true, nodes, &|pw| {
                    choose_product_method_and_price(
                        quot_t_bits,
                        quot_t_bits,
                        false,
                        false,
                        pw,
                        shares,
                        cal,
                        ram_transform_budget,
                        overrides,
                    )
                    .1
                    .total_sec
                        * (pw as f64)
                });
                let div_par_w = (active_w / div_conc.max(1)).max(1);
                // The passes over the numbers run at cache speed where the nodes at work fit it.
                let div_fits_cache = shares.fits_l3_domains(
                    div_conc as u64,
                    (quot_t_bits / 8).saturating_mul(3),
                    div_par_w,
                    7,
                    8,
                );
                let div_node_par = div_conc as f64;

                let (_, sp_t_mp) = choose_product_method_and_price(
                    quot_t_bits,
                    quot_t_bits,
                    false,
                    false,
                    div_par_w,
                    shares,
                    cal,
                    ram_transform_budget,
                    overrides,
                );
                let (_, sp_q_mp) = choose_product_method_and_price(
                    quot_q_bits,
                    quot_q_bits,
                    false,
                    false,
                    div_par_w,
                    shares,
                    cal,
                    ram_transform_budget,
                    overrides,
                );
                let (_, sp_p_mp) = choose_product_method_and_price(
                    quot_p_bits,
                    quot_p_bits,
                    false,
                    false,
                    div_par_w,
                    shares,
                    cal,
                    ram_transform_budget,
                    overrides,
                );
                let sp_scan_bw = if div_fits_cache {
                    (cal.ram_bw_bps * 8.0).max(400e9)
                } else {
                    cal.ram_bw_bps.max(1e9)
                };
                let sp_mem_sec = (((quot_t_bits + quot_q_bits + quot_p_bits) * 2 / 8) as f64)
                    * (nodes as f64)
                    / sp_scan_bw;
                let sp_t_level_sec = level_wall_sec(
                    &sp_t_mp,
                    (1.80 / 3.0) * (nodes as f64),
                    div_node_par,
                );
                let sp_q_level_sec = level_wall_sec(
                    &sp_q_mp,
                    (1.80 / 3.0) * (nodes as f64),
                    div_node_par,
                );
                let sp_p_level_sec = level_wall_sec(
                    &sp_p_mp,
                    (1.80 / 3.0) * ((nodes - 1) as f64),
                    div_node_par,
                );
                let sp_level_sec = sp_t_level_sec + sp_q_level_sec + sp_p_level_sec + sp_mem_sec;

                let seq_call_ovh_sec = ((3.0 * (nodes as f64)) * 6.0e-7) / (active_w as f64);
                let sp_call_ovh_sec = ((nodes as f64) * 1.8e-5) / (active_w as f64);
                let (best_form, nodes_div_sec) =
                    if seq_level_sec + seq_call_ovh_sec <= sp_level_sec + sp_call_ovh_sec {
                        (DivForm::SequentialHensel, seq_level_sec + seq_call_ovh_sec)
                    } else {
                        (DivForm::SingleProduct, sp_level_sec + sp_call_ovh_sec)
                    };

                let fac_half_bits = (factor_bits / 2).max(64);
                let (_, fac_mp) = choose_product_method_and_price(
                    fac_half_bits,
                    fac_half_bits,
                    false,
                    false,
                    shares.active_workers,
                    shares,
                    cal,
                    ram_transform_budget,
                    overrides,
                );
                let inv_setup_sec = if best_form == DivForm::SequentialHensel {
                    0.0
                } else {
                    let half_q = (quot_t_bits / 2).max(64);
                    let (_, inv_fv_mp) = choose_product_method_and_price(
                        factor_bits.min(half_q),
                        half_q,
                        false,
                        false,
                        shares.active_workers,
                        shares,
                        cal,
                        ram_transform_budget,
                        overrides,
                    );
                    let (_, inv_vk_mp) = choose_product_method_and_price(
                        half_q,
                        half_q,
                        false,
                        false,
                        shares.active_workers,
                        shares,
                        cal,
                        ram_transform_budget,
                        overrides,
                    );
                    (inv_fv_mp.total_sec + inv_vk_mp.total_sec) * 1.75
                        + (sp_t_mp.total_sec + sp_q_mp.total_sec + sp_p_mp.total_sec)
                            * (1.0 / 3.0)
                            / div_node_par.max(1.0)
                };
                let tree_mult = ((block_terms as f64).max(2.0).log2() * 0.17).clamp(1.1, 2.2);
                let factor_setup_sec = fac_mp.total_sec * tree_mult
                    + inv_setup_sec
                    + ((block_terms as f64) * 1.5) / cal.sieve_ops_per_sec.max(1e7);
                let div_ratio_w = if best_form == DivForm::SingleProduct {
                    0.15
                } else {
                    0.5
                };
                let nodes_div_sec =
                    nodes_div_sec * (1.0 + div_ratio_w * (level_cost_ratio(quot_t_bits) - 1.0));
                let div_level_sec = factor_setup_sec + nodes_div_sec;
                if dp_trace {
                    eprintln!(
                        "[TRACE DP] div lv={lv} after_div={prev_d} form={best_form:?} conc={div_conc} quot_mbit={:.1} nodes={:.1}ms setup={:.1}ms",
                        (quot_t_bits as f64) * 1e-6,
                        nodes_div_sec * 1e3,
                        factor_setup_sec * 1e3
                    );
                }

                let total_with_div = after_merge + div_level_sec;
                if total_with_div < dp[lv][lv] {
                    dp[lv][lv] = total_with_div;
                    parent[lv][lv] = Some((prev_d, Some(best_form)));
                }
            }
        }
    }

    // Backtrack from best state at `lv == h` (including root `Finish` `cq_odd * z_final` + linear setup).
    let root_terms = leaf_len.saturating_mul(1u64 << h) as f64;
    let z_bits = (((term_count as f64) * 47.110_403_207) as u64 + 128).max(64);
    let finish_root_cost = |d: usize| -> f64 {
        let t_bpt = bpt_after_div(d);
        let root_q_bits = ((root_terms * (t_bpt - 18.0).max(16.0)).ceil() as u64).max(64);
        let (_, cq_z_mp) = choose_product_method_and_price(
            root_q_bits.min(z_bits.saturating_add(384)),
            z_bits,
            false,
            false,
            shares.active_workers,
            shares,
            cal,
            ram_transform_budget,
            overrides,
        );
        let setup_bytes = root_terms * (t_bpt / 8.0) * 8.0;
        cq_z_mp.total_sec + setup_bytes / cal.ram_bw_bps.max(1e9)
    };
    let mut best_d = 0usize;
    let mut best_root_total = dp[h][0] + finish_root_cost(0);
    for d in 1..h {
        let cand = dp[h][d] + finish_root_cost(d);
        if cand < best_root_total {
            best_root_total = cand;
            best_d = d;
        }
    }

    let mut rev_plans = Vec::new();
    let mut cur_state = best_d;
    for lv in (1..=h).rev() {
        if let Some((prev_state, maybe_form)) = parent[lv][cur_state] {
            if let Some(form) = maybe_form {
                let block_terms = leaf_len.saturating_mul(1u64 << lv);
                let span_m = (lv - prev_state) as u64;
                rev_plans.push(DivLevelPlan {
                    level: lv as u32,
                    prev_div_level: prev_state as u32,
                    block_terms,
                    form,
                    est_factor_bits: 3 * span_m * block_terms,
                });
            }
            cur_state = prev_state;
        }
    }
    rev_plans.reverse();
    if dp_trace {
        let line: Vec<String> =
            rev_plans.iter().map(|d| format!("{}:{:?}", d.level, d.form)).collect();
        eprintln!("[TRACE DP] chosen {} est_total={:.2}s", line.join(","), best_root_total);
    }
    rev_plans
}

/// Choose the optimal `MulMethod` and compute its `ProductPrice` for `(a_bits, b_bits)`.
/// Guarantees that no transform coefficient or outer-transform part is a multiple of 61 bits!
#[allow(clippy::too_many_arguments)]
pub fn choose_product_method_and_price(
    a_bits: u64,
    b_bits: u64,
    fused_sum: bool,
    exact_only: bool,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    overrides: &PlanOverrides,
) -> (MulMethod, ProductPrice) {
    if overrides.split_override == SplitFormOverride::Auto {
        #[derive(Clone)]
        struct PriceMemoSlot {
            k0: u64,
            k1: u64,
            k2: u64,
            k3: u64,
            val: (MulMethod, ProductPrice),
        }
        thread_local! {
            static PRICE_MEMO: std::cell::RefCell<Vec<PriceMemoSlot>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let (lo_b, hi_b) = if a_bits <= b_bits {
            (a_bits, b_bits)
        } else {
            (b_bits, a_bits)
        };
        let mut cal_sig = cal
            .ntt_bf_per_sec_per_worker
            .to_bits()
            .rotate_left(13)
            ^ cal.fft_bf_per_sec_per_worker.to_bits().rotate_left(5)
            ^ cal.ram_bw_bps.to_bits().rotate_left(29)
            ^ cal.crt_ops_per_sec_per_worker.to_bits().rotate_left(47)
            ^ cal.basecase_limbs_sq_per_sec.to_bits().rotate_left(37)
            ^ cal.overlap_alpha.to_bits().rotate_left(23)
            ^ cal.sweep_step_sec.to_bits().rotate_left(41)
            ^ ((cal.karatsuba_max_limbs as u64) << 16)
            ^ (cal.basecase_max_limbs as u64)
            ^ (u64::from(cal.nt_effective) << 63)
            ^ (cal.devices.len() as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        for (di, d) in cal.devices.iter().enumerate() {
            cal_sig = cal_sig.rotate_left(9)
                ^ d.idle_rate_bps.to_bits()
                ^ d.beside_sweep_rate_bps.to_bits().rotate_left(19)
                ^ ((di as u64) << 3);
        }
        let sh_sig = shares
            .min_l2_share
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ shares.min_l1_share.rotate_left(11)
            ^ shares.min_l3_share.rotate_left(17)
            ^ shares.total_l3.rotate_left(33)
            ^ shares.min_l3_domain_size.rotate_left(45)
            ^ shares.num_l3_domains.rotate_left(7)
            ^ shares.fft_l1_sub_n.rotate_left(19)
            ^ shares.total_ram.rotate_left(25)
            ^ shares.total_vram.rotate_left(39)
            ^ shares.total_ssd.rotate_left(53)
            ^ shares.worker_count.rotate_left(48);
        let k0 = cal_sig ^ sh_sig ^ ram_transform_budget.rotate_left(7);
        let k1 = lo_b;
        let k2 = hi_b;
        let k3 = ((parallel_workers as u64) << 10)
            | ((overrides.max_ntt_primes as u64) << 2)
            | (u64::from(fused_sum) << 1)
            | u64::from(exact_only);
        let h = (k0
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(k1.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add(k2.wrapping_mul(0x94D0_49BB_1331_11EB))
            .wrapping_add(k3.wrapping_mul(0xD6E8_FEB8_6659_FD93))
            >> 53) as usize;
        if let Some(hit) = PRICE_MEMO.with(|cell| {
            let g = cell.borrow();
            if g.len() == 2048 {
                let s = &g[h];
                if s.k0 == k0 && s.k1 == k1 && s.k2 == k2 && s.k3 == k3 {
                    return Some(s.val.clone());
                }
            }
            None
        }) {
            return hit;
        }
        let res = choose_product_method_and_price_uncached(
            lo_b,
            hi_b,
            fused_sum,
            exact_only,
            parallel_workers,
            shares,
            cal,
            ram_transform_budget,
            overrides,
        );
        PRICE_MEMO.with(|cell| {
            let mut g = cell.borrow_mut();
            if g.len() != 2048 {
                g.resize(
                    2048,
                    PriceMemoSlot {
                        k0: 0,
                        k1: 0,
                        k2: 0,
                        k3: 0,
                        val: (
                            MulMethod::Basecase,
                            ProductPrice::combine(0.0, 0.0, 0.0, 0, 0.0),
                        ),
                    },
                );
            }
            g[h] = PriceMemoSlot {
                k0,
                k1,
                k2,
                k3,
                val: res.clone(),
            };
        });
        return res;
    }
    choose_product_method_and_price_uncached(
        a_bits,
        b_bits,
        fused_sum,
        exact_only,
        parallel_workers,
        shares,
        cal,
        ram_transform_budget,
        overrides,
    )
}

#[allow(clippy::too_many_arguments)]
fn choose_product_method_and_price_uncached(
    a_bits: u64,
    b_bits: u64,
    fused_sum: bool,
    exact_only: bool,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    overrides: &PlanOverrides,
) -> (MulMethod, ProductPrice) {
    let a_limbs = a_bits.div_ceil(64).max(1);
    let b_limbs = b_bits.div_ceil(64).max(1);
    let min_limbs = a_limbs.min(b_limbs);
    let max_limbs = a_limbs.max(b_limbs);
    let out_bits = a_bits.saturating_add(b_bits).saturating_add(64);
    let workers = parallel_workers.max(1) as f64;

    // 1. Basecase & Karatsuba for small limb counts.
    let base_long_cap = (cal.basecase_max_limbs * 2).max(cal.karatsuba_max_limbs) as u64;
    let row_stream_cap = if cal.mul_base_rows_max < (cal.basecase_max_limbs as u64) {
        cal.mul_base_rows_max
    } else {
        0
    };
    if (min_limbs <= (cal.basecase_max_limbs as u64) && max_limbs <= base_long_cap)
        || min_limbs <= row_stream_cap
    {
        let ops = (a_limbs as f64) * (b_limbs as f64);
        let comp = ops / cal.basecase_limbs_sq_per_sec.max(1e6);
        let bytes = (a_limbs + b_limbs) * 16;
        let traf = (bytes as f64) / (cal.ram_bw_bps * 4.0).max(1e9);
        return (
            MulMethod::Basecase,
            ProductPrice::combine(comp, traf, cal.overlap_alpha, bytes, ops * 0.25),
        );
    }

    if max_limbs <= (cal.karatsuba_max_limbs as u64) {
        let ops = (max_limbs as f64).powf(1.585);
        let comp = ops / cal.basecase_limbs_sq_per_sec.max(1e6);
        let bytes = max_limbs * 48;
        let traf = (bytes as f64) / (cal.ram_bw_bps * 3.0).max(1e9);
        return (
            MulMethod::Karatsuba {
                basecase_cutoff: cal.basecase_max_limbs,
            },
            ProductPrice::combine(comp, traf, cal.overlap_alpha, bytes, ops * 0.25),
        );
    }

    // Honor explicit split override for Gate 2 testing!
    match overrides.split_override {
        SplitFormOverride::ForceHalving(levels) => {
            let mut sub_ov = overrides.clone();
            sub_ov.split_override = SplitFormOverride::Auto;
            let half_op = a_bits
                .max(b_bits)
                .div_ceil(1u64 << levels)
                .next_multiple_of(64)
                .saturating_add((levels as u64) + 128);
            let (inner, inner_p) = choose_product_method_and_price(
                half_op,
                half_op,
                false,
                true,
                parallel_workers,
                shares,
                cal,
                ram_transform_budget,
                &sub_ov,
            );
            let mult = 3.0f64.powi(levels as i32);
            return (
                MulMethod::SplitHalving {
                    levels,
                    inner: Box::new(inner),
                },
                ProductPrice::combine(
                    inner_p.compute_sec * mult,
                    inner_p.traffic_sec * mult,
                    cal.overlap_alpha,
                    ((inner_p.bytes_moved as f64) * mult) as u64,
                    inner_p.butterflies_equiv * mult,
                ),
            );
        }
        SplitFormOverride::ForceOuter(k_parts) => {
            if let Some((m, p)) = build_outer_split_plan(
                out_bits,
                k_parts,
                parallel_workers,
                shares,
                cal,
                ram_transform_budget,
                overrides.max_ntt_primes,
            ) {
                return (m, p);
            }
        }
        SplitFormOverride::Auto => {}
    }

    // 2. Right-angle Complex FFT when `!exact_only` and working set fits worker's L2 share (Section 5).
    let mut fft_cand: Option<(MulMethod, ProductPrice)> = None;
    if !exact_only {
        let fft_budget = shares
            .min_l2_share
            .min((ram_transform_budget / 3).max(32 * 1024));
        if let Some((n_complex, coeff_bits)) = choose_fft_params(out_bits, fused_sum, fft_budget) {
            debug_assert_ne!(coeff_bits % 61, 0);
            let bf = 3.0 * ((n_complex / 2) as f64) * (n_complex.ilog2() as f64);
            let edge_bf = 5.5 * (n_complex as f64);
            let l1_sub = shares.fft_l1_sub_n;
            let n_u64 = n_complex as u64;
            let fft_scale = if n_u64 >= l1_sub.saturating_mul(4) {
                1.30
            } else if n_u64 > l1_sub {
                1.12
            } else {
                1.0
            };
            let comp = (bf + edge_bf) * fft_scale / cal.fft_bf_per_sec_per_worker.max(1e6);
            let ws_bytes = (n_complex as u64) * 16;
            let bytes = ws_bytes * 3;
            let cache_mult = if ws_bytes <= shares.min_l1_share {
                12.0
            } else if ws_bytes <= shares.min_l1_share.saturating_mul(2) {
                8.0
            } else {
                5.0
            };
            let traf = (bytes as f64) / (cal.ram_bw_bps * cache_mult).max(1e9);
            let price = ProductPrice::combine(comp, traf, cal.overlap_alpha, bytes, bf);
            if parallel_workers <= 1 || out_bits <= 65_536 {
                return (
                    MulMethod::Fft {
                        n_complex,
                        coeff_bits,
                    },
                    price,
                );
            }
            fft_cand = Some((
                MulMethod::Fft {
                    n_complex,
                    coeff_bits,
                },
                price,
            ));
        }
    }

    // 3. Single-transform NTT search across `(log_n, k_primes, coeff_bits, packed, grid)`.
    if let Some((spec, price)) = search_best_ntt_spec(
        out_bits,
        fused_sum,
        false,
        16,
        workers,
        shares,
        cal,
        ram_transform_budget,
        overrides.max_ntt_primes,
    ) {
        if let Some((fm, fp)) = fft_cand {
            let sync_overhead_sec = 1.5e-5;
            if fp.total_sec <= price.total_sec + sync_overhead_sec {
                return (fm, fp);
            }
        }
        return (MulMethod::Ntt(spec), price);
    }
    if let Some((fm, fp)) = fft_cand {
        return (fm, fp);
    }

    // 4. Product exceeds single-transform capacity (Section 5: > 1.82e15 bits, or constrained
    //    `max_ntt_primes` in Gate 2): price both SplitHalving and SplitOuter (`mod 2^m + 1`) and
    //    pick the faster!
    let mut best_split: Option<(MulMethod, ProductPrice)> = None;

    for levels in 1u32..=4 {
        let half_op = a_bits
            .max(b_bits)
            .div_ceil(1u64 << levels)
            .next_multiple_of(64)
            .saturating_add((levels as u64) + 128);
        let sub_out = half_op.saturating_mul(2).saturating_add(64);
        if let Some((inner_spec, inner_p)) = search_best_ntt_spec(
            sub_out,
            false,
            false,
            16,
            workers,
            shares,
            cal,
            ram_transform_budget,
            overrides.max_ntt_primes,
        ) {
            let mult = 3.0f64.powi(levels as i32);
            let p = ProductPrice::combine(
                inner_p.compute_sec * mult,
                inner_p.traffic_sec * mult,
                cal.overlap_alpha,
                ((inner_p.bytes_moved as f64) * mult) as u64,
                inner_p.butterflies_equiv * mult,
            );
            best_split = Some((
                MulMethod::SplitHalving {
                    levels,
                    inner: Box::new(MulMethod::Ntt(inner_spec)),
                },
                p,
            ));
            break;
        }
    }

    for k_parts in [4usize, 8, 16, 32, 64] {
        if let Some((cand_m, cand_p)) = build_outer_split_plan(
            out_bits,
            k_parts,
            parallel_workers,
            shares,
            cal,
            ram_transform_budget,
            overrides.max_ntt_primes,
        ) {
            if best_split
                .as_ref()
                .is_none_or(|(_, bp)| cand_p.total_sec < bp.total_sec)
            {
                best_split = Some((cand_m, cand_p));
            }
        }
    }

    best_split.expect("SplitHalving or SplitOuter must cover every 64-bit product size")
}

/// Choose an exact wrapped `NttSpec` (`mod 2^L - 1`) with `L >= min_transform_bits` and
/// `coeff_bits % 61 != 0` for `newton.rs` and `radix.rs`.
pub fn choose_wrapped_ntt_spec(
    min_transform_bits: u64,
    operand_max_bits: u64,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_ntt_primes: usize,
) -> NttSpec {
    choose_wrapped_ntt_spec_amortized(
        min_transform_bits,
        operand_max_bits,
        parallel_workers,
        shares,
        cal,
        ram_transform_budget,
        max_ntt_primes,
        1,
    )
}

/// Among the shapes of a wrapped product with a reused transform that are priced within this
/// factor of the cheapest, the one with the fewest points is taken.
///
/// The prices of two such shapes can lie within a thousandth of each other, and the choice then
/// went by the noise of the start-up calibration, one way in one run and the other in the next.
/// Measured on the base conversion's level of 14 Mbit nodes, where 2^16 points with nine primes
/// and 2^17 points with five are priced alike (sixteen products at once, a worker each): the
/// product took 4.9 ms with the former and 6.3 ms with the latter, and the level below ran a
/// quarter faster beside the former. The conversion alone, the former against the latter:
/// equal at 10^8 digits, 0.3% shorter at 10^9, 1.2% at 2.5 * 10^9, 1.4% at 5 * 10^9.
pub const WRAPPED_FEWER_POINTS_MARGIN: f64 = 1.03;

/// Choose an exact wrapped `NttSpec` (`mod 2^L - 1`) when one operand's forward transform is
/// reused across `reuse_count` products (`reuse_count == 1` prices a standalone product).
#[allow(clippy::too_many_arguments)]
pub fn choose_wrapped_ntt_spec_amortized(
    min_transform_bits: u64,
    operand_max_bits: u64,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_ntt_primes: usize,
    reuse_count: u64,
) -> NttSpec {
    let workers = parallel_workers.max(1) as f64;
    let target = min_transform_bits.max(128);
    // When `operand_max_bits <= target`, operands never fold before `ntt_split`, so unsigned cyclic
    // convolution sum is strictly `< n * 2^{2b}` (`extra_sum_bit = false`).
    let extra_sum_bit = operand_max_bits > target;
    let (mut spec, _) = if reuse_count >= 2 && !extra_sum_bit {
        let mut all: Vec<(NttSpec, ProductPrice)> = Vec::new();
        search_best_ntt_spec_uncached(
            target,
            false,
            false,
            16,
            workers,
            shares,
            cal,
            ram_transform_budget,
            max_ntt_primes.max(3),
            reuse_count,
            Some(&mut all),
        )
        .map(|(best_spec, best_price)| {
            let limit = best_price.total_sec * WRAPPED_FEWER_POINTS_MARGIN;
            all.into_iter()
                .filter(|(_, p)| p.total_sec <= limit)
                .min_by(|a, b| a.0.n.cmp(&b.0.n).then(a.1.total_sec.total_cmp(&b.1.total_sec)))
                .unwrap_or((best_spec, best_price))
        })
    } else {
        search_best_ntt_spec(
            target,
            extra_sum_bit,
            false,
            16,
            workers,
            shares,
            cal,
            ram_transform_budget,
            max_ntt_primes.max(3),
        )
    }
    .expect("wrapped transform fits single NTT");
    spec.wrap_cyclic = true;
    spec
}

#[allow(clippy::too_many_arguments)]
fn build_outer_split_plan(
    out_bits: u64,
    k_parts: usize,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_ntt_primes: usize,
) -> Option<(MulMethod, ProductPrice)> {
    if !k_parts.is_power_of_two() || k_parts < 4 {
        return None;
    }
    let mut part_bits_w = out_bits.saturating_add(128).div_ceil(k_parts as u64).max(16);
    if part_bits_w % 61 == 0 {
        part_bits_w += 1;
    }
    let min_m = 2 * part_bits_w + (k_parts.ilog2() as u64) + 8;

    let workers = parallel_workers.max(1) as f64;
    let (mut inner_spec, inner_price) = search_best_ntt_spec(
        min_m,
        false,
        true, // negacyclic = true!
        k_parts.max(16),
        workers,
        shares,
        cal,
        ram_transform_budget,
        max_ntt_primes.max(3),
    )?;

    let m_bits = (inner_spec.n as u64) * (inner_spec.coeff_bits as u64);
    if m_bits < min_m {
        return None;
    }
    inner_spec.negacyclic = true;

    let kf = k_parts as f64;
    let outer_shift_bytes = (k_parts as u64) * m_bits.div_ceil(8) * (k_parts.ilog2() as u64) * 2;
    let outer_traf = (outer_shift_bytes as f64) / cal.ram_bw_bps.max(1e9);
    let price = ProductPrice::combine(
        inner_price.compute_sec * kf,
        inner_price.traffic_sec * kf + outer_traf,
        cal.overlap_alpha,
        inner_price
            .bytes_moved
            .saturating_mul(k_parts as u64)
            .saturating_add(outer_shift_bytes),
        inner_price.butterflies_equiv * kf,
    );
    Some((
        MulMethod::SplitOuter {
            k_parts,
            part_bits_w,
            inner_ntt: inner_spec,
        },
        price,
    ))
}

#[derive(Clone, Copy)]
struct NttSpecCacheEntry {
    slot_key: u64,
    val: Option<(NttSpec, ProductPrice)>,
}

struct NttSpecTlsCache {
    ctx_key: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, bool),
    slots: [NttSpecCacheEntry; 32],
}

impl NttSpecTlsCache {
    const fn new() -> Self {
        Self {
            ctx_key: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false),
            slots: [NttSpecCacheEntry {
                slot_key: 0,
                val: None,
            }; 32],
        }
    }
}

thread_local! {
    static NTT_SPEC_TLS_CACHE: std::cell::RefCell<NttSpecTlsCache> =
        const { std::cell::RefCell::new(NttSpecTlsCache::new()) };
}

#[allow(clippy::too_many_arguments)]
fn search_best_ntt_spec(
    target_bits: u64,
    extra_sum_bit: bool,
    negacyclic: bool,
    min_n_floor: usize,
    workers: f64,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_ntt_primes: usize,
) -> Option<(NttSpec, ProductPrice)> {
    let max_k = max_ntt_primes.clamp(1, 18);
    let cacheable = (workers - 1.0).abs() < 1e-9
        && min_n_floor <= 16
        && max_k == 18
        && target_bits >= 16_384
        && target_bits <= (u64::MAX >> 4);
    if cacheable {
        let q_target = (target_bits + 127) & !127;
        let slot_key = (q_target << 2) | ((extra_sum_bit as u64) << 1) | (negacyclic as u64);
        let idx = (((q_target >> 7)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((extra_sum_bit as u64) << 61)
            ^ ((negacyclic as u64) << 60))
            >> 59) as usize;
        let cur_ctx = (
            shares.min_l2_share,
            shares.min_l3_share,
            shares.total_l3,
            shares.min_l3_domain_size,
            shares.num_l3_domains,
            shares.worker_count,
            ram_transform_budget,
            cal.ntt_bf_per_sec_per_worker.to_bits(),
            cal.crt_ops_per_sec_per_worker.to_bits(),
            cal.ram_bw_bps.to_bits(),
            cal.nt_effective,
        );
        if let Some(hit) = NTT_SPEC_TLS_CACHE.with(|cell| {
            let mut c = cell.borrow_mut();
            if c.ctx_key != cur_ctx {
                c.ctx_key = cur_ctx;
                c.slots.fill(NttSpecCacheEntry {
                    slot_key: 0,
                    val: None,
                });
                None
            } else if c.slots[idx].slot_key == slot_key {
                Some(c.slots[idx].val)
            } else {
                None
            }
        }) {
            return hit;
        }
        let res = search_best_ntt_spec_uncached(
            q_target,
            extra_sum_bit,
            negacyclic,
            min_n_floor,
            workers,
            shares,
            cal,
            ram_transform_budget,
            max_k,
            0,
            None,
        );
        NTT_SPEC_TLS_CACHE.with(|cell| {
            let mut c = cell.borrow_mut();
            c.slots[idx] = NttSpecCacheEntry { slot_key, val: res };
        });
        return res;
    }
    search_best_ntt_spec_uncached(
        target_bits,
        extra_sum_bit,
        negacyclic,
        min_n_floor,
        workers,
        shares,
        cal,
        ram_transform_budget,
        max_k,
        0,
        None,
    )
}

/// Every transform shape the planner weighs for a product of `a_bits` by `b_bits` (`fused_sum`
/// as for a fused merge) on `parallel_workers` workers, each with the price it gives it; the
/// planner takes the cheapest. For the engine's price check, which times them in place.
#[allow(clippy::too_many_arguments)]
pub fn list_ntt_specs_with_prices(
    a_bits: u64,
    b_bits: u64,
    fused_sum: bool,
    parallel_workers: usize,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_ntt_primes: usize,
) -> Vec<(NttSpec, ProductPrice)> {
    let mut all = Vec::new();
    let _ = search_best_ntt_spec_uncached(
        a_bits.saturating_add(b_bits).saturating_add(64),
        fused_sum,
        false,
        16,
        parallel_workers.max(1) as f64,
        shares,
        cal,
        ram_transform_budget,
        max_ntt_primes.clamp(1, 18),
        0,
        Some(&mut all),
    );
    all
}

/// More nodes of a level run at once only where that makes a node cheaper by more than this
/// factor: where the prices are level (the largest products), fewer at once hold less memory.
pub const MORE_AT_ONCE_MARGIN: f64 = 0.98;

/// Whether `c` merge nodes of a level whose products take the grid `sp` may run at once, by the
/// room their transforms take in the last-level cache. A node's three grids (two operands and
/// their product, eight bytes a residue) are what it keeps coming back to. Where they are
/// larger than the cache nothing is to be saved there, and what a product loses between its
/// workers decides: such nodes run with four workers each. Smaller nodes run as many at once
/// as keep all their grids within six tenths of the cache, and one at once beyond that.
///
/// Measured (sixteen workers, 96 MB of last-level cache) on every level of runs of 10^9 and
/// 5 * 10^9 digits with 1, 2, 4 and 8 nodes at once: grids of 14 MB a node were cheapest at 4
/// at once and 30% dearer at 8, of 22 to 28 MB at 2 (17 to 23% dearer at 4), of 44 to 88 MB at
/// 1, and grids of 150 MB and more at 4, where 1 at once cost 10 to 21% more.
pub fn nodes_at_once_fit(c: usize, sp: &NttSpec, shares: &Shares) -> bool {
    if c <= 1 {
        return true;
    }
    let grids = 3u64
        .saturating_mul(sp.k_primes as u64)
        .saturating_mul(sp.n as u64)
        .saturating_mul(8);
    let single_node_l3 = shares.effective_l3_bytes(1, sp.rows);
    if grids >= single_node_l3 {
        c <= (shares.active_workers / 4).max(1)
    } else {
        shares.fits_l3_domains(c as u64, grids, sp.rows, 6, 10)
    }
}

/// What a pair of products that shares the transform of its common operand costs when the two
/// would take different grids and both are made in the larger: in products of the larger.
/// Measured between 1.66 and 2.08 (the price check, form 2).
pub const UNEQUAL_PAIR_COST: f64 = 1.85;

/// Cost of sharing `fwd(b)` across `(x * b, y * b)` in units of the larger product's cost:
/// packed 2D grids with identical `(n, rows)` run Sweep 1 of `(x, y)` in one pass and stream
/// `b` once (`~1.50..1.58` measured); same-`n` unpacked/1D NTT avoids a full transform (`1.74`);
/// cross-`n` packed (`1.72`) and general (`UNEQUAL_PAIR_COST`) stay conservative.
pub fn pair_shared_cost_factor(big_m: &MulMethod, small_m: &MulMethod) -> f64 {
    match (big_m, small_m) {
        (MulMethod::Ntt(sb), MulMethod::Ntt(ss)) => {
            if sb.n == ss.n && sb.rows == ss.rows {
                if sb.packed && sb.rows >= 64 {
                    1.62
                } else {
                    1.65
                }
            } else if sb.packed && sb.rows >= 64 {
                1.72
            } else {
                UNEQUAL_PAIR_COST
            }
        }
        _ => UNEQUAL_PAIR_COST,
    }
}

/// What a transform costs a residue besides its butterflies, in butterflies: twisting, the
/// pointwise product, packing and moving, in cache. Fitted to the price check (0.3 ns where a
/// butterfly takes 0.19 with every hardware thread busy).
const RESIDUE_BASE_OPS: f64 = 1.58;

/// Bytes a transform moves through memory a residue once the products running at the same time
/// have outgrown the last-level cache. Fitted to the price check.
const RESIDUE_TRAFFIC_BYTES: f64 = 12.0;

/// What a worker of a product loses at the ends of its passes, waiting for the others: seconds
/// a worker a product of three transforms. Measured: products of 7 to 16 megabits cost 0.6 to
/// 0.9 ms more worker time on sixteen workers than on four.
const WORKER_SYNC_SEC: f64 = 250.0e-6;

/// What every further worker of a product costs it in worker time. Measured: products of half
/// a gigabit to two gigabits cost 5 to 12% more worker time on sixteen workers than on four.
const WORKER_LOSS: f64 = 0.0055;

/// One recombination a coefficient, in the calibration's unit (its benchmark, one split and one
/// recombination at four primes and 80 bits, is 16.82 units a coefficient; a unit is 0.45 ns on
/// the machine the table was taken on). Measured with every hardware thread busy
/// (`time_crt_and_split_on_all_threads`): the kernel's block path serves up to nine primes and
/// 208 bits; beyond nine primes a two-dimensional grid falls off the block path, and beyond
/// sixteen primes recombination is far slower.
fn crt_units_per_coeff(k_primes: usize, coeff_bits: u64, one_dimensional: bool) -> f64 {
    let k = k_primes as f64;
    let block = 0.294 * k * k + 1.918 * k;
    let odd_2d_mult = if !one_dimensional && (coeff_bits & 1) == 1 && coeff_bits > 64 {
        1.25
    } else {
        1.0
    };
    let seg_2d_ovh = if !one_dimensional { 8.0 } else { 0.0 };
    if !one_dimensional && (k_primes >= 10 || coeff_bits > 224) && k_primes <= 16 && coeff_bits <= 384 {
        return (320.0 + 42.0 * (k - 9.0).max(0.0)).max(block * 2.60);
    }
    if k_primes <= 5 && coeff_bits <= 128 {
        block * odd_2d_mult + seg_2d_ovh
    } else if k_primes <= 5 && coeff_bits <= 240 {
        let m = if coeff_bits <= 192 {
            1.10
        } else if coeff_bits <= 208 {
            1.25
        } else {
            1.45
        };
        block * m * odd_2d_mult + seg_2d_ovh
    } else if k_primes <= 9 && coeff_bits <= 192 {
        let m = if one_dimensional { 1.05 } else { 1.22 };
        block * m * odd_2d_mult + seg_2d_ovh
    } else if k_primes <= 9 && coeff_bits <= 208 {
        let m = if one_dimensional { 1.30 } else { 1.62 };
        block * m * odd_2d_mult + seg_2d_ovh
    } else if k_primes <= 9 && coeff_bits <= 240 {
        let m = if one_dimensional { 1.55 } else { 1.85 };
        block * m * odd_2d_mult + seg_2d_ovh
    } else if k_primes <= 16 && coeff_bits <= 384 {
        if one_dimensional {
            (130.0 + 22.0 * (k - 9.0)).max(block * 1.55)
        } else {
            (320.0 + 42.0 * (k - 9.0)).max(block * 2.60)
        }
    } else {
        825.0
    }
}

/// One split of an operand a coefficient, for all its primes, in the same unit: 48-bit chunks
/// extracted once per tile or segment and reduced modulo each prime.
fn split_units_per_coeff(k_primes: usize, coeff_bits: u64, one_dimensional: bool) -> f64 {
    let k = k_primes as f64;
    let chunks = coeff_bits.div_ceil(48).max(1) as f64;
    if one_dimensional && coeff_bits <= 240 && k_primes <= 10 {
        if coeff_bits <= 192 {
            chunks * (0.774 + 0.133 * k)
        } else {
            chunks * (0.95 + 0.16 * k)
        }
    } else if coeff_bits <= 52 {
        k * 0.686
    } else if coeff_bits <= 64 {
        k * 0.929
    } else if coeff_bits <= 192 {
        chunks * (0.75 + 0.18 * k)
    } else if coeff_bits <= 240 {
        chunks * (0.90 + 0.22 * k)
    } else if coeff_bits <= 256 {
        k * 3.81
    } else if coeff_bits <= 288 {
        k * 6.64
    } else if coeff_bits <= 336 {
        k * 13.05
    } else if coeff_bits <= 384 {
        k * 19.0
    } else {
        k * 59.7
    }
}

#[allow(clippy::too_many_arguments)]
fn search_best_ntt_spec_uncached(
    target_bits: u64,
    extra_sum_bit: bool,
    negacyclic: bool,
    min_n_floor: usize,
    workers: f64,
    shares: &Shares,
    cal: &Calibration,
    ram_transform_budget: u64,
    max_k: usize,
    reuse_b: u64,
    mut all: Option<&mut Vec<(NttSpec, ProductPrice)>>,
) -> Option<(NttSpec, ProductPrice)> {
    let mut best: Option<(NttSpec, ProductPrice)> = None;

    let min_n_needed = target_bits
        .div_ceil(384)
        .max(min_n_floor.max(16) as u64)
        .next_power_of_two();
    let min_log_n = min_n_needed.ilog2().max(4);
    let max_log_n = if negacyclic { 43u32 } else { 44u32 };

    for log_n in min_log_n..=max_log_n {
        let n_u64 = 1u64 << log_n;
        if n_u64 > (usize::MAX as u64) / 16 {
            break;
        }
        let two_adic = if negacyclic { log_n + 1 } else { log_n };
        let prefix_bits = largest_primes_prefix_bits_slice(two_adic.max(40), max_k);
        let min_k = 3.min(max_k);
        if prefix_bits.len() < min_k {
            continue;
        }

        let mut b = target_bits.div_ceil(n_u64);
        if b < 8 {
            if target_bits <= n_u64 * 8 {
                b = 8;
            } else {
                continue;
            }
        }
        if b % 61 == 0 {
            b += 1;
        }
        if negacyclic && (n_u64 * b) % 61 == 0 {
            b += 1;
            if b % 61 == 0 {
                b += 1;
            }
        }
        if b > 384 {
            continue;
        }

        // Minimum prime product bits needed: `2 * b + log_n + extra_sum_bit + 2`.
        let needed_prime_bits = (2 * b) as f64 + (log_n as f64) + if extra_sum_bit { 3.5 } else { 2.5 };
        let mut chosen_k = None;
        for (idx, &sum_p_bits) in prefix_bits.iter().enumerate().skip(min_k - 1) {
            if sum_p_bits >= needed_prime_bits {
                chosen_k = Some(idx + 1);
                break;
            }
        }
        let Some(k_primes) = chosen_k else {
            continue;
        };
        // Prefer even `b` (`(32 * b) % 64 == 0`) so 32-lane vectorized CRT segments align to
        // limb boundaries when staying within the same `k_primes` and without crossing `b == 49`.
        if (b & 1) == 1 && b != 49 {
            let b_cand = b + 1;
            if b_cand <= 384
                && b_cand % 61 != 0
                && (!negacyclic || (n_u64 * b_cand) % 61 != 0)
            {
                let need_cand =
                    (2 * b_cand) as f64 + (log_n as f64) + if extra_sum_bit { 3.5 } else { 2.5 };
                if prefix_bits[k_primes - 1] >= need_cand {
                    b = b_cand;
                }
            }
        }
        let _ = largest_primes_desc_slice;

        let n = n_u64 as usize;
        let par_w_usize = (workers as usize).max(1);
        let fits_ram_1d = (n_u64)
            .saturating_mul(k_primes as u64)
            .saturating_mul(24)
            <= ram_transform_budget.max(64 * 1024);
        let min_store_range_bytes = cal.devices.first().map_or(0, |d| d.min_batch_range_bytes);
        let (rows, cols, tile_cols, extra_sweeps) = choose_grid_shape(
            n,
            shares.min_l2_share,
            shares.min_l3_share,
            cal.nt_effective,
            par_w_usize,
            k_primes,
            b,
            fits_ram_1d,
            extra_sum_bit,
            ram_transform_budget,
            min_store_range_bytes,
        );

        let mut unpacked_fits_l3 = false;
        for packed in [false, true] {
            if packed && (unpacked_fits_l3 || rows == 1 || cols % 8 != 0 || tile_cols % 8 != 0) {
                continue;
            }
            let row_bytes = if packed {
                let base = (cols / 8) * 50;
                if rows > 1 && base.is_multiple_of(4096) {
                    base + 512
                } else {
                    base
                }
            } else if rows > 1 && cols >= 512 {
                cols * 8 + 512
            } else {
                cols * 8
            };
            let grid_bytes_one_op = (k_primes as u64)
                .saturating_mul(rows as u64)
                .saturating_mul(row_bytes as u64);
            let work_single_primes = if extra_sum_bit { 3u64 } else { 2u64 };
            let single_prime_bytes = (rows as u64).saturating_mul(row_bytes as u64);
            let can_stream_pbp = rows > 1 && cols >= 1024 && extra_sweeps == 0;
            let total_res_bytes = if can_stream_pbp {
                grid_bytes_one_op.saturating_add(single_prime_bytes.saturating_mul(work_single_primes))
            } else {
                grid_bytes_one_op
                    .saturating_mul(2)
                    .saturating_add(single_prime_bytes.saturating_mul(work_single_primes))
            };
            let out_of_core = total_res_bytes > ram_transform_budget.max(64 * 1024);
            let conc_transforms = (shares.worker_count / (par_w_usize.max(1) as u64)).max(1);
            let doms_spanned = if rows > 1 {
                shares.domains_spanned_by_workers(par_w_usize)
            } else {
                1
            };
            let cross_dom_frac = if doms_spanned > 1 {
                ((doms_spanned - 1) as f64) / (doms_spanned as f64)
            } else {
                0.0
            };
            let eff_l3 = shares.effective_l3_bytes(conc_transforms, rows);
            let l3_budget = eff_l3 * 7 / 8;
            let unpacked_row_b = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
            let prime_bytes_one_op = (rows as u64).saturating_mul(unpacked_row_b as u64);
            let out_grids = if extra_sum_bit { 3u64 } else { 1u64 };
            let work_primes = if extra_sum_bit { 3u64 } else if packed { 2u64 } else { 1u64 };
            let total_l3_per_tf = grid_bytes_one_op
                .saturating_mul(out_grids)
                .saturating_add(prime_bytes_one_op.saturating_mul(work_primes));
            let fits_l3_all = !out_of_core
                && doms_spanned == 1
                && shares.fits_l3_domains(conc_transforms, total_l3_per_tf, rows, 7, 8);
            if !packed {
                let pair_l3_per_tf = grid_bytes_one_op
                    .saturating_mul(if extra_sum_bit { 3 } else { 2 })
                    .saturating_add(prime_bytes_one_op.saturating_mul(2));
                unpacked_fits_l3 = !out_of_core
                    && (if doms_spanned == 1 {
                        shares.fits_l3_domains(conc_transforms, pair_l3_per_tf, rows, 1, 2)
                    } else {
                        pair_l3_per_tf <= shares.min_l3_domain_size / 2
                    });
            }

            // Unified NTT pricing model across 1D/2D in-RAM, `extra_sweeps > 0`, and `out_of_core`
            // (`SUPI_BENCH_PRICE` residue/butterfly/split/CRT model + Section 10 beside-sweep &
            // idle-rate out-of-core store overlap, normalized to 1 product).
            let raw_one_prod_bytes = target_bits / 8;
            let l3_staged_2_fits = doms_spanned == 1
                && shares.fits_l3_domains(
                    conc_transforms,
                    prime_bytes_one_op.saturating_mul(2),
                    rows,
                    7,
                    8,
                );
            let work_ws_primes = if l3_staged_2_fits || !extra_sum_bit { 2 } else { 3 };
            let work_ws_fits = doms_spanned == 1
                && shares.fits_l3_domains(
                    conc_transforms,
                    prime_bytes_one_op.saturating_mul(work_ws_primes),
                    rows,
                    7,
                    8,
                );
            let sweeps_mult = if rows == 1 {
                2u64
            } else if extra_sweeps == 0 && !out_of_core && (fits_l3_all || work_ws_fits) {
                2u64
            } else {
                5u64 + 4u64 * (extra_sweeps as u64)
            };
            let raw_passes_2d = if rows > 1 { k_primes as u64 + 1 } else { 2u64 };
            // Section 11: in out-of-core column sweeps, each range is checked in a pass of
            // `check_n` before it leaves RAM (Sweep 1) and as it arrives in RAM (Sweep 3),
            // and the plan counts that pass as RAM traffic.
            let col_check_bytes = if out_of_core {
                if rows > 1 {
                    if reuse_b >= 2 {
                        grid_bytes_one_op
                            .saturating_mul(2)
                            .saturating_add(grid_bytes_one_op / (reuse_b as u64))
                    } else {
                        grid_bytes_one_op.saturating_mul(3)
                    }
                } else {
                    grid_bytes_one_op.saturating_mul(2)
                }
            } else {
                0
            };
            let store_bytes_moved = grid_bytes_one_op
                .saturating_mul(sweeps_mult)
                .saturating_add(raw_one_prod_bytes.saturating_mul(raw_passes_2d));
            let bytes_moved = store_bytes_moved.saturating_add(col_check_bytes);

            let sw2_bpe: u64 = 24;
            let sw2_l3_bpe: u64 = if extra_sum_bit { 40 } else { 24 };
            let sw2_spills_l2 = rows > 1 && (cols as u64) * sw2_bpe > shares.min_l2_share;
            let sw2_spills_l3 = rows > 1 && (cols as u64) * sw2_l3_bpe > shares.min_l3_share;
            let sw2_spill_ratio = if rows > 1 {
                ((cols as f64) * (sw2_bpe as f64)) / (shares.min_l2_share.max(4096) as f64)
            } else {
                0.0
            };
            let sw2_bf_mult = if !sw2_spills_l2 {
                1.0
            } else if !sw2_spills_l3 {
                if sw2_spill_ratio <= 2.0 {
                    1.08
                } else if sw2_spill_ratio <= 4.0 {
                    1.16
                } else {
                    1.32
                }
            } else if sw2_spill_ratio <= 2.0 {
                1.28
            } else if sw2_spill_ratio <= 4.0 {
                1.65
            } else {
                2.15
            };

            let (tf, sp, cr, norm) = if extra_sum_bit {
                (8.0f64, 5.0f64, 2.0f64, 2.35f64)
            } else if reuse_b >= 2 {
                let inv_r = 1.0 / (reuse_b as f64);
                (2.0 + inv_r, 1.0 + inv_r, 1.0, 1.0)
            } else {
                (3.0, 2.0, 1.0, 1.0)
            };
            let n_f = n_u64 as f64;
            let k_f = k_primes as f64;
            let residue_tf = tf * n_f * k_f;
            let bf_ops = tf * (n_f / 2.0) * (log_n as f64) * k_f * sw2_bf_mult;
            let sub_row_ops = if rows >= 1024 {
                0.65 * residue_tf
            } else if rows >= 512 {
                0.25 * residue_tf
            } else if rows >= 256 {
                0.10 * residue_tf
            } else {
                0.0
            };
            let pack_res_ops = if rows > 1 {
                if packed {
                    2.05 * residue_tf
                } else {
                    0.85 * residue_tf
                }
            } else {
                0.0
            };
            let scalar_tile_ops = if rows > 1 && tile_cols < 32 {
                1.80 * residue_tf
            } else {
                0.0
            };
            let extra_sweep_ops = (extra_sweeps as f64) * 1.35 * residue_tf;
            let plane = n_f * 8.0;
            let l2_spill_ops = if rows == 1
                && shares.min_l2_share > 0
                && 2.0 * plane > shares.min_l2_share as f64
            {
                2.20 * residue_tf
            } else {
                0.0
            };
            let base_ops = RESIDUE_BASE_OPS * residue_tf
                + sub_row_ops
                + pack_res_ops
                + scalar_tile_ops
                + extra_sweep_ops
                + l2_spill_ops;
            let cs_units = n_f
                * (cr * crt_units_per_coeff(k_primes, b, rows == 1)
                    + sp * split_units_per_coeff(k_primes, b, rows == 1));
            let comp_thread = (bf_ops + base_ops) / cal.ntt_bf_per_sec_per_worker.max(1e6)
                + cs_units / cal.crt_ops_per_sec_per_worker.max(1e6);

            let grid = if rows > 1 {
                grid_bytes_one_op as f64
            } else {
                k_f * plane
            };
            let raw = raw_one_prod_bytes as f64;
            let ws_one = if extra_sum_bit {
                3.0 * grid + 3.0 * plane + 3.0 * raw
            } else if rows > 1 {
                2.0 * grid + 2.0 * plane + 2.0 * raw
            } else {
                grid + 2.0 * plane + 2.0 * raw
            };
            let ws_all = if !extra_sum_bit && reuse_b >= 2 && conc_transforms >= 2 {
                grid + (conc_transforms as f64) * (if rows > 1 { grid + raw } else { grid + 2.0 * plane + raw })
            } else {
                ws_one * (conc_transforms as f64)
            };
            let l3_all = eff_l3.max(1) as f64;
            let raw_spill = if ws_all <= l3_all {
                0.0
            } else {
                (ws_all - l3_all) / ws_all
            };
            let spill = raw_spill.max(cross_dom_frac);
            let pbp_cap = l3_all * 0.90;
            let single_pbp_ws = 2u64.saturating_mul(single_prime_bytes);
            let pbp_ws = single_pbp_ws.saturating_mul(conc_transforms) as f64;
            let pbp_spill = if rows <= 1 {
                0.0
            } else if (single_pbp_ws as f64) > l3_all * 1.40 {
                1.0 + 0.15 * (((single_pbp_ws as f64) / (l3_all * 1.40) - 1.0).min(2.0))
            } else if pbp_ws <= pbp_cap {
                0.0
            } else {
                (pbp_ws - pbp_cap) / pbp_ws
            };
            let base_res_traf = if packed && rows > 1 {
                RESIDUE_TRAFFIC_BYTES * (50.0 / 64.0)
            } else {
                RESIDUE_TRAFFIC_BYTES
            };
            let res_traf_bytes = base_res_traf * (1.0 + 0.65 * (extra_sweeps as f64));
            let traf_thread = (spill * res_traf_bytes + pbp_spill * 3.5)
                * residue_tf
                * (shares.worker_count.max(1) as f64)
                / cal.ram_bw_bps.max(1e9);

            // Workers of one product wait for each other at the end of every pass and share its phases.
            let (comp_thread, traf_thread) = if workers > 1.0 {
                let loss = 1.0 + WORKER_LOSS * (workers - 1.0);
                let packed_all_rows = packed
                    && cols >= 256
                    && cols.is_multiple_of(256)
                    && (conc_transforms <= 2
                        || 2 * single_prime_bytes > l3_budget / conc_transforms.max(1)
                        || reuse_b >= 2);
                let row_tasks = if (fits_l3_all && !packed) || packed_all_rows {
                    k_primes * rows
                } else {
                    rows
                };
                let min_pass_tasks = (cols / tile_cols.max(1)).min(row_tasks).max(1) as f64;
                let sync_mult = if rows > 1 && min_pass_tasks < 8.0 * workers {
                    let ratio = (8.0 * workers) / min_pass_tasks;
                    if !out_of_core { ratio * ratio } else { ratio }
                } else {
                    1.0
                };
                (
                    comp_thread * loss + workers * WORKER_SYNC_SEC * sync_mult * (tf / 3.0),
                    traf_thread * loss,
                )
            } else {
                (comp_thread, traf_thread)
            };

            let in_ram_comp_sec = comp_thread / workers / norm;
            let dram_traf_sec = traf_thread / workers / norm;
            let bf_equiv = (bf_ops + base_ops * 0.25 + cs_units * 0.5) / norm;

            let (comp_sec, traf_sec, eff_alpha) = if !out_of_core {
                let alpha = if extra_sweeps == 0 && doms_spanned == 1 {
                    1.0
                } else {
                    cal.overlap_alpha.max(0.75)
                };
                (in_ram_comp_sec, dram_traf_sec, alpha)
            } else if let Some(d) = cal.devices.first() {
                // Section 10: price the compute-covered transfer portion with `beside_sweep_rate_bps`
                // and `sweep_step_beside_sec`, and any uncovered remainder at `idle_rate_bps`.
                let beside_bps = d.beside_sweep_rate_bps.max(1e6);
                let idle_bps = d.idle_rate_bps.max(1e6);
                let beside_step_ratio =
                    (d.sweep_step_beside_sec / cal.sweep_step_sec.max(1e-9)).clamp(1.0, 4.0);
                let max_covered = (in_ram_comp_sec * beside_step_ratio) * beside_bps;
                let bm = store_bytes_moved as f64;
                let covered_bytes = bm.min(max_covered);
                let rem_bytes = bm - covered_bytes;
                let covered_frac = if max_covered > 0.0 {
                    (covered_bytes / max_covered).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let ooc_comp_sec =
                    in_ram_comp_sec * (1.0 + covered_frac * (beside_step_ratio - 1.0));
                let store_sec = (covered_bytes / beside_bps) + (rem_bytes / idle_bps);
                let check_pass_sec = (col_check_bytes as f64) / cal.ram_bw_bps.max(1e9);
                let dram_stage_sec =
                    dram_traf_sec.max(bm / (cal.ram_bw_bps * 2.0).max(1e9)) + check_pass_sec;
                (ooc_comp_sec, store_sec + dram_stage_sec, cal.overlap_alpha)
            } else {
                let fallback_store_sec =
                    (store_bytes_moved as f64) / (cal.ram_bw_bps * 0.1).max(1e6);
                let check_pass_sec = (col_check_bytes as f64) / cal.ram_bw_bps.max(1e9);
                (
                    in_ram_comp_sec,
                    dram_traf_sec + fallback_store_sec + check_pass_sec,
                    cal.overlap_alpha,
                )
            };

            let price = ProductPrice::combine(
                comp_sec,
                traf_sec,
                eff_alpha,
                bytes_moved,
                bf_equiv,
            );
            let spec = NttSpec {
                n,
                rows,
                cols,
                tile_cols,
                extra_sweeps,
                k_primes,
                coeff_bits: b as u32,
                packed,
                use_nt: cal.nt_effective && rows > 1,
                negacyclic,
                wrap_cyclic: false,
                out_of_core,
                par_workers: (workers as usize).max(1),
            };

            if let Some(v) = all.as_deref_mut() {
                v.push((spec, price));
            }

            let no_store_backing = shares.total_vram == 0 && shares.total_ssd == 0;
            let better = match &best {
                None => true,
                Some((bs, bp)) => {
                    if no_store_backing && bs.out_of_core != spec.out_of_core {
                        !spec.out_of_core
                    } else {
                        price.total_sec < bp.total_sec
                    }
                }
            };
            if better {
                best = Some((spec, price));
            }
        }

        // Once `k_primes <= min_k && b <= 32`, larger `n` only increases transform length and cost monotonically.
        if k_primes <= min_k && b <= 32 && best.is_some() {
            break;
        }
    }

    best
}

/// Size `(rows, cols, tile_cols, extra_sweeps)` to worker L2 and L3 shares and out-of-core
/// Store range batch bounds (Section 5, 6, & 10).
#[allow(clippy::too_many_arguments)]
pub fn choose_grid_shape(
    n: usize,
    l2_share: u64,
    l3_share: u64,
    nt_effective: bool,
    par_workers: usize,
    k_primes: usize,
    coeff_bits: u64,
    fits_ram_1d: bool,
    extra_sum_bit: bool,
    ram_transform_budget: u64,
    min_store_range_bytes: u64,
) -> (usize, usize, usize, u32) {
    let l2 = l2_share.max(4096);
    let l3 = l3_share.max(l2);
    let pw = par_workers.max(1);
    // Keep 1D `rows == 1` strictly when `par_workers <= 1` and the worker's 1D transform
    // working set (`16 * n` bytes across plane and twiddles, matching `tier::ntt_l2_sub_n`)
    // fits within its L2 cache share (Section 6), or for tiny `n <= 1024`!
    if fits_ram_1d && pw <= 1 && (n as u64) * 16 <= l2.min(l3) {
        return (1, n, n, 0);
    }
    if fits_ram_1d && n <= 1024 && (n as u64) * 24 <= l3 && (n as u64) * 16 <= l2 {
        return (1, n, n, 0);
    }

    // Sweep 2 per-row 1D transform + twiddles (`16 * cols` bytes) fits comfortably in `l2`
    // (`max_cols_l2 = 16,384` on `512 KiB` L2 to prevent SMT L2 thrashing at small `rows`),
    // while `while rows > pref_rows_l2` scales `cols` up to `32,768` (`16 * cols <= l2`) for `n >= 2M`.
    let row_l3_bpe: u64 = if extra_sum_bit { 40 } else { 24 };
    let max_cols_l2 = (((l2 / 16).max(8) as usize).next_power_of_two() >> 1).max(8);

    // Sweep 1/3 column-tile working set (`rows * tile_cols * 8`, using streaming NT stores when
    // `nt_effective`) fits `l2`, keeping 32-lane vectorized split and 50-bit packing up to `rows = 2048`.
    let bytes_per_elem_l2 = if nt_effective { 12u64 } else { 20u64 };
    let max_tile_elems = if nt_effective && l2 >= 256 * 1024 {
        (l2 / 8) as usize
    } else {
        (((l2 / bytes_per_elem_l2).max(16) as usize).next_power_of_two() >> 1).max(8)
    };

    let min_par_rows_raw = if pw >= 4 && n >= pw.saturating_mul(4 * 512) {
        let max_r_for_sw1 = crate::tier::largest_pow2_le((n / (pw * 512)) as u64).max(16) as usize;
        (pw.next_power_of_two() * 4).min(max_r_for_sw1).min(64)
    } else if pw >= 2 && n >= pw.saturating_mul(2 * 512) {
        let max_r_for_sw1 = crate::tier::largest_pow2_le((n / (pw * 512)) as u64).max(16) as usize;
        (pw.next_power_of_two() * 2).max(16).min(max_r_for_sw1).min(64)
    } else {
        16
    };
    let min_par_rows = crate::tier::largest_pow2_le(min_par_rows_raw as u64).max(2) as usize;
    let raw_cols = max_cols_l2.min((n / min_par_rows).max(8)).min(n / 2).max(8);
    let mut cols = crate::tier::largest_pow2_le(raw_cols as u64).max(8) as usize;
    if cols >= n {
        return (1, n, n, 0);
    }
    let mut rows = n / cols;
    debug_assert!(rows.is_power_of_two() && cols.is_power_of_two() && rows * cols == n);
    let mut extra_sweeps = 0u32;

    // Prefer `tile_cols >= 32` and bound `rows` by L2 segment capacity and L2 dTLB reach
    // (each Sweep 1/3 gather/scatter touches `rows` 4 KiB pages at a time), growing `cols` up to
    // `16 * cols <= l2` (`32,768` on `512 KiB` L2), up to `4 * l2` (`cols <= 131,072`) when `rows > 1024`,
    // and up to `row_l3_bpe * cols <= l3` when `rows > 2048`!
    let _ = coeff_bits;
    let dtlb_rows_cap = if l2 >= 256 * 1024 {
        2048usize
    } else {
        1usize << 1024usize.min((3072 / 2).max(2)).ilog2()
    };
    let max_rows_tile = (max_tile_elems / 32).max(2).min(dtlb_rows_cap);
    let pref_rows_l2 = max_rows_tile.min(64);
    while rows > pref_rows_l2
        && ((cols * 2) as u64) * 16 <= l2
        && ((cols * 2) as u64) * row_l3_bpe <= l3
        && cols * 2 < n
    {
        rows >>= 1;
        cols <<= 1;
    }
    let l2_tile_rows_cap = ((l2 / (32 * 8)).max(2) as usize).min(1024);
    while rows > max_rows_tile.min(l2_tile_rows_cap)
        && ((cols * 2) as u64) * 16 <= l2.saturating_mul(8)
        && ((cols * 2) as u64) * row_l3_bpe <= l3
        && cols * 2 < n
    {
        rows >>= 1;
        cols <<= 1;
    }
    // Once `cols` has reached the single-pass L2 bound (`16 * cols >= l2`, entering L2
    // sub-blocking in Sweep 2), prefer `rows <= 512` while `16 * cols <= 2 * l2` (`cols <= 65,536`
    // on `512 KiB` L2, fitting physical-core L2 while halving `rows` in Sweeps 1 & 3).
    let l2_subblock_rows_cap = ((l2 / (64 * 8)).max(2) as usize).min(512);
    while (cols as u64) * 16 >= l2
        && rows > max_rows_tile.min(l2_subblock_rows_cap)
        && ((cols * 2) as u64) * 16 <= l2.saturating_mul(2)
        && ((cols * 2) as u64) * row_l3_bpe <= l3
        && cols * 2 < n
    {
        rows >>= 1;
        cols <<= 1;
    }
    while rows > max_rows_tile.min(2048)
        && ((cols * 2) as u64) * row_l3_bpe <= l3
        && cols * 2 < n
    {
        rows >>= 1;
        cols <<= 1;
    }

    // When `rows` itself exceeds `max_tile_elems / 8` (tiny cache shares in Gate 2!), cap `rows`
    // so at least an 8-column tile fits L2 and add `extra_sweeps` inside `cols` (Section 5)!
    let max_rows_l2 = (max_tile_elems / 8).max(2);
    while rows > max_rows_l2 && cols * 2 < n {
        rows >>= 1;
        cols <<= 1;
    }

    // Section 10: in out-of-core sweeps, column sweeps touch `Store` in batches of strided ranges,
    // each range at least `min_store_range_bytes`. Where one RAM tile of the store is too small for
    // one such range from every row, shrink outer `rows` (growing `cols`) so `cols` decomposes into
    // inner column sweeps as Section 5 describes!
    let min_grid_bytes_packed =
        ((k_primes as u64).saturating_mul(n as u64).saturating_mul(50)) / 8;
    let is_ooc = min_store_range_bytes > 0
        && min_grid_bytes_packed > ram_transform_budget.max(64 * 1024);
    let tile_budget = (ram_transform_budget / 2).max(min_store_range_bytes);
    let mut ooc_tile_too_small = false;
    if is_ooc && (rows as u64).saturating_mul(min_store_range_bytes) > tile_budget {
        ooc_tile_too_small = true;
        while rows > 2
            && cols * 2 < n
            && (rows as u64).saturating_mul(min_store_range_bytes) > tile_budget
        {
            rows >>= 1;
            cols <<= 1;
        }
    }

    let eff_row_cap = if ooc_tile_too_small {
        l3.min(tile_budget.max(4096))
    } else {
        l3
    };
    let mut rem_cols = cols;
    while (rem_cols as u64) * row_l3_bpe > eff_row_cap && rem_cols >= 16 {
        extra_sweeps += 1;
        let log_c = rem_cols.ilog2();
        rem_cols >>= (log_c / 2).max(1);
    }
    if ooc_tile_too_small && extra_sweeps == 0 && cols >= 16 {
        extra_sweeps = 1;
    }

    // Size Sweep 1 `tile_cols` (`32` for `rows >= 512`, `64` for `rows <= 256` where `rows * 64 * 8`
    // stays within a quarter of L2, keeping 64-lane blocks and 50-bit pack active!).
    let tc_cap = if rows >= 128 && cols >= pw.saturating_mul(32) {
        32
    } else {
        64
    };
    let par_cap_tc = if cols >= pw.saturating_mul(tc_cap) {
        (cols / pw).max(tc_cap)
    } else if cols >= pw.saturating_mul(8) {
        (cols / pw).max(8)
    } else {
        cols
    };
    let raw_tc = (max_tile_elems / rows)
        .min(par_cap_tc)
        .min(tc_cap)
        .max(8)
        .min(cols);
    let tile_cols = (1usize << raw_tc.ilog2()).min(cols).max(1);
    (rows, cols, tile_cols, extra_sweeps)
}
