//! Gate 4 — Faster (`DESIGN.md` Section 13 & Section 15.4):
//! - Deterministic noise-free `Plan` evaluation on the three canonical profile classes:
//!   compute-bound, traffic-bound, and out-of-core (including overlap split between
//!   `beside_sweep_rate_bps` and `idle_rate_bps`).
//! - Exact deterministic accounting of kernel-table work (`WorkSnapshot`), memory/store bytes
//!   moved, and segmented-sieve operations (`sieve_ops`) under supplied `Profile` + `Calibration`.
//! - Isolated kernel timing confirming bound `x86-64-v4` assembly kernels strictly outperform the
//!   portable reference kernels beyond timing spread and lower recomputed planned time.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use engine::plan::{Calibration, DeviceCalibration, OutputFormat, PaceBound, Plan, PlanOverrides};
use engine::radix::DigitSink;
use engine::refk::{KernelTable, Kernels, NttPrime};
use engine::store::{ExtentMember, MemberKind, TestMember};
use engine::tier::{Profile, Shares};
use engine::{EngineConfig, OutputSinkFactory, run};

struct VecSink {
    buf: Mutex<Vec<u8>>,
}

impl DigitSink for VecSink {
    fn write_at(&self, offset: u64, bytes: &[u8]) -> Result<(), String> {
        let mut g = self.buf.lock().unwrap();
        let off = offset as usize;
        g[off..(off + bytes.len())].copy_from_slice(bytes);
        Ok(())
    }
    fn flush(&self) -> Result<(), String> {
        Ok(())
    }
}

struct VecSinkFactory;

impl OutputSinkFactory for VecSinkFactory {
    fn create_full_size(&self, full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        Ok(Box::new(VecSink {
            buf: Mutex::new(vec![0u8; full_size_bytes as usize]),
        }))
    }
    fn open_existing_exact(&self, expected_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        Ok(Box::new(VecSink {
            buf: Mutex::new(vec![0u8; expected_bytes as usize]),
        }))
    }
}

#[test]
fn gate4_deterministic_planned_time_across_compute_traffic_and_ooc_profiles() {
    let digits = 1_000_000_000u64; // 10^9 digits

    // 1. Compute-bound profile: modest worker butterfly rate, very high RAM bandwidth.
    let comp_prof = Profile::synthetic(
        16,
        16,
        48 * 1024,
        1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let comp_shares = Shares::from_profile(&comp_prof);
    let mut comp_cal = Calibration::synthetic(1.5e9, 1.6e12, true, Vec::new());
    comp_cal.overlap_alpha = 0.0;

    // 2. Traffic-bound profile: many fast cores, constrained memory bandwidth.
    let traf_prof = Profile::synthetic(
        96,
        96,
        48 * 1024,
        1024 * 1024,
        128 * 1024 * 1024,
        128 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let traf_shares = Shares::from_profile(&traf_prof);
    let mut traf_cal = Calibration::synthetic(8.0e9, 40.0e9, true, Vec::new());
    traf_cal.overlap_alpha = 0.0;

    // 3. Out-of-core profile: 64 MiB RAM + VRAM/SSD store devices with distinct `idle` & `beside` rates.
    let ooc_prof = Profile::synthetic(
        16,
        16,
        48 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024, // 64 MiB RAM forces 10^9 digits out of core!
        &[16 * 1024 * 1024 * 1024],
        &[64 * 1024 * 1024 * 1024],
    );
    let ooc_shares = Shares::from_profile(&ooc_prof);
    let ooc_devs = vec![
        DeviceCalibration {
            name: "vram0".to_string(),
            idle_rate_bps: 28.0e9,
            beside_sweep_rate_bps: 22.0e9,
            sweep_step_beside_sec: 1.0e-4,
            min_batch_range_bytes: 262_144,
        },
        DeviceCalibration {
            name: "ssd0".to_string(),
            idle_rate_bps: 7.0e9,
            beside_sweep_rate_bps: 5.8e9,
            sweep_step_beside_sec: 1.0e-4,
            min_batch_range_bytes: 262_144,
        },
    ];
    let ooc_cal = Calibration::synthetic(3.5e9, 90.0e9, true, ooc_devs);

    let p_comp_a =
        Plan::build(digits, OutputFormat::Text, &comp_shares, &comp_cal, PlanOverrides::default())
            .unwrap();
    let p_comp_b =
        Plan::build(digits, OutputFormat::Text, &comp_shares, &comp_cal, PlanOverrides::default())
            .unwrap();

    let p_traf_a =
        Plan::build(digits, OutputFormat::Text, &traf_shares, &traf_cal, PlanOverrides::default())
            .unwrap();
    let p_traf_b =
        Plan::build(digits, OutputFormat::Text, &traf_shares, &traf_cal, PlanOverrides::default())
            .unwrap();

    let p_ooc_a =
        Plan::build(digits, OutputFormat::Text, &ooc_shares, &ooc_cal, PlanOverrides::default())
            .unwrap();
    let p_ooc_b =
        Plan::build(digits, OutputFormat::Text, &ooc_shares, &ooc_cal, PlanOverrides::default())
            .unwrap();

    // Verify canonical profile regimes & non-regression baseline ceilings (Section 15.4)!
    assert_eq!(p_comp_a.pace_bound, PaceBound::ComputeBound);
    assert!(!p_comp_a.out_of_core);
    assert_eq!(p_traf_a.pace_bound, PaceBound::TrafficBound);
    assert!(!p_traf_a.out_of_core);
    assert!(p_ooc_a.out_of_core);
    assert!(!p_comp_a.dividing_levels.is_empty());
    assert!(
        p_comp_a.planned_total_sec > 0.0 && p_comp_a.planned_total_sec < 120.0,
        "Compute-bound 10^9 baseline regression: {}s",
        p_comp_a.planned_total_sec
    );
    assert!(
        p_traf_a.planned_total_sec > 0.0 && p_traf_a.planned_total_sec < 120.0,
        "Traffic-bound 10^9 baseline regression: {}s",
        p_traf_a.planned_total_sec
    );
    assert!(
        p_ooc_a.planned_total_sec > 0.0 && p_ooc_a.planned_total_sec < 600.0,
        "Out-of-core 10^9 baseline regression: {}s",
        p_ooc_a.planned_total_sec
    );

    // Bit-for-bit noise-free determinism!
    for (pa, pb) in [(&p_comp_a, &p_comp_b), (&p_traf_a, &p_traf_b), (&p_ooc_a, &p_ooc_b)] {
        assert_eq!(pa.planned_total_sec.to_bits(), pb.planned_total_sec.to_bits());
        assert_eq!(
            pa.planned_compute_sec.to_bits(),
            pb.planned_compute_sec.to_bits()
        );
        assert_eq!(
            pa.planned_traffic_sec.to_bits(),
            pb.planned_traffic_sec.to_bits()
        );
        assert_eq!(pa.planned_bytes_moved, pb.planned_bytes_moved);
        assert_eq!(
            pa.planned_butterflies.to_bits(),
            pb.planned_butterflies.to_bits()
        );
        assert_eq!(pa.term_count, pb.term_count);
        assert_eq!(pa.tree_height, pb.tree_height);
        assert_eq!(pa.leaf_len, pb.leaf_len);
        assert_eq!(pa.dividing_levels.len(), pb.dividing_levels.len());
    }

    // Sensitivity check 1: Faster kernel bodies strictly lower compute-bound planned time and raise none!
    let mut faster_comp_cal = comp_cal.clone();
    faster_comp_cal.ntt_bf_per_sec_per_worker *= 1.5;
    faster_comp_cal.fft_bf_per_sec_per_worker *= 1.5;
    faster_comp_cal.crt_ops_per_sec_per_worker *= 1.5;
    let p_comp_faster = Plan::build(
        digits,
        OutputFormat::Text,
        &comp_shares,
        &faster_comp_cal,
        PlanOverrides::default(),
    )
    .unwrap();
    assert!(
        p_comp_faster.planned_total_sec < p_comp_a.planned_total_sec,
        "Faster kernels must lower compute-bound planned time ({} vs {})",
        p_comp_faster.planned_total_sec,
        p_comp_a.planned_total_sec
    );

    // Sensitivity check 2: Higher memory bandwidth strictly lowers traffic-bound planned time!
    let mut faster_traf_cal = traf_cal.clone();
    faster_traf_cal.ram_bw_bps *= 1.5;
    let p_traf_faster = Plan::build(
        digits,
        OutputFormat::Text,
        &traf_shares,
        &faster_traf_cal,
        PlanOverrides::default(),
    )
    .unwrap();
    assert!(
        p_traf_faster.planned_total_sec < p_traf_a.planned_total_sec,
        "Higher RAM bandwidth must lower traffic-bound planned time ({} vs {})",
        p_traf_faster.planned_total_sec,
        p_traf_a.planned_total_sec
    );

    // Sensitivity check 3: Higher store `beside_sweep_rate_bps` & `idle_rate_bps` strictly lower out-of-core planned time!
    let mut faster_ooc_cal = ooc_cal.clone();
    for d in &mut faster_ooc_cal.devices {
        d.idle_rate_bps *= 2.0;
        d.beside_sweep_rate_bps *= 2.0;
    }
    let p_ooc_faster = Plan::build(
        digits,
        OutputFormat::Text,
        &ooc_shares,
        &faster_ooc_cal,
        PlanOverrides::default(),
    )
    .unwrap();
    assert!(
        p_ooc_faster.planned_total_sec < p_ooc_a.planned_total_sec,
        "Faster store transfer rates must lower out-of-core planned time ({} vs {})",
        p_ooc_faster.planned_total_sec,
        p_ooc_a.planned_total_sec
    );
}

#[test]
fn gate4_exact_work_and_byte_counters_at_kernel_and_store_seams() {
    // 1. In-memory deterministic counter check (with reduced splitting active so `sieve_ops > 0`).
    let inmem_prof = Profile::synthetic(
        2,
        2,
        16 * 1024,
        256 * 1024,
        4 * 1024 * 1024,
        64 * 1024 * 1024,
        &[],
        &[],
    );
    let inmem_cal = Calibration::synthetic(3.0e9, 50.0e9, true, Vec::new());

    let run_inmem_counted = || {
        let (mut kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
        kernels.count_work = true;
        let mut cfg = EngineConfig::new(
            20_000,
            OutputFormat::Text,
            "g4_inmem",
            inmem_prof.clone(),
            &kernels,
            &VecSinkFactory,
        );
        cfg.supplied_calibration = Some(inmem_cal.clone());
        cfg.overrides.force_dividing_levels =
            Some(vec![(2, engine::plan::DivForm::SingleProduct)]);
        run(&cfg).unwrap()
    };

    let r1 = run_inmem_counted();
    let r2 = run_inmem_counted();
    assert_eq!(r1.dec_hash_w0, r2.dec_hash_w0);
    assert_eq!(
        r1.work_stats, r2.work_stats,
        "Kernel-table work counts must be exact and deterministic!"
    );
    assert_eq!(
        r1.sieve_ops, r2.sieve_ops,
        "Segmented-sieve op counts must be exact and deterministic!"
    );
    assert_eq!(r1.store_bytes_moved, 0);
    assert!(r1.sieve_ops > 0, "Expected non-zero sieve ops at 20,000 digits");
    assert!(r1.work_stats.limb_ops > 0);
    assert!(r1.work_stats.fft_butterflies + r1.work_stats.ntt_butterflies > 0);
    assert!(r1.work_stats.bytes_moved > 0);
    assert!(r1.work_stats.check_limbs > 0);
    assert!(r1.work_stats.digit_words > 0);

    // 2. Out-of-core deterministic counter check (verifying exact `store_bytes_moved > 0` at extent seam).
    let ooc_prof = Profile::synthetic(
        2,
        2,
        4 * 1024,
        4 * 1024,
        0,
        4 * 1024,
        &[8 * 1024 * 1024],
        &[],
    );
    let mut ooc_cal = Calibration::synthetic(2.5e9, 40.0e9, true, Vec::new());
    ooc_cal.basecase_max_limbs = 8;
    ooc_cal.karatsuba_max_limbs = 16;

    let run_ooc_counted = || {
        let (mut kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
        kernels.count_work = true;
        let vram = Arc::new(TestMember::new("vram0", MemberKind::Vram, 8 * 1024 * 1024));
        let mut rates = HashMap::new();
        rates.insert("vram0".to_string(), 30.0e9);
        let mut cfg = EngineConfig::new(
            25_000,
            OutputFormat::Text,
            "g4_ooc",
            ooc_prof.clone(),
            &kernels,
            &VecSinkFactory,
        );
        cfg.supplied_calibration = Some(ooc_cal.clone());
        cfg.supplied_member_rates = Some(rates);
        cfg.members = vec![vram as Arc<dyn ExtentMember>];
        run(&cfg).unwrap()
    };

    let o1 = run_ooc_counted();
    let o2 = run_ooc_counted();
    assert_eq!(o1.dec_hash_w0, o2.dec_hash_w0);
    assert!(
        o1.store_bytes_moved > 0,
        "Expected positive store_bytes_moved on out-of-core profile"
    );
    assert_eq!(
        o1.store_bytes_moved, o2.store_bytes_moved,
        "Extent store byte counts must be exact and deterministic!"
    );
    assert!(
        o1.store_loader_batches > 0 && o1.store_parked_peak_bytes > 0 && o1.store_parked_live_bytes == 0,
        "Out-of-core run must drive overlapped Loader batches ({}) and park/free live numbers (peak={}, live={})",
        o1.store_loader_batches,
        o1.store_parked_peak_bytes,
        o1.store_parked_live_bytes
    );
    assert_eq!(o1.store_loader_batches, o2.store_loader_batches);
    assert_eq!(o1.store_parked_peak_bytes, o2.store_parked_peak_bytes);
    assert_eq!(o1.work_stats, o2.work_stats);
    assert_eq!(o1.sieve_ops, o2.sieve_ops);
}

#[test]
fn gate4_isolated_assembly_vs_reference_kernel_speedup() {
    let (asm_k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_count(), 24);
    let ref_t = KernelTable::reference();

    // Benchmark 1: 4096-point `ntt_fwd` + `ntt_inv` in L1/L2 (median of 7 batches).
    let n = 4096usize;
    let p_int = (159u64 << 42) + 1;
    let pr = NttPrime::new(p_int, 1);
    let tw = vec![1.0f64; n];
    let mut buf_asm = vec![42.0f64; n];
    let mut buf_ref = vec![42.0f64; n];

    let mut asm_times = Vec::with_capacity(7);
    let mut ref_times = Vec::with_capacity(7);
    for _ in 0..7 {
        let t0 = Instant::now();
        for _ in 0..64 {
            unsafe {
                (asm_k.active.ntt_fwd)(buf_asm.as_mut_ptr(), tw.as_ptr(), n as u64, &pr);
                (asm_k.active.ntt_inv)(buf_asm.as_mut_ptr(), tw.as_ptr(), n as u64, &pr);
            }
        }
        std::hint::black_box(buf_asm[0]);
        asm_times.push(t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        for _ in 0..64 {
            unsafe {
                (ref_t.ntt_fwd)(buf_ref.as_mut_ptr(), tw.as_ptr(), n as u64, &pr);
                (ref_t.ntt_inv)(buf_ref.as_mut_ptr(), tw.as_ptr(), n as u64, &pr);
            }
        }
        std::hint::black_box(buf_ref[0]);
        ref_times.push(t1.elapsed().as_secs_f64());
    }
    asm_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ref_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ntt_asm_med = asm_times[3];
    let ntt_ref_med = ref_times[3];
    assert!(
        ntt_asm_med * 1.5 < ntt_ref_med,
        "x86-64-v4 ntt_fwd/inv must beat portable ref well beyond spread (asm={ntt_asm_med:.6}s, ref={ntt_ref_med:.6}s)"
    );

    // Benchmark 2: `check_n` (`mod 2^61 - 1` + two locating primes) on 8,192 limbs.
    let limbs: Vec<u64> = (0..8192u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut out_w = [0u64; 3];
    let tc_a = Instant::now();
    for _ in 0..128 {
        unsafe {
            (asm_k.active.check_n)(limbs.as_ptr(), limbs.len() as u64, &mut out_w, 3);
        }
        std::hint::black_box(out_w);
    }
    let check_asm_sec = tc_a.elapsed().as_secs_f64();

    let tc_r = Instant::now();
    for _ in 0..128 {
        unsafe {
            (ref_t.check_n)(limbs.as_ptr(), limbs.len() as u64, &mut out_w, 3);
        }
        std::hint::black_box(out_w);
    }
    let check_ref_sec = tc_r.elapsed().as_secs_f64();
    assert!(
        check_asm_sec < check_ref_sec,
        "x86-64-v4 check_n must beat portable ref (asm={check_asm_sec:.6}s, ref={check_ref_sec:.6}s)"
    );

    // Plug the live measured assembly vs reference butterfly rates into `Plan::build` and confirm
    // the bound assembly kernels strictly lower recomputed planned time (Section 15.4)!
    let bf_batch = 64.0 * 2.0 * 0.5 * (n as f64) * (n.ilog2() as f64);
    let asm_bf_rate = bf_batch / ntt_asm_med.max(1e-9);
    let ref_bf_rate = bf_batch / ntt_ref_med.max(1e-9);
    let prof = Profile::synthetic(
        8,
        8,
        48 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let shares = Shares::from_profile(&prof);
    let cal_asm = Calibration::synthetic(asm_bf_rate, 200.0e9, true, Vec::new());
    let cal_ref = Calibration::synthetic(ref_bf_rate, 200.0e9, true, Vec::new());
    let plan_asm = Plan::build(100_000_000, OutputFormat::Text, &shares, &cal_asm, PlanOverrides::default()).unwrap();
    let plan_ref = Plan::build(100_000_000, OutputFormat::Text, &shares, &cal_ref, PlanOverrides::default()).unwrap();
    assert!(
        plan_asm.planned_total_sec < plan_ref.planned_total_sec,
        "Measured assembly kernel throughput must strictly lower recomputed planned time ({}s vs {}s)",
        plan_asm.planned_total_sec,
        plan_ref.planned_total_sec
    );
}

#[test]
fn gate4_workspace_cargo_profiles_enforce_section_12_build_settings() {
    let ws_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let root_toml = std::fs::read_to_string(ws_root.join("Cargo.toml")).expect("read root Cargo.toml");

    // Section 12: release profile optimization, whole-program LTO, single codegen unit, panic=abort;
    // test & dev profiles keep integer overflow checks and debug assertions enabled.
    for required in [
        "[profile.release]",
        "opt-level = 3",
        "lto = \"fat\"",
        "codegen-units = 1",
        "panic = \"abort\"",
        "[profile.test]",
        "[profile.dev]",
        "overflow-checks = true",
        "debug-assertions = true",
    ] {
        assert!(
            root_toml.contains(required),
            "Root Cargo.toml missing required Section 12 setting `{required}`"
        );
    }

    // Section 7 & 12: zero external crates across all workspace manifests, and plain-arch Rust build.
    assert!(
        !ws_root.join(".cargo").join("config.toml").exists(),
        "Rust crates must build for the plain architecture without .cargo/config.toml target-cpu overrides"
    );
    for member in ["engine", "kernels", "platform", "cli"] {
        let m_toml = std::fs::read_to_string(ws_root.join(member).join("Cargo.toml"))
            .expect("read member Cargo.toml");
        let mut in_deps = false;
        for line in m_toml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_deps = trimmed.contains("dependencies");
                continue;
            }
            if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            assert!(
                trimmed.contains("path ="),
                "External crate dependency forbidden by Section 7 in {member}/Cargo.toml: `{trimmed}`"
            );
        }
    }
}

#[test]
fn gate4_calibration_persistence_24_kernels_core_reranking_and_live_store_probe() {
    use engine::par::{Scheduler, ThreadPlacer};
    use engine::plan_only;
    use engine::refk::NUM_KERNELS;
    use engine::store::{Store, form_stripes};

    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let prof = Profile::synthetic(
        4,
        8,
        32 * 1024,
        512 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        &[],
        &[],
    );

    // 1. Static calibration persistence & reuse, invalidation on build/kernel/profile change,
    //    and MOD_W0 corruption rejection.
    let cache_path = std::env::temp_dir().join(format!(
        "supi_gate4_cal_test_{}.bin",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&cache_path);

    let digits = 10_000u64;
    let mut cfg1 = EngineConfig::new(digits, OutputFormat::Text, "cal_cold", prof.clone(), &kernels, &VecSinkFactory);
    cfg1.calibration_cache_path = Some(cache_path.clone());
    cfg1.build_id = 0xCAFE_1111_2222_3333;
    let rep1 = run(&cfg1).expect("cold run saves static calibration");
    assert!(!rep1.calibration_reused, "Cold run must measure static calibration fresh");
    assert!(cache_path.is_file(), "Static calibration file must be persisted");

    let (_, cal_warm, _) = plan_only(&cfg1).expect("warm plan_only reuses static calibration");
    assert!(cal_warm.reused_static, "Matching profile + build_id + bound_mask must reuse static calibration");
    assert_eq!(cal_warm.kernel_rates.len(), NUM_KERNELS);
    for (idx, &rate) in cal_warm.kernel_rates.iter().enumerate() {
        assert!(
            rate.is_finite() && rate >= 1.0e6,
            "Kernel #{idx} ({}) rate must be measured positive: {rate}",
            engine::refk::KERNEL_NAMES[idx]
        );
    }
    assert!(
        cal_warm.ram_bw_bps > 0.0
            && cal_warm.sweep_step_sec > 0.0
            && (0.05..=1.0).contains(&cal_warm.overlap_alpha),
        "Dynamic startup figures (ram_bw_bps, sweep_step_sec, overlap_alpha) must be measured fresh"
    );

    let rep2 = run(&cfg1).expect("warm run reuses static calibration");
    assert!(rep2.calibration_reused, "Warm run must reuse persisted static calibration");
    assert_eq!(rep2.dec_hash_w0, rep1.dec_hash_w0);

    // Changing `build_id` must invalidate persisted static calibration.
    let mut cfg_new_build = EngineConfig::new(digits, OutputFormat::Text, "cal_nb", prof.clone(), &kernels, &VecSinkFactory);
    cfg_new_build.calibration_cache_path = Some(cache_path.clone());
    cfg_new_build.build_id = 0xCAFE_9999_8888_7777;
    let (_, cal_nb, _) = plan_only(&cfg_new_build).unwrap();
    assert!(!cal_nb.reused_static, "Changed build_id must invalidate cached static calibration");

    // Corrupting a byte in the persisted cache file must fail MOD_W0 verification cleanly.
    let mut raw = std::fs::read(&cache_path).unwrap();
    raw[16] ^= 0x40;
    std::fs::write(&cache_path, &raw).unwrap();
    let (_, cal_corrupt, _) = plan_only(&cfg_new_build).unwrap();
    assert!(!cal_corrupt.reused_static, "Corrupted calibration cache must be rejected by w0 check word");
    let _ = std::fs::remove_file(&cache_path);

    // 2. Heterogeneous core `ntt_fwd` rates must re-rank `Shares` AND `Scheduler` worker placement
    //    strictly fastest-core-first!
    struct RecordingPlacer {
        first_placed_tid: std::sync::atomic::AtomicU64,
    }
    static REC_PLACER: RecordingPlacer = RecordingPlacer {
        first_placed_tid: std::sync::atomic::AtomicU64::new(u64::MAX),
    };
    impl ThreadPlacer for RecordingPlacer {
        fn place_current_thread(&self, hw_thread: u64, _os_places_threads: bool) {
            let _ = self.first_placed_tid.compare_exchange(
                u64::MAX,
                hw_thread,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }

    let mut het_cal = Calibration::synthetic(4.0e9, 100.0e9, true, Vec::new());
    // Core 3 is fastest (9G), then core 1 (5G), core 2 (3G), core 0 (1G).
    // On `Profile::synthetic(4, 8, ...)`, core `c` owns threads `[2*c, 2*c + 1]`, so core 3's
    // primary thread is `hw_thread == 6`.
    het_cal.core_ntt_bf_per_sec = vec![(0, 1.0e9), (1, 5.0e9), (2, 3.0e9), (3, 9.0e9)];

    let mut het_cfg = EngineConfig::new(5_000, OutputFormat::Text, "het_rank", prof.clone(), &kernels, &VecSinkFactory);
    het_cfg.supplied_calibration = Some(het_cal);
    het_cfg.thread_placer = Some(&REC_PLACER);
    let (het_shares, _, _) = plan_only(&het_cfg).unwrap();
    let top4_cores: Vec<u64> = het_shares.workers[..4].iter().map(|w| w.core_id).collect();
    assert_eq!(
        top4_cores,
        vec![3, 1, 2, 0],
        "Shares must rank primary threads of cores strictly fastest-first, got {top4_cores:?}"
    );
    REC_PLACER.first_placed_tid.store(u64::MAX, std::sync::atomic::Ordering::SeqCst);
    let _ = run(&het_cfg).expect("heterogeneous ranked run");
    assert_eq!(
        REC_PLACER.first_placed_tid.load(std::sync::atomic::Ordering::SeqCst),
        6,
        "Scheduler must place coordinator worker 0 on fastest core 3's primary hw_thread 6"
    );

    // 3. Live concurrent Store contention probe (`idle_rate_bps`, `min_batch_range_bytes`,
    //    `beside_sweep_rate_bps`, `sweep_step_beside_sec`).
    let shares = Shares::from_profile(&prof);
    let sched = Scheduler::new(&shares, None);
    let m0: Arc<dyn ExtentMember> = Arc::new(TestMember::new("live-ssd-0", MemberKind::Ssd, 4 * 1024 * 1024));
    let m1: Arc<dyn ExtentMember> = Arc::new(TestMember::new("live-ssd-1", MemberKind::Ssd, 4 * 1024 * 1024));
    let mut store = Store::new(form_stripes(vec![(m0, 2.0e9), (m1, 2.0e9)], 512));
    let live_cal = Calibration::measure(&kernels, &shares, &sched, Some(&mut store));
    assert_eq!(live_cal.devices.len(), 1);
    let dcal = &live_cal.devices[0];
    assert!(
        dcal.idle_rate_bps > 0.0
            && dcal.beside_sweep_rate_bps > 0.0
            && dcal.sweep_step_beside_sec >= live_cal.sweep_step_sec
            && dcal.min_batch_range_bytes >= 4096,
        "Live Store contention probe must populate positive idle/beside rates, beside step time, and batch range: {dcal:?}"
    );
}

#[test]
fn gate4_calibrated_sqr_base_crossover_and_l3_domain_ranking() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let prof = Profile::synthetic(
        2,
        4,
        48 * 1024,
        1024 * 1024,
        16 * 1024 * 1024,
        256 * 1024 * 1024,
        &[],
        &[],
    );
    let shares = Shares::from_profile(&prof);
    let sched = engine::par::Scheduler::new(&shares, None);
    let cache_path = std::env::temp_dir().join("supi_sec4_static_cal_test.bin");
    let _ = std::fs::remove_file(&cache_path);

    let cal_fresh = Calibration::measure_with_cache(
        &kernels,
        &prof,
        &shares,
        &sched,
        None,
        Some(&cache_path),
        0x5EC4_CA11_B001,
    );
    assert!(!cal_fresh.reused_static);
    assert_eq!(
        kernels.sqr_rows_max(),
        cal_fresh.sqr_base_rows_max,
        "Calibration::measure_with_cache must populate Kernels.sqr_base_rows_max"
    );
    assert_eq!(kernels.fft_sub_n(), shares.fft_l1_sub_n);

    // Reset atomics to u64::MAX and verify warm cache reload restores both crossovers and cache blocks!
    kernels
        .mul_base_rows_max
        .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    kernels
        .sqr_base_rows_max
        .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    kernels
        .fft_l1_sub_n
        .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);

    let cal_warm = Calibration::measure_with_cache(
        &kernels,
        &prof,
        &shares,
        &sched,
        None,
        Some(&cache_path),
        0x5EC4_CA11_B001,
    );
    let _ = std::fs::remove_file(&cache_path);
    assert!(cal_warm.reused_static);
    assert_eq!(cal_warm.mul_base_rows_max, cal_fresh.mul_base_rows_max);
    assert_eq!(cal_warm.sqr_base_rows_max, cal_fresh.sqr_base_rows_max);
    assert_eq!(kernels.sqr_rows_max(), cal_fresh.sqr_base_rows_max);
    assert_eq!(kernels.fft_sub_n(), shares.fft_l1_sub_n);

    // Multi-CCD L3 domain tie-breaking in `Shares::rank_by_throughput`:
    // 2 CCDs (`8` cores = `4` cores/CCD, `16` threads): CCD 1 is faster than CCD 0, and equal-speed
    // cores within each CCD stay contiguous by `l3_domain`.
    let ccd_prof = Profile::synthetic_multi_domain(
        8,
        16,
        2,
        48 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let mut ccd_shares = Shares::from_profile(&ccd_prof);
    let tps: Vec<f64> = ccd_shares
        .workers
        .iter()
        .map(|w| if w.l3_domain == 1 { 6.0e9 } else { 5.0e9 })
        .collect();
    ccd_shares.rank_by_throughput(&tps);
    // First 8 workers must be primary threads (`smt_slot == 0`), with all 4 cores of faster CCD 1 first!
    assert!(ccd_shares.workers[..8].iter().all(|w| w.smt_slot == 0));
    assert!(ccd_shares.workers[..4].iter().all(|w| w.l3_domain == 1));
    assert!(ccd_shares.workers[4..8].iter().all(|w| w.l3_domain == 0));
}

#[test]
fn gate4_section3_single_loader_no_helper_pools_and_column_check_pricing() {
    use engine::mul::MulMethod;
    use engine::plan::{DeviceCalibration, choose_product_method_and_price};

    let ws_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let store_src =
        std::fs::read_to_string(ws_root.join("engine").join("store.rs")).expect("read store.rs");
    let vram_src =
        std::fs::read_to_string(ws_root.join("platform").join("vram.rs")).expect("read vram.rs");

    // Section 6 & Section 10: one worker per hardware thread plus one `Loader` thread — zero extra
    // unpinned helper thread pools in `store.rs` or `vram.rs`, and direct `VK_EXT_external_memory_host`
    // import without a persistent 256 MiB bounce window.
    assert!(
        !store_src.contains("StoreWorkPool") && !store_src.contains("supi-store-"),
        "engine/store.rs must not spawn extra StoreWorkPool helper threads"
    );
    assert!(
        !vram_src.contains("VramCopyPool") && !vram_src.contains("supi-vram-copy"),
        "platform/vram.rs must not spawn extra VramCopyPool helper threads"
    );

    // Section 11: out-of-core column sweeps check each range in a pass of `check_n` before it
    // leaves RAM (Sweep 1) and as it arrives in RAM (Sweep 3), counted as RAM traffic in the plan.
    let prof = Profile::synthetic(
        4,
        4,
        32 * 1024,
        512 * 1024,
        8 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
        &[64 * 1024 * 1024 * 1024],
        &[],
    );
    let sh = Shares::from_profile(&prof);
    let dev_cal = vec![DeviceCalibration {
        name: "vram-0".to_string(),
        idle_rate_bps: 28.0e9,
        beside_sweep_rate_bps: 24.0e9,
        sweep_step_beside_sec: 1.0e-5,
        min_batch_range_bytes: 65_536,
    }];
    let cal = Calibration::synthetic(4.0e9, 120.0e9, true, dev_cal);
    let ov = PlanOverrides::default();

    let (m_ram, p_ram) = choose_product_method_and_price(
        8_000_000,
        8_000_000,
        false,
        false,
        4,
        &sh,
        &cal,
        64 * 1024 * 1024 * 1024,
        &ov,
    );
    let (m_ooc, p_ooc) = choose_product_method_and_price(
        8_000_000,
        8_000_000,
        false,
        false,
        4,
        &sh,
        &cal,
        256 * 1024,
        &ov,
    );
    assert!(matches!(m_ram, MulMethod::Ntt(s) if !s.out_of_core));
    assert!(matches!(m_ooc, MulMethod::Ntt(s) if s.out_of_core));
    assert!(
        p_ooc.bytes_moved > p_ram.bytes_moved,
        "Out-of-core NTT ProductPrice must include column-sweep check_n traffic ({} vs {})",
        p_ooc.bytes_moved,
        p_ram.bytes_moved
    );
}
