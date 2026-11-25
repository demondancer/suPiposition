//! Gate 2 (Section 15.2): Correct at every scale and across every row of Section 11's ladder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use engine::check::{
    CorrectionOutcome, DualCheck, Ladder, Telemetry, compute_limbs_all3, try_correct_single_limb,
};
use engine::fft::FftTableCache;
use engine::limb::BigNum;
use engine::mul::{MulCtx, MulFaultHook, MulMethod};
use engine::ntt::NttTableCache;
use engine::par::Scheduler;
use engine::plan::{
    Calibration, DeviceCalibration, DivForm, MergeForm, OutputFormat, Plan, PlanError,
    PlanOverrides, SplitFormOverride, choose_product_method_and_price, choose_wrapped_ntt_spec,
};
use engine::radix::{DigitSink, LeafFaultHook, MemoryDigitSink, run_conversion};
use engine::refk::{Kernels, MOD_W0, SUPI_ABI_VERSION};
use engine::store::{
    ByteRange, ExtentMember, MemberKind, Store, StoreError, TestFault, TestMember, form_stripes,
};
use engine::tier::{Profile, Shares};
use engine::{EngineConfig, EngineError, OutputSinkFactory, plan_only, run};

struct SharedSink(Arc<MemoryDigitSink>);
impl DigitSink for SharedSink {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        self.0.write_at(file_offset, bytes)
    }
    fn flush(&self) -> Result<(), String> {
        self.0.flush()
    }
}

struct MemFactory {
    sink: Arc<MemoryDigitSink>,
    fail_create: bool,
}

impl MemFactory {
    fn new(size: usize) -> Self {
        Self {
            sink: Arc::new(MemoryDigitSink::new(size)),
            fail_create: false,
        }
    }
}

impl OutputSinkFactory for MemFactory {
    fn create_full_size(&self, _full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        if self.fail_create {
            return Err("Disk full: cannot preallocate output file at full size".to_string());
        }
        Ok(Box::new(SharedSink(Arc::clone(&self.sink))))
    }
    fn open_existing_exact(&self, expected_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        let len = self.sink.buf.lock().unwrap().len() as u64;
        if len != expected_bytes {
            return Err(format!("Expected {expected_bytes} bytes, found {len}"));
        }
        Ok(Box::new(SharedSink(Arc::clone(&self.sink))))
    }
}

fn verify_method_no_61_multiple(m: &MulMethod) {
    match m {
        MulMethod::Basecase | MulMethod::Karatsuba { .. } => {}
        MulMethod::Fft { coeff_bits, .. } => {
            assert_ne!(coeff_bits % 61, 0, "FFT coeff_bits={coeff_bits} divisible by 61");
        }
        MulMethod::Ntt(spec) => {
            assert_ne!(
                spec.coeff_bits % 61,
                0,
                "NTT coeff_bits={} divisible by 61",
                spec.coeff_bits
            );
        }
        MulMethod::SplitHalving { inner, .. } => verify_method_no_61_multiple(inner),
        MulMethod::SplitOuter {
            part_bits_w,
            inner_ntt,
            ..
        } => {
            assert_ne!(
                part_bits_w % 61,
                0,
                "SplitOuter part_bits_w={part_bits_w} divisible by 61"
            );
            assert_ne!(inner_ntt.coeff_bits % 61, 0);
        }
    }
}

#[test]
fn gate2_synthetic_profiles_every_scale_and_no_61_multiples() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let cal = Calibration::synthetic(
        2.0e9,
        80.0e9,
        true,
        vec![DeviceCalibration {
            name: "stripe-0".to_string(),
            idle_rate_bps: 20.0e9,
            beside_sweep_rate_bps: 17.0e9,
            sweep_step_beside_sec: 1.2e-5,
            min_batch_range_bytes: 32768,
        }],
    );

    // 1. 1-core tiny cache & tiny RAM -> forces extra_sweeps > 0 and out_of_core Store!
    let tiny_prof = Profile::synthetic(
        1,
        1,
        4 * 1024,
        8 * 1024,
        0,
        2 * 1024 * 1024,
        &[64 * 1024 * 1024],
        &[128 * 1024 * 1024],
    );
    let tiny_shares = Shares::from_profile(&tiny_prof);
    let tiny_plan =
        Plan::build(500_000, OutputFormat::Text, &tiny_shares, &cal, PlanOverrides::default())
            .expect("tiny plan");
    assert!(tiny_plan.out_of_core, "tiny RAM must trigger out-of-core plan");

    let (m_tiny, _) = choose_product_method_and_price(
        8_000_000,
        8_000_000,
        false,
        true,
        1,
        &tiny_shares,
        &cal,
        256 * 1024,
        &PlanOverrides::default(),
    );
    if let MulMethod::Ntt(spec) = m_tiny {
        assert!(
            spec.extra_sweeps > 0,
            "tiny 8 KiB L2 share must add extra column sweeps, got {spec:?}"
        );
        assert!(spec.out_of_core);
    } else {
        panic!("Expected Ntt with extra_sweeps, got {m_tiny:?}");
    }

    // 2. Near-reference-machine profile (u64::MAX - 1 counts) & multi-trillion digit plan!
    let huge_prof = Profile::synthetic(
        u64::MAX - 1,
        u64::MAX - 1,
        1 << 20,
        1 << 24,
        1 << 34,
        u64::MAX - 1,
        &[u64::MAX / 4],
        &[u64::MAX / 4],
    );
    let huge_fac = MemFactory::new(16);
    let mut cfg_huge = EngineConfig::new(
        10_000_000_000_000, // 10^13 digits
        OutputFormat::Packed19,
        "huge.pk19",
        huge_prof,
        &kernels,
        &huge_fac,
    );
    cfg_huge.supplied_calibration = Some(cal.clone());
    let (huge_shares, _, huge_plan) = plan_only(&cfg_huge).expect("10^13 digit plan near 2^64 machine");
    assert_eq!(huge_shares.cores, u64::MAX - 1);
    assert!(huge_plan.planned_total_sec.is_finite() && huge_plan.planned_total_sec > 0.0);

    // 3. Confirm no plan ever cuts a coefficient or outer-transform part at a multiple of 61 bits!
    let std_shares = Shares::from_profile(&Profile::synthetic(
        8,
        16,
        32 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
        &[],
        &[],
    ));
    let mut ov_short = PlanOverrides::default();
    ov_short.max_ntt_primes = 2; // forces SplitHalving / SplitOuter on modest bit counts too!

    for exp in 10..=52u32 {
        let bits = (1u64 << exp) + 61 * 7;
        for exact_only in [false, true] {
            for ov in [&PlanOverrides::default(), &ov_short] {
                let (m, _) = choose_product_method_and_price(
                    bits,
                    bits,
                    false,
                    exact_only,
                    16,
                    &std_shares,
                    &cal,
                    16 * 1024 * 1024 * 1024,
                    ov,
                );
                verify_method_no_61_multiple(&m);
            }
        }
        if exp <= 38 {
            let w_spec = choose_wrapped_ntt_spec(
                bits,
                bits * 2,
                16,
                &std_shares,
                &cal,
                16 * 1024 * 1024 * 1024,
                18,
            );
            assert_ne!(w_spec.coeff_bits % 61, 0);
        }
    }
}

#[test]
fn gate2_forms_divisions_merges_splits_out_of_core_and_checkpoints() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let digits = 4_000u64;
    let out_len = (digits as usize) + 2;

    // Reference baseline run
    let base_prof = Profile::synthetic(4, 4, 32 * 1024, 512 * 1024, 4 * 1024 * 1024, 256 * 1024 * 1024, &[], &[]);
    let base_fac = MemFactory::new(out_len);
    let base_cfg = EngineConfig::new(digits, OutputFormat::Text, "base", base_prof.clone(), &kernels, &base_fac);
    let base_rep = run(&base_cfg).expect("baseline");
    let expected_bytes = base_fac.sink.buf.lock().unwrap().clone();

    // 1. Multi-level dividing factors + SequentialHensel + SingleProduct + Separate & Fused merges
    for merge_form in [MergeForm::Separate, MergeForm::Fused] {
        let fac = MemFactory::new(out_len);
        let mut cfg = EngineConfig::new(digits, OutputFormat::Text, "forms", base_prof.clone(), &kernels, &fac);
        cfg.overrides.force_dividing_levels = Some(vec![
            (2, DivForm::SequentialHensel),
            (5, DivForm::SingleProduct), // spans 3 levels (5 - 2 = 3)!
        ]);
        cfg.overrides.force_merge_form = Some(merge_form);
        let rep = run(&cfg).expect("forced div & merge forms");
        assert_eq!(rep.dec_hash_w0, base_rep.dec_hash_w0);
        assert_eq!(*fac.sink.buf.lock().unwrap(), expected_bytes);
    }

    // 2. Both SplitHalving and SplitOuter forced on real pi run
    for split_ov in [
        SplitFormOverride::ForceHalving(1),
        SplitFormOverride::ForceOuter(4),
    ] {
        let fac = MemFactory::new(out_len);
        let mut cfg = EngineConfig::new(digits, OutputFormat::Text, "split", base_prof.clone(), &kernels, &fac);
        cfg.overrides.split_override = split_ov;
        let rep = run(&cfg).expect("forced split override");
        assert_eq!(rep.dec_hash_w0, base_rep.dec_hash_w0);
        assert_eq!(*fac.sink.buf.lock().unwrap(), expected_bytes);
    }

    // 3. Out-of-core Store + tiny cache + 0, 1, and 3 SSD TestMembers + frequent checkpoints
    for num_ssds in [0usize, 1, 3] {
        let ssd_cap = 8 * 1024 * 1024usize;
        let vram_cap = 4 * 1024 * 1024usize;
        let ssd_sizes: Vec<u64> = vec![ssd_cap as u64; num_ssds];
        let ooc_prof = Profile::synthetic(
            2,
            2,
            8 * 1024,
            16 * 1024,
            0,
            512 * 1024, // Tiny 512 KiB RAM forces Store usage!
            &[vram_cap as u64],
            &ssd_sizes,
        );
        let mut members: Vec<Arc<dyn ExtentMember>> = vec![Arc::new(TestMember::new(
            "vram-0",
            MemberKind::Vram,
            vram_cap,
        ))];
        for i in 0..num_ssds {
            members.push(Arc::new(TestMember::new(
                format!("ssd-{i}"),
                MemberKind::Ssd,
                ssd_cap,
            )));
        }
        let fac = MemFactory::new(out_len);
        let mut cfg = EngineConfig::new(digits, OutputFormat::Text, "ooc", ooc_prof, &kernels, &fac);
        cfg.members = members;
        cfg.overrides.checkpoint_mult = 0.0; // write checkpoints at every stage!
        let rep = run(&cfg).expect("ooc + ssd profile");
        assert_eq!(rep.dec_hash_w0, base_rep.dec_hash_w0);
        assert_eq!(*fac.sink.buf.lock().unwrap(), expected_bytes);
        if num_ssds == 0 {
            assert_eq!(rep.checkpoints_written, 0, "0 SSDs must write 0 checkpoints");
        } else {
            assert!(
                rep.checkpoints_written >= 3,
                "checkpoint_mult=0 with {num_ssds} SSDs must checkpoint Series+Finish+Convert, got {}",
                rep.checkpoints_written
            );
        }
    }
}

#[test]
fn gate2_stripes_with_unequal_sizes_and_remainders() {
    // Section 10 rule: fastest members left, largest count on rate ties, equal slice bytes from
    // smallest in stripe, and larger member's remainder returns to the pool!
    let m_fast_big: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("fast-big", MemberKind::Vram, 4 * 1024 * 1024));
    let m_fast_small: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("fast-small", MemberKind::Vram, 2 * 1024 * 1024));
    let m_slow: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("slow-ssd", MemberKind::Ssd, 2 * 1024 * 1024));

    let devices = form_stripes(
        vec![
            (m_fast_big, 20.0e9),
            (m_fast_small, 20.0e9),
            (m_slow, 4.0e9),
        ],
        512, // 4 KiB data + 24 B check -> 8 KiB aligned slot
    );

    // Expected:
    // 1. `fast-big` (20 GB/s, 4 MiB) + `fast-small` (20 GB/s, 2 MiB) -> 2 * 20 = 40 GB/s > 3 * 4 = 12 GB/s!
    //    Forms a 2-wide stripe of 2 MiB each (40 GB/s), leaving 2 MiB remainder on `fast-big`!
    // 2. Remainder of `fast-big` (20 GB/s, 2 MiB) alone -> 20 GB/s > 2 * 4 = 8 GB/s!
    // 3. `slow-ssd` (4 GB/s, 2 MiB) alone -> 4 GB/s!
    assert_eq!(devices.len(), 3, "Expected 3 ranked extent devices");
    assert_eq!(devices[0].stripe_width(), 2);
    assert_eq!(devices[1].stripe_width(), 1);
    assert_eq!(devices[1].member_ids(), vec!["fast-big".to_string()]);
    assert_eq!(devices[2].stripe_width(), 1);
    assert_eq!(devices[2].member_ids(), vec!["slow-ssd".to_string()]);
    assert!(devices[0].beside_sweep_rate_bps > devices[1].beside_sweep_rate_bps);
    assert!(devices[1].beside_sweep_rate_bps > devices[2].beside_sweep_rate_bps);
}

#[test]
fn gate2_ladder_every_row_of_section_11() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);

    // --- Row 1a: Single-limb locator & in-place Rung-1 correction vs 2-limb & 8-limb burst ---
    let mut limbs: Vec<u64> = (0..4096).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55).collect();
    let expected3 = compute_limbs_all3(&kernels, &limbs);
    let orig = limbs.clone();

    for idx in [0usize, 1, 77, 2048, 4095] {
        // Random replacement, single bit flip, and +-1 off
        for corrupted in [
            limbs[idx] ^ 0xDEAD_BEEF_CAFE_BABE,
            limbs[idx] ^ (1u64 << 37),
            limbs[idx].wrapping_add(1),
        ] {
            limbs[idx] = corrupted;
            let actual3 = compute_limbs_all3(&kernels, &limbs);
            let out = try_correct_single_limb(&mut limbs, actual3, expected3);
            assert_eq!(out, CorrectionOutcome::Corrected { limb_index: idx });
            assert_eq!(limbs, orig, "Rung 1 must restore exact limb at idx={idx}");
        }
    }

    // Two wrong limbs -> BeyondOneLimb
    limbs[10] ^= 1;
    limbs[2000] ^= 2;
    let act_two = compute_limbs_all3(&kernels, &limbs);
    assert_eq!(
        try_correct_single_limb(&mut limbs, act_two, expected3),
        CorrectionOutcome::BeyondOneLimb
    );
    limbs.copy_from_slice(&orig);

    // 8-limb burst -> BeyondOneLimb
    for j in 500..508 {
        limbs[j] ^= 0x1234_5678_9ABC_DEF0;
    }
    let act_burst = compute_limbs_all3(&kernels, &limbs);
    assert_eq!(
        try_correct_single_limb(&mut limbs, act_burst, expected3),
        CorrectionOutcome::BeyondOneLimb
    );

    // --- Row 3: DualCheck 2-of-3 repair at rest vs double corruption ---
    let tel = Telemetry::default();
    let mut dc = DualCheck::new(123456789);
    dc.c1 ^= 0xFF;
    assert!(dc.verify_or_repair_at_rest(123456789, &tel));
    assert_eq!(dc.c0, 123456789);
    assert_eq!(dc.c1, 123456789);
    assert_eq!(tel.corrections.load(Ordering::Relaxed), 1);

    dc.c0 = 111;
    dc.c1 = 222;
    assert!(
        !dc.verify_or_repair_at_rest(123456789, &tel),
        "Number agreeing with neither copy of w0 must fail at-rest check (Rung 3)"
    );

    // --- Row 11: Bad assembly kernel body at startup falls back to reference ---
    unsafe extern "C" fn bad_add_n(dst: *mut u64, a: *const u64, b: *const u64, n: u64) -> u64 {
        let c = unsafe { engine::refk::ref_add_n(dst, a, b, n) };
        if n > 0 {
            unsafe { *dst ^= 1 };
        }
        c
    }
    unsafe extern "C" fn bad_fft_pmul(dst: *mut f64, a: *const f64, b: *const f64, n: u64) {
        unsafe { engine::refk::ref_fft_pmul(dst, a, b, n) };
        if n > 0 {
            unsafe { *dst += 100.0 };
        }
    }
    unsafe extern "C" fn bind_with_two_bad(table: *mut usize, abi: u64) {
        unsafe {
            kernels::supi_bind(table, abi);
            if !table.is_null() && abi == SUPI_ABI_VERSION {
                *table.add(0) = bad_add_n as *const () as usize; // index 0: add_n
                *table.add(9) = bad_fft_pmul as *const () as usize; // index 9: fft_pmul (restores 6..12)
            }
        }
    }
    let (k_repaired, rep_bad) = Kernels::bind_and_verify(bind_with_two_bad);
    assert!(rep_bad.mismatched.contains(&"add_n"));
    assert!(rep_bad.mismatched.contains(&"fft_pmul"));
    assert_eq!(rep_bad.mismatched.len(), 7); // add_n + 6 FFT family kernels restored as a unit
    assert_eq!(k_repaired.bound_mask.count_ones(), 17);

    // Baseline 3,000-digit pi for end-to-end fault-injection runs
    let digits = 3_000u64;
    let out_len = (digits as usize) + 2;
    let prof = Profile::synthetic(2, 4, 32 * 1024, 512 * 1024, 4 * 1024 * 1024, 128 * 1024 * 1024, &[], &[]);
    let ref_fac = MemFactory::new(out_len);
    let _ref_rep = run(&EngineConfig::new(digits, OutputFormat::Text, "ref", prof.clone(), &kernels, &ref_fac)).unwrap();
    let expected_pi = ref_fac.sink.buf.lock().unwrap().clone();

    // --- Row 1b: Whole-product 1-limb fault -> Rung 1 in-place repair & 2-limb fault -> Rung 2 MakeAgain ---
    struct OneThenTwoLimbHook {
        hits: AtomicUsize,
    }
    impl MulFaultHook for OneThenTwoLimbHook {
        fn on_product_limbs(
            &self,
            point_id: u64,
            attempt: u32,
            limbs: &mut engine::limb::LimbVec,
            emitted_w0: &mut u64,
            _fft_err: &mut f64,
        ) {
            // Inject into whole Series/Finish transform products (`point_id < 0x5000_0000`)
            if point_id < 0x5000_0000 && attempt == 0 && limbs.len() >= 8 {
                let n = self.hits.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // Single-limb damage -> Rung 1 repair in place!
                    limbs[2] ^= 0xCAFE_F00D_1234_5678;
                    *emitted_w0 = (*emitted_w0 + 17) % MOD_W0;
                } else if n == 1 {
                    // Two-limb damage -> Rung 1 sees BeyondOneLimb, escalates to Rung 2 MakeAgain!
                    limbs[1] ^= 0x1111_2222;
                    limbs[5] ^= 0x3333_4444;
                    *emitted_w0 = (*emitted_w0 + 29) % MOD_W0;
                }
            }
        }
    }
    let hook_prod = OneThenTwoLimbHook {
        hits: AtomicUsize::new(0),
    };
    let fac_p = MemFactory::new(out_len);
    let mut low_kara_cal = Calibration::synthetic(2.0e9, 60.0e9, true, Vec::new());
    low_kara_cal.basecase_max_limbs = 8;
    low_kara_cal.karatsuba_max_limbs = 16;
    let mut cfg_p = EngineConfig::new(digits, OutputFormat::Text, "prod_flt", prof.clone(), &kernels, &fac_p);
    cfg_p.supplied_calibration = Some(low_kara_cal.clone());
    cfg_p.mul_fault_hook = Some(&hook_prod);
    let rep_p = run(&cfg_p).expect("product fault recovery");
    assert_eq!(*fac_p.sink.buf.lock().unwrap(), expected_pi);
    assert!(
        rep_p.telemetry_summary.contains("corrections=1") && !rep_p.telemetry_summary.contains("retries=0"),
        "Expected Rung 1 correction + Rung 2 retry in telemetry, got {}",
        rep_p.telemetry_summary
    );

    // --- Row 1c: Leaf digit corruption -> conversion parent node catches & climbs Rung 2 ---
    struct LeafCorruptOnce(AtomicBool);
    impl LeafFaultHook for LeafCorruptOnce {
        fn on_leaf_word(&self, leaf_idx: u64, attempt: u32, word_19: &mut u64, _emitted_w0: &mut u64) {
            if leaf_idx == 2 && attempt == 0 && !self.0.swap(true, Ordering::Relaxed) {
                *word_19 = (*word_19 + 1) % 10_000_000_000_000_000_000;
            }
        }
    }
    let hook_leaf = LeafCorruptOnce(AtomicBool::new(false));
    let fac_l = MemFactory::new(out_len);
    let mut cfg_l = EngineConfig::new(digits, OutputFormat::Text, "leaf_flt", prof.clone(), &kernels, &fac_l);
    cfg_l.leaf_fault_hook = Some(&hook_leaf);
    let rep_l = run(&cfg_l).expect("leaf fault recovery");
    assert!(hook_leaf.0.load(Ordering::Relaxed));
    assert_eq!(*fac_l.sink.buf.lock().unwrap(), expected_pi);
    assert!(!rep_l.telemetry_summary.contains("retries=0"));

    // --- Row 2: Lowered FFT rounding error bound -> climbs Rung 2 as NTT product ---
    let fac_fft = MemFactory::new(out_len);
    let mut cfg_fft = EngineConfig::new(digits, OutputFormat::Text, "fft_bound", prof.clone(), &kernels, &fac_fft);
    cfg_fft.supplied_calibration = Some(low_kara_cal);
    cfg_fft.fft_err_bound = 0.0; // Every non-trivial FFT product exceeds 0.0 and re-runs as exact NTT!
    let rep_fft = run(&cfg_fft).expect("FFT bound fallback to NTT");
    assert_eq!(*fac_fft.sink.buf.lock().unwrap(), expected_pi);
    assert!(!rep_fft.telemetry_summary.contains("retries=0"));

    // --- Row 4 & 5: Stored piece Rung-1 repair + write-back, and Drop/Misplace/Fail faults ---
    let tm = Arc::new(TestMember::new("vram-test", MemberKind::Vram, 2 * 1024 * 1024));
    let devs = form_stripes(vec![(Arc::clone(&tm) as Arc<dyn ExtentMember>, 20.0e9)], 512);
    let store = Store::new(devs);
    let ext = store.allocate(4096 * 2).unwrap();
    let payload: Vec<u64> = (0..1024).map(|i| i * 7 + 3).collect();
    store.write_all(&ext, &payload, &kernels).unwrap();

    // Corrupt 1 limb inside piece 0 on `tm` -> `read_all` must repair in place AND write back!
    let base_piece_off = ext.round_offset_bytes as usize;
    tm.corrupt_u64_at(base_piece_off + 24, 0xABCD_1234_5678_9EF0);
    let st_tel = Telemetry::default();
    let mut read_back = vec![0u64; 1024];
    store.read_all(&ext, &mut read_back, &kernels, &st_tel).unwrap();
    assert_eq!(read_back, payload);
    assert_eq!(st_tel.corrections.load(Ordering::Relaxed), 1);
    // Second read should now see clean data (confirming write-back of repaired piece!)
    store.read_all(&ext, &mut read_back, &kernels, &st_tel).unwrap();
    assert_eq!(st_tel.corrections.load(Ordering::Relaxed), 1);

    // DropNextWrite & MisplaceNextWrite leave stale/wrong piece data -> detected via expected check word or read failure
    tm.inject_fault(TestFault::FailNextRead);
    assert_eq!(
        store.read_all(&ext, &mut read_back, &kernels, &st_tel),
        Err(StoreError::IoFailed)
    );
    tm.inject_fault(TestFault::FailNextWrite);
    assert_eq!(
        store.write_all(&ext, &payload, &kernels),
        Err(StoreError::IoFailed)
    );
    let ext2 = store.allocate(4096).unwrap();
    tm.inject_fault(TestFault::DropNextWrite);
    store.write_all(&ext2, &payload[..512], &kernels).unwrap();
    let mut dropped_read = vec![0u64; 512];
    store.read_all(&ext2, &mut dropped_read, &kernels, &st_tel).unwrap();
    assert_ne!(
        &dropped_read[..],
        &payload[..512],
        "Dropped write leaves all-zero piece whose own 0 check words match until parent number w0 catches it!"
    );
    tm.inject_fault(TestFault::MisplaceNextWrite(8192));
    store.write_all(&ext2, &payload[..512], &kernels).unwrap();

    // --- Row 7 & 8: GPU stops answering -> excluded & run succeeds; Last SSD stops answering -> Rung 4 Stop ---
    // Case A: Stops answering at startup probe -> excluded cleanly
    let bad_vram = Arc::new(TestMember::new("vram-bad", MemberKind::Vram, 4 * 1024 * 1024));
    bad_vram.inject_fault(TestFault::StopAnswering);
    let good_ssd1 = Arc::new(TestMember::new("ssd-0", MemberKind::Ssd, 8 * 1024 * 1024));
    let good_ssd2 = Arc::new(TestMember::new("ssd-1", MemberKind::Ssd, 8 * 1024 * 1024));
    let ooc_prof = Profile::synthetic(
        2,
        2,
        4 * 1024,
        4 * 1024,
        0,
        4 * 1024, // 4 KiB RAM forces out-of-core Store!
        &[4 * 1024 * 1024],
        &[8 * 1024 * 1024, 8 * 1024 * 1024],
    );
    let fac_ex = MemFactory::new(out_len);
    let mut rates = HashMap::new();
    rates.insert("vram-bad".to_string(), 50.0e9);
    rates.insert("ssd-0".to_string(), 5.0e9);
    rates.insert("ssd-1".to_string(), 5.0e9);
    let mut cfg_ex = EngineConfig::new(digits, OutputFormat::Text, "excl", ooc_prof.clone(), &kernels, &fac_ex);
    cfg_ex.members = vec![bad_vram, Arc::clone(&good_ssd1) as Arc<dyn ExtentMember>, Arc::clone(&good_ssd2) as Arc<dyn ExtentMember>];
    cfg_ex.supplied_member_rates = Some(rates.clone());
    let rep_ex = run(&cfg_ex).expect("VRAM exclusion recovery");
    assert!(
        rep_ex.excluded_devices.contains(&"vram-bad".to_string()),
        "Dead GPU must be excluded, got {:?}",
        rep_ex.excluded_devices
    );
    assert_eq!(*fac_ex.sink.buf.lock().unwrap(), expected_pi);

    // Case B: Stops answering mid-run during out-of-core NTT transform -> rollback excludes it & finishes!
    let digits_ooc = 25_000u64;
    let out_len_ooc = (digits_ooc + 2) as usize;
    let fac_base_ooc = MemFactory::new(out_len_ooc);
    run(&EngineConfig::new(
        digits_ooc,
        OutputFormat::Text,
        "base_ooc",
        prof.clone(),
        &kernels,
        &fac_base_ooc,
    ))
    .unwrap();
    let expected_pi_ooc = fac_base_ooc.sink.buf.lock().unwrap().clone();

    let bad_vram_mid = Arc::new(TestMember::new("vram-bad", MemberKind::Vram, 4 * 1024 * 1024));
    bad_vram_mid.inject_fault(TestFault::StopAnswering);
    let mut ooc_cal = Calibration::synthetic(2.0e9, 40.0e9, true, Vec::new());
    ooc_cal.basecase_max_limbs = 8;
    ooc_cal.karatsuba_max_limbs = 16;
    let fac_ex2 = MemFactory::new(out_len_ooc);
    let mut cfg_ex2 = EngineConfig::new(digits_ooc, OutputFormat::Text, "excl_mid", ooc_prof, &kernels, &fac_ex2);
    cfg_ex2.supplied_calibration = Some(ooc_cal);
    cfg_ex2.supplied_member_rates = Some(rates);
    cfg_ex2.members = vec![
        bad_vram_mid,
        Arc::new(TestMember::new("ssd-0", MemberKind::Ssd, 8 * 1024 * 1024)),
        Arc::new(TestMember::new("ssd-1", MemberKind::Ssd, 8 * 1024 * 1024)),
    ];
    let rep_ex2 = run(&cfg_ex2).expect("mid-run VRAM exclusion recovery");
    assert!(
        rep_ex2.excluded_devices.contains(&"vram-bad".to_string()),
        "Mid-run dead GPU must be excluded, got {:?}",
        rep_ex2.excluded_devices
    );
    assert_eq!(*fac_ex2.sink.buf.lock().unwrap(), expected_pi_ooc);

    // Last SSD stops answering during checkpoint -> Rung 4 FatalStop!
    let dead_last_ssd = Arc::new(TestMember::new("ssd-only", MemberKind::Ssd, 8 * 1024 * 1024));
    dead_last_ssd.inject_fault(TestFault::StopAnswering);
    let last_prof = Profile::synthetic(2, 2, 16 * 1024, 64 * 1024, 0, 16 * 1024 * 1024, &[], &[8 * 1024 * 1024]);
    let fac_last = MemFactory::new(out_len);
    let mut cfg_last = EngineConfig::new(digits, OutputFormat::Text, "last_ssd", last_prof, &kernels, &fac_last);
    cfg_last.members = vec![dead_last_ssd];
    cfg_last.overrides.checkpoint_mult = 0.0;
    let err_last = run(&cfg_last).unwrap_err();
    assert!(matches!(err_last, EngineError::FatalStop(_)), "Expected Rung 4 FatalStop, got {err_last:?}");

    // --- Row 10 & Capacity preflight: Output file write failure, create failure, InsufficientCapacity ---
    let fac_wfail = MemFactory::new(out_len);
    fac_wfail.sink.fail_writes.store(true, Ordering::Relaxed);
    let err_wf = run(&EngineConfig::new(digits, OutputFormat::Text, "wf", prof.clone(), &kernels, &fac_wfail)).unwrap_err();
    assert!(matches!(err_wf, EngineError::FatalStop(_)));

    let mut fac_cfail = MemFactory::new(out_len);
    fac_cfail.fail_create = true;
    let err_cf = run(&EngineConfig::new(digits, OutputFormat::Text, "cf", prof.clone(), &kernels, &fac_cfail)).unwrap_err();
    assert!(matches!(err_cf, EngineError::Plan(PlanError::OutputFileError(_))));

    let tiny_unholdable = Profile::synthetic(1, 1, 4096, 8192, 0, 64 * 1024, &[], &[]);
    let err_cap = run(&EngineConfig::new(
        5_000_000,
        OutputFormat::Text,
        "unholdable",
        tiny_unholdable,
        &kernels,
        &MemFactory::new(16),
    ))
    .unwrap_err();
    assert!(matches!(
        err_cap,
        EngineError::Plan(PlanError::InsufficientCapacity { .. })
    ));
}

#[test]
fn gate2_runs_of_zeros_and_nines_inside_and_at_end() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let prof = Profile::synthetic(2, 2, 32 * 1024, 512 * 1024, 4 * 1024 * 1024, 128 * 1024 * 1024, &[], &[]);
    let shares = Shares::from_profile(&prof);
    let sched = Scheduler::new(&shares, None);
    let cal = Calibration::synthetic(2.0e9, 60.0e9, true, Vec::new());
    let fft_cache = FftTableCache::default();
    let ntt_cache = NttTableCache::default();
    let ladder = Ladder::default();
    let tel = Telemetry::default();
    let ctx = MulCtx::new(&kernels, &fft_cache, &ntt_cache, &sched, None, &ladder, &tel);

    // Helper: build `floor(num * 2^{64 * k_limbs} / den)` as a `BigNum` for exact rational tests!
    let make_rational_frac = |num: u128, den: u128, k_limbs: usize| -> BigNum {
        let mut limbs = vec![0u64; k_limbs];
        let mut rem = num;
        for idx in (0..k_limbs).rev() {
            // Compute `(rem << 64) / den` via two 32-bit shifts in u128
            let hi = (rem << 32) / den;
            let r1 = (rem << 32) % den;
            let lo = (r1 << 32) / den;
            rem = (r1 << 32) % den;
            limbs[idx] = ((hi << 32) | lo) as u64;
        }
        BigNum::from_limbs(limbs, false, 0, &kernels)
    };

    let digits = 50u64;
    let plan = Plan::build(digits, OutputFormat::Text, &shares, &cal, PlanOverrides::default()).unwrap();
    let k_in = 8usize;

    // 1. Exact binary fraction `3 + 1/8 = 3.12500000000000000000000...` (run of 0s extending past end!)
    let sink_z = MemoryDigitSink::new((digits as usize) + 2);
    let frac_125 = make_rational_frac(1, 8, k_in);
    let res_z = run_conversion(
        &ctx,
        &plan,
        &shares,
        &cal,
        3,
        &frac_125,
        k_in,
        &sink_z,
        None,
        None,
        None,
    )
    .unwrap();
    let txt_z = String::from_utf8(sink_z.into_bytes()).unwrap();
    assert_eq!(txt_z, format!("3.125{}", "0".repeat(47)));
    assert!(
        res_z.digits_in_doubt,
        "Trailing all-zeros in dropped guard leaf must report digits_in_doubt = true"
    );

    // 2. Fraction just under `3.5`: `3 + (50000... - 0) - epsilon` -> `0.4999999999999999999...` (run of 9s at end!)
    let mut limbs_nines = vec![u64::MAX; k_in];
    limbs_nines[k_in - 1] = (1u64 << 63) - 1; // 0.5 - 2^{-512}: 0.49999999999999999999999999...
    let frac_nines = BigNum::from_limbs(limbs_nines, false, 0, &kernels);
    let sink_9 = MemoryDigitSink::new((digits as usize) + 2);
    let res_9 = run_conversion(
        &ctx,
        &plan,
        &shares,
        &cal,
        3,
        &frac_nines,
        k_in,
        &sink_9,
        None,
        None,
        None,
    )
    .unwrap();
    let txt_9 = String::from_utf8(sink_9.into_bytes()).unwrap();
    assert_eq!(txt_9, format!("3.4{}", "9".repeat(49)));
    assert!(
        res_9.digits_in_doubt,
        "Trailing all-nines in dropped guard leaf must report digits_in_doubt = true"
    );

    // 3. Runs of 0s and 9s INSIDE the number, followed by non-trivial digits (`1 / 999_999_999_999_999_999 + 1 / 7` etc.):
    //    `1_000_000_000_000_000_007 / 10^25 + 1 / 700_000_000_000_000_000_000_000_000` -> Let's use `num / den = 123_000_000_000_000_000_999_999_999_999_3 / 10^31 + 1/3 * 10^{-35}`:
    //    Even simpler: `1 / 7_000_000_000_000_000_000` starts with `0.000000000000000000142857142857...`
    //    (18 zeros inside, then non-repeating/non-9/non-0 `142857...` at the guard leaf!), and
    //    `6_999_999_999_999_999_999 / 7_000_000_000_000_000_000` starts with `0.999999999999999999857142857...`!
    let den_7 = 7_000_000_000_000_000_000u128;
    let frac_in0 = make_rational_frac(1, den_7, k_in);
    let sink_in0 = MemoryDigitSink::new((digits as usize) + 2);
    let res_in0 = run_conversion(
        &ctx,
        &plan,
        &shares,
        &cal,
        3,
        &frac_in0,
        k_in,
        &sink_in0,
        None,
        None,
        None,
    )
    .unwrap();
    let txt_in0 = String::from_utf8(sink_in0.into_bytes()).unwrap();
    assert!(txt_in0.starts_with("3.000000000000000000142857142857"));
    assert!(!res_in0.digits_in_doubt, "Interior 0s with non-zero tail must NOT be in doubt");

    let frac_in9 = make_rational_frac(den_7 - 1, den_7, k_in);
    let sink_in9 = MemoryDigitSink::new((digits as usize) + 2);
    let res_in9 = run_conversion(
        &ctx,
        &plan,
        &shares,
        &cal,
        3,
        &frac_in9,
        k_in,
        &sink_in9,
        None,
        None,
        None,
    )
    .unwrap();
    let txt_in9 = String::from_utf8(sink_in9.into_bytes()).unwrap();
    assert!(txt_in9.starts_with("3.999999999999999999857142857142"));
    assert!(!res_in9.digits_in_doubt, "Interior 9s with non-nine tail must NOT be in doubt");
}

#[test]
fn gate2_live_vulkan_vram_member_transfer_roundtrip() {
    platform::vram::set_test_vk_validation(true);
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let (entries, members) = platform::vram::probe_vram_members();
    let (val_requested, val_active) = platform::vram::last_probe_validation_state();
    if members.is_empty() {
        // Allowed on headless CI without a discrete transfer-queue GPU, but on RTX 5070 it binds!
        println!("No discrete Vulkan 1.4 transfer-only VRAM member detected; skipping live copy");
        return;
    }
    assert!(
        val_requested,
        "Automated discrete-GPU test must request VK_LAYER_KHRONOS_validation (Section 12)"
    );
    assert_eq!(entries.len(), members.len());
    println!(
        "Probed {} discrete GPU VRAM member(s): id={} cap={:.2} GiB vk_validation_active={}",
        members.len(),
        entries[0].id,
        (entries[0].size as f64) / (1024.0 * 1024.0 * 1024.0),
        val_active
    );

    platform::vram::reset_vram_transfer_stats();
    let devs = form_stripes(vec![(Arc::clone(&members[0]), 28.0e9)], 4096);
    let store = Store::new(devs);
    for round in 0..2u64 {
        let ext = store.allocate(256 * 1024).expect("allocate 256 KiB in VRAM");
        let src: Vec<u64> = (0..(256 * 1024 / 8))
            .map(|i| (i as u64).wrapping_mul(0x517C_C1B7_2722_0A95) ^ (0xA5A5 + round))
            .collect();
        store
            .write_all(&ext, &src, &kernels)
            .expect("Vulkan transfer-queue write batch");
        let tel = Telemetry::default();
        let mut dst = vec![0u64; src.len()];
        store
            .read_all(&ext, &mut dst, &kernels, &tel)
            .expect("Vulkan transfer-queue read batch");
        assert_eq!(dst, src, "VRAM extent roundtrip must match bit-for-bit with all 3 check words");
        assert_eq!(tel.corrections.load(Ordering::Relaxed), 0);
        store.free(ext);
    }
    let (direct_imports, bounce_fallbacks) = platform::vram::vram_transfer_stats();
    assert!(
        direct_imports >= 4 && bounce_fallbacks == 0,
        "VRAM transfers must import caller 64KiB-aligned buffers directly via VK_EXT_external_memory_host (direct={direct_imports}, bounce={bounce_fallbacks})"
    );
}

/// Runs of zeros and nines that reach through the nodes of the conversion whose products go
/// through a transform. Such a node takes its product modulo `2^L - 1` and reads the right
/// child's fraction from just below the top; that is exact unless a carry could have crossed
/// from the wrapped integer part at the bottom, which the node tells by the bits in between
/// being all zero, and then it takes the whole product instead. A fraction that ends in zeros or
/// in nines exercises both the wrapped and whole-product paths, next to fractions with ordinary
/// digits, and every digit is compared.
#[test]
fn gate2_runs_of_zeros_and_nines_through_transform_nodes() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let prof = Profile::synthetic(2, 2, 32 * 1024, 512 * 1024, 4 * 1024 * 1024, 128 * 1024 * 1024, &[], &[]);
    let shares = Shares::from_profile(&prof);
    let sched = Scheduler::new(&shares, None);
    let cal = Calibration::synthetic(2.0e9, 60.0e9, true, Vec::new());
    let fft_cache = FftTableCache::default();
    let ntt_cache = NttTableCache::default();
    let ladder = Ladder::default();
    let tel = Telemetry::default();
    let ctx = MulCtx::new(&kernels, &fft_cache, &ntt_cache, &sched, None, &ladder, &tel);

    let digits = 60_000u64;
    let plan = Plan::build(digits, OutputFormat::Text, &shares, &cal, PlanOverrides::default()).unwrap();
    let k_in = ((digits as f64) * std::f64::consts::LOG2_10 / 64.0).ceil() as usize + 8;

    // floor(num * 2^(64 * k_in) / den) as limbs, den below 2^64
    let rational = |num: u128, den: u128| -> Vec<u64> {
        let mut limbs = vec![0u64; k_in];
        let mut rem = num % den;
        for idx in (0..k_in).rev() {
            let hi = (rem << 32) / den;
            let r1 = (rem << 32) % den;
            let lo = (r1 << 32) / den;
            rem = (r1 << 32) % den;
            limbs[idx] = ((hi << 32) | lo) as u64;
        }
        limbs
    };
    let convert = |limbs: Vec<u64>| -> (String, bool) {
        let frac = BigNum::from_limbs(limbs, false, 0, &kernels);
        let sink = MemoryDigitSink::new((digits as usize) + 2);
        let res = run_conversion(&ctx, &plan, &shares, &cal, 3, &frac, k_in, &sink, None, None, None).unwrap();
        (String::from_utf8(sink.into_bytes()).unwrap(), res.digits_in_doubt)
    };
    let n = digits as usize;

    // 1/8: three digits, then zeros to the end
    let (txt, doubt) = convert(rational(1, 8));
    assert_eq!(txt, format!("3.125{}", "0".repeat(n - 3)));
    assert!(doubt);

    // just under 1/2: a four, then nines to the end
    let mut limbs = vec![u64::MAX; k_in];
    limbs[k_in - 1] = (1u64 << 63) - 1;
    let (txt, doubt) = convert(limbs);
    assert_eq!(txt, format!("3.4{}", "9".repeat(n - 1)));
    assert!(doubt);

    // 1/8 plus one unit in the last limb: the fraction left after the first digits is a few
    // bits at the very bottom, so every transform node sees nothing but zeros below the top
    // (and, the integer part being larger than what is left, all ones once it wraps): each of
    // them has to fall back to the whole product. The digits are those of 1/8.
    let mut limbs = rational(1, 8);
    limbs[0] = 1;
    let (txt, doubt) = convert(limbs);
    assert_eq!(txt, format!("3.125{}", "0".repeat(n - 3)));
    assert!(doubt);
    // and one unit less than 1/8: nines after 3.124
    let mut limbs = rational(1, 8);
    limbs[k_in - 1] -= 1;
    limbs.iter_mut().take(k_in - 1).for_each(|w| *w = u64::MAX);
    let (txt, doubt) = convert(limbs);
    assert_eq!(txt, format!("3.124{}", "9".repeat(n - 3)));
    assert!(doubt);

    // 1/3 and 1/7: ordinary digits all the way
    let (txt, doubt) = convert(rational(1, 3));
    assert_eq!(txt, format!("3.{}", "3".repeat(n)));
    assert!(!doubt);
    let (txt, doubt) = convert(rational(1, 7));
    let sevenths: String = "142857".chars().cycle().take(n).collect();
    assert_eq!(txt, format!("3.{sevenths}"));
    assert!(!doubt);

    // A long run of zeros, then ordinary digits: 1/7 moved down by 300 limbs, 2^(-19200) / 7.
    let mut limbs = rational(1, 7);
    limbs.drain(..300);
    limbs.resize(k_in, 0);
    let expect_zeros = (19200.0 * 2f64.log10() + 7f64.log10()).floor() as usize;
    let (txt, doubt) = convert(limbs.clone());
    let zeros = txt[2..].bytes().take_while(|&c| c == b'0').count();
    assert_eq!(zeros, expect_zeros, "zeros after the point");
    assert!(!doubt);
    // Its complement to all ones is nines where this has zeros, and digit by digit the two add
    // to nine (the last few digits apart, where the two round down separately).
    let comp: Vec<u64> = limbs.iter().map(|&w| !w).collect();
    let (txt_c, _) = convert(comp);
    assert!(txt_c[2..].bytes().take(expect_zeros).all(|c| c == b'9'), "nines where the other has zeros");
    let sum_ok = txt[2..n - 20]
        .bytes()
        .zip(txt_c[2..n - 20].bytes())
        .all(|(a, b)| (a - b'0') + (b - b'0') == 9);
    assert!(sum_ok, "a fraction and its complement must add to nines");
}

#[test]
fn gate2_exact_quotient_rung1_repair_and_rung2_factor_inverse_remake() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let digits = 12_000u64;
    let out_len = (digits as usize) + 2;
    let prof = Profile::synthetic(
        2,
        2,
        32 * 1024,
        512 * 1024,
        4 * 1024 * 1024,
        128 * 1024 * 1024,
        &[],
        &[],
    );
    let ref_fac = MemFactory::new(out_len);
    let base_rep = run(&EngineConfig::new(
        digits,
        OutputFormat::Text,
        "quot_base",
        prof.clone(),
        &kernels,
        &ref_fac,
    ))
    .unwrap();
    let expected_pi = ref_fac.sink.buf.lock().unwrap().clone();

    struct QuotientFaultHook {
        hits: AtomicUsize,
    }
    impl MulFaultHook for QuotientFaultHook {
        fn on_product_limbs(
            &self,
            point_id: u64,
            attempt: u32,
            limbs: &mut engine::limb::LimbVec,
            emitted_w0: &mut u64,
            _fft_err: &mut f64,
        ) {
            // Target dividing-level SingleProduct cached-inverse quotients (`0x2000_0000` prefix).
            if (point_id & 0xF000_0000) == 0x2000_0000 && attempt == 0 && limbs.len() >= 4 {
                let n = self.hits.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // 1-limb quotient corruption -> Rung 1 single-limb repair in place!
                    limbs[1] ^= 0x1357_9BDF_2468_ACE0;
                    *emitted_w0 = (*emitted_w0 + 11) % MOD_W0;
                } else if n == 1 {
                    // 2-limb quotient corruption -> Rung 2 remakes factor & Hensel inverse!
                    limbs[0] ^= 0xAAAA_5555_1111_2222;
                    limbs[2] ^= 0x3333_4444_7777_8888;
                    *emitted_w0 = (*emitted_w0 + 23) % MOD_W0;
                }
            }
        }
    }

    let q_hook = QuotientFaultHook {
        hits: AtomicUsize::new(0),
    };
    let mut cal = Calibration::synthetic(4.0e9, 80.0e9, true, Vec::new());
    cal.basecase_max_limbs = 4;
    cal.karatsuba_max_limbs = 8;
    let fac_q = MemFactory::new(out_len);
    let mut cfg_q = EngineConfig::new(digits, OutputFormat::Text, "quot_flt", prof, &kernels, &fac_q);
    cfg_q.supplied_calibration = Some(cal);
    cfg_q.overrides.force_dividing_levels = Some(vec![(4, DivForm::SingleProduct)]);
    cfg_q.mul_fault_hook = Some(&q_hook);

    let rep_q = run(&cfg_q).expect("quotient Rung 1 + Rung 2 recovery");
    assert!(
        q_hook.hits.load(Ordering::Relaxed) >= 2,
        "Quotient fault hook must fire on SingleProduct dividing level, hits={}",
        q_hook.hits.load(Ordering::Relaxed)
    );
    assert_eq!(rep_q.dec_hash_w0, base_rep.dec_hash_w0);
    assert_eq!(*fac_q.sink.buf.lock().unwrap(), expected_pi);
    assert!(
        !rep_q.telemetry_summary.contains("corrections=0")
            && !rep_q.telemetry_summary.contains("retries=0"),
        "Expected both Rung-1 correction and Rung-2 factor+inverse remake retry: {}",
        rep_q.telemetry_summary
    );
}

#[test]
fn gate2_second_rollback_at_same_point_stops_at_rung4_and_timeouts_exclude() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let digits = 3_000u64;
    let out_len = (digits as usize) + 2;
    let prof = Profile::synthetic(
        2,
        2,
        32 * 1024,
        512 * 1024,
        4 * 1024 * 1024,
        64 * 1024 * 1024,
        &[],
        &[8 * 1024 * 1024, 8 * 1024 * 1024],
    );

    let ref_fac = MemFactory::new(out_len);
    run(&EngineConfig::new(
        digits,
        OutputFormat::Text,
        "rb_ref",
        prof.clone(),
        &kernels,
        &ref_fac,
    ))
    .unwrap();
    let expected_pi = ref_fac.sink.buf.lock().unwrap().clone();

    // 1. Persistent unrecoverable fault at the SAME `point_id` must roll back once (Rung 3) and
    //    stop cleanly at Rung 4 (`FatalStop`) on the second rollback at that point (Section 11)!
    struct SamePointAlwaysBad {
        target_pid: Mutex<Option<u64>>,
    }
    impl MulFaultHook for SamePointAlwaysBad {
        fn on_product_limbs(
            &self,
            point_id: u64,
            _attempt: u32,
            limbs: &mut engine::limb::LimbVec,
            emitted_w0: &mut u64,
            _fft_err: &mut f64,
        ) {
            if point_id >= 0x3000_0000 && point_id < 0x4000_0000 && limbs.len() >= 6 {
                let mut g = self.target_pid.lock().unwrap();
                let pid = *g.get_or_insert(point_id);
                if pid == point_id {
                    limbs[0] ^= 0xDEAD_0001;
                    limbs[2] ^= 0xDEAD_0002;
                    limbs[4] ^= 0xDEAD_0003;
                    *emitted_w0 = (*emitted_w0 + 31) % MOD_W0;
                }
            }
        }
    }

    let mut low_cal = Calibration::synthetic(2.0e9, 60.0e9, true, Vec::new());
    low_cal.basecase_max_limbs = 4;
    low_cal.karatsuba_max_limbs = 8;

    let live_events: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let reporter = |msg: &str| {
        live_events.lock().unwrap().push(msg.to_string());
    };
    let hook_same = SamePointAlwaysBad {
        target_pid: Mutex::new(None),
    };
    let ssd0 = Arc::new(TestMember::new("ssd-0", MemberKind::Ssd, 8 * 1024 * 1024));
    let ssd1 = Arc::new(TestMember::new("ssd-1", MemberKind::Ssd, 8 * 1024 * 1024));
    let fac_same = MemFactory::new(out_len);
    let mut cfg_same =
        EngineConfig::new(digits, OutputFormat::Text, "same_pt", prof.clone(), &kernels, &fac_same);
    cfg_same.supplied_calibration = Some(low_cal.clone());
    cfg_same.members = vec![ssd0, ssd1];
    cfg_same.overrides.checkpoint_mult = 0.0;
    cfg_same.mul_fault_hook = Some(&hook_same);
    cfg_same.progress_reporter = Some(&reporter);

    let err_same = run(&cfg_same).expect_err("2nd rollback at same point must stop at Rung 4");
    match &err_same {
        EngineError::FatalStop(msg) => {
            assert!(
                msg.contains("rollbacks=2"),
                "Must stop on the 2nd rollback at the same point (not after 8!), got: {msg}"
            );
        }
        other => panic!("Expected Rung 4 FatalStop, got {other:?}"),
    }
    let captured = live_events.lock().unwrap().clone();
    assert!(
        captured.iter().any(|l| l.starts_with("Checkpoint ") && l.contains("cores=[") && l.contains("devices=[")),
        "Telemetry must be reported at checkpoints: {captured:?}"
    );
    assert!(
        captured.iter().any(|l| l.starts_with("Return ") && l.contains("rollbacks=1")),
        "Telemetry must be reported at rollback #1 before stopping on rollback #2: {captured:?}"
    );

    // 2. Transient fault that rolls back ONCE at a point and succeeds on replay must finish cleanly
    //    and report Checkpoint, Return, and End telemetry lines!
    struct FailPointFirstEpochOnly {
        target_pid: Mutex<Option<u64>>,
        hits: AtomicUsize,
    }
    impl MulFaultHook for FailPointFirstEpochOnly {
        fn on_product_limbs(
            &self,
            point_id: u64,
            _attempt: u32,
            limbs: &mut engine::limb::LimbVec,
            emitted_w0: &mut u64,
            _fft_err: &mut f64,
        ) {
            if point_id >= 0x3000_0000 && point_id < 0x4000_0000 && limbs.len() >= 6 {
                let mut g = self.target_pid.lock().unwrap();
                let pid = *g.get_or_insert(point_id);
                if pid == point_id && self.hits.fetch_add(1, Ordering::Relaxed) < 3 {
                    limbs[0] ^= 0xBEEF_0001;
                    limbs[2] ^= 0xBEEF_0002;
                    *emitted_w0 = (*emitted_w0 + 19) % MOD_W0;
                }
            }
        }
    }
    let hook_once = FailPointFirstEpochOnly {
        target_pid: Mutex::new(None),
        hits: AtomicUsize::new(0),
    };
    let ssd_a = Arc::new(TestMember::new("ssd-a", MemberKind::Ssd, 8 * 1024 * 1024));
    let ssd_b = Arc::new(TestMember::new("ssd-b", MemberKind::Ssd, 8 * 1024 * 1024));
    let fac_once = MemFactory::new(out_len);
    let mut cfg_once =
        EngineConfig::new(digits, OutputFormat::Text, "once_pt", prof.clone(), &kernels, &fac_once);
    cfg_once.supplied_calibration = Some(low_cal);
    cfg_once.members = vec![ssd_a, ssd_b];
    cfg_once.overrides.checkpoint_mult = 0.0;
    cfg_once.mul_fault_hook = Some(&hook_once);
    let rep_once = run(&cfg_once).expect("single rollback recovery");
    assert_eq!(rep_once.rollbacks, 1);
    assert_eq!(*fac_once.sink.buf.lock().unwrap(), expected_pi);
    assert!(rep_once.event_reports.iter().any(|l| l.starts_with("Checkpoint ")));
    assert!(rep_once.event_reports.iter().any(|l| l.starts_with("Return ")));
    assert!(rep_once.event_reports.iter().any(|l| l.starts_with("End ")));
    assert!(
        !rep_once.telemetry_summary.starts_with("scrubs=0 "),
        "Checkpoint patrol checks must increment scrubs: {}",
        rep_once.telemetry_summary
    );

    // 3. SSD `Timeout` & `FailNextWrite`: redundant SSD is excluded without counting rollbacks;
    //    last SSD stops at Rung 4 (`FatalStop`, Section 11)!
    let timeout_ssd = Arc::new(TestMember::new("ssd-timeout", MemberKind::Ssd, 8 * 1024 * 1024));
    timeout_ssd.inject_fault(TestFault::Timeout);
    let healthy_ssd = Arc::new(TestMember::new("ssd-ok", MemberKind::Ssd, 8 * 1024 * 1024));
    let fac_to = MemFactory::new(out_len);
    let mut cfg_to = EngineConfig::new(digits, OutputFormat::Text, "to_excl", prof, &kernels, &fac_to);
    cfg_to.members = vec![timeout_ssd, healthy_ssd];
    cfg_to.overrides.checkpoint_mult = 0.0;
    let rep_to = run(&cfg_to).expect("redundant SSD timeout excluded");
    assert!(rep_to.excluded_devices.contains(&"ssd-timeout".to_string()));
    assert_eq!(rep_to.rollbacks, 0, "Device-exclusion recovery must not count against rollbacks");
    assert_eq!(*fac_to.sink.buf.lock().unwrap(), expected_pi);

    for last_fault in [TestFault::Timeout, TestFault::FailNextWrite] {
        let last_ssd = Arc::new(TestMember::new("ssd-last", MemberKind::Ssd, 8 * 1024 * 1024));
        last_ssd.inject_fault(last_fault);
        let one_ssd_prof = Profile::synthetic(
            2,
            2,
            16 * 1024,
            64 * 1024,
            0,
            16 * 1024 * 1024,
            &[],
            &[8 * 1024 * 1024],
        );
        let fac_l = MemFactory::new(out_len);
        let mut cfg_l =
            EngineConfig::new(digits, OutputFormat::Text, "last_flt", one_ssd_prof, &kernels, &fac_l);
        cfg_l.members = vec![last_ssd];
        cfg_l.overrides.checkpoint_mult = 0.0;
        let err_l = run(&cfg_l).expect_err("Last SSD write failure or timeout must stop at Rung 4");
        assert!(
            matches!(err_l, EngineError::FatalStop(_)),
            "Fault {last_fault:?} on last SSD must be FatalStop, got {err_l:?}"
        );
    }

    // 4. Redundant SSD whose stored piece sends the run back a SECOND time is excluded (`dev_gb >= 2`,
    //    Section 11: "or its piece sends the run back a second time -> It is left out")!
    struct PieceSendsBackTwiceSsd {
        inner: TestMember,
        epoch: AtomicUsize,
        epoch_reads: AtomicUsize,
    }
    impl ExtentMember for PieceSendsBackTwiceSsd {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn kind(&self) -> MemberKind {
            MemberKind::Ssd
        }
        fn alignment(&self) -> u64 {
            4096
        }
        fn capacity(&self) -> u64 {
            self.inner.capacity()
        }
        fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
            self.inner.allocate(bytes)
        }
        fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
            self.inner.read(ranges, host_buf)?;
            if ranges.iter().any(|r| r.device_offset == 0) {
                self.epoch.fetch_add(1, Ordering::Relaxed);
                self.epoch_reads.store(0, Ordering::Relaxed);
                return Ok(());
            }
            // Corrupt 2 limbs on one non-cached Store product read in Epoch 1 (`dev_gb = 1`,
            // `faults = 1`) and once again in Epoch 2 (`dev_gb = 2`, `faults = 2 < 4` -> excluded
            // specifically by the 2nd-rollback rule!).
            if ranges.iter().any(|r| r.device_offset >= 900_000 && r.len >= 64) {
                let ep = self.epoch.load(Ordering::Relaxed);
                let r_idx = self.epoch_reads.fetch_add(1, Ordering::Relaxed);
                if (ep == 1 || ep == 2) && r_idx < 2 && host_buf.len() >= 32 {
                    host_buf[0] ^= 0xFF;
                    host_buf[16] ^= 0xAA;
                }
            }
            Ok(())
        }
        fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
            self.inner.write(ranges, host_buf)
        }
        fn free(&self, device_offset: u64, bytes: u64) {
            self.inner.free(device_offset, bytes);
        }
    }

    let digits_ooc = 25_000u64;
    let out_len_ooc = (digits_ooc + 2) as usize;
    let fac_ooc_ref = MemFactory::new(out_len_ooc);
    run(&EngineConfig::new(
        digits_ooc,
        OutputFormat::Text,
        "ooc_2gb_ref",
        Profile::synthetic(2, 2, 32 * 1024, 512 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, &[], &[]),
        &kernels,
        &fac_ooc_ref,
    ))
    .unwrap();
    let expected_pi_ooc = fac_ooc_ref.sink.buf.lock().unwrap().clone();

    let flaky_piece_ssd: Arc<dyn ExtentMember> = Arc::new(PieceSendsBackTwiceSsd {
        inner: TestMember::new("ssd-piece-2x", MemberKind::Ssd, 16 * 1024 * 1024),
        epoch: AtomicUsize::new(0),
        epoch_reads: AtomicUsize::new(0),
    });
    let clean_ssd: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("ssd-clean", MemberKind::Ssd, 16 * 1024 * 1024));
    let mut ooc_cal = Calibration::synthetic(2.5e9, 40.0e9, true, Vec::new());
    ooc_cal.basecase_max_limbs = 8;
    ooc_cal.karatsuba_max_limbs = 16;
    let mut rates2 = HashMap::new();
    rates2.insert("ssd-piece-2x".to_string(), 6.0e9);
    rates2.insert("ssd-clean".to_string(), 6.0e9);
    let ooc2_prof = Profile::synthetic(
        2,
        2,
        4 * 1024,
        4 * 1024,
        0,
        4 * 1024,
        &[],
        &[16 * 1024 * 1024, 16 * 1024 * 1024],
    );
    let fac_2gb = MemFactory::new(out_len_ooc);
    let mut cfg_2gb =
        EngineConfig::new(digits_ooc, OutputFormat::Text, "ooc_2gb", ooc2_prof, &kernels, &fac_2gb);
    cfg_2gb.supplied_calibration = Some(ooc_cal);
    cfg_2gb.supplied_member_rates = Some(rates2);
    cfg_2gb.members = vec![flaky_piece_ssd, clean_ssd];
    let rep_2gb = run(&cfg_2gb).expect("SSD excluded after sending run back a 2nd time");
    assert!(
        rep_2gb.excluded_devices.contains(&"ssd-piece-2x".to_string()),
        "SSD whose piece sends the run back a 2nd time must be excluded: {:?}",
        rep_2gb.excluded_devices
    );
    assert_eq!(
        rep_2gb.rollbacks, 1,
        "1st rollback counted, 2nd rollback excluded the SSD without counting against rollbacks"
    );
    assert_eq!(*fac_2gb.sink.buf.lock().unwrap(), expected_pi_ooc);
}

#[test]
fn gate2_three_arenas_numa_routing_subtree_lifo_and_zero_leak_teardown() {
    use engine::arena::{MappingReport, RegionMapper};
    use engine::tier::RamNode;
    use std::alloc::{GlobalAlloc, Layout, System};
    const REGION_PAGE: usize = 4096;

    struct NumaTrackingMapper {
        reserves: [AtomicUsize; 4],
        commits: [AtomicUsize; 4],
        releases: AtomicUsize,
    }

    static NUMA_MAPPER: NumaTrackingMapper = NumaTrackingMapper {
        reserves: [
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        ],
        commits: [
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        ],
        releases: AtomicUsize::new(0),
    };

    impl RegionMapper for NumaTrackingMapper {
        fn reserve_region(
            &self,
            bytes: usize,
            numa_node: usize,
        ) -> Option<(*mut u8, usize, MappingReport)> {
            let cap = bytes.div_ceil(REGION_PAGE) * REGION_PAGE;
            let layout = Layout::from_size_align(cap.max(REGION_PAGE), REGION_PAGE).ok()?;
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if ptr.is_null() {
                return None;
            }
            self.reserves[numa_node.min(3)].fetch_add(1, Ordering::Relaxed);
            Some((ptr, cap.max(REGION_PAGE), MappingReport::default()))
        }

        fn commit_range(
            &self,
            _ptr: *mut u8,
            _bytes: usize,
            numa_node: usize,
        ) -> Result<MappingReport, ()> {
            self.commits[numa_node.min(3)].fetch_add(1, Ordering::Relaxed);
            Ok(MappingReport::default())
        }

        unsafe fn decommit_range(&self, _ptr: *mut u8, _bytes: usize) {}

        unsafe fn release_region(&self, base: *mut u8, reserved_bytes: usize) {
            let cap = reserved_bytes.div_ceil(REGION_PAGE) * REGION_PAGE;
            if let Ok(layout) = Layout::from_size_align(cap.max(REGION_PAGE), REGION_PAGE) {
                unsafe { System.dealloc(base, layout) };
            }
            self.releases.fetch_add(1, Ordering::Relaxed);
        }
    }

    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let digits = 60_000u64;
    let out_len = (digits as usize) + 2;

    // Two NUMA nodes (threads 0,1 on node 0; threads 2,3 on node 1) with 64 MiB per node.
    let mut prof = Profile::synthetic(
        4,
        4,
        32 * 1024,
        256 * 1024,
        4 * 1024 * 1024,
        128 * 1024 * 1024,
        &[],
        &[],
    );
    prof.ram = vec![
        RamNode {
            size: 64 * 1024 * 1024,
            threads: vec![0, 1],
        },
        RamNode {
            size: 64 * 1024 * 1024,
            threads: vec![2, 3],
        },
    ];

    let mut cal = Calibration::synthetic(4.0e9, 80.0e9, true, Vec::new());
    cal.basecase_max_limbs = 8;
    cal.karatsuba_max_limbs = 24;

    let ref_fac = MemFactory::new(out_len);
    let mut ref_cfg =
        EngineConfig::new(digits, OutputFormat::Text, "arena_ref", prof.clone(), &kernels, &ref_fac);
    ref_cfg.supplied_calibration = Some(cal.clone());
    let ref_rep = run(&ref_cfg).expect("reference run");

    let fac = MemFactory::new(out_len);
    let mut cfg = EngineConfig::new(digits, OutputFormat::Text, "arena_numa", prof, &kernels, &fac);
    cfg.supplied_calibration = Some(cal);
    cfg.region_mapper = Some(&NUMA_MAPPER);
    let rep = run(&cfg).expect("NUMA 3-arena run");

    assert_eq!(rep.dec_hash_w0, ref_rep.dec_hash_w0);
    assert_eq!(*fac.sink.buf.lock().unwrap(), *ref_fac.sink.buf.lock().unwrap());

    // All three Section-10 arenas (`numbers` subtree LIFO + upper region, `scratch`, `transforms`)
    // must be actively populated during the run and return cleanly to 0 live bytes at teardown!
    assert!(
        rep.arena_usage.numbers_peak_bytes > 0,
        "Numbers arena must record non-zero peak usage: {:?}",
        rep.arena_usage
    );
    assert!(
        rep.arena_usage.numbers_subtree_lifo_peak_bytes > 0,
        "Worker subtree LIFO slabs in Numbers arena must be actively used: {:?}",
        rep.arena_usage
    );
    assert!(
        rep.arena_usage.numbers_region_peak_bytes > 0,
        "Upper plan-assigned Regions in Numbers arena must be actively used: {:?}",
        rep.arena_usage
    );
    assert!(
        rep.arena_usage.scratch_peak_bytes > 0,
        "Per-worker Scratch arenas must record non-zero peak usage: {:?}",
        rep.arena_usage
    );
    assert!(
        rep.arena_usage.transforms_peak_bytes > 0,
        "Transforms arena must record non-zero peak usage: {:?}",
        rep.arena_usage
    );
    assert_eq!(
        (
            rep.arena_usage.numbers_live_bytes,
            rep.arena_usage.scratch_live_bytes,
            rep.arena_usage.transforms_live_bytes,
        ),
        (0, 0, 0),
        "All three arenas must drop back to 0 live bytes at run teardown: {:?}",
        rep.arena_usage
    );
    assert!(
        NUMA_MAPPER.reserves[0].load(Ordering::Relaxed) > 0
            && NUMA_MAPPER.reserves[1].load(Ordering::Relaxed) > 0
            && NUMA_MAPPER.commits[0].load(Ordering::Relaxed) > 0
            && NUMA_MAPPER.commits[1].load(Ordering::Relaxed) > 0,
        "Both NUMA node 0 and NUMA node 1 must receive reserve_region and commit_range calls"
    );
}

#[test]
fn gate2_cli_profile_and_supplied_calibration_plan_only_is_deterministic() {
    let bin = env!("CARGO_BIN_EXE_suPiposition");
    let run_cli = |cal_arg: &str| -> String {
        let out = std::process::Command::new(bin)
            .args([
                "100000000",
                "--plan-only",
                "--no-vram",
                "--profile",
                "8,16,32K,512K,32M,64G",
                "--calibration",
                cal_arg,
            ])
            .output()
            .expect("spawn suPiposition --plan-only");
        assert!(
            out.status.success(),
            "CLI --plan-only --calibration failed: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).expect("utf8 stdout")
    };

    let fast_1 = run_cli("5G,180G,1,3G,32,128");
    let fast_2 = run_cli("5G,180G,1,3G,32,128");
    assert_eq!(
        fast_1, fast_2,
        "--profile + --calibration --plan-only must be 100% deterministic across invocations"
    );

    let slow = run_cli("1G,30G,0,1G,16,64");
    let parse_planned_sec = |s: &str| -> f64 {
        let idx = s.find("planned_sec=").expect("planned_sec in output") + "planned_sec=".len();
        s[idx..].split_whitespace().next().unwrap().parse::<f64>().unwrap()
    };
    let fast_sec = parse_planned_sec(&fast_1);
    let slow_sec = parse_planned_sec(&slow);
    assert!(
        fast_sec < slow_sec,
        "Faster --calibration figures must strictly reduce planned_sec ({fast_sec} vs {slow_sec})"
    );
}

#[test]
fn gate2_section4_unbounded_workers_multi_ccd_l3_and_fft_beyond_32k() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let cal = Calibration::synthetic(4.0e9, 200.0e9, true, Vec::new());

    // 1. No `4,096` worker ceiling: 8,192 mapped threads materialize in full, and near-2^64
    //    synthetic profiles preserve exact `worker_count` and build plans without overflow.
    let prof_8k = Profile::synthetic(
        4_096,
        8_192,
        32 * 1024,
        512 * 1024,
        32 * 1024 * 1024 * 64,
        512 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let shares_8k = Shares::from_profile(&prof_8k);
    assert_eq!(shares_8k.worker_count, 8_192);
    assert_eq!(shares_8k.active_workers, 8_192);
    assert_eq!(shares_8k.workers.len(), 8_192);
    let plan_8k = Plan::build(
        1_000_000,
        OutputFormat::Text,
        &shares_8k,
        &cal,
        PlanOverrides::default(),
    )
    .expect("Plan::build on 8,192 workers");
    assert!(plan_8k.planned_total_sec.is_finite() && plan_8k.planned_total_sec > 0.0);

    let prof_huge = Profile::synthetic(
        u64::MAX / 2,
        u64::MAX - 1,
        64 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        u64::MAX / 2,
        &[],
        &[],
    );
    let shares_huge = Shares::from_profile(&prof_huge);
    assert_eq!(shares_huge.worker_count, u64::MAX - 1);
    let plan_huge = Plan::build(
        1_000_000,
        OutputFormat::Text,
        &shares_huge,
        &cal,
        PlanOverrides::default(),
    )
    .expect("Plan::build on near-2^64 synthetic profile");
    assert!(plan_huge.planned_total_sec.is_finite() && plan_huge.planned_total_sec > 0.0);

    // 2. Multi-CCD L3 domains (`12 x 32 MiB` EPYC vs `1 x 384 MiB` unified L3): identical
    //    `total_l3` must NOT fool single-domain or misaligned working-set checks!
    let epyc_prof = Profile::synthetic_multi_domain(
        96,
        192,
        12,
        48 * 1024,
        1024 * 1024,
        32 * 1024 * 1024,
        512 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let uni_prof = Profile::synthetic(
        96,
        192,
        48 * 1024,
        1024 * 1024,
        384 * 1024 * 1024,
        512 * 1024 * 1024 * 1024,
        &[],
        &[],
    );
    let epyc_sh = Shares::from_profile(&epyc_prof);
    let uni_sh = Shares::from_profile(&uni_prof);
    assert_eq!(epyc_sh.total_l3, uni_sh.total_l3);
    assert_eq!(epyc_sh.num_l3_domains, 12);
    assert_eq!(epyc_sh.min_l3_domain_size, 32 * 1024 * 1024);
    assert_eq!(uni_sh.num_l3_domains, 1);
    assert_eq!(uni_sh.min_l3_domain_size, 384 * 1024 * 1024);
    assert!(
        !epyc_sh.fits_l3_domains(1, 64 * 1024 * 1024, 1, 7, 8)
            && uni_sh.fits_l3_domains(1, 64 * 1024 * 1024, 1, 7, 8),
        "A 64 MiB 1D working set spills a 32 MiB CCD L3 but fits a 384 MiB unified L3"
    );
    // Pricing a 2D transform with `rows == 4` across 192 workers (`conc_transforms == 1`):
    // 4 rows can touch at most 4 of the 12 CCDs (`128 MiB` effective L3) vs `384 MiB` unified L3!
    assert_eq!(epyc_sh.effective_l3_bytes(1, 4), 128 * 1024 * 1024);
    assert_eq!(uni_sh.effective_l3_bytes(1, 4), 384 * 1024 * 1024);
    let (_, price_epyc) = choose_product_method_and_price(
        64_000_000,
        64_000_000,
        false,
        true,
        192,
        &epyc_sh,
        &cal,
        256 * 1024 * 1024 * 1024,
        &PlanOverrides::default(),
    );
    let (_, price_uni) = choose_product_method_and_price(
        64_000_000,
        64_000_000,
        false,
        true,
        192,
        &uni_sh,
        &cal,
        256 * 1024 * 1024 * 1024,
        &PlanOverrides::default(),
    );
    assert_ne!(
        price_epyc.total_sec.to_bits(),
        price_uni.total_sec.to_bits(),
        "Planner pricing must distinguish 12 x 32 MiB CCD L3 from 1 x 384 MiB unified L3"
    );

    // 3. No hardcoded `n_complex <= 32_768` FFT ceiling when L2 share holds larger complex transforms.
    let (n_big_fft, cb_big_fft) =
        engine::fft::choose_fft_params(2_000_000, false, 8 * 1024 * 1024)
            .expect("8 MiB L2 share must admit n_complex = 65,536 FFT");
    assert_eq!(n_big_fft, 65_536);
    assert!(cb_big_fft >= 16);
    kernels.configure_from_shares(&uni_sh);
    let fft_cache = FftTableCache::default();
    let a_big = vec![0x9E37_79B9_7F4A_7C15u64; 14_000];
    let b_big = vec![0xD1B5_4A32_D192_ED03u64; 14_000];
    let (prod_big, w0_big, err_big) =
        engine::fft::fft_mul(&kernels, &fft_cache, &a_big, &b_big, n_big_fft, cb_big_fft)
            .expect("65,536-point complex FFT product must stay within FFT_MAX_ERR_BOUND");
    assert!(err_big <= 0.25, "65,536-point FFT rounding error={err_big}");
    let mut dummy = [0u64; 3];
    let exp_w0 =
        unsafe { (kernels.active.check_n)(prod_big.as_ptr(), prod_big.len() as u64, &mut dummy, 0) };
    assert_eq!(w0_big, exp_w0);

    // 4. End-to-end `engine::run` across tiny-cache vs large-cache synthetic profiles drives
    //    distinct `Kernels` cache sub-block sizes and produces identical verified digits!
    let digits = 20_000u64;
    let out_len = (digits as usize) + 2;
    let tiny_prof = Profile::synthetic(
        2,
        2,
        16 * 1024,
        64 * 1024,
        512 * 1024,
        128 * 1024 * 1024,
        &[],
        &[],
    );
    let large_prof = Profile::synthetic(
        2,
        2,
        64 * 1024,
        2 * 1024 * 1024,
        32 * 1024 * 1024,
        256 * 1024 * 1024,
        &[],
        &[],
    );
    let fac_tiny = MemFactory::new(out_len);
    let mut cfg_tiny =
        EngineConfig::new(digits, OutputFormat::Text, "tiny_c", tiny_prof, &kernels, &fac_tiny);
    cfg_tiny.supplied_calibration = Some(cal.clone());
    let rep_tiny = run(&cfg_tiny).expect("tiny-cache run");
    let tiny_blocks = (
        kernels.fft_sub_n(),
        kernels.ntt_l1_sub_n.load(Ordering::Relaxed),
        kernels.ntt_l2_sub_n.load(Ordering::Relaxed),
        kernels.ntt_cols_l1_sub_rows.load(Ordering::Relaxed),
        kernels.ntt_cols_l2_sub_rows.load(Ordering::Relaxed),
    );
    assert_eq!(tiny_blocks, (512, 1024, 4096, 128, 512));

    let fac_large = MemFactory::new(out_len);
    let mut cfg_large =
        EngineConfig::new(digits, OutputFormat::Text, "large_c", large_prof, &kernels, &fac_large);
    cfg_large.supplied_calibration = Some(cal);
    let rep_large = run(&cfg_large).expect("large-cache run");
    let large_blocks = (
        kernels.fft_sub_n(),
        kernels.ntt_l1_sub_n.load(Ordering::Relaxed),
        kernels.ntt_l2_sub_n.load(Ordering::Relaxed),
        kernels.ntt_cols_l1_sub_rows.load(Ordering::Relaxed),
        kernels.ntt_cols_l2_sub_rows.load(Ordering::Relaxed),
    );
    assert_eq!(large_blocks, (2048, 4096, 131_072, 512, 16_384));
    assert_eq!(rep_tiny.dec_hash_w0, rep_large.dec_hash_w0);
    assert_eq!(
        *fac_tiny.sink.buf.lock().unwrap(),
        *fac_large.sink.buf.lock().unwrap()
    );
}

#[test]
fn gate2_section3_store_tiled_sweeps_parked_numbers_and_member_free() {
    use engine::mul::PreparedParkWrite;
    use engine::plan::choose_grid_shape;

    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);

    // 1. Real `ExtentMember::free` and coalescing span reuse on BOTH striped `TestMember`s and a
    //    real OS SSD file member (`> 3x` total capacity allocated & freed without `OutOfSpace`).
    let tm0 = Arc::new(TestMember::new("free-0", MemberKind::Ssd, 1024 * 1024));
    let tm1 = Arc::new(TestMember::new("free-1", MemberKind::Ssd, 1024 * 1024));
    let stripe_store = Store::new(form_stripes(
        vec![
            (Arc::clone(&tm0) as Arc<dyn ExtentMember>, 4.0e9),
            (Arc::clone(&tm1) as Arc<dyn ExtentMember>, 4.0e9),
        ],
        512,
    ));
    let tel_free = Telemetry::default();
    for iter in 0..10u64 {
        let ext = stripe_store
            .allocate(512 * 1024)
            .expect("freed stripe spans must be reused across >2.5x capacity");
        let words: Vec<u64> = (0..(512 * 1024 / 8))
            .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ iter)
            .collect();
        stripe_store.write_all(&ext, &words, &kernels).unwrap();
        let mut back = vec![0u64; words.len()];
        stripe_store
            .read_all(&ext, &mut back, &kernels, &tel_free)
            .unwrap();
        assert_eq!(back, words);
        stripe_store.free(ext);
        assert_eq!(tm0.live_allocated_bytes(), 0);
        assert_eq!(tm1.live_allocated_bytes(), 0);
    }

    let ssd_tmp = std::env::temp_dir().join(format!(
        "supi_sec3_ssd_free_{}.bin",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&ssd_tmp);
    {
        let (_entry, ssd_mem) =
            platform::open_ssd_path(&ssd_tmp, Some(1024 * 1024)).expect("open 1 MiB OS SSD member");
        let os_store = Store::new(form_stripes(vec![(ssd_mem, 4.0e9)], 512));
        for iter in 0..8u64 {
            let ext = os_store
                .allocate(384 * 1024)
                .expect("OS SSD member must reclaim freed spans across >3x capacity");
            let words: Vec<u64> = (0..(384 * 1024 / 8))
                .map(|i| (i as u64).wrapping_add(iter * 17 + 3))
                .collect();
            os_store.write_all(&ext, &words, &kernels).unwrap();
            let mut back = vec![0u64; words.len()];
            os_store
                .read_all(&ext, &mut back, &kernels, &tel_free)
                .unwrap();
            assert_eq!(back, words);
            os_store.free(ext);
        }
    }
    let _ = std::fs::remove_file(&ssd_tmp);

    // 2. Planner `choose_grid_shape` (Section 10 line 289): when one RAM tile of the store cannot
    //    hold `rows * min_store_range_bytes`, outer `rows` shrinks and `cols` adds `extra_sweeps`.
    let (rows_roomy, _cols_roomy, _, sweeps_roomy) = choose_grid_shape(
        1 << 20,
        512 * 1024,
        32 * 1024 * 1024,
        true,
        8,
        6,
        48,
        false,
        false,
        512 * 1024 * 1024,
        262_144,
    );
    let (rows_tight, cols_tight, _, sweeps_tight) = choose_grid_shape(
        1 << 20,
        512 * 1024,
        32 * 1024 * 1024,
        true,
        8,
        6,
        48,
        false,
        false,
        4 * 1024 * 1024,
        262_144,
    );
    assert!(
        rows_tight <= 8 && rows_tight < rows_roomy && sweeps_tight > sweeps_roomy,
        "Tight RAM store tile must shrink outer rows ({rows_roomy} -> {rows_tight}, cols={cols_tight}) and add inner sweeps ({sweeps_roomy} -> {sweeps_tight})"
    );

    // 3. Direct `ParkedBigNum` Loader roundtrip + Rung-1 single-limb repair & write-back to member!
    let park_tm = Arc::new(TestMember::new("park-tm", MemberKind::Vram, 2 * 1024 * 1024));
    let park_store = Store::new(form_stripes(
        vec![(Arc::clone(&park_tm) as Arc<dyn ExtentMember>, 20.0e9)],
        64,
    ));
    let park_prof = Profile::synthetic(
        2,
        2,
        32 * 1024,
        256 * 1024,
        2 * 1024 * 1024,
        64 * 1024 * 1024,
        &[2 * 1024 * 1024],
        &[],
    );
    let park_sh = Shares::from_profile(&park_prof);
    let park_sched = Scheduler::new(&park_sh, None);
    let fft_c = FftTableCache::default();
    let ntt_c = NttTableCache::default();
    let park_lad = Ladder::default();
    let park_tel = Telemetry::default();
    let park_ctx = MulCtx::new(
        &kernels,
        &fft_c,
        &ntt_c,
        &park_sched,
        Some(&park_store),
        &park_lad,
        &park_tel,
    );
    let orig_limbs: Vec<u64> = (0..200u64)
        .map(|i| i.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x55AA_1234)
        .collect();
    let expected_bn = BigNum::from_limbs(orig_limbs.clone(), true, 19, &kernels);
    let mut to_park = BigNum::from_limbs(orig_limbs.clone(), true, 19, &kernels);
    let prep = PreparedParkWrite::prepare_bignum(&park_ctx, 0x7701, &mut to_park, 2)
        .expect("prepare_bignum")
        .expect("Store has space");
    assert!(to_park.is_zero(), "Parking must release host BigNum limbs immediately");
    let (parked_bn, write_job) = prep.split_write_job();
    park_store
        .submit_loader_scoped(|| write_job.execute(&park_store))
        .wait()
        .expect("Loader write parked BigNum");
    assert!(park_store.parked_live_bytes() >= 200 * 8);

    // Flip 1 limb (word #2 at byte offset 16) in piece 0 on the backing `TestMember`!
    park_tm.corrupt_u64_at(16, 0xDEAD_BEEF_CAFE_0001);
    let recovered_bn = parked_bn
        .unpark_bignum_sync(&park_ctx, 0x7701, 2)
        .expect("Rung-1 repair on parked BigNum piece");
    assert_eq!(recovered_bn.limbs.as_slice(), expected_bn.limbs.as_slice());
    assert_eq!(recovered_bn.neg, expected_bn.neg);
    assert_eq!(recovered_bn.shift, expected_bn.shift);
    assert_eq!(recovered_bn.w0(), expected_bn.w0());
    assert_eq!(park_tel.corrections.load(Ordering::Relaxed), 1);
    assert_eq!(park_store.parked_live_bytes(), 0);
    assert_eq!(park_tm.live_allocated_bytes(), 0);
    // Verify the repaired limb was written back to the storage medium before release!
    let mut raw_check = [0u8; 24];
    park_tm
        .read(
            &[ByteRange {
                device_offset: 0,
                host_offset: 0,
                len: 24,
            }],
            &mut raw_check,
        )
        .unwrap();
    let repaired_word2 = u64::from_le_bytes(raw_check[16..24].try_into().unwrap());
    assert_eq!(
        repaired_word2, orig_limbs[2],
        "Rung-1 piece repair must write the fixed piece back to the member"
    );

    // 4. End-to-end Store-tiled out-of-core `engine::run` at `120,000` digits with a `1 MiB` RAM
    //    profile (`ram_transform_budget <= 512 KiB`, smaller than full prime grids!), 2-member
    //    striping, overlapped `Loader` batches, parked live numbers, AND Rung-1 piece repair!
    struct SingleSweepBitFlipMember {
        inner: TestMember,
        flipped: AtomicBool,
    }
    impl ExtentMember for SingleSweepBitFlipMember {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn kind(&self) -> MemberKind {
            MemberKind::Ssd
        }
        fn alignment(&self) -> u64 {
            4096
        }
        fn capacity(&self) -> u64 {
            self.inner.capacity()
        }
        fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
            self.inner.allocate(bytes)
        }
        fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
            self.inner.read(ranges, host_buf)?;
            // Corrupt a single limb once on a Store-backed extent read past the checkpoint bank!
            if ranges.iter().any(|r| r.device_offset >= 2 * 1024 * 1024 && r.len >= 64)
                && !self.flipped.swap(true, Ordering::Relaxed)
                && host_buf.len() >= 16
            {
                host_buf[8] ^= 0x80;
            }
            Ok(())
        }
        fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
            self.inner.write(ranges, host_buf)
        }
        fn free(&self, device_offset: u64, bytes: u64) {
            self.inner.free(device_offset, bytes);
        }
    }

    let digits = 120_000u64;
    let out_len = (digits as usize) + 2;
    let ref_prof = Profile::synthetic(
        2,
        2,
        32 * 1024,
        512 * 1024,
        4 * 1024 * 1024,
        128 * 1024 * 1024,
        &[],
        &[],
    );
    let ref_fac = MemFactory::new(out_len);
    let ref_rep = run(&EngineConfig::new(
        digits,
        OutputFormat::Text,
        "sec3_ref",
        ref_prof,
        &kernels,
        &ref_fac,
    ))
    .expect("in-RAM 120k reference");
    let expected_pi = ref_fac.sink.buf.lock().unwrap().clone();

    let ooc_prof = Profile::synthetic(
        2,
        2,
        16 * 1024,
        64 * 1024,
        0,
        1024 * 1024,
        &[],
        &[32 * 1024 * 1024, 32 * 1024 * 1024],
    );
    let mut ooc_cal = Calibration::synthetic(4.0e9, 60.0e9, true, Vec::new());
    ooc_cal.basecase_max_limbs = 8;
    ooc_cal.karatsuba_max_limbs = 16;
    let mut rates = HashMap::new();
    rates.insert("sec3-ssd0".to_string(), 8.0e9);
    rates.insert("sec3-ssd1".to_string(), 8.0e9);

    let m0 = Arc::new(SingleSweepBitFlipMember {
        inner: TestMember::new("sec3-ssd0", MemberKind::Ssd, 32 * 1024 * 1024),
        flipped: AtomicBool::new(false),
    });
    let m1 = Arc::new(TestMember::new("sec3-ssd1", MemberKind::Ssd, 32 * 1024 * 1024));
    let ooc_fac = MemFactory::new(out_len);
    let mut ooc_cfg = EngineConfig::new(digits, OutputFormat::Text, "sec3_ooc", ooc_prof, &kernels, &ooc_fac);
    ooc_cfg.supplied_calibration = Some(ooc_cal);
    ooc_cfg.supplied_member_rates = Some(rates);
    ooc_cfg.members = vec![
        Arc::clone(&m0) as Arc<dyn ExtentMember>,
        Arc::clone(&m1) as Arc<dyn ExtentMember>,
    ];

    let ooc_rep = run(&ooc_cfg).expect("Store-tiled out-of-core 120k run");
    assert_eq!(ooc_rep.dec_hash_w0, ref_rep.dec_hash_w0);
    assert_eq!(*ooc_fac.sink.buf.lock().unwrap(), expected_pi);
    assert!(
        ooc_rep.store_loader_batches > 0,
        "Out-of-core sweeps and parked numbers must drive overlapped Loader batches"
    );
    assert!(
        ooc_rep.store_parked_peak_bytes > 0 && ooc_rep.store_parked_live_bytes == 0,
        "Live numbers must be parked in Store during the run and freed to 0 by teardown: peak={}, live={}",
        ooc_rep.store_parked_peak_bytes,
        ooc_rep.store_parked_live_bytes
    );
    assert!(
        ooc_rep.store_tile_peak_bytes > 0
            && ooc_rep.store_tile_peak_bytes <= ooc_rep.plan.ram_transform_budget
            && ooc_rep.store_tile_peak_bytes <= 512 * 1024,
        "Out-of-core NTT sweeps must stay within the RAM tile budget rather than allocating whole grids: tile_peak={} vs budget={}",
        ooc_rep.store_tile_peak_bytes,
        ooc_rep.plan.ram_transform_budget
    );
    assert!(
        m0.flipped.load(Ordering::Relaxed)
            && !ooc_rep.telemetry_summary.contains("corrections=0")
            && ooc_rep.rollbacks == 0,
        "Injected single-limb Store read error must be repaired in place at Rung 1 with 0 rollbacks: {}",
        ooc_rep.telemetry_summary
    );
}

/// Described machines far larger than the one the tests run on (hundreds of workers, several
/// last-level caches and memory nodes): the run is planned for each of them, started with that
/// many workers, and has to write the digits that the run planned for the detected machine writes.
#[test]
fn gate2_described_large_machines_write_the_same_digits() {
    let bin = env!("CARGO_BIN_EXE_suPiposition");
    let digits = if cfg!(debug_assertions) {
        "1000000"
    } else {
        "20000000"
    };
    let dir = std::env::temp_dir();
    let hash_of = |tag: &str, profile: Option<&str>| -> String {
        let out_path = dir.join(format!("supi_gate2_large_{}_{tag}.txt", std::process::id()));
        let mut args: Vec<String> = vec![
            digits.to_string(),
            "--no-vram".to_string(),
            "-o".to_string(),
            out_path.to_string_lossy().into_owned(),
        ];
        if let Some(p) = profile {
            args.extend([
                "--profile".to_string(),
                p.to_string(),
                "--calibration".to_string(),
                "5G,180G,1,3G,32,128".to_string(),
            ]);
        }
        let out = std::process::Command::new(bin)
            .args(&args)
            .output()
            .expect("spawn suPiposition");
        let _ = std::fs::remove_file(&out_path);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "the run for {tag} failed: {text}");
        let at = text
            .find("Dec Hash = ")
            .unwrap_or_else(|| panic!("no hash in the output of {tag}: {text}"))
            + "Dec Hash = ".len();
        text[at..].split_whitespace().next().unwrap().to_string()
    };
    let here = hash_of("detected", None);
    if digits == "1000000" {
        assert_eq!(here, "404089929205932130", "the first million digits");
    }
    for (tag, profile) in [
        ("two_sockets_384_threads_24_caches", "192,384,48K,1M,24x32M,1T"),
        ("four_tiles_256_cores", "256,256,48K,2M,4x320M,1T"),
        ("1024_threads_64_caches", "512,1024,48K,1M,64x32M,4T"),
    ] {
        assert_eq!(
            hash_of(tag, Some(profile)),
            here,
            "digits under the described machine {tag}"
        );
    }
}

/// A merge whose two P were lent out for a stolen `P1 * P2` has to keep the form that the lending
/// was decided on. On this described machine (256 one-thread cores under four last-level caches)
/// one merge of a run to a hundred million digits once chose its form a second time with the
/// right P gone, ran the fused form with the empty left P, and the run wrote wrong digits with
/// every check in order.
#[test]
fn gate2_a_merge_with_its_p_lent_out_keeps_its_form() {
    if cfg!(debug_assertions) {
        // A hundred million digits: a test for the optimized build.
        return;
    }
    let bin = env!("CARGO_BIN_EXE_suPiposition");
    let out_path =
        std::env::temp_dir().join(format!("supi_gate2_lent_p_{}.txt", std::process::id()));
    let out = std::process::Command::new(bin)
        .args([
            "100000000",
            "--no-vram",
            "--profile",
            "256,256,48K,2M,4x320M,1T",
            "--calibration",
            "5G,180G,1,3G,32,128",
            "-o",
        ])
        .arg(&out_path)
        .output()
        .expect("spawn suPiposition");
    let _ = std::fs::remove_file(&out_path);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "the run failed: {text}");
    assert!(
        text.contains("Dec Hash = 199405406068015690"),
        "the first hundred million digits under the described machine: {text}"
    );
}
