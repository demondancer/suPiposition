//! Gate 3 — Resumable (`DESIGN.md` Section 11 & Section 15.3):
//! - Stop at every checkpoint stage (`Series`, `Finish`, `Convert`) and resume in a new OS process
//!   with SSD members backed by real files (`Text` and `Packed19`).
//! - Resume under the same profile/calibration and under changed ones, including changed SSD/VRAM
//!   rates that regroup stripes.
//! - Resume with the file of any one of 2+ SSDs removed (and remaining SSD files renamed & passed
//!   in reversed order), reconstructing lost checkpoint pieces via `add_n` / `sub_n` parity.
//! - Resume cleanly after stopping mid-record write (`--truncate-record-once`) or mid-output write.
//! - Verify every startup case of Section 11: record from another run (with and without
//!   `--discard`), wrong `format_version`, records where none is whole (with and without
//!   `--discard`), missing output file, shortened output file, unreadable checkpoint beyond parity,
//!   and persisted excluded devices across resume.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use engine::check::{CheckpointRecord, RECORD_FORMAT_VERSION, StageRecord};
use engine::plan::{Calibration, OutputFormat};
use engine::radix::DigitSink;
use engine::refk::Kernels;
use engine::store::{ByteRange, ExtentMember, MemberKind, TestFault, TestMember};
use engine::tier::Profile;
use engine::{
    CheckpointStageKind, EngineConfig, EngineError, OutputSinkFactory, RECORD_REGION_BYTES,
    RECORD_SLOT_BYTES, StopCondition, run,
};

struct TempTestDir {
    path: PathBuf,
}

impl TempTestDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("supi_gate3_{tag}_{pid}_{nanos}"));
        fs::create_dir_all(&p).unwrap();
        Self { path: p }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn run_cli(args: &[&str]) -> Output {
    let bin = env!("CARGO_BIN_EXE_suPiposition");
    Command::new(bin)
        .args(args)
        .output()
        .expect("Failed to spawn suPiposition binary")
}

#[test]
fn gate3_stop_and_resume_each_checkpoint_kind_new_process() {
    let dir = TempTestDir::new("stages");
    let digits = "10000";

    // 1. Clean straight-through baselines (`Text` and `Packed19`).
    let base_txt = dir.file("base.txt");
    let out_bt = run_cli(&[digits, "-o", base_txt.to_str().unwrap(), "--no-vram"]);
    assert!(
        out_bt.status.success(),
        "Baseline text run failed: {}",
        String::from_utf8_lossy(&out_bt.stderr)
    );
    let expected_txt = fs::read(&base_txt).unwrap();
    assert_eq!(expected_txt.len(), 10_002);

    let base_p19 = dir.file("base.p19");
    let out_bp = run_cli(&[
        digits,
        "-o",
        base_p19.to_str().unwrap(),
        "--format",
        "packed19",
        "--no-vram",
    ]);
    assert!(
        out_bp.status.success(),
        "Baseline packed19 run failed: {}",
        String::from_utf8_lossy(&out_bp.stderr)
    );
    let expected_p19 = fs::read(&base_p19).unwrap();

    // 2. Stop & resume in a new process at each checkpoint stage (`series`, `finish`, `convert`).
    for stage in ["series", "finish", "convert"] {
        let out_file = dir.file(&format!("pi_{stage}.txt"));
        let ssd0 = dir.file(&format!("ssd0_{stage}.bin"));
        let ssd1 = dir.file(&format!("ssd1_{stage}.bin"));
        let s0_spec = format!("{}@8M", ssd0.display());
        let s1_spec = format!("{}@8M", ssd1.display());

        // Process 1: run until first checkpoint of `stage`, then stop cleanly (exit code 75).
        let p1 = run_cli(&[
            digits,
            "-o",
            out_file.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_spec,
            "--ssd",
            &s1_spec,
            "--checkpoint-mult",
            "0.0",
            "--stop-after-stage",
            stage,
        ]);
        assert_eq!(
            p1.status.code(),
            Some(75),
            "Stage '{stage}' process 1 expected exit 75, got {:?}\nstdout: {}\nstderr: {}",
            p1.status.code(),
            String::from_utf8_lossy(&p1.stdout),
            String::from_utf8_lossy(&p1.stderr)
        );
        let p1_stdout = String::from_utf8_lossy(&p1.stdout);
        assert!(
            p1_stdout.contains("STOPPED_FOR_TEST"),
            "Expected STOPPED_FOR_TEST marker: {p1_stdout}"
        );

        // Output file must already exist at full preallocated size (`10,002` bytes).
        let mid_bytes = fs::read(&out_file).unwrap();
        assert_eq!(mid_bytes.len(), expected_txt.len());
        if stage == "convert" {
            // In `Convert`, the left subtree digits (`"3.14159..."`) were flushed before checkpoint,
            // while the trailing leaves remain unwritten zeros!
            assert_eq!(&mid_bytes[..64], &expected_txt[..64]);
            assert_eq!(mid_bytes[mid_bytes.len() - 1], 0u8);
        }

        // Process 2: resume in a brand-new OS process and run to completion!
        let p2 = run_cli(&[
            digits,
            "-o",
            out_file.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_spec,
            "--ssd",
            &s1_spec,
        ]);
        assert!(
            p2.status.success(),
            "Stage '{stage}' process 2 resume failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&p2.stdout),
            String::from_utf8_lossy(&p2.stderr)
        );
        let p2_stdout = String::from_utf8_lossy(&p2.stdout);
        assert!(
            p2_stdout.contains("resumed=Some("),
            "Process 2 must report resumed serial: {p2_stdout}"
        );
        assert_eq!(
            fs::read(&out_file).unwrap(),
            expected_txt,
            "Resumed digits after '{stage}' checkpoint differed from baseline!"
        );

        // Finishing a run must clear its SSD record regions (`[0..131072)` all zeros).
        for ssd_path in [&ssd0, &ssd1] {
            let raw = fs::read(ssd_path).unwrap();
            assert!(
                raw[..(RECORD_REGION_BYTES as usize)].iter().all(|&b| b == 0),
                "SSD record region on {} was not cleared after completion!",
                ssd_path.display()
            );
        }
    }

    // Also test `Packed19` stop & resume across `convert`!
    let out_p19 = dir.file("pi_conv.p19");
    let ssd0_p = dir.file("ssd0_p19.bin");
    let ssd1_p = dir.file("ssd1_p19.bin");
    let sp0 = format!("{}@8M", ssd0_p.display());
    let sp1 = format!("{}@8M", ssd1_p.display());
    let pp1 = run_cli(&[
        digits,
        "-o",
        out_p19.to_str().unwrap(),
        "--format",
        "packed19",
        "--no-vram",
        "--ssd",
        &sp0,
        "--ssd",
        &sp1,
        "--checkpoint-mult",
        "0.0",
        "--stop-after-stage",
        "convert",
    ]);
    assert_eq!(pp1.status.code(), Some(75));
    let pp2 = run_cli(&[
        digits,
        "-o",
        out_p19.to_str().unwrap(),
        "--format",
        "packed19",
        "--no-vram",
        "--ssd",
        &sp0,
        "--ssd",
        &sp1,
    ]);
    assert!(pp2.status.success());
    assert_eq!(fs::read(&out_p19).unwrap(), expected_p19);
}

#[test]
fn gate3_resume_under_changed_profile_and_regrouped_stripes() {
    let dir = TempTestDir::new("changed_prof");
    let digits = "12000";

    let base_out = dir.file("base.txt");
    assert!(run_cli(&[digits, "-o", base_out.to_str().unwrap(), "--no-vram"]).status.success());
    let expected = fs::read(&base_out).unwrap();

    // 1. Multi-process: start with 4-core roomy L1/L2/RAM profile, stop at `series`, and resume
    //    under a 1-core tiny-L1/tiny-RAM out-of-core profile!
    for stage in ["series", "finish", "convert"] {
        let out_f = dir.file(&format!("prof_{stage}.txt"));
        let s0 = format!("{}@8M", dir.file(&format!("s0_{stage}.bin")).display());
        let s1 = format!("{}@8M", dir.file(&format!("s1_{stage}.bin")).display());

        let r1 = run_cli(&[
            digits,
            "-o",
            out_f.to_str().unwrap(),
            "--no-vram",
            "--profile",
            "4,8,32K,1M,16M,64M",
            "--ssd",
            &s0,
            "--ssd",
            &s1,
            "--checkpoint-mult",
            "0.0",
            "--stop-after-stage",
            stage,
        ]);
        assert_eq!(r1.status.code(), Some(75));

        let r2 = run_cli(&[
            digits,
            "-o",
            out_f.to_str().unwrap(),
            "--no-vram",
            "--profile",
            "1,1,4K,8K,0,64K",
            "--ssd",
            &s0,
            "--ssd",
            &s1,
        ]);
        assert!(
            r2.status.success(),
            "Resume under changed profile failed at '{stage}': {}",
            String::from_utf8_lossy(&r2.stderr)
        );
        assert_eq!(fs::read(&out_f).unwrap(), expected);
    }

    // 2. Regrouped stripe rates + changed calibration figures across resume:
    //    Run 1 stripes 3 SSDs at equal 5 GB/s; Run 2 changes rates to `(9 GB/s, 9 GB/s, 1 GB/s)`
    //    so only 2 stripe together and the 3rd ranks apart, with tiny RAM so `Store` uses the
    //    newly regrouped stripes while reading the 3-SSD checkpoint!
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let mem_ssd0 = Arc::new(TestMember::new("ssd-a", MemberKind::Ssd, 8 * 1024 * 1024));
    let mem_ssd1 = Arc::new(TestMember::new("ssd-b", MemberKind::Ssd, 8 * 1024 * 1024));
    let mem_ssd2 = Arc::new(TestMember::new("ssd-c", MemberKind::Ssd, 8 * 1024 * 1024));

    let shared_disk = dir.file("regroup.txt");
    let out_fac = platform::make_output_factory(&shared_disk);

    let prof1 = Profile::synthetic(
        4,
        4,
        32 * 1024,
        512 * 1024,
        4 * 1024 * 1024,
        64 * 1024 * 1024,
        &[],
        &[8 * 1024 * 1024; 3],
    );
    let mut rates1 = HashMap::new();
    rates1.insert("ssd-a".to_string(), 5.0e9);
    rates1.insert("ssd-b".to_string(), 5.0e9);
    rates1.insert("ssd-c".to_string(), 5.0e9);

    let mut cfg1 = EngineConfig::new(
        12_000,
        OutputFormat::Text,
        "regroup_id",
        prof1,
        &kernels,
        out_fac.as_ref(),
    );
    cfg1.members = vec![
        Arc::clone(&mem_ssd0) as Arc<dyn ExtentMember>,
        Arc::clone(&mem_ssd1) as Arc<dyn ExtentMember>,
        Arc::clone(&mem_ssd2) as Arc<dyn ExtentMember>,
    ];
    cfg1.supplied_member_rates = Some(rates1);
    cfg1.supplied_calibration = Some(Calibration::synthetic(3.0e9, 60.0e9, true, Vec::new()));
    cfg1.overrides.checkpoint_mult = 0.0;
    cfg1.stop_condition = Some(StopCondition::AfterStageCheckpoint(
        CheckpointStageKind::Series,
    ));
    let stop_err = run(&cfg1).unwrap_err();
    assert!(matches!(stop_err, EngineError::StoppedForTest { .. }));

    // Resume with 2 cores, tiny 16 KiB RAM (forcing out-of-core Store), changed calibration, and
    // skewed SSD rates `(9 GB/s, 9 GB/s, 1 GB/s)` + reversed member order!
    let prof2 = Profile::synthetic(
        2,
        2,
        4 * 1024,
        8 * 1024,
        0,
        16 * 1024,
        &[],
        &[8 * 1024 * 1024; 3],
    );
    let mut rates2 = HashMap::new();
    rates2.insert("ssd-a".to_string(), 9.0e9);
    rates2.insert("ssd-b".to_string(), 9.0e9);
    rates2.insert("ssd-c".to_string(), 1.0e9);

    let mut cfg2 = EngineConfig::new(
        12_000,
        OutputFormat::Text,
        "regroup_id",
        prof2,
        &kernels,
        out_fac.as_ref(),
    );
    cfg2.members = vec![mem_ssd2, mem_ssd0, mem_ssd1];
    cfg2.supplied_member_rates = Some(rates2);
    cfg2.supplied_calibration = Some(Calibration::synthetic(1.2e9, 18.0e9, false, Vec::new()));
    let rep2 = run(&cfg2).expect("Resume with regrouped stripes");
    assert_eq!(rep2.resumed_from_serial, Some(1));
    assert_eq!(fs::read(&shared_disk).unwrap(), expected);
}

#[test]
fn gate3_resume_with_any_one_ssd_file_removed_and_reordered_names() {
    let dir = TempTestDir::new("lost_ssd");
    let digits = "10000";

    let base_out = dir.file("base.txt");
    assert!(run_cli(&[digits, "-o", base_out.to_str().unwrap(), "--no-vram"]).status.success());
    let expected = fs::read(&base_out).unwrap();

    // Test removing EACH of the 3 SSDs (`col = 0` data, `col = 1` data, `col = 2` `add_n` parity)!
    for remove_idx in 0..3usize {
        let out_f = dir.file(&format!("out_rm{remove_idx}.txt"));
        let ssds: [PathBuf; 3] = [
            dir.file(&format!("c{remove_idx}_0.bin")),
            dir.file(&format!("c{remove_idx}_1.bin")),
            dir.file(&format!("c{remove_idx}_2.bin")),
        ];
        let specs: Vec<String> = ssds.iter().map(|p| format!("{}@8M", p.display())).collect();

        // Process 1: write 3-SSD parity-striped `finish` checkpoint and stop.
        let p1 = run_cli(&[
            digits,
            "-o",
            out_f.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &specs[0],
            "--ssd",
            &specs[1],
            "--ssd",
            &specs[2],
            "--checkpoint-mult",
            "0.0",
            "--stop-after-stage",
            "finish",
        ]);
        assert_eq!(p1.status.code(), Some(75));

        // Delete `ssds[remove_idx]` completely from disk!
        fs::remove_file(&ssds[remove_idx]).unwrap();
        assert!(!ssds[remove_idx].exists());

        // Rename the surviving 2 SSD files and pass them in REVERSED order to Process 2
        // (and for `remove_idx == 1`, also pass a fresh blank replacement file in the middle to
        // verify both 2-arg omission and 3-arg blank replacement!).
        let surv: Vec<usize> = (0..3).filter(|&i| i != remove_idx).collect();
        let ren_hi = dir.file(&format!("renamed_hi_{remove_idx}.bin"));
        let ren_lo = dir.file(&format!("renamed_lo_{remove_idx}.bin"));
        fs::rename(&ssds[surv[1]], &ren_hi).unwrap();
        fs::rename(&ssds[surv[0]], &ren_lo).unwrap();

        let spec_hi = format!("{}@8M", ren_hi.display());
        let spec_lo = format!("{}@8M", ren_lo.display());

        let p2 = if remove_idx == 1 {
            let blank_mid = format!("{}@8M", ssds[1].display());
            run_cli(&[
                digits,
                "-o",
                out_f.to_str().unwrap(),
                "--no-vram",
                "--ssd",
                &spec_hi,
                "--ssd",
                &blank_mid,
                "--ssd",
                &spec_lo,
            ])
        } else {
            run_cli(&[
                digits,
                "-o",
                out_f.to_str().unwrap(),
                "--no-vram",
                "--ssd",
                &spec_hi,
                "--ssd",
                &spec_lo,
            ])
        };

        assert!(
            p2.status.success(),
            "Resume with SSD #{remove_idx} removed failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&p2.stdout),
            String::from_utf8_lossy(&p2.stderr)
        );
        assert_eq!(
            fs::read(&out_f).unwrap(),
            expected,
            "Digits differed when SSD #{remove_idx} was removed!"
        );
    }
}

#[test]
fn gate3_stop_while_writing_record_or_output_file() {
    let dir = TempTestDir::new("partial_writes");
    let digits = "10000";

    let base_out = dir.file("base.txt");
    assert!(run_cli(&[digits, "-o", base_out.to_str().unwrap(), "--no-vram"]).status.success());
    let expected = fs::read(&base_out).unwrap();

    // --- Case A: Interrupted mid-record write (`--truncate-record-once`) falls back to earlier whole record! ---
    let out_rec = dir.file("partial_rec.txt");
    let ssd0 = dir.file("prec_ssd0.bin");
    let ssd1 = dir.file("prec_ssd1.bin");
    let s0 = format!("{}@8M", ssd0.display());
    let s1 = format!("{}@8M", ssd1.display());

    // Step 1: Write valid checkpoint serial=1 (into slot 1) and stop.
    let p1 = run_cli(&[
        digits,
        "-o",
        out_rec.to_str().unwrap(),
        "--no-vram",
        "--ssd",
        &s0,
        "--ssd",
        &s1,
        "--checkpoint-mult",
        "0.0",
        "--stop-after-ckpt",
        "1",
    ]);
    assert_eq!(p1.status.code(), Some(75));

    // Step 2: Resume from serial=1 and stop HALFWAY through writing record serial=2 (slot 0)!
    let p2 = run_cli(&[
        digits,
        "-o",
        out_rec.to_str().unwrap(),
        "--no-vram",
        "--ssd",
        &s0,
        "--ssd",
        &s1,
        "--checkpoint-mult",
        "0.0",
        "--truncate-record-once",
    ]);
    assert_eq!(p2.status.code(), Some(75));

    // Confirm slot 0 on `ssd0` now holds a non-zero truncated record while slot 1 holds serial=1.
    let raw_ssd0 = fs::read(&ssd0).unwrap();
    assert!(
        raw_ssd0[..(RECORD_SLOT_BYTES as usize)]
            .iter()
            .any(|&b| b != 0),
        "Slot 0 must contain partial record bytes"
    );

    // Step 3: Start Process 3 normally -> must ignore partial slot 0, resume from whole serial=1,
    // and produce exact digits!
    let p3 = run_cli(&[
        digits,
        "-o",
        out_rec.to_str().unwrap(),
        "--no-vram",
        "--ssd",
        &s0,
        "--ssd",
        &s1,
    ]);
    assert!(
        p3.status.success(),
        "Process 3 failed after truncated record write:\n{}",
        String::from_utf8_lossy(&p3.stderr)
    );
    let p3_out = String::from_utf8_lossy(&p3.stdout);
    assert!(
        p3_out.contains("resumed=Some(1)"),
        "Expected resume from earlier whole serial=1, got: {p3_out}"
    );
    assert_eq!(fs::read(&out_rec).unwrap(), expected);

    // --- Case B: Interrupted while writing output file after a `Convert` checkpoint! ---
    let out_conv = dir.file("partial_out.txt");
    let cs0 = format!("{}@8M", dir.file("cout_s0.bin").display());
    let cs1 = format!("{}@8M", dir.file("cout_s1.bin").display());
    let cp1 = run_cli(&[
        digits,
        "-o",
        out_conv.to_str().unwrap(),
        "--no-vram",
        "--ssd",
        &cs0,
        "--ssd",
        &cs1,
        "--checkpoint-mult",
        "0.0",
        "--stop-after-stage",
        "convert",
    ]);
    assert_eq!(cp1.status.code(), Some(75));

    // Simulate the process having partially scribbled un-flushed garbage into the STILL-COVERED
    // trailing region of the output file (`[9800..10002)`) right before losing power!
    {
        let mut bytes = fs::read(&out_conv).unwrap();
        for b in &mut bytes[9800..] {
            *b = b'X';
        }
        fs::write(&out_conv, &bytes).unwrap();
    }

    // Resume in Process 2 -> overwrites all leaves still covered by the checkpoint and restores exact digits!
    let cp2 = run_cli(&[
        digits,
        "-o",
        out_conv.to_str().unwrap(),
        "--no-vram",
        "--ssd",
        &cs0,
        "--ssd",
        &cs1,
    ]);
    assert!(cp2.status.success());
    assert_eq!(fs::read(&out_conv).unwrap(), expected);
}

#[test]
fn gate3_every_startup_case_of_section_11() {
    let dir = TempTestDir::new("startup_cases");
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);

    let base_out = dir.file("base5k.txt");
    assert!(run_cli(&["5000", "-o", base_out.to_str().unwrap(), "--no-vram"]).status.success());
    let expected_5k = fs::read(&base_out).unwrap();

    // Helper to create a stopped checkpoint for `5000` digits on `(ssd0, ssd1, out_path)`.
    let make_stopped = |tag: &str| -> (PathBuf, PathBuf, PathBuf, String, String) {
        let out_p = dir.file(&format!("{tag}.txt"));
        let s0_p = dir.file(&format!("{tag}_s0.bin"));
        let s1_p = dir.file(&format!("{tag}_s1.bin"));
        let s0_arg = format!("{}@8M", s0_p.display());
        let s1_arg = format!("{}@8M", s1_p.display());
        let r = run_cli(&[
            "5000",
            "-o",
            out_p.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
            "--checkpoint-mult",
            "0.0",
            "--stop-after-stage",
            "finish",
        ]);
        assert_eq!(r.status.code(), Some(75));
        (out_p, s0_p, s1_p, s0_arg, s1_arg)
    };

    // -------------------------------------------------------------------------
    // Case 1: Record left by another run -> fails without `--discard`, succeeds with `--discard`!
    // -------------------------------------------------------------------------
    {
        let (out_p, _s0_p, _s1_p, s0_arg, s1_arg) = make_stopped("mismatch");

        // Different `<digits>` without `--discard` -> fails!
        let bad_digits = run_cli(&[
            "6000",
            "-o",
            out_p.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!bad_digits.status.success());
        assert!(
            String::from_utf8_lossy(&bad_digits.stderr).contains("belongs to another run"),
            "stderr: {}",
            String::from_utf8_lossy(&bad_digits.stderr)
        );

        // Different `--format` without `--discard` -> fails!
        let bad_fmt = run_cli(&[
            "5000",
            "-o",
            out_p.to_str().unwrap(),
            "--format",
            "packed19",
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!bad_fmt.status.success());

        // Different `-o` without `--discard` -> fails!
        let other_out = dir.file("other_name.txt");
        let bad_out = run_cli(&[
            "5000",
            "-o",
            other_out.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!bad_out.status.success());

        // Same SSDs WITH `--discard` -> discards old record and completes fresh run cleanly!
        let ok_discard = run_cli(&[
            "5000",
            "-o",
            other_out.to_str().unwrap(),
            "--discard",
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(ok_discard.status.success());
        assert_eq!(fs::read(&other_out).unwrap(), expected_5k);
    }

    // -------------------------------------------------------------------------
    // Case 2: Record of another format version -> fails with `WrongRecordVersion`!
    // -------------------------------------------------------------------------
    {
        let ssd_ver = Arc::new(TestMember::new("ssd-ver", MemberKind::Ssd, 4 * 1024 * 1024));
        let wrong_rec = CheckpointRecord {
            format_version: RECORD_FORMAT_VERSION + 9,
            serial: 1,
            ssd_member_id: "0:ssd-ver".to_string(),
            digits: 5000,
            output_path: "ver.txt".to_string(),
            format_text: true,
            piece_limbs: 512,
            stripe_width: 1,
            data_offset_bytes: RECORD_REGION_BYTES,
            excluded_devices: Vec::new(),
            stage: StageRecord::Finish {
                precision_limbs: 64,
                step_index: 0,
            },
            numbers: Vec::new(),
        };
        let enc = wrong_rec.encode(&kernels);
        let mut slot = vec![0u8; RECORD_SLOT_BYTES as usize];
        slot[..enc.len()].copy_from_slice(&enc);
        ssd_ver
            .write(
                &[ByteRange {
                    device_offset: 0,
                    host_offset: 0,
                    len: slot.len(),
                }],
                &slot,
            )
            .unwrap();

        // Also test via real file in CLI!
        let ver_ssd_path = dir.file("ver_ssd.bin");
        let mut file_img = vec![0u8; 4 * 1024 * 1024];
        file_img[..enc.len()].copy_from_slice(&enc);
        fs::write(&ver_ssd_path, &file_img).unwrap();
        let ver_out = dir.file("ver.txt");
        let cli_ver = run_cli(&[
            "5000",
            "-o",
            ver_out.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &format!("{}@4M", ver_ssd_path.display()),
        ]);
        assert!(!cli_ver.status.success());
        assert!(
            String::from_utf8_lossy(&cli_ver.stderr).contains("incompatible format version"),
            "stderr: {}",
            String::from_utf8_lossy(&cli_ver.stderr)
        );

        let fac = platform::make_output_factory(&ver_out);
        let prof = Profile::synthetic(2, 2, 32 * 1024, 512 * 1024, 0, 32 * 1024 * 1024, &[], &[4 * 1024 * 1024]);
        let mut cfg = EngineConfig::new(5000, OutputFormat::Text, "ver.txt", prof, &kernels, fac.as_ref());
        cfg.members = vec![ssd_ver];
        assert_eq!(
            run(&cfg).unwrap_err(),
            EngineError::WrongRecordVersion(RECORD_FORMAT_VERSION + 9)
        );
    }

    // -------------------------------------------------------------------------
    // Case 3: Records of which none is whole -> fails without `--discard`, succeeds with `--discard`!
    // -------------------------------------------------------------------------
    {
        let corrupt_ssd = dir.file("corrupt_rec.bin");
        let mut img = vec![0u8; 8 * 1024 * 1024];
        img[0..16].copy_from_slice(b"CORRUPT_SLOT_0!!");
        img[(RECORD_SLOT_BYTES as usize)..(RECORD_SLOT_BYTES as usize + 16)]
            .copy_from_slice(b"CORRUPT_SLOT_1!!");
        fs::write(&corrupt_ssd, &img).unwrap();

        let out_c = dir.file("corrupt_out.txt");
        let spec_c = format!("{}@8M", corrupt_ssd.display());
        let res_fail = run_cli(&[
            "5000",
            "-o",
            out_c.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &spec_c,
        ]);
        assert!(!res_fail.status.success());
        assert!(
            String::from_utf8_lossy(&res_fail.stderr).contains("none is whole"),
            "stderr: {}",
            String::from_utf8_lossy(&res_fail.stderr)
        );

        let res_ok = run_cli(&[
            "5000",
            "-o",
            out_c.to_str().unwrap(),
            "--discard",
            "--no-vram",
            "--ssd",
            &spec_c,
        ]);
        assert!(res_ok.status.success());
        assert_eq!(fs::read(&out_c).unwrap(), expected_5k);
    }

    // -------------------------------------------------------------------------
    // Case 4 & 5: Missing output file and shortened output file on resume -> fail!
    // -------------------------------------------------------------------------
    {
        let (out_p, _s0_p, _s1_p, s0_arg, s1_arg) = make_stopped("out_checks");

        // Shorten output file to 100 bytes -> must fail!
        fs::write(&out_p, vec![b'3'; 100]).unwrap();
        let short_res = run_cli(&[
            "5000",
            "-o",
            out_p.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!short_res.status.success());
        assert!(
            String::from_utf8_lossy(&short_res.stderr).contains("output file invalid"),
            "stderr: {}",
            String::from_utf8_lossy(&short_res.stderr)
        );

        // Remove output file completely -> must fail!
        fs::remove_file(&out_p).unwrap();
        let miss_res = run_cli(&[
            "5000",
            "-o",
            out_p.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!miss_res.status.success());
        assert!(
            String::from_utf8_lossy(&miss_res.stderr).contains("output file invalid"),
            "stderr: {}",
            String::from_utf8_lossy(&miss_res.stderr)
        );
    }

    // -------------------------------------------------------------------------
    // Case 6: Checkpoint that cannot be read whole (both SSDs' data banks wiped beyond 1-limb
    //         correction while records stay whole) -> Rung 4 `CheckpointUnreadable`!
    // -------------------------------------------------------------------------
    {
        let (out_p, s0_p, s1_p, s0_arg, s1_arg) = make_stopped("unreadable_ckpt");
        for sp in [&s0_p, &s1_p] {
            let mut bytes = fs::read(sp).unwrap();
            let bank_start = RECORD_REGION_BYTES as usize;
            for b in &mut bytes[bank_start..] {
                *b ^= 0xFF;
            }
            fs::write(sp, &bytes).unwrap();
        }
        let unrd = run_cli(&[
            "5000",
            "-o",
            out_p.to_str().unwrap(),
            "--no-vram",
            "--ssd",
            &s0_arg,
            "--ssd",
            &s1_arg,
        ]);
        assert!(!unrd.status.success());
        assert!(
            String::from_utf8_lossy(&unrd.stderr).contains("Checkpoint could not be read whole"),
            "stderr: {}",
            String::from_utf8_lossy(&unrd.stderr)
        );
    }

    // -------------------------------------------------------------------------
    // Case 7: Device left out in Run 1 is recorded in the checkpoint record and stays left out
    //         when Run 2 resumes!
    // -------------------------------------------------------------------------
    {
        struct MemOutFac(PathBuf);
        impl OutputSinkFactory for MemOutFac {
            fn create_full_size(&self, n: u64) -> Result<Box<dyn DigitSink>, String> {
                platform::make_output_factory(&self.0).create_full_size(n)
            }
            fn open_existing_exact(&self, n: u64) -> Result<Box<dyn DigitSink>, String> {
                platform::make_output_factory(&self.0).open_existing_exact(n)
            }
        }
        let _ = Path::new(".");
        let ex_out = dir.file("excl_resume.txt");
        let fac = MemOutFac(ex_out.clone());

        let flaky_gpu = Arc::new(TestMember::new("gpu-flaky", MemberKind::Vram, 4 * 1024 * 1024));
        flaky_gpu.inject_fault(TestFault::StopAnswering);
        let ssd0 = Arc::new(TestMember::new("ssd-0", MemberKind::Ssd, 8 * 1024 * 1024));
        let ssd1 = Arc::new(TestMember::new("ssd-1", MemberKind::Ssd, 8 * 1024 * 1024));

        let prof = Profile::synthetic(
            2,
            2,
            16 * 1024,
            64 * 1024,
            0,
            32 * 1024 * 1024,
            &[4 * 1024 * 1024],
            &[8 * 1024 * 1024, 8 * 1024 * 1024],
        );
        let mut cfg1 = EngineConfig::new(5000, OutputFormat::Text, "excl_res", prof.clone(), &kernels, &fac);
        cfg1.members = vec![
            Arc::clone(&flaky_gpu) as Arc<dyn ExtentMember>,
            Arc::clone(&ssd0) as Arc<dyn ExtentMember>,
            Arc::clone(&ssd1) as Arc<dyn ExtentMember>,
        ];
        cfg1.overrides.checkpoint_mult = 0.0;
        cfg1.stop_condition = Some(StopCondition::AfterStageCheckpoint(
            CheckpointStageKind::Finish,
        ));
        // Startup probe excludes `gpu-flaky`, writes `Finish` checkpoint with `excluded_devices = ["gpu-flaky"]`, and stops!
        let err1 = run(&cfg1).unwrap_err();
        assert!(matches!(err1, EngineError::StoppedForTest { .. }));

        // Even if a fresh `gpu-flaky` device appears healthy in Run 2, the record keeps it excluded!
        let healthy_gpu_same_id =
            Arc::new(TestMember::new("gpu-flaky", MemberKind::Vram, 4 * 1024 * 1024));
        let mut cfg2 = EngineConfig::new(5000, OutputFormat::Text, "excl_res", prof, &kernels, &fac);
        cfg2.members = vec![healthy_gpu_same_id, ssd0, ssd1];
        let rep2 = run(&cfg2).expect("Resume with persisted excluded GPU");
        assert!(
            rep2.excluded_devices.contains(&"gpu-flaky".to_string()),
            "Excluded GPU from checkpoint record must stay excluded on resume: {:?}",
            rep2.excluded_devices
        );
        assert_eq!(fs::read(&ex_out).unwrap(), expected_5k);
    }
}

#[test]
fn gate3_checkpoint_rate_striping_whole_readback_and_series_divform_replan() {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use engine::plan::{Calibration, DivForm, PlanOverrides};
    use engine::select_checkpoint_ssds;
    use engine::store::StoreError;

    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);

    // -------------------------------------------------------------------------
    // 1. Section 10 checkpoint rate striping: drops slow outlier SSD while keeping >= 2 SSD parity!
    // -------------------------------------------------------------------------
    let s_fast0: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("nvme-0", MemberKind::Ssd, 8 * 1024 * 1024));
    let s_fast1: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("nvme-1", MemberKind::Ssd, 8 * 1024 * 1024));
    let s_slow: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("usb-slow", MemberKind::Ssd, 8 * 1024 * 1024));

    // 2 fast (6 GB/s) + 1 slow (0.4 GB/s): k=2 gives 12 GB/s vs k=3 giving 1.2 GB/s -> picks 2 fast!
    let picked_2 = select_checkpoint_ssds(&[
        (Arc::clone(&s_fast0), 6.0e9),
        (Arc::clone(&s_fast1), 6.0e9),
        (Arc::clone(&s_slow), 0.4e9),
    ]);
    assert_eq!(picked_2.len(), 2);
    assert_eq!(picked_2[0].id(), "nvme-0");
    assert_eq!(picked_2[1].id(), "nvme-1");

    // 2 unequal SSDs (10 GB/s + 2 GB/s): still takes both so `add_n`/`sub_n` parity holds!
    let picked_pair =
        select_checkpoint_ssds(&[(Arc::clone(&s_fast0), 10.0e9), (Arc::clone(&s_slow), 2.0e9)]);
    assert_eq!(picked_pair.len(), 2);

    // -------------------------------------------------------------------------
    // 2. Whole-checkpoint readback catches cross-round corruption & excludes bad SSD!
    // -------------------------------------------------------------------------
    struct CrossRoundAliasSsd {
        inner: TestMember,
        writes: AtomicUsize,
    }
    impl ExtentMember for CrossRoundAliasSsd {
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
            self.inner.read(ranges, host_buf)
        }
        fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
            self.inner.write(ranges, host_buf)?;
            let w = self.writes.fetch_add(1, Ordering::Relaxed);
            // When round >= 1 is written (or retried) in the checkpoint bank (`>= 131072`),
            // silently clobber round 0 (`offset 131072`) so immediate piece-by-piece readback
            // would have missed it, whereas whole-checkpoint readback catches it!
            if ranges.iter().any(|r| r.device_offset >= 131_072) && w >= 1 {
                self.inner.corrupt_u64_at(131_072 + 16, 0xDEAD_BEEF_CAFE_BABE);
            }
            Ok(())
        }
        fn free(&self, device_offset: u64, bytes: u64) {
            self.inner.free(device_offset, bytes);
        }
    }

    struct SharedMemSink(Arc<Mutex<Vec<u8>>>);
    impl DigitSink for SharedMemSink {
        fn write_at(&self, offset: u64, bytes: &[u8]) -> Result<(), String> {
            let mut g = self.0.lock().unwrap();
            let off = offset as usize;
            g[off..(off + bytes.len())].copy_from_slice(bytes);
            Ok(())
        }
        fn flush(&self) -> Result<(), String> {
            Ok(())
        }
    }
    struct SharedMemFactory(Arc<Mutex<Vec<u8>>>);
    impl OutputSinkFactory for SharedMemFactory {
        fn create_full_size(&self, n: u64) -> Result<Box<dyn DigitSink>, String> {
            *self.0.lock().unwrap() = vec![0u8; n as usize];
            Ok(Box::new(SharedMemSink(Arc::clone(&self.0))))
        }
        fn open_existing_exact(&self, n: u64) -> Result<Box<dyn DigitSink>, String> {
            let len = self.0.lock().unwrap().len() as u64;
            if len != n {
                return Err(format!("size {len} != {n}"));
            }
            Ok(Box::new(SharedMemSink(Arc::clone(&self.0))))
        }
    }

    let digits = 15_000u64;
    let base_buf = Arc::new(Mutex::new(Vec::new()));
    let base_fac = SharedMemFactory(Arc::clone(&base_buf));
    let prof = Profile::synthetic(
        2,
        2,
        32 * 1024,
        512 * 1024,
        4 * 1024 * 1024,
        64 * 1024 * 1024,
        &[],
        &[8 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024],
    );
    let base_rep = run(&EngineConfig::new(
        digits,
        OutputFormat::Text,
        "g3_base",
        prof.clone(),
        &kernels,
        &base_fac,
    ))
    .unwrap();
    let expected_pi = base_buf.lock().unwrap().clone();

    let bad_alias_ssd: Arc<dyn ExtentMember> = Arc::new(CrossRoundAliasSsd {
        inner: TestMember::new("ssd-alias", MemberKind::Ssd, 8 * 1024 * 1024),
        writes: AtomicUsize::new(0),
    });
    let good_ssd_a: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("ssd-good-a", MemberKind::Ssd, 8 * 1024 * 1024));
    let good_ssd_b: Arc<dyn ExtentMember> =
        Arc::new(TestMember::new("ssd-good-b", MemberKind::Ssd, 8 * 1024 * 1024));
    let alias_buf = Arc::new(Mutex::new(Vec::new()));
    let alias_fac = SharedMemFactory(Arc::clone(&alias_buf));
    let mut cfg_alias = EngineConfig::new(
        digits,
        OutputFormat::Text,
        "g3_alias",
        prof.clone(),
        &kernels,
        &alias_fac,
    );
    cfg_alias.members = vec![bad_alias_ssd, good_ssd_a, good_ssd_b];
    cfg_alias.overrides.checkpoint_mult = 0.0;
    let rep_alias = run(&cfg_alias).expect("whole-checkpoint readback excludes aliasing SSD");
    assert!(
        rep_alias.excluded_devices.contains(&"ssd-alias".to_string()),
        "Whole-checkpoint readback must catch round-0 clobber and exclude ssd-alias: {:?}",
        rep_alias.excluded_devices
    );
    assert_eq!(*alias_buf.lock().unwrap(), expected_pi);

    // -------------------------------------------------------------------------
    // 3. Resumed Series checkpoint keeps `dividing_mask` while re-planning `DivForm`!
    // -------------------------------------------------------------------------
    let run_series_resume_with_cal = |resumed_cal: Calibration| -> (DivForm, Vec<u8>, u64) {
        let ck_s0 = Arc::new(TestMember::new("ck-0", MemberKind::Ssd, 8 * 1024 * 1024));
        let ck_s1 = Arc::new(TestMember::new("ck-1", MemberKind::Ssd, 8 * 1024 * 1024));
        let ck_slow = Arc::new(TestMember::new("ck-slow", MemberKind::Ssd, 8 * 1024 * 1024));
        let mut rates = HashMap::new();
        rates.insert("ck-0".to_string(), 6.0e9);
        rates.insert("ck-1".to_string(), 6.0e9);
        rates.insert("ck-slow".to_string(), 0.4e9);

        let out_buf = Arc::new(Mutex::new(Vec::new()));
        let fac = SharedMemFactory(Arc::clone(&out_buf));

        let mut cfg1 = EngineConfig::new(
            digits,
            OutputFormat::Text,
            "replan_div",
            prof.clone(),
            &kernels,
            &fac,
        );
        cfg1.members = vec![
            Arc::clone(&ck_s0) as Arc<dyn ExtentMember>,
            Arc::clone(&ck_s1) as Arc<dyn ExtentMember>,
            Arc::clone(&ck_slow) as Arc<dyn ExtentMember>,
        ];
        cfg1.supplied_member_rates = Some(rates.clone());
        cfg1.overrides = PlanOverrides {
            checkpoint_mult: 0.0,
            force_dividing_levels: Some(vec![(4, DivForm::SingleProduct)]),
            ..PlanOverrides::default()
        };
        cfg1.stop_condition = Some(StopCondition::AfterStageCheckpoint(
            CheckpointStageKind::Series,
        ));
        let stop_err = run(&cfg1).unwrap_err();
        assert!(matches!(stop_err, EngineError::StoppedForTest { stage: CheckpointStageKind::Series, .. }));

        // Verify slow 3rd SSD (`0.4 GB/s`) was NOT included in the checkpoint stripe!
        let mut slow_hdr = vec![0u8; 4096];
        ck_slow
            .read(
                &[ByteRange {
                    device_offset: 0,
                    host_offset: 0,
                    len: 4096,
                }],
                &mut slow_hdr,
            )
            .unwrap();
        assert!(
            slow_hdr.iter().all(|&b| b == 0),
            "Slow outlier SSD must be left out of checkpoint stripe by Section 10 rate rule"
        );

        // Resume with ONLY `ck_s1` (proving `ck_s0`+`ck_s1` 2-way parity recovers with `ck_s0` missing!)
        // and NO `force_dividing_levels` override, supplying `resumed_cal` so `Plan::build` re-prices `DivForm`!
        let resume_prof = Profile::synthetic(
            2,
            2,
            32 * 1024,
            512 * 1024,
            4 * 1024 * 1024,
            64 * 1024 * 1024,
            &[],
            &[8 * 1024 * 1024],
        );
        let mut cfg2 = EngineConfig::new(
            digits,
            OutputFormat::Text,
            "replan_div",
            resume_prof,
            &kernels,
            &fac,
        );
        cfg2.members = vec![ck_s1];
        cfg2.supplied_calibration = Some(resumed_cal);
        let rep2 = run(&cfg2).expect("resume Series with re-planned DivForm");
        assert_eq!(rep2.resumed_from_serial, Some(1));
        assert_eq!(
            rep2.plan.dividing_levels.len(),
            1,
            "Resumed Series plan must preserve checkpoint dividing_mask"
        );
        assert_eq!(rep2.plan.dividing_levels[0].level, 4);
        let chosen_form = rep2.plan.dividing_levels[0].form;
        let final_bytes = out_buf.lock().unwrap().clone();
        (chosen_form, final_bytes, rep2.dec_hash_w0)
    };

    // Calibration A: very fast sequential limb division relative to transforms -> chooses `SequentialHensel`!
    let mut cal_seq = Calibration::synthetic(1.0e7, 60.0e9, true, Vec::new());
    cal_seq.basecase_limbs_sq_per_sec = 5.0e11;
    let (form_seq, bytes_seq, hash_seq) = run_series_resume_with_cal(cal_seq);
    assert_eq!(
        form_seq,
        DivForm::SequentialHensel,
        "Resumed Series run must re-plan DivForm to SequentialHensel when limb rate dominates"
    );
    assert_eq!(hash_seq, base_rep.dec_hash_w0);
    assert_eq!(bytes_seq, expected_pi);

    // Calibration B: fast transforms relative to basecase -> chooses `SingleProduct`!
    let mut cal_sp = Calibration::synthetic(2.5e10, 200.0e9, true, Vec::new());
    cal_sp.basecase_limbs_sq_per_sec = 2.0e6;
    cal_sp.basecase_max_limbs = 4;
    cal_sp.karatsuba_max_limbs = 8;
    let (form_sp, bytes_sp, hash_sp) = run_series_resume_with_cal(cal_sp);
    assert_eq!(
        form_sp,
        DivForm::SingleProduct,
        "Resumed Series run must re-plan DivForm to SingleProduct when transform rate dominates"
    );
    assert_eq!(hash_sp, base_rep.dec_hash_w0);
    assert_eq!(bytes_sp, expected_pi);
}
