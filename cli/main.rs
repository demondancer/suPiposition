//! Contract:
//! - Inputs: Command-line arguments (`<digits>`, `-o <path>`, `--format text|packed19`, `--ssd
//!   <path>[@<size>]`, `--discard`, `--no-vram`, `--ref-kernels`, `--plan-only`, `--profile`,
//!   `--calibration`, and checkpoint/test flags).
//! - Outputs: Startup kernel-binding (including mismatched/untested kernel entries) & topology
//!   report, deterministic plan summary, live checkpoint/return telemetry lines, and verified `pi`
//!   digits + `Dec Hash` + end-of-run telemetry report (or non-zero exit code with diagnostic).
//! - Data format: Plain UTF-8 reports on `stdout` / `stderr`.
//! - Work & bytes moved: Joins `kernels::supi_bind`, `platform::probe_platform`, and `engine::run`;
//!   stacks and small heap stay on the standard allocator while `engine::arena::ArenaSet` manages
//!   the run's `numbers`, `scratch`, and `transforms` arenas.

use std::path::PathBuf;
use std::process::ExitCode;

use engine::plan::{Calibration, DeviceCalibration, OutputFormat, PlanOverrides};
use engine::refk::Kernels;
use engine::tier::Profile;
use engine::{CheckpointStageKind, EngineConfig, StopCondition, plan_only, run};

fn parse_byte_count(s: &str) -> Result<u64, String> {
    let clean = s.trim().replace('_', "");
    if clean.is_empty() {
        return Err("Empty number".to_string());
    }
    let (num_part, mult) = match clean.as_bytes().last().copied().unwrap() {
        b'K' | b'k' => (&clean[..clean.len() - 1], 1024u64),
        b'M' | b'm' => (&clean[..clean.len() - 1], 1024 * 1024),
        b'G' | b'g' => (&clean[..clean.len() - 1], 1024 * 1024 * 1024),
        b'T' | b't' => (&clean[..clean.len() - 1], 1024 * 1024 * 1024 * 1024),
        _ => (clean.as_str(), 1u64),
    };
    if let Some((mant, exp)) = num_part.split_once(['e', 'E']) {
        let m: u64 = mant
            .parse()
            .map_err(|_| format!("Invalid integer mantissa in '{s}'"))?;
        let e: u32 = exp
            .parse()
            .map_err(|_| format!("Invalid exponent in '{s}'"))?;
        let pow10 = 10u64
            .checked_pow(e)
            .ok_or_else(|| format!("Overflow in '{s}'"))?;
        return m
            .checked_mul(pow10)
            .and_then(|v| v.checked_mul(mult))
            .ok_or_else(|| format!("Overflow in '{s}'"));
    }
    let base: u64 = num_part
        .parse()
        .map_err(|_| format!("Invalid integer '{s}'"))?;
    base.checked_mul(mult)
        .ok_or_else(|| format!("Overflow in '{s}'"))
}

fn parse_ssd_arg(arg: &str) -> Result<(PathBuf, Option<u64>), String> {
    if let Some((p, sz)) = arg.rsplit_once('@') {
        let bytes = parse_byte_count(sz)?;
        Ok((PathBuf::from(p), Some(bytes)))
    } else {
        Ok((PathBuf::from(arg), None))
    }
}

fn parse_synthetic_profile(spec: &str) -> Result<Profile, String> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() < 6 {
        return Err(
            "--profile expects <cores>,<threads>,<l1>,<l2>,<l3>,<ram>[,<vram>,<ssd>] (<l3> may be <n>x<size>)"
                .to_string(),
        );
    }
    let cores = parse_byte_count(parts[0])?;
    let threads = parse_byte_count(parts[1])?;
    let l1 = parse_byte_count(parts[2])?;
    let l2 = parse_byte_count(parts[3])?;
    // The last-level cache may be written `<n>x<size>`: that many separate caches of that size,
    // each with its own memory node, the cores shared out evenly among them (the sockets of a
    // large system, the tiles or core groups of one chip).
    let (l3_caches, l3) = match parts[4].split_once(['x', 'X']) {
        Some((n, size)) => (parse_byte_count(n)?, parse_byte_count(size)?),
        None => (1, parse_byte_count(parts[4])?),
    };
    let ram = parse_byte_count(parts[5])?;
    let vram = if parts.len() >= 7 {
        let v = parse_byte_count(parts[6])?;
        if v > 0 { vec![v] } else { Vec::new() }
    } else {
        Vec::new()
    };
    let ssd = if parts.len() >= 8 {
        let s = parse_byte_count(parts[7])?;
        if s > 0 { vec![s] } else { Vec::new() }
    } else {
        Vec::new()
    };
    if l3_caches > 1 {
        Ok(Profile::synthetic_multi_domain(
            cores, threads, l3_caches, l1, l2, l3, ram, &vram, &ssd,
        ))
    } else {
        Ok(Profile::synthetic(cores, threads, l1, l2, l3, ram, &vram, &ssd))
    }
}

fn parse_rate_f64(s: &str) -> Result<f64, String> {
    let clean = s.trim().replace('_', "");
    if clean.is_empty() {
        return Err("Empty rate".to_string());
    }
    let (num_part, mult) = match clean.as_bytes().last().copied().unwrap() {
        b'K' | b'k' => (&clean[..clean.len() - 1], 1.0e3f64),
        b'M' | b'm' => (&clean[..clean.len() - 1], 1.0e6f64),
        b'G' | b'g' => (&clean[..clean.len() - 1], 1.0e9f64),
        b'T' | b't' => (&clean[..clean.len() - 1], 1.0e12f64),
        _ => (clean.as_str(), 1.0f64),
    };
    let val: f64 = num_part
        .parse()
        .map_err(|_| format!("Invalid float rate '{s}'"))?;
    let out = val * mult;
    if !out.is_finite() || out <= 0.0 {
        return Err(format!("Rate '{s}' must be positive and finite"));
    }
    Ok(out)
}

fn parse_supplied_calibration(spec: &str) -> Result<Calibration, String> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() < 2 {
        return Err(
            "--calibration expects <ntt_bf>,<ram_bw>[,<nt:0|1>[,<sieve_ops>[,<base_max>,<kara_max>[,<dev_idle>,<dev_beside>]]]]"
                .to_string(),
        );
    }
    let ntt_bf = parse_rate_f64(parts[0])?;
    let ram_bw = parse_rate_f64(parts[1])?;
    let nt_eff = if parts.len() >= 3 {
        match parts[2].trim() {
            "1" | "true" | "nt" => true,
            "0" | "false" => false,
            other => {
                return Err(format!(
                    "Invalid non-temporal flag '{other}' in --calibration (expected 0 or 1)"
                ));
            }
        }
    } else {
        true
    };
    let mut devs = Vec::new();
    if parts.len() >= 8 {
        let idle = parse_rate_f64(parts[6])?;
        let beside = parse_rate_f64(parts[7])?;
        devs.push(DeviceCalibration {
            name: "supplied-store-0".to_string(),
            idle_rate_bps: idle,
            beside_sweep_rate_bps: beside,
            sweep_step_beside_sec: 1.15e-5,
            min_batch_range_bytes: 65_536,
        });
    }
    let mut cal = Calibration::synthetic(ntt_bf, ram_bw, nt_eff, devs);
    if parts.len() >= 4 {
        cal.sieve_ops_per_sec = parse_rate_f64(parts[3])?;
    }
    if parts.len() >= 6 {
        let bm = parse_byte_count(parts[4])? as usize;
        let km = parse_byte_count(parts[5])? as usize;
        if bm == 0 || km <= bm {
            return Err("Invalid basecase/Karatsuba crossovers in --calibration".to_string());
        }
        cal.basecase_max_limbs = bm;
        cal.karatsuba_max_limbs = km;
    }
    Ok(cal)
}

fn print_usage() {
    eprintln!(
        "Usage: suPiposition <digits> -o <output> [options]\n\
         Options:\n  \
           --format text|packed19       Output format (default: text)\n  \
           --ssd <path>[@<size>]        Add SSD file/device member (repeatable)\n  \
           --discard                    Discard existing SSD checkpoint records\n  \
           --no-vram                    Do not use discrete GPU VRAM\n  \
           --ref-kernels                Force portable reference kernels\n  \
           --plan-only                  Compute and print deterministic plan without running\n  \
           --profile <c,t,l1,l2,l3,ram> Override hardware profile with synthetic counts\n  \
           \x20                            (l3 as <n>x<size>: n separate caches)\n  \
           --calibration <bf,bw[,...]>  Supply calibration figures without host measurement\n  \
           --checkpoint-mult <f>        Checkpoint write threshold multiplier\n  \
           --stop-after-stage <stage>   Stop after first series|finish|convert checkpoint"
    );
}

fn main() -> ExitCode {
    // A panic in any thread ends the run at once, in every build. A worker that stopped would
    // otherwise leave the rest waiting for it for ever.
    let report_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        report_panic(info);
        std::process::abort();
    }));
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return if args.is_empty() {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }

    let mut digits_opt: Option<u64> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut format = OutputFormat::Text;
    let mut ssd_specs: Vec<(PathBuf, Option<u64>)> = Vec::new();
    let mut discard_records = false;
    let mut enable_vram = true;
    let mut force_ref_kernels = false;
    let mut is_plan_only = false;
    let mut synthetic_prof: Option<Profile> = None;
    let mut supplied_cal: Option<Calibration> = None;
    let mut overrides = PlanOverrides::default();
    let mut stop_condition: Option<StopCondition> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for -o");
                    return ExitCode::from(2);
                };
                output_path = Some(PathBuf::from(v));
            }
            "--format" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --format");
                    return ExitCode::from(2);
                };
                format = match v.as_str() {
                    "text" => OutputFormat::Text,
                    "packed19" => OutputFormat::Packed19,
                    other => {
                        eprintln!("Error: unknown format '{other}' (expected text or packed19)");
                        return ExitCode::from(2);
                    }
                };
            }
            "--ssd" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --ssd");
                    return ExitCode::from(2);
                };
                match parse_ssd_arg(v) {
                    Ok(spec) => ssd_specs.push(spec),
                    Err(e) => {
                        eprintln!("Error in --ssd: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--discard" => discard_records = true,
            "--no-vram" => enable_vram = false,
            "--ref-kernels" => force_ref_kernels = true,
            "--plan-only" => is_plan_only = true,
            "--profile" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --profile");
                    return ExitCode::from(2);
                };
                match parse_synthetic_profile(v) {
                    Ok(p) => synthetic_prof = Some(p),
                    Err(e) => {
                        eprintln!("Error in --profile: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--calibration" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --calibration");
                    return ExitCode::from(2);
                };
                match parse_supplied_calibration(v) {
                    Ok(c) => supplied_cal = Some(c),
                    Err(e) => {
                        eprintln!("Error in --calibration: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--checkpoint-mult" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --checkpoint-mult");
                    return ExitCode::from(2);
                };
                let Ok(mult) = v.parse::<f64>() else {
                    eprintln!("Error: invalid float for --checkpoint-mult");
                    return ExitCode::from(2);
                };
                overrides.checkpoint_mult = mult;
            }
            "--stop-after-stage" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --stop-after-stage");
                    return ExitCode::from(2);
                };
                let st = match v.as_str() {
                    "series" => CheckpointStageKind::Series,
                    "finish" => CheckpointStageKind::Finish,
                    "convert" => CheckpointStageKind::Convert,
                    other => {
                        eprintln!("Error: unknown stage '{other}'");
                        return ExitCode::from(2);
                    }
                };
                stop_condition = Some(StopCondition::AfterStageCheckpoint(st));
            }
            "--stop-after-ckpt" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("Error: missing value for --stop-after-ckpt");
                    return ExitCode::from(2);
                };
                let Ok(n) = v.parse::<u64>() else {
                    eprintln!("Error: invalid integer for --stop-after-ckpt");
                    return ExitCode::from(2);
                };
                stop_condition = Some(StopCondition::AfterAnyCheckpoint(n));
            }
            "--truncate-record-once" => {
                stop_condition = Some(StopCondition::TruncateRecordWriteOnce);
            }
            arg if !arg.starts_with('-') && digits_opt.is_none() => match parse_byte_count(arg) {
                Ok(d) => digits_opt = Some(d),
                Err(e) => {
                    eprintln!("Error parsing digits '{arg}': {e}");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("Error: unexpected argument '{other}'");
                print_usage();
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let Some(digits) = digits_opt else {
        eprintln!("Error: missing <digits> count");
        print_usage();
        return ExitCode::from(2);
    };

    let out_path = output_path.unwrap_or_else(|| PathBuf::from("pi.txt"));

    // 1. Bind & startup-verify kernels!
    let (mut kernels, bind_rep) = if force_ref_kernels {
        (Kernels::reference_only(), None)
    } else {
        let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
        (k, Some(rep))
    };
    // A traced run lists where each bound kernel starts, as an offset from `supi_bind`, so that a
    // sampled profile can be attributed kernel by kernel.
    if std::env::var_os("SUPI_TRACE").is_some() && bind_rep.is_some() {
        let base = kernels::supi_bind as *const () as usize;
        let ptrs = kernels.active.as_raw_ptrs();
        let list: Vec<String> = engine::refk::KERNEL_NAMES
            .iter()
            .zip(ptrs.iter())
            .map(|(n, &p)| format!("{n}={:#x}", p.wrapping_sub(base)))
            .collect();
        eprintln!("[TRACE KTABLE] {}", list.join(" "));
    }
    // SUPI_KPROF=1 times every transform, movement, digit and check-word kernel of a traced run;
    // SUPI_KPROF=2 times the limb kernels too.
    // (SUPI_BENCH_MUL times one product of a given shape in place of the run: see the engine.)
    // (SUPI_BENCH_PRICE times every shape the planner weighs for a product beside its price;
    // SUPI_TRACE_PLAN prints the estimates the dividing levels are chosen from, and the choice.)
    // (SUPI_BENCH_CONV=<rounds> times the base conversion alone, on a made-up fraction of the
    // run's size, in place of the run.)
    if let Some(v) = std::env::var_os("SUPI_KPROF") {
        kernels.enable_timing(v == "2");
    }

    if let Some(rep) = &bind_rep {
        let bound_cnt = rep.bound_mask.count_ones();
        if rep.untested.is_empty() {
            println!(
                "suPiposition kernels: {bound_cnt}/24 assembly bound (mask=0x{:06x}, mismatched={:?})",
                rep.bound_mask, rep.mismatched
            );
        } else {
            println!(
                "suPiposition kernels: {bound_cnt}/24 assembly bound (mask=0x{:06x}, mismatched={:?}, untested={:?})",
                rep.bound_mask, rep.mismatched, rep.untested
            );
        }
    } else {
        println!("suPiposition kernels: portable reference mode (24/24 refk)");
    }

    // 2. Probe platform topology, VRAM, and SSDs.
    let is_host_profile = synthetic_prof.is_none();
    let probe_vram = enable_vram && (is_host_profile || !is_plan_only);
    let penv = match platform::probe_platform(&ssd_specs, probe_vram) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("Platform initialization failed: {e}");
            return ExitCode::from(1);
        }
    };

    let mut profile = synthetic_prof.unwrap_or(penv.profile);
    if probe_vram && profile.vram.is_empty() {
        for m in &penv.members {
            if m.kind() == engine::store::MemberKind::Vram {
                profile.vram.push(engine::tier::DeviceEntry {
                    id: m.id().to_string(),
                    size: m.capacity(),
                    align: m.alignment(),
                });
            }
        }
    }
    if !ssd_specs.is_empty() && profile.ssd.is_empty() {
        for m in &penv.members {
            if m.kind() == engine::store::MemberKind::Ssd {
                profile.ssd.push(engine::tier::DeviceEntry {
                    id: m.id().to_string(),
                    size: m.capacity(),
                    align: m.alignment(),
                });
            }
        }
    }

    let out_factory = platform::make_output_factory(&out_path);
    let out_id = out_path.to_string_lossy().into_owned();

    let mut cfg = EngineConfig::new(
        digits,
        format,
        out_id,
        profile,
        &kernels,
        out_factory.as_ref(),
    );
    let live_reporter = |msg: &str| {
        println!("{msg}");
    };
    cfg.discard_records = discard_records;
    cfg.supplied_calibration = supplied_cal;
    cfg.build_id = kernels::SUPI_BUILD_ID;
    if is_host_profile && cfg.supplied_calibration.is_none() {
        cfg.calibration_cache_path =
            Some(std::env::temp_dir().join("suPiposition_calibration_v1.bin"));
    }
    cfg.overrides = overrides;
    cfg.members = penv.members;
    cfg.region_mapper = Some(penv.region_mapper);
    cfg.thread_placer = Some(penv.thread_placer);
    cfg.stop_condition = stop_condition;
    cfg.progress_reporter = Some(&live_reporter);

    if is_plan_only {
        match plan_only(&cfg) {
            Ok((shares, _cal, plan)) => {
                println!(
                    "Profile: cores={} threads={} L1={} KiB L2={} KiB L3={} KiB RAM={:.2} GiB VRAM={:.2} GiB SSD={:.2} GiB",
                    shares.cores,
                    shares.threads,
                    shares.min_l1_share / 1024,
                    shares.min_l2_share / 1024,
                    shares.min_l3_share / 1024,
                    (shares.total_ram as f64) / (1024.0 * 1024.0 * 1024.0),
                    (shares.total_vram as f64) / (1024.0 * 1024.0 * 1024.0),
                    (shares.total_ssd as f64) / (1024.0 * 1024.0 * 1024.0),
                );
                println!(
                    "Plan: digits={} terms={} height={} leaf_len={} dividing_levels={} out_of_core={} bound={:?} planned_sec={:.4}",
                    plan.digits,
                    plan.term_count,
                    plan.tree_height,
                    plan.leaf_len,
                    plan.dividing_levels.len(),
                    plan.out_of_core,
                    plan.pace_bound,
                    plan.planned_total_sec,
                );
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Plan error: {e}");
                return ExitCode::from(1);
            }
        }
    }

    match run(&cfg) {
        Ok(rep) => {
            let comp_sec = rep.series_sec + rep.finish_sec + rep.convert_sec;
            println!(
                "Completed {} digits in {:.3} s (compute={:.3}s: series={:.3}s finish={:.3}s convert={:.3}s | calib={:.3}s) | Dec Hash = {} | digits_in_doubt = {}",
                rep.plan.digits,
                rep.wall_sec,
                comp_sec,
                rep.series_sec,
                rep.finish_sec,
                rep.convert_sec,
                rep.calib_sec,
                rep.dec_hash_w0,
                rep.digits_in_doubt
            );
            println!(
                "Arenas: huge={} refused={} pinned={} pin_refused={} | store_moved={} B | ckpts={} resumed={:?} rollbacks={} excluded={:?} | {}",
                rep.arena_report.huge_pages_used,
                rep.arena_report.huge_pages_refused,
                rep.arena_report.pinned_used,
                rep.arena_report.pin_refused,
                rep.store_bytes_moved,
                rep.checkpoints_written,
                rep.resumed_from_serial,
                rep.rollbacks,
                rep.excluded_devices,
                rep.telemetry_summary,
            );
            ExitCode::SUCCESS
        }
        Err(engine::EngineError::StoppedForTest { serial, stage }) => {
            println!("STOPPED_FOR_TEST serial={serial} stage={stage:?}");
            ExitCode::from(75)
        }
        Err(e) => {
            eprintln!("suPiposition failed: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_synthetic_profile;

    #[test]
    fn a_described_machine_may_have_several_last_level_caches() {
        let one = parse_synthetic_profile("8,16,48K,1M,96M,48G").unwrap();
        assert_eq!((one.cores, one.threads, one.l3.len()), (8, 16, 1));
        assert_eq!(one.l3[0].size, 96 * 1024 * 1024);
        let many = parse_synthetic_profile("192,384,48K,1M,24x32M,1T").unwrap();
        assert_eq!((many.cores, many.threads, many.l3.len()), (192, 384, 24));
        assert!(many.l3.iter().all(|c| c.size == 32 * 1024 * 1024 && c.threads.len() == 16));
        assert_eq!(many.ram.len(), 24);
        assert!(parse_synthetic_profile("8,16,48K,1M,x32M,48G").is_err());
        assert!(parse_synthetic_profile("8,16,48K,1M").is_err());
    }
}
