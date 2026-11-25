//! Contract:
//! - Inputs: `EngineConfig` (target `digits`, `OutputFormat`, `output_id`, `discard_records`,
//!   `Profile`, optional supplied `Calibration` and member rates, `PlanOverrides`, bound `Kernels`,
//!   `ExtentMember`s, `OutputSinkFactory`, optional `RegionMapper`, `ThreadPlacer`,
//!   `progress_reporter`, and optional Gate 2/3 hooks).
//! - Outputs: `(Shares, Calibration, Plan)` (for `plan_only`) or `RunReport` with verified digits
//!   written to the output sink, `Dec Hash` check word (`mod 2^61 - 1`), `digits_in_doubt`, exact
//!   kernel/sieve/store counters, checkpoint serials, and telemetry reports at every checkpoint,
//!   return, and end of run.
//! - Data format: Standard library types only; dual-slot `CheckpointRecord`s in `[0..131072)` and
//!   dual-bank Section-10 rate-striped parity checkpoint pieces read back whole below
//!   `store_base_offset` on participating SSD members.
//! - Work & bytes moved: Orchestrates `tier` -> `Calibration` -> `Plan` -> `ArenaSet` ->
//!   `splitting` -> `newton` -> `radix`, climbing the 4-rung correctness ladder across products,
//!   quotients, stored pieces, checkpoints, timeouts, and device exclusions.

pub mod arena;
pub mod check;
pub mod fft;
pub mod limb;
pub mod mul;
pub mod newton;
pub mod ntt;
pub mod par;
pub mod plan;
pub mod radix;
pub mod refk;
pub mod splitting;
pub mod store;
pub mod tier;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::arena::{ArenaSet, MappingReport, RegionMapper};
use crate::check::{
    CheckpointRecord, CorrectionOutcome, DualCheck, Ladder, RECORD_FORMAT_VERSION, RecordError,
    Rung, SavedNumberMeta, StageRecord, Telemetry, build_parity_piece, compute_limbs_all3,
    compute_limbs_w0, reconstruct_missing_piece, try_correct_single_limb, whole_value_w0,
};
use crate::fft::{FFT_MAX_ERR_BOUND, FftTableCache};
use crate::limb::BigNum;
use crate::mul::{MulCtx, MulError, MulFaultHook};
use crate::newton::{NewtonCheckpointSink, NewtonResumeState, run_finish};
use crate::ntt::NttTableCache;
use crate::par::{Scheduler, ThreadPlacer};
use crate::plan::{Calibration, OutputFormat, Plan, PlanError, PlanOverrides};
use crate::radix::{
    DigitSink, LeafFaultHook, PendingFraction, RadixCheckpointSink, RadixResumeState,
    run_conversion, run_conversion_owned,
};
use crate::refk::{Kernels, WorkSnapshot};
use crate::splitting::{BlockTriple, SeriesCheckpointSink, SeriesResumeState, run_series};
use crate::store::{ByteRange, ExtentMember, MemberKind, Store, StoreError, form_stripes};
use crate::tier::{Profile, Shares};

pub const RECORD_SLOT_BYTES: u64 = 65_536; // 64 KiB per record slot (2 slots = 128 KiB)
pub const RECORD_REGION_BYTES: u64 = RECORD_SLOT_BYTES * 2;
pub const CKPT_PIECE_LIMBS: usize = 512; // 4096 data bytes per checkpoint piece

/// Factory opening or validating the output digit sink so `engine` stays decoupled from OS file APIs.
pub trait OutputSinkFactory: Send + Sync {
    /// Create/truncate output sink to exact `full_size_bytes` for a fresh run.
    fn create_full_size(&self, full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String>;
    /// Open existing output sink for a resumed run, failing if missing or its byte length != `expected_bytes`.
    fn open_existing_exact(&self, expected_bytes: u64) -> Result<Box<dyn DigitSink>, String>;
}

/// Optional Gate-3 hook to halt cleanly after writing a specific checkpoint (or mid-record write).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCondition {
    AfterAnyCheckpoint(u64),
    AfterStageCheckpoint(CheckpointStageKind),
    TruncateRecordWriteOnce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointStageKind {
    Series,
    Finish,
    Convert,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    Plan(PlanError),
    RecordMismatch {
        found_digits: u64,
        found_output: String,
        found_text: bool,
    },
    WrongRecordVersion(u32),
    NoWholeRecord,
    OutputFileSizeMismatch(String),
    CheckpointUnreadable(String),
    StoppedForTest {
        serial: u64,
        stage: CheckpointStageKind,
    },
    FatalStop(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Plan(PlanError::InsufficientCapacity {
                needed_bytes,
                available_bytes,
                detail,
            }) => write!(
                f,
                "System cannot hold run (needed {needed_bytes} B, available {available_bytes} B): {detail}"
            ),
            EngineError::Plan(PlanError::OutputFileError(msg)) => {
                write!(f, "Output file preflight failed: {msg}")
            }
            EngineError::RecordMismatch {
                found_digits,
                found_output,
                found_text,
            } => write!(
                f,
                "Existing checkpoint record belongs to another run (digits={found_digits}, output={found_output}, text={found_text}); pass --discard to start fresh"
            ),
            EngineError::WrongRecordVersion(v) => write!(
                f,
                "Checkpoint record has incompatible format version {v} (expected {RECORD_FORMAT_VERSION})"
            ),
            EngineError::NoWholeRecord => write!(
                f,
                "SSD checkpoint records are damaged and none is whole; pass --discard to overwrite"
            ),
            EngineError::OutputFileSizeMismatch(msg) => {
                write!(f, "Resumed run output file invalid: {msg}")
            }
            EngineError::CheckpointUnreadable(msg) => {
                write!(f, "Checkpoint could not be read whole: {msg}")
            }
            EngineError::StoppedForTest { serial, stage } => {
                write!(f, "Stopped after checkpoint serial={serial} ({stage:?})")
            }
            EngineError::FatalStop(msg) => write!(f, "Fatal stop (Rung 4): {msg}"),
        }
    }
}

pub struct EngineConfig<'a> {
    pub digits: u64,
    pub format: OutputFormat,
    pub output_id: String,
    pub discard_records: bool,
    pub profile: Profile,
    pub supplied_calibration: Option<Calibration>,
    pub supplied_member_rates: Option<HashMap<String, f64>>,
    pub calibration_cache_path: Option<std::path::PathBuf>,
    pub build_id: u64,
    pub overrides: PlanOverrides,
    pub kernels: &'a Kernels,
    pub members: Vec<Arc<dyn ExtentMember>>,
    pub output_factory: &'a dyn OutputSinkFactory,
    pub region_mapper: Option<&'static dyn RegionMapper>,
    pub thread_placer: Option<&'static dyn ThreadPlacer>,
    pub fft_err_bound: f64,
    pub mul_fault_hook: Option<&'a dyn MulFaultHook>,
    pub leaf_fault_hook: Option<&'a dyn LeafFaultHook>,
    pub stop_condition: Option<StopCondition>,
    pub progress_reporter: Option<&'a (dyn Fn(&str) + Send + Sync)>,
}

impl<'a> EngineConfig<'a> {
    pub fn new(
        digits: u64,
        format: OutputFormat,
        output_id: impl Into<String>,
        profile: Profile,
        kernels: &'a Kernels,
        output_factory: &'a dyn OutputSinkFactory,
    ) -> Self {
        Self {
            digits,
            format,
            output_id: output_id.into(),
            discard_records: false,
            profile,
            supplied_calibration: None,
            supplied_member_rates: None,
            calibration_cache_path: None,
            build_id: 0,
            overrides: PlanOverrides::default(),
            kernels,
            members: Vec::new(),
            output_factory,
            region_mapper: None,
            thread_placer: None,
            fft_err_bound: FFT_MAX_ERR_BOUND,
            mul_fault_hook: None,
            leaf_fault_hook: None,
            stop_condition: None,
            progress_reporter: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunReport {
    pub dec_hash_w0: u64,
    pub digits_in_doubt: bool,
    pub sieve_ops: u64,
    pub work_stats: WorkSnapshot,
    pub store_bytes_moved: u64,
    pub store_loader_batches: u64,
    pub store_tile_peak_bytes: u64,
    pub store_parked_peak_bytes: u64,
    pub store_parked_live_bytes: u64,
    pub checkpoints_written: u64,
    pub resumed_from_serial: Option<u64>,
    pub rollbacks: u64,
    pub excluded_devices: Vec<String>,
    pub arena_report: MappingReport,
    pub arena_usage: crate::arena::ArenaUsageStats,
    pub calibration_reused: bool,
    pub telemetry_summary: String,
    pub event_reports: Vec<String>,
    pub calib_sec: f64,
    pub series_sec: f64,
    pub finish_sec: f64,
    pub convert_sec: f64,
    pub wall_sec: f64,
    pub plan: Plan,
}

/// Select the SSD stripe for checkpoints by the Section 10 rate rule (`k * min_rate`, preferring
/// larger `k` on ties), keeping `k >= 2` whenever 2+ usable SSDs exist so `add_n`/`sub_n` parity
/// holds across every checkpoint round (Section 11).
pub fn select_checkpoint_ssds(
    ssds_with_rates: &[(Arc<dyn ExtentMember>, f64)],
) -> Vec<Arc<dyn ExtentMember>> {
    let mut pool: Vec<(Arc<dyn ExtentMember>, f64)> = ssds_with_rates
        .iter()
        .filter(|(m, _)| m.capacity() >= RECORD_REGION_BYTES + 65_536)
        .map(|(m, r)| (Arc::clone(m), r.max(1.0)))
        .collect();
    if pool.len() <= 1 {
        return pool.into_iter().map(|(m, _)| m).collect();
    }
    pool.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| b.0.capacity().cmp(&a.0.capacity()))
            .then_with(|| a.0.id().cmp(b.0.id()))
    });
    let mut best_k = 2usize;
    let mut best_rate = 2.0 * pool[1].1;
    for k in 3..=pool.len() {
        let cand_rate = (k as f64) * pool[k - 1].1;
        if cand_rate + 1e-6 >= best_rate {
            best_rate = cand_rate;
            best_k = k;
        }
    }
    pool.into_iter().take(best_k).map(|(m, _)| m).collect()
}

/// Compute checkpoint reservation on an SSD member (`[0 .. store_base_offset)`).
/// Sized so narrowing a stripe down to a single data column after SSD exclusions still fits.
fn ssd_checkpoint_layout(ssd_cap: u64, digits: u64, _num_ssds: usize) -> (u64, u64) {
    if ssd_cap < RECORD_REGION_BYTES + 65_536 {
        return (0, 0);
    }
    let align = 4096u64;
    let est_state_bytes = (((digits as f64) * plan::LOG2_10 / 8.0).ceil() as u64)
        .saturating_mul(7)
        .saturating_add(262_144);
    let per_ssd_need = (est_state_bytes + 65_536).div_ceil(align) * align;
    let max_allow = ((ssd_cap - RECORD_REGION_BYTES) / 3 / align) * align;
    let bank_bytes = per_ssd_need.min(max_allow).max(16_384);
    let store_base = RECORD_REGION_BYTES + 2 * bank_bytes;
    if store_base >= ssd_cap {
        (0, ssd_cap.min(RECORD_REGION_BYTES))
    } else {
        (bank_bytes, store_base)
    }
}

fn rerank_shares(shares: &mut Shares, core_ntt: &[(u64, f64)]) -> bool {
    if core_ntt.is_empty() || shares.os_places_threads {
        return false;
    }
    let before: Vec<(u64, u64)> = shares
        .workers
        .iter()
        .map(|w| (w.core_id, w.hw_thread))
        .collect();
    let mut tps = Vec::with_capacity(shares.workers.len());
    for w in &shares.workers {
        let rate = core_ntt
            .iter()
            .find(|&&(cid, _)| cid == w.core_id)
            .map_or(1.0, |&(_, r)| r);
        tps.push(rate);
    }
    shares.rank_by_throughput(&tps);
    shares
        .workers
        .iter()
        .map(|w| (w.core_id, w.hw_thread))
        .ne(before.into_iter())
}

/// Deterministic `--plan-only` entry point.
pub fn plan_only(cfg: &EngineConfig<'_>) -> Result<(Shares, Calibration, Plan), EngineError> {
    let mut shares = Shares::from_profile(&cfg.profile);
    let cal = if let Some(c) = &cfg.supplied_calibration {
        rerank_shares(&mut shares, &c.core_ntt_bf_per_sec);
        c.clone()
    } else {
        let sched = Scheduler::new(&shares, cfg.thread_placer);
        let c = Calibration::measure_with_cache(
            cfg.kernels,
            &cfg.profile,
            &shares,
            &sched,
            None,
            cfg.calibration_cache_path.as_deref(),
            cfg.build_id,
        );
        rerank_shares(&mut shares, &c.core_ntt_bf_per_sec);
        c
    };
    cfg.kernels.configure_from_shares(&shares);
    cfg.kernels
        .mul_base_rows_max
        .store(cal.mul_base_rows_max, std::sync::atomic::Ordering::Relaxed);
    cfg.kernels
        .sqr_base_rows_max
        .store(cal.sqr_base_rows_max, std::sync::atomic::Ordering::Relaxed);
    let plan = Plan::build(cfg.digits, cfg.format, &shares, &cal, cfg.overrides.clone())
        .map_err(EngineError::Plan)?;
    Ok((shares, cal, plan))
}

#[derive(Clone, Debug)]
enum LoadedResume {
    Series {
        term_count: u64,
        leaf_len: u64,
        dividing_mask: u64,
        state: SeriesResumeState,
    },
    Finish {
        q_root: BigNum,
        t_root: BigNum,
        state: NewtonResumeState,
    },
    Convert {
        leaf_digits: usize,
        state: RadixResumeState,
    },
}

struct DiscoveredCheckpoint {
    serial: u64,
    canonical: CheckpointRecord,
    slot_members: Vec<Option<Arc<dyn ExtentMember>>>,
}

fn parse_slot_prefix(ssd_member_id: &str) -> Option<usize> {
    let (left, _) = ssd_member_id.split_once(':')?;
    left.parse::<usize>().ok()
}

fn inspect_ssd_records(
    ssds: &[Arc<dyn ExtentMember>],
    kernels: &Kernels,
) -> Result<Option<DiscoveredCheckpoint>, EngineError> {
    if ssds.is_empty() {
        return Ok(None);
    }

    let mut valid_records: Vec<(Arc<dyn ExtentMember>, CheckpointRecord)> = Vec::new();
    let mut wrong_version: Option<u32> = None;
    let mut saw_corrupt_nonblank = false;
    let mut eligible_ssds = 0usize;
    let mut lost_ssds = 0usize;

    let mut slot_buf = vec![0u8; RECORD_SLOT_BYTES as usize];
    for ssd in ssds {
        if ssd.capacity() < RECORD_REGION_BYTES {
            continue;
        }
        eligible_ssds += 1;
        let mut best_on_ssd: Option<CheckpointRecord> = None;
        let mut ssd_lost = false;
        for slot_idx in 0..2u64 {
            slot_buf.fill(0);
            let range = [ByteRange {
                device_offset: slot_idx * RECORD_SLOT_BYTES,
                host_offset: 0,
                len: RECORD_SLOT_BYTES as usize,
            }];
            if let Err(e) = ssd.read(&range, &mut slot_buf) {
                if e == StoreError::DeviceLost {
                    ssd_lost = true;
                }
                continue;
            }
            if slot_buf.iter().all(|&b| b == 0) {
                continue;
            }
            match CheckpointRecord::decode(&slot_buf, kernels) {
                Ok(rec) => {
                    if best_on_ssd
                        .as_ref()
                        .is_none_or(|prev| rec.serial > prev.serial)
                    {
                        best_on_ssd = Some(rec);
                    }
                }
                Err(RecordError::WrongVersion(v)) => {
                    wrong_version = Some(v);
                }
                Err(RecordError::Corrupt | RecordError::EmptyOrTruncated) => {
                    saw_corrupt_nonblank = true;
                }
            }
        }
        if ssd_lost {
            lost_ssds += 1;
        }
        if let Some(rec) = best_on_ssd {
            valid_records.push((Arc::clone(ssd), rec));
        }
    }

    if valid_records.is_empty() {
        if eligible_ssds > 0 && lost_ssds == eligible_ssds {
            return Err(EngineError::FatalStop(format!(
                "Last SSD '{}' stopped answering during record inspection",
                ssds[0].id()
            )));
        }
        if let Some(v) = wrong_version {
            return Err(EngineError::WrongRecordVersion(v));
        }
        if saw_corrupt_nonblank {
            return Err(EngineError::NoWholeRecord);
        }
        return Ok(None);
    }

    let newest_serial = valid_records.iter().map(|(_, r)| r.serial).max().unwrap();
    let newest_set: Vec<(Arc<dyn ExtentMember>, CheckpointRecord)> = valid_records
        .into_iter()
        .filter(|(_, r)| r.serial == newest_serial)
        .collect();

    let canonical = newest_set[0].1.clone();
    let width = (canonical.stripe_width as usize).max(1);
    let mut slot_members: Vec<Option<Arc<dyn ExtentMember>>> = vec![None; width];
    for (ssd, rec) in newest_set {
        let idx = parse_slot_prefix(&rec.ssd_member_id).unwrap_or(0);
        if idx < width {
            slot_members[idx] = Some(ssd);
        }
    }

    Ok(Some(DiscoveredCheckpoint {
        serial: newest_serial,
        canonical,
        slot_members,
    }))
}

fn clear_all_ssd_records(ssds: &[Arc<dyn ExtentMember>]) {
    let zeros = vec![0u8; RECORD_REGION_BYTES as usize];
    let range = [ByteRange {
        device_offset: 0,
        host_offset: 0,
        len: RECORD_REGION_BYTES as usize,
    }];
    for ssd in ssds {
        if ssd.capacity() >= RECORD_REGION_BYTES {
            let _ = ssd.write(&range, &zeros);
        }
    }
}

fn piece_slot_stride(ssds: &[Option<Arc<dyn ExtentMember>>], piece_limbs: usize) -> u64 {
    let max_align = ssds
        .iter()
        .flatten()
        .map(|m| m.alignment().max(4096))
        .max()
        .unwrap_or(4096);
    let raw = (piece_limbs as u64) * 8 + 24;
    raw.div_ceil(max_align) * max_align
}

/// Reconstruct all saved numbers from a `DiscoveredCheckpoint`, repairing single-limb piece errors
/// in place (Rung 1) and reconstructing any 1 missing/damaged SSD column per round via `add_n` /
/// `sub_n` parity (Rung 2, Section 11).
fn load_checkpoint_state(
    disc: &DiscoveredCheckpoint,
    kernels: &Kernels,
    telemetry: &Telemetry,
) -> Result<LoadedResume, EngineError> {
    let rec = &disc.canonical;
    let m = (rec.stripe_width as usize).max(1);
    let piece_limbs = (rec.piece_limbs as usize).max(1);
    let slot_bytes = piece_slot_stride(&disc.slot_members, piece_limbs) as usize;
    let data_cols = if m >= 2 { m - 1 } else { 1 };

    let total_saved_limbs: usize = rec.numbers.iter().map(|n| n.limb_len as usize).sum();
    let total_data_pieces = total_saved_limbs.div_ceil(piece_limbs);
    let num_rounds = total_data_pieces.div_ceil(data_cols);

    let mut flat_limbs = Vec::with_capacity(num_rounds * data_cols * piece_limbs);
    let mut raw_buf = vec![0u8; slot_bytes];

    for r_idx in 0..num_rounds {
        let dev_off = rec.data_offset_bytes + (r_idx as u64) * (slot_bytes as u64);
        let mut round_pieces: Vec<Option<Vec<u64>>> = vec![None; m];

        for col in 0..m {
            let Some(ssd) = &disc.slot_members[col] else {
                continue;
            };
            raw_buf.fill(0);
            let range = [ByteRange {
                device_offset: dev_off,
                host_offset: 0,
                len: slot_bytes,
            }];
            if let Err(e) = ssd.read(&range, &mut raw_buf) {
                if e == StoreError::DeviceLost {
                    telemetry.mark_device_stopped(ssd.id());
                }
                telemetry.record_retry(0, 0, Some(ssd.id()));
                continue;
            }
            telemetry.record_scrub();
            let data_bytes = piece_limbs * 8;
            let mut limbs = Vec::with_capacity(piece_limbs);
            for chunk in raw_buf[..data_bytes].chunks_exact(8) {
                limbs.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
            let exp_w0 =
                u64::from_le_bytes(raw_buf[data_bytes..(data_bytes + 8)].try_into().unwrap());
            let exp_w1 = u64::from_le_bytes(
                raw_buf[(data_bytes + 8)..(data_bytes + 16)]
                    .try_into()
                    .unwrap(),
            );
            let exp_w2 = u64::from_le_bytes(
                raw_buf[(data_bytes + 16)..(data_bytes + 24)]
                    .try_into()
                    .unwrap(),
            );
            let expected = [exp_w0, exp_w1, exp_w2];
            let actual = compute_limbs_all3(kernels, &limbs);
            if actual == expected {
                round_pieces[col] = Some(limbs);
            } else {
                // Rung 1: check 2-of-3 check-word agreement OR single-limb repair!
                let match_cnt = usize::from(actual[0] == expected[0])
                    + usize::from(actual[1] == expected[1])
                    + usize::from(actual[2] == expected[2]);
                let repaired = if match_cnt >= 2 {
                    telemetry.record_correction(0, 0, Some(ssd.id()));
                    true
                } else {
                    match try_correct_single_limb(&mut limbs, actual, expected) {
                        CorrectionOutcome::Clean | CorrectionOutcome::Corrected { .. } => {
                            telemetry.record_correction(0, 0, Some(ssd.id()));
                            true
                        }
                        CorrectionOutcome::BeyondOneLimb => false,
                    }
                };

                if repaired {
                    let fresh = compute_limbs_all3(kernels, &limbs);
                    for (i, &w) in limbs.iter().enumerate() {
                        raw_buf[(i * 8)..((i + 1) * 8)].copy_from_slice(&w.to_le_bytes());
                    }
                    raw_buf[data_bytes..(data_bytes + 8)].copy_from_slice(&fresh[0].to_le_bytes());
                    raw_buf[(data_bytes + 8)..(data_bytes + 16)]
                        .copy_from_slice(&fresh[1].to_le_bytes());
                    raw_buf[(data_bytes + 16)..(data_bytes + 24)]
                        .copy_from_slice(&fresh[2].to_le_bytes());
                    if ssd.write(&range, &raw_buf).is_err() {
                        telemetry.mark_device_stopped(ssd.id());
                        telemetry.record_retry(0, 0, Some(ssd.id()));
                    }
                    round_pieces[col] = Some(limbs);
                } else {
                    telemetry.record_retry(0, 0, Some(ssd.id()));
                }
            }
        }

        // Check if any data column `0..data_cols` is missing and reconstruct from parity round!
        for col in 0..data_cols {
            if round_pieces[col].is_none() {
                if m < 2 {
                    return Err(EngineError::CheckpointUnreadable(format!(
                        "Single-SSD checkpoint piece at round {r_idx} failed check words beyond 1 limb"
                    )));
                }
                let refs: Vec<Option<&[u64]>> =
                    round_pieces.iter().map(|opt| opt.as_deref()).collect();
                let mut recov = vec![0u64; piece_limbs];
                if !reconstruct_missing_piece(kernels, &refs, col, &mut recov) {
                    return Err(EngineError::CheckpointUnreadable(format!(
                        "More than one SSD piece lost in checkpoint round {r_idx}"
                    )));
                }
                telemetry.record_retry(0, 0, None);
                round_pieces[col] = Some(recov);
            }
            flat_limbs.extend_from_slice(round_pieces[col].as_ref().unwrap());
        }
    }

    // Slice `flat_limbs` back into individual saved numbers & verify every number's `w0`!
    let mut cursor = 0usize;
    let mut restored: Vec<(SavedNumberMeta, BigNum)> = Vec::with_capacity(rec.numbers.len());
    for meta in &rec.numbers {
        let len = meta.limb_len as usize;
        if cursor + len > flat_limbs.len() {
            return Err(EngineError::CheckpointUnreadable(
                "Checkpoint payload shorter than saved number metadata".to_string(),
            ));
        }
        let limbs = crate::limb::slice_to_limb_vec(&flat_limbs[cursor..(cursor + len)]);
        cursor += len;
        let raw_w0 = compute_limbs_w0(kernels, &limbs);
        let full_w0 = whole_value_w0(raw_w0, meta.neg, meta.shift);
        if full_w0 != meta.w0 {
            return Err(EngineError::CheckpointUnreadable(format!(
                "Saved number tag={:#x} check word mismatch (expected {}, got {})",
                meta.tag, meta.w0, full_w0
            )));
        }
        let bn = BigNum {
            limbs,
            neg: meta.neg,
            shift: meta.shift,
            chk: DualCheck::new(full_w0),
        };
        restored.push((meta.clone(), bn));
    }

    match &rec.stage {
        StageRecord::Series {
            term_count,
            leaf_len,
            dividing_mask,
            ..
        } => {
            let mut nodes: HashMap<(u32, u64), BlockTriple> = HashMap::new();
            for (meta, bn) in restored {
                let lv = (meta.tag >> 56) as u32;
                let node_idx = (meta.tag >> 2) & ((1u64 << 54) - 1);
                let which = meta.tag & 3;
                let entry = nodes.entry((lv, node_idx)).or_insert_with(|| BlockTriple {
                    p: BigNum::zero(),
                    q: BigNum::zero(),
                    t: BigNum::zero(),
                });
                match which {
                    0 => entry.p = bn,
                    1 => entry.q = bn,
                    2 => entry.t = bn,
                    _ => {}
                }
            }
            Ok(LoadedResume::Series {
                term_count: *term_count,
                leaf_len: *leaf_len,
                dividing_mask: *dividing_mask,
                state: SeriesResumeState { nodes },
            })
        }
        StageRecord::Finish {
            precision_limbs, ..
        } => {
            let mut q_root = BigNum::zero();
            let mut t_root = BigNum::zero();
            let mut z = BigNum::zero();
            let mut u = BigNum::zero();
            let mut step_limbs = *precision_limbs as usize;
            let mut acc_bits = (*precision_limbs) * 64 - 32;
            for (meta, bn) in restored {
                match meta.tag {
                    0 => q_root = bn,
                    1 => t_root = bn,
                    2 => z = bn,
                    3 => u = bn,
                    4 if bn.limbs.len() >= 2 => {
                        step_limbs = bn.limbs[0] as usize;
                        acc_bits = bn.limbs[1];
                    }
                    _ => {}
                }
            }
            Ok(LoadedResume::Finish {
                q_root,
                t_root,
                state: NewtonResumeState {
                    step_limbs,
                    acc_bits,
                    z,
                    u,
                },
            })
        }
        StageRecord::Convert { leaf_digits, .. } => {
            let mut headers: HashMap<u64, Vec<u64>> = HashMap::new();
            let mut payloads: HashMap<u64, Vec<u64>> = HashMap::new();
            let mut completed_w0: HashMap<(u64, u64), u64> = HashMap::new();
            for (meta, bn) in restored {
                let kind = meta.tag & 0xF000_0000;
                let idx = meta.tag & 0x0FFF_FFFF;
                if kind == 0x1000_0000 {
                    headers.insert(idx, bn.into_limbs().to_vec());
                } else if kind == 0x2000_0000 {
                    payloads.insert(idx, bn.into_limbs().to_vec());
                } else if meta.tag == 0x3000_0000 {
                    for ch in bn.limbs.chunks_exact(3) {
                        completed_w0.insert((ch[0], ch[1]), ch[2]);
                    }
                }
            }
            let mut indices: Vec<u64> = headers.keys().copied().collect();
            indices.sort_unstable();
            let mut fractions = Vec::with_capacity(indices.len());
            for idx in indices {
                let hdr = &headers[&idx];
                if hdr.len() >= 4 {
                    let frac_limbs = payloads.remove(&idx).unwrap_or_default();
                    let req = if hdr[3] == u64::MAX {
                        None
                    } else {
                        Some(hdr[3])
                    };
                    fractions.push(PendingFraction {
                        start_leaf: hdr[0],
                        leaf_count: hdr[1],
                        frac_limbs,
                        frac_k_limbs: hdr[2] as usize,
                        required_w0: req,
                    });
                }
            }
            Ok(LoadedResume::Convert {
                leaf_digits: *leaf_digits as usize,
                state: RadixResumeState {
                    fractions,
                    completed_w0,
                },
            })
        }
    }
}

struct CheckpointWriterState {
    ssd_pool: Vec<(Arc<dyn ExtentMember>, f64)>,
    excluded_devices: Vec<String>,
    event_reports: Vec<String>,
    next_serial: u64,
    bank_bytes: u64,
    checkpoints_written: u64,
    truncated_once: bool,
    stopped_stage: Option<(u64, CheckpointStageKind)>,
}

struct CheckpointManager<'a> {
    kernels: &'a Kernels,
    telemetry: &'a Telemetry,
    digits: u64,
    format: OutputFormat,
    output_id: String,
    digit_sink: &'a dyn DigitSink,
    est_write_sec: f64,
    checkpoint_mult: f64,
    stop_condition: Option<StopCondition>,
    progress_reporter: Option<&'a (dyn Fn(&str) + Send + Sync)>,
    state: Mutex<CheckpointWriterState>,
}

fn pack_ckpt_piece_slot(kernels: &Kernels, limbs: &[u64], wbuf: &mut [u8]) {
    let dbytes = CKPT_PIECE_LIMBS * 8;
    wbuf.fill(0);
    let [w0, w1, w2] = compute_limbs_all3(kernels, limbs);
    for (i, &w) in limbs.iter().enumerate() {
        wbuf[(i * 8)..((i + 1) * 8)].copy_from_slice(&w.to_le_bytes());
    }
    wbuf[dbytes..(dbytes + 8)].copy_from_slice(&w0.to_le_bytes());
    wbuf[(dbytes + 8)..(dbytes + 16)].copy_from_slice(&w1.to_le_bytes());
    wbuf[(dbytes + 16)..(dbytes + 24)].copy_from_slice(&w2.to_le_bytes());
}

impl<'a> CheckpointManager<'a> {
    fn should_write(&self, work_sec_since_last: f64) -> bool {
        let g = self.state.lock().unwrap();
        if g.ssd_pool.is_empty() || g.bank_bytes == 0 {
            return false;
        }
        if self.checkpoint_mult <= 0.0 {
            return true;
        }
        work_sec_since_last >= self.checkpoint_mult * self.est_write_sec
    }

    fn write_stage_checkpoint(
        &self,
        stage_kind: CheckpointStageKind,
        stage_rec: StageRecord,
        mut items: Vec<(u64, BigNum)>,
    ) -> Result<bool, MulError> {
        // Patrol check every number at rest before writing checkpoint (Section 11)!
        for (tag, bn) in &mut items {
            self.telemetry.record_scrub();
            if !bn.verify_at_rest(self.kernels, self.telemetry) {
                return Err(MulError::NeedRollback(
                    0xC000_0000_0000_0000 | (*tag & 0x0FFF_FFFF_FFFF_FFFF),
                ));
            }
        }

        if stage_kind == CheckpointStageKind::Convert {
            // Section 11: flush output file to device before a Convert checkpoint counts!
            self.digit_sink.flush().map_err(MulError::FatalStop)?;
        }

        let mut metas = Vec::with_capacity(items.len());
        let mut flat_limbs = Vec::new();
        for (tag, bn) in &items {
            metas.push(SavedNumberMeta {
                tag: *tag,
                neg: bn.neg,
                shift: bn.shift,
                limb_len: bn.limbs.len() as u64,
                w0: bn.chk.c0,
            });
            flat_limbs.extend_from_slice(&bn.limbs);
        }

        let mut g = self.state.lock().unwrap();
        loop {
            let ckpt_ssds = select_checkpoint_ssds(&g.ssd_pool);
            if ckpt_ssds.is_empty() || g.bank_bytes == 0 {
                return Ok(false);
            }

            let m = ckpt_ssds.len();
            let data_cols = if m >= 2 { m - 1 } else { 1 };
            let max_align = ckpt_ssds
                .iter()
                .map(|s| s.alignment().max(4096))
                .max()
                .unwrap_or(4096);
            let slot_bytes =
                (((CKPT_PIECE_LIMBS as u64) * 8 + 24).div_ceil(max_align) * max_align) as usize;
            let dbytes = CKPT_PIECE_LIMBS * 8;

            let serial = g.next_serial;
            let bank_idx = serial & 1;
            let data_offset_bytes = RECORD_REGION_BYTES + bank_idx * g.bank_bytes;

            let total_data_pieces = flat_limbs.len().div_ceil(CKPT_PIECE_LIMBS).max(1);
            let num_rounds = total_data_pieces.div_ceil(data_cols);
            let needed_bank = (num_rounds as u64) * (slot_bytes as u64);
            if needed_bank > g.bank_bytes {
                return Ok(false);
            }

            let build_round_cols = |r_idx: usize, parity_buf: &mut [u64]| -> Vec<Vec<u64>> {
                let mut col_limbs: Vec<Vec<u64>> = Vec::with_capacity(m);
                for c in 0..data_cols {
                    let p_idx = r_idx * data_cols + c;
                    let start = p_idx * CKPT_PIECE_LIMBS;
                    let mut piece = vec![0u64; CKPT_PIECE_LIMBS];
                    if start < flat_limbs.len() {
                        let end = (start + CKPT_PIECE_LIMBS).min(flat_limbs.len());
                        piece[..(end - start)].copy_from_slice(&flat_limbs[start..end]);
                    }
                    col_limbs.push(piece);
                }
                if m >= 2 {
                    let refs: Vec<&[u64]> = col_limbs.iter().map(|v| v.as_slice()).collect();
                    build_parity_piece(self.kernels, &refs, parity_buf);
                    col_limbs.push(parity_buf.to_vec());
                }
                col_limbs
            };

            const BATCH_ROUNDS: usize = 256;
            let batch_max_rounds = num_rounds.clamp(1, BATCH_ROUNDS);
            let batch_cap_bytes = batch_max_rounds * slot_bytes;
            let slot_align = (max_align as usize).max(65_536);
            let dummy_tr = || {
                Arc::new((
                    std::sync::atomic::AtomicU64::new(0),
                    std::sync::atomic::AtomicU64::new(0),
                ))
            };
            let mut bad_ssd_id: Option<(String, StoreError)> = None;
            let mut parity_limbs = vec![0u64; CKPT_PIECE_LIMBS];
            let mut wbuf_slot =
                crate::store::AlignedSlotBuf::new_tracked(batch_cap_bytes, slot_align, dummy_tr());
            let mut rbuf_cols: Vec<crate::store::AlignedSlotBuf> = (0..m)
                .map(|_| {
                    crate::store::AlignedSlotBuf::new_tracked(
                        batch_cap_bytes,
                        slot_align,
                        dummy_tr(),
                    )
                })
                .collect();
            let mut single_w =
                crate::store::AlignedSlotBuf::new_tracked(slot_bytes, slot_align, dummy_tr());
            let mut single_r =
                crate::store::AlignedSlotBuf::new_tracked(slot_bytes, slot_align, dummy_tr());

            // Phase 1: Write all rounds across the selected checkpoint SSD stripe in aligned DMA batches.
            let mut r_base = 0usize;
            while r_base < num_rounds && bad_ssd_id.is_none() {
                let b_cnt = (num_rounds - r_base).min(BATCH_ROUNDS);
                let b_bytes = b_cnt * slot_bytes;
                let batch_cols: Vec<Vec<Vec<u64>>> = (r_base..(r_base + b_cnt))
                    .map(|r_idx| build_round_cols(r_idx, &mut parity_limbs))
                    .collect();
                for (col, ssd) in ckpt_ssds.iter().enumerate() {
                    let w_all = unsafe { &mut wbuf_slot.as_import_slice_mut()[..b_bytes] };
                    for (bi, col_limbs) in batch_cols.iter().enumerate() {
                        let s_off = bi * slot_bytes;
                        pack_ckpt_piece_slot(
                            self.kernels,
                            &col_limbs[col],
                            &mut w_all[s_off..(s_off + slot_bytes)],
                        );
                    }
                    let b_range = [ByteRange {
                        device_offset: data_offset_bytes + (r_base as u64) * (slot_bytes as u64),
                        host_offset: 0,
                        len: b_bytes,
                    }];
                    let t_bw = Instant::now();
                    if ssd.write(&b_range, w_all).is_ok()
                        && t_bw.elapsed().as_secs_f64() <= MEMBER_CALL_TIMEOUT_SEC
                    {
                        continue;
                    }
                    for bi in 0..b_cnt {
                        let r_idx = r_base + bi;
                        let s_off = bi * slot_bytes;
                        let sw = unsafe { &mut single_w.as_import_slice_mut()[..slot_bytes] };
                        sw.copy_from_slice(&w_all[s_off..(s_off + slot_bytes)]);
                        let range = [ByteRange {
                            device_offset: data_offset_bytes + (r_idx as u64) * (slot_bytes as u64),
                            host_offset: 0,
                            len: slot_bytes,
                        }];
                        let mut wrote = false;
                        let mut last_err = StoreError::IoFailed;
                        for attempt in 0..2 {
                            if attempt > 0 {
                                self.telemetry.record_retry(0, 0, Some(ssd.id()));
                            }
                            let t_w = Instant::now();
                            match ssd.write(&range, sw) {
                                Ok(())
                                    if t_w.elapsed().as_secs_f64() <= MEMBER_CALL_TIMEOUT_SEC =>
                                {
                                    wrote = true;
                                    break;
                                }
                                Ok(()) => {
                                    self.telemetry.mark_device_stopped(ssd.id());
                                    last_err = StoreError::DeviceLost;
                                }
                                Err(e) => {
                                    self.telemetry.mark_device_stopped(ssd.id());
                                    last_err = e;
                                }
                            }
                        }
                        if !wrote {
                            bad_ssd_id = Some((ssd.id().to_string(), last_err));
                            break;
                        }
                    }
                    if bad_ssd_id.is_some() {
                        break;
                    }
                }
                r_base += b_cnt;
            }

            // Phase 2: Read back the ENTIRE checkpoint whole across all rounds and columns,
            // retrying any disagreeing piece write once on Rung 2 before writing the record
            // (Section 11: "A checkpoint counts once it has been read back whole and its record
            // is written, and the one before it is kept until then").
            let mut reassembled_limbs = Vec::with_capacity(if bad_ssd_id.is_none() {
                num_rounds * data_cols * CKPT_PIECE_LIMBS
            } else {
                0
            });
            if bad_ssd_id.is_none() {
                let mut r_base = 0usize;
                while r_base < num_rounds && bad_ssd_id.is_none() {
                    let b_cnt = (num_rounds - r_base).min(BATCH_ROUNDS);
                    let b_bytes = b_cnt * slot_bytes;
                    let b_range = [ByteRange {
                        device_offset: data_offset_bytes + (r_base as u64) * (slot_bytes as u64),
                        host_offset: 0,
                        len: b_bytes,
                    }];
                    let mut col_batch_ok = vec![false; m];
                    for (col, ssd) in ckpt_ssds.iter().enumerate() {
                        let r_all = unsafe { &mut rbuf_cols[col].as_import_slice_mut()[..b_bytes] };
                        r_all.fill(0);
                        let t_br = Instant::now();
                        let br_res = ssd.read(&b_range, r_all);
                        let br_to = t_br.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC;
                        match br_res {
                            Ok(()) if !br_to => {
                                col_batch_ok[col] = true;
                            }
                            Ok(()) => {
                                self.telemetry.mark_device_stopped(ssd.id());
                            }
                            Err(e) => {
                                if e == StoreError::DeviceLost || br_to {
                                    self.telemetry.mark_device_stopped(ssd.id());
                                }
                            }
                        }
                    }

                    for bi in 0..b_cnt {
                        let r_idx = r_base + bi;
                        let col_limbs = build_round_cols(r_idx, &mut parity_limbs);
                        let dev_off = data_offset_bytes + (r_idx as u64) * (slot_bytes as u64);
                        let range = [ByteRange {
                            device_offset: dev_off,
                            host_offset: 0,
                            len: slot_bytes,
                        }];
                        let s_off = bi * slot_bytes;

                        for (col, ssd) in ckpt_ssds.iter().enumerate() {
                            let wbuf = unsafe { &mut single_w.as_import_slice_mut()[..slot_bytes] };
                            pack_ckpt_piece_slot(self.kernels, &col_limbs[col], wbuf);
                            let r_piece = unsafe {
                                &mut rbuf_cols[col].as_import_slice_mut()
                                    [s_off..(s_off + slot_bytes)]
                            };
                            let first_ok = if col_batch_ok[col] {
                                self.telemetry.record_scrub();
                                r_piece == wbuf
                            } else {
                                let rbuf =
                                    unsafe { &mut single_r.as_import_slice_mut()[..slot_bytes] };
                                rbuf.fill(0);
                                let t_r = Instant::now();
                                let read_res = ssd.read(&range, rbuf);
                                let r_timed_out =
                                    t_r.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC;
                                self.telemetry.record_scrub();
                                match read_res {
                                    Ok(()) if !r_timed_out && rbuf == wbuf => {
                                        r_piece.copy_from_slice(rbuf);
                                        true
                                    }
                                    Ok(()) if r_timed_out => {
                                        self.telemetry.mark_device_stopped(ssd.id());
                                        false
                                    }
                                    Ok(()) => false,
                                    Err(e) => {
                                        if e == StoreError::DeviceLost || r_timed_out {
                                            self.telemetry.mark_device_stopped(ssd.id());
                                        }
                                        false
                                    }
                                }
                            };

                            if !first_ok {
                                // Rung 2: write the disagreeing piece again and read it back!
                                self.telemetry.record_retry(0, 0, Some(ssd.id()));
                                let t_w2 = Instant::now();
                                let w2_res = ssd.write(&range, wbuf);
                                let w2_to = t_w2.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC;
                                let mut retry_ok = false;
                                let mut last_err = StoreError::IntegrityBeyondRepair;
                                match w2_res {
                                    Ok(()) if !w2_to => {
                                        let rbuf = unsafe {
                                            &mut single_r.as_import_slice_mut()[..slot_bytes]
                                        };
                                        rbuf.fill(0);
                                        let t_r2 = Instant::now();
                                        match ssd.read(&range, rbuf) {
                                            Ok(())
                                                if t_r2.elapsed().as_secs_f64()
                                                    <= MEMBER_CALL_TIMEOUT_SEC
                                                    && rbuf == wbuf =>
                                            {
                                                self.telemetry.record_scrub();
                                                r_piece.copy_from_slice(rbuf);
                                                retry_ok = true;
                                            }
                                            Ok(())
                                                if t_r2.elapsed().as_secs_f64()
                                                    > MEMBER_CALL_TIMEOUT_SEC =>
                                            {
                                                self.telemetry.mark_device_stopped(ssd.id());
                                                last_err = StoreError::DeviceLost;
                                            }
                                            Ok(()) => {
                                                self.telemetry.record_scrub();
                                                self.telemetry.record_failure(0, 0, Some(ssd.id()));
                                                last_err = StoreError::IntegrityBeyondRepair;
                                            }
                                            Err(e) => {
                                                if e == StoreError::DeviceLost {
                                                    self.telemetry.mark_device_stopped(ssd.id());
                                                }
                                                self.telemetry.record_failure(0, 0, Some(ssd.id()));
                                                last_err = e;
                                            }
                                        }
                                    }
                                    Ok(()) => {
                                        self.telemetry.mark_device_stopped(ssd.id());
                                        self.telemetry.record_failure(0, 0, Some(ssd.id()));
                                        last_err = StoreError::DeviceLost;
                                    }
                                    Err(e) => {
                                        self.telemetry.mark_device_stopped(ssd.id());
                                        self.telemetry.record_failure(0, 0, Some(ssd.id()));
                                        last_err = e;
                                    }
                                }
                                if !retry_ok {
                                    bad_ssd_id = Some((ssd.id().to_string(), last_err));
                                    break;
                                }
                            }

                            if col < data_cols {
                                for chunk in r_piece[..dbytes].chunks_exact(8) {
                                    reassembled_limbs
                                        .push(u64::from_le_bytes(chunk.try_into().unwrap()));
                                }
                            }
                        }
                        if bad_ssd_id.is_some() {
                            break;
                        }
                    }
                    r_base += b_cnt;
                }
            }

            if let Some((bad_id, err)) = bad_ssd_id {
                if g.ssd_pool.len() > 1 {
                    g.ssd_pool.retain(|(s, _)| s.id() != bad_id);
                    if !g.excluded_devices.iter().any(|d| d == &bad_id) {
                        g.excluded_devices.push(bad_id);
                    }
                    continue;
                } else {
                    if err == StoreError::DeviceLost || self.telemetry.is_device_stopped(&bad_id) {
                        return Err(MulError::FatalStop(format!(
                            "Last SSD '{bad_id}' stopped answering during checkpoint write"
                        )));
                    }
                    return Err(MulError::NeedRollback(0xC200_0000_0000_0000 | serial));
                }
            }

            // Verify every reassembled saved number's `w0` whole before writing the record!
            let mut cursor = 0usize;
            for meta in &metas {
                let len = meta.limb_len as usize;
                if cursor + len > reassembled_limbs.len() {
                    return Err(MulError::NeedRollback(
                        0xC100_0000_0000_0000 | (meta.tag & 0x0FFF_FFFF_FFFF_FFFF),
                    ));
                }
                let raw_w0 = compute_limbs_w0(self.kernels, &reassembled_limbs[cursor..(cursor + len)]);
                cursor += len;
                if whole_value_w0(raw_w0, meta.neg, meta.shift) != meta.w0 {
                    return Err(MulError::NeedRollback(
                        0xC100_0000_0000_0000 | (meta.tag & 0x0FFF_FFFF_FFFF_FFFF),
                    ));
                }
            }

            if self.stop_condition == Some(StopCondition::TruncateRecordWriteOnce)
                && !g.truncated_once
            {
                g.truncated_once = true;
                let rec_slot_off = (serial & 1) * RECORD_SLOT_BYTES;
                let rec = CheckpointRecord {
                    format_version: RECORD_FORMAT_VERSION,
                    serial,
                    ssd_member_id: format!("0:{}", ckpt_ssds[0].id()),
                    digits: self.digits,
                    output_path: self.output_id.clone(),
                    format_text: self.format == OutputFormat::Text,
                    piece_limbs: CKPT_PIECE_LIMBS as u64,
                    stripe_width: m as u32,
                    data_offset_bytes,
                    excluded_devices: g.excluded_devices.clone(),
                    stage: stage_rec.clone(),
                    numbers: metas.clone(),
                };
                let full_frame = rec.encode(self.kernels);
                let mut partial_slot = vec![0u8; RECORD_SLOT_BYTES as usize];
                let cut = (full_frame.len() / 2).max(12);
                partial_slot[..cut].copy_from_slice(&full_frame[..cut]);
                let range = [ByteRange {
                    device_offset: rec_slot_off,
                    host_offset: 0,
                    len: RECORD_SLOT_BYTES as usize,
                }];
                let _ = ckpt_ssds[0].write(&range, &partial_slot);
                g.stopped_stage = Some((serial, stage_kind));
                return Err(MulError::FatalStop("TEST_STOP".to_string()));
            }

            let rec_slot_off = (serial & 1) * RECORD_SLOT_BYTES;
            let rec_range = [ByteRange {
                device_offset: rec_slot_off,
                host_offset: 0,
                len: RECORD_SLOT_BYTES as usize,
            }];

            let mut bad_rec_ssd: Option<(String, StoreError)> = None;
            for (col, ssd) in ckpt_ssds.iter().enumerate() {
                let rec = CheckpointRecord {
                    format_version: RECORD_FORMAT_VERSION,
                    serial,
                    ssd_member_id: format!("{col}:{}", ssd.id()),
                    digits: self.digits,
                    output_path: self.output_id.clone(),
                    format_text: self.format == OutputFormat::Text,
                    piece_limbs: CKPT_PIECE_LIMBS as u64,
                    stripe_width: m as u32,
                    data_offset_bytes,
                    excluded_devices: g.excluded_devices.clone(),
                    stage: stage_rec.clone(),
                    numbers: metas.clone(),
                };
                let encoded = rec.encode(self.kernels);
                if encoded.len() > RECORD_SLOT_BYTES as usize {
                    return Ok(false);
                }
                let mut slot_w = vec![0u8; RECORD_SLOT_BYTES as usize];
                slot_w[..encoded.len()].copy_from_slice(&encoded);
                let mut slot_r = vec![0u8; RECORD_SLOT_BYTES as usize];

                let mut ok = false;
                let mut last_err = StoreError::IoFailed;
                for attempt in 0..2 {
                    if attempt > 0 {
                        self.telemetry.record_retry(0, 0, Some(ssd.id()));
                    }
                    let t_w = Instant::now();
                    match ssd.write(&rec_range, &slot_w) {
                        Ok(()) if t_w.elapsed().as_secs_f64() <= MEMBER_CALL_TIMEOUT_SEC => {
                            let t_r = Instant::now();
                            match ssd.read(&rec_range, &mut slot_r) {
                                Ok(())
                                    if t_r.elapsed().as_secs_f64() <= MEMBER_CALL_TIMEOUT_SEC
                                        && slot_r[..encoded.len()] == encoded[..] =>
                                {
                                    self.telemetry.record_scrub();
                                    if CheckpointRecord::decode(&slot_r, self.kernels).as_ref()
                                        == Ok(&rec)
                                    {
                                        ok = true;
                                        break;
                                    }
                                    last_err = StoreError::IntegrityBeyondRepair;
                                }
                                Ok(())
                                    if t_r.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC =>
                                {
                                    self.telemetry.mark_device_stopped(ssd.id());
                                    last_err = StoreError::DeviceLost;
                                }
                                Ok(()) => {
                                    self.telemetry.record_scrub();
                                    last_err = StoreError::IntegrityBeyondRepair;
                                }
                                Err(e) => {
                                    if e == StoreError::DeviceLost {
                                        self.telemetry.mark_device_stopped(ssd.id());
                                    }
                                    last_err = e;
                                }
                            }
                        }
                        Ok(()) => {
                            self.telemetry.mark_device_stopped(ssd.id());
                            last_err = StoreError::DeviceLost;
                        }
                        Err(e) => {
                            self.telemetry.mark_device_stopped(ssd.id());
                            last_err = e;
                        }
                    }
                }
                if !ok {
                    self.telemetry.record_failure(0, 0, Some(ssd.id()));
                    bad_rec_ssd = Some((ssd.id().to_string(), last_err));
                    break;
                }
            }

            if let Some((bad_id, err)) = bad_rec_ssd {
                if g.ssd_pool.len() > 1 {
                    g.ssd_pool.retain(|(s, _)| s.id() != bad_id);
                    if !g.excluded_devices.iter().any(|d| d == &bad_id) {
                        g.excluded_devices.push(bad_id);
                    }
                    continue;
                } else {
                    if err == StoreError::DeviceLost || self.telemetry.is_device_stopped(&bad_id) {
                        return Err(MulError::FatalStop(format!(
                            "Last SSD '{bad_id}' stopped answering during record write"
                        )));
                    }
                    return Err(MulError::NeedRollback(0xC300_0000_0000_0000 | serial));
                }
            }

            g.next_serial += 1;
            g.checkpoints_written += 1;

            let ckpt_report = format!(
                "Checkpoint serial={serial} stage={stage_kind:?} | {}",
                self.telemetry.summary_line()
            );
            g.event_reports.push(ckpt_report.clone());
            if let Some(cb) = self.progress_reporter {
                cb(&ckpt_report);
            }

            let should_stop = match self.stop_condition {
                Some(StopCondition::AfterAnyCheckpoint(target_n)) => {
                    g.checkpoints_written >= target_n
                }
                Some(StopCondition::AfterStageCheckpoint(target_st)) => target_st == stage_kind,
                _ => false,
            };
            if should_stop {
                g.stopped_stage = Some((serial, stage_kind));
                return Err(MulError::FatalStop("TEST_STOP".to_string()));
            }

            return Ok(true);
        }
    }
}

struct SeriesCkptAdapter<'a, 'b> {
    mgr: &'a CheckpointManager<'b>,
    term_count: u64,
    leaf_len: u64,
    dividing_mask: u64,
}

impl SeriesCheckpointSink for SeriesCkptAdapter<'_, '_> {
    fn maybe_checkpoint_series(
        &self,
        completed_level: u32,
        live_nodes: &[(u32, u64, &BlockTriple)],
        work_sec_since_last: f64,
    ) -> Result<bool, MulError> {
        if !self.mgr.should_write(work_sec_since_last) {
            return Ok(false);
        }
        let mut items = Vec::with_capacity(live_nodes.len() * 3);
        for &(lv, idx, tr) in live_nodes {
            let base = ((lv as u64) << 56) | (idx << 2);
            items.push((base, tr.p.clone()));
            items.push((base | 1, tr.q.clone()));
            items.push((base | 2, tr.t.clone()));
        }
        let stage = StageRecord::Series {
            term_count: self.term_count,
            leaf_len: self.leaf_len,
            dividing_mask: self.dividing_mask,
            completed_leaves: completed_level as u64,
        };
        self.mgr
            .write_stage_checkpoint(CheckpointStageKind::Series, stage, items)
    }
}

impl NewtonCheckpointSink for CheckpointManager<'_> {
    fn should_checkpoint(&self, work_sec_since_last: f64) -> bool {
        self.should_write(work_sec_since_last)
    }

    fn maybe_checkpoint_newton(
        &self,
        state: &NewtonResumeState,
        q_root: &BigNum,
        t_root: &BigNum,
        work_sec_since_last: f64,
    ) -> Result<bool, MulError> {
        if !self.should_write(work_sec_since_last) {
            return Ok(false);
        }
        let meta_bn = BigNum::from_limbs(
            vec![state.step_limbs as u64, state.acc_bits],
            false,
            0,
            self.kernels,
        );
        let items = vec![
            (0u64, q_root.clone()),
            (1u64, t_root.clone()),
            (2u64, state.z.clone()),
            (3u64, state.u.clone()),
            (4u64, meta_bn),
        ];
        let stage = StageRecord::Finish {
            precision_limbs: state.step_limbs as u64,
            step_index: 0,
        };
        self.write_stage_checkpoint(CheckpointStageKind::Finish, stage, items)
    }
}

struct RadixCkptAdapter<'a, 'b> {
    mgr: &'a CheckpointManager<'b>,
    conv_leaf_digits: u64,
}

impl RadixCheckpointSink for RadixCkptAdapter<'_, '_> {
    fn maybe_checkpoint_radix(
        &self,
        pending: &[PendingFraction],
        completed_w0: &[(u64, u64, u64)],
        work_sec_since_last: f64,
    ) -> Result<bool, MulError> {
        if !self.mgr.should_write(work_sec_since_last) {
            return Ok(false);
        }
        let mut items = Vec::with_capacity(pending.len() * 2 + 1);
        for (i, pf) in pending.iter().enumerate() {
            let idx = i as u64;
            let hdr_limbs = vec![
                pf.start_leaf,
                pf.leaf_count,
                pf.frac_k_limbs as u64,
                pf.required_w0.unwrap_or(u64::MAX),
            ];
            items.push((
                0x1000_0000 | idx,
                BigNum::from_limbs(hdr_limbs, false, 0, self.mgr.kernels),
            ));
            items.push((
                0x2000_0000 | idx,
                BigNum::from_limbs(pf.frac_limbs.clone(), false, 0, self.mgr.kernels),
            ));
        }
        if !completed_w0.is_empty() {
            let mut flat_w0 = Vec::with_capacity(completed_w0.len() * 3);
            for &(s, c, w) in completed_w0 {
                flat_w0.push(s);
                flat_w0.push(c);
                flat_w0.push(w);
            }
            items.push((
                0x3000_0000,
                BigNum::from_limbs(flat_w0, false, 0, self.mgr.kernels),
            ));
        }
        let stage = StageRecord::Convert {
            leaf_digits: self.conv_leaf_digits,
            completed_leaves: completed_w0.len() as u64,
        };
        self.mgr
            .write_stage_checkpoint(CheckpointStageKind::Convert, stage, items)
    }
}

const MEMBER_CALL_TIMEOUT_SEC: f64 = 5.0;
const DEVICE_FAULT_LIMIT: u64 = 6;

/// Offset-translating `ExtentMember` wrapper that reserves `[0 .. base_offset)` for SSD checkpoint
/// banks while dealing every `allocate` and `free` call directly to `inner`, releasing all remaining
/// spans on `Drop` across Rung-3 rollbacks or run teardown.
struct BumpOffsetMember {
    inner: Arc<dyn ExtentMember>,
    base_offset: u64,
    usable_bytes: u64,
    reserved_head: bool,
    live_spans: Mutex<Vec<(u64, u64)>>,
    telemetry: Arc<Telemetry>,
}

impl BumpOffsetMember {
    fn new(inner: Arc<dyn ExtentMember>, base_offset: u64, telemetry: Arc<Telemetry>) -> Self {
        let cap = inner.capacity();
        let usable = cap.saturating_sub(base_offset);
        let reserved_head = if base_offset > 0 && base_offset <= cap {
            inner.allocate(base_offset).is_ok()
        } else {
            false
        };
        Self {
            inner,
            base_offset,
            usable_bytes: usable,
            reserved_head,
            live_spans: Mutex::new(Vec::new()),
            telemetry,
        }
    }
}

impl Drop for BumpOffsetMember {
    fn drop(&mut self) {
        let spans = std::mem::take(&mut *self.live_spans.lock().unwrap());
        for (raw_off, bytes) in spans {
            self.inner.free(raw_off, bytes);
        }
        if self.reserved_head {
            self.inner.free(0, self.base_offset);
        }
    }
}

impl ExtentMember for BumpOffsetMember {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn kind(&self) -> MemberKind {
        self.inner.kind()
    }
    fn alignment(&self) -> u64 {
        self.inner.alignment()
    }
    fn capacity(&self) -> u64 {
        self.usable_bytes
    }
    fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
        if bytes > self.usable_bytes {
            return Err(StoreError::OutOfSpace);
        }
        let align = self.inner.alignment().max(1);
        let al = bytes.checked_add(align - 1).ok_or(StoreError::OutOfSpace)? / align * align;
        match self.inner.allocate(al) {
            Ok(raw_off) => {
                if raw_off < self.base_offset
                    || raw_off.saturating_add(al) > self.base_offset.saturating_add(self.usable_bytes)
                {
                    self.inner.free(raw_off, al);
                    return Err(StoreError::OutOfSpace);
                }
                self.live_spans.lock().unwrap().push((raw_off, al));
                Ok(raw_off - self.base_offset)
            }
            Err(e) => {
                if e == StoreError::DeviceLost {
                    self.telemetry.mark_device_stopped(self.id());
                    self.telemetry.record_failure(0, 0, Some(self.id()));
                }
                Err(e)
            }
        }
    }
    fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
        let shifted: Vec<ByteRange> = ranges
            .iter()
            .map(|r| ByteRange {
                device_offset: self.base_offset + r.device_offset,
                host_offset: r.host_offset,
                len: r.len,
            })
            .collect();
        let t0 = Instant::now();
        let res = self.inner.read(&shifted, host_buf);
        let timed_out = t0.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC;
        match res {
            Ok(()) if timed_out => {
                self.telemetry.mark_device_stopped(self.id());
                self.telemetry.record_failure(0, 0, Some(self.id()));
                Err(StoreError::DeviceLost)
            }
            Ok(()) => Ok(()),
            Err(e) => {
                if e == StoreError::DeviceLost || timed_out {
                    self.telemetry.mark_device_stopped(self.id());
                }
                self.telemetry.record_failure(0, 0, Some(self.id()));
                Err(e)
            }
        }
    }
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
        let shifted: Vec<ByteRange> = ranges
            .iter()
            .map(|r| ByteRange {
                device_offset: self.base_offset + r.device_offset,
                host_offset: r.host_offset,
                len: r.len,
            })
            .collect();
        let t0 = Instant::now();
        let res = self.inner.write(&shifted, host_buf);
        let timed_out = t0.elapsed().as_secs_f64() > MEMBER_CALL_TIMEOUT_SEC;
        match res {
            Ok(()) if timed_out => {
                self.telemetry.mark_device_stopped(self.id());
                self.telemetry.record_failure(0, 0, Some(self.id()));
                Err(StoreError::DeviceLost)
            }
            Ok(()) => Ok(()),
            Err(e) => {
                // Section 11: a write failure, DeviceLost, or timeout means the device stopped answering.
                self.telemetry.mark_device_stopped(self.id());
                self.telemetry.record_failure(0, 0, Some(self.id()));
                Err(e)
            }
        }
    }
    fn free(&self, device_offset: u64, bytes: u64) {
        let align = self.inner.alignment().max(1);
        let Some(al) = bytes.checked_add(align - 1).map(|v| v / align * align) else {
            return;
        };
        let raw_off = self.base_offset + device_offset;
        {
            let mut g = self.live_spans.lock().unwrap();
            if let Some(pos) = g.iter().position(|&(o, b)| o == raw_off && b == al) {
                g.swap_remove(pos);
            }
        }
        self.inner.free(raw_off, al);
    }
}

/// Execute a full or resumed `suPiposition` calculation (Section 3, 10, 11).
/// Diagnostic: with `SUPI_BENCH_MUL=<a_mbit>,<b_mbit>,<log2 n>,<rows>,<cols>,<tile_cols>,<k>,
/// <coeff_bits>,<iterations>[,<workers>[,<packed 0|1>[,<streaming stores 0|1>]]]` set, one
/// transform product of that shape is timed in
/// place of the run, in the run's own process state (allocator, profile, scheduler), and the
/// sweep, kernel and idle traces of every iteration are printed (`SUPI_TRACE`, `SUPI_KPROF`).
/// It is how a change to the large products is measured in seconds instead of by a whole run.
/// Returns whether a benchmark ran.
fn bench_one_product(
    kernels: &crate::refk::Kernels,
    ntt_cache: &crate::ntt::NttTableCache,
    sched: &crate::par::Scheduler,
    telemetry: &crate::check::Telemetry,
    workers: usize,
) -> Result<bool, EngineError> {
    let Ok(v) = std::env::var("SUPI_BENCH_MUL") else {
        return Ok(false);
    };
    let f: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if f.len() < 9 {
        return Ok(false);
    }
    let (a_mbit, b_mbit) = (f[0], f[1]);
    let spec = ntt::NttSpec {
        n: 1usize << f[2],
        rows: f[3],
        cols: f[4],
        tile_cols: f[5],
        extra_sweeps: 0,
        k_primes: f[6],
        coeff_bits: f[7] as u32,
        packed: f[3] > 1 && f.get(10).copied().unwrap_or(1) != 0,
        use_nt: f.get(11).copied().unwrap_or(1) != 0,
        negacyclic: false,
        wrap_cyclic: false,
        out_of_core: false,
        par_workers: f.get(9).copied().unwrap_or(workers),
    };
    let make_limbs = |mbit: usize, seed: u64| -> Vec<u64> {
        let mut x = seed;
        (0..(mbit * 1_000_000 / 64))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect()
    };
    let a = make_limbs(a_mbit, 0x1234_5678_9ABC_DEF1);
    let b = make_limbs(b_mbit, 0x0FED_CBA9_8765_4321);
    let out_limbs = a.len() + b.len();
    eprintln!("[BENCH MUL] {spec:?} a={a_mbit} Mbit b={b_mbit} Mbit");
    for it in 0..f[8] {
        let t0 = Instant::now();
        let r = ntt::ntt_mul_slices(kernels, ntt_cache, sched, None, telemetry, &a, &b, spec, out_limbs);
        let sec = t0.elapsed().as_secs_f64();
        let (limbs, w0) =
            r.map_err(|_| EngineError::FatalStop("benchmark product failed".to_string()))?;
        eprintln!("[BENCH MUL] iteration {it}: {:.1} ms, check word {w0:016x}", sec * 1e3);
        crate::ntt::dump_and_reset_ntt_trace("bench");
        crate::refk::dump_and_reset_kernel_times("bench", sec, workers);
        crate::par::dump_and_reset_par_times("bench", sec, workers);
        crate::limb::put_limb_buf(limbs);
    }
    crate::arena::trace_active_arenas("bench");
    Ok(true)
}

/// Diagnostic: with `SUPI_BENCH_CONV=<rounds>` set, the base conversion of a made-up fraction
/// of the run's size is timed in place of the run, `rounds` times, in the run's own process
/// state (allocator, profile, scheduler, plan). Its digits go nowhere, so the time is the
/// conversion's own work without the output file's. In a trace run (`SUPI_TRACE`,
/// `SUPI_KPROF`) the conversion's trace lines and the sweep, kernel and idle times of each
/// round follow. It is how a change to the conversion is measured in the two seconds a
/// conversion of 10^9 digits takes instead of by a whole run. Returns whether it ran.
fn bench_conversion(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
) -> Result<bool, EngineError> {
    let Ok(v) = std::env::var("SUPI_BENCH_CONV") else {
        return Ok(false);
    };
    let rounds = v.trim().parse::<usize>().unwrap_or(1).max(1);
    struct NoSink;
    impl radix::DigitSink for NoSink {
        fn write_at(&self, _file_offset: u64, _bytes: &[u8]) -> Result<(), String> {
            Ok(())
        }
        fn flush(&self) -> Result<(), String> {
            Ok(())
        }
    }
    let workers = shares.active_workers.max(1);
    let num_leaves = plan.digits.div_ceil(radix::LEAF_DIGITS as u64) + 1;
    let k = radix::frac_limbs_for_digits(num_leaves * (radix::LEAF_DIGITS as u64));
    eprintln!("[BENCH CONV] {} digits, a fraction of {k} limbs, {rounds} round(s)", plan.digits);
    let mut times: Vec<f64> = Vec::with_capacity(rounds);
    for it in 0..rounds {
        let mut x = 0x1234_5678_9ABC_DEF1u64 ^ ((it as u64) << 32);
        let limbs: Vec<u64> = (0..k)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        let frac = BigNum::from_limbs_par(limbs, false, 0, ctx.kernels, ctx.sched, workers);
        let t0 = Instant::now();
        let res = radix::run_conversion_owned(
            ctx, plan, shares, cal, 3, frac, k, &NoSink, None, None, None,
        );
        let sec = t0.elapsed().as_secs_f64();
        let conv = res.map_err(|e| {
            EngineError::FatalStop(format!("benchmark conversion failed: {e:?}"))
        })?;
        eprintln!(
            "[BENCH CONV] round {it}: {:.1} ms, digits' check word {:016x}",
            sec * 1e3,
            conv.dec_hash_w0
        );
        times.push(sec * 1e3);
        crate::ntt::dump_and_reset_ntt_trace("bench");
        crate::refk::dump_and_reset_kernel_times("bench", sec, workers);
        crate::par::dump_and_reset_par_times("bench", sec, workers);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "[BENCH CONV] best {:.1} ms, median {:.1} ms",
        times[0],
        times[times.len() / 2]
    );
    crate::arena::trace_active_arenas("bench");
    Ok(true)
}

/// Diagnostic: with `SUPI_BENCH_PRICE=<form>,<kbit>,<workers>,<rounds>[,<kbit of b>[,<log2 n>]]`
/// set, every transform shape the planner weighs for one product (or only those of `log2 n`
/// points) is timed in place of the run and printed
/// beside the price the planner gives it. `form` 0 is the plain product of a number of `kbit`
/// kilobits by one as long (or of `kbit of b`); `form` 1 is the fused merge of two nodes whose T
/// has `kbit` kilobits (T and Q in one pass, with P and the odd part of Q in the proportions
/// they have in the series); `form` 2 is the pair of products `T1 * Q2` and `Q1 * Q2` of such
/// nodes that shares the transform of `Q2`. A product has
/// `workers` workers, and as many products run at once as all the workers allow, each on
/// operands of its own that are not in cache, which is how a level of the series meets them.
/// This is how the planner's prices are checked against the machine. In a trace run
/// (`SUPI_TRACE`, `SUPI_KPROF`) the sweep, kernel and idle times of each shape's timed rounds
/// follow its line. Returns whether a benchmark ran.
fn bench_price_check(
    ctx: &MulCtx<'_>,
    plan: &Plan,
    shares: &Shares,
    cal: &Calibration,
) -> Result<bool, EngineError> {
    use crate::mul::MulMethod;
    let Ok(v) = std::env::var("SUPI_BENCH_PRICE") else {
        return Ok(false);
    };
    let f: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if f.len() < 4 {
        return Ok(false);
    }
    let (form, kbit, pw, rounds) = (f[0], f[1].max(64), f[2].max(1), f[3].max(1));
    let conc = (shares.active_workers.max(1) / pw).max(1);
    let bits = (kbit as u64) * 1000;
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    let mut make = |nbits: u64, shift: u64| -> BigNum {
        let n = (nbits.div_ceil(64) as usize).max(1);
        let mut v: Vec<u64> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
            .collect();
        let top_bits = (nbits - ((n as u64) - 1) * 64) as u32;
        if top_bits < 64 {
            v[n - 1] &= (1u64 << top_bits) - 1;
        }
        v[n - 1] |= 1u64 << (top_bits - 1);
        v[0] |= 1;
        BigNum::from_limbs(v, false, shift, ctx.kernels)
    };
    // One set of operands a product: (a, b) for the plain product; (P1, Q1, T1, P2, Q2, T2) for
    // the merge, Q held as its odd part and a shift as the series holds it.
    let q_odd = bits * 39 / 50;
    let q_shift = bits - q_odd;
    let p_bits = bits * 3 / 5;
    let b_bits = f.get(4).map_or(bits, |&kb| (kb.max(64) as u64) * 1000);
    let set_bits = match form {
        0 => bits + b_bits,
        2 => bits + 2 * q_odd,
        _ => 2 * (bits + q_odd + p_bits),
    };
    let n_sets = (((256u64 << 23) / set_bits.max(1)) as usize).clamp(2 * conc, 64 * conc) / conc * conc;
    let sets: Vec<[BigNum; 6]> = (0..n_sets)
        .map(|_| {
            if form == 0 {
                [
                    make(bits, 0),
                    make(b_bits, 0),
                    BigNum::zero(),
                    BigNum::zero(),
                    BigNum::zero(),
                    BigNum::zero(),
                ]
            } else if form == 2 {
                [
                    make(bits, 0),
                    make(q_odd, q_shift),
                    make(q_odd, q_shift),
                    BigNum::zero(),
                    BigNum::zero(),
                    BigNum::zero(),
                ]
            } else {
                [
                    make(p_bits, 0),
                    make(q_odd, q_shift),
                    make(bits, 0),
                    make(p_bits, 0),
                    make(q_odd, q_shift),
                    make(bits, 0),
                ]
            }
        })
        .collect();

    let fused = form == 1;
    // the second operand the shape is sized for
    let other_bits = match form {
        0 => b_bits,
        2 => q_odd,
        _ => bits,
    };
    let own = plan.select_mul_method(bits, other_bits, fused, false, pw, shares, cal);
    let mut cands = crate::plan::list_ntt_specs_with_prices(
        bits,
        other_bits,
        fused,
        pw,
        shares,
        cal,
        plan.ram_transform_budget,
        plan.overrides.max_ntt_primes,
    );
    let cheapest = cands.iter().map(|c| c.1.total_sec).fold(f64::INFINITY, f64::min);
    cands.retain(|c| !c.0.out_of_core && c.1.total_sec <= 8.0 * cheapest);
    if let Some(&log_n) = f.get(5) {
        cands.retain(|c| c.0.n.ilog2() as usize == log_n);
    }
    let traced = std::env::var_os("SUPI_TRACE").is_some();
    let mut methods: Vec<(MulMethod, Option<crate::plan::ProductPrice>)> = cands
        .iter()
        .map(|(sp, pr)| (MulMethod::Ntt(*sp), Some(*pr)))
        .collect();
    if !matches!(own, MulMethod::Ntt(_)) {
        methods.insert(0, (own.clone(), None));
    }
    eprintln!(
        "[PRICE] form={form} kbit={kbit} workers={pw} at_once={conc} sets={n_sets} | the planner takes {}",
        match &own {
            MulMethod::Ntt(sp) => format!(
                "n=2^{} R={} k={} cb={} pk={}",
                sp.n.ilog2(),
                sp.rows,
                sp.k_primes,
                sp.coeff_bits,
                u8::from(sp.packed)
            ),
            other => format!("{other:?}"),
        }
    );
    let sets_addr = sets.as_ptr() as usize;
    let mut fastest = (f64::INFINITY, String::new());
    let mut own_ms = f64::NAN;
    for (method, price) in &methods {
        let declined = std::sync::atomic::AtomicBool::new(false);
        let (mut best, mut sum) = (f64::INFINITY, 0.0f64);
        // one round to build tables and warm up, then the timed rounds
        for round in 0..=rounds {
            let t0 = Instant::now();
            ctx.sched.par_for_max(conc, 1, conc, |i| {
                // every product of a round has a set of its own
                let set = unsafe {
                    &mut *(sets_addr as *mut [BigNum; 6]).add((round * conc + i) % n_sets)
                };
                if form == 0 {
                    let (lo, hi) = set.split_at_mut(1);
                    match ctx.mul_whole(0x7000_0000 + i as u64, method, &mut lo[0], &mut hi[0]) {
                        Ok(r) => drop(r),
                        Err(_) => declined.store(true, Ordering::Relaxed),
                    }
                } else if form == 2 {
                    match ctx.try_mul_pair_shared_b(
                        method,
                        pw,
                        plan.ram_transform_budget,
                        &set[0],
                        &set[1],
                        &set[2],
                    ) {
                        Some(r) => drop(r),
                        None => declined.store(true, Ordering::Relaxed),
                    }
                } else {
                    match ctx.try_ntt_fused_merge_triple(
                        method,
                        pw,
                        plan.ram_transform_budget,
                        &set[0],
                        &set[1],
                        &set[2],
                        &set[3],
                        &set[4],
                        &set[5],
                        true,
                        false,
                    ) {
                        Some(r) => drop(r),
                        None => declined.store(true, Ordering::Relaxed),
                    }
                }
            });
            let sec = t0.elapsed().as_secs_f64();
            if round > 0 {
                best = best.min(sec);
                sum += sec;
            } else if traced {
                // the warm-up round's counts are not the shape's
                crate::ntt::dump_and_reset_ntt_trace("warm-up");
                crate::refk::dump_and_reset_kernel_times("warm-up", sec, shares.active_workers);
                crate::par::dump_and_reset_par_times("warm-up", sec, shares.active_workers);
            }
        }
        let desc = match method {
            MulMethod::Ntt(sp) => format!(
                "n=2^{} R={} C={} tc={} k={} cb={} pk={}",
                sp.n.ilog2(),
                sp.rows,
                sp.cols,
                sp.tile_cols,
                sp.k_primes,
                sp.coeff_bits,
                u8::from(sp.packed)
            ),
            other => format!("{other:?}"),
        };
        if declined.load(Ordering::Relaxed) {
            eprintln!("[PRICE]   {desc} | declined");
            continue;
        }
        // a round is `conc` products at once, `pw` workers each
        let mean_ms = sum / (rounds as f64) * (pw as f64) * 1e3;
        let best_ms = best * (pw as f64) * 1e3;
        let model = match price {
            Some(pr) => format!(
                "model {:.3} (compute {:.3} traffic {:.3})",
                pr.total_sec * (pw as f64) * 1e3,
                pr.compute_sec * (pw as f64) * 1e3,
                pr.traffic_sec * (pw as f64) * 1e3
            ),
            None => "model -".to_string(),
        };
        eprintln!("[PRICE]   {desc} | {model} | measured {mean_ms:.3} best {best_ms:.3} thread-ms");
        if traced {
            crate::ntt::dump_and_reset_ntt_trace("shape");
            crate::refk::dump_and_reset_kernel_times("shape", sum, shares.active_workers);
            crate::par::dump_and_reset_par_times("shape", sum, shares.active_workers);
        }
        if *method == own {
            own_ms = mean_ms;
        }
        if mean_ms < fastest.0 {
            fastest = (mean_ms, desc);
        }
    }
    eprintln!(
        "[PRICE] the planner's shape {own_ms:.3}, the fastest {:.3} ({}): {:.1}% lost",
        fastest.0,
        fastest.1,
        (own_ms / fastest.0 - 1.0) * 100.0
    );
    Ok(true)
}

pub fn run(cfg: &EngineConfig<'_>) -> Result<RunReport, EngineError> {
    let t_start = Instant::now();

    let all_ssds: Vec<Arc<dyn ExtentMember>> = cfg
        .members
        .iter()
        .filter(|m| m.kind() == MemberKind::Ssd)
        .cloned()
        .collect();

    if cfg.discard_records {
        clear_all_ssd_records(&all_ssds);
    }

    let telemetry = Arc::new(Telemetry::default());
    let ladder = Ladder::new(8);

    // 1. Inspect existing checkpoint records on SSDs before writing anything (Section 11 "Starting").
    let initial_disc = if cfg.discard_records {
        None
    } else {
        inspect_ssd_records(&all_ssds, cfg.kernels)?
    };

    let mut excluded_devices: Vec<String> = Vec::new();
    let mut resumed_from_serial: Option<u64> = None;
    let mut next_serial: u64 = 1;

    if let Some(disc) = &initial_disc {
        let rec = &disc.canonical;
        if rec.digits != cfg.digits
            || rec.output_path != cfg.output_id
            || rec.format_text != (cfg.format == OutputFormat::Text)
        {
            return Err(EngineError::RecordMismatch {
                found_digits: rec.digits,
                found_output: rec.output_path.clone(),
                found_text: rec.format_text,
            });
        }
        excluded_devices = rec.excluded_devices.clone();
        resumed_from_serial = Some(disc.serial);
        next_serial = disc.serial + 1;
    }

    let expected_out_bytes = cfg.format.file_bytes(cfg.digits).ok_or_else(|| {
        EngineError::Plan(PlanError::InsufficientCapacity {
            needed_bytes: u64::MAX,
            available_bytes: 0,
            detail: "Output file size overflows 64 bits".to_string(),
        })
    })?;

    // Validate existing output file on resume BEFORE touching anything!
    let mut digit_sink_opt: Option<Box<dyn DigitSink>> = if initial_disc.is_some() {
        let s = cfg
            .output_factory
            .open_existing_exact(expected_out_bytes)
            .map_err(EngineError::OutputFileSizeMismatch)?;
        Some(s)
    } else {
        None
    };

    // Load initial resume payload now to confirm the checkpoint is readable whole (Section 11).
    let mut current_resume: Option<LoadedResume> = if let Some(disc) = &initial_disc {
        Some(load_checkpoint_state(disc, cfg.kernels, &telemetry)?)
    } else {
        None
    };

    let mut rollbacks = 0u64;
    let mut total_checkpoints_written = 0u64;
    let mut all_event_reports: Vec<String> = Vec::new();

    loop {
        let mut cur_profile = cfg.profile.clone();
        cur_profile
            .vram
            .retain(|d| !excluded_devices.iter().any(|ex| ex == &d.id));
        cur_profile
            .ssd
            .retain(|d| !excluded_devices.iter().any(|ex| ex == &d.id));

        let active_ssds: Vec<Arc<dyn ExtentMember>> = all_ssds
            .iter()
            .filter(|m| !excluded_devices.iter().any(|ex| ex == m.id()))
            .cloned()
            .collect();
        let active_vram: Vec<Arc<dyn ExtentMember>> = cfg
            .members
            .iter()
            .filter(|m| {
                m.kind() == MemberKind::Vram && !excluded_devices.iter().any(|ex| ex == m.id())
            })
            .cloned()
            .collect();

        let min_ssd_cap = active_ssds
            .iter()
            .map(|s| s.capacity())
            .min()
            .unwrap_or(0);
        let (bank_bytes, store_base) =
            ssd_checkpoint_layout(min_ssd_cap, cfg.digits, active_ssds.len());

        let probe_member_idle = |member: &dyn ExtentMember, fallback: f64| -> f64 {
            if cfg.supplied_calibration.is_some()
                || cfg!(debug_assertions)
                || member.capacity() < 256 * 1024 * 1024
            {
                return fallback;
            }
            let align = (member.alignment().max(4096) as usize).max(65_536);
            let raw_probe = if member.capacity() >= 2 * 1024 * 1024 * 1024 {
                256 * 1024 * 1024usize
            } else if member.capacity() >= 512 * 1024 * 1024 {
                64 * 1024 * 1024usize
            } else {
                16 * 1024 * 1024usize
            };
            let probe_len = (raw_probe / align) * align;
            let Ok(dev_off) = member.allocate(probe_len as u64) else {
                return fallback;
            };
            let mut slot_buf = crate::store::AlignedSlotBuf::new_tracked(
                probe_len,
                align,
                Arc::new((
                    std::sync::atomic::AtomicU64::new(0),
                    std::sync::atomic::AtomicU64::new(0),
                )),
            );
            let warm_range = [ByteRange {
                device_offset: dev_off,
                host_offset: 0,
                len: align,
            }];
            let range = [ByteRange {
                device_offset: dev_off,
                host_offset: 0,
                len: probe_len,
            }];
            unsafe {
                let sl = &mut slot_buf.as_import_slice_mut()[..probe_len];
                sl.fill(0xA5);
                let _ = member
                    .write(&warm_range, sl)
                    .and_then(|()| member.read(&warm_range, sl));
            }
            let t0 = Instant::now();
            let ok = unsafe {
                let sl = &mut slot_buf.as_import_slice_mut()[..probe_len];
                member.write(&range, sl).is_ok() && member.read(&range, sl).is_ok()
            };
            let sec = t0.elapsed().as_secs_f64().max(1e-9);
            drop(slot_buf);
            member.free(dev_off, probe_len as u64);
            if ok {
                let bps = ((2 * probe_len) as f64 / sec).max(1.0e7);
                if std::env::var_os("SUPI_TRACE").is_some() {
                    eprintln!("[TRACE MEMBER] {} idle={:.3}GB/s", member.id(), bps * 1e-9);
                }
                bps
            } else {
                fallback
            }
        };

        let profile_ram: u64 = cur_profile.ram.iter().map(|n| n.size).sum();
        let est_out_bits =
            ((cfg.digits.saturating_add(1024) as f64) * plan::LOG2_10).ceil() as u64 + 4096;
        let may_use_store = !cfg.members.is_empty()
            && (profile_ram <= 512 * 1024 * 1024
                || (est_out_bits / 8).saturating_mul(28)
                    > profile_ram.saturating_sub(profile_ram / 8));
        let may_probe_ssd = may_use_store
            || cfg.overrides.checkpoint_mult <= 0.0
            || cfg.digits >= 10_000_000;

        let mut store_members: Vec<(Arc<dyn ExtentMember>, f64)> = Vec::new();
        for v in &active_vram {
            let wrapped = Arc::new(BumpOffsetMember::new(
                Arc::clone(v),
                0,
                Arc::clone(&telemetry),
            ));
            if wrapped.capacity() >= 65_536 {
                let rate = cfg
                    .supplied_member_rates
                    .as_ref()
                    .and_then(|m| m.get(v.id()).copied())
                    .unwrap_or_else(|| {
                        if may_use_store {
                            probe_member_idle(wrapped.as_ref(), 24.0e9)
                        } else {
                            24.0e9
                        }
                    });
                store_members.push((wrapped, rate));
            }
        }
        let mut ssd_with_rates: Vec<(Arc<dyn ExtentMember>, f64)> =
            Vec::with_capacity(active_ssds.len());
        for s in &active_ssds {
            let carved = Arc::new(BumpOffsetMember::new(
                Arc::clone(s),
                store_base,
                Arc::clone(&telemetry),
            ));
            let rate = cfg
                .supplied_member_rates
                .as_ref()
                .and_then(|m| m.get(s.id()).copied())
                .unwrap_or_else(|| {
                    if may_probe_ssd && carved.capacity() >= 65_536 {
                        probe_member_idle(carved.as_ref(), 5.0e9)
                    } else {
                        5.0e9
                    }
                });
            ssd_with_rates.push((Arc::clone(s), rate));
            if carved.capacity() >= 65_536 {
                store_members.push((carved, rate));
            }
        }

        for d in &mut cur_profile.ssd {
            d.size = d.size.saturating_sub(store_base);
        }

        let mut shares = Shares::from_profile(&cur_profile);
        cfg.kernels.configure_from_shares(&shares);
        if let Some(c) = &cfg.supplied_calibration {
            rerank_shares(&mut shares, &c.core_ntt_bf_per_sec);
        }
        let mut sched = Scheduler::new(&shares, cfg.thread_placer);

        let store_piece_limbs = if shares.total_ram < 4 * 1024 * 1024 {
            1024usize
        } else {
            4096usize
        };
        let devices = form_stripes(store_members, store_piece_limbs);
        let mut store_opt = if devices.is_empty() {
            None
        } else {
            Some(Store::new(devices))
        };

        let t_cal0 = Instant::now();
        let cal = if let Some(c) = &cfg.supplied_calibration {
            c.clone()
        } else {
            let c = Calibration::measure_with_cache(
                cfg.kernels,
                &cur_profile,
                &shares,
                &sched,
                if may_use_store {
                    store_opt.as_mut()
                } else {
                    None
                },
                cfg.calibration_cache_path.as_deref(),
                cfg.build_id,
            );
            // If any store member stopped answering during the startup probe, exclude it now!
            let _ = telemetry.take_rollback_device_gobacks();
            let mut calib_excluded = false;
            {
                let dev_faults = telemetry.device_faults.lock().unwrap();
                let mut rem_ssd_cnt = all_ssds
                    .iter()
                    .filter(|s| !excluded_devices.iter().any(|ex| ex == s.id()))
                    .count();
                for m in &cfg.members {
                    if excluded_devices.iter().any(|ex| ex == m.id()) {
                        continue;
                    }
                    let stopped = telemetry.is_device_stopped(m.id());
                    let faults = dev_faults.get(m.id()).copied().unwrap_or(0);
                    if m.kind() == MemberKind::Ssd && rem_ssd_cnt <= 1 && stopped {
                        return Err(EngineError::FatalStop(format!(
                            "Last SSD '{}' stopped answering during calibration",
                            m.id()
                        )));
                    }
                    if stopped || faults >= DEVICE_FAULT_LIMIT {
                        let can_exclude = match m.kind() {
                            MemberKind::Vram => true,
                            MemberKind::Ssd => {
                                if rem_ssd_cnt > 1 {
                                    rem_ssd_cnt -= 1;
                                    true
                                } else {
                                    false
                                }
                            }
                        };
                        if can_exclude {
                            excluded_devices.push(m.id().to_string());
                            calib_excluded = true;
                        }
                    }
                }
            }
            if calib_excluded {
                continue;
            }
            if rerank_shares(&mut shares, &c.core_ntt_bf_per_sec) {
                drop(sched);
                sched = Scheduler::new(&shares, cfg.thread_placer);
            }
            c
        };
        // The crossovers of `mul_base` and `sqr_base` go to the kernels with every call from here on.
        cfg.kernels
            .mul_base_rows_max
            .store(cal.mul_base_rows_max, std::sync::atomic::Ordering::Relaxed);
        cfg.kernels
            .sqr_base_rows_max
            .store(cal.sqr_base_rows_max, std::sync::atomic::Ordering::Relaxed);
        let calib_sec = t_cal0.elapsed().as_secs_f64();

        let mut overrides = cfg.overrides.clone();
        if let Some(LoadedResume::Series {
            term_count,
            leaf_len,
            dividing_mask,
            ..
        }) = &current_resume
        {
            overrides.force_series_shape = Some((*term_count, *leaf_len));
            // Section 11: keep the Series tree shape & which levels divide, while re-planning
            // each dividing level's `DivForm` under the resumed `Shares` and `Calibration`!
            if overrides.force_dividing_levels.is_none() && overrides.force_dividing_mask.is_none()
            {
                overrides.force_dividing_mask = Some(*dividing_mask);
            }
        }
        if let Some(LoadedResume::Convert { leaf_digits, .. }) = &current_resume {
            overrides.force_conv_leaf_digits = Some(*leaf_digits);
        }

        let plan =
            Plan::build(cfg.digits, cfg.format, &shares, &cal, overrides).map_err(EngineError::Plan)?;

        if digit_sink_opt.is_none() {
            let s = cfg
                .output_factory
                .create_full_size(plan.output_file_bytes)
                .map_err(|e| EngineError::Plan(PlanError::OutputFileError(e)))?;
            digit_sink_opt = Some(s);
        }
        let digit_sink = digit_sink_opt.as_deref().unwrap();

        let aw = shares.active_workers.max(1);
        let worker_specs: Vec<crate::arena::ArenaWorkerSpec> = (0..aw)
            .map(|wi| {
                let w = shares.workers.get(wi);
                crate::arena::ArenaWorkerSpec {
                    worker_idx: wi,
                    numa_node: w.map_or(0, |ws| ws.numa_node as usize),
                    cache_share_bytes: w.map_or(plan.ram_scratch_per_worker, |ws| {
                        ws.l2_share.max(plan.ram_scratch_per_worker)
                    }),
                }
            })
            .collect();
        let (num_arena_budget, trans_arena_budget, usable_ram_cap) =
            if cfg.region_mapper.is_none() && !plan.out_of_core {
                let est_need = (plan.target_frac_bits / 8)
                    .saturating_mul(24)
                    .saturating_add(16 * 1024 * 1024);
                (
                    plan.ram_numbers_budget.min(est_need),
                    plan.ram_transform_budget.min(est_need),
                    shares.total_ram.min(est_need.saturating_mul(2)),
                )
            } else {
                (
                    plan.ram_numbers_budget,
                    plan.ram_transform_budget,
                    shares.total_ram,
                )
            };
        let arenas = Arc::new(ArenaSet::new_with_workers(
            &worker_specs,
            num_arena_budget,
            trans_arena_budget,
            usable_ram_cap,
            cfg.region_mapper,
        ));
        crate::arena::register_arena_set(&arenas);
        if !plan.out_of_core && !plan.roomy_run() {
            let soft_cap = ((shares.total_ram / 32).saturating_mul(31)) as usize;
            arenas.set_soft_cap(soft_cap.max(64 * 1024 * 1024));
        }

        let subtree_touch = if plan.digits >= 100_000 {
            256 * 1024
        } else {
            16 * 1024
        };
        let region_touch = if plan.digits >= 1_000_000 {
            2 * 1024 * 1024
        } else {
            64 * 1024
        };
        sched.bind_caller_as_worker0();
        sched.on_each_worker(|| {
            let wi = crate::par::current_worker_idx().min(aw - 1);
            crate::arena::bind_thread_arenas(Arc::clone(&arenas), wi);
            let scratch_touch =
                (worker_specs[wi].cache_share_bytes as usize).clamp(16 * 1024, 512 * 1024);
            arenas.first_touch_worker(wi, scratch_touch, subtree_touch, region_touch);
        });

        struct ArenaRunGuard<'s>(&'s Scheduler);
        impl Drop for ArenaRunGuard<'_> {
            fn drop(&mut self) {
                self.0.bind_caller_as_worker0();
                self.0.on_each_worker(|| {
                    limb::clear_tls_scratch();
                    splitting::clear_tls_scratch();
                    radix::clear_tls_scratch();
                    fft::clear_tls_fwd_pool();
                    ntt::clear_tls_bufs();
                    crate::arena::unbind_thread_arenas();
                });
            }
        }
        let arena_guard = ArenaRunGuard(&sched);

        let fft_cache = FftTableCache::default();
        let ntt_cache = NttTableCache::default();
        ntt_cache.set_l3_prime_cap_bytes(((shares.effective_l3_bytes(1, aw) * 7) / 8) as usize);
        ntt_cache.set_ooc_tile_budget(plan.ram_transform_budget);
        ntt_cache.set_max_tw_bytes(
            if plan.out_of_core && plan.ram_transform_budget < 256 * 1024 * 1024 {
                32 * 1024 * 1024
            } else {
                512 * 1024 * 1024
            },
        );
        // In a run that may need all the memory there is, no pair takes a work grid a prime: at
        // 5e9 digits in 48 GB even those of the small pairs, six levels below the top, moved the
        // top level's last block a gigabyte further out, to where the machine starts paging.
        ntt_cache.set_work_grid_room(if plan.roomy_run() {
            u64::MAX
        } else {
            (plan.ram_transform_budget / 8).saturating_mul(7)
        });
        if !plan.out_of_core
            && current_resume.is_none()
            && plan.digits >= 1_000_000
        {
            let last_div = plan.dividing_levels.last().map(|d| d.level).unwrap_or(0);
            let root_bpt = if last_div == 0 {
                58.0 + 3.0 * ((plan.term_count.max(2) as f64) / std::f64::consts::E).log2()
            } else {
                68.5 + 3.0 * (plan.tree_height.saturating_sub(last_div) as f64)
            };
            let est_s_bits =
                ((((plan.term_count / 2).max(1) as f64) * root_bpt * 1.05) as u64) + 2048;
            let est_f_bits =
                (plan.target_frac_bits.saturating_mul(103) / 100).saturating_add(2048);
            let mut max_r = 0usize;
            let mut max_c = 0usize;
            let mut max_k = 0usize;
            for (b, fused) in [(est_s_bits, true), (est_f_bits, false)] {
                if let mul::MulMethod::Ntt(s) =
                    plan.select_mul_method(b, b, fused, false, aw, &shares, &cal)
                {
                    if !s.out_of_core && s.rows > 1 {
                        max_r = max_r.max(s.rows);
                        max_c = max_c.max(s.cols);
                        max_k = max_k.max(s.k_primes);
                    }
                }
            }
            if max_r > 1 && (!plan.roomy_run() || plan.digits >= 10_000_000) {
                let max_plane_words = (ntt_cache.l3_prime_cap_bytes() / aw / 8).max(65_536);
                ntt::prewarm_tls_scratch(&sched, max_r, max_c, max_k.max(12), max_plane_words);
            }
            if cfg.region_mapper.is_some() && plan.roomy_run() && plan.digits >= 10_000_000 {
                let target_b = (plan.target_frac_bits / 8).max(1024 * 1024) as usize;
                let max_num = (num_arena_budget as usize).saturating_mul(3) / 4;
                let max_tr = (trans_arena_budget as usize).saturating_mul(3) / 4;
                let num_low = ((target_b * 5) / 8).min(max_num / 4);
                let num_high = target_b.saturating_mul(6).min(max_num.saturating_sub(num_low));
                let tr_low = ((target_b * 3) / 8).min(max_tr / 4);
                let tr_high = target_b.saturating_mul(11).min(max_tr.saturating_sub(tr_low));
                arenas.prewarm_working_set(&sched, num_low, num_high, tr_low, tr_high);
            }
        }
        let sieve_ops_counter = AtomicU64::new(0);

        let ckpt_mgr = CheckpointManager {
            kernels: cfg.kernels,
            telemetry: &telemetry,
            digits: cfg.digits,
            format: cfg.format,
            output_id: cfg.output_id.clone(),
            digit_sink,
            est_write_sec: plan.est_checkpoint_write_sec,
            checkpoint_mult: plan.checkpoint_mult,
            stop_condition: cfg.stop_condition,
            progress_reporter: cfg.progress_reporter,
            state: Mutex::new(CheckpointWriterState {
                ssd_pool: ssd_with_rates,
                excluded_devices: excluded_devices.clone(),
                event_reports: Vec::new(),
                next_serial,
                bank_bytes,
                checkpoints_written: total_checkpoints_written,
                truncated_once: false,
                stopped_stage: None,
            }),
        };

        let mut ctx = MulCtx::new(
            cfg.kernels,
            &fft_cache,
            &ntt_cache,
            &sched,
            store_opt.as_ref(),
            &ladder,
            &telemetry,
        );
        ctx.fft_err_bound = cfg.fft_err_bound;
        ctx.fault_hook = cfg.mul_fault_hook;

        if bench_one_product(cfg.kernels, &ntt_cache, &sched, &telemetry, shares.active_workers)? {
            return Err(EngineError::FatalStop(
                "benchmark of one product done (SUPI_BENCH_MUL)".to_string(),
            ));
        }
        if bench_price_check(&ctx, &plan, &shares, &cal)? {
            return Err(EngineError::FatalStop(
                "price check done (SUPI_BENCH_PRICE)".to_string(),
            ));
        }
        if bench_conversion(&ctx, &plan, &shares, &cal)? {
            return Err(EngineError::FatalStop(
                "benchmark of the conversion done (SUPI_BENCH_CONV)".to_string(),
            ));
        }

        let mut div_mask = 0u64;
        for d in &plan.dividing_levels {
            if d.level < 63 {
                div_mask |= 1u64 << d.level;
            }
        }
        let series_ckpt = SeriesCkptAdapter {
            mgr: &ckpt_mgr,
            term_count: plan.term_count,
            leaf_len: plan.leaf_len,
            dividing_mask: div_mask,
        };
        let radix_ckpt = RadixCkptAdapter {
            mgr: &ckpt_mgr,
            conv_leaf_digits: plan.conv_leaf_digits as u64,
        };

        let can_ckpt = plan.checkpoint_enabled && !active_ssds.is_empty();
        let s_ckpt_ref: Option<&dyn SeriesCheckpointSink> = if can_ckpt {
            Some(&series_ckpt)
        } else {
            None
        };
        let n_ckpt_ref: Option<&dyn NewtonCheckpointSink> = if can_ckpt {
            Some(&ckpt_mgr)
        } else {
            None
        };
        let r_ckpt_ref: Option<&dyn RadixCheckpointSink> = if can_ckpt {
            Some(&radix_ckpt)
        } else {
            None
        };

        let mut series_sec = 0.0f64;
        let mut finish_sec = 0.0f64;
        let mut convert_sec = 0.0f64;

        let attempt_res: Result<(u64, bool), MulError> = (|| match &current_resume {
            Some(LoadedResume::Convert { state, .. }) => {
                let tc = Instant::now();
                let conv = run_conversion(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    3,
                    &BigNum::zero(),
                    0,
                    digit_sink,
                    cfg.leaf_fault_hook,
                    Some(state),
                    r_ckpt_ref,
                )?;
                convert_sec = tc.elapsed().as_secs_f64();
                Ok((conv.dec_hash_w0, conv.digits_in_doubt))
            }
            Some(LoadedResume::Finish {
                q_root,
                t_root,
                state,
            }) => {
                let mut q = q_root.clone();
                let mut t = t_root.clone();
                let tf = Instant::now();
                let (int_part, pi_frac, out_frac_limbs) = run_finish(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    &mut q,
                    &mut t,
                    Some(state),
                    n_ckpt_ref,
                )?;
                finish_sec = tf.elapsed().as_secs_f64();
                if !plan.out_of_core && !plan.roomy_run() {
                    crate::arena::trim_active_arenas_above(((shares.total_ram / 32).saturating_mul(31)) as usize);
                }
                let tc = Instant::now();
                let conv = run_conversion_owned(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    int_part,
                    pi_frac,
                    out_frac_limbs,
                    digit_sink,
                    cfg.leaf_fault_hook,
                    None,
                    r_ckpt_ref,
                )?;
                convert_sec = tc.elapsed().as_secs_f64();
                Ok((conv.dec_hash_w0, conv.digits_in_doubt))
            }
            Some(LoadedResume::Series { state, .. }) => {
                let ts = Instant::now();
                let (mut q, mut t) = run_series(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    &sieve_ops_counter,
                    Some(state),
                    s_ckpt_ref,
                )?;
                series_sec = ts.elapsed().as_secs_f64();
                if !plan.out_of_core && !plan.roomy_run() {
                    crate::arena::trim_active_arenas_above(((shares.total_ram / 32).saturating_mul(31)) as usize);
                }
                let tf = Instant::now();
                let (int_part, pi_frac, out_frac_limbs) = run_finish(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    &mut q,
                    &mut t,
                    None,
                    n_ckpt_ref,
                )?;
                finish_sec = tf.elapsed().as_secs_f64();
                if !plan.out_of_core && !plan.roomy_run() {
                    crate::arena::trim_active_arenas_above(((shares.total_ram / 32).saturating_mul(31)) as usize);
                }
                let tc = Instant::now();
                let conv = run_conversion_owned(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    int_part,
                    pi_frac,
                    out_frac_limbs,
                    digit_sink,
                    cfg.leaf_fault_hook,
                    None,
                    r_ckpt_ref,
                )?;
                convert_sec = tc.elapsed().as_secs_f64();
                Ok((conv.dec_hash_w0, conv.digits_in_doubt))
            }
            None => {
                let do_trace = std::env::var_os("SUPI_TRACE").is_some();
                if do_trace {
                    eprintln!(
                        "[TRACE] pre-series wall={:.3}s calib={calib_sec:.3}s div_levels={:?}",
                        t_start.elapsed().as_secs_f64(),
                        plan.dividing_levels
                    );
                }
                let ts = Instant::now();
                let (mut q, mut t) = run_series(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    &sieve_ops_counter,
                    None,
                    s_ckpt_ref,
                )?;
                series_sec = ts.elapsed().as_secs_f64();
                if do_trace {
                    crate::ntt::dump_and_reset_ntt_trace("series");
                    crate::refk::dump_and_reset_kernel_times("series", series_sec, shares.active_workers);
                    crate::par::dump_and_reset_par_times("series", series_sec, shares.active_workers);
                    crate::arena::trace_active_arenas("series");
                }
                if !plan.out_of_core && !plan.roomy_run() {
                    crate::arena::trim_active_arenas_above(((shares.total_ram / 32).saturating_mul(31)) as usize);
                }
                let tf = Instant::now();
                let (int_part, pi_frac, out_frac_limbs) = run_finish(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    &mut q,
                    &mut t,
                    None,
                    n_ckpt_ref,
                )?;
                finish_sec = tf.elapsed().as_secs_f64();
                if do_trace {
                    crate::ntt::dump_and_reset_ntt_trace("finish");
                    crate::refk::dump_and_reset_kernel_times("finish", finish_sec, shares.active_workers);
                    crate::par::dump_and_reset_par_times("finish", finish_sec, shares.active_workers);
                    crate::arena::trace_active_arenas("finish");
                }
                if !plan.out_of_core && !plan.roomy_run() {
                    crate::arena::trim_active_arenas_above(((shares.total_ram / 32).saturating_mul(31)) as usize);
                }
                let tc = Instant::now();
                let conv = run_conversion_owned(
                    &ctx,
                    &plan,
                    &shares,
                    &cal,
                    int_part,
                    pi_frac,
                    out_frac_limbs,
                    digit_sink,
                    cfg.leaf_fault_hook,
                    None,
                    r_ckpt_ref,
                )?;
                convert_sec = tc.elapsed().as_secs_f64();
                if do_trace {
                    crate::ntt::dump_and_reset_ntt_trace("convert");
                    crate::refk::dump_and_reset_kernel_times("convert", convert_sec, shares.active_workers);
                    crate::par::dump_and_reset_par_times("convert", convert_sec, shares.active_workers);
                    crate::arena::trace_active_arenas("convert");
                }
                Ok((conv.dec_hash_w0, conv.digits_in_doubt))
            }
        })();

        // Release worker TLS scratch and unbind `arenas` on all workers before snapshotting final
        // arena live/peak counters or reloading any rollback resume state.
        drop(ctx);
        drop(ntt_cache);
        drop(fft_cache);
        drop(arena_guard);

        let (mgr_excl, mgr_next_ser, mgr_written, mgr_stopped, mgr_events) = {
            let mut g = ckpt_mgr.state.lock().unwrap();
            (
                g.excluded_devices.clone(),
                g.next_serial,
                g.checkpoints_written,
                g.stopped_stage,
                std::mem::take(&mut g.event_reports),
            )
        };
        excluded_devices = mgr_excl;
        next_serial = mgr_next_ser;
        total_checkpoints_written = mgr_written;
        all_event_reports.extend(mgr_events);

        match attempt_res {
            Ok((dec_hash_w0, digits_in_doubt)) => {
                let t_fl = Instant::now();
                digit_sink.flush().map_err(EngineError::FatalStop)?;
                convert_sec += t_fl.elapsed().as_secs_f64();
                clear_all_ssd_records(&all_ssds);

                let (
                    store_bytes_moved,
                    store_loader_batches,
                    store_tile_peak_bytes,
                    store_parked_peak_bytes,
                    store_parked_live_bytes,
                ) = store_opt.as_ref().map_or((0, 0, 0, 0, 0), |st| {
                    (
                        st.total_bytes_moved(),
                        st.total_loader_batches(),
                        st.tile_peak_bytes(),
                        st.parked_peak_bytes(),
                        st.parked_live_bytes(),
                    )
                });
                let final_tel = telemetry.summary_line();
                let end_report = format!(
                    "End digits={} rollbacks={rollbacks} excluded={excluded_devices:?} | {final_tel}",
                    cfg.digits
                );
                all_event_reports.push(end_report);
                let arena_report = arenas.combined_report();
                let arena_usage = arenas.usage_stats();

                return Ok(RunReport {
                    dec_hash_w0,
                    digits_in_doubt,
                    sieve_ops: sieve_ops_counter.load(Ordering::Relaxed),
                    work_stats: cfg.kernels.stats.snapshot(),
                    store_bytes_moved,
                    store_loader_batches,
                    store_tile_peak_bytes,
                    store_parked_peak_bytes,
                    store_parked_live_bytes,
                    checkpoints_written: total_checkpoints_written,
                    resumed_from_serial,
                    rollbacks,
                    excluded_devices,
                    arena_report,
                    arena_usage,
                    calibration_reused: cal.reused_static,
                    telemetry_summary: final_tel,
                    event_reports: all_event_reports,
                    calib_sec,
                    series_sec,
                    finish_sec,
                    convert_sec,
                    wall_sec: t_start.elapsed().as_secs_f64(),
                    plan,
                });
            }
            Err(MulError::FatalStop(msg)) if msg == "TEST_STOP" => {
                let (ser, st) = mgr_stopped.unwrap_or((0, CheckpointStageKind::Series));
                return Err(EngineError::StoppedForTest {
                    serial: ser,
                    stage: st,
                });
            }
            Err(MulError::NeedRollback(point_id)) => {
                let _ = telemetry.take_rollback_device_gobacks();
                let mut newly_excluded = false;
                {
                    let dev_faults = telemetry.device_faults.lock().unwrap();
                    let mut rem_ssd_cnt = all_ssds
                        .iter()
                        .filter(|s| !excluded_devices.iter().any(|ex| ex == s.id()))
                        .count();
                    for m in &cfg.members {
                        if excluded_devices.iter().any(|ex| ex == m.id()) {
                            continue;
                        }
                        let stopped = telemetry.is_device_stopped(m.id());
                        let cnt = dev_faults.get(m.id()).copied().unwrap_or(0);
                        let dev_gb = telemetry.device_goback_count(m.id());

                        if m.kind() == MemberKind::Ssd && rem_ssd_cnt <= 1 && stopped {
                            return Err(EngineError::FatalStop(format!(
                                "Last SSD '{}' stopped answering",
                                m.id()
                            )));
                        }

                        // Section 11: exclude a non-last SSD or any GPU when it stops answering,
                        // its failures pass a set count, or its piece sends the run back a 2nd time.
                        if stopped || cnt >= DEVICE_FAULT_LIMIT || dev_gb >= 2 {
                            let can_exclude = match m.kind() {
                                MemberKind::Vram => true,
                                MemberKind::Ssd => {
                                    if rem_ssd_cnt > 1 {
                                        rem_ssd_cnt -= 1;
                                        true
                                    } else {
                                        false
                                    }
                                }
                            };
                            if can_exclude {
                                excluded_devices.push(m.id().to_string());
                                newly_excluded = true;
                            }
                        }
                    }
                }

                ladder.reset_below_goback();

                if !newly_excluded {
                    rollbacks += 1;
                    let rung = ladder.record_goback(point_id);
                    if rung == Rung::Stop || rollbacks > ladder.max_gobacks {
                        return Err(EngineError::FatalStop(format!(
                            "Second rollback at point {point_id:#x} or exceeded rollback limit (rollbacks={rollbacks})"
                        )));
                    }
                }

                let ret_report = format!(
                    "Return point={point_id:#x} rollbacks={rollbacks} excluded={excluded_devices:?} | {}",
                    telemetry.summary_line()
                );
                all_event_reports.push(ret_report.clone());
                if let Some(cb) = cfg.progress_reporter {
                    cb(&ret_report);
                }

                let rem_ssds: Vec<Arc<dyn ExtentMember>> = all_ssds
                    .iter()
                    .filter(|s| !excluded_devices.iter().any(|ex| ex == s.id()))
                    .cloned()
                    .collect();
                if let Some(disc) = inspect_ssd_records(&rem_ssds, cfg.kernels)? {
                    current_resume = Some(load_checkpoint_state(&disc, cfg.kernels, &telemetry)?);
                    resumed_from_serial = Some(disc.serial);
                    next_serial = disc.serial + 1;
                } else {
                    current_resume = None;
                }
            }
            Err(MulError::FatalStop(msg)) => {
                return Err(EngineError::FatalStop(msg));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_never_names_arch_os_or_inline_asm() {
        let forbidden: &[&str] = &[
            concat!("x86", "_64"),
            concat!("aarch", "64"),
            concat!("arm", "64"),
            concat!("avx", "512"),
            concat!("ne", "on"),
            concat!("sv", "e2"),
            concat!("win", "dows"),
            concat!("lin", "ux"),
            concat!("mac", "os"),
            concat!("win", "32"),
            concat!("global", "_asm"),
            concat!("as", "m!"),
        ];

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut checked = 0usize;
        for entry in std::fs::read_dir(dir).expect("read engine dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            checked += 1;
            let src = std::fs::read_to_string(&path).expect("read rs source");
            let lower = src.to_ascii_lowercase();
            for &bad in forbidden {
                assert!(
                    !lower.contains(bad),
                    "Architectural/OS/ASM boundary violation: {:?} contains forbidden token '{}'",
                    path.file_name().unwrap(),
                    bad
                );
            }
        }
        assert_eq!(
            checked, 15,
            "engine must have exactly its 15 single-file modules"
        );
    }
}
