//! Contract:
//! - Inputs: `ExtentMember` implementations (SSD files/devices, discrete GPU VRAM, or synthetic
//!   fault-injecting `TestMember`s), measured member transfer rates, piece size in limbs, and
//!   `Kernels`, `Scheduler`, + `Telemetry` references.
//! - Outputs: Ranked striped `ExtentDevice`s formed by the Section 10 rule (splitting larger
//!   members' remainders back into the pool and dealing all five member calls directly to stripe
//!   members), extents dealt in equal pieces with all 3 check words `(w0, w1, w2)` verified and
//!   Rung-1 repaired + written back on access, 64 KiB-aligned `AlignedSlotBuf` buffers for direct
//!   host-memory DMA import, and a single async `Loader` thread for overlapped transfers.
//! - Data format: Each stored piece on a member is `piece_limbs * 8` data bytes followed by 24
//!   bytes of little-endian `[w0, w1, w2]`, rounded up to member alignment.
//! - Work & bytes moved: One `ExtentMember::read`/`write` batch call per active stripe member per
//!   batch (coalescing contiguous slots up to 64 MiB), worker-executed `check_n` piece check words
//!   (inline in row sweeps or counted as column-sweep traffic), plus Rung-1 write-back when a
//!   single-limb piece error is repaired; exact byte counts tracked at the seam.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::check::{CorrectionOutcome, Telemetry, compute_limbs_all3, try_correct_single_limb};
use crate::par::Scheduler;
use crate::refk::Kernels;

pub const HOST_IMPORT_ALIGN: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreError {
    OutOfSpace,
    IoFailed,
    DeviceLost,
    IntegrityBeyondRepair,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Vram,
    Ssd,
}

#[derive(Clone, Copy, Debug)]
pub struct ByteRange {
    pub device_offset: u64,
    pub host_offset: usize,
    pub len: usize,
}

/// Coalescing best-fit span allocator shared by every `ExtentMember` implementation so `free`
/// genuinely reclaims and coalesces device space for subsequent allocations.
#[derive(Default, Debug)]
pub struct MemberSpanAlloc {
    next_alloc: u64,
    free_spans: Vec<(u64, u64)>,
    live_bytes: u64,
}

impl MemberSpanAlloc {
    pub const fn new() -> Self {
        Self {
            next_alloc: 0,
            free_spans: Vec::new(),
            live_bytes: 0,
        }
    }

    #[inline]
    pub fn next_alloc(&self) -> u64 {
        self.next_alloc
    }

    #[inline]
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    pub fn allocate(&mut self, bytes: u64, align: u64, cap: u64) -> Result<u64, StoreError> {
        let a = align.max(1);
        let al = bytes
            .checked_add(a - 1)
            .ok_or(StoreError::OutOfSpace)?
            / a
            * a;
        if al == 0 {
            return Ok(0);
        }
        let mut best: Option<(usize, u64)> = None;
        for (i, &(_, sz)) in self.free_spans.iter().enumerate() {
            if sz >= al && best.is_none_or(|(_, bsz)| sz < bsz) {
                best = Some((i, sz));
            }
        }
        if let Some((pos, sz)) = best {
            let off = self.free_spans[pos].0;
            if sz > al {
                self.free_spans[pos] = (off + al, sz - al);
            } else {
                self.free_spans.remove(pos);
            }
            self.live_bytes = self.live_bytes.saturating_add(al);
            return Ok(off);
        }
        let end = self
            .next_alloc
            .checked_add(al)
            .ok_or(StoreError::OutOfSpace)?;
        if end > cap {
            return Err(StoreError::OutOfSpace);
        }
        let off = self.next_alloc;
        self.next_alloc = end;
        self.live_bytes = self.live_bytes.saturating_add(al);
        Ok(off)
    }

    pub fn free(&mut self, offset: u64, bytes: u64, align: u64) {
        let a = align.max(1);
        let Some(al) = bytes.checked_add(a - 1).map(|v| v / a * a) else {
            return;
        };
        if al == 0 || offset.saturating_add(al) > self.next_alloc {
            return;
        }
        let idx = self.free_spans.partition_point(|&(o, _)| o < offset);
        if idx > 0 {
            let (po, pl) = self.free_spans[idx - 1];
            if po + pl > offset {
                return;
            }
        }
        if idx < self.free_spans.len() && offset + al > self.free_spans[idx].0 {
            return;
        }
        self.live_bytes = self.live_bytes.saturating_sub(al);
        self.free_spans.insert(idx, (offset, al));
        if idx + 1 < self.free_spans.len()
            && self.free_spans[idx].0 + self.free_spans[idx].1 == self.free_spans[idx + 1].0
        {
            self.free_spans[idx].1 += self.free_spans[idx + 1].1;
            self.free_spans.remove(idx + 1);
        }
        if idx > 0
            && self.free_spans[idx - 1].0 + self.free_spans[idx - 1].1 == self.free_spans[idx].0
        {
            self.free_spans[idx - 1].1 += self.free_spans[idx].1;
            self.free_spans.remove(idx);
        }
        while let Some(&(last_off, last_sz)) = self.free_spans.last() {
            if last_off + last_sz == self.next_alloc {
                self.next_alloc = last_off;
                self.free_spans.pop();
            } else {
                break;
            }
        }
    }

    pub fn is_range_completely_free(&self, start: u64, len: u64) -> bool {
        if len == 0 || start >= self.next_alloc {
            return true;
        }
        let end = start.saturating_add(len).min(self.next_alloc);
        for &(fo, fl) in &self.free_spans {
            if fo <= start && fo + fl >= end {
                return true;
            }
        }
        false
    }
}

/// The five-call interface shared by every member (VRAM or SSD) and every extent device (Section 10).
pub trait ExtentMember: Send + Sync {
    fn id(&self) -> &str;
    fn kind(&self) -> MemberKind;
    fn alignment(&self) -> u64;

    /// Call 1: usable byte capacity of this member.
    fn capacity(&self) -> u64;
    /// Call 2: allocate a contiguous span of `bytes` on this member, returning `device_offset`.
    fn allocate(&self, bytes: u64) -> Result<u64, StoreError>;
    /// Call 3: read a batch of `ranges` into `host_buf` in one submission.
    fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError>;
    /// Call 4: write a batch of `ranges` from `host_buf` in one submission.
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError>;
    /// Call 5: free a previously allocated span.
    fn free(&self, device_offset: u64, bytes: u64);
}

struct MemberSlice {
    member: Arc<dyn ExtentMember>,
    quota_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtentHandle {
    pub device_idx: usize,
    pub extent_id: u64,
    pub round_offset_bytes: u64,
    pub num_pieces: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ExtentRange {
    pub piece_start: u64,
    pub piece_count: u64,
    pub host_limb_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SlotPieceEntry {
    pub piece_id: u64,
    pub buf_byte_off: usize,
    pub host_limb_offset: usize,
}

#[derive(Clone, Debug)]
pub struct MemberTransferPlan {
    pub m_idx: usize,
    pub buf_base: usize,
    pub buf_cap: usize,
    pub ranges: Vec<ByteRange>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct SlotBatchPlan {
    pub extent: ExtentHandle,
    pub piece_limbs: usize,
    pub piece_slot_bytes: usize,
    pub required_buf_bytes: usize,
    pub pieces: Vec<SlotPieceEntry>,
    pub member_transfers: Vec<MemberTransferPlan>,
}

pub struct ExtentDevice {
    pub name: String,
    slices: Vec<MemberSlice>,
    pub piece_limbs: usize,
    pub piece_slot_bytes: u64,
    pub idle_rate_bps: f64,
    pub beside_sweep_rate_bps: f64,
    pub sweep_step_beside_sec: f64,
    pub min_batch_range_bytes: u64,
    next_extent_id: AtomicU64,
    extents: Mutex<(u64, Vec<(u64, Vec<u64>)>)>, // (used_member_bytes, [(extent_id, per_member_bases)])
}

impl ExtentDevice {
    pub fn new(
        name: String,
        slices: Vec<(Arc<dyn ExtentMember>, u64)>,
        piece_limbs: usize,
        idle_rate_bps: f64,
    ) -> Self {
        debug_assert!(!slices.is_empty() && piece_limbs > 0);
        let max_align = slices
            .iter()
            .map(|(m, _)| m.alignment().max(4096))
            .max()
            .unwrap_or(4096);
        let raw_slot = (piece_limbs as u64) * 8 + 24; // limbs + (w0, w1, w2)
        let piece_slot_bytes = raw_slot.div_ceil(max_align) * max_align;

        let mapped = slices
            .into_iter()
            .map(|(member, slice_bytes)| MemberSlice {
                member,
                quota_bytes: (slice_bytes / piece_slot_bytes) * piece_slot_bytes,
            })
            .collect();

        Self {
            name,
            slices: mapped,
            piece_limbs,
            piece_slot_bytes,
            idle_rate_bps,
            beside_sweep_rate_bps: idle_rate_bps * 0.85,
            sweep_step_beside_sec: 0.0,
            min_batch_range_bytes: piece_slot_bytes,
            next_extent_id: AtomicU64::new(1),
            extents: Mutex::new((0, Vec::new())),
        }
    }

    #[inline]
    pub fn stripe_width(&self) -> usize {
        self.slices.len()
    }

    #[inline]
    pub fn required_host_align(&self) -> usize {
        if self.slices.iter().any(|s| s.member.kind() == MemberKind::Vram) {
            HOST_IMPORT_ALIGN
        } else {
            self.slices
                .iter()
                .map(|s| (s.member.alignment() as usize).max(4096))
                .max()
                .unwrap_or(4096)
        }
    }

    #[inline]
    pub fn upper_slot_buf_bytes(&self, num_pieces: usize) -> usize {
        let align = self.required_host_align();
        let slot = self.piece_slot_bytes as usize;
        if num_pieces == 0 {
            return align;
        }
        let active_w = self.slices.len().max(1).min(num_pieces);
        let per_m = num_pieces.div_ceil(active_w);
        active_w * ((per_m * slot).div_ceil(align) * align)
    }

    pub fn member_ids(&self) -> Vec<String> {
        self.slices.iter().map(|s| s.member.id().to_string()).collect()
    }

    /// Usable data payload capacity in bytes (excluding check-word/alignment slot overhead).
    pub fn capacity(&self) -> u64 {
        let rounds = self.slices[0].quota_bytes / self.piece_slot_bytes;
        rounds * (self.slices.len() as u64) * (self.piece_limbs as u64) * 8
    }

    /// Allocate an extent by dealing `allocate(member_span)` to every member of the stripe.
    pub fn allocate(&self, device_idx: usize, payload_bytes: u64) -> Result<ExtentHandle, StoreError> {
        let piece_data_bytes = (self.piece_limbs as u64) * 8;
        let num_pieces = payload_bytes.div_ceil(piece_data_bytes).max(1);
        let width = self.slices.len() as u64;
        let rounds_needed = num_pieces.div_ceil(width);
        let member_span = rounds_needed * self.piece_slot_bytes;

        let mut g = self.extents.lock().unwrap();
        if g.0.saturating_add(member_span) > self.slices[0].quota_bytes {
            return Err(StoreError::OutOfSpace);
        }

        let mut bases: Vec<u64> = Vec::with_capacity(self.slices.len());
        for sl in &self.slices {
            match sl.member.allocate(member_span) {
                Ok(off) => bases.push(off),
                Err(e) => {
                    for (done_sl, &done_off) in self.slices.iter().zip(bases.iter()) {
                        done_sl.member.free(done_off, member_span);
                    }
                    return Err(e);
                }
            }
        }

        let extent_id = self.next_extent_id.fetch_add(1, Ordering::Relaxed);
        let round_offset_bytes = bases[0];
        g.0 += member_span;
        let pos = g.1.partition_point(|&(id, _)| id < extent_id);
        g.1.insert(pos, (extent_id, bases));

        Ok(ExtentHandle {
            device_idx,
            extent_id,
            round_offset_bytes,
            num_pieces,
            payload_bytes,
        })
    }

    /// Free an extent by dealing `free(base, member_span)` to every member of the stripe.
    pub fn free(&self, ext: ExtentHandle) {
        let width = self.slices.len() as u64;
        let rounds = ext.num_pieces.div_ceil(width);
        let member_span = rounds * self.piece_slot_bytes;
        let bases_opt = {
            let mut g = self.extents.lock().unwrap();
            if let Ok(pos) = g.1.binary_search_by_key(&ext.extent_id, |&(id, _)| id) {
                g.0 = g.0.saturating_sub(member_span);
                Some(g.1.remove(pos).1)
            } else {
                None
            }
        };
        if let Some(bases) = bases_opt {
            for (sl, &base_off) in self.slices.iter().zip(bases.iter()) {
                sl.member.free(base_off, member_span);
            }
        }
    }

    fn lookup_member_bases(&self, extent_id: u64) -> Result<Vec<u64>, StoreError> {
        let g = self.extents.lock().unwrap();
        let pos = g
            .1
            .binary_search_by_key(&extent_id, |&(id, _)| id)
            .map_err(|_| StoreError::IoFailed)?;
        Ok(g.1[pos].1.clone())
    }

    /// Build a zero-copy `SlotBatchPlan` partitioning `ranges` across stripe members so every
    /// member's slice in `AlignedSlotBuf` is 64 KiB-aligned and 64 KiB-multiple sized for direct
    /// DMA / `VK_EXT_external_memory_host` caller-buffer import.
    pub fn plan_slot_batch(
        &self,
        ext: &ExtentHandle,
        ranges: &[ExtentRange],
    ) -> Result<SlotBatchPlan, StoreError> {
        let width = self.slices.len();
        let slot = self.piece_slot_bytes as usize;
        let p_limbs = self.piece_limbs;
        let member_bases = self.lookup_member_bases(ext.extent_id)?;

        let total_pieces: usize = ranges.iter().map(|r| r.piece_count as usize).sum();
        let align = self.required_host_align();
        if total_pieces == 0 {
            return Ok(SlotBatchPlan {
                extent: *ext,
                piece_limbs: p_limbs,
                piece_slot_bytes: slot,
                required_buf_bytes: align,
                pieces: Vec::new(),
                member_transfers: Vec::new(),
            });
        }

        let mut member_counts = vec![0usize; width];
        for r in ranges {
            for p_off in 0..(r.piece_count as usize) {
                let piece_id = r.piece_start + (p_off as u64);
                if piece_id >= ext.num_pieces {
                    return Err(StoreError::IoFailed);
                }
                member_counts[(piece_id as usize) % width] += 1;
            }
        }

        let mut member_buf_base = vec![0usize; width];
        let mut member_buf_cap = vec![0usize; width];
        let mut cur_byte_off = 0usize;
        for m in 0..width {
            member_buf_base[m] = cur_byte_off;
            if member_counts[m] > 0 {
                let raw = member_counts[m] * slot;
                let cap = raw.div_ceil(align) * align;
                member_buf_cap[m] = cap;
                cur_byte_off += cap;
            }
        }

        let mut member_cur = vec![0usize; width];
        let mut per_member: Vec<Vec<ByteRange>> = (0..width).map(|_| Vec::new()).collect();
        let mut per_member_bytes = vec![0u64; width];
        let mut pieces = Vec::with_capacity(total_pieces);

        for r in ranges {
            for p_off in 0..(r.piece_count as usize) {
                let piece_id = r.piece_start + (p_off as u64);
                let limb_start = r.host_limb_offset + p_off * p_limbs;
                let m_idx = (piece_id as usize) % width;
                let local_idx = member_cur[m_idx];
                member_cur[m_idx] = local_idx + 1;

                let rel_host_off = local_idx * slot;
                let abs_buf_off = member_buf_base[m_idx] + rel_host_off;
                pieces.push(SlotPieceEntry {
                    piece_id,
                    buf_byte_off: abs_buf_off,
                    host_limb_offset: limb_start,
                });

                let round_idx = piece_id / (width as u64);
                let dev_off = member_bases[m_idx] + round_idx * self.piece_slot_bytes;
                push_coalesced_range(&mut per_member[m_idx], dev_off, rel_host_off, slot);
                per_member_bytes[m_idx] += slot as u64;
            }
        }

        let mut member_transfers = Vec::with_capacity(width);
        for m_idx in 0..width {
            if !per_member[m_idx].is_empty() {
                member_transfers.push(MemberTransferPlan {
                    m_idx,
                    buf_base: member_buf_base[m_idx],
                    buf_cap: member_buf_cap[m_idx],
                    ranges: std::mem::take(&mut per_member[m_idx]),
                    total_bytes: per_member_bytes[m_idx],
                });
            }
        }

        Ok(SlotBatchPlan {
            extent: *ext,
            piece_limbs: p_limbs,
            piece_slot_bytes: slot,
            required_buf_bytes: cur_byte_off.max(align),
            pieces,
            member_transfers,
        })
    }

    /// Issue a pure-DMA write batch for pre-checked slots in `slot_buf` (callable on `Loader` thread
    /// with zero compute and zero `memcpy`).
    pub fn write_planned_slots(
        &self,
        plan: &SlotBatchPlan,
        slot_buf: &AlignedSlotBuf,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        if plan.member_transfers.is_empty() {
            return Ok(());
        }
        std::sync::atomic::fence(Ordering::SeqCst);
        let full = unsafe { slot_buf.as_import_slice() };
        if plan.member_transfers.len() == 1 {
            let mt = &plan.member_transfers[0];
            let sub = &full[mt.buf_base..(mt.buf_base + mt.buf_cap)];
            self.slices[mt.m_idx].member.write(&mt.ranges, sub)?;
            bytes_counter.fetch_add(mt.total_bytes, Ordering::Relaxed);
            return Ok(());
        }

        let mut results: Vec<Result<(), StoreError>> =
            (0..plan.member_transfers.len()).map(|_| Ok(())).collect();
        thread::scope(|s| {
            for (mt, res_slot) in plan.member_transfers.iter().zip(results.iter_mut()) {
                let sl = &self.slices[mt.m_idx];
                let sub = &full[mt.buf_base..(mt.buf_base + mt.buf_cap)];
                s.spawn(move || {
                    let r = sl.member.write(&mt.ranges, sub);
                    if r.is_ok() {
                        bytes_counter.fetch_add(mt.total_bytes, Ordering::Relaxed);
                    }
                    *res_slot = r;
                });
            }
        });
        for r in results {
            r?;
        }
        Ok(())
    }

    /// Issue a pure-DMA read batch into `slot_buf` (callable on `Loader` thread with zero compute
    /// and zero `memcpy`; workers then scrub check words inline or in a `check_n` pass).
    pub fn read_planned_slots(
        &self,
        plan: &SlotBatchPlan,
        slot_buf: &mut AlignedSlotBuf,
        telemetry: &Telemetry,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        if plan.member_transfers.is_empty() {
            return Ok(());
        }
        slot_buf.ensure_capacity(plan.required_buf_bytes);
        std::sync::atomic::fence(Ordering::SeqCst);
        if plan.member_transfers.len() == 1 {
            let mt = &plan.member_transfers[0];
            let full = unsafe { slot_buf.as_import_slice_mut() };
            let sub = &mut full[mt.buf_base..(mt.buf_base + mt.buf_cap)];
            if let Err(e) = self.slices[mt.m_idx].member.read(&mt.ranges, sub) {
                let mid = self.slices[mt.m_idx].member.id();
                if e == StoreError::DeviceLost {
                    telemetry.mark_device_stopped(mid);
                }
                telemetry.record_failure(0, 0, Some(mid));
                return Err(e);
            }
            bytes_counter.fetch_add(mt.total_bytes, Ordering::Relaxed);
            return Ok(());
        }

        // Split `slot_buf` into disjoint mutable member slices in ascending `buf_base` order:
        let full = unsafe { slot_buf.as_import_slice_mut() };
        let mut rem = full;
        let mut prev_end = 0usize;
        let mut member_subs: Vec<(&MemberTransferPlan, &mut [u8])> =
            Vec::with_capacity(plan.member_transfers.len());
        for mt in &plan.member_transfers {
            let skip = mt.buf_base - prev_end;
            let (_, after_skip) = rem.split_at_mut(skip);
            let (sub, after_sub) = after_skip.split_at_mut(mt.buf_cap);
            prev_end = mt.buf_base + mt.buf_cap;
            rem = after_sub;
            member_subs.push((mt, sub));
        }

        let mut results: Vec<(usize, Result<(), StoreError>)> =
            member_subs.iter().map(|(mt, _)| (mt.m_idx, Ok(()))).collect();
        thread::scope(|s| {
            for ((mt, sub), res_slot) in member_subs.into_iter().zip(results.iter_mut()) {
                let sl = &self.slices[mt.m_idx];
                s.spawn(move || {
                    let r = sl.member.read(&mt.ranges, sub);
                    if r.is_ok() {
                        bytes_counter.fetch_add(mt.total_bytes, Ordering::Relaxed);
                    }
                    res_slot.1 = r;
                });
            }
        });
        for (m_idx, r) in results {
            if let Err(e) = r {
                let mid = self.slices[m_idx].member.id();
                if e == StoreError::DeviceLost {
                    telemetry.mark_device_stopped(mid);
                }
                telemetry.record_failure(0, 0, Some(mid));
                return Err(e);
            }
        }
        Ok(())
    }

    /// Write back a single repaired piece slot to its stripe member (Section 11: "A piece that is
    /// put right is written back").
    pub fn write_single_piece_slot(
        &self,
        ext: &ExtentHandle,
        piece_id: u64,
        slot_bytes: &[u8],
        telemetry: &Telemetry,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        let slot = self.piece_slot_bytes as usize;
        debug_assert!(slot_bytes.len() >= slot);
        let member_bases = self.lookup_member_bases(ext.extent_id)?;
        let width = self.slices.len();
        let m_idx = (piece_id as usize) % width;
        let round_idx = piece_id / (width as u64);
        let dev_off = member_bases[m_idx] + round_idx * self.piece_slot_bytes;
        let mut wb_stage = AlignedSlotBuf::new_uninit(slot);
        unsafe {
            let dst = wb_stage.as_import_slice_mut();
            dst[..slot].copy_from_slice(&slot_bytes[..slot]);
        }
        let wb = [ByteRange {
            device_offset: dev_off,
            host_offset: 0,
            len: slot,
        }];
        let dev_id = self.slices[m_idx].member.id();
        if let Err(e) = self.slices[m_idx]
            .member
            .write(&wb, unsafe { wb_stage.as_import_slice() })
        {
            telemetry.mark_device_stopped(dev_id);
            telemetry.record_failure(0, 0, Some(dev_id));
            return Err(e);
        }
        bytes_counter.fetch_add(slot as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Compute and append `[w0, w1, w2]` in place for a slot whose data bytes `[0 .. piece_limbs*8)`
    /// are already populated in `slot_bytes` (e.g. inline in Sweep 2 while hot in cache, or during
    /// a Sweep 1 `check_n` pass). When `count_as_traffic` is true, charges `piece_limbs * 8` bytes
    /// to `kernels.stats.bytes_moved` (Section 11 column-sweep `check_n` pass).
    #[inline]
    pub fn compute_slot_check_in_place(
        &self,
        slot_bytes: &mut [u8],
        kernels: &Kernels,
        count_as_traffic: bool,
    ) {
        let slot = self.piece_slot_bytes as usize;
        let p_limbs = self.piece_limbs;
        let p_bytes = p_limbs * 8;
        debug_assert!(slot_bytes.len() >= slot && (slot_bytes.as_ptr() as usize).is_multiple_of(8));
        if count_as_traffic && kernels.count_work {
            kernels
                .stats
                .bytes_moved
                .fetch_add(p_bytes as u64, Ordering::Relaxed);
        }
        let words3 = {
            #[cfg(target_endian = "little")]
            {
                let limb_slice =
                    unsafe { std::slice::from_raw_parts(slot_bytes.as_ptr() as *const u64, p_limbs) };
                compute_limbs_all3(kernels, limb_slice)
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut tmp = vec![0u64; p_limbs];
                copy_le_bytes_to_u64(&slot_bytes[..p_bytes], &mut tmp);
                compute_limbs_all3(kernels, &tmp)
            }
        };
        slot_bytes[p_bytes..(p_bytes + 8)].copy_from_slice(&words3[0].to_le_bytes());
        slot_bytes[(p_bytes + 8)..(p_bytes + 16)].copy_from_slice(&words3[1].to_le_bytes());
        slot_bytes[(p_bytes + 16)..(p_bytes + 24)].copy_from_slice(&words3[2].to_le_bytes());
        if p_bytes + 24 < slot {
            slot_bytes[(p_bytes + 24)..slot].fill(0);
        }
    }

    /// Scrub `[0 .. piece_limbs*8)` against `[w0, w1, w2]` in place, repairing single-limb errors
    /// via Rung 1 and writing the repaired piece back to `Store` (Section 11).
    pub fn scrub_or_repair_slot_in_place(
        &self,
        ext: &ExtentHandle,
        piece_id: u64,
        slot_bytes: &mut [u8],
        kernels: &Kernels,
        telemetry: &Telemetry,
        bytes_counter: &AtomicU64,
        count_as_traffic: bool,
    ) -> Result<(), StoreError> {
        let slot = self.piece_slot_bytes as usize;
        let p_limbs = self.piece_limbs;
        let p_bytes = p_limbs * 8;
        debug_assert!(slot_bytes.len() >= slot && (slot_bytes.as_ptr() as usize).is_multiple_of(8));
        if count_as_traffic && kernels.count_work {
            kernels
                .stats
                .bytes_moved
                .fetch_add(p_bytes as u64, Ordering::Relaxed);
        }

        let exp_w0 = u64::from_le_bytes(slot_bytes[p_bytes..(p_bytes + 8)].try_into().unwrap());
        let exp_w1 =
            u64::from_le_bytes(slot_bytes[(p_bytes + 8)..(p_bytes + 16)].try_into().unwrap());
        let exp_w2 =
            u64::from_le_bytes(slot_bytes[(p_bytes + 16)..(p_bytes + 24)].try_into().unwrap());
        let expected = [exp_w0, exp_w1, exp_w2];

        telemetry.record_scrub();

        #[cfg(target_endian = "little")]
        let actual = {
            let limb_slice =
                unsafe { std::slice::from_raw_parts(slot_bytes.as_ptr() as *const u64, p_limbs) };
            compute_limbs_all3(kernels, limb_slice)
        };
        #[cfg(not(target_endian = "little"))]
        let mut fallback_buf = {
            let mut tmp = vec![0u64; p_limbs];
            copy_le_bytes_to_u64(&slot_bytes[..p_bytes], &mut tmp);
            tmp
        };
        #[cfg(not(target_endian = "little"))]
        let actual = compute_limbs_all3(kernels, &fallback_buf);

        if actual == expected {
            return Ok(());
        }

        let width = self.slices.len();
        let m_idx = (piece_id as usize) % width;
        let dev_id = self.slices[m_idx].member.id();
        let match_cnt = usize::from(actual[0] == expected[0])
            + usize::from(actual[1] == expected[1])
            + usize::from(actual[2] == expected[2]);
        let repaired = if match_cnt >= 2 {
            telemetry.record_correction(0, 0, Some(dev_id));
            true
        } else {
            #[cfg(target_endian = "little")]
            let outcome = {
                let limb_mut = unsafe {
                    std::slice::from_raw_parts_mut(slot_bytes.as_mut_ptr() as *mut u64, p_limbs)
                };
                try_correct_single_limb(limb_mut, actual, expected)
            };
            #[cfg(not(target_endian = "little"))]
            let outcome = {
                let o = try_correct_single_limb(&mut fallback_buf, actual, expected);
                if matches!(o, CorrectionOutcome::Corrected { .. }) {
                    unsafe { copy_u64_to_le_bytes(&fallback_buf, slot_bytes.as_mut_ptr()) };
                }
                o
            };
            match outcome {
                CorrectionOutcome::Clean => true,
                CorrectionOutcome::Corrected { .. } => {
                    telemetry.record_correction(0, 0, Some(dev_id));
                    true
                }
                CorrectionOutcome::BeyondOneLimb => false,
            }
        };

        if !repaired {
            telemetry.record_failure(0, 0, Some(dev_id));
            return Err(StoreError::IntegrityBeyondRepair);
        }

        self.compute_slot_check_in_place(slot_bytes, kernels, false);
        self.write_single_piece_slot(ext, piece_id, &slot_bytes[..slot], telemetry, bytes_counter)
    }

    /// Write a batch of piece ranges of `ext` from `src_limbs`, computing all 3 check words per
    /// piece (across `Scheduler` workers when provided, with zero extra helper threads) and
    /// issuing a single batch write to each stripe member.
    pub fn write_ranges(
        &self,
        ext: &ExtentHandle,
        ranges: &[ExtentRange],
        src_limbs: &[u64],
        kernels: &Kernels,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        self.write_ranges_with_sched(ext, ranges, src_limbs, kernels, None, 1, bytes_counter)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_ranges_with_sched(
        &self,
        ext: &ExtentHandle,
        ranges: &[ExtentRange],
        src_limbs: &[u64],
        kernels: &Kernels,
        sched: Option<&Scheduler>,
        max_workers: usize,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        let plan = self.plan_slot_batch(ext, ranges)?;
        if plan.pieces.is_empty() {
            return Ok(());
        }
        let slot = plan.piece_slot_bytes;
        let p_limbs = plan.piece_limbs;
        let slot_buf = AlignedSlotBuf::new_uninit(plan.required_buf_bytes);
        let base_addr = slot_buf.ptr as usize;

        let pack_one = |pe: &SlotPieceEntry| {
            let avail = src_limbs.len().saturating_sub(pe.host_limb_offset).min(p_limbs);
            let dst_slot =
                unsafe { std::slice::from_raw_parts_mut((base_addr + pe.buf_byte_off) as *mut u8, slot) };
            unsafe {
                if avail > 0 {
                    copy_u64_to_le_bytes(
                        &src_limbs[pe.host_limb_offset..(pe.host_limb_offset + avail)],
                        dst_slot.as_mut_ptr(),
                    );
                }
                if avail < p_limbs {
                    std::ptr::write_bytes(
                        dst_slot.as_mut_ptr().add(avail * 8),
                        0,
                        (p_limbs - avail) * 8,
                    );
                }
            }
            self.compute_slot_check_in_place(dst_slot, kernels, false);
        };

        if let Some(sc) = sched.filter(|_| max_workers > 1 && plan.pieces.len() >= 4) {
            sc.par_for_max(plan.pieces.len(), 1, max_workers, |idx| {
                pack_one(&plan.pieces[idx]);
            });
        } else {
            for pe in &plan.pieces {
                pack_one(pe);
            }
        }

        self.write_planned_slots(&plan, &slot_buf, bytes_counter)
    }

    /// Read a batch of piece ranges of `ext` into `dst_limbs`, scrubbing every piece against its
    /// 3 check words `(w0, w1, w2)` and repairing + writing back any single-limb error (Section 11).
    pub fn read_ranges(
        &self,
        ext: &ExtentHandle,
        ranges: &[ExtentRange],
        dst_limbs: &mut [u64],
        kernels: &Kernels,
        telemetry: &Telemetry,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        self.read_ranges_with_sched(
            ext,
            ranges,
            dst_limbs,
            kernels,
            telemetry,
            None,
            1,
            bytes_counter,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_ranges_with_sched(
        &self,
        ext: &ExtentHandle,
        ranges: &[ExtentRange],
        dst_limbs: &mut [u64],
        kernels: &Kernels,
        telemetry: &Telemetry,
        sched: Option<&Scheduler>,
        max_workers: usize,
        bytes_counter: &AtomicU64,
    ) -> Result<(), StoreError> {
        let plan = self.plan_slot_batch(ext, ranges)?;
        if plan.pieces.is_empty() {
            return Ok(());
        }
        let mut slot_buf = AlignedSlotBuf::new_uninit(plan.required_buf_bytes);
        self.read_planned_slots(&plan, &mut slot_buf, telemetry, bytes_counter)?;

        let slot = plan.piece_slot_bytes;
        let p_limbs = plan.piece_limbs;
        let p_bytes = p_limbs * 8;
        let base_addr = slot_buf.ptr as usize;

        let mut monotonic_non_overlapping = true;
        if ranges.len() > 1 {
            let mut prev_end = 0usize;
            for r in ranges {
                let len_limbs = (r.piece_count as usize).saturating_mul(p_limbs);
                if r.host_limb_offset < prev_end {
                    monotonic_non_overlapping = false;
                    break;
                }
                prev_end = r.host_limb_offset.saturating_add(len_limbs);
            }
        }

        if let Some(sc) = sched.filter(|_| {
            monotonic_non_overlapping && max_workers > 1 && plan.pieces.len() >= 4
        }) {
            let dst_addr = dst_limbs.as_mut_ptr() as usize;
            let dst_len = dst_limbs.len();
            let err_flag = AtomicBool::new(false);
            let first_err: Mutex<Option<StoreError>> = Mutex::new(None);
            sc.par_for_max(plan.pieces.len(), 1, max_workers, |idx| {
                if err_flag.load(Ordering::Relaxed) {
                    return;
                }
                let pe = &plan.pieces[idx];
                let slot_mut = unsafe {
                    std::slice::from_raw_parts_mut((base_addr + pe.buf_byte_off) as *mut u8, slot)
                };
                if let Err(e) = self.scrub_or_repair_slot_in_place(
                    ext,
                    pe.piece_id,
                    slot_mut,
                    kernels,
                    telemetry,
                    bytes_counter,
                    false,
                ) {
                    err_flag.store(true, Ordering::Relaxed);
                    let mut g = first_err.lock().unwrap();
                    if g.is_none() {
                        *g = Some(e);
                    }
                    return;
                }
                let copy_limbs = dst_len.saturating_sub(pe.host_limb_offset).min(p_limbs);
                if copy_limbs > 0 {
                    unsafe {
                        let dst_slice = std::slice::from_raw_parts_mut(
                            (dst_addr as *mut u64).add(pe.host_limb_offset),
                            copy_limbs,
                        );
                        copy_le_bytes_to_u64(&slot_mut[..(copy_limbs * 8)], dst_slice);
                    }
                }
            });
            if let Some(e) = first_err.into_inner().unwrap() {
                return Err(e);
            }
            return Ok(());
        }

        let raw_bytes = unsafe { slot_buf.as_import_slice_mut() };
        for pe in &plan.pieces {
            let slot_mut = &mut raw_bytes[pe.buf_byte_off..(pe.buf_byte_off + slot)];
            self.scrub_or_repair_slot_in_place(
                ext,
                pe.piece_id,
                slot_mut,
                kernels,
                telemetry,
                bytes_counter,
                false,
            )?;
            let copy_limbs = dst_limbs.len().saturating_sub(pe.host_limb_offset).min(p_limbs);
            if copy_limbs > 0 {
                copy_le_bytes_to_u64(
                    &slot_mut[..(copy_limbs * 8)],
                    &mut dst_limbs[pe.host_limb_offset..(pe.host_limb_offset + copy_limbs)],
                );
            }
            let _ = p_bytes;
        }
        Ok(())
    }
}

const MAX_COALESCE_BYTES: usize = 64 * 1024 * 1024;

#[inline]
fn push_coalesced_range(ranges: &mut Vec<ByteRange>, dev_off: u64, host_off: usize, len: usize) {
    if let Some(last) = ranges.last_mut()
        && last.device_offset + (last.len as u64) == dev_off
        && last.host_offset + last.len == host_off
        && last.len + len <= MAX_COALESCE_BYTES
    {
        last.len += len;
        return;
    }
    ranges.push(ByteRange {
        device_offset: dev_off,
        host_offset: host_off,
        len,
    });
}

#[inline]
unsafe fn copy_u64_to_le_bytes(src: &[u64], dst_bytes: *mut u8) {
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst_bytes, src.len() * 8);
    }
    #[cfg(not(target_endian = "little"))]
    unsafe {
        for (i, &w) in src.iter().enumerate() {
            std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), dst_bytes.add(i * 8), 8);
        }
    }
}

#[inline]
fn copy_le_bytes_to_u64(src_bytes: &[u8], dst: &mut [u64]) {
    debug_assert_eq!(src_bytes.len(), dst.len() * 8);
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_bytes.as_ptr(),
            dst.as_mut_ptr() as *mut u8,
            src_bytes.len(),
        );
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (i, out) in dst.iter_mut().enumerate() {
            *out = u64::from_le_bytes(src_bytes[(i * 8)..((i + 1) * 8)].try_into().unwrap());
        }
    }
}

static LIVE_SLOT_BUFS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
type HostBufReleaseHook = Box<dyn Fn(usize, usize) -> bool + Send + Sync>;
static SLOT_RELEASE_HOOKS: Mutex<Vec<HostBufReleaseHook>> = Mutex::new(Vec::new());

/// Register a callback invoked synchronously right before an `AlignedSlotBuf` allocation is
/// returned to the allocator (`false` return removes a dropped device listener).
pub fn register_host_buf_release_hook(hook: HostBufReleaseHook) {
    if let Ok(mut g) = SLOT_RELEASE_HOOKS.lock() {
        g.push(hook);
    }
}

/// If `[span_addr .. span_addr + span_len)` lies inside a live `AlignedSlotBuf`, return its full
/// `(base_addr, byte_capacity)` so DMA backends can reuse a persistent host import.
pub fn lookup_registered_slot_buf(span_addr: usize, span_len: usize) -> Option<(usize, usize)> {
    let end = span_addr.checked_add(span_len)?;
    let g = LIVE_SLOT_BUFS.lock().ok()?;
    g.iter()
        .copied()
        .find(|&(base, cap)| span_addr >= base && end <= base.saturating_add(cap))
}

fn register_slot_buf_alloc(addr: usize, cap: usize) {
    if let Ok(mut g) = LIVE_SLOT_BUFS.lock() {
        g.push((addr, cap));
    }
}

fn unregister_slot_buf_alloc(addr: usize, cap: usize) {
    if let Ok(mut g) = LIVE_SLOT_BUFS.lock() {
        if let Some(pos) = g.iter().position(|&(b, c)| b == addr && c == cap) {
            g.swap_remove(pos);
        }
    }
    if let Ok(mut hooks) = SLOT_RELEASE_HOOKS.lock() {
        hooks.retain(|h| h(addr, cap));
    }
}

struct RawStageBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl Drop for RawStageBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unregister_slot_buf_alloc(self.ptr as usize, self.layout.size());
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

thread_local! {
    static TLS_STAGE_BUF: std::cell::RefCell<Option<RawStageBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// 64 KiB-aligned, 64 KiB-multiple host slot buffer suitable for zero-copy direct DMA (`OVERLAPPED`
/// unbuffered SSD I/O and `VK_EXT_external_memory_host` GPU transfer import).
pub struct AlignedSlotBuf {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
    tracked: Option<Arc<(AtomicU64, AtomicU64)>>,
}

unsafe impl Send for AlignedSlotBuf {}
unsafe impl Sync for AlignedSlotBuf {}

impl AlignedSlotBuf {
    pub fn new_uninit(bytes: usize) -> Self {
        let req_len = bytes.max(HOST_IMPORT_ALIGN).div_ceil(HOST_IMPORT_ALIGN) * HOST_IMPORT_ALIGN;
        if req_len <= 128 * 1024 * 1024 {
            let reused = TLS_STAGE_BUF
                .try_with(|cell| {
                    let mut opt = cell.borrow_mut();
                    if let Some(ref cur) = *opt
                        && cur.layout.size() >= req_len
                    {
                        let mut taken = opt.take().unwrap();
                        let p = taken.ptr;
                        let lay = taken.layout;
                        taken.ptr = std::ptr::null_mut();
                        return Some((p, lay));
                    }
                    None
                })
                .ok()
                .flatten();
            if let Some((ptr, layout)) = reused {
                return Self {
                    ptr,
                    len: bytes,
                    layout,
                    tracked: None,
                };
            }
        }
        let alloc_len = if req_len <= 16 * 1024 * 1024 {
            req_len.next_power_of_two().max(HOST_IMPORT_ALIGN)
        } else {
            req_len
        };
        let layout =
            std::alloc::Layout::from_size_align(alloc_len, HOST_IMPORT_ALIGN).expect("slot layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        register_slot_buf_alloc(ptr as usize, layout.size());
        Self {
            ptr,
            len: bytes,
            layout,
            tracked: None,
        }
    }

    pub fn new_tracked(
        bytes: usize,
        align: usize,
        live_peak: Arc<(AtomicU64, AtomicU64)>,
    ) -> Self {
        let a = align.max(4096).next_power_of_two();
        let alloc_len = bytes.max(a).div_ceil(a) * a;
        let layout = std::alloc::Layout::from_size_align(alloc_len, a).expect("slot layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        register_slot_buf_alloc(ptr as usize, layout.size());
        let prev = live_peak.0.fetch_add(alloc_len as u64, Ordering::Relaxed);
        live_peak
            .1
            .fetch_max(prev + (alloc_len as u64), Ordering::Relaxed);
        Self {
            ptr,
            len: bytes,
            layout,
            tracked: Some(live_peak),
        }
    }

    pub fn ensure_capacity(&mut self, min_bytes: usize) {
        if self.layout.size() >= min_bytes {
            self.len = self.len.max(min_bytes);
            return;
        }
        let a = self.layout.align().max(4096);
        let new_alloc = min_bytes.div_ceil(a) * a;
        let new_lay = std::alloc::Layout::from_size_align(new_alloc, a).expect("grow slot layout");
        let new_ptr = unsafe { std::alloc::alloc_zeroed(new_lay) };
        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(new_lay);
        }
        register_slot_buf_alloc(new_ptr as usize, new_lay.size());
        let old_ptr = self.ptr;
        let old_lay = self.layout;
        if let Some(ref tr) = self.tracked {
            tr.0.fetch_sub(old_lay.size() as u64, Ordering::Relaxed);
            let prev = tr.0.fetch_add(new_alloc as u64, Ordering::Relaxed);
            tr.1.fetch_max(prev + (new_alloc as u64), Ordering::Relaxed);
        }
        if !old_ptr.is_null() {
            unregister_slot_buf_alloc(old_ptr as usize, old_lay.size());
            unsafe { std::alloc::dealloc(old_ptr, old_lay) };
        }
        self.ptr = new_ptr;
        self.len = min_bytes;
        self.layout = new_lay;
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.layout.size()
    }

    #[inline]
    pub fn used_len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn as_import_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    #[inline]
    pub unsafe fn as_import_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedSlotBuf {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let p = self.ptr;
        let lay = self.layout;
        self.ptr = std::ptr::null_mut();
        if let Some(tr) = self.tracked.take() {
            tr.0.fetch_sub(lay.size() as u64, Ordering::Relaxed);
            unregister_slot_buf_alloc(p as usize, lay.size());
            unsafe { std::alloc::dealloc(p, lay) };
            return;
        }
        if lay.size() <= 256 * 1024 * 1024 {
            let stored = TLS_STAGE_BUF
                .try_with(|cell| {
                    let mut opt = cell.borrow_mut();
                    if opt.as_ref().is_none_or(|cur| cur.layout.size() < lay.size()) {
                        *opt = Some(RawStageBuf { ptr: p, layout: lay });
                        true
                    } else {
                        false
                    }
                })
                .unwrap_or(false);
            if stored {
                return;
            }
        }
        unregister_slot_buf_alloc(p as usize, lay.size());
        unsafe { std::alloc::dealloc(p, lay) };
    }
}

/// Form stripes from rated members according to Section 10:
/// "A stripe takes the fastest members left, as many as give it the highest rate, and the larger
/// count where two counts tie. That rate is counted as the number of its members times the rate
/// of the slowest... Each member gives its stripe the same number of bytes, as many as the
/// smallest has, and what a larger member has beyond that is a member again."
pub fn form_stripes(
    members_with_rates: Vec<(Arc<dyn ExtentMember>, f64)>,
    piece_limbs: usize,
) -> Vec<ExtentDevice> {
    #[derive(Clone)]
    struct PoolItem {
        member: Arc<dyn ExtentMember>,
        rate: f64,
        rem_bytes: u64,
    }

    let min_useful = ((piece_limbs as u64) * 8 + 4096) * 2;
    let mut pool: Vec<PoolItem> = Vec::new();
    for (m, rate) in members_with_rates {
        let align = m.alignment().max(4096);
        let cap = (m.capacity() / align) * align;
        if cap >= min_useful {
            pool.push(PoolItem {
                member: m,
                rate: rate.max(1.0),
                rem_bytes: cap,
            });
        }
    }

    let mut devices = Vec::new();
    let mut stripe_id = 0usize;
    while !pool.is_empty() {
        // Sort fastest first, breaking ties by larger remaining capacity.
        pool.sort_by(|a, b| {
            b.rate
                .total_cmp(&a.rate)
                .then_with(|| b.rem_bytes.cmp(&a.rem_bytes))
        });

        let mut best_k = 1usize;
        let mut best_rate = pool[0].rate;
        for k in 2..=pool.len() {
            let cand_rate = (k as f64) * pool[k - 1].rate;
            if cand_rate + 1e-6 >= best_rate {
                best_rate = cand_rate;
                best_k = k;
            }
        }

        let max_align = pool[..best_k]
            .iter()
            .map(|it| it.member.alignment().max(4096))
            .max()
            .unwrap_or(4096);
        let raw_slot = (piece_limbs as u64) * 8 + 24;
        let slot_bytes = raw_slot.div_ceil(max_align) * max_align;

        let min_bytes = pool[..best_k]
            .iter()
            .map(|it| (it.rem_bytes / slot_bytes) * slot_bytes)
            .min()
            .unwrap_or(0);
        if min_bytes < min_useful || min_bytes < slot_bytes {
            pool.retain(|it| it.rem_bytes >= min_useful);
            if min_bytes < slot_bytes {
                break;
            }
            continue;
        }

        let mut slices = Vec::with_capacity(best_k);
        for item in &mut pool[..best_k] {
            slices.push((Arc::clone(&item.member), min_bytes));
            item.rem_bytes -= min_bytes;
        }
        pool.retain(|it| it.rem_bytes >= min_useful);

        let dev = ExtentDevice::new(
            format!("stripe-{stripe_id}"),
            slices,
            piece_limbs,
            best_rate,
        );
        if dev.capacity() > 0 {
            devices.push(dev);
            stripe_id += 1;
        }
    }

    // Rank extent devices by transfer rate beside sweep step descending.
    devices.sort_by(|a, b| b.beside_sweep_rate_bps.total_cmp(&a.beside_sweep_rate_bps));
    devices
}

/// The Store: ranked list of striped `ExtentDevice`s plus the single dedicated `Loader` thread
/// (`supi-loader`, Section 6 & Section 10).
pub struct Store {
    pub devices: Vec<ExtentDevice>,
    pub bytes_moved: AtomicU64,
    pub loader_batches: AtomicU64,
    tile_bytes: Arc<(AtomicU64, AtomicU64)>,
    parked_bytes: AtomicU64,
    parked_peak_bytes: AtomicU64,
    loader_queue: Arc<(Mutex<VecDeque<Box<dyn FnOnce() + Send>>>, Condvar, AtomicBool)>,
    loader_thread: Option<JoinHandle<()>>,
}

impl Store {
    pub fn new(devices: Vec<ExtentDevice>) -> Self {
        let loader_queue: Arc<(Mutex<VecDeque<Box<dyn FnOnce() + Send>>>, Condvar, AtomicBool)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new(), AtomicBool::new(false)));

        let loader_thread = if !devices.is_empty() {
            let q = Arc::clone(&loader_queue);
            Some(
                thread::Builder::new()
                    .name("supi-loader".to_string())
                    .spawn(move || {
                        loop {
                            let job = {
                                let mut guard = q.0.lock().unwrap();
                                while guard.is_empty() && !q.2.load(Ordering::Acquire) {
                                    guard = q.1.wait(guard).unwrap();
                                }
                                if q.2.load(Ordering::Acquire) && guard.is_empty() {
                                    return;
                                }
                                guard.pop_front()
                            };
                            if let Some(f) = job {
                                f();
                            }
                        }
                    })
                    .expect("spawn store loader thread"),
            )
        } else {
            None
        };

        Self {
            devices,
            bytes_moved: AtomicU64::new(0),
            loader_batches: AtomicU64::new(0),
            tile_bytes: Arc::new((AtomicU64::new(0), AtomicU64::new(0))),
            parked_bytes: AtomicU64::new(0),
            parked_peak_bytes: AtomicU64::new(0),
            loader_queue,
            loader_thread,
        }
    }

    pub fn total_capacity(&self) -> u64 {
        self.devices.iter().map(|d| d.capacity()).sum()
    }

    pub fn total_bytes_moved(&self) -> u64 {
        self.bytes_moved.load(Ordering::Relaxed)
    }

    pub fn total_loader_batches(&self) -> u64 {
        self.loader_batches.load(Ordering::Relaxed)
    }

    pub fn tile_live_bytes(&self) -> u64 {
        self.tile_bytes.0.load(Ordering::Relaxed)
    }

    pub fn tile_peak_bytes(&self) -> u64 {
        self.tile_bytes.1.load(Ordering::Relaxed)
    }

    pub fn required_host_align(&self) -> usize {
        self.devices
            .iter()
            .map(|d| d.required_host_align())
            .max()
            .unwrap_or(4096)
    }

    pub fn upper_slot_buf_bytes(&self, num_pieces: usize) -> usize {
        self.devices
            .iter()
            .map(|d| d.upper_slot_buf_bytes(num_pieces))
            .max()
            .unwrap_or(4096)
    }

    pub fn alloc_tile_slot_buf(&self, bytes: usize) -> AlignedSlotBuf {
        AlignedSlotBuf::new_tracked(bytes, self.required_host_align(), Arc::clone(&self.tile_bytes))
    }

    pub fn note_parked_alloc(&self, bytes: u64) {
        let prev = self.parked_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.parked_peak_bytes
            .fetch_max(prev.saturating_add(bytes), Ordering::Relaxed);
    }

    pub fn note_parked_free(&self, bytes: u64) {
        self.parked_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    pub fn parked_live_bytes(&self) -> u64 {
        self.parked_bytes.load(Ordering::Relaxed)
    }

    pub fn parked_peak_bytes(&self) -> u64 {
        self.parked_peak_bytes.load(Ordering::Relaxed)
    }

    pub fn allocate(&self, payload_bytes: u64) -> Result<ExtentHandle, StoreError> {
        for (idx, dev) in self.devices.iter().enumerate() {
            if let Ok(h) = dev.allocate(idx, payload_bytes) {
                return Ok(h);
            }
        }
        Err(StoreError::OutOfSpace)
    }

    pub fn free(&self, handle: ExtentHandle) {
        if let Some(dev) = self.devices.get(handle.device_idx) {
            dev.free(handle);
        }
    }

    pub fn write_all(
        &self,
        handle: &ExtentHandle,
        limbs: &[u64],
        kernels: &Kernels,
    ) -> Result<(), StoreError> {
        let range = [ExtentRange {
            piece_start: 0,
            piece_count: handle.num_pieces,
            host_limb_offset: 0,
        }];
        self.devices[handle.device_idx].write_ranges(
            handle,
            &range,
            limbs,
            kernels,
            &self.bytes_moved,
        )
    }

    pub fn write_all_par(
        &self,
        handle: &ExtentHandle,
        limbs: &[u64],
        kernels: &Kernels,
        sched: &Scheduler,
        max_workers: usize,
    ) -> Result<(), StoreError> {
        let range = [ExtentRange {
            piece_start: 0,
            piece_count: handle.num_pieces,
            host_limb_offset: 0,
        }];
        self.devices[handle.device_idx].write_ranges_with_sched(
            handle,
            &range,
            limbs,
            kernels,
            Some(sched),
            max_workers,
            &self.bytes_moved,
        )
    }

    pub fn read_all(
        &self,
        handle: &ExtentHandle,
        dst_limbs: &mut [u64],
        kernels: &Kernels,
        telemetry: &Telemetry,
    ) -> Result<(), StoreError> {
        let range = [ExtentRange {
            piece_start: 0,
            piece_count: handle.num_pieces,
            host_limb_offset: 0,
        }];
        self.devices[handle.device_idx].read_ranges(
            handle,
            &range,
            dst_limbs,
            kernels,
            telemetry,
            &self.bytes_moved,
        )
    }

    pub fn read_all_par(
        &self,
        handle: &ExtentHandle,
        dst_limbs: &mut [u64],
        kernels: &Kernels,
        telemetry: &Telemetry,
        sched: &Scheduler,
        max_workers: usize,
    ) -> Result<(), StoreError> {
        let range = [ExtentRange {
            piece_start: 0,
            piece_count: handle.num_pieces,
            host_limb_offset: 0,
        }];
        self.devices[handle.device_idx].read_ranges_with_sched(
            handle,
            &range,
            dst_limbs,
            kernels,
            telemetry,
            Some(sched),
            max_workers,
            &self.bytes_moved,
        )
    }

    /// Submit an asynchronous batch transfer to the dedicated `Loader` thread while workers sweep
    /// or work on other numbers, returning a handle that waits until the transfer completes.
    pub fn submit_loader_batch<F, R>(&self, f: F) -> LoaderWaitHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.loader_batches.fetch_add(1, Ordering::Relaxed);
        let pair = Arc::new((Mutex::new(None::<R>), Condvar::new()));
        if self.loader_thread.is_none() {
            let r = f();
            *pair.0.lock().unwrap() = Some(r);
            return LoaderWaitHandle { pair };
        }
        let pair_cl = Arc::clone(&pair);
        {
            let mut guard = self.loader_queue.0.lock().unwrap();
            guard.push_back(Box::new(move || {
                let res = f();
                let mut g = pair_cl.0.lock().unwrap();
                *g = Some(res);
                pair_cl.1.notify_one();
            }));
            self.loader_queue.1.notify_one();
        }
        LoaderWaitHandle { pair }
    }

    /// Submit a lifetime-scoped batch transfer to the `Loader` thread (`'a` borrows `&Store`,
    /// `&Telemetry`, etc.). The returned `LoaderScopedHandle<'a, R>` blocks in `Drop` if not yet
    /// awaited, guaranteeing the job completes before any `'a` borrow ends.
    pub fn submit_loader_scoped<'a, F, R>(&'a self, f: F) -> LoaderScopedHandle<'a, R>
    where
        F: FnOnce() -> R + Send + 'a,
        R: Send + 'static,
    {
        self.loader_batches.fetch_add(1, Ordering::Relaxed);
        let pair: Arc<(Mutex<Option<std::thread::Result<R>>>, Condvar)> =
            Arc::new((Mutex::new(None), Condvar::new()));
        if self.loader_thread.is_none() {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            *pair.0.lock().unwrap() = Some(r);
            return LoaderScopedHandle {
                pair,
                waited: false,
                _lt: std::marker::PhantomData,
            };
        }
        let pair_cl = Arc::clone(&pair);
        let job: Box<dyn FnOnce() + Send + 'a> = Box::new(move || {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let mut g = pair_cl.0.lock().unwrap();
            *g = Some(res);
            pair_cl.1.notify_one();
        });
        let static_job: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(job) };
        {
            let mut guard = self.loader_queue.0.lock().unwrap();
            guard.push_back(static_job);
            self.loader_queue.1.notify_one();
        }
        LoaderScopedHandle {
            pair,
            waited: false,
            _lt: std::marker::PhantomData,
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.loader_queue.2.store(true, Ordering::Release);
        self.loader_queue.1.notify_all();
        if let Some(h) = self.loader_thread.take() {
            let _ = h.join();
        }
    }
}

pub struct LoaderWaitHandle<R> {
    pair: Arc<(Mutex<Option<R>>, Condvar)>,
}

impl<R> LoaderWaitHandle<R> {
    pub fn wait(self) -> R {
        let mut g = self.pair.0.lock().unwrap();
        while g.is_none() {
            g = self.pair.1.wait(g).unwrap();
        }
        g.take().unwrap()
    }
}

pub struct LoaderScopedHandle<'a, R: Send + 'static> {
    pair: Arc<(Mutex<Option<std::thread::Result<R>>>, Condvar)>,
    waited: bool,
    _lt: std::marker::PhantomData<&'a ()>,
}

impl<'a, R: Send + 'static> LoaderScopedHandle<'a, R> {
    pub fn wait(mut self) -> R {
        self.waited = true;
        let mut g = self.pair.0.lock().unwrap();
        while g.is_none() {
            g = self.pair.1.wait(g).unwrap();
        }
        match g.take().unwrap() {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

impl<R: Send + 'static> Drop for LoaderScopedHandle<'_, R> {
    fn drop(&mut self) {
        if !self.waited {
            let mut g = self.pair.0.lock().unwrap();
            while g.is_none() {
                g = self.pair.1.wait(g).unwrap();
            }
        }
    }
}

// ============================================================================
// In-Memory Fault-Injecting TestMember for Gate 2 & Gate 3 (Section 15)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestFault {
    None,
    FailNextRead,
    FailNextWrite,
    DropNextWrite,
    MisplaceNextWrite(u64),
    Timeout,
    StopAnswering,
}

pub struct TestMember {
    id: String,
    kind: MemberKind,
    data: Mutex<Vec<u8>>,
    alloc: Mutex<MemberSpanAlloc>,
    fault: Mutex<TestFault>,
}

impl TestMember {
    pub fn new(id: impl Into<String>, kind: MemberKind, bytes: usize) -> Self {
        Self {
            id: id.into(),
            kind,
            data: Mutex::new(vec![0u8; bytes]),
            alloc: Mutex::new(MemberSpanAlloc::new()),
            fault: Mutex::new(TestFault::None),
        }
    }

    pub fn inject_fault(&self, f: TestFault) {
        *self.fault.lock().unwrap() = f;
    }

    pub fn live_allocated_bytes(&self) -> u64 {
        self.alloc.lock().unwrap().live_bytes()
    }

    pub fn corrupt_u64_at(&self, byte_offset: usize, xor_mask: u64) {
        let mut buf = self.data.lock().unwrap();
        if byte_offset + 8 <= buf.len() {
            let cur = u64::from_le_bytes(buf[byte_offset..(byte_offset + 8)].try_into().unwrap());
            buf[byte_offset..(byte_offset + 8)].copy_from_slice(&(cur ^ xor_mask).to_le_bytes());
        }
    }
}

impl ExtentMember for TestMember {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> MemberKind {
        self.kind
    }
    fn alignment(&self) -> u64 {
        4096
    }
    fn capacity(&self) -> u64 {
        self.data.lock().unwrap().len() as u64
    }
    fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
        let cap = self.capacity();
        self.alloc.lock().unwrap().allocate(bytes, 4096, cap)
    }
    fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
        {
            let mut flt = self.fault.lock().unwrap();
            match *flt {
                TestFault::StopAnswering | TestFault::Timeout => return Err(StoreError::DeviceLost),
                TestFault::FailNextRead => {
                    *flt = TestFault::None;
                    return Err(StoreError::IoFailed);
                }
                _ => {}
            }
        }
        let data = self.data.lock().unwrap();
        for r in ranges {
            let doff = r.device_offset as usize;
            if doff + r.len > data.len() || r.host_offset + r.len > host_buf.len() {
                return Err(StoreError::IoFailed);
            }
            host_buf[r.host_offset..(r.host_offset + r.len)]
                .copy_from_slice(&data[doff..(doff + r.len)]);
        }
        Ok(())
    }
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
        let mut misplace = 0u64;
        {
            let mut flt = self.fault.lock().unwrap();
            match *flt {
                TestFault::StopAnswering | TestFault::Timeout => return Err(StoreError::DeviceLost),
                TestFault::FailNextWrite => {
                    *flt = TestFault::None;
                    return Err(StoreError::IoFailed);
                }
                TestFault::DropNextWrite => {
                    *flt = TestFault::None;
                    return Ok(());
                }
                TestFault::MisplaceNextWrite(delta) => {
                    *flt = TestFault::None;
                    misplace = delta;
                }
                TestFault::None | TestFault::FailNextRead => {}
            }
        }
        let mut data = self.data.lock().unwrap();
        for r in ranges {
            let doff =
                ((r.device_offset + misplace) as usize) % data.len().saturating_sub(r.len).max(1);
            if doff + r.len > data.len() || r.host_offset + r.len > host_buf.len() {
                return Err(StoreError::IoFailed);
            }
            data[doff..(doff + r.len)]
                .copy_from_slice(&host_buf[r.host_offset..(r.host_offset + r.len)]);
        }
        Ok(())
    }
    fn free(&self, device_offset: u64, bytes: u64) {
        self.alloc.lock().unwrap().free(device_offset, bytes, 4096);
    }
}
