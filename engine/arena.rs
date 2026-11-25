//! Contract:
//! - Inputs: Memory backing provider (`RegionMapper`) or page-aligned fallback, worker topology
//!   (`(worker_idx, numa_node, cache_share_bytes)`), and capacity budgets from `Shares` and `Plan`
//!   for `numbers`, per-worker `scratch`, and shared `transforms`.
//! - Outputs: `ArenaSet` (`std::alloc::Allocator` arenas for `numbers`, each worker's `scratch`,
//!   and shared `transforms`, plus `Copy` allocator handles `NumbersAlloc`, `ScratchAlloc`, and
//!   `TransformsAlloc` for `Vec<T, A>` and `Box<T, A>`), NUMA-local worker first-touch, and
//!   combined `MappingReport` + `ArenaStats`.
//! - Data format: 64-byte cache-line aligned LIFO/coalescing sub-allocations in per-worker slabs
//!   and whole-page (4096-byte) best-fit coalescing blocks in plan-assigned NUMA `Region`s.
//! - Work & bytes moved: O(1) LIFO bump/pop in worker subtree and scratch regions; O(log blocks)
//!   best-fit neighbor-coalescing in plan-assigned `Region`s so committed pages fault once per run.

use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappingReport {
    pub huge_pages_used: bool,
    pub huge_pages_refused: bool,
    pub pinned_used: bool,
    pub pin_refused: bool,
}

impl MappingReport {
    #[inline]
    pub fn merge(&mut self, other: Self) {
        self.huge_pages_used |= other.huge_pages_used;
        self.huge_pages_refused |= other.huge_pages_refused;
        self.pinned_used |= other.pinned_used;
        self.pin_refused |= other.pin_refused;
    }
}

/// OS-agnostic trait handed in from `platform` so `engine` never names an OS.
pub trait RegionMapper: Send + Sync {
    /// Reserve `bytes` of virtual address space near `numa_node`.
    fn reserve_region(
        &self,
        bytes: usize,
        numa_node: usize,
    ) -> Option<(*mut u8, usize, MappingReport)>;
    /// Commit and optionally pin `[ptr .. ptr + len)` on `numa_node` when first needed.
    fn commit_range(
        &self,
        ptr: *mut u8,
        len: usize,
        numa_node: usize,
    ) -> Result<MappingReport, ()>;
    /// Return physical backing for `[ptr .. ptr + len)` while keeping the virtual range reserved.
    unsafe fn decommit_range(&self, _ptr: *mut u8, _len: usize) {}
    /// Release a previously reserved region.
    unsafe fn release_region(&self, base: *mut u8, reserved_len: usize);
}

/// Fallback page-aligned mapper used when `EngineConfig.region_mapper` is `None` (e.g. synthetic
/// unit/gate tests), so all three `Allocator` arenas still run identically without OS hooks.
struct SystemRegionMapper;

impl RegionMapper for SystemRegionMapper {
    fn reserve_region(
        &self,
        bytes: usize,
        _numa_node: usize,
    ) -> Option<(*mut u8, usize, MappingReport)> {
        let len = bytes.div_ceil(REGION_PAGE) * REGION_PAGE;
        if len == 0 {
            return None;
        }
        let layout = Layout::from_size_align(len, REGION_PAGE).ok()?;
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        if p.is_null() {
            None
        } else {
            Some((
                p,
                len,
                MappingReport {
                    huge_pages_refused: true,
                    pin_refused: true,
                    ..MappingReport::default()
                },
            ))
        }
    }

    fn commit_range(
        &self,
        _ptr: *mut u8,
        _len: usize,
        _numa_node: usize,
    ) -> Result<MappingReport, ()> {
        Ok(MappingReport {
            huge_pages_refused: true,
            pin_refused: true,
            ..MappingReport::default()
        })
    }

    unsafe fn release_region(&self, base: *mut u8, reserved_len: usize) {
        if !base.is_null() && reserved_len > 0 {
            if let Ok(layout) = Layout::from_size_align(reserved_len, REGION_PAGE) {
                unsafe { std::alloc::dealloc(base, layout) };
            }
        }
    }
}

static SYSTEM_REGION_MAPPER: SystemRegionMapper = SystemRegionMapper;

/// Blocks of at least this many bytes in `Numbers` (above subtree depth) and `Transforms` come
/// from whole-page `Region`s; smaller blocks come from per-worker 64-byte LIFO/coalescing slabs.
pub const REGION_MIN_BLOCK: usize = 64 * 1024;
/// Blocks of at least this many bytes in a `Region` are cut from the high end downward, smaller
/// ones from its start upward so transient multi-megabyte workspaces never split smaller blocks.
pub const REGION_LARGE_BLOCK: usize = 8 * 1024 * 1024;
const REGION_PAGE: usize = 4096;
/// Unused middle of a `Region` is committed in steps of up to 64 MiB.
const REGION_COMMIT_PAGES: u64 = ((64 * 1024 * 1024) / REGION_PAGE) as u64;
/// Minimum alignment (one 64-byte cache line) for all arena allocations.
const ALIGN_FLOOR: usize = 64;
/// Per-worker `LifoSlab` commits backing memory in 2 MiB steps.
const LIFO_COMMIT_LINES: u64 = ((2 * 1024 * 1024) / ALIGN_FLOOR) as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaKind {
    Numbers,
    Scratch,
    Transforms,
}

/// Summary of peak and live byte usage across the three Section 10 arenas.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaUsageStats {
    pub numbers_live_bytes: u64,
    pub numbers_peak_bytes: u64,
    pub numbers_subtree_lifo_peak_bytes: u64,
    pub numbers_region_peak_bytes: u64,
    pub scratch_live_bytes: u64,
    pub scratch_peak_bytes: u64,
    pub transforms_live_bytes: u64,
    pub transforms_peak_bytes: u64,
}

/// The free blocks of one zone, by address (for joining neighbours) and by size (for best fit),
/// indexed by 64-bit unit counts so regions and slabs scale to the full 2^64 reference machine.
#[derive(Default)]
struct FreeList {
    /// First unit -> unit count. No two blocks are adjacent.
    by_addr: BTreeMap<u64, u64>,
    /// The same blocks as `(unit count, first unit)`.
    by_size: BTreeSet<(u64, u64)>,
}

impl FreeList {
    fn insert(&mut self, start: u64, len: u64) {
        self.by_addr.insert(start, len);
        self.by_size.insert((len, start));
    }

    fn remove(&mut self, start: u64, len: u64) {
        self.by_addr.remove(&start);
        self.by_size.remove(&(len, start));
    }

    /// Take `need` units from the smallest block that has them; the rest of it stays free.
    fn take_best_fit(&mut self, need: u64) -> Option<u64> {
        let &(len, start) = self.by_size.range((need, 0)..).next()?;
        self.remove(start, len);
        if len > need {
            self.insert(start + need, len - need);
        }
        Some(start)
    }

    /// Take `need` pages from the smallest block that has a backed `[start, start + need)` head
    /// or `[start + len - need, start + len)` tail.
    fn take_best_fit_backed(&mut self, need: u64, unbacked: &BTreeMap<u64, u64>) -> Option<u64> {
        for &(len, st) in self.by_size.range((need, 0)..) {
            if !Region::has_unbacked(unbacked, st, st + need) {
                self.remove(st, len);
                if len > need {
                    self.insert(st + need, len - need);
                }
                return Some(st);
            }
            if len > need && !Region::has_unbacked(unbacked, st + len - need, st + len) {
                self.remove(st, len);
                self.insert(st, len - need);
                return Some(st + len - need);
            }
        }
        None
    }

    /// Join `[start, start + len)` with the free blocks touching it, which leave the list, and
    /// return the joined stretch.
    fn join(&mut self, mut start: u64, mut len: u64, with_below: bool) -> (u64, u64) {
        if with_below {
            if let Some((&prev, &prev_len)) = self.by_addr.range(..start).next_back() {
                if prev + prev_len == start {
                    self.remove(prev, prev_len);
                    start = prev;
                    len += prev_len;
                }
            }
        }
        if let Some(&next_len) = self.by_addr.get(&(start + len)) {
            self.remove(start + len, next_len);
            len += next_len;
        }
        (start, len)
    }

    /// Extend a live block ending at `end` by `extra` units out of the free block starting there.
    fn take_at(&mut self, end: u64, extra: u64) -> bool {
        match self.by_addr.get(&end).copied() {
            Some(next_len) if next_len >= extra => {
                self.remove(end, next_len);
                if next_len > extra {
                    self.insert(end + extra, next_len - extra);
                }
                true
            }
            _ => false,
        }
    }
}

/// Per-worker 64-byte cache-line LIFO slab with O(1) bump/pop on LIFO discipline and neighbor
/// coalescing when a recursive merge allocates its parent before dropping its two children.
struct LifoSlab {
    base: usize,
    total_lines: u64,
    committed_lines: u64,
    head_lines: u64,
    touched_lines: u64,
    numa_node: usize,
    mapper: &'static dyn RegionMapper,
    free: FreeList,
    live_lines: u64,
    peak_live_lines: u64,
    report: MappingReport,
}

impl LifoSlab {
    fn new(mapper: &'static dyn RegionMapper, bytes: usize, numa_node: usize) -> Option<Self> {
        if bytes < REGION_PAGE {
            return None;
        }
        let (ptr, len, report) = mapper.reserve_region(bytes, numa_node)?;
        if ptr.is_null() || !(ptr as usize).is_multiple_of(ALIGN_FLOOR) {
            if !ptr.is_null() {
                unsafe { mapper.release_region(ptr, len) };
            }
            return None;
        }
        let total_lines = (len / ALIGN_FLOOR) as u64;
        Some(Self {
            base: ptr as usize,
            total_lines,
            committed_lines: 0,
            head_lines: 0,
            touched_lines: 0,
            numa_node,
            mapper,
            free: FreeList::default(),
            live_lines: 0,
            peak_live_lines: 0,
            report,
        })
    }

    fn ensure_committed(&mut self, needed_lines: u64) -> bool {
        if needed_lines <= self.committed_lines {
            return true;
        }
        if needed_lines > self.total_lines {
            return false;
        }
        let step = LIFO_COMMIT_LINES.max(1);
        let target = needed_lines
            .div_ceil(step)
            .saturating_mul(step)
            .min(self.total_lines);
        let from_bytes = (self.committed_lines as usize) * ALIGN_FLOOR;
        let commit_bytes = ((target - self.committed_lines) as usize) * ALIGN_FLOOR;
        let ptr = (self.base + from_bytes) as *mut u8;
        match self.mapper.commit_range(ptr, commit_bytes, self.numa_node) {
            Ok(rep) => {
                self.report.merge(rep);
                self.committed_lines = target;
                true
            }
            Err(()) => false,
        }
    }

    /// Commit and write-touch `[0 .. touch_bytes)` on the calling worker thread so physical pages
    /// are faulted onto that worker's local NUMA node before compute begins.
    fn first_touch(&mut self, touch_bytes: usize) {
        let want_lines = (touch_bytes.div_ceil(ALIGN_FLOOR) as u64).min(self.total_lines);
        if want_lines == 0 || !self.ensure_committed(want_lines) {
            return;
        }
        let end_bytes = (want_lines as usize) * ALIGN_FLOOR;
        let mut off = (self.touched_lines as usize) * ALIGN_FLOOR;
        while off < end_bytes {
            unsafe {
                std::ptr::write_volatile((self.base + off) as *mut u8, 0);
            }
            off += REGION_PAGE;
        }
        self.touched_lines = self.touched_lines.max(want_lines);
    }

    /// Allocate at least `bytes` (aligned to `align <= 4096`), returning `(ptr, cap_bytes, virgin)`.
    fn take(&mut self, bytes: usize, align: usize) -> Option<(*mut u8, usize, bool)> {
        if bytes == 0 || align > REGION_PAGE {
            return None;
        }
        let need = bytes.div_ceil(ALIGN_FLOOR) as u64;
        if need == 0 || need > self.total_lines {
            return None;
        }
        if align <= ALIGN_FLOOR {
            if let Some(start) = self.free.take_best_fit(need) {
                self.live_lines += need;
                self.peak_live_lines = self.peak_live_lines.max(self.live_lines);
                let ptr = (self.base + (start as usize) * ALIGN_FLOOR) as *mut u8;
                return Some((ptr, (need as usize) * ALIGN_FLOOR, false));
            }
        }
        let align_lines = (align.max(ALIGN_FLOOR) / ALIGN_FLOOR) as u64;
        let aligned_start = if align_lines <= 1 {
            self.head_lines
        } else {
            self.head_lines.div_ceil(align_lines) * align_lines
        };
        let new_head = aligned_start.checked_add(need)?;
        if new_head > self.total_lines || !self.ensure_committed(new_head) {
            return None;
        }
        if aligned_start > self.head_lines {
            let pad = aligned_start - self.head_lines;
            let (js, jl) = self.free.join(self.head_lines, pad, true);
            self.free.insert(js, jl);
        }
        let virgin = aligned_start >= self.touched_lines;
        self.head_lines = new_head;
        self.touched_lines = self.touched_lines.max(new_head);
        self.live_lines += need;
        self.peak_live_lines = self.peak_live_lines.max(self.live_lines);
        let ptr = (self.base + (aligned_start as usize) * ALIGN_FLOOR) as *mut u8;
        Some((ptr, (need as usize) * ALIGN_FLOOR, virgin))
    }

    unsafe fn give(&mut self, ptr: *mut u8, bytes: usize) {
        let start = ((ptr as usize - self.base) / ALIGN_FLOOR) as u64;
        let len = bytes.div_ceil(ALIGN_FLOOR) as u64;
        self.live_lines = self.live_lines.saturating_sub(len);
        let (j_start, j_len) = self.free.join(start, len, true);
        if j_start + j_len == self.head_lines {
            self.head_lines = j_start;
        } else {
            self.free.insert(j_start, j_len);
        }
    }

    unsafe fn resize(&mut self, ptr: *mut u8, old_bytes: usize, new_bytes: usize) -> Option<usize> {
        let start = ((ptr as usize - self.base) / ALIGN_FLOOR) as u64;
        let old_l = old_bytes.div_ceil(ALIGN_FLOOR) as u64;
        let new_l = new_bytes.div_ceil(ALIGN_FLOOR) as u64;
        if new_l == 0 || new_l > self.total_lines {
            return None;
        }
        if new_l == old_l {
            return Some((new_l as usize) * ALIGN_FLOOR);
        }
        if new_l < old_l {
            let rem = old_l - new_l;
            self.live_lines = self.live_lines.saturating_sub(rem);
            let (js, jl) = self.free.join(start + new_l, rem, false);
            if js + jl == self.head_lines {
                self.head_lines = js;
            } else {
                self.free.insert(js, jl);
            }
            return Some((new_l as usize) * ALIGN_FLOOR);
        }
        let extra = new_l - old_l;
        let end = start + old_l;
        if end == self.head_lines {
            let new_head = end.checked_add(extra)?;
            if !self.ensure_committed(new_head) {
                return None;
            }
            self.head_lines = new_head;
            self.touched_lines = self.touched_lines.max(new_head);
            self.live_lines += extra;
            self.peak_live_lines = self.peak_live_lines.max(self.live_lines);
            Some((new_l as usize) * ALIGN_FLOOR)
        } else if self.free.take_at(end, extra) {
            self.live_lines += extra;
            self.peak_live_lines = self.peak_live_lines.max(self.live_lines);
            Some((new_l as usize) * ALIGN_FLOOR)
        } else {
            None
        }
    }
}

impl Drop for LifoSlab {
    fn drop(&mut self) {
        if self.base != 0 && self.total_lines > 0 {
            unsafe {
                self.mapper.release_region(
                    self.base as *mut u8,
                    (self.total_lines as usize) * ALIGN_FLOOR,
                );
            }
            self.base = 0;
            self.total_lines = 0;
        }
    }
}

// ---- Whole-page best-fit coalescing Region ---------------------------------------------------

struct RegionState {
    /// End of the low zone, where the small blocks are: pages `[0, top)` are live or in `low`.
    top: u64,
    /// Highest `top` reached. Pages at or above it were never in the low zone.
    touched: u64,
    low: FreeList,
    /// Start of the high zone, where the large blocks are: pages `[bottom, pages)` are live or
    /// in `high`.
    bottom: u64,
    /// Lowest `bottom` reached.
    touched_from: u64,
    high: FreeList,
    /// Disjoint unbacked page intervals `start -> len` (initially `0 -> pages`).
    unbacked: BTreeMap<u64, u64>,
    /// Exact count of backed pages (`pages - sum(unbacked)`).
    backed_pages: u64,
    /// Soft ceiling on `backed_pages`.
    soft_cap_pages: u64,
    had_decommit: bool,
    live_pages: u64,
    peak_live_pages: u64,
    /// Requests the range could not serve.
    refused: u64,
    report: MappingReport,
}

/// Counters of a region, in bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegionStats {
    pub reserved: usize,
    pub soft_cap: usize,
    pub committed: usize,
    pub touched: usize,
    pub live: usize,
    pub peak_live: usize,
    pub free_blocks: usize,
    pub refused: u64,
    pub report: MappingReport,
}

/// One reserved range of whole pages on a specific `numa_node` with two zones that grow toward
/// each other: small blocks from the start, large blocks from the end.
pub struct Region {
    base: usize,
    pages: u64,
    large_pages: u64,
    numa_node: usize,
    mapper: &'static dyn RegionMapper,
    shared_backed: Option<Arc<AtomicU64>>,
    state: Mutex<RegionState>,
}

impl Region {
    /// Reserve `bytes` through `mapper` on NUMA node `0`.
    pub fn new(mapper: &'static dyn RegionMapper, bytes: usize) -> Option<Self> {
        Self::with_large_block(mapper, bytes, REGION_LARGE_BLOCK)
    }

    /// As `new`, with the size from which a block counts as large.
    pub fn with_large_block(
        mapper: &'static dyn RegionMapper,
        bytes: usize,
        large_block: usize,
    ) -> Option<Self> {
        Self::with_large_block_on_node(mapper, bytes, large_block, 0, None)
    }

    /// Reserve `bytes` on `numa_node`, optionally sharing a physical-page counter across arenas.
    pub fn with_large_block_on_node(
        mapper: &'static dyn RegionMapper,
        bytes: usize,
        large_block: usize,
        numa_node: usize,
        shared_backed: Option<Arc<AtomicU64>>,
    ) -> Option<Self> {
        if bytes < REGION_MIN_BLOCK {
            return None;
        }
        let (base, len, report) = mapper.reserve_region(bytes, numa_node)?;
        if base.is_null() || !(base as usize).is_multiple_of(REGION_PAGE) {
            if !base.is_null() {
                unsafe { mapper.release_region(base, len) };
            }
            return None;
        }
        let pages = (len / REGION_PAGE) as u64;
        let mut unbacked = BTreeMap::new();
        if pages > 0 {
            unbacked.insert(0, pages);
        }
        Some(Self {
            base: base as usize,
            pages,
            large_pages: (Self::pages_for(large_block) as u64).max(1),
            numa_node,
            mapper,
            shared_backed,
            state: Mutex::new(RegionState {
                top: 0,
                touched: 0,
                low: FreeList::default(),
                bottom: pages,
                touched_from: pages,
                high: FreeList::default(),
                unbacked,
                backed_pages: 0,
                soft_cap_pages: pages,
                had_decommit: false,
                live_pages: 0,
                peak_live_pages: 0,
                refused: 0,
                report,
            }),
        })
    }

    #[inline]
    fn lock(&self) -> MutexGuard<'_, RegionState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        let a = ptr as usize;
        a >= self.base && a - self.base < (self.pages as usize) * REGION_PAGE
    }

    #[inline]
    fn page_of(&self, ptr: *const u8) -> u64 {
        ((ptr as usize - self.base) / REGION_PAGE) as u64
    }

    #[inline]
    fn pages_for(bytes: usize) -> usize {
        bytes.div_ceil(REGION_PAGE)
    }

    /// Commit and first-touch up to `touch_bytes` from the low end of this `Region` on the
    /// calling worker thread so pages fault onto `self.numa_node`.
    pub fn first_touch_initial(&self, touch_bytes: usize) {
        let want_pages = (touch_bytes.div_ceil(REGION_PAGE) as u64).min(self.pages);
        if want_pages == 0 {
            return;
        }
        let mut s = self.lock();
        if self.commit_unbacked_in(&mut s, 0, want_pages).is_err() {
            return;
        }
        s.touched = s.touched.max(want_pages);
        let base = self.base;
        drop(s);
        let end_bytes = (want_pages as usize) * REGION_PAGE;
        let mut off = 0usize;
        while off < end_bytes {
            unsafe {
                std::ptr::write_volatile((base + off) as *mut u8, 0);
            }
            off += REGION_PAGE;
        }
    }

    /// Commit `[0, low_bytes)` at the low end and `[size - high_bytes, size)` at the high end,
    /// mark them touched, and return the newly committed `(addr, byte_len)` spans for parallel
    /// worker page-touching on `self.numa_node`.
    pub fn prepare_parallel_touch(
        &self,
        low_bytes: usize,
        high_bytes: usize,
    ) -> [(usize, usize); 2] {
        let low_p = (low_bytes.div_ceil(REGION_PAGE) as u64).min(self.pages);
        let max_high_p = self.pages.saturating_sub(low_p);
        let high_p = (high_bytes.div_ceil(REGION_PAGE) as u64).min(max_high_p);
        let high_from = self.pages - high_p;
        let mut s = self.lock();
        let mut r0 = (0usize, 0usize);
        let mut r1 = (0usize, 0usize);
        if low_p > s.touched && self.commit_unbacked_in(&mut s, 0, low_p).is_ok() {
            let old_t = s.touched;
            s.touched = low_p;
            r0 = (
                self.base + (old_t as usize) * REGION_PAGE,
                ((low_p - old_t) as usize) * REGION_PAGE,
            );
        }
        if high_from < s.touched_from
            && self.commit_unbacked_in(&mut s, high_from, self.pages).is_ok()
        {
            let old_tf = s.touched_from;
            s.touched_from = high_from;
            r1 = (
                self.base + (high_from as usize) * REGION_PAGE,
                ((old_tf - high_from) as usize) * REGION_PAGE,
            );
        }
        [r0, r1]
    }

    /// Commit pages `[from, to)` on `self.numa_node`; false when the mapper refuses.
    fn commit(&self, s: &mut RegionState, from: u64, to: u64) -> bool {
        let ptr = (self.base + (from as usize) * REGION_PAGE) as *mut u8;
        match self
            .mapper
            .commit_range(ptr, ((to - from) as usize) * REGION_PAGE, self.numa_node)
        {
            Ok(rep) => {
                s.report.merge(rep);
                true
            }
            Err(()) => false,
        }
    }

    #[inline]
    fn has_unbacked(unbacked: &BTreeMap<u64, u64>, from: u64, to: u64) -> bool {
        if from >= to {
            return false;
        }
        if let Some((&u_s, &u_l)) = unbacked.range(..=from).next_back()
            && u_s + u_l > from
        {
            return true;
        }
        unbacked.range((from + 1)..to).next().is_some()
    }

    fn decommit_span(&self, s: &mut RegionState, from: u64, to: u64) {
        if from >= to || s.report.huge_pages_used {
            return;
        }
        let mut overlaps: Vec<(u64, u64)> = Vec::new();
        if let Some((&u_s, &u_l)) = s.unbacked.range(..=from).next_back()
            && u_s + u_l >= from
        {
            overlaps.push((u_s, u_l));
        }
        for (&u_s, &u_l) in s.unbacked.range((from + 1)..=to) {
            overlaps.push((u_s, u_l));
        }
        let mut cur = from;
        let mut freed_total = 0u64;
        for &(u_s, u_l) in &overlaps {
            if u_s > cur {
                let gap_end = u_s.min(to);
                let ptr = (self.base + (cur as usize) * REGION_PAGE) as *mut u8;
                unsafe {
                    self.mapper
                        .decommit_range(ptr, ((gap_end - cur) as usize) * REGION_PAGE);
                }
                let delta = gap_end - cur;
                s.backed_pages = s.backed_pages.saturating_sub(delta);
                freed_total += delta;
                s.had_decommit = true;
            }
            cur = cur.max(u_s + u_l);
        }
        if cur < to {
            let ptr = (self.base + (cur as usize) * REGION_PAGE) as *mut u8;
            unsafe {
                self.mapper
                    .decommit_range(ptr, ((to - cur) as usize) * REGION_PAGE);
            }
            let delta = to - cur;
            s.backed_pages = s.backed_pages.saturating_sub(delta);
            freed_total += delta;
            s.had_decommit = true;
        }
        if freed_total > 0
            && let Some(sh) = &self.shared_backed
        {
            let _ = sh.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(freed_total))
            });
        }
        for &(u_s, _) in &overlaps {
            s.unbacked.remove(&u_s);
        }
        let merged_from = overlaps.first().map_or(from, |&(u_s, _)| u_s.min(from));
        let merged_to = overlaps
            .last()
            .map_or(to, |&(u_s, u_l)| (u_s + u_l).max(to));
        s.unbacked.insert(merged_from, merged_to - merged_from);
    }

    fn reclaim_free_to_cap(
        &self,
        s: &mut RegionState,
        target_cap: u64,
        prot_from: u64,
        prot_to: u64,
    ) {
        if s.backed_pages <= target_cap || s.report.huge_pages_used {
            return;
        }
        let decommit_avoiding_prot = |s: &mut RegionState, a: u64, b: u64| {
            if a >= b {
                return;
            }
            if prot_from >= prot_to || b <= prot_from || a >= prot_to {
                self.decommit_span(s, a, b);
            } else {
                if a < prot_from {
                    self.decommit_span(s, a, prot_from);
                }
                if prot_to < b && s.backed_pages > target_cap {
                    self.decommit_span(s, prot_to, b);
                }
            }
        };
        if s.top < s.bottom {
            let (top, bottom) = (s.top, s.bottom);
            decommit_avoiding_prot(s, top, bottom);
        }
        if s.backed_pages <= target_cap {
            return;
        }
        let mut cands: Vec<(u64, u64)> = s
            .low
            .by_size
            .iter()
            .rev()
            .chain(s.high.by_size.iter().rev())
            .filter(|&&(len, _)| len >= 64)
            .map(|&(len, st)| (st, st + len))
            .collect();
        cands.sort_by(|x, y| (y.1 - y.0).cmp(&(x.1 - x.0)));
        for (a, b) in cands {
            if s.backed_pages <= target_cap {
                break;
            }
            decommit_avoiding_prot(s, a, b);
        }
    }

    fn commit_unbacked_in(
        &self,
        s: &mut RegionState,
        from: u64,
        to: u64,
    ) -> Result<bool, ()> {
        if from >= to {
            return Ok(false);
        }
        let mut pieces: Vec<(u64, u64, u64, u64)> = Vec::new();
        if let Some((&u_s, &u_l)) = s.unbacked.range(..=from).next_back()
            && u_s + u_l > from
        {
            pieces.push((u_s, u_l, from, (u_s + u_l).min(to)));
        }
        for (&u_s, &u_l) in s.unbacked.range((from + 1)..to) {
            pieces.push((u_s, u_l, u_s, (u_s + u_l).min(to)));
        }
        if pieces.is_empty() {
            return Ok(false);
        }
        let need_new: u64 = pieces.iter().map(|&(_, _, c_s, c_e)| c_e - c_s).sum();
        let cap = s.soft_cap_pages;
        if s.backed_pages + need_new > cap {
            self.reclaim_free_to_cap(s, cap.saturating_sub(need_new), from, to);
        }
        for (u_s, u_l, c_s, c_e) in pieces {
            if !self.commit(s, c_s, c_e) {
                return Err(());
            }
            s.unbacked.remove(&u_s);
            let u_e = u_s + u_l;
            if u_s < c_s {
                s.unbacked.insert(u_s, c_s - u_s);
            }
            if c_e < u_e {
                s.unbacked.insert(c_e, u_e - c_e);
            }
            let delta = c_e - c_s;
            s.backed_pages += delta;
            if let Some(sh) = &self.shared_backed {
                sh.fetch_add(delta, Ordering::Relaxed);
            }
        }
        Ok(true)
    }

    fn raise_top(&self, s: &mut RegionState, new_top: u64) -> Option<bool> {
        if new_top > s.bottom {
            return None;
        }
        let step_end = if s.soft_cap_pages < self.pages {
            new_top
        } else {
            new_top
                .div_ceil(REGION_COMMIT_PAGES)
                .saturating_mul(REGION_COMMIT_PAGES)
                .min(s.bottom)
        };
        let committed = self.commit_unbacked_in(s, s.top, step_end).ok()?;
        s.top = new_top;
        if new_top > s.touched {
            s.touched = new_top;
        }
        Some(committed)
    }

    fn lower_bottom(&self, s: &mut RegionState, new_bottom: u64) -> Option<bool> {
        if new_bottom < s.top {
            return None;
        }
        let step_start = if s.soft_cap_pages < self.pages {
            new_bottom
        } else {
            (new_bottom / REGION_COMMIT_PAGES)
                .saturating_mul(REGION_COMMIT_PAGES)
                .max(s.top)
        };
        let committed = self.commit_unbacked_in(s, step_start, s.bottom).ok()?;
        s.bottom = new_bottom;
        if new_bottom < s.touched_from {
            s.touched_from = new_bottom;
        }
        Some(committed)
    }

    /// Take a block of at least `bytes` bytes (`(*mut u8, is_os_zero, needs_fault)`).
    pub fn take_ext(&self, bytes: usize) -> Option<(*mut u8, bool, bool)> {
        let need = Self::pages_for(bytes) as u64;
        if need == 0 {
            return None;
        }
        let mut s = self.lock();
        if need > self.pages {
            s.refused += 1;
            return None;
        }
        let large = need >= self.large_pages;
        let had_decomm = s.had_decommit;
        let found = if large {
            {
                let sr = &mut *s;
                sr.high
                    .take_best_fit_backed(need, &sr.unbacked)
                    .map(|start| (start, false, false))
            }
            .or_else(|| {
                let sr = &mut *s;
                sr.low
                    .take_best_fit_backed(need, &sr.unbacked)
                    .map(|start| (start, false, false))
            })
            .or_else(|| {
                let new_bottom = s.bottom.checked_sub(need)?;
                if new_bottom >= s.top
                    && new_bottom >= s.touched_from
                    && !Self::has_unbacked(&s.unbacked, new_bottom, s.bottom)
                {
                    s.bottom = new_bottom;
                    Some((new_bottom, false, false))
                } else {
                    None
                }
            })
            .or_else(|| {
                let start = s.top;
                let new_top = start.checked_add(need)?;
                if new_top <= s.touched
                    && new_top <= s.bottom
                    && !Self::has_unbacked(&s.unbacked, start, new_top)
                {
                    s.top = new_top;
                    Some((start, false, false))
                } else {
                    None
                }
            })
            .or_else(|| {
                let start = s.high.take_best_fit(need)?;
                match self.commit_unbacked_in(&mut s, start, start + need) {
                    Ok(fault) => Some((start, false, fault)),
                    Err(()) => {
                        s.high.insert(start, need);
                        None
                    }
                }
            })
            .or_else(|| {
                let start = s.low.take_best_fit(need)?;
                match self.commit_unbacked_in(&mut s, start, start + need) {
                    Ok(fault) => Some((start, false, fault)),
                    Err(()) => {
                        s.low.insert(start, need);
                        None
                    }
                }
            })
            .or_else(|| {
                let new_bottom = s.bottom.checked_sub(need)?;
                if new_bottom >= s.top && new_bottom >= s.touched_from {
                    let fault = self.lower_bottom(&mut s, new_bottom)?;
                    Some((new_bottom, false, fault))
                } else {
                    None
                }
            })
            .or_else(|| {
                let end = s.bottom;
                let new_bottom = end.checked_sub(need)?;
                let zero = !had_decomm && end <= s.touched_from && new_bottom >= s.touched;
                let fault = self.lower_bottom(&mut s, new_bottom)?;
                Some((new_bottom, zero, fault || zero))
            })
        } else {
            {
                let sr = &mut *s;
                sr.low
                    .take_best_fit_backed(need, &sr.unbacked)
                    .map(|start| (start, false, false))
            }
            .or_else(|| {
                let start = s.top;
                let new_top = start.checked_add(need)?;
                if new_top <= s.touched
                    && new_top <= s.bottom
                    && !Self::has_unbacked(&s.unbacked, start, new_top)
                {
                    s.top = new_top;
                    Some((start, false, false))
                } else {
                    None
                }
            })
            .or_else(|| {
                let start = s.low.take_best_fit(need)?;
                match self.commit_unbacked_in(&mut s, start, start + need) {
                    Ok(fault) => Some((start, false, fault)),
                    Err(()) => {
                        s.low.insert(start, need);
                        None
                    }
                }
            })
            .or_else(|| {
                let start = s.top;
                let new_top = start.checked_add(need)?;
                let zero = !had_decomm && start >= s.touched && new_top <= s.touched_from;
                let fault = self.raise_top(&mut s, new_top)?;
                Some((start, zero, fault || zero))
            })
            .or_else(|| {
                let sr = &mut *s;
                sr.high
                    .take_best_fit_backed(need, &sr.unbacked)
                    .map(|start| (start, false, false))
            })
            .or_else(|| {
                let start = s.high.take_best_fit(need)?;
                match self.commit_unbacked_in(&mut s, start, start + need) {
                    Ok(fault) => Some((start, false, fault)),
                    Err(()) => {
                        s.high.insert(start, need);
                        None
                    }
                }
            })
        };
        let Some((start, zero, needs_fault)) = found else {
            s.refused += 1;
            return None;
        };
        s.live_pages += need;
        s.peak_live_pages = s.peak_live_pages.max(s.live_pages);
        Some((
            (self.base + (start as usize) * REGION_PAGE) as *mut u8,
            zero,
            needs_fault,
        ))
    }

    #[inline]
    pub fn take(&self, bytes: usize) -> Option<(*mut u8, bool)> {
        self.take_ext(bytes).map(|(p, z, _)| (p, z))
    }

    fn release(&self, s: &mut RegionState, start: u64, len: u64, with_below: bool) {
        if start >= s.bottom {
            let (start, len) = s.high.join(start, len, with_below);
            if start == s.bottom {
                s.bottom = start + len;
            } else {
                s.high.insert(start, len);
            }
        } else {
            let (start, len) = s.low.join(start, len, with_below);
            if start + len == s.top {
                s.top = start;
            } else {
                s.low.insert(start, len);
            }
        }
    }

    pub unsafe fn give(&self, ptr: *mut u8, bytes: usize) {
        let start = self.page_of(ptr);
        let len = Self::pages_for(bytes) as u64;
        let mut s = self.lock();
        s.live_pages = s.live_pages.saturating_sub(len);
        self.release(&mut s, start, len, true);
    }

    pub unsafe fn resize(&self, ptr: *mut u8, old: usize, new: usize) -> bool {
        let start = self.page_of(ptr);
        let old_p = Self::pages_for(old) as u64;
        let new_p = Self::pages_for(new) as u64;
        if new_p == 0 || new_p > self.pages {
            return false;
        }
        if new_p == old_p {
            return true;
        }
        let mut s = self.lock();
        if new_p < old_p {
            s.live_pages = s.live_pages.saturating_sub(old_p - new_p);
            self.release(&mut s, start + new_p, old_p - new_p, false);
            return true;
        }
        let extra = new_p - old_p;
        let end = start + old_p;
        let grown = if start >= s.bottom {
            if s.high.take_at(end, extra) {
                if self.commit_unbacked_in(&mut s, end, end + extra).is_err() {
                    s.high.insert(end, extra);
                    false
                } else {
                    true
                }
            } else {
                false
            }
        } else if end == s.top {
            end.checked_add(extra)
                .is_some_and(|t| self.raise_top(&mut s, t).is_some())
        } else if s.low.take_at(end, extra) {
            if self.commit_unbacked_in(&mut s, end, end + extra).is_err() {
                s.low.insert(end, extra);
                false
            } else {
                true
            }
        } else {
            false
        };
        if grown {
            s.live_pages += extra;
            s.peak_live_pages = s.peak_live_pages.max(s.live_pages);
        }
        grown
    }

    pub fn set_soft_cap(&self, cap_bytes: usize) {
        let mut s = self.lock();
        let cap_p = ((cap_bytes / REGION_PAGE) as u64).clamp(1, self.pages);
        s.soft_cap_pages = cap_p;
        self.reclaim_free_to_cap(&mut s, cap_p, 0, 0);
    }

    pub fn trim_excess_above(&self, keep_committed_bytes: usize) {
        let mut s = self.lock();
        let keep_pages = ((keep_committed_bytes / REGION_PAGE) as u64).min(self.pages);
        self.reclaim_free_to_cap(&mut s, keep_pages, 0, 0);
    }

    pub fn stats(&self) -> RegionStats {
        let s = self.lock();
        let bytes = |pages: u64| (pages as usize) * REGION_PAGE;
        RegionStats {
            reserved: bytes(self.pages),
            soft_cap: bytes(s.soft_cap_pages),
            committed: (s.backed_pages as usize) * REGION_PAGE,
            touched: bytes(s.touched.min(s.touched_from)) + bytes(self.pages - s.touched_from),
            live: (s.live_pages as usize) * REGION_PAGE,
            peak_live: (s.peak_live_pages as usize) * REGION_PAGE,
            free_blocks: s.low.by_addr.len() + s.high.by_addr.len(),
            refused: s.refused,
            report: s.report,
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe {
            self.mapper
                .release_region(self.base as *mut u8, (self.pages as usize) * REGION_PAGE)
        };
    }
}

// ---- Section 10 Three Standard-Allocator Arenas ---------------------------------------------

struct NodeRegionSlot {
    numa_node: usize,
    primary: Option<Region>,
    overflow_bytes: usize,
    mapper: &'static dyn RegionMapper,
    shared_backed: Option<Arc<AtomicU64>>,
    overflow: Mutex<Option<Region>>,
}

impl NodeRegionSlot {
    fn new(
        numa_node: usize,
        primary_bytes: usize,
        overflow_bytes: usize,
        mapper: &'static dyn RegionMapper,
        shared_backed: Option<Arc<AtomicU64>>,
    ) -> Self {
        let primary = Region::with_large_block_on_node(
            mapper,
            primary_bytes,
            REGION_LARGE_BLOCK,
            numa_node,
            shared_backed.clone(),
        );
        Self {
            numa_node,
            primary,
            overflow_bytes,
            mapper,
            shared_backed,
            overflow: Mutex::new(None),
        }
    }

    fn owns(&self, ptr: *const u8) -> bool {
        if self.primary.as_ref().is_some_and(|r| r.owns(ptr)) {
            return true;
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        ov.as_ref().is_some_and(|r| r.owns(ptr))
    }

    fn take_ext(&self, bytes: usize) -> Option<(*mut u8, usize, bool, bool)> {
        let cap_bytes = bytes.div_ceil(REGION_PAGE) * REGION_PAGE;
        if let Some(r) = &self.primary
            && let Some((p, z, nf)) = r.take_ext(bytes)
        {
            return Some((p, cap_bytes, z, nf));
        }
        if self.overflow_bytes >= REGION_MIN_BLOCK {
            let mut ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
            if ov.is_none() {
                let want = self.overflow_bytes.max(cap_bytes.saturating_mul(2));
                *ov = Region::with_large_block_on_node(
                    self.mapper,
                    want,
                    REGION_LARGE_BLOCK,
                    self.numa_node,
                    self.shared_backed.clone(),
                );
            }
            if let Some(r) = ov.as_ref()
                && let Some((p, z, nf)) = r.take_ext(bytes)
            {
                return Some((p, cap_bytes, z, nf));
            }
        }
        None
    }

    unsafe fn give(&self, ptr: *mut u8, bytes: usize) -> bool {
        if let Some(r) = &self.primary
            && r.owns(ptr)
        {
            unsafe { r.give(ptr, bytes) };
            return true;
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref()
            && r.owns(ptr)
        {
            unsafe { r.give(ptr, bytes) };
            return true;
        }
        false
    }

    unsafe fn resize(&self, ptr: *mut u8, old_bytes: usize, new_bytes: usize) -> Option<usize> {
        let new_cap = new_bytes.div_ceil(REGION_PAGE) * REGION_PAGE;
        if let Some(r) = &self.primary
            && r.owns(ptr)
        {
            return unsafe { r.resize(ptr, old_bytes, new_bytes) }.then_some(new_cap);
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref()
            && r.owns(ptr)
        {
            return unsafe { r.resize(ptr, old_bytes, new_bytes) }.then_some(new_cap);
        }
        None
    }

    fn first_touch_initial(&self, bytes: usize) {
        if let Some(r) = &self.primary {
            r.first_touch_initial(bytes);
        }
    }

    fn prepare_parallel_touch(&self, low_bytes: usize, high_bytes: usize) -> [(usize, usize); 2] {
        match &self.primary {
            Some(r) => r.prepare_parallel_touch(low_bytes, high_bytes),
            None => [(0, 0); 2],
        }
    }

    fn set_soft_cap(&self, cap_bytes: usize) {
        if let Some(r) = &self.primary {
            r.set_soft_cap(cap_bytes);
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref() {
            r.set_soft_cap(cap_bytes);
        }
    }

    fn trim_excess_above(&self, keep_bytes: usize) {
        if let Some(r) = &self.primary {
            r.trim_excess_above(keep_bytes);
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref() {
            r.trim_excess_above(0);
        }
    }

    fn report(&self) -> MappingReport {
        let mut rep = MappingReport::default();
        if let Some(r) = &self.primary {
            rep.merge(r.stats().report);
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref() {
            rep.merge(r.stats().report);
        }
        rep
    }

    fn stats_sum(&self) -> RegionStats {
        let mut acc = RegionStats::default();
        let add = |acc: &mut RegionStats, st: RegionStats| {
            acc.reserved = acc.reserved.saturating_add(st.reserved);
            acc.soft_cap = acc.soft_cap.saturating_add(st.soft_cap);
            acc.committed = acc.committed.saturating_add(st.committed);
            acc.touched = acc.touched.saturating_add(st.touched);
            acc.live = acc.live.saturating_add(st.live);
            acc.peak_live = acc.peak_live.saturating_add(st.peak_live);
            acc.free_blocks = acc.free_blocks.saturating_add(st.free_blocks);
            acc.refused = acc.refused.saturating_add(st.refused);
            acc.report.merge(st.report);
        };
        if let Some(r) = &self.primary {
            add(&mut acc, r.stats());
        }
        let ov = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ov.as_ref() {
            add(&mut acc, r.stats());
        }
        acc
    }
}

#[repr(align(64))]
struct WorkerSlabSlot {
    base: usize,
    cap_bytes: usize,
    slab: Mutex<Option<LifoSlab>>,
}

impl WorkerSlabSlot {
    fn new(slab: Option<LifoSlab>) -> Self {
        let (base, cap_bytes) = match &slab {
            Some(s) => (s.base, (s.total_lines as usize) * ALIGN_FLOOR),
            None => (0, 0),
        };
        Self {
            base,
            cap_bytes,
            slab: Mutex::new(slab),
        }
    }

    #[inline]
    fn owns(&self, ptr: *const u8) -> bool {
        let a = ptr as usize;
        self.cap_bytes != 0 && a >= self.base && a - self.base < self.cap_bytes
    }
}

/// One of the three Section 10 arenas (`numbers`, a worker's `scratch`, or shared `transforms`),
/// implementing `std::alloc::Allocator` directly for `Vec<T, &Arena>` and `Box<T, &Arena>`.
pub struct Arena {
    pub kind: ArenaKind,
    /// Cache-line-isolated per-worker LIFO slabs with lock-free `(base, cap_bytes)` bounds.
    worker_slabs: Vec<WorkerSlabSlot>,
    /// Plan-assigned NUMA-node `Region`s (`Numbers` above subtree depth, `Transforms` shared grids).
    node_regions: Vec<NodeRegionSlot>,
    /// Worker index -> index in `node_regions`.
    worker_node_slot: Vec<usize>,
    sys_bytes_in_use: AtomicUsize,
    sys_peak_bytes: AtomicUsize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Construct a standalone single-node `Arena` (also supports direct unit tests).
    pub fn new(
        kind: ArenaKind,
        reserve_bytes: usize,
        numa_node: usize,
        mapper: Option<&'static dyn RegionMapper>,
    ) -> Self {
        let m = mapper.unwrap_or(&SYSTEM_REGION_MAPPER);
        match kind {
            ArenaKind::Scratch => {
                let slab = LifoSlab::new(m, reserve_bytes.max(REGION_MIN_BLOCK), numa_node);
                Self {
                    kind,
                    worker_slabs: vec![WorkerSlabSlot::new(slab)],
                    node_regions: Vec::new(),
                    worker_node_slot: vec![0],
                    sys_bytes_in_use: AtomicUsize::new(0),
                    sys_peak_bytes: AtomicUsize::new(0),
                }
            }
            ArenaKind::Numbers | ArenaKind::Transforms => {
                let lifo_bytes = (reserve_bytes / 8).max(REGION_MIN_BLOCK);
                let reg_bytes = reserve_bytes.saturating_sub(lifo_bytes).max(REGION_MIN_BLOCK);
                let slab = LifoSlab::new(m, lifo_bytes, numa_node);
                let slot = NodeRegionSlot::new(numa_node, reg_bytes, reg_bytes, m, None);
                Self {
                    kind,
                    worker_slabs: vec![WorkerSlabSlot::new(slab)],
                    node_regions: vec![slot],
                    worker_node_slot: vec![0],
                    sys_bytes_in_use: AtomicUsize::new(0),
                    sys_peak_bytes: AtomicUsize::new(0),
                }
            }
        }
    }

    pub fn current_bytes(&self) -> usize {
        let mut sum = self.sys_bytes_in_use.load(Ordering::Relaxed);
        for ws in &self.worker_slabs {
            let g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_ref() {
                sum = sum.saturating_add((s.live_lines as usize) * ALIGN_FLOOR);
            }
        }
        for nr in &self.node_regions {
            sum = sum.saturating_add(nr.stats_sum().live);
        }
        sum
    }

    pub fn peak_bytes(&self) -> usize {
        let mut sum = self.sys_peak_bytes.load(Ordering::Relaxed);
        sum = sum.saturating_add(self.worker_lifo_peak_bytes() as usize);
        sum = sum.saturating_add(self.region_peak_bytes() as usize);
        sum
    }

    pub fn worker_lifo_peak_bytes(&self) -> u64 {
        let mut sum = 0u64;
        for ws in &self.worker_slabs {
            let g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_ref() {
                sum = sum.saturating_add(s.peak_live_lines * (ALIGN_FLOOR as u64));
            }
        }
        sum
    }

    pub fn region_peak_bytes(&self) -> u64 {
        let mut sum = 0u64;
        for nr in &self.node_regions {
            sum = sum.saturating_add(nr.stats_sum().peak_live as u64);
        }
        sum
    }

    pub fn report(&self) -> MappingReport {
        let mut rep = MappingReport::default();
        for ws in &self.worker_slabs {
            let g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_ref() {
                rep.merge(s.report);
            }
        }
        for nr in &self.node_regions {
            rep.merge(nr.report());
        }
        rep
    }

    pub fn alloc_vec<T>(&self, capacity: usize) -> Vec<T, &Arena> {
        let mut v = Vec::new_in(self);
        if capacity > 0 {
            v.reserve_exact(capacity);
        }
        v
    }

    pub fn alloc_box<T>(&self, value: T) -> Box<T, &Arena> {
        Box::new_in(value, self)
    }

    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        self.worker_slabs.iter().any(|ws| ws.owns(ptr))
            || self.node_regions.iter().any(|nr| nr.owns(ptr))
    }

    /// Core allocation returning `(NonNull<[u8]>, is_zero, needs_touch)`.
    #[inline]
    fn allocate_detailed(
        &self,
        layout: Layout,
        worker_hint: usize,
        in_subtree: bool,
    ) -> Result<(NonNull<[u8]>, bool, bool), AllocError> {
        if layout.size() == 0 {
            let align = layout.align().max(ALIGN_FLOOR);
            let dangling = NonNull::new(align as *mut u8).ok_or(AllocError)?;
            return Ok((NonNull::slice_from_raw_parts(dangling, 0), true, false));
        }
        let align = layout.align().max(ALIGN_FLOOR);
        let size = layout
            .size()
            .checked_add(align - 1)
            .map(|v| v & !(align - 1))
            .ok_or(AllocError)?;

        let slab_idx = if self.worker_slabs.is_empty() {
            0
        } else {
            worker_hint.min(self.worker_slabs.len() - 1)
        };
        let prefer_slab = match self.kind {
            ArenaKind::Scratch => true,
            ArenaKind::Numbers => {
                align <= REGION_PAGE
                    && (in_subtree || size < REGION_PAGE || self.node_regions.is_empty())
            }
            ArenaKind::Transforms => {
                align <= REGION_PAGE && (size < REGION_PAGE || self.node_regions.is_empty())
            }
        };

        if prefer_slab && slab_idx < self.worker_slabs.len() && size <= self.worker_slabs[slab_idx].cap_bytes {
            let mut g = self.worker_slabs[slab_idx]
                .slab
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(slab) = g.as_mut()
                && let Some((ptr, cap, virgin)) = slab.take(size, align)
            {
                let nn = NonNull::new(ptr).ok_or(AllocError)?;
                return Ok((NonNull::slice_from_raw_parts(nn, cap), false, virgin));
            }
        }

        if align <= REGION_PAGE && !self.node_regions.is_empty() {
            let pref_slot = self
                .worker_node_slot
                .get(worker_hint)
                .copied()
                .unwrap_or(0)
                .min(self.node_regions.len() - 1);
            for off in 0..self.node_regions.len() {
                let slot_i = (pref_slot + off) % self.node_regions.len();
                if let Some((ptr, cap, is_zero, needs_touch)) =
                    self.node_regions[slot_i].take_ext(size)
                {
                    let nn = NonNull::new(ptr).ok_or(AllocError)?;
                    return Ok((NonNull::slice_from_raw_parts(nn, cap), is_zero, needs_touch));
                }
            }
        }

        // System fallback only when both worker slab and region are exhausted (or align > 4096).
        let sys_layout = Layout::from_size_align(size, align).map_err(|_| AllocError)?;
        let raw = unsafe { std::alloc::alloc(sys_layout) };
        let nn = NonNull::new(raw).ok_or(AllocError)?;
        let used = self.sys_bytes_in_use.fetch_add(size, Ordering::Relaxed) + size;
        self.sys_peak_bytes.fetch_max(used, Ordering::Relaxed);
        Ok((NonNull::slice_from_raw_parts(nn, size), false, true))
    }

    #[inline]
    unsafe fn deallocate_inner(&self, ptr: NonNull<u8>, layout: Layout, worker_hint: usize) -> bool {
        if layout.size() == 0 {
            return true;
        }
        let align = layout.align().max(ALIGN_FLOOR);
        let size = (layout.size() + align - 1) & !(align - 1);
        let p = ptr.as_ptr();

        let n_slabs = self.worker_slabs.len();
        if n_slabs > 0 {
            let first = worker_hint.min(n_slabs - 1);
            if self.worker_slabs[first].owns(p) {
                let mut g = self.worker_slabs[first].slab.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(slab) = g.as_mut() {
                    let cap = size.div_ceil(ALIGN_FLOOR) * ALIGN_FLOOR;
                    unsafe { slab.give(p, cap) };
                    return true;
                }
            }
            for (idx, ws) in self.worker_slabs.iter().enumerate() {
                if idx != first && ws.owns(p) {
                    let mut g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(slab) = g.as_mut() {
                        let cap = size.div_ceil(ALIGN_FLOOR) * ALIGN_FLOOR;
                        unsafe { slab.give(p, cap) };
                        return true;
                    }
                }
            }
        }
        for nr in &self.node_regions {
            if nr.owns(p) && unsafe { nr.give(p, size) } {
                return true;
            }
        }
        false
    }

    #[inline]
    unsafe fn resize_in_place_inner(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        worker_hint: usize,
    ) -> Option<NonNull<[u8]>> {
        let old_align = old_layout.align().max(ALIGN_FLOOR);
        let new_align = new_layout.align().max(ALIGN_FLOOR);
        if new_align > old_align && !(ptr.as_ptr() as usize).is_multiple_of(new_align) {
            return None;
        }
        let old_size = (old_layout.size() + old_align - 1) & !(old_align - 1);
        let new_size = (new_layout.size() + new_align - 1) & !(new_align - 1);
        let p = ptr.as_ptr();

        let n_slabs = self.worker_slabs.len();
        if n_slabs > 0 {
            let first = worker_hint.min(n_slabs - 1);
            if self.worker_slabs[first].owns(p) {
                let mut g = self.worker_slabs[first].slab.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(slab) = g.as_mut() {
                    let old_cap = old_size.div_ceil(ALIGN_FLOOR) * ALIGN_FLOOR;
                    let new_cap = unsafe { slab.resize(p, old_cap, new_size) }?;
                    return Some(NonNull::slice_from_raw_parts(ptr, new_cap));
                }
            }
            for (idx, ws) in self.worker_slabs.iter().enumerate() {
                if idx != first && ws.owns(p) {
                    let mut g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(slab) = g.as_mut() {
                        let old_cap = old_size.div_ceil(ALIGN_FLOOR) * ALIGN_FLOOR;
                        let new_cap = unsafe { slab.resize(p, old_cap, new_size) }?;
                        return Some(NonNull::slice_from_raw_parts(ptr, new_cap));
                    }
                }
            }
        }
        for nr in &self.node_regions {
            if nr.owns(p) {
                let old_cap = old_size.div_ceil(REGION_PAGE) * REGION_PAGE;
                let new_cap = unsafe { nr.resize(p, old_cap, new_size) }?;
                return Some(NonNull::slice_from_raw_parts(ptr, new_cap));
            }
        }
        None
    }
}

unsafe impl Allocator for Arena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (wi, sub) = tls_worker_and_subtree();
        self.allocate_detailed(layout, wi, sub).map(|(s, _, _)| s)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (wi, sub) = tls_worker_and_subtree();
        let (slice, is_zero, _) = self.allocate_detailed(layout, wi, sub)?;
        if !is_zero && slice.len() > 0 {
            unsafe {
                std::ptr::write_bytes(slice.as_ptr() as *mut u8, 0, slice.len());
            }
        }
        Ok(slice)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
        if unsafe { self.deallocate_inner(ptr, layout, wi) } {
            return;
        }
        let align = layout.align().max(ALIGN_FLOOR);
        let size = (layout.size() + align - 1) & !(align - 1);
        if let Ok(sys_layout) = Layout::from_size_align(size, align) {
            self.sys_bytes_in_use.fetch_sub(size, Ordering::Relaxed);
            unsafe { std::alloc::dealloc(ptr.as_ptr(), sys_layout) };
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }
        let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
        if let Some(grown) = unsafe { self.resize_in_place_inner(ptr, old_layout, new_layout, wi) } {
            return Ok(grown);
        }
        let fresh = self.allocate(new_layout)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                fresh.as_ptr() as *mut u8,
                old_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }
        Ok(fresh)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        if old_layout.size() == 0 {
            return self.allocate_zeroed(new_layout);
        }
        let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
        if let Some(grown) = unsafe { self.resize_in_place_inner(ptr, old_layout, new_layout, wi) } {
            let tail = grown.len().saturating_sub(old_layout.size());
            if tail > 0 {
                unsafe {
                    std::ptr::write_bytes(
                        (grown.as_ptr() as *mut u8).add(old_layout.size()),
                        0,
                        tail,
                    );
                }
            }
            return Ok(grown);
        }
        let fresh = self.allocate_zeroed(new_layout)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                fresh.as_ptr() as *mut u8,
                old_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }
        Ok(fresh)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() <= old_layout.size());
        if new_layout.size() == 0 {
            unsafe { self.deallocate(ptr, old_layout) };
            let align = new_layout.align().max(ALIGN_FLOOR);
            let dangling = NonNull::new(align as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }
        let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
        if let Some(shrunk) = unsafe { self.resize_in_place_inner(ptr, old_layout, new_layout, wi) } {
            return Ok(shrunk);
        }
        let fresh = self.allocate(new_layout)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                fresh.as_ptr() as *mut u8,
                new_layout.size(),
            );
            self.deallocate(ptr, old_layout);
        }
        Ok(fresh)
    }
}

/// Worker topology descriptor passed when building NUMA-aware, worker-first-touched `ArenaSet`.
#[derive(Clone, Copy, Debug)]
pub struct ArenaWorkerSpec {
    pub worker_idx: usize,
    pub numa_node: usize,
    pub cache_share_bytes: u64,
}

/// The three arenas of Section 10: `numbers` (per-worker LIFO regions for subtrees + plan-assigned
/// NUMA regions above that depth), per-worker `scratch` (sized to each worker's cache shares), and
/// shared `transforms`.
pub struct ArenaSet {
    pub numbers: Arena,
    pub scratch: Vec<Arena>,
    pub transforms: Arena,
    shared_backed_pages: Arc<AtomicU64>,
    soft_cap_bytes: AtomicUsize,
}

impl ArenaSet {
    /// Simple constructor (single NUMA node 0, without worker pool first-touch).
    pub fn new(
        num_workers: usize,
        total_ram_budget: u64,
        scratch_per_worker: u64,
        mapper: Option<&'static dyn RegionMapper>,
    ) -> Self {
        let workers: Vec<ArenaWorkerSpec> = (0..num_workers.max(1))
            .map(|wi| ArenaWorkerSpec {
                worker_idx: wi,
                numa_node: 0,
                cache_share_bytes: scratch_per_worker,
            })
            .collect();
        let scratch_total = scratch_per_worker.saturating_mul(workers.len() as u64);
        let rem = total_ram_budget.saturating_sub(scratch_total);
        let num_b = (rem / 8) * 3;
        let trans_b = rem.saturating_sub(num_b);
        Self::new_with_workers(&workers, num_b, trans_b, total_ram_budget, mapper)
    }

    /// Build the three NUMA-aware Section 10 arenas sized to the profile/plan RAM budgets.
    pub fn new_with_workers(
        workers: &[ArenaWorkerSpec],
        numbers_budget: u64,
        transforms_budget: u64,
        usable_ram_cap: u64,
        mapper: Option<&'static dyn RegionMapper>,
    ) -> Self {
        let m = mapper.unwrap_or(&SYSTEM_REGION_MAPPER);
        let is_sys_fallback = mapper.is_none();
        let nw = workers.len().max(1);
        let shared_backed = Arc::new(AtomicU64::new(0));

        // Discover distinct NUMA nodes in worker rank order.
        let mut nodes: Vec<(usize, usize)> = Vec::new(); // (numa_node, worker_count)
        let mut worker_node_slot = Vec::with_capacity(nw);
        for w in workers {
            if let Some(pos) = nodes.iter().position(|&(n, _)| n == w.numa_node) {
                nodes[pos].1 += 1;
                worker_node_slot.push(pos);
            } else {
                let pos = nodes.len();
                nodes.push((w.numa_node, 1));
                worker_node_slot.push(pos);
            }
        }
        if nodes.is_empty() {
            nodes.push((0, 1));
            worker_node_slot.push(0);
        }

        // 1. Per-worker `scratch` arenas sized to each worker's cache shares (with virtual room
        //    for 2D NTT tile scratch when using an OS virtual-memory `RegionMapper`).
        let mut scratch = Vec::with_capacity(nw);
        let max_virt_pw = ((usable_ram_cap / 4) / (nw as u64)).max(REGION_MIN_BLOCK as u64) as usize;
        for w in workers {
            let cache_b = (w.cache_share_bytes as usize).clamp(REGION_MIN_BLOCK, max_virt_pw);
            let reserve_b = if is_sys_fallback {
                cache_b.min(max_virt_pw)
            } else {
                cache_b.saturating_mul(8).min(max_virt_pw.max(cache_b))
            };
            let slab = LifoSlab::new(m, reserve_b, w.numa_node);
            scratch.push(Arena {
                kind: ArenaKind::Scratch,
                worker_slabs: vec![WorkerSlabSlot::new(slab)],
                node_regions: Vec::new(),
                worker_node_slot: vec![0],
                sys_bytes_in_use: AtomicUsize::new(0),
                sys_peak_bytes: AtomicUsize::new(0),
            });
        }

        // 2. `numbers` arena: per-worker LIFO regions for subtrees + plan-assigned NUMA regions.
        let num_total = (numbers_budget as usize).max(REGION_MIN_BLOCK * nw);
        let num_lifo_each = (num_total / nw / 4).max(REGION_MIN_BLOCK);
        let mut num_worker_slabs = Vec::with_capacity(nw);
        for w in workers {
            num_worker_slabs.push(WorkerSlabSlot::new(LifoSlab::new(m, num_lifo_each, w.numa_node)));
        }
        let num_region_total = num_total
            .saturating_sub(num_lifo_each.saturating_mul(nw) / 2)
            .max(REGION_MIN_BLOCK * nodes.len());
        let mut num_node_regions = Vec::with_capacity(nodes.len());
        for &(numa_node, w_cnt) in &nodes {
            let share = ((num_region_total as u128 * w_cnt as u128) / (nw as u128)) as usize;
            let primary_b = share.max(REGION_MIN_BLOCK);
            let overflow_b = if is_sys_fallback {
                primary_b
            } else {
                ((usable_ram_cap as usize) / nodes.len()).max(primary_b)
            };
            num_node_regions.push(NodeRegionSlot::new(
                numa_node,
                primary_b,
                overflow_b,
                m,
                Some(Arc::clone(&shared_backed)),
            ));
        }
        let numbers = Arena {
            kind: ArenaKind::Numbers,
            worker_slabs: num_worker_slabs,
            node_regions: num_node_regions,
            worker_node_slot: worker_node_slot.clone(),
            sys_bytes_in_use: AtomicUsize::new(0),
            sys_peak_bytes: AtomicUsize::new(0),
        };

        // 3. `transforms` arena: per-worker small L1/L2 slabs + shared NUMA regions.
        let trans_total = (transforms_budget as usize).max(REGION_MIN_BLOCK * nw);
        let trans_small_each = (trans_total / nw / 8).max(REGION_MIN_BLOCK);
        let mut trans_worker_slabs = Vec::with_capacity(nw);
        for w in workers {
            trans_worker_slabs.push(WorkerSlabSlot::new(LifoSlab::new(
                m,
                trans_small_each,
                w.numa_node,
            )));
        }
        let trans_region_total = trans_total
            .saturating_sub(trans_small_each.saturating_mul(nw) / 2)
            .max(REGION_MIN_BLOCK * nodes.len());
        let mut trans_node_regions = Vec::with_capacity(nodes.len());
        for &(numa_node, w_cnt) in &nodes {
            let share = ((trans_region_total as u128 * w_cnt as u128) / (nw as u128)) as usize;
            let primary_b = share.max(REGION_MIN_BLOCK);
            let overflow_b = if is_sys_fallback {
                primary_b
            } else {
                ((usable_ram_cap as usize) / nodes.len()).max(primary_b)
            };
            trans_node_regions.push(NodeRegionSlot::new(
                numa_node,
                primary_b,
                overflow_b,
                m,
                Some(Arc::clone(&shared_backed)),
            ));
        }
        let transforms = Arena {
            kind: ArenaKind::Transforms,
            worker_slabs: trans_worker_slabs,
            node_regions: trans_node_regions,
            worker_node_slot,
            sys_bytes_in_use: AtomicUsize::new(0),
            sys_peak_bytes: AtomicUsize::new(0),
        };

        Self {
            numbers,
            scratch,
            transforms,
            shared_backed_pages: shared_backed,
            soft_cap_bytes: AtomicUsize::new(usable_ram_cap as usize),
        }
    }

    /// First-touch worker `wi`'s local `scratch[wi]`, `numbers` subtree LIFO slab, `transforms`
    /// small slab, and (for the lead worker on each NUMA node) the initial pages of `numbers` and
    /// `transforms` plan-assigned `Region`s on that worker's NUMA node (Section 10).
    pub fn first_touch_worker(
        &self,
        wi: usize,
        scratch_touch_bytes: usize,
        subtree_touch_bytes: usize,
        region_touch_bytes: usize,
    ) {
        if let Some(sc) = self.scratch.get(wi)
            && let Some(ws) = sc.worker_slabs.first()
        {
            let mut g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_mut() {
                s.first_touch(scratch_touch_bytes);
            }
        }
        if let Some(ws) = self.numbers.worker_slabs.get(wi) {
            let mut g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_mut() {
                s.first_touch(subtree_touch_bytes);
            }
        }
        if let Some(ws) = self.transforms.worker_slabs.get(wi) {
            let mut g = ws.slab.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.as_mut() {
                s.first_touch(subtree_touch_bytes / 2);
            }
        }
        if region_touch_bytes > 0
            && let Some(&slot_idx) = self.numbers.worker_node_slot.get(wi)
        {
            // Only the first worker mapped to `slot_idx` touches the node's shared region header.
            let is_node_lead = self
                .numbers
                .worker_node_slot
                .iter()
                .position(|&s| s == slot_idx)
                == Some(wi);
            if is_node_lead {
                if let Some(nr) = self.numbers.node_regions.get(slot_idx) {
                    nr.first_touch_initial(region_touch_bytes);
                }
                if let Some(tr) = self.transforms.node_regions.get(slot_idx) {
                    tr.first_touch_initial(region_touch_bytes);
                }
            }
        }
    }

    /// Pre-commit and parallel-touch the working low/high spans of `numbers` and `transforms`
    /// across all workers on their respective NUMA nodes before the computation begins.
    pub fn prewarm_working_set(
        &self,
        sched: &crate::par::Scheduler,
        num_low_bytes: usize,
        num_high_bytes: usize,
        trans_low_bytes: usize,
        trans_high_bytes: usize,
    ) {
        let nw = self.numbers.worker_node_slot.len().max(1);
        let n_slots = self.numbers.node_regions.len();
        if n_slots == 0 {
            return;
        }
        let mut slot_spans: Vec<[(usize, usize); 4]> = Vec::with_capacity(n_slots);
        let mut slot_w_cnt: Vec<usize> = Vec::with_capacity(n_slots);
        let mut any_pages = false;
        for slot_idx in 0..n_slots {
            let w_cnt = self
                .numbers
                .worker_node_slot
                .iter()
                .filter(|&&s| s == slot_idx)
                .count()
                .max(1);
            let scale = |b: usize| ((b as u128 * w_cnt as u128) / (nw as u128)) as usize;
            let nr_spans = self
                .numbers
                .node_regions
                .get(slot_idx)
                .map_or([(0, 0); 2], |nr| {
                    nr.prepare_parallel_touch(scale(num_low_bytes), scale(num_high_bytes))
                });
            let tr_spans = self
                .transforms
                .node_regions
                .get(slot_idx)
                .map_or([(0, 0); 2], |tr| {
                    tr.prepare_parallel_touch(scale(trans_low_bytes), scale(trans_high_bytes))
                });
            let spans = [nr_spans[0], nr_spans[1], tr_spans[0], tr_spans[1]];
            if spans.iter().any(|&(_, len)| len > 0) {
                any_pages = true;
            }
            slot_spans.push(spans);
            slot_w_cnt.push(w_cnt);
        }
        if !any_pages {
            return;
        }
        sched.on_each_worker(|| {
            let wi = crate::par::current_worker_idx().min(nw - 1);
            let Some(&slot_idx) = self.numbers.worker_node_slot.get(wi) else {
                return;
            };
            let w_cnt = slot_w_cnt[slot_idx].max(1);
            let rank = self.numbers.worker_node_slot[..wi]
                .iter()
                .filter(|&&s| s == slot_idx)
                .count();
            for &(base, len) in &slot_spans[slot_idx] {
                if len == 0 {
                    continue;
                }
                let pages = len / REGION_PAGE;
                let p0 = (pages * rank) / w_cnt;
                let p1 = (pages * (rank + 1)) / w_cnt;
                for p in p0..p1 {
                    unsafe {
                        std::ptr::write_volatile((base + p * REGION_PAGE) as *mut u8, 0);
                    }
                }
            }
        });
    }

    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        self.numbers.owns(ptr)
            || self.transforms.owns(ptr)
            || self.scratch.iter().any(|s| s.owns(ptr))
    }

    pub fn set_soft_cap(&self, cap_bytes: usize) {
        self.soft_cap_bytes.store(cap_bytes, Ordering::Relaxed);
        let n_nodes = self.numbers.node_regions.len().max(1);
        let num_cap = ((cap_bytes / 8).saturating_mul(5) / n_nodes).max(REGION_MIN_BLOCK);
        let trans_cap = ((cap_bytes / 8).saturating_mul(6) / n_nodes).max(REGION_MIN_BLOCK);
        for nr in &self.numbers.node_regions {
            nr.set_soft_cap(num_cap);
        }
        for tr in &self.transforms.node_regions {
            tr.set_soft_cap(trans_cap);
        }
    }

    pub fn trim_above(&self, keep_bytes: usize) {
        let cur_backed = (self.shared_backed_pages.load(Ordering::Relaxed) as usize) * REGION_PAGE;
        if cur_backed <= keep_bytes {
            return;
        }
        let half = keep_bytes / 2;
        let n_nodes = self.numbers.node_regions.len().max(1);
        for nr in &self.numbers.node_regions {
            nr.trim_excess_above(half / n_nodes);
        }
        for tr in &self.transforms.node_regions {
            tr.trim_excess_above(half / n_nodes);
        }
    }

    pub fn combined_report(&self) -> MappingReport {
        let mut r = self.numbers.report();
        r.merge(self.transforms.report());
        for s in &self.scratch {
            r.merge(s.report());
        }
        r
    }

    pub fn usage_stats(&self) -> ArenaUsageStats {
        let mut sc_live = 0u64;
        let mut sc_peak = 0u64;
        for s in &self.scratch {
            sc_live = sc_live.saturating_add(s.current_bytes() as u64);
            sc_peak = sc_peak.saturating_add(s.peak_bytes() as u64);
        }
        ArenaUsageStats {
            numbers_live_bytes: self.numbers.current_bytes() as u64,
            numbers_peak_bytes: self.numbers.peak_bytes() as u64,
            numbers_subtree_lifo_peak_bytes: self.numbers.worker_lifo_peak_bytes(),
            numbers_region_peak_bytes: self.numbers.region_peak_bytes(),
            scratch_live_bytes: sc_live,
            scratch_peak_bytes: sc_peak,
            transforms_live_bytes: self.transforms.current_bytes() as u64,
            transforms_peak_bytes: self.transforms.peak_bytes() as u64,
        }
    }

    pub fn transforms_region_stats(&self) -> RegionStats {
        let mut acc = RegionStats::default();
        for tr in &self.transforms.node_regions {
            let st = tr.stats_sum();
            acc.reserved = acc.reserved.saturating_add(st.reserved);
            acc.soft_cap = acc.soft_cap.saturating_add(st.soft_cap);
            acc.committed = acc.committed.saturating_add(st.committed);
            acc.touched = acc.touched.saturating_add(st.touched);
            acc.live = acc.live.saturating_add(st.live);
            acc.peak_live = acc.peak_live.saturating_add(st.peak_live);
            acc.free_blocks = acc.free_blocks.saturating_add(st.free_blocks);
            acc.refused = acc.refused.saturating_add(st.refused);
            acc.report.merge(st.report);
        }
        acc.soft_cap = self.soft_cap_bytes.load(Ordering::Relaxed);
        acc.live = self
            .numbers
            .current_bytes()
            .saturating_add(self.transforms.current_bytes());
        acc
    }
}

// ---- Thread-bound ArenaSet installation & `Copy` Allocator Handles ---------------------------

static LIVE_ARENA_SETS: RwLock<Vec<Weak<ArenaSet>>> = RwLock::new(Vec::new());

thread_local! {
    static TLS_ARENA: std::cell::RefCell<Option<Arc<ArenaSet>>> = const {
        std::cell::RefCell::new(None)
    };
    static TLS_ARENA_PTR: Cell<*const ArenaSet> = const { Cell::new(std::ptr::null()) };
    static TLS_WORKER_IDX: Cell<usize> = const { Cell::new(0) };
    static TLS_SUBTREE_DEPTH: Cell<u32> = const { Cell::new(0) };
    static TLS_BYPASS_DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[inline]
fn tls_worker_and_subtree() -> (usize, bool) {
    let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
    let sub = TLS_SUBTREE_DEPTH.try_with(|c| c.get() > 0).unwrap_or(false);
    (wi, sub)
}

#[inline]
fn arena_bypassed() -> bool {
    TLS_BYPASS_DEPTH.try_with(|c| c.get() > 0).unwrap_or(false)
}

#[inline]
fn with_active_arenas<R>(f: impl FnOnce(&ArenaSet, usize, bool) -> R) -> Option<R> {
    if arena_bypassed() {
        return None;
    }
    let raw = TLS_ARENA_PTR.try_with(|c| c.get()).unwrap_or(std::ptr::null());
    if raw.is_null() {
        return None;
    }
    let (wi, sub) = tls_worker_and_subtree();
    // SAFETY: `TLS_ARENA_PTR` is set alongside `TLS_ARENA = Some(arc)` and cleared before dropping `arc`.
    Some(f(unsafe { &*raw }, wi, sub))
}

#[inline]
fn with_owner_arenas<R>(ptr: *const u8, mut f: impl FnMut(&ArenaSet, usize) -> Option<R>) -> Option<R> {
    let wi = TLS_WORKER_IDX.try_with(|c| c.get()).unwrap_or(0);
    let raw = TLS_ARENA_PTR.try_with(|c| c.get()).unwrap_or(std::ptr::null());
    if !raw.is_null() {
        let local = unsafe { &*raw };
        if let Some(r) = f(local, wi) {
            return Some(r);
        }
    }
    let reg = LIVE_ARENA_SETS.read().unwrap_or_else(|e| e.into_inner());
    for w in reg.iter() {
        if let Some(a) = w.upgrade()
            && !std::ptr::eq(Arc::as_ptr(&a), raw)
            && a.owns(ptr)
        {
            return f(&a, wi);
        }
    }
    None
}

/// Bind `arenas` onto the calling thread with worker index `worker_idx`.
pub fn bind_thread_arenas(arenas: Arc<ArenaSet>, worker_idx: usize) {
    let raw = Arc::as_ptr(&arenas);
    let _ = TLS_WORKER_IDX.try_with(|c| c.set(worker_idx));
    let _ = TLS_SUBTREE_DEPTH.try_with(|c| c.set(0));
    let _ = TLS_ARENA.try_with(|cell| {
        *cell.borrow_mut() = Some(arenas);
    });
    let _ = TLS_ARENA_PTR.try_with(|c| c.set(raw));
}

/// Unbind any `ArenaSet` from the calling thread.
pub fn unbind_thread_arenas() {
    let _ = TLS_ARENA_PTR.try_with(|c| c.set(std::ptr::null()));
    let _ = TLS_SUBTREE_DEPTH.try_with(|c| c.set(0));
    let _ = TLS_ARENA.try_with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Register an `ArenaSet` in the live-owner registry for the duration of a run.
pub fn register_arena_set(arenas: &Arc<ArenaSet>) {
    let mut reg = LIVE_ARENA_SETS.write().unwrap_or_else(|e| e.into_inner());
    reg.retain(|w| w.strong_count() > 0);
    reg.push(Arc::downgrade(arenas));
}

/// RAII scope marking that the current worker is running a subtree alone so `NumbersAlloc`
/// routes allocations to this worker's LIFO region (Section 10).
pub struct SubtreeLifoScope {
    prev: u32,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl SubtreeLifoScope {
    #[inline]
    pub fn enter() -> Self {
        let prev = TLS_SUBTREE_DEPTH
            .try_with(|c| {
                let p = c.get();
                c.set(p.saturating_add(1));
                p
            })
            .unwrap_or(0);
        Self {
            prev,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Temporarily step out of subtree LIFO mode so an escaping frontier result allocates in the
    /// plan-assigned `Numbers` region above the subtree depth.
    #[inline]
    pub fn pause() -> Self {
        let prev = TLS_SUBTREE_DEPTH.try_with(|c| c.replace(0)).unwrap_or(0);
        Self {
            prev,
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for SubtreeLifoScope {
    #[inline]
    fn drop(&mut self) {
        let _ = TLS_SUBTREE_DEPTH.try_with(|c| c.set(self.prev));
    }
}

struct SystemAllocGuard;

impl Drop for SystemAllocGuard {
    #[inline]
    fn drop(&mut self) {
        let _ = TLS_BYPASS_DEPTH.try_with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Run `f` routing allocations on the current thread to the system allocator (e.g. permanent
/// static twiddle tables).
#[inline]
pub fn with_system_alloc<R>(f: impl FnOnce() -> R) -> R {
    let _ = TLS_BYPASS_DEPTH.try_with(|c| c.set(c.get() + 1));
    let _guard = SystemAllocGuard;
    f()
}

#[inline]
fn sys_alloc(layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
    if layout.size() == 0 {
        let align = layout.align().max(ALIGN_FLOOR);
        let dangling = NonNull::new(align as *mut u8).ok_or(AllocError)?;
        return Ok(NonNull::slice_from_raw_parts(dangling, 0));
    }
    let align = layout.align().max(ALIGN_FLOOR);
    let size = layout
        .size()
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or(AllocError)?;
    let al = Layout::from_size_align(size, align).map_err(|_| AllocError)?;
    let p = unsafe {
        if zeroed {
            System.alloc_zeroed(al)
        } else {
            System.alloc(al)
        }
    };
    let nn = NonNull::new(p).ok_or(AllocError)?;
    Ok(NonNull::slice_from_raw_parts(nn, size))
}

#[inline]
unsafe fn sys_dealloc(ptr: NonNull<u8>, layout: Layout) {
    if layout.size() == 0 {
        return;
    }
    let align = layout.align().max(ALIGN_FLOOR);
    let size = (layout.size() + align - 1) & !(align - 1);
    if let Ok(al) = Layout::from_size_align(size, align) {
        unsafe { System.dealloc(ptr.as_ptr(), al) };
    }
}

impl ArenaSet {
    #[inline]
    fn select_arena(&self, kind: ArenaKind, wi: usize) -> &Arena {
        match kind {
            ArenaKind::Numbers => &self.numbers,
            ArenaKind::Scratch => &self.scratch[wi.min(self.scratch.len() - 1)],
            ArenaKind::Transforms => &self.transforms,
        }
    }

    #[inline]
    unsafe fn dealloc_by_kind(
        &self,
        kind: ArenaKind,
        ptr: NonNull<u8>,
        layout: Layout,
        wi: usize,
    ) -> bool {
        match kind {
            ArenaKind::Numbers => unsafe { self.numbers.deallocate_inner(ptr, layout, wi) },
            ArenaKind::Scratch => {
                let first = wi.min(self.scratch.len().saturating_sub(1));
                if let Some(sc) = self.scratch.get(first)
                    && unsafe { sc.deallocate_inner(ptr, layout, 0) }
                {
                    return true;
                }
                for (idx, sc) in self.scratch.iter().enumerate() {
                    if idx != first && unsafe { sc.deallocate_inner(ptr, layout, 0) } {
                        return true;
                    }
                }
                false
            }
            ArenaKind::Transforms => unsafe { self.transforms.deallocate_inner(ptr, layout, wi) },
        }
    }

    #[inline]
    unsafe fn resize_by_kind(
        &self,
        kind: ArenaKind,
        ptr: NonNull<u8>,
        old_l: Layout,
        new_l: Layout,
        wi: usize,
    ) -> Option<NonNull<[u8]>> {
        match kind {
            ArenaKind::Numbers => unsafe {
                self.numbers.resize_in_place_inner(ptr, old_l, new_l, wi)
            },
            ArenaKind::Scratch => {
                let first = wi.min(self.scratch.len().saturating_sub(1));
                if let Some(sc) = self.scratch.get(first)
                    && sc.owns(ptr.as_ptr())
                {
                    return unsafe { sc.resize_in_place_inner(ptr, old_l, new_l, 0) };
                }
                for (idx, sc) in self.scratch.iter().enumerate() {
                    if idx != first && sc.owns(ptr.as_ptr()) {
                        return unsafe { sc.resize_in_place_inner(ptr, old_l, new_l, 0) };
                    }
                }
                None
            }
            ArenaKind::Transforms => unsafe {
                self.transforms.resize_in_place_inner(ptr, old_l, new_l, wi)
            },
        }
    }
}

macro_rules! impl_arena_handle {
    ($name:ident, $kind:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        unsafe impl Allocator for $name {
            #[inline]
            fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                if let Some(res) = with_active_arenas(|a, wi, sub| {
                    a.select_arena($kind, wi)
                        .allocate_detailed(layout, wi, sub)
                        .map(|(s, _, _)| s)
                }) {
                    return res;
                }
                sys_alloc(layout, false)
            }

            #[inline]
            fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                if let Some(res) = with_active_arenas(|a, wi, sub| {
                    let (slice, is_zero, _) =
                        a.select_arena($kind, wi).allocate_detailed(layout, wi, sub)?;
                    if !is_zero && slice.len() > 0 {
                        unsafe {
                            std::ptr::write_bytes(slice.as_ptr() as *mut u8, 0, slice.len());
                        }
                    }
                    Ok(slice)
                }) {
                    return res;
                }
                sys_alloc(layout, true)
            }

            #[inline]
            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                if layout.size() == 0 {
                    return;
                }
                if with_owner_arenas(ptr.as_ptr(), |owner, wi| {
                    unsafe { owner.dealloc_by_kind($kind, ptr, layout, wi) }.then_some(())
                })
                .is_some()
                {
                    return;
                }
                let align = layout.align().max(ALIGN_FLOOR);
                let size = (layout.size() + align - 1) & !(align - 1);
                let _ = with_active_arenas(|a, wi, _| {
                    let _ = a.select_arena($kind, wi).sys_bytes_in_use.try_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |cur| Some(cur.saturating_sub(size)),
                    );
                });
                unsafe { sys_dealloc(ptr, layout) };
            }

            unsafe fn grow(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                debug_assert!(new_layout.size() >= old_layout.size());
                if old_layout.size() == 0 {
                    return self.allocate(new_layout);
                }
                if let Some(grown) = with_owner_arenas(ptr.as_ptr(), |owner, wi| unsafe {
                    owner.resize_by_kind($kind, ptr, old_layout, new_layout, wi)
                }) {
                    return Ok(grown);
                }
                let fresh = self.allocate(new_layout)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ptr.as_ptr(),
                        fresh.as_ptr() as *mut u8,
                        old_layout.size(),
                    );
                    self.deallocate(ptr, old_layout);
                }
                Ok(fresh)
            }

            unsafe fn grow_zeroed(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                debug_assert!(new_layout.size() >= old_layout.size());
                if old_layout.size() == 0 {
                    return self.allocate_zeroed(new_layout);
                }
                if let Some(grown) = with_owner_arenas(ptr.as_ptr(), |owner, wi| unsafe {
                    owner.resize_by_kind($kind, ptr, old_layout, new_layout, wi)
                }) {
                    let tail = grown.len().saturating_sub(old_layout.size());
                    if tail > 0 {
                        unsafe {
                            std::ptr::write_bytes(
                                (grown.as_ptr() as *mut u8).add(old_layout.size()),
                                0,
                                tail,
                            );
                        }
                    }
                    return Ok(grown);
                }
                let fresh = self.allocate_zeroed(new_layout)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ptr.as_ptr(),
                        fresh.as_ptr() as *mut u8,
                        old_layout.size(),
                    );
                    self.deallocate(ptr, old_layout);
                }
                Ok(fresh)
            }

            unsafe fn shrink(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                debug_assert!(new_layout.size() <= old_layout.size());
                if new_layout.size() == 0 {
                    unsafe { self.deallocate(ptr, old_layout) };
                    let align = new_layout.align().max(ALIGN_FLOOR);
                    let dangling = NonNull::new(align as *mut u8).ok_or(AllocError)?;
                    return Ok(NonNull::slice_from_raw_parts(dangling, 0));
                }
                if let Some(shrunk) = with_owner_arenas(ptr.as_ptr(), |owner, wi| unsafe {
                    owner.resize_by_kind($kind, ptr, old_layout, new_layout, wi)
                }) {
                    return Ok(shrunk);
                }
                let fresh = self.allocate(new_layout)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ptr.as_ptr(),
                        fresh.as_ptr() as *mut u8,
                        new_layout.size(),
                    );
                    self.deallocate(ptr, old_layout);
                }
                Ok(fresh)
            }
        }
    };
}

impl_arena_handle!(NumbersAlloc, ArenaKind::Numbers);
impl_arena_handle!(ScratchAlloc, ArenaKind::Scratch);
impl_arena_handle!(TransformsAlloc, ArenaKind::Transforms);

/// Allocate an uninitialized `Vec<u64, NumbersAlloc>` of capacity `>= min_cap`, returning
/// `(vec, needs_touch)` so callers know whether the backing pages are already faulted.
#[inline]
pub fn alloc_numbers_u64_uninit(min_cap: usize) -> (Vec<u64, NumbersAlloc>, bool) {
    if min_cap == 0 {
        return (Vec::new_in(NumbersAlloc), false);
    }
    let layout = Layout::array::<u64>(min_cap).expect("numbers u64 layout overflow");
    if let Some(Ok((slice, _, needs_touch))) =
        with_active_arenas(|a, wi, sub| a.numbers.allocate_detailed(layout, wi, sub))
    {
        let cap_u64 = slice.len() / 8;
        let v = unsafe { Vec::from_raw_parts_in(slice.as_ptr() as *mut u64, 0, cap_u64, NumbersAlloc) };
        return (v, needs_touch);
    }
    let slice = sys_alloc(layout, false).unwrap_or_else(|_| std::alloc::handle_alloc_error(layout));
    let cap_u64 = slice.len() / 8;
    let v = unsafe { Vec::from_raw_parts_in(slice.as_ptr() as *mut u64, 0, cap_u64, NumbersAlloc) };
    (v, true)
}

/// Allocate a zero-initialized `Vec<u64, NumbersAlloc>` of length `len`, skipping `write_bytes`
/// when `Numbers` hands out OS-zeroed virgin pages.
#[inline]
pub fn alloc_numbers_u64_zeroed(len: usize) -> Vec<u64, NumbersAlloc> {
    if len == 0 {
        return Vec::new_in(NumbersAlloc);
    }
    let layout = Layout::array::<u64>(len).expect("numbers u64 layout overflow");
    let slice = NumbersAlloc
        .allocate_zeroed(layout)
        .unwrap_or_else(|_| std::alloc::handle_alloc_error(layout));
    let cap_u64 = slice.len() / 8;
    unsafe { Vec::from_raw_parts_in(slice.as_ptr() as *mut u64, len, cap_u64, NumbersAlloc) }
}

/// Allocate a 64-byte cache-line aligned `Vec<u8, TransformsAlloc>` of capacity `>= byte_cap`,
/// returning `(vec, needs_touch)`.
#[inline]
pub fn alloc_transforms_bytes_uninit(byte_cap: usize) -> (Vec<u8, TransformsAlloc>, bool) {
    if byte_cap == 0 {
        return (Vec::new_in(TransformsAlloc), false);
    }
    let layout = Layout::from_size_align(byte_cap, ALIGN_FLOOR).expect("transforms byte layout");
    if let Some(Ok((slice, _, needs_touch))) =
        with_active_arenas(|a, wi, sub| a.transforms.allocate_detailed(layout, wi, sub))
    {
        let v = unsafe {
            Vec::from_raw_parts_in(slice.as_ptr() as *mut u8, 0, slice.len(), TransformsAlloc)
        };
        return (v, needs_touch);
    }
    let slice = sys_alloc(layout, false).unwrap_or_else(|_| std::alloc::handle_alloc_error(layout));
    let v = unsafe {
        Vec::from_raw_parts_in(slice.as_ptr() as *mut u8, 0, slice.len(), TransformsAlloc)
    };
    (v, true)
}

/// Counters of the active thread's `ArenaSet` (for work-grid headroom checks and trace output).
pub fn active_arenas_stats() -> Option<RegionStats> {
    with_active_arenas(|a, _, _| a.transforms_region_stats())
}

/// Live bytes currently held in the active `ArenaSet`'s `numbers` arena (`0` if unbound).
pub fn active_numbers_live_bytes() -> u64 {
    with_active_arenas(|a, _, _| a.numbers.current_bytes() as u64).unwrap_or(0)
}

/// Trim unused physical backing across the active `ArenaSet` above `keep_bytes`.
pub fn trim_active_arenas_above(keep_bytes: usize) {
    let _ = with_active_arenas(|a, _, _| {
        a.trim_above(keep_bytes);
    });
}

/// Trace runs: print one line summarizing `numbers`, `scratch`, and `transforms` arena usage.
pub fn trace_active_arenas(tag: &str) {
    if let Some(u) = with_active_arenas(|a, _, _| a.usage_stats()) {
        let mb = |b: u64| (b as f64) / 1048576.0;
        eprintln!(
            "[TRACE ARENA {tag}] num_live={:.0}MB(peak={:.0}MB lifo_pk={:.0}MB) trans_live={:.0}MB(peak={:.0}MB) scratch_pk={:.0}MB",
            mb(u.numbers_live_bytes),
            mb(u.numbers_peak_bytes),
            mb(u.numbers_subtree_lifo_peak_bytes),
            mb(u.transforms_live_bytes),
            mb(u.transforms_peak_bytes),
            mb(u.scratch_peak_bytes),
        );
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    struct HeapMapper;

    impl RegionMapper for HeapMapper {
        fn reserve_region(
            &self,
            bytes: usize,
            _node: usize,
        ) -> Option<(*mut u8, usize, MappingReport)> {
            let layout = Layout::from_size_align(bytes, REGION_PAGE).ok()?;
            let p = unsafe { System.alloc_zeroed(layout) };
            (!p.is_null()).then_some((p, bytes, MappingReport::default()))
        }

        fn commit_range(
            &self,
            _ptr: *mut u8,
            _len: usize,
            _node: usize,
        ) -> Result<MappingReport, ()> {
            Ok(MappingReport::default())
        }

        unsafe fn release_region(&self, base: *mut u8, reserved_len: usize) {
            let layout = Layout::from_size_align(reserved_len, REGION_PAGE).unwrap();
            unsafe { System.dealloc(base, layout) };
        }
    }

    static HEAP_MAPPER: HeapMapper = HeapMapper;
    const PG: usize = REGION_PAGE;

    fn region(pages: usize) -> Region {
        Region::with_large_block(&HEAP_MAPPER, pages * PG, 16 * PG).unwrap()
    }

    fn check_list(list: &FreeList, lo: u64, hi: u64) -> u64 {
        assert_eq!(list.by_addr.len(), list.by_size.len());
        let mut prev_end = None;
        let mut free_pages = 0u64;
        for (&start, &len) in &list.by_addr {
            assert!(len > 0);
            assert!(list.by_size.contains(&(len, start)));
            assert!(
                start > lo || (start == lo && lo == 0),
                "a free block may not touch the middle"
            );
            assert!(start >= lo && start + len <= hi);
            if let Some(pe) = prev_end {
                assert!(start > pe, "free blocks may not touch or overlap");
            }
            prev_end = Some(start + len);
            free_pages += len as u64;
        }
        free_pages
    }

    fn check_invariants(r: &Region) {
        let s = r.lock();
        assert!(s.top <= s.bottom && s.bottom <= r.pages);
        assert!(s.top <= s.touched && s.touched_from <= s.bottom);
        for (&start, &len) in &s.low.by_addr {
            assert!(start + len < s.top);
        }
        let free_low = check_list(&s.low, 0, s.top);
        let free_high = check_list(&s.high, s.bottom, r.pages);
        let in_zones = (s.top as u64) + ((r.pages - s.bottom) as u64);
        assert_eq!(free_low + free_high + s.live_pages, in_zones);
    }

    #[test]
    fn large_blocks_are_reused_and_neighbours_join() {
        let r = region(256);
        let (a, za) = r.take(16 * PG).unwrap();
        let (b, _) = r.take(16 * PG).unwrap();
        let (c, _) = r.take(16 * PG).unwrap();
        assert!(za);
        assert_eq!(a as usize, r.base + 240 * PG);
        assert_eq!(a as usize - b as usize, 16 * PG);
        unsafe {
            r.give(a, 16 * PG);
            check_invariants(&r);
            r.give(b, 16 * PG);
        }
        check_invariants(&r);
        let (d, zd) = r.take(32 * PG).unwrap();
        assert_eq!(d, b);
        assert!(!zd);
        unsafe {
            r.give(c, 16 * PG);
            r.give(d, 32 * PG);
        }
        check_invariants(&r);
        let st = r.stats();
        assert_eq!((st.live, st.free_blocks), (0, 0));
        assert_eq!(st.touched, 48 * PG);
        assert!(r.take(257 * PG).is_none());
        assert_eq!(r.stats().refused, 1);
    }

    #[test]
    fn small_blocks_come_from_the_start_and_leave_large_ones_whole() {
        let r = region(256);
        let base = r.base;
        let (big1, _) = r.take(40 * PG).unwrap();
        let (s1, z1) = r.take(4 * PG).unwrap();
        let (big2, _) = r.take(40 * PG).unwrap();
        let (s2, _) = r.take(4 * PG).unwrap();
        assert!(z1);
        assert_eq!(big1 as usize, base + 216 * PG);
        assert_eq!(big2 as usize, base + 176 * PG);
        assert_eq!(s1 as usize, base);
        assert_eq!(s2 as usize, base + 4 * PG);
        unsafe {
            r.give(big1, 40 * PG);
            r.give(big2, 40 * PG);
        }
        check_invariants(&r);
        let (big3, _) = r.take(80 * PG).unwrap();
        assert_eq!(big3 as usize, base + 176 * PG);
        unsafe {
            r.give(s1, 4 * PG);
            check_invariants(&r);
            r.give(s2, 4 * PG);
            r.give(big3, 80 * PG);
        }
        check_invariants(&r);
        let st = r.stats();
        assert_eq!((st.live, st.free_blocks), (0, 0));
        assert_eq!(st.touched, (80 + 8) * PG);
    }

    #[test]
    fn a_large_block_reuses_freed_space_among_the_small_ones() {
        let r = region(256);
        let smalls: Vec<*mut u8> = (0..8).map(|_| r.take(8 * PG).unwrap().0).collect();
        unsafe {
            for &p in &smalls[1..5] {
                r.give(p, 8 * PG);
            }
        }
        check_invariants(&r);
        let (big, zero) = r.take(24 * PG).unwrap();
        assert_eq!(big as usize, r.base + 8 * PG);
        assert!(!zero);
        assert_eq!(r.stats().touched, 64 * PG);
        unsafe {
            r.give(big, 24 * PG);
            for &p in smalls[..1].iter().chain(&smalls[5..]) {
                r.give(p, 8 * PG);
            }
        }
        check_invariants(&r);
        assert_eq!((r.stats().live, r.stats().free_blocks), (0, 0));
    }

    #[test]
    fn a_small_block_goes_among_the_large_ones_only_when_the_middle_is_used_up() {
        let r = region(64);
        let (big1, _) = r.take(16 * PG).unwrap();
        let (big2, _) = r.take(16 * PG).unwrap();
        let (big3, _) = r.take(16 * PG).unwrap();
        let (s1, _) = r.take(15 * PG).unwrap();
        assert!(r.take(4 * PG).is_none());
        unsafe { r.give(big2, 16 * PG) };
        let (s2, _) = r.take(4 * PG).unwrap();
        assert_eq!(s2, big2);
        check_invariants(&r);
        unsafe {
            r.give(s2, 4 * PG);
            r.give(big1, 16 * PG);
            r.give(big3, 16 * PG);
            r.give(s1, 15 * PG);
        }
        check_invariants(&r);
        assert_eq!((r.stats().live, r.stats().free_blocks), (0, 0));
    }

    #[test]
    fn resize_in_place_grows_and_shrinks() {
        for unit in [1usize, 16] {
            let r = region(1024);
            let (a, _) = r.take(4 * unit * PG).unwrap();
            unsafe {
                if unit == 16 {
                    assert!(!r.resize(a, 4 * unit * PG, 10 * unit * PG));
                    r.give(a, 4 * unit * PG);
                    let (pad, _) = r.take(6 * unit * PG).unwrap();
                    let (a2, _) = r.take(4 * unit * PG).unwrap();
                    assert_eq!(a2, a.sub(6 * unit * PG));
                    r.give(pad, 6 * unit * PG);
                    assert!(r.resize(a2, 4 * unit * PG, 10 * unit * PG));
                    check_invariants(&r);
                    r.give(a2, 10 * unit * PG);
                    check_invariants(&r);
                    assert_eq!(r.stats().live, 0);
                    continue;
                }
                assert!(r.resize(a, 4 * PG, 10 * PG));
                let (b, _) = r.take(4 * PG).unwrap();
                assert_eq!(b as usize - a as usize, 10 * PG);
                assert!(!r.resize(a, 10 * PG, 11 * PG));
                assert!(r.resize(a, 10 * PG, 3 * PG));
                check_invariants(&r);
                assert!(r.resize(a, 3 * PG, 9 * PG));
                assert!(!r.resize(a, 9 * PG, 11 * PG));
                check_invariants(&r);
                r.give(b, 4 * PG);
                r.give(a, 9 * PG);
            }
            check_invariants(&r);
            assert_eq!(r.stats().live, 0);
        }
    }

    #[test]
    fn random_traffic_never_overlaps_or_loses_pages() {
        let r = region(4096);
        let mut live: Vec<(*mut u8, usize, u8)> = Vec::new();
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for step in 0..30_000u32 {
            let roll = next() % 100;
            if roll < 45 || live.is_empty() {
                let bytes = 1 + (next() as usize % (96 * PG));
                if let Some((p, zero)) = r.take(bytes) {
                    let tag = (step % 251) as u8 + 1;
                    unsafe {
                        if zero {
                            let sl = std::slice::from_raw_parts(p, bytes);
                            assert!(sl.iter().all(|&b| b == 0), "a block reported zero is not");
                        }
                        std::ptr::write_bytes(p, tag, bytes);
                    }
                    live.push((p, bytes, tag));
                }
            } else if roll < 85 {
                let i = next() as usize % live.len();
                let (p, bytes, tag) = live.swap_remove(i);
                unsafe {
                    let sl = std::slice::from_raw_parts(p, bytes);
                    assert!(sl.iter().all(|&b| b == tag), "a live block was overwritten");
                    r.give(p, bytes);
                }
            } else {
                let i = next() as usize % live.len();
                let (p, bytes, tag) = live[i];
                let new_bytes = 1 + (next() as usize % (96 * PG));
                if unsafe { r.resize(p, bytes, new_bytes) } {
                    unsafe {
                        let keep = bytes.min(new_bytes);
                        let sl = std::slice::from_raw_parts(p, keep);
                        assert!(sl.iter().all(|&b| b == tag), "resize changed kept bytes");
                        std::ptr::write_bytes(p, tag, new_bytes);
                    }
                    live[i].1 = new_bytes;
                }
            }
            if step % 499 == 0 {
                check_invariants(&r);
            }
        }
        for (p, bytes, tag) in live.drain(..) {
            unsafe {
                let sl = std::slice::from_raw_parts(p, bytes);
                assert!(sl.iter().all(|&b| b == tag));
                r.give(p, bytes);
            }
        }
        check_invariants(&r);
        let st = r.stats();
        assert_eq!((st.live, st.free_blocks), (0, 0));
    }

    #[test]
    fn three_arenas_back_standard_vec_and_box_and_coalesce_lifo() {
        let specs = [
            ArenaWorkerSpec {
                worker_idx: 0,
                numa_node: 0,
                cache_share_bytes: 1024 * 1024,
            },
            ArenaWorkerSpec {
                worker_idx: 1,
                numa_node: 1,
                cache_share_bytes: 1024 * 1024,
            },
        ];
        let arenas = Arc::new(ArenaSet::new_with_workers(
            &specs,
            16 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            Some(&HEAP_MAPPER),
        ));
        register_arena_set(&arenas);
        arenas.first_touch_worker(0, 64 * 1024, 64 * 1024, 64 * 1024);
        bind_thread_arenas(Arc::clone(&arenas), 0);

        // Direct `&Arena` Vec and Box:
        let mut direct_v = arenas.numbers.alloc_vec::<u64>(512);
        direct_v.extend(0..512);
        let direct_b = arenas.scratch[0].alloc_box([42u64; 16]);
        assert_eq!(direct_b[15], 42);
        drop(direct_b);
        drop(direct_v);

        // Subtree LIFO merge pattern (`left`, `right`, `parent` allocated before `left`/`right` drop):
        {
            let _sub = SubtreeLifoScope::enter();
            let left: Vec<u64, NumbersAlloc> = {
                let mut v = Vec::with_capacity_in(4096, NumbersAlloc);
                v.resize(4096, 11);
                v
            };
            let right: Vec<u64, NumbersAlloc> = {
                let mut v = Vec::with_capacity_in(4096, NumbersAlloc);
                v.resize(4096, 22);
                v
            };
            let parent: Vec<u64, NumbersAlloc> = {
                let mut v = Vec::with_capacity_in(8192, NumbersAlloc);
                v.resize(8192, 33);
                v
            };
            drop(left);
            drop(right);
            assert_eq!(parent[8191], 33);
            drop(parent);
        }
        assert_eq!(arenas.numbers.current_bytes(), 0);
        assert!(arenas.numbers.worker_lifo_peak_bytes() > 0);

        // Plan-assigned region above subtree depth + transform & scratch handles:
        let mut upper_num: Vec<u64, NumbersAlloc> = Vec::with_capacity_in(32_768, NumbersAlloc);
        upper_num.resize(32_768, 7);
        let mut grid: Vec<u8, TransformsAlloc> = Vec::with_capacity_in(262_144, TransformsAlloc);
        grid.resize(262_144, 9);
        let sc_box: Box<[f64; 128], ScratchAlloc> = Box::new_in([1.5; 128], ScratchAlloc);
        assert_eq!(sc_box[0], 1.5);
        assert!(arenas.numbers.region_peak_bytes() >= 262_144);
        assert!(arenas.transforms.peak_bytes() >= 262_144);
        assert!(arenas.scratch[0].peak_bytes() >= 1024);
        drop(sc_box);
        drop(grid);
        drop(upper_num);
        assert_eq!(arenas.numbers.current_bytes(), 0);
        assert_eq!(arenas.transforms.current_bytes(), 0);
        assert_eq!(arenas.scratch[0].current_bytes(), 0);
        unbind_thread_arenas();
    }

    #[test]
    fn region_and_freelist_index_beyond_16_tib_u32_page_limit() {
        // Synthetic virtual mapper returning a 32 TiB (> u32::MAX * 4 KiB = 16 TiB) address span
        // without touching host RAM so 64-bit page indexing in `Region` and `FreeList` is verified.
        struct Virtual32TiBMapper;
        impl RegionMapper for Virtual32TiBMapper {
            fn reserve_region(
                &self,
                bytes: usize,
                _numa_node: usize,
            ) -> Option<(*mut u8, usize, MappingReport)> {
                Some((0x1000_0000usize as *mut u8, bytes, MappingReport::default()))
            }
            fn commit_range(
                &self,
                _ptr: *mut u8,
                _len: usize,
                _numa_node: usize,
            ) -> Result<MappingReport, ()> {
                Ok(MappingReport::default())
            }
            unsafe fn release_region(&self, _base: *mut u8, _reserved_len: usize) {}
        }
        static V32T: Virtual32TiBMapper = Virtual32TiBMapper;

        let span_32_tib: usize = 32 * 1024 * 1024 * 1024 * 1024;
        let r = Region::new(&V32T, span_32_tib).expect("32 TiB region");
        assert!(r.pages > u32::MAX as u64);
        assert_eq!(r.stats().reserved, span_32_tib);

        // Large allocation cut from the high end (`start > u32::MAX` pages):
        let big_bytes = 64 * 1024 * 1024;
        let (hi_ptr, _) = r.take(big_bytes).expect("high-zone block above 16 TiB");
        assert_eq!(hi_ptr as usize, 0x1000_0000usize + span_32_tib - big_bytes);
        assert!(r.owns(hi_ptr));
        unsafe { r.give(hi_ptr, big_bytes) };
        check_invariants(&r);
        assert_eq!(r.stats().live, 0);
    }
}
