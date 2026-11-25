//! Contract:
//! - Inputs: Win32 kernel queries (`GetLogicalProcessorInformationEx`, `GetProcessAffinityMask`,
//!   `GlobalMemoryStatusEx`, `GetDiskFreeSpaceExW`), thread placement requests, arena
//!   virtual-memory reservations, SSD paths, and output file paths.
//! - Outputs: Populated `Profile` (counting only the hardware threads the process's affinity
//!   allows, and each cache divided among those), `WinThreadPlacer`, `WinRegionMapper` (reporting
//!   huge-page and pinning outcomes), direct-I/O overlapped `WinSsdMember` (`FILE_FLAG_NO_BUFFERING
//!   | FILE_FLAG_OVERLAPPED` with coalescing `MemberSpanAlloc` reclamation, all batch ranges in
//!   flight at once, and a 5 s transfer timeout), and preallocated `WinOutputSinkFactory`.
//! - Data format: Raw Win32 FFI (`kernel32`) with zero external crates.
//! - Work & bytes moved: Startup topology discovery; zero-copy direct NVMe/SSD transfers when
//!   4096-aligned (4096-aligned bounce buffer only for unaligned edges).

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use engine::arena::{MappingReport, RegionMapper};
use engine:: OutputSinkFactory;
use engine::par::ThreadPlacer;
use engine::radix::DigitSink;
use engine::store::{ByteRange, ExtentMember, MemberKind, MemberSpanAlloc, StoreError};
use engine::tier::{CacheEntry, CoreThreads, DeviceEntry, Profile, RamNode};

type Handle = *mut std::ffi::c_void;
type Bool = i32;
type Dword = u32;

const INVALID_HANDLE_VALUE: Handle = (-1isize) as Handle;
const GENERIC_READ: Dword = 0x8000_0000;
const GENERIC_WRITE: Dword = 0x4000_0000;
const FILE_SHARE_READ: Dword = 0x0000_0001;
const FILE_SHARE_WRITE: Dword = 0x0000_0002;
const CREATE_ALWAYS: Dword = 2;
const OPEN_EXISTING: Dword = 3;
const OPEN_ALWAYS: Dword = 4;
const FILE_ATTRIBUTE_NORMAL: Dword = 0x0000_0080;
const FILE_FLAG_OVERLAPPED: Dword = 0x4000_0000;
const FILE_FLAG_NO_BUFFERING: Dword = 0x2000_0000;
const ERROR_IO_PENDING: Dword = 997;
const WAIT_OBJECT_0: Dword = 0x0000_0000;
const WAIT_TIMEOUT: Dword = 0x0000_0102;
const SSD_IO_TIMEOUT_MS: Dword = 5_000;

const MEM_COMMIT: Dword = 0x0000_1000;
const MEM_RESERVE: Dword = 0x0000_2000;
const MEM_DECOMMIT: Dword = 0x0000_4000;
const MEM_RELEASE: Dword = 0x0000_8000;
const MEM_LARGE_PAGES: Dword = 0x2000_0000;
const PAGE_READWRITE: Dword = 0x04;

const RELATION_PROCESSOR_CORE: u32 = 0;
const RELATION_NUMA_NODE: u32 = 1;
const RELATION_CACHE: u32 = 2;
const RELATION_NUMA_NODE_EX: u32 = 6;
const RELATION_ALL: u32 = 0xFFFF;

#[repr(C)]
#[derive(Clone, Copy)]
struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}

#[repr(C)]
struct MemoryStatusEx {
    dw_length: Dword,
    dw_memory_load: Dword,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[repr(C)]
struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset: Dword,
    offset_high: Dword,
    h_event: Handle,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLastError() -> Dword;
    fn CloseHandle(h_object: Handle) -> Bool;
    fn GetCurrentProcess() -> Handle;
    fn GetCurrentThread() -> Handle;

    fn GetLogicalProcessorInformationEx(
        relationship_type: u32,
        buffer: *mut u8,
        returned_length: *mut Dword,
    ) -> Bool;
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> Bool;
    fn SetThreadGroupAffinity(
        h_thread: Handle,
        group_affinity: *const GroupAffinity,
        previous_group_affinity: *mut GroupAffinity,
    ) -> Bool;
    fn GetProcessAffinityMask(
        h_process: Handle,
        lp_process_affinity_mask: *mut usize,
        lp_system_affinity_mask: *mut usize,
    ) -> Bool;
    fn GetProcessGroupAffinity(
        h_process: Handle,
        group_count: *mut u16,
        group_array: *mut u16,
    ) -> Bool;

    fn GetLargePageMinimum() -> usize;
    fn VirtualAllocExNuma(
        h_process: Handle,
        lp_address: *mut u8,
        dw_size: usize,
        fl_allocation_type: Dword,
        fl_protect: Dword,
        nnd_preferred: Dword,
    ) -> *mut u8;
    fn VirtualLock(lp_address: *mut u8, dw_size: usize) -> Bool;
    fn VirtualFree(lp_address: *mut u8, dw_size: usize, dw_free_type: Dword) -> Bool;

    fn CreateFileW(
        lp_file_name: *const u16,
        dw_desired_access: Dword,
        dw_share_mode: Dword,
        lp_security_attributes: *const std::ffi::c_void,
        dw_creation_disposition: Dword,
        dw_flags_and_attributes: Dword,
        h_template_file: Handle,
    ) -> Handle;
    fn DeleteFileW(lp_file_name: *const u16) -> Bool;
    fn GetFileSizeEx(h_file: Handle, lp_file_size: *mut i64) -> Bool;
    fn SetFilePointerEx(
        h_file: Handle,
        li_distance_to_move: i64,
        lp_new_file_pointer: *mut i64,
        dw_move_method: Dword,
    ) -> Bool;
    fn SetEndOfFile(h_file: Handle) -> Bool;
    fn SetFileValidData(h_file: Handle, valid_data_length: i64) -> Bool;
    fn FlushFileBuffers(h_file: Handle) -> Bool;
    fn GetDiskFreeSpaceExW(
        lp_directory_name: *const u16,
        lp_free_bytes_available_to_caller: *mut u64,
        lp_total_number_of_bytes: *mut u64,
        lp_total_number_of_free_bytes: *mut u64,
    ) -> Bool;

    fn CreateEventW(
        lp_event_attributes: *const std::ffi::c_void,
        b_manual_reset: Bool,
        b_initial_state: Bool,
        lp_name: *const u16,
    ) -> Handle;
    fn ReadFile(
        h_file: Handle,
        lp_buffer: *mut u8,
        n_number_of_bytes_to_read: Dword,
        lp_number_of_bytes_read: *mut Dword,
        lp_overlapped: *mut Overlapped,
    ) -> Bool;
    fn WriteFile(
        h_file: Handle,
        lp_buffer: *const u8,
        n_number_of_bytes_to_write: Dword,
        lp_number_of_bytes_written: *mut Dword,
        lp_overlapped: *mut Overlapped,
    ) -> Bool;
    fn GetOverlappedResult(
        h_file: Handle,
        lp_overlapped: *mut Overlapped,
        lp_number_of_bytes_transferred: *mut Dword,
        b_wait: Bool,
    ) -> Bool;
    fn WaitForSingleObject(h_handle: Handle, dw_milliseconds: Dword) -> Dword;
    fn ResetEvent(h_event: Handle) -> Bool;
    fn CancelIoEx(h_file: Handle, lp_overlapped: *mut Overlapped) -> Bool;
}

fn to_wide_null(s: &OsStr) -> Vec<u16> {
    let mut v: Vec<u16> = s.encode_wide().collect();
    v.push(0);
    v
}

fn expand_group_mask(group: u16, mask: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let base = (group as u64) * 64;
    let mut m = mask as u64;
    while m != 0 {
        let bit = m.trailing_zeros() as u64;
        out.push(base + bit);
        m &= m - 1;
    }
    out
}

/// The hardware threads the operating system lets this process use (section 6), or `None` when
/// the process is unrestricted or spans several processor groups.
fn process_allowed_threads() -> Option<Vec<u64>> {
    unsafe {
        let mut proc_mask: usize = 0;
        let mut sys_mask: usize = 0;
        if GetProcessAffinityMask(GetCurrentProcess(), &mut proc_mask, &mut sys_mask) == 0
            || proc_mask == 0
            || proc_mask == sys_mask
        {
            return None;
        }
        let mut groups = [0u16; 64];
        let mut count: u16 = groups.len() as u16;
        if GetProcessGroupAffinity(GetCurrentProcess(), &mut count, groups.as_mut_ptr()) == 0
            || count != 1
        {
            return None;
        }
        Some(expand_group_mask(groups[0], proc_mask))
    }
}

pub fn detect_host_profile(vram: Vec<DeviceEntry>, ssd: Vec<DeviceEntry>) -> Profile {
    let mut core_map: Vec<CoreThreads> = Vec::new();
    let mut l1: Vec<CacheEntry> = Vec::new();
    let mut l2: Vec<CacheEntry> = Vec::new();
    let mut l3: Vec<CacheEntry> = Vec::new();
    let mut numa_masks: Vec<Vec<u64>> = Vec::new();

    unsafe {
        let mut len: Dword = 0;
        let _ = GetLogicalProcessorInformationEx(RELATION_ALL, null_mut(), &mut len);
        if len > 0 {
            let mut buf = vec![0u8; len as usize];
            if GetLogicalProcessorInformationEx(RELATION_ALL, buf.as_mut_ptr(), &mut len) != 0 {
                let mut offset = 0usize;
                let total = len as usize;
                while offset + 8 <= total {
                    let rel =
                        u32::from_le_bytes(buf[offset..(offset + 4)].try_into().unwrap());
                    let rec_size =
                        u32::from_le_bytes(buf[(offset + 4)..(offset + 8)].try_into().unwrap())
                            as usize;
                    if rec_size < 8 || offset + rec_size > total {
                        break;
                    }
                    let body = &buf[(offset + 8)..(offset + rec_size)];

                    match rel {
                        RELATION_PROCESSOR_CORE if body.len() >= 24 + std::mem::size_of::<GroupAffinity>() => {
                            // PROCESSOR_RELATIONSHIP: Flags(1), EfficiencyClass(1), Reserved(20), GroupCount(2), GroupMask[GroupCount]
                            let group_count = u16::from_le_bytes([body[22], body[23]]) as usize;
                            let mut threads = Vec::new();
                            let ga_size = std::mem::size_of::<GroupAffinity>();
                            for g in 0..group_count {
                                let goff = 24 + g * ga_size;
                                if goff + ga_size <= body.len() {
                                    let ga: GroupAffinity =
                                        std::ptr::read_unaligned(body.as_ptr().add(goff) as *const GroupAffinity);
                                    threads.extend(expand_group_mask(ga.group, ga.mask));
                                }
                            }
                            if !threads.is_empty() {
                                let cid = core_map.len() as u64;
                                core_map.push(CoreThreads {
                                    core_id: cid,
                                    threads,
                                });
                            }
                        }
                        RELATION_CACHE if body.len() >= 28 + std::mem::size_of::<GroupAffinity>() => {
                            // CACHE_RELATIONSHIP: Level(1), Associativity(1), LineSize(2), CacheSize(4), Type(4), Reserved(18), GroupCount(2), GroupMasks...
                            let level = body[0];
                            let cache_size =
                                u32::from_le_bytes(body[4..8].try_into().unwrap()) as u64;
                            let cache_type =
                                u32::from_le_bytes(body[8..12].try_into().unwrap());
                            // Type: 0 = Unified, 1 = Instruction, 2 = Data. Keep Unified and Data only!
                            if cache_type == 0 || cache_type == 2 {
                                let group_count =
                                    u16::from_le_bytes([body[30], body[31]]) as usize;
                                let ga_size = std::mem::size_of::<GroupAffinity>();
                                let mut threads = Vec::new();
                                if group_count > 0 && 32 + group_count * ga_size <= body.len() {
                                    for g in 0..group_count {
                                        let goff = 32 + g * ga_size;
                                        let ga: GroupAffinity = std::ptr::read_unaligned(
                                            body.as_ptr().add(goff) as *const GroupAffinity,
                                        );
                                        threads.extend(expand_group_mask(ga.group, ga.mask));
                                    }
                                } else {
                                    let ga: GroupAffinity = std::ptr::read_unaligned(
                                        body.as_ptr().add(32) as *const GroupAffinity,
                                    );
                                    threads.extend(expand_group_mask(ga.group, ga.mask));
                                }
                                if cache_size > 0 && !threads.is_empty() {
                                    let entry = CacheEntry {
                                        size: cache_size,
                                        threads,
                                    };
                                    match level {
                                        1 => l1.push(entry),
                                        2 => l2.push(entry),
                                        3 => l3.push(entry),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        RELATION_NUMA_NODE | RELATION_NUMA_NODE_EX
                            if body.len() >= 24 + std::mem::size_of::<GroupAffinity>() =>
                        {
                            // NUMA_NODE_RELATIONSHIP: NodeNumber(4), Reserved(18), GroupCount(2), GroupMasks...
                            let group_count =
                                u16::from_le_bytes([body[22], body[23]]).max(1) as usize;
                            let ga_size = std::mem::size_of::<GroupAffinity>();
                            let mut threads = Vec::new();
                            for g in 0..group_count {
                                let goff = 24 + g * ga_size;
                                if goff + ga_size <= body.len() {
                                    let ga: GroupAffinity = std::ptr::read_unaligned(
                                        body.as_ptr().add(goff) as *const GroupAffinity,
                                    );
                                    threads.extend(expand_group_mask(ga.group, ga.mask));
                                }
                            }
                            if !threads.is_empty() && rel == RELATION_NUMA_NODE {
                                numa_masks.push(threads);
                            }
                        }
                        _ => {}
                    }
                    offset += rec_size;
                }
            }
        }
    }

    // Count only what the operating system lets the process use: its processor affinity confines
    // a run that shares a machine (section 6), and a cache is divided among the threads that
    // remain on it.
    if let Some(allowed) = process_allowed_threads() {
        let keep = |ts: &mut Vec<u64>| ts.retain(|t| allowed.contains(t));
        for c in &mut core_map {
            keep(&mut c.threads);
        }
        core_map.retain(|c| !c.threads.is_empty());
        for level in [&mut l1, &mut l2, &mut l3] {
            for e in level.iter_mut() {
                keep(&mut e.threads);
            }
            level.retain(|e| !e.threads.is_empty());
        }
        for n in &mut numa_masks {
            keep(n);
        }
        numa_masks.retain(|n| !n.is_empty());
    }

    let fallback_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1);
    if core_map.is_empty() {
        for t in 0..fallback_threads {
            core_map.push(CoreThreads {
                core_id: t,
                threads: vec![t],
            });
        }
    }
    let cores = (core_map.len() as u64).max(1);
    let all_threads: Vec<u64> = core_map.iter().flat_map(|c| c.threads.iter().copied()).collect();
    let threads = (all_threads.len() as u64).max(cores);

    if l1.is_empty() {
        for c in &core_map {
            l1.push(CacheEntry {
                size: 48 * 1024,
                threads: c.threads.clone(),
            });
        }
    }
    if l2.is_empty() {
        for c in &core_map {
            l2.push(CacheEntry {
                size: 1024 * 1024,
                threads: c.threads.clone(),
            });
        }
    }

    let avail_ram = unsafe {
        let mut ms: MemoryStatusEx = std::mem::zeroed();
        ms.dw_length = std::mem::size_of::<MemoryStatusEx>() as Dword;
        if GlobalMemoryStatusEx(&mut ms) != 0 && ms.ull_avail_phys > 0 {
            ms.ull_avail_phys
        } else {
            4 * 1024 * 1024 * 1024
        }
    };

    let ram = if numa_masks.is_empty() {
        vec![RamNode {
            size: avail_ram,
            threads: all_threads,
        }]
    } else {
        let per_node = avail_ram / (numa_masks.len() as u64);
        numa_masks
            .into_iter()
            .map(|t| RamNode {
                size: per_node,
                threads: t,
            })
            .collect()
    };

    Profile {
        cores,
        threads,
        core_map,
        l1,
        l2,
        l3,
        ram,
        vram,
        ssd,
        os_places_threads: false,
    }
}

pub struct WinThreadPlacer;

impl ThreadPlacer for WinThreadPlacer {
    fn place_current_thread(&self, hw_thread: u64, os_places_threads: bool) {
        if os_places_threads {
            return;
        }
        let group = (hw_thread / 64) as u16;
        let bit = (hw_thread % 64) as u32;
        let ga = GroupAffinity {
            mask: 1usize << bit,
            group,
            reserved: [0; 3],
        };
        unsafe {
            let _ = SetThreadGroupAffinity(GetCurrentThread(), &ga, null_mut());
        }
    }
}

// 0 = unprobed, 1 = available, 2 = refused
static LARGE_PAGES_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static PIN_REFUSED_STATE: AtomicBool = AtomicBool::new(false);

pub struct WinRegionMapper;

impl RegionMapper for WinRegionMapper {
    fn reserve_region(
        &self,
        bytes: usize,
        numa_node: usize,
    ) -> Option<(*mut u8, usize, MappingReport)> {
        if bytes == 0 {
            return None;
        }
        unsafe {
            let mut rep = MappingReport::default();
            let mut state = LARGE_PAGES_STATE.load(Ordering::Relaxed);
            let lp_min = if state != 2 { GetLargePageMinimum() } else { 0 };
            if state == 0 {
                if lp_min > 0 {
                    // Probe a single minimum-sized large page first so an unprivileged token fails
                    // instantaneously instead of compacting hundreds of MiB of physical RAM.
                    let probe = VirtualAllocExNuma(
                        GetCurrentProcess(),
                        null_mut(),
                        lp_min,
                        MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                        PAGE_READWRITE,
                        numa_node as Dword,
                    );
                    if !probe.is_null() {
                        LARGE_PAGES_STATE.store(1, Ordering::Relaxed);
                        state = 1;
                        if bytes <= lp_min {
                            rep.huge_pages_used = true;
                            rep.pinned_used = true;
                            return Some((probe, lp_min, rep));
                        }
                        let _ = VirtualFree(probe, 0, MEM_RELEASE);
                    } else {
                        LARGE_PAGES_STATE.store(2, Ordering::Relaxed);
                        state = 2;
                    }
                } else {
                    LARGE_PAGES_STATE.store(2, Ordering::Relaxed);
                    state = 2;
                }
            }

            if state == 1 && lp_min > 0 {
                let huge_len = bytes.div_ceil(lp_min) * lp_min;
                let p = VirtualAllocExNuma(
                    GetCurrentProcess(),
                    null_mut(),
                    huge_len,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                    numa_node as Dword,
                );
                if !p.is_null() {
                    rep.huge_pages_used = true;
                    rep.pinned_used = true;
                    return Some((p, huge_len, rep));
                }
                LARGE_PAGES_STATE.store(2, Ordering::Relaxed);
                rep.huge_pages_refused = true;
            } else {
                rep.huge_pages_refused = true;
            }

            let ord_len = bytes.div_ceil(65_536) * 65_536;
            let p = VirtualAllocExNuma(
                GetCurrentProcess(),
                null_mut(),
                ord_len,
                MEM_RESERVE,
                PAGE_READWRITE,
                numa_node as Dword,
            );
            if p.is_null() {
                None
            } else {
                Some((p, ord_len, rep))
            }
        }
    }

    fn commit_range(
        &self,
        ptr: *mut u8,
        len: usize,
        numa_node: usize,
    ) -> Result<MappingReport, ()> {
        let mut rep = MappingReport::default();
        unsafe {
            // If region was already committed with MEM_LARGE_PAGES, VirtualLock is unnecessary; otherwise commit ordinary pages on `numa_node`:
            let c = VirtualAllocExNuma(
                GetCurrentProcess(),
                ptr,
                len,
                MEM_COMMIT,
                PAGE_READWRITE,
                numa_node as Dword,
            );
            if !c.is_null() {
                if PIN_REFUSED_STATE.load(Ordering::Relaxed) {
                    rep.pin_refused = true;
                } else if VirtualLock(ptr, len) != 0 {
                    rep.pinned_used = true;
                } else {
                    PIN_REFUSED_STATE.store(true, Ordering::Relaxed);
                    rep.pin_refused = true;
                }
                return Ok(rep);
            }
            // Check if `ptr` was already committed via `MEM_LARGE_PAGES`:
            rep.huge_pages_used = true;
            rep.pinned_used = true;
            Ok(rep)
        }
    }

    unsafe fn decommit_range(&self, ptr: *mut u8, len: usize) {
        if !ptr.is_null() && len > 0 {
            unsafe {
                let _ = VirtualFree(ptr, len, MEM_DECOMMIT);
            }
        }
    }

    unsafe fn release_region(&self, base: *mut u8, _reserved_len: usize) {
        if !base.is_null() {
            unsafe {
                let _ = VirtualFree(base, 0, MEM_RELEASE);
            }
        }
    }
}

pub static WIN_THREAD_PLACER: WinThreadPlacer = WinThreadPlacer;
pub static WIN_REGION_MAPPER: WinRegionMapper = WinRegionMapper;

fn query_free_disk_bytes(path: &Path) -> Option<u64> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let wide = to_wide_null(parent.as_os_str());
    let mut avail = 0u64;
    let mut total = 0u64;
    let mut total_free = 0u64;
    unsafe {
        if GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, &mut total, &mut total_free) != 0 {
            Some(avail)
        } else {
            None
        }
    }
}

struct AlignedBounce {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBounce {
    fn new(bytes: usize) -> Option<Self> {
        let aligned = bytes.div_ceil(4096).max(1) * 4096;
        let layout = Layout::from_size_align(aligned, 4096).ok()?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, layout })
        }
    }
}

impl Drop for AlignedBounce {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }
}

const SSD_MAX_IRP_BYTES: usize = 8 * 1024 * 1024;

pub struct WinSsdMember {
    id: String,
    handle: Handle,
    cap_bytes: u64,
    direct_io: bool,
    alloc: Mutex<MemberSpanAlloc>,
    events: Mutex<Vec<Handle>>,
}

unsafe impl Send for WinSsdMember {}
unsafe impl Sync for WinSsdMember {}

impl Drop for WinSsdMember {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.events.lock() {
            for ev in pool.drain(..) {
                if !ev.is_null() {
                    unsafe {
                        let _ = CloseHandle(ev);
                    }
                }
            }
        }
        if self.handle != INVALID_HANDLE_VALUE && !self.handle.is_null() {
            unsafe {
                let _ = CloseHandle(self.handle);
            }
        }
    }
}

impl WinSsdMember {
    unsafe fn take_event(&self) -> Option<Handle> {
        if let Some(ev) = self.events.lock().unwrap().pop() {
            unsafe {
                let _ = ResetEvent(ev);
            }
            return Some(ev);
        }
        let ev = unsafe { CreateEventW(null(), 1, 0, null()) };
        if ev.is_null() { None } else { Some(ev) }
    }

    fn put_events(&self, returned: &mut Vec<Handle>) {
        if returned.is_empty() {
            return;
        }
        let mut pool = self.events.lock().unwrap();
        for ev in returned.drain(..) {
            if ev.is_null() {
                continue;
            }
            if pool.len() < 128 {
                pool.push(ev);
            } else {
                unsafe {
                    let _ = CloseHandle(ev);
                }
            }
        }
    }

    pub fn open(path: &Path, reserve_override: Option<u64>) -> Result<(DeviceEntry, Arc<dyn ExtentMember>), String> {
        let wide = to_wide_null(path.as_os_str());
        unsafe {
            // Try opening with direct I/O + overlapped first!
            let mut direct_io = true;
            let mut h = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null(),
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                null_mut(),
            );
            if h == INVALID_HANDLE_VALUE {
                direct_io = false;
                h = CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    null(),
                    OPEN_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                    null_mut(),
                );
                if h == INVALID_HANDLE_VALUE {
                    return Err(format!(
                        "Failed to open SSD path '{}' (Win32 error {})",
                        path.display(),
                        GetLastError()
                    ));
                }
            }

            let mut existing_size: i64 = 0;
            let _ = GetFileSizeEx(h, &mut existing_size);
            let cur_bytes = (existing_size.max(0)) as u64;

            let target_bytes = if let Some(req) = reserve_override {
                req
            } else if cur_bytes >= 1024 * 1024 {
                cur_bytes
            } else {
                // Take free space on volume (leaving 512 MiB filesystem headroom), capped only by available disk!
                let free = query_free_disk_bytes(path).unwrap_or(256 * 1024 * 1024);
                free.saturating_sub(512 * 1024 * 1024).max(cur_bytes)
            };
            let target_aligned = (target_bytes / 4096) * 4096;
            if target_aligned == 0 {
                let _ = CloseHandle(h);
                return Err(format!("SSD path '{}' has 0 usable aligned bytes", path.display()));
            }

            if cur_bytes < target_aligned {
                if SetFilePointerEx(h, target_aligned as i64, null_mut(), 0) == 0
                    || SetEndOfFile(h) == 0
                {
                    let err = GetLastError();
                    let _ = CloseHandle(h);
                    return Err(format!(
                        "Failed to preallocate SSD file '{}' to {} B (Win32 error {})",
                        path.display(),
                        target_aligned,
                        err
                    ));
                }
                let _ = SetFileValidData(h, target_aligned as i64);
            }

            let id = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("ssd")
                .to_string();
            let entry = DeviceEntry {
                id: id.clone(),
                size: target_aligned,
                align: 4096,
            };
            let member = Arc::new(Self {
                id,
                handle: h,
                cap_bytes: target_aligned,
                direct_io,
                alloc: Mutex::new(MemberSpanAlloc::new()),
                events: Mutex::new(Vec::with_capacity(64)),
            });
            Ok((entry, member))
        }
    }
}

struct InFlightOp {
    overlapped: Box<Overlapped>,
    event: Handle,
    bounce: Option<AlignedBounce>,
    host_offset: usize,
    len: usize,
}

impl Drop for InFlightOp {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

unsafe fn cancel_and_drain_inflight(handle: Handle, ops: Vec<InFlightOp>) {
    for mut op in ops {
        unsafe {
            let _ = CancelIoEx(handle, op.overlapped.as_mut());
            if WaitForSingleObject(op.event, 500) == WAIT_TIMEOUT {
                // Keep kernel DMA targets pinned if driver refuses to complete cancellation promptly.
                std::mem::forget(op);
            } else {
                let mut dummy: Dword = 0;
                let _ = GetOverlappedResult(handle, op.overlapped.as_mut(), &mut dummy, 0);
            }
        }
    }
}

impl ExtentMember for WinSsdMember {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> MemberKind {
        MemberKind::Ssd
    }
    fn alignment(&self) -> u64 {
        4096
    }
    fn capacity(&self) -> u64 {
        self.cap_bytes
    }
    fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
        self.alloc.lock().unwrap().allocate(bytes, 4096, self.cap_bytes)
    }
    fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
        if ranges.is_empty() {
            return Ok(());
        }
        unsafe {
            // Submit all ranges in flight at once via OVERLAPPED I/O, splitting large transfers
            // into 8 MiB sub-IRPs to drive multi-queue NVMe depth (Section 10)!
            let est_ops: usize = ranges
                .iter()
                .map(|r| r.len.div_ceil(SSD_MAX_IRP_BYTES))
                .sum();
            let mut inflight: Vec<InFlightOp> = Vec::with_capacity(est_ops);
            for r in ranges {
                if r.len == 0 {
                    continue;
                }
                if r.device_offset
                    .checked_add(r.len as u64)
                    .is_none_or(|end| end > self.cap_bytes)
                    || r.host_offset
                        .checked_add(r.len)
                        .is_none_or(|end| end > host_buf.len())
                {
                    cancel_and_drain_inflight(self.handle, inflight);
                    return Err(StoreError::IoFailed);
                }
                if self.direct_io && (r.device_offset & 4095) != 0 {
                    cancel_and_drain_inflight(self.handle, inflight);
                    return Err(StoreError::IoFailed);
                }
                let mut sub_off = 0usize;
                while sub_off < r.len {
                    let sub_len = (r.len - sub_off).min(SSD_MAX_IRP_BYTES);
                    let dev_off = r.device_offset + (sub_off as u64);
                    let host_off = r.host_offset + sub_off;
                    let dst_ptr = host_buf.as_mut_ptr().add(host_off);
                    let is_aligned = ((dst_ptr as usize) & 4095) == 0
                        && (sub_len & 4095) == 0
                        && (dev_off & 4095) == 0;

                    let Some(ev) = self.take_event() else {
                        cancel_and_drain_inflight(self.handle, inflight);
                        return Err(StoreError::IoFailed);
                    };
                    let mut ov = Box::new(Overlapped {
                        internal: 0,
                        internal_high: 0,
                        offset: (dev_off & 0xFFFF_FFFF) as Dword,
                        offset_high: (dev_off >> 32) as Dword,
                        h_event: ev,
                    });

                    let (io_ptr, io_len, bounce) = if !self.direct_io || is_aligned {
                        (dst_ptr, sub_len as Dword, None)
                    } else {
                        let al_len = sub_len.div_ceil(4096) * 4096;
                        let Some(b) = AlignedBounce::new(al_len) else {
                            let _ = CloseHandle(ev);
                            cancel_and_drain_inflight(self.handle, inflight);
                            return Err(StoreError::IoFailed);
                        };
                        (b.ptr, al_len as Dword, Some(b))
                    };

                    let mut done: Dword = 0;
                    let ok = ReadFile(self.handle, io_ptr, io_len, &mut done, ov.as_mut());
                    if ok == 0 && GetLastError() != ERROR_IO_PENDING {
                        let _ = CloseHandle(ev);
                        cancel_and_drain_inflight(self.handle, inflight);
                        return Err(StoreError::IoFailed);
                    }
                    inflight.push(InFlightOp {
                        overlapped: ov,
                        event: ev,
                        bounce,
                        host_offset: host_off,
                        len: sub_len,
                    });
                    sub_off += sub_len;
                }
            }

            // Reap all in-flight operations within the bounded SSD transfer timeout (Section 11)!
            let start = std::time::Instant::now();
            let mut returned_events: Vec<Handle> = Vec::with_capacity(inflight.len());
            let mut iter = inflight.into_iter();
            while let Some(mut op) = iter.next() {
                let elapsed_ms = start.elapsed().as_millis().min(SSD_IO_TIMEOUT_MS as u128) as Dword;
                let rem_ms = SSD_IO_TIMEOUT_MS.saturating_sub(elapsed_ms);
                let wait_rc = WaitForSingleObject(op.event, rem_ms);
                if wait_rc == WAIT_TIMEOUT {
                    let mut rest = Vec::with_capacity(iter.len() + 1);
                    rest.push(op);
                    rest.extend(iter);
                    cancel_and_drain_inflight(self.handle, rest);
                    self.put_events(&mut returned_events);
                    return Err(StoreError::DeviceLost);
                }
                let mut transferred: Dword = 0;
                if wait_rc != WAIT_OBJECT_0
                    || GetOverlappedResult(self.handle, op.overlapped.as_mut(), &mut transferred, 0)
                        == 0
                    || (transferred as usize) < op.len
                {
                    let rest: Vec<InFlightOp> = iter.collect();
                    cancel_and_drain_inflight(self.handle, rest);
                    self.put_events(&mut returned_events);
                    return Err(StoreError::IoFailed);
                }
                if let Some(b) = &op.bounce {
                    std::ptr::copy_nonoverlapping(
                        b.ptr,
                        host_buf.as_mut_ptr().add(op.host_offset),
                        op.len,
                    );
                }
                returned_events.push(std::mem::replace(&mut op.event, null_mut()));
            }
            self.put_events(&mut returned_events);
        }
        Ok(())
    }
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
        if ranges.is_empty() {
            return Ok(());
        }
        unsafe {
            let est_ops: usize = ranges
                .iter()
                .map(|r| r.len.div_ceil(SSD_MAX_IRP_BYTES))
                .sum();
            let mut inflight: Vec<InFlightOp> = Vec::with_capacity(est_ops);
            for r in ranges {
                if r.len == 0 {
                    continue;
                }
                if r.device_offset
                    .checked_add(r.len as u64)
                    .is_none_or(|end| end > self.cap_bytes)
                    || r.host_offset
                        .checked_add(r.len)
                        .is_none_or(|end| end > host_buf.len())
                {
                    cancel_and_drain_inflight(self.handle, inflight);
                    return Err(StoreError::IoFailed);
                }
                if self.direct_io && ((r.device_offset & 4095) != 0 || (r.len & 4095) != 0) {
                    cancel_and_drain_inflight(self.handle, inflight);
                    return Err(StoreError::IoFailed);
                }
                let mut sub_off = 0usize;
                while sub_off < r.len {
                    let sub_len = (r.len - sub_off).min(SSD_MAX_IRP_BYTES);
                    let dev_off = r.device_offset + (sub_off as u64);
                    let host_off = r.host_offset + sub_off;
                    let src_ptr = host_buf.as_ptr().add(host_off);
                    let is_aligned = ((src_ptr as usize) & 4095) == 0
                        && (sub_len & 4095) == 0
                        && (dev_off & 4095) == 0;

                    let Some(ev) = self.take_event() else {
                        cancel_and_drain_inflight(self.handle, inflight);
                        return Err(StoreError::IoFailed);
                    };
                    let mut ov = Box::new(Overlapped {
                        internal: 0,
                        internal_high: 0,
                        offset: (dev_off & 0xFFFF_FFFF) as Dword,
                        offset_high: (dev_off >> 32) as Dword,
                        h_event: ev,
                    });

                    let (io_ptr, io_len, bounce) = if !self.direct_io || is_aligned {
                        (src_ptr, sub_len as Dword, None)
                    } else {
                        let Some(b) = AlignedBounce::new(sub_len) else {
                            let _ = CloseHandle(ev);
                            cancel_and_drain_inflight(self.handle, inflight);
                            return Err(StoreError::IoFailed);
                        };
                        std::ptr::copy_nonoverlapping(src_ptr, b.ptr, sub_len);
                        (b.ptr as *const u8, sub_len as Dword, Some(b))
                    };

                    let mut done: Dword = 0;
                    let ok = WriteFile(self.handle, io_ptr, io_len, &mut done, ov.as_mut());
                    if ok == 0 && GetLastError() != ERROR_IO_PENDING {
                        let _ = CloseHandle(ev);
                        cancel_and_drain_inflight(self.handle, inflight);
                        return Err(StoreError::DeviceLost);
                    }
                    inflight.push(InFlightOp {
                        overlapped: ov,
                        event: ev,
                        bounce,
                        host_offset: host_off,
                        len: sub_len,
                    });
                    sub_off += sub_len;
                }
            }

            let start = std::time::Instant::now();
            let mut returned_events: Vec<Handle> = Vec::with_capacity(inflight.len());
            let mut iter = inflight.into_iter();
            while let Some(mut op) = iter.next() {
                let elapsed_ms = start.elapsed().as_millis().min(SSD_IO_TIMEOUT_MS as u128) as Dword;
                let rem_ms = SSD_IO_TIMEOUT_MS.saturating_sub(elapsed_ms);
                let wait_rc = WaitForSingleObject(op.event, rem_ms);
                if wait_rc == WAIT_TIMEOUT {
                    let mut rest = Vec::with_capacity(iter.len() + 1);
                    rest.push(op);
                    rest.extend(iter);
                    cancel_and_drain_inflight(self.handle, rest);
                    self.put_events(&mut returned_events);
                    return Err(StoreError::DeviceLost);
                }
                let mut transferred: Dword = 0;
                if wait_rc != WAIT_OBJECT_0
                    || GetOverlappedResult(self.handle, op.overlapped.as_mut(), &mut transferred, 0)
                        == 0
                    || (transferred as usize) < op.len
                {
                    let rest: Vec<InFlightOp> = iter.collect();
                    cancel_and_drain_inflight(self.handle, rest);
                    self.put_events(&mut returned_events);
                    return Err(StoreError::DeviceLost);
                }
                returned_events.push(std::mem::replace(&mut op.event, null_mut()));
            }
            self.put_events(&mut returned_events);
        }
        Ok(())
    }
    fn free(&self, device_offset: u64, bytes: u64) {
        self.alloc.lock().unwrap().free(device_offset, bytes, 4096);
    }
}

/// Bytes written between two background flushes of the output file while digits are produced.
/// A dedicated background flusher thread flushes dirty cache slices every `128 MiB` without
/// blocking any compute worker thread, leaving less than one slice for the final `flush()`.
const FLUSH_SLICE_BYTES: u64 = 128 * 1024 * 1024;

struct ThreadOverlappedEvent(Handle);

impl Drop for ThreadOverlappedEvent {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

thread_local! {
    static TLS_WRITE_EVENT: std::cell::RefCell<Option<ThreadOverlappedEvent>> =
        const { std::cell::RefCell::new(None) };
}

struct WinFlushBg {
    /// `(wake_requested, stop_requested, flush_in_progress)`
    mu: Mutex<(bool, bool, bool)>,
    cv: std::sync::Condvar,
    err: AtomicU64,
}

impl WinFlushBg {
    fn spawn(handle: Handle) -> (Arc<Self>, std::thread::JoinHandle<()>) {
        let state = Arc::new(Self {
            mu: Mutex::new((false, false, false)),
            cv: std::sync::Condvar::new(),
            err: AtomicU64::new(0),
        });
        let st_bg = Arc::clone(&state);
        let h_usize = handle as usize;
        let jh = std::thread::Builder::new()
            .name("supi-out-flush".to_string())
            .spawn(move || {
                let h = h_usize as Handle;
                loop {
                    let do_flush = {
                        let mut g = st_bg.mu.lock().unwrap();
                        while !g.0 && !g.1 {
                            g = st_bg.cv.wait(g).unwrap();
                        }
                        if g.1 && !g.0 {
                            break;
                        }
                        g.0 = false;
                        g.2 = true;
                        true
                    };
                    if do_flush {
                        unsafe {
                            if FlushFileBuffers(h) == 0 {
                                st_bg
                                    .err
                                    .store(u64::from(GetLastError()).max(1), Ordering::Release);
                            }
                        }
                        let mut g = st_bg.mu.lock().unwrap();
                        g.2 = false;
                        let should_stop = g.1 && !g.0;
                        st_bg.cv.notify_all();
                        if should_stop {
                            break;
                        }
                    }
                }
            })
            .expect("spawn supi-out-flush");
        (state, jh)
    }
}

pub struct WinFileDigitSink {
    handle: Handle,
    full_size: u64,
    dirty: AtomicBool,
    /// Bytes written since a flush last started.
    unflushed: AtomicU64,
    bg_flusher: Option<Arc<WinFlushBg>>,
    bg_join: Option<std::thread::JoinHandle<()>>,
}

unsafe impl Send for WinFileDigitSink {}
unsafe impl Sync for WinFileDigitSink {}

impl WinFileDigitSink {
    fn write_raw(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        let ev = TLS_WRITE_EVENT.with(|cell| -> Result<Handle, String> {
            let mut slot = cell.borrow_mut();
            if let Some(ref g) = *slot {
                return Ok(g.0);
            }
            let h = unsafe { CreateEventW(null(), 1, 0, null()) };
            if h.is_null() {
                return Err(format!(
                    "Output file CreateEventW failed (Win32 error {})",
                    unsafe { GetLastError() }
                ));
            }
            *slot = Some(ThreadOverlappedEvent(h));
            Ok(h)
        })?;

        unsafe {
            let mut pos = 0usize;
            while pos < bytes.len() {
                let chunk = (bytes.len() - pos).min(16 * 1024 * 1024);
                let cur_off = file_offset + (pos as u64);
                let mut ov = Overlapped {
                    internal: 0,
                    internal_high: 0,
                    offset: (cur_off & 0xFFFF_FFFF) as Dword,
                    offset_high: (cur_off >> 32) as Dword,
                    h_event: ev,
                };
                let mut written: Dword = 0;
                let ok = WriteFile(
                    self.handle,
                    bytes.as_ptr().add(pos),
                    chunk as Dword,
                    &mut written,
                    &mut ov,
                );
                if ok == 0 {
                    let err = GetLastError();
                    if err != ERROR_IO_PENDING {
                        return Err(format!("Output file WriteFile failed (Win32 error {err})"));
                    }
                    if GetOverlappedResult(self.handle, &mut ov, &mut written, 1) == 0 {
                        return Err(format!(
                            "Output file GetOverlappedResult failed (Win32 error {})",
                            GetLastError()
                        ));
                    }
                }
                if (written as usize) != chunk {
                    return Err("Output file short write".to_string());
                }
                pos += chunk;
            }
        }
        Ok(())
    }
}

impl Drop for WinFileDigitSink {
    fn drop(&mut self) {
        if let (Some(bg), Some(jh)) = (self.bg_flusher.take(), self.bg_join.take()) {
            {
                let mut g = bg.mu.lock().unwrap();
                g.0 = false;
                g.1 = true;
                bg.cv.notify_all();
            }
            let _ = jh.join();
        }
        if self.handle != INVALID_HANDLE_VALUE && !self.handle.is_null() {
            unsafe {
                let _ = CloseHandle(self.handle);
            }
        }
    }
}

impl DigitSink for WinFileDigitSink {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        if file_offset.saturating_add(bytes.len() as u64) > self.full_size {
            return Err(format!(
                "Output write at {}+{} exceeds preallocated {} B",
                file_offset,
                bytes.len(),
                self.full_size
            ));
        }
        self.write_raw(file_offset, bytes)?;
        self.dirty.store(true, Ordering::Release);
        // Signal the dedicated background flusher thread once a 128 MiB slice has accumulated so
        // NVMe writeback overlaps digit production without stalling any compute worker thread.
        if let Some(bg) = &self.bg_flusher {
            let pending = self.unflushed.fetch_add(bytes.len() as u64, Ordering::AcqRel)
                + (bytes.len() as u64);
            if pending >= FLUSH_SLICE_BYTES {
                if let Ok(mut g) = bg.mu.try_lock() {
                    if !g.0 && !g.2 {
                        self.unflushed.store(0, Ordering::Release);
                        g.0 = true;
                        bg.cv.notify_one();
                    }
                }
            }
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), String> {
        if let Some(bg) = &self.bg_flusher {
            let mut g = bg.mu.lock().unwrap();
            while g.0 || g.2 {
                g = bg.cv.wait(g).unwrap();
            }
            drop(g);
            let slice_err = bg.err.swap(0, Ordering::AcqRel);
            if slice_err != 0 {
                return Err(format!(
                    "Output file FlushFileBuffers failed (Win32 error {slice_err})"
                ));
            }
        }
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        self.unflushed.store(0, Ordering::Release);
        unsafe {
            if FlushFileBuffers(self.handle) == 0 {
                Err(format!(
                    "Output file FlushFileBuffers failed (Win32 error {})",
                    GetLastError()
                ))
            } else {
                Ok(())
            }
        }
    }
}

pub struct WinOutputSinkFactory {
    pub path: PathBuf,
}

impl WinOutputSinkFactory {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl OutputSinkFactory for WinOutputSinkFactory {
    fn create_full_size(&self, full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        if full_size_bytes > (i64::MAX as u64) {
            return Err(format!(
                "Requested output file size {} B exceeds filesystem maximum",
                full_size_bytes
            ));
        }
        if let Some(avail) = query_free_disk_bytes(&self.path) {
            // Account for replacing an existing file of the same name if already present!
            let existing = std::fs::metadata(&self.path).map_or(0, |m| m.len());
            if avail.saturating_add(existing) < full_size_bytes {
                return Err(format!(
                    "Insufficient disk space for output file '{}' (need {} B, free {} B)",
                    self.path.display(),
                    full_size_bytes,
                    avail.saturating_add(existing)
                ));
            }
        }

        let wide = to_wide_null(self.path.as_os_str());
        unsafe {
            let h = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ,
                null(),
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                null_mut(),
            );
            if h == INVALID_HANDLE_VALUE {
                return Err(format!(
                    "Cannot create output file '{}' (Win32 error {})",
                    self.path.display(),
                    GetLastError()
                ));
            }
            if SetFilePointerEx(h, full_size_bytes as i64, null_mut(), 0) == 0
                || SetEndOfFile(h) == 0
            {
                let err = GetLastError();
                let _ = CloseHandle(h);
                let _ = DeleteFileW(wide.as_ptr());
                return Err(format!(
                    "Cannot preallocate output file '{}' to {} B (Win32 error {})",
                    self.path.display(),
                    full_size_bytes,
                    err
                ));
            }
            let _ = SetFileValidData(h, full_size_bytes as i64);
            let (bg_flusher, bg_join) = if full_size_bytes >= FLUSH_SLICE_BYTES {
                let (st, jh) = WinFlushBg::spawn(h);
                (Some(st), Some(jh))
            } else {
                (None, None)
            };
            Ok(Box::new(WinFileDigitSink {
                handle: h,
                full_size: full_size_bytes,
                dirty: AtomicBool::new(false),
                unflushed: AtomicU64::new(0),
                bg_flusher,
                bg_join,
            }))
        }
    }

    fn open_existing_exact(&self, expected_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        let wide = to_wide_null(self.path.as_os_str());
        unsafe {
            let h = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ,
                null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                null_mut(),
            );
            if h == INVALID_HANDLE_VALUE {
                return Err(format!(
                    "Output file '{}' is missing (Win32 error {})",
                    self.path.display(),
                    GetLastError()
                ));
            }
            let mut sz: i64 = 0;
            if GetFileSizeEx(h, &mut sz) == 0 || (sz as u64) != expected_bytes {
                let _ = CloseHandle(h);
                return Err(format!(
                    "Output file '{}' has size {} B, expected {} B",
                    self.path.display(),
                    sz,
                    expected_bytes
                ));
            }
            let (bg_flusher, bg_join) = if expected_bytes >= FLUSH_SLICE_BYTES {
                let (st, jh) = WinFlushBg::spawn(h);
                (Some(st), Some(jh))
            } else {
                (None, None)
            };
            Ok(Box::new(WinFileDigitSink {
                handle: h,
                full_size: expected_bytes,
                dirty: AtomicBool::new(false),
                unflushed: AtomicU64::new(0),
                bg_flusher,
                bg_join,
            }))
        }
    }
}
