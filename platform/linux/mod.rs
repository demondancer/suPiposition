//! Contract:
//! - Inputs: Linux sysfs/procfs topology (`/sys/devices/system/cpu`, `/proc/meminfo`), thread
//!   affinity requests (`sched_setaffinity`), huge-page `mmap` / `madvise` / `mlock` reservations,
//!   SSD paths, and output file paths.
//! - Outputs: Populated `Profile`, `LinuxThreadPlacer`, `LinuxRegionMapper`, `LinuxSsdMember`
//!   (with coalescing `MemberSpanAlloc` reclamation), and `LinuxOutputSinkFactory`.
//! - Data format: Standard library + raw `libc` FFI symbols with zero external crates.
//! - Work & bytes moved: Startup sysfs topology discovery; `pread`/`pwrite` random-access file I/O
//!   with 5 s call timeout and write-failure `DeviceLost` escalation.

use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(not(unix))]
use crate::unix_stand_in::FileExt;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::{Arc, Mutex};

use engine::OutputSinkFactory;
use engine::arena::{MappingReport, RegionMapper};
use engine::par::ThreadPlacer;
use engine::radix::DigitSink;
use engine::store::{ByteRange, ExtentMember, MemberKind, MemberSpanAlloc, StoreError};
use engine::tier::{CacheEntry, CoreThreads, DeviceEntry, Profile, RamNode};

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn mlock(addr: *const u8, len: usize) -> i32;
    fn munlock(addr: *const u8, len: usize) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    fn syscall(num: i64, ...) -> i64;
}

/// `mbind` has no wrapper in the C library itself; its number is that of the system call table.
#[cfg(target_arch = "x86_64")]
const SYS_MBIND: i64 = 237;
#[cfg(not(target_arch = "x86_64"))]
const SYS_MBIND: i64 = 235;
const MPOL_PREFERRED: usize = 1;

/// The system's own number of each memory node of the profile, in the profile's order.
static MEMORY_NODE_IDS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

/// Asks for the pages of `[ptr, ptr + len)` to be taken from the memory node the profile lists
/// at `node_index`, whichever thread touches them first; where that node is full the kernel
/// takes them elsewhere. A system with one node, or one that did not name its nodes, is left to
/// the thread that touches a page first.
fn prefer_node(ptr: *mut u8, len: usize, node_index: usize) {
    let Some(ids) = MEMORY_NODE_IDS.get() else {
        return;
    };
    if ids.len() < 2 {
        return;
    }
    let Some(&node) = ids.get(node_index) else {
        return;
    };
    if node >= (1 << 16) {
        return;
    }
    let words = (node / 64) as usize + 1;
    let mut mask = vec![0u64; words];
    mask[(node / 64) as usize] = 1u64 << (node % 64);
    unsafe {
        // (the kernel reads one bit fewer than the count it is given)
        let _ = syscall(
            SYS_MBIND,
            ptr as usize,
            len,
            MPOL_PREFERRED,
            mask.as_ptr() as usize,
            words * 64 + 1,
            0usize,
        );
    }
}

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_HUGETLB: i32 = 0x40000;
const MADV_HUGEPAGE: i32 = 14;
const MADV_DONTNEED: i32 = 4;

pub fn detect_host_profile(vram: Vec<DeviceEntry>, ssd: Vec<DeviceEntry>) -> Profile {
    // What the system itself says of the threads this process may use, of every cache and the
    // threads that share it, and of its memory nodes (`sysfs`). The figures further down are
    // only for a system that does not say.
    if let Some((mut profile, node_ids)) = crate::sysfs::profile_and_nodes_from_reader(&|path| {
        std::fs::read_to_string(Path::new("/").join(path)).ok()
    }) {
        let _ = MEMORY_NODE_IDS.set(node_ids);
        profile.vram = vram;
        profile.ssd = ssd;
        return profile;
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1);
    let mut avail_ram = 4 * 1024 * 1024 * 1024u64;
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        avail_ram = kb * 1024;
                    }
                }
            }
        }
    }
    let mut core_map = Vec::new();
    let mut l1 = Vec::new();
    let mut l2 = Vec::new();
    let mut all_t = Vec::new();
    for t in 0..threads {
        all_t.push(t);
        core_map.push(CoreThreads {
            core_id: t,
            threads: vec![t],
        });
        l1.push(CacheEntry {
            size: 48 * 1024,
            threads: vec![t],
        });
        l2.push(CacheEntry {
            size: 1024 * 1024,
            threads: vec![t],
        });
    }
    let l3 = vec![CacheEntry {
        size: 32 * 1024 * 1024,
        threads: all_t.clone(),
    }];
    let ram = vec![RamNode {
        size: avail_ram,
        threads: all_t,
    }];
    Profile {
        cores: threads,
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

pub struct LinuxThreadPlacer;
impl ThreadPlacer for LinuxThreadPlacer {
    fn place_current_thread(&self, hw_thread: u64, os_places_threads: bool) {
        // A mask as long as the thread's number needs: systems number their threads well
        // past 1024.
        if os_places_threads || hw_thread >= (1 << 22) {
            return;
        }
        let words = ((hw_thread / 64) as usize + 1).max(16);
        let mut mask = vec![0u64; words];
        mask[(hw_thread / 64) as usize] = 1u64 << (hw_thread % 64);
        unsafe {
            let _ = sched_setaffinity(0, words * 8, mask.as_ptr());
        }
    }
}

pub struct LinuxRegionMapper;
impl RegionMapper for LinuxRegionMapper {
    fn reserve_region(
        &self,
        bytes: usize,
        numa_node: usize,
    ) -> Option<(*mut u8, usize, MappingReport)> {
        if bytes == 0 {
            return None;
        }
        let len = bytes.div_ceil(2 * 1024 * 1024) * (2 * 1024 * 1024);
        let mut rep = MappingReport::default();
        unsafe {
            let p_huge = mmap(
                null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            );
            if (p_huge as isize) != -1 && !p_huge.is_null() {
                rep.huge_pages_used = true;
                prefer_node(p_huge, len, numa_node);
                return Some((p_huge, len, rep));
            }
            let p = mmap(
                null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            if (p as isize) == -1 || p.is_null() {
                return None;
            }
            prefer_node(p, len, numa_node);
            if madvise(p, len, MADV_HUGEPAGE) == 0 {
                rep.huge_pages_used = true;
            } else {
                rep.huge_pages_refused = true;
            }
            Some((p, len, rep))
        }
    }

    fn commit_range(
        &self,
        ptr: *mut u8,
        len: usize,
        numa_node: usize,
    ) -> Result<MappingReport, ()> {
        let mut rep = MappingReport::default();
        // (pinning takes the pages at once, by this thread: the node is named first)
        prefer_node(ptr, len, numa_node);
        unsafe {
            if mlock(ptr, len) == 0 {
                rep.pinned_used = true;
            } else {
                rep.pin_refused = true;
            }
        }
        Ok(rep)
    }

    /// Gives the pages back and keeps the range: the next touch finds fresh pages. Pinned pages
    /// are let go first, which the kernel asks for. Without this the memory the engine counts as
    /// returned would stay resident, and a job held to a memory limit would pass it.
    unsafe fn decommit_range(&self, ptr: *mut u8, len: usize) {
        if !ptr.is_null() && len > 0 {
            unsafe {
                let _ = munlock(ptr, len);
                let _ = madvise(ptr, len, MADV_DONTNEED);
            }
        }
    }

    unsafe fn release_region(&self, base: *mut u8, reserved_len: usize) {
        if !base.is_null() && reserved_len > 0 {
            unsafe {
                let _ = munmap(base, reserved_len);
            }
        }
    }
}

pub static LINUX_THREAD_PLACER: LinuxThreadPlacer = LinuxThreadPlacer;
pub static LINUX_REGION_MAPPER: LinuxRegionMapper = LinuxRegionMapper;

pub struct LinuxSsdMember {
    id: String,
    file: File,
    cap_bytes: u64,
    alloc: Mutex<MemberSpanAlloc>,
}

impl LinuxSsdMember {
    pub fn open(
        path: &Path,
        reserve_override: Option<u64>,
    ) -> Result<(DeviceEntry, Arc<dyn ExtentMember>), String> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| format!("Failed to open SSD '{}': {e}", path.display()))?;
        let cur = f.metadata().map_or(0, |m| m.len());
        let cap = (reserve_override.unwrap_or(cur.max(256 * 1024 * 1024)) / 4096) * 4096;
        if cur < cap {
            f.set_len(cap)
                .map_err(|e| format!("Failed to size SSD '{}': {e}", path.display()))?;
        }
        let id = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("ssd")
            .to_string();
        let entry = DeviceEntry {
            id: id.clone(),
            size: cap,
            align: 4096,
        };
        Ok((
            entry,
            Arc::new(Self {
                id,
                file: f,
                cap_bytes: cap,
                alloc: Mutex::new(MemberSpanAlloc::new()),
            }),
        ))
    }
}

impl ExtentMember for LinuxSsdMember {
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
        let t0 = std::time::Instant::now();
        for r in ranges {
            self.file
                .read_exact_at(&mut host_buf[r.host_offset..(r.host_offset + r.len)], r.device_offset)
                .map_err(|_| StoreError::IoFailed)?;
            if t0.elapsed().as_secs_f64() > 5.0 {
                return Err(StoreError::DeviceLost);
            }
        }
        Ok(())
    }
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
        let t0 = std::time::Instant::now();
        for r in ranges {
            self.file
                .write_all_at(&host_buf[r.host_offset..(r.host_offset + r.len)], r.device_offset)
                .map_err(|_| StoreError::DeviceLost)?;
            if t0.elapsed().as_secs_f64() > 5.0 {
                return Err(StoreError::DeviceLost);
            }
        }
        Ok(())
    }
    fn free(&self, device_offset: u64, bytes: u64) {
        self.alloc.lock().unwrap().free(device_offset, bytes, 4096);
    }
}

struct LinuxDigitSink {
    file: File,
    full_size: u64,
}

impl DigitSink for LinuxDigitSink {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        if file_offset + (bytes.len() as u64) > self.full_size {
            return Err("Write past preallocated size".to_string());
        }
        self.file
            .write_all_at(bytes, file_offset)
            .map_err(|e| format!("Output write failed: {e}"))
    }
    fn flush(&self) -> Result<(), String> {
        self.file
            .sync_all()
            .map_err(|e| format!("Output sync failed: {e}"))
    }
}

pub struct LinuxOutputSinkFactory {
    pub path: PathBuf,
}

impl LinuxOutputSinkFactory {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl OutputSinkFactory for LinuxOutputSinkFactory {
    fn create_full_size(&self, full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| format!("Cannot create '{}': {e}", self.path.display()))?;
        f.set_len(full_size_bytes)
            .map_err(|e| format!("Cannot preallocate '{}' to {full_size_bytes} B: {e}", self.path.display()))?;
        Ok(Box::new(LinuxDigitSink {
            file: f,
            full_size: full_size_bytes,
        }))
    }
    fn open_existing_exact(&self, expected_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| format!("Output '{}' missing: {e}", self.path.display()))?;
        let sz = f.metadata().map_err(|e| e.to_string())?.len();
        if sz != expected_bytes {
            return Err(format!("Output '{}' size {sz} != {expected_bytes}", self.path.display()));
        }
        Ok(Box::new(LinuxDigitSink {
            file: f,
            full_size: expected_bytes,
        }))
    }
}
