//! Contract:
//! - Inputs: macOS `sysctlbyname` topology queries, OS-placed worker policy (`os_places_threads =
//!   true`), `mmap` / `mlock` reservations, SSD paths, and output file paths.
//! - Outputs: Populated `Profile`, `MacThreadPlacer`, `MacRegionMapper`, `MacSsdMember` (with
//!   coalescing `MemberSpanAlloc` reclamation), and `MacOutputSinkFactory`.
//! - Data format: Standard library + raw `libSystem` FFI symbols with zero external crates.
//! - Work & bytes moved: Startup `sysctl` discovery; `pread`/`pwrite` random-access file I/O with
//!   5 s call timeout and write-failure `DeviceLost` escalation.

use std::ffi::c_char;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(not(unix))]
use crate::unix_stand_in::FileExt;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

use engine::OutputSinkFactory;
use engine::arena::{MappingReport, RegionMapper};
use engine::par::ThreadPlacer;
use engine::radix::DigitSink;
use engine::store::{ByteRange, ExtentMember, MemberKind, MemberSpanAlloc, StoreError};
use engine::tier::{CacheEntry, CoreThreads, DeviceEntry, Profile, RamNode};

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut u8,
        oldlenp: *mut usize,
        newp: *const u8,
        newlen: usize,
    ) -> i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn mlock(addr: *const u8, len: usize) -> i32;
}

fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    unsafe {
        let mut val = 0u64;
        let mut sz = std::mem::size_of::<u64>();
        if sysctlbyname(name.as_ptr(), (&mut val as *mut u64) as *mut u8, &mut sz, null(), 0) == 0 {
            if sz == 4 {
                Some((val as u32) as u64)
            } else {
                Some(val)
            }
        } else {
            None
        }
    }
}

pub fn detect_host_profile(vram: Vec<DeviceEntry>, ssd: Vec<DeviceEntry>) -> Profile {
    let cores = sysctl_u64(c"hw.physicalcpu").unwrap_or(1).max(1);
    let threads = sysctl_u64(c"hw.logicalcpu").unwrap_or(cores).max(cores);
    let memsize = sysctl_u64(c"hw.memsize").unwrap_or(8 * 1024 * 1024 * 1024);
    let l1_sz = sysctl_u64(c"hw.perflevel0.l1dcachesize").unwrap_or(128 * 1024);
    let l2_sz = sysctl_u64(c"hw.perflevel0.l2cachesize").unwrap_or(16 * 1024 * 1024);

    let mut core_map = Vec::new();
    let mut l1 = Vec::new();
    let mut all_t = Vec::new();
    for t in 0..threads {
        all_t.push(t);
        core_map.push(CoreThreads {
            core_id: t,
            threads: vec![t],
        });
        l1.push(CacheEntry {
            size: l1_sz,
            threads: vec![t],
        });
    }
    let l2 = vec![CacheEntry {
        size: l2_sz,
        threads: all_t.clone(),
    }];
    let ram = vec![RamNode {
        size: (memsize * 3) / 4,
        threads: all_t,
    }];

    Profile {
        cores,
        threads,
        core_map,
        l1,
        l2,
        l3: Vec::new(),
        ram,
        vram,
        ssd,
        // Section 6: "On macOS, which exposes no call to place a thread on a chosen core, the operating system places the workers."
        os_places_threads: true,
    }
}

pub struct MacThreadPlacer;
impl ThreadPlacer for MacThreadPlacer {
    fn place_current_thread(&self, _hw_thread: u64, _os_places_threads: bool) {}
}

pub struct MacRegionMapper;
impl RegionMapper for MacRegionMapper {
    fn reserve_region(
        &self,
        bytes: usize,
        _numa_node: usize,
    ) -> Option<(*mut u8, usize, MappingReport)> {
        if bytes == 0 {
            return None;
        }
        let len = bytes.div_ceil(2 * 1024 * 1024) * (2 * 1024 * 1024);
        unsafe {
            let p = mmap(null_mut(), len, 3, 0x1002, -1, 0);
            if (p as isize) == -1 || p.is_null() {
                None
            } else {
                Some((
                    p,
                    len,
                    MappingReport {
                        huge_pages_refused: true,
                        ..MappingReport::default()
                    },
                ))
            }
        }
    }

    fn commit_range(
        &self,
        ptr: *mut u8,
        len: usize,
        _numa_node: usize,
    ) -> Result<MappingReport, ()> {
        let mut rep = MappingReport::default();
        unsafe {
            if mlock(ptr, len) == 0 {
                rep.pinned_used = true;
            } else {
                rep.pin_refused = true;
            }
        }
        Ok(rep)
    }

    unsafe fn release_region(&self, base: *mut u8, reserved_len: usize) {
        if !base.is_null() && reserved_len > 0 {
            unsafe {
                let _ = munmap(base, reserved_len);
            }
        }
    }
}

pub static MAC_THREAD_PLACER: MacThreadPlacer = MacThreadPlacer;
pub static MAC_REGION_MAPPER: MacRegionMapper = MacRegionMapper;

pub struct MacSsdMember {
    id: String,
    file: File,
    cap_bytes: u64,
    alloc: Mutex<MemberSpanAlloc>,
}

impl MacSsdMember {
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

impl ExtentMember for MacSsdMember {
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

struct MacDigitSink {
    file: File,
    full_size: u64,
}

impl DigitSink for MacDigitSink {
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

pub struct MacOutputSinkFactory {
    pub path: PathBuf,
}

impl MacOutputSinkFactory {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl OutputSinkFactory for MacOutputSinkFactory {
    fn create_full_size(&self, full_size_bytes: u64) -> Result<Box<dyn DigitSink>, String> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| format!("Cannot create '{}': {e}", self.path.display()))?;
        f.set_len(full_size_bytes)
            .map_err(|e| format!("Cannot preallocate '{}': {e}", self.path.display()))?;
        Ok(Box::new(MacDigitSink {
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
        Ok(Box::new(MacDigitSink {
            file: f,
            full_size: expected_bytes,
        }))
    }
}
