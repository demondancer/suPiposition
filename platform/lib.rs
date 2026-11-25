//! Contract:
//! - Inputs: SSD `(PathBuf, Option<u64>)` path and optional preallocation-byte specs, `enable_vram`
//!   flag, and output file path.
//! - Outputs: OS-neutral `PlatformEnv` uniting `Profile`, `RegionMapper`, `ThreadPlacer`, opened
//!   SSD + discrete-GPU VRAM `ExtentMember`s, and `OutputSinkFactory`.
//! - Data format: Dispatches `#[cfg(target_os = ...)]` across `windows/`, `linux/`, and `macos/`
//!   plus Vulkan 1.4 `vram.rs`.
//! - Work & bytes moved: Startup hardware topology + Vulkan 1.4 transfer-queue probing and OS
//!   virtual-memory/direct-I/O handle binding.

pub mod sysfs;
pub mod vram;

// The Linux and macOS layers are built only on their own systems. The tests on Windows build
// them as well, against a stand-in for the positional file calls of Unix, so that a change to
// the engine's interfaces that breaks either of them shows wherever the tests are run.
#[cfg(all(test, target_os = "windows"))]
mod unix_stand_in {
    pub trait FileExt {
        fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
        fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()>;
    }
    impl FileExt for std::fs::File {
        fn read_exact_at(&self, _buf: &mut [u8], _offset: u64) -> std::io::Result<()> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
        fn write_all_at(&self, _buf: &[u8], _offset: u64) -> std::io::Result<()> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
    }
}
#[cfg(all(test, target_os = "windows"))]
#[path = "linux/mod.rs"]
#[allow(dead_code)]
mod linux_built_for_the_tests;
#[cfg(all(test, target_os = "windows"))]
#[path = "macos/mod.rs"]
#[allow(dead_code)]
mod macos_built_for_the_tests;

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use engine::OutputSinkFactory;
use engine::arena::RegionMapper;
use engine::par::ThreadPlacer;
use engine::store::ExtentMember;
use engine::tier::{DeviceEntry, Profile};

pub struct PlatformEnv {
    pub profile: Profile,
    pub members: Vec<Arc<dyn ExtentMember>>,
    pub region_mapper: &'static dyn RegionMapper,
    pub thread_placer: &'static dyn ThreadPlacer,
}

pub fn open_ssd_path(
    path: &Path,
    reserve_override: Option<u64>,
) -> Result<(DeviceEntry, Arc<dyn ExtentMember>), String> {
    #[cfg(target_os = "windows")]
    {
        windows::WinSsdMember::open(path, reserve_override)
    }
    #[cfg(target_os = "linux")]
    {
        linux::LinuxSsdMember::open(path, reserve_override)
    }
    #[cfg(target_os = "macos")]
    {
        macos::MacSsdMember::open(path, reserve_override)
    }
}

pub fn make_output_factory(path: impl Into<PathBuf>) -> Box<dyn OutputSinkFactory> {
    #[cfg(target_os = "windows")]
    {
        Box::new(windows::WinOutputSinkFactory::new(path))
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LinuxOutputSinkFactory::new(path))
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(macos::MacOutputSinkFactory::new(path))
    }
}

pub fn probe_platform(
    ssd_specs: &[(PathBuf, Option<u64>)],
    enable_vram: bool,
) -> Result<PlatformEnv, String> {
    let (vram_entries, vram_members) = if enable_vram {
        vram::probe_vram_members()
    } else {
        (Vec::new(), Vec::new())
    };

    let mut ssd_entries = Vec::with_capacity(ssd_specs.len());
    let mut members: Vec<Arc<dyn ExtentMember>> = vram_members;

    for (path, size_opt) in ssd_specs {
        let (entry, member) = open_ssd_path(path, *size_opt)?;
        ssd_entries.push(entry);
        members.push(member);
    }

    #[cfg(target_os = "windows")]
    {
        let profile = windows::detect_host_profile(vram_entries, ssd_entries);
        Ok(PlatformEnv {
            profile,
            members,
            region_mapper: &windows::WIN_REGION_MAPPER,
            thread_placer: &windows::WIN_THREAD_PLACER,
        })
    }
    #[cfg(target_os = "linux")]
    {
        let profile = linux::detect_host_profile(vram_entries, ssd_entries);
        Ok(PlatformEnv {
            profile,
            members,
            region_mapper: &linux::LINUX_REGION_MAPPER,
            thread_placer: &linux::LINUX_THREAD_PLACER,
        })
    }
    #[cfg(target_os = "macos")]
    {
        let profile = macos::detect_host_profile(vram_entries, ssd_entries);
        Ok(PlatformEnv {
            profile,
            members,
            region_mapper: &macos::MAC_REGION_MAPPER,
            thread_placer: &macos::MAC_THREAD_PLACER,
        })
    }
}
