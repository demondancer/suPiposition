//! Contract:
//! - Inputs: Dynamic system Vulkan 1.4 loader (`vulkan-1.dll`, `libvulkan.so.1`, or
//!   `libvulkan.1.dylib`), `SUPI_VK_VALIDATION` environment override (`0`/`1`), and automated-test
//!   validation toggle (`set_test_vk_validation` / `debug_assertions`).
//! - Outputs: Discrete GPU `DeviceEntry` descriptors (named by driver `deviceUUID` and sized from
//!   `VK_EXT_memory_budget`) and `VramMember` implementations of the 5-call `ExtentMember` trait
//!   (including coalescing free-list reclamation of `DeviceSlab`s), requesting
//!   `VK_LAYER_KHRONOS_validation` automatically under automated tests (with clean fallback if the
//!   host lacks the SDK layer).
//! - Data format: 256 MiB device-local `VkBuffer` slabs addressed linearly as `[0 .. capacity)`,
//!   paired with direct `VK_EXT_external_memory_host` caller-buffer imports (zero persistent
//!   staging bounce buffer and zero helper copy threads).
//! - Work & bytes moved: Storage only — requires a transfer-only queue family (`VK_QUEUE_TRANSFER_BIT`
//!   with neither `GRAPHICS` nor `COMPUTE`), imports page/64KiB-aligned caller buffers directly via
//!   `VK_EXT_external_memory_host`, and submits each read/write batch as `vkCmdCopyBuffer` DMA
//!   transfers with a 5 s fence timeout.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use engine::store::{ByteRange, ExtentMember, HOST_IMPORT_ALIGN, MemberKind, MemberSpanAlloc, StoreError};
use engine::tier::DeviceEntry;

type VkFlags = u32;
type VkBool32 = u32;
type VkDeviceSize = u64;
type VkStructureType = i32;
type VkResult = i32;

type VkInstance = *mut c_void;
type VkPhysicalDevice = *mut c_void;
type VkDevice = *mut c_void;
type VkQueue = *mut c_void;
type VkCommandBuffer = *mut c_void;
type VkBuffer = u64;
type VkDeviceMemory = u64;
type VkCommandPool = u64;
type VkFence = u64;

const VK_SUCCESS: VkResult = 0;
const VK_TIMEOUT: VkResult = 2;
const VK_ERROR_DEVICE_LOST: VkResult = -4;

const VK_API_VERSION_1_4: u32 = (1 << 22) | (4 << 12);
const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

const VK_QUEUE_GRAPHICS_BIT: VkFlags = 0x0000_0001;
const VK_QUEUE_COMPUTE_BIT: VkFlags = 0x0000_0002;
const VK_QUEUE_TRANSFER_BIT: VkFlags = 0x0000_0004;

const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags = 0x0000_0001;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 0x0000_0004;
const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkFlags = 0x0000_0008;
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkFlags = 0x0000_0001;

const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkFlags = 0x0000_0001;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkFlags = 0x0000_0002;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

const VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: VkFlags = 0x0000_0001;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkFlags = 0x0000_0002;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 0x0000_0001;

const VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT: u32 = 0x0000_0080;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: VkStructureType = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: VkStructureType = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: VkStructureType = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: VkStructureType = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: VkStructureType = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: VkStructureType = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: VkStructureType = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: VkStructureType = 42;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: VkStructureType = 1_000_059_001;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: VkStructureType = 1_000_059_006;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: VkStructureType = 1_000_071_004;
const VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO: VkStructureType = 1_000_072_000;
const VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT: VkStructureType = 1_000_178_000;
const VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT: VkStructureType = 1_000_178_001;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT: VkStructureType =
    1_000_178_002;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: VkStructureType =
    1_000_237_000;

const VK_MAX_MEMORY_TYPES: usize = 32;
static TEST_VK_VALIDATION: AtomicBool = AtomicBool::new(false);
static LAST_PROBE_VALIDATION_REQUESTED: AtomicBool = AtomicBool::new(false);
static LAST_PROBE_VALIDATION_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Enable `VK_LAYER_KHRONOS_validation` by default during automated test runs (Section 12).
pub fn set_test_vk_validation(enable: bool) {
    TEST_VK_VALIDATION.store(enable, Ordering::Relaxed);
}

/// Returns `(validation_requested, validation_layer_active)` for the most recent `probe_vram_members` call.
pub fn last_probe_validation_state() -> (bool, bool) {
    (
        LAST_PROBE_VALIDATION_REQUESTED.load(Ordering::Relaxed),
        LAST_PROBE_VALIDATION_ACTIVE.load(Ordering::Relaxed),
    )
}
const VK_MAX_MEMORY_HEAPS: usize = 16;
const VK_UUID_SIZE: usize = 16;
const VK_LUID_SIZE: usize = 8;

const MAX_IMPORT_WINDOW_BYTES: usize = 512 * 1024 * 1024; // max 512 MiB per host import window
const SLAB_GRANULARITY: u64 = 256 * 1024 * 1024; // 256 MiB indexed device-local slab slots
const TRANSFER_TIMEOUT_NS: u64 = 5_000_000_000; // 5 s timeout per batch

static VRAM_DIRECT_IMPORTS: AtomicU64 = AtomicU64::new(0);
static VRAM_BOUNCE_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Reset `(direct_imports, bounce_fallbacks)` counters for automated tests.
pub fn reset_vram_transfer_stats() {
    VRAM_DIRECT_IMPORTS.store(0, Ordering::Relaxed);
    VRAM_BOUNCE_FALLBACKS.store(0, Ordering::Relaxed);
}

/// Return `(direct_imports, bounce_fallbacks)` across all `VramMember` transfers since reset.
pub fn vram_transfer_stats() -> (u64, u64) {
    (
        VRAM_DIRECT_IMPORTS.load(Ordering::Relaxed),
        VRAM_BOUNCE_FALLBACKS.load(Ordering::Relaxed),
    )
}

#[repr(C)]
struct VkApplicationInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queue_flags: VkFlags,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: VkExtent3D,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtensionProperties {
    extension_name: [u8; 256],
    spec_version: u32,
}

#[repr(C)]
struct VkPhysicalDevicePropertiesRaw {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    // Remaining `limits` + `sparseProperties` bytes totaling 824 bytes on 64-bit targets:
    _rest: [u8; 824 - 20 - 256 - VK_UUID_SIZE],
}

#[repr(C)]
struct VkPhysicalDeviceProperties2 {
    s_type: VkStructureType,
    p_next: *mut c_void,
    properties: VkPhysicalDevicePropertiesRaw,
}

#[repr(C)]
struct VkPhysicalDeviceIDProperties {
    s_type: VkStructureType,
    p_next: *mut c_void,
    device_uuid: [u8; VK_UUID_SIZE],
    driver_uuid: [u8; VK_UUID_SIZE],
    device_luid: [u8; VK_LUID_SIZE],
    device_node_mask: u32,
    device_luid_valid: VkBool32,
}

#[repr(C)]
struct VkPhysicalDeviceExternalMemoryHostPropertiesEXT {
    s_type: VkStructureType,
    p_next: *mut c_void,
    min_imported_host_pointer_alignment: VkDeviceSize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    property_flags: VkFlags,
    heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: VkDeviceSize,
    flags: VkFlags,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties2 {
    s_type: VkStructureType,
    p_next: *mut c_void,
    memory_properties: VkPhysicalDeviceMemoryProperties,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryBudgetPropertiesEXT {
    s_type: VkStructureType,
    p_next: *mut c_void,
    heap_budget: [VkDeviceSize; VK_MAX_MEMORY_HEAPS],
    heap_usage: [VkDeviceSize; VK_MAX_MEMORY_HEAPS],
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
    p_enabled_features: *const c_void,
}

#[repr(C)]
struct VkExternalMemoryBufferCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    handle_types: u32,
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    size: VkDeviceSize,
    usage: VkFlags,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkMemoryHostPointerPropertiesEXT {
    s_type: VkStructureType,
    p_next: *mut c_void,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkImportMemoryHostPointerInfoEXT {
    s_type: VkStructureType,
    p_next: *const c_void,
    handle_type: u32,
    p_host_pointer: *mut c_void,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    allocation_size: VkDeviceSize,
    memory_type_index: u32,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkBufferCopy {
    src_offset: VkDeviceSize,
    dst_offset: VkDeviceSize,
    size: VkDeviceSize,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const u64,
    p_wait_dst_stage_mask: *const VkFlags,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: VkStructureType,
    p_next: *const c_void,
    flags: VkFlags,
}

type PfnVkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const c_char) -> Option<unsafe extern "system" fn()>;
type PfnVkCreateInstance =
    unsafe extern "system" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> VkResult;
type PfnVkDestroyInstance = unsafe extern "system" fn(VkInstance, *const c_void);
type PfnVkEnumeratePhysicalDevices =
    unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;
type PfnVkGetPhysicalDeviceProperties2 =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties2);
type PfnVkGetPhysicalDeviceMemoryProperties2 =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties2);
type PfnVkGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
type PfnVkEnumerateDeviceExtensionProperties = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const c_char,
    *mut u32,
    *mut VkExtensionProperties,
) -> VkResult;
type PfnVkCreateDevice = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> VkResult;
type PfnVkGetDeviceProcAddr =
    unsafe extern "system" fn(VkDevice, *const c_char) -> Option<unsafe extern "system" fn()>;

type PfnVkDestroyDevice = unsafe extern "system" fn(VkDevice, *const c_void);
type PfnVkDeviceWaitIdle = unsafe extern "system" fn(VkDevice) -> VkResult;
type PfnVkGetDeviceQueue = unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue);
type PfnVkCreateBuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> VkResult;
type PfnVkDestroyBuffer = unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);
type PfnVkGetBufferMemoryRequirements =
    unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
type PfnVkAllocateMemory = unsafe extern "system" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> VkResult;
type PfnVkFreeMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
type PfnVkBindBufferMemory =
    unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> VkResult;
type PfnVkGetMemoryHostPointerPropertiesEXT = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const c_void,
    *mut VkMemoryHostPointerPropertiesEXT,
) -> VkResult;
type PfnVkCreateCommandPool = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> VkResult;
type PfnVkDestroyCommandPool = unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void);
type PfnVkAllocateCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> VkResult;
type PfnVkResetCommandBuffer = unsafe extern "system" fn(VkCommandBuffer, VkFlags) -> VkResult;
type PfnVkBeginCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult;
type PfnVkEndCommandBuffer = unsafe extern "system" fn(VkCommandBuffer) -> VkResult;
type PfnVkCmdCopyBuffer =
    unsafe extern "system" fn(VkCommandBuffer, VkBuffer, VkBuffer, u32, *const VkBufferCopy);
type PfnVkCreateFence = unsafe extern "system" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> VkResult;
type PfnVkDestroyFence = unsafe extern "system" fn(VkDevice, VkFence, *const c_void);
type PfnVkResetFences = unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> VkResult;
type PfnVkWaitForFences =
    unsafe extern "system" fn(VkDevice, u32, *const VkFence, VkBool32, u64) -> VkResult;
type PfnVkQueueSubmit =
    unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult;

struct DynLib(*mut c_void);
unsafe impl Send for DynLib {}
unsafe impl Sync for DynLib {}

impl DynLib {
    fn open_vulkan() -> Option<Self> {
        #[cfg(target_os = "windows")]
        unsafe {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn LoadLibraryA(name: *const c_char) -> *mut c_void;
            }
            let h = LoadLibraryA(c"vulkan-1.dll".as_ptr());
            if h.is_null() { None } else { Some(Self(h)) }
        }
        #[cfg(target_os = "linux")]
        unsafe {
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
            }
            let h = dlopen(c"libvulkan.so.1".as_ptr(), 2); // RTLD_NOW = 2
            if h.is_null() { None } else { Some(Self(h)) }
        }
        #[cfg(target_os = "macos")]
        unsafe {
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
            }
            let h = dlopen(c"libvulkan.1.dylib".as_ptr(), 2);
            if h.is_null() { None } else { Some(Self(h)) }
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }

    unsafe fn get_proc(&self, sym: &CStr) -> Option<*const c_void> {
        #[cfg(target_os = "windows")]
        unsafe {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *const c_void;
            }
            let p = GetProcAddress(self.0, sym.as_ptr());
            if p.is_null() { None } else { Some(p) }
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        unsafe {
            unsafe extern "C" {
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *const c_void;
            }
            let p = dlsym(self.0, sym.as_ptr());
            if p.is_null() { None } else { Some(p) }
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            let _ = sym;
            None
        }
    }
}

struct VkInstanceHandle {
    _lib: DynLib,
    instance: VkInstance,
    destroy_instance: PfnVkDestroyInstance,
}

unsafe impl Send for VkInstanceHandle {}
unsafe impl Sync for VkInstanceHandle {}

impl Drop for VkInstanceHandle {
    fn drop(&mut self) {
        if !self.instance.is_null() {
            unsafe { (self.destroy_instance)(self.instance, null()) };
        }
    }
}

struct DeviceFns {
    destroy_device: PfnVkDestroyDevice,
    device_wait_idle: PfnVkDeviceWaitIdle,
    create_buffer: PfnVkCreateBuffer,
    destroy_buffer: PfnVkDestroyBuffer,
    get_buffer_mem_reqs: PfnVkGetBufferMemoryRequirements,
    allocate_memory: PfnVkAllocateMemory,
    free_memory: PfnVkFreeMemory,
    bind_buffer_memory: PfnVkBindBufferMemory,
    get_host_ptr_props: PfnVkGetMemoryHostPointerPropertiesEXT,
    destroy_command_pool: PfnVkDestroyCommandPool,
    reset_command_buffer: PfnVkResetCommandBuffer,
    begin_command_buffer: PfnVkBeginCommandBuffer,
    end_command_buffer: PfnVkEndCommandBuffer,
    cmd_copy_buffer: PfnVkCmdCopyBuffer,
    destroy_fence: PfnVkDestroyFence,
    reset_fences: PfnVkResetFences,
    wait_for_fences: PfnVkWaitForFences,
    queue_submit: PfnVkQueueSubmit,
}

#[derive(Clone, Copy)]
struct DeviceSlab {
    base_offset: u64,
    size_bytes: u64,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
}

#[derive(Clone, Copy)]
struct CachedHostImport {
    base_addr: usize,
    byte_len: usize,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
}

struct VramInner {
    _inst: Arc<VkInstanceHandle>,
    device: VkDevice,
    fns: DeviceFns,
    transfer_queue: VkQueue,
    queue_family_idx: u32,
    device_local_mem_bits: u32,
    coherent_mem_bits: u32,
    cached_coherent_mem_bits: u32,
    cmd_pool: VkCommandPool,
    cmd_buf: VkCommandBuffer,
    fence: VkFence,
    slabs: Vec<Option<DeviceSlab>>,
    cached_imports: Vec<CachedHostImport>,
    alloc: MemberSpanAlloc,
    lost: bool,
}

unsafe impl Send for VramInner {}

impl Drop for VramInner {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.fns.device_wait_idle)(self.device);
            for ci in self.cached_imports.drain(..) {
                (self.fns.destroy_buffer)(self.device, ci.buffer, null());
                (self.fns.free_memory)(self.device, ci.memory, null());
            }
            for sl in self.slabs.drain(..).flatten() {
                (self.fns.destroy_buffer)(self.device, sl.buffer, null());
                (self.fns.free_memory)(self.device, sl.memory, null());
            }
            if self.fence != 0 {
                (self.fns.destroy_fence)(self.device, self.fence, null());
            }
            if self.cmd_pool != 0 {
                (self.fns.destroy_command_pool)(self.device, self.cmd_pool, null());
            }
            (self.fns.destroy_device)(self.device, null());
        }
    }
}

pub struct VramMember {
    id: String,
    usable_bytes: u64,
    align: u64,
    inner: Mutex<VramInner>,
}

impl VramMember {
    fn ensure_slab_slot_locked(
        &self,
        st: &mut VramInner,
        slot_idx: usize,
    ) -> Result<DeviceSlab, StoreError> {
        if slot_idx >= st.slabs.len() {
            return Err(StoreError::OutOfSpace);
        }
        if let Some(s) = st.slabs[slot_idx] {
            return Ok(s);
        }
        let base_offset = (slot_idx as u64) * SLAB_GRANULARITY;
        if base_offset >= self.usable_bytes {
            return Err(StoreError::OutOfSpace);
        }
        let slab_size = (self.usable_bytes - base_offset).min(SLAB_GRANULARITY);
        let bci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: null(),
            flags: 0,
            size: slab_size,
            usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &st.queue_family_idx,
        };
        let mut buf: VkBuffer = 0;
        let rc = unsafe { (st.fns.create_buffer)(st.device, &bci, null(), &mut buf) };
        if rc != VK_SUCCESS {
            return Err(StoreError::OutOfSpace);
        }
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe { (st.fns.get_buffer_mem_reqs)(st.device, buf, &mut reqs) };
        let candidates = reqs.memory_type_bits & st.device_local_mem_bits;
        if candidates == 0 {
            unsafe { (st.fns.destroy_buffer)(st.device, buf, null()) };
            return Err(StoreError::OutOfSpace);
        }
        let mem_type_idx = candidates.trailing_zeros();
        let mai = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: null(),
            allocation_size: reqs.size,
            memory_type_index: mem_type_idx,
        };
        let mut mem: VkDeviceMemory = 0;
        let rc_m = unsafe { (st.fns.allocate_memory)(st.device, &mai, null(), &mut mem) };
        if rc_m != VK_SUCCESS {
            unsafe { (st.fns.destroy_buffer)(st.device, buf, null()) };
            return Err(StoreError::OutOfSpace);
        }
        let rc_b = unsafe { (st.fns.bind_buffer_memory)(st.device, buf, mem, 0) };
        if rc_b != VK_SUCCESS {
            unsafe {
                (st.fns.destroy_buffer)(st.device, buf, null());
                (st.fns.free_memory)(st.device, mem, null());
            }
            return Err(StoreError::IoFailed);
        }
        let slab = DeviceSlab {
            base_offset,
            size_bytes: slab_size,
            buffer: buf,
            memory: mem,
        };
        st.slabs[slot_idx] = Some(slab);
        Ok(slab)
    }

    /// Import a caller-owned aligned host memory window `[ptr .. ptr + span_bytes)` directly into
    /// Vulkan via `VK_EXT_external_memory_host` without allocating or copying a staging bounce buffer.
    fn import_host_window_locked(
        &self,
        st: &VramInner,
        ptr: *const u8,
        span_bytes: usize,
    ) -> Option<(VkBuffer, VkDeviceMemory)> {
        if ptr.is_null()
            || span_bytes == 0
            || !(ptr as usize).is_multiple_of(self.align as usize)
            || !(span_bytes as u64).is_multiple_of(self.align)
        {
            return None;
        }
        unsafe {
            let ext_buf_info = VkExternalMemoryBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                p_next: null(),
                handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            };
            let bci = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: (&ext_buf_info as *const _) as *const c_void,
                flags: 0,
                size: span_bytes as u64,
                usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &st.queue_family_idx,
            };
            let mut host_buf: VkBuffer = 0;
            if (st.fns.create_buffer)(st.device, &bci, null(), &mut host_buf) != VK_SUCCESS {
                return None;
            }

            let mut s_reqs = VkMemoryRequirements {
                size: 0,
                alignment: 0,
                memory_type_bits: 0,
            };
            (st.fns.get_buffer_mem_reqs)(st.device, host_buf, &mut s_reqs);
            let alloc_size = s_reqs.size.div_ceil(self.align) * self.align;
            if alloc_size > (span_bytes as u64) {
                (st.fns.destroy_buffer)(st.device, host_buf, null());
                return None;
            }

            let mut hp_props = VkMemoryHostPointerPropertiesEXT {
                s_type: VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
                p_next: null_mut(),
                memory_type_bits: 0,
            };
            if (st.fns.get_host_ptr_props)(
                st.device,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                ptr as *const c_void,
                &mut hp_props,
            ) != VK_SUCCESS
            {
                (st.fns.destroy_buffer)(st.device, host_buf, null());
                return None;
            }

            let mut valid = s_reqs.memory_type_bits & hp_props.memory_type_bits;
            if (valid & st.cached_coherent_mem_bits) != 0 {
                valid &= st.cached_coherent_mem_bits;
            } else if (valid & st.coherent_mem_bits) != 0 {
                valid &= st.coherent_mem_bits;
            } else {
                valid = 0;
            }
            if valid == 0 {
                (st.fns.destroy_buffer)(st.device, host_buf, null());
                return None;
            }

            let import_info = VkImportMemoryHostPointerInfoEXT {
                s_type: VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                p_next: null(),
                handle_type: VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                p_host_pointer: ptr as *mut c_void,
            };
            let mai = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: (&import_info as *const _) as *const c_void,
                allocation_size: alloc_size,
                memory_type_index: valid.trailing_zeros(),
            };
            let mut host_mem: VkDeviceMemory = 0;
            if (st.fns.allocate_memory)(st.device, &mai, null(), &mut host_mem) != VK_SUCCESS {
                (st.fns.destroy_buffer)(st.device, host_buf, null());
                return None;
            }
            if (st.fns.bind_buffer_memory)(st.device, host_buf, host_mem, 0) != VK_SUCCESS {
                (st.fns.destroy_buffer)(st.device, host_buf, null());
                (st.fns.free_memory)(st.device, host_mem, null());
                return None;
            }
            Some((host_buf, host_mem))
        }
    }

    #[inline]
    unsafe fn destroy_imported_host_locked(st: &VramInner, buf: VkBuffer, mem: VkDeviceMemory) {
        unsafe {
            if buf != 0 {
                (st.fns.destroy_buffer)(st.device, buf, null());
            }
            if mem != 0 {
                (st.fns.free_memory)(st.device, mem, null());
            }
        }
    }

    fn submit_buffer_copies_locked(
        &self,
        st: &mut VramInner,
        imported_host_buf: VkBuffer,
        ops: &[(VkBuffer, VkBufferCopy)],
        to_device: bool,
    ) -> Result<(), StoreError> {
        if ops.is_empty() {
            return Ok(());
        }
        unsafe {
            if (st.fns.reset_fences)(st.device, 1, &st.fence) != VK_SUCCESS {
                st.lost = true;
                return Err(StoreError::DeviceLost);
            }
            if (st.fns.reset_command_buffer)(st.cmd_buf, 0) != VK_SUCCESS {
                st.lost = true;
                return Err(StoreError::DeviceLost);
            }
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: null(),
            };
            if (st.fns.begin_command_buffer)(st.cmd_buf, &begin_info) != VK_SUCCESS {
                return Err(StoreError::IoFailed);
            }

            let mut idx = 0usize;
            let mut batch_regions: Vec<VkBufferCopy> = Vec::with_capacity(ops.len());
            while idx < ops.len() {
                let slab_buf = ops[idx].0;
                batch_regions.clear();
                while idx < ops.len() && ops[idx].0 == slab_buf {
                    batch_regions.push(ops[idx].1);
                    idx += 1;
                }
                let (src_b, dst_b) = if to_device {
                    (imported_host_buf, slab_buf)
                } else {
                    (slab_buf, imported_host_buf)
                };
                (st.fns.cmd_copy_buffer)(
                    st.cmd_buf,
                    src_b,
                    dst_b,
                    batch_regions.len() as u32,
                    batch_regions.as_ptr(),
                );
            }

            if (st.fns.end_command_buffer)(st.cmd_buf) != VK_SUCCESS {
                return Err(StoreError::IoFailed);
            }
            let submit = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: null(),
                p_wait_dst_stage_mask: null(),
                command_buffer_count: 1,
                p_command_buffers: &st.cmd_buf,
                signal_semaphore_count: 0,
                p_signal_semaphores: null(),
            };
            let sub_rc = (st.fns.queue_submit)(st.transfer_queue, 1, &submit, st.fence);
            if sub_rc == VK_ERROR_DEVICE_LOST {
                st.lost = true;
                return Err(StoreError::DeviceLost);
            }
            if sub_rc != VK_SUCCESS {
                return Err(StoreError::IoFailed);
            }
            let wait_rc =
                (st.fns.wait_for_fences)(st.device, 1, &st.fence, 1, TRANSFER_TIMEOUT_NS);
            if wait_rc == VK_TIMEOUT || wait_rc == VK_ERROR_DEVICE_LOST {
                st.lost = true;
                return Err(StoreError::DeviceLost);
            }
            if wait_rc != VK_SUCCESS {
                return Err(StoreError::IoFailed);
            }
        }
        Ok(())
    }

    fn evict_host_buf(&self, base_addr: usize, byte_len: usize) {
        let mut st = self.inner.lock().unwrap();
        let mut i = 0usize;
        while i < st.cached_imports.len() {
            let c = st.cached_imports[i];
            if c.base_addr == base_addr && c.byte_len == byte_len {
                st.cached_imports.swap_remove(i);
                unsafe { Self::destroy_imported_host_locked(&st, c.buffer, c.memory) };
            } else {
                i += 1;
            }
        }
    }

    fn get_or_import_cached_slot_locked(
        &self,
        st: &mut VramInner,
        base_addr: usize,
        full_cap: usize,
    ) -> Option<VkBuffer> {
        if let Some(hit) = st
            .cached_imports
            .iter()
            .find(|c| c.base_addr == base_addr && c.byte_len == full_cap)
        {
            return Some(hit.buffer);
        }
        let (h_buf, h_mem) =
            self.import_host_window_locked(st, base_addr as *const u8, full_cap)?;
        if st.cached_imports.len() >= 32 {
            let old = st.cached_imports.remove(0);
            unsafe { Self::destroy_imported_host_locked(st, old.buffer, old.memory) };
        }
        st.cached_imports.push(CachedHostImport {
            base_addr,
            byte_len: full_cap,
            buffer: h_buf,
            memory: h_mem,
        });
        Some(h_buf)
    }

    fn build_slab_ops_locked(
        &self,
        st: &mut VramInner,
        ranges: &[ByteRange],
        host_bias: usize,
        host_base_off: usize,
        to_device: bool,
    ) -> Result<Vec<(VkBuffer, VkBufferCopy)>, StoreError> {
        let mut ops = Vec::with_capacity(ranges.len());
        for r in ranges {
            let mut rem = r.len;
            let mut cur_dev = r.device_offset;
            let mut cur_host = r.host_offset + host_bias - host_base_off;
            while rem > 0 {
                let slot_idx = (cur_dev / SLAB_GRANULARITY) as usize;
                let slab = self.ensure_slab_slot_locked(st, slot_idx)?;
                let slab_rem = ((slab.base_offset + slab.size_bytes) - cur_dev) as usize;
                let chunk = rem.min(slab_rem);
                let copy = if to_device {
                    VkBufferCopy {
                        src_offset: cur_host as u64,
                        dst_offset: cur_dev - slab.base_offset,
                        size: chunk as u64,
                    }
                } else {
                    VkBufferCopy {
                        src_offset: cur_dev - slab.base_offset,
                        dst_offset: cur_host as u64,
                        size: chunk as u64,
                    }
                };
                ops.push((slab.buffer, copy));
                cur_dev += chunk as u64;
                cur_host += chunk;
                rem -= chunk;
            }
        }
        Ok(ops)
    }

    fn transfer_fallback_aligned_locked(
        &self,
        st: &mut VramInner,
        ranges: &[ByteRange],
        host_ptr: *mut u8,
        to_device: bool,
    ) -> Result<(), StoreError> {
        let win_align = (self.align as usize).max(HOST_IMPORT_ALIGN);
        let mut r_idx = 0usize;
        while r_idx < ranges.len() {
            let mut packed_bytes = 0usize;
            let start_r = r_idx;
            while r_idx < ranges.len() {
                let r_len = ranges[r_idx].len;
                if packed_bytes > 0 && packed_bytes + r_len > MAX_IMPORT_WINDOW_BYTES {
                    break;
                }
                packed_bytes += r_len;
                r_idx += 1;
            }
            if packed_bytes == 0 {
                continue;
            }
            VRAM_BOUNCE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            let span = packed_bytes.div_ceil(win_align) * win_align;
            let layout = Layout::from_size_align(span, win_align).map_err(|_| StoreError::IoFailed)?;
            let tmp_ptr = unsafe { alloc_zeroed(layout) };
            if tmp_ptr.is_null() {
                return Err(StoreError::IoFailed);
            }
            let mut tmp_ranges = Vec::with_capacity(r_idx - start_r);
            let mut cur_tmp_off = 0usize;
            for r in &ranges[start_r..r_idx] {
                if to_device && r.len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (host_ptr as *const u8).add(r.host_offset),
                            tmp_ptr.add(cur_tmp_off),
                            r.len,
                        );
                    }
                }
                tmp_ranges.push(ByteRange {
                    device_offset: r.device_offset,
                    host_offset: cur_tmp_off,
                    len: r.len,
                });
                cur_tmp_off += r.len;
            }

            let res = if let Some((h_buf, h_mem)) =
                self.import_host_window_locked(st, tmp_ptr, span)
            {
                let ops_res = self.build_slab_ops_locked(st, &tmp_ranges, 0, 0, to_device);
                let sub_res =
                    ops_res.and_then(|ops| self.submit_buffer_copies_locked(st, h_buf, &ops, to_device));
                unsafe { Self::destroy_imported_host_locked(st, h_buf, h_mem) };
                sub_res
            } else {
                Err(StoreError::IoFailed)
            };

            if res.is_ok() && !to_device {
                let mut back_off = 0usize;
                for r in &ranges[start_r..r_idx] {
                    if r.len > 0 {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                tmp_ptr.add(back_off),
                                host_ptr.add(r.host_offset),
                                r.len,
                            );
                        }
                    }
                    back_off += r.len;
                }
            }
            unsafe { dealloc(tmp_ptr, layout) };
            res?;
        }
        Ok(())
    }
}

impl ExtentMember for VramMember {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> MemberKind {
        MemberKind::Vram
    }
    fn alignment(&self) -> u64 {
        self.align
    }
    fn capacity(&self) -> u64 {
        self.usable_bytes
    }
    fn allocate(&self, bytes: u64) -> Result<u64, StoreError> {
        let mut st = self.inner.lock().unwrap();
        if st.lost {
            return Err(StoreError::DeviceLost);
        }
        let aligned = bytes.div_ceil(self.align) * self.align;
        let start = st.alloc.allocate(aligned, self.align, self.usable_bytes)?;
        if aligned > 0 {
            let first_slot = (start / SLAB_GRANULARITY) as usize;
            let last_slot = ((start + aligned - 1) / SLAB_GRANULARITY) as usize;
            for slot_idx in first_slot..=last_slot {
                if let Err(e) = self.ensure_slab_slot_locked(&mut st, slot_idx) {
                    st.alloc.free(start, aligned, self.align);
                    return Err(e);
                }
            }
        }
        Ok(start)
    }
    fn read(&self, ranges: &[ByteRange], host_buf: &mut [u8]) -> Result<(), StoreError> {
        let mut st = self.inner.lock().unwrap();
        if st.lost {
            return Err(StoreError::DeviceLost);
        }
        if ranges.is_empty() {
            return Ok(());
        }
        let mut min_h = usize::MAX;
        let mut max_h = 0usize;
        for r in ranges {
            if r.device_offset.saturating_add(r.len as u64) > self.usable_bytes
                || r.host_offset.saturating_add(r.len) > host_buf.len()
            {
                return Err(StoreError::IoFailed);
            }
            if r.len > 0 {
                min_h = min_h.min(r.host_offset);
                max_h = max_h.max(r.host_offset + r.len);
            }
        }
        if max_h == 0 {
            return Ok(());
        }

        let win_align = (self.align as usize).max(HOST_IMPORT_ALIGN);
        let host_ptr_addr = host_buf.as_ptr() as usize;
        if let Some((slot_base, slot_cap)) =
            engine::store::lookup_registered_slot_buf(host_ptr_addr + min_h, max_h - min_h)
            && slot_cap <= MAX_IMPORT_WINDOW_BYTES
            && slot_base.is_multiple_of(win_align)
            && slot_cap.is_multiple_of(win_align)
            && let Some(h_buf) = self.get_or_import_cached_slot_locked(&mut st, slot_base, slot_cap)
        {
            VRAM_DIRECT_IMPORTS.fetch_add(1, Ordering::Relaxed);
            let host_bias = host_ptr_addr - slot_base;
            let ops = self.build_slab_ops_locked(&mut st, ranges, host_bias, 0, false)?;
            return self.submit_buffer_copies_locked(&mut st, h_buf, &ops, false);
        }

        let import_off = min_h & !(win_align - 1);
        let needed_span = (max_h - import_off).div_ceil(win_align) * win_align;
        let import_ptr = unsafe { host_buf.as_ptr().add(import_off) };
        if needed_span <= MAX_IMPORT_WINDOW_BYTES
            && (import_ptr as usize).is_multiple_of(win_align)
            && import_off + needed_span <= host_buf.len()
            && let Some((h_buf, h_mem)) =
                self.import_host_window_locked(&st, import_ptr, needed_span)
        {
            VRAM_DIRECT_IMPORTS.fetch_add(1, Ordering::Relaxed);
            let ops_res = self.build_slab_ops_locked(&mut st, ranges, 0, import_off, false);
            let res = ops_res
                .and_then(|ops| self.submit_buffer_copies_locked(&mut st, h_buf, &ops, false));
            unsafe { Self::destroy_imported_host_locked(&st, h_buf, h_mem) };
            return res;
        }

        self.transfer_fallback_aligned_locked(&mut st, ranges, host_buf.as_mut_ptr(), false)
    }
    fn write(&self, ranges: &[ByteRange], host_buf: &[u8]) -> Result<(), StoreError> {
        let mut st = self.inner.lock().unwrap();
        if st.lost {
            return Err(StoreError::DeviceLost);
        }
        if ranges.is_empty() {
            return Ok(());
        }
        let mut min_h = usize::MAX;
        let mut max_h = 0usize;
        for r in ranges {
            if r.device_offset.saturating_add(r.len as u64) > self.usable_bytes
                || r.host_offset.saturating_add(r.len) > host_buf.len()
            {
                return Err(StoreError::IoFailed);
            }
            if r.len > 0 {
                min_h = min_h.min(r.host_offset);
                max_h = max_h.max(r.host_offset + r.len);
            }
        }
        if max_h == 0 {
            return Ok(());
        }

        let win_align = (self.align as usize).max(HOST_IMPORT_ALIGN);
        let host_ptr_addr = host_buf.as_ptr() as usize;
        if let Some((slot_base, slot_cap)) =
            engine::store::lookup_registered_slot_buf(host_ptr_addr + min_h, max_h - min_h)
            && slot_cap <= MAX_IMPORT_WINDOW_BYTES
            && slot_base.is_multiple_of(win_align)
            && slot_cap.is_multiple_of(win_align)
            && let Some(h_buf) = self.get_or_import_cached_slot_locked(&mut st, slot_base, slot_cap)
        {
            VRAM_DIRECT_IMPORTS.fetch_add(1, Ordering::Relaxed);
            let host_bias = host_ptr_addr - slot_base;
            let ops = self.build_slab_ops_locked(&mut st, ranges, host_bias, 0, true)?;
            return self.submit_buffer_copies_locked(&mut st, h_buf, &ops, true);
        }

        let import_off = min_h & !(win_align - 1);
        let needed_span = (max_h - import_off).div_ceil(win_align) * win_align;
        let import_ptr = unsafe { host_buf.as_ptr().add(import_off) };
        if needed_span <= MAX_IMPORT_WINDOW_BYTES
            && (import_ptr as usize).is_multiple_of(win_align)
            && import_off + needed_span <= host_buf.len()
            && let Some((h_buf, h_mem)) =
                self.import_host_window_locked(&st, import_ptr, needed_span)
        {
            VRAM_DIRECT_IMPORTS.fetch_add(1, Ordering::Relaxed);
            let ops_res = self.build_slab_ops_locked(&mut st, ranges, 0, import_off, true);
            let res =
                ops_res.and_then(|ops| self.submit_buffer_copies_locked(&mut st, h_buf, &ops, true));
            unsafe { Self::destroy_imported_host_locked(&st, h_buf, h_mem) };
            return res;
        }

        self.transfer_fallback_aligned_locked(
            &mut st,
            ranges,
            host_buf.as_ptr() as *mut u8,
            true,
        )
    }
    fn free(&self, device_offset: u64, bytes: u64) {
        let mut st = self.inner.lock().unwrap();
        let aligned = bytes.div_ceil(self.align) * self.align;
        if aligned == 0 {
            return;
        }
        st.alloc.free(device_offset, aligned, self.align);
        let first_slot = (device_offset / SLAB_GRANULARITY) as usize;
        let last_slot = ((device_offset + aligned - 1) / SLAB_GRANULARITY) as usize;
        for slot_idx in first_slot..=last_slot.min(st.slabs.len().saturating_sub(1)) {
            if let Some(sl) = st.slabs[slot_idx]
                && st.alloc.is_range_completely_free(sl.base_offset, sl.size_bytes)
            {
                unsafe {
                    (st.fns.destroy_buffer)(st.device, sl.buffer, null());
                    (st.fns.free_memory)(st.device, sl.memory, null());
                }
                st.slabs[slot_idx] = None;
            }
        }
    }
}

fn format_uuid_hex(uuid: &[u8; VK_UUID_SIZE]) -> String {
    let mut s = String::with_capacity(36);
    s.push_str("gpu-");
    for &b in uuid {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn cstr_bytes_eq(buf: &[u8], target: &[u8]) -> bool {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    &buf[..end] == target
}

/// Probe discrete Vulkan 1.4 GPUs and open transfer-only `VramMember`s.
pub fn probe_vram_members() -> (Vec<DeviceEntry>, Vec<Arc<dyn ExtentMember>>) {
    let Some(lib) = DynLib::open_vulkan() else {
        return (Vec::new(), Vec::new());
    };

    unsafe {
        let Some(gipa_raw) = lib.get_proc(c"vkGetInstanceProcAddr") else {
            return (Vec::new(), Vec::new());
        };
        let get_inst_proc: PfnVkGetInstanceProcAddr = std::mem::transmute(gipa_raw);

        let Some(create_inst_fn) = get_inst_proc(null_mut(), c"vkCreateInstance".as_ptr()) else {
            return (Vec::new(), Vec::new());
        };
        let vk_create_instance: PfnVkCreateInstance = std::mem::transmute(create_inst_fn);

        let want_validation = match std::env::var("SUPI_VK_VALIDATION").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => TEST_VK_VALIDATION.load(Ordering::Relaxed) || cfg!(debug_assertions),
        };
        LAST_PROBE_VALIDATION_REQUESTED.store(want_validation, Ordering::Relaxed);
        LAST_PROBE_VALIDATION_ACTIVE.store(false, Ordering::Relaxed);
        let layer_name = c"VK_LAYER_KHRONOS_validation".as_ptr();
        let layers = [layer_name];

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: null(),
            p_application_name: c"suPiposition".as_ptr(),
            application_version: 1,
            p_engine_name: c"suPiposition".as_ptr(),
            engine_version: 1,
            api_version: VK_API_VERSION_1_4,
        };

        let mut instance: VkInstance = null_mut();
        let mut ici = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: if want_validation { 1 } else { 0 },
            pp_enabled_layer_names: if want_validation { layers.as_ptr() } else { null() },
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
        };

        if vk_create_instance(&ici, null(), &mut instance) == VK_SUCCESS {
            LAST_PROBE_VALIDATION_ACTIVE.store(want_validation, Ordering::Relaxed);
        } else if want_validation {
            ici.enabled_layer_count = 0;
            ici.pp_enabled_layer_names = null();
            if vk_create_instance(&ici, null(), &mut instance) != VK_SUCCESS {
                return (Vec::new(), Vec::new());
            }
        } else {
            return (Vec::new(), Vec::new());
        }

        macro_rules! inst_fn {
            ($name:expr, $ty:ty) => {{
                let Some(f) = get_inst_proc(instance, $name.as_ptr()) else {
                    return (Vec::new(), Vec::new());
                };
                std::mem::transmute::<unsafe extern "system" fn(), $ty>(f)
            }};
        }

        let destroy_instance = inst_fn!(c"vkDestroyInstance", PfnVkDestroyInstance);
        let enum_phys = inst_fn!(c"vkEnumeratePhysicalDevices", PfnVkEnumeratePhysicalDevices);
        let get_props2 = inst_fn!(
            c"vkGetPhysicalDeviceProperties2",
            PfnVkGetPhysicalDeviceProperties2
        );
        let get_mem_props2 = inst_fn!(
            c"vkGetPhysicalDeviceMemoryProperties2",
            PfnVkGetPhysicalDeviceMemoryProperties2
        );
        let get_queue_props = inst_fn!(
            c"vkGetPhysicalDeviceQueueFamilyProperties",
            PfnVkGetPhysicalDeviceQueueFamilyProperties
        );
        let enum_dev_exts = inst_fn!(
            c"vkEnumerateDeviceExtensionProperties",
            PfnVkEnumerateDeviceExtensionProperties
        );
        let create_device = inst_fn!(c"vkCreateDevice", PfnVkCreateDevice);
        let get_dev_proc = inst_fn!(c"vkGetDeviceProcAddr", PfnVkGetDeviceProcAddr);

        let inst_arc = Arc::new(VkInstanceHandle {
            _lib: lib,
            instance,
            destroy_instance,
        });

        let mut phys_count = 0u32;
        if enum_phys(inst_arc.instance, &mut phys_count, null_mut()) != VK_SUCCESS
            || phys_count == 0
        {
            return (Vec::new(), Vec::new());
        }
        let mut phys_devs = vec![null_mut(); phys_count as usize];
        if enum_phys(inst_arc.instance, &mut phys_count, phys_devs.as_mut_ptr()) != VK_SUCCESS {
            return (Vec::new(), Vec::new());
        }

        let mut entries = Vec::new();
        let mut members: Vec<Arc<dyn ExtentMember>> = Vec::new();

        for pdev in phys_devs {
            let mut ext_host_props = VkPhysicalDeviceExternalMemoryHostPropertiesEXT {
                s_type: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
                p_next: null_mut(),
                min_imported_host_pointer_alignment: 4096,
            };
            let mut id_props = VkPhysicalDeviceIDProperties {
                s_type: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
                p_next: (&mut ext_host_props as *mut _) as *mut c_void,
                device_uuid: [0; VK_UUID_SIZE],
                driver_uuid: [0; VK_UUID_SIZE],
                device_luid: [0; VK_LUID_SIZE],
                device_node_mask: 0,
                device_luid_valid: 0,
            };
            let mut props2: VkPhysicalDeviceProperties2 = std::mem::zeroed();
            props2.s_type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.p_next = (&mut id_props as *mut _) as *mut c_void;
            get_props2(pdev, &mut props2);

            if props2.properties.device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                || props2.properties.api_version < VK_API_VERSION_1_4
            {
                continue;
            }

            // Require a transfer-only queue family: `TRANSFER` set and neither `GRAPHICS` nor `COMPUTE` set!
            let mut qf_count = 0u32;
            get_queue_props(pdev, &mut qf_count, null_mut());
            if qf_count == 0 {
                continue;
            }
            let mut qfs = vec![
                VkQueueFamilyProperties {
                    queue_flags: 0,
                    queue_count: 0,
                    timestamp_valid_bits: 0,
                    min_image_transfer_granularity: VkExtent3D {
                        width: 0,
                        height: 0,
                        depth: 0
                    },
                };
                qf_count as usize
            ];
            get_queue_props(pdev, &mut qf_count, qfs.as_mut_ptr());
            let Some(transfer_qf) = qfs.iter().position(|q| {
                q.queue_count > 0
                    && (q.queue_flags & VK_QUEUE_TRANSFER_BIT) != 0
                    && (q.queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0
            }) else {
                continue;
            };

            // Require `VK_EXT_external_memory_host` and `VK_EXT_memory_budget`!
            let mut ext_count = 0u32;
            if enum_dev_exts(pdev, null(), &mut ext_count, null_mut()) != VK_SUCCESS
                || ext_count == 0
            {
                continue;
            }
            let mut exts = vec![
                VkExtensionProperties {
                    extension_name: [0; 256],
                    spec_version: 0,
                };
                ext_count as usize
            ];
            if enum_dev_exts(pdev, null(), &mut ext_count, exts.as_mut_ptr()) != VK_SUCCESS {
                continue;
            }
            let has_ext_host = exts
                .iter()
                .any(|e| cstr_bytes_eq(&e.extension_name, b"VK_EXT_external_memory_host"));
            let has_budget = exts
                .iter()
                .any(|e| cstr_bytes_eq(&e.extension_name, b"VK_EXT_memory_budget"));
            if !has_ext_host || !has_budget {
                continue;
            }

            // Query device-local memory budget!
            let mut budget_props = VkPhysicalDeviceMemoryBudgetPropertiesEXT {
                s_type: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
                p_next: null_mut(),
                heap_budget: [0; VK_MAX_MEMORY_HEAPS],
                heap_usage: [0; VK_MAX_MEMORY_HEAPS],
            };
            let mut mem_props2: VkPhysicalDeviceMemoryProperties2 = std::mem::zeroed();
            mem_props2.s_type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            mem_props2.p_next = (&mut budget_props as *mut _) as *mut c_void;
            get_mem_props2(pdev, &mut mem_props2);

            let mp = &mem_props2.memory_properties;
            let mut best_heap_free = 0u64;
            let mut dev_local_mem_bits = 0u32;

            for h in 0..(mp.memory_heap_count as usize) {
                if (mp.memory_heaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 {
                    let free = budget_props.heap_budget[h].saturating_sub(budget_props.heap_usage[h]);
                    if free > best_heap_free {
                        best_heap_free = free;
                    }
                }
            }
            for t in 0..(mp.memory_type_count as usize) {
                let mt = mp.memory_types[t];
                let h = mt.heap_index as usize;
                if (mt.property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
                    && h < (mp.memory_heap_count as usize)
                    && (mp.memory_heaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
                {
                    dev_local_mem_bits |= 1u32 << t;
                }
            }

            // Leave 512 MiB driver/desktop headroom on the discrete GPU heap.
            let usable = best_heap_free.saturating_sub(512 * 1024 * 1024);
            let align = ext_host_props
                .min_imported_host_pointer_alignment
                .max(4096)
                .next_power_of_two();
            let usable = (usable / SLAB_GRANULARITY) * SLAB_GRANULARITY;
            if usable < SLAB_GRANULARITY || dev_local_mem_bits == 0 {
                continue;
            }

            // Open `VkDevice` with ONLY the transfer-only queue family and the two required extensions!
            let prio = 1.0f32;
            let qci = VkDeviceQueueCreateInfo {
                s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                queue_family_index: transfer_qf as u32,
                queue_count: 1,
                p_queue_priorities: &prio,
            };
            let req_exts = [
                c"VK_EXT_external_memory_host".as_ptr(),
                c"VK_EXT_memory_budget".as_ptr(),
            ];
            let dci = VkDeviceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                queue_create_info_count: 1,
                p_queue_create_infos: &qci,
                enabled_layer_count: 0,
                pp_enabled_layer_names: null(),
                enabled_extension_count: req_exts.len() as u32,
                pp_enabled_extension_names: req_exts.as_ptr(),
                p_enabled_features: null(),
            };
            let mut device: VkDevice = null_mut();
            if create_device(pdev, &dci, null(), &mut device) != VK_SUCCESS || device.is_null() {
                continue;
            }

            macro_rules! dev_fn {
                ($name:expr, $ty:ty) => {{
                    let Some(f) = get_dev_proc(device, $name.as_ptr()) else {
                        continue;
                    };
                    std::mem::transmute::<unsafe extern "system" fn(), $ty>(f)
                }};
            }

            let get_queue = dev_fn!(c"vkGetDeviceQueue", PfnVkGetDeviceQueue);
            let create_cmd_pool = dev_fn!(c"vkCreateCommandPool", PfnVkCreateCommandPool);
            let alloc_cmd_bufs = dev_fn!(c"vkAllocateCommandBuffers", PfnVkAllocateCommandBuffers);
            let create_fence = dev_fn!(c"vkCreateFence", PfnVkCreateFence);

            let fns = DeviceFns {
                destroy_device: dev_fn!(c"vkDestroyDevice", PfnVkDestroyDevice),
                device_wait_idle: dev_fn!(c"vkDeviceWaitIdle", PfnVkDeviceWaitIdle),
                create_buffer: dev_fn!(c"vkCreateBuffer", PfnVkCreateBuffer),
                destroy_buffer: dev_fn!(c"vkDestroyBuffer", PfnVkDestroyBuffer),
                get_buffer_mem_reqs: dev_fn!(
                    c"vkGetBufferMemoryRequirements",
                    PfnVkGetBufferMemoryRequirements
                ),
                allocate_memory: dev_fn!(c"vkAllocateMemory", PfnVkAllocateMemory),
                free_memory: dev_fn!(c"vkFreeMemory", PfnVkFreeMemory),
                bind_buffer_memory: dev_fn!(c"vkBindBufferMemory", PfnVkBindBufferMemory),
                get_host_ptr_props: dev_fn!(
                    c"vkGetMemoryHostPointerPropertiesEXT",
                    PfnVkGetMemoryHostPointerPropertiesEXT
                ),
                destroy_command_pool: dev_fn!(c"vkDestroyCommandPool", PfnVkDestroyCommandPool),
                reset_command_buffer: dev_fn!(c"vkResetCommandBuffer", PfnVkResetCommandBuffer),
                begin_command_buffer: dev_fn!(c"vkBeginCommandBuffer", PfnVkBeginCommandBuffer),
                end_command_buffer: dev_fn!(c"vkEndCommandBuffer", PfnVkEndCommandBuffer),
                cmd_copy_buffer: dev_fn!(c"vkCmdCopyBuffer", PfnVkCmdCopyBuffer),
                destroy_fence: dev_fn!(c"vkDestroyFence", PfnVkDestroyFence),
                reset_fences: dev_fn!(c"vkResetFences", PfnVkResetFences),
                wait_for_fences: dev_fn!(c"vkWaitForFences", PfnVkWaitForFences),
                queue_submit: dev_fn!(c"vkQueueSubmit", PfnVkQueueSubmit),
            };

            let mut transfer_queue: VkQueue = null_mut();
            get_queue(device, transfer_qf as u32, 0, &mut transfer_queue);

            // Verify `VK_EXT_external_memory_host` imports a 64 KiB caller-aligned window cleanly
            // on this GPU, then release the probe window immediately (zero persistent bounce RAM).
            let probe_align = (align as usize).max(HOST_IMPORT_ALIGN);
            let Ok(probe_layout) = Layout::from_size_align(probe_align, probe_align) else {
                (fns.destroy_device)(device, null());
                continue;
            };
            let probe_ptr = alloc_zeroed(probe_layout);
            if probe_ptr.is_null() {
                (fns.destroy_device)(device, null());
                continue;
            }

            let ext_buf_info = VkExternalMemoryBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                p_next: null(),
                handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            };
            let qf_u32 = transfer_qf as u32;
            let sbci = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: (&ext_buf_info as *const _) as *const c_void,
                flags: 0,
                size: probe_align as u64,
                usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &qf_u32,
            };
            let mut probe_buf: VkBuffer = 0;
            if (fns.create_buffer)(device, &sbci, null(), &mut probe_buf) != VK_SUCCESS {
                dealloc(probe_ptr, probe_layout);
                (fns.destroy_device)(device, null());
                continue;
            }

            let mut s_reqs = VkMemoryRequirements {
                size: 0,
                alignment: 0,
                memory_type_bits: 0,
            };
            (fns.get_buffer_mem_reqs)(device, probe_buf, &mut s_reqs);

            let mut hp_props = VkMemoryHostPointerPropertiesEXT {
                s_type: VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
                p_next: null_mut(),
                memory_type_bits: 0,
            };
            if (fns.get_host_ptr_props)(
                device,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                probe_ptr as *const c_void,
                &mut hp_props,
            ) != VK_SUCCESS
            {
                (fns.destroy_buffer)(device, probe_buf, null());
                dealloc(probe_ptr, probe_layout);
                (fns.destroy_device)(device, null());
                continue;
            }

            let mut coherent_bits = 0u32;
            let mut cached_coherent_bits = 0u32;
            for t in 0..(mp.memory_type_count as usize) {
                let flags = mp.memory_types[t].property_flags;
                if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 {
                    coherent_bits |= 1u32 << t;
                    if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0 {
                        cached_coherent_bits |= 1u32 << t;
                    }
                }
            }
            let mut valid_host_bits = s_reqs.memory_type_bits & hp_props.memory_type_bits;
            if (valid_host_bits & cached_coherent_bits) != 0 {
                valid_host_bits &= cached_coherent_bits;
            } else if (valid_host_bits & coherent_bits) != 0 {
                valid_host_bits &= coherent_bits;
            } else {
                valid_host_bits = 0;
            }
            let import_alloc_size = s_reqs.size.div_ceil(align) * align;
            if valid_host_bits == 0 || import_alloc_size > (probe_align as u64) {
                (fns.destroy_buffer)(device, probe_buf, null());
                dealloc(probe_ptr, probe_layout);
                (fns.destroy_device)(device, null());
                continue;
            }

            let import_info = VkImportMemoryHostPointerInfoEXT {
                s_type: VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                p_next: null(),
                handle_type: VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                p_host_pointer: probe_ptr as *mut c_void,
            };
            let smai = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: (&import_info as *const _) as *const c_void,
                allocation_size: import_alloc_size,
                memory_type_index: valid_host_bits.trailing_zeros(),
            };
            let mut probe_mem: VkDeviceMemory = 0;
            let probe_ok = (fns.allocate_memory)(device, &smai, null(), &mut probe_mem) == VK_SUCCESS
                && (fns.bind_buffer_memory)(device, probe_buf, probe_mem, 0) == VK_SUCCESS;
            (fns.destroy_buffer)(device, probe_buf, null());
            if probe_mem != 0 {
                (fns.free_memory)(device, probe_mem, null());
            }
            dealloc(probe_ptr, probe_layout);
            if !probe_ok {
                (fns.destroy_device)(device, null());
                continue;
            }

            let cpci = VkCommandPoolCreateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: null(),
                flags: VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                    | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queue_family_index: qf_u32,
            };
            let mut cmd_pool: VkCommandPool = 0;
            let mut cmd_buf: VkCommandBuffer = null_mut();
            let mut fence: VkFence = 0;
            let fci = VkFenceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                p_next: null(),
                flags: 0,
            };
            let pool_ok = create_cmd_pool(device, &cpci, null(), &mut cmd_pool) == VK_SUCCESS;
            let cb_ok = pool_ok && {
                let cbai = VkCommandBufferAllocateInfo {
                    s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: null(),
                    command_pool: cmd_pool,
                    level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    command_buffer_count: 1,
                };
                alloc_cmd_bufs(device, &cbai, &mut cmd_buf) == VK_SUCCESS
            };
            let fence_ok = cb_ok && create_fence(device, &fci, null(), &mut fence) == VK_SUCCESS;

            if !fence_ok {
                if fence != 0 {
                    (fns.destroy_fence)(device, fence, null());
                }
                if cmd_pool != 0 {
                    (fns.destroy_command_pool)(device, cmd_pool, null());
                }
                (fns.destroy_device)(device, null());
                continue;
            }

            let num_slots = usable.div_ceil(SLAB_GRANULARITY) as usize;
            let dev_id = format_uuid_hex(&id_props.device_uuid);
            entries.push(DeviceEntry {
                id: dev_id.clone(),
                size: usable,
                align,
            });
            let vram_member = Arc::new(VramMember {
                id: dev_id,
                usable_bytes: usable,
                align,
                inner: Mutex::new(VramInner {
                    _inst: Arc::clone(&inst_arc),
                    device,
                    fns,
                    transfer_queue,
                    queue_family_idx: qf_u32,
                    device_local_mem_bits: dev_local_mem_bits,
                    coherent_mem_bits: coherent_bits,
                    cached_coherent_mem_bits: cached_coherent_bits,
                    cmd_pool,
                    cmd_buf,
                    fence,
                    slabs: vec![None; num_slots],
                    cached_imports: Vec::new(),
                    alloc: MemberSpanAlloc::new(),
                    lost: false,
                }),
            });
            let weak_vram = Arc::downgrade(&vram_member);
            engine::store::register_host_buf_release_hook(Box::new(move |addr, cap| {
                if let Some(strong) = weak_vram.upgrade() {
                    strong.evict_host_buf(addr, cap);
                    true
                } else {
                    false
                }
            }));
            members.push(vram_member);
        }

        (entries, members)
    }
}
