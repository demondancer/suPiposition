//! Contract:
//! - Inputs: `table: *mut usize` pointing to 24 kernel function pointers, and `abi: u64` layout version.
//! - Outputs: Overwrites supported entries in `table` in place when the CPU and OS satisfy the target
//!   architecture file's requirements and `abi` matches (otherwise leaves `table` untouched), and
//!   exposes deterministic compile-time `SUPI_BUILD_ID` for static calibration cache keying.
//! - Data format: C ABI function pointer array of length 24 and 64-bit `SUPI_BUILD_ID`.
//! - Work & bytes moved: One startup CPU probe; moves at most 192 bytes into `table`.

include!(concat!(env!("OUT_DIR"), "/glue.rs"));
