//! Contract:
//! - Inputs: Pointers, 64-bit lengths, and plain `#[repr(C)]` parameter structs for all 24 leaf
//!   kernels, plus a `BindFn` callback (`supi_bind`) for startup kernel table binding and self-test.
//! - Outputs: Exact reference results for limb, NTT, movement, digit, and check-word kernels, and
//!   reference complex-double weighted-convolution results for the 6-kernel FFT chain; bound and
//!   startup-verified `Kernels` dispatch table.
//! - Data format: 64-bit limbs least-significant first; complex doubles interleaved `(re, im)`;
//!   NTT residues as integer-valued `f64` in `[0, p)`; 50-bit packed residues as 50-byte blocks per
//!   8 residues; decimal digits as 19-digit `u64` words or ASCII bytes.
//! - Work & bytes moved: Each kernel performs only the arithmetic and memory accesses of the slice
//!   lengths passed in its arguments; optional atomic counters record exact kernel work at the
//!   seam, and a traced run can time every call through the table (`Kernels::enable_timing`).

use std::sync::atomic::{AtomicU64, Ordering};

pub const SUPI_ABI_VERSION: u64 = 6;
pub const NUM_KERNELS: usize = 24;

pub const MOD_W0: u64 = (1u64 << 61) - 1;
pub const PRIME_LOC1: u64 = (159u64 << 42) + 1;
pub const PRIME_LOC2: u64 = (115u64 << 42) + 1;

pub const KERNEL_NAMES: [&str; NUM_KERNELS] = [
    "add_n",
    "sub_n",
    "mul_1",
    "addmul_1",
    "mul_base",
    "sqr_base",
    "fft_split",
    "fft_fwd",
    "fft_inv",
    "fft_pmul",
    "fft_pmuladd",
    "fft_carry",
    "ntt_split",
    "ntt_fwd",
    "ntt_inv",
    "ntt_fwd_cols",
    "ntt_inv_cols",
    "ntt_pmul",
    "ntt_pmuladd",
    "ntt_crt",
    "gather",
    "scatter",
    "to_digits",
    "check_n",
];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FftSplitCfg {
    pub in_limbs: u64,
    pub n_complex: u64,
    pub coeff_bits: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FftCarryCfg {
    pub out_limbs: u64,
    pub n_complex: u64,
    pub coeff_bits: u64,
    /// Out: the carry out of the last coefficient, an `i64`. The limbs written are the sum of
    /// the coefficients below `2^(2 * n_complex * coeff_bits)` (and this carry above it where
    /// it is positive and `out_limbs` leaves room). A product longer than the transform comes
    /// out modulo `2^L + 1`; its caller needs this carry to tie the limbs to the product.
    pub top_carry: u64,
    pub max_err: f64,
    pub check_word: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttPrime {
    pub p: f64,
    pub inv_p: f64,
    pub scale: f64,
    pub l1_sub: u64,
    pub l2_sub: u64,
    pub fwd_tw: *const f64,
    pub pmul_b: *const f64,
}

unsafe impl Send for NttPrime {}
unsafe impl Sync for NttPrime {}

impl NttPrime {
    #[inline]
    pub fn new(p_int: u64, scale_int: u64) -> Self {
        Self::with_blocks(p_int, scale_int, u64::MAX, u64::MAX)
    }

    #[inline]
    pub fn with_blocks(p_int: u64, scale_int: u64, l1_sub: u64, l2_sub: u64) -> Self {
        let p = p_int as f64;
        Self {
            p,
            inv_p: 1.0 / p,
            scale: scale_int as f64,
            l1_sub,
            l2_sub,
            fwd_tw: std::ptr::null(),
            pmul_b: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttColsCfg {
    pub rows: u64,
    pub cols: u64,
    pub l1_sub_rows: u64,
    pub l2_sub_rows: u64,
    pub scratch: *mut f64,
}

impl NttColsCfg {
    #[inline]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows: rows as u64,
            cols: cols as u64,
            l1_sub_rows: u64::MAX,
            l2_sub_rows: u64::MAX,
            scratch: std::ptr::null_mut(),
        }
    }

    #[inline]
    pub fn with_blocks(
        rows: usize,
        cols: usize,
        l1_sub_rows: u64,
        l2_sub_rows: u64,
        scratch: *mut f64,
    ) -> Self {
        Self {
            rows: rows as u64,
            cols: cols as u64,
            l1_sub_rows,
            l2_sub_rows,
            scratch,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttSplitCfg {
    pub in_limbs: u64,
    pub start_coeff: u64,
    pub count: u64,
    pub coeff_bits: u64,
    pub col_stride: u64,
    pub cols_per_row: u64,
    pub p: f64,
    pub inv_p: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttVecCfg {
    pub n: u64,
    pub p: f64,
    pub inv_p: f64,
    pub sub_second: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttCrtCfg {
    pub k_primes: u64,
    pub count: u64,
    pub coeff_bits: u64,
    pub out_limbs: u64,
    pub signed: u64,
    pub wrap_cyclic: u64,
    pub primes: *const f64,
    pub inv_primes: *const f64,
    pub garner_inv: *const f64,
    pub prefix_prods: *const u64,
    pub k_words: u64,
    pub carry_limbs: *mut u64,
    pub segs: NttCrtSegs,
}

/// Many equal streams in one `ntt_crt` call. With `segments >= 2`, stream `s` reads
/// `residues[i] + s * res_stride`, writes the config's `out_limbs` limbs at
/// `out + s * out_stride`, starts from a zero carry, leaves its carry at
/// `carry_limbs + s * carry_stride` and its check word at `w0[s * w0_stride]`. The call then
/// returns 0, `carry_limbs` is not null and `wrap_cyclic` is 0. With `segments <= 1` the call
/// is one stream and the other fields are not read.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttCrtSegs {
    pub segments: u64,
    /// In residues.
    pub res_stride: u64,
    /// In limbs.
    pub out_stride: u64,
    /// In words.
    pub carry_stride: u64,
    pub w0: *mut u64,
    pub w0_stride: u64,
}

impl NttCrtSegs {
    /// One stream.
    pub const ONE: Self = Self {
        segments: 0,
        res_stride: 0,
        out_stride: 0,
        carry_stride: 0,
        w0: std::ptr::null_mut(),
        w0_stride: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoveCfg {
    pub rows: u64,
    pub cols: u64,
    pub outer_stride_bytes: u64,
    pub packed: u64,
    pub use_nt: u64,
}

pub type FnAddN = unsafe extern "C" fn(*mut u64, *const u64, *const u64, u64) -> u64;
pub type FnSubN = unsafe extern "C" fn(*mut u64, *const u64, *const u64, u64) -> u64;
pub type FnMul1 = unsafe extern "C" fn(*mut u64, *const u64, u64, u64) -> u64;
pub type FnAddMul1 = unsafe extern "C" fn(*mut u64, *const u64, u64, u64) -> u64;
pub type FnMulBase = unsafe extern "C" fn(*mut u64, *const u64, *const u64, *const [u64; 3]);
pub type FnSqrBase = unsafe extern "C" fn(*mut u64, *const u64, u64, u64);

pub type FnFftSplit = unsafe extern "C" fn(*mut f64, *const u64, *const f64, *const FftSplitCfg);
pub type FnFftFwd = unsafe extern "C" fn(*mut f64, *const f64, u64, u64);
pub type FnFftInv = unsafe extern "C" fn(*mut f64, *const f64, u64, u64);
pub type FnFftPmul = unsafe extern "C" fn(*mut f64, *const f64, *const f64, u64);
pub type FnFftPmulAdd = unsafe extern "C" fn(*mut f64, *const [*const f64; 4], u64, u64);
pub type FnFftCarry = unsafe extern "C" fn(*mut u64, *const f64, *const f64, *mut FftCarryCfg);

pub type FnNttSplit = unsafe extern "C" fn(*mut f64, *const u64, *const NttSplitCfg);
pub type FnNttFwd = unsafe extern "C" fn(*mut f64, *const f64, u64, *const NttPrime);
pub type FnNttInv = unsafe extern "C" fn(*mut f64, *const f64, u64, *const NttPrime);
pub type FnNttFwdCols = unsafe extern "C" fn(*mut f64, *const f64, *const NttColsCfg, *const NttPrime);
pub type FnNttInvCols = unsafe extern "C" fn(*mut f64, *const f64, *const NttColsCfg, *const NttPrime);
pub type FnNttPmul = unsafe extern "C" fn(*mut f64, *const f64, *const f64, *const NttVecCfg);
pub type FnNttPmulAdd = unsafe extern "C" fn(*mut f64, *const [*const f64; 4], *const NttVecCfg);
pub type FnNttCrt = unsafe extern "C" fn(*mut u64, *const *const f64, *mut NttCrtCfg) -> u64;

pub type FnGather = unsafe extern "C" fn(*mut f64, *const u8, *const MoveCfg);
pub type FnScatter = unsafe extern "C" fn(*mut u8, *const f64, *const MoveCfg);

pub type FnToDigits = unsafe extern "C" fn(*mut u8, *const u64, u64, u64) -> u64;
pub type FnCheckN = unsafe extern "C" fn(*const u64, u64, *mut [u64; 3], u64) -> u64;

pub type BindFn = unsafe extern "C" fn(*mut usize, u64);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelTable {
    pub add_n: FnAddN,
    pub sub_n: FnSubN,
    pub mul_1: FnMul1,
    pub addmul_1: FnAddMul1,
    pub mul_base: FnMulBase,
    pub sqr_base: FnSqrBase,

    pub fft_split: FnFftSplit,
    pub fft_fwd: FnFftFwd,
    pub fft_inv: FnFftInv,
    pub fft_pmul: FnFftPmul,
    pub fft_pmuladd: FnFftPmulAdd,
    pub fft_carry: FnFftCarry,

    pub ntt_split: FnNttSplit,
    pub ntt_fwd: FnNttFwd,
    pub ntt_inv: FnNttInv,
    pub ntt_fwd_cols: FnNttFwdCols,
    pub ntt_inv_cols: FnNttInvCols,
    pub ntt_pmul: FnNttPmul,
    pub ntt_pmuladd: FnNttPmulAdd,
    pub ntt_crt: FnNttCrt,

    pub gather: FnGather,
    pub scatter: FnScatter,

    pub to_digits: FnToDigits,
    pub check_n: FnCheckN,
}

impl KernelTable {
    pub const fn reference() -> Self {
        Self {
            add_n: ref_add_n,
            sub_n: ref_sub_n,
            mul_1: ref_mul_1,
            addmul_1: ref_addmul_1,
            mul_base: ref_mul_base,
            sqr_base: ref_sqr_base,
            fft_split: ref_fft_split,
            fft_fwd: ref_fft_fwd,
            fft_inv: ref_fft_inv,
            fft_pmul: ref_fft_pmul,
            fft_pmuladd: ref_fft_pmuladd,
            fft_carry: ref_fft_carry,
            ntt_split: ref_ntt_split,
            ntt_fwd: ref_ntt_fwd,
            ntt_inv: ref_ntt_inv,
            ntt_fwd_cols: ref_ntt_fwd_cols,
            ntt_inv_cols: ref_ntt_inv_cols,
            ntt_pmul: ref_ntt_pmul,
            ntt_pmuladd: ref_ntt_pmuladd,
            ntt_crt: ref_ntt_crt,
            gather: ref_gather,
            scatter: ref_scatter,
            to_digits: ref_to_digits,
            check_n: ref_check_n,
        }
    }

    pub fn as_raw_ptrs(&self) -> [usize; NUM_KERNELS] {
        // SAFETY: KernelTable is #[repr(C)] with exactly 24 function pointers.
        unsafe { std::mem::transmute_copy(self) }
    }

    pub unsafe fn from_raw_ptrs(ptrs: &[usize; NUM_KERNELS]) -> Self {
        unsafe { std::mem::transmute_copy(ptrs) }
    }
}

#[derive(Default)]
pub struct WorkStats {
    pub limb_ops: AtomicU64,
    pub fft_butterflies: AtomicU64,
    pub ntt_butterflies: AtomicU64,
    pub crt_ops: AtomicU64,
    pub bytes_moved: AtomicU64,
    pub check_limbs: AtomicU64,
    pub digit_words: AtomicU64,
}

impl WorkStats {
    pub fn snapshot(&self) -> WorkSnapshot {
        WorkSnapshot {
            limb_ops: self.limb_ops.load(Ordering::Relaxed),
            fft_butterflies: self.fft_butterflies.load(Ordering::Relaxed),
            ntt_butterflies: self.ntt_butterflies.load(Ordering::Relaxed),
            crt_ops: self.crt_ops.load(Ordering::Relaxed),
            bytes_moved: self.bytes_moved.load(Ordering::Relaxed),
            check_limbs: self.check_limbs.load(Ordering::Relaxed),
            digit_words: self.digit_words.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkSnapshot {
    pub limb_ops: u64,
    pub fft_butterflies: u64,
    pub ntt_butterflies: u64,
    pub crt_ops: u64,
    pub bytes_moved: u64,
    pub check_limbs: u64,
    pub digit_words: u64,
}

#[derive(Clone, Debug)]
pub struct BindReport {
    pub bound_mask: u32,
    pub mismatched: Vec<&'static str>,
    pub untested: Vec<&'static str>,
}

impl BindReport {
    pub fn bound_count(&self) -> u32 {
        self.bound_mask.count_ones()
    }
}

pub struct Kernels {
    pub active: KernelTable,
    pub reference: KernelTable,
    pub bound_mask: u32,
    pub stats: WorkStats,
    pub count_work: bool,
    /// Calibrated row-vs-vector crossover for `mul_base` (`u64::MAX` before calibration).
    pub mul_base_rows_max: AtomicU64,
    /// Calibrated scalar-triangle-vs-vector crossover for `sqr_base` (`u64::MAX` before calibration).
    pub sqr_base_rows_max: AtomicU64,
    /// Cache-derived L1 sub-transform complex count for `fft_fwd` and `fft_inv`.
    pub fft_l1_sub_n: AtomicU64,
    /// Cache-derived L1 sub-transform length for `ntt_fwd` and `ntt_inv`.
    pub ntt_l1_sub_n: AtomicU64,
    /// Cache-derived L2 sub-transform length for `ntt_fwd` and `ntt_inv`.
    pub ntt_l2_sub_n: AtomicU64,
    /// Cache-derived L1 strip row block for `ntt_fwd_cols` and `ntt_inv_cols`.
    pub ntt_cols_l1_sub_rows: AtomicU64,
    /// Cache-derived L2 strip row block for `ntt_fwd_cols` and `ntt_inv_cols`.
    pub ntt_cols_l2_sub_rows: AtomicU64,
}

impl Kernels {
    pub fn reference_only() -> Self {
        let reference = KernelTable::reference();
        Self {
            active: reference,
            reference,
            bound_mask: 0,
            stats: WorkStats::default(),
            count_work: false,
            mul_base_rows_max: AtomicU64::new(u64::MAX),
            sqr_base_rows_max: AtomicU64::new(u64::MAX),
            fft_l1_sub_n: AtomicU64::new(u64::MAX),
            ntt_l1_sub_n: AtomicU64::new(u64::MAX),
            ntt_l2_sub_n: AtomicU64::new(u64::MAX),
            ntt_cols_l1_sub_rows: AtomicU64::new(u64::MAX),
            ntt_cols_l2_sub_rows: AtomicU64::new(u64::MAX),
        }
    }

    /// Populate every cache-derived sub-block parameter on `Kernels` from the run's `Shares`.
    pub fn configure_from_shares(&self, shares: &crate::tier::Shares) {
        self.fft_l1_sub_n
            .store(shares.fft_l1_sub_n, Ordering::Relaxed);
        self.ntt_l1_sub_n
            .store(shares.ntt_l1_sub_n, Ordering::Relaxed);
        self.ntt_l2_sub_n
            .store(shares.ntt_l2_sub_n, Ordering::Relaxed);
        self.ntt_cols_l1_sub_rows
            .store(shares.ntt_cols_l1_sub_rows, Ordering::Relaxed);
        self.ntt_cols_l2_sub_rows
            .store(shares.ntt_cols_l2_sub_rows, Ordering::Relaxed);
    }

    /// The lengths `mul_base` takes for operands of `na` and `nb` limbs.
    #[inline(always)]
    pub fn mul_base_lens(&self, na: usize, nb: usize) -> [u64; 3] {
        [na as u64, nb as u64, self.mul_base_rows_max.load(Ordering::Relaxed)]
    }

    #[inline(always)]
    pub fn sqr_rows_max(&self) -> u64 {
        self.sqr_base_rows_max.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn fft_sub_n(&self) -> u64 {
        self.fft_l1_sub_n.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ntt_prime(&self, p_int: u64, scale_int: u64) -> NttPrime {
        NttPrime::with_blocks(
            p_int,
            scale_int,
            self.ntt_l1_sub_n.load(Ordering::Relaxed),
            self.ntt_l2_sub_n.load(Ordering::Relaxed),
        )
    }

    /// Number of `f64` elements needed when `strip_rows` exceeds stack-local scratch (`> 4096`
    /// rows = `> 256 KiB`, i.e. `> 1/32` of an 8 MiB worker stack); returns `0` when the strip
    /// fits in the worker's hot stack frame (`NttColsCfg.scratch = null`).
    #[inline(always)]
    pub fn ntt_cols_scratch_f64s(&self, rows: usize) -> usize {
        let l1 = self.ntt_cols_l1_sub_rows.load(Ordering::Relaxed);
        let l2 = self.ntt_cols_l2_sub_rows.load(Ordering::Relaxed);
        let r = rows as u64;
        let strip_rows = if r <= l2 { r } else { l1.min(r) };
        if strip_rows <= 4096 {
            0
        } else {
            (strip_rows as usize).saturating_mul(8).saturating_add(8)
        }
    }

    #[inline(always)]
    pub fn ntt_cols_cfg(&self, rows: usize, cols: usize, scratch_buf: &mut [f64]) -> NttColsCfg {
        let l1 = self.ntt_cols_l1_sub_rows.load(Ordering::Relaxed);
        let l2 = self.ntt_cols_l2_sub_rows.load(Ordering::Relaxed);
        let r = rows as u64;
        let strip_rows = if r <= l2 { r } else { l1.min(r) };
        let need_f64s = (strip_rows as usize).saturating_mul(8);
        let scratch = if need_f64s > 0 && scratch_buf.len() >= need_f64s {
            let base = scratch_buf.as_mut_ptr() as usize;
            let aligned = (base + 63) & !63usize;
            let off_f64 = (aligned - base) / std::mem::size_of::<f64>();
            if off_f64 + need_f64s <= scratch_buf.len() {
                unsafe { scratch_buf.as_mut_ptr().add(off_f64) }
            } else {
                std::ptr::null_mut()
            }
        } else {
            std::ptr::null_mut()
        };
        NttColsCfg {
            rows: r,
            cols: cols as u64,
            l1_sub_rows: l1,
            l2_sub_rows: l2,
            scratch,
        }
    }

    pub fn bind_and_verify(bind_fn: BindFn) -> (Self, BindReport) {
        let reference = KernelTable::reference();
        let ref_ptrs = reference.as_raw_ptrs();
        let mut cand_ptrs = ref_ptrs;
        unsafe {
            bind_fn(cand_ptrs.as_mut_ptr(), SUPI_ABI_VERSION);
        }
        let mut cand = unsafe { KernelTable::from_raw_ptrs(&cand_ptrs) };
        let mut bound_mask: u32 = 0;
        for i in 0..NUM_KERNELS {
            if cand_ptrs[i] != ref_ptrs[i] {
                bound_mask |= 1u32 << i;
            }
        }

        let mut mismatched = Vec::new();
        let mut untested = Vec::new();

        // FFT family (indices 6..12) is bound, tested, and restored as one unit (DESIGN.md s7).
        let fft_mask = 0b111111u32 << 6;
        if (bound_mask & fft_mask) != 0 {
            if (bound_mask & fft_mask) != fft_mask || !selftest_fft_chain(&cand, &reference) {
                for idx in 6..12 {
                    if (bound_mask & (1u32 << idx)) != 0 {
                        mismatched.push(KERNEL_NAMES[idx]);
                    }
                    cand_ptrs[idx] = ref_ptrs[idx];
                }
                bound_mask &= !fft_mask;
                cand = unsafe { KernelTable::from_raw_ptrs(&cand_ptrs) };
            }
        } else {
            for idx in 6..12 {
                untested.push(KERNEL_NAMES[idx]);
            }
        }

        // Test each of the other 18 kernels individually against its reference.
        for idx in (0..6).chain(12..24) {
            if (bound_mask & (1u32 << idx)) != 0 {
                if !selftest_single_kernel(idx, &cand, &reference) {
                    mismatched.push(KERNEL_NAMES[idx]);
                    cand_ptrs[idx] = ref_ptrs[idx];
                    bound_mask &= !(1u32 << idx);
                    cand = unsafe { KernelTable::from_raw_ptrs(&cand_ptrs) };
                }
            } else {
                untested.push(KERNEL_NAMES[idx]);
            }
        }

        let report = BindReport {
            bound_mask,
            mismatched,
            untested,
        };
        let kernels = Self {
            active: cand,
            reference,
            bound_mask,
            stats: WorkStats::default(),
            count_work: false,
            mul_base_rows_max: AtomicU64::new(u64::MAX),
            sqr_base_rows_max: AtomicU64::new(u64::MAX),
            fft_l1_sub_n: AtomicU64::new(u64::MAX),
            ntt_l1_sub_n: AtomicU64::new(u64::MAX),
            ntt_l2_sub_n: AtomicU64::new(u64::MAX),
            ntt_cols_l1_sub_rows: AtomicU64::new(u64::MAX),
            ntt_cols_l2_sub_rows: AtomicU64::new(u64::MAX),
        };
        (kernels, report)
    }

    pub fn with_reference_view(&self) -> Self {
        Self {
            active: self.reference,
            reference: self.reference,
            bound_mask: 0,
            stats: WorkStats::default(),
            count_work: self.count_work,
            mul_base_rows_max: AtomicU64::new(self.mul_base_rows_max.load(Ordering::Relaxed)),
            sqr_base_rows_max: AtomicU64::new(self.sqr_base_rows_max.load(Ordering::Relaxed)),
            fft_l1_sub_n: AtomicU64::new(self.fft_l1_sub_n.load(Ordering::Relaxed)),
            ntt_l1_sub_n: AtomicU64::new(self.ntt_l1_sub_n.load(Ordering::Relaxed)),
            ntt_l2_sub_n: AtomicU64::new(self.ntt_l2_sub_n.load(Ordering::Relaxed)),
            ntt_cols_l1_sub_rows: AtomicU64::new(self.ntt_cols_l1_sub_rows.load(Ordering::Relaxed)),
            ntt_cols_l2_sub_rows: AtomicU64::new(self.ntt_cols_l2_sub_rows.load(Ordering::Relaxed)),
        }
    }
}

// ============================================================================
// Exact Arithmetic Helpers
// ============================================================================

#[inline(always)]
pub fn mod61_reduce(x: u128) -> u64 {
    let m = MOD_W0 as u128;
    let s1 = (x & m) + (x >> 61);
    let s2 = (s1 & m) + (s1 >> 61);
    let r = s2 as u64;
    if r >= MOD_W0 { r - MOD_W0 } else { r }
}

#[inline(always)]
pub fn mod61_add(a: u64, b: u64) -> u64 {
    debug_assert!(a < MOD_W0 && b < MOD_W0);
    let s = a + b;
    if s >= MOD_W0 { s - MOD_W0 } else { s }
}

#[inline(always)]
pub fn mod61_sub(a: u64, b: u64) -> u64 {
    debug_assert!(a < MOD_W0 && b < MOD_W0);
    if a >= b { a - b } else { a + MOD_W0 - b }
}

#[inline(always)]
pub fn mod61_mul(a: u64, b: u64) -> u64 {
    debug_assert!(a < MOD_W0 && b < MOD_W0);
    mod61_reduce((a as u128) * (b as u128))
}

#[inline(always)]
pub fn mod61_pow2(bit_shift: u64) -> u64 {
    let rem = (bit_shift % 61) as u32;
    1u64 << rem
}

#[inline(always)]
pub fn fma_mul_mod(a: f64, b: f64, p: f64, inv_p: f64) -> f64 {
    debug_assert!(a >= 0.0 && a < p && b >= 0.0 && b < p);
    let h = a * b;
    let l = a.mul_add(b, -h);
    let q = (h * inv_p).round_ties_even();
    let r_hi = (-q).mul_add(p, h);
    let r = r_hi + l;
    let out = if r < 0.0 { r + p } else { r };
    debug_assert_eq!(
        out as u64 as u128,
        ((a as u64 as u128) * (b as u64 as u128)) % (p as u64 as u128)
    );
    out
}

#[inline(always)]
pub fn fma_add_mod(a: f64, b: f64, p: f64) -> f64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

#[inline(always)]
pub fn fma_sub_mod(a: f64, b: f64, p: f64) -> f64 {
    let d = a - b;
    if d < 0.0 { d + p } else { d }
}

// ============================================================================
// Family 1: Reference Limb Kernels (6)
// ============================================================================

pub unsafe extern "C" fn ref_add_n(dst: *mut u64, a: *const u64, b: *const u64, n: u64) -> u64 {
    let len = n as usize;
    let mut carry = 0u64;
    for i in 0..len {
        let av = unsafe { *a.add(i) };
        let bv = unsafe { *b.add(i) };
        let (s1, c1) = av.overflowing_add(bv);
        let (s2, c2) = s1.overflowing_add(carry);
        unsafe { *dst.add(i) = s2 };
        carry = (c1 as u64) + (c2 as u64);
    }
    carry
}

pub unsafe extern "C" fn ref_sub_n(dst: *mut u64, a: *const u64, b: *const u64, n: u64) -> u64 {
    let len = n as usize;
    let mut borrow = 0u64;
    for i in 0..len {
        let av = unsafe { *a.add(i) };
        let bv = unsafe { *b.add(i) };
        let (d1, b1) = av.overflowing_sub(bv);
        let (d2, b2) = d1.overflowing_sub(borrow);
        unsafe { *dst.add(i) = d2 };
        borrow = (b1 as u64) + (b2 as u64);
    }
    borrow
}

pub unsafe extern "C" fn ref_mul_1(dst: *mut u64, a: *const u64, n: u64, m: u64) -> u64 {
    let len = n as usize;
    let mul = m as u128;
    let mut carry = 0u128;
    for i in 0..len {
        let av = unsafe { *a.add(i) } as u128;
        let prod = av * mul + carry;
        unsafe { *dst.add(i) = prod as u64 };
        carry = prod >> 64;
    }
    carry as u64
}

pub unsafe extern "C" fn ref_addmul_1(dst: *mut u64, a: *const u64, n: u64, m: u64) -> u64 {
    let len = n as usize;
    let mul = m as u128;
    let mut carry = 0u128;
    for i in 0..len {
        let av = unsafe { *a.add(i) } as u128;
        let dv = unsafe { *dst.add(i) } as u128;
        let prod = av * mul + dv + carry;
        unsafe { *dst.add(i) = prod as u64 };
        carry = prod >> 64;
    }
    carry as u64
}

/// `dst[..na + nb] = a * b`. `lens` is `[na, nb, rows_max]`: the limbs of the two operands, and
/// the length of the shorter operand up to which a body with two ways of multiplying stays on
/// its rows of limbs. The plan measures that length (`Calibration::mul_base_rows_max`); it
/// changes no result, and this plain form, which has only rows, takes no notice of it.
pub unsafe extern "C" fn ref_mul_base(
    dst: *mut u64,
    a: *const u64,
    b: *const u64,
    lens: *const [u64; 3],
) {
    let [na, nb, _rows_max] = unsafe { *lens };
    let na = na as usize;
    let nb = nb as usize;
    debug_assert!(na > 0 && nb > 0);
    for i in 0..(na + nb) {
        unsafe { *dst.add(i) = 0 };
    }
    for j in 0..nb {
        let bj = unsafe { *b.add(j) };
        let c = unsafe { ref_addmul_1(dst.add(j), a, na as u64, bj) };
        unsafe { *dst.add(j + na) = c };
    }
}

pub unsafe extern "C" fn ref_sqr_base(dst: *mut u64, a: *const u64, n: u64, _rows_max: u64) {
    let len = n as usize;
    if len == 0 {
        return;
    }
    for i in 0..(2 * len) {
        unsafe { *dst.add(i) = 0 };
    }
    for i in 0..len {
        let ai = unsafe { *a.add(i) };
        if i + 1 < len {
            let c = unsafe { ref_addmul_1(dst.add(2 * i + 1), a.add(i + 1), (len - 1 - i) as u64, ai) };
            unsafe { *dst.add(i + len) = c };
        }
    }
    let mut shift_carry = 0u64;
    for i in 0..(2 * len) {
        let v = unsafe { *dst.add(i) };
        unsafe { *dst.add(i) = (v << 1) | shift_carry };
        shift_carry = v >> 63;
    }
    let mut carry = 0u128;
    for i in 0..len {
        let ai = unsafe { *a.add(i) } as u128;
        let sq = ai * ai;
        let lo = (sq as u64) as u128 + (unsafe { *dst.add(2 * i) } as u128) + carry;
        unsafe { *dst.add(2 * i) = lo as u64 };
        let hi = (sq >> 64) + (unsafe { *dst.add(2 * i + 1) } as u128) + (lo >> 64);
        unsafe { *dst.add(2 * i + 1) = hi as u64 };
        carry = hi >> 64;
    }
}

// ============================================================================
// Family 2: Reference FFT Kernels (6) — Right-Angle Weighted Complex FFT
// ============================================================================

#[inline(always)]
fn extract_bits_u64(limbs: &[u64], bit_pos: usize, width: usize) -> u64 {
    debug_assert!(width > 0 && width <= 32);
    let limb_idx = bit_pos >> 6;
    if limb_idx >= limbs.len() {
        return 0;
    }
    let bit_off = bit_pos & 63;
    let mask = (1u64 << width) - 1;
    let mut val = limbs[limb_idx] >> bit_off;
    if bit_off + width > 64 && limb_idx + 1 < limbs.len() {
        val |= limbs[limb_idx + 1] << (64 - bit_off);
    }
    val & mask
}

pub unsafe extern "C" fn ref_fft_split(
    out_complex: *mut f64,
    limbs: *const u64,
    weights: *const f64,
    cfg: *const FftSplitCfg,
) {
    let cfg = unsafe { *cfg };
    let in_len = cfg.in_limbs as usize;
    let n = cfg.n_complex as usize;
    let b = cfg.coeff_bits as usize;
    debug_assert!(n.is_power_of_two() && b > 0 && b <= 24 && b % 61 != 0);

    let limb_slice = if in_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(limbs, in_len) }
    };

    let m_digits = 2 * n;
    let half = 1i64 << (b - 1);
    let base = 1i64 << b;
    let mut digits = vec![0i64; m_digits];
    let mut borrow_carry = 0i64;
    let total_in_bits = in_len * 64;
    for k in 0..m_digits {
        let bit_pos = k * b;
        let raw = if bit_pos < total_in_bits {
            extract_bits_u64(limb_slice, bit_pos, b) as i64
        } else {
            0
        };
        let mut v = raw + borrow_carry;
        if v >= half && k + 1 < m_digits {
            v -= base;
            borrow_carry = 1;
        } else if v >= base && k + 1 < m_digits {
            v -= base;
            borrow_carry = 1;
        } else {
            borrow_carry = 0;
        }
        digits[k] = v;
    }

    for j in 0..n {
        let xr = digits[j] as f64;
        let xi = digits[j + n] as f64;
        let wr = unsafe { *weights.add(2 * j) };
        let wi = unsafe { *weights.add(2 * j + 1) };
        unsafe {
            *out_complex.add(2 * j) = xr.mul_add(wr, -(xi * wi));
            *out_complex.add(2 * j + 1) = xr.mul_add(wi, xi * wr);
        }
    }
}

pub unsafe extern "C" fn ref_fft_fwd(
    buf: *mut f64,
    twiddles: *const f64,
    n_complex: u64,
    _l1_sub_n: u64,
) {
    let n = n_complex as usize;
    debug_assert!(n.is_power_of_two());
    let mut b_blocks = 1usize;
    let mut m = n >> 1;
    while m >= 1 {
        for j in 0..b_blocks {
            let wr = unsafe { *twiddles.add(2 * (b_blocks + j)) };
            let wi = unsafe { *twiddles.add(2 * (b_blocks + j) + 1) };
            let base = j * 2 * m;
            for i in 0..m {
                let i0 = 2 * (base + i);
                let i1 = 2 * (base + m + i);
                let ur = unsafe { *buf.add(i0) };
                let ui = unsafe { *buf.add(i0 + 1) };
                let xr = unsafe { *buf.add(i1) };
                let xi = unsafe { *buf.add(i1 + 1) };
                let vr = xr.mul_add(wr, -(xi * wi));
                let vi = xr.mul_add(wi, xi * wr);
                unsafe {
                    *buf.add(i0) = ur + vr;
                    *buf.add(i0 + 1) = ui + vi;
                    *buf.add(i1) = ur - vr;
                    *buf.add(i1 + 1) = ui - vi;
                }
            }
        }
        b_blocks <<= 1;
        m >>= 1;
    }
}

pub unsafe extern "C" fn ref_fft_inv(
    buf: *mut f64,
    inv_twiddles: *const f64,
    n_complex: u64,
    _l1_sub_n: u64,
) {
    let n = n_complex as usize;
    debug_assert!(n.is_power_of_two());
    let mut m = 1usize;
    while m < n {
        let b_blocks = n / (2 * m);
        for j in 0..b_blocks {
            let wr = unsafe { *inv_twiddles.add(2 * (b_blocks + j)) };
            let wi = unsafe { *inv_twiddles.add(2 * (b_blocks + j) + 1) };
            let base = j * 2 * m;
            for i in 0..m {
                let i0 = 2 * (base + i);
                let i1 = 2 * (base + m + i);
                let ur = unsafe { *buf.add(i0) };
                let ui = unsafe { *buf.add(i0 + 1) };
                let vr = unsafe { *buf.add(i1) };
                let vi = unsafe { *buf.add(i1 + 1) };
                let dr = ur - vr;
                let di = ui - vi;
                unsafe {
                    *buf.add(i0) = ur + vr;
                    *buf.add(i0 + 1) = ui + vi;
                    *buf.add(i1) = dr.mul_add(wr, -(di * wi));
                    *buf.add(i1 + 1) = dr.mul_add(wi, di * wr);
                }
            }
        }
        m <<= 1;
    }
    let scale = 1.0 / (n as f64);
    for i in 0..(2 * n) {
        unsafe { *buf.add(i) *= scale };
    }
}

pub unsafe extern "C" fn ref_fft_pmul(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    n_complex: u64,
) {
    let n = n_complex as usize;
    for j in 0..n {
        let ar = unsafe { *a.add(2 * j) };
        let ai = unsafe { *a.add(2 * j + 1) };
        let br = unsafe { *b.add(2 * j) };
        let bi = unsafe { *b.add(2 * j + 1) };
        unsafe {
            *dst.add(2 * j) = ar.mul_add(br, -(ai * bi));
            *dst.add(2 * j + 1) = ar.mul_add(bi, ai * br);
        }
    }
}

pub unsafe extern "C" fn ref_fft_pmuladd(
    dst: *mut f64,
    ptrs: *const [*const f64; 4],
    n_complex: u64,
    sub_second: u64,
) {
    let [a1, b1, a2, b2] = unsafe { *ptrs };
    let n = n_complex as usize;
    let sign = if sub_second != 0 { -1.0 } else { 1.0 };
    for j in 0..n {
        let a1r = unsafe { *a1.add(2 * j) };
        let a1i = unsafe { *a1.add(2 * j + 1) };
        let b1r = unsafe { *b1.add(2 * j) };
        let b1i = unsafe { *b1.add(2 * j + 1) };

        let a2r = unsafe { *a2.add(2 * j) };
        let a2i = unsafe { *a2.add(2 * j + 1) };
        let b2r = unsafe { *b2.add(2 * j) };
        let b2i = unsafe { *b2.add(2 * j + 1) };

        let p1r = a1r.mul_add(b1r, -(a1i * b1i));
        let p1i = a1r.mul_add(b1i, a1i * b1r);
        let p2r = a2r.mul_add(b2r, -(a2i * b2i));
        let p2i = a2r.mul_add(b2i, a2i * b2r);

        unsafe {
            *dst.add(2 * j) = p1r + sign * p2r;
            *dst.add(2 * j + 1) = p1i + sign * p2i;
        }
    }
}

pub unsafe extern "C" fn ref_fft_carry(
    out_limbs: *mut u64,
    buf: *const f64,
    inv_weights: *const f64,
    cfg: *mut FftCarryCfg,
) {
    let c = unsafe { &mut *cfg };
    let out_len = c.out_limbs as usize;
    let n = c.n_complex as usize;
    let b = c.coeff_bits as usize;

    let mut digits = vec![0i64; 2 * n];
    let mut max_err = 0.0f64;
    for j in 0..n {
        let zr = unsafe { *buf.add(2 * j) };
        let zi = unsafe { *buf.add(2 * j + 1) };
        let wr = unsafe { *inv_weights.add(2 * j) };
        let wi = unsafe { *inv_weights.add(2 * j + 1) };
        let yr = zr.mul_add(wr, -(zi * wi));
        let yi = zr.mul_add(wi, zi * wr);
        let rr = yr.round_ties_even();
        let ri = yi.round_ties_even();
        let er = (yr - rr).abs();
        let ei = (yi - ri).abs();
        if er > max_err {
            max_err = er;
        }
        if ei > max_err {
            max_err = ei;
        }
        digits[j] = rr as i64;
        digits[j + n] = ri as i64;
    }

    for i in 0..out_len {
        unsafe { *out_limbs.add(i) = 0 };
    }

    let mask = (1i128 << b) - 1;
    let mut carry: i128 = 0;
    let mut bit_buf: u128 = 0;
    let mut bit_cnt: usize = 0;
    let mut limb_idx: usize = 0;

    for &d in &digits {
        let val = (d as i128) + carry;
        let coeff = (val & mask) as u128;
        carry = val >> b;
        bit_buf |= coeff << bit_cnt;
        bit_cnt += b;
        while bit_cnt >= 64 {
            if limb_idx < out_len {
                unsafe { *out_limbs.add(limb_idx) = bit_buf as u64 };
            }
            limb_idx += 1;
            bit_buf >>= 64;
            bit_cnt -= 64;
        }
    }
    if carry > 0 {
        bit_buf |= (carry as u128) << bit_cnt;
    }
    while limb_idx < out_len && bit_buf != 0 {
        unsafe { *out_limbs.add(limb_idx) = bit_buf as u64 };
        limb_idx += 1;
        bit_buf >>= 64;
    }

    let mut dummy = [0u64; 3];
    let w0 = unsafe { ref_check_n(out_limbs, c.out_limbs, &mut dummy, 0) };
    c.top_carry = carry as i64 as u64;
    c.max_err = max_err;
    c.check_word = w0;
}

// ============================================================================
// Family 3: Reference NTT Kernels (8)
// ============================================================================

#[inline]
fn extract_coeff_mod_p(limbs: &[u64], bit_pos: usize, width: usize, p: f64, inv_p: f64) -> f64 {
    if bit_pos >= limbs.len() * 64 {
        return 0.0;
    }
    if width <= 32 {
        let v = extract_bits_u64(limbs, bit_pos, width) as f64;
        return if v >= p { v - p } else { v };
    }
    // Extract in 32-bit chunks from most-significant chunk down to least-significant chunk,
    // evaluating modulo p via exact f64 FMA modular multiplication.
    let two32_mod = ((1u64 << 32) % (p as u64)) as f64;
    let num_chunks = width.div_ceil(32);
    let mut acc = 0.0f64;
    for chunk in (0..num_chunks).rev() {
        let c_bit = bit_pos + chunk * 32;
        let c_w = (width - chunk * 32).min(32);
        let part = extract_bits_u64(limbs, c_bit, c_w) as f64;
        let part_red = if part >= p { part - p } else { part };
        if chunk + 1 < num_chunks {
            acc = fma_mul_mod(acc, two32_mod, p, inv_p);
        }
        acc = fma_add_mod(acc, part_red, p);
    }
    acc
}

#[inline]
fn extract_bits_u64_wide(limbs: &[u64], bit_pos: usize, width: usize) -> u64 {
    debug_assert!(width > 0 && width <= 64);
    let limb_idx = bit_pos >> 6;
    if limb_idx >= limbs.len() {
        return 0;
    }
    let bit_off = bit_pos & 63;
    let mut val = limbs[limb_idx] >> bit_off;
    if bit_off + width > 64 && limb_idx + 1 < limbs.len() {
        val |= limbs[limb_idx + 1] << (64 - bit_off);
    }
    if width < 64 {
        val & ((1u64 << width) - 1)
    } else {
        val
    }
}

/// Stride in `f64` elements between 48-bit planes produced/consumed by planar `ntt_split`
/// (`p == 0.0` extract and `in_limbs == u64::MAX` Horner reduce). Rounds `count >= 32` up to
/// a multiple of 8 plus 8 (`+64` bytes) so every plane starts 64-byte aligned and consecutive
/// planes stagger across cache sets instead of aliasing at power-of-two strides.
#[inline]
pub const fn ntt_planar_stride(count: usize) -> usize {
    if count >= 32 {
        ((count + 7) & !7) + 8
    } else {
        count
    }
}

pub unsafe extern "C" fn ref_ntt_split(
    dst: *mut f64,
    limbs: *const u64,
    cfg: *const NttSplitCfg,
) {
    let c = unsafe { *cfg };
    let count = c.count as usize;
    if count == 0 {
        return;
    }
    let b = c.coeff_bits as usize;
    if c.in_limbs == u64::MAX {
        let num_chunks = b.div_ceil(48);
        let stride_el = ntt_planar_stride(count);
        let total_el = (num_chunks - 1) * stride_el + count;
        let planes = unsafe { std::slice::from_raw_parts(limbs as *const f64, total_el) };
        let two48_mod = ((1u64 << 48) % (c.p as u64)) as f64;
        for i in 0..count {
            let mut acc = 0.0f64;
            for ch in (0..num_chunks).rev() {
                let mut part = planes[ch * stride_el + i];
                if part >= c.p {
                    part -= c.p;
                }
                if ch + 1 < num_chunks {
                    acc = fma_mul_mod(acc, two48_mod, c.p, c.inv_p);
                }
                acc = fma_add_mod(acc, part, c.p);
            }
            unsafe {
                *dst.add(i) = acc;
            }
        }
        return;
    }

    let in_len = c.in_limbs as usize;
    let limb_slice = if in_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(limbs, in_len) }
    };
    let start = c.start_coeff as usize;
    let stride = c.col_stride as usize;
    let cols = c.cols_per_row as usize;

    if c.p == 0.0 {
        let num_chunks = b.div_ceil(48);
        let stride_el = ntt_planar_stride(count);
        let emit_one = |out_i: usize, idx: usize| {
            let bit_pos = idx * b;
            for ch in 0..num_chunks {
                let c_bit = bit_pos + ch * 48;
                let c_w = (b - ch * 48).min(48);
                let part = if c_bit < limb_slice.len() * 64 {
                    extract_bits_u64_wide(limb_slice, c_bit, c_w) as f64
                } else {
                    0.0
                };
                unsafe {
                    *dst.add(ch * stride_el + out_i) = part;
                }
            }
        };
        if stride == 0 {
            for i in 0..count {
                emit_one(i, start + i);
            }
        } else {
            debug_assert!(cols > 0 && count % cols == 0);
            let rows = count / cols;
            for r in 0..rows {
                for col in 0..cols {
                    emit_one(r * cols + col, start + r * stride + col);
                }
            }
        }
        return;
    }

    if stride == 0 {
        for i in 0..count {
            let idx = start + i;
            unsafe {
                *dst.add(i) = extract_coeff_mod_p(limb_slice, idx * b, b, c.p, c.inv_p);
            }
        }
    } else {
        debug_assert!(cols > 0 && count % cols == 0);
        let rows = count / cols;
        for r in 0..rows {
            for col in 0..cols {
                let idx = start + r * stride + col;
                unsafe {
                    *dst.add(r * cols + col) =
                        extract_coeff_mod_p(limb_slice, idx * b, b, c.p, c.inv_p);
                }
            }
        }
    }
}

pub unsafe extern "C" fn ref_ntt_fwd(
    buf: *mut f64,
    twiddles: *const f64,
    n: u64,
    prime: *const NttPrime,
) {
    let n = n as usize;
    debug_assert!(n.is_power_of_two());
    let pr = unsafe { *prime };
    let p = pr.p;
    let inv_p = pr.inv_p;

    let mut b_blocks = 1usize;
    let mut m = n >> 1;
    while m >= 1 {
        for j in 0..b_blocks {
            let w = unsafe { *twiddles.add(b_blocks + j) };
            let base = j * 2 * m;
            for i in 0..m {
                let u = unsafe { *buf.add(base + i) };
                let x = unsafe { *buf.add(base + m + i) };
                let v = fma_mul_mod(x, w, p, inv_p);
                unsafe {
                    *buf.add(base + i) = fma_add_mod(u, v, p);
                    *buf.add(base + m + i) = fma_sub_mod(u, v, p);
                }
            }
        }
        b_blocks <<= 1;
        m >>= 1;
    }
}

pub unsafe extern "C" fn ref_ntt_inv(
    buf: *mut f64,
    inv_twiddles: *const f64,
    n: u64,
    prime: *const NttPrime,
) {
    let pr = unsafe { *prime };
    if !pr.fwd_tw.is_null() {
        let mut pr_fwd = pr;
        pr_fwd.scale = 1.0;
        pr_fwd.fwd_tw = std::ptr::null();
        pr_fwd.pmul_b = std::ptr::null();
        unsafe {
            ref_ntt_fwd(buf, pr.fwd_tw, n, &pr_fwd);
        }
    }
    if !pr.pmul_b.is_null() {
        let vcfg = NttVecCfg {
            n,
            p: pr.p,
            inv_p: pr.inv_p,
            sub_second: 0,
        };
        unsafe {
            ref_ntt_pmul(buf, buf as *const f64, pr.pmul_b, &vcfg);
        }
    }
    let n = n as usize;
    debug_assert!(n.is_power_of_two());
    let p = pr.p;
    let inv_p = pr.inv_p;

    let mut m = 1usize;
    while m < n {
        let b_blocks = n / (2 * m);
        for j in 0..b_blocks {
            let w_inv = unsafe { *inv_twiddles.add(b_blocks + j) };
            let base = j * 2 * m;
            for i in 0..m {
                let u = unsafe { *buf.add(base + i) };
                let v = unsafe { *buf.add(base + m + i) };
                let sum = fma_add_mod(u, v, p);
                let diff = fma_sub_mod(u, v, p);
                unsafe {
                    *buf.add(base + i) = sum;
                    *buf.add(base + m + i) = fma_mul_mod(diff, w_inv, p, inv_p);
                }
            }
        }
        m <<= 1;
    }

    if pr.scale != 1.0 {
        for i in 0..n {
            let v = unsafe { *buf.add(i) };
            unsafe { *buf.add(i) = fma_mul_mod(v, pr.scale, p, inv_p) };
        }
    }
}

pub unsafe extern "C" fn ref_ntt_fwd_cols(
    buf: *mut f64,
    twiddles: *const f64,
    cfg: *const NttColsCfg,
    prime: *const NttPrime,
) {
    let c = unsafe { *cfg };
    let rows = c.rows as usize;
    let cols = c.cols as usize;
    debug_assert!(rows.is_power_of_two());
    let pr = unsafe { *prime };
    let p = pr.p;
    let inv_p = pr.inv_p;

    let mut b_blocks = 1usize;
    let mut m = rows >> 1;
    while m >= 1 {
        for j in 0..b_blocks {
            let w = unsafe { *twiddles.add(b_blocks + j) };
            let base_row = j * 2 * m;
            for r in 0..m {
                let r0 = (base_row + r) * cols;
                let r1 = (base_row + m + r) * cols;
                for c in 0..cols {
                    let u = unsafe { *buf.add(r0 + c) };
                    let x = unsafe { *buf.add(r1 + c) };
                    let v = fma_mul_mod(x, w, p, inv_p);
                    unsafe {
                        *buf.add(r0 + c) = fma_add_mod(u, v, p);
                        *buf.add(r1 + c) = fma_sub_mod(u, v, p);
                    }
                }
            }
        }
        b_blocks <<= 1;
        m >>= 1;
    }
}

pub unsafe extern "C" fn ref_ntt_inv_cols(
    buf: *mut f64,
    inv_twiddles: *const f64,
    cfg: *const NttColsCfg,
    prime: *const NttPrime,
) {
    let c = unsafe { *cfg };
    let rows = c.rows as usize;
    let cols = c.cols as usize;
    debug_assert!(rows.is_power_of_two());
    let pr = unsafe { *prime };
    let p = pr.p;
    let inv_p = pr.inv_p;

    let mut m = 1usize;
    while m < rows {
        let b_blocks = rows / (2 * m);
        for j in 0..b_blocks {
            let w_inv = unsafe { *inv_twiddles.add(b_blocks + j) };
            let base_row = j * 2 * m;
            for r in 0..m {
                let r0 = (base_row + r) * cols;
                let r1 = (base_row + m + r) * cols;
                for c in 0..cols {
                    let u = unsafe { *buf.add(r0 + c) };
                    let v = unsafe { *buf.add(r1 + c) };
                    let sum = fma_add_mod(u, v, p);
                    let diff = fma_sub_mod(u, v, p);
                    unsafe {
                        *buf.add(r0 + c) = sum;
                        *buf.add(r1 + c) = fma_mul_mod(diff, w_inv, p, inv_p);
                    }
                }
            }
        }
        m <<= 1;
    }

    if pr.scale != 1.0 {
        let total = rows * cols;
        for i in 0..total {
            let v = unsafe { *buf.add(i) };
            unsafe { *buf.add(i) = fma_mul_mod(v, pr.scale, p, inv_p) };
        }
    }
}

pub unsafe extern "C" fn ref_ntt_pmul(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    cfg: *const NttVecCfg,
) {
    let c = unsafe { *cfg };
    let n = c.n as usize;
    if c.sub_second != 0 {
        let cols = c.sub_second as usize;
        debug_assert!(cols > 0 && n.is_multiple_of(cols));
        let rows = n / cols;
        for r in 0..rows {
            let sf = unsafe { *b.add(r) };
            let base = r * cols;
            for col in 0..cols {
                let idx = base + col;
                let tv = unsafe { *dst.add(idx) };
                let av = unsafe { *a.add(idx) };
                let t = fma_mul_mod(tv, av, c.p, c.inv_p);
                unsafe { *dst.add(idx) = fma_mul_mod(t, sf, c.p, c.inv_p) };
            }
        }
        return;
    }
    for i in 0..n {
        let av = unsafe { *a.add(i) };
        let bv = unsafe { *b.add(i) };
        unsafe { *dst.add(i) = fma_mul_mod(av, bv, c.p, c.inv_p) };
    }
}

pub unsafe extern "C" fn ref_ntt_pmuladd(
    dst: *mut f64,
    ptrs: *const [*const f64; 4],
    cfg: *const NttVecCfg,
) {
    let [a1, b1, a2, b2] = unsafe { *ptrs };
    let c = unsafe { *cfg };
    let n = c.n as usize;
    for i in 0..n {
        let p1 = if b1.is_null() {
            unsafe { *a1.add(i) }
        } else {
            fma_mul_mod(unsafe { *a1.add(i) }, unsafe { *b1.add(i) }, c.p, c.inv_p)
        };
        let p2 = fma_mul_mod(unsafe { *a2.add(i) }, unsafe { *b2.add(i) }, c.p, c.inv_p);
        let res = if c.sub_second != 0 {
            fma_sub_mod(p1, p2, c.p)
        } else {
            fma_add_mod(p1, p2, c.p)
        };
        unsafe { *dst.add(i) = res };
    }
}

#[inline]
fn big_add_mul_u64(acc: &mut [u64], prod_words: &[u64], v: u64) {
    let mut carry = 0u128;
    let mul = v as u128;
    for i in 0..prod_words.len() {
        let p = (prod_words[i] as u128) * mul + (acc[i] as u128) + carry;
        acc[i] = p as u64;
        carry = p >> 64;
    }
}

#[inline]
fn big_sub_assign(acc: &mut [u64], sub_words: &[u64]) {
    let mut borrow = 0u64;
    for i in 0..acc.len() {
        let (d1, b1) = acc[i].overflowing_sub(sub_words[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        acc[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
    }
}

#[inline]
fn big_add_signed(acc: &mut [u64], carry: &[u64]) {
    let mut c = 0u64;
    for i in 0..acc.len() {
        let (s1, c1) = acc[i].overflowing_add(carry[i]);
        let (s2, c2) = s1.overflowing_add(c);
        acc[i] = s2;
        c = (c1 as u64) + (c2 as u64);
    }
}

#[inline]
fn big_ashr(acc: &[u64], out: &mut [u64], shift: usize) {
    let k = acc.len();
    let sign_fill = if (acc[k - 1] as i64) < 0 { u64::MAX } else { 0u64 };
    let word_shift = shift >> 6;
    let bit_shift = shift & 63;
    for i in 0..k {
        let lo = if i + word_shift < k {
            acc[i + word_shift]
        } else {
            sign_fill
        };
        let hi = if bit_shift == 0 {
            0
        } else if i + word_shift + 1 < k {
            acc[i + word_shift + 1]
        } else {
            sign_fill
        };
        out[i] = if bit_shift == 0 {
            lo
        } else {
            (lo >> bit_shift) | (hi << (64 - bit_shift))
        };
    }
}

pub unsafe extern "C" fn ref_ntt_crt(
    out_limbs: *mut u64,
    residues: *const *const f64,
    cfg: *mut NttCrtCfg,
) -> u64 {
    let c = unsafe { &*cfg };
    let segs = c.segs;
    if segs.segments < 2 {
        return unsafe { ref_ntt_crt_one(out_limbs, residues, cfg) };
    }
    // Many streams: each one is the single-stream contract on its own pointers.
    let k_primes = c.k_primes as usize;
    let res = unsafe { std::slice::from_raw_parts(residues, k_primes) };
    let mut seg_res = [std::ptr::null::<f64>(); 18];
    for s in 0..(segs.segments as usize) {
        for i in 0..k_primes {
            seg_res[i] = unsafe { res[i].add(s * (segs.res_stride as usize)) };
        }
        let carry = unsafe { c.carry_limbs.add(s * (segs.carry_stride as usize)) };
        for w in 0..(c.k_words as usize) {
            unsafe { *carry.add(w) = 0 };
        }
        let mut one = NttCrtCfg {
            carry_limbs: carry,
            segs: NttCrtSegs::ONE,
            ..*c
        };
        let w0 = unsafe {
            ref_ntt_crt_one(
                out_limbs.add(s * (segs.out_stride as usize)),
                seg_res.as_ptr(),
                &mut one,
            )
        };
        unsafe { *segs.w0.add(s * (segs.w0_stride as usize)) = w0 };
    }
    0
}

unsafe fn ref_ntt_crt_one(
    out_limbs: *mut u64,
    residues: *const *const f64,
    cfg: *mut NttCrtCfg,
) -> u64 {
    let c = unsafe { &mut *cfg };
    let k_primes = c.k_primes as usize;
    let count = c.count as usize;
    let b = c.coeff_bits as usize;
    let out_len = c.out_limbs as usize;
    let k_words = c.k_words as usize;
    debug_assert!(k_primes >= 1 && k_primes <= 18 && k_words >= 2 && k_words <= 18);

    let primes = unsafe { std::slice::from_raw_parts(c.primes, k_primes) };
    let inv_primes = unsafe { std::slice::from_raw_parts(c.inv_primes, k_primes) };
    let garner_inv = unsafe { std::slice::from_raw_parts(c.garner_inv, k_primes) };
    let prefix_prods =
        unsafe { std::slice::from_raw_parts(c.prefix_prods, (k_primes + 1) * k_words) };
    let res_ptrs = unsafe { std::slice::from_raw_parts(residues, k_primes) };

    for i in 0..out_len {
        unsafe { *out_limbs.add(i) = 0 };
    }

    let mut carry = vec![0u64; k_words];
    if !c.carry_limbs.is_null() {
        for w in 0..k_words {
            carry[w] = unsafe { *c.carry_limbs.add(w) };
        }
    }

    let mut v = [0f64; 18];
    let mut coeff_val = vec![0u64; k_words];
    let mut next_carry = vec![0u64; k_words];

    let mut bit_pos = 0usize;
    for idx in 0..count {
        v[0] = unsafe { *res_ptrs[0].add(idx) };
        for i in 1..k_primes {
            let pi = primes[i];
            let inv_pi = inv_primes[i];
            let mut acc = v[i - 1];
            if acc >= pi {
                acc -= pi;
            }
            for j in (0..(i - 1)).rev() {
                let pj = primes[j];
                let pj_mod = if pj >= pi { pj - pi } else { pj };
                let vj_mod = if v[j] >= pi { v[j] - pi } else { v[j] };
                acc = fma_mul_mod(acc, pj_mod, pi, inv_pi);
                acc = fma_add_mod(acc, vj_mod, pi);
            }
            let ri = unsafe { *res_ptrs[i].add(idx) };
            let diff = fma_sub_mod(ri, acc, pi);
            v[i] = fma_mul_mod(diff, garner_inv[i], pi, inv_pi);
        }

        coeff_val.fill(0);
        coeff_val[0] = v[0] as u64;
        for i in 1..k_primes {
            let prod_slice = &prefix_prods[(i * k_words)..((i + 1) * k_words)];
            big_add_mul_u64(&mut coeff_val, prod_slice, v[i] as u64);
        }

        if c.signed != 0 {
            let top_p = primes[k_primes - 1] as u64;
            if (v[k_primes - 1] as u64) > (top_p >> 1) {
                let p_tot = &prefix_prods[(k_primes * k_words)..((k_primes + 1) * k_words)];
                big_sub_assign(&mut coeff_val, p_tot);
            }
        }

        big_add_signed(&mut coeff_val, &carry);

        // Extract low `b` bits of `coeff_val` and write to `out_limbs` at `bit_pos`.
        write_bits_from_words(out_limbs, out_len, bit_pos, b, &coeff_val);
        bit_pos += b;

        big_ashr(&coeff_val, &mut next_carry, b);
        carry.copy_from_slice(&next_carry);
    }

    if !c.carry_limbs.is_null() {
        for w in 0..k_words {
            unsafe { *c.carry_limbs.add(w) = carry[w] };
        }
    } else if c.wrap_cyclic != 0 {
        // Fold remaining carry out of L = count * b back into bit 0 (mod 2^L - 1).
        fold_cyclic_carry(out_limbs, out_len, bit_pos, &carry);
    } else {
        // Flush remaining positive carry into remaining output limbs above bit_pos.
        let rem_bits = out_len.saturating_sub(bit_pos >> 6) * 64 + 64;
        let max_flush = rem_bits.min(k_words * 64);
        if max_flush > 0 && bit_pos < out_len * 64 {
            let width = (out_len * 64 - bit_pos).min(max_flush);
            write_bits_from_words(out_limbs, out_len, bit_pos, width, &carry);
        }
    }

    let mut dummy = [0u64; 3];
    unsafe { ref_check_n(out_limbs, c.out_limbs, &mut dummy, 0) }
}

#[inline]
fn write_bits_from_words(out_limbs: *mut u64, out_len: usize, bit_pos: usize, width: usize, src: &[u64]) {
    let mut written = 0usize;
    while written < width {
        let dst_bit = bit_pos + written;
        let dst_limb = dst_bit >> 6;
        if dst_limb >= out_len {
            break;
        }
        let dst_off = dst_bit & 63;
        let chunk = (width - written).min(64 - dst_off);
        let src_word = written >> 6;
        let src_off = written & 63;
        let mut bits = src[src_word] >> src_off;
        if src_off + chunk > 64 && src_word + 1 < src.len() {
            bits |= src[src_word + 1] << (64 - src_off);
        }
        if chunk < 64 {
            bits &= (1u64 << chunk) - 1;
        }
        unsafe {
            *out_limbs.add(dst_limb) |= bits << dst_off;
        }
        written += chunk;
    }
}

pub(crate) fn fold_cyclic_carry(out_limbs: *mut u64, out_len: usize, total_bits: usize, carry_words: &[u64]) {
    if out_len == 0 || total_bits == 0 {
        return;
    }
    let neg = (carry_words[carry_words.len() - 1] as i64) < 0;
    let mut mag = carry_words.to_vec();
    if neg {
        let mut b = 0u64;
        for w in &mut mag {
            let (d1, b1) = 0u64.overflowing_sub(*w);
            let (d2, b2) = d1.overflowing_sub(b);
            *w = d2;
            b = (b1 as u64) + (b2 as u64);
        }
    }
    if mag.iter().all(|&x| x == 0) {
        return;
    }

    let top_limb = (total_bits - 1) >> 6;
    let top_bits = total_bits - top_limb * 64;
    let top_mask = if top_bits == 64 { u64::MAX } else { (1u64 << top_bits) - 1 };

    if !neg {
        let mut c = 0u64;
        for i in 0..=top_limb.min(out_len - 1) {
            let addv = if i < mag.len() { mag[i] } else { 0 };
            let cur = unsafe { *out_limbs.add(i) };
            let (s1, c1) = cur.overflowing_add(addv);
            let (s2, c2) = s1.overflowing_add(c);
            if i == top_limb {
                unsafe { *out_limbs.add(i) = s2 & top_mask };
                c = if top_bits == 64 {
                    (c1 as u64) + (c2 as u64)
                } else {
                    (s2 >> top_bits) + (((c1 as u64) + (c2 as u64)) << (64 - top_bits))
                };
            } else {
                unsafe { *out_limbs.add(i) = s2 };
                c = (c1 as u64) + (c2 as u64);
            }
        }
        while c > 0 {
            let mut cc = c;
            c = 0;
            for i in 0..=top_limb.min(out_len - 1) {
                let cur = unsafe { *out_limbs.add(i) };
                let (s, ov) = cur.overflowing_add(cc);
                if i == top_limb {
                    let extra = if top_bits == 64 { ov as u64 } else { s >> top_bits };
                    unsafe { *out_limbs.add(i) = s & top_mask };
                    c = extra;
                } else {
                    unsafe { *out_limbs.add(i) = s };
                    cc = ov as u64;
                    if cc == 0 {
                        break;
                    }
                }
            }
        }
    } else {
        let mut b = 0u64;
        for i in 0..=top_limb.min(out_len - 1) {
            let subv = if i < mag.len() { mag[i] } else { 0 };
            let cur = unsafe { *out_limbs.add(i) };
            let (d1, b1) = cur.overflowing_sub(subv);
            let (d2, b2) = d1.overflowing_sub(b);
            if i == top_limb {
                let under = ((b1 as u64) + (b2 as u64) > 0) || (top_bits < 64 && (d2 >> top_bits) != 0);
                unsafe { *out_limbs.add(i) = d2 & top_mask };
                b = under as u64;
            } else {
                unsafe { *out_limbs.add(i) = d2 };
                b = (b1 as u64) + (b2 as u64);
            }
        }
        if b > 0 {
            // Borrowed below 0 modulo 2^L - 1 -> subtract 1 more!
            let mut bb = 1u64;
            for i in 0..=top_limb.min(out_len - 1) {
                let cur = unsafe { *out_limbs.add(i) };
                let (d, un) = cur.overflowing_sub(bb);
                unsafe { *out_limbs.add(i) = if i == top_limb { d & top_mask } else { d } };
                bb = un as u64;
                if bb == 0 {
                    break;
                }
            }
        }
    }

    // Canonicalize 2^L - 1 -> 0 so output is strictly in [0, 2^L - 1).
    if top_limb < out_len {
        let mut all_ones = true;
        for i in 0..=top_limb {
            let expected = if i == top_limb { top_mask } else { u64::MAX };
            if unsafe { *out_limbs.add(i) } != expected {
                all_ones = false;
                break;
            }
        }
        if all_ones {
            for i in 0..=top_limb {
                unsafe { *out_limbs.add(i) = 0 };
            }
        }
    }
}

// ============================================================================
// Family 4: Reference Movement Kernels (2) — Strided & 50-Bit Packed Gather/Scatter
// ============================================================================

#[inline]
pub fn pack_8_residues_50bit(src: &[f64; 8], dst: &mut [u8; 50]) {
    dst.fill(0);
    for (j, &val) in src.iter().enumerate() {
        let u = val as u64;
        debug_assert!(u < (1u64 << 50));
        let bit_pos = j * 50;
        let byte_idx = bit_pos >> 3;
        let bit_off = bit_pos & 7;
        let shifted = (u as u128) << bit_off;
        let bytes = shifted.to_le_bytes();
        let n_copy = (50 - byte_idx).min(8);
        for k in 0..n_copy {
            dst[byte_idx + k] |= bytes[k];
        }
    }
}

#[inline]
pub fn unpack_8_residues_50bit(src: &[u8; 50], dst: &mut [f64; 8]) {
    let mask = (1u64 << 50) - 1;
    for j in 0..8 {
        let bit_pos = j * 50;
        let byte_idx = bit_pos >> 3;
        let bit_off = bit_pos & 7;
        let mut buf = [0u8; 8];
        let n_copy = (50 - byte_idx).min(8);
        buf[..n_copy].copy_from_slice(&src[byte_idx..(byte_idx + n_copy)]);
        let raw = u64::from_le_bytes(buf);
        let hi_byte = if byte_idx + 8 < 50 { src[byte_idx + 8] as u64 } else { 0 };
        let val = if bit_off == 0 {
            raw & mask
        } else {
            ((raw >> bit_off) | (hi_byte << (64 - bit_off))) & mask
        };
        dst[j] = val as f64;
    }
}

pub unsafe extern "C" fn ref_gather(tile_dst: *mut f64, outer_src: *const u8, cfg: *const MoveCfg) {
    let c = unsafe { *cfg };
    let rows = c.rows as usize;
    let cols = c.cols as usize;
    let stride = c.outer_stride_bytes as usize;

    if c.packed == 0 {
        for r in 0..rows {
            let src_row = unsafe { outer_src.add(r * stride) as *const f64 };
            let dst_row = unsafe { tile_dst.add(r * cols) };
            unsafe { std::ptr::copy_nonoverlapping(src_row, dst_row, cols) };
        }
    } else {
        debug_assert!(cols % 8 == 0);
        let blocks = cols / 8;
        let mut tmp = [0f64; 8];
        for r in 0..rows {
            let row_ptr = unsafe { outer_src.add(r * stride) };
            let dst_row = unsafe { tile_dst.add(r * cols) };
            for b in 0..blocks {
                let blk_bytes: &[u8; 50] =
                    unsafe { &*(row_ptr.add(b * 50) as *const [u8; 50]) };
                unpack_8_residues_50bit(blk_bytes, &mut tmp);
                unsafe { std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst_row.add(b * 8), 8) };
            }
        }
    }
}

pub unsafe extern "C" fn ref_scatter(outer_dst: *mut u8, tile_src: *const f64, cfg: *const MoveCfg) {
    let c = unsafe { *cfg };
    let rows = c.rows as usize;
    let cols = c.cols as usize;
    let stride = c.outer_stride_bytes as usize;

    if (c.packed & 1) == 0 {
        let src_stride_bytes = if (c.packed >> 8) != 0 {
            (c.packed >> 8) as usize
        } else {
            cols * 8
        };
        let src_u8 = tile_src as *const u8;
        for r in 0..rows {
            let src_row = unsafe { src_u8.add(r * src_stride_bytes) as *const f64 };
            let dst_row = unsafe { outer_dst.add(r * stride) as *mut f64 };
            unsafe { std::ptr::copy_nonoverlapping(src_row, dst_row, cols) };
        }
    } else {
        debug_assert!(cols % 8 == 0);
        let blocks = cols / 8;
        let mut packed_blk = [0u8; 50];
        for r in 0..rows {
            let src_row = unsafe { tile_src.add(r * cols) };
            let row_ptr = unsafe { outer_dst.add(r * stride) };
            for b in 0..blocks {
                let src_blk: &[f64; 8] =
                    unsafe { &*(src_row.add(b * 8) as *const [f64; 8]) };
                pack_8_residues_50bit(src_blk, &mut packed_blk);
                unsafe { std::ptr::copy_nonoverlapping(packed_blk.as_ptr(), row_ptr.add(b * 50), 50) };
            }
        }
    }
}

// ============================================================================
// Family 5: Reference Digits and Check-Word Kernels (2)
// ============================================================================

pub const POW10_19_MOD_W0: u64 = (10_000_000_000_000_000_000u128 % (MOD_W0 as u128)) as u64;

pub unsafe extern "C" fn ref_to_digits(
    out_buf: *mut u8,
    words_19dig: *const u64,
    n_words: u64,
    format_text: u64,
) -> u64 {
    let n = n_words as usize;
    let mut w0 = 0u64;
    for i in 0..n {
        let val = unsafe { *words_19dig.add(i) };
        debug_assert!(val < 10_000_000_000_000_000_000);
        let v_mod = mod61_reduce(val as u128);
        w0 = mod61_add(mod61_mul(w0, POW10_19_MOD_W0), v_mod);

        if format_text != 0 {
            let mut rem = val;
            let dst = unsafe { out_buf.add(i * 19) };
            for pos in (0..19).rev() {
                let dig = (rem % 10) as u8;
                rem /= 10;
                unsafe { *dst.add(pos) = b'0' + dig };
            }
        } else {
            let bytes = val.to_le_bytes();
            let dst = unsafe { out_buf.add(i * 8) };
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, 8) };
        }
    }
    w0
}

pub unsafe extern "C" fn ref_check_n(
    limbs: *const u64,
    n: u64,
    out_words: *mut [u64; 3],
    all_three: u64,
) -> u64 {
    let len = n as usize;
    let mut w0 = 0u64;
    for j in (0..len).rev() {
        let limb = unsafe { *limbs.add(j) };
        let l_mod = mod61_reduce(limb as u128);
        // Multiply w0 by 2^64 ≡ 8 (mod 2^61 - 1) via shifts and additions.
        let shifted = mod61_reduce((w0 as u128) << 3);
        w0 = mod61_add(shifted, l_mod);
    }

    if all_three == 0 {
        if !out_words.is_null() {
            unsafe { (*out_words)[0] = w0 };
        }
        return w0;
    }

    let p1 = PRIME_LOC1 as f64;
    let inv_p1 = 1.0 / p1;
    let two32_1 = ((1u64 << 32) % PRIME_LOC1) as f64;
    let two64_1 = ((1u128 << 64) % (PRIME_LOC1 as u128)) as f64;

    let p2 = PRIME_LOC2 as f64;
    let inv_p2 = 1.0 / p2;
    let two32_2 = ((1u64 << 32) % PRIME_LOC2) as f64;
    let two64_2 = ((1u128 << 64) % (PRIME_LOC2 as u128)) as f64;

    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    for j in (0..len).rev() {
        let limb = unsafe { *limbs.add(j) };
        let lo = (limb as u32) as f64;
        let hi = (limb >> 32) as f64;

        let v1 = fma_add_mod(lo, fma_mul_mod(hi, two32_1, p1, inv_p1), p1);
        acc1 = fma_add_mod(fma_mul_mod(acc1, two64_1, p1, inv_p1), v1, p1);

        let v2 = fma_add_mod(lo, fma_mul_mod(hi, two32_2, p2, inv_p2), p2);
        acc2 = fma_add_mod(fma_mul_mod(acc2, two64_2, p2, inv_p2), v2, p2);
    }

    let w1 = acc1 as u64;
    let w2 = acc2 as u64;
    if !out_words.is_null() {
        unsafe { *out_words = [w0, w1, w2] };
    }
    w0
}

// ============================================================================
// Startup Self-Test Against Reference (Section 7 & Section 11)
// ============================================================================

struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

pub fn build_fft_tables(n_complex: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert!(n_complex.is_power_of_two());
    let mut weights = vec![0.0f64; 2 * n_complex];
    let mut inv_weights = vec![0.0f64; 2 * n_complex];
    let step_w = std::f64::consts::PI / (2.0 * (n_complex as f64));
    for j in 0..n_complex {
        let ang = (j as f64) * step_w;
        let (s, c) = ang.sin_cos();
        weights[2 * j] = c;
        weights[2 * j + 1] = s;
        inv_weights[2 * j] = c;
        inv_weights[2 * j + 1] = -s;
    }

    let mut tw = vec![0.0f64; 2 * n_complex];
    let mut inv_tw = vec![0.0f64; 2 * n_complex];
    let mut b_blocks = 1usize;
    let mut s_bits = 0u32;
    while b_blocks < n_complex {
        let idx_scale = (2 * n_complex) / b_blocks;
        for j in 0..b_blocks {
            let br = if s_bits == 0 {
                0
            } else {
                j.reverse_bits() >> (usize::BITS - s_bits)
            };
            let idx = br * idx_scale;
            let (cos_v, sin_v) = if idx < n_complex {
                (weights[2 * idx], -weights[2 * idx + 1])
            } else if idx == n_complex {
                (0.0, -1.0)
            } else {
                let m = idx - n_complex;
                (-weights[2 * m + 1], -weights[2 * m])
            };
            tw[2 * (b_blocks + j)] = cos_v;
            tw[2 * (b_blocks + j) + 1] = sin_v;
            inv_tw[2 * (b_blocks + j)] = cos_v;
            inv_tw[2 * (b_blocks + j) + 1] = -sin_v;
        }
        b_blocks <<= 1;
        s_bits += 1;
    }
    (weights, inv_weights, tw, inv_tw)
}

pub fn mod_pow_u64(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut res = 1u64;
    let m128 = m as u128;
    base %= m;
    while exp > 0 {
        if (exp & 1) != 0 {
            res = (((res as u128) * (base as u128)) % m128) as u64;
        }
        base = (((base as u128) * (base as u128)) % m128) as u64;
        exp >>= 1;
    }
    res
}

pub fn primitive_root_ntt(p: u64) -> u64 {
    // Factor p - 1 = c * 2^t where c < 1024 for all primes < 2^50 with t >= 40.
    let phi = p - 1;
    let tz = phi.trailing_zeros();
    let mut c = phi >> tz;
    let mut factors = vec![2u64];
    let mut d = 3u64;
    while d * d <= c {
        if c % d == 0 {
            factors.push(d);
            while c % d == 0 {
                c /= d;
            }
        }
        d += 2;
    }
    if c > 1 {
        factors.push(c);
    }
    let mut g = 2u64;
    loop {
        if factors.iter().all(|&f| mod_pow_u64(g, phi / f, p) != 1) {
            return g;
        }
        g += 1;
    }
}

pub fn build_ntt_tables(p: u64, n: usize, negacyclic: bool) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(n.is_power_of_two());
    let g = primitive_root_ntt(p);
    let mut tw = vec![0.0f64; n.max(1)];
    let mut inv_tw = vec![0.0f64; n.max(1)];
    if n <= 1 {
        return (tw, inv_tw);
    }
    let two_n = (2 * n) as u64;
    debug_assert_eq!((p - 1) % two_n, 0);
    let omega_2n = mod_pow_u64(g, (p - 1) / two_n, p);
    let omega_2n_inv = mod_pow_u64(omega_2n, p - 2, p);

    let p_f = p as f64;
    let inv_p_f = 1.0 / p_f;
    let log_n = n.ilog2() as usize;
    let mut pow2_fwd = [0.0f64; 64];
    let mut pow2_inv = [0.0f64; 64];
    pow2_fwd[0] = omega_2n as f64;
    pow2_inv[0] = omega_2n_inv as f64;
    for t in 1..log_n {
        pow2_fwd[t] = fma_mul_mod(pow2_fwd[t - 1], pow2_fwd[t - 1], p_f, inv_p_f);
        pow2_inv[t] = fma_mul_mod(pow2_inv[t - 1], pow2_inv[t - 1], p_f, inv_p_f);
    }

    for s_bits in 0..log_n {
        let b_blocks = 1usize << s_bits;
        let step_shift = log_n - 1 - s_bits;
        tw[b_blocks] = if negacyclic { pow2_fwd[step_shift] } else { 1.0 };
        inv_tw[b_blocks] = if negacyclic { pow2_inv[step_shift] } else { 1.0 };

        for q in 0..s_bits {
            let m_sub = 1usize << q;
            let mf = pow2_fwd[log_n - 1 - q];
            let mi = pow2_inv[log_n - 1 - q];
            for k in 0..m_sub {
                let src_f = tw[b_blocks + k];
                let src_i = inv_tw[b_blocks + k];
                tw[b_blocks + m_sub + k] = fma_mul_mod(src_f, mf, p_f, inv_p_f);
                inv_tw[b_blocks + m_sub + k] = fma_mul_mod(src_i, mi, p_f, inv_p_f);
            }
        }
    }
    (tw, inv_tw)
}

fn selftest_fft_chain(cand: &KernelTable, reference: &KernelTable) -> bool {
    let mut rng = XorShift64(0x5A17_BEEF_CAFE_1234);
    for &(n_complex, coeff_bits, a_num, b_num, den) in &[
        (16usize, 16u64, 1u64, 1u64, 2u64),
        (64, 16, 1, 1, 2),
        (64, 18, 9, 5, 8),
        (64, 18, 29, 1, 16),
        (256, 18, 1, 1, 2),
        (256, 20, 9, 5, 8),
        (256, 20, 29, 1, 16),
        (2048, 14, 1, 1, 2),
    ] {
        let (w, iw, tw, itw) = build_fft_tables(n_complex);
        let a_limbs = (((n_complex as u64) * coeff_bits * a_num) / (64 * den)).max(1) as usize;
        let b_limbs = (((n_complex as u64) * coeff_bits * b_num) / (64 * den)).max(1) as usize;
        let out_limbs = a_limbs + b_limbs + 2;
        let mut a = vec![0u64; a_limbs];
        let mut b = vec![0u64; b_limbs];
        let mut a2 = vec![0u64; a_limbs];
        let mut b2 = vec![0u64; b_limbs];
        for i in 0..a_limbs {
            a[i] = rng.next() | (1u64 << 62);
            a2[i] = rng.next() >> 3;
        }
        for i in 0..b_limbs {
            b[i] = rng.next() | (1u64 << 62);
            b2[i] = rng.next() >> 3;
        }
        let scfg_a = FftSplitCfg {
            in_limbs: a_limbs as u64,
            n_complex: n_complex as u64,
            coeff_bits,
        };
        let scfg_b = FftSplitCfg {
            in_limbs: b_limbs as u64,
            n_complex: n_complex as u64,
            coeff_bits,
        };

        let run_chain = |kt: &KernelTable, mode: u8, l1_sub_n: u64| -> (Vec<u64>, u64, f64) {
            let mut fa = vec![0.0f64; 2 * n_complex];
            let mut fb = vec![0.0f64; 2 * n_complex];
            let mut fc = vec![0.0f64; 2 * n_complex];
            let mut out = vec![0u64; out_limbs];
            let mut ccfg = FftCarryCfg {
                out_limbs: out_limbs as u64,
                n_complex: n_complex as u64,
                coeff_bits,
                top_carry: 0,
                max_err: 0.0,
                check_word: 0,
            };
            unsafe {
                (kt.fft_split)(fa.as_mut_ptr(), a.as_ptr(), w.as_ptr(), &scfg_a);
                (kt.fft_split)(fb.as_mut_ptr(), b.as_ptr(), w.as_ptr(), &scfg_b);
                (kt.fft_fwd)(fa.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                (kt.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                if mode > 0 {
                    let mut fa2 = vec![0.0f64; 2 * n_complex];
                    let mut fb2 = vec![0.0f64; 2 * n_complex];
                    (kt.fft_split)(fa2.as_mut_ptr(), a2.as_ptr(), w.as_ptr(), &scfg_a);
                    (kt.fft_split)(fb2.as_mut_ptr(), b2.as_ptr(), w.as_ptr(), &scfg_b);
                    (kt.fft_fwd)(fa2.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                    (kt.fft_fwd)(fb2.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                    let ptrs = [fa.as_ptr(), fb.as_ptr(), fa2.as_ptr(), fb2.as_ptr()];
                    let sub_second = u64::from(mode == 2);
                    (kt.fft_pmuladd)(fc.as_mut_ptr(), &ptrs, n_complex as u64, sub_second);
                } else {
                    (kt.fft_pmul)(fc.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
                }
                (kt.fft_inv)(fc.as_mut_ptr(), itw.as_ptr(), n_complex as u64, l1_sub_n);
                (kt.fft_carry)(out.as_mut_ptr(), fc.as_ptr(), iw.as_ptr(), &mut ccfg);
            }
            (out, ccfg.check_word, ccfg.max_err)
        };

        for &l1_sub_n in &[u64::MAX, 64, 128, 256, 512, 1024] {
            for mode in [0u8, 1, 2] {
                let (o_c, w_c, e_c) = run_chain(cand, mode, l1_sub_n);
                let (o_r, w_r, _) = run_chain(reference, mode, l1_sub_n);
                if o_c != o_r || w_c != w_r || !(e_c < 0.25) {
                    return false;
                }
            }
        }
    }
    true
}

fn selftest_single_kernel(idx: usize, cand: &KernelTable, reference: &KernelTable) -> bool {
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15 ^ (idx as u64));
    match idx {
        0 => {
            for &n in &[0usize, 1, 3, 8, 17, 64, 67] {
                let a: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let b: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let mut dc = vec![0u64; n];
                let mut dr = vec![0u64; n];
                let cc = unsafe { (cand.add_n)(dc.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
                let cr = unsafe { (reference.add_n)(dr.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
                if dc != dr || cc != cr {
                    return false;
                }
            }
            true
        }
        1 => {
            for &n in &[0usize, 1, 3, 8, 17, 64, 67] {
                let a: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let b: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let mut dc = vec![0u64; n];
                let mut dr = vec![0u64; n];
                let cc = unsafe { (cand.sub_n)(dc.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
                let cr = unsafe { (reference.sub_n)(dr.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
                if dc != dr || cc != cr {
                    return false;
                }
            }
            true
        }
        2 => {
            for &n in &[0usize, 1, 4, 8, 19, 65] {
                let a: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let m = rng.next();
                let mut dc = vec![0u64; n];
                let mut dr = vec![0u64; n];
                let cc = unsafe { (cand.mul_1)(dc.as_mut_ptr(), a.as_ptr(), n as u64, m) };
                let cr = unsafe { (reference.mul_1)(dr.as_mut_ptr(), a.as_ptr(), n as u64, m) };
                if dc != dr || cc != cr {
                    return false;
                }
            }
            true
        }
        3 => {
            for &n in &[0usize, 1, 4, 8, 19, 65] {
                let a: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let init: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                let m = rng.next();
                let mut dc = init.clone();
                let mut dr = init;
                let cc = unsafe { (cand.addmul_1)(dc.as_mut_ptr(), a.as_ptr(), n as u64, m) };
                let cr = unsafe { (reference.addmul_1)(dr.as_mut_ptr(), a.as_ptr(), n as u64, m) };
                if dc != dr || cc != cr {
                    return false;
                }
            }
            true
        }
        4 => {
            // A kernel may have code of its own for each short length and for each place a long
            // row enters its loop: every length up to 24 limbs, then some long shapes.
            let mut shapes: Vec<(usize, usize)> = (1..=24usize).map(|na| (na, na.min(1 + na % 4))).collect();
            shapes.extend_from_slice(&[
                (8, 8),
                (17, 9),
                (33, 19),
                (45, 44),
                (64, 64),
                (65, 65),
                (70, 61),
                (96, 80),
                (130, 65),
                (140, 20),
                (160, 72),
                (140, 135),
            ]);
            for &(na, nb) in &shapes {
                let ones = na % 5 == 0;
                let a: Vec<u64> = (0..na).map(|_| if ones { u64::MAX } else { rng.next() }).collect();
                let b: Vec<u64> = (0..nb).map(|_| if ones { u64::MAX } else { rng.next() }).collect();
                let mut dc = vec![0u64; na + nb];
                let mut dr = vec![0u64; na + nb];
                for rows_max in [u64::MAX, 16, 8, 0] {
                    let lens = [na as u64, nb as u64, rows_max];
                    unsafe {
                        (cand.mul_base)(dc.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                        (reference.mul_base)(dr.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                    }
                    if dc != dr {
                        return false;
                    }
                }
            }
            true
        }
        5 => {
            for &n in &[1usize, 2, 4, 8, 12, 17, 33, 64, 68, 96, 135, 195] {
                let ones = n == 96;
                let a: Vec<u64> = (0..n).map(|_| if ones { u64::MAX } else { rng.next() }).collect();
                let mut dc = vec![0u64; 2 * n];
                let mut dr = vec![0u64; 2 * n];
                for rows_max in [u64::MAX, 16, 8, 0] {
                    unsafe {
                        (cand.sqr_base)(dc.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                        (reference.sqr_base)(dr.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                    }
                    if dc != dr {
                        return false;
                    }
                }
            }
            true
        }
        12 => {
            let p = PRIME_LOC1;
            let pr = NttPrime::new(p, 1);
            let limbs: Vec<u64> = (0..180).map(|_| rng.next()).collect();
            let count = 33usize;
            for &b in &[27u64, 50, 96, 204, 300] {
                let cfg = NttSplitCfg {
                    in_limbs: limbs.len() as u64,
                    start_coeff: 3,
                    count: count as u64,
                    coeff_bits: b,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                let mut dc = vec![0.0f64; count];
                let mut dr = vec![0.0f64; count];
                unsafe {
                    (cand.ntt_split)(dc.as_mut_ptr(), limbs.as_ptr(), &cfg);
                    (reference.ntt_split)(dr.as_mut_ptr(), limbs.as_ptr(), &cfg);
                }
                if dc != dr {
                    return false;
                }
            }
            // Strided 2D extraction + planar extract (`p == 0`) + planar Horner reduction (`in_limbs == u64::MAX`).
            let cfg_2d = NttSplitCfg {
                in_limbs: limbs.len() as u64,
                start_coeff: 2,
                count: 32,
                coeff_bits: 75,
                col_stride: 12,
                cols_per_row: 8,
                p: pr.p,
                inv_p: pr.inv_p,
            };
            let mut dc_2d = vec![0.0f64; 32];
            let mut dr_2d = vec![0.0f64; 32];
            unsafe {
                (cand.ntt_split)(dc_2d.as_mut_ptr(), limbs.as_ptr(), &cfg_2d);
                (reference.ntt_split)(dr_2d.as_mut_ptr(), limbs.as_ptr(), &cfg_2d);
            }
            if dc_2d != dr_2d {
                return false;
            }
            let b_pl = 204usize;
            let chunks = b_pl.div_ceil(48);
            let big_limbs: Vec<u64> = (0..1800).map(|_| rng.next()).collect();
            for &cnt in &[17usize, 33, 512] {
                let stride_el = ntt_planar_stride(cnt);
                let cfg_ext = NttSplitCfg {
                    in_limbs: big_limbs.len() as u64,
                    start_coeff: 1,
                    count: cnt as u64,
                    coeff_bits: b_pl as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: 0.0,
                    inv_p: 0.0,
                };
                let mut pc = vec![0.0f64; chunks * stride_el];
                let mut pr_buf = vec![0.0f64; chunks * stride_el];
                unsafe {
                    (cand.ntt_split)(pc.as_mut_ptr(), big_limbs.as_ptr(), &cfg_ext);
                    (reference.ntt_split)(pr_buf.as_mut_ptr(), big_limbs.as_ptr(), &cfg_ext);
                }
                if pc != pr_buf {
                    return false;
                }
                let cfg_red = NttSplitCfg {
                    in_limbs: u64::MAX,
                    start_coeff: 0,
                    count: cnt as u64,
                    coeff_bits: b_pl as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                let mut rc = vec![0.0f64; cnt];
                let mut rr = vec![0.0f64; cnt];
                unsafe {
                    (cand.ntt_split)(rc.as_mut_ptr(), pc.as_ptr() as *const u64, &cfg_red);
                    (reference.ntt_split)(rr.as_mut_ptr(), pr_buf.as_ptr() as *const u64, &cfg_red);
                }
                if rc != rr {
                    return false;
                }
            }
            true
        }
        13 | 14 => {
            let p = PRIME_LOC1;
            for &n in &[1usize, 2, 8, 16, 32, 64, 256, 2048, 4096, 65536] {
                let (tw, itw) = build_ntt_tables(p, n, false);
                let inv_n = mod_pow_u64(n as u64, p - 2, p);
                let init: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                for &(l1_sub, l2_sub) in &[
                    (u64::MAX, u64::MAX),
                    (64, 256),
                    (128, 1024),
                    (2048, 32768),
                ] {
                    let pr_fwd = NttPrime::with_blocks(p, 1, l1_sub, l2_sub);
                    let pr_inv = NttPrime::with_blocks(p, inv_n, l1_sub, l2_sub);
                    let mut dc = init.clone();
                    let mut dr = init.clone();
                    unsafe {
                        if idx == 13 {
                            (cand.ntt_fwd)(dc.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
                            (reference.ntt_fwd)(dr.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
                        } else {
                            (cand.ntt_inv)(dc.as_mut_ptr(), itw.as_ptr(), n as u64, &pr_inv);
                            (reference.ntt_inv)(dr.as_mut_ptr(), itw.as_ptr(), n as u64, &pr_inv);
                        }
                    }
                    if dc != dr {
                        return false;
                    }
                    if idx == 14 {
                        let b_vec: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                        let mut pr_pi = pr_inv;
                        pr_pi.pmul_b = b_vec.as_ptr();
                        let mut dc_pi = init.clone();
                        let mut dr_pi = init.clone();
                        unsafe {
                            (cand.ntt_inv)(dc_pi.as_mut_ptr(), itw.as_ptr(), n as u64, &pr_pi);
                            (reference.ntt_inv)(
                                dr_pi.as_mut_ptr(),
                                itw.as_ptr(),
                                n as u64,
                                &pr_pi,
                            );
                        }
                        if dc_pi != dr_pi {
                            return false;
                        }

                        let mut pr_fpi = pr_inv;
                        pr_fpi.fwd_tw = tw.as_ptr();
                        pr_fpi.pmul_b = b_vec.as_ptr();
                        let mut dc_fpi = init.clone();
                        let mut dr_fpi = init.clone();
                        unsafe {
                            (cand.ntt_inv)(dc_fpi.as_mut_ptr(), itw.as_ptr(), n as u64, &pr_fpi);
                            (reference.ntt_inv)(
                                dr_fpi.as_mut_ptr(),
                                itw.as_ptr(),
                                n as u64,
                                &pr_fpi,
                            );
                        }
                        if dc_fpi != dr_fpi {
                            return false;
                        }

                        let mut dc_sq = init.clone();
                        let mut dr_sq = init.clone();
                        let mut pr_sq_c = pr_inv;
                        pr_sq_c.fwd_tw = tw.as_ptr();
                        pr_sq_c.pmul_b = dc_sq.as_ptr();
                        let mut pr_sq_r = pr_inv;
                        pr_sq_r.fwd_tw = tw.as_ptr();
                        pr_sq_r.pmul_b = dr_sq.as_ptr();
                        unsafe {
                            (cand.ntt_inv)(dc_sq.as_mut_ptr(), itw.as_ptr(), n as u64, &pr_sq_c);
                            (reference.ntt_inv)(
                                dr_sq.as_mut_ptr(),
                                itw.as_ptr(),
                                n as u64,
                                &pr_sq_r,
                            );
                        }
                        if dc_sq != dr_sq {
                            return false;
                        }
                    }
                }
            }
            true
        }
        15 | 16 => {
            let p = PRIME_LOC1;
            let mut scratch_vec = vec![0.0f64; 512 * 8 + 16];
            let sb_addr = scratch_vec.as_mut_ptr() as usize;
            let sb_aligned = ((sb_addr + 63) & !63usize) as *mut f64;
            for &(rows, cols) in &[
                (4usize, 8usize),
                (16, 8),
                (32, 16),
                (64, 8),
                (128, 8),
                (16, 11),
                (512, 8),
            ] {
                let (tw, itw) = build_ntt_tables(p, rows, false);
                let inv_r = mod_pow_u64(rows as u64, p - 2, p);
                let pr_fwd = NttPrime::new(p, 1);
                let pr_inv = NttPrime::new(p, inv_r);
                let init: Vec<f64> = (0..(rows * cols)).map(|_| (rng.next() % p) as f64).collect();
                for &(l1_r, l2_r, use_scr) in &[
                    (u64::MAX, u64::MAX, true),
                    (16, 64, true),
                    (64, 256, false),
                    (256, 4096, true),
                ] {
                    let mut dc = init.clone();
                    let mut dr = init.clone();
                    let cfg = NttColsCfg::with_blocks(
                        rows,
                        cols,
                        l1_r,
                        l2_r,
                        if use_scr { sb_aligned } else { std::ptr::null_mut() },
                    );
                    unsafe {
                        if idx == 15 {
                            (cand.ntt_fwd_cols)(dc.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
                            (reference.ntt_fwd_cols)(dr.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
                        } else {
                            (cand.ntt_inv_cols)(dc.as_mut_ptr(), itw.as_ptr(), &cfg, &pr_inv);
                            (reference.ntt_inv_cols)(dr.as_mut_ptr(), itw.as_ptr(), &cfg, &pr_inv);
                        }
                    }
                    if dc != dr {
                        return false;
                    }
                }
            }
            true
        }
        17 => {
            let p = PRIME_LOC1;
            let pr = NttPrime::new(p, 1);
            for &n in &[0usize, 1, 8, 19, 64] {
                let a: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                let b: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                let mut dc = vec![0.0f64; n];
                let mut dr = vec![0.0f64; n];
                let cfg = NttVecCfg {
                    n: n as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                    sub_second: 0,
                };
                unsafe {
                    (cand.ntt_pmul)(dc.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &cfg);
                    (reference.ntt_pmul)(dr.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &cfg);
                }
                if dc != dr {
                    return false;
                }
            }
            // Also verify fused row-scale mode (`sub_second == cols > 0`).
            let rows = 4usize;
            let cols = 16usize;
            let n = rows * cols;
            let a: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
            let row_scales: Vec<f64> = (0..rows).map(|_| (rng.next() % p) as f64).collect();
            let init: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
            let mut dc = init.clone();
            let mut dr = init;
            let cfg_rs = NttVecCfg {
                n: n as u64,
                p: pr.p,
                inv_p: pr.inv_p,
                sub_second: cols as u64,
            };
            unsafe {
                (cand.ntt_pmul)(dc.as_mut_ptr(), a.as_ptr(), row_scales.as_ptr(), &cfg_rs);
                (reference.ntt_pmul)(dr.as_mut_ptr(), a.as_ptr(), row_scales.as_ptr(), &cfg_rs);
            }
            dc == dr
        }
        18 => {
            let p = PRIME_LOC1;
            let pr = NttPrime::new(p, 1);
            for &sub in &[0u64, 1] {
                for &null_b1 in &[false, true] {
                    let n = 37usize;
                    let a1: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                    let b1: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                    let a2: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                    let b2: Vec<f64> = (0..n).map(|_| (rng.next() % p) as f64).collect();
                    let ptrs = [
                        a1.as_ptr(),
                        if null_b1 { std::ptr::null() } else { b1.as_ptr() },
                        a2.as_ptr(),
                        b2.as_ptr(),
                    ];
                    let cfg = NttVecCfg {
                        n: n as u64,
                        p: pr.p,
                        inv_p: pr.inv_p,
                        sub_second: sub,
                    };
                    let mut dc = vec![0.0f64; n];
                    let mut dr = vec![0.0f64; n];
                    unsafe {
                        (cand.ntt_pmuladd)(dc.as_mut_ptr(), &ptrs, &cfg);
                        (reference.ntt_pmuladd)(dr.as_mut_ptr(), &ptrs, &cfg);
                    }
                    if dc != dr {
                        return false;
                    }
                }
            }
            true
        }
        19 => {
            let all_ps = crate::ntt::largest_primes_for_two_adic(40, 12);
            let test_crt = |rng: &mut XorShift64,
                            k: usize,
                            b: u64,
                            count: usize,
                            signed: u64,
                            wrap_cyclic: u64,
                            segments: usize|
             -> bool {
                let ps = &all_ps[..k];
                let k_words = k.max(4);
                let primes_f: Vec<f64> = ps.iter().map(|&x| x as f64).collect();
                let inv_f: Vec<f64> = ps.iter().map(|&x| 1.0 / (x as f64)).collect();
                let mut garner = vec![0.0f64; k];
                let mut prefix = vec![0u64; (k + 1) * k_words];
                prefix[0] = 1;
                for i in 0..k {
                    if i > 0 {
                        let mut m_mod = 1u64;
                        for &pj in &ps[..i] {
                            m_mod =
                                (((m_mod as u128) * ((pj % ps[i]) as u128)) % (ps[i] as u128))
                                    as u64;
                        }
                        garner[i] = mod_pow_u64(m_mod, ps[i] - 2, ps[i]) as f64;
                    }
                    let prev = prefix[(i * k_words)..((i + 1) * k_words)].to_vec();
                    let dst_s = &mut prefix[((i + 1) * k_words)..((i + 2) * k_words)];
                    let mut c = 0u128;
                    for w in 0..k_words {
                        let prd = (prev[w] as u128) * (ps[i] as u128) + c;
                        dst_s[w] = prd as u64;
                        c = prd >> 64;
                    }
                }
                let total_res = segments * count;
                let streams: Vec<Vec<f64>> = ps
                    .iter()
                    .map(|&pi| (0..total_res).map(|_| (rng.next() % pi) as f64).collect())
                    .collect();
                let r_ptrs: Vec<*const f64> = streams.iter().map(|v| v.as_ptr()).collect();
                let out_per_seg = (((count as u64) * b).div_ceil(64) as usize) + k_words + 2;
                let mut dc = vec![0u64; segments * out_per_seg];
                let mut dr = vec![0u64; segments * out_per_seg];
                if segments <= 1 {
                    let mut cfg_c = NttCrtCfg {
                        k_primes: k as u64,
                        count: count as u64,
                        coeff_bits: b,
                        out_limbs: out_per_seg as u64,
                        signed,
                        wrap_cyclic,
                        primes: primes_f.as_ptr(),
                        inv_primes: inv_f.as_ptr(),
                        garner_inv: garner.as_ptr(),
                        prefix_prods: prefix.as_ptr(),
                        k_words: k_words as u64,
                        carry_limbs: std::ptr::null_mut(),
                        segs: NttCrtSegs::ONE,
                    };
                    let mut cfg_r = cfg_c;
                    let wc = unsafe { (cand.ntt_crt)(dc.as_mut_ptr(), r_ptrs.as_ptr(), &mut cfg_c) };
                    let wr = unsafe {
                        (reference.ntt_crt)(dr.as_mut_ptr(), r_ptrs.as_ptr(), &mut cfg_r)
                    };
                    dc == dr && wc == wr
                } else {
                    let mut carry_c = vec![0u64; segments * k_words];
                    let mut carry_r = vec![0u64; segments * k_words];
                    let mut w0_c = vec![0u64; segments];
                    let mut w0_r = vec![0u64; segments];
                    let mut cfg_c = NttCrtCfg {
                        k_primes: k as u64,
                        count: count as u64,
                        coeff_bits: b,
                        out_limbs: out_per_seg as u64,
                        signed,
                        wrap_cyclic: 0,
                        primes: primes_f.as_ptr(),
                        inv_primes: inv_f.as_ptr(),
                        garner_inv: garner.as_ptr(),
                        prefix_prods: prefix.as_ptr(),
                        k_words: k_words as u64,
                        carry_limbs: carry_c.as_mut_ptr(),
                        segs: NttCrtSegs {
                            segments: segments as u64,
                            res_stride: count as u64,
                            out_stride: out_per_seg as u64,
                            carry_stride: k_words as u64,
                            w0: w0_c.as_mut_ptr(),
                            w0_stride: 1,
                        },
                    };
                    let mut cfg_r = NttCrtCfg {
                        carry_limbs: carry_r.as_mut_ptr(),
                        segs: NttCrtSegs {
                            w0: w0_r.as_mut_ptr(),
                            ..cfg_c.segs
                        },
                        ..cfg_c
                    };
                    unsafe {
                        (cand.ntt_crt)(dc.as_mut_ptr(), r_ptrs.as_ptr(), &mut cfg_c);
                        (reference.ntt_crt)(dr.as_mut_ptr(), r_ptrs.as_ptr(), &mut cfg_r);
                    }
                    dc == dr && carry_c == carry_r && w0_c == w0_r
                }
            };
            // Exercise 3..5-prime fast body (+ block path count >= 64, signed, wrap_cyclic),
            // 6..9-prime body, 10..18-prime general wide body, and multi-segment NttCrtSegs!
            test_crt(&mut rng, 3, 58, 33, 0, 0, 1)
                && test_crt(&mut rng, 4, 80, 65, 0, 1, 1)
                && test_crt(&mut rng, 5, 92, 65, 1, 0, 1)
                && test_crt(&mut rng, 8, 196, 65, 0, 0, 1)
                && test_crt(&mut rng, 12, 260, 33, 0, 0, 1)
                && test_crt(&mut rng, 4, 64, 32, 0, 0, 2)
        }
        20 | 21 => {
            // Includes `cols = 256` (32 blocks of 8) to exercise register-direct and NT block bodies.
            for &cols in &[16usize, 256] {
                for &packed in &[0u64, 1] {
                    for &use_nt in &[0u64, 1] {
                        let rows = 4usize;
                        let stride_bytes = if packed == 0 {
                            (cols + 16) * 8
                        } else {
                            (cols / 4) * 50
                        };
                        let tile_in: Vec<f64> = (0..(rows * cols))
                            .map(|_| (rng.next() & ((1u64 << 50) - 1)) as f64)
                            .collect();
                        if idx == 21 {
                            let mut oc = vec![0u8; rows * stride_bytes];
                            let mut or = vec![0u8; rows * stride_bytes];
                            let cfg = MoveCfg {
                                rows: rows as u64,
                                cols: cols as u64,
                                outer_stride_bytes: stride_bytes as u64,
                                packed,
                                use_nt,
                            };
                            unsafe {
                                (cand.scatter)(oc.as_mut_ptr(), tile_in.as_ptr(), &cfg);
                                (reference.scatter)(or.as_mut_ptr(), tile_in.as_ptr(), &cfg);
                            }
                            if oc != or {
                                return false;
                            }
                            if packed == 0 {
                                let src_stride_words = cols + 2;
                                let src_strided: Vec<f64> = (0..(rows * src_stride_words))
                                    .map(|_| (rng.next() & ((1u64 << 50) - 1)) as f64)
                                    .collect();
                                let mut oc_s = vec![0u8; rows * stride_bytes];
                                let mut or_s = vec![0u8; rows * stride_bytes];
                                let cfg_s = MoveCfg {
                                    rows: rows as u64,
                                    cols: cols as u64,
                                    outer_stride_bytes: stride_bytes as u64,
                                    packed: ((src_stride_words * 8) as u64) << 8,
                                    use_nt,
                                };
                                unsafe {
                                    (cand.scatter)(oc_s.as_mut_ptr(), src_strided.as_ptr(), &cfg_s);
                                    (reference.scatter)(
                                        or_s.as_mut_ptr(),
                                        src_strided.as_ptr(),
                                        &cfg_s,
                                    );
                                }
                                if oc_s != or_s {
                                    return false;
                                }
                            }
                        } else {
                            let mut outer = vec![0u8; rows * stride_bytes];
                            let cfg = MoveCfg {
                                rows: rows as u64,
                                cols: cols as u64,
                                outer_stride_bytes: stride_bytes as u64,
                                packed,
                                use_nt,
                            };
                            unsafe {
                                (reference.scatter)(outer.as_mut_ptr(), tile_in.as_ptr(), &cfg);
                            }
                            let mut tc = vec![0.0f64; rows * cols];
                            let mut tr = vec![0.0f64; rows * cols];
                            unsafe {
                                (cand.gather)(tc.as_mut_ptr(), outer.as_ptr(), &cfg);
                                (reference.gather)(tr.as_mut_ptr(), outer.as_ptr(), &cfg);
                            }
                            if tc != tr {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        }
        22 => {
            for &fmt in &[0u64, 1] {
                for &n in &[3usize, 19, 24] {
                    let words: Vec<u64> = (0..n)
                        .map(|_| rng.next() % 10_000_000_000_000_000_000u64)
                        .collect();
                    let b_len = if fmt != 0 { n * 19 } else { n * 8 };
                    let mut bc = vec![0u8; b_len];
                    let mut br = vec![0u8; b_len];
                    let wc =
                        unsafe { (cand.to_digits)(bc.as_mut_ptr(), words.as_ptr(), n as u64, fmt) };
                    let wr = unsafe {
                        (reference.to_digits)(br.as_mut_ptr(), words.as_ptr(), n as u64, fmt)
                    };
                    if bc != br || wc != wr {
                        return false;
                    }
                }
            }
            true
        }
        23 => {
            for &n in &[0usize, 1, 7, 8, 16, 61, 64, 130] {
                let limbs: Vec<u64> = (0..n).map(|_| rng.next()).collect();
                for &all_three in &[0u64, 1] {
                    let mut oc = [0u64; 3];
                    let mut or = [0u64; 3];
                    let wc = unsafe { (cand.check_n)(limbs.as_ptr(), n as u64, &mut oc, all_three) };
                    let wr = unsafe { (reference.check_n)(limbs.as_ptr(), n as u64, &mut or, all_three) };
                    if oc != or || wc != wr {
                        return false;
                    }
                }
            }
            true
        }
        _ => true,
    }
}

// ============================================================================
// Kernel timing at the seam (trace runs only)
// ============================================================================
//
// With timing enabled, every entry of the active table is a shim that times the bound kernel
// and adds to the calling thread's own counters. It is what shows, for a traced run, how its
// time divides between the kernels and the portable code around them (gate 4, Section 15).

const KT_SLOTS: usize = 256;

#[repr(align(128))]
struct KtSlot {
    ns: [AtomicU64; NUM_KERNELS],
    calls: [AtomicU64; NUM_KERNELS],
}

#[allow(clippy::declare_interior_mutable_const)]
const KT_ZERO: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const KT_SLOT_INIT: KtSlot = KtSlot {
    ns: [KT_ZERO; NUM_KERNELS],
    calls: [KT_ZERO; NUM_KERNELS],
};
static KT_TABLE: [KtSlot; KT_SLOTS] = [KT_SLOT_INIT; KT_SLOTS];
static KT_REAL: std::sync::OnceLock<KernelTable> = std::sync::OnceLock::new();
static KT_NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static KT_ID: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

#[inline]
fn kt_real() -> &'static KernelTable {
    KT_REAL.get().expect("kernel timing enabled")
}

#[inline]
fn kt_add(kernel: usize, t0: std::time::Instant) {
    let ns = t0.elapsed().as_nanos() as u64;
    let id = KT_ID.with(|c| {
        let mut v = c.get();
        if v == usize::MAX {
            v = KT_NEXT.fetch_add(1, Ordering::Relaxed) % KT_SLOTS;
            c.set(v);
        }
        v
    });
    KT_TABLE[id].ns[kernel].fetch_add(ns, Ordering::Relaxed);
    KT_TABLE[id].calls[kernel].fetch_add(1, Ordering::Relaxed);
}

macro_rules! kt_shim {
    ($name:ident, $field:ident, $idx:expr, ($($arg:ident : $ty:ty),*) $(-> $ret:ty)?) => {
        unsafe extern "C" fn $name($($arg: $ty),*) $(-> $ret)? {
            let t0 = std::time::Instant::now();
            let r = unsafe { (kt_real().$field)($($arg),*) };
            kt_add($idx, t0);
            r
        }
    };
}

kt_shim!(kt_add_n, add_n, 0, (a: *mut u64, b: *const u64, c: *const u64, n: u64) -> u64);
kt_shim!(kt_sub_n, sub_n, 1, (a: *mut u64, b: *const u64, c: *const u64, n: u64) -> u64);
kt_shim!(kt_mul_1, mul_1, 2, (a: *mut u64, b: *const u64, n: u64, m: u64) -> u64);
kt_shim!(kt_addmul_1, addmul_1, 3, (a: *mut u64, b: *const u64, n: u64, m: u64) -> u64);
kt_shim!(kt_mul_base, mul_base, 4, (a: *mut u64, b: *const u64, c: *const u64, l: *const [u64; 3]));
kt_shim!(kt_sqr_base, sqr_base, 5, (a: *mut u64, b: *const u64, n: u64, rm: u64));
kt_shim!(kt_fft_split, fft_split, 6, (a: *mut f64, b: *const u64, w: *const f64, c: *const FftSplitCfg));
kt_shim!(kt_fft_fwd, fft_fwd, 7, (a: *mut f64, t: *const f64, n: u64, s: u64));
kt_shim!(kt_fft_inv, fft_inv, 8, (a: *mut f64, t: *const f64, n: u64, s: u64));
kt_shim!(kt_fft_pmul, fft_pmul, 9, (o: *mut f64, a: *const f64, b: *const f64, n: u64));
kt_shim!(kt_fft_pmuladd, fft_pmuladd, 10, (o: *mut f64, p: *const [*const f64; 4], n: u64, s: u64));
kt_shim!(kt_fft_carry, fft_carry, 11, (o: *mut u64, a: *const f64, w: *const f64, c: *mut FftCarryCfg));
kt_shim!(kt_ntt_split, ntt_split, 12, (o: *mut f64, a: *const u64, c: *const NttSplitCfg));
kt_shim!(kt_ntt_fwd, ntt_fwd, 13, (a: *mut f64, t: *const f64, n: u64, p: *const NttPrime));
kt_shim!(kt_ntt_inv, ntt_inv, 14, (a: *mut f64, t: *const f64, n: u64, p: *const NttPrime));
kt_shim!(kt_ntt_fwd_cols, ntt_fwd_cols, 15, (a: *mut f64, t: *const f64, c: *const NttColsCfg, p: *const NttPrime));
kt_shim!(kt_ntt_inv_cols, ntt_inv_cols, 16, (a: *mut f64, t: *const f64, c: *const NttColsCfg, p: *const NttPrime));
kt_shim!(kt_ntt_pmul, ntt_pmul, 17, (o: *mut f64, a: *const f64, b: *const f64, c: *const NttVecCfg));
kt_shim!(kt_ntt_pmuladd, ntt_pmuladd, 18, (o: *mut f64, p: *const [*const f64; 4], c: *const NttVecCfg));
kt_shim!(kt_ntt_crt, ntt_crt, 19, (o: *mut u64, r: *const *const f64, c: *mut NttCrtCfg) -> u64);
kt_shim!(kt_gather, gather, 20, (d: *mut f64, s: *const u8, c: *const MoveCfg));
kt_shim!(kt_scatter, scatter, 21, (d: *mut u8, s: *const f64, c: *const MoveCfg));
kt_shim!(kt_to_digits, to_digits, 22, (o: *mut u8, w: *const u64, n: u64, f: u64) -> u64);
kt_shim!(kt_check_n, check_n, 23, (l: *const u64, n: u64, o: *mut [u64; 3], a: u64) -> u64);

impl Kernels {
    /// Route every active kernel through a timing shim. `limbs_too` also times the six limb
    /// kernels, whose calls are short enough for the clock to weigh on them.
    pub fn enable_timing(&mut self, limbs_too: bool) {
        let real = *KT_REAL.get_or_init(|| self.active);
        let mut t = KernelTable {
            add_n: kt_add_n,
            sub_n: kt_sub_n,
            mul_1: kt_mul_1,
            addmul_1: kt_addmul_1,
            mul_base: kt_mul_base,
            sqr_base: kt_sqr_base,
            fft_split: kt_fft_split,
            fft_fwd: kt_fft_fwd,
            fft_inv: kt_fft_inv,
            fft_pmul: kt_fft_pmul,
            fft_pmuladd: kt_fft_pmuladd,
            fft_carry: kt_fft_carry,
            ntt_split: kt_ntt_split,
            ntt_fwd: kt_ntt_fwd,
            ntt_inv: kt_ntt_inv,
            ntt_fwd_cols: kt_ntt_fwd_cols,
            ntt_inv_cols: kt_ntt_inv_cols,
            ntt_pmul: kt_ntt_pmul,
            ntt_pmuladd: kt_ntt_pmuladd,
            ntt_crt: kt_ntt_crt,
            gather: kt_gather,
            scatter: kt_scatter,
            to_digits: kt_to_digits,
            check_n: kt_check_n,
        };
        if !limbs_too {
            t.add_n = real.add_n;
            t.sub_n = real.sub_n;
            t.mul_1 = real.mul_1;
            t.addmul_1 = real.addmul_1;
            t.mul_base = real.mul_base;
            t.sqr_base = real.sqr_base;
        }
        self.active = t;
    }
}

/// Print and clear the kernel times gathered since the last call; silent when timing is off.
pub fn dump_and_reset_kernel_times(tag: &str, wall_sec: f64, workers: usize) {
    if KT_REAL.get().is_none() {
        return;
    }
    let mut total_ns = 0u64;
    let mut line = String::new();
    for k in 0..NUM_KERNELS {
        let mut ns = 0u64;
        let mut calls = 0u64;
        for slot in &KT_TABLE {
            ns += slot.ns[k].swap(0, Ordering::Relaxed);
            calls += slot.calls[k].swap(0, Ordering::Relaxed);
        }
        if calls > 0 {
            total_ns += ns;
            line.push_str(&format!(" {}={:.0}ms/{}", KERNEL_NAMES[k], ns as f64 * 1e-6, calls));
        }
    }
    let thread_sec = wall_sec * (workers as f64);
    eprintln!(
        "[TRACE KERNELS {tag}] wall={:.0}ms x{} workers = {:.0}ms; in kernels {:.0}ms ({:.0}%) |{}",
        wall_sec * 1e3,
        workers,
        thread_sec * 1e3,
        total_ns as f64 * 1e-6,
        if thread_sec > 0.0 { 100.0 * (total_ns as f64) * 1e-9 / thread_sec } else { 0.0 },
        line
    );
}
