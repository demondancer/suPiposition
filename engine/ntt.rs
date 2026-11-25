//! Contract:
//! - Inputs: Unsigned limb slices, `NttSpec` (transform length `n = rows * cols`, `k_primes`
//!   primes `< 2^50`, `coeff_bits` with `coeff_bits % 61 != 0`, `packed` 50-bit rest format,
//!   `negacyclic` root flag, and optional out-of-core `Store`), `Scheduler`, and `Kernels`.
//! - Outputs: Exact whole or wrapped (`mod 2^{n*coeff_bits} +- 1`) products, squares, and fused
//!   two-product linear combinations (`Q2*T1 +- P1*T2`), returning `(limbs, check_word_w0)`, plus
//!   reusable `NttCachedFwd`.
//! - Data format: Residues in `[0, p)` as `f64` inside caches, stored as either 64-bit `f64` or
//!   50-bit packed blocks (50 bytes per 8 residues) across RAM or out-of-core `Store` extents.
//! - Work & bytes moved: 3-sweep cache-tiled grid (`ntt_split`+`ntt_fwd_cols`+`scatter`, row
//!   `gather`+`ntt_fwd`+`ntt_pmul`+`ntt_inv`+`scatter`, `gather`+`ntt_inv_cols`+`ntt_crt` + carry
//!   stitch), moving each residue array to/from memory once per sweep (`~40n` bits unpacked,
//!   `~32n` bits packed), recursively adding sweeps when cache or store tiles require multi-level
//!   grids; out-of-core transforms sweep one RAM-bounded tile at a time in strided range batches
//!   with `Loader` read/write overlap, inline row-sweep piece checks, and column-sweep `check_n`
//!   passes counted as traffic.

use std::alloc::Allocator;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::arena::{ScratchAlloc, TransformsAlloc};
use crate::check::Telemetry;
use crate::limb::{LimbVec, trim_limbs};
use crate::par::Scheduler;
use crate::refk::{
    Kernels, MoveCfg, NttCrtCfg, NttCrtSegs, NttPrime, NttSplitCfg, NttVecCfg, build_ntt_tables,
    mod_pow_u64, primitive_root_ntt,
};
use crate::store::{AlignedSlotBuf, ExtentHandle, ExtentRange, SlotBatchPlan, Store, StoreError};

pub type TransU64Vec = Vec<u64, TransformsAlloc>;

struct NttTlsScratch {
    tile: Vec<f64, ScratchAlloc>,
    col_scratch: Vec<f64, ScratchAlloc>,
    tw_row: Vec<f64, ScratchAlloc>,
    tw_step: Vec<f64, ScratchAlloc>,
    s_u64: Vec<u64, ScratchAlloc>,
    ra1: Vec<f64, ScratchAlloc>,
    rb1: Vec<f64, ScratchAlloc>,
    ra2: Vec<f64, ScratchAlloc>,
    rb2: Vec<f64, ScratchAlloc>,
    rc: Vec<f64, ScratchAlloc>,
    prime_segs: Vec<Vec<f64, ScratchAlloc>, ScratchAlloc>,
    row_ptrs: Vec<usize, ScratchAlloc>,
}

impl NttTlsScratch {
    fn new() -> Self {
        Self {
            tile: Vec::new_in(ScratchAlloc),
            col_scratch: Vec::new_in(ScratchAlloc),
            tw_row: Vec::new_in(ScratchAlloc),
            tw_step: Vec::new_in(ScratchAlloc),
            s_u64: Vec::new_in(ScratchAlloc),
            ra1: Vec::new_in(ScratchAlloc),
            rb1: Vec::new_in(ScratchAlloc),
            ra2: Vec::new_in(ScratchAlloc),
            rb2: Vec::new_in(ScratchAlloc),
            rc: Vec::new_in(ScratchAlloc),
            prime_segs: Vec::new_in(ScratchAlloc),
            row_ptrs: Vec::new_in(ScratchAlloc),
        }
    }
}

#[inline(always)]
unsafe fn call_ntt_fwd(kernels: &Kernels, buf: *mut f64, tw: *const f64, n: u64, pr: &NttPrime) {
    let l1 = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let l2 = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    let f = kernels.active.ntt_fwd;
    if pr.l1_sub == l1 && pr.l2_sub == l2 {
        unsafe {
            f(buf, tw, n, pr);
        }
    } else {
        let mut pr_bound = *pr;
        pr_bound.l1_sub = l1;
        pr_bound.l2_sub = l2;
        unsafe {
            f(buf, tw, n, &pr_bound);
        }
    }
}

#[inline(always)]
unsafe fn call_ntt_inv(
    kernels: &Kernels,
    buf: *mut f64,
    inv_tw: *const f64,
    n: u64,
    pr: &NttPrime,
) {
    let l1 = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let l2 = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    let f = kernels.active.ntt_inv;
    if pr.l1_sub == l1 && pr.l2_sub == l2 {
        unsafe {
            f(buf, inv_tw, n, pr);
        }
    } else {
        let mut pr_bound = *pr;
        pr_bound.l1_sub = l1;
        pr_bound.l2_sub = l2;
        unsafe {
            f(buf, inv_tw, n, &pr_bound);
        }
    }
}

#[inline(always)]
unsafe fn call_ntt_pmul_inv(
    kernels: &Kernels,
    buf: *mut f64,
    inv_tw: *const f64,
    pmul_b: *const f64,
    n: u64,
    pr_inv: &NttPrime,
) {
    let l1 = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let l2 = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    if l2 > 0 && n > l2.saturating_mul(4) {
        let mut pr_clean = *pr_inv;
        pr_clean.fwd_tw = std::ptr::null();
        pr_clean.pmul_b = std::ptr::null();
        let vcfg = NttVecCfg {
            n,
            p: pr_inv.p,
            inv_p: pr_inv.inv_p,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(buf, buf, pmul_b, &vcfg);
            call_ntt_inv(kernels, buf, inv_tw, n, &pr_clean);
        }
        return;
    }
    let mut pr_bound = *pr_inv;
    pr_bound.l1_sub = l1;
    pr_bound.l2_sub = l2;
    pr_bound.fwd_tw = std::ptr::null();
    pr_bound.pmul_b = pmul_b;
    unsafe {
        (kernels.active.ntt_inv)(buf, inv_tw, n, &pr_bound);
    }
}

#[inline(always)]
unsafe fn call_ntt_fwd_pmul_inv(
    kernels: &Kernels,
    buf: *mut f64,
    fwd_tw: *const f64,
    inv_tw: *const f64,
    pmul_b: *const f64,
    n: u64,
    pr_inv: &NttPrime,
) {
    let l1 = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let l2 = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    if l2 > 0 && n > l2 {
        let mut pr_clean = *pr_inv;
        pr_clean.fwd_tw = std::ptr::null();
        pr_clean.pmul_b = std::ptr::null();
        unsafe {
            call_ntt_fwd(kernels, buf, fwd_tw, n, &pr_clean);
            call_ntt_pmul_inv(kernels, buf, inv_tw, pmul_b, n, pr_inv);
        }
        return;
    }
    let mut pr_bound = *pr_inv;
    pr_bound.l1_sub = l1;
    pr_bound.l2_sub = l2;
    pr_bound.fwd_tw = fwd_tw;
    pr_bound.pmul_b = pmul_b;
    unsafe {
        (kernels.active.ntt_inv)(buf, inv_tw, n, &pr_bound);
    }
}

#[inline(always)]
unsafe fn call_ntt_fwd_cols(
    kernels: &Kernels,
    buf: *mut f64,
    tw: *const f64,
    shape: &[u64; 2],
    pr: &NttPrime,
    col_scratch: &mut Vec<f64, ScratchAlloc>,
) {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let need = kernels.ntt_cols_scratch_f64s(rows);
    if col_scratch.len() < need {
        col_scratch.resize(need, 0.0);
    }
    let cfg = kernels.ntt_cols_cfg(rows, cols, col_scratch);
    unsafe {
        (kernels.active.ntt_fwd_cols)(buf, tw, &cfg, pr);
    }
}

#[inline(always)]
unsafe fn call_ntt_inv_cols(
    kernels: &Kernels,
    buf: *mut f64,
    inv_tw: *const f64,
    shape: &[u64; 2],
    pr: &NttPrime,
    col_scratch: &mut Vec<f64, ScratchAlloc>,
) {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let need = kernels.ntt_cols_scratch_f64s(rows);
    if col_scratch.len() < need {
        col_scratch.resize(need, 0.0);
    }
    let cfg = kernels.ntt_cols_cfg(rows, cols, col_scratch);
    unsafe {
        (kernels.active.ntt_inv_cols)(buf, inv_tw, &cfg, pr);
    }
}

struct NttTls1dFused {
    m_ch: Vec<f64, ScratchAlloc>,
    m_t_res: Vec<f64, ScratchAlloc>,
    m_q_res: Vec<f64, ScratchAlloc>,
    m_p_res: Vec<f64, ScratchAlloc>,
}

impl NttTls1dFused {
    fn new() -> Self {
        Self {
            m_ch: Vec::new_in(ScratchAlloc),
            m_t_res: Vec::new_in(ScratchAlloc),
            m_q_res: Vec::new_in(ScratchAlloc),
            m_p_res: Vec::new_in(ScratchAlloc),
        }
    }
}

thread_local! {
    static NTT_TLS_SCRATCH: RefCell<NttTlsScratch> = RefCell::new(NttTlsScratch::new());
    static NTT_TLS_1D_FUSED: RefCell<NttTls1dFused> = RefCell::new(NttTls1dFused::new());
}

struct Fused1dLease(Option<NttTls1dFused>);

impl Fused1dLease {
    #[inline]
    fn acquire() -> Self {
        let s = NTT_TLS_1D_FUSED
            .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), NttTls1dFused::new()));
        Self(Some(s))
    }

    #[inline]
    fn get_mut(&mut self) -> &mut NttTls1dFused {
        self.0.as_mut().unwrap()
    }
}

impl Drop for Fused1dLease {
    #[inline]
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            NTT_TLS_1D_FUSED.with(|cell| {
                *cell.borrow_mut() = s;
            });
        }
    }
}

struct ScratchLease(Option<NttTlsScratch>);

impl ScratchLease {
    #[inline]
    fn acquire() -> Self {
        let s = NTT_TLS_SCRATCH
            .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), NttTlsScratch::new()));
        Self(Some(s))
    }

    #[inline]
    fn get_mut(&mut self) -> &mut NttTlsScratch {
        self.0.as_mut().unwrap()
    }
}

impl Drop for ScratchLease {
    #[inline]
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            NTT_TLS_SCRATCH.with(|cell| {
                *cell.borrow_mut() = s;
            });
        }
    }
}

static NTT_TRACE_TW_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_SW1_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_SW2_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_SW3_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_1D_FWD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_1D_INV_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_1D_CRT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_ALLOC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_TRACE_W0_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// Trace runs: the workers' idle time counted while each phase ran (see `par::idle_since_mark`).
static NTT_IDLE_ALLOC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_IDLE_SW1_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_IDLE_SW2_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_IDLE_SW3_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_IDLE_W0_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_SW1_SPLIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_SW1_COLS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_SW1_SCAT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_SW3_GATH_COLS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NTT_SW3_CRT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn ntt_trace_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("SUPI_TRACE").is_some())
}

/// Trace runs only: how often each product routine ran in each transform shape, and for how long.
static NTT_TRACE_PATHS: std::sync::Mutex<Vec<(String, u64, u64)>> = std::sync::Mutex::new(Vec::new());

/// Times one call of a product routine for the trace; inert outside a trace run.
struct PathTimer(Option<(String, std::time::Instant)>);

impl PathTimer {
    #[inline]
    fn start(name: &str, spec: &NttSpec, limb_lens: &[usize]) -> Self {
        if !ntt_trace_enabled() {
            return PathTimer(None);
        }
        let mbits: Vec<usize> = limb_lens.iter().map(|&l| (l * 64 + 500_000) / 1_000_000).collect();
        let key = format!(
            "{name} n=2^{} R={} C={} tc={} k={} cb={} pk={} pw={} Mbit={mbits:?}",
            spec.n.max(1).ilog2(),
            spec.rows,
            spec.cols,
            spec.tile_cols,
            spec.k_primes,
            spec.coeff_bits,
            u8::from(spec.packed),
            spec.par_workers,
        );
        PathTimer(Some((key, std::time::Instant::now())))
    }

    #[inline]
    fn tag(&mut self, extra: &str) {
        if let Some((key, _)) = &mut self.0 {
            key.push(' ');
            key.push_str(extra);
        }
    }
}

impl Drop for PathTimer {
    fn drop(&mut self) {
        if let Some((key, t0)) = self.0.take() {
            let ns = t0.elapsed().as_nanos() as u64;
            let mut g = NTT_TRACE_PATHS.lock().unwrap();
            if let Some(e) = g.iter_mut().find(|e| e.0 == key) {
                e.1 += 1;
                e.2 += ns;
            } else {
                g.push((key, 1, ns));
            }
        }
    }
}

pub fn dump_and_reset_ntt_trace(tag: &str) {
    let tw = NTT_TRACE_TW_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let sw1 = NTT_TRACE_SW1_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let sw2 = NTT_TRACE_SW2_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let sw3 = NTT_TRACE_SW3_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let f1 = NTT_TRACE_1D_FWD_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let i1 = NTT_TRACE_1D_INV_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let c1 = NTT_TRACE_1D_CRT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let al = NTT_TRACE_ALLOC_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let w0 = NTT_TRACE_W0_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let s1_sp = NTT_SW1_SPLIT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let s1_co = NTT_SW1_COLS_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let s1_sc = NTT_SW1_SCAT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let s3_gc = NTT_SW3_GATH_COLS_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let s3_cr = NTT_SW3_CRT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let mut paths = std::mem::take(&mut *NTT_TRACE_PATHS.lock().unwrap());
    paths.sort_by_key(|e| std::cmp::Reverse(e.2));
    // Every routine's calls and time, whatever the shapes: which product forms a level ran in.
    let mut by_name: Vec<(&str, u64, u64)> = Vec::new();
    for (key, calls, ns) in &paths {
        let name = key.split(' ').next().unwrap_or("");
        if let Some(e) = by_name.iter_mut().find(|e| e.0 == name) {
            e.1 += calls;
            e.2 += ns;
        } else {
            by_name.push((name, *calls, *ns));
        }
    }
    if !by_name.is_empty() {
        let line: Vec<String> = by_name
            .iter()
            .map(|(name, calls, ns)| format!("{name}: {calls} call(s) {:.1}ms", *ns as f64 * 1e-6))
            .collect();
        eprintln!("    [forms {tag}] {}", line.join(" | "));
    }
    for (key, calls, ns) in paths.into_iter().take(10) {
        eprintln!("    [path {tag}] {:.1}ms in {calls} call(s): {key}", ns as f64 * 1e-6);
    }
    let idle = |c: &std::sync::atomic::AtomicU64| c.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let (i_al, i_1, i_2, i_3, i_w0) = (
        idle(&NTT_IDLE_ALLOC_NS),
        idle(&NTT_IDLE_SW1_NS),
        idle(&NTT_IDLE_SW2_NS),
        idle(&NTT_IDLE_SW3_NS),
        idle(&NTT_IDLE_W0_NS),
    );
    if i_al + i_1 + i_2 + i_3 + i_w0 > 0.0 {
        eprintln!(
            "[TRACE NTT {tag}] workers idle before and at the start of: alloc {i_al:.1}ms, sweep 1 {i_1:.1}ms, sweep 2 {i_2:.1}ms, sweep 3 {i_3:.1}ms, check word {i_w0:.1}ms"
        );
    }
    eprintln!(
        "[TRACE NTT {tag}] tw_build={tw:.2}ms alloc={al:.2}ms 1D(fwd={f1:.2}ms inv={i1:.2}ms crt={c1:.2}ms) 3SW(sw1={sw1:.2}ms sw2={sw2:.2}ms sw3={sw3:.2}ms w0={w0:.2}ms) | sw1_cpu(sp={s1_sp:.0} cols+tw={s1_co:.0} scat={s1_sc:.0}) sw3_cpu(gath+cols={s3_gc:.0} crt={s3_cr:.0})"
    );
}

#[inline]
fn resize_f64_buf<A: Allocator>(v: &mut Vec<f64, A>, len: usize) {
    if v.len() > len {
        v.truncate(len);
    } else if v.len() < len {
        v.resize(len, 0.0);
    }
}

/// Deterministic Miller-Rabin primality test exact for all 64-bit integers.
pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let d = (n - 1) >> (n - 1).trailing_zeros();
    let s = (n - 1).trailing_zeros();
    for a in [2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            continue;
        }
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut comp = true;
        for _ in 1..s {
            x = (((x as u128) * (x as u128)) % (n as u128)) as u64;
            if x == n - 1 {
                comp = false;
                break;
            }
        }
        if comp {
            return false;
        }
    }
    true
}

/// Cached descending prime tables `[c * 2^t + 1 < 2^50]` and cumulative `log2` bit sums
/// for `t in 40..=44` (up to 18 primes each).
fn cached_ntt_primes_desc() -> &'static [([u64; 18], [f64; 18], usize); 5] {
    static TABLE: std::sync::OnceLock<[([u64; 18], [f64; 18], usize); 5]> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut arr = [([0u64; 18], [0.0f64; 18], 0usize); 5];
        for (idx, t) in (40u32..=44).enumerate() {
            let max_c = (1u64 << (50 - t)) - 1;
            let mut cnt = 0usize;
            let mut sum_bits = 0.0f64;
            let mut c = max_c;
            while c >= 1 && cnt < 18 {
                let cand = (c << t) + 1;
                if cand < (1u64 << 50) && is_prime_u64(cand) {
                    arr[idx].0[cnt] = cand;
                    sum_bits += (cand as f64).log2();
                    arr[idx].1[cnt] = sum_bits;
                    cnt += 1;
                }
                c -= 1;
            }
            arr[idx].2 = cnt;
        }
        arr
    })
}

/// Zero-allocation slice of up to `max_count` largest NTT primes in descending order for `2^t`.
#[inline]
pub fn largest_primes_desc_slice(min_t: u32, max_count: usize) -> &'static [u64] {
    let t = min_t.max(40);
    if t >= 45 {
        return &[];
    }
    let (ref row, _, len) = cached_ntt_primes_desc()[(t - 40) as usize];
    &row[..max_count.min(len)]
}

/// Zero-allocation slice of cumulative `sum_{j <= i} log2(p_j)` for `largest_primes_desc_slice`.
#[inline]
pub fn largest_primes_prefix_bits_slice(min_t: u32, max_count: usize) -> &'static [f64] {
    let t = min_t.max(40);
    if t >= 45 {
        return &[];
    }
    let (_, ref pref, len) = cached_ntt_primes_desc()[(t - 40) as usize];
    &pref[..max_count.min(len)]
}

/// Return up to `max_count` largest NTT primes `c * 2^t + 1 < 2^50` supporting 2-adic root `2^t`.
pub fn largest_primes_for_two_adic(min_t: u32, max_count: usize) -> Vec<u64> {
    let desc = largest_primes_desc_slice(min_t, max_count);
    let mut out = Vec::with_capacity(desc.len());
    for &p in desc.iter().rev() {
        out.push(p);
    }
    // Sorted ascending `p_0 < p_1 < ... < p_{K-1}` so Garner CRT reductions are monotonic!
    out
}

#[derive(Clone, Debug)]
pub struct CrtPlan {
    pub primes_u64: Vec<u64>,
    pub primes_f64: Vec<f64>,
    pub inv_primes_f64: Vec<f64>,
    pub garner_inv: Vec<f64>,
    pub prefix_prods: Vec<u64>,
    pub k_words: usize,
    pub total_bits_f64: f64,
}

impl CrtPlan {
    pub fn build(primes_asc: Vec<u64>) -> Self {
        let k = primes_asc.len();
        debug_assert!((1..=18).contains(&k));
        let total_bits_f64: f64 = primes_asc.iter().map(|&p| (p as f64).log2()).sum();
        let k_words = ((total_bits_f64.ceil() as usize) + 6).div_ceil(64).max(2);

        let primes_f64: Vec<f64> = primes_asc.iter().map(|&p| p as f64).collect();
        let inv_primes_f64: Vec<f64> = primes_asc.iter().map(|&p| 1.0 / (p as f64)).collect();
        let mut garner_inv = vec![0.0f64; k];
        let mut prefix_prods = vec![0u64; (k + 1) * k_words];
        prefix_prods[0] = 1;

        for i in 0..k {
            if i > 0 {
                let mut m_mod = 1u64;
                for &pj in &primes_asc[..i] {
                    m_mod = (((m_mod as u128) * ((pj % primes_asc[i]) as u128))
                        % (primes_asc[i] as u128)) as u64;
                }
                garner_inv[i] = mod_pow_u64(m_mod, primes_asc[i] - 2, primes_asc[i]) as f64;
            }
            let prev = prefix_prods[(i * k_words)..((i + 1) * k_words)].to_vec();
            let dst = &mut prefix_prods[((i + 1) * k_words)..((i + 2) * k_words)];
            let mut carry = 0u128;
            for w in 0..k_words {
                let prd = (prev[w] as u128) * (primes_asc[i] as u128) + carry;
                dst[w] = prd as u64;
                carry = prd >> 64;
            }
        }

        Self {
            primes_u64: primes_asc,
            primes_f64,
            inv_primes_f64,
            garner_inv,
            prefix_prods,
            k_words,
            total_bits_f64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TableKey {
    p: u64,
    n: usize,
    negacyclic: bool,
}

pub struct PrimeTables {
    pub tw: Vec<f64>,
    pub inv_tw: Vec<f64>,
    pub g_root: u64,
    pub pr_fwd: NttPrime,
    pub pr_inv: NttPrime,
    pub inv_n: u64,
}

/// 64-byte cache-line aligned transform byte buffer backed by an ordinary `Vec<u8, TransformsAlloc>`
/// in the `Transforms` standard-allocator arena (Section 10).
pub struct AlignedBytes {
    buf: Vec<u8, TransformsAlloc>,
    needs_touch: bool,
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        let mut out = Self::with_capacity(self.buf.len());
        if !self.buf.is_empty() {
            out.buf.extend_from_slice(&self.buf);
            out.needs_touch = false;
        }
        out
    }
}

impl AlignedBytes {
    #[inline]
    pub fn with_capacity(byte_cap: usize) -> Self {
        let aligned_cap = byte_cap.div_ceil(64) * 64;
        let (buf, needs_touch) = crate::arena::alloc_transforms_bytes_uninit(aligned_cap);
        Self { buf, needs_touch }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_byte_len: usize) {
        debug_assert!(new_byte_len <= self.buf.capacity());
        unsafe {
            self.buf.set_len(new_byte_len);
        }
    }

    #[inline]
    pub unsafe fn set_len_initialized(&mut self, new_byte_len: usize) {
        debug_assert!(new_byte_len <= self.buf.capacity());
        unsafe {
            self.buf.set_len(new_byte_len);
        }
        self.needs_touch = false;
    }

    #[inline]
    pub fn as_padded_u64_slice(&mut self) -> &[u64] {
        let byte_len = self.buf.len();
        let limb_len = byte_len.div_ceil(8);
        let padded_bytes = limb_len * 8;
        debug_assert!(padded_bytes <= self.buf.capacity());
        if padded_bytes > byte_len {
            unsafe {
                std::ptr::write_bytes(self.buf.as_mut_ptr().add(byte_len), 0, padded_bytes - byte_len);
            }
        }
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u64, limb_len) }
    }

    #[inline]
    pub fn truncate(&mut self, new_byte_len: usize) {
        self.buf.truncate(new_byte_len);
    }

    pub fn resize_zeroed(&mut self, new_byte_len: usize) {
        if new_byte_len <= self.buf.len() {
            self.buf.truncate(new_byte_len);
            return;
        }
        let aligned_cap = new_byte_len.div_ceil(64) * 64;
        if aligned_cap > self.buf.capacity() {
            self.buf.reserve_exact(aligned_cap - self.buf.len());
        }
        let old_len = self.buf.len();
        unsafe {
            std::ptr::write_bytes(
                self.buf.as_mut_ptr().add(old_len),
                0,
                new_byte_len - old_len,
            );
            self.buf.set_len(new_byte_len);
        }
        self.needs_touch = false;
    }
}

pub fn clear_tls_bufs() {
    let mut sc = NttTlsScratch::new();
    let _ = NTT_TLS_SCRATCH.try_with(|cell| {
        sc = std::mem::replace(&mut *cell.borrow_mut(), NttTlsScratch::new());
    });
    drop(sc);
    clear_tls_1d_fused();
}

pub fn clear_tls_1d_fused() {
    let mut drop_bufs = NttTls1dFused::new();
    let _ = NTT_TLS_1D_FUSED.try_with(|cell| {
        drop_bufs = std::mem::replace(&mut *cell.borrow_mut(), NttTls1dFused::new());
    });
    drop(drop_bufs);
}

fn prewarm_tls_scratch_one(
    max_rows: usize,
    max_cols: usize,
    max_k: usize,
    max_plane_words: usize,
) {
    let mut lease = ScratchLease::acquire();
    let sc = lease.get_mut();
    *sc = NttTlsScratch::new();
    let tile_cap = max_rows.saturating_mul(256).max(524_288);
    let tw_step_cap = max_rows
        .min(2048)
        .saturating_mul(256)
        .max(max_plane_words)
        .max(262_144);
    let row_cap = max_cols.max(32 * 256).max(max_k.saturating_mul(max_rows).saturating_mul(8));
    let ra2_cap = max_rows.saturating_mul(208).max(row_cap);
    sc.tile.resize(tile_cap, 0.0);
    sc.tw_row.resize(max_rows.max(64), 1.0);
    sc.tw_step.resize(tw_step_cap, 0.0);
    sc.s_u64.resize(max_rows.max(64), 0);
    sc.ra1.resize(row_cap, 0.0);
    sc.rb1.resize(row_cap, 0.0);
    sc.ra2.resize(ra2_cap, 0.0);
    sc.rb2.resize(32_768, 0.0);
    sc.rc.resize(32_768, 0.0);
    sc.prime_segs.resize_with(max_k, || Vec::new_in(ScratchAlloc));
    for pi in 0..max_k {
        sc.prime_segs[pi].resize(pi * 40 + max_rows.saturating_mul(32), 0.0);
    }
    sc.row_ptrs.resize(max_k, 0);
}

pub fn prewarm_tls_scratch(
    sched: &Scheduler,
    max_rows: usize,
    max_cols: usize,
    max_k: usize,
    max_plane_words: usize,
) {
    if max_rows <= 1 && max_cols == 0 {
        return;
    }
    sched.on_each_worker(|| {
        prewarm_tls_scratch_one(max_rows, max_cols, max_k, max_plane_words);
    });
}

type TwInflight = Arc<(Mutex<Option<Arc<PrimeTables>>>, std::sync::Condvar)>;
type Tile0Inflight = Arc<(Mutex<Option<Arc<Vec<f64>>>>, std::sync::Condvar)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileTwKey {
    p: u64,
    n: usize,
    rows: usize,
    tile_cols: usize,
    negacyclic: bool,
    inverse: bool,
}

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static TLS_CRT: RefCell<HashMap<(u64, u32, usize), Arc<CrtPlan>>> =
        RefCell::new(HashMap::new());
    static TLS_TW: RefCell<HashMap<(u64, TableKey), Arc<PrimeTables>>> =
        RefCell::new(HashMap::new());
}

pub struct NttTableCache {
    cache_id: u64,
    crt_cache: Mutex<HashMap<(u32, usize), Arc<CrtPlan>>>,
    tw_cache: Mutex<HashMap<TableKey, Arc<PrimeTables>>>,
    tw_inflight: Mutex<HashMap<TableKey, TwInflight>>,
    tile0_cache: Mutex<HashMap<TileTwKey, Arc<Vec<f64>>>>,
    tile0_inflight: Mutex<HashMap<TileTwKey, Tile0Inflight>>,
    tile0_bytes: AtomicUsize,
    max_tw_bytes: AtomicUsize,
    l3_prime_cap_bytes: AtomicUsize,
    work_grid_room: AtomicU64,
    ooc_tile_budget: AtomicU64,
}

impl Default for NttTableCache {
    fn default() -> Self {
        Self {
            cache_id: NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            crt_cache: Mutex::new(HashMap::new()),
            tw_cache: Mutex::new(HashMap::new()),
            tw_inflight: Mutex::new(HashMap::new()),
            tile0_cache: Mutex::new(HashMap::new()),
            tile0_inflight: Mutex::new(HashMap::new()),
            tile0_bytes: AtomicUsize::new(0),
            max_tw_bytes: AtomicUsize::new(512 * 1024 * 1024),
            l3_prime_cap_bytes: AtomicUsize::new(0),
            work_grid_room: AtomicU64::new(u64::MAX),
            ooc_tile_budget: AtomicU64::new(64 * 1024 * 1024),
        }
    }
}

impl NttTableCache {
    pub fn set_max_tw_bytes(&self, bytes: usize) {
        self.max_tw_bytes.store(bytes, Ordering::Relaxed);
    }

    pub fn set_l3_prime_cap_bytes(&self, cap_bytes: usize) {
        self.l3_prime_cap_bytes.store(cap_bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn l3_prime_cap_bytes(&self) -> usize {
        let v = self.l3_prime_cap_bytes.load(Ordering::Relaxed);
        if v > 0 { v } else { 32 * 1024 * 1024 }
    }

    /// The most a shared-operand pair's grids may take, in bytes, for the shared operand to be
    /// given a work grid a prime instead of one. Unbounded until set.
    pub fn set_work_grid_room(&self, bytes: u64) {
        self.work_grid_room.store(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn work_grid_room(&self) -> u64 {
        self.work_grid_room.load(Ordering::Relaxed)
    }

    pub fn set_ooc_tile_budget(&self, bytes: u64) {
        self.ooc_tile_budget.store(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn ooc_tile_budget_bytes(&self) -> u64 {
        let v = self.ooc_tile_budget.load(Ordering::Relaxed);
        if v > 0 { v } else { 64 * 1024 * 1024 }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tile0_if_unclaimed(
        &self,
        kernels: &Kernels,
        p: u64,
        g_root: u64,
        n: usize,
        rows: usize,
        tile_cols: usize,
        negacyclic: bool,
        inverse: bool,
    ) {
        let key = TileTwKey {
            p,
            n,
            rows,
            tile_cols,
            negacyclic,
            inverse,
        };
        if self.tile0_cache.lock().unwrap().contains_key(&key) {
            return;
        }
        let pair = {
            let mut inf = self.tile0_inflight.lock().unwrap();
            if self.tile0_cache.lock().unwrap().contains_key(&key) || inf.contains_key(&key) {
                return;
            }
            let pair = Arc::new((Mutex::new(None), std::sync::Condvar::new()));
            inf.insert(key, Arc::clone(&pair));
            pair
        };
        let built = crate::arena::with_system_alloc(|| {
            self.build_tile0_twiddles_uncached(
                key, kernels, p, g_root, n, rows, tile_cols, negacyclic, inverse,
            )
        });
        {
            let mut inf = self.tile0_inflight.lock().unwrap();
            inf.remove(&key);
        }
        let (lock, cvar) = &*pair;
        *lock.lock().unwrap() = Some(built);
        cvar.notify_all();
    }

    #[allow(clippy::too_many_arguments)]
    fn get_tile0_twiddles(
        &self,
        kernels: &Kernels,
        p: u64,
        g_root: u64,
        n: usize,
        rows: usize,
        tile_cols: usize,
        negacyclic: bool,
        inverse: bool,
    ) -> Arc<Vec<f64>> {
        let key = TileTwKey {
            p,
            n,
            rows,
            tile_cols,
            negacyclic,
            inverse,
        };
        if let Some(found) = self.tile0_cache.lock().unwrap().get(&key).cloned() {
            return found;
        }
        let (wait_pair, is_builder) = {
            let mut inf = self.tile0_inflight.lock().unwrap();
            if let Some(found) = self.tile0_cache.lock().unwrap().get(&key).cloned() {
                return found;
            }
            if let Some(existing) = inf.get(&key) {
                (Arc::clone(existing), false)
            } else {
                let pair = Arc::new((Mutex::new(None), std::sync::Condvar::new()));
                inf.insert(key, Arc::clone(&pair));
                (pair, true)
            }
        };
        if !is_builder {
            let (lock, cvar) = &*wait_pair;
            let mut ready = lock.lock().unwrap();
            while ready.is_none() {
                ready = cvar.wait(ready).unwrap();
            }
            return Arc::clone(ready.as_ref().unwrap());
        }
        let built = crate::arena::with_system_alloc(|| {
            self.build_tile0_twiddles_uncached(
                key, kernels, p, g_root, n, rows, tile_cols, negacyclic, inverse,
            )
        });
        {
            let mut inf = self.tile0_inflight.lock().unwrap();
            inf.remove(&key);
        }
        let (lock, cvar) = &*wait_pair;
        *lock.lock().unwrap() = Some(Arc::clone(&built));
        cvar.notify_all();
        built
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tile0_twiddles_uncached(
        &self,
        key: TileTwKey,
        kernels: &Kernels,
        p: u64,
        g_root: u64,
        n: usize,
        rows: usize,
        tile_cols: usize,
        negacyclic: bool,
        inverse: bool,
    ) -> Arc<Vec<f64>> {
        let t0 = std::time::Instant::now();
        let tile_elems = rows * tile_cols;
        let cols = if rows > 0 { n / rows } else { 0 };
        let num_tiles = if tile_cols > 0 { cols / tile_cols } else { 0 };
        let max_tw_bytes = self.max_tw_bytes.load(Ordering::Relaxed);
        let cache_full = num_tiles > 1
            && num_tiles * tile_elems == n
            && n <= 131_072
            && self.tile0_bytes.load(Ordering::Relaxed) + n * 8 <= max_tw_bytes / 4;
        let use_bg_rs = num_tiles > 256;
        let num_hi = if use_bg_rs { num_tiles.div_ceil(64) } else { 0 };
        let rs_total_len = if use_bg_rs {
            tile_elems
                .saturating_add((64 + num_hi).saturating_mul(rows))
                .saturating_add(1)
        } else {
            tile_elems
                .saturating_add(num_tiles.saturating_mul(rows))
                .saturating_add(1)
        };
        let append_row_scales = !cache_full
            && rows > 1
            && tile_cols > 1
            && num_tiles > 1
            && num_tiles * tile_elems == n
            && rs_total_len * 8 <= 32 * 1024 * 1024
            && self.tile0_bytes.load(Ordering::Relaxed) + rs_total_len * 8 <= max_tw_bytes;

        let pr = NttPrime::new(p, 1);
        let psi = build_psi_params(p, g_root, n, negacyclic, inverse);
        let mut tr = Vec::new();
        let mut ts = Vec::new();

        let mut buf = if cache_full {
            let mut full = vec![1.0f64; n];
            twist_tile_rows(
                kernels,
                &mut full[..tile_elems],
                rows,
                tile_cols,
                0,
                psi,
                pr.p,
                pr.inv_p,
                &mut tr,
                &mut ts,
            );
            let mu = barrett_mu_p50(p);
            let s_row = rows.ilog2() as usize;
            let mut s_u64 = vec![1u64; rows];
            let mut s_gen = [1u64; 32];
            if s_row > 0 {
                s_gen[s_row - 1] = pow_mod_p50(psi.root_sq_u64, tile_cols as u64, p, mu);
                for idx in (0..(s_row - 1)).rev() {
                    let nxt = s_gen[idx + 1];
                    s_gen[idx] = mul_mod_p50(nxt, nxt, p, mu);
                }
            }
            s_u64[0] = if negacyclic {
                pow_mod_p50(psi.first_u64, tile_cols as u64, p, mu)
            } else {
                1
            };
            for (m, &gm) in s_gen.iter().enumerate().take(s_row) {
                let half = 1usize << m;
                for j in 0..half {
                    s_u64[half + j] = mul_mod_p50(s_u64[j], gm, p, mu);
                }
            }
            let mut step_tile = vec![1.0f64; tile_elems];
            for r in 0..rows {
                step_tile[(r * tile_cols)..((r + 1) * tile_cols)].fill(s_u64[r] as f64);
            }
            let start_r = usize::from(!negacyclic);
            let row_skip = start_r * tile_cols;
            let n_mul = tile_elems - row_skip;
            if n_mul > 0 {
                let vcfg = NttVecCfg {
                    n: n_mul as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                    sub_second: 0,
                };
                for ct in 1..num_tiles {
                    let prev_off = (ct - 1) * tile_elems + row_skip;
                    let cur_off = ct * tile_elems + row_skip;
                    unsafe {
                        (kernels.active.ntt_pmul)(
                            full.as_mut_ptr().add(cur_off),
                            full.as_ptr().add(prev_off),
                            step_tile.as_ptr().add(row_skip),
                            &vcfg,
                        );
                    }
                }
            }
            full
        } else {
            let mut tile0 = Vec::with_capacity(if append_row_scales {
                rs_total_len
            } else {
                tile_elems
            });
            tile0.resize(tile_elems, 1.0f64);
            twist_tile_rows(
                kernels,
                &mut tile0[..tile_elems],
                rows,
                tile_cols,
                0,
                psi,
                pr.p,
                pr.inv_p,
                &mut tr,
                &mut ts,
            );
            tile0
        };
        if inverse && rows > 1 {
            let mu = barrett_mu_p50(p);
            let inv_rows = pow_mod_p50(rows as u64, p - 2, p, mu);
            for x in &mut buf {
                *x = mul_mod_p50(*x as u64, inv_rows, p, mu) as f64;
            }
        }
        if append_row_scales {
            buf.resize(rs_total_len, 1.0f64);
            buf[rs_total_len - 1] = if use_bg_rs { -1.0 } else { 0.0 };
            let mut s_step = Vec::new();
            fill_row_scales_u64(rows, tile_cols, psi, p, &mut s_step);
            let rs_base = tile_elems;
            for r in 0..rows {
                buf[rs_base + rows + r] = s_step[r] as f64;
            }
            let start_r = usize::from(!negacyclic);
            if rows > start_r {
                let vcfg_rs = NttVecCfg {
                    n: (rows - start_r) as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                    sub_second: 0,
                };
                let step_off = rs_base + rows + start_r;
                let lo_limit = if use_bg_rs { 64 } else { num_tiles };
                for ct in 2..lo_limit {
                    let prev_off = rs_base + (ct - 1) * rows + start_r;
                    let cur_off = rs_base + ct * rows + start_r;
                    unsafe {
                        (kernels.active.ntt_pmul)(
                            buf.as_mut_ptr().add(cur_off),
                            buf.as_ptr().add(prev_off),
                            buf.as_ptr().add(step_off),
                            &vcfg_rs,
                        );
                    }
                }
                if use_bg_rs && num_hi > 1 {
                    let hi_base = rs_base + 64 * rows;
                    let lo63_off = rs_base + 63 * rows + start_r;
                    let hi1_off = hi_base + rows + start_r;
                    unsafe {
                        (kernels.active.ntt_pmul)(
                            buf.as_mut_ptr().add(hi1_off),
                            buf.as_ptr().add(lo63_off),
                            buf.as_ptr().add(step_off),
                            &vcfg_rs,
                        );
                    }
                    for hi in 2..num_hi {
                        let prev_hi = hi_base + (hi - 1) * rows + start_r;
                        let cur_hi = hi_base + hi * rows + start_r;
                        unsafe {
                            (kernels.active.ntt_pmul)(
                                buf.as_mut_ptr().add(cur_hi),
                                buf.as_ptr().add(prev_hi),
                                buf.as_ptr().add(hi1_off),
                                &vcfg_rs,
                            );
                        }
                    }
                }
            }
        }
        NTT_TRACE_TW_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let added_bytes = buf.len() * 8;
        let arc = Arc::new(buf);
        let mut g = self.tile0_cache.lock().unwrap();
        if let Some(existing) = g.get(&key) {
            return Arc::clone(existing);
        }
        self.tile0_bytes.fetch_add(added_bytes, Ordering::Relaxed);
        g.insert(key, Arc::clone(&arc));
        arc
    }

    pub fn take_byte_buf(&self, len: usize) -> AlignedBytes {
        let mut v = AlignedBytes::with_capacity(len);
        v.resize_zeroed(len);
        v
    }

    pub fn take_byte_bufs(&self, count: usize, len: usize) -> Vec<AlignedBytes> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut v = AlignedBytes::with_capacity(len);
            v.resize_zeroed(len);
            out.push(v);
        }
        out
    }

    pub fn take_byte_bufs_uninit(&self, count: usize, len: usize) -> Vec<AlignedBytes> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(AlignedBytes::with_capacity(len));
        }
        out
    }

    pub fn take_byte_bufs_committed(
        &self,
        count: usize,
        len: usize,
        sched: &Scheduler,
        max_workers: usize,
    ) -> Vec<AlignedBytes> {
        self.take_byte_bufs_committed_impl(count, len, sched, max_workers)
    }

    pub fn take_byte_bufs_transient_committed(
        &self,
        count: usize,
        len: usize,
        sched: &Scheduler,
        max_workers: usize,
    ) -> Vec<AlignedBytes> {
        self.take_byte_bufs_committed_impl(count, len, sched, max_workers)
    }

    fn take_byte_bufs_committed_impl(
        &self,
        count: usize,
        len: usize,
        sched: &Scheduler,
        max_workers: usize,
    ) -> Vec<AlignedBytes> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(AlignedBytes::with_capacity(len));
        }
        if len >= 512 * 1024 {
            let mut touch_ptrs: Vec<usize> = Vec::new();
            for buf in &mut out {
                if buf.needs_touch {
                    touch_ptrs.push(buf.as_mut_ptr() as usize);
                    buf.needs_touch = false;
                }
            }
            let touch_cnt = touch_ptrs.len();
            if touch_cnt > 0 {
                let pages_per_buf = len.div_ceil(4096);
                let total_pages = touch_cnt * pages_per_buf;
                let nw = max_workers.min(sched.num_workers()).max(1);
                if nw <= 1 || total_pages < 64 {
                    for &base_addr in &touch_ptrs {
                        let base = base_addr as *mut u8;
                        let mut off = 0usize;
                        while off < len {
                            unsafe {
                                std::ptr::write_volatile(base.add(off), 0);
                            }
                            off += 4096;
                        }
                    }
                } else {
                    let pages_per_w = total_pages.div_ceil(nw);
                    let ptrs_ref = touch_ptrs.as_slice();
                    sched.par_for_leaf_max(nw, 1, nw, |wi| {
                        let p0 = wi * pages_per_w;
                        let p1 = ((wi + 1) * pages_per_w).min(total_pages);
                        for p in p0..p1 {
                            let b = p / pages_per_buf;
                            let off = (p % pages_per_buf) * 4096;
                            if off < len {
                                let base = ptrs_ref[b] as *mut u8;
                                unsafe {
                                    std::ptr::write_volatile(base.add(off), 0);
                                }
                            }
                        }
                    });
                }
            }
        }
        out
    }

    #[inline]
    pub fn put_byte_bufs(&self, bufs: Vec<AlignedBytes>) {
        drop(bufs);
    }

    pub fn take_u64_buf(&self, len: usize) -> TransU64Vec {
        let mut v = Vec::with_capacity_in(len, TransformsAlloc);
        v.resize(len, 0);
        v
    }

    pub fn take_u64_buf_uninit(&self, len: usize) -> TransU64Vec {
        let mut v = Vec::with_capacity_in(len, TransformsAlloc);
        unsafe {
            v.set_len(len);
        }
        v
    }

    #[inline]
    pub fn put_u64_buf(&self, v: TransU64Vec) {
        drop(v);
    }

    #[inline]
    pub fn get_crt_ref(&self, min_t: u32, k_primes: usize) -> &CrtPlan {
        let key = (min_t.max(40), k_primes);
        if let Some(ptr) = TLS_CRT.with(|cell| {
            cell.borrow()
                .get(&(self.cache_id, key.0, key.1))
                .map(Arc::as_ptr)
        }) {
            return unsafe { &*ptr };
        }
        let arc = self.get_crt(min_t, k_primes);
        unsafe { &*Arc::as_ptr(&arc) }
    }

    pub fn get_crt(&self, min_t: u32, k_primes: usize) -> Arc<CrtPlan> {
        let key = (min_t.max(40), k_primes);
        let cid = self.cache_id;
        if let Some(hit) = TLS_CRT.with(|cell| cell.borrow().get(&(cid, key.0, key.1)).cloned()) {
            return hit;
        }
        let found = {
            let g = self.crt_cache.lock().unwrap();
            g.get(&key).cloned()
        };
        let res = if let Some(v) = found {
            v
        } else {
            crate::arena::with_system_alloc(|| {
                let ps = largest_primes_for_two_adic(key.0, key.1);
                let plan = Arc::new(CrtPlan::build(ps));
                let mut g = self.crt_cache.lock().unwrap();
                Arc::clone(g.entry(key).or_insert(plan))
            })
        };
        TLS_CRT.with(|cell| {
            let mut map = cell.borrow_mut();
            if map.len() >= 128 {
                map.clear();
            }
            map.insert((cid, key.0, key.1), Arc::clone(&res));
        });
        res
    }

    #[inline]
    pub fn get_twiddles_ref(&self, p: u64, n: usize, negacyclic: bool) -> &PrimeTables {
        let key = TableKey { p, n, negacyclic };
        if let Some(ptr) = TLS_TW.with(|cell| {
            cell.borrow().get(&(self.cache_id, key)).map(Arc::as_ptr)
        }) {
            return unsafe { &*ptr };
        }
        let arc = self.get_twiddles(p, n, negacyclic);
        unsafe { &*Arc::as_ptr(&arc) }
    }

    pub fn get_twiddles(&self, p: u64, n: usize, negacyclic: bool) -> Arc<PrimeTables> {
        let key = TableKey { p, n, negacyclic };
        let cid = self.cache_id;
        if let Some(hit) = TLS_TW.with(|cell| cell.borrow().get(&(cid, key)).cloned()) {
            return hit;
        }
        {
            let g = self.tw_cache.lock().unwrap();
            if let Some(v) = g.get(&key) {
                let cloned = Arc::clone(v);
                TLS_TW.with(|cell| {
                    let mut map = cell.borrow_mut();
                    if map.len() >= 256 {
                        map.clear();
                    }
                    map.insert((cid, key), Arc::clone(&cloned));
                });
                return cloned;
            }
        }
        let (wait_pair, is_builder) = {
            let mut inf = self.tw_inflight.lock().unwrap();
            {
                let g = self.tw_cache.lock().unwrap();
                if let Some(v) = g.get(&key) {
                    let cloned = Arc::clone(v);
                    TLS_TW.with(|cell| {
                        let mut map = cell.borrow_mut();
                        if map.len() >= 256 {
                            map.clear();
                        }
                        map.insert((cid, key), Arc::clone(&cloned));
                    });
                    return cloned;
                }
            }
            if let Some(existing) = inf.get(&key) {
                (Arc::clone(existing), false)
            } else {
                let pair = Arc::new((Mutex::new(None), std::sync::Condvar::new()));
                inf.insert(key, Arc::clone(&pair));
                (pair, true)
            }
        };

        if !is_builder {
            let (lock, cvar) = &*wait_pair;
            let mut ready = lock.lock().unwrap();
            while ready.is_none() {
                ready = cvar.wait(ready).unwrap();
            }
            let cloned = Arc::clone(ready.as_ref().unwrap());
            TLS_TW.with(|cell| {
                let mut map = cell.borrow_mut();
                if map.len() >= 256 {
                    map.clear();
                }
                map.insert((cid, key), Arc::clone(&cloned));
            });
            return cloned;
        }

        let t0 = std::time::Instant::now();
        let (tw, inv_tw) = crate::arena::with_system_alloc(|| build_ntt_tables(p, n, negacyclic));
        let g_root = primitive_root_ntt(p);
        let pr_fwd = NttPrime::new(p, 1);
        let inv_n = if n > 0 {
            mod_pow_u64(n as u64, p - 2, p)
        } else {
            1
        };
        let pr_inv = NttPrime::new(p, inv_n);
        NTT_TRACE_TW_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let entry = crate::arena::with_system_alloc(|| {
            Arc::new(PrimeTables {
                tw,
                inv_tw,
                g_root,
                pr_fwd,
                pr_inv,
                inv_n,
            })
        });
        {
            let mut g = self.tw_cache.lock().unwrap();
            g.insert(key, Arc::clone(&entry));
        }
        {
            let mut inf = self.tw_inflight.lock().unwrap();
            inf.remove(&key);
        }
        let (lock, cvar) = &*wait_pair;
        *lock.lock().unwrap() = Some(Arc::clone(&entry));
        cvar.notify_all();
        TLS_TW.with(|cell| {
            let mut map = cell.borrow_mut();
            if map.len() >= 256 {
                map.clear();
            }
            map.insert((cid, key), Arc::clone(&entry));
        });
        entry
    }

    fn build_twiddles_if_unclaimed(&self, p: u64, n: usize, negacyclic: bool) {
        let key = TableKey { p, n, negacyclic };
        if self.tw_cache.lock().unwrap().contains_key(&key) {
            return;
        }
        let wait_pair = {
            let mut inf = self.tw_inflight.lock().unwrap();
            if self.tw_cache.lock().unwrap().contains_key(&key) || inf.contains_key(&key) {
                return;
            }
            let pair = Arc::new((Mutex::new(None), std::sync::Condvar::new()));
            inf.insert(key, Arc::clone(&pair));
            pair
        };
        let t0 = std::time::Instant::now();
        let (tw, inv_tw) = crate::arena::with_system_alloc(|| build_ntt_tables(p, n, negacyclic));
        let g_root = primitive_root_ntt(p);
        let pr_fwd = NttPrime::new(p, 1);
        let inv_n = if n > 0 {
            mod_pow_u64(n as u64, p - 2, p)
        } else {
            1
        };
        let pr_inv = NttPrime::new(p, inv_n);
        NTT_TRACE_TW_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let entry = crate::arena::with_system_alloc(|| {
            Arc::new(PrimeTables {
                tw,
                inv_tw,
                g_root,
                pr_fwd,
                pr_inv,
                inv_n,
            })
        });
        {
            let mut g = self.tw_cache.lock().unwrap();
            g.insert(key, Arc::clone(&entry));
        }
        {
            let mut inf = self.tw_inflight.lock().unwrap();
            inf.remove(&key);
        }
        let (lock, cvar) = &*wait_pair;
        *lock.lock().unwrap() = Some(Arc::clone(&entry));
        cvar.notify_all();
        let cid = self.cache_id;
        TLS_TW.with(|cell| {
            let mut map = cell.borrow_mut();
            if map.len() >= 256 {
                map.clear();
            }
            map.insert((cid, key), entry);
        });
    }

    pub fn get_twiddles_refs_par(
        &self,
        sched: &Scheduler,
        par_workers: usize,
        primes: &[u64],
        k: usize,
        n: usize,
        negacyclic: bool,
    ) -> Vec<&PrimeTables> {
        let cid = self.cache_id;
        let all_in_tls = TLS_TW.with(|cell| {
            let m = cell.borrow();
            (0..k).all(|pi| {
                m.contains_key(&(
                    cid,
                    TableKey {
                        p: primes[pi],
                        n,
                        negacyclic,
                    },
                ))
            })
        });
        if !all_in_tls && par_workers > 1 && k > 1 && n >= 256 {
            let any_missing = {
                let g = self.tw_cache.lock().unwrap();
                (0..k).any(|pi| {
                    !g.contains_key(&TableKey {
                        p: primes[pi],
                        n,
                        negacyclic,
                    })
                })
            };
            if any_missing {
                sched.par_for_max(k, 1, par_workers, |pi| {
                    self.build_twiddles_if_unclaimed(primes[pi], n, negacyclic);
                });
            }
        }
        (0..k)
            .map(|pi| self.get_twiddles_ref(primes[pi], n, negacyclic))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn get_tile0_twiddles_par(
        &self,
        kernels: &Kernels,
        sched: &Scheduler,
        par_workers: usize,
        primes: &[u64],
        col_tables: &[&PrimeTables],
        k: usize,
        n: usize,
        rows: usize,
        tile_cols: usize,
        negacyclic: bool,
        inverse: bool,
    ) -> Vec<Arc<Vec<f64>>> {
        {
            let g = self.tile0_cache.lock().unwrap();
            let mut out = Vec::with_capacity(k);
            for &p in primes.iter().take(k) {
                let key = TileTwKey {
                    p,
                    n,
                    rows,
                    tile_cols,
                    negacyclic,
                    inverse,
                };
                if let Some(v) = g.get(&key) {
                    out.push(Arc::clone(v));
                } else {
                    break;
                }
            }
            if out.len() == k {
                return out;
            }
        }
        if par_workers > 1 && k > 1 {
            sched.par_for_max(k, 1, par_workers, |pi| {
                self.build_tile0_if_unclaimed(
                    kernels,
                    primes[pi],
                    col_tables[pi].g_root,
                    n,
                    rows,
                    tile_cols,
                    negacyclic,
                    inverse,
                );
            });
        }
        (0..k)
            .map(|pi| {
                self.get_tile0_twiddles(
                    kernels,
                    primes[pi],
                    col_tables[pi].g_root,
                    n,
                    rows,
                    tile_cols,
                    negacyclic,
                    inverse,
                )
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttSpec {
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub tile_cols: usize,
    pub extra_sweeps: u32,
    pub k_primes: usize,
    pub coeff_bits: u32,
    pub packed: bool,
    pub use_nt: bool,
    pub negacyclic: bool,
    pub wrap_cyclic: bool,
    pub out_of_core: bool,
    pub par_workers: usize,
}

impl NttSpec {
    #[inline]
    pub fn min_two_adic(&self) -> u32 {
        let log_n = self.n.ilog2();
        if self.negacyclic { log_n + 1 } else { log_n }
    }

    #[inline]
    pub fn outer_row_bytes(&self) -> usize {
        if self.packed {
            debug_assert_eq!(self.cols % 8, 0);
            let base = (self.cols / 8) * 50;
            if self.rows > 1 && base.is_multiple_of(4096) {
                base + 512
            } else {
                base
            }
        } else if self.rows > 1 && self.cols >= 512 {
            self.cols * 8 + 512
        } else {
            self.cols * 8
        }
    }
}

pub enum ResidueStorage {
    InRam {
        bufs: Vec<AlignedBytes>,
    }, // per prime: `rows * outer_row_bytes` (or `n * 8` when `rows == 1 && !packed`)
    InStore(Vec<crate::store::ExtentHandle>),
}

pub struct NttCachedFwd {
    pub spec: NttSpec,
    /// True when rows are already transformed into frequency domain (`rows == 1` or reusable
    /// `rows > 1` in-RAM cache), allowing Sweep 2 to skip redundant `ntt_fwd` row passes!
    pub rows_transformed: bool,
    pub storage: ResidueStorage,
}

impl NttCachedFwd {
    pub fn take_in_ram_bufs(&mut self) -> Option<Vec<AlignedBytes>> {
        match &mut self.storage {
            ResidueStorage::InRam { bufs, .. } if !bufs.is_empty() => Some(std::mem::take(bufs)),
            _ => None,
        }
    }

    pub fn release_store(&mut self, store: Option<&Store>) {
        if let ResidueStorage::InStore(handles) = &mut self.storage {
            if let Some(st) = store {
                for h in handles.drain(..) {
                    st.free(h);
                }
            } else {
                handles.clear();
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PsiParams {
    first_u64: u64,
    root_sq_u64: u64,
    negacyclic: bool,
}

/// Precompute Bailey 4-step twist generators `(first_u64, root_sq_u64)` where
/// `psi[bitrev_R(br)] = first * root_sq^br mod p` for `br in 0..rows`.
fn build_psi_params(
    p_u64: u64,
    g_root: u64,
    n: usize,
    negacyclic: bool,
    inverse: bool,
) -> PsiParams {
    let two_n = (2 * n) as u64;
    let om = mod_pow_u64(g_root, (p_u64 - 1) / two_n, p_u64);
    let root_u64 = if inverse {
        mod_pow_u64(om, p_u64 - 2, p_u64)
    } else {
        om
    };
    let root_sq_u64 = (((root_u64 as u128) * (root_u64 as u128)) % (p_u64 as u128)) as u64;
    let first_u64 = if negacyclic { root_u64 } else { 1 };
    PsiParams {
        first_u64,
        root_sq_u64,
        negacyclic,
    }
}

#[inline(always)]
fn barrett_mu_p50(p_u64: u64) -> u64 {
    if p_u64 > (1u64 << 49) {
        ((1u128 << 113) / (p_u64 as u128)) as u64
    } else {
        0
    }
}

#[inline(always)]
fn mul_mod_p50(a: u64, b: u64, p_u64: u64, mu: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    if mu != 0 {
        let hi = (prod >> 49) as u64;
        let q = (((hi as u128) * (mu as u128)) >> 64) as u64;
        let mut r = (prod - (q as u128) * (p_u64 as u128)) as u64;
        if r >= p_u64 {
            r -= p_u64;
        }
        if r >= p_u64 {
            r -= p_u64;
        }
        debug_assert_eq!(r, (prod % (p_u64 as u128)) as u64);
        r
    } else {
        (prod % (p_u64 as u128)) as u64
    }
}

#[inline]
fn pow_mod_p50(mut base: u64, mut exp: u64, p_u64: u64, mu: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) != 0 {
            acc = mul_mod_p50(acc, base, p_u64, mu);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod_p50(base, base, p_u64, mu);
        }
    }
    acc
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn fill_geom_row_vec(
    kernels: &Kernels,
    dst: &mut [f64],
    w_u64: u64,
    col_start: usize,
    p_u64: u64,
    mu: u64,
    p_f: f64,
    inv_p_f: f64,
    scratch: &mut [f64],
) {
    let len = dst.len();
    if len == 0 {
        return;
    }
    if w_u64 == 1 {
        dst.fill(1.0);
        return;
    }
    let w_base = if col_start == 0 {
        1u64
    } else {
        pow_mod_p50(w_u64, col_start as u64, p_u64, mu)
    };
    let init = len.min(16);
    let mut cur = w_base;
    dst[0] = cur as f64;
    for slot in dst.iter_mut().take(init).skip(1) {
        cur = mul_mod_p50(cur, w_u64, p_u64, mu);
        *slot = cur as f64;
    }
    let mut done = init;
    if done < len {
        let mut w_pow = pow_mod_p50(w_u64, done as u64, p_u64, mu);
        while done < len {
            let chunk = (len - done).min(done);
            scratch[..chunk].fill(w_pow as f64);
            let vcfg = NttVecCfg {
                n: chunk as u64,
                p: p_f,
                inv_p: inv_p_f,
                sub_second: 0,
            };
            unsafe {
                (kernels.active.ntt_pmul)(
                    dst.as_mut_ptr().add(done),
                    dst.as_ptr(),
                    scratch.as_ptr(),
                    &vcfg,
                );
            }
            done += chunk;
            if done < len {
                w_pow = mul_mod_p50(w_pow, w_pow, p_u64, mu);
            }
        }
    }
}

/// Multiply each row `r in 0..rows` of `tile` (`rows * tile_cols`) in-place inside L1/L2 by
/// `psi[r]^(col_start + tc)` (`tc in 0..tile_cols`), building the `rows * tile_cols` twist tile
/// in `log2(rows)` vector `ntt_pmul` doublings and applying it in one contiguous `ntt_pmul` call.
#[allow(clippy::too_many_arguments)]
fn twist_tile_rows<A1: Allocator, A2: Allocator>(
    kernels: &Kernels,
    tile: &mut [f64],
    rows: usize,
    tile_cols: usize,
    col_start: usize,
    psi: PsiParams,
    p_f: f64,
    inv_p_f: f64,
    tw_row: &mut Vec<f64, A1>,
    tw_step: &mut Vec<f64, A2>,
) {
    if rows == 0
        || tile_cols == 0
        || (col_start == 0 && tile_cols == 1)
        || (!psi.negacyclic && rows <= 1)
    {
        return;
    }
    let p_u64 = p_f as u64;
    let mu = barrett_mu_p50(p_u64);
    let s_row = rows.ilog2() as usize;
    let total_elems = rows * tile_cols;
    let max_half_elems = (rows / 2).max(1) * tile_cols;
    resize_f64_buf(tw_row, total_elems);
    resize_f64_buf(tw_step, max_half_elems);

    let mut gen_m = [1u64; 32];
    if s_row > 0 {
        gen_m[s_row - 1] = psi.root_sq_u64;
        for idx in (0..(s_row - 1)).rev() {
            let prev = gen_m[idx + 1];
            gen_m[idx] = mul_mod_p50(prev, prev, p_u64, mu);
        }
    }

    if !psi.negacyclic {
        tw_row[..tile_cols].fill(1.0);
    } else {
        fill_geom_row_vec(
            kernels,
            &mut tw_row[..tile_cols],
            psi.first_u64,
            col_start,
            p_u64,
            mu,
            p_f,
            inv_p_f,
            &mut tw_step[..tile_cols],
        );
    }

    for (m, &w_m) in gen_m.iter().enumerate().take(s_row) {
        let half_rows = 1usize << m;
        let block_len = half_rows * tile_cols;
        if m == 0 && !psi.negacyclic {
            fill_geom_row_vec(
                kernels,
                &mut tw_row[tile_cols..(2 * tile_cols)],
                w_m,
                col_start,
                p_u64,
                mu,
                p_f,
                inv_p_f,
                &mut tw_step[..tile_cols],
            );
            continue;
        }
        fill_geom_row_vec(
            kernels,
            &mut tw_step[..tile_cols],
            w_m,
            col_start,
            p_u64,
            mu,
            p_f,
            inv_p_f,
            &mut tw_row[block_len..(block_len + tile_cols)],
        );
        let mut filled = tile_cols;
        while filled < block_len {
            let copy_n = filled.min(block_len - filled);
            tw_step.copy_within(0..copy_n, filled);
            filled += copy_n;
        }
        let step_vcfg = NttVecCfg {
            n: block_len as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tw_row.as_mut_ptr().add(block_len),
                tw_row.as_ptr(),
                tw_step.as_ptr(),
                &step_vcfg,
            );
        }
    }

    if !psi.negacyclic {
        let off = tile_cols;
        let n_mul = total_elems - off;
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tile.as_ptr().add(off),
                tw_row.as_ptr().add(off),
                &vcfg,
            );
        }
    } else {
        let vcfg = NttVecCfg {
            n: total_elems as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr(),
                tile.as_ptr(),
                tw_row.as_ptr(),
                &vcfg,
            );
        }
    }
}

fn fill_row_scales_u64<A: Allocator>(
    rows: usize,
    col_start: usize,
    psi: PsiParams,
    p_u64: u64,
    s_u64: &mut Vec<u64, A>,
) {
    let mu = barrett_mu_p50(p_u64);
    let s_row = rows.ilog2() as usize;
    if s_u64.len() < rows {
        s_u64.resize(rows, 1);
    }
    let mut s_gen = [1u64; 32];
    if s_row > 0 {
        s_gen[s_row - 1] = pow_mod_p50(psi.root_sq_u64, col_start as u64, p_u64, mu);
        for idx in (0..(s_row - 1)).rev() {
            let nxt = s_gen[idx + 1];
            s_gen[idx] = mul_mod_p50(nxt, nxt, p_u64, mu);
        }
    }
    s_u64[0] = if psi.negacyclic {
        pow_mod_p50(psi.first_u64, col_start as u64, p_u64, mu)
    } else {
        1
    };
    for (m, &gm) in s_gen.iter().enumerate().take(s_row) {
        let half = 1usize << m;
        for j in 0..half {
            s_u64[half + j] = mul_mod_p50(s_u64[j], gm, p_u64, mu);
        }
    }
}

#[inline(always)]
fn tw0_is_full(tw0: &[f64], min_len: usize) -> bool {
    (tw0.len() & 1) == 0 && tw0.len() >= min_len
}

#[inline(always)]
fn tw0_has_row_scales(tw0: &[f64], rows: usize, tile_cols: usize) -> bool {
    (tw0.len() & 1) == 1 && tw0.len() > rows * tile_cols + rows
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tw0_resolve_row_scales<'a, A: Allocator>(
    kernels: &Kernels,
    tw0: &'a [f64],
    rows: usize,
    tile_cols: usize,
    col_start: usize,
    start_r: usize,
    p_f: f64,
    inv_p_f: f64,
    tw_row: &'a mut Vec<f64, A>,
) -> Option<&'a [f64]> {
    if (tw0.len() & 1) == 1 && tile_cols > 0 && col_start.is_multiple_of(tile_cols) {
        let ct = col_start / tile_cols;
        let tile_elems = rows * tile_cols;
        let tag = tw0[tw0.len() - 1];
        if tag == 0.0 {
            let rs_off = tile_elems + ct * rows;
            if rs_off + rows < tw0.len() {
                return Some(&tw0[rs_off..(rs_off + rows)]);
            }
        } else if tag == -1.0 {
            let lo = ct & 63;
            let hi = ct >> 6;
            let lo_off = tile_elems + lo * rows;
            let hi_off = tile_elems + (64 + hi) * rows;
            if hi_off + rows < tw0.len() {
                if hi == 0 {
                    return Some(&tw0[lo_off..(lo_off + rows)]);
                }
                if lo == 0 {
                    return Some(&tw0[hi_off..(hi_off + rows)]);
                }
                if tw_row.len() < rows {
                    tw_row.resize(rows, 1.0);
                }
                for slot in tw_row.iter_mut().take(start_r) {
                    *slot = 1.0;
                }
                if rows > start_r {
                    let vcfg = NttVecCfg {
                        n: (rows - start_r) as u64,
                        p: p_f,
                        inv_p: inv_p_f,
                        sub_second: 0,
                    };
                    unsafe {
                        (kernels.active.ntt_pmul)(
                            tw_row.as_mut_ptr().add(start_r),
                            tw0.as_ptr().add(lo_off + start_r),
                            tw0.as_ptr().add(hi_off + start_r),
                            &vcfg,
                        );
                    }
                }
                return Some(&tw_row[..rows]);
            }
        }
    }
    None
}

/// Fast single-pass application of cached `tw_tile0` (`psi[r]^tc mod p` for `col_start == 0`),
/// fusing row scale `psi[r]^col_start mod p` directly inside `ntt_pmul` (`sub_second = tile_cols`).
#[allow(clippy::too_many_arguments)]
fn apply_tile_twist<A1: Allocator, A2: Allocator>(
    kernels: &Kernels,
    tile: &mut [f64],
    tw_tile0: &[f64],
    rows: usize,
    tile_cols: usize,
    col_start: usize,
    psi: PsiParams,
    p_u64: u64,
    p_f: f64,
    inv_p_f: f64,
    tw_row: &mut Vec<f64, A1>,
    s_u64: &mut Vec<u64, A2>,
) {
    if rows == 0
        || tile_cols == 0
        || (col_start == 0 && tile_cols == 1)
        || (!psi.negacyclic && rows <= 1)
    {
        return;
    }
    let total_elems = rows * tile_cols;
    let start_r = usize::from(!psi.negacyclic);
    let off = start_r * tile_cols;
    let n_mul = total_elems - off;
    if n_mul == 0 {
        return;
    }
    if tw0_is_full(tw_tile0, (col_start + tile_cols) * rows) {
        let tw_off = col_start * rows + off;
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tile.as_ptr().add(off),
                tw_tile0.as_ptr().add(tw_off),
                &vcfg,
            );
        }
        return;
    }
    if col_start == 0 {
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tile.as_ptr().add(off),
                tw_tile0.as_ptr().add(off),
                &vcfg,
            );
        }
        return;
    }
    if let Some(rs) = tw0_resolve_row_scales(
        kernels, tw_tile0, rows, tile_cols, col_start, start_r, p_f, inv_p_f, tw_row,
    ) {
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: tile_cols as u64,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tw_tile0.as_ptr().add(off),
                rs.as_ptr().add(start_r),
                &vcfg,
            );
        }
        return;
    }

    fill_row_scales_u64(rows, col_start, psi, p_u64, s_u64);
    if tw_row.len() < rows {
        tw_row.resize(rows, 1.0);
    }
    for r in start_r..rows {
        tw_row[r] = s_u64[r] as f64;
    }
    let vcfg = NttVecCfg {
        n: n_mul as u64,
        p: p_f,
        inv_p: inv_p_f,
        sub_second: tile_cols as u64,
    };
    unsafe {
        (kernels.active.ntt_pmul)(
            tile.as_mut_ptr().add(off),
            tw_tile0.as_ptr().add(off),
            tw_row.as_ptr().add(start_r),
            &vcfg,
        );
    }
}

/// Inverse tile untwist with `rows^{-1} mod p` already folded into `tw_tile0`.
/// Eliminates the separate `ZMM_SCALE4` pass inside `ntt_inv_cols` (`scale = 1.0`).
#[allow(clippy::too_many_arguments)]
fn apply_tile_twist_inv_scaled<A1: Allocator, A2: Allocator>(
    kernels: &Kernels,
    tile: &mut [f64],
    tw_tile0: &[f64],
    rows: usize,
    tile_cols: usize,
    col_start: usize,
    psi: PsiParams,
    p_u64: u64,
    p_f: f64,
    inv_p_f: f64,
    tw_row: &mut Vec<f64, A1>,
    s_u64: &mut Vec<u64, A2>,
) {
    if rows <= 1 || tile_cols == 0 {
        if psi.negacyclic && rows == 1 && tile_cols > 0 {
            apply_tile_twist(
                kernels,
                tile,
                tw_tile0,
                rows,
                tile_cols,
                col_start,
                psi,
                p_u64,
                p_f,
                inv_p_f,
                tw_row,
                s_u64,
            );
        }
        return;
    }
    let total_elems = rows * tile_cols;
    if tw0_is_full(tw_tile0, (col_start + tile_cols) * rows) {
        let tw_off = col_start * rows;
        let vcfg = NttVecCfg {
            n: total_elems as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr(),
                tile.as_ptr(),
                tw_tile0.as_ptr().add(tw_off),
                &vcfg,
            );
        }
        return;
    }
    if col_start == 0 {
        let vcfg = NttVecCfg {
            n: total_elems as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr(),
                tile.as_ptr(),
                tw_tile0.as_ptr(),
                &vcfg,
            );
        }
        return;
    }

    let start_r = usize::from(!psi.negacyclic);
    if start_r > 0 {
        let vcfg0 = NttVecCfg {
            n: tile_cols as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr(),
                tile.as_ptr(),
                tw_tile0.as_ptr(),
                &vcfg0,
            );
        }
    }
    let off = start_r * tile_cols;
    let n_mul = total_elems - off;
    if let Some(rs) = tw0_resolve_row_scales(
        kernels, tw_tile0, rows, tile_cols, col_start, start_r, p_f, inv_p_f, tw_row,
    ) {
        if n_mul > 0 {
            let vcfg = NttVecCfg {
                n: n_mul as u64,
                p: p_f,
                inv_p: inv_p_f,
                sub_second: tile_cols as u64,
            };
            unsafe {
                (kernels.active.ntt_pmul)(
                    tile.as_mut_ptr().add(off),
                    tw_tile0.as_ptr().add(off),
                    rs.as_ptr().add(start_r),
                    &vcfg,
                );
            }
        }
        return;
    }

    fill_row_scales_u64(rows, col_start, psi, p_u64, s_u64);
    if tw_row.len() < rows {
        tw_row.resize(rows, 1.0);
    }
    for r in start_r..rows {
        tw_row[r] = s_u64[r] as f64;
    }
    if n_mul > 0 {
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: tile_cols as u64,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tw_tile0.as_ptr().add(off),
                tw_row.as_ptr().add(start_r),
                &vcfg,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_tile_twist_cg4_sub<A1: Allocator, A2: Allocator>(
    kernels: &Kernels,
    tile: &mut [f64],
    tw_tile0: &[f64],
    step64: &[u64],
    rows: usize,
    tile_cols: usize,
    col_start: usize,
    sub: usize,
    psi: PsiParams,
    p_u64: u64,
    p_f: f64,
    inv_p_f: f64,
    tw_row: &mut Vec<f64, A1>,
    s_u64: &mut Vec<u64, A2>,
) {
    if rows == 0
        || tile_cols == 0
        || (col_start == 0 && tile_cols == 1)
        || (!psi.negacyclic && rows <= 1)
    {
        return;
    }
    let total_elems = rows * tile_cols;
    let start_r = usize::from(!psi.negacyclic);
    let off = start_r * tile_cols;
    let n_mul = total_elems - off;
    if n_mul == 0 {
        return;
    }
    if tw0_is_full(tw_tile0, (col_start + tile_cols) * rows) {
        let tw_off = col_start * rows + off;
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tile.as_ptr().add(off),
                tw_tile0.as_ptr().add(tw_off),
                &vcfg,
            );
        }
        return;
    }
    if col_start == 0 {
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: 0,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tile.as_ptr().add(off),
                tw_tile0.as_ptr().add(off),
                &vcfg,
            );
        }
        return;
    }
    if let Some(rs) = tw0_resolve_row_scales(
        kernels, tw_tile0, rows, tile_cols, col_start, start_r, p_f, inv_p_f, tw_row,
    ) {
        let vcfg = NttVecCfg {
            n: n_mul as u64,
            p: p_f,
            inv_p: inv_p_f,
            sub_second: tile_cols as u64,
        };
        unsafe {
            (kernels.active.ntt_pmul)(
                tile.as_mut_ptr().add(off),
                tw_tile0.as_ptr().add(off),
                rs.as_ptr().add(start_r),
                &vcfg,
            );
        }
        return;
    }

    if s_u64.len() < rows {
        s_u64.resize(rows, 1);
    }
    if sub == 0 {
        fill_row_scales_u64(rows, col_start, psi, p_u64, s_u64);
    } else if col_start == tile_cols {
        s_u64[..rows].copy_from_slice(step64);
    } else {
        let mu = barrett_mu_p50(p_u64);
        for r in start_r..rows {
            s_u64[r] = mul_mod_p50(s_u64[r], step64[r], p_u64, mu);
        }
    }
    if tw_row.len() < rows {
        tw_row.resize(rows, 1.0);
    }
    for r in start_r..rows {
        tw_row[r] = s_u64[r] as f64;
    }
    let vcfg = NttVecCfg {
        n: n_mul as u64,
        p: p_f,
        inv_p: inv_p_f,
        sub_second: tile_cols as u64,
    };
    unsafe {
        (kernels.active.ntt_pmul)(
            tile.as_mut_ptr().add(off),
            tw_tile0.as_ptr().add(off),
            tw_row.as_ptr().add(start_r),
            &vcfg,
        );
    }
}

fn compute_limbs_w0_parallel(
    kernels: &Kernels,
    sched: &Scheduler,
    limbs: &[u64],
    max_workers: usize,
) -> u64 {
    if max_workers <= 1 || limbs.len() < 16384 {
        return crate::check::compute_limbs_w0(kernels, limbs);
    }
    let chunk = 8192usize;
    let num_chunks = limbs.len().div_ceil(chunk);
    let mut chunk_w0s = vec![0u64; num_chunks];
    let cw_ptr = chunk_w0s.as_mut_ptr() as usize;
    sched.par_for_leaf_max(num_chunks, 1, max_workers, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(limbs.len());
        let w = crate::check::compute_limbs_w0(kernels, &limbs[start..end]);
        unsafe {
            *(cw_ptr as *mut u64).add(ci) = w;
        }
    });
    let step_shift = crate::refk::mod61_pow2((chunk as u64) * 3);
    let mut shift = 1u64;
    let mut total_w0 = 0u64;
    for &cw in &chunk_w0s {
        total_w0 = crate::refk::mod61_add(total_w0, crate::refk::mod61_mul(cw, shift));
        shift = crate::refk::mod61_mul(shift, step_shift);
    }
    total_w0
}

#[inline(always)]
fn mod61_mul8(x: u64) -> u64 {
    const M: u64 = (1u64 << 61) - 1;
    let s = ((x << 3) & M) + (x >> 58);
    if s >= M { s - M } else { s }
}

#[inline(always)]
fn w0_update_with_shift(w0: &mut u64, shift: u64, old_val: u64, new_val: u64) {
    if old_val == new_val {
        return;
    }
    let old_m = crate::refk::mod61_reduce(old_val as u128);
    let new_m = crate::refk::mod61_reduce(new_val as u128);
    let delta = crate::refk::mod61_sub(new_m, old_m);
    *w0 = crate::refk::mod61_add(*w0, crate::refk::mod61_mul(delta, shift));
}

#[inline]
fn stitch_signed_carry_slice(
    slice: &mut [u64],
    rel_off: usize,
    c_slice: &[u64],
    off_shift: u64,
    w0: &mut u64,
) -> i64 {
    let kw = c_slice.len();
    if kw == 0 || rel_off >= slice.len() {
        return 0;
    }
    let mut any_nz = 0u64;
    for &v in c_slice {
        any_nz |= v;
    }
    if any_nz == 0 {
        return 0;
    }
    let is_neg = (c_slice[kw - 1] as i64) < 0;
    if !is_neg {
        let mut clen = kw;
        while clen > 0 && c_slice[clen - 1] == 0 {
            clen -= 1;
        }
        let n_add = clen.min(slice.len() - rel_off);
        let mut carry = 0u64;
        let mut sh = off_shift;
        for j in 0..n_add {
            let old = slice[rel_off + j];
            let (s1, c1) = old.overflowing_add(c_slice[j]);
            let (s2, c2) = s1.overflowing_add(carry);
            w0_update_with_shift(w0, sh, old, s2);
            slice[rel_off + j] = s2;
            carry = (c1 as u64) + (c2 as u64);
            sh = mod61_mul8(sh);
        }
        let mut idx = rel_off + n_add;
        while carry > 0 && idx < slice.len() {
            let old = slice[idx];
            let (s, ov) = old.overflowing_add(carry);
            w0_update_with_shift(w0, sh, old, s);
            slice[idx] = s;
            carry = u64::from(ov);
            idx += 1;
            sh = mod61_mul8(sh);
        }
        carry as i64
    } else {
        let mut mag = [0u64; 20];
        let mut b = 0u64;
        for j in 0..kw {
            let (d1, b1) = 0u64.overflowing_sub(c_slice[j]);
            let (d2, b2) = d1.overflowing_sub(b);
            mag[j] = d2;
            b = (b1 as u64) + (b2 as u64);
        }
        let mut clen = kw;
        while clen > 0 && mag[clen - 1] == 0 {
            clen -= 1;
        }
        let n_sub = clen.min(slice.len() - rel_off);
        let mut brw = 0u64;
        let mut sh = off_shift;
        for j in 0..n_sub {
            let old = slice[rel_off + j];
            let (d1, b1) = old.overflowing_sub(mag[j]);
            let (d2, b2) = d1.overflowing_sub(brw);
            w0_update_with_shift(w0, sh, old, d2);
            slice[rel_off + j] = d2;
            brw = (b1 as u64) + (b2 as u64);
            sh = mod61_mul8(sh);
        }
        let mut idx = rel_off + n_sub;
        while brw > 0 && idx < slice.len() {
            let old = slice[idx];
            let (d, un) = old.overflowing_sub(brw);
            w0_update_with_shift(w0, sh, old, d);
            slice[idx] = d;
            brw = u64::from(un);
            idx += 1;
            sh = mod61_mul8(sh);
        }
        -(brw as i64)
    }
}

#[inline(always)]
fn w0_update_limb(w0: &mut u64, idx: usize, old_val: u64, new_val: u64) {
    if old_val == new_val {
        return;
    }
    let shift = crate::refk::mod61_pow2((idx as u64) * 3);
    let old_m = crate::refk::mod61_reduce(old_val as u128);
    let new_m = crate::refk::mod61_reduce(new_val as u128);
    let delta = crate::refk::mod61_sub(new_m, old_m);
    *w0 = crate::refk::mod61_add(*w0, crate::refk::mod61_mul(delta, shift));
}

fn fold_cyclic_carry_w0(
    out: &mut [u64],
    total_bits: usize,
    carry_words: &[u64],
    w0: &mut u64,
) {
    let out_len = out.len();
    if out_len == 0 || total_bits == 0 || carry_words.is_empty() {
        return;
    }
    let neg = (carry_words[carry_words.len() - 1] as i64) < 0;
    let mut mag_stack = [0u64; 32];
    let mut mag_vec: Vec<u64>;
    let mag: &mut [u64] = if carry_words.len() <= 32 {
        mag_stack[..carry_words.len()].copy_from_slice(carry_words);
        &mut mag_stack[..carry_words.len()]
    } else {
        mag_vec = carry_words.to_vec();
        &mut mag_vec
    };
    if neg {
        let mut b = 0u64;
        for w in mag.iter_mut() {
            let (d1, b1) = 0u64.overflowing_sub(*w);
            let (d2, b2) = d1.overflowing_sub(b);
            *w = d2;
            b = (b1 as u64) + (b2 as u64);
        }
    }
    let mut mag_len = mag.len();
    while mag_len > 0 && mag[mag_len - 1] == 0 {
        mag_len -= 1;
    }
    if mag_len == 0 {
        return;
    }

    let top_limb = (total_bits - 1) >> 6;
    let top_bits = total_bits - top_limb * 64;
    let top_mask = if top_bits == 64 {
        u64::MAX
    } else {
        (1u64 << top_bits) - 1
    };
    let limit = top_limb.min(out_len - 1);

    if !neg {
        let mut c = 0u64;
        let mut i = 0usize;
        while i <= limit {
            if i >= mag_len && c == 0 {
                if top_limb <= limit && top_bits < 64 && i <= top_limb {
                    let cur = out[top_limb];
                    let new_v = cur & top_mask;
                    w0_update_limb(w0, top_limb, cur, new_v);
                    out[top_limb] = new_v;
                    c = cur >> top_bits;
                }
                break;
            }
            let addv = if i < mag_len { mag[i] } else { 0 };
            let cur = out[i];
            let (s1, c1) = cur.overflowing_add(addv);
            let (s2, c2) = s1.overflowing_add(c);
            if i == top_limb {
                let new_v = s2 & top_mask;
                w0_update_limb(w0, i, cur, new_v);
                out[i] = new_v;
                c = if top_bits == 64 {
                    (c1 as u64) + (c2 as u64)
                } else {
                    (s2 >> top_bits) + (((c1 as u64) + (c2 as u64)) << (64 - top_bits))
                };
            } else {
                w0_update_limb(w0, i, cur, s2);
                out[i] = s2;
                c = (c1 as u64) + (c2 as u64);
            }
            i += 1;
        }
        while c > 0 {
            let mut cc = c;
            c = 0;
            for j in 0..=limit {
                let cur = out[j];
                let (s, ov) = cur.overflowing_add(cc);
                if j == top_limb {
                    let extra = if top_bits == 64 { ov as u64 } else { s >> top_bits };
                    let new_v = s & top_mask;
                    w0_update_limb(w0, j, cur, new_v);
                    out[j] = new_v;
                    c = extra;
                } else {
                    w0_update_limb(w0, j, cur, s);
                    out[j] = s;
                    cc = ov as u64;
                    if cc == 0 {
                        if top_limb <= limit && top_bits < 64 {
                            let cur_t = out[top_limb];
                            let new_t = cur_t & top_mask;
                            w0_update_limb(w0, top_limb, cur_t, new_t);
                            out[top_limb] = new_t;
                            c = cur_t >> top_bits;
                        }
                        break;
                    }
                }
            }
        }
    } else {
        let mut b = 0u64;
        let mut i = 0usize;
        while i <= limit {
            if i >= mag_len && b == 0 {
                if top_limb <= limit && top_bits < 64 && i <= top_limb {
                    let cur = out[top_limb];
                    let under = (cur >> top_bits) != 0;
                    let new_v = cur & top_mask;
                    w0_update_limb(w0, top_limb, cur, new_v);
                    out[top_limb] = new_v;
                    b = u64::from(under);
                }
                break;
            }
            let subv = if i < mag_len { mag[i] } else { 0 };
            let cur = out[i];
            let (d1, b1) = cur.overflowing_sub(subv);
            let (d2, b2) = d1.overflowing_sub(b);
            if i == top_limb {
                let under = ((b1 as u64) + (b2 as u64) > 0)
                    || (top_bits < 64 && (d2 >> top_bits) != 0);
                let new_v = d2 & top_mask;
                w0_update_limb(w0, i, cur, new_v);
                out[i] = new_v;
                b = u64::from(under);
            } else {
                w0_update_limb(w0, i, cur, d2);
                out[i] = d2;
                b = (b1 as u64) + (b2 as u64);
            }
            i += 1;
        }
        if b > 0 {
            let mut bb = 1u64;
            for j in 0..=limit {
                let cur = out[j];
                let (d, un) = cur.overflowing_sub(bb);
                let new_v = if j == top_limb { d & top_mask } else { d };
                w0_update_limb(w0, j, cur, new_v);
                out[j] = new_v;
                bb = u64::from(un);
                if bb == 0 {
                    break;
                }
            }
        }
    }

    if top_limb < out_len {
        let mut all_ones = true;
        for (i, &val) in out.iter().enumerate().take(top_limb + 1) {
            let expected = if i == top_limb { top_mask } else { u64::MAX };
            if val != expected {
                all_ones = false;
                break;
            }
        }
        if all_ones {
            for (i, slot) in out.iter_mut().enumerate().take(top_limb + 1) {
                let old = *slot;
                w0_update_limb(w0, i, old, 0);
                *slot = 0;
            }
        }
    }
}

/// Execute a 1D row forward transform of length `len`, adding column sub-sweeps recursively when
/// `extra_sweeps > 0` (Section 5: "takes one more column sweep in each direction for each further
/// level of the grid").
fn row_fwd_multi_sweep(
    kernels: &Kernels,
    buf: &mut [f64],
    tw: &[f64],
    pr_fwd: &NttPrime,
    extra_sweeps: u32,
) {
    let len = buf.len();
    if extra_sweeps == 0 || len < 4 {
        unsafe {
            call_ntt_fwd(kernels, buf.as_mut_ptr(), tw.as_ptr(), len as u64, pr_fwd);
        }
        return;
    }
    let log_len = len.ilog2();
    let sub_rows = 1usize << (log_len / 2).max(1);
    let sub_cols = len / sub_rows;
    let sub_tile = sub_cols.min(16).max(1);
    let mcfg = MoveCfg {
        rows: sub_rows as u64,
        cols: sub_tile as u64,
        outer_stride_bytes: (sub_cols * 8) as u64,
        packed: 0,
        use_nt: 0,
    };
    let shape = [sub_rows as u64, sub_tile as u64];
    let mut tile = vec![0.0f64; sub_rows * sub_tile];
    let mut col_scratch = Vec::new_in(ScratchAlloc);
    for c0 in (0..sub_cols).step_by(sub_tile) {
        unsafe {
            (kernels.active.gather)(
                tile.as_mut_ptr(),
                buf.as_ptr().add(c0) as *const u8,
                &mcfg,
            );
            call_ntt_fwd_cols(
                kernels,
                tile.as_mut_ptr(),
                tw.as_ptr(),
                &shape,
                pr_fwd,
                &mut col_scratch,
            );
            (kernels.active.scatter)(
                buf.as_mut_ptr().add(c0) as *mut u8,
                tile.as_ptr(),
                &mcfg,
            );
        }
    }
    if kernels.count_work {
        kernels
            .stats
            .bytes_moved
            .fetch_add((len * 16) as u64, Ordering::Relaxed);
    }
    let mut sub_tw = vec![0.0f64; sub_cols.max(1)];
    for sr in 0..sub_rows {
        let mut b = 1usize;
        while b < sub_cols {
            let src_base = sub_rows * b + sr * b;
            sub_tw[b..(2 * b)].copy_from_slice(&tw[src_base..(src_base + b)]);
            b <<= 1;
        }
        row_fwd_multi_sweep(
            kernels,
            &mut buf[(sr * sub_cols)..((sr + 1) * sub_cols)],
            &sub_tw,
            pr_fwd,
            extra_sweeps - 1,
        );
    }
}

fn row_inv_multi_sweep(
    kernels: &Kernels,
    buf: &mut [f64],
    inv_tw: &[f64],
    p_u64: u64,
    scale_total: u64,
    extra_sweeps: u32,
) {
    let len = buf.len();
    if extra_sweeps == 0 || len < 4 {
        let pr_inv = NttPrime::new(p_u64, scale_total);
        unsafe {
            call_ntt_inv(kernels, buf.as_mut_ptr(), inv_tw.as_ptr(), len as u64, &pr_inv);
        }
        return;
    }
    let log_len = len.ilog2();
    let sub_rows = 1usize << (log_len / 2).max(1);
    let sub_cols = len / sub_rows;
    let mut sub_inv_tw = vec![0.0f64; sub_cols.max(1)];
    for sr in 0..sub_rows {
        let mut b = 1usize;
        while b < sub_cols {
            let src_base = sub_rows * b + sr * b;
            sub_inv_tw[b..(2 * b)].copy_from_slice(&inv_tw[src_base..(src_base + b)]);
            b <<= 1;
        }
        row_inv_multi_sweep(
            kernels,
            &mut buf[(sr * sub_cols)..((sr + 1) * sub_cols)],
            &sub_inv_tw,
            p_u64,
            1,
            extra_sweeps - 1,
        );
    }
    let sub_tile = sub_cols.min(16).max(1);
    let mcfg = MoveCfg {
        rows: sub_rows as u64,
        cols: sub_tile as u64,
        outer_stride_bytes: (sub_cols * 8) as u64,
        packed: 0,
        use_nt: 0,
    };
    let shape = [sub_rows as u64, sub_tile as u64];
    let pr_col_inv = NttPrime::new(p_u64, scale_total);
    let mut tile = vec![0.0f64; sub_rows * sub_tile];
    let mut col_scratch = Vec::new_in(ScratchAlloc);
    for c0 in (0..sub_cols).step_by(sub_tile) {
        unsafe {
            (kernels.active.gather)(
                tile.as_mut_ptr(),
                buf.as_ptr().add(c0) as *const u8,
                &mcfg,
            );
            call_ntt_inv_cols(
                kernels,
                tile.as_mut_ptr(),
                inv_tw.as_ptr(),
                &shape,
                &pr_col_inv,
                &mut col_scratch,
            );
            (kernels.active.scatter)(
                buf.as_mut_ptr().add(c0) as *mut u8,
                tile.as_ptr(),
                &mcfg,
            );
        }
    }
    if kernels.count_work {
        kernels
            .stats
            .bytes_moved
            .fetch_add((len * 16) as u64, Ordering::Relaxed);
    }
}

#[inline]
fn nz_elems_1d(in_limbs: usize, coeff_bits: u32, n: usize) -> usize {
    let total_bits = (in_limbs as u64) * 64;
    if total_bits == 0 || n == 0 {
        return 0;
    }
    let raw = total_bits.div_ceil(coeff_bits as u64) as usize;
    raw.next_multiple_of(32).min(n)
}

#[inline]
fn tile_nz_elems(
    in_limbs: usize,
    coeff_bits: u32,
    rows: usize,
    cols: usize,
    col_start: usize,
    tile_cols: usize,
) -> usize {
    let total_bits = (in_limbs as u64) * 64;
    let b = coeff_bits as u64;
    let first_bit = (col_start as u64) * b;
    if total_bits <= first_bit || rows == 0 || tile_cols == 0 {
        return 0;
    }
    let row_step_bits = (cols as u64) * b;
    let nz_rows = (((total_bits - 1 - first_bit) / row_step_bits) as usize + 1).min(rows);
    (nz_rows * tile_cols).next_multiple_of(32).min(rows * tile_cols)
}

/// Forward transform (1D when `rows == 1`, or Sweep 1 column transform when `rows > 1`).
pub fn ntt_forward(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    limbs: &[u64],
    spec: NttSpec,
) -> Result<NttCachedFwd, crate::store::StoreError> {
    ntt_forward_inner(kernels, cache, sched, store, limbs, spec, false)
}

/// Forward transform pre-completing row `ntt_fwd` when in RAM (`rows > 1 && !out_of_core`) so
/// operands reused `>= 2` times never repeat row transforms in Sweep 2.
pub fn ntt_forward_reusable(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    limbs: &[u64],
    spec: NttSpec,
) -> Result<NttCachedFwd, crate::store::StoreError> {
    ntt_forward_inner(kernels, cache, sched, store, limbs, spec, true)
}

fn ntt_forward_inner(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    limbs: &[u64],
    spec: NttSpec,
    reusable: bool,
) -> Result<NttCachedFwd, crate::store::StoreError> {
    debug_assert_eq!(spec.rows * spec.cols, spec.n);
    debug_assert_ne!(spec.coeff_bits % 61, 0);
    let _pt = PathTimer::start(if reusable { "fwd-reusable" } else { "fwd" }, &spec, &[limbs.len()]);

    let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
    let k = spec.k_primes;

    if spec.out_of_core
        && let Some(st) = store
    {
        return ntt_forward_out_of_core(kernels, cache, sched, st, limbs, spec, crt);
    }

    if spec.rows == 1 {
        let n = spec.n;
        let row_bytes = spec.outer_row_bytes();
        let t_al = std::time::Instant::now();
        let mut bufs = cache.take_byte_bufs_uninit(k, row_bytes);
        NTT_TRACE_ALLOC_NS.fetch_add(t_al.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        let bufs_ptr = bufs.as_mut_ptr() as usize;
        let t_f1 = std::time::Instant::now();

        let nz_1d = nz_elems_1d(limbs.len(), spec.coeff_bits, n);
        let p_indep_1d = !spec.packed
            && k >= 2
            && spec.coeff_bits <= 52
            && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
        let p_chunk_1d = !spec.packed
            && !p_indep_1d
            && k >= 2
            && (53..=384).contains(&spec.coeff_bits)
            && crt.primes_u64[k - 1] > (1u64 << 48);
        if p_indep_1d {
            let t0 = cache.get_twiddles_ref(crt.primes_u64[0], n, spec.negacyclic);
            let pr0 = &t0.pr_fwd;
            let f0_ptr = bufs[0].as_mut_ptr() as *mut f64;
            unsafe {
                if nz_1d > 0 {
                    let scfg0 = NttSplitCfg {
                        in_limbs: limbs.len() as u64,
                        start_coeff: 0,
                        count: nz_1d as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: pr0.p,
                        inv_p: pr0.inv_p,
                    };
                    (kernels.active.ntt_split)(f0_ptr, limbs.as_ptr(), &scfg0);
                }
                if nz_1d < n {
                    std::ptr::write_bytes(f0_ptr.add(nz_1d), 0, n - nz_1d);
                }
                for b in bufs.iter_mut().take(k).skip(1) {
                    let dst_f = b.as_mut_ptr() as *mut f64;
                    if nz_1d > 0 {
                        std::ptr::copy_nonoverlapping(f0_ptr, dst_f, nz_1d);
                    }
                    if nz_1d < n {
                        std::ptr::write_bytes(dst_f.add(nz_1d), 0, n - nz_1d);
                    }
                }
            }
        } else if p_chunk_1d {
            let num_ch = (spec.coeff_bits as usize).div_ceil(48);
            let blk_cap = 1024usize.min(n).max(1);
            let ch_bytes_len =
                (num_ch * crate::refk::ntt_planar_stride(blk_cap) * 8).max(64);
            let mut ch_buf = AlignedBytes::with_capacity(ch_bytes_len);
            let ch_ptr = ch_buf.as_mut_ptr() as *mut f64;
            let mut prs = [NttPrime::new(2, 1); 18];
            for (pi, slot) in prs.iter_mut().enumerate().take(k) {
                *slot = cache
                    .get_twiddles_ref(crt.primes_u64[pi], n, spec.negacyclic)
                    .pr_fwd;
            }
            let mut pos = 0usize;
            while pos < nz_1d {
                let cnt = (nz_1d - pos).min(blk_cap);
                let ext_cfg = NttSplitCfg {
                    in_limbs: limbs.len() as u64,
                    start_coeff: pos as u64,
                    count: cnt as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: 0.0,
                    inv_p: 0.0,
                };
                unsafe {
                    (kernels.active.ntt_split)(ch_ptr, limbs.as_ptr(), &ext_cfg);
                }
                for (pi, b_dst) in bufs.iter_mut().enumerate().take(k) {
                    let red_cfg = NttSplitCfg {
                        in_limbs: u64::MAX,
                        start_coeff: 0,
                        count: cnt as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: prs[pi].p,
                        inv_p: prs[pi].inv_p,
                    };
                    unsafe {
                        let dst_f = (b_dst.as_mut_ptr() as *mut f64).add(pos);
                        (kernels.active.ntt_split)(dst_f, ch_ptr as *const u64, &red_cfg);
                    }
                }
                pos += cnt;
            }
            if nz_1d < n {
                for b_dst in bufs.iter_mut().take(k) {
                    unsafe {
                        std::ptr::write_bytes(
                            (b_dst.as_mut_ptr() as *mut f64).add(nz_1d),
                            0,
                            n - nz_1d,
                        );
                    }
                }
            }
        }

        if spec.par_workers <= 1 && !spec.packed {
            for pi in 0..k {
                let p_u64 = crt.primes_u64[pi];
                let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
                let pr = &tables.pr_fwd;
                let f_ptr = bufs[pi].as_mut_ptr() as *mut f64;
                unsafe {
                    if !p_indep_1d && !p_chunk_1d {
                        if nz_1d > 0 {
                            let scfg = NttSplitCfg {
                                in_limbs: limbs.len() as u64,
                                start_coeff: 0,
                                count: nz_1d as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: 0,
                                cols_per_row: 0,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            (kernels.active.ntt_split)(f_ptr, limbs.as_ptr(), &scfg);
                        }
                        if nz_1d < n {
                            std::ptr::write_bytes(f_ptr.add(nz_1d), 0, n - nz_1d);
                        }
                    }
                    call_ntt_fwd(kernels, f_ptr, tables.tw.as_ptr(), n as u64, pr);
                }
                if kernels.count_work && n > 1 {
                    kernels.stats.ntt_butterflies.fetch_add(
                        ((n as u64) / 2) * (n.ilog2() as u64),
                        Ordering::Relaxed,
                    );
                    kernels
                        .stats
                        .bytes_moved
                        .fetch_add((limbs.len() * 8 + row_bytes) as u64, Ordering::Relaxed);
                }
            }
        } else {
            sched.par_for_init_max(
                k,
                1,
                spec.par_workers,
                ScratchLease::acquire,
                |lease, pi| {
                    let p_u64 = crt.primes_u64[pi];
                    let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
                    let pr = &tables.pr_fwd;
                    let scfg = NttSplitCfg {
                        in_limbs: limbs.len() as u64,
                        start_coeff: 0,
                        count: nz_1d as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: pr.p,
                        inv_p: pr.inv_p,
                    };
                    let dst_bytes = unsafe { &mut *(bufs_ptr as *mut AlignedBytes).add(pi) };
                    if !spec.packed {
                        let f_ptr = dst_bytes.as_mut_ptr() as *mut f64;
                        unsafe {
                            if !p_indep_1d && !p_chunk_1d {
                                if nz_1d > 0 {
                                    (kernels.active.ntt_split)(f_ptr, limbs.as_ptr(), &scfg);
                                }
                                if nz_1d < n {
                                    std::ptr::write_bytes(f_ptr.add(nz_1d), 0, n - nz_1d);
                                }
                            }
                            call_ntt_fwd(kernels, f_ptr, tables.tw.as_ptr(), n as u64, pr);
                        }
                    } else {
                        let tile = &mut lease.get_mut().tile;
                        resize_f64_buf(tile, n);
                        let mcfg = MoveCfg {
                            rows: 1,
                            cols: n as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: 1,
                            use_nt: u64::from(spec.use_nt),
                        };
                        unsafe {
                            if nz_1d > 0 {
                                (kernels.active.ntt_split)(
                                    tile.as_mut_ptr(),
                                    limbs.as_ptr(),
                                    &scfg,
                                );
                            }
                            if nz_1d < n {
                                std::ptr::write_bytes(tile.as_mut_ptr().add(nz_1d), 0, n - nz_1d);
                            }
                            call_ntt_fwd(kernels, 
                                tile.as_mut_ptr(),
                                tables.tw.as_ptr(),
                                n as u64,
                                pr,
                            );
                            (kernels.active.scatter)(dst_bytes.as_mut_ptr(), tile.as_ptr(), &mcfg);
                        }
                    }
                    if kernels.count_work && n > 1 {
                        kernels.stats.ntt_butterflies.fetch_add(
                            ((n as u64) / 2) * (n.ilog2() as u64),
                            Ordering::Relaxed,
                        );
                        kernels
                            .stats
                            .bytes_moved
                            .fetch_add((limbs.len() * 8 + row_bytes) as u64, Ordering::Relaxed);
                    }
                },
            );
        }
        for b in &mut bufs {
            unsafe {
                b.set_len(row_bytes);
            }
        }
        NTT_TRACE_1D_FWD_NS.fetch_add(t_f1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        return finish_cached_fwd(spec, true, bufs, row_bytes, cache, store, kernels);
    }

    // 3-Sweep Grid Layout — Sweep 1: Column sweep (`ntt_split` + `ntt_fwd_cols` + `scatter`).
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_cols = spec.tile_cols.min(cols).max(1);
    debug_assert_eq!(cols % tile_cols, 0);
    let num_col_tiles = cols / tile_cols;
    let row_bytes = spec.outer_row_bytes();

    let t_al = std::time::Instant::now();
    let total_grid_bytes = rows * row_bytes;
    let mut bufs = cache.take_byte_bufs_committed(k, total_grid_bytes, sched, spec.par_workers);
    NTT_TRACE_ALLOC_NS.fetch_add(t_al.elapsed().as_nanos() as u64, Ordering::Relaxed);
    NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    let bufs_ptr = bufs.as_mut_ptr() as usize;

    let col_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        rows,
        spec.negacyclic,
    );
    let psi_fwd: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                spec.negacyclic,
                false,
            )
        })
        .collect();

    let tw_tile0_fwd = cache.get_tile0_twiddles_par(
        kernels,
        sched,
        spec.par_workers,
        &crt.primes_u64,
        &col_tables,
        k,
        spec.n,
        rows,
        tile_cols,
        spec.negacyclic,
        false,
    );
    let prime_col_fwd: Vec<NttPrime> = (0..k).map(|pi| col_tables[pi].pr_fwd).collect();

    // Prime-by-prime Sweep 1 + Row `ntt_fwd` fusion when `reusable` so cached transforms
    // always return `rows_transformed = true` without redundant row passes on shared reuse.
    let unpacked_row_bytes = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
    let unpacked_prime_bytes = rows * unpacked_row_bytes;
    let est_conc = sched.num_workers().div_ceil(spec.par_workers.max(1)).max(1);
    let my_l3_cap = cache.l3_prime_cap_bytes() / est_conc;
    if reusable
        && spec.extra_sweeps == 0
        && num_col_tiles >= spec.par_workers.max(1)
    {
        let dram_fwd = unpacked_prime_bytes > my_l3_cap / 2;
        let eff_tc = if rows >= 128 && cols.is_multiple_of(32) {
            tile_cols.min(32)
        } else {
            tile_cols
        };
        let eff_tw_tile0: Vec<Arc<Vec<f64>>> = if eff_tc == tile_cols {
            tw_tile0_fwd.clone()
        } else {
            cache.get_tile0_twiddles_par(
                kernels,
                sched,
                spec.par_workers,
                &crt.primes_u64,
                &col_tables,
                k,
                spec.n,
                rows,
                eff_tc,
                spec.negacyclic,
                false,
            )
        };
        let row0_tables = cache.get_twiddles_refs_par(
            sched,
            spec.par_workers,
            &crt.primes_u64,
            k,
            cols,
            false,
        );
        let pack_sw1 = can_pack_sweep1(&spec, eff_tc);
        let need_tmp = (spec.packed || dram_fwd) && !pack_sw1;
        let mut tmp_unpacked = if need_tmp {
            cache.take_byte_bufs_transient_committed(1, unpacked_prime_bytes, sched, spec.par_workers)
        } else {
            Vec::new()
        };
        let tmp_unpacked_ptr = if need_tmp {
            tmp_unpacked[0].as_mut_ptr() as usize
        } else {
            0usize
        };
        let sw1_nt = u64::from(spec.use_nt && (dram_fwd || k * total_grid_bytes > my_l3_cap));
        let row_chunk = (rows / (spec.par_workers.max(1) * 8)).clamp(1, sw2_chunk_cap(cols));
        let sw1_once_packed = pack_sw1 && sweep1_groups_256(&spec, eff_tc);
        let sw1_once_unpacked = !sw1_once_packed
            && !need_tmp
            && !spec.packed
            && (cols / eff_tc) >= spec.par_workers.max(1);
        if sw1_once_packed || sw1_once_unpacked {
            let bases: Vec<usize> = bufs.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &eff_tw_tile0);
            if sw1_once_packed {
                ntt_sweep1_all_primes(
                    kernels,
                    cache,
                    sched,
                    &[(limbs, &bases, row_bytes)],
                    &spec,
                    eff_tc,
                    &primes,
                    sw1_nt,
                );
            } else {
                ntt_sweep1_all_primes_unpacked(
                    kernels,
                    cache,
                    sched,
                    &[(limbs, &bases, row_bytes)],
                    &spec,
                    eff_tc,
                    &primes,
                    sw1_nt,
                );
            }
            let t_sw2_f = std::time::Instant::now();
            let sw2_w = sw2_workers(sched, &spec);
            let chunk_cap = sw2_chunk_cap(cols);
            let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
            if pack_sw1 {
                let out_nt = ROW_SWEEP_STREAMS;
                sched.par_for_init_max(
                    k * rows,
                    all_chunk,
                    sw2_w,
                    || {
                        let mut lease = ScratchLease::acquire();
                        resize_f64_buf(&mut lease.get_mut().ra1, cols);
                        lease
                    },
                    |lease, task| {
                        let (pi, r) = (task / rows, task % rows);
                        let sc = lease.get_mut();
                        let pr = prime_col_fwd[pi];
                        let row_tw_ptr = row0_tables[pi].tw.as_ptr();
                        let mcfg_pack = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: 1,
                            use_nt: out_nt,
                        };
                        unsafe {
                            let dst_ptr = (bases[pi] as *mut u8).add(r * row_bytes);
                            (kernels.active.gather)(
                                sc.ra1.as_mut_ptr(),
                                dst_ptr as *const u8,
                                &mcfg_pack,
                            );
                            call_ntt_fwd(kernels, 
                                sc.ra1.as_mut_ptr(),
                                row_tw_ptr,
                                cols as u64,
                                &pr,
                            );
                            (kernels.active.scatter)(dst_ptr, sc.ra1.as_ptr(), &mcfg_pack);
                        }
                    },
                );
            } else {
                sched.par_for_leaf_max(k * rows, all_chunk, sw2_w, |task| {
                    let (pi, r) = (task / rows, task % rows);
                    let pr = prime_col_fwd[pi];
                    let row_tw_ptr = row0_tables[pi].tw.as_ptr();
                    unsafe {
                        let row_ptr = (bases[pi] as *mut u8).add(r * row_bytes) as *mut f64;
                        call_ntt_fwd(kernels, row_ptr, row_tw_ptr, cols as u64, &pr);
                    }
                });
            }
            for b in &mut bufs {
                unsafe {
                    b.set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2_f.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        } else {
            for pi in 0..k {
                let p_u64 = crt.primes_u64[pi];
                let pr = prime_col_fwd[pi];
                let dst_stride = if need_tmp {
                    unpacked_row_bytes
                } else {
                    row_bytes
                };
                let sw1_dst = if need_tmp {
                    tmp_unpacked_ptr
                } else {
                    bufs[pi].as_mut_ptr() as usize
                };
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &[(limbs, sw1_dst, dst_stride)],
                    &spec,
                    eff_tc,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &eff_tw_tile0[pi],
                    sw1_nt,
                    pack_sw1,
                );
                let t_sw2_f = std::time::Instant::now();
                let row_tw_ptr = row0_tables[pi].tw.as_ptr() as usize;
                if pack_sw1 || dram_fwd {
                    let out_nt = ROW_SWEEP_STREAMS;
                    sched.par_for_init_max(
                        rows,
                        row_chunk,
                        spec.par_workers,
                        || {
                            let mut lease = ScratchLease::acquire();
                            resize_f64_buf(&mut lease.get_mut().ra1, cols);
                            lease
                        },
                        |lease, r| {
                            let sc = lease.get_mut();
                            let mcfg_pack = MoveCfg {
                                rows: 1,
                                cols: cols as u64,
                                outer_stride_bytes: row_bytes as u64,
                                packed: u64::from(spec.packed),
                                use_nt: out_nt,
                            };
                            unsafe {
                                let dst_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(r * row_bytes);
                                if pack_sw1 {
                                    (kernels.active.gather)(
                                        sc.ra1.as_mut_ptr(),
                                        dst_ptr as *const u8,
                                        &mcfg_pack,
                                    );
                                } else {
                                    let src_row = (tmp_unpacked_ptr as *const u8)
                                        .add(r * unpacked_row_bytes)
                                        as *const f64;
                                    std::ptr::copy_nonoverlapping(src_row, sc.ra1.as_mut_ptr(), cols);
                                }
                                call_ntt_fwd(kernels, 
                                    sc.ra1.as_mut_ptr(),
                                    row_tw_ptr as *const f64,
                                    cols as u64,
                                    &pr,
                                );
                                (kernels.active.scatter)(dst_ptr, sc.ra1.as_ptr(), &mcfg_pack);
                            }
                        },
                    );
                } else {
                    sched.par_for_leaf_max(rows, row_chunk, spec.par_workers, |r| {
                        let row_off = r * row_bytes;
                        unsafe {
                            if spec.packed {
                                let tmp_row_ptr = (tmp_unpacked_ptr as *mut u8)
                                    .add(r * unpacked_row_bytes)
                                    as *mut f64;
                                call_ntt_fwd(kernels, 
                                    tmp_row_ptr,
                                    row_tw_ptr as *const f64,
                                    cols as u64,
                                    &pr,
                                );
                                let mcfg_pack = MoveCfg {
                                    rows: 1,
                                    cols: cols as u64,
                                    outer_stride_bytes: row_bytes as u64,
                                    packed: 1,
                                    use_nt: u64::from(
                                        spec.use_nt
                                            && k * total_grid_bytes + unpacked_prime_bytes > my_l3_cap,
                                    ),
                                };
                                let dst_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off);
                                (kernels.active.scatter)(dst_ptr, tmp_row_ptr as *const f64, &mcfg_pack);
                            } else {
                                let row_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off) as *mut f64;
                                call_ntt_fwd(kernels, 
                                    row_ptr,
                                    row_tw_ptr as *const f64,
                                    cols as u64,
                                    &pr,
                                );
                            }
                        }
                    });
                }
                unsafe {
                    bufs[pi].set_len(total_grid_bytes);
                }
                NTT_TRACE_SW2_NS.fetch_add(t_sw2_f.elapsed().as_nanos() as u64, Ordering::Relaxed);
                NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
            }
        }
        if need_tmp {
            unsafe {
                tmp_unpacked[0].set_len(unpacked_prime_bytes);
            }
            cache.put_byte_bufs(tmp_unpacked);
        }
        if kernels.count_work {
            let bf_tot =
                ((spec.n as u64) / 2) * ((rows.ilog2() + cols.ilog2()) as u64) * (k as u64);
            kernels
                .stats
                .ntt_butterflies
                .fetch_add(bf_tot, Ordering::Relaxed);
            kernels
                .stats
                .bytes_moved
                .fetch_add((3 * k * rows * row_bytes) as u64, Ordering::Relaxed);
        }
        return finish_cached_fwd(spec, true, bufs, total_grid_bytes, cache, store, kernels);
    }

    let p_indep_split =
        spec.coeff_bits <= 52 && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
    let p_chunk_split = !p_indep_split
        && k >= 2
        && (53..=384).contains(&spec.coeff_bits)
        && crt.primes_u64[k - 1] > (1u64 << 48);
    let num_split_chunks = (spec.coeff_bits as usize).div_ceil(48);
    let tile_elems = rows * tile_cols;
    let pw = spec.par_workers.max(1);
    let sw1_use_nt = spec.use_nt && 2 * k * total_grid_bytes > my_l3_cap;
    let sw1_nt_flag = u64::from(sw1_use_nt);
    let eff_tc_nr = if rows >= 128 && cols.is_multiple_of(32) {
        tile_cols.min(32)
    } else {
        tile_cols
    };
    let used_all_primes_sw1 = !reusable
        && spec.extra_sweeps == 0
        && !spec.out_of_core
        && spec.packed
        && sweep1_groups_256(&spec, eff_tc_nr);
    let t_sw1 = std::time::Instant::now();
    if used_all_primes_sw1 {
        let eff_tw_tile0: Vec<Arc<Vec<f64>>> = if eff_tc_nr == tile_cols {
            tw_tile0_fwd.clone()
        } else {
            cache.get_tile0_twiddles_par(
                kernels,
                sched,
                spec.par_workers,
                &crt.primes_u64,
                &col_tables,
                k,
                spec.n,
                rows,
                eff_tc_nr,
                spec.negacyclic,
                false,
            )
        };
        let bases: Vec<usize> = bufs.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
        let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &eff_tw_tile0);
        ntt_sweep1_all_primes(
            kernels,
            cache,
            sched,
            &[(limbs, &bases, row_bytes)],
            &spec,
            eff_tc_nr,
            &primes,
            sw1_nt_flag,
        );
    } else if spec.packed
        && rows >= 1024
        && rows <= 4096
        && tile_cols == 64
        && num_col_tiles.is_multiple_of(4)
        && (num_col_tiles / 4) >= pw
    {
        let num_groups = num_col_tiles / 4;
        sched.par_for_init_max(
            num_groups,
            1,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, tile_elems);
                lease
            },
            |lease, cg4| {
                let sc = lease.get_mut();
                let shape = [rows as u64, 64];
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: 64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: 1,
                    use_nt: sw1_nt_flag,
                };
                for sub in 0..4 {
                    let ct = cg4 * 4 + sub;
                    let col_start = ct * 64;
                    let col_byte_off = (col_start / 8) * 50;
                    let nz = tile_nz_elems(
                        limbs.len(),
                        spec.coeff_bits,
                        rows,
                        cols,
                        col_start,
                        64,
                    );
                    if nz == 0 {
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                            for pi in 0..k {
                                let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(col_byte_off);
                                (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                            }
                        }
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        continue;
                    }
                    let use_sub_chunks = p_chunk_split && nz >= 32;
                    if p_indep_split {
                        let t_sp0 = std::time::Instant::now();
                        resize_f64_buf(&mut sc.tw_step, nz);
                        let scfg0 = NttSplitCfg {
                            in_limbs: limbs.len() as u64,
                            start_coeff: col_start as u64,
                            count: nz as u64,
                            coeff_bits: spec.coeff_bits as u64,
                            col_stride: cols as u64,
                            cols_per_row: 64,
                            p: prime_col_fwd[0].p,
                            inv_p: prime_col_fwd[0].inv_p,
                        };
                        unsafe {
                            (kernels.active.ntt_split)(
                                sc.tw_step.as_mut_ptr(),
                                limbs.as_ptr(),
                                &scfg0,
                            );
                        }
                        NTT_SW1_SPLIT_NS
                            .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    } else if use_sub_chunks {
                        let t_sp0 = std::time::Instant::now();
                        resize_f64_buf(
                            &mut sc.tw_step,
                            num_split_chunks * crate::refk::ntt_planar_stride(nz),
                        );
                        let ext_cfg = NttSplitCfg {
                            in_limbs: limbs.len() as u64,
                            start_coeff: col_start as u64,
                            count: nz as u64,
                            coeff_bits: spec.coeff_bits as u64,
                            col_stride: cols as u64,
                            cols_per_row: 64,
                            p: 0.0,
                            inv_p: 0.0,
                        };
                        unsafe {
                            (kernels.active.ntt_split)(
                                sc.tw_step.as_mut_ptr(),
                                limbs.as_ptr(),
                                &ext_cfg,
                            );
                        }
                        NTT_SW1_SPLIT_NS
                            .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    for pi in 0..k {
                        let p_u64 = crt.primes_u64[pi];
                        let pr = prime_col_fwd[pi];
                        let tw_col_ptr = col_tables[pi].tw.as_ptr();
                        let tw_t0 = &tw_tile0_fwd[pi];
                        let psi = psi_fwd[pi];
                        let t_sp0 = std::time::Instant::now();
                        if p_indep_split {
                            sc.tile[..nz].copy_from_slice(&sc.tw_step[..nz]);
                            if nz < tile_elems {
                                unsafe {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        } else if use_sub_chunks {
                            let red_cfg = NttSplitCfg {
                                in_limbs: u64::MAX,
                                start_coeff: 0,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: 0,
                                cols_per_row: 0,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    sc.tw_step.as_ptr() as *const u64,
                                    &red_cfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        } else {
                            let scfg = NttSplitCfg {
                                in_limbs: limbs.len() as u64,
                                start_coeff: col_start as u64,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: 64,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    limbs.as_ptr(),
                                    &scfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        }
                        let t_co0 = std::time::Instant::now();
                        unsafe {
                            call_ntt_fwd_cols(
                                kernels,
                                sc.tile.as_mut_ptr(),
                                tw_col_ptr,
                                &shape,
                                &pr,
                                &mut sc.col_scratch,
                            );
                        }
                        apply_tile_twist(
                            kernels,
                            &mut sc.tile[..tile_elems],
                            tw_t0,
                            rows,
                            64,
                            col_start,
                            psi,
                            p_u64,
                            pr.p,
                            pr.inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(col_byte_off);
                            (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                        }
                        if !p_indep_split {
                            NTT_SW1_SPLIT_NS.fetch_add(
                                t_co0.duration_since(t_sp0).as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                        }
                        NTT_SW1_COLS_NS.fetch_add(
                            t_sc0.duration_since(t_co0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            },
        );
    } else if spec.packed
        && rows <= 2048
        && (tile_cols == 32 || tile_cols == 64)
        && cols >= 256
        && cols.is_multiple_of(256)
        && (cols / 256) >= pw
        && 256usize
            .saturating_mul(rows)
            .saturating_mul(8)
            .saturating_mul(pw)
            <= cache.l3_prime_cap_bytes()
    {
        let gt = 256 / tile_cols;
        let num_groups = cols / 256;
        let strip_max = 32usize.min(rows);
        let tile_pl_stride = crate::refk::ntt_planar_stride(tile_elems);
        let use_cg_chunks = p_chunk_split
            && rows <= 2048
            && gt
                .saturating_mul(num_split_chunks)
                .saturating_mul(tile_pl_stride)
                .saturating_mul(8)
                .saturating_mul(pw)
                <= cache.l3_prime_cap_bytes() / 2;
        let step_tc_fwd: Vec<Vec<u64>> = (0..k)
            .map(|pi| {
                let mut s = Vec::new();
                if !tw0_is_full(&tw_tile0_fwd[pi], cols * rows)
                    && !tw0_has_row_scales(&tw_tile0_fwd[pi], rows, tile_cols)
                {
                    fill_row_scales_u64(rows, tile_cols, psi_fwd[pi], crt.primes_u64[pi], &mut s);
                }
                s
            })
            .collect();
        sched.par_for_init_max(
            num_groups,
            1,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, gt * tile_elems);
                if p_indep_split {
                    resize_f64_buf(&mut sc.tw_step, gt * tile_elems);
                } else if use_cg_chunks {
                    resize_f64_buf(&mut sc.tw_step, gt * num_split_chunks * tile_pl_stride);
                }
                resize_f64_buf(&mut sc.ra1, strip_max * 256);
                lease
            },
            |lease, cg| {
                let sc = lease.get_mut();
                let shape = [rows as u64, tile_cols as u64];
                let group_byte_off = cg * 1600;
                let mut nz_arr = [0usize; 8];
                let mut any_nz = false;
                for sub in 0..gt {
                    let nz = tile_nz_elems(
                        limbs.len(),
                        spec.coeff_bits,
                        rows,
                        cols,
                        (cg * gt + sub) * tile_cols,
                        tile_cols,
                    );
                    nz_arr[sub] = nz;
                    if nz > 0 {
                        any_nz = true;
                    }
                }
                if !any_nz {
                    let t_sc0 = std::time::Instant::now();
                    unsafe {
                        std::ptr::write_bytes(sc.ra1.as_mut_ptr(), 0, strip_max * 256);
                        for pi in 0..k {
                            let base_ptr =
                                (*(bufs_ptr as *mut AlignedBytes).add(pi)).as_mut_ptr();
                            if sw1_nt_flag != 0 {
                                let mut r0 = 0usize;
                                while r0 < rows {
                                    let sr = (rows - r0).min(strip_max);
                                    let mcfg_strip = MoveCfg {
                                        rows: sr as u64,
                                        cols: 256,
                                        outer_stride_bytes: row_bytes as u64,
                                        packed: 1,
                                        use_nt: sw1_nt_flag,
                                    };
                                    (kernels.active.scatter)(
                                        base_ptr.add(r0 * row_bytes + group_byte_off),
                                        sc.ra1.as_ptr(),
                                        &mcfg_strip,
                                    );
                                    r0 += sr;
                                }
                            } else {
                                for r in 0..rows {
                                    std::ptr::write_bytes(
                                        base_ptr.add(r * row_bytes + group_byte_off),
                                        0,
                                        1600,
                                    );
                                }
                            }
                        }
                    }
                    NTT_SW1_SCAT_NS
                        .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    return;
                }
                if p_indep_split {
                    let t_sp0 = std::time::Instant::now();
                    for sub in 0..gt {
                        let ct = cg * gt + sub;
                        let col_start = ct * tile_cols;
                        let nz = nz_arr[sub];
                        unsafe {
                            let dst_sub = sc.tw_step.as_mut_ptr().add(sub * tile_elems);
                            if nz > 0 {
                                let scfg0 = NttSplitCfg {
                                    in_limbs: limbs.len() as u64,
                                    start_coeff: col_start as u64,
                                    count: nz as u64,
                                    coeff_bits: spec.coeff_bits as u64,
                                    col_stride: cols as u64,
                                    cols_per_row: tile_cols as u64,
                                    p: prime_col_fwd[0].p,
                                    inv_p: prime_col_fwd[0].inv_p,
                                };
                                (kernels.active.ntt_split)(dst_sub, limbs.as_ptr(), &scfg0);
                            }
                            if nz < tile_elems {
                                std::ptr::write_bytes(dst_sub.add(nz), 0, tile_elems - nz);
                            }
                        }
                    }
                    NTT_SW1_SPLIT_NS
                        .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                } else if use_cg_chunks {
                    let t_sp0 = std::time::Instant::now();
                    for sub in 0..gt {
                        let ct = cg * gt + sub;
                        let col_start = ct * tile_cols;
                        let nz = nz_arr[sub];
                        if nz > 0 {
                            let ext_cfg = NttSplitCfg {
                                in_limbs: limbs.len() as u64,
                                start_coeff: col_start as u64,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: tile_cols as u64,
                                p: 0.0,
                                inv_p: 0.0,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tw_step
                                        .as_mut_ptr()
                                        .add(sub * num_split_chunks * tile_pl_stride),
                                    limbs.as_ptr(),
                                    &ext_cfg,
                                );
                            }
                        }
                    }
                    NTT_SW1_SPLIT_NS
                        .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                for pi in 0..k {
                    let p_u64 = crt.primes_u64[pi];
                    let pr = prime_col_fwd[pi];
                    let tw_col_ptr = col_tables[pi].tw.as_ptr();
                    let tw_t0 = &tw_tile0_fwd[pi];
                    let psi = psi_fwd[pi];
                    if p_indep_split {
                        let t_co0 = std::time::Instant::now();
                        let work_buf: &mut [f64] = if pi + 1 < k {
                            for sub in 0..gt {
                                let base = sub * tile_elems;
                                let nz = nz_arr[sub];
                                if nz > 0 {
                                    sc.tile[base..(base + nz)]
                                        .copy_from_slice(&sc.tw_step[base..(base + nz)]);
                                }
                                if nz < tile_elems {
                                    unsafe {
                                        std::ptr::write_bytes(
                                            sc.tile.as_mut_ptr().add(base + nz),
                                            0,
                                            tile_elems - nz,
                                        );
                                    }
                                }
                            }
                            &mut sc.tile[..(gt * tile_elems)]
                        } else {
                            &mut sc.tw_step[..(gt * tile_elems)]
                        };
                        for sub in 0..gt {
                            if nz_arr[sub] == 0 {
                                continue;
                            }
                            let ct = cg * gt + sub;
                            let col_start = ct * tile_cols;
                            let sub_slice =
                                &mut work_buf[(sub * tile_elems)..((sub + 1) * tile_elems)];
                            unsafe {
                                call_ntt_fwd_cols(
                                    kernels,
                                    sub_slice.as_mut_ptr(),
                                    tw_col_ptr,
                                    &shape,
                                    &pr,
                                    &mut sc.col_scratch,
                                );
                            }
                            apply_tile_twist_cg4_sub(
                                kernels,
                                sub_slice,
                                tw_t0,
                                &step_tc_fwd[pi],
                                rows,
                                tile_cols,
                                col_start,
                                sub,
                                psi,
                                p_u64,
                                pr.p,
                                pr.inv_p,
                                &mut sc.tw_row,
                                &mut sc.s_u64,
                            );
                        }
                        let t_sc0 = std::time::Instant::now();
                        let mut r0 = 0usize;
                        while r0 < rows {
                            let sr = (rows - r0).min(strip_max);
                            unsafe {
                                let dst_base = sc.ra1.as_mut_ptr();
                                let src_base = work_buf.as_ptr();
                                for r in 0..sr {
                                    let d = dst_base.add(r * 256);
                                    let row_idx = r0 + r;
                                    for sub in 0..gt {
                                        std::ptr::copy_nonoverlapping(
                                            src_base.add(sub * tile_elems + row_idx * tile_cols),
                                            d.add(sub * tile_cols),
                                            tile_cols,
                                        );
                                    }
                                }
                                let mcfg_strip = MoveCfg {
                                    rows: sr as u64,
                                    cols: 256,
                                    outer_stride_bytes: row_bytes as u64,
                                    packed: 1,
                                    use_nt: sw1_nt_flag,
                                };
                                let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(r0 * row_bytes + group_byte_off);
                                (kernels.active.scatter)(outer_ptr, sc.ra1.as_ptr(), &mcfg_strip);
                            }
                            r0 += sr;
                        }
                        NTT_SW1_COLS_NS.fetch_add(
                            t_sc0.duration_since(t_co0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    } else {
                        for sub in 0..gt {
                            let ct = cg * gt + sub;
                            let col_start = ct * tile_cols;
                            let nz = nz_arr[sub];
                            let sub_slice =
                                &mut sc.tile[(sub * tile_elems)..((sub + 1) * tile_elems)];
                            if nz == 0 {
                                unsafe {
                                    std::ptr::write_bytes(sub_slice.as_mut_ptr(), 0, tile_elems);
                                }
                                continue;
                            }
                            let t_sp0 = std::time::Instant::now();
                            if use_cg_chunks {
                                let red_cfg = NttSplitCfg {
                                    in_limbs: u64::MAX,
                                    start_coeff: 0,
                                    count: nz as u64,
                                    coeff_bits: spec.coeff_bits as u64,
                                    col_stride: 0,
                                    cols_per_row: 0,
                                    p: pr.p,
                                    inv_p: pr.inv_p,
                                };
                                unsafe {
                                    (kernels.active.ntt_split)(
                                        sub_slice.as_mut_ptr(),
                                        sc.tw_step
                                            .as_ptr()
                                            .add(sub * num_split_chunks * tile_pl_stride)
                                            as *const u64,
                                        &red_cfg,
                                    );
                                    if nz < tile_elems {
                                        std::ptr::write_bytes(
                                            sub_slice.as_mut_ptr().add(nz),
                                            0,
                                            tile_elems - nz,
                                        );
                                    }
                                }
                            } else {
                                let scfg = NttSplitCfg {
                                    in_limbs: limbs.len() as u64,
                                    start_coeff: col_start as u64,
                                    count: nz as u64,
                                    coeff_bits: spec.coeff_bits as u64,
                                    col_stride: cols as u64,
                                    cols_per_row: tile_cols as u64,
                                    p: pr.p,
                                    inv_p: pr.inv_p,
                                };
                                unsafe {
                                    (kernels.active.ntt_split)(
                                        sub_slice.as_mut_ptr(),
                                        limbs.as_ptr(),
                                        &scfg,
                                    );
                                    if nz < tile_elems {
                                        std::ptr::write_bytes(
                                            sub_slice.as_mut_ptr().add(nz),
                                            0,
                                            tile_elems - nz,
                                        );
                                    }
                                }
                            }
                            let t_co0 = std::time::Instant::now();
                            unsafe {
                                call_ntt_fwd_cols(
                                    kernels,
                                    sub_slice.as_mut_ptr(),
                                    tw_col_ptr,
                                    &shape,
                                    &pr,
                                    &mut sc.col_scratch,
                                );
                            }
                            apply_tile_twist_cg4_sub(
                                kernels,
                                sub_slice,
                                tw_t0,
                                &step_tc_fwd[pi],
                                rows,
                                tile_cols,
                                col_start,
                                sub,
                                psi,
                                p_u64,
                                pr.p,
                                pr.inv_p,
                                &mut sc.tw_row,
                                &mut sc.s_u64,
                            );
                            NTT_SW1_SPLIT_NS.fetch_add(
                                t_co0.duration_since(t_sp0).as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                            NTT_SW1_COLS_NS.fetch_add(
                                t_co0.elapsed().as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                        }
                        let t_sc0 = std::time::Instant::now();
                        let mut r0 = 0usize;
                        while r0 < rows {
                            let sr = (rows - r0).min(strip_max);
                            unsafe {
                                let dst_base = sc.ra1.as_mut_ptr();
                                let src_base = sc.tile.as_ptr();
                                for r in 0..sr {
                                    let d = dst_base.add(r * 256);
                                    let row_idx = r0 + r;
                                    for sub in 0..gt {
                                        std::ptr::copy_nonoverlapping(
                                            src_base.add(sub * tile_elems + row_idx * tile_cols),
                                            d.add(sub * tile_cols),
                                            tile_cols,
                                        );
                                    }
                                }
                                let mcfg_strip = MoveCfg {
                                    rows: sr as u64,
                                    cols: 256,
                                    outer_stride_bytes: row_bytes as u64,
                                    packed: 1,
                                    use_nt: sw1_nt_flag,
                                };
                                let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(r0 * row_bytes + group_byte_off);
                                (kernels.active.scatter)(outer_ptr, sc.ra1.as_ptr(), &mcfg_strip);
                            }
                            r0 += sr;
                        }
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            },
        );
    } else if !p_indep_split
        && num_col_tiles.is_multiple_of(4)
        && (num_col_tiles / 4) >= pw
    {
        let num_groups = num_col_tiles / 4;
        let use_tile_chunks = p_chunk_split && rows <= 4096;
        sched.par_for_init_max(
            num_groups,
            1,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, tile_elems);
                lease
            },
            |lease, cg4| {
                let sc = lease.get_mut();
                let shape = [rows as u64, tile_cols as u64];
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: tile_cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: sw1_nt_flag,
                };
                for sub in 0..4 {
                    let ct = cg4 * 4 + sub;
                    let col_start = ct * tile_cols;
                    let nz = tile_nz_elems(
                        limbs.len(),
                        spec.coeff_bits,
                        rows,
                        cols,
                        col_start,
                        tile_cols,
                    );
                    let col_byte_off = if spec.packed {
                        (col_start / 8) * 50
                    } else {
                        col_start * 8
                    };
                    if nz == 0 {
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                            for pi in 0..k {
                                let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(col_byte_off);
                                (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                            }
                        }
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        continue;
                    }
                    let use_sub_ch = use_tile_chunks && nz >= 32;
                    if use_sub_ch {
                        let t_sp0 = std::time::Instant::now();
                        resize_f64_buf(
                            &mut sc.tw_step,
                            num_split_chunks * crate::refk::ntt_planar_stride(nz),
                        );
                        let ext_cfg = NttSplitCfg {
                            in_limbs: limbs.len() as u64,
                            start_coeff: col_start as u64,
                            count: nz as u64,
                            coeff_bits: spec.coeff_bits as u64,
                            col_stride: cols as u64,
                            cols_per_row: tile_cols as u64,
                            p: 0.0,
                            inv_p: 0.0,
                        };
                        unsafe {
                            (kernels.active.ntt_split)(
                                sc.tw_step.as_mut_ptr(),
                                limbs.as_ptr(),
                                &ext_cfg,
                            );
                        }
                        NTT_SW1_SPLIT_NS
                            .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    for pi in 0..k {
                        let p_u64 = crt.primes_u64[pi];
                        let pr = prime_col_fwd[pi];
                        let tw_col_ptr = col_tables[pi].tw.as_ptr();
                        let tw_t0 = &tw_tile0_fwd[pi];
                        let psi = psi_fwd[pi];
                        let t_sp0 = std::time::Instant::now();
                        if use_sub_ch {
                            let red_cfg = NttSplitCfg {
                                in_limbs: u64::MAX,
                                start_coeff: 0,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: 0,
                                cols_per_row: 0,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    sc.tw_step.as_ptr() as *const u64,
                                    &red_cfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        } else {
                            let scfg = NttSplitCfg {
                                in_limbs: limbs.len() as u64,
                                start_coeff: col_start as u64,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: tile_cols as u64,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    limbs.as_ptr(),
                                    &scfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        }
                        let t_co0 = std::time::Instant::now();
                        unsafe {
                            call_ntt_fwd_cols(
                                kernels,
                                sc.tile.as_mut_ptr(),
                                tw_col_ptr,
                                &shape,
                                &pr,
                                &mut sc.col_scratch,
                            );
                        }
                        apply_tile_twist(
                            kernels,
                            &mut sc.tile[..tile_elems],
                            tw_t0,
                            rows,
                            tile_cols,
                            col_start,
                            psi,
                            p_u64,
                            pr.p,
                            pr.inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(col_byte_off);
                            (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                        }
                        NTT_SW1_SPLIT_NS.fetch_add(
                            t_co0.duration_since(t_sp0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_COLS_NS.fetch_add(
                            t_sc0.duration_since(t_co0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            },
        );
    } else if num_col_tiles >= pw {
        let use_tile_chunks = p_chunk_split && tile_elems <= 32_768;
        let sw1_chunk = if spec.packed && tile_cols == 64 && num_col_tiles >= pw * 16 {
            4
        } else {
            (64 / tile_cols)
                .max(1)
                .min((num_col_tiles / (pw * 8)).max(1))
                .min(4)
        };
        sched.par_for_init_max(
            num_col_tiles,
            sw1_chunk,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, tile_elems);
                if p_indep_split {
                    resize_f64_buf(&mut sc.tw_step, tile_elems);
                } else if use_tile_chunks {
                    resize_f64_buf(
                        &mut sc.tw_step,
                        num_split_chunks * crate::refk::ntt_planar_stride(tile_elems),
                    );
                }
                lease
            },
            |lease, ct| {
                let sc = lease.get_mut();
                let col_start = ct * tile_cols;
                let nz = tile_nz_elems(
                    limbs.len(),
                    spec.coeff_bits,
                    rows,
                    cols,
                    col_start,
                    tile_cols,
                );
                let shape = [rows as u64, tile_cols as u64];
                let col_byte_off = if spec.packed {
                    (col_start / 8) * 50
                } else {
                    col_start * 8
                };
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: tile_cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: sw1_nt_flag,
                };
                if nz == 0 {
                    let t_sc0 = std::time::Instant::now();
                    unsafe {
                        std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                        for pi in 0..k {
                            let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(col_byte_off);
                            (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                        }
                    }
                    NTT_SW1_SCAT_NS
                        .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    return;
                }
                if p_indep_split {
                    let t_sp0 = std::time::Instant::now();
                    let scfg0 = NttSplitCfg {
                        in_limbs: limbs.len() as u64,
                        start_coeff: col_start as u64,
                        count: nz as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: cols as u64,
                        cols_per_row: tile_cols as u64,
                        p: prime_col_fwd[0].p,
                        inv_p: prime_col_fwd[0].inv_p,
                    };
                    unsafe {
                        (kernels.active.ntt_split)(
                            sc.tw_step.as_mut_ptr(),
                            limbs.as_ptr(),
                            &scfg0,
                        );
                        if nz < tile_elems {
                            std::ptr::write_bytes(
                                sc.tw_step.as_mut_ptr().add(nz),
                                0,
                                tile_elems - nz,
                            );
                        }
                    }
                    NTT_SW1_SPLIT_NS
                        .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    for pi in 0..k {
                        let p_u64 = crt.primes_u64[pi];
                        let pr = prime_col_fwd[pi];
                        let t_co0 = std::time::Instant::now();
                        let buf_ref: &mut [f64] = if pi + 1 < k {
                            sc.tile[..nz].copy_from_slice(&sc.tw_step[..nz]);
                            if nz < tile_elems {
                                unsafe {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                            &mut sc.tile[..tile_elems]
                        } else {
                            &mut sc.tw_step[..tile_elems]
                        };
                        unsafe {
                            call_ntt_fwd_cols(
                                kernels,
                                buf_ref.as_mut_ptr(),
                                col_tables[pi].tw.as_ptr(),
                                &shape,
                                &pr,
                                &mut sc.col_scratch,
                            );
                        }
                        apply_tile_twist(
                            kernels,
                            buf_ref,
                            &tw_tile0_fwd[pi],
                            rows,
                            tile_cols,
                            col_start,
                            psi_fwd[pi],
                            p_u64,
                            pr.p,
                            pr.inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(col_byte_off);
                            (kernels.active.scatter)(outer_ptr, buf_ref.as_ptr(), &mcfg);
                        }
                        NTT_SW1_COLS_NS.fetch_add(
                            t_sc0.duration_since(t_co0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                } else {
                    if use_tile_chunks {
                        let t_sp0 = std::time::Instant::now();
                        let ext_cfg = NttSplitCfg {
                            in_limbs: limbs.len() as u64,
                            start_coeff: col_start as u64,
                            count: nz as u64,
                            coeff_bits: spec.coeff_bits as u64,
                            col_stride: cols as u64,
                            cols_per_row: tile_cols as u64,
                            p: 0.0,
                            inv_p: 0.0,
                        };
                        unsafe {
                            (kernels.active.ntt_split)(
                                sc.tw_step.as_mut_ptr(),
                                limbs.as_ptr(),
                                &ext_cfg,
                            );
                        }
                        NTT_SW1_SPLIT_NS
                            .fetch_add(t_sp0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    for pi in 0..k {
                        let p_u64 = crt.primes_u64[pi];
                        let pr = prime_col_fwd[pi];
                        let t_sp0 = std::time::Instant::now();
                        if use_tile_chunks {
                            let red_cfg = NttSplitCfg {
                                in_limbs: u64::MAX,
                                start_coeff: 0,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: 0,
                                cols_per_row: 0,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    sc.tw_step.as_ptr() as *const u64,
                                    &red_cfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        } else {
                            let scfg = NttSplitCfg {
                                in_limbs: limbs.len() as u64,
                                start_coeff: col_start as u64,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: tile_cols as u64,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            unsafe {
                                (kernels.active.ntt_split)(
                                    sc.tile.as_mut_ptr(),
                                    limbs.as_ptr(),
                                    &scfg,
                                );
                                if nz < tile_elems {
                                    std::ptr::write_bytes(
                                        sc.tile.as_mut_ptr().add(nz),
                                        0,
                                        tile_elems - nz,
                                    );
                                }
                            }
                        }
                        let t_co0 = std::time::Instant::now();
                        unsafe {
                            call_ntt_fwd_cols(
                                kernels,
                                sc.tile.as_mut_ptr(),
                                col_tables[pi].tw.as_ptr(),
                                &shape,
                                &pr,
                                &mut sc.col_scratch,
                            );
                        }
                        apply_tile_twist(
                            kernels,
                            &mut sc.tile[..tile_elems],
                            &tw_tile0_fwd[pi],
                            rows,
                            tile_cols,
                            col_start,
                            psi_fwd[pi],
                            p_u64,
                            pr.p,
                            pr.inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                        let t_sc0 = std::time::Instant::now();
                        unsafe {
                            let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(col_byte_off);
                            (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                        }
                        NTT_SW1_SPLIT_NS.fetch_add(
                            t_co0.duration_since(t_sp0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_COLS_NS.fetch_add(
                            t_sc0.duration_since(t_co0).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        NTT_SW1_SCAT_NS
                            .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            },
        );
    } else {
        let total_sw1 = k * num_col_tiles;
        let sw1_chunk = (128 / tile_cols)
            .max(1)
            .min((total_sw1 / (pw * 8)).max(1))
            .min(4);
        sched.par_for_init_max(
            total_sw1,
            sw1_chunk,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, rows * tile_cols);
                lease
            },
            |lease, work_id| {
                let sc = lease.get_mut();
                let pi = work_id / num_col_tiles;
                let ct = work_id % num_col_tiles;
                let col_start = ct * tile_cols;
                let tile_elems_w = rows * tile_cols;
                let nz = tile_nz_elems(
                    limbs.len(),
                    spec.coeff_bits,
                    rows,
                    cols,
                    col_start,
                    tile_cols,
                );
                let p_u64 = crt.primes_u64[pi];
                let pr = prime_col_fwd[pi];

                let shape = [rows as u64, tile_cols as u64];
                let col_byte_off = if spec.packed {
                    (col_start / 8) * 50
                } else {
                    col_start * 8
                };
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: tile_cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: sw1_nt_flag,
                };
                if nz == 0 {
                    let t_sc0 = std::time::Instant::now();
                    unsafe {
                        std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems_w);
                        let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(col_byte_off);
                        (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                    }
                    NTT_SW1_SCAT_NS
                        .fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    return;
                }
                let scfg = NttSplitCfg {
                    in_limbs: limbs.len() as u64,
                    start_coeff: col_start as u64,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: cols as u64,
                    cols_per_row: tile_cols as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                let t_sp0 = std::time::Instant::now();
                unsafe {
                    (kernels.active.ntt_split)(sc.tile.as_mut_ptr(), limbs.as_ptr(), &scfg);
                    if nz < tile_elems_w {
                        std::ptr::write_bytes(sc.tile.as_mut_ptr().add(nz), 0, tile_elems_w - nz);
                    }
                }
                let t_co0 = std::time::Instant::now();
                unsafe {
                    call_ntt_fwd_cols(
                        kernels,
                        sc.tile.as_mut_ptr(),
                        col_tables[pi].tw.as_ptr(),
                        &shape,
                        &pr,
                        &mut sc.col_scratch,
                    );
                }
                apply_tile_twist(
                    kernels,
                    &mut sc.tile[..tile_elems_w],
                    &tw_tile0_fwd[pi],
                    rows,
                    tile_cols,
                    col_start,
                    psi_fwd[pi],
                    p_u64,
                    pr.p,
                    pr.inv_p,
                    &mut sc.tw_row,
                    &mut sc.s_u64,
                );
                let t_sc0 = std::time::Instant::now();
                unsafe {
                    let outer_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                        .as_mut_ptr()
                        .add(col_byte_off);
                    (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                }
                NTT_SW1_SPLIT_NS
                    .fetch_add(t_co0.duration_since(t_sp0).as_nanos() as u64, Ordering::Relaxed);
                NTT_SW1_COLS_NS
                    .fetch_add(t_sc0.duration_since(t_co0).as_nanos() as u64, Ordering::Relaxed);
                NTT_SW1_SCAT_NS.fetch_add(t_sc0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            },
        );
    }
    for b in &mut bufs {
        unsafe {
            b.set_len(total_grid_bytes);
        }
    }
    if !used_all_primes_sw1 {
        NTT_TRACE_SW1_NS.fetch_add(t_sw1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }

    if kernels.count_work {
        let bf_cols = ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
        kernels
            .stats
            .ntt_butterflies
            .fetch_add(bf_cols, Ordering::Relaxed);
        kernels
            .stats
            .bytes_moved
            .fetch_add((k * rows * row_bytes) as u64, Ordering::Relaxed);
    }

    let mut rows_transformed = false;
    if reusable {
        let row0_tables: Vec<&PrimeTables> =
            cache.get_twiddles_refs_par(sched, spec.par_workers, &crt.primes_u64, k, cols, false);
        let sw2_w = sw2_workers(sched, &spec);
        let t_sw1r = std::time::Instant::now();
        if !spec.packed && spec.extra_sweeps == 0 {
            sched.par_for_max(k * rows, 1, sw2_w, |work_id| {
                let pi = work_id / rows;
                let r = work_id % rows;
                let row_off = r * row_bytes;
                unsafe {
                    let row_ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                        .as_mut_ptr()
                        .add(row_off) as *mut f64;
                    call_ntt_fwd(kernels, 
                        row_ptr,
                        row0_tables[pi].tw.as_ptr(),
                        cols as u64,
                        &row0_tables[pi].pr_fwd,
                    );
                }
            });
        } else {
            sched.par_for_init_max(
                k * rows,
                1,
                sw2_w,
                || {
                    let mut lease = ScratchLease::acquire();
                    resize_f64_buf(&mut lease.get_mut().ra1, cols);
                    lease
                },
                |lease, work_id| {
                    let sc = lease.get_mut();
                    let pi = work_id / rows;
                    let r = work_id % rows;
                    let row_off = r * row_bytes;
                    let mcfg = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: row_bytes as u64,
                        packed: u64::from(spec.packed),
                        use_nt: sw1_nt_flag,
                    };
                    unsafe {
                        let ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(row_off);
                        (kernels.active.gather)(sc.ra1.as_mut_ptr(), ptr, &mcfg);
                    }
                    row_fwd_multi_sweep(
                        kernels,
                        &mut sc.ra1,
                        &row0_tables[pi].tw,
                        &row0_tables[pi].pr_fwd,
                        spec.extra_sweeps,
                    );
                    unsafe {
                        let ptr = (*(bufs_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(row_off);
                        (kernels.active.scatter)(ptr, sc.ra1.as_ptr(), &mcfg);
                    }
                },
            );
        }
        NTT_TRACE_SW1_NS.fetch_add(t_sw1r.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        if kernels.count_work {
            let bf_rows = ((spec.n as u64) / 2) * (cols.ilog2() as u64) * (k as u64);
            kernels
                .stats
                .ntt_butterflies
                .fetch_add(bf_rows, Ordering::Relaxed);
            kernels
                .stats
                .bytes_moved
                .fetch_add((2 * k * rows * row_bytes) as u64, Ordering::Relaxed);
        }
        rows_transformed = true;
    }

    finish_cached_fwd(
        spec,
        rows_transformed,
        bufs,
        total_grid_bytes,
        cache,
        store,
        kernels,
    )
}

#[inline]
fn ooc_elem_bytes(count: usize, packed: bool) -> usize {
    if packed {
        debug_assert_eq!(count % 8, 0);
        (count / 8) * 50
    } else {
        count * 8
    }
}

#[inline]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn row_piece_shift(r: usize, pieces_per_row: usize, stripe_width: usize) -> usize {
    if pieces_per_row <= 1 || stripe_width <= 1 {
        return 0;
    }
    let g = gcd_usize(pieces_per_row, stripe_width);
    if g <= 1 {
        return 0;
    }
    let period = (stripe_width / g).max(1);
    (r / period) & (g - 1)
}

#[derive(Clone, Copy, Debug)]
struct StoreGridLayout {
    rows: usize,
    cols: usize,
    packed: bool,
    piece_data_bytes: usize,
    cols_per_piece: usize,
    pieces_per_row: usize,
    rows_per_piece: usize,
    row_units_per_prime: usize,
    total_pieces_per_prime: usize,
    row_stride_in_piece: usize,
    used_bytes_per_piece: usize,
}

impl StoreGridLayout {
    fn build(spec: &NttSpec, piece_limbs: usize) -> Self {
        let rows = spec.rows.max(1);
        let cols = spec.cols.max(1);
        let packed = spec.packed;
        let piece_data_bytes = piece_limbs * 8;
        let mut cols_per_piece = cols;
        while ooc_elem_bytes(cols_per_piece, packed) > piece_data_bytes && cols_per_piece > 1 {
            cols_per_piece >>= 1;
        }
        debug_assert!(!packed || cols_per_piece % 8 == 0);
        let pieces_per_row = (cols / cols_per_piece).max(1);
        let cp_data_bytes = ooc_elem_bytes(cols_per_piece, packed);
        let (rows_per_piece, row_stride_in_piece) = if pieces_per_row == 1 {
            let stride = cp_data_bytes.next_multiple_of(8).max(8);
            let max_rpp = (piece_data_bytes / stride).max(1);
            let rpp = (1usize << max_rpp.ilog2()).min(rows).max(1);
            (rpp, stride)
        } else {
            (1usize, cp_data_bytes)
        };
        let row_units_per_prime = (rows / rows_per_piece).max(1);
        let total_pieces_per_prime = row_units_per_prime * pieces_per_row;
        let used_bytes_per_piece = if pieces_per_row == 1 {
            rows_per_piece * row_stride_in_piece
        } else {
            cp_data_bytes
        };
        Self {
            rows,
            cols,
            packed,
            piece_data_bytes,
            cols_per_piece,
            pieces_per_row,
            rows_per_piece,
            row_units_per_prime,
            total_pieces_per_prime,
            row_stride_in_piece,
            used_bytes_per_piece,
        }
    }

    fn col_batch_ranges(&self, cp0: usize, cp_batch: usize, stripe_width: usize) -> Vec<ExtentRange> {
        if self.pieces_per_row == 1 {
            return vec![ExtentRange {
                piece_start: 0,
                piece_count: self.total_pieces_per_prime as u64,
                host_limb_offset: 0,
            }];
        }
        (0..self.rows)
            .map(|r| {
                let sh = row_piece_shift(r, self.pieces_per_row, stripe_width);
                let p0 = cp0 ^ (sh & !(cp_batch - 1));
                ExtentRange {
                    piece_start: (r * self.pieces_per_row + p0) as u64,
                    piece_count: cp_batch as u64,
                    host_limb_offset: 0,
                }
            })
            .collect()
    }

    #[inline]
    fn col_slot_and_off(
        &self,
        plan: &SlotBatchPlan,
        r: usize,
        cp0: usize,
        cp_batch: usize,
        cp: usize,
        stripe_width: usize,
    ) -> (usize, usize) {
        if self.pieces_per_row == 1 {
            let p_idx = r / self.rows_per_piece;
            let sub_r = r % self.rows_per_piece;
            (
                plan.pieces[p_idx].buf_byte_off,
                sub_r * self.row_stride_in_piece,
            )
        } else {
            let sh = row_piece_shift(r, self.pieces_per_row, stripe_width);
            let p_off = (cp - cp0) ^ (sh & (cp_batch - 1));
            (plan.pieces[r * cp_batch + p_off].buf_byte_off, 0)
        }
    }
}

struct StoreHandlesGuard<'a> {
    store: &'a Store,
    handles: Vec<ExtentHandle>,
}

impl<'a> StoreHandlesGuard<'a> {
    fn into_handles(mut self) -> Vec<ExtentHandle> {
        std::mem::take(&mut self.handles)
    }
}

impl<'a> Drop for StoreHandlesGuard<'a> {
    fn drop(&mut self) {
        for h in self.handles.drain(..) {
            self.store.free(h);
        }
    }
}

struct PrimeSlotStage {
    plans: Vec<SlotBatchPlan>,
    bufs: Vec<AlignedSlotBuf>,
}

impl PrimeSlotStage {
    fn new(st: &Store, max_primes: usize, buf_bytes_each: usize) -> Self {
        let mut bufs = Vec::with_capacity(max_primes);
        for _ in 0..max_primes {
            bufs.push(st.alloc_tile_slot_buf(buf_bytes_each));
        }
        Self {
            plans: Vec::with_capacity(max_primes),
            bufs,
        }
    }

    fn ensure_buf_capacities(&mut self) {
        for (idx, plan) in self.plans.iter().enumerate() {
            self.bufs[idx].ensure_capacity(plan.required_buf_bytes);
        }
    }

    fn write_all_to_store(&self, st: &Store) -> Result<(), StoreError> {
        for (idx, plan) in self.plans.iter().enumerate() {
            st.devices[plan.extent.device_idx].write_planned_slots(
                plan,
                &self.bufs[idx],
                &st.bytes_moved,
            )?;
        }
        Ok(())
    }

    fn read_all_from_store(&mut self, st: &Store, telemetry: &Telemetry) -> Result<(), StoreError> {
        for (idx, plan) in self.plans.iter().enumerate() {
            st.devices[plan.extent.device_idx].read_planned_slots(
                plan,
                &mut self.bufs[idx],
                telemetry,
                &st.bytes_moved,
            )?;
        }
        Ok(())
    }
}

#[inline]
fn upper_slot_buf_bytes(st: &Store, pieces_per_prime: usize) -> usize {
    st.upper_slot_buf_bytes(pieces_per_prime)
}

/// Out-of-core forward NTT (Sweep 1): computes RAM-sized column-piece tiles into 64 KiB-aligned
/// `AlignedSlotBuf`s, runs the Section 11 `check_n` piece check pass in parallel before each tile
/// leaves RAM, and overlaps `Loader` batch writes (`submit_loader_scoped`) with worker compute.
fn ntt_forward_out_of_core(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    st: &Store,
    limbs: &[u64],
    spec: NttSpec,
    crt: &CrtPlan,
) -> Result<NttCachedFwd, StoreError> {
    if st.devices.is_empty() {
        return Err(StoreError::OutOfSpace);
    }
    let k = spec.k_primes;
    let lay = StoreGridLayout::build(&spec, st.devices[0].piece_limbs);
    let ext_payload_bytes = (lay.total_pieces_per_prime * lay.piece_data_bytes) as u64;

    let mut guard = StoreHandlesGuard {
        store: st,
        handles: Vec::with_capacity(k),
    };
    for _ in 0..k {
        guard.handles.push(st.allocate(ext_payload_bytes)?);
    }

    let budget = cache.ooc_tile_budget_bytes().max(65_536) as usize;
    let half_budget = (budget / 2).max(1);
    let min_range_bytes = st.devices.first().map_or(0, |d| d.min_batch_range_bytes);
    let max_slot_bytes = st
        .devices
        .iter()
        .map(|d| d.piece_slot_bytes as usize)
        .max()
        .unwrap_or(65_536);

    let (cp_batch, primes_in_batch) = if lay.pieces_per_row == 1 {
        let per_p = upper_slot_buf_bytes(st, lay.total_pieces_per_prime);
        let mut p_cnt = (half_budget / per_p).clamp(1, k);
        if p_cnt == k && k >= 2 {
            p_cnt = (k / 2).max(1);
        }
        (1usize, p_cnt)
    } else {
        let min_cp_raw = min_range_bytes.div_ceil(max_slot_bytes as u64).max(1) as usize;
        let mut cp_b = min_cp_raw.next_power_of_two().min(lay.pieces_per_row);
        while cp_b > 1 && upper_slot_buf_bytes(st, lay.rows * cp_b) > half_budget {
            cp_b >>= 1;
        }
        let per_p0 = upper_slot_buf_bytes(st, lay.rows * cp_b);
        let mut p_cnt = (half_budget / per_p0).clamp(1, k);
        while p_cnt == k && cp_b * 2 <= (lay.pieces_per_row / 2).max(1) {
            let next_p = upper_slot_buf_bytes(st, lay.rows * cp_b * 2);
            if k.saturating_mul(next_p) <= half_budget {
                cp_b *= 2;
            } else {
                break;
            }
        }
        if p_cnt == k && cp_b == lay.pieces_per_row && k >= 2 {
            p_cnt = (k / 2).max(1);
        }
        (cp_b, p_cnt)
    };

    let pieces_per_p_batch = if lay.pieces_per_row == 1 {
        lay.total_pieces_per_prime
    } else {
        lay.rows * cp_batch
    };
    let buf_bytes_each = upper_slot_buf_bytes(st, pieces_per_p_batch);
    let mut stages = [
        PrimeSlotStage::new(st, primes_in_batch, buf_bytes_each),
        PrimeSlotStage::new(st, primes_in_batch, buf_bytes_each),
    ];

    // Build `(p0, p_cnt, cp0, cp_cnt)` batch schedule:
    let mut batches: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut cp0 = 0usize;
    while cp0 < lay.pieces_per_row {
        let cp_cnt = (lay.pieces_per_row - cp0).min(cp_batch);
        let mut p0 = 0usize;
        while p0 < k {
            let p_cnt = (k - p0).min(primes_in_batch);
            batches.push((p0, p_cnt, cp0, cp_cnt));
            p0 += p_cnt;
        }
        cp0 += cp_cnt;
    }

    let t_sw1 = std::time::Instant::now();
    if spec.rows == 1 {
        let n = spec.n;
        let nz_1d = nz_elems_1d(limbs.len(), spec.coeff_bits, n);
        let fill_1d_batch =
            |stage: &mut PrimeSlotStage, (p0, p_cnt, cp0, cp_cnt): (usize, usize, usize, usize)| -> Result<(), StoreError> {
                stage.plans.clear();
                for pi_loc in 0..p_cnt {
                    let h = &guard.handles[p0 + pi_loc];
                    let dev = &st.devices[h.device_idx];
                    let ranges = lay.col_batch_ranges(cp0, cp_cnt, dev.stripe_width());
                    stage.plans.push(dev.plan_slot_batch(h, &ranges)?);
                }
                stage.ensure_buf_capacities();
                let buf_addrs: Vec<usize> = stage.bufs[..p_cnt]
                    .iter_mut()
                    .map(|b| b.as_mut_ptr() as usize)
                    .collect();
                let plans_ref = stage.plans.as_slice();
                sched.par_for_init_max(
                    p_cnt,
                    1,
                    spec.par_workers,
                    ScratchLease::acquire,
                    |lease, pi_loc| {
                        let pi = p0 + pi_loc;
                        let p_u64 = crt.primes_u64[pi];
                        let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
                        let pr = &tables.pr_fwd;
                        let tile = &mut lease.get_mut().tile;
                        resize_f64_buf(tile, n);
                        unsafe {
                            if nz_1d > 0 {
                                let scfg = NttSplitCfg {
                                    in_limbs: limbs.len() as u64,
                                    start_coeff: 0,
                                    count: nz_1d as u64,
                                    coeff_bits: spec.coeff_bits as u64,
                                    col_stride: 0,
                                    cols_per_row: 0,
                                    p: pr.p,
                                    inv_p: pr.inv_p,
                                };
                                (kernels.active.ntt_split)(tile.as_mut_ptr(), limbs.as_ptr(), &scfg);
                            }
                            if nz_1d < n {
                                std::ptr::write_bytes(tile.as_mut_ptr().add(nz_1d), 0, n - nz_1d);
                            }
                            call_ntt_fwd(kernels, tile.as_mut_ptr(), tables.tw.as_ptr(), n as u64, pr);
                        }
                        let plan = &plans_ref[pi_loc];
                        let dev = &st.devices[plan.extent.device_idx];
                        let w = dev.stripe_width();
                        let base_ptr = buf_addrs[pi_loc] as *mut u8;
                        let mcfg = MoveCfg {
                            rows: 1,
                            cols: lay.cols_per_piece as u64,
                            outer_stride_bytes: lay.row_stride_in_piece as u64,
                            packed: u64::from(lay.packed),
                            use_nt: 0,
                        };
                        for cp_off in 0..cp_cnt {
                            let cp = cp0 + cp_off;
                            let (slot_off, row_off) = lay.col_slot_and_off(plan, 0, cp0, cp_cnt, cp, w);
                            unsafe {
                                let slot_ptr = base_ptr.add(slot_off);
                                (kernels.active.scatter)(
                                    slot_ptr.add(row_off),
                                    tile.as_ptr().add(cp * lay.cols_per_piece),
                                    &mcfg,
                                );
                                if lay.used_bytes_per_piece < lay.piece_data_bytes {
                                    std::ptr::write_bytes(
                                        slot_ptr.add(lay.used_bytes_per_piece),
                                        0,
                                        lay.piece_data_bytes - lay.used_bytes_per_piece,
                                    );
                                }
                                let slot_slice =
                                    std::slice::from_raw_parts_mut(slot_ptr, plan.piece_slot_bytes);
                                dev.compute_slot_check_in_place(slot_slice, kernels, true);
                            }
                        }
                    },
                );
                Ok(())
            };

        let (s_left, s_right) = stages.split_at_mut(1);
        fill_1d_batch(&mut s_left[0], batches[0])?;
        for b_idx in 1..batches.len() {
            let (prev_s, cur_s) = if (b_idx & 1) == 1 {
                (&s_left[0], &mut s_right[0])
            } else {
                (&s_right[0], &mut s_left[0])
            };
            let loader_h = st.submit_loader_scoped(|| prev_s.write_all_to_store(st));
            let compute_res = fill_1d_batch(cur_s, batches[b_idx]);
            let io_res = loader_h.wait();
            io_res?;
            compute_res?;
        }
        let last_s = if ((batches.len() - 1) & 1) == 0 {
            &s_left[0]
        } else {
            &s_right[0]
        };
        st.submit_loader_scoped(|| last_s.write_all_to_store(st)).wait()?;

        NTT_TRACE_1D_FWD_NS.fetch_add(t_sw1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if kernels.count_work && n > 1 {
            let bf = ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64);
            kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
            kernels
                .stats
                .bytes_moved
                .fetch_add((limbs.len() * 8 + k * spec.outer_row_bytes()) as u64, Ordering::Relaxed);
        }
        return Ok(NttCachedFwd {
            spec,
            rows_transformed: true,
            storage: ResidueStorage::InStore(guard.into_handles()),
        });
    }

    // 2D (`rows > 1`) Sweep 1:
    let rows = lay.rows;
    let cols = lay.cols;
    let mut eff_tc = spec.tile_cols.min(lay.cols_per_piece).max(1);
    if lay.packed && eff_tc < 8 {
        eff_tc = 8.min(lay.cols_per_piece);
    }
    debug_assert_eq!(lay.cols_per_piece % eff_tc, 0);
    let tiles_per_cp = lay.cols_per_piece / eff_tc;
    let tile_elems = rows * eff_tc;

    let col_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        rows,
        spec.negacyclic,
    );
    let psi_fwd: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                spec.negacyclic,
                false,
            )
        })
        .collect();
    let tw_tile0_fwd = cache.get_tile0_twiddles_par(
        kernels,
        sched,
        spec.par_workers,
        &crt.primes_u64,
        &col_tables,
        k,
        spec.n,
        rows,
        eff_tc,
        spec.negacyclic,
        false,
    );
    let prime_col_fwd: Vec<NttPrime> = (0..k).map(|pi| col_tables[pi].pr_fwd).collect();

    let fill_sw1_batch =
        |stage: &mut PrimeSlotStage, (p0, p_cnt, cp0, cp_cnt): (usize, usize, usize, usize)| -> Result<(), StoreError> {
            stage.plans.clear();
            for pi_loc in 0..p_cnt {
                let h = &guard.handles[p0 + pi_loc];
                let dev = &st.devices[h.device_idx];
                let ranges = lay.col_batch_ranges(cp0, cp_cnt, dev.stripe_width());
                stage.plans.push(dev.plan_slot_batch(h, &ranges)?);
            }
            stage.ensure_buf_capacities();
            let buf_addrs: Vec<usize> = stage.bufs[..p_cnt]
                .iter_mut()
                .map(|b| b.as_mut_ptr() as usize)
                .collect();
            let plans_ref = stage.plans.as_slice();
            let tiles_in_batch = cp_cnt * tiles_per_cp;
            let total_tasks = p_cnt * tiles_in_batch;

            sched.par_for_init_max(
                total_tasks,
                1,
                spec.par_workers,
                || {
                    let mut lease = ScratchLease::acquire();
                    resize_f64_buf(&mut lease.get_mut().tile, tile_elems);
                    lease
                },
                |lease, task_id| {
                    let sc = lease.get_mut();
                    let pi_loc = task_id / tiles_in_batch;
                    let t_loc = task_id % tiles_in_batch;
                    let pi = p0 + pi_loc;
                    let ct = cp0 * tiles_per_cp + t_loc;
                    let col_start = ct * eff_tc;
                    let cp = col_start / lay.cols_per_piece;
                    let local_col = col_start % lay.cols_per_piece;
                    let local_col_byte_off = ooc_elem_bytes(local_col, lay.packed);

                    let nz = tile_nz_elems(
                        limbs.len(),
                        spec.coeff_bits,
                        rows,
                        cols,
                        col_start,
                        eff_tc,
                    );
                    let p_u64 = crt.primes_u64[pi];
                    let pr = prime_col_fwd[pi];
                    let shape = [rows as u64, eff_tc as u64];

                    if nz == 0 {
                        unsafe {
                            std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                        }
                    } else {
                        let scfg = NttSplitCfg {
                            in_limbs: limbs.len() as u64,
                            start_coeff: col_start as u64,
                            count: nz as u64,
                            coeff_bits: spec.coeff_bits as u64,
                            col_stride: cols as u64,
                            cols_per_row: eff_tc as u64,
                            p: pr.p,
                            inv_p: pr.inv_p,
                        };
                        unsafe {
                            (kernels.active.ntt_split)(sc.tile.as_mut_ptr(), limbs.as_ptr(), &scfg);
                            if nz < tile_elems {
                                std::ptr::write_bytes(sc.tile.as_mut_ptr().add(nz), 0, tile_elems - nz);
                            }
                            call_ntt_fwd_cols(
                                kernels,
                                sc.tile.as_mut_ptr(),
                                col_tables[pi].tw.as_ptr(),
                                &shape,
                                &pr,
                                &mut sc.col_scratch,
                            );
                        }
                        apply_tile_twist(
                            kernels,
                            &mut sc.tile[..tile_elems],
                            &tw_tile0_fwd[pi],
                            rows,
                            eff_tc,
                            col_start,
                            psi_fwd[pi],
                            p_u64,
                            pr.p,
                            pr.inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                    }

                    let plan = &plans_ref[pi_loc];
                    let w = st.devices[plan.extent.device_idx].stripe_width();
                    let base_ptr = buf_addrs[pi_loc] as *mut u8;
                    if lay.pieces_per_row == 1 {
                        let mcfg = MoveCfg {
                            rows: lay.rows_per_piece as u64,
                            cols: eff_tc as u64,
                            outer_stride_bytes: lay.row_stride_in_piece as u64,
                            packed: u64::from(lay.packed),
                            use_nt: 0,
                        };
                        for p_idx in 0..lay.total_pieces_per_prime {
                            let slot_off = plan.pieces[p_idx].buf_byte_off;
                            unsafe {
                                (kernels.active.scatter)(
                                    base_ptr.add(slot_off + local_col_byte_off),
                                    sc.tile.as_ptr().add(p_idx * lay.rows_per_piece * eff_tc),
                                    &mcfg,
                                );
                            }
                        }
                    } else {
                        let mcfg_row = MoveCfg {
                            rows: 1,
                            cols: eff_tc as u64,
                            outer_stride_bytes: lay.row_stride_in_piece as u64,
                            packed: u64::from(lay.packed),
                            use_nt: 0,
                        };
                        for r in 0..rows {
                            let (slot_off, _) = lay.col_slot_and_off(plan, r, cp0, cp_cnt, cp, w);
                            unsafe {
                                let dst = base_ptr.add(slot_off + local_col_byte_off);
                                if lay.packed {
                                    (kernels.active.scatter)(
                                        dst,
                                        sc.tile.as_ptr().add(r * eff_tc),
                                        &mcfg_row,
                                    );
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        sc.tile.as_ptr().add(r * eff_tc),
                                        dst as *mut f64,
                                        eff_tc,
                                    );
                                }
                            }
                        }
                    }
                },
            );

            // Section 11: check every outgoing column-sweep piece in a parallel `check_n` pass
            // before it leaves RAM (`count_as_traffic = true`).
            let pieces_each = plans_ref[0].pieces.len();
            let total_slots = p_cnt * pieces_each;
            sched.par_for_leaf_max(total_slots, 1, spec.par_workers, |slot_idx| {
                let pi_loc = slot_idx / pieces_each;
                let pe_idx = slot_idx % pieces_each;
                let plan = &plans_ref[pi_loc];
                let dev = &st.devices[plan.extent.device_idx];
                let slot_off = plan.pieces[pe_idx].buf_byte_off;
                let base_ptr = buf_addrs[pi_loc] as *mut u8;
                unsafe {
                    if lay.used_bytes_per_piece < lay.piece_data_bytes {
                        std::ptr::write_bytes(
                            base_ptr.add(slot_off + lay.used_bytes_per_piece),
                            0,
                            lay.piece_data_bytes - lay.used_bytes_per_piece,
                        );
                    }
                    let slot_mut =
                        std::slice::from_raw_parts_mut(base_ptr.add(slot_off), plan.piece_slot_bytes);
                    dev.compute_slot_check_in_place(slot_mut, kernels, true);
                }
            });
            Ok(())
        };

    let (s_left, s_right) = stages.split_at_mut(1);
    fill_sw1_batch(&mut s_left[0], batches[0])?;
    for b_idx in 1..batches.len() {
        let (prev_s, cur_s) = if (b_idx & 1) == 1 {
            (&s_left[0], &mut s_right[0])
        } else {
            (&s_right[0], &mut s_left[0])
        };
        let loader_h = st.submit_loader_scoped(|| prev_s.write_all_to_store(st));
        let compute_res = fill_sw1_batch(cur_s, batches[b_idx]);
        let io_res = loader_h.wait();
        io_res?;
        compute_res?;
    }
    let last_s = if ((batches.len() - 1) & 1) == 0 {
        &s_left[0]
    } else {
        &s_right[0]
    };
    st.submit_loader_scoped(|| last_s.write_all_to_store(st)).wait()?;

    NTT_TRACE_SW1_NS.fetch_add(t_sw1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    if kernels.count_work {
        let bf_cols = ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
        kernels.stats.ntt_butterflies.fetch_add(bf_cols, Ordering::Relaxed);
        kernels
            .stats
            .bytes_moved
            .fetch_add((limbs.len() * 8 + k * rows * spec.outer_row_bytes()) as u64, Ordering::Relaxed);
    }

    Ok(NttCachedFwd {
        spec,
        rows_transformed: false,
        storage: ResidueStorage::InStore(guard.into_handles()),
    })
}

/// Out-of-core Sweep 2 (sequential row tiles + 3-way `Loader` overlap + inline cache check words)
/// and Sweep 3 (strided column-piece batches + `Loader` prefetch + `check_n` scrub + CRT).
#[allow(clippy::too_many_arguments)]
fn ntt_combine_out_of_core(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    st: &Store,
    telemetry: &Telemetry,
    a1: &mut NttCachedFwd,
    mut b1: Option<NttOperand<'_>>,
    mut second: Option<(NttOperand<'_>, NttOperand<'_>, bool)>,
    out_limbs: usize,
) -> Result<(LimbVec, u64), StoreError> {
    if st.devices.is_empty() {
        return Err(StoreError::DeviceLost);
    }
    let spec = a1.spec;
    let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
    let k = spec.k_primes;
    let signed = spec.negacyclic || second.as_ref().map(|(_, _, sub)| *sub).unwrap_or(false);
    let same_ab1 = b1.is_none();
    let has_second = second.is_some();
    let sub_second = second.as_ref().map(|(_, _, sub)| *sub).unwrap_or(false);

    let lay = StoreGridLayout::build(&spec, st.devices[0].piece_limbs);
    let rows = lay.rows;
    let cols = lay.cols;

    let a1_rt = a1.rows_transformed;
    let b1_rt = b1.as_ref().map(|b| b.as_ref().rows_transformed).unwrap_or(a1_rt);
    let (a2_rt, b2_rt) = second
        .as_ref()
        .map(|(a2, b2, _)| (a2.as_ref().rows_transformed, b2.as_ref().rows_transformed))
        .unwrap_or((false, false));

    let b1_wb = matches!(&b1, Some(NttOperand::ReuseInPlace(op)) if !op.rows_transformed && matches!(op.storage, ResidueStorage::InStore(_)));
    let a2_wb = matches!(&second, Some((NttOperand::ReuseInPlace(op), _, _)) if !op.rows_transformed && matches!(op.storage, ResidueStorage::InStore(_)));
    let b2_wb = matches!(&second, Some((_, NttOperand::ReuseInPlace(op), _)) if !op.rows_transformed && matches!(op.storage, ResidueStorage::InStore(_)));

    // Take `a1`'s `InStore` handles as `prod_handles` (or allocate temporary `prod_handles` if `a1` was `InRam`):
    let (a1_ram_opt, prod_handles_vec) = match &mut a1.storage {
        ResidueStorage::InStore(handles) if handles.len() == k => {
            (None, std::mem::take(handles))
        }
        ResidueStorage::InRam { bufs } => {
            let taken = std::mem::take(bufs);
            let ext_payload_bytes = (lay.total_pieces_per_prime * lay.piece_data_bytes) as u64;
            let mut ph = Vec::with_capacity(k);
            for _ in 0..k {
                match st.allocate(ext_payload_bytes) {
                    Ok(h) => ph.push(h),
                    Err(e) => {
                        for h_prev in ph {
                            st.free(h_prev);
                        }
                        cache.put_byte_bufs(taken);
                        return Err(e);
                    }
                }
            }
            (Some(taken), ph)
        }
        _ => return Err(StoreError::DeviceLost),
    };
    let a1_ram_guard = OwnedSlotsGuard {
        cache,
        s1: a1_ram_opt.unwrap_or_default(),
        s2: Vec::new(),
        s3: Vec::new(),
        s4: Vec::new(),
    };
    let prod_guard = StoreHandlesGuard {
        store: st,
        handles: prod_handles_vec,
    };

    enum OpRef<'r> {
        Store(&'r [ExtentHandle], NttSpec),
        Ram(&'r [AlignedBytes], NttSpec),
    }

    let op_a1 = if !a1_ram_guard.s1.is_empty() {
        OpRef::Ram(a1_ram_guard.s1.as_slice(), spec)
    } else {
        OpRef::Store(prod_guard.handles.as_slice(), spec)
    };
    let op_b1 = b1.as_ref().map(|b| match &b.as_ref().storage {
        ResidueStorage::InStore(h) => OpRef::Store(h.as_slice(), b.as_ref().spec),
        ResidueStorage::InRam { bufs } => OpRef::Ram(bufs.as_slice(), b.as_ref().spec),
    });
    let op_a2 = second.as_ref().map(|(a2, _, _)| match &a2.as_ref().storage {
        ResidueStorage::InStore(h) => OpRef::Store(h.as_slice(), a2.as_ref().spec),
        ResidueStorage::InRam { bufs } => OpRef::Ram(bufs.as_slice(), a2.as_ref().spec),
    });
    let op_b2 = second.as_ref().map(|(_, b2, _)| match &b2.as_ref().storage {
        ResidueStorage::InStore(h) => OpRef::Store(h.as_slice(), b2.as_ref().spec),
        ResidueStorage::InRam { bufs } => OpRef::Ram(bufs.as_slice(), b2.as_ref().spec),
    });

    let in_store_mask = [
        matches!(op_a1, OpRef::Store(..)),
        op_b1.as_ref().is_some_and(|o| matches!(o, OpRef::Store(..))),
        op_a2.as_ref().is_some_and(|o| matches!(o, OpRef::Store(..))),
        op_b2.as_ref().is_some_and(|o| matches!(o, OpRef::Store(..))),
    ];
    let wb_mask = [b1_wb, a2_wb, b2_wb];
    let active_streams =
        in_store_mask.iter().filter(|&&x| x).count() + usize::from(!in_store_mask[0]);

    // Size sequential Sweep 2 steps within `ooc_tile_budget_bytes` (reusing `in_ops` buffers in
    // place for both `ReuseInPlace` write-backs and `prod_out` whenever `a1` lives in `Store`):
    let budget = cache.ooc_tile_budget_bytes().max(65_536) as usize;
    let stream_half_budget = (budget / (2 * active_streams.max(1))).max(1);
    let one_unit_buf = upper_slot_buf_bytes(st, lay.pieces_per_row);
    let max_units_ram = (stream_half_budget / one_unit_buf.max(1)).max(1);
    let total_units = k * lay.row_units_per_prime;
    let target_units = max_units_ram.min((total_units / 4).max(1));

    #[derive(Clone, Copy)]
    struct Sw2Step {
        p_start: usize,
        p_count: usize,
        u_start: usize,
        u_count: usize,
    }

    let mut sw2_steps: Vec<Sw2Step> = Vec::new();
    let (sw2_max_p, sw2_max_u) = if target_units >= lay.row_units_per_prime {
        let mut p_step = (target_units / lay.row_units_per_prime).clamp(1, k);
        if p_step == k && k >= 2 {
            p_step = (k / 2).max(1);
        }
        let mut p0 = 0usize;
        while p0 < k {
            let p_cnt = (k - p0).min(p_step);
            sw2_steps.push(Sw2Step {
                p_start: p0,
                p_count: p_cnt,
                u_start: 0,
                u_count: lay.row_units_per_prime,
            });
            p0 += p_cnt;
        }
        (p_step, lay.row_units_per_prime)
    } else {
        let mut u_step = (1usize << target_units.ilog2()).min(lay.row_units_per_prime).max(1);
        while u_step > 1
            && upper_slot_buf_bytes(st, u_step * lay.pieces_per_row) > stream_half_budget
        {
            u_step >>= 1;
        }
        for pi in 0..k {
            let mut u0 = 0usize;
            while u0 < lay.row_units_per_prime {
                let u_cnt = (lay.row_units_per_prime - u0).min(u_step);
                sw2_steps.push(Sw2Step {
                    p_start: pi,
                    p_count: 1,
                    u_start: u0,
                    u_count: u_cnt,
                });
                u0 += u_cnt;
            }
        }
        (1usize, u_step)
    };

    struct Sw2IoSet {
        in_ops: [Option<PrimeSlotStage>; 4],
        wb_mask: [bool; 3],
        prod_out_ram: Option<PrimeSlotStage>,
    }

    impl Sw2IoSet {
        fn new(
            st: &Store,
            in_mask: [bool; 4],
            wb_mask: [bool; 3],
            max_p: usize,
            buf_bytes: usize,
        ) -> Self {
            Self {
                in_ops: std::array::from_fn(|i| {
                    in_mask[i].then(|| PrimeSlotStage::new(st, max_p, buf_bytes))
                }),
                wb_mask,
                prod_out_ram: (!in_mask[0]).then(|| PrimeSlotStage::new(st, max_p, buf_bytes)),
            }
        }

        fn read_inputs(&mut self, st: &Store, telemetry: &Telemetry) -> Result<(), StoreError> {
            for stage in self.in_ops.iter_mut().flatten() {
                stage.read_all_from_store(st, telemetry)?;
            }
            Ok(())
        }

        fn write_outputs(&self, st: &Store) -> Result<(), StoreError> {
            for idx in 0..3 {
                if self.wb_mask[idx]
                    && let Some(ref stage) = self.in_ops[idx + 1]
                {
                    stage.write_all_to_store(st)?;
                }
            }
            if let Some(ref pout) = self.prod_out_ram {
                pout.write_all_to_store(st)
            } else if let Some(ref a1_stg) = self.in_ops[0] {
                a1_stg.write_all_to_store(st)
            } else {
                Ok(())
            }
        }
    }

    let sw2_buf_each = upper_slot_buf_bytes(st, sw2_max_u * lay.pieces_per_row);
    let mut sw2_sets = [
        Sw2IoSet::new(st, in_store_mask, wb_mask, sw2_max_p, sw2_buf_each),
        Sw2IoSet::new(st, in_store_mask, wb_mask, sw2_max_p, sw2_buf_each),
    ];

    let plan_sw2_step = |set: &mut Sw2IoSet, step: Sw2Step| -> Result<(), StoreError> {
        let range = [ExtentRange {
            piece_start: (step.u_start * lay.pieces_per_row) as u64,
            piece_count: (step.u_count * lay.pieces_per_row) as u64,
            host_limb_offset: 0,
        }];
        let op_refs: [Option<&OpRef<'_>>; 4] = [
            Some(&op_a1),
            op_b1.as_ref(),
            op_a2.as_ref(),
            op_b2.as_ref(),
        ];
        for idx in 0..4 {
            if let (Some(stage), Some(OpRef::Store(handles, _))) = (&mut set.in_ops[idx], op_refs[idx]) {
                stage.plans.clear();
                for pi_loc in 0..step.p_count {
                    let h = &handles[step.p_start + pi_loc];
                    stage.plans.push(st.devices[h.device_idx].plan_slot_batch(h, &range)?);
                }
                stage.ensure_buf_capacities();
            }
        }
        if let Some(ref mut pout) = set.prod_out_ram {
            pout.plans.clear();
            for pi_loc in 0..step.p_count {
                let h = &prod_guard.handles[step.p_start + pi_loc];
                pout.plans
                    .push(st.devices[h.device_idx].plan_slot_batch(h, &range)?);
            }
            pout.ensure_buf_capacities();
        }
        Ok(())
    };

    let row_negacyclic = spec.rows == 1 && spec.negacyclic;
    let row0_tables: Vec<&PrimeTables> =
        cache.get_twiddles_refs_par(sched, spec.par_workers, &crt.primes_u64, k, cols, row_negacyclic);
    let ntt_l1_sub = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let ntt_l2_sub = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    let prime_row_meta: Vec<(NttPrime, NttPrime, u64)> = (0..k)
        .map(|pi| {
            let mut pf = row0_tables[pi].pr_fwd;
            let mut pv = row0_tables[pi].pr_inv;
            pf.l1_sub = ntt_l1_sub;
            pf.l2_sub = ntt_l2_sub;
            pv.l1_sub = ntt_l1_sub;
            pv.l2_sub = ntt_l2_sub;
            (pf, pv, row0_tables[pi].inv_n)
        })
        .collect();

    let run_sw2_workers = |set: &mut Sw2IoSet, step: Sw2Step| -> Result<(), StoreError> {
        let in_addrs: [Vec<usize>; 4] = std::array::from_fn(|i| {
            set.in_ops[i].as_mut().map_or_else(Vec::new, |stg| {
                stg.bufs[..step.p_count]
                    .iter_mut()
                    .map(|b| b.as_mut_ptr() as usize)
                    .collect()
            })
        });
        let wb_addrs: [&[usize]; 3] = [
            in_addrs[1].as_slice(),
            in_addrs[2].as_slice(),
            in_addrs[3].as_slice(),
        ];
        let prod_ram_addrs: Vec<usize> = set.prod_out_ram.as_mut().map_or_else(Vec::new, |pout| {
            pout.bufs[..step.p_count]
                .iter_mut()
                .map(|b| b.as_mut_ptr() as usize)
                .collect()
        });
        let prod_addrs: &[usize] = if set.prod_out_ram.is_some() {
            prod_ram_addrs.as_slice()
        } else {
            in_addrs[0].as_slice()
        };

        let in_plans: [&[SlotBatchPlan]; 4] = std::array::from_fn(|i| {
            set.in_ops[i]
                .as_ref()
                .map_or(&[][..], |stg| stg.plans.as_slice())
        });
        let wb_plans: [&[SlotBatchPlan]; 3] = [in_plans[1], in_plans[2], in_plans[3]];
        let prod_plans: &[SlotBatchPlan] = if let Some(ref pout) = set.prod_out_ram {
            pout.plans.as_slice()
        } else {
            in_plans[0]
        };

        let err_flag = AtomicBool::new(false);
        let first_err: Mutex<Option<StoreError>> = Mutex::new(None);
        let sw2_w = sw2_workers(sched, &spec);
        let total_tasks = step.p_count * step.u_count;
        let count_1d_scrub = spec.rows == 1;

        let load_op_row = |op_idx: usize,
                           op_ref: &OpRef<'_>,
                           pi: usize,
                           pi_loc: usize,
                           u_loc: usize,
                           sub_r: usize,
                           r: usize,
                           dst_f64: &mut [f64]|
         -> Result<(), StoreError> {
            match op_ref {
                OpRef::Ram(bufs, op_spec) => {
                    let rb = op_spec.outer_row_bytes();
                    let mcfg = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: rb as u64,
                        packed: u64::from(op_spec.packed),
                        use_nt: 0,
                    };
                    unsafe {
                        let src = bufs[pi].as_ptr().add(r * rb);
                        (kernels.active.gather)(dst_f64.as_mut_ptr(), src, &mcfg);
                    }
                    Ok(())
                }
                OpRef::Store(_, op_spec) => {
                    let op_lay = StoreGridLayout::build(op_spec, st.devices[0].piece_limbs);
                    let plan = &in_plans[op_idx][pi_loc];
                    let dev = &st.devices[plan.extent.device_idx];
                    let w = dev.stripe_width();
                    let base_ptr = in_addrs[op_idx][pi_loc] as *mut u8;
                    if op_lay.pieces_per_row == 1 {
                        let pe = &plan.pieces[u_loc];
                        if sub_r == 0 {
                            let slot_mut = unsafe {
                                std::slice::from_raw_parts_mut(
                                    base_ptr.add(pe.buf_byte_off),
                                    plan.piece_slot_bytes,
                                )
                            };
                            dev.scrub_or_repair_slot_in_place(
                                &plan.extent,
                                pe.piece_id,
                                slot_mut,
                                kernels,
                                telemetry,
                                &st.bytes_moved,
                                count_1d_scrub,
                            )?;
                        }
                        let mcfg = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: op_lay.row_stride_in_piece as u64,
                            packed: u64::from(op_lay.packed),
                            use_nt: 0,
                        };
                        unsafe {
                            let src =
                                base_ptr.add(pe.buf_byte_off + sub_r * op_lay.row_stride_in_piece);
                            (kernels.active.gather)(dst_f64.as_mut_ptr(), src, &mcfg);
                        }
                    } else {
                        let sh = row_piece_shift(r, op_lay.pieces_per_row, w);
                        let mcfg = MoveCfg {
                            rows: 1,
                            cols: op_lay.cols_per_piece as u64,
                            outer_stride_bytes: op_lay.row_stride_in_piece as u64,
                            packed: u64::from(op_lay.packed),
                            use_nt: 0,
                        };
                        for p_in_row in 0..op_lay.pieces_per_row {
                            let pe = &plan.pieces[u_loc * op_lay.pieces_per_row + p_in_row];
                            let slot_mut = unsafe {
                                std::slice::from_raw_parts_mut(
                                    base_ptr.add(pe.buf_byte_off),
                                    plan.piece_slot_bytes,
                                )
                            };
                            dev.scrub_or_repair_slot_in_place(
                                &plan.extent,
                                pe.piece_id,
                                slot_mut,
                                kernels,
                                telemetry,
                                &st.bytes_moved,
                                count_1d_scrub,
                            )?;
                            let cp = p_in_row ^ sh;
                            unsafe {
                                (kernels.active.gather)(
                                    dst_f64.as_mut_ptr().add(cp * op_lay.cols_per_piece),
                                    slot_mut.as_ptr(),
                                    &mcfg,
                                );
                            }
                        }
                    }
                    Ok(())
                }
            }
        };

        let store_out_row = |plans: &[SlotBatchPlan],
                             addrs: &[usize],
                             out_lay: &StoreGridLayout,
                             pi_loc: usize,
                             u_loc: usize,
                             sub_r: usize,
                             r: usize,
                             src_f64: &[f64],
                             count_traffic: bool| {
            let plan = &plans[pi_loc];
            let dev = &st.devices[plan.extent.device_idx];
            let w = dev.stripe_width();
            let base_ptr = addrs[pi_loc] as *mut u8;
            if out_lay.pieces_per_row == 1 {
                let pe = &plan.pieces[u_loc];
                let mcfg = MoveCfg {
                    rows: 1,
                    cols: cols as u64,
                    outer_stride_bytes: out_lay.row_stride_in_piece as u64,
                    packed: u64::from(out_lay.packed),
                    use_nt: ROW_SWEEP_STREAMS,
                };
                unsafe {
                    let slot_ptr = base_ptr.add(pe.buf_byte_off);
                    (kernels.active.scatter)(
                        slot_ptr.add(sub_r * out_lay.row_stride_in_piece),
                        src_f64.as_ptr(),
                        &mcfg,
                    );
                    if sub_r + 1 == out_lay.rows_per_piece {
                        if out_lay.used_bytes_per_piece < out_lay.piece_data_bytes {
                            std::ptr::write_bytes(
                                slot_ptr.add(out_lay.used_bytes_per_piece),
                                0,
                                out_lay.piece_data_bytes - out_lay.used_bytes_per_piece,
                            );
                        }
                        let slot_mut = std::slice::from_raw_parts_mut(slot_ptr, plan.piece_slot_bytes);
                        dev.compute_slot_check_in_place(slot_mut, kernels, count_traffic);
                    }
                }
            } else {
                let sh = row_piece_shift(r, out_lay.pieces_per_row, w);
                let mcfg = MoveCfg {
                    rows: 1,
                    cols: out_lay.cols_per_piece as u64,
                    outer_stride_bytes: out_lay.row_stride_in_piece as u64,
                    packed: u64::from(out_lay.packed),
                    use_nt: ROW_SWEEP_STREAMS,
                };
                for cp in 0..out_lay.pieces_per_row {
                    let p_in_row = cp ^ sh;
                    let pe = &plan.pieces[u_loc * out_lay.pieces_per_row + p_in_row];
                    unsafe {
                        let slot_ptr = base_ptr.add(pe.buf_byte_off);
                        (kernels.active.scatter)(
                            slot_ptr,
                            src_f64.as_ptr().add(cp * out_lay.cols_per_piece),
                            &mcfg,
                        );
                        if out_lay.used_bytes_per_piece < out_lay.piece_data_bytes {
                            std::ptr::write_bytes(
                                slot_ptr.add(out_lay.used_bytes_per_piece),
                                0,
                                out_lay.piece_data_bytes - out_lay.used_bytes_per_piece,
                            );
                        }
                        // Section 11: compute piece check words inline while data is hot in worker cache!
                        let slot_mut = std::slice::from_raw_parts_mut(slot_ptr, plan.piece_slot_bytes);
                        dev.compute_slot_check_in_place(slot_mut, kernels, count_traffic);
                    }
                }
            }
        };

        sched.par_for_init_max(
            total_tasks,
            1,
            sw2_w,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.ra1, cols);
                if !same_ab1 {
                    resize_f64_buf(&mut sc.rb1, cols);
                }
                if has_second {
                    resize_f64_buf(&mut sc.ra2, cols);
                    resize_f64_buf(&mut sc.rb2, cols);
                }
                resize_f64_buf(&mut sc.rc, cols);
                lease
            },
            |lease, task_id| {
                if err_flag.load(Ordering::Relaxed) {
                    return;
                }
                let sc = lease.get_mut();
                let pi_loc = task_id / step.u_count;
                let u_loc = task_id % step.u_count;
                let pi = step.p_start + pi_loc;
                let u_abs = step.u_start + u_loc;
                let p_u64 = crt.primes_u64[pi];
                let (ref pr_fwd, ref pr_inv, inv_cols) = prime_row_meta[pi];
                let r_tw = &row0_tables[pi].tw;
                let r_inv_tw = &row0_tables[pi].inv_tw;

                for sub_r in 0..lay.rows_per_piece {
                    let r = u_abs * lay.rows_per_piece + sub_r;
                    if let Err(e) =
                        load_op_row(0, &op_a1, pi, pi_loc, u_loc, sub_r, r, &mut sc.ra1[..cols])
                    {
                        err_flag.store(true, Ordering::Relaxed);
                        let mut g = first_err.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                        return;
                    }
                    if !a1_rt {
                        row_fwd_multi_sweep(
                            kernels,
                            &mut sc.ra1[..cols],
                            r_tw,
                            pr_fwd,
                            spec.extra_sweeps,
                        );
                    }

                    let pb1_ptr = if let Some(ref b1_ref) = op_b1 {
                        if let Err(e) =
                            load_op_row(1, b1_ref, pi, pi_loc, u_loc, sub_r, r, &mut sc.rb1[..cols])
                        {
                            err_flag.store(true, Ordering::Relaxed);
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                        let b1_spec = match b1_ref {
                            OpRef::Store(_, s) | OpRef::Ram(_, s) => *s,
                        };
                        if !b1_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.rb1[..cols],
                                r_tw,
                                pr_fwd,
                                b1_spec.extra_sweeps,
                            );
                            if b1_wb {
                                let wb_lay =
                                    StoreGridLayout::build(&b1_spec, st.devices[0].piece_limbs);
                                store_out_row(
                                    wb_plans[0],
                                    wb_addrs[0],
                                    &wb_lay,
                                    pi_loc,
                                    u_loc,
                                    sub_r,
                                    r,
                                    &sc.rb1[..cols],
                                    false,
                                );
                            }
                        }
                        sc.rb1.as_ptr()
                    } else {
                        sc.ra1.as_ptr()
                    };

                    if let (Some(a2_ref), Some(b2_ref)) = (&op_a2, &op_b2) {
                        if let Err(e) =
                            load_op_row(2, a2_ref, pi, pi_loc, u_loc, sub_r, r, &mut sc.ra2[..cols])
                        {
                            err_flag.store(true, Ordering::Relaxed);
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                        let a2_spec = match a2_ref {
                            OpRef::Store(_, s) | OpRef::Ram(_, s) => *s,
                        };
                        if !a2_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.ra2[..cols],
                                r_tw,
                                pr_fwd,
                                a2_spec.extra_sweeps,
                            );
                            if a2_wb {
                                let wb_lay =
                                    StoreGridLayout::build(&a2_spec, st.devices[0].piece_limbs);
                                store_out_row(
                                    wb_plans[1],
                                    wb_addrs[1],
                                    &wb_lay,
                                    pi_loc,
                                    u_loc,
                                    sub_r,
                                    r,
                                    &sc.ra2[..cols],
                                    false,
                                );
                            }
                        }

                        if let Err(e) =
                            load_op_row(3, b2_ref, pi, pi_loc, u_loc, sub_r, r, &mut sc.rb2[..cols])
                        {
                            err_flag.store(true, Ordering::Relaxed);
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                        let b2_spec = match b2_ref {
                            OpRef::Store(_, s) | OpRef::Ram(_, s) => *s,
                        };
                        if !b2_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.rb2[..cols],
                                r_tw,
                                pr_fwd,
                                b2_spec.extra_sweeps,
                            );
                            if b2_wb {
                                let wb_lay =
                                    StoreGridLayout::build(&b2_spec, st.devices[0].piece_limbs);
                                store_out_row(
                                    wb_plans[2],
                                    wb_addrs[2],
                                    &wb_lay,
                                    pi_loc,
                                    u_loc,
                                    sub_r,
                                    r,
                                    &sc.rb2[..cols],
                                    false,
                                );
                            }
                        }

                        let ptrs = [sc.ra1.as_ptr(), pb1_ptr, sc.ra2.as_ptr(), sc.rb2.as_ptr()];
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_inv.p,
                            inv_p: pr_inv.inv_p,
                            sub_second: u64::from(sub_second),
                        };
                        unsafe {
                            (kernels.active.ntt_pmuladd)(sc.rc.as_mut_ptr(), &ptrs, &vcfg);
                        }
                    } else {
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_inv.p,
                            inv_p: pr_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            (kernels.active.ntt_pmul)(
                                sc.rc.as_mut_ptr(),
                                sc.ra1.as_ptr(),
                                pb1_ptr,
                                &vcfg,
                            );
                        }
                    }

                    row_inv_multi_sweep(
                        kernels,
                        &mut sc.rc[..cols],
                        r_inv_tw,
                        p_u64,
                        inv_cols,
                        spec.extra_sweeps,
                    );
                    store_out_row(
                        prod_plans,
                        &prod_addrs,
                        &lay,
                        pi_loc,
                        u_loc,
                        sub_r,
                        r,
                        &sc.rc[..cols],
                        count_1d_scrub,
                    );
                }
            },
        );

        if let Some(e) = first_err.into_inner().unwrap() {
            return Err(e);
        }
        Ok(())
    };

    // Execute Sweep 2 with 3-way `Loader` overlap (`write(s - 1)` + `read(s + 1)` beside `compute(s)`):
    let t_sw2 = std::time::Instant::now();
    {
        let (s_left, s_right) = sw2_sets.split_at_mut(1);
        plan_sw2_step(&mut s_left[0], sw2_steps[0])?;
        let s0_ref = &mut s_left[0];
        st.submit_loader_scoped(|| s0_ref.read_inputs(st, telemetry)).wait()?;

        let num_steps = sw2_steps.len();
        for s_idx in 0..num_steps {
            let (cur_set, io_set) = if (s_idx & 1) == 0 {
                (&mut s_left[0], &mut s_right[0])
            } else {
                (&mut s_right[0], &mut s_left[0])
            };
            let need_write_prev = s_idx > 0;
            let next_step_opt = (s_idx + 1 < num_steps).then(|| sw2_steps[s_idx + 1]);
            let loader_h = (need_write_prev || next_step_opt.is_some()).then(|| {
                st.submit_loader_scoped(|| -> Result<(), StoreError> {
                    if need_write_prev {
                        io_set.write_outputs(st)?;
                    }
                    if let Some(next_step) = next_step_opt {
                        plan_sw2_step(io_set, next_step)?;
                        io_set.read_inputs(st, telemetry)?;
                    }
                    Ok(())
                })
            });

            let compute_res = run_sw2_workers(cur_set, sw2_steps[s_idx]);
            if let Some(h) = loader_h {
                h.wait()?;
            }
            compute_res?;
        }

        let last_set = if ((num_steps - 1) & 1) == 0 {
            &s_left[0]
        } else {
            &s_right[0]
        };
        st.submit_loader_scoped(|| last_set.write_outputs(st)).wait()?;
    }
    NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
    NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

    // Drop Sweep 2 staging sets and any `a1` RAM buffers before Sweep 3 allocates its tile sets:
    drop(sw2_sets);
    drop(a1_ram_guard);

    if b1_wb && let Some(NttOperand::ReuseInPlace(op)) = &mut b1 {
        op.rows_transformed = true;
    }
    if a2_wb && let Some((NttOperand::ReuseInPlace(op), _, _)) = &mut second {
        op.rows_transformed = true;
    }
    if b2_wb && let Some((_, NttOperand::ReuseInPlace(op), _)) = &mut second {
        op.rows_transformed = true;
    }

    let fwd_row_ops = u64::from(!a1_rt)
        + u64::from(!same_ab1 && !b1_rt)
        + if has_second {
            u64::from(!a2_rt) + u64::from(!b2_rt)
        } else {
            0
        };

    // ========================================================================
    // Out-of-core Sweep 3: Strided column-piece batches + Loader prefetch + CRT
    // ========================================================================
    let t_sw3 = std::time::Instant::now();
    let col_tables = if rows > 1 {
        cache.get_twiddles_refs_par(
            sched,
            spec.par_workers,
            &crt.primes_u64,
            k,
            rows,
            spec.negacyclic,
        )
    } else {
        Vec::new()
    };
    let psi_inv: Vec<PsiParams> = if rows > 1 {
        (0..k)
            .map(|pi| {
                build_psi_params(
                    crt.primes_u64[pi],
                    col_tables[pi].g_root,
                    spec.n,
                    spec.negacyclic,
                    true,
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let prime_col_inv: Vec<NttPrime> = if rows > 1 {
        (0..k).map(|pi| col_tables[pi].pr_inv).collect()
    } else {
        Vec::new()
    };

    let b_usize = spec.coeff_bits as usize;
    let seg_cols_opt = if lay.cols_per_piece >= 32
        && lay.cols_per_piece.is_multiple_of(32)
        && (32 * b_usize).is_multiple_of(64)
        && rows >= 128
    {
        Some(32usize)
    } else if lay.cols_per_piece >= 64
        && lay.cols_per_piece.is_multiple_of(64)
        && (64 * b_usize).is_multiple_of(64)
    {
        Some(64usize)
    } else if lay.cols_per_piece >= 32
        && lay.cols_per_piece.is_multiple_of(32)
        && (32 * b_usize).is_multiple_of(64)
    {
        Some(32usize)
    } else if (lay.cols_per_piece * b_usize).is_multiple_of(64) {
        Some(lay.cols_per_piece)
    } else {
        None
    };

    let half_sw3_budget = (budget / 2).max(1);
    let min_range_bytes = st.devices.first().map_or(0, |d| d.min_batch_range_bytes);
    let max_slot_bytes = st
        .devices
        .iter()
        .map(|d| d.piece_slot_bytes as usize)
        .max()
        .unwrap_or(65_536);

    let (sw3_cp_batch, sw3_p_sub) = if lay.pieces_per_row == 1 {
        let per_p = upper_slot_buf_bytes(st, lay.total_pieces_per_prime);
        let p_sub = (half_sw3_budget / per_p).clamp(1, k);
        (1usize, p_sub)
    } else {
        let min_cp_raw = min_range_bytes.div_ceil(max_slot_bytes as u64).max(1) as usize;
        let mut cp_b = min_cp_raw.next_power_of_two().min(lay.pieces_per_row);
        while cp_b > 1 && k.saturating_mul(upper_slot_buf_bytes(st, rows * cp_b)) > half_sw3_budget {
            cp_b >>= 1;
        }
        let per_p1 = upper_slot_buf_bytes(st, rows * cp_b);
        if k.saturating_mul(per_p1) <= half_sw3_budget {
            while cp_b * 2 <= (lay.pieces_per_row / 2).max(1) {
                let next_p = upper_slot_buf_bytes(st, rows * cp_b * 2);
                if k.saturating_mul(next_p) <= half_sw3_budget {
                    cp_b *= 2;
                } else {
                    break;
                }
            }
            (cp_b, k)
        } else {
            let p_sub = (half_sw3_budget / per_p1).clamp(1, k);
            (1usize, p_sub)
        }
    };

    let sw3_pieces_per_p = if lay.pieces_per_row == 1 {
        lay.total_pieces_per_prime
    } else {
        rows * sw3_cp_batch
    };
    let sw3_buf_each = upper_slot_buf_bytes(st, sw3_pieces_per_p);
    let mut sw3_stages = [
        PrimeSlotStage::new(st, sw3_p_sub, sw3_buf_each),
        PrimeSlotStage::new(st, sw3_p_sub, sw3_buf_each),
    ];

    let mut sw3_batches: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut cp0 = 0usize;
    while cp0 < lay.pieces_per_row {
        let cp_cnt = (lay.pieces_per_row - cp0).min(sw3_cp_batch);
        let mut p0 = 0usize;
        while p0 < k {
            let p_cnt = (k - p0).min(sw3_p_sub);
            sw3_batches.push((p0, p_cnt, cp0, cp_cnt));
            p0 += p_cnt;
        }
        cp0 += cp_cnt;
    }

    let plan_sw3_batch =
        |stage: &mut PrimeSlotStage, (p0, p_cnt, cp0, cp_cnt): (usize, usize, usize, usize)| -> Result<(), StoreError> {
            stage.plans.clear();
            for pi_loc in 0..p_cnt {
                let h = &prod_guard.handles[p0 + pi_loc];
                let dev = &st.devices[h.device_idx];
                let ranges = lay.col_batch_ranges(cp0, cp_cnt, dev.stripe_width());
                stage.plans.push(dev.plan_slot_batch(h, &ranges)?);
            }
            Ok(())
        };

    let scrub_sw3_stage = |stage: &mut PrimeSlotStage, p_cnt: usize| -> Result<(), StoreError> {
        let buf_addrs: Vec<usize> = stage.bufs[..p_cnt]
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();
        let plans_ref = stage.plans.as_slice();
        let pieces_each = plans_ref[0].pieces.len();
        let total_slots = p_cnt * pieces_each;
        let err_flag = AtomicBool::new(false);
        let first_err: Mutex<Option<StoreError>> = Mutex::new(None);
        let count_sw3_check = spec.rows > 1;

        sched.par_for_leaf_max(total_slots, 1, spec.par_workers, |slot_idx| {
            if err_flag.load(Ordering::Relaxed) {
                return;
            }
            let pi_loc = slot_idx / pieces_each;
            let pe_idx = slot_idx % pieces_each;
            let plan = &plans_ref[pi_loc];
            let dev = &st.devices[plan.extent.device_idx];
            let pe = &plan.pieces[pe_idx];
            let slot_mut = unsafe {
                std::slice::from_raw_parts_mut(
                    (buf_addrs[pi_loc] as *mut u8).add(pe.buf_byte_off),
                    plan.piece_slot_bytes,
                )
            };
            if let Err(e) = dev.scrub_or_repair_slot_in_place(
                &plan.extent,
                pe.piece_id,
                slot_mut,
                kernels,
                telemetry,
                &st.bytes_moved,
                count_sw3_check,
            ) {
                err_flag.store(true, Ordering::Relaxed);
                let mut g = first_err.lock().unwrap();
                if g.is_none() {
                    *g = Some(e);
                }
            }
        });
        if let Some(e) = first_err.into_inner().unwrap() {
            return Err(e);
        }
        Ok(())
    };

    if let Some(seg_cols) = seg_cols_opt {
        let tw_seg0_inv = if rows > 1 {
            cache.get_tile0_twiddles_par(
                kernels,
                sched,
                spec.par_workers,
                &crt.primes_u64,
                &col_tables,
                k,
                spec.n,
                rows,
                seg_cols,
                spec.negacyclic,
                true,
            )
        } else {
            Vec::new()
        };

        let num_col_groups = cols / seg_cols;
        let groups_per_cp = lay.cols_per_piece / seg_cols;
        let seg_limbs = (seg_cols * b_usize) / 64;
        let can_skip_crt_2d = !signed && !spec.negacyclic && !spec.wrap_cyclic;
        let needed_rows = if can_skip_crt_2d {
            (out_limbs + 2)
                .div_ceil(num_col_groups * seg_limbs)
                .min(rows)
                .max(1)
        } else {
            rows
        };
        let total_segs = needed_rows * num_col_groups;
        let total_limbs = total_segs * seg_limbs;
        let m_bits = (spec.n as u64) * (spec.coeff_bits as u64);
        let full_limbs = (total_limbs + crt.k_words + 2).max(out_limbs);

        let mut out = crate::limb::take_limb_buf_committed(full_limbs, sched, spec.par_workers);
        let mut seg_carries = cache.take_u64_buf_uninit(total_segs * crt.k_words);
        let mut seg_w0s = cache.take_u64_buf_uninit(total_segs);
        let out_ptr = out.as_mut_ptr() as usize;
        let carries_ptr = seg_carries.as_mut_ptr() as usize;
        let w0s_ptr = seg_w0s.as_mut_ptr() as usize;

        // When `sw3_p_sub < k` (ultra-tiny synthetic RAM budget), stage untwisted primes of 1 `cp` here:
        let mut sub_prime_slab: Vec<f64, TransformsAlloc> = if sw3_p_sub < k {
            let mut v = Vec::with_capacity_in(k * rows * lay.cols_per_piece, TransformsAlloc);
            v.resize(k * rows * lay.cols_per_piece, 0.0);
            v
        } else {
            Vec::new_in(TransformsAlloc)
        };
        let slab_ptr = sub_prime_slab.as_mut_ptr() as usize;

        let (s_left, s_right) = sw3_stages.split_at_mut(1);
        plan_sw3_batch(&mut s_left[0], sw3_batches[0])?;
        let s0_ref = &mut s_left[0];
        st.submit_loader_scoped(|| s0_ref.read_all_from_store(st, telemetry)).wait()?;

        for b_idx in 0..sw3_batches.len() {
            let (p0, p_cnt, cp0, cp_cnt) = sw3_batches[b_idx];
            let (cur_s, next_s) = if (b_idx & 1) == 0 {
                (&mut s_left[0], &mut s_right[0])
            } else {
                (&mut s_right[0], &mut s_left[0])
            };
            let has_next = b_idx + 1 < sw3_batches.len();
            if has_next {
                plan_sw3_batch(next_s, sw3_batches[b_idx + 1])?;
            }
            let loader_h = has_next.then(|| {
                st.submit_loader_scoped(|| next_s.read_all_from_store(st, telemetry))
            });

            let scrub_res = scrub_sw3_stage(cur_s, p_cnt);
            if scrub_res.is_ok() {
                let buf_addrs: Vec<usize> = cur_s.bufs[..p_cnt]
                    .iter()
                    .map(|b| b.as_ptr() as usize)
                    .collect();
                let plans_ref = cur_s.plans.as_slice();
                let cgs_in_batch = cp_cnt * groups_per_cp;
                let seg_elems = rows * seg_cols;

                if sw3_p_sub == k {
                    sched.par_for_init_max(
                        cgs_in_batch,
                        1,
                        spec.par_workers,
                        || {
                            let mut lease = ScratchLease::acquire();
                            let sc = lease.get_mut();
                            if sc.prime_segs.len() < k {
                                sc.prime_segs.resize_with(k, || Vec::new_in(ScratchAlloc));
                            }
                            for pi in 0..k {
                                resize_f64_buf(&mut sc.prime_segs[pi], pi * 40 + seg_elems);
                            }
                            if sc.row_ptrs.len() < k {
                                sc.row_ptrs.resize(k, 0);
                            }
                            lease
                        },
                        |lease, cg_loc| {
                            let sc = lease.get_mut();
                            let cg = cp0 * groups_per_cp + cg_loc;
                            let col_start = cg * seg_cols;
                            let cp = col_start / lay.cols_per_piece;
                            let local_col = col_start % lay.cols_per_piece;
                            let local_col_byte_off = ooc_elem_bytes(local_col, lay.packed);
                            let shape = [rows as u64, seg_cols as u64];

                            for pi in 0..k {
                                let plan = &plans_ref[pi];
                                let w = st.devices[plan.extent.device_idx].stripe_width();
                                let base_ptr = buf_addrs[pi] as *const u8;
                                let seg_slice =
                                    &mut sc.prime_segs[pi][(pi * 40)..(pi * 40 + seg_elems)];

                                if lay.pieces_per_row == 1 {
                                    let mcfg = MoveCfg {
                                        rows: lay.rows_per_piece as u64,
                                        cols: seg_cols as u64,
                                        outer_stride_bytes: lay.row_stride_in_piece as u64,
                                        packed: u64::from(lay.packed),
                                        use_nt: 0,
                                    };
                                    for p_idx in 0..lay.total_pieces_per_prime {
                                        let slot_off = plan.pieces[p_idx].buf_byte_off;
                                        unsafe {
                                            (kernels.active.gather)(
                                                seg_slice
                                                    .as_mut_ptr()
                                                    .add(p_idx * lay.rows_per_piece * seg_cols),
                                                base_ptr.add(slot_off + local_col_byte_off),
                                                &mcfg,
                                            );
                                        }
                                    }
                                } else {
                                    let mcfg_row = MoveCfg {
                                        rows: 1,
                                        cols: seg_cols as u64,
                                        outer_stride_bytes: lay.row_stride_in_piece as u64,
                                        packed: u64::from(lay.packed),
                                        use_nt: 0,
                                    };
                                    for r in 0..rows {
                                        let (slot_off, _) =
                                            lay.col_slot_and_off(plan, r, cp0, cp_cnt, cp, w);
                                        unsafe {
                                            let src = base_ptr.add(slot_off + local_col_byte_off);
                                            if lay.packed {
                                                (kernels.active.gather)(
                                                    seg_slice.as_mut_ptr().add(r * seg_cols),
                                                    src,
                                                    &mcfg_row,
                                                );
                                            } else {
                                                std::ptr::copy_nonoverlapping(
                                                    src as *const f64,
                                                    seg_slice.as_mut_ptr().add(r * seg_cols),
                                                    seg_cols,
                                                );
                                            }
                                        }
                                    }
                                }

                                if rows > 1 {
                                    apply_tile_twist_inv_scaled(
                                        kernels,
                                        seg_slice,
                                        &tw_seg0_inv[pi],
                                        rows,
                                        seg_cols,
                                        col_start,
                                        psi_inv[pi],
                                        crt.primes_u64[pi],
                                        prime_col_inv[pi].p,
                                        prime_col_inv[pi].inv_p,
                                        &mut sc.tw_row,
                                        &mut sc.s_u64,
                                    );
                                    unsafe {
                                        call_ntt_inv_cols(
                                            kernels,
                                            seg_slice.as_mut_ptr(),
                                            col_tables[pi].inv_tw.as_ptr(),
                                            &shape,
                                            &col_tables[pi].pr_fwd,
                                            &mut sc.col_scratch,
                                        );
                                    }
                                }
                            }

                            for pi in 0..k {
                                sc.row_ptrs[pi] =
                                    unsafe { sc.prime_segs[pi].as_ptr().add(pi * 40) } as usize;
                            }
                            let mut ccfg = NttCrtCfg {
                                k_primes: k as u64,
                                count: seg_cols as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                out_limbs: seg_limbs as u64,
                                signed: u64::from(signed),
                                wrap_cyclic: 0,
                                primes: crt.primes_f64.as_ptr(),
                                inv_primes: crt.inv_primes_f64.as_ptr(),
                                garner_inv: crt.garner_inv.as_ptr(),
                                prefix_prods: crt.prefix_prods.as_ptr(),
                                k_words: crt.k_words as u64,
                                segs: NttCrtSegs {
                                    segments: needed_rows as u64,
                                    res_stride: seg_cols as u64,
                                    out_stride: (num_col_groups * seg_limbs) as u64,
                                    carry_stride: (num_col_groups * crt.k_words) as u64,
                                    w0: unsafe { (w0s_ptr as *mut u64).add(cg) },
                                    w0_stride: num_col_groups as u64,
                                },
                                carry_limbs: unsafe {
                                    (carries_ptr as *mut u64).add(cg * crt.k_words)
                                },
                            };
                            unsafe {
                                ntt_crt_streams(
                                    kernels,
                                    (out_ptr as *mut u64).add(cg * seg_limbs),
                                    sc.row_ptrs.as_ptr() as *const *const f64,
                                    &mut ccfg,
                                );
                            }
                        },
                    );
                } else {
                    // Sub-batched primes for 1 `cp`: untwist + `inv_cols` into `sub_prime_slab`,
                    // then run `ntt_crt_streams` once `p0 + p_cnt == k`!
                    let tasks = p_cnt * cgs_in_batch;
                    sched.par_for_init_max(
                        tasks,
                        1,
                        spec.par_workers,
                        || {
                            let mut lease = ScratchLease::acquire();
                            resize_f64_buf(&mut lease.get_mut().tile, seg_elems);
                            lease
                        },
                        |lease, tid| {
                            let sc = lease.get_mut();
                            let pi_loc = tid / cgs_in_batch;
                            let cg_loc = tid % cgs_in_batch;
                            let pi = p0 + pi_loc;
                            let cg = cp0 * groups_per_cp + cg_loc;
                            let col_start = cg * seg_cols;
                            let cp = col_start / lay.cols_per_piece;
                            let local_col = col_start % lay.cols_per_piece;
                            let local_col_byte_off = ooc_elem_bytes(local_col, lay.packed);
                            let shape = [rows as u64, seg_cols as u64];
                            let plan = &plans_ref[pi_loc];
                            let w = st.devices[plan.extent.device_idx].stripe_width();
                            let base_ptr = buf_addrs[pi_loc] as *const u8;

                            let dst_seg = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (slab_ptr as *mut f64)
                                        .add((pi * groups_per_cp + cg_loc) * seg_elems),
                                    seg_elems,
                                )
                            };
                            if lay.pieces_per_row == 1 {
                                let mcfg = MoveCfg {
                                    rows: lay.rows_per_piece as u64,
                                    cols: seg_cols as u64,
                                    outer_stride_bytes: lay.row_stride_in_piece as u64,
                                    packed: u64::from(lay.packed),
                                    use_nt: 0,
                                };
                                for p_idx in 0..lay.total_pieces_per_prime {
                                    let slot_off = plan.pieces[p_idx].buf_byte_off;
                                    unsafe {
                                        (kernels.active.gather)(
                                            dst_seg
                                                .as_mut_ptr()
                                                .add(p_idx * lay.rows_per_piece * seg_cols),
                                            base_ptr.add(slot_off + local_col_byte_off),
                                            &mcfg,
                                        );
                                    }
                                }
                            } else {
                                let mcfg_row = MoveCfg {
                                    rows: 1,
                                    cols: seg_cols as u64,
                                    outer_stride_bytes: lay.row_stride_in_piece as u64,
                                    packed: u64::from(lay.packed),
                                    use_nt: 0,
                                };
                                for r in 0..rows {
                                    let (slot_off, _) =
                                        lay.col_slot_and_off(plan, r, cp0, cp_cnt, cp, w);
                                    unsafe {
                                        (kernels.active.gather)(
                                            dst_seg.as_mut_ptr().add(r * seg_cols),
                                            base_ptr.add(slot_off + local_col_byte_off),
                                            &mcfg_row,
                                        );
                                    }
                                }
                            }
                            if rows > 1 {
                                apply_tile_twist_inv_scaled(
                                    kernels,
                                    dst_seg,
                                    &tw_seg0_inv[pi],
                                    rows,
                                    seg_cols,
                                    col_start,
                                    psi_inv[pi],
                                    crt.primes_u64[pi],
                                    prime_col_inv[pi].p,
                                    prime_col_inv[pi].inv_p,
                                    &mut sc.tw_row,
                                    &mut sc.s_u64,
                                );
                                unsafe {
                                    call_ntt_inv_cols(
                                        kernels,
                                        dst_seg.as_mut_ptr(),
                                        col_tables[pi].inv_tw.as_ptr(),
                                        &shape,
                                        &col_tables[pi].pr_fwd,
                                        &mut sc.col_scratch,
                                    );
                                }
                            }
                        },
                    );

                    if p0 + p_cnt == k {
                        sched.par_for_leaf_max(cgs_in_batch, 1, spec.par_workers, |cg_loc| {
                            let cg = cp0 * groups_per_cp + cg_loc;
                            let mut row_ptrs = [std::ptr::null::<f64>(); 18];
                            for (pi, slot) in row_ptrs.iter_mut().enumerate().take(k) {
                                *slot = unsafe {
                                    (slab_ptr as *const f64)
                                        .add((pi * groups_per_cp + cg_loc) * seg_elems)
                                };
                            }
                            let mut ccfg = NttCrtCfg {
                                k_primes: k as u64,
                                count: seg_cols as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                out_limbs: seg_limbs as u64,
                                signed: u64::from(signed),
                                wrap_cyclic: 0,
                                primes: crt.primes_f64.as_ptr(),
                                inv_primes: crt.inv_primes_f64.as_ptr(),
                                garner_inv: crt.garner_inv.as_ptr(),
                                prefix_prods: crt.prefix_prods.as_ptr(),
                                k_words: crt.k_words as u64,
                                segs: NttCrtSegs {
                                    segments: needed_rows as u64,
                                    res_stride: seg_cols as u64,
                                    out_stride: (num_col_groups * seg_limbs) as u64,
                                    carry_stride: (num_col_groups * crt.k_words) as u64,
                                    w0: unsafe { (w0s_ptr as *mut u64).add(cg) },
                                    w0_stride: num_col_groups as u64,
                                },
                                carry_limbs: unsafe {
                                    (carries_ptr as *mut u64).add(cg * crt.k_words)
                                },
                            };
                            unsafe {
                                ntt_crt_streams(
                                    kernels,
                                    (out_ptr as *mut u64).add(cg * seg_limbs),
                                    row_ptrs.as_ptr(),
                                    &mut ccfg,
                                );
                            }
                        });
                    }
                }
            }

            if let Some(h) = loader_h {
                h.wait()?;
            }
            scrub_res?;
        }

        drop(sub_prime_slab);
        drop(sw3_stages);
        drop(prod_guard);

        unsafe {
            if full_limbs > total_limbs {
                std::ptr::write_bytes(
                    (out_ptr as *mut u64).add(total_limbs),
                    0,
                    full_limbs - total_limbs,
                );
            }
            out.set_len(full_limbs);
        }
        NTT_TRACE_SW3_NS.fetch_add(t_sw3.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW3_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

        let w0 = finalize_segmented_w0(
            kernels,
            sched,
            cache,
            &mut out,
            seg_carries,
            seg_w0s,
            total_segs,
            seg_limbs,
            crt.k_words,
            total_limbs,
            out_limbs,
            m_bits,
            &spec,
        );
        if kernels.count_work {
            let n_u = spec.n as u64;
            let bf = if rows > 1 {
                (n_u / 2)
                    * (fwd_row_ops * (cols.ilog2() as u64) + ((cols.ilog2() + rows.ilog2()) as u64))
                    * (k as u64)
            } else if n_u > 1 {
                (((n_u / 2) * (n_u.ilog2() as u64)) + n_u) * (k as u64)
            } else {
                0
            };
            kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
            kernels
                .stats
                .crt_ops
                .fetch_add((spec.n * k * k) as u64, Ordering::Relaxed);
            kernels.stats.bytes_moved.fetch_add(
                ((2 + fwd_row_ops as usize) * k * rows * spec.outer_row_bytes()) as u64,
                Ordering::Relaxed,
            );
        }
        return Ok((out, w0));
    }

    // Tiny `cols < 64` fallback when `(cols * coeff_bits) % 64 != 0` (`pieces_per_row == 1`):
    let tile_cols = spec.tile_cols.min(cols).max(1);
    let tw_tile0_inv = if rows > 1 {
        cache.get_tile0_twiddles_par(
            kernels,
            sched,
            spec.par_workers,
            &crt.primes_u64,
            &col_tables,
            k,
            spec.n,
            rows,
            tile_cols,
            spec.negacyclic,
            true,
        )
    } else {
        Vec::new()
    };
    let n = spec.n;
    let mut natural: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
    let nat_ptr = natural.as_mut_ptr() as usize;

    let (s_left, s_right) = sw3_stages.split_at_mut(1);
    plan_sw3_batch(&mut s_left[0], sw3_batches[0])?;
    let s0_ref = &mut s_left[0];
    st.submit_loader_scoped(|| s0_ref.read_all_from_store(st, telemetry)).wait()?;

    for b_idx in 0..sw3_batches.len() {
        let (p0, p_cnt, _, _) = sw3_batches[b_idx];
        let (cur_s, next_s) = if (b_idx & 1) == 0 {
            (&mut s_left[0], &mut s_right[0])
        } else {
            (&mut s_right[0], &mut s_left[0])
        };
        let has_next = b_idx + 1 < sw3_batches.len();
        if has_next {
            plan_sw3_batch(next_s, sw3_batches[b_idx + 1])?;
        }
        let loader_h = has_next.then(|| {
            st.submit_loader_scoped(|| next_s.read_all_from_store(st, telemetry))
        });
        let scrub_res = scrub_sw3_stage(cur_s, p_cnt);
        if scrub_res.is_ok() {
            let buf_addrs: Vec<usize> = cur_s.bufs[..p_cnt]
                .iter()
                .map(|b| b.as_ptr() as usize)
                .collect();
            let plans_ref = cur_s.plans.as_slice();
            let num_col_tiles = cols / tile_cols;
            sched.par_for_init_max(
                p_cnt * num_col_tiles,
                1,
                spec.par_workers,
                || {
                    let mut lease = ScratchLease::acquire();
                    resize_f64_buf(&mut lease.get_mut().tile, rows * tile_cols);
                    lease
                },
                |lease, tid| {
                    let sc = lease.get_mut();
                    let pi_loc = tid / num_col_tiles;
                    let ct = tid % num_col_tiles;
                    let pi = p0 + pi_loc;
                    let col_start = ct * tile_cols;
                    let col_byte_off = ooc_elem_bytes(col_start, lay.packed);
                    let plan = &plans_ref[pi_loc];
                    let base_ptr = buf_addrs[pi_loc] as *const u8;
                    let mcfg_in = MoveCfg {
                        rows: lay.rows_per_piece as u64,
                        cols: tile_cols as u64,
                        outer_stride_bytes: lay.row_stride_in_piece as u64,
                        packed: u64::from(lay.packed),
                        use_nt: 0,
                    };
                    for p_idx in 0..lay.total_pieces_per_prime {
                        let slot_off = plan.pieces[p_idx].buf_byte_off;
                        unsafe {
                            (kernels.active.gather)(
                                sc.tile
                                    .as_mut_ptr()
                                    .add(p_idx * lay.rows_per_piece * tile_cols),
                                base_ptr.add(slot_off + col_byte_off),
                                &mcfg_in,
                            );
                        }
                    }
                    if rows > 1 {
                        let shape = [rows as u64, tile_cols as u64];
                        apply_tile_twist_inv_scaled(
                            kernels,
                            &mut sc.tile[..(rows * tile_cols)],
                            &tw_tile0_inv[pi],
                            rows,
                            tile_cols,
                            col_start,
                            psi_inv[pi],
                            crt.primes_u64[pi],
                            prime_col_inv[pi].p,
                            prime_col_inv[pi].inv_p,
                            &mut sc.tw_row,
                            &mut sc.s_u64,
                        );
                        unsafe {
                            call_ntt_inv_cols(
                                kernels,
                                sc.tile.as_mut_ptr(),
                                col_tables[pi].inv_tw.as_ptr(),
                                &shape,
                                &col_tables[pi].pr_fwd,
                                &mut sc.col_scratch,
                            );
                        }
                    }
                    let mcfg_out = MoveCfg {
                        rows: rows as u64,
                        cols: tile_cols as u64,
                        outer_stride_bytes: (cols * 8) as u64,
                        packed: 0,
                        use_nt: 0,
                    };
                    unsafe {
                        let dst = (*(nat_ptr as *mut Vec<f64>).add(pi))
                            .as_mut_ptr()
                            .add(col_start);
                        (kernels.active.scatter)(dst as *mut u8, sc.tile.as_ptr(), &mcfg_out);
                    }
                },
            );
        }
        if let Some(h) = loader_h {
            h.wait()?;
        }
        scrub_res?;
    }

    drop(sw3_stages);
    drop(prod_guard);

    let m_bits = (n as u64) * (spec.coeff_bits as u64);
    let crt_out_limbs = if spec.negacyclic {
        (m_bits.div_ceil(64) as usize) + 1
    } else {
        out_limbs
    };
    let mut out = crate::limb::take_limb_buf_uninit(crt_out_limbs);
    unsafe {
        out.set_len(crt_out_limbs);
    }
    let mut fermat_carry = [0u64; 20];
    let mut res_ptrs = [std::ptr::null::<f64>(); 20];
    for (pi, slot) in res_ptrs.iter_mut().enumerate().take(k) {
        *slot = natural[pi].as_ptr();
    }
    let mut ccfg = NttCrtCfg {
        k_primes: k as u64,
        count: n as u64,
        coeff_bits: spec.coeff_bits as u64,
        out_limbs: crt_out_limbs as u64,
        signed: u64::from(signed),
        wrap_cyclic: u64::from(spec.wrap_cyclic && !spec.negacyclic),
        primes: crt.primes_f64.as_ptr(),
        inv_primes: crt.inv_primes_f64.as_ptr(),
        garner_inv: crt.garner_inv.as_ptr(),
        prefix_prods: crt.prefix_prods.as_ptr(),
        k_words: crt.k_words as u64,
        segs: NttCrtSegs::ONE,
        carry_limbs: if spec.negacyclic {
            fermat_carry.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
    };
    let mut w0 = unsafe { (kernels.active.ntt_crt)(out.as_mut_ptr(), res_ptrs.as_ptr(), &mut ccfg) };
    if spec.negacyclic {
        let cur_len = out.len();
        fold_fermat_carry_in_place(
            &mut out,
            cur_len,
            m_bits,
            &fermat_carry[..crt.k_words],
            kernels,
        );
        w0 = crate::check::compute_limbs_w0(kernels, &out);
    } else {
        trim_limbs(&mut out);
    }
    if kernels.count_work {
        let n_u = spec.n as u64;
        let bf = (n_u / 2) * ((cols.ilog2() + rows.ilog2()) as u64) * (k as u64) * 2;
        kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
        kernels
            .stats
            .crt_ops
            .fetch_add((spec.n * k * k) as u64, Ordering::Relaxed);
        kernels.stats.bytes_moved.fetch_add(
            ((2 + fwd_row_ops as usize) * k * rows * spec.outer_row_bytes()) as u64,
            Ordering::Relaxed,
        );
    }
    Ok((out, w0))
}

fn finish_cached_fwd(
    spec: NttSpec,
    rows_transformed: bool,
    bufs: Vec<AlignedBytes>,
    grid_bytes: usize,
    _cache: &NttTableCache,
    _store: Option<&Store>,
    _kernels: &Kernels,
) -> Result<NttCachedFwd, crate::store::StoreError> {
    let _ = grid_bytes;
    Ok(NttCachedFwd {
        spec,
        rows_transformed,
        storage: ResidueStorage::InRam { bufs },
    })
}

fn materialize_storage<'a>(
    storage: &'a ResidueStorage,
    _cache: &NttTableCache,
    _spec: &NttSpec,
    _store: Option<&Store>,
    _kernels: &Kernels,
    _telemetry: &Telemetry,
    _owned_slot: &'a mut Vec<AlignedBytes>,
) -> Result<&'a [AlignedBytes], crate::store::StoreError> {
    match storage {
        ResidueStorage::InRam { bufs, .. } => Ok(bufs.as_slice()),
        ResidueStorage::InStore(_) => Err(crate::store::StoreError::DeviceLost),
    }
}

struct OwnedSlotsGuard<'a> {
    cache: &'a NttTableCache,
    s1: Vec<AlignedBytes>,
    s2: Vec<AlignedBytes>,
    s3: Vec<AlignedBytes>,
    s4: Vec<AlignedBytes>,
}

impl<'a> Drop for OwnedSlotsGuard<'a> {
    fn drop(&mut self) {
        self.cache.put_byte_bufs(std::mem::take(&mut self.s1));
        self.cache.put_byte_bufs(std::mem::take(&mut self.s2));
        self.cache.put_byte_bufs(std::mem::take(&mut self.s3));
        self.cache.put_byte_bufs(std::mem::take(&mut self.s4));
    }
}

pub enum NttOperand<'a> {
    Shared(&'a NttCachedFwd),
    Owned(&'a mut NttCachedFwd),
    ReuseInPlace(&'a mut NttCachedFwd),
}

impl<'a> NttOperand<'a> {
    #[inline]
    pub fn as_ref(&self) -> &NttCachedFwd {
        match self {
            Self::Shared(r) => r,
            Self::Owned(r) | Self::ReuseInPlace(r) => r,
        }
    }
}

fn prep_operand_sweep2<'a>(
    op: &mut NttOperand<'_>,
    cache: &NttTableCache,
    spec: &NttSpec,
    allow_packed_inplace: bool,
    store: Option<&Store>,
    kernels: &Kernels,
    telemetry: &Telemetry,
    owned_slot: &'a mut Vec<AlignedBytes>,
) -> Result<(usize, bool, bool), crate::store::StoreError> {
    let rt = op.as_ref().rows_transformed;
    let can_inplace = spec.extra_sweeps == 0
        && !rt
        && match op {
            NttOperand::Owned(c) => {
                !spec.packed
                    && matches!(&c.storage, ResidueStorage::InRam { bufs, .. } if !bufs.is_empty())
            }
            NttOperand::ReuseInPlace(c) => {
                (!spec.packed || allow_packed_inplace)
                    && matches!(&c.storage, ResidueStorage::InRam { bufs, .. } if !bufs.is_empty())
            }
            NttOperand::Shared(_) => false,
        };
    if can_inplace {
        if let NttOperand::Owned(c) | NttOperand::ReuseInPlace(c) = op {
            c.rows_transformed = true;
            if let ResidueStorage::InRam { bufs, .. } = &mut c.storage {
                return Ok((bufs.as_mut_ptr() as usize, false, true));
            }
        }
    }
    let slice = materialize_storage(
        &op.as_ref().storage,
        cache,
        spec,
        store,
        kernels,
        telemetry,
        owned_slot,
    )?;
    Ok((slice.as_ptr() as usize, rt, false))
}

/// The row sweep writes its rows through the cache, never past it. A row it writes is one it
/// has just read (or lies next to one), so the lines are in cache and an ordinary store costs
/// nothing more, whereas a streaming store must push each line to memory at once and stalls on
/// it. Measured on one product at a time, the sweep took 97 ms against 127 at 1024 rows, 56
/// against 65 at 512, 26 against 32 at 256 and 11.8 against 13.8 at 128, and a run of 1e9
/// digits 20.40 s against 20.96 s. (Sweep 1, which fills a grid it has not read, does stream.)
const ROW_SWEEP_STREAMS: u64 = 0;

/// Workers of the row sweep: all the product has. It used to be held to one a core where two
/// rows of a core's two workers do not fit its second-level cache together; that paid only
/// while the sweep streamed its output (`ROW_SWEEP_STREAMS`). Writing through the cache, one
/// product at a time on 16 hardware threads of 8 cores took 101 ms in this sweep on 8 workers,
/// 97 on 12 and 91 on 16 at 1024 rows of 32768 columns, and 13.7, 11.9 and 11.2 ms at 128 rows.
#[inline]
fn sw2_workers(_sched: &Scheduler, spec: &NttSpec) -> usize {
    spec.par_workers.max(1)
}

/// Complete product (or fused 2-product linear combination) from forward-transformed operands!
/// `a1` is single-use and owned by the caller (its in-RAM residue grid is reused in-place for
/// Sweep 2 when single-sweep, both unpacked and 50-bit packed); `b1 == None` computes `a1 * a1`.
#[allow(clippy::too_many_arguments)]
pub fn ntt_combine_and_inverse(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    telemetry: &Telemetry,
    a1: &mut NttCachedFwd,
    mut b1: Option<NttOperand<'_>>,
    mut second: Option<(NttOperand<'_>, NttOperand<'_>, bool)>,
    out_limbs: usize,
) -> Result<(LimbVec, u64), crate::store::StoreError> {
    let spec = a1.spec;
    let any_in_store = matches!(a1.storage, ResidueStorage::InStore(_))
        || b1
            .as_ref()
            .is_some_and(|b| matches!(b.as_ref().storage, ResidueStorage::InStore(_)))
        || second.as_ref().is_some_and(|(a2, b2, _)| {
            matches!(a2.as_ref().storage, ResidueStorage::InStore(_))
                || matches!(b2.as_ref().storage, ResidueStorage::InStore(_))
        });
    if (spec.out_of_core || any_in_store)
        && let Some(st) = store
    {
        return ntt_combine_out_of_core(
            kernels, cache, sched, st, telemetry, a1, b1, second, out_limbs,
        );
    }

    let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
    let k = spec.k_primes;
    let signed = spec.negacyclic || second.as_ref().map(|(_, _, sub)| *sub).unwrap_or(false);
    let same_ab1 = b1.is_none();
    let mut _pt = PathTimer::start("combine", &spec, &[out_limbs]);
    _pt.tag(&format!(
        "square={same_ab1} second={} a1_rows_done={}",
        second.is_some(),
        a1.rows_transformed
    ));
    let a1_rt = a1.rows_transformed;
    let b1_rt = b1.as_ref().map(|b| b.as_ref().rows_transformed).unwrap_or(a1_rt);
    let (a2_rt, b2_rt) = second
        .as_ref()
        .map(|(a2, b2, _)| (a2.as_ref().rows_transformed, b2.as_ref().rows_transformed))
        .unwrap_or((false, false));

    let mut slots = OwnedSlotsGuard {
        cache,
        s1: Vec::new(),
        s2: Vec::new(),
        s3: Vec::new(),
        s4: Vec::new(),
    };

    if spec.rows == 1 {
        let bufs_second = if let Some((ref a2, ref b2, sub)) = second {
            let ba2 = materialize_storage(
                &a2.as_ref().storage,
                cache,
                &spec,
                store,
                kernels,
                telemetry,
                &mut slots.s3,
            )?;
            let bb2 = materialize_storage(
                &b2.as_ref().storage,
                cache,
                &spec,
                store,
                kernels,
                telemetry,
                &mut slots.s4,
            )?;
            Some((ba2, bb2, sub))
        } else {
            None
        };
        let n = spec.n;
        let mut inplace_a1 = if !spec.packed {
            a1.take_in_ram_bufs()
        } else {
            None
        };
        let bufs_a1 = if inplace_a1.is_none() {
            materialize_storage(
                &a1.storage,
                cache,
                &spec,
                store,
                kernels,
                telemetry,
                &mut slots.s1,
            )?
        } else {
            &[][..]
        };
        let bufs_b1 = if let Some(ref b1_op) = b1 {
            materialize_storage(
                &b1_op.as_ref().storage,
                cache,
                &spec,
                store,
                kernels,
                telemetry,
                &mut slots.s2,
            )?
        } else {
            bufs_a1
        };
        let res_stride = res_stride_1d(n);
        let t_al = std::time::Instant::now();
        let mut prod_buf = if inplace_a1.is_none() {
            cache.take_u64_buf_uninit(k * res_stride)
        } else {
            Vec::new_in(TransformsAlloc)
        };
        NTT_TRACE_ALLOC_NS.fetch_add(t_al.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        let prod_base = prod_buf.as_mut_ptr() as usize;
        let inplace_ptr = inplace_a1
            .as_mut()
            .map_or(0usize, |g| g.as_mut_ptr() as usize);
        let t_i1 = std::time::Instant::now();

        if spec.par_workers <= 1 && !spec.packed {
            for pi in 0..k {
                let p_u64 = crt.primes_u64[pi];
                let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
                let pr_inv = &tables.pr_inv;
                let (dst_ptr, p_a1) = if inplace_ptr != 0 {
                    let p = unsafe {
                        (*(inplace_ptr as *mut AlignedBytes).add(pi)).as_mut_ptr() as *mut f64
                    };
                    (p, p as *const f64)
                } else {
                    (
                        unsafe { (prod_base as *mut f64).add(pi * res_stride) },
                        bufs_a1[pi].as_ptr() as *const f64,
                    )
                };
                let p_b1 = if same_ab1 {
                    p_a1
                } else {
                    bufs_b1[pi].as_ptr() as *const f64
                };
                if let Some((ba2, bb2, sub)) = bufs_second {
                    let ptrs = [
                        p_a1,
                        p_b1,
                        ba2[pi].as_ptr() as *const f64,
                        bb2[pi].as_ptr() as *const f64,
                    ];
                    let vcfg = NttVecCfg {
                        n: n as u64,
                        p: pr_inv.p,
                        inv_p: pr_inv.inv_p,
                        sub_second: u64::from(sub),
                    };
                    unsafe {
                        (kernels.active.ntt_pmuladd)(dst_ptr, &ptrs, &vcfg);
                        call_ntt_inv(kernels, 
                            dst_ptr,
                            tables.inv_tw.as_ptr(),
                            n as u64,
                            pr_inv,
                        );
                    }
                } else if inplace_ptr != 0 {
                    unsafe {
                        call_ntt_pmul_inv(
                            kernels,
                            dst_ptr,
                            tables.inv_tw.as_ptr(),
                            p_b1,
                            n as u64,
                            pr_inv,
                        );
                    }
                } else {
                    let vcfg = NttVecCfg {
                        n: n as u64,
                        p: pr_inv.p,
                        inv_p: pr_inv.inv_p,
                        sub_second: 0,
                    };
                    unsafe {
                        (kernels.active.ntt_pmul)(dst_ptr, p_a1, p_b1, &vcfg);
                        call_ntt_inv(kernels, 
                            dst_ptr,
                            tables.inv_tw.as_ptr(),
                            n as u64,
                            pr_inv,
                        );
                    }
                }
                if kernels.count_work && n > 1 {
                    kernels.stats.ntt_butterflies.fetch_add(
                        ((n as u64) / 2) * (n.ilog2() as u64) + (n as u64),
                        Ordering::Relaxed,
                    );
                }
            }
        } else {
            sched.par_for_init_max(
                k,
                1,
                spec.par_workers,
                ScratchLease::acquire,
                |lease, pi| {
                    let sc = lease.get_mut();
                    let p_u64 = crt.primes_u64[pi];
                    let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
                    let pr_inv = &tables.pr_inv;
                    let dst_ptr = if inplace_ptr != 0 {
                        unsafe {
                            (*(inplace_ptr as *mut AlignedBytes).add(pi)).as_mut_ptr() as *mut f64
                        }
                    } else {
                        unsafe { (prod_base as *mut f64).add(pi * res_stride) }
                    };

                    let mcfg = MoveCfg {
                        rows: 1,
                        cols: n as u64,
                        outer_stride_bytes: spec.outer_row_bytes() as u64,
                        packed: u64::from(spec.packed),
                        use_nt: 0,
                    };

                    let p_a1 = if inplace_ptr != 0 {
                        dst_ptr as *const f64
                    } else if spec.packed {
                        resize_f64_buf(&mut sc.ra1, n);
                        unsafe {
                            (kernels.active.gather)(
                                sc.ra1.as_mut_ptr(),
                                bufs_a1[pi].as_ptr(),
                                &mcfg,
                            )
                        };
                        sc.ra1.as_ptr()
                    } else {
                        bufs_a1[pi].as_ptr() as *const f64
                    };
                    let p_b1 = if same_ab1 {
                        p_a1
                    } else if spec.packed {
                        resize_f64_buf(&mut sc.rb1, n);
                        unsafe {
                            (kernels.active.gather)(
                                sc.rb1.as_mut_ptr(),
                                bufs_b1[pi].as_ptr(),
                                &mcfg,
                            )
                        };
                        sc.rb1.as_ptr()
                    } else {
                        bufs_b1[pi].as_ptr() as *const f64
                    };

                    if let Some((ba2, bb2, sub)) = bufs_second {
                        let p_a2 = if spec.packed {
                            resize_f64_buf(&mut sc.ra2, n);
                            unsafe {
                                (kernels.active.gather)(
                                    sc.ra2.as_mut_ptr(),
                                    ba2[pi].as_ptr(),
                                    &mcfg,
                                )
                            };
                            sc.ra2.as_ptr()
                        } else {
                            ba2[pi].as_ptr() as *const f64
                        };
                        let p_b2 = if spec.packed {
                            resize_f64_buf(&mut sc.rb2, n);
                            unsafe {
                                (kernels.active.gather)(
                                    sc.rb2.as_mut_ptr(),
                                    bb2[pi].as_ptr(),
                                    &mcfg,
                                )
                            };
                            sc.rb2.as_ptr()
                        } else {
                            bb2[pi].as_ptr() as *const f64
                        };
                        let ptrs = [p_a1, p_b1, p_a2, p_b2];
                        let vcfg = NttVecCfg {
                            n: n as u64,
                            p: pr_inv.p,
                            inv_p: pr_inv.inv_p,
                            sub_second: u64::from(sub),
                        };
                        unsafe {
                            (kernels.active.ntt_pmuladd)(dst_ptr, &ptrs, &vcfg);
                            call_ntt_inv(kernels, 
                                dst_ptr,
                                tables.inv_tw.as_ptr(),
                                n as u64,
                                pr_inv,
                            );
                        }
                    } else if inplace_ptr != 0 {
                        unsafe {
                            call_ntt_pmul_inv(
                                kernels,
                                dst_ptr,
                                tables.inv_tw.as_ptr(),
                                p_b1,
                                n as u64,
                                pr_inv,
                            );
                        }
                    } else {
                        let vcfg = NttVecCfg {
                            n: n as u64,
                            p: pr_inv.p,
                            inv_p: pr_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            (kernels.active.ntt_pmul)(dst_ptr, p_a1, p_b1, &vcfg);
                            call_ntt_inv(kernels, 
                                dst_ptr,
                                tables.inv_tw.as_ptr(),
                                n as u64,
                                pr_inv,
                            );
                        }
                    }
                    if kernels.count_work && n > 1 {
                        kernels.stats.ntt_butterflies.fetch_add(
                            ((n as u64) / 2) * (n.ilog2() as u64) + (n as u64),
                            Ordering::Relaxed,
                        );
                    }
                },
            );
        }
        NTT_TRACE_1D_INV_NS.fetch_add(t_i1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_c1 = std::time::Instant::now();

        // Recombine modulo CRT! (Parallelized in 512-element chunks when `par_workers >= 2 && n >= 4096`.)
        let m_bits = (n as u64) * (spec.coeff_bits as u64);
        let can_skip_crt_1d = !signed && !spec.negacyclic && !spec.wrap_cyclic;
        if spec.par_workers >= 2 && n >= 4096 {
            let seg_cols = 512usize;
            let num_segs = n / seg_cols;
            let seg_limbs = (seg_cols * (spec.coeff_bits as usize)) / 64;
            let active_segs = if can_skip_crt_1d {
                (out_limbs + 2).div_ceil(seg_limbs).min(num_segs).max(1)
            } else {
                num_segs
            };
            let active_limbs = active_segs * seg_limbs;
            let alloc_limbs = out_limbs.max(active_limbs + crt.k_words + 2);
            let mut out = crate::limb::take_limb_buf_uninit(alloc_limbs);
            let mut seg_carries = cache.take_u64_buf_uninit(active_segs * crt.k_words);
            let mut seg_w0s = cache.take_u64_buf_uninit(active_segs);
            let out_ptr = out.as_mut_ptr() as usize;
            let carries_ptr = seg_carries.as_mut_ptr() as usize;
            let w0s_ptr = seg_w0s.as_mut_ptr() as usize;
            let prod_ptr = prod_buf.as_ptr() as usize;

            sched.par_for_leaf_max(active_segs, 1, spec.par_workers, |seg| {
                let mut seg_ptrs = [std::ptr::null::<f64>(); 18];
                for (pi, slot) in seg_ptrs.iter_mut().enumerate().take(k) {
                    *slot = unsafe {
                        if inplace_ptr != 0 {
                            (*(inplace_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .cast::<f64>()
                                .add(seg * seg_cols)
                        } else {
                            (prod_ptr as *const f64).add(pi * res_stride + seg * seg_cols)
                        }
                    };
                }
                let seg_out_ptr = unsafe { (out_ptr as *mut u64).add(seg * seg_limbs) };
                let seg_carry_ptr = unsafe { (carries_ptr as *mut u64).add(seg * crt.k_words) };
                for w in 0..crt.k_words {
                    unsafe { *seg_carry_ptr.add(w) = 0 };
                }
                let mut ccfg = NttCrtCfg {
                    k_primes: k as u64,
                    count: seg_cols as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    out_limbs: seg_limbs as u64,
                    signed: u64::from(signed),
                    wrap_cyclic: 0,
                    primes: crt.primes_f64.as_ptr(),
                    inv_primes: crt.inv_primes_f64.as_ptr(),
                    garner_inv: crt.garner_inv.as_ptr(),
                    prefix_prods: crt.prefix_prods.as_ptr(),
                    k_words: crt.k_words as u64,
                    segs: NttCrtSegs::ONE,
                    carry_limbs: seg_carry_ptr,
                };
                unsafe {
                    let sw = (kernels.active.ntt_crt)(seg_out_ptr, seg_ptrs.as_ptr(), &mut ccfg);
                    *(w0s_ptr as *mut u64).add(seg) = sw;
                }
            });
            unsafe {
                if alloc_limbs > active_limbs {
                    std::ptr::write_bytes(
                        (out_ptr as *mut u64).add(active_limbs),
                        0,
                        alloc_limbs - active_limbs,
                    );
                }
                out.set_len(alloc_limbs);
            }

            if kernels.count_work {
                kernels
                    .stats
                    .crt_ops
                    .fetch_add((n * k * k) as u64, Ordering::Relaxed);
            }
            if let Some(grid) = inplace_a1 {
                cache.put_byte_bufs(grid);
            } else {
                cache.put_u64_buf(prod_buf);
            }
            let w0 = finalize_segmented_w0(
                kernels,
                sched,
                cache,
                &mut out,
                seg_carries,
                seg_w0s,
                active_segs,
                seg_limbs,
                crt.k_words,
                active_limbs,
                out_limbs,
                m_bits,
                &spec,
            );
            NTT_TRACE_1D_CRT_NS.fetch_add(t_c1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Ok((out, w0));
        }

        let crt_out_limbs = if spec.negacyclic {
            (m_bits.div_ceil(64) as usize) + 1
        } else {
            out_limbs
        };
        let (active_count, call_limbs) = if can_skip_crt_1d {
            cut_crt_stream(crt_out_limbs, spec.coeff_bits as usize, n)
        } else {
            (n, crt_out_limbs)
        };
        let mut out = crate::limb::take_limb_buf_uninit(call_limbs);
        unsafe {
            out.set_len(call_limbs);
        }
        let mut fermat_carry = if spec.negacyclic {
            vec![0u64; crt.k_words]
        } else {
            Vec::new()
        };
        let mut res_ptrs = [std::ptr::null::<f64>(); 18];
        for (pi, slot) in res_ptrs.iter_mut().enumerate().take(k) {
            *slot = if let Some(ref grid) = inplace_a1 {
                grid[pi].as_ptr() as *const f64
            } else {
                unsafe { (prod_buf.as_ptr() as *const f64).add(pi * res_stride) }
            };
        }
        let mut ccfg = NttCrtCfg {
            k_primes: k as u64,
            count: active_count as u64,
            coeff_bits: spec.coeff_bits as u64,
            out_limbs: call_limbs as u64,
            signed: u64::from(signed),
            wrap_cyclic: u64::from(spec.wrap_cyclic && !spec.negacyclic),
            primes: crt.primes_f64.as_ptr(),
            inv_primes: crt.inv_primes_f64.as_ptr(),
            garner_inv: crt.garner_inv.as_ptr(),
            prefix_prods: crt.prefix_prods.as_ptr(),
            k_words: crt.k_words as u64,
            segs: NttCrtSegs::ONE,
            carry_limbs: if spec.negacyclic {
                fermat_carry.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
        };
        if kernels.count_work {
            kernels
                .stats
                .crt_ops
                .fetch_add((n * k * k) as u64, Ordering::Relaxed);
        }
        let mut w0 =
            unsafe { (kernels.active.ntt_crt)(out.as_mut_ptr(), res_ptrs.as_ptr(), &mut ccfg) };
        if call_limbs > crt_out_limbs && drop_limbs_above(&mut out, crt_out_limbs) {
            w0 = compute_limbs_w0_parallel(kernels, sched, &out, spec.par_workers);
        }
        NTT_TRACE_1D_CRT_NS.fetch_add(t_c1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(grid) = inplace_a1 {
            cache.put_byte_bufs(grid);
        } else {
            cache.put_u64_buf(prod_buf);
        }
        if spec.negacyclic {
            let cur_len = out.len();
            fold_fermat_carry_in_place(&mut out, cur_len, m_bits, &fermat_carry, kernels);
            w0 = compute_limbs_w0_parallel(kernels, sched, &out, spec.par_workers);
        } else {
            trim_limbs(&mut out);
        }
        return Ok((out, w0));
    }

    // ========================================================================
    // Sweep 2 (Row sweep) + Sweep 3 (Column sweep back + CRT + Stitch)!
    // ========================================================================
    let rows = spec.rows;
    let cols = spec.cols;
    let row_bytes = spec.outer_row_bytes();
    let total_grid_bytes = rows * row_bytes;
    let sw2_nt_flag = ROW_SWEEP_STREAMS;

    let row0_tables: Vec<&PrimeTables> = (0..k)
        .map(|pi| cache.get_twiddles_ref(crt.primes_u64[pi], cols, false))
        .collect();

    let ntt_l1_sub = kernels.ntt_l1_sub_n.load(Ordering::Relaxed);
    let ntt_l2_sub = kernels.ntt_l2_sub_n.load(Ordering::Relaxed);
    let prime_row_meta: Vec<(NttPrime, NttPrime, u64)> = (0..k)
        .map(|pi| {
            let mut pf = row0_tables[pi].pr_fwd;
            let mut pv = row0_tables[pi].pr_inv;
            pf.l1_sub = ntt_l1_sub;
            pf.l2_sub = ntt_l2_sub;
            pv.l1_sub = ntt_l1_sub;
            pv.l2_sub = ntt_l2_sub;
            (pf, pv, row0_tables[pi].inv_n)
        })
        .collect();

    let has_second = second.is_some();
    let can_steal_a1 = spec.extra_sweeps == 0
        && match &a1.storage {
            ResidueStorage::InRam { bufs, .. } => !bufs.is_empty(),
            ResidueStorage::InStore(handles) => !handles.is_empty(),
        };
    let prep_b1 = if let Some(ref mut b1_op) = b1 {
        Some(prep_operand_sweep2(
            b1_op,
            cache,
            &spec,
            can_steal_a1,
            store,
            kernels,
            telemetry,
            &mut slots.s2,
        )?)
    } else {
        None
    };
    let prep_second = if let Some((ref mut a2_op, ref mut b2_op, sub)) = second {
        let pa2 = prep_operand_sweep2(
            a2_op,
            cache,
            &spec,
            can_steal_a1,
            store,
            kernels,
            telemetry,
            &mut slots.s3,
        )?;
        let pb2 = prep_operand_sweep2(
            b2_op,
            cache,
            &spec,
            can_steal_a1,
            store,
            kernels,
            telemetry,
            &mut slots.s4,
        )?;
        Some((pa2, pb2, sub))
    } else {
        None
    };

    // Fast in-place Sweep 2 when single-sweep (both in-RAM and materialized Store grids):
    // Reuse `a1`'s residue grid directly as `prod_grid` (`5` sweeps total, zero `prod_grid` alloc),
    // and transform single-use `Owned` operands in-place in their own grids when `!spec.packed`!
    let stolen_a1 = if spec.extra_sweeps == 0 {
        if let Some(g) = a1.take_in_ram_bufs() {
            Some(g)
        } else {
            materialize_storage(
                &a1.storage,
                cache,
                &spec,
                store,
                kernels,
                telemetry,
                &mut slots.s1,
            )?;
            if slots.s1.is_empty() {
                None
            } else {
                Some(std::mem::take(&mut slots.s1))
            }
        }
    } else {
        None
    };

    let t_sw2 = std::time::Instant::now();
    let prod_grid = if let Some(mut grid) = stolen_a1 {
        let grid_ptr = grid.as_mut_ptr() as usize;
        let need_ra1 = spec.packed;
        let need_rb1 = prep_b1.is_some_and(|(_, rt, inp)| spec.packed || (!rt && !inp))
            || prep_second.is_some_and(|((_, rt, inp), _, _)| spec.packed || (!rt && !inp));
        let need_ra2 = prep_second.is_some_and(|(_, (_, rt, inp), _)| spec.packed || (!rt && !inp));

        let sw2_w = sw2_workers(sched, &spec);
        let sw2_chunk = ((k * rows) / (sw2_w * 8)).clamp(1, sw2_chunk_cap(cols));
        sched.par_for_init_max(
            k * rows,
            sw2_chunk,
            sw2_w,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                if need_ra1 {
                    resize_f64_buf(&mut sc.ra1, cols);
                }
                if need_rb1 {
                    resize_f64_buf(&mut sc.rb1, cols);
                }
                if need_ra2 {
                    resize_f64_buf(&mut sc.ra2, cols);
                }
                lease
            },
            |lease, work_id| {
                let sc = lease.get_mut();
                let pi = work_id / rows;
                let r = work_id % rows;
                let (ref pr_fwd, ref pr_inv, _) = prime_row_meta[pi];
                let r_tw = &row0_tables[pi].tw;
                let r_inv_tw = &row0_tables[pi].inv_tw;
                let row_off = r * row_bytes;
                let mcfg_op = MoveCfg {
                    rows: 1,
                    cols: cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: 0,
                };

                let grid_row_u8 = unsafe {
                    (*(grid_ptr as *mut AlignedBytes).add(pi))
                        .as_mut_ptr()
                        .add(row_off)
                };
                let work_row_ptr = if spec.packed {
                    let mcfg_a1_in = MoveCfg {
                        use_nt: 0,
                        ..mcfg_op
                    };
                    unsafe {
                        (kernels.active.gather)(sc.ra1.as_mut_ptr(), grid_row_u8, &mcfg_a1_in);
                        if !a1_rt && prep_second.is_some() {
                            call_ntt_fwd(kernels, 
                                sc.ra1.as_mut_ptr(),
                                r_tw.as_ptr(),
                                cols as u64,
                                pr_fwd,
                            );
                        }
                    }
                    sc.ra1.as_mut_ptr()
                } else {
                    let p = grid_row_u8 as *mut f64;
                    if !a1_rt && prep_second.is_some() {
                        unsafe {
                            call_ntt_fwd(kernels, p, r_tw.as_ptr(), cols as u64, pr_fwd);
                        }
                    }
                    p
                };

                let mcfg_wb = MoveCfg {
                    use_nt: sw2_nt_flag,
                    ..mcfg_op
                };
                let pb1_ptr = if let Some((b1_ptr, b1_p_rt, b1_inp)) = prep_b1 {
                    if b1_inp {
                        if spec.packed {
                            unsafe {
                                let dst_u8 = (*(b1_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off);
                                (kernels.active.gather)(sc.rb1.as_mut_ptr(), dst_u8, &mcfg_op);
                                call_ntt_fwd(kernels, 
                                    sc.rb1.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                                (kernels.active.scatter)(dst_u8, sc.rb1.as_ptr(), &mcfg_wb);
                            }
                            sc.rb1.as_ptr()
                        } else {
                            let p = unsafe {
                                (*(b1_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off) as *mut f64
                            };
                            unsafe {
                                call_ntt_fwd(kernels, p, r_tw.as_ptr(), cols as u64, pr_fwd);
                            }
                            p as *const f64
                        }
                    } else if b1_p_rt && !spec.packed {
                        unsafe {
                            (*(b1_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(b1_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), src, &mcfg_op);
                            if !b1_p_rt {
                                call_ntt_fwd(kernels, 
                                    sc.rb1.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                            }
                        }
                        sc.rb1.as_ptr()
                    }
                } else {
                    work_row_ptr as *const f64
                };

                if let Some(((a2_ptr, a2_p_rt, a2_inp), (b2_ptr, b2_p_rt, b2_inp), sub)) =
                    prep_second
                {
                    let vcfg_mul = NttVecCfg {
                        n: cols as u64,
                        p: pr_inv.p,
                        inv_p: pr_inv.inv_p,
                        sub_second: 0,
                    };
                    unsafe {
                        (kernels.active.ntt_pmul)(
                            work_row_ptr,
                            work_row_ptr as *const f64,
                            pb1_ptr,
                            &vcfg_mul,
                        );
                    }
                    let pa2_ptr = if a2_inp {
                        if spec.packed {
                            unsafe {
                                let dst_u8 = (*(a2_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off);
                                (kernels.active.gather)(sc.rb1.as_mut_ptr(), dst_u8, &mcfg_op);
                                call_ntt_fwd(kernels, 
                                    sc.rb1.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                                (kernels.active.scatter)(dst_u8, sc.rb1.as_ptr(), &mcfg_wb);
                            }
                            sc.rb1.as_ptr()
                        } else {
                            let p = unsafe {
                                (*(a2_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off) as *mut f64
                            };
                            unsafe {
                                call_ntt_fwd(kernels, p, r_tw.as_ptr(), cols as u64, pr_fwd);
                            }
                            p as *const f64
                        }
                    } else if a2_p_rt && !spec.packed {
                        unsafe {
                            (*(a2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(a2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), src, &mcfg_op);
                            if !a2_p_rt {
                                call_ntt_fwd(kernels, 
                                    sc.rb1.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                            }
                        }
                        sc.rb1.as_ptr()
                    };
                    let pb2_ptr = if b2_inp {
                        if spec.packed {
                            unsafe {
                                let dst_u8 = (*(b2_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off);
                                (kernels.active.gather)(sc.ra2.as_mut_ptr(), dst_u8, &mcfg_op);
                                call_ntt_fwd(kernels, 
                                    sc.ra2.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                                (kernels.active.scatter)(dst_u8, sc.ra2.as_ptr(), &mcfg_wb);
                            }
                            sc.ra2.as_ptr()
                        } else {
                            let p = unsafe {
                                (*(b2_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(row_off) as *mut f64
                            };
                            unsafe {
                                call_ntt_fwd(kernels, p, r_tw.as_ptr(), cols as u64, pr_fwd);
                            }
                            p as *const f64
                        }
                    } else if b2_p_rt && !spec.packed {
                        unsafe {
                            (*(b2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(b2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.ra2.as_mut_ptr(), src, &mcfg_op);
                            if !b2_p_rt {
                                call_ntt_fwd(kernels, 
                                    sc.ra2.as_mut_ptr(),
                                    r_tw.as_ptr(),
                                    cols as u64,
                                    pr_fwd,
                                );
                            }
                        }
                        sc.ra2.as_ptr()
                    };
                    let ptrs = [
                        work_row_ptr as *const f64,
                        std::ptr::null(),
                        pa2_ptr,
                        pb2_ptr,
                    ];
                    let vcfg = NttVecCfg {
                        n: cols as u64,
                        p: pr_inv.p,
                        inv_p: pr_inv.inv_p,
                        sub_second: u64::from(sub),
                    };
                    unsafe {
                        (kernels.active.ntt_pmuladd)(work_row_ptr, &ptrs, &vcfg);
                        call_ntt_inv(
                            kernels,
                            work_row_ptr,
                            r_inv_tw.as_ptr(),
                            cols as u64,
                            pr_inv,
                        );
                    }
                } else if !a1_rt {
                    unsafe {
                        call_ntt_fwd_pmul_inv(
                            kernels,
                            work_row_ptr,
                            r_tw.as_ptr(),
                            r_inv_tw.as_ptr(),
                            pb1_ptr,
                            cols as u64,
                            pr_inv,
                        );
                    }
                } else {
                    unsafe {
                        call_ntt_pmul_inv(
                            kernels,
                            work_row_ptr,
                            r_inv_tw.as_ptr(),
                            pb1_ptr,
                            cols as u64,
                            pr_inv,
                        );
                    }
                }

                if spec.packed {
                    let mcfg_sw2_out = MoveCfg {
                        use_nt: sw2_nt_flag,
                        ..mcfg_op
                    };
                    unsafe {
                        (kernels.active.scatter)(
                            grid_row_u8,
                            work_row_ptr as *const f64,
                            &mcfg_sw2_out,
                        );
                    }
                }
            },
        );
        grid
    } else {
        let bufs_a1 = materialize_storage(
            &a1.storage,
            cache,
            &spec,
            store,
            kernels,
            telemetry,
            &mut slots.s1,
        )?;
        let a1_ptr = bufs_a1.as_ptr() as usize;
        let t_al2 = std::time::Instant::now();
        let mut grid = cache.take_byte_bufs_transient_committed(k, total_grid_bytes, sched, spec.par_workers);
        NTT_TRACE_ALLOC_NS.fetch_add(t_al2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        let prod_grid_ptr = grid.as_mut_ptr() as usize;

        let sw2_w = sw2_workers(sched, &spec);
        let sw2_chunk = ((k * rows) / (sw2_w * 8)).clamp(1, sw2_chunk_cap(cols));
        sched.par_for_init_max(
            k * rows,
            sw2_chunk,
            sw2_w,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                if !a1_rt || spec.packed {
                    resize_f64_buf(&mut sc.ra1, cols);
                }
                if !same_ab1 && (!b1_rt || spec.packed) {
                    resize_f64_buf(&mut sc.rb1, cols);
                }
                if has_second {
                    if !a2_rt || spec.packed {
                        resize_f64_buf(&mut sc.ra2, cols);
                    }
                    if !b2_rt || spec.packed {
                        resize_f64_buf(&mut sc.rb2, cols);
                    }
                }
                resize_f64_buf(&mut sc.rc, cols);
                lease
            },
            |lease, work_id| {
                let sc = lease.get_mut();
                let pi = work_id / rows;
                let r = work_id % rows;
                let p_u64 = crt.primes_u64[pi];
                let (ref pr_fwd, ref pr_inv, inv_cols) = prime_row_meta[pi];
                let r_tw = &row0_tables[pi].tw;
                let r_inv_tw = &row0_tables[pi].inv_tw;

                let mcfg = MoveCfg {
                    rows: 1,
                    cols: cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: sw2_nt_flag,
                };
                let row_off = r * row_bytes;
                let pa1_ptr = if a1_rt && !spec.packed {
                    unsafe {
                        (*(a1_ptr as *const AlignedBytes).add(pi))
                            .as_ptr()
                            .add(row_off) as *const f64
                    }
                } else {
                    unsafe {
                        let src = (*(a1_ptr as *const AlignedBytes).add(pi))
                            .as_ptr()
                            .add(row_off);
                        (kernels.active.gather)(sc.ra1.as_mut_ptr(), src, &mcfg);
                    }
                    if !a1_rt {
                        row_fwd_multi_sweep(kernels, &mut sc.ra1, r_tw, pr_fwd, spec.extra_sweeps);
                    }
                    sc.ra1.as_ptr()
                };
                let pb1_ptr = if let Some((b1_ptr, _, _)) = prep_b1 {
                    if b1_rt && !spec.packed {
                        unsafe {
                            (*(b1_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(b1_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), src, &mcfg);
                        }
                        if !b1_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.rb1,
                                r_tw,
                                pr_fwd,
                                spec.extra_sweeps,
                            );
                        }
                        sc.rb1.as_ptr()
                    }
                } else {
                    pa1_ptr
                };

                if let Some(((a2_ptr, _, _), (b2_ptr, _, _), sub)) = prep_second {
                    let pa2_ptr = if a2_rt && !spec.packed {
                        unsafe {
                            (*(a2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(a2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.ra2.as_mut_ptr(), src, &mcfg);
                        }
                        if !a2_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.ra2,
                                r_tw,
                                pr_fwd,
                                spec.extra_sweeps,
                            );
                        }
                        sc.ra2.as_ptr()
                    };
                    let pb2_ptr = if b2_rt && !spec.packed {
                        unsafe {
                            (*(b2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off) as *const f64
                        }
                    } else {
                        unsafe {
                            let src = (*(b2_ptr as *const AlignedBytes).add(pi))
                                .as_ptr()
                                .add(row_off);
                            (kernels.active.gather)(sc.rb2.as_mut_ptr(), src, &mcfg);
                        }
                        if !b2_rt {
                            row_fwd_multi_sweep(
                                kernels,
                                &mut sc.rb2,
                                r_tw,
                                pr_fwd,
                                spec.extra_sweeps,
                            );
                        }
                        sc.rb2.as_ptr()
                    };
                    unsafe {
                        let ptrs = [pa1_ptr, pb1_ptr, pa2_ptr, pb2_ptr];
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_inv.p,
                            inv_p: pr_inv.inv_p,
                            sub_second: u64::from(sub),
                        };
                        (kernels.active.ntt_pmuladd)(sc.rc.as_mut_ptr(), &ptrs, &vcfg);
                    }
                } else {
                    let vcfg = NttVecCfg {
                        n: cols as u64,
                        p: pr_inv.p,
                        inv_p: pr_inv.inv_p,
                        sub_second: 0,
                    };
                    unsafe {
                        (kernels.active.ntt_pmul)(sc.rc.as_mut_ptr(), pa1_ptr, pb1_ptr, &vcfg);
                    }
                }

                row_inv_multi_sweep(
                    kernels,
                    &mut sc.rc,
                    r_inv_tw,
                    p_u64,
                    inv_cols,
                    spec.extra_sweeps,
                );
                unsafe {
                    let dst_ptr = (*(prod_grid_ptr as *mut AlignedBytes).add(pi))
                        .as_mut_ptr()
                        .add(row_off);
                    (kernels.active.scatter)(dst_ptr, sc.rc.as_ptr(), &mcfg);
                }
            },
        );
        for b in &mut grid {
            unsafe {
                b.set_len(total_grid_bytes);
            }
        }
        grid
    };
    NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
    NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    drop(slots);

    let fwd_row_ops = u64::from(!a1_rt)
        + u64::from(!same_ab1 && !b1_rt)
        + if has_second {
            u64::from(!a2_rt) + u64::from(!b2_rt)
        } else {
            0
        };
    ntt_sweep3_and_crt(
        kernels,
        cache,
        sched,
        &spec,
        crt,
        prod_grid,
        signed,
        out_limbs,
        Some(fwd_row_ops),
    )
}

#[allow(clippy::too_many_arguments)]
fn ntt_sweep3_and_crt(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    spec: &NttSpec,
    crt: &CrtPlan,
    prod_grid: Vec<AlignedBytes>,
    signed: bool,
    out_limbs: usize,
    count_work_fwd_row_ops: Option<u64>,
) -> Result<(LimbVec, u64), crate::store::StoreError> {
    let k = spec.k_primes;
    let rows = spec.rows;
    let cols = spec.cols;
    let row_bytes = spec.outer_row_bytes();

    // Sweep 3: Column sweep back (`gather` -> untwist -> `ntt_inv_cols` -> `ntt_crt` + boundary stitch!)
    let col_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        rows,
        spec.negacyclic,
    );
    let psi_inv: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                spec.negacyclic,
                true,
            )
        })
        .collect();
    let prime_col_inv: Vec<NttPrime> = (0..k).map(|pi| col_tables[pi].pr_inv).collect();

    let tile_cols = spec.tile_cols.min(cols).max(1);
    let num_col_tiles = cols / tile_cols;

    let b_usize = spec.coeff_bits as usize;
    let pw_sw3 = spec.par_workers.max(1);
    let min_groups = if pw_sw3 > 1 { pw_sw3 * 2 } else { 1 };
    let l2_share = sched.min_l2_share() as usize;
    let fits_seg_fast = |seg: usize| -> bool {
        cols >= seg * min_groups
            && cols.is_multiple_of(seg)
            && (seg * b_usize).is_multiple_of(64)
            && rows * seg * 8 <= l2_share / 16
            && k * rows * seg * 8 <= (l2_share * 5) / 8
    };
    let fits_seg_l2 = |seg: usize| -> bool {
        cols >= seg * min_groups
            && cols.is_multiple_of(seg)
            && (seg * b_usize).is_multiple_of(64)
            && rows * seg * 8 <= l2_share / 4
            && k * rows * seg * 8 <= (l2_share * 7) / 8
    };
    let seg_cols_opt = if fits_seg_fast(64) {
        Some(64)
    } else if fits_seg_fast(32) {
        Some(32)
    } else if fits_seg_fast(128) {
        Some(128)
    } else if fits_seg_l2(64) {
        Some(64)
    } else if fits_seg_l2(32) {
        Some(32)
    } else if cols >= 32 * min_groups
        && cols.is_multiple_of(32)
        && (32 * b_usize).is_multiple_of(64)
    {
        Some(32)
    } else if cols >= 64 * min_groups
        && cols.is_multiple_of(64)
        && (64 * b_usize).is_multiple_of(64)
    {
        Some(64)
    } else if cols >= 32 && cols.is_multiple_of(32) && (32 * b_usize).is_multiple_of(64) {
        Some(32)
    } else if cols >= 64 && cols.is_multiple_of(64) && (64 * b_usize).is_multiple_of(64) {
        Some(64)
    } else if fits_seg_l2(16) {
        Some(16)
    } else if fits_seg_l2(8) {
        Some(8)
    } else if (tile_cols * b_usize).is_multiple_of(64) {
        Some(tile_cols)
    } else if (cols * b_usize).is_multiple_of(64) {
        Some(cols)
    } else {
        None
    };

    if let Some(seg_cols) = seg_cols_opt {
        let tw_seg0_inv = cache.get_tile0_twiddles_par(
            kernels,
            sched,
            spec.par_workers,
            &crt.primes_u64,
            &col_tables,
            k,
            spec.n,
            rows,
            seg_cols,
            spec.negacyclic,
            true,
        );
        let num_col_groups = cols / seg_cols;
        let seg_limbs = (seg_cols * b_usize) / 64;
        let can_skip_crt_2d = !signed && !spec.negacyclic && !spec.wrap_cyclic;
        let needed_rows = if can_skip_crt_2d {
            (out_limbs + 2)
                .div_ceil(num_col_groups * seg_limbs)
                .min(rows)
                .max(1)
        } else {
            rows
        };
        let total_segs = needed_rows * num_col_groups;
        let total_limbs = total_segs * seg_limbs;
        let m_bits = (spec.n as u64) * (spec.coeff_bits as u64);
        let full_limbs = (total_limbs + crt.k_words + 2).max(out_limbs);

        let t_al3 = std::time::Instant::now();
        let mut out = crate::limb::take_limb_buf_committed(full_limbs, sched, spec.par_workers);
        let mut seg_carries = cache.take_u64_buf_uninit(total_segs * crt.k_words);
        let mut seg_w0s = cache.take_u64_buf_uninit(total_segs);
        NTT_TRACE_ALLOC_NS.fetch_add(t_al3.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        let out_ptr = out.as_mut_ptr() as usize;
        let carries_ptr = seg_carries.as_mut_ptr() as usize;
        let w0s_ptr = seg_w0s.as_mut_ptr() as usize;

        let t_sw3 = std::time::Instant::now();
        let cg_chunk = if spec.packed
            && 256usize.is_multiple_of(seg_cols)
            && num_col_groups >= pw_sw3 * (256 / seg_cols) * 8
        {
            256 / seg_cols
        } else if k * rows * seg_cols >= 4_096 {
            1
        } else {
            (128 / seg_cols)
                .max(1)
                .min((num_col_groups / (pw_sw3 * 8)).max(1))
                .min(4)
        };
        sched.par_for_init_max(
            num_col_groups,
            cg_chunk,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                if sc.prime_segs.len() < k {
                    sc.prime_segs.resize_with(k, || Vec::new_in(ScratchAlloc));
                }
                for pi in 0..k {
                    resize_f64_buf(&mut sc.prime_segs[pi], pi * 40 + rows * seg_cols);
                }
                if sc.row_ptrs.len() < k {
                    sc.row_ptrs.resize(k, 0);
                }
                lease
            },
            |lease, cg| {
                let sc = lease.get_mut();
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: seg_cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed: u64::from(spec.packed),
                    use_nt: u64::from(
                        spec.use_nt && k * rows * row_bytes > cache.l3_prime_cap_bytes(),
                    ),
                };
                let shape = [rows as u64, seg_cols as u64];

                let col_start = cg * seg_cols;
                let col_byte_off = if spec.packed {
                    (col_start / 8) * 50
                } else {
                    col_start * 8
                };
                let seg_elems = rows * seg_cols;
                for pi in 0..k {
                    let seg_slice = &mut sc.prime_segs[pi][(pi * 40)..(pi * 40 + seg_elems)];
                    let pr_col = if rows > 1 {
                        &col_tables[pi].pr_fwd
                    } else {
                        &prime_col_inv[pi]
                    };
                    unsafe {
                        (kernels.active.gather)(
                            seg_slice.as_mut_ptr(),
                            prod_grid[pi].as_ptr().add(col_byte_off),
                            &mcfg,
                        );
                    }
                    apply_tile_twist_inv_scaled(
                        kernels,
                        seg_slice,
                        &tw_seg0_inv[pi],
                        rows,
                        seg_cols,
                        col_start,
                        psi_inv[pi],
                        crt.primes_u64[pi],
                        prime_col_inv[pi].p,
                        prime_col_inv[pi].inv_p,
                        &mut sc.tw_row,
                        &mut sc.s_u64,
                    );
                    unsafe {
                        call_ntt_inv_cols(
                            kernels,
                            seg_slice.as_mut_ptr(),
                            col_tables[pi].inv_tw.as_ptr(),
                            &shape,
                            pr_col,
                            &mut sc.col_scratch,
                        );
                    }
                }

                // The rows of this column group are one stream each, segment
                // `r * num_col_groups + cg` of the product: one call recombines them all.
                for pi in 0..k {
                    sc.row_ptrs[pi] = unsafe { sc.prime_segs[pi].as_ptr().add(pi * 40) } as usize;
                }
                let mut ccfg = NttCrtCfg {
                    k_primes: k as u64,
                    count: seg_cols as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    out_limbs: seg_limbs as u64,
                    signed: u64::from(signed),
                    wrap_cyclic: 0,
                    primes: crt.primes_f64.as_ptr(),
                    inv_primes: crt.inv_primes_f64.as_ptr(),
                    garner_inv: crt.garner_inv.as_ptr(),
                    prefix_prods: crt.prefix_prods.as_ptr(),
                    k_words: crt.k_words as u64,
                    segs: NttCrtSegs {
                        segments: needed_rows as u64,
                        res_stride: seg_cols as u64,
                        out_stride: (num_col_groups * seg_limbs) as u64,
                        carry_stride: (num_col_groups * crt.k_words) as u64,
                        w0: unsafe { (w0s_ptr as *mut u64).add(cg) },
                        w0_stride: num_col_groups as u64,
                    },
                    carry_limbs: unsafe { (carries_ptr as *mut u64).add(cg * crt.k_words) },
                };
                unsafe {
                    ntt_crt_streams(
                        kernels,
                        (out_ptr as *mut u64).add(cg * seg_limbs),
                        sc.row_ptrs.as_ptr() as *const *const f64,
                        &mut ccfg,
                    );
                }
            },
        );

        // Recycle `prod_grid` back to `NttTableCache` immediately after Sweep 3 gather finishes!
        cache.put_byte_bufs(prod_grid);

        unsafe {
            if full_limbs > total_limbs {
                std::ptr::write_bytes(
                    (out_ptr as *mut u64).add(total_limbs),
                    0,
                    full_limbs - total_limbs,
                );
            }
            out.set_len(full_limbs);
        }
        NTT_TRACE_SW3_NS.fetch_add(t_sw3.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW3_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

        let t_w0 = std::time::Instant::now();
        let w0 = finalize_segmented_w0(
            kernels,
            sched,
            cache,
            &mut out,
            seg_carries,
            seg_w0s,
            total_segs,
            seg_limbs,
            crt.k_words,
            total_limbs,
            out_limbs,
            m_bits,
            &spec,
        );
        NTT_TRACE_W0_NS.fetch_add(t_w0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_W0_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

        if let Some(fwd_row_ops) = count_work_fwd_row_ops {
            if kernels.count_work {
                let n = spec.n as u64;
                let bf = (n / 2)
                    * (fwd_row_ops * (cols.ilog2() as u64) + ((cols.ilog2() + rows.ilog2()) as u64))
                    * (k as u64);
                kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
                kernels
                    .stats
                    .crt_ops
                    .fetch_add((spec.n * k * k) as u64, Ordering::Relaxed);
                kernels
                    .stats
                    .bytes_moved
                    .fetch_add(((2 + fwd_row_ops as usize) * k * rows * row_bytes) as u64, Ordering::Relaxed);
            }
        }

        return Ok((out, w0));
    }

    // Fallback only for tiny `cols < 64` when `(cols * coeff_bits) % 64 != 0`:
    let tw_tile0_inv: Vec<Arc<Vec<f64>>> = (0..k)
        .map(|pi| {
            cache.get_tile0_twiddles(
                kernels,
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                rows,
                tile_cols,
                spec.negacyclic,
                true,
            )
        })
        .collect();
    let n = spec.n;
    let mut natural: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
    let nat_ptr = natural.as_mut_ptr() as usize;

    sched.par_for_init_max(
        k * num_col_tiles,
        1,
        spec.par_workers,
        || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            resize_f64_buf(&mut sc.tile, rows * tile_cols);
            lease
        },
        |lease, work_id| {
            let sc = lease.get_mut();
            let pi = work_id / num_col_tiles;
            let ct = work_id % num_col_tiles;
            let col_start = ct * tile_cols;
            let col_byte_off = if spec.packed {
                (col_start / 8) * 50
            } else {
                col_start * 8
            };
            let mcfg_in = MoveCfg {
                rows: rows as u64,
                cols: tile_cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: 0,
            };
            let mcfg_out = MoveCfg {
                rows: rows as u64,
                cols: tile_cols as u64,
                outer_stride_bytes: (cols * 8) as u64,
                packed: 0,
                use_nt: 0,
            };
            let shape = [rows as u64, tile_cols as u64];
            unsafe {
                (kernels.active.gather)(
                    sc.tile.as_mut_ptr(),
                    prod_grid[pi].as_ptr().add(col_byte_off),
                    &mcfg_in,
                );
            }
            apply_tile_twist_inv_scaled(
                kernels,
                &mut sc.tile,
                &tw_tile0_inv[pi],
                rows,
                tile_cols,
                col_start,
                psi_inv[pi],
                crt.primes_u64[pi],
                prime_col_inv[pi].p,
                prime_col_inv[pi].inv_p,
                &mut sc.tw_row,
                &mut sc.s_u64,
            );
            let pr_col = if rows > 1 {
                &col_tables[pi].pr_fwd
            } else {
                &prime_col_inv[pi]
            };
            unsafe {
                call_ntt_inv_cols(
                    kernels,
                    sc.tile.as_mut_ptr(),
                    col_tables[pi].inv_tw.as_ptr(),
                    &shape,
                    pr_col,
                    &mut sc.col_scratch,
                );
                let dst = (*(nat_ptr as *mut Vec<f64>).add(pi))
                    .as_mut_ptr()
                    .add(col_start);
                (kernels.active.scatter)(dst as *mut u8, sc.tile.as_ptr(), &mcfg_out);
            }
        },
    );

    cache.put_byte_bufs(prod_grid);

    if count_work_fwd_row_ops.is_some() && kernels.count_work {
        let n_u = spec.n as u64;
        let bf = (n_u / 2) * ((cols.ilog2() + rows.ilog2()) as u64) * (k as u64) * 2;
        kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
        kernels
            .stats
            .crt_ops
            .fetch_add((spec.n * k * k) as u64, Ordering::Relaxed);
        kernels
            .stats
            .bytes_moved
            .fetch_add((4 * k * rows * row_bytes) as u64, Ordering::Relaxed);
    }

    let m_bits = (n as u64) * (spec.coeff_bits as u64);
    let crt_out_limbs = if spec.negacyclic {
        (m_bits.div_ceil(64) as usize) + 1
    } else {
        out_limbs
    };
    let mut out = crate::limb::take_limb_buf_uninit(crt_out_limbs);
    unsafe {
        out.set_len(crt_out_limbs);
    }
    let mut fermat_carry = [0u64; 20];
    debug_assert!(crt.k_words <= fermat_carry.len());
    let mut res_ptrs = [std::ptr::null::<f64>(); 20];
    for (pi, slot) in res_ptrs.iter_mut().enumerate().take(k) {
        *slot = natural[pi].as_ptr();
    }
    let mut ccfg = NttCrtCfg {
        k_primes: k as u64,
        count: n as u64,
        coeff_bits: spec.coeff_bits as u64,
        out_limbs: crt_out_limbs as u64,
        signed: u64::from(signed),
        wrap_cyclic: u64::from(spec.wrap_cyclic && !spec.negacyclic),
        primes: crt.primes_f64.as_ptr(),
        inv_primes: crt.inv_primes_f64.as_ptr(),
        garner_inv: crt.garner_inv.as_ptr(),
        prefix_prods: crt.prefix_prods.as_ptr(),
        k_words: crt.k_words as u64,
        segs: NttCrtSegs::ONE,
        carry_limbs: if spec.negacyclic {
            fermat_carry.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
    };
    let mut w0 = unsafe { (kernels.active.ntt_crt)(out.as_mut_ptr(), res_ptrs.as_ptr(), &mut ccfg) };
    if spec.negacyclic {
        let cur_len = out.len();
        fold_fermat_carry_in_place(&mut out, cur_len, m_bits, &fermat_carry[..crt.k_words], kernels);
        w0 = crate::check::compute_limbs_w0(kernels, &out);
    } else {
        trim_limbs(&mut out);
    }
    Ok((out, w0))
}

/// Where the single recombination call of a plain product stops: the count of coefficients and
/// the limbs their digits fill. A product of `crt_out_limbs` limbs needs no coefficient above
/// that length. The count is a whole number of 64s, whose digits end on a limb boundary, and
/// the call is given an output that holds all of them, so that the kernel never has to cut a
/// digit short; `drop_limbs_above` then takes the surplus off.
fn cut_crt_stream(crt_out_limbs: usize, coeff_bits: usize, n: usize) -> (usize, usize) {
    let count = (crt_out_limbs * 64)
        .div_ceil(coeff_bits)
        .next_multiple_of(64)
        .min(n)
        .max(1);
    let stream_limbs = (count * coeff_bits).div_ceil(64);
    (count, stream_limbs.max(crt_out_limbs))
}

/// Shorten `out` to `len` limbs. Returns whether a dropped limb was not zero, in which case
/// the check word that came with the longer number no longer describes it.
fn drop_limbs_above(out: &mut LimbVec, len: usize) -> bool {
    let changed = out[len..].iter().any(|&x| x != 0);
    out.truncate(len);
    changed
}

/// Recombine the equal streams that `cfg.segs` describes with one kernel call. Every stream
/// starts from a zero carry and leaves its carry and its check word where `cfg` says. A single
/// stream goes through the one-stream form of the contract, which takes a carry in and returns
/// the check word.
unsafe fn ntt_crt_streams(
    kernels: &Kernels,
    out: *mut u64,
    residues: *const *const f64,
    cfg: &mut NttCrtCfg,
) {
    if cfg.segs.segments >= 2 {
        unsafe { (kernels.active.ntt_crt)(out, residues, cfg) };
        return;
    }
    let w0_ptr = cfg.segs.w0;
    cfg.segs = NttCrtSegs::ONE;
    unsafe {
        for w in 0..(cfg.k_words as usize) {
            *cfg.carry_limbs.add(w) = 0;
        }
        *w0_ptr = (kernels.active.ntt_crt)(out, residues, cfg);
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_segmented_w0(
    kernels: &Kernels,
    sched: &Scheduler,
    cache: &NttTableCache,
    out: &mut LimbVec,
    seg_carries: TransU64Vec,
    seg_w0s: TransU64Vec,
    num_segs: usize,
    seg_limbs: usize,
    k_words: usize,
    total_limbs: usize,
    out_limbs: usize,
    m_bits: u64,
    spec: &NttSpec,
) -> u64 {
    let step_shift = crate::refk::mod61_pow2((seg_limbs as u64) * 3);
    let mut total_w0 = 0u64;
    let can_par = spec.par_workers > 1 && num_segs >= 256 && seg_limbs > k_words + 2;
    let nw = if can_par {
        spec.par_workers.min(num_segs / 64).max(1)
    } else {
        1
    };

    if nw > 1 {
        let mut worker_w0s = vec![0u64; nw];
        let mut boundary_deltas = vec![0i64; nw];
        let out_len = out.len();
        let out_ptr = out.as_mut_ptr() as usize;
        let w0_ptr = worker_w0s.as_mut_ptr() as usize;
        let bd_ptr = boundary_deltas.as_mut_ptr() as usize;
        let seg_carries_ref = &seg_carries;
        let seg_w0s_ref = &seg_w0s;

        sched.par_for_leaf_max(nw, 1, spec.par_workers, |wi| {
            let s_begin = (wi * num_segs) / nw;
            let s_end = ((wi + 1) * num_segs) / nw;

            let mut w_shift = crate::refk::mod61_pow2(((s_begin * seg_limbs) as u64) * 3);
            let mut local_w0 = 0u64;
            for &sw in &seg_w0s_ref[s_begin..s_end] {
                local_w0 =
                    crate::refk::mod61_add(local_w0, crate::refk::mod61_mul(sw, w_shift));
                w_shift = crate::refk::mod61_mul(w_shift, step_shift);
            }

            let l_begin = s_begin * seg_limbs;
            let l_end = if wi + 1 == nw {
                out_len
            } else {
                s_end * seg_limbs
            };
            let c_begin = if wi == 0 { 0 } else { s_begin - 1 };
            let c_end = if wi + 1 == nw { num_segs } else { s_end - 1 };

            let my_slice = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut u64).add(l_begin), l_end - l_begin)
            };
            let mut c_shift = crate::refk::mod61_pow2((((c_begin + 1) * seg_limbs) as u64) * 3);
            let mut b_delta = 0i64;
            for seg in c_begin..c_end {
                let rel_off = (seg + 1) * seg_limbs - l_begin;
                let c_slice = &seg_carries_ref[(seg * k_words)..((seg + 1) * k_words)];
                b_delta +=
                    stitch_signed_carry_slice(my_slice, rel_off, c_slice, c_shift, &mut local_w0);
                c_shift = crate::refk::mod61_mul(c_shift, step_shift);
            }

            unsafe {
                *(w0_ptr as *mut u64).add(wi) = local_w0;
                *(bd_ptr as *mut i64).add(wi) = b_delta;
            }
        });

        for &w in &worker_w0s {
            total_w0 = crate::refk::mod61_add(total_w0, w);
        }
        for (wi, &d) in boundary_deltas.iter().enumerate().take(nw - 1) {
            if d != 0 {
                let s_end = ((wi + 1) * num_segs) / nw;
                let b_off = s_end * seg_limbs;
                let b_shift = crate::refk::mod61_pow2((b_off as u64) * 3);
                let w_arr = [d as u64];
                let _ = stitch_signed_carry_slice(out, b_off, &w_arr, b_shift, &mut total_w0);
            }
        }
        cache.put_u64_buf(seg_w0s);
        cache.put_u64_buf(seg_carries);
    } else {
        let mut w_shift = 1u64;
        for &sw in seg_w0s.iter().take(num_segs) {
            total_w0 = crate::refk::mod61_add(total_w0, crate::refk::mod61_mul(sw, w_shift));
            w_shift = crate::refk::mod61_mul(w_shift, step_shift);
        }
        cache.put_u64_buf(seg_w0s);

        let mut c_shift = step_shift;
        for seg in 0..num_segs {
            let c_slice = &seg_carries[(seg * k_words)..((seg + 1) * k_words)];
            let _ = stitch_signed_carry_slice(
                out,
                (seg + 1) * seg_limbs,
                c_slice,
                c_shift,
                &mut total_w0,
            );
            c_shift = crate::refk::mod61_mul(c_shift, step_shift);
        }
        cache.put_u64_buf(seg_carries);
    }

    if spec.negacyclic {
        let hc_len = out.len().saturating_sub(total_limbs);
        if hc_len <= 32 {
            let mut hc_stack = [0u64; 32];
            hc_stack[..hc_len].copy_from_slice(&out[total_limbs..]);
            fold_fermat_carry_in_place_w0(
                out,
                total_limbs,
                m_bits,
                &hc_stack[..hc_len],
                kernels,
                sched,
                spec.par_workers,
                &mut total_w0,
            );
        } else {
            let high_carry = out[total_limbs..].to_vec();
            fold_fermat_carry_in_place_w0(
                out,
                total_limbs,
                m_bits,
                &high_carry,
                kernels,
                sched,
                spec.par_workers,
                &mut total_w0,
            );
        }
    } else if spec.wrap_cyclic {
        let hc_len = out.len().saturating_sub(total_limbs);
        let mut hc_stack = [0u64; 32];
        let hc_vec;
        let high_carry: &[u64] = if hc_len <= 32 {
            hc_stack[..hc_len].copy_from_slice(&out[total_limbs..]);
            &hc_stack[..hc_len]
        } else {
            hc_vec = out[total_limbs..].to_vec();
            &hc_vec
        };
        for i in total_limbs.min(out.len())..out.len() {
            let old = out[i];
            w0_update_limb(&mut total_w0, i, old, 0);
        }
        out.truncate(total_limbs);
        if out.len() > out_limbs {
            for i in out_limbs..out.len() {
                let old = out[i];
                w0_update_limb(&mut total_w0, i, old, 0);
            }
            out.truncate(out_limbs);
        } else if out.len() < out_limbs {
            out.resize(out_limbs, 0);
        }
        fold_cyclic_carry_w0(out, total_limbs * 64, high_carry, &mut total_w0);
        trim_limbs(out);
    } else {
        if out.len() > out_limbs {
            for i in out_limbs..out.len() {
                let old = out[i];
                w0_update_limb(&mut total_w0, i, old, 0);
            }
            out.truncate(out_limbs);
        }
        trim_limbs(out);
    }

    debug_assert_eq!(
        total_w0,
        unsafe {
            let mut tmp = [0u64; 3];
            (kernels.active.check_n)(out.as_ptr(), out.len() as u64, &mut tmp, 0)
        }
    );
    total_w0
}

#[allow(clippy::too_many_arguments)]
fn fold_fermat_carry_in_place_w0(
    out: &mut LimbVec,
    low_len: usize,
    m_bits: u64,
    carry_words: &[u64],
    kernels: &Kernels,
    sched: &Scheduler,
    par_workers: usize,
    w0: &mut u64,
) {
    let carry_neg = !carry_words.is_empty() && (carry_words[carry_words.len() - 1] as i64) < 0;
    if carry_words.len() > 32 || m_bits < 128 {
        *out = fold_fermat_carry(kernels, &out[..low_len.min(out.len())], m_bits, carry_words);
        *w0 = compute_limbs_w0_parallel(kernels, sched, out, par_workers);
        return;
    }
    let mut c_buf = [0u64; 32];
    let mut c_len = carry_words.len();
    c_buf[..c_len].copy_from_slice(carry_words);
    if carry_neg {
        let mut b = 0u64;
        for w in &mut c_buf[..c_len] {
            let (d1, b1) = 0u64.overflowing_sub(*w);
            let (d2, b2) = d1.overflowing_sub(b);
            *w = d2;
            b = (b1 as u64) + (b2 as u64);
        }
    }
    while c_len > 0 && c_buf[c_len - 1] == 0 {
        c_len -= 1;
    }
    if (c_len as u64) * 64 > m_bits {
        *out = fold_fermat_carry(kernels, &out[..low_len.min(out.len())], m_bits, carry_words);
        *w0 = compute_limbs_w0_parallel(kernels, sched, out, par_workers);
        return;
    }

    let top_word = (m_bits / 64) as usize;
    let top_bit = (m_bits & 63) as u32;
    let mod_limbs = top_word + 1;

    let valid_low = low_len.min(out.len());
    for i in valid_low..out.len() {
        let old = out[i];
        w0_update_limb(w0, i, old, 0);
    }
    out.truncate(valid_low);
    if out.len() < mod_limbs {
        out.resize(mod_limbs, 0);
    } else if out.len() > mod_limbs {
        for i in mod_limbs..out.len() {
            let old = out[i];
            w0_update_limb(w0, i, old, 0);
        }
        out.truncate(mod_limbs);
    }
    let old_top = out[top_word];
    let masked_top = if top_bit == 0 {
        0
    } else {
        old_top & ((1u64 << top_bit) - 1)
    };
    w0_update_limb(w0, top_word, old_top, masked_top);
    out[top_word] = masked_top;

    if c_len == 0 {
        crate::limb::trim_limbs(out);
        return;
    }

    if !carry_neg {
        let mut borrow = 0u64;
        for i in 0..mod_limbs {
            let c_word = if i < c_len { c_buf[i] } else { 0 };
            if c_word == 0 && borrow == 0 {
                break;
            }
            let old = out[i];
            let (d1, b1) = old.overflowing_sub(c_word);
            let (d2, b2) = d1.overflowing_sub(borrow);
            w0_update_limb(w0, i, old, d2);
            out[i] = d2;
            borrow = u64::from(b1) + u64::from(b2);
        }
        if borrow != 0 {
            for (idx, slot) in out.iter_mut().enumerate().take(mod_limbs) {
                let old = *slot;
                let (s, c) = old.overflowing_add(1);
                w0_update_limb(w0, idx, old, s);
                *slot = s;
                if !c {
                    break;
                }
            }
            let old_t = out[top_word];
            let new_t = old_t.wrapping_add(1u64 << top_bit);
            w0_update_limb(w0, top_word, old_t, new_t);
            out[top_word] = new_t;
        }
    } else {
        let mut carry = 0u64;
        for i in 0..mod_limbs {
            let c_word = if i < c_len { c_buf[i] } else { 0 };
            if c_word == 0 && carry == 0 {
                break;
            }
            let old = out[i];
            let (s1, k1) = old.overflowing_add(c_word);
            let (s2, k2) = s1.overflowing_add(carry);
            w0_update_limb(w0, i, old, s2);
            out[i] = s2;
            carry = u64::from(k1) + u64::from(k2);
        }
        let hi_bits = out[top_word] >> top_bit;
        let ge_mod = if carry > 0 || hi_bits > 1 {
            true
        } else if hi_bits == 0 {
            false
        } else {
            let lo_mask = if top_bit == 0 {
                0u64
            } else {
                (1u64 << top_bit) - 1
            };
            (out[top_word] & lo_mask) != 0 || out[..top_word].iter().any(|&v| v != 0)
        };
        if ge_mod {
            for (idx, slot) in out.iter_mut().enumerate().take(mod_limbs) {
                let old = *slot;
                let (d, b) = old.overflowing_sub(1);
                w0_update_limb(w0, idx, old, d);
                *slot = d;
                if !b {
                    break;
                }
            }
            let old_t = out[top_word];
            let new_t = old_t.wrapping_sub(1u64 << top_bit);
            w0_update_limb(w0, top_word, old_t, new_t);
            out[top_word] = new_t;
        }
    }
    crate::limb::trim_limbs(out);
}

/// Zero-allocation in-place Fermat wrap `out[..low_len] - carry_words` modulo `2^m_bits + 1`
/// into canonical `[0, 2^m_bits]` when `m_bits >= 128` and `carry_words` fits below `2^m_bits`.
pub fn fold_fermat_carry_in_place(
    out: &mut LimbVec,
    low_len: usize,
    m_bits: u64,
    carry_words: &[u64],
    kernels: &Kernels,
) {
    let carry_neg = !carry_words.is_empty() && (carry_words[carry_words.len() - 1] as i64) < 0;
    if carry_words.len() > 32 || m_bits < 128 {
        *out = fold_fermat_carry(kernels, &out[..low_len.min(out.len())], m_bits, carry_words);
        return;
    }
    let mut c_buf = [0u64; 32];
    let mut c_len = carry_words.len();
    c_buf[..c_len].copy_from_slice(carry_words);
    if carry_neg {
        let mut b = 0u64;
        for w in &mut c_buf[..c_len] {
            let (d1, b1) = 0u64.overflowing_sub(*w);
            let (d2, b2) = d1.overflowing_sub(b);
            *w = d2;
            b = (b1 as u64) + (b2 as u64);
        }
    }
    while c_len > 0 && c_buf[c_len - 1] == 0 {
        c_len -= 1;
    }
    if (c_len as u64) * 64 > m_bits {
        *out = fold_fermat_carry(kernels, &out[..low_len.min(out.len())], m_bits, carry_words);
        return;
    }

    let top_word = (m_bits / 64) as usize;
    let top_bit = (m_bits & 63) as u32;
    let mod_limbs = top_word + 1;

    let valid_low = low_len.min(out.len());
    out.truncate(valid_low);
    if out.len() < mod_limbs {
        out.resize(mod_limbs, 0);
    } else if out.len() > mod_limbs {
        out.truncate(mod_limbs);
    }
    if top_bit == 0 {
        out[top_word] = 0;
    } else {
        out[top_word] &= (1u64 << top_bit) - 1;
    }

    if c_len == 0 {
        crate::limb::trim_limbs(out);
        return;
    }

    if !carry_neg {
        // out = (low - c_mag) mod (2^m_bits + 1)
        let mut borrow = 0u64;
        for i in 0..mod_limbs {
            let c_word = if i < c_len { c_buf[i] } else { 0 };
            if c_word == 0 && borrow == 0 {
                break;
            }
            let (d1, b1) = out[i].overflowing_sub(c_word);
            let (d2, b2) = d1.overflowing_sub(borrow);
            out[i] = d2;
            borrow = u64::from(b1) + u64::from(b2);
        }
        if borrow != 0 {
            for w in &mut out[..mod_limbs] {
                let (s, c) = w.overflowing_add(1);
                *w = s;
                if !c {
                    break;
                }
            }
            out[top_word] = out[top_word].wrapping_add(1u64 << top_bit);
        }
    } else {
        // out = (low + c_mag) mod (2^m_bits + 1)
        let mut carry = 0u64;
        for i in 0..mod_limbs {
            let c_word = if i < c_len { c_buf[i] } else { 0 };
            if c_word == 0 && carry == 0 {
                break;
            }
            let (s1, k1) = out[i].overflowing_add(c_word);
            let (s2, k2) = s1.overflowing_add(carry);
            out[i] = s2;
            carry = u64::from(k1) + u64::from(k2);
        }
        let hi_bits = out[top_word] >> top_bit;
        let ge_mod = if carry > 0 || hi_bits > 1 {
            true
        } else if hi_bits == 0 {
            false
        } else {
            let lo_mask = if top_bit == 0 {
                0u64
            } else {
                (1u64 << top_bit) - 1
            };
            (out[top_word] & lo_mask) != 0 || out[..top_word].iter().any(|&w| w != 0)
        };
        if ge_mod {
            for w in &mut out[..mod_limbs] {
                let (d, b) = w.overflowing_sub(1);
                *w = d;
                if !b {
                    break;
                }
            }
            out[top_word] = out[top_word].wrapping_sub(1u64 << top_bit);
        }
    }
    crate::limb::trim_limbs(out);
}

/// Fold `low_limbs - carry_words * 1` modulo `2^m_bits + 1` into canonical `[0, 2^m_bits]`.
pub fn fold_fermat_carry(
    kernels: &Kernels,
    low_limbs: &[u64],
    m_bits: u64,
    carry_words: &[u64],
) -> LimbVec {
    let mut modulus = vec![0u64; (m_bits / 64 + 1) as usize];
    modulus[0] = 1;
    modulus[(m_bits / 64) as usize] |= 1u64 << (m_bits & 63);
    crate::limb::trim_limbs(&mut modulus);

    let mut low = low_limbs.to_vec();
    // Mask `low` strictly to `m_bits`.
    let top_word = (m_bits / 64) as usize;
    let top_bit = (m_bits & 63) as u32;
    if top_word < low.len() {
        if top_bit == 0 {
            low.truncate(top_word);
        } else {
            low[top_word] &= (1u64 << top_bit) - 1;
            low.truncate(top_word + 1);
        }
    }
    crate::limb::trim_limbs(&mut low);

    let carry_neg = !carry_words.is_empty() && (carry_words[carry_words.len() - 1] as i64) < 0;
    let mut c_mag = carry_words.to_vec();
    if carry_neg {
        let mut b = 0u64;
        for w in &mut c_mag {
            let (d1, b1) = 0u64.overflowing_sub(*w);
            let (d2, b2) = d1.overflowing_sub(b);
            *w = d2;
            b = (b1 as u64) + (b2 as u64);
        }
    }
    crate::limb::trim_limbs(&mut c_mag);

    // Reduce `c_mag` modulo `2^m_bits + 1` via alternating m-bit slices!
    let mut acc = crate::limb::BigNum::from_limbs(low, false, 0, kernels);
    let mut pos = 0u64;
    let total_c_bits = (c_mag.len() as u64) * 64;
    // We want `low - C`, where `C = (-1)^{carry_neg} * sum_j chunk_j * (-1)^j`.
    let mut chunk_idx = 0usize;
    while pos < total_c_bits {
        let slice = extract_bit_window(&c_mag, pos, m_bits);
        if !slice.is_empty() {
            // Sign of `chunk_j` in `-C`: if `carry_neg` is false (`C > 0`), `-C` has `-chunk_0 + chunk_1 - ...`
            // If `carry_neg` is true (`C = -|C|`), `-C = +|C|` has `+chunk_0 - chunk_1 + ...`
            let add_term = carry_neg ^ (chunk_idx % 2 == 1);
            let term_bn = crate::limb::BigNum::from_limbs(slice, !add_term, 0, kernels);
            acc = crate::limb::add_bignum(kernels, &acc, &term_bn);
        }
        pos += m_bits;
        chunk_idx += 1;
    }

    let mod_bn = crate::limb::BigNum::from_limbs(modulus.clone(), false, 0, kernels);
    while acc.neg {
        acc = crate::limb::add_bignum(kernels, &acc, &mod_bn);
    }
    while crate::limb::cmp_limbs(&acc.limbs, &modulus) != std::cmp::Ordering::Less {
        acc = crate::limb::sub_bignum(kernels, &acc, &mod_bn);
    }
    acc.into_limbs()
}

fn extract_bit_window(limbs: &[u64], start_bit: u64, width_bits: u64) -> LimbVec {
    if limbs.is_empty() || width_bits == 0 || start_bit >= (limbs.len() as u64) * 64 {
        return crate::limb::empty_limb_vec();
    }
    let mut shifted = crate::limb::shr_limbs(limbs, start_bit);
    let top_word = (width_bits / 64) as usize;
    let top_bit = (width_bits & 63) as u32;
    if top_word < shifted.len() {
        if top_bit == 0 {
            shifted.truncate(top_word);
        } else {
            shifted[top_word] &= (1u64 << top_bit) - 1;
            shifted.truncate(top_word + 1);
        }
    }
    crate::limb::trim_limbs(&mut shifted);
    shifted
}

#[derive(Clone, Copy)]
enum Prep1dMode {
    PIndep { off: usize },
    PChunk { off: usize },
    Direct,
}

#[derive(Clone, Copy)]
struct Prep1dOp {
    nz: usize,
    mode: Prep1dMode,
}

#[inline]
fn ch_words_for_1d(nz: usize, p_indep: bool, p_chunk: bool, num_ch: usize) -> usize {
    let pad = if nz >= 512 { 40 } else { 0 };
    if p_indep {
        nz + pad
    } else if p_chunk && nz >= 32 {
        num_ch * crate::refk::ntt_planar_stride(nz) + pad
    } else {
        0
    }
}

#[allow(clippy::too_many_arguments)]
fn prep_1d_operand(
    kernels: &Kernels,
    limbs: &[u64],
    n: usize,
    coeff_bits: u32,
    p_indep: bool,
    p_chunk: bool,
    num_ch: usize,
    pr0: &NttPrime,
    m_ch: &mut [f64],
    cur_off: &mut usize,
) -> Prep1dOp {
    let nz = nz_elems_1d(limbs.len(), coeff_bits, n);
    let pad = if nz >= 512 { 40 } else { 0 };
    if p_indep {
        let off = *cur_off;
        *cur_off += nz + pad;
        if nz > 0 {
            let scfg0 = NttSplitCfg {
                in_limbs: limbs.len() as u64,
                start_coeff: 0,
                count: nz as u64,
                coeff_bits: coeff_bits as u64,
                col_stride: 0,
                cols_per_row: 0,
                p: pr0.p,
                inv_p: pr0.inv_p,
            };
            unsafe {
                (kernels.active.ntt_split)(m_ch.as_mut_ptr().add(off), limbs.as_ptr(), &scfg0);
            }
        }
        Prep1dOp {
            nz,
            mode: Prep1dMode::PIndep { off },
        }
    } else if p_chunk && nz >= 32 {
        let off = *cur_off;
        *cur_off += num_ch * crate::refk::ntt_planar_stride(nz) + pad;
        let ext_cfg = NttSplitCfg {
            in_limbs: limbs.len() as u64,
            start_coeff: 0,
            count: nz as u64,
            coeff_bits: coeff_bits as u64,
            col_stride: 0,
            cols_per_row: 0,
            p: 0.0,
            inv_p: 0.0,
        };
        unsafe {
            (kernels.active.ntt_split)(m_ch.as_mut_ptr().add(off), limbs.as_ptr(), &ext_cfg);
        }
        Prep1dOp {
            nz,
            mode: Prep1dMode::PChunk { off },
        }
    } else {
        Prep1dOp {
            nz,
            mode: Prep1dMode::Direct,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn split_1d_only(
    kernels: &Kernels,
    op_limbs: &[u64],
    prep: Prep1dOp,
    m_ch_ptr: *const f64,
    pr: &NttPrime,
    n: usize,
    coeff_bits: u32,
    dst: *mut f64,
) {
    match prep.mode {
        Prep1dMode::PIndep { off } => {
            if prep.nz > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(m_ch_ptr.add(off), dst, prep.nz);
                }
            }
            if prep.nz < n {
                unsafe {
                    std::ptr::write_bytes(dst.add(prep.nz), 0, n - prep.nz);
                }
            }
        }
        Prep1dMode::PChunk { off } => {
            if prep.nz > 0 {
                let red_cfg = NttSplitCfg {
                    in_limbs: u64::MAX,
                    start_coeff: 0,
                    count: prep.nz as u64,
                    coeff_bits: coeff_bits as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                unsafe {
                    (kernels.active.ntt_split)(dst, m_ch_ptr.add(off) as *const u64, &red_cfg);
                }
            }
            if prep.nz < n {
                unsafe {
                    std::ptr::write_bytes(dst.add(prep.nz), 0, n - prep.nz);
                }
            }
        }
        Prep1dMode::Direct => {
            if prep.nz > 0 {
                let scfg = NttSplitCfg {
                    in_limbs: op_limbs.len() as u64,
                    start_coeff: 0,
                    count: prep.nz as u64,
                    coeff_bits: coeff_bits as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                unsafe {
                    (kernels.active.ntt_split)(dst, op_limbs.as_ptr(), &scfg);
                }
            }
            if prep.nz < n {
                unsafe {
                    std::ptr::write_bytes(dst.add(prep.nz), 0, n - prep.nz);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn eval_1d_fwd(
    kernels: &Kernels,
    op_limbs: &[u64],
    prep: Prep1dOp,
    m_ch_ptr: *const f64,
    tables: &PrimeTables,
    n: usize,
    coeff_bits: u32,
    dst: *mut f64,
) {
    let pr = &tables.pr_fwd;
    unsafe {
        split_1d_only(kernels, op_limbs, prep, m_ch_ptr, pr, n, coeff_bits, dst);
        call_ntt_fwd(kernels, dst, tables.tw.as_ptr(), n as u64, pr);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn eval_1d_fwd_pmul_inv(
    kernels: &Kernels,
    op_limbs: &[u64],
    prep: Prep1dOp,
    m_ch_ptr: *const f64,
    tables: &PrimeTables,
    n: usize,
    coeff_bits: u32,
    dst: *mut f64,
    pmul_b: *const f64,
) {
    unsafe {
        split_1d_only(
            kernels,
            op_limbs,
            prep,
            m_ch_ptr,
            &tables.pr_fwd,
            n,
            coeff_bits,
            dst,
        );
        call_ntt_fwd_pmul_inv(
            kernels,
            dst,
            tables.tw.as_ptr(),
            tables.inv_tw.as_ptr(),
            pmul_b,
            n as u64,
            &tables.pr_inv,
        );
    }
}

#[inline]
fn res_stride_1d(n: usize) -> usize {
    if n >= 512 { n + 40 } else { n }
}

#[allow(clippy::too_many_arguments)]
fn crt_1d_from_flat(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    spec: &NttSpec,
    crt: &CrtPlan,
    flat_res: &[f64],
    res_stride: usize,
    signed: bool,
    out_limbs: usize,
) -> (LimbVec, u64) {
    let t_c1 = ntt_trace_enabled().then(std::time::Instant::now);
    let k = spec.k_primes;
    let n = spec.n;
    let m_bits = (n as u64) * (spec.coeff_bits as u64);
    let can_skip_crt = !spec.out_of_core && !signed && !spec.negacyclic && !spec.wrap_cyclic;

    if spec.par_workers >= 2 && n >= 4096 {
        let seg_cols = 512usize;
        let num_segs = n / seg_cols;
        let seg_limbs = (seg_cols * (spec.coeff_bits as usize)) / 64;
        let active_segs = if can_skip_crt {
            (out_limbs + 2).div_ceil(seg_limbs).min(num_segs).max(1)
        } else {
            num_segs
        };
        let active_limbs = active_segs * seg_limbs;
        let alloc_limbs = out_limbs.max(active_limbs + crt.k_words + 2);
        let mut out = crate::limb::take_limb_buf_uninit(alloc_limbs);
        let mut seg_carries = cache.take_u64_buf_uninit(active_segs * crt.k_words);
        let mut seg_w0s = cache.take_u64_buf_uninit(active_segs);
        let out_ptr = out.as_mut_ptr() as usize;
        let carries_ptr = seg_carries.as_mut_ptr() as usize;
        let w0s_ptr = seg_w0s.as_mut_ptr() as usize;
        let prod_ptr = flat_res.as_ptr() as usize;

        sched.par_for_leaf_max(active_segs, 1, spec.par_workers, |seg| {
            let mut seg_ptrs = [std::ptr::null::<f64>(); 18];
            for (pi, slot) in seg_ptrs.iter_mut().enumerate().take(k) {
                *slot = unsafe { (prod_ptr as *const f64).add(pi * res_stride + seg * seg_cols) };
            }
            let seg_out_ptr = unsafe { (out_ptr as *mut u64).add(seg * seg_limbs) };
            let seg_carry_ptr = unsafe { (carries_ptr as *mut u64).add(seg * crt.k_words) };
            for w in 0..crt.k_words {
                unsafe { *seg_carry_ptr.add(w) = 0 };
            }
            let mut ccfg = NttCrtCfg {
                k_primes: k as u64,
                count: seg_cols as u64,
                coeff_bits: spec.coeff_bits as u64,
                out_limbs: seg_limbs as u64,
                signed: u64::from(signed),
                wrap_cyclic: 0,
                primes: crt.primes_f64.as_ptr(),
                inv_primes: crt.inv_primes_f64.as_ptr(),
                garner_inv: crt.garner_inv.as_ptr(),
                prefix_prods: crt.prefix_prods.as_ptr(),
                k_words: crt.k_words as u64,
                segs: NttCrtSegs::ONE,
                carry_limbs: seg_carry_ptr,
            };
            unsafe {
                let sw = (kernels.active.ntt_crt)(seg_out_ptr, seg_ptrs.as_ptr(), &mut ccfg);
                *(w0s_ptr as *mut u64).add(seg) = sw;
            }
        });
        unsafe {
            if alloc_limbs > active_limbs {
                std::ptr::write_bytes(
                    (out_ptr as *mut u64).add(active_limbs),
                    0,
                    alloc_limbs - active_limbs,
                );
            }
            out.set_len(alloc_limbs);
        }
        if kernels.count_work {
            kernels
                .stats
                .crt_ops
                .fetch_add((n * k * k) as u64, Ordering::Relaxed);
        }
        let w0 = finalize_segmented_w0(
            kernels,
            sched,
            cache,
            &mut out,
            seg_carries,
            seg_w0s,
            active_segs,
            seg_limbs,
            crt.k_words,
            active_limbs,
            out_limbs,
            m_bits,
            spec,
        );
        if let Some(t) = t_c1 {
            NTT_TRACE_1D_CRT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        return (out, w0);
    }

    let crt_out_limbs = if spec.negacyclic {
        (m_bits.div_ceil(64) as usize) + 1
    } else {
        out_limbs
    };
    let (active_count, call_limbs) = if can_skip_crt {
        cut_crt_stream(crt_out_limbs, spec.coeff_bits as usize, n)
    } else {
        (n, crt_out_limbs)
    };
    let mut out = crate::limb::take_limb_buf_uninit(call_limbs);
    unsafe {
        out.set_len(call_limbs);
    }
    let mut fermat_carry = [0u64; 20];
    debug_assert!(crt.k_words <= fermat_carry.len());
    let mut res_ptrs = [std::ptr::null::<f64>(); 18];
    for (pi, slot) in res_ptrs.iter_mut().enumerate().take(k) {
        *slot = unsafe { flat_res.as_ptr().add(pi * res_stride) };
    }
    let mut ccfg = NttCrtCfg {
        k_primes: k as u64,
        count: active_count as u64,
        coeff_bits: spec.coeff_bits as u64,
        out_limbs: call_limbs as u64,
        signed: u64::from(signed),
        wrap_cyclic: u64::from(spec.wrap_cyclic && !spec.negacyclic),
        primes: crt.primes_f64.as_ptr(),
        inv_primes: crt.inv_primes_f64.as_ptr(),
        garner_inv: crt.garner_inv.as_ptr(),
        prefix_prods: crt.prefix_prods.as_ptr(),
        k_words: crt.k_words as u64,
        segs: NttCrtSegs::ONE,
        carry_limbs: if spec.negacyclic {
            fermat_carry.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
    };
    if kernels.count_work {
        kernels
            .stats
            .crt_ops
            .fetch_add((n * k * k) as u64, Ordering::Relaxed);
    }
    let mut w0 = unsafe { (kernels.active.ntt_crt)(out.as_mut_ptr(), res_ptrs.as_ptr(), &mut ccfg) };
    if call_limbs > crt_out_limbs && drop_limbs_above(&mut out, crt_out_limbs) {
        w0 = compute_limbs_w0_parallel(kernels, sched, &out, spec.par_workers);
    }
    if let Some(t) = t_c1 {
        NTT_TRACE_1D_CRT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if spec.negacyclic {
        let cur_len = out.len();
        fold_fermat_carry_in_place(&mut out, cur_len, m_bits, &fermat_carry[..crt.k_words], kernels);
        w0 = compute_limbs_w0_parallel(kernels, sched, &out, spec.par_workers);
    } else {
        trim_limbs(&mut out);
    }
    (out, w0)
}

#[inline]
fn can_pack_sweep1(spec: &NttSpec, tile_cols: usize) -> bool {
    spec.packed
        && (tile_cols == 32 || tile_cols == 64)
        && spec.cols.is_multiple_of(256)
        && (spec.cols / 256) >= spec.par_workers.max(1)
}

/// Sweep 1 for one operand, one prime and one group of 256 columns of a packed grid: split,
/// column transform and twist tile by tile, then the packed rows of the group go to the grid.
/// `staged` (see `sweep1_stages_groups`) packs each sub-tile as soon as it is made, into
/// `sc.ra2`, and writes the staged rows in one pass; without it the sub-tiles are kept in
/// `sc.tile` and packed row strip by row strip at the end. `ra1_is_zero` tells whether `sc.ra1`
/// still holds the zero strip of an earlier call.
#[allow(clippy::too_many_arguments)]
fn sweep1_group_one_op(
    kernels: &Kernels,
    sc: &mut NttTlsScratch,
    spec: &NttSpec,
    tile_cols: usize,
    cg: usize,
    limbs_i: &[u64],
    dst_base: usize,
    dst_row_bytes: usize,
    p_u64: u64,
    col_tab: &PrimeTables,
    psi: PsiParams,
    tw0_slice: &[f64],
    step_tc: &[u64],
    use_nt: u64,
    ra1_is_zero: &mut bool,
    staged: bool,
    pre_sub_nz: Option<&[usize; 8]>,
    plane_addrs: Option<&[usize; 8]>,
) {
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_elems = rows * tile_cols;
    let group_tiles = 256 / tile_cols;
    let strip_max = 32usize.min(rows);
    let pr = col_tab.pr_fwd;
    let col_tw_ptr = col_tab.tw.as_ptr() as usize;
    let shape = [rows as u64, tile_cols as u64];
    let group_byte_off = cg * 1600;
    let mut local_nz = [0usize; 8];
    let sub_nz: &[usize; 8] = if let Some(nz_ref) = pre_sub_nz {
        nz_ref
    } else {
        for sub in 0..group_tiles {
            local_nz[sub] = tile_nz_elems(
                limbs_i.len(),
                spec.coeff_bits,
                rows,
                cols,
                (cg * group_tiles + sub) * tile_cols,
                tile_cols,
            );
        }
        &local_nz
    };
    let any_nz = sub_nz[..group_tiles].iter().any(|&nz| nz > 0);
    if !any_nz {
        unsafe {
            let base_ptr = (dst_base as *mut u8).add(group_byte_off);
            if use_nt != 0 {
                if !*ra1_is_zero {
                    std::ptr::write_bytes(sc.ra1.as_mut_ptr(), 0, strip_max * 256);
                    *ra1_is_zero = true;
                }
                let mut r0 = 0usize;
                while r0 < rows {
                    let sr = (rows - r0).min(strip_max);
                    let mcfg_strip = MoveCfg {
                        rows: sr as u64,
                        cols: 256,
                        outer_stride_bytes: dst_row_bytes as u64,
                        packed: 1,
                        use_nt,
                    };
                    (kernels.active.scatter)(
                        base_ptr.add(r0 * dst_row_bytes),
                        sc.ra1.as_ptr(),
                        &mcfg_strip,
                    );
                    r0 += sr;
                }
            } else {
                for r in 0..rows {
                    std::ptr::write_bytes(
                        base_ptr.add(r * dst_row_bytes),
                        0,
                        1600,
                    );
                }
            }
        }
        return;
    }
    for sub in 0..group_tiles {
        let ct = cg * group_tiles + sub;
        let col_start = ct * tile_cols;
        let nz = sub_nz[sub];
        // Staged: one sub-tile at a time, which stays in cache from the split to the pack, and
        // its place in the packed rows of the group. Else: every sub-tile of the group is kept.
        let sub_slice = if staged {
            &mut sc.tile[..tile_elems]
        } else {
            &mut sc.tile[(sub * tile_elems)..((sub + 1) * tile_elems)]
        };
        let sub_pack_bytes = tile_cols * 50 / 8;
        let stage_ptr = unsafe {
            if use_nt == 0 {
                (dst_base as *mut u8).add(group_byte_off + sub * sub_pack_bytes)
            } else {
                (sc.ra2.as_mut_ptr() as *mut u8).add(sub * sub_pack_bytes)
            }
        };
        if nz == 0 {
            unsafe {
                if staged {
                    let zero_stride = if use_nt == 0 { dst_row_bytes } else { 1600 };
                    for r in 0..rows {
                        std::ptr::write_bytes(stage_ptr.add(r * zero_stride), 0, sub_pack_bytes);
                    }
                } else {
                    std::ptr::write_bytes(sub_slice.as_mut_ptr(), 0, tile_elems);
                }
            }
            continue;
        }
        unsafe {
            if let Some(addrs) = plane_addrs
                && addrs[sub] != 0
            {
                let red_cfg = NttSplitCfg {
                    in_limbs: u64::MAX,
                    start_coeff: 0,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                (kernels.active.ntt_split)(
                    sub_slice.as_mut_ptr(),
                    addrs[sub] as *const u64,
                    &red_cfg,
                );
            } else {
                let scfg = NttSplitCfg {
                    in_limbs: limbs_i.len() as u64,
                    start_coeff: col_start as u64,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: cols as u64,
                    cols_per_row: tile_cols as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                (kernels.active.ntt_split)(
                    sub_slice.as_mut_ptr(),
                    limbs_i.as_ptr(),
                    &scfg,
                );
            }
            if nz < tile_elems {
                std::ptr::write_bytes(
                    sub_slice.as_mut_ptr().add(nz),
                    0,
                    tile_elems - nz,
                );
            }
            call_ntt_fwd_cols(
                kernels,
                sub_slice.as_mut_ptr(),
                col_tw_ptr as *const f64,
                &shape,
                &pr,
                &mut sc.col_scratch,
            );
        }
        apply_tile_twist_cg4_sub(
            kernels,
            sub_slice,
            tw0_slice,
            step_tc,
            rows,
            tile_cols,
            col_start,
            sub,
            psi,
            p_u64,
            pr.p,
            pr.inv_p,
            &mut sc.tw_row,
            &mut sc.s_u64,
        );
        if staged {
            let mcfg_pack = MoveCfg {
                rows: rows as u64,
                cols: tile_cols as u64,
                outer_stride_bytes: if use_nt == 0 {
                    dst_row_bytes as u64
                } else {
                    1600
                },
                packed: 1,
                use_nt: 0,
            };
            unsafe {
                (kernels.active.scatter)(stage_ptr, sub_slice.as_ptr(), &mcfg_pack);
            }
        }
    }
    if staged {
        if use_nt != 0 {
            // The staged group, 1600 bytes a row, goes to the grid as it is: 25 whole cache lines
            // a row.
            let mcfg_rows = MoveCfg {
                rows: rows as u64,
                cols: 200,
                outer_stride_bytes: dst_row_bytes as u64,
                packed: 0,
                use_nt,
            };
            unsafe {
                let outer_ptr = (dst_base as *mut u8).add(group_byte_off);
                (kernels.active.scatter)(outer_ptr, sc.ra2.as_ptr(), &mcfg_rows);
            }
        }
        return;
    }
    *ra1_is_zero = false;
    let mut r0 = 0usize;
    while r0 < rows {
        let sr = (rows - r0).min(strip_max);
        unsafe {
            let dst_ra1 = sc.ra1.as_mut_ptr();
            let src_tile = sc.tile.as_ptr();
            for r in 0..sr {
                let d = dst_ra1.add(r * 256);
                let row_idx = r0 + r;
                for sub in 0..group_tiles {
                    std::ptr::copy_nonoverlapping(
                        src_tile.add(sub * tile_elems + row_idx * tile_cols),
                        d.add(sub * tile_cols),
                        tile_cols,
                    );
                }
            }
            let mcfg_strip = MoveCfg {
                rows: sr as u64,
                cols: 256,
                outer_stride_bytes: dst_row_bytes as u64,
                packed: 1,
                use_nt,
            };
            let outer_ptr =
                (dst_base as *mut u8).add(r0 * dst_row_bytes + group_byte_off);
            (kernels.active.scatter)(outer_ptr, sc.ra1.as_ptr(), &mcfg_strip);
        }
        r0 += sr;
    }
}

#[inline]
fn sw2_chunk_cap(cols: usize) -> usize {
    if cols >= 8_192 {
        1
    } else if cols >= 4_096 {
        2
    } else {
        4
    }
}

/// True when a group of sweep 1 is staged: its unpacked sub-tiles, `rows * 256` residues, are
/// more than half the worker's share of the second-level cache, so they would be read back
/// from further away to be packed. Measured on one product at a time (16 workers): at 1024
/// rows staging takes a product from 354 to 328 ms, at 512 from 163 to 156, at 256 from 79.0 to
/// 76.6; at 128 rows and below it gains nothing, and at 32 it costs 2%.
#[inline]
fn sweep1_stages_groups(sched: &Scheduler, spec: &NttSpec) -> bool {
    (spec.rows as u64) * 256 * 8 > sched.min_l2_share() / 2
}

/// True when sweep 1 of `spec` writes its packed rows a group of 256 columns at a time.
#[inline]
fn sweep1_groups_256(spec: &NttSpec, tile_cols: usize) -> bool {
    (16..=2048).contains(&spec.rows)
        && (tile_cols == 32 || tile_cols == 64)
        && spec.cols >= 256
        && spec.cols.is_multiple_of(256)
        && (spec.cols / 256) >= spec.par_workers.max(1)
}

/// One prime's share of a sweep 1 that serves several primes in one pass over the operands.
struct Sweep1Prime<'a> {
    p_u64: u64,
    col_tab: &'a PrimeTables,
    psi: PsiParams,
    tw0: &'a [f64],
}

/// The per-prime inputs of an all-primes sweep 1, taken from the tables of a product.
fn sweep1_primes<'a>(
    primes_u64: &[u64],
    col_tables: &[&'a PrimeTables],
    psi: &[PsiParams],
    tw0: &'a [Arc<Vec<f64>>],
) -> Vec<Sweep1Prime<'a>> {
    (0..col_tables.len())
        .map(|pi| Sweep1Prime {
            p_u64: primes_u64[pi],
            col_tab: col_tables[pi],
            psi: psi[pi],
            tw0: &tw0[pi],
        })
        .collect()
}

/// Whether `extra_bytes` more of work grids (bringing the operation's total grids to
/// `total_op_bytes`) may be taken so that an operand is read once for all primes: live bytes plus
/// all grids of the operation must fit comfortably inside both the region's physical-backing cap
/// and three quarters of its address range. Without a region the answer is no.
fn room_for_work_grids(total_op_bytes: usize, extra_bytes: usize) -> bool {
    crate::arena::active_arenas_stats().is_some_and(|st| {
        let live_after = st.live.saturating_add(total_op_bytes);
        let cap_limit = (st.soft_cap / 16).saturating_mul(15);
        let range_limit = (st.reserved / 4).saturating_mul(3);
        live_after <= cap_limit.min(range_limit)
            && st.live.saturating_add(extra_bytes.saturating_mul(2)) <= range_limit
    })
}

/// Staged sub-tile chunk (`sub0..sub1`, `1 <= sub1 - sub0 <= group_tiles`) for one prime:
/// reduces from pre-extracted 48-bit planes in `plane_addrs`, runs column NTT + twist into
/// `sc.tile[..tile_elems]`, packs into `sc.ra2`, carries any sub-cache-line tail (`< 64` bytes per
/// row) in `sc.ra1[carry_base..]` into the next chunk, and scatters strictly 64-byte-aligned whole
/// cache lines to `dst_base + cg * 1600 + emit_byte_off`.
#[allow(clippy::too_many_arguments)]
fn sweep1_staged_chunk_one_op(
    kernels: &Kernels,
    sc: &mut NttTlsScratch,
    spec: &NttSpec,
    tile_cols: usize,
    cg: usize,
    sub0: usize,
    sub1: usize,
    sub_nz: &[usize; 8],
    plane_addrs: &[usize; 8],
    limbs_i: &[u64],
    dst_base: usize,
    dst_row_bytes: usize,
    p_u64: u64,
    col_tab: &PrimeTables,
    psi: PsiParams,
    tw0_slice: &[f64],
    step_tc: &[u64],
    use_nt: u64,
    carry_base_words: usize,
) {
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_elems = rows * tile_cols;
    let group_tiles = 256 / tile_cols;
    let sub_pack_bytes = tile_cols * 50 / 8;
    let chunk_subs = sub1 - sub0;
    let chunk_row_bytes = chunk_subs * sub_pack_bytes;
    let prev_byte_end = sub0 * sub_pack_bytes;
    let cur_byte_end = sub1 * sub_pack_bytes;
    let emit_byte_off = prev_byte_end & !63;
    let prev_rem_bytes = prev_byte_end - emit_byte_off;
    let prev_rem_words = prev_rem_bytes / 8;
    let total_row_bytes = prev_rem_bytes + chunk_row_bytes;
    let total_row_words = total_row_bytes / 8;
    let next_emit_end = cur_byte_end & !63;
    let emit_bytes = next_emit_end - emit_byte_off;
    let emit_words = emit_bytes / 8;
    let next_rem_words = (cur_byte_end - next_emit_end) / 8;
    let pr = col_tab.pr_fwd;
    let col_tw_ptr = col_tab.tw.as_ptr();
    let shape = [rows as u64, tile_cols as u64];

    if use_nt == 0 {
        for sub in sub0..sub1 {
            let ct = cg * group_tiles + sub;
            let col_start = ct * tile_cols;
            let nz = sub_nz[sub];
            let dst_sub_ptr =
                unsafe { (dst_base as *mut u8).add(cg * 1600 + sub * sub_pack_bytes) };
            if nz == 0 {
                unsafe {
                    for r in 0..rows {
                        std::ptr::write_bytes(dst_sub_ptr.add(r * dst_row_bytes), 0, sub_pack_bytes);
                    }
                }
                continue;
            }
            let sub_slice = &mut sc.tile[..tile_elems];
            unsafe {
                if plane_addrs[sub] != 0 {
                    let red_cfg = NttSplitCfg {
                        in_limbs: u64::MAX,
                        start_coeff: 0,
                        count: nz as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: pr.p,
                        inv_p: pr.inv_p,
                    };
                    (kernels.active.ntt_split)(
                        sub_slice.as_mut_ptr(),
                        plane_addrs[sub] as *const u64,
                        &red_cfg,
                    );
                } else {
                    let scfg = NttSplitCfg {
                        in_limbs: limbs_i.len() as u64,
                        start_coeff: col_start as u64,
                        count: nz as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: cols as u64,
                        cols_per_row: tile_cols as u64,
                        p: pr.p,
                        inv_p: pr.inv_p,
                    };
                    (kernels.active.ntt_split)(sub_slice.as_mut_ptr(), limbs_i.as_ptr(), &scfg);
                }
                if nz < tile_elems {
                    std::ptr::write_bytes(sub_slice.as_mut_ptr().add(nz), 0, tile_elems - nz);
                }
                call_ntt_fwd_cols(
                    kernels,
                    sub_slice.as_mut_ptr(),
                    col_tw_ptr,
                    &shape,
                    &pr,
                    &mut sc.col_scratch,
                );
            }
            apply_tile_twist_cg4_sub(
                kernels,
                sub_slice,
                tw0_slice,
                step_tc,
                rows,
                tile_cols,
                col_start,
                if sub == sub0 { 0 } else { sub },
                psi,
                p_u64,
                pr.p,
                pr.inv_p,
                &mut sc.tw_row,
                &mut sc.s_u64,
            );
            let mcfg_pack = MoveCfg {
                rows: rows as u64,
                cols: tile_cols as u64,
                outer_stride_bytes: dst_row_bytes as u64,
                packed: 1,
                use_nt: 0,
            };
            unsafe {
                (kernels.active.scatter)(dst_sub_ptr, sub_slice.as_ptr(), &mcfg_pack);
            }
        }
        return;
    }

    if prev_rem_words > 0 {
        unsafe {
            let carry_src = (sc.ra1.as_ptr() as *const u64).add(carry_base_words);
            let ra2_u64 = sc.ra2.as_mut_ptr() as *mut u64;
            for r in 0..rows {
                std::ptr::copy_nonoverlapping(
                    carry_src.add(r * 8),
                    ra2_u64.add(r * total_row_words),
                    prev_rem_words,
                );
            }
        }
    }

    for sub in sub0..sub1 {
        let ct = cg * group_tiles + sub;
        let col_start = ct * tile_cols;
        let nz = sub_nz[sub];
        let stage_ptr = unsafe {
            (sc.ra2.as_mut_ptr() as *mut u8).add(prev_rem_bytes + (sub - sub0) * sub_pack_bytes)
        };
        if nz == 0 {
            unsafe {
                for r in 0..rows {
                    std::ptr::write_bytes(stage_ptr.add(r * total_row_bytes), 0, sub_pack_bytes);
                }
            }
            continue;
        }
        let sub_slice = &mut sc.tile[..tile_elems];
        unsafe {
            if plane_addrs[sub] != 0 {
                let red_cfg = NttSplitCfg {
                    in_limbs: u64::MAX,
                    start_coeff: 0,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                (kernels.active.ntt_split)(
                    sub_slice.as_mut_ptr(),
                    plane_addrs[sub] as *const u64,
                    &red_cfg,
                );
            } else {
                let scfg = NttSplitCfg {
                    in_limbs: limbs_i.len() as u64,
                    start_coeff: col_start as u64,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: cols as u64,
                    cols_per_row: tile_cols as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                (kernels.active.ntt_split)(sub_slice.as_mut_ptr(), limbs_i.as_ptr(), &scfg);
            }
            if nz < tile_elems {
                std::ptr::write_bytes(sub_slice.as_mut_ptr().add(nz), 0, tile_elems - nz);
            }
            call_ntt_fwd_cols(
                kernels,
                sub_slice.as_mut_ptr(),
                col_tw_ptr,
                &shape,
                &pr,
                &mut sc.col_scratch,
            );
        }
        apply_tile_twist_cg4_sub(
            kernels,
            sub_slice,
            tw0_slice,
            step_tc,
            rows,
            tile_cols,
            col_start,
            if sub == sub0 { 0 } else { sub },
            psi,
            p_u64,
            pr.p,
            pr.inv_p,
            &mut sc.tw_row,
            &mut sc.s_u64,
        );
        let mcfg_pack = MoveCfg {
            rows: rows as u64,
            cols: tile_cols as u64,
            outer_stride_bytes: total_row_bytes as u64,
            packed: 1,
            use_nt: 0,
        };
        unsafe {
            (kernels.active.scatter)(stage_ptr, sub_slice.as_ptr(), &mcfg_pack);
        }
    }

    if next_rem_words > 0 {
        unsafe {
            let carry_dst = (sc.ra1.as_mut_ptr() as *mut u64).add(carry_base_words);
            let ra2_u64 = sc.ra2.as_mut_ptr() as *mut u64;
            for r in 0..rows {
                let row_src = ra2_u64.add(r * total_row_words);
                std::ptr::copy_nonoverlapping(
                    row_src.add(emit_words),
                    carry_dst.add(r * 8),
                    next_rem_words,
                );
            }
        }
    }

    if emit_words > 0 {
        let mcfg_rows = MoveCfg {
            rows: rows as u64,
            cols: emit_words as u64,
            outer_stride_bytes: dst_row_bytes as u64,
            packed: (total_row_bytes as u64) << 8,
            use_nt,
        };
        unsafe {
            let outer_ptr = (dst_base as *mut u8).add(cg * 1600 + emit_byte_off);
            (kernels.active.scatter)(outer_ptr, sc.ra2.as_ptr(), &mcfg_rows);
        }
    }
}

/// Sweep 1 for several primes in one pass. A worker takes a group of 256 columns and, operand by
/// operand, makes that group for every prime before it moves on, so the operand's limbs under
/// the group are read from memory once and from the cache for every prime after the first.
/// `ops` gives, per operand, its limbs, its grid for each entry of `primes` (a base address), and
/// the row stride of those grids. The grids are packed; `sweep1_groups_256` must hold.
#[allow(clippy::too_many_arguments)]
fn ntt_sweep1_all_primes(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    ops: &[(&[u64], &[usize], usize)],
    spec: &NttSpec,
    tile_cols_arg: usize,
    primes: &[Sweep1Prime<'_>],
    use_nt: u64,
) {
    if ops.is_empty() || primes.is_empty() {
        return;
    }
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_cols = tile_cols_arg.min(cols).max(1);
    debug_assert!(sweep1_groups_256(spec, tile_cols));
    let tile_elems = rows * tile_cols;
    let group_tiles = 256 / tile_cols;
    let sub_pack_bytes = tile_cols * 50 / 8;
    let strip_max = 32usize.min(rows);
    let step_tcs: Vec<Vec<u64>> = primes
        .iter()
        .map(|pc| {
            let mut v = Vec::new();
            if !tw0_is_full(pc.tw0, cols * rows)
                && !tw0_has_row_scales(pc.tw0, rows, tile_cols)
            {
                fill_row_scales_u64(rows, tile_cols, pc.psi, pc.p_u64, &mut v);
            }
            v
        })
        .collect();
    let t_sw1 = ntt_trace_enabled().then(std::time::Instant::now);
    let staged = sweep1_stages_groups(sched, spec);
    let num_groups = cols / 256;
    let pw = spec.par_workers.max(1);
    let active_workers = sched.num_workers().max(pw).max(1);
    let n_ops = ops.len();
    let k_primes = primes.len();
    let units = n_ops * k_primes;

    // Split penultimate groups by operand (preserving 100% planar reuse across primes!) and only
    // the very last few groups by `(operand, prime)` for microsecond tail load balance.
    let prime_fine_groups = if pw > 1 {
        if n_ops >= 2 {
            (pw / 8).clamp(1, 2).min(num_groups / 4)
        } else {
            (pw / 4).clamp(1, 4).min(num_groups / 4)
        }
    } else {
        0
    };
    let op_fine_groups = if pw > 1 && n_ops >= 2 {
        pw.min((num_groups - prime_fine_groups) / 2)
    } else {
        0
    };
    let whole_groups = num_groups - op_fine_groups - prime_fine_groups;
    let op_fine_tasks = op_fine_groups * n_ops;
    let prime_fine_tasks = prime_fine_groups * units;

    let can_planar = k_primes >= 2
        && (53..=384).contains(&spec.coeff_bits)
        && primes.iter().all(|pc| pc.p_u64 > (1u64 << 48));
    let num_ch = (spec.coeff_bits as usize).div_ceil(48);
    let staged_bytes = if staged && use_nt != 0 {
        spec.rows * 1600
    } else {
        0
    };
    let l3_per_worker = if use_nt != 0 {
        cache.l3_prime_cap_bytes() / active_workers
    } else {
        (cache.l3_prime_cap_bytes() * 3 / 4) / active_workers
    };
    let tile_bytes = if staged {
        tile_elems * 8
    } else {
        group_tiles * tile_elems * 8
    };
    let max_full_plane_words = l3_per_worker
        .saturating_sub(staged_bytes + tile_bytes)
        / 8;

    sched.par_for_init_max(
        whole_groups + op_fine_tasks + prime_fine_tasks,
        1,
        spec.par_workers,
        || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            let tile_len = if staged {
                tile_elems
            } else {
                group_tiles * tile_elems
            };
            resize_f64_buf(&mut sc.tile, tile_len);
            resize_f64_buf(&mut sc.ra1, (strip_max * 256).max(k_primes * rows * 8));
            if staged && use_nt != 0 {
                resize_f64_buf(&mut sc.ra2, spec.rows * 208);
            }
            lease
        },
        |lease, task| {
            let sc = lease.get_mut();
            let mut ra1_is_zero = false;
            let (cg, only_op, only_prime) = if task < whole_groups {
                (task, None, None)
            } else if task < whole_groups + op_fine_tasks {
                let f = task - whole_groups;
                (whole_groups + f / n_ops, Some(f % n_ops), None)
            } else {
                let f = task - whole_groups - op_fine_tasks;
                let u = f % units;
                (
                    whole_groups + op_fine_groups + f / units,
                    Some(u / k_primes),
                    Some(u % k_primes),
                )
            };
            for (oi, &(limbs_i, dst_bases, dst_row_bytes)) in ops.iter().enumerate() {
                if only_op.is_some_and(|want_oi| want_oi != oi) {
                    continue;
                }
                let mut sub_nz = [0usize; 8];
                let mut max_sub_nz = 0usize;
                let mut total_plane_words = 0usize;
                for sub in 0..group_tiles {
                    let nz = tile_nz_elems(
                        limbs_i.len(),
                        spec.coeff_bits,
                        rows,
                        cols,
                        (cg * group_tiles + sub) * tile_cols,
                        tile_cols,
                    );
                    sub_nz[sub] = nz;
                    if nz > max_sub_nz {
                        max_sub_nz = nz;
                    }
                    if nz >= 32 {
                        total_plane_words += num_ch * crate::refk::ntt_planar_stride(nz);
                    }
                }

                // When staged and a full 8-sub-tile group exceeds one worker's L3 budget, split
                // the group into `chunk_subs` sub-tiles (`4`, `2`, or `1`) per pass so 100% of
                // sub-tiles extract 48-bit planes once (`1x` DRAM read) while keeping both
                // `sc.tw_step` and the active slice of `sc.ra2` hot inside cache!
                if only_prime.is_none()
                    && can_planar
                    && staged
                    && total_plane_words > max_full_plane_words
                    && max_sub_nz >= 32
                {
                    let max_sub_pl = crate::refk::ntt_planar_stride(max_sub_nz);
                    let mut chunk_subs = group_tiles / 2;
                    while chunk_subs > 1 {
                        let chunk_ra2_bytes = rows * chunk_subs * sub_pack_bytes;
                        let chunk_plane_bytes = chunk_subs * num_ch * max_sub_pl * 8;
                        if chunk_ra2_bytes + chunk_plane_bytes + tile_bytes <= l3_per_worker {
                            break;
                        }
                        chunk_subs /= 2;
                    }
                    let chunk_subs = chunk_subs.max(1);
                    if use_nt != 0 {
                        resize_f64_buf(&mut sc.ra1, k_primes * rows * 8);
                    }
                    let mut sub0 = 0usize;
                    while sub0 < group_tiles {
                        let sub1 = (sub0 + chunk_subs).min(group_tiles);
                        let mut plane_offs = [usize::MAX; 8];
                        let mut cur_words = 0usize;
                        for sub in sub0..sub1 {
                            if sub_nz[sub] >= 32 {
                                plane_offs[sub] = cur_words;
                                cur_words += num_ch * crate::refk::ntt_planar_stride(sub_nz[sub]);
                            }
                        }
                        let mut plane_addrs = [0usize; 8];
                        if cur_words > 0 {
                            resize_f64_buf(&mut sc.tw_step, cur_words);
                            for sub in sub0..sub1 {
                                let off = plane_offs[sub];
                                if off != usize::MAX {
                                    let ext_cfg = NttSplitCfg {
                                        in_limbs: limbs_i.len() as u64,
                                        start_coeff: ((cg * group_tiles + sub) * tile_cols)
                                            as u64,
                                        count: sub_nz[sub] as u64,
                                        coeff_bits: spec.coeff_bits as u64,
                                        col_stride: cols as u64,
                                        cols_per_row: tile_cols as u64,
                                        p: 0.0,
                                        inv_p: 0.0,
                                    };
                                    unsafe {
                                        let dst_p = sc.tw_step.as_mut_ptr().add(off);
                                        (kernels.active.ntt_split)(
                                            dst_p,
                                            limbs_i.as_ptr(),
                                            &ext_cfg,
                                        );
                                        plane_addrs[sub] = dst_p as usize;
                                    }
                                }
                            }
                        }
                        for (pi, (pc, (&dst_base, step_tc))) in primes
                            .iter()
                            .zip(dst_bases.iter().zip(step_tcs.iter()))
                            .enumerate()
                        {
                            sweep1_staged_chunk_one_op(
                                kernels,
                                sc,
                                spec,
                                tile_cols,
                                cg,
                                sub0,
                                sub1,
                                &sub_nz,
                                &plane_addrs,
                                limbs_i,
                                dst_base,
                                dst_row_bytes,
                                pc.p_u64,
                                pc.col_tab,
                                pc.psi,
                                pc.tw0,
                                step_tc,
                                use_nt,
                                pi * rows * 8,
                            );
                        }
                        sub0 = sub1;
                    }
                    continue;
                }

                let mut plane_offs = [usize::MAX; 8];
                let mut cur_words = 0usize;
                if only_prime.is_none() && can_planar && max_full_plane_words > 0 {
                    for sub in 0..group_tiles {
                        let need = num_ch * crate::refk::ntt_planar_stride(sub_nz[sub]);
                        if sub_nz[sub] >= 32 && cur_words + need <= max_full_plane_words {
                            plane_offs[sub] = cur_words;
                            cur_words += need;
                        }
                    }
                }
                let mut plane_addrs = [0usize; 8];
                let have_planes = cur_words > 0;
                if have_planes {
                    resize_f64_buf(&mut sc.tw_step, cur_words);
                    for sub in 0..group_tiles {
                        let off = plane_offs[sub];
                        if off != usize::MAX {
                            let ext_cfg = NttSplitCfg {
                                in_limbs: limbs_i.len() as u64,
                                start_coeff: ((cg * group_tiles + sub) * tile_cols) as u64,
                                count: sub_nz[sub] as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: tile_cols as u64,
                                p: 0.0,
                                inv_p: 0.0,
                            };
                            unsafe {
                                let dst_p = sc.tw_step.as_mut_ptr().add(off);
                                (kernels.active.ntt_split)(
                                    dst_p,
                                    limbs_i.as_ptr(),
                                    &ext_cfg,
                                );
                                plane_addrs[sub] = dst_p as usize;
                            }
                        }
                    }
                }
                let pass_addrs = have_planes.then_some(&plane_addrs);
                for (pi, (pc, (&dst_base, step_tc))) in primes
                    .iter()
                    .zip(dst_bases.iter().zip(step_tcs.iter()))
                    .enumerate()
                {
                    if only_prime.is_some_and(|want_pi| want_pi != pi) {
                        continue;
                    }
                    sweep1_group_one_op(
                        kernels,
                        sc,
                        spec,
                        tile_cols,
                        cg,
                        limbs_i,
                        dst_base,
                        dst_row_bytes,
                        pc.p_u64,
                        pc.col_tab,
                        pc.psi,
                        pc.tw0,
                        step_tc,
                        use_nt,
                        &mut ra1_is_zero,
                        staged,
                        Some(&sub_nz),
                        pass_addrs,
                    );
                }
            }
        },
    );
    if let Some(t) = t_sw1 {
        NTT_TRACE_SW1_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }
}

/// Sweep 1 for several primes in one pass into unpacked grids (`!spec.packed`).
/// Each worker takes one column tile `ct`, extracts 48-bit planes once into L1/L2 cache,
/// and produces that tile across all `primes` before advancing to the next tile.
#[allow(clippy::too_many_arguments)]
fn ntt_sweep1_all_primes_unpacked(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    ops: &[(&[u64], &[usize], usize)],
    spec: &NttSpec,
    tile_cols_arg: usize,
    primes: &[Sweep1Prime<'_>],
    use_nt: u64,
) {
    if ops.is_empty() || primes.is_empty() {
        return;
    }
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_cols = tile_cols_arg.min(cols).max(1);
    let num_col_tiles = cols / tile_cols;
    let tile_elems = rows * tile_cols;
    let pw = spec.par_workers.max(1);
    let n_ops = ops.len();
    let k_primes = primes.len();
    let units = n_ops * k_primes;
    let prime_fine_tiles = if pw > 1 {
        if n_ops >= 2 {
            (pw / 8).clamp(1, 2).min(num_col_tiles / 4)
        } else {
            (pw / 4).clamp(1, 4).min(num_col_tiles / 4)
        }
    } else {
        0
    };
    let op_fine_tiles = if pw > 1 && n_ops >= 2 {
        pw.min((num_col_tiles - prime_fine_tiles) / 2)
    } else {
        0
    };
    let whole_tiles = num_col_tiles - op_fine_tiles - prime_fine_tiles;
    let op_fine_tasks = op_fine_tiles * n_ops;
    let prime_fine_tasks = prime_fine_tiles * units;
    let can_planar = k_primes >= 2
        && (53..=384).contains(&spec.coeff_bits)
        && primes.iter().all(|pc| pc.p_u64 > (1u64 << 48));
    let can_indep = k_primes >= 2
        && spec.coeff_bits <= 52
        && primes.iter().all(|pc| (1u64 << spec.coeff_bits) <= pc.p_u64);
    let num_ch = (spec.coeff_bits as usize).div_ceil(48);
    let max_plane_words =
        (cache.l3_prime_cap_bytes() / sched.num_workers().max(1) / 8).max(65_536);
    let t_sw1 = ntt_trace_enabled().then(std::time::Instant::now);

    sched.par_for_init_max(
        whole_tiles + op_fine_tasks + prime_fine_tasks,
        1,
        spec.par_workers,
        || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            resize_f64_buf(&mut sc.tile, tile_elems);
            lease
        },
        |lease, task| {
            let sc = lease.get_mut();
            let (ct, only_op, only_prime) = if task < whole_tiles {
                (task, None, None)
            } else if task < whole_tiles + op_fine_tasks {
                let f = task - whole_tiles;
                (whole_tiles + f / n_ops, Some(f % n_ops), None)
            } else {
                let f = task - whole_tiles - op_fine_tasks;
                let u = f % units;
                (
                    whole_tiles + op_fine_tiles + f / units,
                    Some(u / k_primes),
                    Some(u % k_primes),
                )
            };
            let col_start = ct * tile_cols;
            let col_byte_off = col_start * 8;
            let shape = [rows as u64, tile_cols as u64];

            for (oi, &(limbs_i, dst_bases, dst_row_bytes)) in ops.iter().enumerate() {
                if only_op.is_some_and(|want_oi| want_oi != oi) {
                    continue;
                }
                let nz = tile_nz_elems(
                    limbs_i.len(),
                    spec.coeff_bits,
                    rows,
                    cols,
                    col_start,
                    tile_cols,
                );
                let mcfg = MoveCfg {
                    rows: rows as u64,
                    cols: tile_cols as u64,
                    outer_stride_bytes: dst_row_bytes as u64,
                    packed: 0,
                    use_nt,
                };
                if nz == 0 {
                    let mut zeroed = false;
                    for (pi, &dst_base) in dst_bases.iter().enumerate() {
                        if only_prime.is_some_and(|want_pi| want_pi != pi) {
                            continue;
                        }
                        unsafe {
                            let outer_ptr = (dst_base as *mut u8).add(col_byte_off);
                            if use_nt != 0 {
                                if !zeroed {
                                    std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                                    zeroed = true;
                                }
                                (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                            } else {
                                for r in 0..rows {
                                    std::ptr::write_bytes(
                                        outer_ptr.add(r * dst_row_bytes) as *mut f64,
                                        0,
                                        tile_cols,
                                    );
                                }
                            }
                        }
                    }
                    continue;
                }

                let pl_words = num_ch * crate::refk::ntt_planar_stride(nz);
                let use_planar = only_prime.is_none()
                    && can_planar
                    && nz >= 32
                    && pl_words <= max_plane_words;
                let use_indep = only_prime.is_none() && !use_planar && can_indep;
                if use_planar {
                    resize_f64_buf(&mut sc.tw_step, pl_words);
                    let ext_cfg = NttSplitCfg {
                        in_limbs: limbs_i.len() as u64,
                        start_coeff: col_start as u64,
                        count: nz as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: cols as u64,
                        cols_per_row: tile_cols as u64,
                        p: 0.0,
                        inv_p: 0.0,
                    };
                    unsafe {
                        (kernels.active.ntt_split)(
                            sc.tw_step.as_mut_ptr(),
                            limbs_i.as_ptr(),
                            &ext_cfg,
                        );
                    }
                } else if use_indep {
                    resize_f64_buf(&mut sc.tw_step, nz);
                    let pr0 = primes[0].col_tab.pr_fwd;
                    let scfg0 = NttSplitCfg {
                        in_limbs: limbs_i.len() as u64,
                        start_coeff: col_start as u64,
                        count: nz as u64,
                        coeff_bits: spec.coeff_bits as u64,
                        col_stride: cols as u64,
                        cols_per_row: tile_cols as u64,
                        p: pr0.p,
                        inv_p: pr0.inv_p,
                    };
                    unsafe {
                        (kernels.active.ntt_split)(
                            sc.tw_step.as_mut_ptr(),
                            limbs_i.as_ptr(),
                            &scfg0,
                        );
                    }
                }

                for (pi, (pc, &dst_base)) in primes.iter().zip(dst_bases.iter()).enumerate() {
                    if only_prime.is_some_and(|want_pi| want_pi != pi) {
                        continue;
                    }
                    let pr = pc.col_tab.pr_fwd;
                    unsafe {
                        if use_planar {
                            let red_cfg = NttSplitCfg {
                                in_limbs: u64::MAX,
                                start_coeff: 0,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: 0,
                                cols_per_row: 0,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            (kernels.active.ntt_split)(
                                sc.tile.as_mut_ptr(),
                                sc.tw_step.as_ptr() as *const u64,
                                &red_cfg,
                            );
                        } else if use_indep {
                            sc.tile[..nz].copy_from_slice(&sc.tw_step[..nz]);
                        } else {
                            let scfg = NttSplitCfg {
                                in_limbs: limbs_i.len() as u64,
                                start_coeff: col_start as u64,
                                count: nz as u64,
                                coeff_bits: spec.coeff_bits as u64,
                                col_stride: cols as u64,
                                cols_per_row: tile_cols as u64,
                                p: pr.p,
                                inv_p: pr.inv_p,
                            };
                            (kernels.active.ntt_split)(sc.tile.as_mut_ptr(), limbs_i.as_ptr(), &scfg);
                        }
                        if nz < tile_elems {
                            std::ptr::write_bytes(sc.tile.as_mut_ptr().add(nz), 0, tile_elems - nz);
                        }
                        call_ntt_fwd_cols(
                            kernels,
                            sc.tile.as_mut_ptr(),
                            pc.col_tab.tw.as_ptr(),
                            &shape,
                            &pr,
                            &mut sc.col_scratch,
                        );
                    }
                    apply_tile_twist(
                        kernels,
                        &mut sc.tile[..tile_elems],
                        pc.tw0,
                        rows,
                        tile_cols,
                        col_start,
                        pc.psi,
                        pc.p_u64,
                        pr.p,
                        pr.inv_p,
                        &mut sc.tw_row,
                        &mut sc.s_u64,
                    );
                    unsafe {
                        let outer_ptr = (dst_base as *mut u8).add(col_byte_off);
                        (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
                    }
                }
            }
        },
    );
    if let Some(t) = t_sw1 {
        NTT_TRACE_SW1_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ntt_sweep1_multi_prime_unpacked(
    kernels: &Kernels,
    sched: &Scheduler,
    ops: &[(&[u64], usize, usize)],
    spec: &NttSpec,
    tile_cols_arg: usize,
    p_u64: u64,
    col_tab: &PrimeTables,
    psi: PsiParams,
    tw0_slice: &[f64],
    use_nt: u64,
) {
    ntt_sweep1_multi_prime_ext(
        kernels,
        sched,
        ops,
        spec,
        tile_cols_arg,
        p_u64,
        col_tab,
        psi,
        tw0_slice,
        use_nt,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn ntt_sweep1_multi_prime_ext(
    kernels: &Kernels,
    sched: &Scheduler,
    ops: &[(&[u64], usize, usize)],
    spec: &NttSpec,
    tile_cols_arg: usize,
    p_u64: u64,
    col_tab: &PrimeTables,
    psi: PsiParams,
    tw0_slice: &[f64],
    use_nt: u64,
    dst_packed: bool,
) {
    if ops.is_empty() {
        return;
    }
    debug_assert!(ops.len() <= 8);
    let rows = spec.rows;
    let cols = spec.cols;
    let tile_cols = tile_cols_arg.min(cols).max(1);
    let num_col_tiles = cols / tile_cols;
    let tile_elems = rows * tile_cols;
    let pw = spec.par_workers.max(1);
    let sw1_chunk = (64 / tile_cols)
        .max(1)
        .min((num_col_tiles / (pw * 8)).max(1))
        .min(4);
    let pr = col_tab.pr_fwd;
    let col_tw_ptr = col_tab.tw.as_ptr() as usize;
    let may_share_tw = ops.len() >= 3 && !tw0_is_full(tw0_slice, cols * rows);

    let run_one_tile = |sc: &mut NttTlsScratch, ct: usize| {
        let col_start = ct * tile_cols;
        let col_byte_off = if dst_packed {
            (col_start / 8) * 50
        } else {
            col_start * 8
        };
        let shape = [rows as u64, tile_cols as u64];

        let mut nz_arr = [0usize; 8];
        let mut active_ops = 0usize;
        for (idx, &(limbs_i, _, _)) in ops.iter().enumerate() {
            let nz = tile_nz_elems(
                limbs_i.len(),
                spec.coeff_bits,
                rows,
                cols,
                col_start,
                tile_cols,
            );
            nz_arr[idx] = nz;
            if nz > 0 {
                active_ops += 1;
            }
        }

        let skip_twist = rows == 0
            || tile_cols == 0
            || (col_start == 0 && tile_cols == 1)
            || (!psi.negacyclic && rows <= 1);
        let start_r = usize::from(!psi.negacyclic);
        let off = start_r * tile_cols;
        let has_full_tw0 = tw0_is_full(tw0_slice, (col_start + tile_cols) * rows);
        let shared_tw =
            active_ops >= 3 && col_start > 0 && !skip_twist && !has_full_tw0 && tile_elems > off;

        if shared_tw {
            let vcfg_tw = NttVecCfg {
                n: (tile_elems - off) as u64,
                p: pr.p,
                inv_p: pr.inv_p,
                sub_second: tile_cols as u64,
            };
            sc.ra1[off..tile_elems].fill(1.0);
            if let Some(rs) = tw0_resolve_row_scales(
                kernels,
                tw0_slice,
                rows,
                tile_cols,
                col_start,
                start_r,
                pr.p,
                pr.inv_p,
                &mut sc.tw_row,
            ) {
                unsafe {
                    (kernels.active.ntt_pmul)(
                        sc.ra1.as_mut_ptr().add(off),
                        tw0_slice.as_ptr().add(off),
                        rs.as_ptr().add(start_r),
                        &vcfg_tw,
                    );
                }
            } else {
                fill_row_scales_u64(rows, col_start, psi, p_u64, &mut sc.s_u64);
                if sc.tw_row.len() < rows {
                    sc.tw_row.resize(rows, 1.0);
                }
                for r in start_r..rows {
                    sc.tw_row[r] = sc.s_u64[r] as f64;
                }
                unsafe {
                    (kernels.active.ntt_pmul)(
                        sc.ra1.as_mut_ptr().add(off),
                        tw0_slice.as_ptr().add(off),
                        sc.tw_row.as_ptr().add(start_r),
                        &vcfg_tw,
                    );
                }
            }
        }

        for (idx, &(limbs_i, dst_base, dst_row_bytes)) in ops.iter().enumerate() {
            let nz = nz_arr[idx];
            if nz == 0 {
                continue;
            }
            let mcfg = MoveCfg {
                rows: rows as u64,
                cols: tile_cols as u64,
                outer_stride_bytes: dst_row_bytes as u64,
                packed: u64::from(dst_packed),
                use_nt,
            };
            unsafe {
                let scfg = NttSplitCfg {
                    in_limbs: limbs_i.len() as u64,
                    start_coeff: col_start as u64,
                    count: nz as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    col_stride: cols as u64,
                    cols_per_row: tile_cols as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                };
                (kernels.active.ntt_split)(sc.tile.as_mut_ptr(), limbs_i.as_ptr(), &scfg);
                if nz < tile_elems {
                    std::ptr::write_bytes(sc.tile.as_mut_ptr().add(nz), 0, tile_elems - nz);
                }
            }
            unsafe {
                call_ntt_fwd_cols(
                    kernels,
                    sc.tile.as_mut_ptr(),
                    col_tw_ptr as *const f64,
                    &shape,
                    &pr,
                    &mut sc.col_scratch,
                );
            }
            if shared_tw {
                let vcfg_mul = NttVecCfg {
                    n: (tile_elems - off) as u64,
                    p: pr.p,
                    inv_p: pr.inv_p,
                    sub_second: 0,
                };
                unsafe {
                    (kernels.active.ntt_pmul)(
                        sc.tile.as_mut_ptr().add(off),
                        sc.tile.as_ptr().add(off),
                        sc.ra1.as_ptr().add(off),
                        &vcfg_mul,
                    );
                }
            } else {
                apply_tile_twist(
                    kernels,
                    &mut sc.tile[..tile_elems],
                    tw0_slice,
                    rows,
                    tile_cols,
                    col_start,
                    psi,
                    p_u64,
                    pr.p,
                    pr.inv_p,
                    &mut sc.tw_row,
                    &mut sc.s_u64,
                );
            }
            unsafe {
                let outer_ptr = (dst_base as *mut u8).add(col_byte_off);
                (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg);
            }
        }

        if active_ops < ops.len() {
            let mut zeroed = false;
            for (idx, &(_, dst_base, dst_row_bytes)) in ops.iter().enumerate() {
                if nz_arr[idx] != 0 {
                    continue;
                }
                unsafe {
                    let outer_ptr = (dst_base as *mut u8).add(col_byte_off);
                    if dst_packed {
                        let tile_pk_bytes = (tile_cols / 8) * 50;
                        if use_nt != 0 && tile_cols >= 32 {
                            if !zeroed {
                                std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                                zeroed = true;
                            }
                            let mcfg_z = MoveCfg {
                                rows: rows as u64,
                                cols: tile_cols as u64,
                                outer_stride_bytes: dst_row_bytes as u64,
                                packed: 1,
                                use_nt,
                            };
                            (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg_z);
                        } else {
                            for r in 0..rows {
                                std::ptr::write_bytes(
                                    outer_ptr.add(r * dst_row_bytes),
                                    0,
                                    tile_pk_bytes,
                                );
                            }
                        }
                    } else if use_nt != 0 {
                        if !zeroed {
                            std::ptr::write_bytes(sc.tile.as_mut_ptr(), 0, tile_elems);
                            zeroed = true;
                        }
                        let mcfg_z = MoveCfg {
                            rows: rows as u64,
                            cols: tile_cols as u64,
                            outer_stride_bytes: dst_row_bytes as u64,
                            packed: 0,
                            use_nt,
                        };
                        (kernels.active.scatter)(outer_ptr, sc.tile.as_ptr(), &mcfg_z);
                    } else {
                        for r in 0..rows {
                            std::ptr::write_bytes(
                                outer_ptr.add(r * dst_row_bytes) as *mut f64,
                                0,
                                tile_cols,
                            );
                        }
                    }
                }
            }
        }
    };

    let t_sw1 = ntt_trace_enabled().then(std::time::Instant::now);
    let coalesce_256 = dst_packed && sweep1_groups_256(spec, tile_cols);
    if coalesce_256 {
        let group_tiles = 256 / tile_cols;
        let num_groups = cols / 256;
        let strip_max = 32usize.min(rows);
        let mut step_tc = Vec::new();
        if !tw0_is_full(tw0_slice, cols * rows)
            && !tw0_has_row_scales(tw0_slice, rows, tile_cols)
        {
            fill_row_scales_u64(rows, tile_cols, psi, p_u64, &mut step_tc);
        }
        let staged = sweep1_stages_groups(sched, spec);
        let n_ops = ops.len();
        let op_fine = if pw > 1 && n_ops >= 2 {
            pw.min(num_groups / 3)
        } else {
            0
        };
        let whole_groups = num_groups - op_fine;
        sched.par_for_init_max(
            whole_groups + op_fine * n_ops,
            1,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, group_tiles * tile_elems);
                resize_f64_buf(&mut sc.ra1, strip_max * 256);
                if staged {
                    resize_f64_buf(&mut sc.ra2, spec.rows * 200);
                }
                lease
            },
            |lease, task| {
                let sc = lease.get_mut();
                let mut ra1_is_zero = false;
                let (cg, only_oi) = if task < whole_groups {
                    (task, None)
                } else {
                    let f = task - whole_groups;
                    (whole_groups + f / n_ops, Some(f % n_ops))
                };
                for (oi, &(limbs_i, dst_base, dst_row_bytes)) in ops.iter().enumerate() {
                    if only_oi.is_some_and(|w| w != oi) {
                        continue;
                    }
                    sweep1_group_one_op(
                        kernels,
                        sc,
                        spec,
                        tile_cols,
                        cg,
                        limbs_i,
                        dst_base,
                        dst_row_bytes,
                        p_u64,
                        col_tab,
                        psi,
                        tw0_slice,
                        &step_tc,
                        use_nt,
                        &mut ra1_is_zero,
                        staged,
                        None,
                        None,
                    );
                }
            },
        );
    } else if dst_packed {
        debug_assert!(cols.is_multiple_of(256) && 256usize.is_multiple_of(tile_cols));
        let group_tiles = 256 / tile_cols;
        let num_groups = cols / 256;
        let grp_chunk = (num_groups / (pw * 8)).clamp(1, 2);
        sched.par_for_init_max(
            num_groups,
            grp_chunk,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, tile_elems);
                if may_share_tw {
                    resize_f64_buf(&mut sc.ra1, tile_elems);
                }
                lease
            },
            |lease, cg| {
                let sc = lease.get_mut();
                for sub in 0..group_tiles {
                    run_one_tile(sc, cg * group_tiles + sub);
                }
            },
        );
    } else {
        sched.par_for_init_max(
            num_col_tiles,
            sw1_chunk,
            spec.par_workers,
            || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                resize_f64_buf(&mut sc.tile, tile_elems);
                if may_share_tw {
                    resize_f64_buf(&mut sc.ra1, tile_elems);
                }
                lease
            },
            |lease, ct| {
                run_one_tile(lease.get_mut(), ct);
            },
        );
    }
    if let Some(t) = t_sw1 {
        NTT_TRACE_SW1_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_SW1_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ntt_sweep1_single_prime_unpacked(
    kernels: &Kernels,
    sched: &Scheduler,
    limbs: &[u64],
    spec: &NttSpec,
    tile_cols_arg: usize,
    p_u64: u64,
    col_tab: &PrimeTables,
    psi: PsiParams,
    tw0_slice: &[f64],
    dst_base: usize,
    dst_row_bytes: usize,
) {
    ntt_sweep1_multi_prime_unpacked(
        kernels,
        sched,
        &[(limbs, dst_base, dst_row_bytes)],
        spec,
        tile_cols_arg,
        p_u64,
        col_tab,
        psi,
        tw0_slice,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_mul_with_cached_op(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    telemetry: &Telemetry,
    a_limbs: &[u64],
    mut b_op: NttOperand<'_>,
    spec: NttSpec,
    out_limbs: usize,
) -> Result<(LimbVec, u64), crate::store::StoreError> {
    let b_spec = b_op.as_ref().spec;
    let _pt = PathTimer::start("mul_cached", &spec, &[a_limbs.len(), out_limbs]);
    let same_grid = spec.n == b_spec.n
        && spec.rows == b_spec.rows
        && spec.cols == b_spec.cols
        && spec.k_primes == b_spec.k_primes
        && spec.negacyclic == b_spec.negacyclic;
    let b_in_ram = matches!(
        &b_op.as_ref().storage,
        ResidueStorage::InRam { bufs, .. } if bufs.len() == spec.k_primes
    );

    if same_grid
        && b_in_ram
        && spec.rows == 1
        && !spec.packed
        && !b_spec.packed
        && !spec.out_of_core
    {
        let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
        let k = spec.k_primes;
        let n = spec.n;
        let p_indep =
            k >= 2 && spec.coeff_bits <= 52 && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
        let num_ch = (spec.coeff_bits as usize).div_ceil(48);
        let nz_a = nz_elems_1d(a_limbs.len(), spec.coeff_bits, n);
        let p_chunk_cand = !p_indep
            && k >= 2
            && (53..=384).contains(&spec.coeff_bits)
            && crt.primes_u64[k - 1] > (1u64 << 48);
        let max_1d_ch_bytes =
            (cache.l3_prime_cap_bytes() / sched.num_workers().max(1)).max(512 * 1024);
        let p_chunk = p_chunk_cand
            && ch_words_for_1d(nz_a, p_indep, true, num_ch) * 8 <= max_1d_ch_bytes;
        let t0 = cache.get_twiddles_ref(crt.primes_u64[0], n, spec.negacyclic);

        let ResidueStorage::InRam { bufs: b_bufs, .. } = &b_op.as_ref().storage else {
            unreachable!();
        };
        let b_bufs_ptr = b_bufs.as_ptr() as usize;

        let mut f_lease = Fused1dLease::acquire();
        let f1d = f_lease.get_mut();
        let ch_need = ch_words_for_1d(nz_a, p_indep, p_chunk, num_ch);
        let res_stride = res_stride_1d(n);
        resize_f64_buf(&mut f1d.m_ch, ch_need);
        resize_f64_buf(&mut f1d.m_t_res, k * res_stride);

        let t_f1 = ntt_trace_enabled().then(std::time::Instant::now);
        let mut cur_off = 0usize;
        let prep_a = prep_1d_operand(
            kernels,
            a_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let m_ch_ptr = f1d.m_ch.as_ptr() as usize;
        let res_ptr = f1d.m_t_res.as_mut_ptr() as usize;

        let eval_prime_cached = |pi: usize| {
            let p_u64 = crt.primes_u64[pi];
            let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
            unsafe {
                let dst = (res_ptr as *mut f64).add(pi * res_stride);
                let pb = (*(b_bufs_ptr as *const AlignedBytes).add(pi)).as_ptr() as *const f64;
                eval_1d_fwd_pmul_inv(
                    kernels,
                    a_limbs,
                    prep_a,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    dst,
                    pb,
                );
            }
        };
        if spec.par_workers <= 1 || k <= 1 {
            for pi in 0..k {
                eval_prime_cached(pi);
            }
        } else {
            sched.par_for_leaf_max(k, 1, spec.par_workers, eval_prime_cached);
        }
        if let Some(t) = t_f1 {
            NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if kernels.count_work && n > 1 {
            let bf_fwd = ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64);
            let bf_inv = (((n as u64) / 2) * (n.ilog2() as u64) + (n as u64)) * (k as u64);
            kernels
                .stats
                .ntt_butterflies
                .fetch_add(bf_fwd + bf_inv, Ordering::Relaxed);
            kernels
                .stats
                .bytes_moved
                .fetch_add((a_limbs.len() * 8 + k * n * 8) as u64, Ordering::Relaxed);
        }
        return Ok(crt_1d_from_flat(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &f1d.m_t_res[..(k * res_stride)],
            res_stride,
            spec.negacyclic,
            out_limbs,
        ));
    }

    if same_grid
        && b_in_ram
        && spec.rows > 1
        && spec.cols >= 64
        && spec.extra_sweeps == 0
        && b_spec.extra_sweeps == 0
        && !spec.out_of_core
    {
        let rows = spec.rows;
        let cols = spec.cols;
        let unpacked_row_bytes = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
        let unpacked_prime_bytes = rows * unpacked_row_bytes;
        let est_conc = sched.num_workers().div_ceil(spec.par_workers.max(1)).max(1);
        let my_l3_cap = cache.l3_prime_cap_bytes() / est_conc;
        let dram_pbp = unpacked_prime_bytes > my_l3_cap / 2;
        let base_tc = spec.tile_cols.min(cols).max(1);
        let tile_cols = if rows >= 128 && cols.is_multiple_of(32) {
            base_tc.min(32)
        } else {
            base_tc
        };
        let pack_sw1 = can_pack_sweep1(&spec, tile_cols);
        let need_g0 = (spec.packed || dram_pbp) && !pack_sw1;
        {
            let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
            let k = spec.k_primes;
            let row_bytes = spec.outer_row_bytes();
            let total_grid_bytes = rows * row_bytes;
            let b_packed = b_spec.packed;
            let b_row_bytes = b_spec.outer_row_bytes();

            let (b_bufs_ptr, b_rt, b_inplace) = match &mut b_op {
                NttOperand::ReuseInPlace(bc) => {
                    let rt = bc.rows_transformed;
                    if !rt {
                        bc.rows_transformed = true;
                    }
                    let ResidueStorage::InRam { bufs, .. } = &mut bc.storage else {
                        unreachable!();
                    };
                    (bufs.as_mut_ptr() as usize, rt, !rt)
                }
                NttOperand::Owned(bc) => {
                    let rt = bc.rows_transformed;
                    let inplace = !rt && !b_packed;
                    if inplace {
                        bc.rows_transformed = true;
                    }
                    let ResidueStorage::InRam { bufs, .. } = &mut bc.storage else {
                        unreachable!();
                    };
                    (bufs.as_mut_ptr() as usize, rt, inplace)
                }
                NttOperand::Shared(bc) => {
                    let rt = bc.rows_transformed;
                    let ResidueStorage::InRam { bufs, .. } = &bc.storage else {
                        unreachable!();
                    };
                    (bufs.as_ptr() as usize, rt, false)
                }
            };

            let t_al = std::time::Instant::now();
            let mut prod_grid =
                cache.take_byte_bufs_transient_committed(k, total_grid_bytes, sched, spec.par_workers);
            let mut g0 = if need_g0 {
                cache.take_byte_bufs_transient_committed(1, unpacked_prime_bytes, sched, spec.par_workers)
            } else {
                Vec::new()
            };
            NTT_TRACE_ALLOC_NS.fetch_add(t_al.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

            let col_tables = cache.get_twiddles_refs_par(
                sched,
                spec.par_workers,
                &crt.primes_u64,
                k,
                rows,
                spec.negacyclic,
            );
            let row_tables = cache.get_twiddles_refs_par(
                sched,
                spec.par_workers,
                &crt.primes_u64,
                k,
                cols,
                false,
            );
            let psi_fwd: Vec<PsiParams> = (0..k)
                .map(|pi| {
                    build_psi_params(
                        crt.primes_u64[pi],
                        col_tables[pi].g_root,
                        spec.n,
                        spec.negacyclic,
                        false,
                    )
                })
                .collect();
            let tw_tile0_fwd = cache.get_tile0_twiddles_par(
                kernels,
                sched,
                spec.par_workers,
                &crt.primes_u64,
                &col_tables,
                k,
                spec.n,
                rows,
                tile_cols,
                spec.negacyclic,
                false,
            );

            let sw2_w = sw2_workers(sched, &spec);
            let chunk_cap = sw2_chunk_cap(cols);
            let row_chunk = (rows / (sw2_w * 8)).clamp(1, chunk_cap);
            let prod_grid_ptr = prod_grid.as_mut_ptr() as usize;
            let a_stride = if need_g0 {
                unpacked_row_bytes
            } else {
                row_bytes
            };
            let sw1_nt = u64::from(spec.use_nt && rows * a_stride > my_l3_cap);
            let once_nt = u64::from(spec.use_nt && k * total_grid_bytes > my_l3_cap);
            let b_wb_nt = u64::from(
                b_spec.use_nt && k * rows * b_row_bytes > cache.l3_prime_cap_bytes(),
            );

            // Packed or in-place unpacked: `a` is made for all primes in one pass over its limbs.
            let sw1_once_packed = pack_sw1 && sweep1_groups_256(&spec, tile_cols);
            let sw1_once_unpacked = !sw1_once_packed
                && !need_g0
                && !spec.packed
                && (cols / tile_cols) >= spec.par_workers.max(1);
            let sw1_once = sw1_once_packed || sw1_once_unpacked;
            if sw1_once_packed {
                let a_bases: Vec<usize> =
                    prod_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
                let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
                ntt_sweep1_all_primes(
                    kernels,
                    cache,
                    sched,
                    &[(a_limbs, &a_bases, row_bytes)],
                    &spec,
                    tile_cols,
                    &primes,
                    once_nt,
                );
            } else if sw1_once_unpacked {
                let a_bases: Vec<usize> =
                    prod_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
                let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
                ntt_sweep1_all_primes_unpacked(
                    kernels,
                    cache,
                    sched,
                    &[(a_limbs, &a_bases, row_bytes)],
                    &spec,
                    tile_cols,
                    &primes,
                    once_nt,
                );
            }

            let sw2_cached_row = |sc: &mut NttTlsScratch, pi: usize, r: usize, a_base: usize| {
                let r_tab = row_tables[pi];
                let pr_fwd = r_tab.pr_fwd;
                let pr_row_inv = r_tab.pr_inv;
                let r_tw_ptr = r_tab.tw.as_ptr() as usize;
                let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
                let b_row_off = r * b_row_bytes;
                let mcfg_b = MoveCfg {
                    rows: 1,
                    cols: cols as u64,
                    outer_stride_bytes: b_row_bytes as u64,
                    packed: u64::from(b_packed),
                    use_nt: if b_packed && !b_inplace { b_wb_nt } else { 0 },
                };
                let pb_ptr = if b_inplace {
                    if b_packed {
                        let mcfg_wb = MoveCfg {
                            use_nt: b_wb_nt,
                            ..mcfg_b
                        };
                        unsafe {
                            let dst_u8 = (*(b_bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(b_row_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), dst_u8, &mcfg_b);
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.scatter)(dst_u8, sc.rb1.as_ptr(), &mcfg_wb);
                        }
                        sc.rb1.as_ptr()
                    } else {
                        let p = unsafe {
                            (*(b_bufs_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(b_row_off) as *mut f64
                        };
                        unsafe {
                            call_ntt_fwd(kernels, 
                                p,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                        }
                        p as *const f64
                    }
                } else if b_rt && !b_packed {
                    unsafe {
                        (*(b_bufs_ptr as *const AlignedBytes).add(pi))
                            .as_ptr()
                            .add(b_row_off) as *const f64
                    }
                } else {
                    unsafe {
                        let src = (*(b_bufs_ptr as *const AlignedBytes).add(pi))
                            .as_ptr()
                            .add(b_row_off);
                        (kernels.active.gather)(sc.rb1.as_mut_ptr(), src, &mcfg_b);
                        if !b_rt {
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                        }
                    }
                    sc.rb1.as_ptr()
                };

                let a_work = if pack_sw1 || dram_pbp {
                    unsafe {
                        let src_u8 = (a_base as *const u8).add(r * a_stride);
                        if pack_sw1 {
                            let mcfg_a = MoveCfg {
                                rows: 1,
                                cols: cols as u64,
                                outer_stride_bytes: a_stride as u64,
                                packed: 1,
                                use_nt: if sw1_once_packed { once_nt } else { sw1_nt },
                            };
                            (kernels.active.gather)(sc.ra1.as_mut_ptr(), src_u8, &mcfg_a);
                        } else {
                            std::ptr::copy_nonoverlapping(
                                src_u8 as *const f64,
                                sc.ra1.as_mut_ptr(),
                                cols,
                            );
                        }
                    }
                    sc.ra1.as_mut_ptr()
                } else {
                    unsafe { (a_base as *mut u8).add(r * a_stride) as *mut f64 }
                };

                unsafe {
                    call_ntt_fwd_pmul_inv(
                        kernels,
                        a_work,
                        r_tw_ptr as *const f64,
                        r_inv_tw_ptr as *const f64,
                        pb_ptr,
                        cols as u64,
                        &pr_row_inv,
                    );
                    if need_g0 || pack_sw1 {
                        let mcfg_out = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: u64::from(spec.packed),
                            use_nt: ROW_SWEEP_STREAMS,
                        };
                        let dst_u8 = (*(prod_grid_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, a_work as *const f64, &mcfg_out);
                    }
                }
            };
            let sw2_cached_scratch = || {
                let mut lease = ScratchLease::acquire();
                let sc = lease.get_mut();
                if pack_sw1 || dram_pbp {
                    resize_f64_buf(&mut sc.ra1, cols);
                }
                if b_packed || !b_rt {
                    resize_f64_buf(&mut sc.rb1, cols);
                }
                lease
            };

            if sw1_once {
                let t_sw2 = std::time::Instant::now();
                let a_bases: Vec<usize> =
                    prod_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
                let chunk_cap = sw2_chunk_cap(cols);
                let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
                sched.par_for_init_max(
                    k * rows,
                    all_chunk,
                    sw2_w,
                    sw2_cached_scratch,
                    |lease, task| {
                        let (pi, r) = (task / rows, task % rows);
                        sw2_cached_row(lease.get_mut(), pi, r, a_bases[pi]);
                    },
                );
                for g in &mut prod_grid {
                    unsafe {
                        g.set_len(total_grid_bytes);
                    }
                }
                NTT_TRACE_SW2_NS
                    .fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                NTT_IDLE_SW2_NS
                    .fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
            } else {
                for pi in 0..k {
                    let p_u64 = crt.primes_u64[pi];
                    let a_base = if need_g0 {
                        g0[0].as_mut_ptr() as usize
                    } else {
                        prod_grid[pi].as_mut_ptr() as usize
                    };
                    ntt_sweep1_multi_prime_ext(
                        kernels,
                        sched,
                        &[(a_limbs, a_base, a_stride)],
                        &spec,
                        tile_cols,
                        p_u64,
                        col_tables[pi],
                        psi_fwd[pi],
                        &tw_tile0_fwd[pi],
                        sw1_nt,
                        pack_sw1,
                    );

                    let t_sw2 = std::time::Instant::now();
                    sched.par_for_init_max(
                        rows,
                        row_chunk,
                        sw2_w,
                        sw2_cached_scratch,
                        |lease, r| {
                            sw2_cached_row(lease.get_mut(), pi, r, a_base);
                        },
                    );
                    unsafe {
                        prod_grid[pi].set_len(total_grid_bytes);
                    }
                    NTT_TRACE_SW2_NS
                        .fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    NTT_IDLE_SW2_NS
                        .fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
                }
            }
            if need_g0 {
                unsafe {
                    g0[0].set_len(unpacked_prime_bytes);
                }
                cache.put_byte_bufs(g0);
            }
            if kernels.count_work {
                let bf_cols = ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
                kernels
                    .stats
                    .ntt_butterflies
                    .fetch_add(bf_cols, Ordering::Relaxed);
                kernels
                    .stats
                    .bytes_moved
                    .fetch_add((a_limbs.len() * 8) as u64, Ordering::Relaxed);
            }
            let fwd_row_ops = 1u64 + u64::from(!b_rt);
            return ntt_sweep3_and_crt(
                kernels,
                cache,
                sched,
                &spec,
                crt,
                prod_grid,
                spec.negacyclic,
                out_limbs,
                Some(fwd_row_ops),
            );
        }
    }

    let mut fa = ntt_forward(kernels, cache, sched, store, a_limbs, spec)?;
    let res = ntt_combine_and_inverse(
        kernels,
        cache,
        sched,
        store,
        telemetry,
        &mut fa,
        Some(b_op),
        None,
        out_limbs,
    );
    fa.release_store(store);
    res
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_mul_slices(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    store: Option<&Store>,
    telemetry: &Telemetry,
    a_limbs: &[u64],
    b_limbs: &[u64],
    spec: NttSpec,
    out_limbs: usize,
) -> Result<(LimbVec, u64), crate::store::StoreError> {
    let same = std::ptr::eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len();
    let _pt = PathTimer::start("mul_slices", &spec, &[a_limbs.len(), b_limbs.len(), out_limbs]);

    if spec.rows == 1 && !spec.packed && !spec.out_of_core {
        let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
        let k = spec.k_primes;
        let n = spec.n;
        let p_indep =
            k >= 2 && spec.coeff_bits <= 52 && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
        let num_ch = (spec.coeff_bits as usize).div_ceil(48);
        let t0 = cache.get_twiddles_ref(crt.primes_u64[0], n, spec.negacyclic);

        let mut f_lease = Fused1dLease::acquire();
        let f1d = f_lease.get_mut();
        let nz_a = nz_elems_1d(a_limbs.len(), spec.coeff_bits, n);
        let nz_b = if same {
            0
        } else {
            nz_elems_1d(b_limbs.len(), spec.coeff_bits, n)
        };
        let p_chunk_cand = !p_indep
            && k >= 2
            && (53..=384).contains(&spec.coeff_bits)
            && crt.primes_u64[k - 1] > (1u64 << 48);
        let cand_ch_words = ch_words_for_1d(nz_a, p_indep, true, num_ch)
            + if same {
                0
            } else {
                ch_words_for_1d(nz_b, p_indep, true, num_ch)
            };
        let max_1d_ch_bytes =
            (cache.l3_prime_cap_bytes() / sched.num_workers().max(1)).max(512 * 1024);
        let p_chunk = p_chunk_cand && cand_ch_words * 8 <= max_1d_ch_bytes;
        let ch_need = ch_words_for_1d(nz_a, p_indep, p_chunk, num_ch)
            + if same {
                0
            } else {
                ch_words_for_1d(nz_b, p_indep, p_chunk, num_ch)
            };
        let res_stride = res_stride_1d(n);
        resize_f64_buf(&mut f1d.m_ch, ch_need);
        resize_f64_buf(&mut f1d.m_t_res, k * res_stride);

        let t_f1 = ntt_trace_enabled().then(std::time::Instant::now);
        let mut cur_off = 0usize;
        let prep_a = prep_1d_operand(
            kernels,
            a_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_b = if same {
            prep_a
        } else {
            prep_1d_operand(
                kernels,
                b_limbs,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            )
        };
        let m_ch_ptr = f1d.m_ch.as_ptr() as usize;
        let res_ptr = f1d.m_t_res.as_mut_ptr() as usize;

        let run_1d_prime = |sc: &mut NttTlsScratch, pi: usize| {
            let p_u64 = crt.primes_u64[pi];
            let tables = cache.get_twiddles_ref(p_u64, n, spec.negacyclic);
            unsafe {
                let dst = (res_ptr as *mut f64).add(pi * res_stride);
                let pb_ptr = if same {
                    dst as *const f64
                } else {
                    let pb = sc.rb1.as_mut_ptr();
                    eval_1d_fwd(
                        kernels,
                        b_limbs,
                        prep_b,
                        m_ch_ptr as *const f64,
                        tables,
                        n,
                        spec.coeff_bits,
                        pb,
                    );
                    pb as *const f64
                };
                eval_1d_fwd_pmul_inv(
                    kernels,
                    a_limbs,
                    prep_a,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    dst,
                    pb_ptr,
                );
            }
        };
        if spec.par_workers <= 1 || k <= 1 {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            if !same {
                resize_f64_buf(&mut sc.rb1, n);
            }
            for pi in 0..k {
                run_1d_prime(sc, pi);
            }
        } else {
            sched.par_for_init_max(
                k,
                1,
                spec.par_workers,
                || {
                    let mut lease = ScratchLease::acquire();
                    if !same {
                        resize_f64_buf(&mut lease.get_mut().rb1, n);
                    }
                    lease
                },
                |lease, pi| run_1d_prime(lease.get_mut(), pi),
            );
        }
        if let Some(t) = t_f1 {
            NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if kernels.count_work && n > 1 {
            let n_fwd = if same { 1u64 } else { 2u64 };
            let bf_fwd = n_fwd * ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64);
            let bf_inv = (((n as u64) / 2) * (n.ilog2() as u64) + (n as u64)) * (k as u64);
            kernels
                .stats
                .ntt_butterflies
                .fetch_add(bf_fwd + bf_inv, Ordering::Relaxed);
            let limb_b = if same {
                a_limbs.len() * 8
            } else {
                (a_limbs.len() + b_limbs.len()) * 8
            };
            kernels
                .stats
                .bytes_moved
                .fetch_add((limb_b + k * n * 8) as u64, Ordering::Relaxed);
        }
        return Ok(crt_1d_from_flat(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &f1d.m_t_res[..(k * res_stride)],
            res_stride,
            spec.negacyclic,
            out_limbs,
        ));
    }

    if spec.rows > 1 && spec.cols >= 64 && spec.extra_sweeps == 0 && !spec.out_of_core {
        let rows = spec.rows;
        let cols = spec.cols;
        let unpacked_row_bytes = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
        let unpacked_prime_bytes = rows * unpacked_row_bytes;
        let est_conc = sched.num_workers().div_ceil(spec.par_workers.max(1)).max(1);
        let my_l3_cap = cache.l3_prime_cap_bytes() / est_conc;
        let sw1_ops = 1usize + usize::from(!same);
        let dram_pbp = sw1_ops * unpacked_prime_bytes > my_l3_cap;
        let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
        let k = spec.k_primes;
        let row_bytes = spec.outer_row_bytes();
        let total_grid_bytes = rows * row_bytes;
        let base_tc = spec.tile_cols.min(cols).max(1);
        let tile_cols = if rows >= 128 && cols.is_multiple_of(32) {
            base_tc.min(32)
        } else {
            base_tc
        };
        let pack_sw1 = can_pack_sweep1(&spec, tile_cols);
        let need_a_work = (spec.packed || dram_pbp) && !pack_sw1;
        // Sweep 1 in one pass for all primes: `a` goes straight into the product grids, and `b`
        // too where there is room for a work grid per prime instead of one.
        let sw1_once_packed = pack_sw1 && sweep1_groups_256(&spec, tile_cols);
        let sw1_once_unpacked = !sw1_once_packed
            && !need_a_work
            && !spec.packed
            && (cols / tile_cols) >= spec.par_workers.max(1);
        let sw1_once_a = sw1_once_packed || sw1_once_unpacked;
        let sw1_once_unpacked_b = sw1_once_unpacked
            && !dram_pbp
            && !same
            && 2 * k * unpacked_prime_bytes <= my_l3_cap
            && room_for_work_grids(
                est_conc.saturating_mul(2 * k * unpacked_prime_bytes),
                (k - 1) * unpacked_prime_bytes,
            );
        let with_grid_a_prime =
            ((2 * k) as u64).saturating_mul(total_grid_bytes as u64);
        let conc_grid_bytes = (est_conc as u64).saturating_mul(with_grid_a_prime);
        let sw1_once_b = (sw1_once_packed
            && !same
            && 2 * total_grid_bytes > my_l3_cap
            && conc_grid_bytes <= cache.work_grid_room()
            && room_for_work_grids(
                conc_grid_bytes.min(usize::MAX as u64) as usize,
                (k - 1) * total_grid_bytes,
            ))
            || sw1_once_unpacked_b;
        let need_work = usize::from(need_a_work)
            + if sw1_once_b { k } else { usize::from(!same) };
        let work_prime_bytes = if pack_sw1 {
            total_grid_bytes
        } else {
            unpacked_prime_bytes
        };
        let work_row_bytes = if pack_sw1 {
            row_bytes
        } else {
            unpacked_row_bytes
        };

        let t_al = std::time::Instant::now();
        let (mut prod_grid, mut work_grids) = if need_work > 0 && work_prime_bytes == total_grid_bytes
        {
            let mut all = cache.take_byte_bufs_transient_committed(
                k + need_work,
                total_grid_bytes,
                sched,
                spec.par_workers,
            );
            let wg = all.split_off(k);
            (all, wg)
        } else {
            let pg = cache.take_byte_bufs_transient_committed(
                k,
                total_grid_bytes,
                sched,
                spec.par_workers,
            );
            let wg = if need_work > 0 {
                cache.take_byte_bufs_transient_committed(
                    need_work,
                    work_prime_bytes,
                    sched,
                    spec.par_workers,
                )
            } else {
                Vec::new()
            };
            (pg, wg)
        };
        NTT_TRACE_ALLOC_NS.fetch_add(t_al.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

        let col_tables = cache.get_twiddles_refs_par(
            sched,
            spec.par_workers,
            &crt.primes_u64,
            k,
            rows,
            spec.negacyclic,
        );
        let row_tables = cache.get_twiddles_refs_par(
            sched,
            spec.par_workers,
            &crt.primes_u64,
            k,
            cols,
            false,
        );
        let psi_fwd: Vec<PsiParams> = (0..k)
            .map(|pi| {
                build_psi_params(
                    crt.primes_u64[pi],
                    col_tables[pi].g_root,
                    spec.n,
                    spec.negacyclic,
                    false,
                )
            })
            .collect();
        let tw_tile0_fwd = cache.get_tile0_twiddles_par(
            kernels,
            sched,
            spec.par_workers,
            &crt.primes_u64,
            &col_tables,
            k,
            spec.n,
            rows,
            tile_cols,
            spec.negacyclic,
            false,
        );

        let sw2_w = sw2_workers(sched, &spec);
        let chunk_cap = sw2_chunk_cap(cols);
        let row_chunk = (rows / (sw2_w * 8)).clamp(1, chunk_cap);
        let prod_grid_ptr = prod_grid.as_mut_ptr() as usize;
        let a_stride = if need_a_work {
            work_row_bytes
        } else {
            row_bytes
        };
        let pbp_ops = if sw1_once_a {
            usize::from(!same && !sw1_once_b)
        } else {
            sw1_ops
        };
        let sw1_nt = u64::from(spec.use_nt && pbp_ops * work_prime_bytes > my_l3_cap);
        let once_bytes = k * total_grid_bytes + if sw1_once_b { k * work_prime_bytes } else { 0 };
        let once_nt = u64::from(spec.use_nt && once_bytes > my_l3_cap);

        if sw1_once_packed {
            let a_bases: Vec<usize> =
                prod_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let b_bases: Vec<usize> =
                work_grids.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
            let mut ops: Vec<(&[u64], &[usize], usize)> = vec![(a_limbs, &a_bases, row_bytes)];
            if sw1_once_b {
                ops.push((b_limbs, &b_bases, work_row_bytes));
            }
            ntt_sweep1_all_primes(kernels, cache, sched, &ops, &spec, tile_cols, &primes, once_nt);
        } else if sw1_once_unpacked {
            let a_bases: Vec<usize> =
                prod_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let b_bases: Vec<usize> =
                work_grids.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
            let mut ops: Vec<(&[u64], &[usize], usize)> = vec![(a_limbs, &a_bases, row_bytes)];
            if sw1_once_b {
                ops.push((b_limbs, &b_bases, work_row_bytes));
            }
            ntt_sweep1_all_primes_unpacked(
                kernels,
                cache,
                sched,
                &ops,
                &spec,
                tile_cols,
                &primes,
                once_nt,
            );
        }

        let out_nt = ROW_SWEEP_STREAMS;
        // One row of sweep 2 for prime `pi` where the rows are gathered from packed or work
        // grids: both operands' rows forward, their pointwise product, the row back, and into
        // the product grid.
        let sw2_row = |sc: &mut NttTlsScratch, pi: usize, r: usize, a_base: usize, b_base: usize| {
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
            let mcfg_in = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: work_row_bytes as u64,
                packed: u64::from(pack_sw1),
                use_nt: if sw1_once_packed { once_nt } else { 0 },
            };
            let mcfg_out = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: out_nt,
            };
            unsafe {
                let pb_ptr = if same {
                    sc.ra1.as_ptr()
                } else {
                    let b_src_u8 = (b_base as *const u8).add(r * work_row_bytes);
                    if pack_sw1 {
                        (kernels.active.gather)(
                            sc.rb1.as_mut_ptr(),
                            b_src_u8,
                            &mcfg_in,
                        );
                    } else {
                        std::ptr::copy_nonoverlapping(
                            b_src_u8 as *const f64,
                            sc.rb1.as_mut_ptr(),
                            cols,
                        );
                    }
                    call_ntt_fwd(
                        kernels,
                        sc.rb1.as_mut_ptr(),
                        r_tw_ptr as *const f64,
                        cols as u64,
                        &pr_fwd,
                    );
                    sc.rb1.as_ptr()
                };
                let a_src_u8 = (a_base as *const u8).add(r * a_stride);
                if pack_sw1 {
                    (kernels.active.gather)(sc.ra1.as_mut_ptr(), a_src_u8, &mcfg_in);
                } else {
                    std::ptr::copy_nonoverlapping(
                        a_src_u8 as *const f64,
                        sc.ra1.as_mut_ptr(),
                        cols,
                    );
                }
                call_ntt_fwd_pmul_inv(
                    kernels,
                    sc.ra1.as_mut_ptr(),
                    r_tw_ptr as *const f64,
                    r_inv_tw_ptr as *const f64,
                    pb_ptr,
                    cols as u64,
                    &pr_row_inv,
                );
                let dst_u8 = (*(prod_grid_ptr as *mut AlignedBytes).add(pi))
                    .as_mut_ptr()
                    .add(r * row_bytes);
                (kernels.active.scatter)(dst_u8, sc.ra1.as_ptr(), &mcfg_out);
            }
        };
        let sw2_scratch = || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            resize_f64_buf(&mut sc.ra1, cols);
            if !same {
                resize_f64_buf(&mut sc.rb1, cols);
            }
            lease
        };

        // Sweep 1 made for every prime: sweep 2 is one loop over all the primes' rows. A loop a
        // prime ends with the workers waiting for its last chunks and starts with their waking.
        let sw2_all_primes_packed = sw1_once_packed && (same || sw1_once_b);
        let sw2_all_primes_inplace = sw1_once_unpacked && !dram_pbp && (same || sw1_once_b);
        let sw2_all_primes = sw2_all_primes_packed || sw2_all_primes_inplace;
        if sw2_all_primes_packed {
            let t_sw2 = std::time::Instant::now();
            let bases: Vec<(usize, usize)> = (0..k)
                .map(|pi| {
                    (
                        prod_grid[pi].as_mut_ptr() as usize,
                        if same {
                            0usize
                        } else {
                            work_grids[pi].as_mut_ptr() as usize
                        },
                    )
                })
                .collect();
            let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
            sched.par_for_init_max(k * rows, all_chunk, sw2_w, sw2_scratch, |lease, task| {
                let (pi, r) = (task / rows, task % rows);
                sw2_row(lease.get_mut(), pi, r, bases[pi].0, bases[pi].1);
            });
            for g in prod_grid.iter_mut() {
                unsafe {
                    g.set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        } else if sw2_all_primes_inplace {
            let t_sw2 = std::time::Instant::now();
            let bases: Vec<(usize, usize)> = (0..k)
                .map(|pi| {
                    (
                        prod_grid[pi].as_mut_ptr() as usize,
                        if same {
                            0usize
                        } else {
                            work_grids[pi].as_mut_ptr() as usize
                        },
                    )
                })
                .collect();
            let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
            sched.par_for_leaf_max(k * rows, all_chunk, sw2_w, |task| {
                let (pi, r) = (task / rows, task % rows);
                let (a_base, b_base) = bases[pi];
                let r_tab = row_tables[pi];
                let pr_fwd = r_tab.pr_fwd;
                let pr_row_inv = r_tab.pr_inv;
                let r_tw_ptr = r_tab.tw.as_ptr() as usize;
                let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
                let a_row = unsafe { (a_base as *mut u8).add(r * a_stride) as *mut f64 };
                unsafe {
                    let pb_ptr = if same {
                        a_row as *const f64
                    } else {
                        let b_row = (b_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        call_ntt_fwd(
                            kernels,
                            b_row,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        b_row as *const f64
                    };
                    call_ntt_fwd_pmul_inv(
                        kernels,
                        a_row,
                        r_tw_ptr as *const f64,
                        r_inv_tw_ptr as *const f64,
                        pb_ptr,
                        cols as u64,
                        &pr_row_inv,
                    );
                }
            });
            for g in prod_grid.iter_mut() {
                unsafe {
                    g.set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }

        for pi in 0..(if sw2_all_primes { 0 } else { k }) {
            let p_u64 = crt.primes_u64[pi];
            let a_base = if need_a_work {
                work_grids[0].as_mut_ptr() as usize
            } else {
                prod_grid[pi].as_mut_ptr() as usize
            };
            let b_base = if same {
                0usize
            } else if sw1_once_b {
                work_grids[pi].as_mut_ptr() as usize
            } else if sw1_once_a {
                let bb = work_grids[0].as_mut_ptr() as usize;
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &[(b_limbs, bb, work_row_bytes)],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
                bb
            } else {
                let b_idx = usize::from(need_a_work);
                let bb = work_grids[b_idx].as_mut_ptr() as usize;
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &[
                        (a_limbs, a_base, a_stride),
                        (b_limbs, bb, work_row_bytes),
                    ],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
                bb
            };
            if same && !sw1_once_a {
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &[(a_limbs, a_base, a_stride)],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
            }

            let t_sw2 = std::time::Instant::now();
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;

            if pack_sw1 || dram_pbp {
                sched.par_for_init_max(rows, row_chunk, sw2_w, sw2_scratch, |lease, r| {
                    sw2_row(lease.get_mut(), pi, r, a_base, b_base);
                });
            } else {
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let a_row = unsafe { (a_base as *mut u8).add(r * a_stride) as *mut f64 };
                    unsafe {
                        let pb_ptr = if same {
                            a_row as *const f64
                        } else {
                            let b_row =
                                (b_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(
                                kernels,
                                b_row,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            b_row as *const f64
                        };
                        call_ntt_fwd_pmul_inv(
                            kernels,
                            a_row,
                            r_tw_ptr as *const f64,
                            r_inv_tw_ptr as *const f64,
                            pb_ptr,
                            cols as u64,
                            &pr_row_inv,
                        );
                        if spec.packed {
                            let mcfg_out = MoveCfg {
                                rows: 1,
                                cols: cols as u64,
                                outer_stride_bytes: row_bytes as u64,
                                packed: 1,
                                use_nt: ROW_SWEEP_STREAMS,
                            };
                            let dst_u8 = (*(prod_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(r * row_bytes);
                            (kernels.active.scatter)(dst_u8, a_row as *const f64, &mcfg_out);
                        }
                    }
                });
            }
            unsafe {
                prod_grid[pi].set_len(total_grid_bytes);
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }
        if need_work > 0 {
            for g in &mut work_grids {
                unsafe {
                    g.set_len(work_prime_bytes);
                }
            }
            cache.put_byte_bufs(work_grids);
        }
        if kernels.count_work {
            let n_fwd = if same { 1u64 } else { 2u64 };
            let bf_cols = n_fwd * ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
            kernels
                .stats
                .ntt_butterflies
                .fetch_add(bf_cols, Ordering::Relaxed);
            let limb_b = if same {
                a_limbs.len() * 8
            } else {
                (a_limbs.len() + b_limbs.len()) * 8
            };
            kernels
                .stats
                .bytes_moved
                .fetch_add(limb_b as u64, Ordering::Relaxed);
        }
        let fwd_row_ops = if same { 1u64 } else { 2u64 };
        return ntt_sweep3_and_crt(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            prod_grid,
            spec.negacyclic,
            out_limbs,
            Some(fwd_row_ops),
        );
    }

    let mut fa = ntt_forward(kernels, cache, sched, store, a_limbs, spec)?;
    if same {
        let res = ntt_combine_and_inverse(
            kernels,
            cache,
            sched,
            store,
            telemetry,
            &mut fa,
            None,
            None,
            out_limbs,
        );
        fa.release_store(store);
        res
    } else {
        let mut fb = match ntt_forward(kernels, cache, sched, store, b_limbs, spec) {
            Ok(v) => v,
            Err(e) => {
                fa.release_store(store);
                return Err(e);
            }
        };
        let res = ntt_combine_and_inverse(
            kernels,
            cache,
            sched,
            store,
            telemetry,
            &mut fa,
            Some(NttOperand::Owned(&mut fb)),
            None,
            out_limbs,
        );
        fa.release_store(store);
        fb.release_store(store);
        res
    }
}

pub struct FusedMergeRaw {
    pub t: (LimbVec, u64),
    pub q: Option<(LimbVec, u64)>,
    pub p: Option<(LimbVec, u64)>,
}

/// Check whether `ntt_fused_merge_triple` fits within L2/L3 cache and `ram_transform_budget`.
#[inline]
pub fn ntt_fused_merge_eligible(
    spec: &NttSpec,
    ram_transform_budget: u64,
    l3_prime_cap_bytes: usize,
    num_workers: usize,
    n_out_grids: usize,
) -> bool {
    if spec.out_of_core || spec.negacyclic || spec.wrap_cyclic || spec.extra_sweeps != 0 {
        return false;
    }
    if spec.rows == 1 {
        return !spec.packed;
    }
    if spec.cols < 64 {
        return false;
    }
    let total_grid_bytes = spec.rows * spec.outer_row_bytes();
    let unpacked_row_bytes = if spec.cols >= 512 {
        spec.cols * 8 + 512
    } else {
        spec.cols * 8
    };
    let unpacked_prime_bytes = spec.rows * unpacked_row_bytes;
    let _ = (l3_prime_cap_bytes, num_workers);
    let base_tc = spec.tile_cols.min(spec.cols).max(1);
    let eff_tc = if spec.rows >= 128 && spec.cols.is_multiple_of(32) {
        base_tc.min(32)
    } else {
        base_tc
    };
    let pack_sw1 = can_pack_sweep1(spec, eff_tc);
    let out_bytes = (n_out_grids.max(1) as u64)
        .saturating_mul(spec.k_primes as u64)
        .saturating_mul(total_grid_bytes as u64);
    let min_work_bytes = if pack_sw1 {
        3u64.saturating_mul(total_grid_bytes as u64)
    } else {
        2u64.saturating_mul(unpacked_prime_bytes as u64)
    };
    out_bytes.saturating_add(min_work_bytes) <= ram_transform_budget.max(64 * 1024)
}

/// Fused prime-by-prime `(T, Q, P)` binary-splitting merge keeping all forward intermediate
/// residues inside L2 (`rows == 1`) or L3 (`rows > 1`).
#[allow(clippy::too_many_arguments)]
pub fn ntt_fused_merge_triple(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    spec: NttSpec,
    ram_transform_budget: u64,
    q2_limbs: &[u64],
    t1_limbs: &[u64],
    p1_limbs: &[u64],
    t2_limbs: &[u64],
    sub_second: bool,
    t_out_limbs: usize,
    q1_and_out: Option<(&[u64], usize)>,
    p2_and_out: Option<(&[u64], usize)>,
) -> Option<FusedMergeRaw> {
    let n_out_grids = 1usize + usize::from(q1_and_out.is_some()) + usize::from(p2_and_out.is_some());
    if !ntt_fused_merge_eligible(
        &spec,
        ram_transform_budget,
        cache.l3_prime_cap_bytes(),
        sched.num_workers(),
        n_out_grids,
    ) {
        return None;
    }
    let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
    let k = spec.k_primes;
    let mut _pt = PathTimer::start(
        "triple",
        &spec,
        &[q2_limbs.len(), t1_limbs.len(), p1_limbs.len(), t2_limbs.len()],
    );
    _pt.tag(&format!("q={} p={}", q1_and_out.is_some(), p2_and_out.is_some()));

    if spec.rows == 1 {
        let n = spec.n;
        let p_indep =
            k >= 2 && spec.coeff_bits <= 52 && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
        let num_ch = (spec.coeff_bits as usize).div_ceil(48);
        let p_chunk_cand = !p_indep
            && k >= 2
            && (53..=384).contains(&spec.coeff_bits)
            && crt.primes_u64[k - 1] > (1u64 << 48);
        let mut cand_ch_words = ch_words_for_1d(
            nz_elems_1d(q2_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        ) + ch_words_for_1d(
            nz_elems_1d(t1_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        ) + ch_words_for_1d(
            nz_elems_1d(p1_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        ) + ch_words_for_1d(
            nz_elems_1d(t2_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        );
        if let Some((q1_l, _)) = q1_and_out {
            cand_ch_words += ch_words_for_1d(
                nz_elems_1d(q1_l.len(), spec.coeff_bits, n),
                p_indep,
                true,
                num_ch,
            );
        }
        if let Some((p2_l, _)) = p2_and_out {
            cand_ch_words += ch_words_for_1d(
                nz_elems_1d(p2_l.len(), spec.coeff_bits, n),
                p_indep,
                true,
                num_ch,
            );
        }
        let max_1d_ch_bytes =
            (cache.l3_prime_cap_bytes() / sched.num_workers().max(1)).max(512 * 1024);
        let p_chunk = p_chunk_cand && cand_ch_words * 8 <= max_1d_ch_bytes;
        let t0 = cache.get_twiddles_ref(crt.primes_u64[0], n, false);

        let mut f_lease = Fused1dLease::acquire();
        let f1d = f_lease.get_mut();
        let ch_need = if p_chunk || p_indep { cand_ch_words } else { 0 };
        let res_stride = res_stride_1d(n);
        resize_f64_buf(&mut f1d.m_ch, ch_need);
        resize_f64_buf(&mut f1d.m_t_res, k * res_stride);
        if q1_and_out.is_some() {
            resize_f64_buf(&mut f1d.m_q_res, k * res_stride);
        }
        if p2_and_out.is_some() {
            resize_f64_buf(&mut f1d.m_p_res, k * res_stride);
        }

        let t_f1 = ntt_trace_enabled().then(std::time::Instant::now);
        let mut cur_off = 0usize;
        let prep_q2 = prep_1d_operand(
            kernels,
            q2_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_t1 = prep_1d_operand(
            kernels,
            t1_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_p1 = prep_1d_operand(
            kernels,
            p1_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_t2 = prep_1d_operand(
            kernels,
            t2_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_q1 = q1_and_out.map(|(q1_l, _)| {
            prep_1d_operand(
                kernels,
                q1_l,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            )
        });
        let prep_p2 = p2_and_out.map(|(p2_l, _)| {
            prep_1d_operand(
                kernels,
                p2_l,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            )
        });

        let m_ch_ptr = f1d.m_ch.as_ptr() as usize;
        let t_res_ptr = f1d.m_t_res.as_mut_ptr() as usize;
        let q_res_ptr = f1d.m_q_res.as_mut_ptr() as usize;
        let p_res_ptr = f1d.m_p_res.as_mut_ptr() as usize;

        let run_fused_prime = |sc: &mut NttTlsScratch, pi: usize| {
            let p_u64 = crt.primes_u64[pi];
            let tables = cache.get_twiddles_ref(p_u64, n, false);
            let pr_inv = &tables.pr_inv;
            let vcfg_mul = NttVecCfg {
                n: n as u64,
                p: pr_inv.p,
                inv_p: pr_inv.inv_p,
                sub_second: 0,
            };
            let vcfg_fma = NttVecCfg {
                n: n as u64,
                p: pr_inv.p,
                inv_p: pr_inv.inv_p,
                sub_second: u64::from(sub_second),
            };
            unsafe {
                let ra1 = sc.ra1.as_mut_ptr();
                let rb1 = sc.rb1.as_mut_ptr();
                // 1. ra1 = fwd(q2)
                eval_1d_fwd(
                    kernels,
                    q2_limbs,
                    prep_q2,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    ra1,
                );
                // 2. Q = IFFT(fwd(q1) * ra1) directly in q_dst
                if let (Some((q1_l, _)), Some(pq1)) = (q1_and_out, prep_q1) {
                    let q_dst = (q_res_ptr as *mut f64).add(pi * res_stride);
                    eval_1d_fwd_pmul_inv(
                        kernels,
                        q1_l,
                        pq1,
                        m_ch_ptr as *const f64,
                        tables,
                        n,
                        spec.coeff_bits,
                        q_dst,
                        ra1 as *const f64,
                    );
                }
                // 3. ra1 = ra1 * fwd(t1)
                eval_1d_fwd(
                    kernels,
                    t1_limbs,
                    prep_t1,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    rb1,
                );
                (kernels.active.ntt_pmul)(
                    ra1,
                    ra1 as *const f64,
                    rb1 as *const f64,
                    &vcfg_mul,
                );
                // 4. rb1 = fwd(p1), t_dst = fwd(t2), T = IFFT(ra1 +- rb1 * t_dst)
                eval_1d_fwd(
                    kernels,
                    p1_limbs,
                    prep_p1,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    rb1,
                );
                let t_dst = (t_res_ptr as *mut f64).add(pi * res_stride);
                eval_1d_fwd(
                    kernels,
                    t2_limbs,
                    prep_t2,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    t_dst,
                );
                let ptrs = [
                    ra1 as *const f64,
                    std::ptr::null(),
                    rb1 as *const f64,
                    t_dst as *const f64,
                ];
                (kernels.active.ntt_pmuladd)(t_dst, &ptrs, &vcfg_fma);
                call_ntt_inv(kernels, t_dst, tables.inv_tw.as_ptr(), n as u64, pr_inv);
                // 5. P = IFFT(rb1 * fwd(p2)) directly in p_dst
                if let (Some((p2_l, _)), Some(pp2)) = (p2_and_out, prep_p2) {
                    let p_dst = (p_res_ptr as *mut f64).add(pi * res_stride);
                    eval_1d_fwd_pmul_inv(
                        kernels,
                        p2_l,
                        pp2,
                        m_ch_ptr as *const f64,
                        tables,
                        n,
                        spec.coeff_bits,
                        p_dst,
                        rb1 as *const f64,
                    );
                }
            }
        };
        if spec.par_workers <= 1 || k <= 1 {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            resize_f64_buf(&mut sc.ra1, n);
            resize_f64_buf(&mut sc.rb1, n);
            for pi in 0..k {
                run_fused_prime(sc, pi);
            }
        } else {
            sched.par_for_init_max(
                k,
                1,
                spec.par_workers,
                || {
                    let mut lease = ScratchLease::acquire();
                    let sc = lease.get_mut();
                    resize_f64_buf(&mut sc.ra1, n);
                    resize_f64_buf(&mut sc.rb1, n);
                    lease
                },
                |lease, pi| run_fused_prime(lease.get_mut(), pi),
            );
        }
        if let Some(t) = t_f1 {
            NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        let n_fwd = 4u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some());
        let n_inv = 1u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some());
        if kernels.count_work && n > 1 {
            let bf = (n_fwd + n_inv) * ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64)
                + (n_inv + 1) * (n as u64) * (k as u64);
            kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
            kernels.stats.bytes_moved.fetch_add(
                ((n_inv as usize) * k * n * 8) as u64,
                Ordering::Relaxed,
            );
        }

        let t = crt_1d_from_flat(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &f1d.m_t_res[..(k * res_stride)],
            res_stride,
            sub_second,
            t_out_limbs,
        );
        let q = q1_and_out.map(|(_, q_out_l)| {
            crt_1d_from_flat(
                kernels,
                cache,
                sched,
                &spec,
                crt,
                &f1d.m_q_res[..(k * res_stride)],
                res_stride,
                false,
                q_out_l,
            )
        });
        let p = p2_and_out.map(|(_, p_out_l)| {
            crt_1d_from_flat(
                kernels,
                cache,
                sched,
                &spec,
                crt,
                &f1d.m_p_res[..(k * res_stride)],
                res_stride,
                false,
                p_out_l,
            )
        });
        return Some(FusedMergeRaw { t, q, p });
    }

    // 2D Prime-by-Prime Fused `(T, Q, P)` Merge!
    let rows = spec.rows;
    let cols = spec.cols;
    let row_bytes = spec.outer_row_bytes();
    let total_grid_bytes = rows * row_bytes;
    let unpacked_row_bytes = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
    let unpacked_prime_bytes = rows * unpacked_row_bytes;
    let est_conc = sched.num_workers().div_ceil(spec.par_workers.max(1)).max(1);
    let my_l3_cap = cache.l3_prime_cap_bytes() / est_conc;
    let out_bytes = (n_out_grids as u64)
        .saturating_mul(k as u64)
        .saturating_mul(total_grid_bytes as u64);
    let base_tc = spec.tile_cols.min(cols).max(1);
    let tile_cols = if rows >= 128 && cols.is_multiple_of(32) {
        base_tc.min(32)
    } else {
        base_tc
    };
    let l3_spills_work = (3 + n_out_grids) * unpacked_prime_bytes > my_l3_cap;
    let fit2_ram = out_bytes.saturating_add(2 * (unpacked_prime_bytes as u64))
        <= ram_transform_budget.max(64 * 1024);
    let use_l3_staged =
        !spec.packed && l3_spills_work && 2 * unpacked_prime_bytes <= my_l3_cap && fit2_ram;
    let pack_sw1 = !use_l3_staged && can_pack_sweep1(&spec, tile_cols);
    let can_out_unpacked = !use_l3_staged
        && !pack_sw1
        && !spec.packed
        && row_bytes == unpacked_row_bytes
        && (cols / tile_cols) >= spec.par_workers.max(1);
    let rf_work_bytes = if pack_sw1 {
        3 * total_grid_bytes
    } else if can_out_unpacked {
        3 * unpacked_prime_bytes
    } else {
        (3 + n_out_grids) * unpacked_prime_bytes
    };
    let dram_row_fused = !use_l3_staged
        && out_bytes.saturating_add(rf_work_bytes as u64) <= ram_transform_budget.max(64 * 1024);
    let use_pack_sw1 = dram_row_fused && pack_sw1;
    let out_in_out_unpacked = dram_row_fused && !use_pack_sw1 && can_out_unpacked;
    _pt.tag(&format!("row_fused={dram_row_fused} packed_work={use_pack_sw1}"));
    let sw1_once_packed = use_pack_sw1 && sweep1_groups_256(&spec, tile_cols);
    let sw1_once_unpacked = out_in_out_unpacked;
    let cand_work_prime_bytes = if use_pack_sw1 {
        total_grid_bytes
    } else {
        unpacked_prime_bytes
    };
    let with_all_work_primes =
        out_bytes.saturating_add(((3 * k) as u64).saturating_mul(cand_work_prime_bytes as u64));
    let conc_all_grid_bytes = (est_conc as u64).saturating_mul(with_all_work_primes);
    let sw1_once_unpacked_all = sw1_once_unpacked
        && (3 + n_out_grids) * k * unpacked_prime_bytes <= my_l3_cap
        && with_all_work_primes <= ram_transform_budget.max(64 * 1024)
        && conc_all_grid_bytes <= cache.work_grid_room()
        && room_for_work_grids(
            conc_all_grid_bytes.min(usize::MAX as u64) as usize,
            3 * k.saturating_sub(1) * unpacked_prime_bytes,
        );
    let sw1_once_packed_all = sw1_once_packed
        && (est_conc <= 2 || 3 * total_grid_bytes > my_l3_cap)
        && with_all_work_primes <= ram_transform_budget.max(64 * 1024)
        && conc_all_grid_bytes <= cache.work_grid_room()
        && room_for_work_grids(
            conc_all_grid_bytes.min(usize::MAX as u64) as usize,
            3 * k.saturating_sub(1) * total_grid_bytes,
        );
    let sw1_once_all = sw1_once_packed_all || sw1_once_unpacked_all;
    let fit3_ram = out_bytes.saturating_add(3 * (unpacked_prime_bytes as u64))
        <= ram_transform_budget.max(64 * 1024);
    let have_g2 = !dram_row_fused && 3 * unpacked_prime_bytes <= my_l3_cap && fit3_ram;
    let num_work = if sw1_once_all {
        (3 * k).max(3)
    } else if use_pack_sw1 || out_in_out_unpacked {
        3
    } else if dram_row_fused {
        3 + n_out_grids
    } else if have_g2 {
        3
    } else {
        2
    };
    let work_prime_bytes = if use_pack_sw1 {
        total_grid_bytes
    } else {
        unpacked_prime_bytes
    };
    let work_row_bytes = if use_pack_sw1 {
        row_bytes
    } else {
        unpacked_row_bytes
    };

    let t_al = ntt_trace_enabled().then(std::time::Instant::now);
    let (mut t_grid, mut q_grid, mut p_grid, mut work_grids) =
        if work_prime_bytes == total_grid_bytes {
            let mut all = cache.take_byte_bufs_transient_committed(
                n_out_grids * k + num_work,
                total_grid_bytes,
                sched,
                spec.par_workers,
            );
            let wg = all.split_off(n_out_grids * k);
            let pg = if p2_and_out.is_some() {
                let rem = all.len() - k;
                all.split_off(rem)
            } else {
                Vec::new()
            };
            let qg = if q1_and_out.is_some() {
                all.split_off(k)
            } else {
                Vec::new()
            };
            (all, qg, pg, wg)
        } else {
            let mut outs = cache.take_byte_bufs_transient_committed(
                n_out_grids * k,
                total_grid_bytes,
                sched,
                spec.par_workers,
            );
            let pg = if p2_and_out.is_some() {
                let rem = outs.len() - k;
                outs.split_off(rem)
            } else {
                Vec::new()
            };
            let qg = if q1_and_out.is_some() {
                outs.split_off(k)
            } else {
                Vec::new()
            };
            let wg = cache.take_byte_bufs_transient_committed(
                num_work,
                work_prime_bytes,
                sched,
                spec.par_workers,
            );
            (outs, qg, pg, wg)
        };
    if let Some(t) = t_al {
        NTT_TRACE_ALLOC_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }

    let col_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        rows,
        false,
    );
    let row_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        cols,
        false,
    );
    let psi_fwd: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                false,
                false,
            )
        })
        .collect();
    let tw_tile0_fwd = cache.get_tile0_twiddles_par(
        kernels,
        sched,
        spec.par_workers,
        &crt.primes_u64,
        &col_tables,
        k,
        spec.n,
        rows,
        tile_cols,
        false,
        false,
    );

    let sw2_w = sw2_workers(sched, &spec);
    let chunk_cap = sw2_chunk_cap(cols);
    let row_chunk = (rows / (sw2_w * 8)).clamp(1, chunk_cap);
    let g0_base = work_grids[0].as_mut_ptr() as usize;
    let g1_base = work_grids[1].as_mut_ptr() as usize;
    let g2_base = if have_g2 || dram_row_fused {
        work_grids[2].as_mut_ptr() as usize
    } else {
        0usize
    };
    let t_grid_ptr = t_grid.as_mut_ptr() as usize;
    let q_grid_ptr = q_grid.as_mut_ptr() as usize;
    let p_grid_ptr = p_grid.as_mut_ptr() as usize;

    if dram_row_fused {
        let w_q2 = work_grids[0].as_mut_ptr() as usize;
        let w_t1 = work_grids[1].as_mut_ptr() as usize;
        let w_p1 = work_grids[2].as_mut_ptr() as usize;
        let sw1_ops = 4usize + usize::from(q1_and_out.is_some()) + usize::from(p2_and_out.is_some());
        let sw1_once = sw1_once_packed || sw1_once_unpacked;
        let pbp_ops = if sw1_once { 3usize } else { sw1_ops };
        let l3_fits_sw1 = pbp_ops * work_prime_bytes <= my_l3_cap;
        let sw1_nt = u64::from(spec.use_nt && !l3_fits_sw1);
        let once_ops_cnt = if sw1_once_all {
            3 + n_out_grids
        } else {
            n_out_grids
        };
        let once_nt = u64::from(spec.use_nt && once_ops_cnt * k * work_prime_bytes > my_l3_cap);
        let out_nt = ROW_SWEEP_STREAMS;
        let l3_inplace_work = !pack_sw1 && (sw1_ops * work_prime_bytes <= my_l3_cap);
        let have_q1 = q1_and_out.is_some();
        let have_p2 = p2_and_out.is_some();
        let mut ops_vec: Vec<(&[u64], usize, usize)> = Vec::with_capacity(6);

        if sw1_once {
            let t_bases: Vec<usize> = t_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let q_bases: Vec<usize> = q_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let p_bases: Vec<usize> = p_grid.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let (q2_bases, t1_bases, p1_bases): (Vec<usize>, Vec<usize>, Vec<usize>) =
                if sw1_once_all {
                    (
                        (0..k)
                            .map(|pi| work_grids[3 * pi].as_mut_ptr() as usize)
                            .collect(),
                        (0..k)
                            .map(|pi| work_grids[3 * pi + 1].as_mut_ptr() as usize)
                            .collect(),
                        (0..k)
                            .map(|pi| work_grids[3 * pi + 2].as_mut_ptr() as usize)
                            .collect(),
                    )
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
            let mut once_ops: Vec<(&[u64], &[usize], usize)> = Vec::with_capacity(6);
            if sw1_once_all {
                once_ops.push((q2_limbs, &q2_bases, work_row_bytes));
                once_ops.push((t1_limbs, &t1_bases, work_row_bytes));
                once_ops.push((p1_limbs, &p1_bases, work_row_bytes));
            }
            once_ops.push((t2_limbs, &t_bases, work_row_bytes));
            if let Some((q1_l, _)) = q1_and_out {
                once_ops.push((q1_l, &q_bases, work_row_bytes));
            }
            if let Some((p2_l, _)) = p2_and_out {
                once_ops.push((p2_l, &p_bases, work_row_bytes));
            }
            if sw1_once_packed {
                ntt_sweep1_all_primes(
                    kernels,
                    cache,
                    sched,
                    &once_ops,
                    &spec,
                    tile_cols,
                    &primes,
                    once_nt,
                );
            } else {
                debug_assert_eq!(work_row_bytes, row_bytes);
                ntt_sweep1_all_primes_unpacked(
                    kernels,
                    cache,
                    sched,
                    &once_ops,
                    &spec,
                    tile_cols,
                    &primes,
                    once_nt,
                );
            }
        }

        if sw1_once_all {
            let t_sw2 = std::time::Instant::now();
            let bases: Vec<[usize; 6]> = (0..k)
                .map(|pi| {
                    [
                        work_grids[3 * pi].as_mut_ptr() as usize,
                        work_grids[3 * pi + 1].as_mut_ptr() as usize,
                        work_grids[3 * pi + 2].as_mut_ptr() as usize,
                        t_grid[pi].as_mut_ptr() as usize,
                        if have_q1 {
                            q_grid[pi].as_mut_ptr() as usize
                        } else {
                            0
                        },
                        if have_p2 {
                            p_grid[pi].as_mut_ptr() as usize
                        } else {
                            0
                        },
                    ]
                })
                .collect();
            let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
            if !pack_sw1 && out_in_out_unpacked {
                sched.par_for_leaf_max(k * rows, all_chunk, sw2_w, |task| {
                    let (pi, r) = (task / rows, task % rows);
                    let [b_q2, b_t1, b_p1, b_t2, b_q1, b_p2] = bases[pi];
                    let r_tab = row_tables[pi];
                    let pr_fwd = r_tab.pr_fwd;
                    let pr_row_inv = r_tab.pr_inv;
                    let r_tw_ptr = r_tab.tw.as_ptr() as usize;
                    let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
                    let r_off = r * work_row_bytes;
                    let vcfg_mul = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: 0,
                    };
                    let vcfg_fma = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: u64::from(sub_second),
                    };
                    unsafe {
                        let q2_r = (b_q2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            q2_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        if have_q1 {
                            let q1_r = (b_q1 as *mut u8).add(r_off) as *mut f64;
                            call_ntt_fwd_pmul_inv(
                                kernels,
                                q1_r,
                                r_tw_ptr as *const f64,
                                r_inv_tw_ptr as *const f64,
                                q2_r as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                        }

                        let t1_r = (b_t1 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            t1_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        (kernels.active.ntt_pmul)(
                            q2_r,
                            q2_r as *const f64,
                            t1_r as *const f64,
                            &vcfg_mul,
                        );

                        let p1_r = (b_p1 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            p1_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );

                        let t2_r = (b_t2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            t2_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        let ptrs = [
                            q2_r as *const f64,
                            std::ptr::null(),
                            p1_r as *const f64,
                            t2_r as *const f64,
                        ];
                        (kernels.active.ntt_pmuladd)(t2_r, &ptrs, &vcfg_fma);
                        call_ntt_inv(kernels, 
                            t2_r,
                            r_inv_tw_ptr as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );

                        if have_p2 {
                            let p2_r = (b_p2 as *mut u8).add(r_off) as *mut f64;
                            call_ntt_fwd_pmul_inv(
                                kernels,
                                p2_r,
                                r_tw_ptr as *const f64,
                                r_inv_tw_ptr as *const f64,
                                p1_r as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                        }
                    }
                });
            } else {
                sched.par_for_init_max(
                    k * rows,
                    all_chunk,
                    sw2_w,
                    || {
                        let mut lease = ScratchLease::acquire();
                        let sc = lease.get_mut();
                        resize_f64_buf(&mut sc.ra1, cols);
                        resize_f64_buf(&mut sc.rb1, cols);
                        resize_f64_buf(&mut sc.ra2, cols);
                        lease
                    },
                    |lease, task| {
                        let sc = lease.get_mut();
                        let (pi, r) = (task / rows, task % rows);
                        let [b_q2, b_t1, b_p1, b_t2, b_q1, b_p2] = bases[pi];
                        let r_tab = row_tables[pi];
                        let pr_fwd = r_tab.pr_fwd;
                        let pr_row_inv = r_tab.pr_inv;
                        let r_tw_ptr = r_tab.tw.as_ptr() as usize;
                        let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
                        let r_off = r * work_row_bytes;
                        let out_off = r * row_bytes;
                        let vcfg_mul = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        let vcfg_fma = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: u64::from(sub_second),
                        };
                        let mcfg_in = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: work_row_bytes as u64,
                            packed: u64::from(pack_sw1),
                            use_nt: if sw1_once_packed && sw2_w <= 8 { once_nt } else { 0 },
                        };
                        let mcfg_out = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: u64::from(spec.packed),
                            use_nt: out_nt,
                        };
                        unsafe {
                            let q2_src = (b_q2 as *const u8).add(r_off);
                            (kernels.active.gather)(sc.ra1.as_mut_ptr(), q2_src, &mcfg_in);
                            call_ntt_fwd(kernels, 
                                sc.ra1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            if have_q1 {
                                let q1_src = (b_q1 as *const u8).add(r_off);
                                (kernels.active.gather)(sc.rb1.as_mut_ptr(), q1_src, &mcfg_in);
                                call_ntt_fwd_pmul_inv(
                                    kernels,
                                    sc.rb1.as_mut_ptr(),
                                    r_tw_ptr as *const f64,
                                    r_inv_tw_ptr as *const f64,
                                    sc.ra1.as_ptr(),
                                    cols as u64,
                                    &pr_row_inv,
                                );
                                let dst_q = (*(q_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_q, sc.rb1.as_ptr(), &mcfg_out);
                            }

                            let t1_src = (b_t1 as *const u8).add(r_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), t1_src, &mcfg_in);
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(
                                sc.ra1.as_mut_ptr(),
                                sc.ra1.as_ptr(),
                                sc.rb1.as_ptr(),
                                &vcfg_mul,
                            );

                            let p1_src = (b_p1 as *const u8).add(r_off);
                            (kernels.active.gather)(sc.rb1.as_mut_ptr(), p1_src, &mcfg_in);
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            let t2_src = (b_t2 as *const u8).add(r_off);
                            (kernels.active.gather)(sc.ra2.as_mut_ptr(), t2_src, &mcfg_in);
                            call_ntt_fwd(kernels, 
                                sc.ra2.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            let ptrs = [
                                sc.ra1.as_ptr(),
                                std::ptr::null(),
                                sc.rb1.as_ptr(),
                                sc.ra2.as_ptr(),
                            ];
                            (kernels.active.ntt_pmuladd)(sc.ra2.as_mut_ptr(), &ptrs, &vcfg_fma);
                            call_ntt_inv(kernels, 
                                sc.ra2.as_mut_ptr(),
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            let dst_t = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(out_off);
                            (kernels.active.scatter)(dst_t, sc.ra2.as_ptr(), &mcfg_out);

                            if have_p2 {
                                let p2_src = (b_p2 as *const u8).add(r_off);
                                (kernels.active.gather)(sc.ra2.as_mut_ptr(), p2_src, &mcfg_in);
                                call_ntt_fwd_pmul_inv(
                                    kernels,
                                    sc.ra2.as_mut_ptr(),
                                    r_tw_ptr as *const f64,
                                    r_inv_tw_ptr as *const f64,
                                    sc.rb1.as_ptr(),
                                    cols as u64,
                                    &pr_row_inv,
                                );
                                let dst_p = (*(p_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_p, sc.ra2.as_ptr(), &mcfg_out);
                            }
                        }
                    },
                );
            }
            for g in t_grid
                .iter_mut()
                .chain(q_grid.iter_mut())
                .chain(p_grid.iter_mut())
            {
                unsafe {
                    g.set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }

        for pi in 0..(if sw1_once_all { 0 } else { k }) {
            let w_t2 = if pack_sw1 || out_in_out_unpacked {
                t_grid[pi].as_mut_ptr() as usize
            } else {
                work_grids[3].as_mut_ptr() as usize
            };
            let w_q1 = if q1_and_out.is_some() {
                if pack_sw1 || out_in_out_unpacked {
                    q_grid[pi].as_mut_ptr() as usize
                } else {
                    work_grids[4].as_mut_ptr() as usize
                }
            } else {
                0usize
            };
            let w_p2 = if p2_and_out.is_some() {
                if pack_sw1 || out_in_out_unpacked {
                    p_grid[pi].as_mut_ptr() as usize
                } else {
                    let p2_idx = 4 + usize::from(q1_and_out.is_some());
                    work_grids[p2_idx].as_mut_ptr() as usize
                }
            } else {
                0usize
            };

            ops_vec.clear();
            ops_vec.push((q2_limbs, w_q2, work_row_bytes));
            ops_vec.push((t1_limbs, w_t1, work_row_bytes));
            ops_vec.push((p1_limbs, w_p1, work_row_bytes));
            ops_vec.push((t2_limbs, w_t2, work_row_bytes));
            if let Some((q1_l, _)) = q1_and_out {
                ops_vec.push((q1_l, w_q1, work_row_bytes));
            }
            if let Some((p2_l, _)) = p2_and_out {
                ops_vec.push((p2_l, w_p2, work_row_bytes));
            }

            let p_u64 = crt.primes_u64[pi];
            if sw1_once {
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &ops_vec[..3],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
            } else if !l3_inplace_work && ops_vec.len() > 3 {
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &ops_vec[..3],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &ops_vec[3..],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
            } else {
                ntt_sweep1_multi_prime_ext(
                    kernels,
                    sched,
                    &ops_vec,
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    sw1_nt,
                    pack_sw1,
                );
            }

            let t_sw2 = std::time::Instant::now();
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;

            if l3_inplace_work {
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let r_off = r * work_row_bytes;
                    let out_off = r * row_bytes;
                    let vcfg_mul = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: 0,
                    };
                    let vcfg_fma = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: u64::from(sub_second),
                    };
                    let mcfg_out = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: row_bytes as u64,
                        packed: u64::from(spec.packed),
                        use_nt: out_nt,
                    };
                    unsafe {
                        let q2_r = (w_q2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            q2_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        if have_q1 {
                            let q1_r = (w_q1 as *mut u8).add(r_off) as *mut f64;
                            call_ntt_fwd_pmul_inv(
                                kernels,
                                q1_r,
                                r_tw_ptr as *const f64,
                                r_inv_tw_ptr as *const f64,
                                q2_r as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            if !out_in_out_unpacked {
                                let dst_q = (*(q_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_q, q1_r as *const f64, &mcfg_out);
                            }
                        }

                        let t1_r = (w_t1 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            t1_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        (kernels.active.ntt_pmul)(
                            q2_r,
                            q2_r as *const f64,
                            t1_r as *const f64,
                            &vcfg_mul,
                        );

                        let p1_r = (w_p1 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            p1_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );

                        let t2_r = (w_t2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(kernels, 
                            t2_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );

                        let ptrs = [
                            q2_r as *const f64,
                            std::ptr::null(),
                            p1_r as *const f64,
                            t2_r as *const f64,
                        ];
                        (kernels.active.ntt_pmuladd)(t2_r, &ptrs, &vcfg_fma);
                        call_ntt_inv(kernels, 
                            t2_r,
                            r_inv_tw_ptr as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        if !out_in_out_unpacked {
                            let dst_t = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(out_off);
                            (kernels.active.scatter)(dst_t, t2_r as *const f64, &mcfg_out);
                        }

                        if have_p2 {
                            let p2_r = (w_p2 as *mut u8).add(r_off) as *mut f64;
                            call_ntt_fwd_pmul_inv(
                                kernels,
                                p2_r,
                                r_tw_ptr as *const f64,
                                r_inv_tw_ptr as *const f64,
                                p1_r as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            if !out_in_out_unpacked {
                                let dst_p = (*(p_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_p, p2_r as *const f64, &mcfg_out);
                            }
                        }
                    }
                });
            } else {
                sched.par_for_init_max(
                    rows,
                    row_chunk,
                    sw2_w,
                    || {
                        let mut lease = ScratchLease::acquire();
                        let sc = lease.get_mut();
                        resize_f64_buf(&mut sc.ra1, cols);
                        resize_f64_buf(&mut sc.rb1, cols);
                        resize_f64_buf(&mut sc.ra2, cols);
                        lease
                    },
                    |lease, r| {
                        let sc = lease.get_mut();
                        let r_off = r * work_row_bytes;
                        let out_off = r * row_bytes;
                        let vcfg_mul = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        let vcfg_fma = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: u64::from(sub_second),
                        };
                        let mcfg_in = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: work_row_bytes as u64,
                            packed: u64::from(pack_sw1),
                            use_nt: 0,
                        };
                        let mcfg_out = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: u64::from(spec.packed),
                            use_nt: out_nt,
                        };
                        unsafe {
                            let q2_src = (w_q2 as *const u8).add(r_off);
                            if pack_sw1 {
                                (kernels.active.gather)(sc.ra1.as_mut_ptr(), q2_src, &mcfg_in);
                            } else {
                                std::ptr::copy_nonoverlapping(
                                    q2_src as *const f64,
                                    sc.ra1.as_mut_ptr(),
                                    cols,
                                );
                            }
                            call_ntt_fwd(kernels, 
                                sc.ra1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            if have_q1 {
                                let q1_src = (w_q1 as *const u8).add(r_off);
                                if pack_sw1 {
                                    (kernels.active.gather)(sc.rb1.as_mut_ptr(), q1_src, &mcfg_in);
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        q1_src as *const f64,
                                        sc.rb1.as_mut_ptr(),
                                        cols,
                                    );
                                }
                                call_ntt_fwd_pmul_inv(
                                    kernels,
                                    sc.rb1.as_mut_ptr(),
                                    r_tw_ptr as *const f64,
                                    r_inv_tw_ptr as *const f64,
                                    sc.ra1.as_ptr(),
                                    cols as u64,
                                    &pr_row_inv,
                                );
                                let dst_q = (*(q_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_q, sc.rb1.as_ptr(), &mcfg_out);
                            }

                            let t1_src = (w_t1 as *const u8).add(r_off);
                            if pack_sw1 {
                                (kernels.active.gather)(sc.rb1.as_mut_ptr(), t1_src, &mcfg_in);
                            } else {
                                std::ptr::copy_nonoverlapping(
                                    t1_src as *const f64,
                                    sc.rb1.as_mut_ptr(),
                                    cols,
                                );
                            }
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(
                                sc.ra1.as_mut_ptr(),
                                sc.ra1.as_ptr(),
                                sc.rb1.as_ptr(),
                                &vcfg_mul,
                            );

                            let p1_src = (w_p1 as *const u8).add(r_off);
                            if pack_sw1 {
                                (kernels.active.gather)(sc.rb1.as_mut_ptr(), p1_src, &mcfg_in);
                            } else {
                                std::ptr::copy_nonoverlapping(
                                    p1_src as *const f64,
                                    sc.rb1.as_mut_ptr(),
                                    cols,
                                );
                            }
                            call_ntt_fwd(kernels, 
                                sc.rb1.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            let t2_src = (w_t2 as *const u8).add(r_off);
                            if pack_sw1 {
                                (kernels.active.gather)(sc.ra2.as_mut_ptr(), t2_src, &mcfg_in);
                            } else {
                                std::ptr::copy_nonoverlapping(
                                    t2_src as *const f64,
                                    sc.ra2.as_mut_ptr(),
                                    cols,
                                );
                            }
                            call_ntt_fwd(kernels, 
                                sc.ra2.as_mut_ptr(),
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );

                            let ptrs = [
                                sc.ra1.as_ptr(),
                                std::ptr::null(),
                                sc.rb1.as_ptr(),
                                sc.ra2.as_ptr(),
                            ];
                            (kernels.active.ntt_pmuladd)(sc.ra2.as_mut_ptr(), &ptrs, &vcfg_fma);
                            call_ntt_inv(kernels, 
                                sc.ra2.as_mut_ptr(),
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            let dst_t = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(out_off);
                            (kernels.active.scatter)(dst_t, sc.ra2.as_ptr(), &mcfg_out);

                            if have_p2 {
                                let p2_src = (w_p2 as *const u8).add(r_off);
                                if pack_sw1 {
                                    (kernels.active.gather)(sc.ra2.as_mut_ptr(), p2_src, &mcfg_in);
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        p2_src as *const f64,
                                        sc.ra2.as_mut_ptr(),
                                        cols,
                                    );
                                }
                                call_ntt_fwd_pmul_inv(
                                    kernels,
                                    sc.ra2.as_mut_ptr(),
                                    r_tw_ptr as *const f64,
                                    r_inv_tw_ptr as *const f64,
                                    sc.rb1.as_ptr(),
                                    cols as u64,
                                    &pr_row_inv,
                                );
                                let dst_p = (*(p_grid_ptr as *mut AlignedBytes).add(pi))
                                    .as_mut_ptr()
                                    .add(out_off);
                                (kernels.active.scatter)(dst_p, sc.ra2.as_ptr(), &mcfg_out);
                            }
                        }
                    },
                );
            }

            unsafe {
                t_grid[pi].set_len(total_grid_bytes);
                if have_q1 {
                    q_grid[pi].set_len(total_grid_bytes);
                }
                if have_p2 {
                    p_grid[pi].set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }
    } else {
        for pi in 0..k {
            let p_u64 = crt.primes_u64[pi];
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
            let mcfg_out = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: ROW_SWEEP_STREAMS,
            };

            if have_g2 {
                // Fused Steps 1..3 in 3 L3 grids: Sweep 1(q2 -> g0, t1 -> g1, [q1 -> g2]) + Row pass!
                if let Some((q1_l, _)) = q1_and_out {
                    ntt_sweep1_multi_prime_unpacked(
                        kernels,
                        sched,
                        &[
                            (q2_limbs, g0_base, unpacked_row_bytes),
                            (t1_limbs, g1_base, unpacked_row_bytes),
                            (q1_l, g2_base, unpacked_row_bytes),
                        ],
                        &spec,
                        tile_cols,
                        p_u64,
                        col_tables[pi],
                        psi_fwd[pi],
                        &tw_tile0_fwd[pi],
                        0,
                    );
                    let t_sw2_qt = std::time::Instant::now();
                    sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            let g0_r =
                                (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            let g1_r =
                                (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            let g2_r =
                                (g2_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(kernels, 
                                g0_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            call_ntt_fwd(kernels, 
                                g1_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            call_ntt_fwd(kernels, 
                                g2_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(
                                g2_r,
                                g2_r as *const f64,
                                g0_r as *const f64,
                                &vcfg,
                            );
                            call_ntt_inv(kernels, 
                                g2_r,
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            let dst_u8 = (*(q_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(r * row_bytes);
                            (kernels.active.scatter)(dst_u8, g2_r as *const f64, &mcfg_out);
                            (kernels.active.ntt_pmul)(
                                g0_r,
                                g0_r as *const f64,
                                g1_r as *const f64,
                                &vcfg,
                            );
                        }
                    });
                    unsafe {
                        q_grid[pi].set_len(total_grid_bytes);
                    }
                    NTT_TRACE_SW2_NS
                        .fetch_add(t_sw2_qt.elapsed().as_nanos() as u64, Ordering::Relaxed);
                } else {
                    ntt_sweep1_multi_prime_unpacked(
                        kernels,
                        sched,
                        &[
                            (q2_limbs, g0_base, unpacked_row_bytes),
                            (t1_limbs, g1_base, unpacked_row_bytes),
                        ],
                        &spec,
                        tile_cols,
                        p_u64,
                        col_tables[pi],
                        psi_fwd[pi],
                        &tw_tile0_fwd[pi],
                        0,
                    );
                    let t_sw2_qt = std::time::Instant::now();
                    sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            let g0_r =
                                (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            let g1_r =
                                (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(kernels, 
                                g0_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            call_ntt_fwd(kernels, 
                                g1_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(
                                g0_r,
                                g0_r as *const f64,
                                g1_r as *const f64,
                                &vcfg,
                            );
                        }
                    });
                    NTT_TRACE_SW2_NS
                        .fetch_add(t_sw2_qt.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            } else if let Some((q1_l, _)) = q1_and_out {
                // 2-grid L3 path (`lv = 19`): Sweep 1(q2 -> g0, q1 -> g1) + Row pass -> q_grid[pi]
                ntt_sweep1_multi_prime_unpacked(
                    kernels,
                    sched,
                    &[
                        (q2_limbs, g0_base, unpacked_row_bytes),
                        (q1_l, g1_base, unpacked_row_bytes),
                    ],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    0,
                );
                let t_sw2_q = std::time::Instant::now();
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let vcfg = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: 0,
                    };
                    unsafe {
                        let g0_r = (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        let g1_r = (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        call_ntt_fwd(kernels, g0_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                        call_ntt_fwd(kernels, g1_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                        (kernels.active.ntt_pmul)(
                            g1_r,
                            g1_r as *const f64,
                            g0_r as *const f64,
                            &vcfg,
                        );
                        call_ntt_inv(kernels, 
                            g1_r,
                            r_inv_tw_ptr as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        let dst_u8 = (*(q_grid_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, g1_r as *const f64, &mcfg_out);
                    }
                });
                unsafe {
                    q_grid[pi].set_len(total_grid_bytes);
                }
                NTT_TRACE_SW2_NS.fetch_add(t_sw2_q.elapsed().as_nanos() as u64, Ordering::Relaxed);
                NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

                ntt_sweep1_single_prime_unpacked(
                    kernels,
                    sched,
                    t1_limbs,
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    g1_base,
                    unpacked_row_bytes,
                );
                let t_sw2_t1 = std::time::Instant::now();
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let vcfg = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: 0,
                    };
                    let mcfg_tmp = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: row_bytes as u64,
                        packed: u64::from(spec.packed),
                        use_nt: 0,
                    };
                    unsafe {
                        let g0_r = (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        let g1_r = (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        call_ntt_fwd(kernels, g1_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                        (kernels.active.ntt_pmul)(
                            g0_r,
                            g0_r as *const f64,
                            g1_r as *const f64,
                            &vcfg,
                        );
                        let dst_u8 = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, g0_r as *const f64, &mcfg_tmp);
                    }
                });
                NTT_TRACE_SW2_NS
                    .fetch_add(t_sw2_t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            } else {
                ntt_sweep1_multi_prime_unpacked(
                    kernels,
                    sched,
                    &[
                        (q2_limbs, g0_base, unpacked_row_bytes),
                        (t1_limbs, g1_base, unpacked_row_bytes),
                    ],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    0,
                );
                let t_sw2_t1 = std::time::Instant::now();
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let vcfg = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: 0,
                    };
                    let mcfg_tmp = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: row_bytes as u64,
                        packed: u64::from(spec.packed),
                        use_nt: 0,
                    };
                    unsafe {
                        let g0_r = (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        let g1_r = (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        call_ntt_fwd(kernels, g0_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                        call_ntt_fwd(kernels, g1_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                        (kernels.active.ntt_pmul)(
                            g0_r,
                            g0_r as *const f64,
                            g1_r as *const f64,
                            &vcfg,
                        );
                        let dst_u8 = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, g0_r as *const f64, &mcfg_tmp);
                    }
                });
                NTT_TRACE_SW2_NS
                    .fetch_add(t_sw2_t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }

            if have_g2 {
                // 4a. 3-grid L3 path (`lv = 13..18`): multi-operand Sweep 1(p1, t2) -> (g1, g2)
                ntt_sweep1_multi_prime_unpacked(
                    kernels,
                    sched,
                    &[
                        (p1_limbs, g1_base, unpacked_row_bytes),
                        (t2_limbs, g2_base, unpacked_row_bytes),
                    ],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    0,
                );
                let t_sw2_t = std::time::Instant::now();
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let vcfg_fma = NttVecCfg {
                        n: cols as u64,
                        p: pr_row_inv.p,
                        inv_p: pr_row_inv.inv_p,
                        sub_second: u64::from(sub_second),
                    };
                    unsafe {
                        let g0_r = (g0_base as *const u8).add(r * unpacked_row_bytes) as *const f64;
                        let g1_r = (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        let g2_r = (g2_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                        call_ntt_fwd(kernels, 
                            g1_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        call_ntt_fwd(kernels, 
                            g2_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        let ptrs = [
                            g0_r,
                            std::ptr::null(),
                            g1_r as *const f64,
                            g2_r as *const f64,
                        ];
                        (kernels.active.ntt_pmuladd)(g2_r, &ptrs, &vcfg_fma);
                        call_ntt_inv(kernels, 
                            g2_r,
                            r_inv_tw_ptr as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        let dst_u8 = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, g2_r as *const f64, &mcfg_out);
                    }
                });
                unsafe {
                    t_grid[pi].set_len(total_grid_bytes);
                }
                NTT_TRACE_SW2_NS
                    .fetch_add(t_sw2_t.elapsed().as_nanos() as u64, Ordering::Relaxed);

                // 5a. If `p2` needed: Sweep1(p2) -> g0, then g1 * fwd(g0) -> inv -> p_grid[pi]
                if let Some((p2_l, _)) = p2_and_out {
                    ntt_sweep1_single_prime_unpacked(
                        kernels,
                        sched,
                        p2_l,
                        &spec,
                        tile_cols,
                        p_u64,
                        col_tables[pi],
                        psi_fwd[pi],
                        &tw_tile0_fwd[pi],
                        g0_base,
                        unpacked_row_bytes,
                    );
                    let t_sw2_p = std::time::Instant::now();
                    sched.par_for_leaf_max(rows, row_chunk, spec.par_workers, |r| {
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            let g1_r =
                                (g1_base as *const u8).add(r * unpacked_row_bytes) as *const f64;
                            let g0_r =
                                (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(kernels, 
                                g0_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(g0_r, g1_r, g0_r as *const f64, &vcfg);
                            call_ntt_inv(kernels, 
                                g0_r,
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            let dst_u8 = (*(p_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(r * row_bytes);
                            (kernels.active.scatter)(dst_u8, g0_r as *const f64, &mcfg_out);
                        }
                    });
                    unsafe {
                        p_grid[pi].set_len(total_grid_bytes);
                    }
                    NTT_TRACE_SW2_NS
                        .fetch_add(t_sw2_p.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            } else {
                // 4b. 2-grid L3 path (`lv = 19`): multi-operand Sweep 1(p1, t2) -> (g0, g1)
                ntt_sweep1_multi_prime_unpacked(
                    kernels,
                    sched,
                    &[
                        (p1_limbs, g0_base, unpacked_row_bytes),
                        (t2_limbs, g1_base, unpacked_row_bytes),
                    ],
                    &spec,
                    tile_cols,
                    p_u64,
                    col_tables[pi],
                    psi_fwd[pi],
                    &tw_tile0_fwd[pi],
                    0,
                );
                let t_sw2_t = std::time::Instant::now();
                sched.par_for_init_max(
                    rows,
                    row_chunk,
                    sw2_w,
                    || {
                        let mut lease = ScratchLease::acquire();
                        resize_f64_buf(&mut lease.get_mut().ra1, cols);
                        lease
                    },
                    |lease, r| {
                        let sc = lease.get_mut();
                        let mcfg_in = MoveCfg {
                            rows: 1,
                            cols: cols as u64,
                            outer_stride_bytes: row_bytes as u64,
                            packed: u64::from(spec.packed),
                            use_nt: 0,
                        };
                        let vcfg_fma = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: u64::from(sub_second),
                        };
                        unsafe {
                            let t_u8 = (*(t_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(r * row_bytes);
                            (kernels.active.gather)(sc.ra1.as_mut_ptr(), t_u8, &mcfg_in);
                            let g0_r =
                                (g0_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            let g1_r =
                                (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(kernels, 
                                g0_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            call_ntt_fwd(kernels, 
                                g1_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            let ptrs = [
                                sc.ra1.as_ptr(),
                                std::ptr::null(),
                                g0_r as *const f64,
                                g1_r as *const f64,
                            ];
                            (kernels.active.ntt_pmuladd)(g1_r, &ptrs, &vcfg_fma);
                            call_ntt_inv(kernels, 
                                g1_r,
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            (kernels.active.scatter)(t_u8, g1_r as *const f64, &mcfg_out);
                        }
                    },
                );
                unsafe {
                    t_grid[pi].set_len(total_grid_bytes);
                }
                NTT_TRACE_SW2_NS
                    .fetch_add(t_sw2_t.elapsed().as_nanos() as u64, Ordering::Relaxed);

                if let Some((p2_l, _)) = p2_and_out {
                    ntt_sweep1_single_prime_unpacked(
                        kernels,
                        sched,
                        p2_l,
                        &spec,
                        tile_cols,
                        p_u64,
                        col_tables[pi],
                        psi_fwd[pi],
                        &tw_tile0_fwd[pi],
                        g1_base,
                        unpacked_row_bytes,
                    );
                    let t_sw2_p = std::time::Instant::now();
                    sched.par_for_leaf_max(rows, row_chunk, spec.par_workers, |r| {
                        let vcfg = NttVecCfg {
                            n: cols as u64,
                            p: pr_row_inv.p,
                            inv_p: pr_row_inv.inv_p,
                            sub_second: 0,
                        };
                        unsafe {
                            let g0_r =
                                (g0_base as *const u8).add(r * unpacked_row_bytes) as *const f64;
                            let g1_r =
                                (g1_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                            call_ntt_fwd(kernels, 
                                g1_r,
                                r_tw_ptr as *const f64,
                                cols as u64,
                                &pr_fwd,
                            );
                            (kernels.active.ntt_pmul)(g1_r, g0_r, g1_r as *const f64, &vcfg);
                            call_ntt_inv(kernels, 
                                g1_r,
                                r_inv_tw_ptr as *const f64,
                                cols as u64,
                                &pr_row_inv,
                            );
                            let dst_u8 = (*(p_grid_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(r * row_bytes);
                            (kernels.active.scatter)(dst_u8, g1_r as *const f64, &mcfg_out);
                        }
                    });
                    unsafe {
                        p_grid[pi].set_len(total_grid_bytes);
                    }
                    NTT_TRACE_SW2_NS
                        .fetch_add(t_sw2_p.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        }
    }

    for g in &mut work_grids {
        unsafe {
            g.set_len(work_prime_bytes);
        }
    }
    cache.put_byte_bufs(work_grids);

    if kernels.count_work {
        let n_fwd = 4u64 + u64::from(q1_and_out.is_some()) + u64::from(p2_and_out.is_some());
        let bf_cols = n_fwd * ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
        kernels
            .stats
            .ntt_butterflies
            .fetch_add(bf_cols, Ordering::Relaxed);
    }

    if (q1_and_out.is_some() || p2_and_out.is_some())
        && out_bytes.saturating_mul(2) <= ram_transform_budget.max(64 * 1024)
        && let Some(res) = ntt_sweep3_and_crt_multi(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &col_tables,
            &mut t_grid,
            sub_second,
            t_out_limbs,
            q1_and_out.map(|(_, ql)| ql),
            &mut q_grid,
            p2_and_out.map(|(_, pl)| pl),
            &mut p_grid,
        )
    {
        return Some(res);
    }

    let Ok(t) = ntt_sweep3_and_crt(
        kernels,
        cache,
        sched,
        &spec,
        crt,
        t_grid,
        sub_second,
        t_out_limbs,
        Some(4),
    ) else {
        if !q_grid.is_empty() {
            cache.put_byte_bufs(q_grid);
        }
        if !p_grid.is_empty() {
            cache.put_byte_bufs(p_grid);
        }
        return None;
    };

    let q = if let Some((_, q_out_l)) = q1_and_out {
        match ntt_sweep3_and_crt(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            q_grid,
            false,
            q_out_l,
            Some(1),
        ) {
            Ok(v) => Some(v),
            Err(_) => {
                if !p_grid.is_empty() {
                    cache.put_byte_bufs(p_grid);
                }
                return None;
            }
        }
    } else {
        None
    };

    let p = if let Some((_, p_out_l)) = p2_and_out {
        match ntt_sweep3_and_crt(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            p_grid,
            false,
            p_out_l,
            Some(1),
        ) {
            Ok(v) => Some(v),
            Err(_) => return None,
        }
    } else {
        None
    };

    Some(FusedMergeRaw { t, q, p })
}

#[allow(clippy::too_many_arguments)]
fn ntt_sweep3_and_crt_multi(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    spec: &NttSpec,
    crt: &CrtPlan,
    col_tables: &[&PrimeTables],
    t_grid: &mut Vec<AlignedBytes>,
    t_signed: bool,
    t_out_limbs: usize,
    q_out_limbs: Option<usize>,
    q_grid: &mut Vec<AlignedBytes>,
    p_out_limbs: Option<usize>,
    p_grid: &mut Vec<AlignedBytes>,
) -> Option<FusedMergeRaw> {
    let k = spec.k_primes;
    let rows = spec.rows;
    let cols = spec.cols;
    let row_bytes = spec.outer_row_bytes();
    let tile_cols = spec.tile_cols.min(cols).max(1);
    let b_usize = spec.coeff_bits as usize;
    let pw_sw3 = spec.par_workers.max(1);
    let min_groups = if pw_sw3 > 1 { pw_sw3 * 2 } else { 1 };

    let l2_share = sched.min_l2_share() as usize;
    let fits_seg_fast = |seg: usize| -> bool {
        cols >= seg * min_groups
            && cols.is_multiple_of(seg)
            && (seg * b_usize).is_multiple_of(64)
            && rows * seg * 8 <= l2_share / 16
            && k * rows * seg * 8 <= (l2_share * 5) / 8
    };
    let fits_seg_l2 = |seg: usize| -> bool {
        cols >= seg * min_groups
            && cols.is_multiple_of(seg)
            && (seg * b_usize).is_multiple_of(64)
            && rows * seg * 8 <= l2_share / 4
            && k * rows * seg * 8 <= (l2_share * 7) / 8
    };
    let seg_cols = if fits_seg_fast(64) {
        64
    } else if fits_seg_fast(32) {
        32
    } else if fits_seg_fast(128) {
        128
    } else if fits_seg_l2(64) {
        64
    } else if fits_seg_l2(32) {
        32
    } else if cols >= 32 * min_groups
        && cols.is_multiple_of(32)
        && (32 * b_usize).is_multiple_of(64)
    {
        32
    } else if cols >= 64 * min_groups
        && cols.is_multiple_of(64)
        && (64 * b_usize).is_multiple_of(64)
    {
        64
    } else if cols >= 32 && cols.is_multiple_of(32) && (32 * b_usize).is_multiple_of(64) {
        32
    } else if cols >= 64 && cols.is_multiple_of(64) && (64 * b_usize).is_multiple_of(64) {
        64
    } else if fits_seg_l2(16) {
        16
    } else if fits_seg_l2(8) {
        8
    } else if (tile_cols * b_usize).is_multiple_of(64) {
        tile_cols
    } else if (cols * b_usize).is_multiple_of(64) {
        cols
    } else {
        return None;
    };

    let psi_inv: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                spec.negacyclic,
                true,
            )
        })
        .collect();
    let prime_col_inv: Vec<NttPrime> = (0..k).map(|pi| col_tables[pi].pr_inv).collect();
    let tw_seg0_inv = cache.get_tile0_twiddles_par(
        kernels,
        sched,
        spec.par_workers,
        &crt.primes_u64,
        col_tables,
        k,
        spec.n,
        rows,
        seg_cols,
        spec.negacyclic,
        true,
    );

    let num_col_groups = cols / seg_cols;
    let seg_limbs = (seg_cols * b_usize) / 64;
    let m_bits = (spec.n as u64) * (spec.coeff_bits as u64);

    struct Sw3Slot {
        grid: Vec<AlignedBytes>,
        signed: bool,
        out_limbs: usize,
        needed_rows: usize,
        total_segs: usize,
        total_limbs: usize,
        full_limbs: usize,
        out: LimbVec,
        seg_carries: TransU64Vec,
        seg_w0s: TransU64Vec,
        out_ptr: usize,
        carries_ptr: usize,
        w0s_ptr: usize,
        fwd_row_ops: u64,
    }

    let make_slot = |grid: Vec<AlignedBytes>, signed: bool, out_l: usize, fwd_ops: u64| -> Sw3Slot {
        let can_skip = !spec.out_of_core && !signed && !spec.negacyclic && !spec.wrap_cyclic;
        let needed_rows = if can_skip {
            (out_l + 2)
                .div_ceil(num_col_groups * seg_limbs)
                .min(rows)
                .max(1)
        } else {
            rows
        };
        let total_segs = needed_rows * num_col_groups;
        let total_limbs = total_segs * seg_limbs;
        let full_limbs = (total_limbs + crt.k_words + 2).max(out_l);
        let t_al3 = std::time::Instant::now();
        let mut out = crate::limb::take_limb_buf_committed(full_limbs, sched, spec.par_workers);
        let mut seg_carries = cache.take_u64_buf_uninit(total_segs * crt.k_words);
        let mut seg_w0s = cache.take_u64_buf_uninit(total_segs);
        NTT_TRACE_ALLOC_NS.fetch_add(t_al3.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        let out_ptr = out.as_mut_ptr() as usize;
        let carries_ptr = seg_carries.as_mut_ptr() as usize;
        let w0s_ptr = seg_w0s.as_mut_ptr() as usize;
        Sw3Slot {
            grid,
            signed,
            out_limbs: out_l,
            needed_rows,
            total_segs,
            total_limbs,
            full_limbs,
            out,
            seg_carries,
            seg_w0s,
            out_ptr,
            carries_ptr,
            w0s_ptr,
            fwd_row_ops: fwd_ops,
        }
    };

    let mut slots: Vec<Sw3Slot> = Vec::with_capacity(3);
    slots.push(make_slot(std::mem::take(t_grid), t_signed, t_out_limbs, 4));
    if let Some(ql) = q_out_limbs {
        slots.push(make_slot(std::mem::take(q_grid), false, ql, 1));
    }
    if let Some(pl) = p_out_limbs {
        slots.push(make_slot(std::mem::take(p_grid), false, pl, 1));
    }

    let t_sw3 = std::time::Instant::now();
    let pw = spec.par_workers.max(1);
    let n_slots = slots.len();
    let fine_cgs = if pw > 1 && n_slots >= 2 && rows >= 128 {
        pw.min(num_col_groups / 2)
    } else {
        0
    };
    let whole_cgs = num_col_groups - fine_cgs;
    let cg_chunk = if fine_cgs > 0 || n_slots * k * rows * seg_cols >= 4_096 {
        1
    } else {
        (128 / seg_cols)
            .max(1)
            .min((num_col_groups / (pw * 8)).max(1))
            .min(4)
    };
    sched.par_for_init_max(
        whole_cgs + fine_cgs * n_slots,
        cg_chunk,
        spec.par_workers,
        || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            if sc.prime_segs.len() < k {
                sc.prime_segs.resize_with(k, || Vec::new_in(ScratchAlloc));
            }
            for pi in 0..k {
                resize_f64_buf(&mut sc.prime_segs[pi], pi * 40 + rows * seg_cols);
            }
            if sc.row_ptrs.len() < k {
                sc.row_ptrs.resize(k, 0);
            }
            lease
        },
        |lease, task| {
            let (cg, only_slot) = if task < whole_cgs {
                (task, None)
            } else {
                let f = task - whole_cgs;
                (whole_cgs + f / n_slots, Some(f % n_slots))
            };
            let sc = lease.get_mut();
            let mcfg = MoveCfg {
                rows: rows as u64,
                cols: seg_cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: u64::from(
                    spec.use_nt && k * rows * row_bytes > cache.l3_prime_cap_bytes(),
                ),
            };
            let shape = [rows as u64, seg_cols as u64];
            let col_start = cg * seg_cols;
            let col_byte_off = if spec.packed {
                (col_start / 8) * 50
            } else {
                col_start * 8
            };
            let seg_elems = rows * seg_cols;

            for (si, sl) in slots.iter().enumerate() {
                if only_slot.is_some_and(|want| want != si) {
                    continue;
                }
                for pi in 0..k {
                    let seg_slice = &mut sc.prime_segs[pi][(pi * 40)..(pi * 40 + seg_elems)];
                    let pr_col = if rows > 1 {
                        &col_tables[pi].pr_fwd
                    } else {
                        &prime_col_inv[pi]
                    };
                    unsafe {
                        (kernels.active.gather)(
                            seg_slice.as_mut_ptr(),
                            sl.grid[pi].as_ptr().add(col_byte_off),
                            &mcfg,
                        );
                    }
                    apply_tile_twist_inv_scaled(
                        kernels,
                        seg_slice,
                        &tw_seg0_inv[pi],
                        rows,
                        seg_cols,
                        col_start,
                        psi_inv[pi],
                        crt.primes_u64[pi],
                        prime_col_inv[pi].p,
                        prime_col_inv[pi].inv_p,
                        &mut sc.tw_row,
                        &mut sc.s_u64,
                    );
                    unsafe {
                        call_ntt_inv_cols(
                            kernels,
                            seg_slice.as_mut_ptr(),
                            col_tables[pi].inv_tw.as_ptr(),
                            &shape,
                            pr_col,
                            &mut sc.col_scratch,
                        );
                    }
                }

                // The rows of this column group are one stream each, segment
                // `r * num_col_groups + cg` of the product: one call recombines them all.
                for pi in 0..k {
                    sc.row_ptrs[pi] = unsafe { sc.prime_segs[pi].as_ptr().add(pi * 40) } as usize;
                }
                let mut ccfg = NttCrtCfg {
                    k_primes: k as u64,
                    count: seg_cols as u64,
                    coeff_bits: spec.coeff_bits as u64,
                    out_limbs: seg_limbs as u64,
                    signed: u64::from(sl.signed),
                    wrap_cyclic: 0,
                    primes: crt.primes_f64.as_ptr(),
                    inv_primes: crt.inv_primes_f64.as_ptr(),
                    garner_inv: crt.garner_inv.as_ptr(),
                    prefix_prods: crt.prefix_prods.as_ptr(),
                    k_words: crt.k_words as u64,
                    segs: NttCrtSegs {
                        segments: sl.needed_rows as u64,
                        res_stride: seg_cols as u64,
                        out_stride: (num_col_groups * seg_limbs) as u64,
                        carry_stride: (num_col_groups * crt.k_words) as u64,
                        w0: unsafe { (sl.w0s_ptr as *mut u64).add(cg) },
                        w0_stride: num_col_groups as u64,
                    },
                    carry_limbs: unsafe { (sl.carries_ptr as *mut u64).add(cg * crt.k_words) },
                };
                unsafe {
                    ntt_crt_streams(
                        kernels,
                        (sl.out_ptr as *mut u64).add(cg * seg_limbs),
                        sc.row_ptrs.as_ptr() as *const *const f64,
                        &mut ccfg,
                    );
                }
            }
        },
    );
    NTT_TRACE_SW3_NS.fetch_add(t_sw3.elapsed().as_nanos() as u64, Ordering::Relaxed);
    NTT_IDLE_SW3_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

    let mut finished: Vec<(LimbVec, u64)> = Vec::with_capacity(slots.len());
    for mut sl in slots {
        cache.put_byte_bufs(sl.grid);
        unsafe {
            if sl.full_limbs > sl.total_limbs {
                std::ptr::write_bytes(
                    (sl.out_ptr as *mut u64).add(sl.total_limbs),
                    0,
                    sl.full_limbs - sl.total_limbs,
                );
            }
            sl.out.set_len(sl.full_limbs);
        }
        let t_w0 = std::time::Instant::now();
        let w0 = finalize_segmented_w0(
            kernels,
            sched,
            cache,
            &mut sl.out,
            sl.seg_carries,
            sl.seg_w0s,
            sl.total_segs,
            seg_limbs,
            crt.k_words,
            sl.total_limbs,
            sl.out_limbs,
            m_bits,
            spec,
        );
        NTT_TRACE_W0_NS.fetch_add(t_w0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_W0_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        if kernels.count_work {
            let n = spec.n as u64;
            let bf = (n / 2)
                * (sl.fwd_row_ops * (cols.ilog2() as u64) + ((cols.ilog2() + rows.ilog2()) as u64))
                * (k as u64);
            kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
            kernels
                .stats
                .crt_ops
                .fetch_add((spec.n * k * k) as u64, Ordering::Relaxed);
            kernels.stats.bytes_moved.fetch_add(
                ((2 + sl.fwd_row_ops as usize) * k * rows * row_bytes) as u64,
                Ordering::Relaxed,
            );
        }
        finished.push((sl.out, w0));
    }

    let mut it = finished.into_iter();
    let t = it.next().unwrap();
    let q = if q_out_limbs.is_some() {
        Some(it.next().unwrap())
    } else {
        None
    };
    let p = if p_out_limbs.is_some() {
        Some(it.next().unwrap())
    } else {
        None
    };
    Some(FusedMergeRaw { t, q, p })
}

/// Fused 5-transform pair `(a1 * b, a2 * b)` sharing `fwd(b)` inside L2 (`rows == 1`) or L3 (`rows > 1`).
#[allow(clippy::too_many_arguments)]
pub fn ntt_mul_pair_shared_b(
    kernels: &Kernels,
    cache: &NttTableCache,
    sched: &Scheduler,
    spec: NttSpec,
    ram_transform_budget: u64,
    a1_limbs: &[u64],
    a1_out_limbs: usize,
    a2_limbs: &[u64],
    a2_out_limbs: usize,
    b_limbs: &[u64],
) -> Option<((LimbVec, u64), (LimbVec, u64))> {
    if spec.out_of_core || spec.negacyclic || spec.wrap_cyclic || spec.extra_sweeps != 0 {
        return None;
    }
    let crt = cache.get_crt_ref(spec.min_two_adic(), spec.k_primes);
    let k = spec.k_primes;
    let mut _pt = PathTimer::start(
        "pair",
        &spec,
        &[a1_limbs.len(), a2_limbs.len(), b_limbs.len()],
    );

    if spec.rows == 1 {
        if spec.packed {
            return None;
        }
        let n = spec.n;
        let p_indep =
            k >= 2 && spec.coeff_bits <= 52 && (1u64 << spec.coeff_bits) <= crt.primes_u64[0];
        let num_ch = (spec.coeff_bits as usize).div_ceil(48);
        let p_chunk_cand = !p_indep
            && k >= 2
            && (53..=384).contains(&spec.coeff_bits)
            && crt.primes_u64[k - 1] > (1u64 << 48);
        let max_1d_ch_bytes =
            (cache.l3_prime_cap_bytes() / sched.num_workers().max(1)).max(512 * 1024);

        let t0 = cache.get_twiddles_ref(crt.primes_u64[0], n, false);
        let mut f_lease = Fused1dLease::acquire();
        let f1d = f_lease.get_mut();
        let res_stride = res_stride_1d(n);

        if spec.par_workers <= 1 && k > 1 && n > 8192 {
            // Single-worker large 1D pair (`n >= 16384`): under full-machine SMT load two threads
            // share physical L2 (`512 KiB` per worker), so keep `m_ch` sized to a single operand
            // (`cur_off = 0` reused per operand) and reuse `m_t_res` across `a1` and `a2` instead
            // of allocating a second `k * res_stride` output buffer.
            let single_ch_words = ch_words_for_1d(
                nz_elems_1d(b_limbs.len(), spec.coeff_bits, n),
                p_indep,
                true,
                num_ch,
            )
            .max(ch_words_for_1d(
                nz_elems_1d(a1_limbs.len(), spec.coeff_bits, n),
                p_indep,
                true,
                num_ch,
            ))
            .max(ch_words_for_1d(
                nz_elems_1d(a2_limbs.len(), spec.coeff_bits, n),
                p_indep,
                true,
                num_ch,
            ));
            let p_chunk = p_chunk_cand && single_ch_words * 8 <= max_1d_ch_bytes;
            let ch_need = if p_chunk || p_indep { single_ch_words } else { 0 };
            resize_f64_buf(&mut f1d.m_ch, ch_need);
            resize_f64_buf(&mut f1d.m_q_res, k * n);
            resize_f64_buf(&mut f1d.m_t_res, k * res_stride);

            let mut tables_vec: Vec<&PrimeTables> = Vec::with_capacity(k);
            tables_vec.push(t0);
            for pi in 1..k {
                tables_vec.push(cache.get_twiddles_ref(crt.primes_u64[pi], n, false));
            }

            let t_f1 = ntt_trace_enabled().then(std::time::Instant::now);
            let mut cur_off = 0usize;
            let prep_b = prep_1d_operand(
                kernels,
                b_limbs,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            );
            let m_ch_ptr = f1d.m_ch.as_ptr();
            let fb_base = f1d.m_q_res.as_mut_ptr();
            for (pi, &tables) in tables_vec.iter().enumerate() {
                unsafe {
                    eval_1d_fwd(
                        kernels,
                        b_limbs,
                        prep_b,
                        m_ch_ptr,
                        tables,
                        n,
                        spec.coeff_bits,
                        fb_base.add(pi * n),
                    );
                }
            }

            cur_off = 0;
            let prep_a1 = prep_1d_operand(
                kernels,
                a1_limbs,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            );
            let m_ch_ptr = f1d.m_ch.as_ptr();
            let dst_base = f1d.m_t_res.as_mut_ptr();
            let fb_const = f1d.m_q_res.as_ptr();
            for (pi, &tables) in tables_vec.iter().enumerate() {
                unsafe {
                    eval_1d_fwd_pmul_inv(
                        kernels,
                        a1_limbs,
                        prep_a1,
                        m_ch_ptr,
                        tables,
                        n,
                        spec.coeff_bits,
                        dst_base.add(pi * res_stride),
                        fb_const.add(pi * n),
                    );
                }
            }
            if let Some(t) = t_f1 {
                NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            let r1 = crt_1d_from_flat(
                kernels,
                cache,
                sched,
                &spec,
                crt,
                &f1d.m_t_res[..(k * res_stride)],
                res_stride,
                false,
                a1_out_limbs,
            );

            let t_f2 = ntt_trace_enabled().then(std::time::Instant::now);
            cur_off = 0;
            let prep_a2 = prep_1d_operand(
                kernels,
                a2_limbs,
                n,
                spec.coeff_bits,
                p_indep,
                p_chunk,
                num_ch,
                &t0.pr_fwd,
                &mut f1d.m_ch,
                &mut cur_off,
            );
            let m_ch_ptr = f1d.m_ch.as_ptr();
            let dst_base = f1d.m_t_res.as_mut_ptr();
            let fb_const = f1d.m_q_res.as_ptr();
            for (pi, &tables) in tables_vec.iter().enumerate() {
                unsafe {
                    eval_1d_fwd_pmul_inv(
                        kernels,
                        a2_limbs,
                        prep_a2,
                        m_ch_ptr,
                        tables,
                        n,
                        spec.coeff_bits,
                        dst_base.add(pi * res_stride),
                        fb_const.add(pi * n),
                    );
                }
            }
            if let Some(t) = t_f2 {
                NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            if kernels.count_work && n > 1 {
                let bf = 5u64 * ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64)
                    + 2u64 * (n as u64) * (k as u64);
                kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
                kernels
                    .stats
                    .bytes_moved
                    .fetch_add((2 * k * n * 8) as u64, Ordering::Relaxed);
            }
            let r2 = crt_1d_from_flat(
                kernels,
                cache,
                sched,
                &spec,
                crt,
                &f1d.m_t_res[..(k * res_stride)],
                res_stride,
                false,
                a2_out_limbs,
            );
            return Some((r1, r2));
        }

        let cand_ch_words = ch_words_for_1d(
            nz_elems_1d(b_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        ) + ch_words_for_1d(
            nz_elems_1d(a1_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        ) + ch_words_for_1d(
            nz_elems_1d(a2_limbs.len(), spec.coeff_bits, n),
            p_indep,
            true,
            num_ch,
        );
        let p_chunk = p_chunk_cand && cand_ch_words * 8 <= max_1d_ch_bytes;
        let ch_need = if p_chunk || p_indep { cand_ch_words } else { 0 };
        resize_f64_buf(&mut f1d.m_ch, ch_need);
        resize_f64_buf(&mut f1d.m_t_res, k * res_stride);
        resize_f64_buf(&mut f1d.m_q_res, k * res_stride);

        let t_f1 = ntt_trace_enabled().then(std::time::Instant::now);
        let mut cur_off = 0usize;
        let prep_b = prep_1d_operand(
            kernels,
            b_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_a1 = prep_1d_operand(
            kernels,
            a1_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );
        let prep_a2 = prep_1d_operand(
            kernels,
            a2_limbs,
            n,
            spec.coeff_bits,
            p_indep,
            p_chunk,
            num_ch,
            &t0.pr_fwd,
            &mut f1d.m_ch,
            &mut cur_off,
        );

        let m_ch_ptr = f1d.m_ch.as_ptr() as usize;
        let res1_ptr = f1d.m_t_res.as_mut_ptr() as usize;
        let res2_ptr = f1d.m_q_res.as_mut_ptr() as usize;

        let eval_prime = |lease: &mut ScratchLease, pi: usize| {
            let sc = lease.get_mut();
            let p_u64 = crt.primes_u64[pi];
            let tables = cache.get_twiddles_ref(p_u64, n, false);
            unsafe {
                let fb = sc.ra1.as_mut_ptr();
                let dst1 = (res1_ptr as *mut f64).add(pi * res_stride);
                let dst2 = (res2_ptr as *mut f64).add(pi * res_stride);
                eval_1d_fwd(
                    kernels,
                    b_limbs,
                    prep_b,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    fb,
                );
                eval_1d_fwd_pmul_inv(
                    kernels,
                    a1_limbs,
                    prep_a1,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    dst1,
                    fb as *const f64,
                );
                eval_1d_fwd_pmul_inv(
                    kernels,
                    a2_limbs,
                    prep_a2,
                    m_ch_ptr as *const f64,
                    tables,
                    n,
                    spec.coeff_bits,
                    dst2,
                    fb as *const f64,
                );
            }
        };
        if spec.par_workers <= 1 || k <= 1 {
            let mut lease = ScratchLease::acquire();
            resize_f64_buf(&mut lease.get_mut().ra1, n);
            for pi in 0..k {
                eval_prime(&mut lease, pi);
            }
        } else {
            sched.par_for_init_max(
                k,
                1,
                spec.par_workers,
                || {
                    let mut lease = ScratchLease::acquire();
                    resize_f64_buf(&mut lease.get_mut().ra1, n);
                    lease
                },
                eval_prime,
            );
        }
        if let Some(t) = t_f1 {
            NTT_TRACE_1D_FWD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if kernels.count_work && n > 1 {
            let bf = 5u64 * ((n as u64) / 2) * (n.ilog2() as u64) * (k as u64)
                + 2u64 * (n as u64) * (k as u64);
            kernels.stats.ntt_butterflies.fetch_add(bf, Ordering::Relaxed);
            kernels
                .stats
                .bytes_moved
                .fetch_add((2 * k * n * 8) as u64, Ordering::Relaxed);
        }
        let r1 = crt_1d_from_flat(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &f1d.m_t_res[..(k * res_stride)],
            res_stride,
            false,
            a1_out_limbs,
        );
        let r2 = crt_1d_from_flat(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &f1d.m_q_res[..(k * res_stride)],
            res_stride,
            false,
            a2_out_limbs,
        );
        return Some((r1, r2));
    }

    let rows = spec.rows;
    let cols = spec.cols;
    if cols < 64 {
        return None;
    }
    let unpacked_row_bytes = if cols >= 512 { cols * 8 + 512 } else { cols * 8 };
    let unpacked_prime_bytes = rows * unpacked_row_bytes;
    let est_conc = sched.num_workers().div_ceil(spec.par_workers.max(1)).max(1);
    let my_l3_cap = cache.l3_prime_cap_bytes() / est_conc;
    let base_tc = spec.tile_cols.min(cols).max(1);
    let tile_cols = if rows >= 128 && cols.is_multiple_of(32) {
        base_tc.min(32)
    } else {
        base_tc
    };
    let row_bytes = spec.outer_row_bytes();
    let total_grid_bytes = rows * row_bytes;
    let out_bytes = 2u64
        .saturating_mul(k as u64)
        .saturating_mul(total_grid_bytes as u64);
    let pack_sw1 = can_pack_sweep1(&spec, tile_cols);
    let rf_work_bytes = if pack_sw1 {
        total_grid_bytes
    } else {
        3 * unpacked_prime_bytes
    };
    let dram_row_fused =
        out_bytes.saturating_add(rf_work_bytes as u64) <= ram_transform_budget.max(64 * 1024);
    let use_pack_sw1 = dram_row_fused && pack_sw1;
    _pt.tag(&format!("row_fused={dram_row_fused} packed_work={use_pack_sw1}"));
    // Packed work in groups of 256 columns: `b` too is made for all primes in one pass, into a
    // work grid a prime, where there is room for them and one such grid is more than half of
    // this product's share of the last-level cache. Below that the one grid `b` is made in
    // prime by prime is still in the cache when the row sweep reads it, which is worth more
    // than reading `b` once and running the row sweep as one loop (measured: a pair of
    // 600 Mbit operands 6 to 13% cheaper with a grid a prime on 4, 8 and 16 workers, one of
    // 75 Mbit on 4 workers of 16 busy 4 to 8% dearer). The run bounds the grids of a pair that
    // may do so (`work_grid_room`): at 5e9 digits in 48 GB the more grids below the top level
    // left that level 1 to 2.6 GB more to commit, up to and past where the machine pages.
    let with_grid_a_prime =
        out_bytes.saturating_add((k as u64).saturating_mul(total_grid_bytes as u64));
    let conc_grid_bytes = (est_conc as u64).saturating_mul(with_grid_a_prime);
    let a_in_out_unpacked = dram_row_fused && !use_pack_sw1 && !spec.packed;
    let sw1_once_unpacked_cand =
        a_in_out_unpacked && (cols / tile_cols) >= spec.par_workers.max(1);
    let sw1_once_unpacked_b = sw1_once_unpacked_cand
        && 3 * k * unpacked_prime_bytes <= my_l3_cap
        && with_grid_a_prime <= ram_transform_budget.max(64 * 1024)
        && conc_grid_bytes <= cache.work_grid_room()
        && room_for_work_grids(
            conc_grid_bytes.min(usize::MAX as u64) as usize,
            (k - 1) * unpacked_prime_bytes,
        );
    let sw1_once_b = (use_pack_sw1
        && sweep1_groups_256(&spec, tile_cols)
        && 2 * total_grid_bytes > my_l3_cap
        && with_grid_a_prime <= ram_transform_budget.max(64 * 1024)
        && conc_grid_bytes <= cache.work_grid_room()
        && room_for_work_grids(
            conc_grid_bytes.min(usize::MAX as u64) as usize,
            (k - 1) * total_grid_bytes,
        ))
        || sw1_once_unpacked_b;
    let need_work = if sw1_once_b {
        k
    } else if use_pack_sw1 || a_in_out_unpacked {
        1usize
    } else if dram_row_fused {
        3usize
    } else {
        1usize + usize::from(spec.packed)
    };
    let work_prime_bytes = if use_pack_sw1 {
        total_grid_bytes
    } else {
        unpacked_prime_bytes
    };
    let work_row_bytes = if use_pack_sw1 {
        row_bytes
    } else {
        unpacked_row_bytes
    };
    if out_bytes.saturating_add((need_work as u64).saturating_mul(work_prime_bytes as u64))
        > ram_transform_budget.max(64 * 1024)
    {
        return None;
    }

    let t_al = ntt_trace_enabled().then(std::time::Instant::now);
    let (mut g1_out, mut g2_out, mut work_grids) = if work_prime_bytes == total_grid_bytes {
        let mut all = cache.take_byte_bufs_transient_committed(
            2 * k + need_work,
            total_grid_bytes,
            sched,
            spec.par_workers,
        );
        let wg = all.split_off(2 * k);
        let g2 = all.split_off(k);
        (all, g2, wg)
    } else {
        let mut outs = cache.take_byte_bufs_transient_committed(
            2 * k,
            total_grid_bytes,
            sched,
            spec.par_workers,
        );
        let g2 = outs.split_off(k);
        let wg = cache.take_byte_bufs_transient_committed(
            need_work,
            work_prime_bytes,
            sched,
            spec.par_workers,
        );
        (outs, g2, wg)
    };
    if let Some(t) = t_al {
        NTT_TRACE_ALLOC_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NTT_IDLE_ALLOC_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
    }

    let col_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        rows,
        false,
    );
    let row_tables = cache.get_twiddles_refs_par(
        sched,
        spec.par_workers,
        &crt.primes_u64,
        k,
        cols,
        false,
    );
    let psi_fwd: Vec<PsiParams> = (0..k)
        .map(|pi| {
            build_psi_params(
                crt.primes_u64[pi],
                col_tables[pi].g_root,
                spec.n,
                false,
                false,
            )
        })
        .collect();
    let tw_tile0_fwd = cache.get_tile0_twiddles_par(
        kernels,
        sched,
        spec.par_workers,
        &crt.primes_u64,
        &col_tables,
        k,
        spec.n,
        rows,
        tile_cols,
        false,
        false,
    );

    let sw2_w = sw2_workers(sched, &spec);
    let chunk_cap = sw2_chunk_cap(cols);
    let row_chunk = (rows / (sw2_w * 8)).clamp(1, chunk_cap);
    let b_base = work_grids[0].as_mut_ptr() as usize;
    let g1_ptr = g1_out.as_mut_ptr() as usize;
    let g2_ptr = g2_out.as_mut_ptr() as usize;
    let a_stride = if (spec.packed || dram_row_fused) && !use_pack_sw1 {
        unpacked_row_bytes
    } else {
        row_bytes
    };

    if dram_row_fused {
        let w_b = work_grids[0].as_mut_ptr() as usize;
        // Packed work: `a1` and `a2` land in the output grids, so one pass over their limbs
        // makes them for all primes. `b` goes with them where it has a work grid a prime;
        // otherwise it is made prime by prime in the one grid it has.
        let sw1_once_packed = pack_sw1 && sweep1_groups_256(&spec, tile_cols);
        let sw1_once_unpacked =
            a_in_out_unpacked && (cols / tile_cols) >= spec.par_workers.max(1);
        let sw1_once = sw1_once_packed || sw1_once_unpacked;
        let pbp_ops = if sw1_once { 1usize } else { 3usize };
        let sw1_nt = u64::from(spec.use_nt && pbp_ops * work_prime_bytes > my_l3_cap);
        let once_bytes = (2 + usize::from(sw1_once_b)) * k * total_grid_bytes;
        let once_nt = u64::from(spec.use_nt && once_bytes > my_l3_cap);
        let out_nt = ROW_SWEEP_STREAMS;
        let l3_inplace_work = !pack_sw1 && (3 * work_prime_bytes <= my_l3_cap);

        if sw1_once_packed {
            let a1_bases: Vec<usize> =
                g1_out.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let a2_bases: Vec<usize> =
                g2_out.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let b_bases: Vec<usize> =
                work_grids.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
            let mut ops: Vec<(&[u64], &[usize], usize)> = vec![
                (a1_limbs, &a1_bases, work_row_bytes),
                (a2_limbs, &a2_bases, work_row_bytes),
            ];
            if sw1_once_b {
                ops.push((b_limbs, &b_bases, work_row_bytes));
            }
            ntt_sweep1_all_primes(kernels, cache, sched, &ops, &spec, tile_cols, &primes, once_nt);
        } else if sw1_once_unpacked {
            debug_assert_eq!(work_row_bytes, row_bytes);
            let a1_bases: Vec<usize> =
                g1_out.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let a2_bases: Vec<usize> =
                g2_out.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let b_bases: Vec<usize> =
                work_grids.iter_mut().map(|g| g.as_mut_ptr() as usize).collect();
            let primes = sweep1_primes(&crt.primes_u64, &col_tables, &psi_fwd, &tw_tile0_fwd);
            let mut ops: Vec<(&[u64], &[usize], usize)> = vec![
                (a1_limbs, &a1_bases, row_bytes),
                (a2_limbs, &a2_bases, row_bytes),
            ];
            if sw1_once_b {
                ops.push((b_limbs, &b_bases, row_bytes));
            }
            ntt_sweep1_all_primes_unpacked(
                kernels,
                cache,
                sched,
                &ops,
                &spec,
                tile_cols,
                &primes,
                once_nt,
            );
        }

        // One row of sweep 2 for prime `pi` where the rows are gathered from packed or work
        // grids: `b`'s row forward, then each of `a1` and `a2` forward, times `b`, back, and
        // into its output grid.
        let pair_row = |sc: &mut NttTlsScratch,
                        pi: usize,
                        r: usize,
                        w_b: usize,
                        w_a1: usize,
                        w_a2: usize| {
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
            let r_off = r * work_row_bytes;
            let out_off = r * row_bytes;
            let mcfg_in = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: work_row_bytes as u64,
                packed: u64::from(pack_sw1),
                use_nt: if sw1_once_packed { once_nt } else { 0 },
            };
            let mcfg_in_b = MoveCfg {
                use_nt: if sw1_once_packed && sw1_once_b {
                    once_nt
                } else {
                    sw1_nt
                },
                ..mcfg_in
            };
            let mcfg_out = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: out_nt,
            };
            unsafe {
                let b_src = (w_b as *const u8).add(r_off);
                if pack_sw1 {
                    (kernels.active.gather)(sc.ra1.as_mut_ptr(), b_src, &mcfg_in_b);
                } else {
                    std::ptr::copy_nonoverlapping(
                        b_src as *const f64,
                        sc.ra1.as_mut_ptr(),
                        cols,
                    );
                }
                call_ntt_fwd(kernels, 
                    sc.ra1.as_mut_ptr(),
                    r_tw_ptr as *const f64,
                    cols as u64,
                    &pr_fwd,
                );
                let b_row_ptr: *const f64 = sc.ra1.as_ptr();

                let a1_src = (w_a1 as *const u8).add(r_off);
                if pack_sw1 {
                    (kernels.active.gather)(sc.rb1.as_mut_ptr(), a1_src, &mcfg_in);
                } else {
                    std::ptr::copy_nonoverlapping(
                        a1_src as *const f64,
                        sc.rb1.as_mut_ptr(),
                        cols,
                    );
                }
                call_ntt_fwd_pmul_inv(
                    kernels,
                    sc.rb1.as_mut_ptr(),
                    r_tw_ptr as *const f64,
                    r_inv_tw_ptr as *const f64,
                    b_row_ptr,
                    cols as u64,
                    &pr_row_inv,
                );
                let dst1 = (*(g1_ptr as *mut AlignedBytes).add(pi))
                    .as_mut_ptr()
                    .add(out_off);
                (kernels.active.scatter)(dst1, sc.rb1.as_ptr(), &mcfg_out);

                let a2_src = (w_a2 as *const u8).add(r_off);
                if pack_sw1 {
                    (kernels.active.gather)(sc.rb1.as_mut_ptr(), a2_src, &mcfg_in);
                } else {
                    std::ptr::copy_nonoverlapping(
                        a2_src as *const f64,
                        sc.rb1.as_mut_ptr(),
                        cols,
                    );
                }
                call_ntt_fwd_pmul_inv(
                    kernels,
                    sc.rb1.as_mut_ptr(),
                    r_tw_ptr as *const f64,
                    r_inv_tw_ptr as *const f64,
                    b_row_ptr,
                    cols as u64,
                    &pr_row_inv,
                );
                let dst2 = (*(g2_ptr as *mut AlignedBytes).add(pi))
                    .as_mut_ptr()
                    .add(out_off);
                (kernels.active.scatter)(dst2, sc.rb1.as_ptr(), &mcfg_out);
            }
        };
        let pair_scratch = || {
            let mut lease = ScratchLease::acquire();
            let sc = lease.get_mut();
            resize_f64_buf(&mut sc.ra1, cols);
            resize_f64_buf(&mut sc.rb1, cols);
            lease
        };

        // `b` made for every prime: sweep 2 is one loop over all the primes' rows. A loop a
        // prime ends with the workers waiting for its last chunks and starts with their waking.
        if sw1_once_b {
            let t_sw2 = std::time::Instant::now();
            let bases: Vec<(usize, usize, usize)> = (0..k)
                .map(|pi| {
                    (
                        work_grids[pi].as_mut_ptr() as usize,
                        g1_out[pi].as_mut_ptr() as usize,
                        g2_out[pi].as_mut_ptr() as usize,
                    )
                })
                .collect();
            let all_chunk = (k * rows / (sw2_w * 8)).clamp(1, chunk_cap);
            if !pack_sw1 && a_in_out_unpacked {
                sched.par_for_leaf_max(k * rows, all_chunk, sw2_w, |task| {
                    let (pi, r) = (task / rows, task % rows);
                    let (w_b, w_a1, w_a2) = bases[pi];
                    let r_tab = row_tables[pi];
                    let pr_fwd = r_tab.pr_fwd;
                    let pr_row_inv = r_tab.pr_inv;
                    let r_tw_ptr = r_tab.tw.as_ptr() as usize;
                    let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
                    let r_off = r * work_row_bytes;
                    unsafe {
                        let b_r = (w_b as *mut u8).add(r_off) as *mut f64;
                        let a1_r = (w_a1 as *mut u8).add(r_off) as *mut f64;
                        let a2_r = (w_a2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(
                            kernels,
                            b_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        call_ntt_fwd_pmul_inv(
                            kernels,
                            a1_r,
                            r_tw_ptr as *const f64,
                            r_inv_tw_ptr as *const f64,
                            b_r as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        call_ntt_fwd_pmul_inv(
                            kernels,
                            a2_r,
                            r_tw_ptr as *const f64,
                            r_inv_tw_ptr as *const f64,
                            b_r as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                    }
                });
            } else {
                sched.par_for_init_max(k * rows, all_chunk, sw2_w, pair_scratch, |lease, task| {
                    let (pi, r) = (task / rows, task % rows);
                    let (w_b, w_a1, w_a2) = bases[pi];
                    pair_row(lease.get_mut(), pi, r, w_b, w_a1, w_a2);
                });
            }
            for g in g1_out.iter_mut().chain(g2_out.iter_mut()) {
                unsafe {
                    g.set_len(total_grid_bytes);
                }
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }

        for pi in 0..(if sw1_once_b { 0 } else { k }) {
            let w_a1 = if pack_sw1 || a_in_out_unpacked {
                g1_out[pi].as_mut_ptr() as usize
            } else {
                work_grids[1].as_mut_ptr() as usize
            };
            let w_a2 = if pack_sw1 || a_in_out_unpacked {
                g2_out[pi].as_mut_ptr() as usize
            } else {
                work_grids[2].as_mut_ptr() as usize
            };
            let p_u64 = crt.primes_u64[pi];
            let ops_all = [
                (b_limbs, w_b, work_row_bytes),
                (a1_limbs, w_a1, work_row_bytes),
                (a2_limbs, w_a2, work_row_bytes),
            ];
            ntt_sweep1_multi_prime_ext(
                kernels,
                sched,
                if sw1_once { &ops_all[..1] } else { &ops_all[..] },
                &spec,
                tile_cols,
                p_u64,
                col_tables[pi],
                psi_fwd[pi],
                &tw_tile0_fwd[pi],
                sw1_nt,
                pack_sw1,
            );

            let t_sw2 = std::time::Instant::now();
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;

            if l3_inplace_work {
                sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                    let r_off = r * work_row_bytes;
                    let out_off = r * row_bytes;
                    let mcfg_out = MoveCfg {
                        rows: 1,
                        cols: cols as u64,
                        outer_stride_bytes: row_bytes as u64,
                        packed: u64::from(spec.packed),
                        use_nt: out_nt,
                    };
                    unsafe {
                        let b_r = (w_b as *mut u8).add(r_off) as *mut f64;
                        let a1_r = (w_a1 as *mut u8).add(r_off) as *mut f64;
                        let a2_r = (w_a2 as *mut u8).add(r_off) as *mut f64;
                        call_ntt_fwd(
                            kernels,
                            b_r,
                            r_tw_ptr as *const f64,
                            cols as u64,
                            &pr_fwd,
                        );
                        call_ntt_fwd_pmul_inv(
                            kernels,
                            a1_r,
                            r_tw_ptr as *const f64,
                            r_inv_tw_ptr as *const f64,
                            b_r as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        if !a_in_out_unpacked {
                            let dst1 = (*(g1_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(out_off);
                            (kernels.active.scatter)(dst1, a1_r as *const f64, &mcfg_out);
                        }

                        call_ntt_fwd_pmul_inv(
                            kernels,
                            a2_r,
                            r_tw_ptr as *const f64,
                            r_inv_tw_ptr as *const f64,
                            b_r as *const f64,
                            cols as u64,
                            &pr_row_inv,
                        );
                        if !a_in_out_unpacked {
                            let dst2 = (*(g2_ptr as *mut AlignedBytes).add(pi))
                                .as_mut_ptr()
                                .add(out_off);
                            (kernels.active.scatter)(dst2, a2_r as *const f64, &mcfg_out);
                        }
                    }
                });
            } else {
                sched.par_for_init_max(rows, row_chunk, sw2_w, pair_scratch, |lease, r| {
                    pair_row(lease.get_mut(), pi, r, w_b, w_a1, w_a2);
                });
            }
            unsafe {
                g1_out[pi].set_len(total_grid_bytes);
                g2_out[pi].set_len(total_grid_bytes);
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }
    } else {
        for pi in 0..k {
            let p_u64 = crt.primes_u64[pi];
            let r_tab = row_tables[pi];
            let pr_fwd = r_tab.pr_fwd;
            let pr_row_inv = r_tab.pr_inv;
            let r_tw_ptr = r_tab.tw.as_ptr() as usize;
            let r_inv_tw_ptr = r_tab.inv_tw.as_ptr() as usize;
            let mcfg_out = MoveCfg {
                rows: 1,
                cols: cols as u64,
                outer_stride_bytes: row_bytes as u64,
                packed: u64::from(spec.packed),
                use_nt: ROW_SWEEP_STREAMS,
            };

            let a1_base = if spec.packed {
                work_grids[1].as_mut_ptr() as usize
            } else {
                g1_out[pi].as_mut_ptr() as usize
            };
            ntt_sweep1_multi_prime_unpacked(
                kernels,
                sched,
                &[
                    (b_limbs, b_base, unpacked_row_bytes),
                    (a1_limbs, a1_base, a_stride),
                ],
                &spec,
                tile_cols,
                p_u64,
                col_tables[pi],
                psi_fwd[pi],
                &tw_tile0_fwd[pi],
                0,
            );
            let t_sw2_1 = std::time::Instant::now();
            sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                unsafe {
                    let b_r = (b_base as *mut u8).add(r * unpacked_row_bytes) as *mut f64;
                    let a_r = (a1_base as *mut u8).add(r * a_stride) as *mut f64;
                    call_ntt_fwd(kernels, b_r, r_tw_ptr as *const f64, cols as u64, &pr_fwd);
                    call_ntt_fwd_pmul_inv(
                        kernels,
                        a_r,
                        r_tw_ptr as *const f64,
                        r_inv_tw_ptr as *const f64,
                        b_r as *const f64,
                        cols as u64,
                        &pr_row_inv,
                    );
                    if spec.packed {
                        let dst_u8 = (*(g1_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, a_r as *const f64, &mcfg_out);
                    }
                }
            });
            unsafe {
                g1_out[pi].set_len(total_grid_bytes);
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2_1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);

            let a2_base = if spec.packed {
                work_grids[1].as_mut_ptr() as usize
            } else {
                g2_out[pi].as_mut_ptr() as usize
            };
            ntt_sweep1_single_prime_unpacked(
                kernels,
                sched,
                a2_limbs,
                &spec,
                tile_cols,
                p_u64,
                col_tables[pi],
                psi_fwd[pi],
                &tw_tile0_fwd[pi],
                a2_base,
                a_stride,
            );
            let t_sw2_2 = std::time::Instant::now();
            sched.par_for_leaf_max(rows, row_chunk, sw2_w, |r| {
                unsafe {
                    let b_r = (b_base as *const u8).add(r * unpacked_row_bytes) as *const f64;
                    let a_r = (a2_base as *mut u8).add(r * a_stride) as *mut f64;
                    call_ntt_fwd_pmul_inv(
                        kernels,
                        a_r,
                        r_tw_ptr as *const f64,
                        r_inv_tw_ptr as *const f64,
                        b_r,
                        cols as u64,
                        &pr_row_inv,
                    );
                    if spec.packed {
                        let dst_u8 = (*(g2_ptr as *mut AlignedBytes).add(pi))
                            .as_mut_ptr()
                            .add(r * row_bytes);
                        (kernels.active.scatter)(dst_u8, a_r as *const f64, &mcfg_out);
                    }
                }
            });
            unsafe {
                g2_out[pi].set_len(total_grid_bytes);
            }
            NTT_TRACE_SW2_NS.fetch_add(t_sw2_2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            NTT_IDLE_SW2_NS.fetch_add(crate::par::idle_since_mark(), Ordering::Relaxed);
        }
    }

    for g in &mut work_grids {
        unsafe {
            g.set_len(work_prime_bytes);
        }
    }
    cache.put_byte_bufs(work_grids);

    if kernels.count_work {
        let bf_cols = 3u64 * ((spec.n as u64) / 2) * (rows.ilog2() as u64) * (k as u64);
        kernels
            .stats
            .ntt_butterflies
            .fetch_add(bf_cols, Ordering::Relaxed);
    }

    let mut empty_p_grid: Vec<AlignedBytes> = Vec::new();
    if out_bytes.saturating_mul(2) <= ram_transform_budget.max(64 * 1024)
        && let Some(res) = ntt_sweep3_and_crt_multi(
            kernels,
            cache,
            sched,
            &spec,
            crt,
            &col_tables,
            &mut g1_out,
            false,
            a1_out_limbs,
            Some(a2_out_limbs),
            &mut g2_out,
            None,
            &mut empty_p_grid,
        )
    {
        return Some((res.t, res.q.unwrap()));
    }

    let Ok(r1) = ntt_sweep3_and_crt(
        kernels,
        cache,
        sched,
        &spec,
        crt,
        g1_out,
        false,
        a1_out_limbs,
        Some(2),
    ) else {
        cache.put_byte_bufs(g2_out);
        return None;
    };
    let r2 = ntt_sweep3_and_crt(
        kernels,
        cache,
        sched,
        &spec,
        crt,
        g2_out,
        false,
        a2_out_limbs,
        Some(1),
    )
    .ok()?;
    Some((r1, r2))
}
