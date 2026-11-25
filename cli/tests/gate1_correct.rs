//! Gate 1 (Section 15.1): Every kernel meets its contract against its reference under differential
//! fuzzing, every product matches its check word, and test runs give the exact same digits as
//! y-cruncher v0.8.7 Build 9547 (Windows package v0.8.7.9547b).

use std::path::PathBuf;
use std::process::Command;

use engine::EngineConfig;
use engine::ntt::{CrtPlan, largest_primes_for_two_adic};
use engine::plan::OutputFormat;
use engine::radix::{MemoryDigitSink, PACKED19_MAGIC};
use engine::refk::{
    FftCarryCfg, FftSplitCfg, KernelTable, Kernels, MoveCfg, NttColsCfg, NttCrtCfg, NttCrtSegs,
    NttPrime, NttSplitCfg, NttVecCfg, SUPI_ABI_VERSION, build_fft_tables, build_ntt_tables,
    mod_pow_u64,
};
use engine::tier::Profile;
use engine::{OutputSinkFactory, run};

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

struct MemFactory {
    buf: std::sync::Arc<MemoryDigitSink>,
}

impl OutputSinkFactory for MemFactory {
    fn create_full_size(
        &self,
        _full_size_bytes: u64,
    ) -> Result<Box<dyn engine::radix::DigitSink>, String> {
        Ok(Box::new(SharedSink(std::sync::Arc::clone(&self.buf))))
    }
    fn open_existing_exact(
        &self,
        _expected_bytes: u64,
    ) -> Result<Box<dyn engine::radix::DigitSink>, String> {
        Ok(Box::new(SharedSink(std::sync::Arc::clone(&self.buf))))
    }
}

struct SharedSink(std::sync::Arc<MemoryDigitSink>);
impl engine::radix::DigitSink for SharedSink {
    fn write_at(&self, file_offset: u64, bytes: &[u8]) -> Result<(), String> {
        self.0.write_at(file_offset, bytes)
    }
    fn flush(&self) -> Result<(), String> {
        self.0.flush()
    }
}

#[test]
fn gate1_bind_and_abi_checks() {
    // 1. Null table pointer must be a safe no-op.
    unsafe {
        kernels::supi_bind(std::ptr::null_mut(), SUPI_ABI_VERSION);
    }

    // 2. Wrong ABI version must leave all 24 entries untouched.
    let ref_tbl = KernelTable::reference();
    let ref_ptrs = ref_tbl.as_raw_ptrs();
    for bad_abi in [0u64, SUPI_ABI_VERSION - 1, SUPI_ABI_VERSION + 1, 99, u64::MAX] {
        let mut ptrs = ref_ptrs;
        unsafe {
            kernels::supi_bind(ptrs.as_mut_ptr(), bad_abi);
        }
        assert_eq!(ptrs, ref_ptrs, "supi_bind must ignore abi={bad_abi}");
    }

    // 3. Startup bind and self-test on this x86-64-v4 + ADX machine must bind 24/24 with 0 mismatches.
    let (kernels, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(
        rep.bound_mask, 0x00FF_FFFF,
        "Expected all 24 assembly kernels bound on x86-64-v4 + ADX host, got mask=0x{:06x}, untested={:?}, mismatched={:?}",
        rep.bound_mask, rep.untested, rep.mismatched
    );
    assert!(rep.mismatched.is_empty());
    assert!(rep.untested.is_empty());
    assert_eq!(kernels.bound_mask, 0x00FF_FFFF);

    // 4. Partial kernel binding must report any unbound/uncompiled kernel bodies in `rep.untested`
    //    (Section 15.1: "Any kernel body not compiled or not tested on the host platform is reported").
    unsafe extern "C" fn partial_bind(table: *mut usize, abi: u64) {
        let ref_ptrs = KernelTable::reference().as_raw_ptrs();
        unsafe {
            kernels::supi_bind(table, abi);
            if !table.is_null() && abi == SUPI_ABI_VERSION {
                *table.add(2) = ref_ptrs[2]; // leave `mul_1` unbound
                *table.add(21) = ref_ptrs[21]; // leave `scatter` unbound
            }
        }
    }
    let (k_part, rep_part) = Kernels::bind_and_verify(partial_bind);
    assert_eq!(rep_part.untested, vec!["mul_1", "scatter"]);
    assert!(rep_part.mismatched.is_empty());
    assert_eq!(k_part.bound_mask, 0x00FF_FFFF & !((1u32 << 2) | (1u32 << 21)));
}

#[test]
fn gate1_every_module_opens_with_contract_header() {
    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut dirs = vec![
        ws_root.join("engine"),
        ws_root.join("kernels"),
        ws_root.join("platform"),
        ws_root.join("platform").join("windows"),
        ws_root.join("platform").join("linux"),
        ws_root.join("platform").join("macos"),
        ws_root.join("cli"),
    ];
    let mut checked = 0usize;
    for dir in dirs.drain(..) {
        for entry in std::fs::read_dir(&dir).expect("read module dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            if path.file_name().and_then(|s| s.to_str()) == Some("build.rs") {
                continue;
            }
            checked += 1;
            let src = std::fs::read_to_string(&path).expect("read source file");
            assert!(
                src.starts_with("//! Contract:\n") || src.starts_with("//! Contract:\r\n"),
                "Module {path:?} must open with `//! Contract:` (Section 7)"
            );
            let header: String = src
                .lines()
                .take_while(|l| l.starts_with("//!"))
                .collect::<Vec<_>>()
                .join("\n");
            for field in ["Inputs:", "Outputs:", "Data format:", "Work & bytes moved:"] {
                assert!(
                    header.contains(field),
                    "Module {path:?} contract header missing `{field}`"
                );
            }
        }
    }
    assert!(checked >= 21, "Expected at least 21 module files, checked {checked}");
}

#[test]
fn gate1_fuzz_family1_limbs() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x1111_2222_3333_4444);

    let lengths = [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 257,
    ];
    for &n in &lengths {
        for trial in 0..8 {
            let mut a = vec![0u64; n];
            let mut b = vec![0u64; n];
            for i in 0..n {
                a[i] = match trial {
                    0 => u64::MAX,
                    1 => 0,
                    2 if i + 1 == n => 0x8000_0000_0000_0000,
                    _ => rng.next_u64(),
                };
                b[i] = match trial {
                    0 => u64::MAX,
                    1 => 1,
                    2 => u64::MAX,
                    _ => rng.next_u64(),
                };
            }

            // add_n & sub_n (separate & aliased dst==a / dst==b)
            let mut dst_asm = vec![0xDEAD_BEEFu64; n + 2];
            let mut dst_ref = vec![0xDEAD_BEEFu64; n + 2];
            let c_asm = unsafe { (asm.add_n)(dst_asm.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
            let c_ref = unsafe { (rfk.add_n)(dst_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((c_asm, &dst_asm), (c_ref, &dst_ref), "add_n n={n}");

            let mut alias_a = a.clone();
            let ca2 = unsafe { (asm.add_n)(alias_a.as_mut_ptr(), alias_a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((ca2, alias_a.as_slice()), (c_ref, &dst_ref[..n]));

            dst_asm.fill(0xCAFE_BABE);
            dst_ref.fill(0xCAFE_BABE);
            let b_asm = unsafe { (asm.sub_n)(dst_asm.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
            let b_ref = unsafe { (rfk.sub_n)(dst_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((b_asm, &dst_asm), (b_ref, &dst_ref), "sub_n n={n}");

            let mut alias_b = b.clone();
            let ba2 = unsafe { (asm.sub_n)(alias_b.as_mut_ptr(), a.as_ptr(), alias_b.as_ptr(), n as u64) };
            assert_eq!((ba2, alias_b.as_slice()), (b_ref, &dst_ref[..n]));

            // The sum made in place (`add_bignum_into_par`) puts the result on either operand
            // of a subtraction and on the first of an addition, with both kernel tables.
            let mut alias_a = a.clone();
            let ba3 = unsafe { (asm.sub_n)(alias_a.as_mut_ptr(), alias_a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((ba3, alias_a.as_slice()), (b_ref, &dst_ref[..n]), "sub_n on a, n={n}");
            let mut alias_a = a.clone();
            let br3 = unsafe { (rfk.sub_n)(alias_a.as_mut_ptr(), alias_a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((br3, alias_a.as_slice()), (b_ref, &dst_ref[..n]), "reference sub_n on a, n={n}");
            let mut alias_b = b.clone();
            let br4 = unsafe { (rfk.sub_n)(alias_b.as_mut_ptr(), a.as_ptr(), alias_b.as_ptr(), n as u64) };
            assert_eq!((br4, alias_b.as_slice()), (b_ref, &dst_ref[..n]), "reference sub_n on b, n={n}");
            let mut alias_a = a.clone();
            let cr3 = unsafe { (rfk.add_n)(alias_a.as_mut_ptr(), alias_a.as_ptr(), b.as_ptr(), n as u64) };
            let mut sum_ref = vec![0u64; n];
            let c_sum = unsafe { (rfk.add_n)(sum_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64) };
            assert_eq!((cr3, &alias_a), (c_sum, &sum_ref), "reference add_n on a, n={n}");

            // mul_1 & addmul_1
            let m = if trial == 0 { u64::MAX } else { rng.next_u64() };
            dst_asm.fill(0x5555_AAAA);
            dst_ref.fill(0x5555_AAAA);
            let m_asm = unsafe { (asm.mul_1)(dst_asm.as_mut_ptr(), a.as_ptr(), n as u64, m) };
            let m_ref = unsafe { (rfk.mul_1)(dst_ref.as_mut_ptr(), a.as_ptr(), n as u64, m) };
            assert_eq!((m_asm, &dst_asm), (m_ref, &dst_ref), "mul_1 n={n}");

            dst_asm[..n].copy_from_slice(&b);
            dst_ref[..n].copy_from_slice(&b);
            let am_asm = unsafe { (asm.addmul_1)(dst_asm.as_mut_ptr(), a.as_ptr(), n as u64, m) };
            let am_ref = unsafe { (rfk.addmul_1)(dst_ref.as_mut_ptr(), a.as_ptr(), n as u64, m) };
            assert_eq!((am_asm, &dst_asm), (am_ref, &dst_ref), "addmul_1 n={n}");

            if n > 0 && n <= 65 {
                for rows_max in [0u64, 13, u64::MAX] {
                    let mut sq_asm = vec![0x9999u64; 2 * n + 2];
                    let mut sq_ref = vec![0x9999u64; 2 * n + 2];
                    unsafe {
                        (asm.sqr_base)(sq_asm.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                        (rfk.sqr_base)(sq_ref.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                    }
                    assert_eq!(sq_asm, sq_ref, "sqr_base n={n} rows_max={rows_max}");
                }
            }
        }
    }

    for &(na, nb) in &[(1usize, 1usize), (3, 1), (7, 4), (8, 8), (17, 9), (33, 16), (64, 33)] {
        for trial in 0..6 {
            let a: Vec<u64> = (0..na)
                .map(|_| if trial == 0 { u64::MAX } else { rng.next_u64() })
                .collect();
            let b: Vec<u64> = (0..nb)
                .map(|_| if trial == 0 { u64::MAX } else { rng.next_u64() })
                .collect();
            for rows_max in [13u64, 0, u64::MAX] {
                let lens = [na as u64, nb as u64, rows_max];
                let mut p_asm = vec![0x7777u64; na + nb + 2];
                let mut p_ref = vec![0x7777u64; na + nb + 2];
                unsafe {
                    (asm.mul_base)(p_asm.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                    (rfk.mul_base)(p_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                }
                assert_eq!(p_asm, p_ref, "mul_base ({na}, {nb}) rows_max {rows_max}");
            }
        }
    }
}

/// `mul_base` has code of its own for every length of a short row and for every place a long
/// row can enter its loop, and a form in the vector registers for the middle sizes: every pair
/// of lengths up to 40 limbs, both ways round, every length of the shorter operand through the
/// vector form's range, and longer shapes up to 140 by 70, on operands that make the carries
/// run (all ones, powers of two and their neighbours, long runs of ones or zeros) as well as
/// on random ones. All ones in both operands is also the largest a column of the vector form
/// can grow. The product must be the reference's in every limb and nothing may be written
/// past it.
#[test]
fn gate1_fuzz_mul_base_shapes() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x6D75_6C5F_6261_7365);

    fn operand(rng: &mut XorShift64, n: usize, pattern: u32) -> Vec<u64> {
        let mut v = vec![0u64; n];
        match pattern {
            0 => v.iter_mut().for_each(|x| *x = rng.next_u64()),
            1 => v.fill(u64::MAX),
            2 => {
                // one bit
                let bit = (rng.next_u64() % (64 * n as u64)) as usize;
                v[bit / 64] = 1u64 << (bit % 64);
            }
            3 => {
                // 2^k - 1: a run of ones from the bottom
                let bits = 1 + (rng.next_u64() % (64 * n as u64)) as usize;
                for (i, x) in v.iter_mut().enumerate() {
                    let lo = 64 * i;
                    *x = if bits >= lo + 64 { u64::MAX } else if bits > lo { (1u64 << (bits - lo)) - 1 } else { 0 };
                }
            }
            4 => {
                // 2^k + 1
                let bit = 1 + (rng.next_u64() % (64 * n as u64 - 1)) as usize;
                v[bit / 64] |= 1u64 << (bit % 64);
                v[0] |= 1;
            }
            5 => {
                // runs of ones and zeros of random lengths
                let mut pos = 0usize;
                let mut one = rng.next_u64() & 1 == 1;
                while pos < 64 * n {
                    let len = 1 + (rng.next_u64() % 97) as usize;
                    if one {
                        for b in pos..(pos + len).min(64 * n) {
                            v[b / 64] |= 1u64 << (b % 64);
                        }
                    }
                    pos += len;
                    one = !one;
                }
            }
            6 => {
                // all ones but for one limb
                v.fill(u64::MAX);
                let i = (rng.next_u64() % n as u64) as usize;
                v[i] = rng.next_u64();
            }
            7 => {
                // random with whole limbs of zeros and of ones
                for x in v.iter_mut() {
                    *x = match rng.next_u64() % 4 {
                        0 => 0,
                        1 => u64::MAX,
                        _ => rng.next_u64(),
                    };
                }
            }
            _ => {} // zero
        }
        v
    }

    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for na in 1..=40usize {
        for nb in 1..=na {
            shapes.push((na, nb));
        }
    }
    for &na in &[41usize, 47, 48, 49, 56, 57, 63, 64, 65, 66, 70, 71, 72, 96, 127, 128, 129, 140] {
        for &nb in &[1usize, 2, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 41, 47, 48, 56, 57, 63, 64, 65, 66, 70] {
            if nb <= na {
                shapes.push((na, nb));
            }
        }
    }
    // The vector form (a shorter operand above the caller's `rows_max` and of up to 64 limbs,
    // under a longer one of up to 128) works on blocks of seven limbs, with as many registers
    // of sums as the shorter operand
    // has vectors of digits and a last store of one to seven bytes: every length of the
    // shorter operand, under longer ones that give every remainder of the lengths and of
    // their sum modulo seven, and at the ends of the range on both sides.
    for nb in 13..=66usize {
        for na in [nb, nb + 1, nb + 2, nb + 3, nb + 4, nb + 5, nb + 6, 2 * nb, 127, 128, 129] {
            if na >= nb && !shapes.contains(&(na, nb)) {
                shapes.push((na, nb));
            }
        }
    }
    let mut cases = 0u64;
    for &(na, nb) in &shapes {
        let rounds = if na <= 40 { 12 } else { 20 };
        for round in 0..rounds {
            let (pa, pb) = match round {
                0 => (0, 0),
                1 => (1, 1),
                2 => (1, 0),
                3 => (0, 1),
                4 => (3, 3),
                5 => (2, 2),
                6 => (3, 4),
                7 => (5, 5),
                8 => (6, 1),
                9 => (7, 7),
                10 => (8, 0),
                11 => (1, 3),
                _ => ((rng.next_u64() % 8) as u32, (rng.next_u64() % 8) as u32),
            };
            let a = operand(&mut rng, na, pa);
            let b = operand(&mut rng, nb, pb);
            // The crossover is the caller's: the usual one, all rows, no rows that the body can
            // spare, and the two settings on either side of this shape's shorter operand. The
            // result is the same for every one of them.
            let guard = 0x5AFE_C0DE_5AFE_C0DEu64;
            let mut p_ref = vec![guard; na + nb + 4];
            let lens_ref = [na as u64, nb as u64, 0];
            unsafe {
                (rfk.mul_base)(p_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens_ref);
            }
            for rows_max in [13u64, u64::MAX, 0, nb as u64, nb as u64 - 1] {
                for swap in [false, true] {
                    let (x, y) = if swap { (&b, &a) } else { (&a, &b) };
                    let lens = [x.len() as u64, y.len() as u64, rows_max];
                    let mut p_asm = vec![guard; na + nb + 4];
                    unsafe {
                        (asm.mul_base)(p_asm.as_mut_ptr(), x.as_ptr(), y.as_ptr(), &lens);
                    }
                    assert_eq!(
                        p_asm, p_ref,
                        "mul_base ({}, {}) rows_max {rows_max} patterns ({pa}, {pb}) round {round}",
                        x.len(),
                        y.len()
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 200_000);
}

#[test]
fn gate1_fuzz_family2_fft() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x5555_6666_7777_8888);

    for &(n_complex, coeff_bits) in &[
        (4usize, 18u64),
        (8, 16),
        (16, 18),
        (32, 18),
        (64, 19),
        (128, 18),
        (256, 18),
        (512, 17),
        (1024, 16),
    ] {
        let (weights, inv_weights, tw, inv_tw) = build_fft_tables(n_complex);
        let max_in_bits = (n_complex * (coeff_bits as usize)).saturating_sub(8);
        let in_limbs = (max_in_bits / 64).max(1);
        let out_limbs = 2 * in_limbs + 2;

        let a: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
        let b: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
        let c: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
        let d: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();

        let scfg = FftSplitCfg {
            in_limbs: in_limbs as u64,
            n_complex: n_complex as u64,
            coeff_bits,
        };

        // 1. Single product via FFT chain vs reference mul_base
        let mut fa = vec![0.0f64; 2 * n_complex];
        let mut fb = vec![0.0f64; 2 * n_complex];
        let mut fp = vec![0.0f64; 2 * n_complex];
        let mut out_asm = vec![0u64; out_limbs];
        let mut ccfg_asm = FftCarryCfg {
            out_limbs: out_limbs as u64,
            n_complex: n_complex as u64,
            coeff_bits,
            top_carry: 0,
            max_err: 0.0,
            check_word: 0,
        };
        unsafe {
            (asm.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &scfg);
            (asm.fft_split)(fb.as_mut_ptr(), b.as_ptr(), weights.as_ptr(), &scfg);
            (asm.fft_fwd)(fa.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
            (asm.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
            (asm.fft_pmul)(fp.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
            (asm.fft_inv)(fp.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
            (asm.fft_carry)(
                out_asm.as_mut_ptr(),
                fp.as_ptr(),
                inv_weights.as_ptr(),
                &mut ccfg_asm,
            );
        }

        let mut out_exact = vec![0u64; out_limbs];
        let lens = [in_limbs as u64, in_limbs as u64, 0];
        unsafe {
            (rfk.mul_base)(out_exact.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
        }
        assert_eq!(out_asm, out_exact, "FFT product mismatch at N={n_complex}");
        assert!(ccfg_asm.max_err < 0.25, "FFT max_err={} too high", ccfg_asm.max_err);
        let mut dummy = [0u64; 3];
        let exp_w0 = unsafe { (rfk.check_n)(out_exact.as_ptr(), out_limbs as u64, &mut dummy, 0) };
        assert_eq!(ccfg_asm.check_word, exp_w0);

        // 2. Fused pmuladd (A*B + C*D and |A*B - C*D|) vs reference FFT chain
        for sub_second in [0u64, 1u64] {
            let mut fc = vec![0.0f64; 2 * n_complex];
            let mut fd = vec![0.0f64; 2 * n_complex];
            unsafe {
                (asm.fft_split)(fc.as_mut_ptr(), c.as_ptr(), weights.as_ptr(), &scfg);
                (asm.fft_split)(fd.as_mut_ptr(), d.as_ptr(), weights.as_ptr(), &scfg);
                (asm.fft_fwd)(fc.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
                (asm.fft_fwd)(fd.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
            }
            let ptrs = [fa.as_ptr(), fb.as_ptr(), fc.as_ptr(), fd.as_ptr()];
            let mut fsum_asm = vec![0.0f64; 2 * n_complex];
            let mut fsum_ref = vec![0.0f64; 2 * n_complex];
            unsafe {
                (asm.fft_pmuladd)(fsum_asm.as_mut_ptr(), &ptrs, n_complex as u64, sub_second);
                (rfk.fft_pmuladd)(fsum_ref.as_mut_ptr(), &ptrs, n_complex as u64, sub_second);
                (asm.fft_inv)(fsum_asm.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
                (rfk.fft_inv)(fsum_ref.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
            }
            let mut o_asm = vec![0u64; out_limbs];
            let mut o_ref = vec![0u64; out_limbs];
            let mut ca = FftCarryCfg {
                out_limbs: out_limbs as u64,
                n_complex: n_complex as u64,
                coeff_bits,
                top_carry: 0,
                max_err: 0.0,
                check_word: 0,
            };
            let mut cr = ca;
            unsafe {
                (asm.fft_carry)(o_asm.as_mut_ptr(), fsum_asm.as_ptr(), inv_weights.as_ptr(), &mut ca);
                (rfk.fft_carry)(o_ref.as_mut_ptr(), fsum_ref.as_ptr(), inv_weights.as_ptr(), &mut cr);
            }
            if sub_second == 0 {
                assert_eq!(o_asm, o_ref, "fft_pmuladd add N={n_complex}");
                assert_eq!(ca.check_word, cr.check_word);
            }
        }
    }
}

/// `fft_carry` on digit streams given directly (unit weights), assembly against reference:
/// carries and borrows that run through many digits, digits at the edges of the ranges the
/// vector paths take, digits as wide as a double holds exactly, fractions to round, and outputs
/// shorter than, as long as, and longer than the digits.
#[test]
fn gate1_fuzz_fft_carry_digit_streams() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x0DD5_7EA4_1234_ABCD);
    let mut cases = 0usize;
    let mut run_len = 0usize;

    fn signed(rng: &mut XorShift64, bound: i64) -> i64 {
        let r = (rng.next_u64() % ((2 * bound + 1) as u64)) as i64;
        r - bound
    }

    for &n in &[1usize, 3, 8, 9, 16, 24, 40, 64, 136, 256, 1024] {
        let unit: Vec<f64> = (0..n).flat_map(|_| [1.0f64, 0.0]).collect();
        for &b in &[8u64, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 30, 31, 32] {
            let two_n = 2 * n;
            let top = 1i64 << b;
            // what a product of balanced digits can reach, and never past 2^52
            let wide = 1i64 << (2 * b + (two_n.ilog2() as u64) + 1).min(52);
            for pattern in 0..18u32 {
                let mut digits: Vec<i64> = match pattern {
                    0 | 13 => (0..two_n).map(|_| signed(&mut rng, wide)).collect(),
                    // digits in range and one carry or borrow that has a run of digits at the
                    // end of their range to go through: runs of every length up to seventy
                    // (some settle in the vector unit's extra rounds, the longer ones do not),
                    // anywhere among the digits, the last ones included
                    14..=17 => {
                        let mut v: Vec<i64> =
                            (0..two_n).map(|_| (rng.next_u64() % (top as u64)) as i64).collect();
                        run_len = run_len % 70 + 1;
                        let len = run_len.min(two_n.saturating_sub(1)).max(1);
                        let at = if pattern == 17 {
                            two_n - len - usize::from(two_n > len)
                        } else {
                            (rng.next_u64() as usize) % (two_n - len + 1)
                        };
                        let carry = pattern % 2 == 0;
                        for d in v[at..at + len].iter_mut() {
                            *d = if carry { top - 1 } else { 0 };
                        }
                        if at > 0 {
                            v[at - 1] = if carry { top } else { -1 };
                        } else {
                            v[0] = if carry { top } else { -1 };
                        }
                        v
                    }
                    1 => (0..two_n).map(|_| signed(&mut rng, top * 8)).collect(),
                    // a carry in at the bottom runs through every digit
                    2 => (0..two_n).map(|i| if i == 0 { top } else { top - 1 }).collect(),
                    // a borrow at the bottom runs through every digit
                    3 => (0..two_n).map(|i| if i == 0 { -1 } else { 0 }).collect(),
                    4 => (0..two_n).map(|_| (rng.next_u64() % (top as u64)) as i64).collect(),
                    // digits already in range but for a few at -1 and 2^b, with runs of zeros
                    // and of 2^b - 1 for their carries to run through
                    5 | 6 => {
                        let mut v: Vec<i64> = (0..two_n)
                            .map(|_| match rng.next_u64() % 8 {
                                0 | 1 => 0,
                                2 | 3 => top - 1,
                                4 => 1,
                                5 => top - 2,
                                _ => (rng.next_u64() % (top as u64)) as i64,
                            })
                            .collect();
                        let hits = if pattern == 5 { 1 + two_n / 16 } else { 1 + two_n / 3 };
                        for _ in 0..hits {
                            let at = (rng.next_u64() as usize) % two_n;
                            v[at] = if rng.next_u64() & 1 == 0 { -1 } else { top };
                        }
                        v
                    }
                    7 => (0..two_n)
                        .map(|_| if rng.next_u64() & 1 == 0 { wide } else { -wide })
                        .collect(),
                    8 => (0..two_n).map(|i| if i % 2 == 0 { wide } else { -wide }).collect(),
                    // mostly zero
                    9 => (0..two_n)
                        .map(|_| if rng.next_u64() % 9 == 0 { signed(&mut rng, wide) } else { 0 })
                        .collect(),
                    // low parts at the ends of the digit range under random high parts, so that
                    // the partly carried digits land on -1, 0, 2^b - 1 and 2^b
                    10 => (0..two_n)
                        .map(|_| {
                            let lo = match rng.next_u64() % 6 {
                                0 => 0,
                                1 => 1,
                                2 => top - 1,
                                3 => top - 2,
                                4 => -1,
                                _ => (rng.next_u64() % (top as u64)) as i64,
                            };
                            let hi = signed(&mut rng, (wide >> b).max(2));
                            (hi * top + lo).clamp(-wide, wide)
                        })
                        .collect(),
                    // every digit as wide as a double holds exactly
                    11 => (0..two_n).map(|_| signed(&mut rng, 1i64 << 52)).collect(),
                    _ => (0..two_n)
                        .map(|_| signed(&mut rng, (1i64 << (3 * b).min(52)).min(wide.max(top))))
                        .collect(),
                };
                if pattern == 6 && two_n >= 4 {
                    // the last digits too
                    digits[two_n - 1] = top;
                    digits[two_n - 2] = top - 1;
                }
                let mut buf = vec![0.0f64; two_n];
                for j in 0..n {
                    buf[2 * j] = digits[j] as f64;
                    buf[2 * j + 1] = digits[j + n] as f64;
                }
                if pattern == 13 {
                    // fractions to round away, where the digit leaves room for them
                    for v in buf.iter_mut() {
                        if v.abs() < (1u64 << 48) as f64 {
                            *v += ((rng.next_u64() % 401) as f64 - 200.0) / 1024.0;
                        }
                    }
                }
                let exact = (two_n * (b as usize)).div_ceil(64);
                for &out_limbs in &[
                    exact,
                    exact + 3,
                    exact + 1,
                    exact.saturating_sub(1).max(1),
                    (exact / 2).max(1),
                    1,
                ] {
                    let mut o_asm = vec![0x5A5A_5A5A_5A5A_5A5Au64; out_limbs + 2];
                    let mut o_ref = vec![0x5A5A_5A5A_5A5A_5A5Au64; out_limbs + 2];
                    let mut ca = FftCarryCfg {
                        out_limbs: out_limbs as u64,
                        n_complex: n as u64,
                        coeff_bits: b,
                        top_carry: 0,
                        max_err: 0.0,
                        check_word: 0,
                    };
                    let mut cr = ca;
                    unsafe {
                        (asm.fft_carry)(o_asm.as_mut_ptr(), buf.as_ptr(), unit.as_ptr(), &mut ca);
                        (rfk.fft_carry)(o_ref.as_mut_ptr(), buf.as_ptr(), unit.as_ptr(), &mut cr);
                    }
                    let what = format!("n={n} b={b} pattern={pattern} out_limbs={out_limbs}");
                    assert_eq!(o_asm, o_ref, "limbs, {what}");
                    assert_eq!(ca.top_carry, cr.top_carry, "top carry, {what}");
                    assert_eq!(ca.check_word, cr.check_word, "check word, {what}");
                    assert_eq!(ca.max_err.to_bits(), cr.max_err.to_bits(), "max_err, {what}");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 20_000);
}

#[test]
fn gate1_fuzz_family3_ntt() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x9999_AAAA_BBBB_CCCC);

    let primes = largest_primes_for_two_adic(40, 18);
    let p0 = primes[0];
    let p0_f = p0 as f64;
    let inv_p0 = 1.0 / p0_f;

    // 1. ntt_split (contiguous & strided across non-61-multiple coeff_bits, full + boundary in_limbs)
    let limbs: Vec<u64> = (0..800).map(|_| rng.next_u64()).collect();
    for &in_l in &[800u64, 14u64] {
        for &cb in &[
            17u64, 31, 32, 47, 59, 60, 62, 64, 75, 92, 96, 108, 127, 128, 129, 136, 144, 152, 160,
            161, 176, 188, 191, 192, 193, 200, 208, 209, 240, 256, 288, 300, 320, 336, 384,
        ] {
            for &(count, stride, cols) in &[
                (1u64, 0u64, 0u64),
                (7, 0, 0),
                (8, 0, 0),
                (9, 0, 0),
                (33, 0, 0),
                (37, 0, 0),
                (45, 0, 0),
                (65, 0, 0),
                (96, 0, 0),
                (24, 16, 8),
                (40, 48, 40),
                (96, 48, 32),
                (21, 13, 7),
            ] {
                let cfg = NttSplitCfg {
                    in_limbs: in_l,
                    start_coeff: 3,
                    count,
                    coeff_bits: cb,
                    col_stride: stride,
                    cols_per_row: cols,
                    p: p0_f,
                    inv_p: inv_p0,
                };
                let mut d_asm = vec![-1.0f64; count as usize + 2];
                let mut d_ref = vec![-1.0f64; count as usize + 2];
                unsafe {
                    (asm.ntt_split)(d_asm.as_mut_ptr(), limbs.as_ptr(), &cfg);
                    (rfk.ntt_split)(d_ref.as_mut_ptr(), limbs.as_ptr(), &cfg);
                }
                assert_eq!(
                    d_asm, d_ref,
                    "ntt_split in_l={in_l} cb={cb} count={count} stride={stride}"
                );
            }
        }
    }

    // 1b. ntt_split planar 48-bit extract (p == 0.0) + planar Horner reduce (in_limbs == u64::MAX)
    for &in_l in &[800u64, 14u64] {
        for &cb in &[
            47u64, 48, 49, 60, 64, 75, 96, 108, 128, 136, 144, 152, 160, 161, 176, 191, 192, 193,
            200, 208, 209, 240, 256, 300, 320, 336, 384,
        ] {
            let num_chunks = (cb as usize).div_ceil(48);
            for &(count, stride, cols) in &[
                (1u64, 0u64, 0u64),
                (7, 0, 0),
                (8, 0, 0),
                (9, 0, 0),
                (32, 0, 0),
                (33, 0, 0),
                (37, 0, 0),
                (64, 0, 0),
                (73, 0, 0),
                (96, 0, 0),
                (24, 16, 8),
                (40, 48, 40),
                (64, 48, 32),
                (96, 48, 32),
                (21, 13, 7),
            ] {
                let cnt = count as usize;
                let stride_el = engine::refk::ntt_planar_stride(cnt);
                let ext_cfg = NttSplitCfg {
                    in_limbs: in_l,
                    start_coeff: 3,
                    count,
                    coeff_bits: cb,
                    col_stride: stride,
                    cols_per_row: cols,
                    p: 0.0,
                    inv_p: 0.0,
                };
                let mut pl_asm = vec![-3.0f64; num_chunks * stride_el + 2];
                let mut pl_ref = vec![-3.0f64; num_chunks * stride_el + 2];
                unsafe {
                    (asm.ntt_split)(pl_asm.as_mut_ptr(), limbs.as_ptr(), &ext_cfg);
                    (rfk.ntt_split)(pl_ref.as_mut_ptr(), limbs.as_ptr(), &ext_cfg);
                }
                assert_eq!(
                    pl_asm, pl_ref,
                    "ntt_split extract in_l={in_l} cb={cb} count={count} stride={stride}"
                );

                let red_cfg = NttSplitCfg {
                    in_limbs: u64::MAX,
                    start_coeff: 0,
                    count,
                    coeff_bits: cb,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: p0_f,
                    inv_p: inv_p0,
                };
                let mut r_asm = vec![-4.0f64; cnt + 2];
                let mut r_ref = vec![-4.0f64; cnt + 2];
                let mut d_direct = vec![-4.0f64; cnt + 2];
                let direct_cfg = NttSplitCfg {
                    p: p0_f,
                    inv_p: inv_p0,
                    ..ext_cfg
                };
                unsafe {
                    (asm.ntt_split)(r_asm.as_mut_ptr(), pl_asm.as_ptr() as *const u64, &red_cfg);
                    (rfk.ntt_split)(r_ref.as_mut_ptr(), pl_ref.as_ptr() as *const u64, &red_cfg);
                    (asm.ntt_split)(d_direct.as_mut_ptr(), limbs.as_ptr(), &direct_cfg);
                }
                assert_eq!(
                    r_asm, r_ref,
                    "ntt_split reduce in_l={in_l} cb={cb} count={count} stride={stride}"
                );
                assert_eq!(
                    r_asm, d_direct,
                    "ntt_split extract+reduce vs direct in_l={in_l} cb={cb} count={count} stride={stride}"
                );
            }
        }
    }

    // 2. ntt_fwd & ntt_inv across every power of 2 (exercising top Radix-16/4/2 + sub_n=4096 Radix-16),
    //    up to 2^18: above 2^11 the forward passes share their reductions in a pattern that
    //    changes with the number of stages, one to seven here
    for &n in &[
        1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144,
    ] {
        for &negacyclic in &[false, true] {
            let (tw, inv_tw) = build_ntt_tables(p0, n, negacyclic);
            let n_inv = mod_pow_u64(n as u64, p0 - 2, p0);
            let pr_fwd = NttPrime::new(p0, 1);
            let pr_inv = NttPrime::new(p0, n_inv);

            let orig: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
            let mut b_asm = orig.clone();
            let mut b_ref = orig.clone();
            unsafe {
                (asm.ntt_fwd)(b_asm.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
                (rfk.ntt_fwd)(b_ref.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
            }
            assert_eq!(b_asm, b_ref, "ntt_fwd n={n} neg={negacyclic}");

            if n == 32768 {
                let mut ns_asm = b_asm.clone();
                let mut ns_ref = b_ref.clone();
                unsafe {
                    (asm.ntt_inv)(ns_asm.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_fwd);
                    (rfk.ntt_inv)(ns_ref.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_fwd);
                }
                assert_eq!(ns_asm, ns_ref, "ntt_inv scale=1 n={n} neg={negacyclic}");
            }

            unsafe {
                (asm.ntt_inv)(b_asm.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_inv);
                (rfk.ntt_inv)(b_ref.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_inv);
            }
            assert_eq!(b_asm, b_ref, "ntt_inv n={n} neg={negacyclic}");
            assert_eq!(b_asm, orig, "ntt roundtrip n={n} neg={negacyclic}");

            // Extremal all-(p-1) inputs into both ntt_fwd and ntt_inv directly!
            let mut ext_asm = vec![(p0 - 1) as f64; n];
            let mut ext_ref = ext_asm.clone();
            unsafe {
                (asm.ntt_fwd)(ext_asm.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
                (rfk.ntt_fwd)(ext_ref.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
            }
            assert_eq!(ext_asm, ext_ref, "ntt_fwd extremal n={n} neg={negacyclic}");
            let mut inv_asm = vec![(p0 - 1) as f64; n];
            let mut inv_ref = inv_asm.clone();
            unsafe {
                (asm.ntt_inv)(inv_asm.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_inv);
                (rfk.ntt_inv)(inv_ref.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_inv);
            }
            assert_eq!(inv_asm, inv_ref, "ntt_inv extremal n={n} neg={negacyclic}");
        }
    }

    // 3. ntt_fwd_cols & ntt_inv_cols across strip-outer Radix-16 & odd column tails
    for &rows in &[1usize, 2, 4, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096] {
        let (tw, inv_tw) = build_ntt_tables(p0, rows, true);
        let r_inv = mod_pow_u64(rows as u64, p0 - 2, p0);
        let pr_fwd = NttPrime::new(p0, 1);
        let pr_inv = NttPrime::new(p0, r_inv);
        let col_list: &[usize] = if rows >= 256 {
            &[1, 8, 9, 16]
        } else {
            &[1, 3, 7, 8, 9, 16, 17, 33]
        };
        for &cols in col_list {
            let orig: Vec<f64> = (0..(rows * cols))
                .map(|_| (rng.next_u64() % p0) as f64)
                .collect();
            let mut b_asm = orig.clone();
            let mut b_ref = orig.clone();
            let mut scratch = vec![0.0f64; rows * 8 + 16];
            let aligned_ptr = {
                let addr = scratch.as_mut_ptr() as usize;
                ((addr + 63) & !63) as *mut f64
            };
            let cfg = NttColsCfg {
                rows: rows as u64,
                cols: cols as u64,
                l1_sub_rows: 256,
                l2_sub_rows: 4096,
                scratch: aligned_ptr,
            };
            unsafe {
                (asm.ntt_fwd_cols)(b_asm.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
                (rfk.ntt_fwd_cols)(b_ref.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
            }
            assert_eq!(b_asm, b_ref, "ntt_fwd_cols rows={rows} cols={cols}");

            unsafe {
                (asm.ntt_inv_cols)(b_asm.as_mut_ptr(), inv_tw.as_ptr(), &cfg, &pr_inv);
                (rfk.ntt_inv_cols)(b_ref.as_mut_ptr(), inv_tw.as_ptr(), &cfg, &pr_inv);
            }
            assert_eq!(b_asm, b_ref, "ntt_inv_cols rows={rows} cols={cols}");
            assert_eq!(b_asm, orig);

            let mut inv_asm = vec![(p0 - 1) as f64; rows * cols];
            let mut inv_ref = inv_asm.clone();
            unsafe {
                (asm.ntt_inv_cols)(inv_asm.as_mut_ptr(), inv_tw.as_ptr(), &cfg, &pr_inv);
                (rfk.ntt_inv_cols)(inv_ref.as_mut_ptr(), inv_tw.as_ptr(), &cfg, &pr_inv);
            }
            assert_eq!(inv_asm, inv_ref, "ntt_inv_cols extremal rows={rows} cols={cols}");
        }
    }

    // 4. ntt_pmul & ntt_pmuladd
    for &n in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129] {
        let a1: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
        let b1: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
        let a2: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
        let b2: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
        let mut o_asm = vec![-2.0f64; n + 1];
        let mut o_ref = vec![-2.0f64; n + 1];
        let vcfg = NttVecCfg {
            n: n as u64,
            p: p0_f,
            inv_p: inv_p0,
            sub_second: 0,
        };
        unsafe {
            (asm.ntt_pmul)(o_asm.as_mut_ptr(), a1.as_ptr(), b1.as_ptr(), &vcfg);
            (rfk.ntt_pmul)(o_ref.as_mut_ptr(), a1.as_ptr(), b1.as_ptr(), &vcfg);
        }
        assert_eq!(o_asm, o_ref, "ntt_pmul n={n}");

        for sub_second in [0u64, 1u64] {
            let vcfg2 = NttVecCfg { sub_second, ..vcfg };
            let ptrs = [a1.as_ptr(), b1.as_ptr(), a2.as_ptr(), b2.as_ptr()];
            unsafe {
                (asm.ntt_pmuladd)(o_asm.as_mut_ptr(), &ptrs, &vcfg2);
                (rfk.ntt_pmuladd)(o_ref.as_mut_ptr(), &ptrs, &vcfg2);
            }
            assert_eq!(o_asm, o_ref, "ntt_pmuladd n={n} sub={sub_second}");

            // Accumulate mode (b1 = null): separate dst and in-place dst == a1
            let acc_ptrs = [a1.as_ptr(), std::ptr::null(), a2.as_ptr(), b2.as_ptr()];
            let mut acc_asm = vec![-5.0f64; n + 1];
            let mut acc_ref = vec![-5.0f64; n + 1];
            unsafe {
                (asm.ntt_pmuladd)(acc_asm.as_mut_ptr(), &acc_ptrs, &vcfg2);
                (rfk.ntt_pmuladd)(acc_ref.as_mut_ptr(), &acc_ptrs, &vcfg2);
            }
            assert_eq!(acc_asm, acc_ref, "ntt_pmuladd acc n={n} sub={sub_second}");

            let mut ip_asm = a1.clone();
            let mut ip_ref = a1.clone();
            let ip_asm_ptrs = [ip_asm.as_ptr(), std::ptr::null(), a2.as_ptr(), b2.as_ptr()];
            let ip_ref_ptrs = [ip_ref.as_ptr(), std::ptr::null(), a2.as_ptr(), b2.as_ptr()];
            unsafe {
                (asm.ntt_pmuladd)(ip_asm.as_mut_ptr(), &ip_asm_ptrs, &vcfg2);
                (rfk.ntt_pmuladd)(ip_ref.as_mut_ptr(), &ip_ref_ptrs, &vcfg2);
            }
            assert_eq!(ip_asm, ip_ref, "ntt_pmuladd acc inplace n={n} sub={sub_second}");
        }
    }

    // 4b. ntt_pmul row-twist mode (sub_second = cols)
    for &rows in &[1usize, 2, 5] {
        for &cols in &[1usize, 7, 8, 9, 32, 33, 64, 65] {
            let n = rows * cols;
            let orig: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
            let tw_tile: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
            let row_sf: Vec<f64> = (0..rows).map(|_| (rng.next_u64() % p0) as f64).collect();
            let mut d_asm = orig.clone();
            let mut d_ref = orig.clone();
            let vcfg_rt = NttVecCfg {
                n: n as u64,
                p: p0_f,
                inv_p: inv_p0,
                sub_second: cols as u64,
            };
            unsafe {
                (asm.ntt_pmul)(d_asm.as_mut_ptr(), tw_tile.as_ptr(), row_sf.as_ptr(), &vcfg_rt);
                (rfk.ntt_pmul)(d_ref.as_mut_ptr(), tw_tile.as_ptr(), row_sf.as_ptr(), &vcfg_rt);
            }
            assert_eq!(d_asm, d_ref, "ntt_pmul row_twist rows={rows} cols={cols}");
        }
    }

    // 5. ntt_crt (unsigned, signed, cyclic wrap including 2^L - 1 -> 0, and carry chaining across k_p=1..18)
    for &k_p in &[1usize, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 18] {
        let crt = CrtPlan::build(primes[..k_p].to_vec());
        let cb_list: &[u64] = if k_p >= 6 {
            &[
                47, 64, 86, 127, 128, 129, 144, 160, 161, 176, 192, 193, 200, 208, 210, 256, 300,
                384,
            ]
        } else if k_p == 3 {
            &[47, 60, 64, 65, 66, 67, 80, 95, 96, 110, 127, 128]
        } else {
            &[47, 60, 64, 75, 86]
        };
        for &cb in cb_list {
            if cb > 64 * (crt.k_words as u64) {
                continue;
            }
            for &count in &[1usize, 7, 8, 9, 32, 37, 45, 65, 96] {
                for &(signed, wrap, null_carry) in &[
                    (0u64, 0u64, false),
                    (0, 0, true),
                    (1, 0, false),
                    (1, 0, true),
                    (0, 1, true),
                ] {
                    let mut res_vecs: Vec<Vec<f64>> = Vec::with_capacity(k_p);
                    let safe_words = crt.k_words.saturating_sub(1);
                    for j in 0..count {
                        let neg_mag = signed != 0 && wrap == 0 && (j & 1) == 1 && j + 1 < count;
                        let extremal_p_minus_1 = signed == 0 && wrap == 0 && (j % 17) == 0;
                        let mut mag_words = [0u64; 16];
                        if wrap != 0 && k_p >= 3 && cb == 64 && count == 8 {
                            mag_words[0] = u64::MAX;
                        } else if wrap != 0 {
                            mag_words[0] = rng.next_u64();
                            mag_words[1] = rng.next_u64() & 3;
                        } else if k_p == 1 {
                            mag_words[0] = rng.next_u64() >> 18;
                        } else if neg_mag && (j % 5) == 1 {
                            mag_words[0] = 1; // Exercise exact -1 across all limbs (including sh==28)!
                        } else if k_p <= 5 {
                            mag_words[0] = rng.next_u64();
                            mag_words[1] = rng.next_u64() >> 4;
                        } else {
                            let full_w = safe_words.min(15);
                            for w in 0..full_w.saturating_sub(1) {
                                mag_words[w] = rng.next_u64();
                            }
                            if full_w > 0 {
                                mag_words[full_w - 1] = rng.next_u64() >> 8;
                            }
                        }
                        for p_idx in 0..k_p {
                            if j == 0 {
                                res_vecs.push(Vec::with_capacity(count));
                            }
                            let pr = crt.primes_u64[p_idx] as u128;
                            let r = if extremal_p_minus_1 {
                                pr - 1
                            } else {
                                let mut rem = 0u128;
                                for &w in mag_words.iter().rev() {
                                    rem = ((rem << 64) | (w as u128)) % pr;
                                }
                                if neg_mag && rem != 0 { pr - rem } else { rem }
                            };
                            res_vecs[p_idx].push(r as f64);
                        }
                    }
                    let ptrs: Vec<*const f64> = res_vecs.iter().map(|v| v.as_ptr()).collect();
                    let extra_limbs = if null_carry { 18 } else { 4 };
                    let out_limbs = ((count as u64) * cb).div_ceil(64) as usize + extra_limbs;
                    let mut out_asm = vec![0xABCDu64; out_limbs];
                    let mut out_ref = vec![0xABCDu64; out_limbs];
                    let mut carry_asm = vec![0u64; crt.k_words + 1];
                    let mut carry_ref = vec![0u64; crt.k_words + 1];
                    if wrap == 0 && signed == 0 && !null_carry {
                        carry_asm[0] = 12345;
                        carry_ref[0] = 12345;
                    }
                    let mut c_asm = NttCrtCfg {
                        k_primes: k_p as u64,
                        count: count as u64,
                        coeff_bits: cb,
                        out_limbs: if wrap != 0 {
                            ((count as u64) * cb).div_ceil(64)
                        } else {
                            out_limbs as u64
                        },
                        signed,
                        wrap_cyclic: wrap,
                        primes: crt.primes_f64.as_ptr(),
                        inv_primes: crt.inv_primes_f64.as_ptr(),
                        garner_inv: crt.garner_inv.as_ptr(),
                        prefix_prods: crt.prefix_prods.as_ptr(),
                        k_words: crt.k_words as u64,
                        segs: NttCrtSegs::ONE,
                        carry_limbs: if null_carry {
                            std::ptr::null_mut()
                        } else {
                            carry_asm.as_mut_ptr()
                        },
                    };
                    let mut c_ref = NttCrtCfg {
                        carry_limbs: if null_carry {
                            std::ptr::null_mut()
                        } else {
                            carry_ref.as_mut_ptr()
                        },
                        ..c_asm
                    };
                    let w0_asm = unsafe { (asm.ntt_crt)(out_asm.as_mut_ptr(), ptrs.as_ptr(), &mut c_asm) };
                    let w0_ref = unsafe { (rfk.ntt_crt)(out_ref.as_mut_ptr(), ptrs.as_ptr(), &mut c_ref) };
                    assert_eq!(
                        out_asm, out_ref,
                        "ntt_crt kp={k_p} cb={cb} cnt={count} s={signed} w={wrap} nc={null_carry}"
                    );
                    assert_eq!(w0_asm, w0_ref);
                    if !null_carry {
                        assert_eq!(carry_asm, carry_ref);
                    }
                }
            }
        }
    }
}

#[test]
fn gate1_fuzz_family4_and_5_movement_digits_check() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_1234);

    // 1. gather & scatter (unpacked + 50-bit packed + NT + unaligned byte offsets)
    for &packed in &[0u64, 1u64] {
        for &use_nt in &[0u64, 1u64] {
            for &(rows, cols) in &[(1u64, 8u64), (3, 16), (4, 24), (2, 136)] {
                let row_bytes = if packed != 0 {
                    (cols / 8) * 50
                } else {
                    cols * 8
                };
                let stride = row_bytes + if packed != 0 { 14 } else { 16 };
                let src: Vec<f64> = (0..(rows * cols))
                    .map(|_| (rng.next_u64() & ((1u64 << 50) - 1)) as f64)
                    .collect();
                // Add an unaligned prefix offset (8 bytes for unpacked, 3 bytes for packed)
                let pad_off = if packed != 0 { 3usize } else { 8usize };
                let total_bytes = pad_off + (rows * stride) as usize + 64;
                let mut mem_asm = vec![0xEEu8; total_bytes];
                let mut mem_ref = vec![0xEEu8; total_bytes];
                let cfg = MoveCfg {
                    rows,
                    cols,
                    outer_stride_bytes: stride,
                    packed,
                    use_nt,
                };
                unsafe {
                    (asm.scatter)(mem_asm.as_mut_ptr().add(pad_off), src.as_ptr(), &cfg);
                    (rfk.scatter)(mem_ref.as_mut_ptr().add(pad_off), src.as_ptr(), &cfg);
                }
                assert_eq!(
                    mem_asm, mem_ref,
                    "scatter packed={packed} nt={use_nt} rows={rows} cols={cols}"
                );

                let mut g_asm = vec![-1.0f64; (rows * cols) as usize + 4];
                let mut g_ref = vec![-1.0f64; (rows * cols) as usize + 4];
                unsafe {
                    (asm.gather)(g_asm.as_mut_ptr(), mem_asm.as_ptr().add(pad_off), &cfg);
                    (rfk.gather)(g_ref.as_mut_ptr(), mem_ref.as_ptr().add(pad_off), &cfg);
                }
                assert_eq!(g_asm, g_ref, "gather packed={packed} nt={use_nt}");
                assert_eq!(&g_asm[..(rows * cols) as usize], src.as_slice());
            }
        }
    }

    // 2. to_digits (text & packed19)
    for &fmt_text in &[0u64, 1u64] {
        for &count in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9, 17, 64] {
            let mut words: Vec<u64> = (0..count)
                .map(|i| match i {
                    0 => 0,
                    1 => 9_999_999_999_999_999_999,
                    2 => 1,
                    _ => rng.next_u64() % 10_000_000_000_000_000_000,
                })
                .collect();
            if words.is_empty() {
                words.push(0);
            }
            let out_len = count * if fmt_text != 0 { 19 } else { 8 } + 16;
            let mut o_asm = vec![0xCCu8; out_len];
            let mut o_ref = vec![0xCCu8; out_len];
            let w_asm = unsafe {
                (asm.to_digits)(o_asm.as_mut_ptr(), words.as_ptr(), count as u64, fmt_text)
            };
            let w_ref = unsafe {
                (rfk.to_digits)(o_ref.as_mut_ptr(), words.as_ptr(), count as u64, fmt_text)
            };
            assert_eq!((w_asm, &o_asm), (w_ref, &o_ref), "to_digits cnt={count} txt={fmt_text}");
        }
    }

    // 3. check_n (w0 alone & all 3 check words + null out_words)
    for &n in &[0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1025] {
        let data: Vec<u64> = (0..n.max(1)).map(|_| rng.next_u64()).collect();
        for &all3 in &[0u64, 1u64] {
            let mut w_asm = [0x99u64; 3];
            let mut w_ref = [0x99u64; 3];
            let r_asm = unsafe { (asm.check_n)(data.as_ptr(), n as u64, &mut w_asm, all3) };
            let r_ref = unsafe { (rfk.check_n)(data.as_ptr(), n as u64, &mut w_ref, all3) };
            assert_eq!((r_asm, w_asm), (r_ref, w_ref), "check_n n={n} all3={all3}");
        }
        let r_null = unsafe { (asm.check_n)(data.as_ptr(), n as u64, std::ptr::null_mut(), 0) };
        let r_ref0 = unsafe { (rfk.check_n)(data.as_ptr(), n as u64, std::ptr::null_mut(), 0) };
        assert_eq!(r_null, r_ref0);
    }
}

#[test]
fn gate1_compare_digits_against_y_cruncher() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let yc_exe = manifest_dir
        .parent()
        .unwrap()
        .join("y-cruncher v0.8.7.9547b")
        .join("y-cruncher.exe");
    if !yc_exe.exists() {
        // y-cruncher is not part of this repository. Where its Windows package is not unpacked
        // beside the sources there is nothing to compare with, and the test says so.
        eprintln!("SKIPPED: no y-cruncher at {yc_exe:?}: the digits were not compared with it");
        return;
    }

    let tmp_dir = std::env::temp_dir().join(format!("supi_gate1_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    // Generate 100,000 verified digits with y-cruncher v0.8.7 Build 9547
    let status = Command::new(&yc_exe)
        .args([
            "pause:-2",
            "skip-warnings",
            "status:none",
            "colors:0",
            "custom",
            "pi",
            "-dec:100000",
            "-o",
        ])
        .arg(&tmp_dir)
        .status()
        .expect("spawn y-cruncher");
    assert!(status.success(), "y-cruncher must exit cleanly");

    let yc_pi = std::fs::read(tmp_dir.join("Pi - Dec - Chudnovsky.txt")).expect("read y-cruncher pi");
    assert_eq!(yc_pi.len(), 100_002);

    let (kernels_asm, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let kernels_ref = Kernels::reference_only();
    let prof = Profile::synthetic(4, 8, 32 * 1024, 512 * 1024, 8 * 1024 * 1024, 512 * 1024 * 1024, &[], &[]);

    // Compare Text output across multiple digit counts (assembly & reference kernels)
    for &digits in &[100u64, 1_000, 10_000, 100_000] {
        let expected = &yc_pi[..(digits as usize + 2)];
        let sink = std::sync::Arc::new(MemoryDigitSink::new(expected.len()));
        let factory = MemFactory {
            buf: std::sync::Arc::clone(&sink),
        };
        let cfg = EngineConfig::new(
            digits,
            OutputFormat::Text,
            "mem",
            prof.clone(),
            &kernels_asm,
            &factory,
        );
        let rep = run(&cfg).expect("run text asm");
        assert!(!rep.digits_in_doubt);
        let got = sink.buf.lock().unwrap().clone();
        if got.as_slice() != expected {
            let first_diff = got
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(expected.len()));
            let total_diffs = got
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            let lo = first_diff.saturating_sub(24);
            let hi_g = (first_diff + 48).min(got.len());
            let hi_e = (first_diff + 48).min(expected.len());
            panic!(
                "Text mismatch vs y-cruncher at {digits} digits: len={} vs {}, first_diff={first_diff} (leaf={}), total_diffs={total_diffs}\n  got: {:?}\n  exp: {:?}",
                got.len(),
                expected.len(),
                first_diff.saturating_sub(2) / 19,
                std::str::from_utf8(&got[lo..hi_g]).unwrap_or("<non-utf8>"),
                std::str::from_utf8(&expected[lo..hi_e]).unwrap_or("<non-utf8>"),
            );
        }

        if digits == 1_000 {
            let sink_r = std::sync::Arc::new(MemoryDigitSink::new(expected.len()));
            let fac_r = MemFactory {
                buf: std::sync::Arc::clone(&sink_r),
            };
            let cfg_r = EngineConfig::new(
                digits,
                OutputFormat::Text,
                "mem_r",
                prof.clone(),
                &kernels_ref,
                &fac_r,
            );
            let rep_r = run(&cfg_r).expect("run text ref");
            assert_eq!(rep_r.dec_hash_w0, rep.dec_hash_w0);
            assert_eq!(sink_r.buf.lock().unwrap().as_slice(), expected);
        }

        if digits == 100_000 {
            // Stress-sweep synthetic calibrations across basecase/karatsuba/FFT/NTT crossovers
            // so calibration drift under parallel load is deterministically covered.
            for &bc_max in &[16usize, 24, 32, 48, 64] {
                for &kara_max in &[48usize, 80, 128, 192, 256] {
                    if kara_max <= bc_max {
                        continue;
                    }
                    for &(ntt_rate, fft_mult, nt_eff) in &[
                        (2.0e9f64, 1.65f64, true),
                        (2.0e9f64, 0.50f64, false),
                        (0.4e9f64, 3.00f64, true),
                    ] {
                        let mut cal = engine::plan::Calibration::synthetic(
                            ntt_rate,
                            4.0e10,
                            nt_eff,
                            Vec::new(),
                        );
                        cal.fft_bf_per_sec_per_worker = ntt_rate * fft_mult;
                        cal.basecase_max_limbs = bc_max;
                        cal.karatsuba_max_limbs = kara_max;
                        let sink_c = std::sync::Arc::new(MemoryDigitSink::new(expected.len()));
                        let fac_c = MemFactory {
                            buf: std::sync::Arc::clone(&sink_c),
                        };
                        let mut cfg_c = EngineConfig::new(
                            digits,
                            OutputFormat::Text,
                            "mem_c",
                            prof.clone(),
                            &kernels_asm,
                            &fac_c,
                        );
                        cfg_c.supplied_calibration = Some(cal);
                        let rep_c = run(&cfg_c).expect("run text asm synthetic cal");
                        assert!(!rep_c.digits_in_doubt);
                        let got_c = sink_c.buf.lock().unwrap().clone();
                        if got_c.as_slice() != expected {
                            let first_diff = got_c
                                .iter()
                                .zip(expected.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            let lo = first_diff.saturating_sub(24);
                            let hi_g = (first_diff + 48).min(got_c.len());
                            let hi_e = (first_diff + 48).min(expected.len());
                            panic!(
                                "Synthetic cal mismatch (bc={bc_max}, kara={kara_max}, fft_mult={fft_mult}, nt={nt_eff}): first_diff={first_diff} (leaf={})\n  got: {:?}\n  exp: {:?}",
                                first_diff.saturating_sub(2) / 19,
                                std::str::from_utf8(&got_c[lo..hi_g]).unwrap_or("<non-utf8>"),
                                std::str::from_utf8(&expected[lo..hi_e]).unwrap_or("<non-utf8>"),
                            );
                        }
                    }
                }
            }
        }
    }

    // Also verify Packed19 format against y-cruncher's 10,000 digits!
    let digits_pk = 10_000u64;
    let pk_bytes = OutputFormat::Packed19.file_bytes(digits_pk).unwrap() as usize;
    let sink_pk = std::sync::Arc::new(MemoryDigitSink::new(pk_bytes));
    let fac_pk = MemFactory {
        buf: std::sync::Arc::clone(&sink_pk),
    };
    let cfg_pk = EngineConfig::new(
        digits_pk,
        OutputFormat::Packed19,
        "mem_pk",
        prof,
        &kernels_asm,
        &fac_pk,
    );
    run(&cfg_pk).expect("run packed19");
    let pk_raw = sink_pk.buf.lock().unwrap().clone();
    assert_eq!(
        u64::from_le_bytes(pk_raw[0..8].try_into().unwrap()),
        PACKED19_MAGIC
    );
    assert_eq!(
        u64::from_le_bytes(pk_raw[8..16].try_into().unwrap()),
        digits_pk
    );
    let mut unpacked_ascii = Vec::with_capacity(10_030);
    unpacked_ascii.extend_from_slice(b"3.");
    let chunks: Vec<&[u8]> = pk_raw[16..].chunks_exact(8).collect();
    let rem_d = (digits_pk as usize) % 19;
    for (idx, chunk) in chunks.iter().enumerate() {
        let val = u64::from_le_bytes((*chunk).try_into().unwrap());
        let width = if idx + 1 == chunks.len() && rem_d > 0 {
            rem_d
        } else {
            19
        };
        let s = format!("{val:0width$}");
        unpacked_ascii.extend_from_slice(s.as_bytes());
    }
    assert_eq!(
        unpacked_ascii.as_slice(),
        &yc_pi[..(digits_pk as usize + 2)]
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

/// The shipped binary takes every large block from the process region (`engine::arena`). At two
/// million digits every number of the upper tree levels is such a block, so this compares that
/// path, digit for digit, with y-cruncher.
#[test]
fn gate1_binary_digits_against_y_cruncher_at_two_million() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let yc_exe = manifest_dir
        .parent()
        .unwrap()
        .join("y-cruncher v0.8.7.9547b")
        .join("y-cruncher.exe");
    if !yc_exe.exists() {
        // y-cruncher is not part of this repository. Where its Windows package is not unpacked
        // beside the sources there is nothing to compare with, and the test says so.
        eprintln!("SKIPPED: no y-cruncher at {yc_exe:?}: the digits were not compared with it");
        return;
    }

    let tmp_dir = std::env::temp_dir().join(format!("supi_gate1_bin_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let digits = 2_000_000usize;
    let status = Command::new(&yc_exe)
        .args([
            "pause:-2",
            "skip-warnings",
            "status:none",
            "colors:0",
            "custom",
            "pi",
            &format!("-dec:{digits}"),
            "-o",
        ])
        .arg(&tmp_dir)
        .status()
        .expect("spawn y-cruncher");
    assert!(status.success(), "y-cruncher must exit cleanly");
    let yc_pi = std::fs::read(tmp_dir.join("Pi - Dec - Chudnovsky.txt")).expect("read y-cruncher pi");
    assert_eq!(yc_pi.len(), digits + 2);

    let out_path = tmp_dir.join("supi.txt");
    let out = Command::new(env!("CARGO_BIN_EXE_suPiposition"))
        .arg(digits.to_string())
        .arg("-o")
        .arg(&out_path)
        .arg("--no-vram")
        .output()
        .expect("spawn suPiposition");
    assert!(
        out.status.success(),
        "suPiposition must exit cleanly: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ours = std::fs::read(&out_path).expect("read suPiposition digits");
    assert_eq!(ours.len(), yc_pi.len());
    if ours != yc_pi {
        let first_diff = ours.iter().zip(yc_pi.iter()).position(|(a, b)| a != b).unwrap();
        panic!("binary differs from y-cruncher at byte {first_diff} of {digits} digits");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

/// `ntt_crt` on whole 32-coefficient batches without carry in, where the assembly kernel places
/// the digits a block at a time: every prime count from 3 to 9, digit widths on both sides of
/// the 64-bit word boundaries and down to the narrowest the block path takes, and values that
/// make a digit land exactly on 2^b so that the carry between digits runs through many of them.
#[test]
fn gate1_fuzz_ntt_crt_block_emit() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x1357_9BDF_2468_ACE0);
    let primes = largest_primes_for_two_adic(40, 18);

    // residues of the integer given by little-endian words
    let residues = |words: &[u64], crt: &CrtPlan, out: &mut [Vec<f64>]| {
        for (pi, &p) in crt.primes_u64.iter().enumerate() {
            let pr = p as u128;
            let mut rem = 0u128;
            for &w in words.iter().rev() {
                rem = ((rem << 64) | (w as u128)) % pr;
            }
            out[pi].push(rem as f64);
        }
    };
    let set_bits = |words: &mut [u64], from: usize, to: usize| {
        for bit in from..to {
            words[bit / 64] |= 1u64 << (bit % 64);
        }
    };

    let mut cases = 0usize;
    for k_p in 3usize..=9 {
        let crt = CrtPlan::build(primes[..k_p].to_vec());
        let kw = crt.k_words;
        let p_tot = &crt.prefix_prods[(k_p * kw)..((k_p + 1) * kw)];
        let top = p_tot.iter().rposition(|&w| w != 0).unwrap();
        let pb = top * 64 + 64 - (p_tot[top].leading_zeros() as usize);
        // the narrowest digit of the block path: three digits, the third of at most 126 bits
        let lo = pb.div_ceil(3).max((pb.saturating_sub(126)).div_ceil(2));
        let hi = if k_p <= 5 { 127 } else { 208 };
        let mut cbs: Vec<usize> = vec![
            lo,
            lo + 1,
            (lo + hi) / 2,
            hi,
            (pb - 62).div_ceil(2),
            (pb - 64) / 2,
        ];
        cbs.extend([63usize, 64, 65, 127, 128, 129, 191, 192, 193]);
        cbs.retain(|&cb| cb >= lo && cb <= hi);
        cbs.sort_unstable();
        cbs.dedup();
        for &cb in &cbs {
            for &count in &[32usize, 64, 96, 128, 192, 256] {
                for mode in 0..3 {
                    for pattern in 0..6 {
                        let mut res: Vec<Vec<f64>> =
                            (0..k_p).map(|_| Vec::with_capacity(count)).collect();
                        for j in 0..count {
                            let mut x = [0u64; 12];
                            match pattern {
                                0 => {
                                    // any value below the product of the primes
                                    for w in x.iter_mut().take(kw) {
                                        *w = rng.next_u64();
                                    }
                                    for bit in (pb - 1)..(kw * 64) {
                                        x[bit / 64] &= !(1u64 << (bit % 64));
                                    }
                                }
                                1 => {
                                    // the largest value: the product minus one
                                    x[..kw].copy_from_slice(p_tot);
                                    let mut i = 0;
                                    while x[i] == 0 {
                                        x[i] = u64::MAX;
                                        i += 1;
                                    }
                                    x[i] -= 1;
                                }
                                2 => {}
                                3 => {
                                    // both low digits all ones and a small third digit: the sum
                                    // of three digits lands on 2^b - 1 or 2^b again and again
                                    set_bits(&mut x, 0, 2 * cb);
                                    let c = rng.next_u64() % 4;
                                    if pb > 2 * cb + 3 {
                                        x[(2 * cb) / 64] |= c << ((2 * cb) % 64);
                                        if (2 * cb) % 64 > 61 {
                                            x[(2 * cb) / 64 + 1] |= c >> (64 - (2 * cb) % 64);
                                        }
                                    }
                                }
                                4 => {
                                    // mostly one all-ones digit, sometimes a random value
                                    if rng.next_u64() % 10 < 7 {
                                        set_bits(&mut x, 0, cb);
                                    } else {
                                        for w in x.iter_mut().take(kw) {
                                            *w = rng.next_u64();
                                        }
                                        for bit in (pb - 1)..(kw * 64) {
                                            x[bit / 64] &= !(1u64 << (bit % 64));
                                        }
                                    }
                                }
                                _ => {
                                    // a single bit, or all bits below it
                                    let bit = (rng.next_u64() as usize) % (pb - 1);
                                    if j % 2 == 0 {
                                        x[bit / 64] = 1u64 << (bit % 64);
                                    } else {
                                        set_bits(&mut x, 0, bit);
                                    }
                                }
                            }
                            residues(&x[..kw], &crt, &mut res);
                        }
                        let ptrs: Vec<*const f64> = res.iter().map(|v| v.as_ptr()).collect();
                        // mode 0: carry returned to the caller; 1: carry flushed into spare
                        // limbs; 2: cyclic wrap of the carry
                        let stream = (count * cb).div_ceil(64);
                        let out_limbs = if mode == 1 { stream + kw + 2 } else { stream };
                        let mut out_asm = vec![0xABCDu64; out_limbs + 2];
                        let mut out_ref = vec![0xABCDu64; out_limbs + 2];
                        let mut carry_asm = vec![0u64; kw + 1];
                        let mut carry_ref = vec![0u64; kw + 1];
                        let mut c_asm = NttCrtCfg {
                            k_primes: k_p as u64,
                            count: count as u64,
                            coeff_bits: cb as u64,
                            out_limbs: out_limbs as u64,
                            signed: 0,
                            wrap_cyclic: u64::from(mode == 2),
                            primes: crt.primes_f64.as_ptr(),
                            inv_primes: crt.inv_primes_f64.as_ptr(),
                            garner_inv: crt.garner_inv.as_ptr(),
                            prefix_prods: crt.prefix_prods.as_ptr(),
                            k_words: kw as u64,
                            segs: NttCrtSegs::ONE,
                            carry_limbs: if mode == 0 {
                                carry_asm.as_mut_ptr()
                            } else {
                                std::ptr::null_mut()
                            },
                        };
                        let mut c_ref = NttCrtCfg {
                            carry_limbs: if mode == 0 {
                                carry_ref.as_mut_ptr()
                            } else {
                                std::ptr::null_mut()
                            },
                            ..c_asm
                        };
                        let w0_asm =
                            unsafe { (asm.ntt_crt)(out_asm.as_mut_ptr(), ptrs.as_ptr(), &mut c_asm) };
                        let w0_ref =
                            unsafe { (rfk.ntt_crt)(out_ref.as_mut_ptr(), ptrs.as_ptr(), &mut c_ref) };
                        let what = format!("kp={k_p} cb={cb} cnt={count} mode={mode} pattern={pattern}");
                        assert_eq!(out_asm, out_ref, "ntt_crt block limbs {what}");
                        assert_eq!(w0_asm, w0_ref, "ntt_crt block w0 {what}");
                        assert_eq!(carry_asm, carry_ref, "ntt_crt block carry {what}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 4000);
}

/// `ntt_crt` on many equal streams in one call (`NttCrtSegs`), against the reference: prime
/// counts and digit widths on both sides of what the assembly kernel's block path takes;
/// streams of 32 coefficients, which it takes two at a time and the last one alone when their
/// number is odd, streams of whole 64s, and lengths and outputs that it leaves to one call per
/// stream; strides wider than a stream, with a sentinel between the streams that no call may
/// touch; carry areas that hold other values on entry; and values that carry through many
/// digits.
#[test]
fn gate1_fuzz_ntt_crt_segments() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x0F1E_2D3C_4B5A_6978);
    let primes = largest_primes_for_two_adic(40, 18);
    const SENTINEL: u64 = 0x5EA7_5EA7_5EA7_5EA7;

    let set_bits = |words: &mut [u64], from: usize, to: usize| {
        for bit in from..to {
            words[bit / 64] |= 1u64 << (bit % 64);
        }
    };

    let mut cases = 0usize;
    let mut block_cases = 0usize;
    for k_p in [2usize, 3, 4, 5, 6, 7, 8, 9, 10, 13] {
        let crt = CrtPlan::build(primes[..k_p].to_vec());
        let kw = crt.k_words;
        let p_tot = &crt.prefix_prods[(k_p * kw)..((k_p + 1) * kw)];
        let top = p_tot.iter().rposition(|&w| w != 0).unwrap();
        let pb = top * 64 + 64 - (p_tot[top].leading_zeros() as usize);
        // the digit widths of the block path: three digits, the third of at most 126 bits
        let lo = pb.div_ceil(3).max((pb.saturating_sub(126)).div_ceil(2));
        let hi = if k_p <= 5 { 127 } else { 208 };
        let mid = (lo + hi) / 2;
        let mut cbs: Vec<usize> = vec![
            lo.saturating_sub(2).max(40),
            lo,
            lo + 1,
            lo + 2,
            mid,
            mid + 1,
            hi - 1,
            hi,
            hi + 1,
            64,
            128,
            192,
        ];
        cbs.retain(|&cb| cb <= 64 * kw && cb <= 384);
        cbs.sort_unstable();
        cbs.dedup();
        for &cb in &cbs {
            for &count in &[32usize, 64, 128, 96, 40] {
                for &segments in &[2usize, 3, 5, 8] {
                    for variant in 0..6usize {
                        let signed = u64::from(variant == 5);
                        let pattern = variant % 5;
                        let exact = (count * cb).div_ceil(64);
                        // most outputs hold exactly the digits of a stream; some are longer,
                        // some are cut short
                        let out_limbs = match (cases + variant) % 7 {
                            5 => exact + 2,
                            6 if exact > 1 => exact - 1,
                            _ => exact,
                        };
                        let wide = (cases + segments) % 3 == 1;
                        let res_stride = if wide { count + 24 } else { count };
                        let out_stride = if wide { out_limbs + 3 } else { out_limbs };
                        let carry_stride = if wide { kw + 2 } else { kw };
                        let w0_stride = if wide { 3 } else { 1 };

                        let mut res: Vec<Vec<f64>> =
                            (0..k_p).map(|_| vec![0.0f64; (segments - 1) * res_stride + count]).collect();
                        for s in 0..segments {
                            for j in 0..count {
                                let mut x = [0u64; 16];
                                match pattern {
                                    0 => {
                                        for w in x.iter_mut().take(kw) {
                                            *w = rng.next_u64();
                                        }
                                        for bit in (pb - 1)..(kw * 64) {
                                            x[bit / 64] &= !(1u64 << (bit % 64));
                                        }
                                    }
                                    1 => {
                                        // the largest value: the product minus one
                                        x[..kw].copy_from_slice(p_tot);
                                        let mut i = 0;
                                        while x[i] == 0 {
                                            x[i] = u64::MAX;
                                            i += 1;
                                        }
                                        x[i] -= 1;
                                    }
                                    2 => {
                                        // both low digits all ones: sums land on 2^b - 1 and 2^b
                                        let ones = (2 * cb).min(pb - 1);
                                        set_bits(&mut x, 0, ones);
                                        if (s + j) % 3 == 0 {
                                            x[0] &= !1;
                                        }
                                    }
                                    3 => {
                                        // mostly one all-ones digit, sometimes any value or zero
                                        match rng.next_u64() % 10 {
                                            0..=6 => set_bits(&mut x, 0, cb.min(pb - 1)),
                                            7 => {}
                                            _ => {
                                                for w in x.iter_mut().take(kw) {
                                                    *w = rng.next_u64();
                                                }
                                                for bit in (pb - 1)..(kw * 64) {
                                                    x[bit / 64] &= !(1u64 << (bit % 64));
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        // a single bit, or all bits below it
                                        let bit = (rng.next_u64() as usize) % (pb - 1);
                                        if j % 2 == 0 {
                                            x[bit / 64] = 1u64 << (bit % 64);
                                        } else {
                                            set_bits(&mut x, 0, bit);
                                        }
                                    }
                                }
                                // a signed call reads every second value as its negative
                                let negate = signed != 0 && j % 2 == 1 && pattern != 1;
                                for (pi, &p) in crt.primes_u64.iter().enumerate() {
                                    let pr = p as u128;
                                    let mut rem = 0u128;
                                    for &w in x[..kw].iter().rev() {
                                        rem = ((rem << 64) | (w as u128)) % pr;
                                    }
                                    if negate && rem != 0 {
                                        rem = pr - rem;
                                    }
                                    res[pi][s * res_stride + j] = rem as f64;
                                }
                            }
                        }
                        let ptrs: Vec<*const f64> = res.iter().map(|v| v.as_ptr()).collect();
                        let out_len = (segments - 1) * out_stride + out_limbs + 2;
                        let carry_len = (segments - 1) * carry_stride + kw + 2;
                        let w0_len = (segments - 1) * w0_stride + 2;
                        let mut out_asm = vec![SENTINEL; out_len];
                        let mut out_ref = vec![SENTINEL; out_len];
                        // carry areas hold other values on entry: every stream starts from zero
                        let mut carry_asm: Vec<u64> = (0..carry_len).map(|_| rng.next_u64()).collect();
                        let mut carry_ref = carry_asm.clone();
                        if wide {
                            for s in 0..segments {
                                for w in kw..carry_stride {
                                    carry_asm[s * carry_stride + w] = SENTINEL;
                                    carry_ref[s * carry_stride + w] = SENTINEL;
                                }
                            }
                        }
                        let mut w0_asm = vec![SENTINEL; w0_len];
                        let mut w0_ref = vec![SENTINEL; w0_len];
                        let mut c_asm = NttCrtCfg {
                            k_primes: k_p as u64,
                            count: count as u64,
                            coeff_bits: cb as u64,
                            out_limbs: out_limbs as u64,
                            signed,
                            wrap_cyclic: 0,
                            primes: crt.primes_f64.as_ptr(),
                            inv_primes: crt.inv_primes_f64.as_ptr(),
                            garner_inv: crt.garner_inv.as_ptr(),
                            prefix_prods: crt.prefix_prods.as_ptr(),
                            k_words: kw as u64,
                            carry_limbs: carry_asm.as_mut_ptr(),
                            segs: NttCrtSegs {
                                segments: segments as u64,
                                res_stride: res_stride as u64,
                                out_stride: out_stride as u64,
                                carry_stride: carry_stride as u64,
                                w0: w0_asm.as_mut_ptr(),
                                w0_stride: w0_stride as u64,
                            },
                        };
                        let mut c_ref = NttCrtCfg {
                            carry_limbs: carry_ref.as_mut_ptr(),
                            segs: NttCrtSegs {
                                w0: w0_ref.as_mut_ptr(),
                                ..c_asm.segs
                            },
                            ..c_asm
                        };
                        let ret_asm =
                            unsafe { (asm.ntt_crt)(out_asm.as_mut_ptr(), ptrs.as_ptr(), &mut c_asm) };
                        let ret_ref =
                            unsafe { (rfk.ntt_crt)(out_ref.as_mut_ptr(), ptrs.as_ptr(), &mut c_ref) };
                        let what = format!(
                            "kp={k_p} cb={cb} cnt={count} segs={segments} variant={variant} out={out_limbs}/{exact} wide={wide}"
                        );
                        assert_eq!(out_asm, out_ref, "ntt_crt segments limbs {what}");
                        assert_eq!(carry_asm, carry_ref, "ntt_crt segments carry {what}");
                        assert_eq!(w0_asm, w0_ref, "ntt_crt segments w0 {what}");
                        assert_eq!((ret_asm, ret_ref), (0, 0), "ntt_crt segments return {what}");
                        cases += 1;
                        // what the block path of the assembly kernel takes
                        if signed == 0
                            && (3..=9).contains(&k_p)
                            && cb >= lo
                            && cb <= hi
                            && out_limbs * 64 == count * cb
                            && (count == 32 || count % 64 == 0)
                        {
                            block_cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 10_000, "{cases}");
    assert!(block_cases > 1_000, "{block_cases}");
}

/// `ntt_split` where the assembly kernel runs its chain over 48-bit chunks without reducing
/// between the steps: moduli at both ends of the range that path takes (just above 2^48, around
/// 2^49, just below 2^50) and the primes of a real run, chunk counts from 2 to 8, and limbs
/// that make every chunk as large as it can be, zero, or alternate between the two. Both the
/// direct form and the planar form (extract once, reduce per modulus) against the reference.
#[test]
fn gate1_fuzz_ntt_split_chains() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x4242_1357_9BDF_0246);
    let real = largest_primes_for_two_adic(40, 18);
    let moduli: Vec<u64> = vec![
        (1u64 << 48) + 1,
        (1u64 << 48) + 21,
        (1u64 << 48) + (1u64 << 47) - 1,
        (1u64 << 49) - 1,
        (1u64 << 49) + 1,
        (1u64 << 49) + (1u64 << 48) + 3,
        (1u64 << 50) - 1,
        (1u64 << 50) - 27,
        real[0],
        real[9],
        real[17],
    ];
    let patterns = 6usize;
    let n_limbs = 1400usize;
    let mut cases = 0usize;
    for pattern in 0..patterns {
        let limbs: Vec<u64> = (0..n_limbs)
            .map(|i| match pattern {
                0 => rng.next_u64(),
                1 => u64::MAX,
                2 => 0,
                3 => {
                    if (i / 3) % 2 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                4 => {
                    if rng.next_u64() % 4 == 0 {
                        rng.next_u64()
                    } else {
                        u64::MAX
                    }
                }
                _ => {
                    if i % 2 == 0 {
                        u64::MAX
                    } else {
                        rng.next_u64()
                    }
                }
            })
            .collect();
        for &cb in &[
            49u64, 64, 95, 96, 97, 108, 130, 144, 145, 152, 164, 186, 192, 193, 200, 208, 209, 240, 241,
            254, 278, 288, 289, 336, 337, 384,
        ] {
            let num_chunks = (cb as usize).div_ceil(48);
            for &count in &[32u64, 40, 64, 97, 200] {
                let cnt = count as usize;
                let stride_el = engine::refk::ntt_planar_stride(cnt);
                for &in_l in &[n_limbs as u64, ((count * cb) / 64).saturating_sub(5).max(1)] {
                    let ext_cfg = NttSplitCfg {
                        in_limbs: in_l,
                        start_coeff: 0,
                        count,
                        coeff_bits: cb,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: 0.0,
                        inv_p: 0.0,
                    };
                    let mut planes = vec![-3.0f64; num_chunks * stride_el + 2];
                    unsafe {
                        (rfk.ntt_split)(planes.as_mut_ptr(), limbs.as_ptr(), &ext_cfg);
                    }
                    for &p in &moduli {
                        let pf = p as f64;
                        let direct_cfg = NttSplitCfg {
                            p: pf,
                            inv_p: 1.0 / pf,
                            ..ext_cfg
                        };
                        let red_cfg = NttSplitCfg {
                            in_limbs: u64::MAX,
                            ..direct_cfg
                        };
                        let mut d_asm = vec![-1.0f64; cnt + 2];
                        let mut d_ref = vec![-1.0f64; cnt + 2];
                        let mut r_asm = vec![-1.0f64; cnt + 2];
                        let mut r_ref = vec![-1.0f64; cnt + 2];
                        unsafe {
                            (asm.ntt_split)(d_asm.as_mut_ptr(), limbs.as_ptr(), &direct_cfg);
                            (rfk.ntt_split)(d_ref.as_mut_ptr(), limbs.as_ptr(), &direct_cfg);
                            (asm.ntt_split)(r_asm.as_mut_ptr(), planes.as_ptr() as *const u64, &red_cfg);
                            (rfk.ntt_split)(r_ref.as_mut_ptr(), planes.as_ptr() as *const u64, &red_cfg);
                        }
                        let what = format!("p={p} cb={cb} count={count} in_l={in_l} pattern={pattern}");
                        assert_eq!(d_asm, d_ref, "ntt_split direct {what}");
                        assert_eq!(r_asm, r_ref, "ntt_split planar reduce {what}");
                        assert_eq!(d_ref, r_ref, "ntt_split direct against planar {what}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 15_000, "{cases}");
}

/// The FFT chain on a product that is longer than the transform. The transform is a right-angle
/// one, so what comes out is the product modulo `2^L + 1` (`L` = the bits of all coefficients):
/// with the product `hi * 2^L + lo`, the `L` bits of the output are `lo - hi + c`, where `c` is
/// the carry out of the last coefficient, which `fft_carry` reports (the digits of the inverse
/// transform are negative as well as positive, so `c` may be either). The base conversion's
/// node products rely on exactly this. Checked here: the assembly chain against the reference
/// chain, limbs, check word and carry, and both against exact arithmetic, on random operands
/// and on operands that give the largest sums and long runs for a borrow to cross.
#[test]
fn gate1_fuzz_fft_wrapped_product() {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    assert_eq!(rep.bound_mask, 0x00FF_FFFF);
    let mut rng = XorShift64::new(0x0BAD_5EED_1234_5678);
    let mut worst_err = 0.0f64;
    let mut negative_carries = 0usize;
    let mut cases = 0usize;
    for &(n_complex, coeff_bits) in &[
        (32usize, 18u64),
        (64, 20),
        (256, 21),
        (512, 21),
        (1024, 21),
        (2048, 21),
        (4096, 20),
        (8192, 19),
    ] {
        let (weights, inv_weights, tw, inv_tw) = build_fft_tables(n_complex);
        let l_bits = 2 * n_complex * (coeff_bits as usize);
        assert_eq!(l_bits % 64, 0);
        let kl = l_bits / 64;
        for pattern in 0..24usize {
            // a is the long operand, a few coefficients short of the transform; b the shorter
            // one, as a power of ten is in a node of the conversion
            let la = kl - (4 * (coeff_bits as usize)).div_ceil(64);
            let lb = match pattern % 3 {
                0 => kl / 2,
                1 => kl / 2 + 3,
                _ => (kl / 3).max(2),
            };
            let mut a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let mut b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            match pattern {
                3 => {
                    a.iter_mut().for_each(|w| *w = u64::MAX);
                    b.iter_mut().for_each(|w| *w = u64::MAX);
                }
                4 => {
                    // the low bits of the product are below the bits above them
                    a.iter_mut().for_each(|w| *w = 0);
                    a[la - 1] = 1u64 << 63;
                    a[0] = 1;
                }
                5 => {
                    // long runs of zeros in the low product
                    for w in a.iter_mut().take(la - 2) {
                        *w = 0;
                    }
                    b.iter_mut().skip(1).for_each(|w| *w = 0);
                    b[0] = 1;
                    b[lb - 1] = 1u64 << 62;
                }
                6 => {
                    a.iter_mut().for_each(|w| *w = 0);
                }
                7 => {
                    // alternating all-ones and zero limbs
                    for (i, w) in a.iter_mut().enumerate() {
                        *w = if i % 2 == 0 { u64::MAX } else { 0 };
                    }
                }
                _ => {}
            }
            let chain = |t: &KernelTable| -> (Vec<u64>, u64, f64, i64) {
                let mut fa = vec![7.5f64; 2 * n_complex + 64];
                let mut fb = vec![7.5f64; 2 * n_complex + 64];
                let mut out = vec![0x5A5Au64; kl + 64];
                let sa = FftSplitCfg {
                    in_limbs: la as u64,
                    n_complex: n_complex as u64,
                    coeff_bits,
                };
                let sb = FftSplitCfg {
                    in_limbs: lb as u64,
                    ..sa
                };
                let mut cc = FftCarryCfg {
                    out_limbs: kl as u64,
                    n_complex: n_complex as u64,
                    coeff_bits,
                    top_carry: 0x7777,
                    max_err: 0.0,
                    check_word: 0,
                };
                unsafe {
                    let chk = |fa: &Vec<f64>, fb: &Vec<f64>, out: &Vec<u64>, step: &str| {
                        assert!(fa[2 * n_complex..].iter().all(|&x| x == 7.5), "fa overrun after {step}");
                        assert!(fb[2 * n_complex..].iter().all(|&x| x == 7.5), "fb overrun after {step}");
                        assert!(out[kl..].iter().all(|&x| x == 0x5A5A), "out overrun after {step}");
                    };
                    (t.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &sa);
                    chk(&fa, &fb, &out, "split a");
                    (t.fft_split)(fb.as_mut_ptr(), b.as_ptr(), weights.as_ptr(), &sb);
                    chk(&fa, &fb, &out, "split b");
                    (t.fft_fwd)(fa.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
                    (t.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
                    chk(&fa, &fb, &out, "fwd");
                    (t.fft_pmul)(fa.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
                    chk(&fa, &fb, &out, "pmul");
                    (t.fft_inv)(fa.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
                    chk(&fa, &fb, &out, "inv");
                    (t.fft_carry)(out.as_mut_ptr(), fa.as_ptr(), inv_weights.as_ptr(), &mut cc);
                    chk(&fa, &fb, &out, "carry");
                }
                assert_eq!(out[kl], 0x5A5A, "wrote past the output");
                out.truncate(kl);
                (out, cc.check_word, cc.max_err, cc.top_carry as i64)
            };
            let (o_asm, w_asm, e_asm, c_asm) = chain(&k.active);
            let (o_ref, w_ref, e_ref, c_ref) = chain(&k.reference);
            let what = format!("n={n_complex} bits={coeff_bits} pattern={pattern}");
            assert_eq!(o_asm, o_ref, "assembly chain against reference chain, {what}");
            assert_eq!(w_asm, w_ref, "check word, {what}");
            assert_eq!(c_asm, c_ref, "carry of the last coefficient, {what}");
            assert!(c_asm.unsigned_abs() < (1u64 << 40), "carry {c_asm}, {what}");
            let mut dummy = [0u64; 3];
            let w_limbs = unsafe { (k.reference.check_n)(o_asm.as_ptr(), kl as u64, &mut dummy, 0) };
            assert_eq!(w_asm, w_limbs, "check word against the limbs, {what}");
            worst_err = worst_err.max(e_asm).max(e_ref);
            assert!(e_asm < 0.25 && e_ref < 0.25, "rounding error {e_asm} {e_ref}, {what}");

            // Exact. The sum of the coefficients is t = (lo - hi + c) + c * 2^L for an integer c:
            // the limbs are t mod 2^L and the reported carry is floor(t / 2^L) = c + e, with
            // e = floor((lo - hi + c) / 2^L), which is -1, 0 or 1 (0 where nothing wraps).
            let mut full = vec![0u64; kl + lb + 1];
            let lens = [la as u64, lb as u64, 0];
            unsafe {
                (k.reference.mul_base)(full.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
            }
            let mut d = full[..kl].to_vec(); // (lo - hi) mod 2^L, and whether lo < hi
            let mut borrow = 0u64;
            for (i, w) in d.iter_mut().enumerate() {
                let h = full.get(kl + i).copied().unwrap_or(0);
                let (d1, b1) = w.overflowing_sub(h);
                let (d2, b2) = d1.overflowing_sub(borrow);
                *w = d2;
                borrow = u64::from(b1 || b2);
            }
            let neg = borrow as i64;
            let mut matched = 0usize;
            let mut matched_e = 0i64;
            for e in [-1i64, 0, 1] {
                let c = c_asm - e;
                // v = d + c modulo 2^L, with the carry (or minus the borrow) out of the top
                let mut v = d.clone();
                let ext = if c < 0 { u64::MAX } else { 0 };
                let mut carry = 0u64;
                for (i, w) in v.iter_mut().enumerate() {
                    let add = if i == 0 { c as u64 } else { ext };
                    let (d1, o1) = w.overflowing_add(add);
                    let (d2, o2) = d1.overflowing_add(carry);
                    *w = d2;
                    carry = u64::from(o1 || o2);
                }
                let co = (carry as i64) - i64::from(c < 0);
                if co - neg == e && v == o_asm {
                    matched += 1;
                    matched_e = e;
                }
            }
            assert_eq!(matched, 1, "chain against exact arithmetic, {what}");
            negative_carries += usize::from(c_asm - matched_e < 0);
            cases += 1;
        }
    }
    assert!(negative_carries > 0 && negative_carries < cases, "{negative_carries} of {cases}");
    println!("largest rounding error of a wrapped product: {worst_err}; negative carries {negative_carries} of {cases}");
}

#[test]
fn section4_parameterized_cache_blocks_and_tiled_vector_basecase() {
    let (k, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let asm = &k.active;
    let rfk = &k.reference;
    let mut rng = XorShift64::new(0x5EC4_0001_BEEF_CAFE);

    // 1. 2D block-tiled `mul_base` above `64 x 128` across `rows_max` and carry cascades.
    for &(na, nb) in &[(96usize, 80usize), (129, 65), (160, 96), (200, 150)] {
        for trial in 0..2 {
            let a: Vec<u64> = (0..na)
                .map(|_| if trial == 0 { u64::MAX } else { rng.next_u64() })
                .collect();
            let b: Vec<u64> = (0..nb)
                .map(|_| if trial == 0 { u64::MAX } else { rng.next_u64() })
                .collect();
            for rows_max in [0u64, 8, 13, 32, u64::MAX] {
                let lens = [na as u64, nb as u64, rows_max];
                let mut d_asm = vec![0xDEAD_BEEF_CAFE_BABEu64; na + nb + 2];
                let mut d_ref = vec![0xDEAD_BEEF_CAFE_BABEu64; na + nb + 2];
                unsafe {
                    (asm.mul_base)(d_asm.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                    (rfk.mul_base)(d_ref.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                }
                assert_eq!(
                    d_asm, d_ref,
                    "mul_base tiled mismatch na={na} nb={nb} trial={trial} rows_max={rows_max}"
                );
            }
        }
    }

    // 2. 2D block-tiled `sqr_base` above `64` limbs across `rows_max` and carry cascades.
    for &n in &[18usize, 64, 65, 68, 96, 135, 200] {
        for trial in 0..2 {
            let a: Vec<u64> = (0..n)
                .map(|_| if trial == 0 { u64::MAX } else { rng.next_u64() })
                .collect();
            for rows_max in [0u64, 8, 13, 32, u64::MAX] {
                let mut d_asm = vec![0xA5A5_5A5A_F0F0_0F0Fu64; 2 * n + 2];
                let mut d_ref = vec![0xA5A5_5A5A_F0F0_0F0Fu64; 2 * n + 2];
                unsafe {
                    (asm.sqr_base)(d_asm.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                    (rfk.sqr_base)(d_ref.as_mut_ptr(), a.as_ptr(), n as u64, rows_max);
                }
                assert_eq!(
                    d_asm, d_ref,
                    "sqr_base tiled mismatch n={n} trial={trial} rows_max={rows_max}"
                );
            }
        }
    }

    // 3. Parameterized `fft_fwd` and `fft_inv` across even and odd `log2(l1_sub_n)` boundaries.
    for &n_complex in &[128usize, 512, 1024, 2048, 4096] {
        let coeff_bits = 18u64;
        let in_limbs = (n_complex * 18) / 64;
        let out_limbs = 2 * in_limbs + 2;
        let a: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
        let b: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
        let (weights, inv_weights, tw, inv_tw) = build_fft_tables(n_complex);
        let scfg = FftSplitCfg {
            in_limbs: in_limbs as u64,
            n_complex: n_complex as u64,
            coeff_bits,
        };
        for l1_sub_n in [64u64, 128, 256, 512, 1024, 2048, u64::MAX] {
            let run_one = |t: &KernelTable| -> (Vec<u64>, u64) {
                let mut fa = vec![0.0f64; 2 * n_complex];
                let mut fb = vec![0.0f64; 2 * n_complex];
                let mut out = vec![0u64; out_limbs];
                let mut cc = FftCarryCfg {
                    out_limbs: out_limbs as u64,
                    n_complex: n_complex as u64,
                    coeff_bits,
                    top_carry: 0,
                    max_err: 0.0,
                    check_word: 0,
                };
                unsafe {
                    (t.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &scfg);
                    (t.fft_split)(fb.as_mut_ptr(), b.as_ptr(), weights.as_ptr(), &scfg);
                    (t.fft_fwd)(fa.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                    (t.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, l1_sub_n);
                    (t.fft_pmul)(fa.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
                    (t.fft_inv)(fa.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, l1_sub_n);
                    (t.fft_carry)(out.as_mut_ptr(), fa.as_ptr(), inv_weights.as_ptr(), &mut cc);
                }
                assert!(cc.max_err < 0.25, "FFT err={} n={n_complex} sub={l1_sub_n}", cc.max_err);
                (out, cc.check_word)
            };
            assert_eq!(
                run_one(asm),
                run_one(rfk),
                "FFT parameterized sub-block mismatch n={n_complex} l1_sub_n={l1_sub_n}"
            );
        }
    }

    // 4. Parameterized `ntt_fwd` and `ntt_inv` across `(l1_sub, l2_sub)` including odd stage gaps.
    let p0 = largest_primes_for_two_adic(42, 1)[0];
    for &n in &[256usize, 1024, 4096, 16384] {
        let (tw, inv_tw) = build_ntt_tables(p0, n, true);
        let n_inv = mod_pow_u64(n as u64, p0 - 2, p0);
        let orig: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p0) as f64).collect();
        for &(l1_sub, l2_sub) in &[
            (64u64, 256u64),
            (64, 512),
            (128, 512),
            (256, 2048),
            (512, 4096),
            (u64::MAX, u64::MAX),
        ] {
            let pr_fwd = NttPrime::with_blocks(p0, 1, l1_sub, l2_sub);
            let pr_inv = NttPrime::with_blocks(p0, n_inv, l1_sub, l2_sub);
            let mut b_asm = orig.clone();
            let mut b_ref = orig.clone();
            unsafe {
                (asm.ntt_fwd)(b_asm.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
                (rfk.ntt_fwd)(b_ref.as_mut_ptr(), tw.as_ptr(), n as u64, &pr_fwd);
            }
            assert_eq!(b_asm, b_ref, "ntt_fwd n={n} l1={l1_sub} l2={l2_sub}");
            unsafe {
                (asm.ntt_inv)(b_asm.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr_inv);
            }
            assert_eq!(b_asm, orig, "ntt_inv roundtrip n={n} l1={l1_sub} l2={l2_sub}");
        }
    }

    // 5. Parameterized `ntt_fwd_cols` & `ntt_inv_cols` with both caller scratch and null fallback.
    for &(rows, cols) in &[(64usize, 16usize), (256, 9), (512, 17), (1024, 16)] {
        let (tw, inv_tw) = build_ntt_tables(p0, rows, false);
        let r_inv = mod_pow_u64(rows as u64, p0 - 2, p0);
        let pr_fwd = NttPrime::new(p0, 1);
        let pr_inv = NttPrime::new(p0, r_inv);
        let orig: Vec<f64> = (0..(rows * cols))
            .map(|_| (rng.next_u64() % p0) as f64)
            .collect();
        let mut scratch_vec = vec![0.0f64; rows * 8 + 16];
        let aligned_scratch = ((scratch_vec.as_mut_ptr() as usize + 63) & !63) as *mut f64;
        for &(l1_r, l2_r) in &[
            (16u64, 64u64),
            (16, 128),
            (32, 128),
            (64, 256),
            (256, 4096),
            (u64::MAX, u64::MAX),
        ] {
            for use_scratch in [true, false] {
                let cfg = NttColsCfg {
                    rows: rows as u64,
                    cols: cols as u64,
                    l1_sub_rows: l1_r,
                    l2_sub_rows: l2_r,
                    scratch: if use_scratch {
                        aligned_scratch
                    } else {
                        std::ptr::null_mut()
                    },
                };
                let mut b_asm = orig.clone();
                let mut b_ref = orig.clone();
                unsafe {
                    (asm.ntt_fwd_cols)(b_asm.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
                    (rfk.ntt_fwd_cols)(b_ref.as_mut_ptr(), tw.as_ptr(), &cfg, &pr_fwd);
                }
                assert_eq!(
                    b_asm, b_ref,
                    "ntt_fwd_cols rows={rows} cols={cols} l1={l1_r} l2={l2_r} scratch={use_scratch}"
                );
                unsafe {
                    (asm.ntt_inv_cols)(b_asm.as_mut_ptr(), inv_tw.as_ptr(), &cfg, &pr_inv);
                }
                assert_eq!(
                    b_asm, orig,
                    "ntt_inv_cols roundtrip rows={rows} cols={cols} l1={l1_r} l2={l2_r} scratch={use_scratch}"
                );
            }
        }
    }
}
