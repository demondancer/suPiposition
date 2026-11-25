//! Gate 4 tooling (Section 15.4): each kernel timed in isolation on one thread, as the gate asks
//! for an edit that leaves the counts of work as they were, and the recombination and split
//! kernels also on every hardware thread at once, which is how a full run meets them (two
//! threads to a core share its execution ports: a kernel is then bound by how many operations
//! it issues, not by how long its dependency chains are). The tests only print; they are
//! ignored in a normal test run and are run on a quiet machine with
//! `cargo test --release --test kernel_timing -- --ignored --nocapture --test-threads=1`.

use std::time::Instant;

use engine::ntt::{CrtPlan, largest_primes_for_two_adic};
use engine::refk::{
    FftCarryCfg, FftSplitCfg, KernelTable, Kernels, MoveCfg, NttCrtCfg, NttCrtSegs, NttPrime,
    NttSplitCfg, NttVecCfg, build_fft_tables, build_ntt_tables,
};

struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Best of `trials` timings of `reps` calls of `f`, in nanoseconds per call.
fn best_ns(trials: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / (reps as f64));
    }
    best
}

fn reps_for(work_units: usize) -> usize {
    (40_000_000 / work_units.max(1)).clamp(3, 200_000)
}

fn tables() -> [(&'static str, KernelTable); 2] {
    let (k, rep) = Kernels::bind_and_verify(kernels::supi_bind);
    println!("bound kernels: {}/24 (mask=0x{:06x})", rep.bound_count(), rep.bound_mask);
    [("asm", k.active), ("ref", k.reference)]
}

#[test]
#[ignore]
fn time_limb_kernels() {
    let mut rng = XorShift64(0x1234_5678_9ABC_DEF1);
    for (name, t) in tables() {
        for &n in &[2usize, 4, 8, 16, 32, 64, 128] {
            let a: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
            let mut out = vec![0u64; 2 * n];
            let lens = [n as u64, n as u64, 13];
            let ns = best_ns(5, reps_for(n * n), || unsafe {
                (t.mul_base)(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
            });
            let ns_sq = best_ns(5, reps_for(n * n), || unsafe {
                (t.sqr_base)(out.as_mut_ptr(), a.as_ptr(), n as u64, 13);
            });
            println!(
                "{name} mul_base {n:>4}x{n:<4} {:>9.1} ns  {:.3} ns/limb^2 | sqr_base {:>9.1} ns  {:.3} ns/limb^2",
                ns,
                ns / ((n * n) as f64),
                ns_sq,
                ns_sq / ((n * n) as f64)
            );
        }
        for &n in &[8usize, 64, 1024, 16384] {
            let a: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
            let mut out = vec![0u64; n + 1];
            let ns_add = best_ns(5, reps_for(n), || unsafe {
                std::hint::black_box((t.add_n)(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64));
            });
            let ns_m1 = best_ns(5, reps_for(n), || unsafe {
                std::hint::black_box((t.mul_1)(out.as_mut_ptr(), a.as_ptr(), n as u64, 0x9E37_79B9_7F4A_7C15));
            });
            let ns_am1 = best_ns(5, reps_for(n), || unsafe {
                std::hint::black_box((t.addmul_1)(out.as_mut_ptr(), a.as_ptr(), n as u64, 0x9E37_79B9_7F4A_7C15));
            });
            let mut w = [0u64; 3];
            let ns_c1 = best_ns(5, reps_for(n), || unsafe {
                std::hint::black_box((t.check_n)(a.as_ptr(), n as u64, &mut w, 0));
            });
            let ns_c3 = best_ns(5, reps_for(n), || unsafe {
                std::hint::black_box((t.check_n)(a.as_ptr(), n as u64, &mut w, 3));
            });
            let nf = n as f64;
            println!(
                "{name} n={n:>6}: add_n {:.3}  mul_1 {:.3}  addmul_1 {:.3}  check_n(w0) {:.3}  check_n(all3) {:.3}  ns/limb",
                ns_add / nf,
                ns_m1 / nf,
                ns_am1 / nf,
                ns_c1 / nf,
                ns_c3 / nf
            );
        }
    }
}

#[test]
#[ignore]
fn time_ntt_transforms() {
    let p = largest_primes_for_two_adic(42, 1)[0];
    let pr = NttPrime::new(p, 1);
    let mut rng = XorShift64(0xABCD_EF01_2345_6789);
    for (name, t) in tables() {
        for &n in &[16usize, 32, 64, 128, 256, 1024, 2048, 4096, 16384, 32768, 65536, 131072, 1 << 20] {
            let (tw, inv_tw) = build_ntt_tables(p, n, false);
            let orig: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
            let mut buf = orig.clone();
            let bf = ((n / 2) * (n.ilog2() as usize)) as f64;
            let reps = reps_for(n * 8);
            let ns_f = best_ns(5, reps, || unsafe {
                (t.ntt_fwd)(buf.as_mut_ptr(), tw.as_ptr(), n as u64, &pr);
            });
            buf.copy_from_slice(&orig);
            let ns_i = best_ns(5, reps, || unsafe {
                (t.ntt_inv)(buf.as_mut_ptr(), inv_tw.as_ptr(), n as u64, &pr);
            });
            println!(
                "{name} ntt n={n:>8}: fwd {:.4} ns/bf ({:.2} G bf/s)  inv {:.4} ns/bf ({:.2} G bf/s)",
                ns_f / bf,
                bf / ns_f,
                ns_i / bf,
                bf / ns_i
            );
        }
        // Column passes on the tile shapes the sweeps use.
        for &(rows, cols) in &[(64usize, 64usize), (1024, 32), (1024, 64), (4096, 16)] {
            let (tw, _) = build_ntt_tables(p, rows, false);
            let n = rows * cols;
            let mut buf: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
            let mut col_scratch = vec![0.0f64; rows * 8 + 16];
            let aligned_ptr = ((col_scratch.as_mut_ptr() as usize + 63) & !63) as *mut f64;
            let cfg_cols = engine::refk::NttColsCfg {
                rows: rows as u64,
                cols: cols as u64,
                l1_sub_rows: 256,
                l2_sub_rows: 4096,
                scratch: aligned_ptr,
            };
            let bf = ((n / 2) * (rows.ilog2() as usize)) as f64;
            let ns = best_ns(5, reps_for(n * 8), || unsafe {
                (t.ntt_fwd_cols)(buf.as_mut_ptr(), tw.as_ptr(), &cfg_cols, &pr);
            });
            println!(
                "{name} ntt_fwd_cols {rows:>5}x{cols:<3}: {:.4} ns/bf ({:.2} G bf/s)",
                ns / bf,
                bf / ns
            );
        }
        for &n in &[4096usize, 32768] {
            let a: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
            let b: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
            let mut o = vec![0.0f64; n];
            let cfg = NttVecCfg {
                n: n as u64,
                p: pr.p,
                inv_p: pr.inv_p,
                sub_second: 0,
            };
            let ns = best_ns(5, reps_for(n * 2), || unsafe {
                (t.ntt_pmul)(o.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &cfg);
            });
            let ptrs = [a.as_ptr(), b.as_ptr(), b.as_ptr(), a.as_ptr()];
            let ns2 = best_ns(5, reps_for(n * 4), || unsafe {
                (t.ntt_pmuladd)(o.as_mut_ptr(), &ptrs, &cfg);
            });
            println!(
                "{name} ntt_pmul n={n:>6}: {:.3} ns/elem | ntt_pmuladd {:.3} ns/elem",
                ns / (n as f64),
                ns2 / (n as f64)
            );
        }
    }
}

#[test]
#[ignore]
fn time_ntt_split_and_crt() {
    let mut rng = XorShift64(0x0F0F_1234_ABCD_9876);
    for (name, t) in tables() {
        for &(k, cb) in &[(3usize, 60u64), (4, 84), (5, 108), (6, 132), (8, 180)] {
            let primes = largest_primes_for_two_adic(42, k);
            let plan = CrtPlan::build(primes);
            let n = 4096usize;
            let in_limbs = (n * (cb as usize)).div_ceil(64);
            let limbs: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
            let chunks = (cb as usize).div_ceil(48);
            let mut planes = vec![0.0f64; chunks * engine::refk::ntt_planar_stride(n) + 8];
            let mut res: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n + 8]).collect();
            // (a) direct split, one call per prime
            let ns_direct = best_ns(5, reps_for(n * k * 4), || {
                for pi in 0..k {
                    let cfg = NttSplitCfg {
                        in_limbs: in_limbs as u64,
                        start_coeff: 0,
                        count: n as u64,
                        coeff_bits: cb,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: plan.primes_f64[pi],
                        inv_p: plan.inv_primes_f64[pi],
                    };
                    unsafe { (t.ntt_split)(res[pi].as_mut_ptr(), limbs.as_ptr(), &cfg) };
                }
            });
            // (b) planar extract once, then one reduction per prime
            let ns_planar = best_ns(5, reps_for(n * k * 4), || {
                let ext = NttSplitCfg {
                    in_limbs: in_limbs as u64,
                    start_coeff: 0,
                    count: n as u64,
                    coeff_bits: cb,
                    col_stride: 0,
                    cols_per_row: 0,
                    p: 0.0,
                    inv_p: 0.0,
                };
                unsafe { (t.ntt_split)(planes.as_mut_ptr(), limbs.as_ptr(), &ext) };
                for pi in 0..k {
                    let red = NttSplitCfg {
                        in_limbs: u64::MAX,
                        start_coeff: 0,
                        count: n as u64,
                        coeff_bits: cb,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: plan.primes_f64[pi],
                        inv_p: plan.inv_primes_f64[pi],
                    };
                    unsafe {
                        (t.ntt_split)(res[pi].as_mut_ptr(), planes.as_ptr() as *const u64, &red)
                    };
                }
            });
            // CRT of valid residues: take residues of random coefficients below the capacity.
            for pi in 0..k {
                let pv = plan.primes_u64[pi];
                for j in 0..n {
                    res[pi][j] = (rng.next_u64() % pv) as f64;
                }
            }
            let out_limbs = (n * (cb as usize)).div_ceil(64) + (plan.k_words) + 4;
            let mut out = vec![0u64; out_limbs];
            let ptrs: Vec<*const f64> = res.iter().map(|v| v.as_ptr()).collect();
            let ns_crt = best_ns(5, reps_for(n * k * 8), || {
                let mut ccfg = NttCrtCfg {
                    k_primes: k as u64,
                    count: n as u64,
                    coeff_bits: cb,
                    out_limbs: out_limbs as u64,
                    signed: 0,
                    wrap_cyclic: 0,
                    primes: plan.primes_f64.as_ptr(),
                    inv_primes: plan.inv_primes_f64.as_ptr(),
                    garner_inv: plan.garner_inv.as_ptr(),
                    prefix_prods: plan.prefix_prods.as_ptr(),
                    k_words: plan.k_words as u64,
                    segs: NttCrtSegs::ONE,
                    carry_limbs: std::ptr::null_mut(),
                };
                unsafe {
                    std::hint::black_box((t.ntt_crt)(out.as_mut_ptr(), ptrs.as_ptr(), &mut ccfg));
                }
            });
            let per = (n * k) as f64;
            println!(
                "{name} k={k} cb={cb:>3}: split direct {:.3}  split planar {:.3}  crt {:.3}  ns per residue",
                ns_direct / per,
                ns_planar / per,
                ns_crt / per
            );
        }
    }
}

/// The call shapes the sweeps really use: short row segments of a column tile.
#[test]
#[ignore]
fn time_split_and_crt_in_tile_shapes() {
    let mut rng = XorShift64(0x5151_7272_9393_B4B4);
    for (name, t) in tables() {
        for &(k, cb, rows, cols, seg) in &[
            (4usize, 84u64, 1024usize, 65536usize, 32usize),
            (4, 84, 1024, 65536, 64),
            (4, 86, 16, 16384, 64),
            (5, 108, 1024, 131072, 32),
            (8, 184, 32, 16384, 64),
        ] {
            let primes = largest_primes_for_two_adic(42, k);
            let plan = CrtPlan::build(primes);
            // A small stand-in integer: the tile reads `rows / 2` strided runs of `seg` coefficients.
            let grid_cols = 4 * seg;
            let nz_rows = rows / 2;
            let in_limbs = (nz_rows * grid_cols * (cb as usize)).div_ceil(64) + 4;
            let limbs: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
            let count = nz_rows * seg;
            let chunks = (cb as usize).div_ceil(48);
            let mut planes = vec![0.0f64; chunks * engine::refk::ntt_planar_stride(count) + 8];
            let mut res: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; rows * seg + 8]).collect();
            let reps = reps_for(count * k * 4);
            let ns_direct = best_ns(5, reps, || {
                for pi in 0..k {
                    let cfg = NttSplitCfg {
                        in_limbs: in_limbs as u64,
                        start_coeff: 0,
                        count: count as u64,
                        coeff_bits: cb,
                        col_stride: grid_cols as u64,
                        cols_per_row: seg as u64,
                        p: plan.primes_f64[pi],
                        inv_p: plan.inv_primes_f64[pi],
                    };
                    unsafe { (t.ntt_split)(res[pi].as_mut_ptr(), limbs.as_ptr(), &cfg) };
                }
            });
            let ns_planar = best_ns(5, reps, || {
                let ext = NttSplitCfg {
                    in_limbs: in_limbs as u64,
                    start_coeff: 0,
                    count: count as u64,
                    coeff_bits: cb,
                    col_stride: grid_cols as u64,
                    cols_per_row: seg as u64,
                    p: 0.0,
                    inv_p: 0.0,
                };
                unsafe { (t.ntt_split)(planes.as_mut_ptr(), limbs.as_ptr(), &ext) };
                for pi in 0..k {
                    let red = NttSplitCfg {
                        in_limbs: u64::MAX,
                        start_coeff: 0,
                        count: count as u64,
                        coeff_bits: cb,
                        col_stride: 0,
                        cols_per_row: 0,
                        p: plan.primes_f64[pi],
                        inv_p: plan.inv_primes_f64[pi],
                    };
                    unsafe {
                        (t.ntt_split)(res[pi].as_mut_ptr(), planes.as_ptr() as *const u64, &red)
                    };
                }
            });
            // CRT: one call per row segment of `seg` coefficients, as sweep 3 makes them.
            for pi in 0..k {
                let pv = plan.primes_u64[pi];
                for j in 0..(rows * seg) {
                    res[pi][j] = (rng.next_u64() % pv) as f64;
                }
            }
            let seg_limbs = (seg * (cb as usize)) / 64;
            let mut out = vec![0u64; rows * seg_limbs + plan.k_words + 8];
            let mut carry = vec![0u64; plan.k_words + 2];
            let ns_crt = best_ns(5, reps_for(rows * seg * k * 8), || {
                for r in 0..rows {
                    let ptrs: [*const f64; 18] = std::array::from_fn(|pi| {
                        if pi < k {
                            unsafe { res[pi].as_ptr().add(r * seg) }
                        } else {
                            std::ptr::null()
                        }
                    });
                    let mut ccfg = NttCrtCfg {
                        k_primes: k as u64,
                        count: seg as u64,
                        coeff_bits: cb,
                        out_limbs: seg_limbs as u64,
                        signed: 0,
                        wrap_cyclic: 0,
                        primes: plan.primes_f64.as_ptr(),
                        inv_primes: plan.inv_primes_f64.as_ptr(),
                        garner_inv: plan.garner_inv.as_ptr(),
                        prefix_prods: plan.prefix_prods.as_ptr(),
                        k_words: plan.k_words as u64,
                        segs: NttCrtSegs::ONE,
                        carry_limbs: carry.as_mut_ptr(),
                    };
                    unsafe {
                        std::hint::black_box((t.ntt_crt)(
                            out.as_mut_ptr().add(r * seg_limbs),
                            ptrs.as_ptr(),
                            &mut ccfg,
                        ));
                    }
                }
            });
            let _ = cols;
            println!(
                "{name} k={k} cb={cb:>3} tile {rows}x{seg}: split direct {:.3}  split planar {:.3} ns per residue | crt per {seg}-coefficient segment {:.3} ns per residue",
                ns_direct / ((count * k) as f64),
                ns_planar / ((count * k) as f64),
                ns_crt / ((rows * seg * k) as f64)
            );
        }
    }
}

/// The root-level shape at one billion digits, on an integer far larger than the caches:
/// 512 nonzero rows of 65,536 coefficients, tiles of 32 columns, one or sixteen threads.
#[test]
#[ignore]
fn time_split_on_a_large_integer() {
    let (k, cb, rows, cols, tc) = (4usize, 84u64, 1024usize, 65536usize, 32usize);
    let nz_rows = rows / 2;
    let in_limbs = (nz_rows * cols * (cb as usize)).div_ceil(64) + 4;
    let mut rng = XorShift64(0xD00D_F00D_1234_5678);
    let limbs: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
    let primes = largest_primes_for_two_adic(42, k);
    let plan = CrtPlan::build(primes);
    let count = nz_rows * tc;
    let chunks = (cb as usize).div_ceil(48);
    let tiles = cols / tc;
    for (name, t) in tables() {
        if name == "ref" {
            continue;
        }
        for &threads in &[1usize, 8, 16] {
            let per_thread = tiles / threads;
            let t0 = Instant::now();
            let stats: Vec<(f64, f64)> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads)
                    .map(|ti| {
                        let limbs = &limbs;
                        let plan = &plan;
                        sc.spawn(move || {
                            let mut planes =
                                vec![0.0f64; chunks * engine::refk::ntt_planar_stride(count) + 8];
                            let mut res = vec![0.0f64; rows * tc + 8];
                            let (mut ns_ext, mut ns_red) = (0.0f64, 0.0f64);
                            for tile in (ti * per_thread)..((ti + 1) * per_thread) {
                                let ext = NttSplitCfg {
                                    in_limbs: in_limbs as u64,
                                    start_coeff: (tile * tc) as u64,
                                    count: count as u64,
                                    coeff_bits: cb,
                                    col_stride: cols as u64,
                                    cols_per_row: tc as u64,
                                    p: 0.0,
                                    inv_p: 0.0,
                                };
                                let a = Instant::now();
                                unsafe { (t.ntt_split)(planes.as_mut_ptr(), limbs.as_ptr(), &ext) };
                                ns_ext += a.elapsed().as_secs_f64() * 1e9;
                                let b = Instant::now();
                                for pi in 0..k {
                                    let red = NttSplitCfg {
                                        in_limbs: u64::MAX,
                                        start_coeff: 0,
                                        count: count as u64,
                                        coeff_bits: cb,
                                        col_stride: 0,
                                        cols_per_row: 0,
                                        p: plan.primes_f64[pi],
                                        inv_p: plan.inv_primes_f64[pi],
                                    };
                                    unsafe {
                                        (t.ntt_split)(res.as_mut_ptr(), planes.as_ptr() as *const u64, &red)
                                    };
                                }
                                ns_red += b.elapsed().as_secs_f64() * 1e9;
                            }
                            (ns_ext, ns_red)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let wall = t0.elapsed().as_secs_f64();
            let n_tiles = (per_thread * threads) as f64;
            let ext: f64 = stats.iter().map(|s| s.0).sum::<f64>() / n_tiles;
            let red: f64 = stats.iter().map(|s| s.1).sum::<f64>() / n_tiles;
            println!(
                "{name} split of a {} MB integer, {threads:>2} thread(s): extract {:.1} us per tile ({:.3} ns per coefficient), reduce x{k} {:.1} us per tile ({:.3} ns per residue); wall {:.3} s",
                in_limbs * 8 / (1 << 20),
                ext * 1e-3,
                ext / (count as f64),
                red * 1e-3,
                red / ((count * k) as f64),
                wall
            );
        }
    }
}

#[test]
#[ignore]
fn time_movement() {
    let p = largest_primes_for_two_adic(42, 1)[0];
    let mut rng = XorShift64(0x7777_3333_1111_5555);
    for (name, t) in tables() {
        // One row at rest, packed and unpacked, in cache.
        for &cols in &[4096usize, 32768] {
            let dense: Vec<f64> = (0..cols).map(|_| (rng.next_u64() % p) as f64).collect();
            let mut back = vec![0.0f64; cols];
            for &packed in &[0u64, 1u64] {
                let row_bytes = if packed == 1 { cols / 8 * 50 } else { cols * 8 };
                let mut outer = vec![0u8; row_bytes + 64];
                let cfg = MoveCfg {
                    rows: 1,
                    cols: cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed,
                    use_nt: 0,
                };
                let ns_s = best_ns(5, reps_for(cols * 2), || unsafe {
                    (t.scatter)(outer.as_mut_ptr(), dense.as_ptr(), &cfg);
                });
                let ns_g = best_ns(5, reps_for(cols * 2), || unsafe {
                    (t.gather)(back.as_mut_ptr(), outer.as_ptr(), &cfg);
                });
                assert_eq!(back, dense);
                println!(
                    "{name} row cols={cols:>6} packed={packed}: scatter {:.3}  gather {:.3}  ns/elem",
                    ns_s / (cols as f64),
                    ns_g / (cols as f64)
                );
            }
        }
        // One tile across many rows (strided), packed and unpacked, grid in cache.
        for &(rows, cols, grid_cols) in &[(1024usize, 32usize, 256usize), (1024, 64, 256), (64, 64, 4096)] {
            let n = rows * cols;
            let dense: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
            let mut back = vec![0.0f64; n];
            for &packed in &[0u64, 1u64] {
                let row_bytes = if packed == 1 { grid_cols / 8 * 50 } else { grid_cols * 8 };
                let mut outer = vec![0u8; rows * row_bytes + 64];
                let cfg = MoveCfg {
                    rows: rows as u64,
                    cols: cols as u64,
                    outer_stride_bytes: row_bytes as u64,
                    packed,
                    use_nt: 0,
                };
                let ns_s = best_ns(5, reps_for(n * 2), || unsafe {
                    (t.scatter)(outer.as_mut_ptr(), dense.as_ptr(), &cfg);
                });
                let ns_g = best_ns(5, reps_for(n * 2), || unsafe {
                    (t.gather)(back.as_mut_ptr(), outer.as_ptr(), &cfg);
                });
                assert_eq!(back, dense);
                println!(
                    "{name} tile {rows}x{cols} packed={packed}: scatter {:.3}  gather {:.3}  ns/elem",
                    ns_s / (n as f64),
                    ns_g / (n as f64)
                );
            }
        }
    }
}

/// `fft_carry` and `fft_split` by themselves, in nanoseconds a digit (a complex point holds
/// two): the digits of a product of balanced digits, unit weights.
#[test]
#[ignore]
fn time_fft_edges() {
    let mut rng = XorShift64(0x0F1E_2D3C_4B5A_6978);
    for (name, t) in tables() {
        for &(n, b) in &[
            (64usize, 22u64),
            (256, 21),
            (1024, 21),
            (2048, 21),
            (4096, 20),
            (8192, 19),
            (256, 16),
            (1024, 16),
            (4096, 16),
        ] {
            let two_n = 2 * n;
            let unit: Vec<f64> = (0..n).flat_map(|_| [1.0f64, 0.0]).collect();
            let wide = 1i64 << (2 * b + (two_n.ilog2() as u64) - 3).min(52);
            let buf: Vec<f64> = (0..two_n)
                .map(|_| ((rng.next_u64() % ((2 * wide + 1) as u64)) as i64 - wide) as f64)
                .collect();
            let out_limbs = (two_n * (b as usize)).div_ceil(64) + 1;
            let mut out = vec![0u64; out_limbs];
            let reps = reps_for(two_n * 4);
            let mut ccfg = FftCarryCfg {
                out_limbs: out_limbs as u64,
                n_complex: n as u64,
                coeff_bits: b,
                top_carry: 0,
                max_err: 0.0,
                check_word: 0,
            };
            let ns_carry = best_ns(7, reps, || unsafe {
                (t.fft_carry)(out.as_mut_ptr(), buf.as_ptr(), unit.as_ptr(), &mut ccfg);
            });
            let mut dummy = [0u64; 3];
            let ns_check = best_ns(7, reps, || unsafe {
                std::hint::black_box((t.check_n)(out.as_ptr(), out_limbs as u64, &mut dummy, 0));
            });
            let in_limbs = ((n * (b as usize)).saturating_sub(64) / 64).max(1);
            let a: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
            let scfg = FftSplitCfg {
                in_limbs: in_limbs as u64,
                n_complex: n as u64,
                coeff_bits: b,
            };
            let (weights, _, _, _) = build_fft_tables(n);
            let mut fa = vec![0.0f64; two_n];
            let ns_split = best_ns(7, reps, || unsafe {
                (t.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &scfg);
            });
            println!(
                "{name} N={n:>5} b={b}: fft_carry {:.3} ns/digit (of it the check word {:.3})  fft_split {:.3} ns/digit",
                ns_carry / (two_n as f64),
                ns_check / (two_n as f64),
                ns_split / (two_n as f64)
            );
        }
    }
}

/// The assembly `fft_carry` as it is bound, without the check against the reference kernel that
/// `bind_and_verify` makes: for experiments that leave parts of the kernel out to see what they
/// cost (its output is then wrong, and a checked table would hand back the reference kernel).
#[test]
#[ignore]
fn time_fft_carry_unchecked() {
    let mut ptrs = KernelTable::reference().as_raw_ptrs();
    unsafe { kernels::supi_bind(ptrs.as_mut_ptr(), engine::refk::SUPI_ABI_VERSION) };
    let t = unsafe { KernelTable::from_raw_ptrs(&ptrs) };
    let mut rng = XorShift64(0x0F1E_2D3C_4B5A_6978);
    for &(n, b) in &[(1024usize, 21u64), (4096, 20), (1024, 16)] {
        let two_n = 2 * n;
        let unit: Vec<f64> = (0..n).flat_map(|_| [1.0f64, 0.0]).collect();
        let wide = 1i64 << (2 * b + (two_n.ilog2() as u64) - 3).min(52);
        let buf: Vec<f64> = (0..two_n)
            .map(|_| ((rng.next_u64() % ((2 * wide + 1) as u64)) as i64 - wide) as f64)
            .collect();
        let out_limbs = (two_n * (b as usize)).div_ceil(64) + 1;
        let mut out = vec![0u64; out_limbs];
        let mut ccfg = FftCarryCfg {
            out_limbs: out_limbs as u64,
            n_complex: n as u64,
            coeff_bits: b,
            top_carry: 0,
            max_err: 0.0,
            check_word: 0,
        };
        let ns = best_ns(7, reps_for(two_n * 4), || unsafe {
            (t.fft_carry)(out.as_mut_ptr(), buf.as_ptr(), unit.as_ptr(), &mut ccfg);
        });
        println!("unchecked N={n:>5} b={b}: fft_carry {:.3} ns/digit", ns / (two_n as f64));
    }
}

#[test]
#[ignore]
fn time_fft_chain() {
    let mut rng = XorShift64(0x2468_ACE0_1357_9BDF);
    for (name, t) in tables() {
        for &n_complex in &[256usize, 1024, 4096, 8192, 16384, 32768] {
            let (weights, inv_weights, tw, inv_tw) = build_fft_tables(n_complex);
            let coeff_bits: u64 = if n_complex <= 2048 { 20 } else { 18 };
            let in_limbs = ((n_complex * (coeff_bits as usize)).saturating_sub(64) / 64).max(1);
            let out_limbs = 2 * in_limbs + 2;
            let a: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..in_limbs).map(|_| rng.next_u64()).collect();
            let scfg = FftSplitCfg {
                in_limbs: in_limbs as u64,
                n_complex: n_complex as u64,
                coeff_bits,
            };
            let mut fa = vec![0.0f64; 2 * n_complex];
            let mut fb = vec![0.0f64; 2 * n_complex];
            let mut fp = vec![0.0f64; 2 * n_complex];
            let mut out = vec![0u64; out_limbs];
            let reps = reps_for(n_complex * 16);
            let ns_split = best_ns(5, reps, || unsafe {
                (t.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &scfg);
            });
            unsafe { (t.fft_split)(fb.as_mut_ptr(), b.as_ptr(), weights.as_ptr(), &scfg) };
            let ns_fwd = best_ns(5, reps, || unsafe {
                (t.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
            });
            let ns_pmul = best_ns(5, reps, || unsafe {
                (t.fft_pmul)(fp.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
            });
            let ns_inv = best_ns(5, reps, || unsafe {
                (t.fft_inv)(fp.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
            });
            // A real product, start to finish.
            let mut max_err = 0.0f64;
            let ns_all = best_ns(5, reps, || unsafe {
                (t.fft_split)(fa.as_mut_ptr(), a.as_ptr(), weights.as_ptr(), &scfg);
                (t.fft_split)(fb.as_mut_ptr(), b.as_ptr(), weights.as_ptr(), &scfg);
                (t.fft_fwd)(fa.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
                (t.fft_fwd)(fb.as_mut_ptr(), tw.as_ptr(), n_complex as u64, 1024);
                (t.fft_pmul)(fp.as_mut_ptr(), fa.as_ptr(), fb.as_ptr(), n_complex as u64);
                (t.fft_inv)(fp.as_mut_ptr(), inv_tw.as_ptr(), n_complex as u64, 1024);
                let mut ccfg = FftCarryCfg {
                    out_limbs: out_limbs as u64,
                    n_complex: n_complex as u64,
                    coeff_bits,
                    top_carry: 0,
                    max_err: 0.0,
                    check_word: 0,
                };
                (t.fft_carry)(out.as_mut_ptr(), fp.as_ptr(), inv_weights.as_ptr(), &mut ccfg);
                max_err = ccfg.max_err;
            });
            let bf = ((n_complex / 2) * (n_complex.ilog2() as usize)) as f64;
            let ns_carry = ns_all - 2.0 * ns_split - 2.0 * ns_fwd - ns_pmul - ns_inv;
            println!(
                "{name} fft N={n_complex:>6} b={coeff_bits} ({in_limbs} limbs): fwd {:.4} ns/bf ({:.2} G/s) inv {:.4} ns/bf | per point: split {:.2} pmul {:.2} carry~{:.2} ns | product {:.1} us = {:.2} ns per product limb (max_err {:.3})",
                ns_fwd / bf,
                bf / ns_fwd,
                ns_inv / bf,
                ns_split / (n_complex as f64),
                ns_pmul / (n_complex as f64),
                ns_carry / (n_complex as f64),
                ns_all * 1e-3,
                ns_all / ((2 * in_limbs) as f64),
                max_err
            );
        }
    }
}

#[test]
#[ignore]
fn time_to_digits() {
    let mut rng = XorShift64(0x1111_2222_3333_4444);
    for (name, t) in tables() {
        let n = 4096usize;
        let words: Vec<u64> = (0..n)
            .map(|_| rng.next_u64() % 10_000_000_000_000_000_000u64)
            .collect();
        let mut out = vec![0u8; n * 19 + 64];
        let ns = best_ns(5, reps_for(n * 16), || unsafe {
            std::hint::black_box((t.to_digits)(out.as_mut_ptr(), words.as_ptr(), n as u64, 19));
        });
        println!("{name} to_digits: {:.3} ns per 19-digit word", ns / (n as f64));
    }
}

/// What one `ntt_crt` call costs beyond its coefficients: calls with no coefficients, with and
/// without output limbs to clear and check, beside calls of the sizes sweep 3 makes.
/// Recombination of the row segments of one column group, as the third sweep of a product does
/// it: one call per row segment against one call for all of them.
#[test]
#[ignore]
fn time_crt_row_segments() {
    let mut rng = XorShift64(0x7777_1234_5678_9ABC);
    for (name, t) in tables() {
        for &(k, cb) in &[
            (3usize, 62u64),
            (4, 84),
            (5, 100),
            (5, 108),
            (6, 130),
            (7, 152),
            (8, 180),
            (9, 200),
        ] {
            let plan = CrtPlan::build(largest_primes_for_two_adic(42, k));
            let kw = plan.k_words;
            for &seg_cols in &[32usize, 64] {
                let rows = 4096 / seg_cols;
                let seg_limbs = seg_cols * (cb as usize) / 64;
                let res: Vec<Vec<f64>> = plan
                    .primes_u64
                    .iter()
                    .map(|&pv| (0..(rows * seg_cols)).map(|_| (rng.next_u64() % pv) as f64).collect())
                    .collect();
                let ptrs: Vec<*const f64> = res.iter().map(|v| v.as_ptr()).collect();
                let mut out = vec![0u64; rows * seg_limbs + 8];
                let mut carry = vec![0u64; rows * kw];
                let mut w0s = vec![0u64; rows];
                let cfg = NttCrtCfg {
                    k_primes: k as u64,
                    count: seg_cols as u64,
                    coeff_bits: cb,
                    out_limbs: seg_limbs as u64,
                    signed: 0,
                    wrap_cyclic: 0,
                    primes: plan.primes_f64.as_ptr(),
                    inv_primes: plan.inv_primes_f64.as_ptr(),
                    garner_inv: plan.garner_inv.as_ptr(),
                    prefix_prods: plan.prefix_prods.as_ptr(),
                    k_words: kw as u64,
                    segs: NttCrtSegs::ONE,
                    carry_limbs: std::ptr::null_mut(),
                };
                let reps = if name == "asm" { 300 } else { 20 };
                let mut row_ptrs = vec![std::ptr::null::<f64>(); k];
                let per_row = best_ns(7, reps, || {
                    for r in 0..rows {
                        for pi in 0..k {
                            row_ptrs[pi] = unsafe { ptrs[pi].add(r * seg_cols) };
                        }
                        let c = unsafe { carry.as_mut_ptr().add(r * kw) };
                        for w in 0..kw {
                            unsafe { *c.add(w) = 0 };
                        }
                        let mut one = NttCrtCfg { carry_limbs: c, ..cfg };
                        unsafe {
                            w0s[r] = (t.ntt_crt)(out.as_mut_ptr().add(r * seg_limbs), row_ptrs.as_ptr(), &mut one);
                        }
                    }
                    std::hint::black_box(&mut out);
                });
                let check: (Vec<u64>, Vec<u64>, Vec<u64>) = (out.clone(), carry.clone(), w0s.clone());
                let one_call = best_ns(7, reps, || {
                    let mut many = NttCrtCfg {
                        carry_limbs: carry.as_mut_ptr(),
                        segs: NttCrtSegs {
                            segments: rows as u64,
                            res_stride: seg_cols as u64,
                            out_stride: seg_limbs as u64,
                            carry_stride: kw as u64,
                            w0: w0s.as_mut_ptr(),
                            w0_stride: 1,
                        },
                        ..cfg
                    };
                    unsafe {
                        (t.ntt_crt)(out.as_mut_ptr(), ptrs.as_ptr(), &mut many);
                    }
                    std::hint::black_box(&mut out);
                });
                assert!(check == (out.clone(), carry.clone(), w0s.clone()), "the two ways differ");
                let coeffs = (rows * seg_cols) as f64;
                println!(
                    "{name} k={k} cb={cb:>3} {rows:>3} segments of {seg_cols}: one call per segment {:6.2} ns/coeff | one call {:6.2} ns/coeff | x{:.2}",
                    per_row / coeffs,
                    one_call / coeffs,
                    per_row / one_call
                );
            }
        }
    }
}

/// The forward transform and the recombination and split kernels on 1, 8 and 16 threads at
/// once, each thread on its own in-cache data: what a thread gets when every core, and then
/// every hardware thread, runs the same kernel, as in a full run. The planner prices a product
/// from these three (butterflies, and split and recombination per coefficient for the product's
/// number of primes), so this is also the table its constants are checked against; the split
/// here is the one a two-dimensional product makes, prime by prime.
#[test]
#[ignore]
fn time_crt_and_split_on_all_threads() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    let t = &kernels.active;
    // The forward transform on the same thread counts: the unit the planner prices a product in.
    {
        let p = largest_primes_for_two_adic(42, 1)[0];
        let pr = NttPrime::new(p, 1);
        for &n in &[4096usize, 32768] {
            let (tw, _) = build_ntt_tables(p, n, false);
            let bf = ((n / 2) * (n.ilog2() as usize)) as f64;
            let mut line = format!("ntt_fwd n={n:>5}:");
            for &threads in &[1usize, 8, 16] {
                let barrier = std::sync::Barrier::new(threads);
                let per_thread: Vec<f64> = std::thread::scope(|sc| {
                    let handles: Vec<_> = (0..threads)
                        .map(|ti| {
                            let (tw, pr, barrier) = (&tw, &pr, &barrier);
                            sc.spawn(move || {
                                let mut rng = XorShift64(0x0BAD_5EED_1234_5678 + ti as u64);
                                let mut buf: Vec<f64> = (0..n).map(|_| (rng.next_u64() % p) as f64).collect();
                                barrier.wait();
                                best_ns(5, reps_for(n * 8), || unsafe {
                                    (t.ntt_fwd)(buf.as_mut_ptr(), tw.as_ptr(), n as u64, pr);
                                }) / bf
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let ns = per_thread.iter().sum::<f64>() / (per_thread.len() as f64);
                line += &format!(" | {threads:>2} thr: {ns:.4} ns/bf");
            }
            println!("{line}");
        }
    }
    // Every number of primes a product uses, at the shortest and the longest coefficient it serves.
    for &(k, cb) in &[
        (3usize, 50u64),
        (3, 62),
        (4, 70),
        (4, 86),
        (5, 94),
        (5, 112),
        (6, 118),
        (6, 136),
        (7, 142),
        (7, 162),
        (8, 166),
        (8, 186),
        (9, 190),
        (9, 208),
        (9, 214),
        (10, 216),
        (10, 232),
        (12, 262),
        (12, 280),
        (14, 310),
        (14, 330),
        (16, 360),
        (16, 378),
        (18, 406),
        (18, 420),
    ] {
        let plan = CrtPlan::build(largest_primes_for_two_adic(42, k));
        let kw = plan.k_words;
        // Ten primes and more are far slower a coefficient: fewer repeats, one segment length.
        let (trials, reps) = if k >= 10 { (3usize, 60usize) } else { (5, 400) };
        let seg_choices: &[usize] = if k >= 10 { &[32] } else { &[32, 64] };
        for &seg_cols in seg_choices {
            let rows = 4096 / seg_cols;
            let seg_limbs = seg_cols * (cb as usize) / 64;
            let mut line = format!("k={k} cb={cb:>3} segments of {seg_cols}:");
            for &threads in &[1usize, 8, 16] {
                let barrier = std::sync::Barrier::new(threads);
                let per_thread: Vec<(f64, f64)> = std::thread::scope(|sc| {
                    let handles: Vec<_> = (0..threads)
                        .map(|ti| {
                            let plan = &plan;
                            let barrier = &barrier;
                            sc.spawn(move || {
                                let mut rng = XorShift64(0x1234_5678_9ABC_DEF1 + ti as u64);
                                let res: Vec<Vec<f64>> = plan
                                    .primes_u64
                                    .iter()
                                    .map(|&pv| {
                                        (0..(rows * seg_cols)).map(|_| (rng.next_u64() % pv) as f64).collect()
                                    })
                                    .collect();
                                let ptrs: Vec<*const f64> = res.iter().map(|v| v.as_ptr()).collect();
                                let limbs: Vec<u64> = (0..(rows * seg_limbs + 8)).map(|_| rng.next_u64()).collect();
                                let mut out = vec![0u64; rows * seg_limbs + 8];
                                let mut carry = vec![0u64; rows * kw];
                                let mut w0s = vec![0u64; rows];
                                let mut grid = vec![0.0f64; rows * seg_cols];
                                barrier.wait();
                                let crt_ns = best_ns(trials, reps, || {
                                    let mut many = NttCrtCfg {
                                        k_primes: k as u64,
                                        count: seg_cols as u64,
                                        coeff_bits: cb,
                                        out_limbs: seg_limbs as u64,
                                        signed: 0,
                                        wrap_cyclic: 0,
                                        primes: plan.primes_f64.as_ptr(),
                                        inv_primes: plan.inv_primes_f64.as_ptr(),
                                        garner_inv: plan.garner_inv.as_ptr(),
                                        prefix_prods: plan.prefix_prods.as_ptr(),
                                        k_words: kw as u64,
                                        segs: NttCrtSegs {
                                            segments: rows as u64,
                                            res_stride: seg_cols as u64,
                                            out_stride: seg_limbs as u64,
                                            carry_stride: kw as u64,
                                            w0: w0s.as_mut_ptr(),
                                            w0_stride: 1,
                                        },
                                        carry_limbs: carry.as_mut_ptr(),
                                    };
                                    unsafe {
                                        (t.ntt_crt)(out.as_mut_ptr(), ptrs.as_ptr(), &mut many);
                                    }
                                    std::hint::black_box(&mut out);
                                });
                                // the split of the same number of coefficients for every prime
                                let split_ns = best_ns(trials, reps, || {
                                    for pi in 0..k {
                                        let scfg = NttSplitCfg {
                                            in_limbs: (rows * seg_limbs) as u64,
                                            start_coeff: 0,
                                            count: (rows * seg_cols) as u64,
                                            coeff_bits: cb,
                                            col_stride: 0,
                                            cols_per_row: 0,
                                            p: plan.primes_f64[pi],
                                            inv_p: plan.inv_primes_f64[pi],
                                        };
                                        unsafe {
                                            (t.ntt_split)(grid.as_mut_ptr(), limbs.as_ptr(), &scfg);
                                        }
                                    }
                                    std::hint::black_box(&mut grid);
                                });
                                let coeffs = (rows * seg_cols) as f64;
                                (crt_ns / coeffs, split_ns / coeffs)
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let n = per_thread.len() as f64;
                let crt = per_thread.iter().map(|p| p.0).sum::<f64>() / n;
                let split = per_thread.iter().map(|p| p.1).sum::<f64>() / n;
                line += &format!(" | {threads:>2} thr: crt {crt:5.2} split {split:5.2} ns/coeff");
            }
            println!("{line}");
        }
    }
}

/// What memory gives on this machine: sequential reads, cached writes, streaming writes and
/// short strided runs, on 8 and on 16 threads at once, each thread on its own 256 MiB. The
/// sweeps of a large product are judged against these figures.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore]
fn time_memory_bandwidth() {
    use std::arch::x86_64::*;
    const BYTES: usize = 256 << 20;
    let run = |threads: usize, what: usize| -> f64 {
        let barrier = std::sync::Barrier::new(threads);
        let secs: Vec<f64> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|ti| {
                    let barrier = &barrier;
                    sc.spawn(move || {
                        let mut buf = vec![ti as u64 + 1; BYTES / 8];
                        let mut dst = vec![0u64; BYTES / 8];
                        // touch everything first
                        for w in dst.iter_mut().step_by(512) {
                            *w = 1;
                        }
                        barrier.wait();
                        let mut best = f64::INFINITY;
                        for _ in 0..3 {
                            let t = Instant::now();
                            match what {
                                0 => {
                                    let mut s = 0u64;
                                    for &w in buf.iter() {
                                        s = s.wrapping_add(w);
                                    }
                                    std::hint::black_box(s);
                                }
                                1 => {
                                    for w in buf.iter_mut() {
                                        *w = 7;
                                    }
                                    std::hint::black_box(&mut buf);
                                }
                                2 => unsafe {
                                    let v = _mm512_set1_epi64(9);
                                    let p = dst.as_mut_ptr() as *mut u8;
                                    let mut off = (64 - (p as usize & 63)) & 63;
                                    while off + 64 <= BYTES {
                                        _mm512_stream_si512(p.add(off) as *mut _, v);
                                        off += 64;
                                    }
                                    _mm_sfence();
                                },
                                3 => {
                                    dst.copy_from_slice(&buf);
                                    std::hint::black_box(&mut dst);
                                }
                                _ => {
                                    // 1024 interleaved streams, 608 bytes from each in turn: the
                                    // way a tile of sweep 1 reads its operand
                                    let rows = 1024usize;
                                    let stride = BYTES / 8 / rows;
                                    let run_w = 76usize;
                                    let mut s = 0u64;
                                    let mut c = 0usize;
                                    while c + run_w <= stride {
                                        for r in 0..rows {
                                            let base = r * stride + c;
                                            for &w in &buf[base..base + run_w] {
                                                s = s.wrapping_add(w);
                                            }
                                        }
                                        c += run_w;
                                    }
                                    std::hint::black_box(s);
                                }
                            }
                            best = best.min(t.elapsed().as_secs_f64());
                        }
                        best
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let avg = secs.iter().sum::<f64>() / (secs.len() as f64);
        (threads as f64) * (BYTES as f64) / avg / 1e9
    };
    for (what, name) in [
        (0usize, "sequential read"),
        (1, "write through the cache"),
        (2, "streaming write"),
        (3, "copy"),
        (4, "1024 interleaved streams, 608 bytes at a time"),
    ] {
        println!(
            "{name:<48} 1 thread {:6.1} GB/s | 8 threads {:6.1} GB/s | 16 threads {:6.1} GB/s",
            run(1, what),
            run(8, what),
            run(16, what)
        );
    }
}

/// What a butterfly costs on this machine's vector pipes, on an array that stays in the first
/// cache: the present form (every value brought back into [0, p) at once: 14 operations), the
/// lazy form (the product reduced to within about p/2 of zero, sums and differences left as
/// they are: 8 operations), and the reduction a lazy pass ends with (3 operations a value).
/// Arithmetic only: no claim that the values are right, which is the kernels' and gate 1's
/// business.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore]
fn time_butterfly_forms() {
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn canonical(u: *mut f64, x: *mut f64, w: *const f64, n: usize, p: __m512d, inv: __m512d) {
        unsafe {
            let zero = _mm512_setzero_pd();
            let mut i = 0;
            while i < n {
                let uu = _mm512_loadu_pd(u.add(i));
                let xx = _mm512_loadu_pd(x.add(i));
                let ww = _mm512_loadu_pd(w.add(i));
                let h = _mm512_mul_pd(xx, ww);
                let q = _mm512_roundscale_pd::<8>(_mm512_mul_pd(h, inv));
                let l = _mm512_fmsub_pd(xx, ww, h);
                let mut v = _mm512_add_pd(_mm512_fnmadd_pd(q, p, h), l);
                let neg = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(v, zero);
                v = _mm512_mask_add_pd(v, neg, v, p);
                let mut s = _mm512_add_pd(uu, v);
                let ge = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(s, p);
                s = _mm512_mask_sub_pd(s, ge, s, p);
                let mut d = _mm512_sub_pd(uu, v);
                let lt = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(d, zero);
                d = _mm512_mask_add_pd(d, lt, d, p);
                _mm512_storeu_pd(u.add(i), s);
                _mm512_storeu_pd(x.add(i), d);
                i += 8;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn lazy(u: *mut f64, x: *mut f64, w: *const f64, n: usize, p: __m512d, inv: __m512d) {
        unsafe {
            let mut i = 0;
            while i < n {
                let uu = _mm512_loadu_pd(u.add(i));
                let xx = _mm512_loadu_pd(x.add(i));
                let ww = _mm512_loadu_pd(w.add(i));
                let h = _mm512_mul_pd(xx, ww);
                let q = _mm512_roundscale_pd::<8>(_mm512_mul_pd(h, inv));
                let l = _mm512_fmsub_pd(xx, ww, h);
                let v = _mm512_add_pd(_mm512_fnmadd_pd(q, p, h), l);
                _mm512_storeu_pd(u.add(i), _mm512_add_pd(uu, v));
                _mm512_storeu_pd(x.add(i), _mm512_sub_pd(uu, v));
                i += 8;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn reduce(u: *mut f64, n: usize, p: __m512d, inv: __m512d) {
        unsafe {
            let mut i = 0;
            while i < n {
                let a = _mm512_loadu_pd(u.add(i));
                let q = _mm512_roundscale_pd::<8>(_mm512_mul_pd(a, inv));
                _mm512_storeu_pd(u.add(i), _mm512_fnmadd_pd(q, p, a));
                i += 8;
            }
        }
    }
    let n = 1024usize; // three arrays of 8 KiB
    let pv = ((1023u64 << 40) + 1) as f64;
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    let mk = |rng: &mut XorShift64| -> Vec<f64> { (0..n).map(|_| (rng.next_u64() % (1u64 << 49)) as f64).collect() };
    let (mut u, mut x, w) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
    let (u0, x0) = (u.clone(), x.clone());
    unsafe {
        let p = _mm512_set1_pd(pv);
        let inv = _mm512_set1_pd(1.0 / pv);
        let reps = 20_000usize;
        let canon_ns = best_ns(9, reps, || canonical(u.as_mut_ptr(), x.as_mut_ptr(), w.as_ptr(), n, p, inv));
        u.copy_from_slice(&u0);
        x.copy_from_slice(&x0);
        // a lazy pass with its closing reduction, so that the values stay bounded over the repeats
        let lazy_ns = best_ns(9, reps, || {
            lazy(u.as_mut_ptr(), x.as_mut_ptr(), w.as_ptr(), n, p, inv);
            reduce(u.as_mut_ptr(), n, p, inv);
            reduce(x.as_mut_ptr(), n, p, inv);
        });
        u.copy_from_slice(&u0);
        x.copy_from_slice(&x0);
        let red_ns = best_ns(9, reps, || {
            reduce(u.as_mut_ptr(), n, p, inv);
            reduce(x.as_mut_ptr(), n, p, inv);
        });
        let per = n as f64;
        println!(
            "per butterfly: present form {:.3} ns | lazy form {:.3} ns | closing reduction of both values {:.3} ns | a 4-level pass: present {:.3}, lazy {:.3} ns",
            canon_ns / per,
            (lazy_ns - red_ns) / per,
            red_ns / per,
            4.0 * canon_ns / per,
            4.0 * (lazy_ns - red_ns) / per + red_ns / per
        );
    }
}

/// The shapes of the classical products of a run of 10^9 digits (long limbs, short limbs,
/// thousands of calls), from a count of every `mul_base` call of one run with the first
/// crossover at 64 limbs: the bins that hold at least a quarter percent of the calls or of the
/// limb products (96.3% of 50.4 million calls, 93.3% of 24.3 billion limb products; the
/// products of shorter operands are made without the kernel).
const MUL_BASE_SHAPES_1B: &[(usize, usize, u32)] = &[
    (7, 7, 2400), (8, 7, 2412), (8, 8, 1153), (9, 7, 4236), (9, 8, 3544), (9, 9, 2908),
    (10, 5, 1291), (10, 9, 156), (11, 5, 129), (11, 6, 2688), (11, 10, 763), (12, 10, 508),
    (12, 11, 2693), (15, 13, 298), (15, 14, 524), (16, 14, 2075), (16, 15, 1735), (16, 16, 708),
    (17, 9, 172), (17, 16, 2545), (17, 17, 134), (18, 10, 1186), (19, 11, 2632), (20, 20, 160),
    (21, 21, 352), (22, 22, 853), (23, 23, 653), (28, 28, 174), (29, 29, 386), (30, 29, 85),
    (30, 30, 921), (31, 30, 185), (31, 31, 184), (32, 27, 73), (33, 20, 101), (33, 21, 106),
    (33, 28, 176), (34, 21, 246), (34, 22, 239), (34, 29, 423), (35, 22, 601), (35, 23, 536),
    (35, 30, 1013), (35, 31, 112), (36, 23, 115), (36, 31, 94), (40, 40, 63), (41, 41, 75),
    (42, 42, 107), (43, 43, 165), (44, 44, 261), (45, 45, 326), (55, 55, 32), (56, 56, 56),
    (57, 57, 76), (58, 58, 118), (59, 58, 21), (59, 59, 182), (60, 59, 32), (60, 60, 281),
    (61, 60, 50), (61, 61, 111), (64, 39, 26), (65, 40, 39), (65, 55, 27), (66, 41, 60),
    (66, 56, 42), (66, 57, 23), (67, 42, 91), (67, 57, 66), (67, 58, 34), (68, 43, 139),
    (68, 58, 103), (68, 59, 51), (69, 43, 26), (69, 44, 212), (69, 59, 161), (69, 60, 76),
    (70, 44, 43), (70, 45, 322), (70, 60, 252), (70, 61, 114),
];

/// `mul_base` on the shapes of a real run, on one thread and on every hardware thread at once
/// (a run meets it on all of them, two to a core). Prints what the calls of the table cost in
/// thread-seconds, by the length of the short operand, some shapes one by one, and the same
/// calls in the mixed order of a run against their times in isolation. With `SUPI_TIME_SHAPES`
/// set it also prints every shape's time (`shape <table> <threads> <long> <short> <ns>`).
#[test]
#[ignore]
fn time_mul_base_shapes() {
    let (kernels, _) = Kernels::bind_and_verify(kernels::supi_bind);
    // The crossover the calls are given: `SUPI_ROWS_MAX` (0: no rows the body can spare; a
    // large number: all rows), else the usual one.
    let rows_max: u64 = std::env::var("SUPI_ROWS_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(13);
    for (name, t) in [("asm", &kernels.active), ("ref", &kernels.reference)] {
        if name == "ref" && std::env::var_os("SUPI_TIME_REF").is_none() {
            continue;
        }
        for &threads in &[1usize, 16] {
            let barrier = std::sync::Barrier::new(threads);
            let per_thread: Vec<(Vec<f64>, f64, f64)> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..threads)
                    .map(|ti| {
                        let barrier = &barrier;
                        sc.spawn(move || {
                            let mut rng = XorShift64(0x51A9_E5_0F_1234_5678 + ti as u64);
                            let a: Vec<u64> = (0..4096).map(|_| rng.next_u64()).collect();
                            let b: Vec<u64> = (0..4096).map(|_| rng.next_u64()).collect();
                            let mut out = vec![0u64; 8192];
                            barrier.wait();
                            let ns: Vec<f64> = MUL_BASE_SHAPES_1B
                                .iter()
                                .map(|&(hi, lo, _)| {
                                    let lens = [hi as u64, lo as u64, rows_max];
                                    best_ns(5, reps_for(hi * lo * 4), || unsafe {
                                        (t.mul_base)(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &lens);
                                    })
                                })
                                .collect();
                            // The calls in a mixed order, shapes drawn by their share of the calls,
                            // operands and results at moving places in their buffers.
                            let total: u64 = MUL_BASE_SHAPES_1B.iter().map(|s| s.2 as u64).sum();
                            let seq: Vec<(usize, usize, usize, usize)> = (0..8192)
                                .map(|_| {
                                    let mut pick = rng.next_u64() % total;
                                    let mut idx = 0usize;
                                    for (i, s) in MUL_BASE_SHAPES_1B.iter().enumerate() {
                                        if pick < s.2 as u64 {
                                            idx = i;
                                            break;
                                        }
                                        pick -= s.2 as u64;
                                    }
                                    let (hi, lo, _) = MUL_BASE_SHAPES_1B[idx];
                                    (idx, hi, lo, (rng.next_u64() % 3000) as usize)
                                })
                                .collect();
                            let mixed = best_ns(7, 4, || {
                                for &(_, hi, lo, off) in &seq {
                                    let lens = [hi as u64, lo as u64, rows_max];
                                    unsafe {
                                        (t.mul_base)(
                                            out.as_mut_ptr().add(off),
                                            a.as_ptr().add(off),
                                            b.as_ptr().add(off / 2),
                                            &lens,
                                        );
                                    }
                                }
                            });
                            let apart: f64 = seq.iter().map(|&(idx, ..)| ns[idx]).sum();
                            (ns, mixed, apart)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let n_thr = per_thread.len() as f64;
            let ns: Vec<f64> = (0..MUL_BASE_SHAPES_1B.len())
                .map(|i| per_thread.iter().map(|p| p.0[i]).sum::<f64>() / n_thr)
                .collect();
            let classes: [(usize, usize); 6] = [(1, 8), (9, 13), (14, 19), (20, 32), (33, 48), (49, 128)];
            let mut by_class = [0.0f64; 6];
            let mut prod_class = [0.0f64; 6];
            for (i, &(hi, lo, kcalls)) in MUL_BASE_SHAPES_1B.iter().enumerate() {
                let c = classes.iter().position(|&(l, h)| lo >= l && lo <= h).unwrap();
                by_class[c] += ns[i] * (kcalls as f64) * 1e3 * 1e-9;
                prod_class[c] += (hi * lo) as f64 * (kcalls as f64) * 1e3;
            }
            let mut line = format!(
                "{name} {threads:>2} thr: the calls of the table take {:.2} thread-s |",
                by_class.iter().sum::<f64>()
            );
            for (c, &(l, h)) in classes.iter().enumerate() {
                line += &format!(
                    " short {l}-{h}: {:.2} s ({:.3} ns a limb product) |",
                    by_class[c],
                    by_class[c] * 1e9 / prod_class[c]
                );
            }
            println!("{line}");
            let mixed = per_thread.iter().map(|p| p.1).sum::<f64>() / n_thr;
            let apart = per_thread.iter().map(|p| p.2).sum::<f64>() / n_thr;
            println!(
                "{name} {threads:>2} thr: 8192 calls in mixed order {:.1} us, the same calls timed apart {:.1} us (x{:.3})",
                mixed * 1e-3,
                apart * 1e-3,
                mixed / apart
            );
            let mut line = format!("{name} {threads:>2} thr, ns a call:");
            for &(hi, lo) in &[(7usize, 7usize), (9, 7), (9, 9), (12, 11), (16, 14), (17, 16), (22, 22), (30, 30), (35, 30), (45, 45), (60, 60), (70, 60)] {
                if let Some(i) = MUL_BASE_SHAPES_1B.iter().position(|s| s.0 == hi && s.1 == lo) {
                    line += &format!(" {hi}x{lo} {:.1}", ns[i]);
                }
            }
            println!("{line}");
            if std::env::var_os("SUPI_TIME_SHAPES").is_some() {
                for (i, &(hi, lo, _)) in MUL_BASE_SHAPES_1B.iter().enumerate() {
                    println!("shape {name} {threads} {hi} {lo} {:.1}", ns[i]);
                }
            }
        }
    }
}
