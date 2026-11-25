# suPiposition

**A multithreaded program for computing π to billions of decimal digits on shared-memory systems.**

## Abstract

suPiposition computes π with the Chudnovsky series [1], evaluated by binary splitting [2, 3]. Large products are computed with a floating-point fast Fourier transform (FFT) or with a multi-prime number-theoretic transform (NTT) [4, 5], organized as a cache-blocked, three-pass (column, row, column) transform related to Bailey's four-step method [6]. The final division and square root are obtained from a single Newton–Raphson iteration for the reciprocal square root, and radix conversion uses a divide-and-conquer algorithm [13]. All inner loops are confined to 24 hand-written assembly kernels that are bound at startup and validated against portable reference implementations.

No block size, crossover point, or thread count is fixed in the source. Every such parameter is derived at startup from the discovered hardware topology (cores, hardware threads, each cache and the threads that share it, NUMA nodes, and available memory) and from a short calibration run, in the tradition of self-adapting numerical software [7, 8]. Every product is verified by a residue checksum, a form of algorithm-based fault tolerance [9].

On the system evaluated in Section 5, suPiposition computes 10⁹ to 5 × 10⁹ digits 1.22 to 1.33 times faster than y-cruncher v0.8.7 [10] in start-to-end time, and 1.15 to 1.24 times faster against that program's reported computation time. The digits of the two programs are identical.

The design is specified in [DESIGN.md](DESIGN.md).

**Project status.** This repository contains the source as of 2026-11-25, published at the author's direction. It does not yet constitute a release in the sense of Section 15 of the design, which requires a speedup of at least 1.3 over y-cruncher in memory at 10¹⁰ to 10¹¹ digits and of at least 1.5 out of core, measured on a single system. Neither criterion has been evaluated: the system available for this work holds at most 5 × 10⁹ digits in memory, and the out-of-core path is incomplete (Section 8).

## 1. System Requirements

- **Processor.** An x86-64 processor at the x86-64-v4 microarchitecture level (AVX-512 F, CD, BW, DQ, and VL) with the ADX extension. On other x86-64 processors the program runs correctly on its portable reference kernels, at substantially lower performance.
- **Toolchain.** Rust 1.100 or later. There are no other build or run-time dependencies.
- **Operating system.** Windows 11 or Linux. Section 7 describes what has been validated on each, and the state of the macOS port.
- **Memory.** Approximately 7.2 GiB per 10⁹ digits (Section 5.4). A run that cannot fit in the memory available to the process is rejected at startup.

## 2. Building and Running

```
cargo build --release
target/release/suPiposition 1000000000 -o pi.txt
```

| Option | Description |
|---|---|
| `-o <file>` | Output file for the digits |
| `--format text\|packed19` | Output format (default: `text`) |
| `--ssd <path>[@<size>]` | Adds an SSD file or device for checkpoints and for data that does not fit in memory (repeatable) |
| `--discard` | Discards checkpoint records found on the SSDs |
| `--no-vram` | Disables the use of discrete-GPU memory as storage |
| `--ref-kernels` | Forces the portable reference kernels |
| `--plan-only` | Prints the hardware profile, the execution plan, and the least memory with which the run is admitted, without running |
| `--profile <cores,threads,l1,l2,l3,ram>` | Plans and runs for a synthetic hardware profile instead of the detected one (sizes in bytes; L1 and L2 per core; L3 and memory in total). Writing L3 as `<n>x<size>` describes `n` separate last-level caches of that size, each with its own NUMA node, as in `256,256,48K,2M,4x320M,1T`. Worker threads of such a run are not pinned |
| `--calibration <...>` | Supplies calibration figures instead of measuring them, which makes a plan deterministic |

Each run reports the wall-clock time of every stage and a 64-bit hash of the digits (`Dec Hash`). The hash is the one y-cruncher prints, so that the results of the two programs can be compared without comparing files.

## 3. Method

The computation has three stages.

1. **Series.** The Chudnovsky series is summed by binary splitting over a balanced tree. At planner-selected levels, a common factor with a closed form is divided out of the recurrence, which reduces the size of the largest operands (DESIGN.md, Section 4).
2. **Final division and square root.** One Newton–Raphson iteration for the reciprocal square root yields both, with wrapped (cyclic) products where part of the result is known in advance.
3. **Radix conversion.** A divide-and-conquer conversion from binary to decimal, in which each node is a middle product by a cached power of ten.

**Multiplication.** Products are computed by the schoolbook method, by Karatsuba's method, by a complex floating-point FFT whose rounding error is checked against a bound on every product, or by an NTT over 3 to 18 primes below 2⁵⁰, with residues held as integer-valued doubles and modular reduction performed by fused multiply-add. The crossover points are measured at startup.

**Calibration and planning.** At startup the program reads the hardware topology from the operating system and measures a small set of calibration figures: kernel throughput, memory bandwidth, and the crossover points. A planner then selects, for every product, the method, the number of primes, the coefficient length, and the transform's block decomposition so that each working set fits the worker's share of the corresponding cache level. The plan classifies each phase as compute-bound or memory-bandwidth-bound, the two regimes of the Roofline model [11], and trades arithmetic for memory traffic accordingly.

**Parallelism.** A fork-join scheduler with work stealing [12] runs one worker thread per hardware thread. At the lower levels of the splitting tree, each subtree is evaluated by a single worker; at the upper levels, each product is itself parallelized across primes and transform blocks.

**Kernels.** The 24 kernels (multiprecision primitives, FFT and NTT passes, data movement, digit conversion, and checksums) are implemented in one assembly file per architecture and are reached through a table. At startup each assembly kernel is compared with its portable reference on random and boundary inputs, and a kernel that differs is replaced by its reference.

## 4. Verification and Validation

**Run-time checking.** Every number carries a checksum: its value modulo the Mersenne prime 2⁶¹ − 1. Because the checksum of a product equals the product of the operands' checksums, each product is verified as soon as it is complete. A failed check is first corrected in place where the error is confined to a single 64-bit word, then recomputed, then recovered from the last checkpoint (DESIGN.md, Section 11).

**Test suite.**

```
cargo test --release
```

The suite comprises four gates, in the terminology of the design.

| Gate | Property | Method |
|---|---|---|
| 1 | Correctness | Each kernel against its reference, bit for bit; complete runs against known digits |
| 2 | Correctness at every scale | Plans for synthetic profiles from a single core to the largest profile the program can describe; complete runs planned for synthetic systems of 256 to 1,024 worker threads under 4 to 64 last-level caches; memory admission (Section 6); fault injection for every recovery path |
| 3 | Checkpoint and restart | Interruption and resumption at every stage |
| 4 | Performance accounting | Deterministic plans and exact counts of work and of bytes moved under fixed profiles and calibration figures |

Two tests of gate 1 compare the output with that of y-cruncher itself. They require its Windows distribution, unpacked as `y-cruncher v0.8.7.9547b/` beside the sources; it is not part of this repository. Where it is absent, those two tests report that the comparison was skipped, and pass.

**A limitation of checksums, and one defect found.** A product checksum establishes that a product was computed correctly. It does not establish that the correct operands were supplied. One defect of this kind has been found and corrected: under a synthetic profile of 256 cores and four last-level caches, one merge of a 10⁸-digit run multiplied by an operand that it had already handed to another worker, and the output was incorrect from digit 90,965,871 onward although every check passed. That configuration is now a regression test. The defect was found only by sweeping synthetic profiles, which is the reason such sweeps are part of the validation reported in Section 6.

## 5. Performance Evaluation

### 5.1 Experimental Setup

|---|---|
| Processor | AMD Ryzen 7 9850X3D: 8 cores, 16 hardware threads, 96 MiB L3 |
| Memory | 48 GB |
| Operating systems | Windows 11; Fedora Linux 45 under WSL 3.0.5.0 (kernel 6.18) on the same hardware |
| Compiler | Rust 1.100.0, release profile |
| Baseline | y-cruncher v0.8.7.9547b (binary `24-ZN5 ~ Komari`, selected by the program) on the same system under Windows 11 |

Times are those that each program reports. For suPiposition this is the wall-clock time from startup to completion, which includes calibration, all run-time checks, and writing the digits to an SSD. All measurements were taken on 2026-11-25.

### 5.2 Time to Solution

| Digits | suPiposition | y-cruncher, start to end | Ratio | y-cruncher, computation | Ratio |
|---:|---:|---:|---:|---:|---:|
| 10⁸ | 1.155 s | 1.174 s | 1.02 | 1.045 s | 0.90 |
| 10⁹ | 13.808 s | 18.339 s | 1.33 | 17.123 s | 1.24 |
| 2.5 × 10⁹ | 41.895 s | 51.505 s | 1.23 | 48.312 s | 1.15 |
| 5 × 10⁹ | 92.415 s | 113.019 s | 1.22 | 106.285 s | 1.15 |

Each ratio is the y-cruncher time divided by the suPiposition time, so a value above 1 favors suPiposition.

- **suPiposition:** the time defined in Section 5.1.
- **y-cruncher, start to end:** its reported start-to-end wall time, which includes its verification of the base conversion. y-cruncher was run without digit output.
- **y-cruncher, computation:** its reported total computation time, which excludes that verification and the program's setup. This is the more conservative baseline.

At 10⁹ to 5 × 10⁹ digits, suPiposition is faster by a factor of 1.22 to 1.33 in start-to-end time and of 1.15 to 1.24 against y-cruncher's computation time. At 10⁸ digits the two programs are equal in start-to-end time to within 2%, and y-cruncher's computation time is the shorter by about 10%.

The three suPiposition runs at each size took 1.168, 1.155, and 1.162 s; 14.318, 14.034, and 13.808 s; 43.249, 41.895, and 42.218 s; and 93.882, 94.097, and 92.415 s, a spread of 1 to 4%. The y-cruncher figures are single runs and are subject to similar variation.

Two conditions differed between the programs. y-cruncher ran on locked 2 MiB pages, whereas suPiposition's request for large pages was refused under the same account, and it ran on 4 KiB pages. y-cruncher's working memory was 0.47, 4.71, 11.8, and 22.3 GiB at the four sizes, less than suPiposition requires (Section 5.4).

Both programs produce the same digits. The 64-bit digit hashes printed by both are:

| Digits | Digit hash |
|---:|---|
| 10⁸ | 199405406068015690 |
| 10⁹ | 807692683230452073 |
| 2.5 × 10⁹ | 1227538786604942620 |
| 5 × 10⁹ | 917012961347625551 |

Under Linux (WSL 3) on the same hardware, 10⁹ digits take 14.8 to 15.7 s, against 13.9 to 14.9 s under Windows in alternating runs the same day, and 2.5 × 10⁹ digits take 44.6 s. These figures include the overhead of virtualization and should not be taken as representative of Linux on bare metal.

### 5.3 Strong Scaling

10⁹ digits, with worker threads restricted by processor affinity (Windows) or by `taskset` (Linux). One hardware thread per core is used except in the last row.

| Cores × threads | Windows time | Speedup | Parallel efficiency | Linux (WSL 3) time | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 × 1 | 103.9 s | 1.00 | 100% | 108.3 s | 1.00 |
| 2 × 1 | 53.5 s | 1.94 | 97% | 55.9 s | 1.94 |
| 4 × 1 | 27.6 s | 3.76 | 94% | 28.8 s | 3.76 |
| 8 × 1 | 15.8 s | 6.59 | 82% | 19.0 s | 5.69 |
| 8 × 2 | 13.9 s | 7.47 | | 14.8 s | 7.34 |

The single-thread run executes at a higher clock frequency than the eight-core runs, so the efficiencies understate the scaling of the algorithm. The lower eight-core figure under WSL reflects the hypervisor's placement of virtual processors, which the guest cannot control.

### 5.4 Memory Footprint

The planner admits a run only if its peak memory commitment fits the memory available to the process (Section 6). The table gives, for each problem size, the largest control-group memory limit tested at which the run was rejected at startup and the smallest tested at which it completed, on Linux with swap disabled.

| Digits | Rejected at | Completed at | Per 10⁹ digits |
|---:|---:|---:|---:|
| 10⁸ | 1,400 MiB | 1,500 MiB | |
| 10⁹ | 7.0 GiB | 7,400 MiB | 7.2 GiB |
| 2.5 × 10⁹ | 16 GiB | 18 GiB | 7.2 GiB |

At 10⁸ digits the footprint is dominated by per-thread scratch buffers, which do not grow with the problem size. No run in these experiments was terminated by the kernel for exceeding its limit.

The planner reports the least memory with which it admits a run (`--plan-only`), which is the figure to request from a resource manager. On the evaluation system it reports 1.42 GiB for 10⁸ digits, 7.05 GiB for 10⁹, and 32.0 GiB for 5 × 10⁹, consistent with the limits in the table.

## 6. Execution on Large Shared-Memory Systems

Nothing in the source is sized for the system of Section 5. The following are taken from the hardware profile at startup.

- **Worker threads.** One per hardware thread that the operating system allows the process to use, with the first thread of every core ranked ahead of any second thread. On Windows, threads beyond the first 64 are reached through processor groups.
- **Concurrency within the series.** Each level of the splitting tree runs as many products concurrently as divide the worker threads evenly, divide the number of nodes at that level, and fit in the cache.
- **Cache topology.** Each cache is recorded with the hardware threads that share it. A system with several last-level caches (one per core complex, tile, or socket) is planned for the cache that the threads of a product actually share, not for the sum.
- **NUMA.** The profile records each NUMA node with its memory and its hardware threads. Memory for a worker thread is requested from that thread's node: with `VirtualAllocExNuma` on Windows, and on Linux with `mbind` under the preferred (non-strict) policy, using the node numbers assigned by the system.
- **Huge pages.** Memory arenas request huge pages: on Linux a reserved huge-page pool if one exists, otherwise transparent huge pages by `madvise`. Memory that the program has finished with is returned to the system (`MADV_DONTNEED` on Linux, decommit on Windows).
- **Resource-managed execution.** On Linux the profile is read from sysfs and procfs and honors the constraints under which a batch system or container runtime places a job:
  - the CPU set (`Cpus_allowed_list`), so that a job confined to a subset of processors plans for that subset and for the caches those processors share;
  - the memory nodes it may use (`Mems_allowed_list`);
  - the memory limit of its control group or of any ancestor (`memory.max`);
  - the CPU-time quota of its control group or of any ancestor (`cpu.max`). A job with a quota but no CPU set runs as many worker threads as the quota has whole processors and leaves their placement to the kernel.
- **Memory admission.** The plan commits memory for three purposes: the multiprecision numbers at their largest, per-thread scratch buffers, and transform buffers. The numbers are given what they require and the transforms the remainder, and the total never exceeds the memory in the profile. A run that cannot be planned within that memory, on a system without storage for out-of-core operation, is rejected at startup with a report of what is missing.

**Effect of honoring a CPU-time quota.** 10⁸ digits under Linux control-group quotas on the system of Section 5, with one worker thread per whole processor of the quota (the behavior described above) and, for comparison, with 16 worker threads under the same quota:

| Quota | Workers matched to the quota | 16 workers | Ratio |
|---:|---:|---:|---:|
| 1 processor | 8.65 s | 23.78 s | 2.7 |
| 4 processors | 2.26 s | 5.40 s | 2.4 |
| 8 processors | 1.45 s | 2.62 s | 1.8 |

**Validation status.** No system larger than that of Section 5 has been available. Larger systems have been exercised only as synthetic hardware profiles (`--profile`): the run is planned for the described system and executed with that system's number of worker threads on the 16 hardware threads available, which validates the plan and the digits but not the performance.

| Configuration | Status |
|---|---|
| Up to 16 hardware threads, and subsets selected by processor affinity (1, 2, 3, 4, 6, 8, 12, and 14 threads) | Executed; digits verified |
| Synthetic profiles: eight systems of 64 to 512 cores with 1 to 24 last-level caches at eleven problem sizes from 2 × 10⁷ to 5 × 10⁸ digits; systems of 17 to 288 cores at 10⁸ digits; eight systems of up to 2,048 hardware threads with up to 128 last-level caches at 10⁸ and 2 × 10⁸ digits; and 256 cores with four last-level caches at 10⁹ and 2.5 × 10⁹ digits | Executed on 16 hardware threads; digits verified in every case |
| Linux hardware-profile reader | Tested against the files of the evaluation system and against a constructed two-socket system of 384 hardware threads, both unconfined and confined by CPU set, memory node, memory limit, and CPU-time quota |
| Linux on the evaluation system (WSL 3): unconfined, and confined by CPU set, memory limit, and CPU-time quota | Built and tested; digits verified to 2.5 × 10⁹ |
| A physical system with more than 16 hardware threads, more than one NUMA node, or more than one last-level cache | Not yet executed |

For a synthetic profile of 256 cores, four 320 MiB last-level caches, and 1 TiB of memory, the planner admits runs of up to 1.5 × 10¹¹ digits in memory and reports a minimum of 596 GiB for 10¹¹ digits. It rejects 2.5 × 10¹¹ digits, for which it requires 2 TiB: at that size the requirement is set by the transform of the largest product, which is computed whole.

## 7. Portability

- **Windows 11.** The development platform. All results in Section 5 other than those marked Linux were obtained there.
- **Linux.** Built with Rust 1.100 on Fedora 45 under WSL 3.0.5.0 (kernel 6.18) without warnings. All 24 kernels bind; both test profiles pass; and runs of 10⁶, 10⁸, 10⁹, and 2.5 × 10⁹ digits produce the correct digits. Not yet exercised: Linux on bare metal, more than 16 hardware threads, more than one NUMA node (the `mbind` call has been validated in isolation only), and a reserved huge-page pool.
- **macOS.** The kernel file's macOS branch assembles with the LLVM assembler, and the platform layer is type-checked by the Windows test build. It has not been built or run.
- **AArch64.** The design specifies a second kernel file, `aarch64.s`. It has not been written.

## 8. Limitations

- **Out-of-core operation is incomplete.** In-memory runs are verified to 5 × 10⁹ digits. The storage path for runs larger than memory currently holds only cached transforms and has been exercised by tests, not by large runs. A run that does not fit in memory is therefore rejected unless storage is supplied, and supplying storage does not yet make arbitrarily large runs possible.
- **The memory model is conservative.** The planner bounds the peak size of the multiprecision numbers by the five inputs and three products of the final merge held simultaneously, and bounds per-thread scratch by eight times each worker's cache share. The measured peak of the numbers is 3 to 20% below its bound, and measured scratch is about half of its bound, so the smallest memory at which a run is admitted is somewhat larger than the smallest at which it could complete.
- **Large-system performance is unmeasured.** The scaling results of Section 5.3 end at 16 hardware threads. Section 6 states what has been validated for larger systems and what has not.
- **Privileges.** Large pages and locked memory are requested and are used where the account has the privilege.

## 9. Repository Organization

| Directory | Contents |
|---|---|
| `engine/` | Arithmetic, planner, scheduler, memory arenas, and storage. It contains no architecture-specific or operating-system-specific code, a property enforced by a test |
| `kernels/` | `x86_64.s`: the 24 assembly kernels and the processor probe |
| `platform/` | Windows, Linux, and macOS layers (hardware profile, thread placement, memory mapping, direct I/O), and GPU memory as storage through Vulkan |
| `cli/` | Command-line interface and the test gates |

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).

## References

1. D. V. Chudnovsky and G. V. Chudnovsky, "Approximations and complex multiplication according to Ramanujan," in *Ramanujan Revisited*, Academic Press, 1988, pp. 375–472.
2. R. P. Brent, "Fast multiple-precision evaluation of elementary functions," *Journal of the ACM*, vol. 23, no. 2, pp. 242–251, 1976.
3. B. Haible and T. Papanikolaou, "Fast multiprecision evaluation of series of rational numbers," in *Algorithmic Number Theory (ANTS-III)*, Lecture Notes in Computer Science, vol. 1423, Springer, 1998, pp. 338–350.
4. A. Schönhage and V. Strassen, "Schnelle Multiplikation großer Zahlen," *Computing*, vol. 7, pp. 281–292, 1971.
5. J. M. Pollard, "The fast Fourier transform in a finite field," *Mathematics of Computation*, vol. 25, no. 114, pp. 365–374, 1971.
6. D. H. Bailey, "FFTs in external or hierarchical memory," *The Journal of Supercomputing*, vol. 4, no. 1, pp. 23–35, 1990.
7. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proceedings of the IEEE*, vol. 93, no. 2, pp. 216–231, 2005.
8. J. Demmel, J. Dongarra, V. Eijkhout, E. Fuentes, A. Petitet, R. Vuduc, R. C. Whaley, and K. Yelick, "Self-adapting linear algebra algorithms and software," *Proceedings of the IEEE*, vol. 93, no. 2, pp. 293–312, 2005.
9. K.-H. Huang and J. A. Abraham, "Algorithm-based fault tolerance for matrix operations," *IEEE Transactions on Computers*, vol. C-33, no. 6, pp. 518–528, 1984.
10. A. J. Yee, "y-cruncher: a multi-threaded pi program," http://www.numberworld.org/y-cruncher/.
11. S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.
12. R. D. Blumofe and C. E. Leiserson, "Scheduling multithreaded computations by work stealing," *Journal of the ACM*, vol. 46, no. 5, pp. 720–748, 1999.
13. R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, Cambridge University Press, 2010.
