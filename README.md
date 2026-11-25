# suPiposition

The fastest Pi calculator there is

Computes decimal digits of π.

The Chudnovsky series is summed by binary splitting, numbers are multiplied by a floating-point FFT or a multi-prime NTT, one Newton iteration gives the square root and the division, and a divide-and-conquer conversion writes the decimal digits. Every inner loop is one of 24 kernels written in assembly. The sizes they work at are not fixed in the source: they are derived at start-up from what the machine has (cores, threads, each cache and the threads that share it, memory) and from a short calibration.

[DESIGN.md](DESIGN.md) is the design the code follows. Its status line predates the code.

## Requirements

- A CPU at x86-64-v4 (AVX-512 F, CD, BW, DQ and VL) with ADX. On any other x86-64 chip the program still runs, on portable reference kernels that are far slower.
- Rust 1.100 or later. There are no other dependencies.
- Developed and tested on Windows 11. For Linux and macOS see [Other systems](#other-systems).

## Build and run

```
cargo build --release
target/release/suPiposition 1000000000 -o pi.txt
```

| Option | Meaning |
|---|---|
| `-o <file>` | Where the digits go |
| `--format text\|packed19` | Output format (default `text`) |
| `--ssd <path>[@<size>]` | Add an SSD file or device for checkpoints and for data that does not fit in memory (repeatable) |
| `--discard` | Discard checkpoint records found on the SSDs |
| `--no-vram` | Do not use the memory of a discrete GPU as storage |
| `--ref-kernels` | Use the portable reference kernels |
| `--plan-only` | Print the plan without running it |
| `--profile <cores,threads,l1,l2,l3,ram>` | Plan and run for a described machine instead of the one detected (bytes; L1 and L2 per core, L3 and RAM in all). `<n>x<size>` as L3 describes `n` separate last-level caches of that size, each with its own memory node: `256,256,48K,2M,4x320M,1T` |
| `--calibration <...>` | Supply calibration figures instead of measuring them |

A run prints the time of each stage and a `Dec Hash` of the digits, the same hash y-cruncher prints, so two programs' results can be compared without comparing files.

## Checks

```
cargo test --release
```

runs four gates: kernels against their reference bit for bit and whole runs against known digits (gate 1), plans for machines from one core to the largest a profile can describe, whole runs planned for described machines of 256 to 1,024 workers under 4 to 64 last-level caches, and the ladder of responses to a failed check (gate 2), stop and resume at every stage (gate 3), and the planner's determinism (gate 4). Every product is checked while it is made by a check word modulo 2⁶¹ − 1.

A check word proves a product, not that the right numbers were multiplied. The one silent fault found so far was of that kind: on a described machine of 256 cores under four caches, one merge of a 10⁸-digit run multiplied by an operand it had lent to another worker, and the digits were wrong from the 90,965,871st on with every check in order. It is fixed, and that run is now a test.

## Results

One machine: Ryzen 7 9800X3D (8 cores, 16 threads, 96 MB L3), 48 GB of memory, Windows 11. Runs of this build on 2026-11-25 (the middle one of three at the first two sizes), digits written to an SSD, time from start to finish.

| Digits | Time | Dec Hash |
|---:|---:|---|
| 10⁸ | 1.16 s | 199405406068015690 |
| 10⁹ | 13.9 s | 807692683230452073 |
| 2.5 × 10⁹ | 42.2 s | 1227538786604942620 |
| 5 × 10⁹ | 94.3 s, 91.5 s | 917012961347625551 |

The hashes equal those of y-cruncher 0.8.7.9547b for the same digits. On this machine y-cruncher's own computation times, measured on 2026-11-17, were 18.1 s, 51.2 s and 113.1 s for the last three rows. Runs differ by about one percent, and by more at the largest size, where five gigabytes of digits are written while it runs.

## Large machines

Nothing in the source is sized for the machine above. What a larger one changes is taken from its profile:

- **Workers.** One per hardware thread the operating system lets the process use, every core's first thread ahead of any second one. On Windows, threads beyond the first 64 are reached through processor groups.
- **How much runs at once.** A level of the series runs as many products at once as share the workers out evenly, go into the level's count of nodes, and have room in the cache.
- **Caches.** Each cache is counted with the threads that share it. A machine with several last-level caches (one per group of cores, per tile or per socket) is planned by the cache a product's workers actually share, not by the sum.
- **Memory nodes.** The profile lists each node with its memory and the threads nearest to it, and a worker's memory is asked for from its own node: by `VirtualAllocExNuma` on Windows, by `mbind` (preferred, not strict) on Linux, under the numbers the system itself gives its nodes.
- **A confined job.** On Linux the profile is read from sysfs and procfs and honours the job's CPU set, its memory nodes and the memory limit of its control group, so a job under a batch system plans for what it was given. Memory the engine has finished with goes back to the system (decommitted on Windows, `MADV_DONTNEED` on Linux), so the resident size follows the plan.

What has and has not been tried:

| | |
|---|---|
| Up to 16 threads, and fewer by processor affinity (3, 6, 8, 12, 14) | Run, digits verified |
| Described machines (`--profile`), run on 16 threads: eight machines of 64 to 512 cores under 1 to 24 last-level caches at eleven digit counts from 2 × 10⁷ to 5 × 10⁸, machines of 17 to 288 cores at 10⁸, eight more of up to 2,048 threads under up to 128 caches at 10⁸ and 2 × 10⁸, and 256 cores under four caches at 10⁹ and 2.5 × 10⁹ | Digits verified in every run; times mean nothing in such a run |
| The Linux profile reader | Tested on this machine's real files and on a made-up two-socket system of 384 threads, whole and as a job confined to one package |
| A real machine with more than 16 threads, more than one memory node, or more than one last-level cache | Never run |

## Other systems

- **Linux.** Checked under WSL 3.0.5.0 (kernel 6.18, Fedora 45), as far as a C compiler reaches: the Linux branch of the kernel file assembles with LLVM's assembler (the one Rust uses) and with the GNU assembler, the object binds all 24 kernels, and its six limb kernels pass 612,000 cases against plain C arithmetic. The profile reader is tested on that system's own sysfs and procfs files, and the calls the layer makes to ask for memory from a node and to give memory back do there what the layer expects of them (tried from C). The FFT and NTT kernels have not been run there. The Rust code of the Linux layer (`platform/linux`) is type-checked against the engine by the Windows test build but has never been built or run on Linux, so a first build may need small fixes.
- **macOS.** The kernel file's macOS branch assembles with LLVM's assembler; the layer is type-checked like the Linux one. Never built or run.
- **Arm.** The design has a second kernel file, `aarch64.s`. It does not exist yet.

## Limits

- Runs that fit in memory are verified to 5 × 10⁹ digits. The path for runs larger than memory exists but only parks cached transforms on storage so far, and has been exercised by tests, not by large runs.
- Large pages and locked memory are requested and used where the account has the privilege.

## Layout

| | |
|---|---|
| `engine/` | The arithmetic, the planner, the scheduler, memory and storage, with no knowledge of any architecture or operating system |
| `kernels/` | `x86_64.s`: the 24 kernels and the CPU probe |
| `platform/` | Windows, Linux and macOS layers, and GPU memory as storage through Vulkan |
| `cli/` | The command line and the gates |

## Licence

Mozilla Public License 2.0; see [LICENSE](LICENSE).
