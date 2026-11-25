//! Contract:
//! - Inputs: `Profile` counting the 8 resources of the 2^64 reference machine (cores, threads,
//!   L1, L2, L3, RAM, VRAM, SSD) and optional core throughput measurements.
//! - Outputs: `Shares` describing exact 64-bit `worker_count` (never capped below 2^64),
//!   materialized `WorkerDesc` rank order (first thread of each core ahead of any second thread,
//!   grouped by L3 domain), per-worker and minimum L1/L2/L3 cache shares with missing or
//!   non-increasing cache levels passed down, per-L3-domain topology (`num_l3_domains`,
//!   `min_l3_domain_size`), working-set kernel sub-block sizes (`fft_l1_sub_n`, `ntt_l1_sub_n`,
//!   `ntt_l2_sub_n`, `ntt_cols_l1_sub_rows`, `ntt_cols_l2_sub_rows`), and RAM/VRAM/SSD capacities.
//! - Data format: All sizes, counts, and percentages of the 2^64 reference machine are `u64`.
//! - Work & bytes moved: O(threads + caches + nodes + devices) startup metadata derivation only.

/// Largest power of two at or below `v` (`0` when `v == 0`).
#[inline]
pub const fn largest_pow2_le(v: u64) -> u64 {
    if v == 0 { 0 } else { 1u64 << v.ilog2() }
}

/// Summary threshold for extreme near-2^64 synthetic profiles whose `WorkerDesc` array cannot
/// fit in host memory during `--plan-only` / Gate 2 synthetic tests; every real topology (and
/// synthetic profiles up to 262,144 threads) materializes 100% of its hardware threads.
const MAX_SYNTHETIC_MATERIALIZED_THREADS: u64 = 262_144;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheEntry {
    pub size: u64,
    pub threads: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamNode {
    pub size: u64,
    pub threads: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEntry {
    pub id: String,
    pub size: u64,
    pub align: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreThreads {
    pub core_id: u64,
    pub threads: Vec<u64>,
}

/// The system profile: 8 resources as 64-bit counts (parts in 2^64 of the reference machine).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub cores: u64,
    pub threads: u64,
    pub core_map: Vec<CoreThreads>,
    pub l1: Vec<CacheEntry>,
    pub l2: Vec<CacheEntry>,
    pub l3: Vec<CacheEntry>,
    pub ram: Vec<RamNode>,
    pub vram: Vec<DeviceEntry>,
    pub ssd: Vec<DeviceEntry>,
    pub os_places_threads: bool,
}

impl Profile {
    pub fn synthetic(
        cores: u64,
        threads: u64,
        l1_per_core: u64,
        l2_per_core: u64,
        l3_total: u64,
        ram_total: u64,
        vram_sizes: &[u64],
        ssd_sizes: &[u64],
    ) -> Self {
        Self::synthetic_multi_domain(
            cores,
            threads,
            1,
            l1_per_core,
            l2_per_core,
            l3_total,
            ram_total,
            vram_sizes,
            ssd_sizes,
        )
    }

    /// Build a synthetic `Profile` partitioned evenly across `domains` separate L3 caches and
    /// NUMA RAM nodes (e.g. multi-CCD or multi-tile server chips).
    #[allow(clippy::too_many_arguments)]
    pub fn synthetic_multi_domain(
        cores: u64,
        threads: u64,
        domains: u64,
        l1_per_core: u64,
        l2_per_core: u64,
        l3_per_domain: u64,
        ram_total: u64,
        vram_sizes: &[u64],
        ssd_sizes: &[u64],
    ) -> Self {
        let cores = cores.max(1);
        let threads = threads.max(cores);
        let domains = domains.max(1).min(cores);
        let materializable = threads <= MAX_SYNTHETIC_MATERIALIZED_THREADS;
        let mut core_map = Vec::with_capacity(if materializable { cores as usize } else { 0 });
        let mut l1 = Vec::new();
        let mut l2 = Vec::new();
        let mut dom_threads: Vec<Vec<u64>> = if materializable {
            vec![Vec::new(); domains as usize]
        } else {
            Vec::new()
        };

        if materializable {
            let t_per_c = (threads / cores).max(1);
            let extra = threads % cores;
            let mut tid = 0u64;
            for c in 0..cores {
                let dom_idx = ((c as u128 * domains as u128) / (cores as u128)) as usize;
                let cnt = t_per_c + u64::from(c < extra);
                let mut ct = Vec::with_capacity(cnt as usize);
                for _ in 0..cnt {
                    ct.push(tid);
                    dom_threads[dom_idx].push(tid);
                    tid += 1;
                }
                if l1_per_core > 0 {
                    l1.push(CacheEntry {
                        size: l1_per_core,
                        threads: ct.clone(),
                    });
                }
                if l2_per_core > 0 {
                    l2.push(CacheEntry {
                        size: l2_per_core,
                        threads: ct.clone(),
                    });
                }
                core_map.push(CoreThreads {
                    core_id: c,
                    threads: ct,
                });
            }
        } else {
            // Extreme synthetic profile near 2^64 reference machine: represent summary shares cleanly.
            if l1_per_core > 0 {
                l1.push(CacheEntry {
                    size: l1_per_core,
                    threads: vec![0],
                });
            }
            if l2_per_core > 0 {
                l2.push(CacheEntry {
                    size: l2_per_core,
                    threads: vec![0],
                });
            }
        }

        let l3 = if l3_per_domain > 0 {
            if materializable {
                dom_threads
                    .iter()
                    .map(|ts| CacheEntry {
                        size: l3_per_domain,
                        threads: ts.clone(),
                    })
                    .collect()
            } else {
                (0..domains.min(1024))
                    .map(|d| CacheEntry {
                        size: l3_per_domain,
                        threads: vec![d],
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };

        let ram = if materializable {
            let base_ram = ram_total / domains;
            let rem_ram = ram_total % domains;
            dom_threads
                .into_iter()
                .enumerate()
                .map(|(di, ts)| RamNode {
                    size: base_ram + if (di as u64) < rem_ram { 1 } else { 0 },
                    threads: ts,
                })
                .collect()
        } else {
            vec![RamNode {
                size: ram_total,
                threads: vec![0],
            }]
        };

        let vram = vram_sizes
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0)
            .map(|(i, &s)| DeviceEntry {
                id: format!("vram-{i}"),
                size: s,
                align: 4096,
            })
            .collect();

        let ssd = ssd_sizes
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0)
            .map(|(i, &s)| DeviceEntry {
                id: format!("ssd-{i}"),
                size: s,
                align: 4096,
            })
            .collect();

        Self {
            cores,
            threads,
            core_map,
            l1,
            l2,
            l3,
            ram,
            vram,
            ssd,
            os_places_threads: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorkerDesc {
    pub worker_idx: usize,
    pub hw_thread: u64,
    pub core_id: u64,
    pub smt_slot: u32,
    pub numa_node: usize,
    pub l3_domain: usize,
    pub l1_share: u64,
    pub l2_share: u64,
    pub l3_share: u64,
}

#[derive(Clone, Debug)]
pub struct Shares {
    pub cores: u64,
    pub threads: u64,
    /// Exact 64-bit count of workers across the 2^64 reference machine (never capped below 2^64).
    pub worker_count: u64,
    /// Materialized OS worker descriptors (`== worker_count` for every real system and every
    /// synthetic profile up to `MAX_SYNTHETIC_MATERIALIZED_THREADS`).
    pub active_workers: usize,
    pub workers: Vec<WorkerDesc>,
    pub min_l1_share: u64,
    pub min_l2_share: u64,
    pub min_l3_share: u64,
    pub num_l3_domains: u64,
    pub min_l3_domain_size: u64,
    pub total_l3: u64,
    pub fft_l1_sub_n: u64,
    pub ntt_l1_sub_n: u64,
    pub ntt_l2_sub_n: u64,
    pub ntt_cols_l1_sub_rows: u64,
    pub ntt_cols_l2_sub_rows: u64,
    pub total_ram: u64,
    pub total_vram: u64,
    pub total_ssd: u64,
    pub os_places_threads: bool,
}

#[inline]
fn thread_list_contains(list: &[u64], tid: u64) -> bool {
    let n = list.len();
    if n == 0 {
        return false;
    }
    let first = list[0];
    let last = list[n - 1];
    if last >= first && last - first + 1 == n as u64 {
        return tid >= first && tid <= last;
    }
    if n >= 8 && (1..n.min(5)).all(|i| list[i - 1] < list[i]) && first <= last {
        if list.binary_search(&tid).is_ok() {
            return true;
        }
    }
    list.contains(&tid)
}

/// Derive the five cache-resident kernel sub-block sizes from `(min_l1_share, min_l2_share)`:
/// - `fft_l1_sub_n`: `16 B` complex data + `8 B` twiddles = `24 B/pt <= min_l1_share` (`>= 64`).
/// - `ntt_l1_sub_n`: `8 B` residue data + `4 B` twiddles = `12 B/pt <= min_l1_share` (`>= 64`).
/// - `ntt_l2_sub_n`: `8 B` residue data + `8 B` twiddles/headroom = `16 B/pt <= min_l2_share` (`>= 2 * ntt_l1_sub_n`).
/// - `ntt_cols_l1_sub_rows`: `64 B` 8-col strip + `32 B` twiddle/tile = `96 B/row <= min_l1_share` (`>= 16`).
/// - `ntt_cols_l2_sub_rows`: `64 B` 8-col strip + `64 B` tile source = `128 B/row <= min_l2_share` (`>= 2 * ntt_cols_l1_sub_rows`).
#[inline]
pub const fn derive_kernel_cache_blocks(
    min_l1_share: u64,
    min_l2_share: u64,
) -> (u64, u64, u64, u64, u64) {
    let fft_l1 = {
        let p = largest_pow2_le(min_l1_share / 24);
        if p < 64 { 64 } else { p }
    };
    let ntt_l1 = {
        let p = largest_pow2_le(min_l1_share / 12);
        if p < 64 { 64 } else { p }
    };
    let ntt_l2 = {
        let floor = ntt_l1.saturating_mul(2);
        let p = largest_pow2_le(min_l2_share / 16);
        if p < floor { floor } else { p }
    };
    let cols_l1 = {
        let p = largest_pow2_le(min_l1_share / 96);
        if p < 16 { 16 } else { p }
    };
    let cols_l2 = {
        let floor = cols_l1.saturating_mul(2);
        let p = largest_pow2_le(min_l2_share / 128);
        if p < floor { floor } else { p }
    };
    (fft_l1, ntt_l1, ntt_l2, cols_l1, cols_l2)
}

impl Shares {
    pub fn from_profile(profile: &Profile) -> Self {
        debug_assert!(profile.cores >= 1 && profile.threads >= profile.cores);

        let total_ram = profile
            .ram
            .iter()
            .fold(0u64, |acc, n| acc.saturating_add(n.size));
        let total_vram = profile
            .vram
            .iter()
            .fold(0u64, |acc, d| acc.saturating_add(d.size));
        let total_ssd = profile
            .ssd
            .iter()
            .fold(0u64, |acc, d| acc.saturating_add(d.size));

        let worker_count = profile.threads.max(1);
        let mapped_threads: usize = profile.core_map.iter().map(|c| c.threads.len()).sum();
        let target_workers = if mapped_threads > 0 {
            mapped_threads.min(profile.threads.min(usize::MAX as u64) as usize)
        } else {
            (profile
                .threads
                .min(MAX_SYNTHETIC_MATERIALIZED_THREADS) as usize)
                .max(1)
        };

        let share_for_thread =
            |entries: &[CacheEntry], hw_tid: u64, fallback_div: u64| -> (u64, usize) {
                for (idx, e) in entries.iter().enumerate() {
                    if thread_list_contains(&e.threads, hw_tid) {
                        let sharers = (e.threads.len() as u64).max(1);
                        return (e.size / sharers, idx);
                    }
                }
                if let Some(first) = entries.first() {
                    return (first.size / fallback_div.max(1), 0);
                }
                (0, 0)
            };

        let threads_per_core = (profile.threads / profile.cores.max(1)).max(1);
        let mut raw_workers = Vec::with_capacity(target_workers);

        if !profile.core_map.is_empty() {
            // Order every core's first thread (smt_slot == 0) ahead of any second thread (DESIGN.md s6).
            let max_slots = profile
                .core_map
                .iter()
                .map(|c| c.threads.len())
                .max()
                .unwrap_or(1);
            for slot in 0..max_slots {
                for core in &profile.core_map {
                    if let Some(&hw_tid) = core.threads.get(slot)
                        && raw_workers.len() < target_workers
                    {
                        let numa_node = profile
                            .ram
                            .iter()
                            .position(|n| thread_list_contains(&n.threads, hw_tid))
                            .unwrap_or(0);
                        let (raw_l1, _) = share_for_thread(&profile.l1, hw_tid, threads_per_core);
                        let (raw_l2, l2_dom) =
                            share_for_thread(&profile.l2, hw_tid, threads_per_core);
                        let (raw_l3, l3_dom_raw) =
                            share_for_thread(&profile.l3, hw_tid, profile.threads);
                        let l3_domain = if !profile.l3.is_empty() {
                            l3_dom_raw
                        } else {
                            l2_dom
                        };
                        let (l1_s, l2_s, l3_s) = normalize_cache_shares(raw_l1, raw_l2, raw_l3);
                        raw_workers.push(WorkerDesc {
                            worker_idx: raw_workers.len(),
                            hw_thread: hw_tid,
                            core_id: core.core_id,
                            smt_slot: slot as u32,
                            numa_node,
                            l3_domain,
                            l1_share: l1_s,
                            l2_share: l2_s,
                            l3_share: l3_s,
                        });
                    }
                }
            }
        }

        if raw_workers.is_empty() {
            let raw_l1 = profile
                .l1
                .first()
                .map(|e| e.size / threads_per_core)
                .unwrap_or(0);
            let raw_l2 = profile
                .l2
                .first()
                .map(|e| e.size / threads_per_core)
                .unwrap_or(0);
            let raw_l3 = if !profile.l3.is_empty() {
                let l3_sum = profile
                    .l3
                    .iter()
                    .fold(0u64, |acc, c| acc.saturating_add(c.size));
                l3_sum / worker_count
            } else {
                0
            };
            let (l1_s, l2_s, l3_s) = normalize_cache_shares(raw_l1, raw_l2, raw_l3);
            let dom_cnt = profile.l3.len().max(1);
            for i in 0..target_workers {
                raw_workers.push(WorkerDesc {
                    worker_idx: i,
                    hw_thread: i as u64,
                    core_id: (i as u64) % profile.cores.max(1),
                    smt_slot: ((i as u64) / profile.cores.max(1)) as u32,
                    numa_node: 0,
                    l3_domain: i % dom_cnt,
                    l1_share: l1_s,
                    l2_share: l2_s,
                    l3_share: l3_s,
                });
            }
        }

        let min_l1_share = raw_workers
            .iter()
            .map(|w| w.l1_share)
            .min()
            .unwrap_or(16 * 1024);
        let min_l2_share = raw_workers
            .iter()
            .map(|w| w.l2_share)
            .min()
            .unwrap_or(min_l1_share);
        let min_l3_share = raw_workers
            .iter()
            .map(|w| w.l3_share)
            .min()
            .unwrap_or(min_l2_share);

        debug_assert!(min_l1_share > 0);
        debug_assert!(min_l2_share >= min_l1_share);
        debug_assert!(min_l3_share >= min_l2_share);

        let (num_l3_domains, min_l3_domain_size, total_l3) = if !profile.l3.is_empty() {
            let mut active_doms: Vec<usize> = if !profile.core_map.is_empty() {
                let mut ds: Vec<usize> = raw_workers
                    .iter()
                    .filter(|w| w.l3_domain < profile.l3.len())
                    .map(|w| w.l3_domain)
                    .collect();
                ds.sort_unstable();
                ds.dedup();
                ds
            } else {
                (0..profile.l3.len()).collect()
            };
            if active_doms.is_empty() {
                active_doms.push(0);
            }
            let sum_l3 = active_doms
                .iter()
                .fold(0u64, |acc, &di| acc.saturating_add(profile.l3[di].size));
            let min_dom = active_doms
                .iter()
                .map(|&di| profile.l3[di].size)
                .min()
                .unwrap_or(min_l3_share)
                .max(min_l3_share);
            (active_doms.len() as u64, min_dom, sum_l3.max(min_dom))
        } else if !profile.l2.is_empty() {
            // Section 6: when L3 is absent, L2 clusters serve as the last-level cache domains.
            let doms = profile.l2.len() as u64;
            let min_dom = profile
                .l2
                .iter()
                .map(|c| c.size)
                .min()
                .unwrap_or(min_l3_share)
                .max(min_l3_share);
            let sum_l2 = profile
                .l2
                .iter()
                .fold(0u64, |acc, c| acc.saturating_add(c.size))
                .max(min_dom);
            (doms, min_dom, sum_l2)
        } else {
            let tot = min_l3_share.saturating_mul(worker_count);
            (1, tot.max(min_l3_share), tot.max(min_l3_share))
        };

        let (
            fft_l1_sub_n,
            ntt_l1_sub_n,
            ntt_l2_sub_n,
            ntt_cols_l1_sub_rows,
            ntt_cols_l2_sub_rows,
        ) = derive_kernel_cache_blocks(min_l1_share, min_l2_share);

        Self {
            cores: profile.cores,
            threads: profile.threads,
            worker_count,
            active_workers: raw_workers.len(),
            workers: raw_workers,
            min_l1_share,
            min_l2_share,
            min_l3_share,
            num_l3_domains,
            min_l3_domain_size,
            total_l3,
            fft_l1_sub_n,
            ntt_l1_sub_n,
            ntt_l2_sub_n,
            ntt_cols_l1_sub_rows,
            ntt_cols_l2_sub_rows,
            total_ram,
            total_vram,
            total_ssd,
            os_places_threads: profile.os_places_threads,
        }
    }

    /// Effective last-level cache bytes usable by `conc_nodes` concurrent nodes, each spanning
    /// `rows` grid rows (`rows == 1` for 1D or broadcast data that cannot partition across L3
    /// domains; `rows > 1` for 2D row-partitioned grids). On single-L3 chips (`num_l3_domains == 1`)
    /// this equals `total_l3`; on multi-CCD / multi-tile chips it bounds each L3 domain by
    /// `min_l3_domain_size` so shared or misaligned node counts never overstate L3 capacity.
    #[inline]
    pub fn effective_l3_bytes(&self, conc_nodes: u64, rows: usize) -> u64 {
        let conc = conc_nodes.max(1);
        let d = self.num_l3_domains.max(1);
        let dom_sz = self.min_l3_domain_size;
        if d == 1 {
            return self.total_l3;
        }
        if conc >= d {
            let nodes_per_dom = conc.div_ceil(d);
            ((dom_sz as u128 * conc as u128) / (nodes_per_dom as u128))
                .min(u64::MAX as u128) as u64
        } else if rows <= 1 {
            conc.saturating_mul(dom_sz)
        } else {
            let doms_per_node = (d / conc).clamp(1, rows as u64);
            conc.saturating_mul(doms_per_node).saturating_mul(dom_sz)
        }
    }

    /// Exact predicate testing whether `conc_nodes` concurrent nodes of `per_node_bytes` each
    /// (`rows` rows per node) fit inside the per-domain L3 budget `numer / denom * domain_size`.
    #[inline]
    pub fn fits_l3_domains(
        &self,
        conc_nodes: u64,
        per_node_bytes: u64,
        rows: usize,
        numer: u64,
        denom: u64,
    ) -> bool {
        let conc = conc_nodes.max(1);
        let d = self.num_l3_domains.max(1);
        let dom_cap = ((self.min_l3_domain_size as u128 * numer as u128) / (denom.max(1) as u128))
            .min(u64::MAX as u128) as u64;
        if d == 1 {
            let tot_cap = ((self.total_l3 as u128 * numer as u128) / (denom.max(1) as u128))
                .min(u64::MAX as u128) as u64;
            return conc.saturating_mul(per_node_bytes) <= tot_cap;
        }
        if conc >= d {
            let nodes_per_dom = conc.div_ceil(d);
            nodes_per_dom.saturating_mul(per_node_bytes) <= dom_cap
        } else if rows <= 1 {
            per_node_bytes <= dom_cap
        } else {
            let doms_per_node = (d / conc).clamp(1, rows as u64);
            per_node_bytes <= doms_per_node.saturating_mul(dom_cap)
        }
    }

    /// Number of distinct L3 domains spanned by a single task running across `par_workers`
    /// workers (`1` on single-L3 chips or when `par_workers` stays within a single L3 domain).
    #[inline]
    pub fn domains_spanned_by_workers(&self, par_workers: usize) -> u64 {
        let d = self.num_l3_domains.max(1);
        if d <= 1 || par_workers <= 1 {
            return 1;
        }
        let w_per_dom = self.worker_count.max(1).div_ceil(d).max(1);
        (par_workers as u64).div_ceil(w_per_dom).clamp(1, d)
    }

    /// Re-rank workers using measured per-worker `ntt_fwd` throughput while keeping every core's
    /// first thread (`smt_slot == 0`) ahead of any second thread (`smt_slot > 0`) (DESIGN.md s6)
    /// and preserving L3-domain locality among equal-speed cores.
    pub fn rank_by_throughput(&mut self, throughputs: &[f64]) {
        if self.os_places_threads || throughputs.len() < self.workers.len() {
            return;
        }
        let mut indexed: Vec<(f64, WorkerDesc)> = self
            .workers
            .drain(..)
            .enumerate()
            .map(|(i, w)| (throughputs[i], w))
            .collect();
        indexed.sort_by(|a, b| {
            a.1.smt_slot
                .cmp(&b.1.smt_slot)
                .then_with(|| b.0.total_cmp(&a.0))
                .then_with(|| a.1.l3_domain.cmp(&b.1.l3_domain))
                .then_with(|| a.1.hw_thread.cmp(&b.1.hw_thread))
        });
        for (new_idx, (_, mut w)) in indexed.into_iter().enumerate() {
            w.worker_idx = new_idx;
            self.workers.push(w);
        }
    }
}

/// How many nodes may run at once with the workers shared out evenly among them: every count
/// that divides the workers, in rising order, from one node with all of them to a node a worker.
/// A count between these would leave workers without a node. What the system has decides the
/// list, so sixteen workers give 1, 2, 4, 8, 16 and twelve give 1, 2, 3, 4, 6, 12. A level of
/// the series takes from it only the counts that also go into its number of nodes: with any
/// other, the level's last nodes are fewer than the count and run with a share of the workers.
pub fn even_splits(workers: usize) -> Vec<usize> {
    let w = workers.max(1);
    let (mut low, mut high) = (Vec::new(), Vec::new());
    let mut d = 1usize;
    while d.saturating_mul(d) <= w {
        if w.is_multiple_of(d) {
            low.push(d);
            if d != w / d {
                high.push(w / d);
            }
        }
        d += 1;
    }
    low.extend(high.into_iter().rev());
    low
}

/// Section 6 rule: "When a cache level is absent, or gives a worker at most what the level below
/// it does, its job passes to that lower level."
fn normalize_cache_shares(raw_l1: u64, raw_l2: u64, raw_l3: u64) -> (u64, u64, u64) {
    let l1 = if raw_l1 == 0 { 16 * 1024 } else { raw_l1 };
    let l2 = if raw_l2 <= l1 { l1 } else { raw_l2 };
    let l3 = if raw_l3 <= l2 { l2 } else { raw_l3 };
    (l1, l2, l3)
}

#[cfg(test)]
mod tests {
    use super::{Profile, Shares, derive_kernel_cache_blocks, even_splits};

    #[test]
    fn even_splits_are_the_divisors_in_rising_order() {
        assert_eq!(even_splits(0), vec![1]);
        assert_eq!(even_splits(1), vec![1]);
        assert_eq!(even_splits(7), vec![1, 7]);
        assert_eq!(even_splits(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(even_splits(16), vec![1, 2, 4, 8, 16]);
        assert_eq!(
            even_splits(192),
            vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
        );
        for w in 1..=600usize {
            let s = even_splits(w);
            let by_trial: Vec<usize> = (1..=w).filter(|d| w.is_multiple_of(*d)).collect();
            assert_eq!(s, by_trial, "workers {w}");
        }
    }

    #[test]
    fn shares_have_no_4096_worker_cap_and_preserve_near_2_pow_64_counts() {
        let p8k = Profile::synthetic(
            4096,
            8192,
            48 * 1024,
            1024 * 1024,
            2 * 1024 * 1024 * 1024,
            1024 * 1024 * 1024 * 1024,
            &[],
            &[],
        );
        let s8k = Shares::from_profile(&p8k);
        assert_eq!(s8k.worker_count, 8192);
        assert_eq!(s8k.active_workers, 8192);
        assert_eq!(s8k.workers.len(), 8192);
        assert!(s8k.workers[..4096].iter().all(|w| w.smt_slot == 0));
        assert!(s8k.workers[4096..].iter().all(|w| w.smt_slot == 1));

        let p_max = Profile::synthetic(
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX - 1,
            &[u64::MAX - 1],
            &[u64::MAX - 1],
        );
        let s_max = Shares::from_profile(&p_max);
        assert_eq!(s_max.worker_count, u64::MAX - 1);
    }

    #[test]
    fn multi_domain_l3_bounds_single_and_concurrent_working_sets_per_domain() {
        // 12-CCD server chip: 96 cores, 12 x 32 MiB L3 = 384 MiB total.
        let p_epyc = Profile::synthetic_multi_domain(
            96,
            96,
            12,
            48 * 1024,
            1024 * 1024,
            32 * 1024 * 1024,
            512 * 1024 * 1024 * 1024,
            &[],
            &[],
        );
        let sh = Shares::from_profile(&p_epyc);
        assert_eq!(sh.num_l3_domains, 12);
        assert_eq!(sh.min_l3_domain_size, 32 * 1024 * 1024);
        assert_eq!(sh.total_l3, 384 * 1024 * 1024);
        assert_eq!(sh.min_l3_share, (32 * 1024 * 1024) / 8);

        // 1D / broadcast buffer cannot exceed one 32 MiB CCD even when conc == 1!
        assert_eq!(sh.effective_l3_bytes(1, 1), 32 * 1024 * 1024);
        assert!(!sh.fits_l3_domains(1, 64 * 1024 * 1024, 1, 1, 1));
        assert!(sh.fits_l3_domains(1, 32 * 1024 * 1024, 1, 1, 1));

        // 8 concurrent nodes across 12 CCDs: integer 1 CCD per node -> 256 MiB, not 384 MiB!
        assert_eq!(sh.effective_l3_bytes(8, 64), 256 * 1024 * 1024);
        assert!(!sh.fits_l3_domains(8, 40 * 1024 * 1024, 64, 1, 1));
        assert!(sh.fits_l3_domains(12, 32 * 1024 * 1024, 64, 1, 1));

        // 16 concurrent nodes across 12 CCDs: up to 2 nodes share one CCD -> 16 MiB per node!
        assert_eq!(sh.effective_l3_bytes(16, 64), 256 * 1024 * 1024);
        assert!(!sh.fits_l3_domains(16, 20 * 1024 * 1024, 64, 1, 1));
        assert!(sh.fits_l3_domains(16, 16 * 1024 * 1024, 64, 1, 1));
    }

    #[test]
    fn kernel_cache_blocks_derive_from_l1_and_l2_shares() {
        assert_eq!(
            derive_kernel_cache_blocks(24 * 1024, 512 * 1024),
            (1024, 2048, 32_768, 256, 4096)
        );
        assert_eq!(
            derive_kernel_cache_blocks(16 * 1024, 256 * 1024),
            (512, 1024, 16_384, 128, 2048)
        );
        assert_eq!(
            derive_kernel_cache_blocks(48 * 1024, 2 * 1024 * 1024),
            (2048, 4096, 131_072, 512, 16_384)
        );
    }
}
