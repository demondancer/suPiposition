//! Contract:
//! - Inputs: a reader of the small text files in which a Linux system describes itself
//!   (`sys/devices/system/cpu/...`, `sys/devices/system/node/...`, `proc/meminfo`,
//!   `proc/self/status`, `proc/self/cgroup`, `sys/fs/cgroup/.../memory.max`), paths given from the
//!   root without a leading slash.
//! - Outputs: the `Profile` of what this process may use: its hardware threads grouped by core,
//!   every data cache with its size and the threads that share it, every memory node with the
//!   memory available on it and the threads nearest to it. `None` where the files do not give
//!   the whole picture; the caller then keeps its own fallback.
//! - Data format: counts and bytes as `u64`; hardware threads keep the numbers the system gives
//!   them, which are the numbers a thread is placed by.
//! - Work & bytes moved: a few short reads per hardware thread, once at startup. The module is
//!   plain text handling and is built and tested on every system, not only where it is used.

use std::collections::BTreeMap;

use engine::tier::{CacheEntry, CoreThreads, Profile, RamNode};

/// A list as the kernel writes it, `0-3,8,10-11`, as the numbers it names in rising order.
pub fn parse_cpu_list(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<u64>().ok(), b.trim().parse::<u64>().ok()),
            None => {
                let v = part.parse::<u64>().ok();
                (v, v)
            }
        };
        if let (Some(lo), Some(hi)) = (lo, hi)
            && lo <= hi
            && hi - lo < (1 << 22)
        {
            out.extend(lo..=hi);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A size as sysfs writes it, `48K`, `1024K`, `16M` or plain bytes.
pub fn parse_size(text: &str) -> Option<u64> {
    let t = text.trim();
    let last = t.chars().last()?;
    let (digits, unit) = match last {
        'K' | 'k' => (&t[..t.len() - 1], 1024u64),
        'M' | 'm' => (&t[..t.len() - 1], 1024 * 1024),
        'G' | 'g' => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        _ => (t, 1),
    };
    digits.trim().parse::<u64>().ok()?.checked_mul(unit)
}

/// The bytes of one entry of a `meminfo` file (`MemAvailable:  123 kB`, `Node 0 MemFree: 123 kB`).
fn meminfo_bytes(text: &str, key: &str) -> Option<u64> {
    for line in text.lines() {
        let Some(at) = line.find(key) else { continue };
        let rest = line[at + key.len()..].trim_start();
        let Some(rest) = rest.strip_prefix(':') else { continue };
        if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) {
            return kb.checked_mul(1024);
        }
    }
    None
}

/// One of the lists of `proc/self/status` (`Cpus_allowed_list`, `Mems_allowed_list`).
fn status_list(status: &str, key: &str) -> Option<Vec<u64>> {
    status
        .lines()
        .find_map(|l| l.strip_prefix(key).and_then(|r| r.trim_start().strip_prefix(':')))
        .map(parse_cpu_list)
        .filter(|v| !v.is_empty())
}

/// The memory limit of the control group this process runs in, the smallest set on the group
/// or on any group above it: where a batch system confines a job, this is what it may use.
fn cgroup_memory_limit(read: &dyn Fn(&str) -> Option<String>) -> Option<u64> {
    let groups = read("proc/self/cgroup")?;
    let path = groups.lines().find_map(|l| l.strip_prefix("0::"))?;
    let mut dir = path.trim().trim_matches('/').to_string();
    let mut limit: Option<u64> = None;
    loop {
        let file = if dir.is_empty() {
            "sys/fs/cgroup/memory.max".to_string()
        } else {
            format!("sys/fs/cgroup/{dir}/memory.max")
        };
        if let Some(v) = read(&file).and_then(|t| t.trim().parse::<u64>().ok()) {
            limit = Some(limit.map_or(v, |l| l.min(v)));
        }
        if dir.is_empty() {
            break;
        }
        dir = match dir.rfind('/') {
            Some(i) => dir[..i].to_string(),
            None => String::new(),
        };
    }
    limit
}

/// The profile of the system `read` describes, without its extent devices (`vram` and `ssd`
/// are left empty for the caller to fill).
pub fn profile_from_reader(read: &dyn Fn(&str) -> Option<String>) -> Option<Profile> {
    profile_and_nodes_from_reader(read).map(|(profile, _)| profile)
}

/// The same profile, and with it the system's own number of each memory node of `ram`, in that
/// order: a job given the nodes 2 and 3 has them as its first and second, and memory for its
/// workers has to be asked from 2 and 3. Empty where the memory is listed as one node that the
/// system did not name.
pub fn profile_and_nodes_from_reader(
    read: &dyn Fn(&str) -> Option<String>,
) -> Option<(Profile, Vec<u64>)> {
    let online = parse_cpu_list(&read("sys/devices/system/cpu/online")?);
    let status = read("proc/self/status").unwrap_or_default();
    // What the operating system lets this process use decides, not what the machine has.
    let usable: Vec<u64> = match status_list(&status, "Cpus_allowed_list") {
        Some(allowed) => online
            .iter()
            .copied()
            .filter(|c| allowed.binary_search(c).is_ok())
            .collect(),
        None => online,
    };
    if usable.is_empty() {
        return None;
    }
    let is_usable = |c: &u64| usable.binary_search(c).is_ok();

    // Cores: the threads that name the same set of siblings.
    let mut by_core: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for &c in &usable {
        let dir = format!("sys/devices/system/cpu/cpu{c}/topology/");
        let siblings = read(&format!("{dir}core_cpus_list"))
            .or_else(|| read(&format!("{dir}thread_siblings_list")))
            .map(|t| parse_cpu_list(&t))
            .unwrap_or_default();
        let key = siblings.first().copied().unwrap_or(c);
        by_core.entry(key).or_default().push(c);
    }
    let core_map: Vec<CoreThreads> = by_core
        .into_values()
        .enumerate()
        .map(|(i, mut threads)| {
            threads.sort_unstable();
            CoreThreads {
                core_id: i as u64,
                threads,
            }
        })
        .collect();

    // Caches that hold data, each once, with those of its sharers that this process may use.
    let mut caches: BTreeMap<(u32, Vec<u64>), u64> = BTreeMap::new();
    for &c in &usable {
        for index in 0..16 {
            let dir = format!("sys/devices/system/cpu/cpu{c}/cache/index{index}/");
            let Some(level) = read(&format!("{dir}level")).and_then(|t| t.trim().parse::<u32>().ok())
            else {
                break;
            };
            let kind = read(&format!("{dir}type")).unwrap_or_default();
            if !matches!(kind.trim(), "Data" | "Unified") {
                continue;
            }
            let Some(size) = read(&format!("{dir}size")).and_then(|t| parse_size(&t)) else {
                continue;
            };
            let mut sharers = read(&format!("{dir}shared_cpu_list"))
                .map(|t| parse_cpu_list(&t))
                .unwrap_or_default();
            sharers.retain(|t| is_usable(t));
            if sharers.is_empty() {
                sharers.push(c);
            }
            caches.insert((level, sharers), size);
        }
    }
    let (mut l1, mut l2, mut l3) = (Vec::new(), Vec::new(), Vec::new());
    for ((level, threads), size) in caches {
        let entry = CacheEntry { size, threads };
        match level {
            1 => l1.push(entry),
            2 => l2.push(entry),
            3 => l3.push(entry),
            _ => {}
        }
    }
    if l1.is_empty() && l2.is_empty() && l3.is_empty() {
        return None;
    }

    // Memory: what the system reports as available, within the limit of the control group, shared
    // out among the memory nodes this process may use as their free memory lies.
    let meminfo = read("proc/meminfo")?;
    let mut available =
        meminfo_bytes(&meminfo, "MemAvailable").or_else(|| meminfo_bytes(&meminfo, "MemFree"))?;
    if let Some(limit) = cgroup_memory_limit(read) {
        available = available.min(limit);
    }
    let mems_allowed = status_list(&status, "Mems_allowed_list");
    let node_ids = read("sys/devices/system/node/online")
        .map(|t| parse_cpu_list(&t))
        .unwrap_or_default();
    let mut nodes: Vec<(u64, Vec<u64>, u64)> = Vec::new();
    for n in node_ids {
        if let Some(allowed) = &mems_allowed
            && allowed.binary_search(&n).is_err()
        {
            continue;
        }
        let dir = format!("sys/devices/system/node/node{n}/");
        let threads: Vec<u64> = read(&format!("{dir}cpulist"))
            .map(|t| parse_cpu_list(&t))
            .unwrap_or_default()
            .into_iter()
            .filter(|c| is_usable(c))
            .collect();
        let free = read(&format!("{dir}meminfo"))
            .and_then(|t| meminfo_bytes(&t, "MemFree"))
            .unwrap_or(0);
        if threads.is_empty() && free == 0 {
            continue;
        }
        nodes.push((n, threads, free));
    }
    let free_sum: u128 = nodes.iter().map(|n| n.2 as u128).sum();
    let (ram, ram_node_ids): (Vec<RamNode>, Vec<u64>) =
        if free_sum == 0 || nodes.iter().all(|n| n.1.is_empty()) {
            (
                vec![RamNode {
                    size: available,
                    threads: usable.clone(),
                }],
                Vec::new(),
            )
        } else {
            let ids: Vec<u64> = nodes.iter().map(|n| n.0).collect();
            (
                nodes
                    .into_iter()
                    .map(|(_, threads, free)| RamNode {
                        size: ((available as u128) * (free as u128) / free_sum) as u64,
                        threads,
                    })
                    .collect(),
                ids,
            )
        };

    let profile = Profile {
        cores: core_map.len() as u64,
        threads: usable.len() as u64,
        core_map,
        l1,
        l2,
        l3,
        ram,
        vram: Vec::new(),
        ssd: Vec::new(),
        os_places_threads: false,
    };
    Some((profile, ram_node_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Files written as `== <path>` lines, each followed by the file's text.
    fn files_of(dump: &str) -> HashMap<String, String> {
        let mut files: HashMap<String, String> = HashMap::new();
        let mut cur: Option<String> = None;
        for line in dump.lines() {
            if let Some(path) = line.strip_prefix("== ") {
                cur = Some(path.trim().to_string());
                files.entry(path.trim().to_string()).or_default();
            } else if let Some(path) = &cur {
                let text = files.get_mut(path).unwrap();
                text.push_str(line);
                text.push('\n');
            }
        }
        files
    }

    fn profile_of(files: &HashMap<String, String>) -> Option<Profile> {
        profile_from_reader(&|p| files.get(p).cloned())
    }

    #[test]
    fn lists_and_sizes_are_read_as_the_kernel_writes_them() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list(" 2 , 0-1 ,2"), vec![0, 1, 2]);
        assert_eq!(parse_cpu_list(""), Vec::<u64>::new());
        assert_eq!(parse_cpu_list("7-3,x,4"), vec![4]);
        assert_eq!(parse_size("48K\n"), Some(48 * 1024));
        assert_eq!(parse_size("98304K"), Some(96 * 1024 * 1024));
        assert_eq!(parse_size("16M"), Some(16 * 1024 * 1024));
        assert_eq!(parse_size("4096"), Some(4096));
        assert_eq!(parse_size("K"), None);
        assert_eq!(parse_size(""), None);
        assert_eq!(
            meminfo_bytes("MemTotal: 10 kB\nMemAvailable:   47537204 kB\n", "MemAvailable"),
            Some(47_537_204 * 1024)
        );
        assert_eq!(
            meminfo_bytes("Node 1 MemFree:        123 kB\n", "MemFree"),
            Some(123 * 1024)
        );
    }

    /// A real system: eight cores with two threads each, one last-level cache for all of them,
    /// one memory node (the files of a Zen 5 desktop chip as Linux shows them).
    #[test]
    fn a_real_eight_core_system_is_read_whole() {
        let files = files_of(include_str!("testdata/sysfs_wsl2_zen5_8c16t.txt"));
        let p = profile_of(&files).expect("profile");
        assert_eq!((p.cores, p.threads), (8, 16));
        assert_eq!(p.core_map.len(), 8);
        for (i, core) in p.core_map.iter().enumerate() {
            assert_eq!(core.threads, vec![2 * i as u64, 2 * i as u64 + 1]);
        }
        assert_eq!(p.l1.len(), 8);
        assert!(p.l1.iter().all(|c| c.size == 48 * 1024 && c.threads.len() == 2));
        assert_eq!(p.l2.len(), 8);
        assert!(p.l2.iter().all(|c| c.size == 1024 * 1024 && c.threads.len() == 2));
        assert_eq!(p.l3.len(), 1);
        assert_eq!(p.l3[0].size, 96 * 1024 * 1024);
        assert_eq!(p.l3[0].threads, (0..16).collect::<Vec<u64>>());
        assert_eq!(p.ram.len(), 1);
        assert_eq!(p.ram[0].size, 47_537_204 * 1024);
        assert_eq!(p.ram[0].threads.len(), 16);
        let shares = engine::tier::Shares::from_profile(&p);
        assert_eq!(shares.active_workers, 16);
        assert_eq!(shares.num_l3_domains, 1);
        assert_eq!(shares.min_l1_share, 24 * 1024);
        assert_eq!(shares.min_l2_share, 512 * 1024);
        assert_eq!(shares.total_l3, 96 * 1024 * 1024);
    }

    /// A made-up large system in the numbering Linux uses: `sockets` packages of `cores` cores,
    /// two threads a core (thread `c` and `c + all cores`), a last-level cache for every
    /// `l3_cores` cores, a memory node a package.
    fn large_system(sockets: u64, cores: u64, l3_cores: u64) -> HashMap<String, String> {
        let all = sockets * cores;
        let mut f: HashMap<String, String> = HashMap::new();
        f.insert("sys/devices/system/cpu/online".into(), format!("0-{}\n", 2 * all - 1));
        f.insert(
            "sys/devices/system/node/online".into(),
            format!("0-{}\n", sockets - 1),
        );
        f.insert(
            "proc/meminfo".into(),
            format!("MemTotal: {} kB\nMemAvailable: {} kB\n", all * 8_388_608, all * 8_000_000),
        );
        for c in 0..all {
            for t in [c, c + all] {
                let d = format!("sys/devices/system/cpu/cpu{t}/");
                f.insert(format!("{d}topology/core_cpus_list"), format!("{c},{}\n", c + all));
                let l3_first = c / l3_cores * l3_cores;
                let l3_last = l3_first + l3_cores - 1;
                let caches = [
                    ("1", "Data", "48K", format!("{c},{}", c + all)),
                    ("1", "Instruction", "32K", format!("{c},{}", c + all)),
                    ("2", "Unified", "1024K", format!("{c},{}", c + all)),
                    (
                        "3",
                        "Unified",
                        "32768K",
                        format!("{l3_first}-{l3_last},{}-{}", l3_first + all, l3_last + all),
                    ),
                ];
                for (i, (level, kind, size, shared)) in caches.iter().enumerate() {
                    let cd = format!("{d}cache/index{i}/");
                    f.insert(format!("{cd}level"), format!("{level}\n"));
                    f.insert(format!("{cd}type"), format!("{kind}\n"));
                    f.insert(format!("{cd}size"), format!("{size}\n"));
                    f.insert(format!("{cd}shared_cpu_list"), format!("{shared}\n"));
                }
            }
        }
        for s in 0..sockets {
            let d = format!("sys/devices/system/node/node{s}/");
            let (a, b) = (s * cores, (s + 1) * cores - 1);
            f.insert(format!("{d}cpulist"), format!("{a}-{b},{}-{}\n", a + all, b + all));
            f.insert(
                format!("{d}meminfo"),
                format!("Node {s} MemTotal: {} kB\nNode {s} MemFree: {} kB\n", cores * 8_388_608, cores * 8_000_000),
            );
        }
        f
    }

    #[test]
    fn a_two_socket_system_of_384_threads_is_read_whole() {
        let files = large_system(2, 96, 8);
        let p = profile_of(&files).expect("profile");
        assert_eq!((p.cores, p.threads), (192, 384));
        assert!(p.core_map.iter().all(|c| c.threads.len() == 2));
        assert_eq!(p.core_map[5].threads, vec![5, 197]);
        assert_eq!((p.l1.len(), p.l2.len(), p.l3.len()), (192, 192, 24));
        assert!(p.l3.iter().all(|c| c.size == 32 * 1024 * 1024 && c.threads.len() == 16));
        assert_eq!(p.ram.len(), 2);
        assert_eq!(p.ram[0].threads.len(), 192);
        assert_eq!(p.ram[0].size + p.ram[1].size, 192 * 8_000_000 * 1024);
        let shares = engine::tier::Shares::from_profile(&p);
        assert_eq!(shares.active_workers, 384);
        assert_eq!(shares.num_l3_domains, 24);
        assert_eq!(shares.min_l3_domain_size, 32 * 1024 * 1024);
        assert_eq!(shares.min_l3_share, 2 * 1024 * 1024);
    }

    /// The system's own numbers of the memory nodes go with the profile: a job given the second
    /// package only has that package's node as its first, and its memory is asked from node 1.
    #[test]
    fn memory_nodes_keep_the_numbers_the_system_gives_them() {
        let nodes_of = |files: &HashMap<String, String>| -> Vec<u64> {
            super::profile_and_nodes_from_reader(&|path| files.get(path).cloned())
                .expect("profile")
                .1
        };
        let mut files = large_system(2, 96, 8);
        assert_eq!(nodes_of(&files), vec![0, 1]);
        files.insert(
            "proc/self/status".into(),
            "Cpus_allowed_list:\t96-191,288-383\nMems_allowed_list:\t1\n".into(),
        );
        let p = profile_of(&files).expect("profile");
        assert_eq!((p.cores, p.threads), (96, 192));
        assert_eq!(p.ram.len(), 1);
        assert_eq!(nodes_of(&files), vec![1]);
        // one node that the system does not name: nothing to ask for by number
        let one = files_of(include_str!("testdata/sysfs_wsl2_zen5_8c16t.txt"));
        assert_eq!(profile_of(&one).expect("profile").ram.len(), 1);
        assert!(nodes_of(&one).len() <= 1);
    }

    /// A job confined by its batch system: the threads of one package, that package's memory
    /// node, and a memory limit on a group above the job's own.
    #[test]
    fn a_confined_job_sees_only_what_it_was_given() {
        let mut files = large_system(2, 96, 8);
        files.insert(
            "proc/self/status".into(),
            "Name:\tx\nCpus_allowed_list:\t0-95,192-287\nMems_allowed_list:\t0\n".into(),
        );
        files.insert("proc/self/cgroup".into(), "0::/batch/job_7/step_0\n".into());
        files.insert("sys/fs/cgroup/batch/job_7/step_0/memory.max".into(), "max\n".into());
        files.insert("sys/fs/cgroup/batch/job_7/memory.max".into(), "68719476736\n".into());
        files.insert("sys/fs/cgroup/memory.max".into(), "max\n".into());
        let p = profile_of(&files).expect("profile");
        assert_eq!((p.cores, p.threads), (96, 192));
        assert_eq!(p.l3.len(), 12);
        assert_eq!(p.ram.len(), 1);
        assert_eq!(p.ram[0].size, 64 * 1024 * 1024 * 1024);
        assert_eq!(p.ram[0].threads.len(), 192);
        // half the threads of every core: each core is seen with the one thread it was given
        files.insert(
            "proc/self/status".into(),
            "Cpus_allowed_list:\t0-191\n".into(),
        );
        let p = profile_of(&files).expect("profile");
        assert_eq!((p.cores, p.threads), (192, 192));
        assert!(p.l1.iter().all(|c| c.threads.len() == 1));
        assert!(p.l3.iter().all(|c| c.threads.len() == 8));
    }

    /// The reader the Linux layer passes, over the real root: on a system without these files
    /// (this test also runs on other systems) it finds nothing and says so.
    #[test]
    fn the_reader_over_a_real_root_is_well_formed() {
        let p = profile_from_reader(&|path| {
            std::fs::read_to_string(std::path::Path::new("/").join(path)).ok()
        });
        if let Some(p) = p {
            assert!(p.cores >= 1 && p.threads >= p.cores);
        }
    }

    #[test]
    fn a_system_that_does_not_describe_itself_gives_none() {
        let mut files = large_system(1, 4, 4);
        assert!(profile_of(&files).is_some());
        files.remove("sys/devices/system/cpu/online");
        assert!(profile_of(&files).is_none());
        let mut files = large_system(1, 4, 4);
        files.retain(|k, _| !k.contains("/cache/"));
        assert!(profile_of(&files).is_none());
        let mut files = large_system(1, 4, 4);
        files.remove("proc/meminfo");
        assert!(profile_of(&files).is_none());
    }
}
