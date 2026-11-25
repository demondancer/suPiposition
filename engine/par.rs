//! Contract:
//! - Inputs: Ranked `Shares` worker list, optional OS `ThreadPlacer` callback, recursive fork-join
//!   closures (`join`) for subtrees, and parallel slice/tile closures (`par_for`) for transforms.
//! - Outputs: Deterministic results identical to sequential execution, executed across persistent
//!   worker threads woken strictly in rank order (fastest core's first thread first). A capped
//!   loop started outside every pool job runs on the top-ranked workers only; a nested one may
//!   use any worker, its cap being a count.
//! - Data format: Stack-scoped erased job descriptors (`JobRef`) for `join` deques and shared
//!   loop descriptors for `par_for`, with zero per-item heap allocation.
//! - Work & bytes moved: O(1) local deque push/pop when uncontended; idle workers sleep on condvars
//!   with zero spin burn.

use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::tier::Shares;

/// OS-agnostic thread placement interface handed in from `platform`.
pub trait ThreadPlacer: Send + Sync {
    fn place_current_thread(&self, hw_thread: u64, os_places_threads: bool);
}

thread_local! {
    static TLS_WORKER_IDX: Cell<usize> = const { Cell::new(0) };
    static TLS_CORE_ID: Cell<u64> = const { Cell::new(0) };
    /// How many pool jobs or loop bodies enclose the code this thread is running.
    static TLS_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Marks the current thread as running inside a pool job for the guard's lifetime.
struct DepthGuard;

impl DepthGuard {
    #[inline]
    fn enter() -> Self {
        TLS_DEPTH.with(|c| c.set(c.get() + 1));
        DepthGuard
    }
}

impl Drop for DepthGuard {
    #[inline]
    fn drop(&mut self) {
        TLS_DEPTH.with(|c| c.set(c.get() - 1));
    }
}

/// A capped loop started outside every pool job is the only work in flight, so its cap names
/// the top-ranked workers (rank order, section 6). A nested loop shares the pool with its
/// siblings, so its cap is a count only and any worker may take its tasks.
#[inline]
fn rank_limit(cap_workers: usize) -> usize {
    if TLS_DEPTH.with(|c| c.get()) == 0 {
        cap_workers
    } else {
        usize::MAX
    }
}

#[inline]
pub fn current_worker_idx() -> usize {
    TLS_WORKER_IDX.with(|c| c.get())
}

#[inline]
pub fn current_core_id() -> u64 {
    TLS_CORE_ID.with(|c| c.get())
}

// Trace-run accounting of the time workers spend without work: between jobs (`idle`) and
// waiting inside `join` / `par_for` for tasks that other workers took (`wait`).
static PAR_IDLE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PAR_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn par_trace_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("SUPI_TRACE").is_some())
}

static PAR_IDLE_MARK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Trace runs: the idle time counted since the last call. A worker's wait is counted when its
/// next job starts, so what a phase reports is the wait before and at the start of it.
pub fn idle_since_mark() -> u64 {
    let now = PAR_IDLE_NS.load(Ordering::Relaxed);
    let prev = PAR_IDLE_MARK_NS.swap(now, Ordering::Relaxed);
    now.saturating_sub(prev)
}

/// Print and clear the idle and waiting time gathered since the last call (trace runs only).
pub fn dump_and_reset_par_times(tag: &str, wall_sec: f64, workers: usize) {
    if !par_trace_enabled() {
        return;
    }
    let idle = PAR_IDLE_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let wait = PAR_WAIT_NS.swap(0, Ordering::Relaxed) as f64 * 1e-6;
    let total = wall_sec * 1e3 * (workers as f64);
    eprintln!(
        "[TRACE PAR {tag}] thread time {total:.0}ms: idle between jobs {idle:.0}ms ({:.0}%), waiting on stolen tasks {wait:.0}ms ({:.0}%)",
        if total > 0.0 { 100.0 * idle / total } else { 0.0 },
        if total > 0.0 { 100.0 * wait / total } else { 0.0 },
    );
}

/// Times a wait loop, leaving out the stolen jobs it runs meanwhile.
struct WaitClock {
    start: Option<std::time::Instant>,
    exec_ns: u64,
}

impl WaitClock {
    #[inline]
    fn start() -> Self {
        Self {
            start: par_trace_enabled().then(std::time::Instant::now),
            exec_ns: 0,
        }
    }

    #[inline]
    unsafe fn run(&mut self, job: JobPtr) {
        if self.start.is_some() {
            let t = std::time::Instant::now();
            unsafe { ((*job.0).exec)(job.0) };
            self.exec_ns += t.elapsed().as_nanos() as u64;
        } else {
            unsafe { ((*job.0).exec)(job.0) };
        }
    }
}

impl Drop for WaitClock {
    #[inline]
    fn drop(&mut self) {
        if let Some(t) = self.start {
            let ns = (t.elapsed().as_nanos() as u64).saturating_sub(self.exec_ns);
            PAR_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy)]
struct JobPtr(*const JobHeader);
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

struct JobHeader {
    exec: unsafe fn(*const JobHeader),
    is_leaf: bool,
    max_worker_idx: usize,
}

struct OuterScopeGuard<'a>(&'a AtomicUsize);

impl Drop for OuterScopeGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct WorkerSlot {
    deque: Mutex<VecDeque<JobPtr>>,
    deque_len: AtomicUsize,
    is_sleeping: AtomicBool,
    sleep_lock: Mutex<bool>, // true = sleeping
    sleep_cv: Condvar,
}

struct PoolInner {
    num_workers: usize,
    shutdown: AtomicBool,
    active_jobs: AtomicUsize,
    outer_scopes: AtomicUsize,
    slots: Vec<WorkerSlot>,
}

impl PoolInner {
    fn wake_fastest_sleeping(&self, count: usize, max_worker_idx: usize) {
        if count == 0 {
            return;
        }
        let mut woken = 0usize;
        let limit = max_worker_idx.min(self.num_workers);
        // Scan strictly in rank order 1..limit (worker 0 is the main coordinator thread).
        for w in 1..limit {
            if !self.slots[w].is_sleeping.load(Ordering::SeqCst) {
                continue;
            }
            let mut did_wake = false;
            {
                let mut sleeping = self.slots[w].sleep_lock.lock().unwrap();
                if *sleeping {
                    *sleeping = false;
                    self.slots[w].is_sleeping.store(false, Ordering::SeqCst);
                    did_wake = true;
                }
            }
            if did_wake {
                self.slots[w].sleep_cv.notify_one();
                woken += 1;
                if woken >= count {
                    break;
                }
            }
        }
    }

    fn try_pop_or_steal(&self, my_idx: usize) -> Option<JobPtr> {
        if self.slots[my_idx].deque_len.load(Ordering::Acquire) > 0 {
            let mut dq = self.slots[my_idx].deque.lock().unwrap();
            if let Some(pos) = dq
                .iter()
                .rposition(|p| unsafe { my_idx < (*p.0).max_worker_idx })
            {
                let job = dq.remove(pos);
                self.slots[my_idx].deque_len.store(dq.len(), Ordering::Release);
                if job.is_some() {
                    return job;
                }
            }
        }
        if self.active_jobs.load(Ordering::Acquire) == 0 {
            return None;
        }
        for offset in 1..self.num_workers {
            let victim = (my_idx + offset) % self.num_workers;
            if self.slots[victim].deque_len.load(Ordering::Acquire) == 0 {
                continue;
            }
            let mut dq = self.slots[victim].deque.lock().unwrap();
            if let Some(pos) = dq
                .iter()
                .position(|p| unsafe { my_idx < (*p.0).max_worker_idx })
            {
                let job = dq.remove(pos);
                self.slots[victim].deque_len.store(dq.len(), Ordering::Release);
                if job.is_some() {
                    return job;
                }
            }
        }
        None
    }

    fn has_eligible_job(&self, my_idx: usize) -> bool {
        if self.active_jobs.load(Ordering::SeqCst) == 0
            && self.slots[my_idx].deque_len.load(Ordering::Acquire) == 0
        {
            return false;
        }
        for slot in &self.slots {
            if slot.deque_len.load(Ordering::Acquire) == 0 {
                continue;
            }
            let dq = slot.deque.lock().unwrap();
            if dq.iter().any(|p| unsafe { my_idx < (*p.0).max_worker_idx }) {
                return true;
            }
        }
        false
    }

    fn try_steal_leaf(&self, my_idx: usize) -> Option<JobPtr> {
        if self.active_jobs.load(Ordering::Acquire) == 0 {
            return None;
        }
        for offset in 1..self.num_workers {
            let victim = (my_idx + offset) % self.num_workers;
            if self.slots[victim].deque_len.load(Ordering::Acquire) == 0 {
                continue;
            }
            let mut dq = self.slots[victim].deque.lock().unwrap();
            if let Some(pos) = dq
                .iter()
                .position(|p| unsafe { (*p.0).is_leaf && my_idx < (*p.0).max_worker_idx })
            {
                let job = dq.remove(pos);
                self.slots[victim].deque_len.store(dq.len(), Ordering::Release);
                if job.is_some() {
                    self.active_jobs.fetch_sub(1, Ordering::SeqCst);
                    return job;
                }
            }
        }
        None
    }
}

pub struct Scheduler {
    inner: Arc<PoolInner>,
    handles: Vec<JoinHandle<()>>,
    placer: Option<&'static dyn ThreadPlacer>,
    os_places: bool,
    worker0_hw_tid: u64,
    worker0_core_id: u64,
    num_cores: usize,
    min_l1_share: u64,
    min_l2_share: u64,
}

impl Scheduler {
    pub fn new(shares: &Shares, placer: Option<&'static dyn ThreadPlacer>) -> Self {
        let num_workers = shares.active_workers.max(1);
        let mut slots = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            slots.push(WorkerSlot {
                deque: Mutex::new(VecDeque::with_capacity(64)),
                deque_len: AtomicUsize::new(0),
                is_sleeping: AtomicBool::new(false),
                sleep_lock: Mutex::new(false),
                sleep_cv: Condvar::new(),
            });
        }

        let inner = Arc::new(PoolInner {
            num_workers,
            shutdown: AtomicBool::new(false),
            active_jobs: AtomicUsize::new(0),
            outer_scopes: AtomicUsize::new(0),
            slots,
        });

        let w0_hw = shares.workers.first().map(|w| w.hw_thread).unwrap_or(0);
        let w0_core = shares.workers.first().map(|w| w.core_id).unwrap_or(0);
        let os_places = shares.os_places_threads;

        if let Some(p) = placer {
            p.place_current_thread(w0_hw, os_places);
        }
        TLS_WORKER_IDX.with(|c| c.set(0));
        TLS_CORE_ID.with(|c| c.set(w0_core));

        let mut handles = Vec::with_capacity(num_workers.saturating_sub(1));
        for w_idx in 1..num_workers {
            let pool = Arc::clone(&inner);
            let hw_tid = shares.workers[w_idx].hw_thread;
            let core_id = shares.workers[w_idx].core_id;
            let handle = thread::Builder::new()
                .name(format!("supi-w{w_idx}"))
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    TLS_WORKER_IDX.with(|c| c.set(w_idx));
                    TLS_CORE_ID.with(|c| c.set(core_id));
                    if let Some(p) = placer {
                        p.place_current_thread(hw_tid, os_places);
                    }
                    worker_loop(&pool, w_idx);
                })
                .expect("spawn worker thread");
            handles.push(handle);
        }

        let num_cores = (shares.cores.min(num_workers as u64) as usize).clamp(1, num_workers);
        let min_l1_share = shares.min_l1_share;
        let min_l2_share = shares.min_l2_share;

        Self {
            inner,
            handles,
            placer,
            os_places,
            worker0_hw_tid: w0_hw,
            worker0_core_id: w0_core,
            num_cores,
            min_l1_share,
            min_l2_share,
        }
    }

    #[inline]
    pub fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    #[inline]
    pub fn num_cores(&self) -> usize {
        self.num_cores
    }

    #[inline]
    pub fn min_l1_share(&self) -> u64 {
        self.min_l1_share
    }

    #[inline]
    pub fn min_l2_share(&self) -> u64 {
        self.min_l2_share
    }

    /// Subtree threshold depth rule from Section 10:
    /// "Below the depth at which the tree has sixteen times as many nodes as workers, a subtree
    /// runs on one worker, and stealing evens out the rest; above that depth, products are parallel
    /// across primes and tiles."
    #[inline]
    pub fn subtree_steal_max_nodes(&self) -> usize {
        16usize.saturating_mul(self.inner.num_workers)
    }

    pub fn bind_caller_as_worker0(&self) {
        TLS_WORKER_IDX.with(|c| c.set(0));
        TLS_CORE_ID.with(|c| c.set(self.worker0_core_id));
        if let Some(p) = self.placer {
            p.place_current_thread(self.worker0_hw_tid, self.os_places);
        }
    }

    /// Fork-join two closures across the work-stealing pool.
    pub fn join<A, B, FA, FB>(&self, fa: FA, fb: FB) -> (A, B)
    where
        A: Send,
        B: Send,
        FA: FnOnce() -> A + Send,
        FB: FnOnce() -> B + Send,
    {
        if self.inner.num_workers <= 1 {
            return (fa(), fb());
        }

        let my_idx = current_worker_idx().min(self.inner.num_workers - 1);

        #[repr(C)]
        struct StackJob<F, R> {
            header: JobHeader,
            func: Mutex<Option<F>>,
            result: Mutex<Option<thread::Result<R>>>,
            thief_idx: AtomicUsize,
            done: AtomicBool,
            waiter_lock: Mutex<()>,
            waiter_cv: Condvar,
        }

        unsafe fn run_stack_job<F: FnOnce() -> R + Send, R: Send>(hdr: *const JobHeader) {
            let job = unsafe { &*(hdr as *const StackJob<F, R>) };
            let waiter_lock = &job.waiter_lock;
            let waiter_cv = &job.waiter_cv;
            let done = &job.done;
            let f_opt = job.func.lock().unwrap().take();
            if let Some(f) = f_opt {
                job.thief_idx.store(current_worker_idx(), Ordering::Release);
                let _depth = DepthGuard::enter();
                let res = catch_unwind(AssertUnwindSafe(f));
                *job.result.lock().unwrap() = Some(res);
                let _g = waiter_lock.lock().unwrap();
                done.store(true, Ordering::Release);
                waiter_cv.notify_one();
                drop(_g);
            }
        }

        self.inner.outer_scopes.fetch_add(1, Ordering::AcqRel);
        let _scope_guard = OuterScopeGuard(&self.inner.outer_scopes);

        let job_b = StackJob::<FB, B> {
            header: JobHeader {
                exec: run_stack_job::<FB, B>,
                is_leaf: false,
                max_worker_idx: usize::MAX,
            },
            func: Mutex::new(Some(fb)),
            result: Mutex::new(None),
            thief_idx: AtomicUsize::new(usize::MAX),
            done: AtomicBool::new(false),
            waiter_lock: Mutex::new(()),
            waiter_cv: Condvar::new(),
        };

        {
            let mut dq = self.inner.slots[my_idx].deque.lock().unwrap();
            dq.push_back(JobPtr(&job_b.header as *const JobHeader));
            self.inner.slots[my_idx]
                .deque_len
                .store(dq.len(), Ordering::Release);
            self.inner.active_jobs.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.wake_fastest_sleeping(1, self.inner.num_workers);

        let _depth = DepthGuard::enter();
        let res_a = catch_unwind(AssertUnwindSafe(fa));

        // Try popping `job_b` back locally first.
        let popped_local = if self.inner.slots[my_idx].deque_len.load(Ordering::Acquire) > 0 {
            let mut dq = self.inner.slots[my_idx].deque.lock().unwrap();
            if let Some(pos) = dq.iter().rposition(|p| std::ptr::eq(p.0, &job_b.header)) {
                dq.remove(pos);
                self.inner.slots[my_idx]
                    .deque_len
                    .store(dq.len(), Ordering::Release);
                self.inner.active_jobs.fetch_sub(1, Ordering::SeqCst);
                true
            } else {
                false
            }
        } else {
            false
        };

        if popped_local {
            let fb_local = job_b.func.lock().unwrap().take().unwrap();
            let res_b = catch_unwind(AssertUnwindSafe(fb_local));
            let a = match res_a {
                Ok(v) => v,
                Err(p) => resume_unwind(p),
            };
            let b = match res_b {
                Ok(v) => v,
                Err(p) => resume_unwind(p),
            };
            return (a, b);
        }

        // Stolen by another worker: leapfrog from the thief's deque (tasks inside `job_b`'s
        // subtree) or steal bounded leaf sweep tasks (`is_leaf == true`) across any worker.
        let mut wait_clock = WaitClock::start();
        while !job_b.done.load(Ordering::Acquire) {
            let thief = job_b.thief_idx.load(Ordering::Acquire);
            let stolen_sub = if thief < self.inner.num_workers
                && self.inner.slots[thief].deque_len.load(Ordering::Acquire) > 0
            {
                let mut dq = self.inner.slots[thief].deque.lock().unwrap();
                let j = dq.pop_front();
                self.inner.slots[thief]
                    .deque_len
                    .store(dq.len(), Ordering::Release);
                if j.is_some() {
                    self.inner.active_jobs.fetch_sub(1, Ordering::SeqCst);
                }
                j
            } else {
                None
            };

            if let Some(sub) = stolen_sub {
                unsafe { wait_clock.run(sub) };
                continue;
            }

            if let Some(leaf) = self.inner.try_steal_leaf(my_idx) {
                unsafe { wait_clock.run(leaf) };
                continue;
            }

            let mut progressed = false;
            for s in 0..32_768u32 {
                if job_b.done.load(Ordering::Acquire) {
                    progressed = true;
                    break;
                }
                if (s & 511) == 511 {
                    let t = job_b.thief_idx.load(Ordering::Acquire);
                    if (t < self.inner.num_workers
                        && self.inner.slots[t].deque_len.load(Ordering::Acquire) > 0)
                        || self.inner.active_jobs.load(Ordering::Acquire) > 0
                    {
                        if let Some(leaf) = self.inner.try_steal_leaf(my_idx) {
                            unsafe { wait_clock.run(leaf) };
                            progressed = true;
                            break;
                        }
                        if t < self.inner.num_workers
                            && self.inner.slots[t].deque_len.load(Ordering::Acquire) > 0
                        {
                            progressed = true;
                            break;
                        }
                    }
                }
                std::hint::spin_loop();
            }
            if !progressed {
                let g = job_b.waiter_lock.lock().unwrap();
                if !job_b.done.load(Ordering::Acquire) {
                    let _ = job_b
                        .waiter_cv
                        .wait_timeout(g, std::time::Duration::from_millis(1));
                }
            }
        }

        drop(wait_clock);
        // Fence against thief's `waiter_lock` unlock so `job_b` stays live until thief exits.
        let _sync = job_b.waiter_lock.lock().unwrap();
        drop(_sync);

        let a = match res_a {
            Ok(v) => v,
            Err(p) => resume_unwind(p),
        };
        let b_res = job_b.result.lock().unwrap().take().unwrap();
        let b = match b_res {
            Ok(v) => v,
            Err(p) => resume_unwind(p),
        };
        (a, b)
    }

    /// Execute `body(item_idx)` in parallel for `0..count`, waking at most as many top-ranked
    /// workers as there are chunks of work.
    #[inline]
    pub fn par_for<F>(&self, count: usize, chunk_size: usize, body: F)
    where
        F: Fn(usize) + Sync,
    {
        self.par_for_max(count, chunk_size, self.inner.num_workers, body);
    }

    /// Execute `f()` once on each of the pool's workers.
    pub fn on_each_worker<F>(&self, f: F)
    where
        F: Fn() + Sync,
    {
        self.bind_caller_as_worker0();
        let nw = self.inner.num_workers;
        if nw <= 1 {
            f();
            return;
        }
        let barrier = std::sync::Barrier::new(nw);
        self.par_for(nw, 1, |_| {
            f();
            barrier.wait();
        });
    }

    /// Execute `body(item_idx)` in parallel for `0..count` across at most `max_workers` workers.
    #[inline]
    pub fn par_for_max<F>(&self, count: usize, chunk_size: usize, max_workers: usize, body: F)
    where
        F: Fn(usize) + Sync,
    {
        self.par_for_inner(count, chunk_size, max_workers, false, body);
    }

    /// Execute a leaf data-parallel loop `0..count` across all workers (`is_leaf = true` so
    /// workers waiting at outer barriers steal chunks immediately without sleeping).
    #[inline]
    pub fn par_for_leaf<F>(&self, count: usize, chunk_size: usize, body: F)
    where
        F: Fn(usize) + Sync,
    {
        self.par_for_inner(count, chunk_size, self.inner.num_workers, true, body);
    }

    /// Execute a leaf data-parallel loop `0..count` across at most `max_workers` workers.
    #[inline]
    pub fn par_for_leaf_max<F>(&self, count: usize, chunk_size: usize, max_workers: usize, body: F)
    where
        F: Fn(usize) + Sync,
    {
        self.par_for_inner(count, chunk_size, max_workers, true, body);
    }

    fn par_for_inner<F>(
        &self,
        count: usize,
        chunk_size: usize,
        max_workers: usize,
        is_leaf: bool,
        body: F,
    ) where
        F: Fn(usize) + Sync,
    {
        let chunk = chunk_size.max(1);
        if count == 0 {
            return;
        }
        let cap_workers = max_workers.min(self.inner.num_workers);
        if count <= chunk || cap_workers <= 1 {
            for i in 0..count {
                body(i);
            }
            std::sync::atomic::fence(Ordering::SeqCst);
            return;
        }

        let total_chunks = count.div_ceil(chunk);
        let tasks = total_chunks.min(cap_workers);
        if tasks <= 1 {
            for i in 0..count {
                body(i);
            }
            std::sync::atomic::fence(Ordering::SeqCst);
            return;
        }

        let next_idx = AtomicUsize::new(0);
        let body_ref: &dyn Fn(usize) = &body;

        #[repr(C)]
        struct ParTask<'a> {
            header: JobHeader,
            slot_idx: usize,
            next_idx: &'a AtomicUsize,
            count: usize,
            chunk: usize,
            body: &'a (dyn Fn(usize) + Sync),
            remaining_tasks: &'a AtomicUsize,
            all_done: &'a AtomicBool,
            done_lock: &'a Mutex<()>,
            done_cv: &'a Condvar,
        }

        // Erase lifetime onto stack frame safely because we block until `all_done` and acquire
        // `done_lock` before returning from `par_for`.
        unsafe fn exec_par_task(hdr: *const JobHeader) {
            let t = unsafe { &*(hdr as *const ParTask<'static>) };
            let next_idx = t.next_idx;
            let count = t.count;
            let chunk = t.chunk;
            let body = t.body;
            let remaining_tasks = t.remaining_tasks;
            let all_done = t.all_done;
            let done_lock = t.done_lock;
            let done_cv = t.done_cv;
            {
                let _depth = DepthGuard::enter();
                loop {
                    let start = next_idx.fetch_add(chunk, Ordering::Relaxed);
                    if start >= count {
                        break;
                    }
                    let end = (start + chunk).min(count);
                    for i in start..end {
                        body(i);
                    }
                }
            }
            if remaining_tasks.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _g = done_lock.lock().unwrap();
                all_done.store(true, Ordering::Release);
                done_cv.notify_one();
                drop(_g);
            }
        }

        let body_sync: &(dyn Fn(usize) + Sync) = &body;
        let _ = body_ref;
        let remaining = AtomicUsize::new(tasks - 1);
        let all_done = AtomicBool::new(false);
        let done_lock = Mutex::new(());
        let done_cv = Condvar::new();

        let my_idx = current_worker_idx().min(self.inner.num_workers - 1);
        let max_worker_idx = rank_limit(cap_workers);
        let deal_direct = max_worker_idx < usize::MAX;
        let mut task_nodes = Vec::with_capacity(tasks - 1);
        for k in 0..(tasks - 1) {
            let slot_idx = if deal_direct {
                let cand = if k < my_idx { k } else { k + 1 };
                cand.min(self.inner.num_workers - 1)
            } else {
                my_idx
            };
            task_nodes.push(ParTask {
                header: JobHeader {
                    exec: exec_par_task,
                    is_leaf,
                    max_worker_idx,
                },
                slot_idx,
                next_idx: &next_idx,
                count,
                chunk,
                body: body_sync,
                remaining_tasks: &remaining,
                all_done: &all_done,
                done_lock: &done_lock,
                done_cv: &done_cv,
            });
        }

        self.inner.active_jobs.fetch_add(tasks - 1, Ordering::SeqCst);
        if deal_direct {
            for node in &task_nodes {
                let s_idx = node.slot_idx;
                let mut dq = self.inner.slots[s_idx].deque.lock().unwrap();
                dq.push_back(JobPtr(&node.header as *const JobHeader));
                self.inner.slots[s_idx]
                    .deque_len
                    .store(dq.len(), Ordering::Release);
            }
        } else {
            let mut dq = self.inner.slots[my_idx].deque.lock().unwrap();
            for node in &task_nodes {
                dq.push_back(JobPtr(&node.header as *const JobHeader));
            }
            self.inner.slots[my_idx]
                .deque_len
                .store(dq.len(), Ordering::Release);
        }
        // Wake fastest sleeping workers first!
        self.inner.wake_fastest_sleeping(tasks - 1, max_worker_idx);

        // Caller also drains chunks!
        let _depth = DepthGuard::enter();
        loop {
            let start = next_idx.fetch_add(chunk, Ordering::Relaxed);
            if start >= count {
                break;
            }
            let end = (start + chunk).min(count);
            for i in start..end {
                body(i);
            }
        }

        // Reclaim any unpopped ParTask nodes locally, then assist leaf sweeps while active thieves finish.
        let mut wait_clock = WaitClock::start();
        loop {
            if all_done.load(Ordering::Acquire) {
                break;
            }
            let mut popped = None;
            for node in &task_nodes {
                let s_idx = node.slot_idx;
                if self.inner.slots[s_idx].deque_len.load(Ordering::Acquire) == 0 {
                    continue;
                }
                let mut dq = self.inner.slots[s_idx].deque.lock().unwrap();
                let target = &node.header as *const JobHeader;
                if let Some(pos) = dq.iter().rposition(|p| std::ptr::eq(p.0, target)) {
                    popped = dq.remove(pos);
                    self.inner.slots[s_idx]
                        .deque_len
                        .store(dq.len(), Ordering::Release);
                    self.inner.active_jobs.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
            }
            if let Some(jp) = popped {
                unsafe { wait_clock.run(jp) };
                continue;
            }
            if !is_leaf {
                if let Some(leaf) = self.inner.try_steal_leaf(my_idx) {
                    unsafe { wait_clock.run(leaf) };
                    continue;
                }
            }
            let mut progressed = false;
            for s in 0..131_072u32 {
                if all_done.load(Ordering::Acquire) {
                    progressed = true;
                    break;
                }
                if !is_leaf && (s & 511) == 511 {
                    if let Some(leaf) = self.inner.try_steal_leaf(my_idx) {
                        unsafe { wait_clock.run(leaf) };
                        progressed = true;
                        break;
                    }
                }
                std::hint::spin_loop();
            }
            if !progressed {
                let g = done_lock.lock().unwrap();
                if !all_done.load(Ordering::Acquire) {
                    let _ = done_cv.wait_timeout(g, std::time::Duration::from_micros(250));
                }
            }
        }
        drop(wait_clock);
        let _sync = done_lock.lock().unwrap();
        drop(_sync);
    }

    /// Execute `0..count` in chunks of `chunk_size` across all workers, initializing one
    /// task-local state `S` lazily per active worker task so scratch buffers stay hot in L1/L2
    /// without per-item heap allocations.
    #[inline]
    pub fn par_for_init<S, Init, F>(&self, count: usize, chunk_size: usize, init: Init, body: F)
    where
        Init: Fn() -> S + Sync,
        F: Fn(&mut S, usize) + Sync,
    {
        self.par_for_init_max(count, chunk_size, self.inner.num_workers, init, body);
    }

    /// Execute `0..count` in chunks of `chunk_size` across at most `max_workers` workers.
    pub fn par_for_init_max<S, Init, F>(
        &self,
        count: usize,
        chunk_size: usize,
        max_workers: usize,
        init: Init,
        body: F,
    ) where
        Init: Fn() -> S + Sync,
        F: Fn(&mut S, usize) + Sync,
    {
        if count == 0 {
            return;
        }
        let chunk = chunk_size.max(1);
        let cap_workers = max_workers.min(self.inner.num_workers);
        if count <= chunk || cap_workers <= 1 {
            let mut st = init();
            for i in 0..count {
                body(&mut st, i);
            }
            return;
        }

        let total_chunks = count.div_ceil(chunk);
        let tasks = total_chunks.min(cap_workers);
        if tasks <= 1 {
            let mut st = init();
            for i in 0..count {
                body(&mut st, i);
            }
            return;
        }

        let next_idx = AtomicUsize::new(0);

        #[repr(C)]
        struct ParInitTask<'a, S> {
            header: JobHeader,
            slot_idx: usize,
            next_idx: &'a AtomicUsize,
            count: usize,
            chunk: usize,
            init: &'a (dyn Fn() -> S + Sync),
            body: &'a (dyn Fn(&mut S, usize) + Sync),
            remaining_tasks: &'a AtomicUsize,
            all_done: &'a AtomicBool,
            done_lock: &'a Mutex<()>,
            done_cv: &'a Condvar,
        }

        unsafe fn exec_par_init_task<S>(hdr: *const JobHeader) {
            let t = unsafe { &*(hdr as *const ParInitTask<'_, S>) };
            let next_idx = t.next_idx;
            let count = t.count;
            let chunk = t.chunk;
            let init = t.init;
            let body = t.body;
            let remaining_tasks = t.remaining_tasks;
            let all_done = t.all_done;
            let done_lock = t.done_lock;
            let done_cv = t.done_cv;
            {
                let _depth = DepthGuard::enter();
                let mut state_opt: Option<S> = None;
                loop {
                    let start = next_idx.fetch_add(chunk, Ordering::Relaxed);
                    if start >= count {
                        break;
                    }
                    let st = state_opt.get_or_insert_with(|| init());
                    let end = (start + chunk).min(count);
                    for i in start..end {
                        body(st, i);
                    }
                }
            }
            if remaining_tasks.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _g = done_lock.lock().unwrap();
                all_done.store(true, Ordering::Release);
                done_cv.notify_one();
                drop(_g);
            }
        }

        let init_sync: &(dyn Fn() -> S + Sync) = &init;
        let body_sync: &(dyn Fn(&mut S, usize) + Sync) = &body;
        let remaining = AtomicUsize::new(tasks - 1);
        let all_done = AtomicBool::new(false);
        let done_lock = Mutex::new(());
        let done_cv = Condvar::new();

        let my_idx = current_worker_idx().min(self.inner.num_workers - 1);
        let max_worker_idx = rank_limit(cap_workers);
        let deal_direct = max_worker_idx < usize::MAX;
        let mut task_nodes = Vec::with_capacity(tasks - 1);
        for k in 0..(tasks - 1) {
            let slot_idx = if deal_direct {
                let cand = if k < my_idx { k } else { k + 1 };
                cand.min(self.inner.num_workers - 1)
            } else {
                my_idx
            };
            task_nodes.push(ParInitTask::<S> {
                header: JobHeader {
                    exec: exec_par_init_task::<S>,
                    is_leaf: true,
                    max_worker_idx,
                },
                slot_idx,
                next_idx: &next_idx,
                count,
                chunk,
                init: init_sync,
                body: body_sync,
                remaining_tasks: &remaining,
                all_done: &all_done,
                done_lock: &done_lock,
                done_cv: &done_cv,
            });
        }

        self.inner.active_jobs.fetch_add(tasks - 1, Ordering::SeqCst);
        if deal_direct {
            for node in &task_nodes {
                let s_idx = node.slot_idx;
                let mut dq = self.inner.slots[s_idx].deque.lock().unwrap();
                dq.push_back(JobPtr(&node.header as *const JobHeader));
                self.inner.slots[s_idx]
                    .deque_len
                    .store(dq.len(), Ordering::Release);
            }
        } else {
            let mut dq = self.inner.slots[my_idx].deque.lock().unwrap();
            for node in &task_nodes {
                dq.push_back(JobPtr(&node.header as *const JobHeader));
            }
            self.inner.slots[my_idx]
                .deque_len
                .store(dq.len(), Ordering::Release);
        }
        self.inner.wake_fastest_sleeping(tasks - 1, max_worker_idx);

        let _depth = DepthGuard::enter();
        let mut caller_state: Option<S> = None;
        loop {
            let start = next_idx.fetch_add(chunk, Ordering::Relaxed);
            if start >= count {
                break;
            }
            let st = caller_state.get_or_insert_with(&init);
            let end = (start + chunk).min(count);
            for i in start..end {
                body(st, i);
            }
        }
        drop(caller_state);

        let mut wait_clock = WaitClock::start();
        loop {
            if all_done.load(Ordering::Acquire) {
                break;
            }
            let mut popped = None;
            for node in &task_nodes {
                let s_idx = node.slot_idx;
                if self.inner.slots[s_idx].deque_len.load(Ordering::Acquire) == 0 {
                    continue;
                }
                let mut dq = self.inner.slots[s_idx].deque.lock().unwrap();
                let target = &node.header as *const JobHeader;
                if let Some(pos) = dq.iter().rposition(|p| std::ptr::eq(p.0, target)) {
                    popped = dq.remove(pos);
                    self.inner.slots[s_idx]
                        .deque_len
                        .store(dq.len(), Ordering::Release);
                    self.inner.active_jobs.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
            }
            if let Some(jp) = popped {
                unsafe { wait_clock.run(jp) };
                continue;
            }
            let mut progressed = false;
            for _ in 0..131_072u32 {
                if all_done.load(Ordering::Acquire) {
                    progressed = true;
                    break;
                }
                std::hint::spin_loop();
            }
            if !progressed {
                let g = done_lock.lock().unwrap();
                if !all_done.load(Ordering::Acquire) {
                    let _ = done_cv.wait_timeout(g, std::time::Duration::from_micros(250));
                }
            }
        }
        drop(wait_clock);
        let _sync = done_lock.lock().unwrap();
        drop(_sync);
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        for slot in &self.inner.slots {
            let mut sl = slot.sleep_lock.lock().unwrap();
            *sl = false;
            slot.is_sleeping.store(false, Ordering::SeqCst);
            slot.sleep_cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(pool: &PoolInner, my_idx: usize) {
    let trace = par_trace_enabled();
    let mut idle_since = trace.then(std::time::Instant::now);
    let mut just_worked = false;
    loop {
        if pool.shutdown.load(Ordering::Acquire) {
            return;
        }
        if pool.slots[my_idx].deque_len.load(Ordering::Acquire) > 0
            || pool.active_jobs.load(Ordering::Acquire) > 0
        {
            if let Some(job) = pool.try_pop_or_steal(my_idx) {
                pool.active_jobs.fetch_sub(1, Ordering::SeqCst);
                if let Some(t) = idle_since {
                    PAR_IDLE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                unsafe { ((*job.0).exec)(job.0) };
                if trace {
                    idle_since = Some(std::time::Instant::now());
                }
                just_worked = true;
                continue;
            }
        }
        if just_worked {
            let mut found = false;
            for spin in 0..131_072u32 {
                if pool.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                if pool.slots[my_idx].deque_len.load(Ordering::Acquire) > 0
                    || ((spin & 3) == 0 && pool.active_jobs.load(Ordering::Acquire) > 0)
                {
                    if let Some(job) = pool.try_pop_or_steal(my_idx) {
                        pool.active_jobs.fetch_sub(1, Ordering::SeqCst);
                        if let Some(t) = idle_since {
                            PAR_IDLE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        unsafe { ((*job.0).exec)(job.0) };
                        if trace {
                            idle_since = Some(std::time::Instant::now());
                        }
                        found = true;
                        break;
                    }
                }
                std::hint::spin_loop();
            }
            if found {
                continue;
            }
            just_worked = false;
        }
        let mut sleeping = pool.slots[my_idx].sleep_lock.lock().unwrap();
        pool.slots[my_idx].is_sleeping.store(true, Ordering::SeqCst);
        if pool.shutdown.load(Ordering::SeqCst) {
            pool.slots[my_idx].is_sleeping.store(false, Ordering::Relaxed);
            return;
        }
        if !pool.has_eligible_job(my_idx) {
            *sleeping = true;
            while *sleeping && !pool.shutdown.load(Ordering::Acquire) {
                sleeping = pool.slots[my_idx].sleep_cv.wait(sleeping).unwrap();
            }
        }
        pool.slots[my_idx].is_sleeping.store(false, Ordering::SeqCst);
    }
}
