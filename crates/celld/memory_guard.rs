// Copyright 2026 Deno Land Inc. Apache-2.0 license.

//! Terminates running JavaScript before the process runs out of memory.
//!
//! The ArrayBuffer budget (`array_buffer_budget`) bounds what goes through
//! V8's embedder allocator. Two kinds of memory never do: a resizable
//! ArrayBuffer (`new ArrayBuffer(n, { maxByteLength })`) and WebAssembly
//! memory. V8 reserves their address space up front and commits pages itself
//! on `resize` and `grow`. Measured on 0.6.3 in a 2 GiB guest: a loop that
//! grows either one is OOM-killed, budget or not, and so is a loop that keeps
//! many small ones.
//!
//! This guard is the backstop for those and for anything else. A thread reads
//! the process's resident plus swapped memory every 10 ms. Above the line it
//! asks every isolate for an interrupt, again every 50 ms while the process
//! stays over: V8 keeps interrupt flags in per-thread state, and an isolate
//! that moves between threads under its `Locker` can lose a flag set while
//! it was unlocked. The callback stays queued and runs with the next request
//! that does land, so a re-request is what delivers it. The interrupt runs on the isolate's
//! own thread, inside whatever JavaScript is running, or at its next entry
//! if it is idle. It collects garbage first and gives the freed memory time
//! to leave (`still_over_after_collection`), so memory that only garbage
//! holds never fails a request. If the process is still
//! over the line, it terminates that execution with "Worker exceeded its
//! memory limit", the way an exceeded CPU limit does, and the node keeps
//! serving.
//!
//! Memory a Worker keeps across requests (a module-scope array) survives a
//! termination. If the process is still over the line `ESCALATE_AFTER` after
//! the first termination, the guard sends the process SIGTERM: the ordinary
//! graceful shutdown hands its cells to peers, and the supervisor starts a
//! fresh node, as it would after a crash, but without the kernel's kill.
//!
//! Two lines, either one trips the guard:
//! - the process holds `CELLD_MEMORY_LIMIT_MB` (default 85% of the memory the
//!   node may use: above the shedding ceiling `CELLD_MAX_RSS_MB`, 80%; 0
//!   disables the whole guard);
//! - the machine has less than `CELLD_MEMORY_FLOOR_MB` available (default 15%
//!   of it, at least 64 MiB; 0 disables this line): MemAvailable, or the
//!   cgroup limit minus its working set, and only while celld holds at least
//!   half its own line. The process line alone is blind to everything else
//!   in the guest. Measured on a 4 GiB Dev cell: 4096 small
//!   resizable buffers got celld OOM-killed below its 90% line, because
//!   dockerd, the agent and the kernel hold the rest.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

/// 10 ms: a fill grows a process by 2-5 GB/s, and a real guest has a few
/// hundred MiB between the guard's lines and the kernel's OOM kill. At 25 ms
/// one Dev attack in four still won that race. Reading /proc twice per sample
/// costs about 0.25% of one core.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(10);

/// `v8::isolate::InterruptCallback`, which the crate does not re-export.
pub type InterruptCallback = unsafe extern "C" fn(v8::UnsafeRawIsolatePtr, *mut c_void);

/// How long the process may stay over the line after a termination before
/// the guard restarts it.
pub const ESCALATE_AFTER: Duration = Duration::from_secs(10);

struct Guard {
    line: u64,
    floor: Option<u64>,
    callback: InterruptCallback,
    isolates: Mutex<HashMap<u64, Entry>>,
    next_id: AtomicU64,
}

/// How long an interrupt waits, after collecting, for freed memory to leave
/// the process before it terminates anything. The isolate is paused for
/// that long only while the process is over the line.
const SETTLE_WAIT: Duration = Duration::from_millis(100);
const SETTLE_POLL: Duration = Duration::from_millis(10);

/// Re-request an interrupt this often while the process stays over the line.
const REREQUEST_INTERVAL_MS: u64 = 50;

/// After a check finishes, interrupts within this long skip the work: queued
/// requests drain in a burst. Counted from the END of a check, and a check in
/// progress blocks both new requests and other callbacks. A check can take
/// `SETTLE_WAIT`, and V8 runs queued interrupts back to back, so a window
/// counted from the start let requests pile up faster than they drained: the
/// isolate never left its interrupt loop (measured, pt.2 draft).
const CHECK_INTERVAL_MS: u64 = 20;

/// The error a memory termination fails a request with. celld collects
/// garbage when it hands such a termination back (see `js.rs`).
pub const LIMIT_ERROR: &str = "Worker exceeded its memory limit";

/// At most this many interrupts outstanding per isolate. An idle isolate
/// that keeps losing flags would otherwise queue one per re-request.
const MAX_OUTSTANDING: usize = 1024;

struct Entry {
    handle: v8::IsolateHandle,
    shared: Arc<Shared>,
}

/// What the sampler and the interrupt callback share for one isolate. The
/// callback receives a strong reference with each interrupt.
#[derive(Default)]
struct Shared {
    outstanding: AtomicUsize,
    last_request_ms: AtomicU64,
    last_done_ms: AtomicU64,
    checking: AtomicBool,
}

/// A check in progress for one isolate; dropping it ends the check.
pub struct Check {
    shared: Arc<Shared>,
}

impl Drop for Check {
    fn drop(&mut self) {
        self.shared.last_done_ms.store(now_ms().max(1), Ordering::Relaxed);
        self.shared.checking.store(false, Ordering::Release);
    }
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn now_ms() -> u64 {
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

static GUARD: OnceLock<Option<Guard>> = OnceLock::new();
static SAMPLER: Once = Once::new();
static TERMINATIONS: AtomicU64 = AtomicU64::new(0);
static ESCALATED: AtomicBool = AtomicBool::new(false);

/// Keeps one isolate under the guard; dropping it lets the isolate go.
pub struct Registration {
    id: u64,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(guard) = guard() {
            guard.isolates.lock().unwrap().remove(&self.id);
        }
    }
}

fn guard() -> Option<&'static Guard> {
    GUARD.get().and_then(Option::as_ref)
}

/// Put an isolate under the guard. `None` when the guard is disabled.
/// `callback` is the interrupt that decides and terminates; it must call
/// [`interrupt_taken`] with its `data` first.
pub fn register(handle: v8::IsolateHandle, callback: InterruptCallback) -> Option<Registration> {
    let guard = GUARD
        .get_or_init(|| {
            let line = line_from_environment()?;
            let floor = floor_from_environment();
            tracing::info!(
                limit_mib = line >> 20,
                floor_mib = floor.map(|bytes| bytes >> 20),
                "memory guard: JavaScript is terminated while the process holds more, or the machine has less available"
            );
            Some(Guard {
                line,
                floor,
                callback,
                isolates: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(0),
            })
        })
        .as_ref()?;
    SAMPLER.call_once(|| {
        std::thread::Builder::new()
            .name("celld-memory-guard".to_string())
            .spawn(run_sampler)
            .expect("spawn the memory guard");
    });
    let id = guard.next_id.fetch_add(1, Ordering::Relaxed);
    guard.isolates.lock().unwrap().insert(
        id,
        Entry {
            handle,
            shared: Arc::new(Shared::default()),
        },
    );
    Some(Registration { id })
}

/// Take back the reference the guard handed V8 with an interrupt. `Some`
/// when this callback should check the process: not while another check runs,
/// and not for the rest of a burst of queued interrupts.
///
/// # Safety
/// `data` must be the pointer the guard passed to `request_interrupt`, and
/// this must be called exactly once per interrupt.
pub unsafe fn interrupt_taken(data: *mut c_void) -> Option<Check> {
    let shared = unsafe { Arc::from_raw(data as *const Shared) };
    shared.outstanding.fetch_sub(1, Ordering::AcqRel);
    let last = shared.last_done_ms.load(Ordering::Relaxed);
    if last != 0 && now_ms().saturating_sub(last) < CHECK_INTERVAL_MS {
        return None;
    }
    if shared.checking.swap(true, Ordering::AcqRel) {
        return None;
    }
    Some(Check { shared })
}

/// Which line the node is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Over {
    /// The process holds at least `CELLD_MEMORY_LIMIT_MB`.
    Process { held: u64, line: u64 },
    /// The machine has at most `CELLD_MEMORY_FLOOR_MB` available.
    Machine { available: u64, floor: u64 },
}

impl Over {
    pub fn describe(&self) -> String {
        match *self {
            Over::Process { held, line } => {
                format!("the node holds {} MiB, the limit is {} MiB", held >> 20, line >> 20)
            }
            Over::Machine { available, floor } => format!(
                "the machine has {} MiB available, the floor is {} MiB",
                available >> 20,
                floor >> 20
            ),
        }
    }
}

/// The line the node is over, if any.
pub fn over_line() -> Option<Over> {
    let guard = guard()?;
    judge(guard.line, guard.floor, process_memory_bytes(), available_bytes())
}

/// The floor trips only while this process holds at least half its own line:
/// a guest short of memory because of something else (a Worker's containers
/// under dockerd) must not have every innocent request terminated, and
/// restarting celld would free nothing there.
fn judge(line: u64, floor: Option<u64>, held: Option<u64>, available: Option<u64>) -> Option<Over> {
    let held = held?;
    if held >= line {
        return Some(Over::Process { held, line });
    }
    match (floor, available) {
        (Some(floor), Some(available)) if available <= floor && held >= line / 2 => {
            Some(Over::Machine { available, floor })
        }
        _ => None,
    }
}

/// After the interrupt's garbage collection: `(held, line)` if the process is
/// still over the line once freed memory has had time to leave.
///
/// A collection does not lower the process's memory at once: V8 frees
/// ArrayBuffer and WebAssembly backing stores on a background sweeper thread
/// after it. Measured right after the collection, the buffers of a request
/// terminated a moment ago still count, and they failed the next request
/// (measured: the request after a terminated `rabmany` and `wasm` attack).
/// So the sweeper gets up to `SETTLE_WAIT` to bring the process under the
/// line. Only a process over the line after that is still holding the memory.
///
/// No `malloc_trim` here: tried, and in a guest at its memory limit with
/// zram it walked swapped heap on every interrupt and stalled the node.
pub fn still_over_after_collection() -> Option<Over> {
    purge_allocator();
    let deadline = Instant::now() + SETTLE_WAIT;
    loop {
        let over = over_line()?;
        if Instant::now() >= deadline {
            return Some(over);
        }
        std::thread::sleep(SETTLE_POLL);
    }
}

/// Give jemalloc's dirty pages back to the kernel now. It keeps freed pages for
/// `dirty_decay_ms` (10 s), and bytes celld hands to JavaScript (fetch bodies,
/// storage reads) live in Rust Vecs, so a terminated request's buffers sat in
/// the allocator for that long and kept the process over the line. Setting an
/// arena's decay times to 0 purges it at once (madvise on jemalloc's own
/// extent records, nothing swapped back in); the old times are put back.
fn purge_allocator() {
    let Ok(arenas) = tikv_jemalloc_ctl::arenas::narenas::read() else {
        return;
    };
    for arena in 0..arenas {
        for knob in ["dirty_decay_ms", "muzzy_decay_ms"] {
            let name = format!("arena.{arena}.{knob}\0");
            // SAFETY: an ssize_t mallctl, read and written with its own type.
            unsafe {
                if let Ok(old) = tikv_jemalloc_ctl::raw::read::<isize>(name.as_bytes()) {
                    let _ = tikv_jemalloc_ctl::raw::write(name.as_bytes(), 0_isize);
                    let _ = tikv_jemalloc_ctl::raw::write(name.as_bytes(), old);
                }
            }
        }
    }
}

/// Count one termination; the sampler escalates only after one.
pub fn record_termination() {
    TERMINATIONS.fetch_add(1, Ordering::Relaxed);
}

fn run_sampler() {
    let guard = guard().expect("the sampler starts only with a guard");
    let mut over_since: Option<Instant> = None;
    let mut terminations_at_crossing = 0;
    loop {
        std::thread::sleep(SAMPLE_INTERVAL);
        let Some(over) = over_line() else {
            over_since = None;
            continue;
        };
        let now = Instant::now();
        let since = *over_since.get_or_insert_with(|| {
            terminations_at_crossing = TERMINATIONS.load(Ordering::Relaxed);
            now
        });
        let terminated = TERMINATIONS.load(Ordering::Relaxed) > terminations_at_crossing;
        if should_escalate(now.duration_since(since), terminated)
            && !ESCALATED.swap(true, Ordering::Relaxed)
        {
            tracing::error!(
                pressure = %over.describe(),
                "memory guard: the node stayed over its memory line after terminating \
                 JavaScript, so memory is held across requests; shutting down for a restart"
            );
            // SAFETY: raising a signal at our own process; celld's handler
            // runs the ordinary graceful shutdown.
            unsafe { libc::raise(libc::SIGTERM) };
        }
        let now_ms = now_ms();
        let isolates = guard.isolates.lock().unwrap();
        for entry in isolates.values() {
            let shared = &entry.shared;
            if shared.checking.load(Ordering::Acquire) {
                continue;
            }
            if !should_request(
                now_ms,
                shared.last_request_ms.load(Ordering::Relaxed),
                shared.outstanding.load(Ordering::Acquire),
            ) {
                continue;
            }
            shared.last_request_ms.store(now_ms.max(1), Ordering::Relaxed);
            shared.outstanding.fetch_add(1, Ordering::AcqRel);
            let data = Arc::into_raw(shared.clone()) as *mut c_void;
            if !entry.handle.request_interrupt(guard.callback, data) {
                // The isolate is gone; V8 will never run the callback.
                // SAFETY: the reference `into_raw` just made.
                drop(unsafe { Arc::from_raw(data as *const Shared) });
                shared.outstanding.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

fn should_request(now_ms: u64, last_request_ms: u64, outstanding: usize) -> bool {
    outstanding < MAX_OUTSTANDING
        && (last_request_ms == 0 || now_ms.saturating_sub(last_request_ms) >= REREQUEST_INTERVAL_MS)
}

fn should_escalate(over_for: Duration, terminated: bool) -> bool {
    terminated && over_for >= ESCALATE_AFTER
}

fn line_from_environment() -> Option<u64> {
    match crate::env_vars::optional::<u64>("CELLD_MEMORY_LIMIT_MB")
        .expect("validated CELLD_MEMORY_LIMIT_MB")
    {
        Some(0) => None,
        Some(megabytes) => Some(megabytes.saturating_mul(1024 * 1024)),
        None => default_line(crate::machine::total_memory_bytes()),
    }
}

fn floor_from_environment() -> Option<u64> {
    match crate::env_vars::optional::<u64>("CELLD_MEMORY_FLOOR_MB")
        .expect("validated CELLD_MEMORY_FLOOR_MB")
    {
        Some(0) => None,
        Some(megabytes) => Some(megabytes.saturating_mul(1024 * 1024)),
        None => default_floor(crate::machine::total_memory_bytes()),
    }
}

/// What the rest of a guest holds whatever its size: dockerd, the in-VM agent,
/// the kernel. About 300 MiB on a Platinum cell. On a 2 GiB cell, 85%/15%
/// left too little of it: the guest thrashed in zram before either line.
const RESERVE_BYTES: u64 = 384 * 1024 * 1024;

/// 15% of the node's memory, and at least `RESERVE_BYTES`, but never more
/// than half.
fn default_floor(total_memory_bytes: Option<u64>) -> Option<u64> {
    total_memory_bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| (bytes / 100 * 15).max(RESERVE_BYTES).min(bytes / 2))
}

/// What the machine has left: the cgroup limit less its working set when the
/// node runs under one, otherwise the kernel's MemAvailable.
#[cfg(target_os = "linux")]
#[allow(clippy::disallowed_methods)] // /proc describes this machine.
fn available_bytes() -> Option<u64> {
    if let Some(limit) = crate::memory::cgroup_limit_bytes() {
        let working_set = crate::memory::sample().cgroup_working_set_bytes?;
        return Some(limit.saturating_sub(working_set));
    }
    parse_meminfo_available(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(not(target_os = "linux"))]
fn available_bytes() -> Option<u64> {
    None
}

fn parse_meminfo_available(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .map(|kb| kb.saturating_mul(1024))
}

/// 85% of the node's memory, and at least `RESERVE_BYTES` below all of it,
/// but never under half. 90% left too little room on a real guest (see
/// `SAMPLE_INTERVAL`), and 85% too little on a 2 GiB one.
fn default_line(total_memory_bytes: Option<u64>) -> Option<u64> {
    total_memory_bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| (bytes / 100 * 85).min(bytes.saturating_sub(RESERVE_BYTES)).max(bytes / 2))
}

/// Resident plus swapped memory of this process. Swapped pages count: in a
/// guest with zram, a process that pages out is still holding that memory.
#[cfg(target_os = "linux")]
#[allow(clippy::disallowed_methods)] // /proc describes this process.
fn process_memory_bytes() -> Option<u64> {
    parse_status(&std::fs::read_to_string("/proc/self/status").ok()?)
}

#[cfg(not(target_os = "linux"))]
fn process_memory_bytes() -> Option<u64> {
    None
}

fn parse_status(status: &str) -> Option<u64> {
    let field = |name: &str| -> Option<u64> {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    };
    let rss_kb = field("VmRSS:")?;
    let swap_kb = field("VmSwap:").unwrap_or(0);
    Some(rss_kb.saturating_add(swap_kb).saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_memory_is_resident_plus_swapped() {
        let status = "Name:\tcelld\nVmRSS:\t  1048576 kB\nRssAnon:\t 1000 kB\nVmSwap:\t   524288 kB\n";
        assert_eq!(parse_status(status), Some(1536 * 1024 * 1024));
        assert_eq!(parse_status("VmRSS:\t2048 kB\n"), Some(2048 * 1024), "no swap line");
        assert_eq!(parse_status("Name:\tcelld\n"), None);
    }

    #[test]
    fn the_default_line_is_eighty_five_percent_of_the_node_with_a_reserve() {
        const MIB: u64 = 1024 * 1024;
        assert_eq!(default_line(Some(4096 * MIB)), Some(4096 * MIB / 100 * 85), "big: 85%");
        assert_eq!(default_line(Some(1966 * MIB)), Some(1966 * MIB - 384 * MIB), "a 2 GiB cell keeps 384 MiB");
        assert_eq!(default_line(Some(512 * MIB)), Some(256 * MIB), "never under half");
        assert_eq!(default_line(None), None, "unknown memory size: no guard");
        assert_eq!(default_line(Some(0)), None);
    }

    #[test]
    fn it_escalates_only_after_a_termination_that_did_not_bring_memory_down() {
        assert!(!should_escalate(Duration::from_secs(60), false), "over the line, but nothing to blame on JS");
        assert!(!should_escalate(Duration::from_secs(9), true), "a termination gets time to free memory");
        assert!(should_escalate(ESCALATE_AFTER, true));
    }

    #[test]
    fn it_re_requests_while_over_the_line_because_a_flag_can_be_lost() {
        assert!(should_request(1000, 0, 0), "first request");
        assert!(!should_request(1030, 1000, 1), "not every sample");
        assert!(should_request(1050, 1000, 1), "again after 50 ms, even with one outstanding (its flag may be lost)");
        assert!(!should_request(9000, 1000, MAX_OUTSTANDING), "bounded per isolate");
    }

    #[test]
    fn a_burst_of_queued_interrupts_checks_once() {
        let shared = Arc::new(Shared::default());
        let hand = |shared: &Arc<Shared>| {
            shared.outstanding.fetch_add(1, Ordering::AcqRel);
            Arc::into_raw(shared.clone()) as *mut c_void
        };
        let (a, b, c) = (hand(&shared), hand(&shared), hand(&shared));
        let check = unsafe { interrupt_taken(a) };
        assert!(check.is_some(), "the first of a burst checks");
        assert!(unsafe { interrupt_taken(b) }.is_none(), "not while that check runs");
        drop(check);
        assert!(unsafe { interrupt_taken(c) }.is_none(), "nor right after it finishes");
        assert!(!shared.checking.load(Ordering::Acquire));
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
        assert_eq!(Arc::strong_count(&shared), 1, "every reference handed to V8 came back");
    }

    #[test]
    fn either_line_trips_the_guard() {
        const MIB: u64 = 1024 * 1024;
        let line = 1843 * MIB;
        let floor = Some(204 * MIB);
        assert_eq!(judge(line, floor, Some(1000 * MIB), Some(900 * MIB)), None);
        assert_eq!(
            judge(line, floor, Some(1900 * MIB), Some(900 * MIB)),
            Some(Over::Process { held: 1900 * MIB, line })
        );
        assert_eq!(
            judge(line, floor, Some(1500 * MIB), Some(150 * MIB)),
            Some(Over::Machine { available: 150 * MIB, floor: 204 * MIB }),
            "the process is under its line, but the guest is out of memory"
        );
        assert_eq!(judge(line, None, Some(1500 * MIB), Some(1)), None, "floor disabled");
        assert_eq!(
            judge(line, floor, Some(300 * MIB), Some(150 * MIB)),
            None,
            "the guest is short, but celld holds little of it (a Worker's containers): not celld's to fix"
        );
        assert_eq!(judge(line, floor, None, Some(1)), None, "nothing readable: nothing to blame");
    }

    #[test]
    fn the_default_floor_is_fifteen_percent_and_at_least_the_reserve() {
        const MIB: u64 = 1024 * 1024;
        assert_eq!(default_floor(Some(4096 * MIB)), Some(4096 * MIB / 100 * 15));
        assert_eq!(default_floor(Some(1966 * MIB)), Some(384 * MIB), "a 2 GiB cell: the reserve");
        assert_eq!(default_floor(Some(512 * MIB)), Some(256 * MIB), "never over half");
        assert_eq!(default_floor(None), None);
    }

    #[test]
    fn mem_available_is_read_from_meminfo() {
        let meminfo = "MemTotal:        4016000 kB\nMemFree:          100000 kB\nMemAvailable:     2048000 kB\n";
        assert_eq!(parse_meminfo_available(meminfo), Some(2048000 * 1024));
        assert_eq!(parse_meminfo_available("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn this_process_reports_its_memory() {
        if cfg!(target_os = "linux") {
            assert!(process_memory_bytes().is_some_and(|bytes| bytes > 0));
        }
    }
}
