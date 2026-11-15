// Copyright 2026 Deno Land Inc. Apache-2.0 license.

//! One budget for the memory every ArrayBuffer in the process holds.
//!
//! `CELLD_V8_HEAP_LIMIT_MB` bounds each isolate's V8 heap, but ArrayBuffer
//! backing stores are not on that heap: V8 asks the embedder's allocator for
//! them, and bytes celld hands to JavaScript arrive in stores of their own.
//! Without a budget, one request could fill gigabytes of `Uint8Array`s in a
//! synchronous loop. Measured on 0.6.3: 4096 one-MiB arrays in a 2 GiB guest
//! swapped the guest until the node missed its lease renewal and fenced itself,
//! taking every cell it hosted down with it.
//!
//! With the budget, the allocation that would cross it is refused instead.
//! V8 throws `RangeError: Array buffer allocation failed`, which the script can
//! catch, and the node keeps serving. Bytes an async op resolves (a fetch body,
//! a storage read) are refused the same way: that promise rejects with a
//! `RangeError`. Bytes that arrive by other routes (RPC arguments, WebSocket
//! messages, props) are already bounded by the size limits on those routes.
//! They are charged to the budget but never refused, so they still push the
//! next JS allocation over the line.
//!
//! `CELLD_ARRAY_BUFFER_LIMIT_MB` sets the budget. The default is half of the
//! memory the node may use: its cgroup limit, otherwise physical memory.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Allocations of this size or smaller are charged but never refused.
///
/// V8 aborts the whole process when an allocation it makes on the embedder's
/// behalf fails (`v8::ArrayBuffer::new` from Rust), and celld makes small ones,
/// such as a 25-byte trace context. Refusing 64 bytes would free nothing anyway.
pub const ALWAYS_GRANTED_BYTES: usize = 64;

/// The budget when the node cannot read how much memory it has.
const FALLBACK_LIMIT_BYTES: usize = 1024 * 1024 * 1024;

/// At most one refusal warning per this many seconds; the count keeps going.
const WARNING_INTERVAL_SECS: u64 = 10;

pub struct Budget {
    used: AtomicUsize,
    limit: usize,
    refused: AtomicU64,
    /// Seconds since `epoch` at the last warning, plus one (0 = never).
    last_warning: AtomicU64,
    epoch: OnceLock<Instant>,
}

impl Budget {
    pub const fn new(limit: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            limit,
            refused: AtomicU64::new(0),
            last_warning: AtomicU64::new(0),
            epoch: OnceLock::new(),
        }
    }

    /// Charge `len` bytes, or refuse them (`false`) when that would cross the
    /// limit. Small allocations are always charged.
    pub fn reserve(&self, len: usize) -> bool {
        if len <= ALWAYS_GRANTED_BYTES {
            self.used.fetch_add(len, Ordering::Relaxed);
            return true;
        }
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            let Some(next) = used.checked_add(len).filter(|next| *next <= self.limit) else {
                self.refused.fetch_add(1, Ordering::Relaxed);
                self.warn(len, used);
                return false;
            };
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return true,
                Err(actual) => used = actual,
            }
        }
    }

    /// Charge `len` bytes whatever the limit says.
    pub fn charge(&self, len: usize) {
        self.used.fetch_add(len, Ordering::Relaxed);
    }

    pub fn release(&self, len: usize) {
        self.used.fetch_sub(len, Ordering::Relaxed);
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    fn warn(&self, requested: usize, used: usize) {
        let now = self.epoch.get_or_init(Instant::now).elapsed().as_secs() + 1;
        let last = self.last_warning.load(Ordering::Relaxed);
        if last != 0 && now < last + WARNING_INTERVAL_SECS {
            return;
        }
        if self
            .last_warning
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            tracing::warn!(
                requested_bytes = requested,
                used_bytes = used,
                limit_bytes = self.limit,
                refused_total = self.refused(),
                "ArrayBuffer allocation refused: the process holds its whole \
                 CELLD_ARRAY_BUFFER_LIMIT_MB budget"
            );
        }
    }
}

static BUDGET: OnceLock<Budget> = OnceLock::new();

/// The process budget, sized from the environment on first use.
pub fn budget() -> &'static Budget {
    BUDGET.get_or_init(|| {
        let (limit, source) = limit_from_environment();
        tracing::info!(limit_mib = limit >> 20, source, "ArrayBuffer budget for this process");
        Budget::new(limit)
    })
}

fn limit_from_environment() -> (usize, &'static str) {
    if let Some(megabytes) = crate::env_vars::positive::<usize>("CELLD_ARRAY_BUFFER_LIMIT_MB")
        .expect("validated CELLD_ARRAY_BUFFER_LIMIT_MB")
    {
        let bytes = megabytes
            .checked_mul(1024 * 1024)
            .expect("validated CELLD_ARRAY_BUFFER_LIMIT_MB range");
        return (bytes, "CELLD_ARRAY_BUFFER_LIMIT_MB");
    }
    match crate::machine::total_memory_bytes() {
        Some(total) => (default_limit(Some(total)), "half of the node's memory"),
        None => (default_limit(None), "fallback: memory size unreadable"),
    }
}

fn default_limit(total_memory_bytes: Option<u64>) -> usize {
    total_memory_bytes
        .map(|bytes| usize::try_from(bytes / 2).unwrap_or(usize::MAX))
        .filter(|bytes| *bytes > 0)
        .unwrap_or(FALLBACK_LIMIT_BYTES)
}

static VTABLE: v8::RustAllocatorVtable<Budget> = v8::RustAllocatorVtable {
    allocate,
    allocate_uninitialized,
    free,
    drop: drop_handle,
};

/// The ArrayBuffer allocator for one isolate. V8 owns and deletes each one;
/// all of them draw on the one `'static` budget.
pub fn allocator() -> v8::UniqueRef<v8::Allocator> {
    // SAFETY: the handle is the process budget, a `'static` that outlives
    // every isolate, and `drop_handle` never frees it.
    unsafe { v8::new_rust_allocator(budget() as *const Budget, &VTABLE) }
}

unsafe extern "C" fn allocate(budget: &Budget, len: usize) -> *mut c_void {
    if !budget.reserve(len) {
        return std::ptr::null_mut();
    }
    // Zeroed, as V8's default allocator hands out (`calloc`). One byte for an
    // empty buffer, so the pointer is never null for a granted request.
    let data = unsafe { libc::calloc(len.max(1), 1) };
    if data.is_null() {
        budget.release(len);
    }
    data
}

unsafe extern "C" fn allocate_uninitialized(budget: &Budget, len: usize) -> *mut c_void {
    if !budget.reserve(len) {
        return std::ptr::null_mut();
    }
    let data = unsafe { libc::malloc(len.max(1)) };
    if data.is_null() {
        budget.release(len);
    }
    data
}

unsafe extern "C" fn free(budget: &Budget, data: *mut c_void, len: usize) {
    unsafe { libc::free(data) };
    budget.release(len);
}

unsafe extern "C" fn drop_handle(_budget: *const Budget) {}

/// Charge-or-refuse policy for bytes Rust hands to JavaScript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Refuse when the budget is spent (an async op's result).
    Refusable,
    /// Charge regardless (routes whose size is bounded elsewhere).
    Charged,
}

/// Move `bytes` into a backing store charged to the budget, without copying.
/// The charge is released when V8 frees the store. `None` when the budget
/// refuses a `Refusable` store; the caller raises the `RangeError`.
pub fn backing_store_from_vec(
    bytes: Vec<u8>,
    admission: Admission,
) -> Option<v8::UniqueRef<v8::BackingStore>> {
    let budget = budget();
    let len = bytes.len();
    match admission {
        Admission::Refusable => {
            if !budget.reserve(len) {
                return None;
            }
        }
        Admission::Charged => budget.charge(len),
    }
    let mut held = std::mem::ManuallyDrop::new(bytes);
    let data = held.as_mut_ptr();
    let capacity = held.capacity();
    // The capacity rides in the deleter data so the Vec is rebuilt exactly.
    // SAFETY: `data`/`len`/`capacity` are the parts of a Vec<u8> this function
    // now owns. V8 calls `release_vec` exactly once, with the same `len`.
    Some(unsafe {
        v8::ArrayBuffer::new_backing_store_from_ptr(
            data.cast(),
            len,
            release_vec,
            capacity as *mut c_void,
        )
    })
}

unsafe extern "C" fn release_vec(data: *mut c_void, len: usize, capacity: *mut c_void) {
    // SAFETY: the parts `backing_store_from_vec` took apart.
    drop(unsafe { Vec::from_raw_parts(data.cast::<u8>(), len, capacity as usize) });
    budget().release(len);
}

/// What V8 says when it refuses an ArrayBuffer, so a refused op result reads
/// the same as a refused `new ArrayBuffer`.
pub const REFUSAL_MESSAGE: &str = "Array buffer allocation failed";

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    #[test]
    fn refuses_the_allocation_that_would_cross_the_limit() {
        let budget = Budget::new(10 * MIB);
        for _ in 0..10 {
            assert!(budget.reserve(MIB));
        }
        assert!(!budget.reserve(MIB), "the eleventh MiB crosses a 10 MiB budget");
        assert_eq!(budget.used(), 10 * MIB, "a refusal charges nothing");
        assert_eq!(budget.refused(), 1);
        budget.release(MIB);
        assert!(budget.reserve(MIB), "a freed MiB can be taken again");
    }

    #[test]
    fn one_allocation_larger_than_the_limit_is_refused() {
        let budget = Budget::new(10 * MIB);
        assert!(!budget.reserve(4096 * MIB));
        assert!(!budget.reserve(usize::MAX), "no overflow on an absurd size");
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn small_allocations_are_charged_but_never_refused() {
        let budget = Budget::new(MIB);
        assert!(budget.reserve(MIB));
        assert!(budget.reserve(ALWAYS_GRANTED_BYTES), "V8 aborts on a refused embedder allocation");
        assert!(budget.reserve(0));
        assert_eq!(budget.used(), MIB + ALWAYS_GRANTED_BYTES);
        assert!(!budget.reserve(ALWAYS_GRANTED_BYTES + 1));
    }

    #[test]
    fn charged_bytes_count_against_the_next_refusable_allocation() {
        let budget = Budget::new(10 * MIB);
        budget.charge(12 * MIB);
        assert!(!budget.reserve(MIB), "charged bytes put the budget over the line");
        budget.release(12 * MIB);
        assert!(budget.reserve(MIB));
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_limit() {
        let budget = std::sync::Arc::new(Budget::new(64 * MIB));
        let granted = std::sync::Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let budget = budget.clone();
                let granted = granted.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        if budget.reserve(MIB) {
                            granted.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(granted.load(Ordering::Relaxed), 64);
        assert_eq!(budget.used(), 64 * MIB);
    }

    #[test]
    fn the_default_is_half_the_memory_the_node_may_use() {
        assert_eq!(default_limit(Some(2048 * MIB as u64)), 1024 * MIB);
        assert_eq!(default_limit(None), FALLBACK_LIMIT_BYTES);
        assert_eq!(default_limit(Some(1)), FALLBACK_LIMIT_BYTES);
    }
}
