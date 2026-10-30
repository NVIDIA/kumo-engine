// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Batched asynchronous readahead for the mmap'd index.
//!
//! The sampler's read path is synchronous page faults: a thread touches an
//! uncached index page, parks in D-state until that single 4 KB read returns,
//! computes for a moment, then faults on the next page. Every thread therefore
//! holds exactly ONE read in flight, so the devices sit at queue depth ~0 and
//! deliver ~38% of their measured ceiling no matter how many threads are added
//! (`docs/explorations/pinterest_omnisage/mmap_iops_ceiling_and_io_uring_proposal.md`).
//!
//! `willneed` breaks that coupling without inverting the traversal: the caller
//! hands it every range a whole frontier is about to touch, and it asks the
//! kernel to start those reads *now*. The reads land while the sampler is still
//! working through the head of the frontier, so the queue actually fills.
//!
//! This is a pure hint — it changes only *when* bytes are paged in, never which
//! bytes are read — so sampler output stays bit-identical (verified by
//! `tests_prefetch.rs`).
//!
//! Off unless `DG_PREFETCH=1`.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

/// Max ranges per `process_madvise` call (kernel `UIO_MAXIOV`).
const MAX_IOV: usize = 1024;

/// Calls between residency probes.
///
/// This has to be large, and the reason is not the probe's own cost. In the
/// cache-resident regime the sampler is taking ~8.6M minor faults/s through
/// `filemap_map_pages`, and every `madvise`/`mincore` call contends with that
/// fault path on `mmap_lock` — a prefetch pass stalls the faulting threads
/// rather than burning CPU (measured: the prefetch run used LESS total CPU,
/// 21.1 s vs 23.0 s, for MORE wall time). So the cost of a probe is set by how
/// many threads it blocks, not by how much work it does, and the only lever is
/// to make probes rare.
///
/// Overridable via `DG_PREFETCH_PROBE_EVERY` for calibration.
const PROBE_EVERY_DEFAULT: u64 = 8192;
const PROBE_SAMPLES: usize = 8;

fn probe_every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("DG_PREFETCH_PROBE_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(PROBE_EVERY_DEFAULT)
    })
}

/// Hit-rate band, in percent, for switching prefetch off and back on.
///
/// A single probe of 8 pages is a noisy estimate — on a 191 GB index in a
/// 174 GiB cache, ~9% of pages are genuinely absent, so an all-8-must-hit rule
/// leaves the gate active on ~half of probes and the resident regression barely
/// improves. Smooth the estimate and separate the thresholds so the decision
/// stops flapping around the boundary.
const HIT_OFF: u32 = 75; // ≥ this smoothed hit rate → stop prefetching
const HIT_ON: u32 = 60; // < this → resume
/// EWMA weight of the newest probe, as a reciprocal (1/4).
const EWMA_SHIFT: u32 = 2;

/// Coalesce two ranges separated by at most this many bytes. Random adjacency
/// gaps below a page cost nothing to bridge (the page is read either way) and
/// bridging slightly larger gaps trades a little read amplification for fewer
/// iovec entries, which is the right trade when the device is the bottleneck.
const COALESCE_GAP: usize = 4096;

/// 0 = not yet read from the environment, 1 = on, 2 = off.
static ENABLED: AtomicU8 = AtomicU8::new(0);

pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("DG_PREFETCH").is_ok_and(|v| v == "1" || v == "true");
            ENABLED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Override the `DG_PREFETCH` setting for the rest of the process.
///
/// Exists so an A/B benchmark can switch modes between runs in ONE process —
/// otherwise the comparison is confounded by each process starting with a cold
/// page cache — and so the determinism test can sample both ways in-process.
pub fn set_enabled(on: bool) {
    ENABLED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

// ---------------------------------------------------------------- the gate
//
// Keeps prefetch out of the way when the pages are already cached.
//
// `MADV_WILLNEED` is NOT free on a hit — the kernel still walks every page of
// every range, and the prefetch pass also repeats the offset decode. In the
// cache-resident regime the sampler's budget is ~115 ns per node, and that
// overhead measured a **4.3× slowdown** on a fully-resident 191 GB index
// (8.61M → 2.02M nodes/s). Prefetch is only ever worth its cost on a miss, so
// sample residency with `mincore` and stand down while the hit rate is high.
//
// Two design points, both learned the hard way:
//
// - The gate is process-wide, not per-shard, and the CALLER checks it before
//   doing any grouping work. Grouping a frontier by shard — an allocation per
//   shard plus a binary search per id — costs more than the gate saves, so
//   gating only inside `prefetch_nodes` left a 30% resident regression.
// - The decision is smoothed and hysteretic. A single 8-page probe is noisy;
//   an all-must-hit rule left the gate active on ~half of probes.
//
// Adaptive in both directions: re-probed every `PROBE_EVERY` calls, so a
// workload that grows past RAM turns prefetch back on by itself.

// The call counter is THREAD-LOCAL, and nothing on the skip path touches a
// shared cache line. That is not micro-optimization: with a shared counter and
// a shared skip tally, two contended atomic RMWs per batch from 64 threads cost
// ~4.6 µs per call and held the cache-resident regression at 1.38x even once
// the gate was skipping 97% of passes. At a ~115 ns/node budget the bookkeeping
// WAS the regression. Partitioning the counter per thread leaves the overall
// probe rate unchanged — calls are spread across threads either way.
thread_local! {
    static CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Start active: the first call probes, and being briefly wrong on a resident
/// workload costs one batch.
static ACTIVE: AtomicBool = AtomicBool::new(true);
/// Smoothed page-cache hit rate, in percent. Starts at 0 (assume misses).
static HIT_RATE: AtomicU32 = AtomicU32::new(0);
/// Diagnostics. Only bumped when a pass actually runs — the common path when
/// the gate is off must stay off any shared line.
static RAN: AtomicU64 = AtomicU64::new(0);

/// `(passes_run, smoothed_hit_rate_pct)` — for benchmarks to confirm the gate
/// is doing what it claims rather than inferring it from throughput.
pub fn stats() -> (u64, u32) {
    (RAN.load(Ordering::Relaxed), HIT_RATE.load(Ordering::Relaxed))
}

/// How many ids a probe looks at while the gate is off.
///
/// A probe must not cost a full prefetch pass. One pass over a ~5,500-node
/// frontier runs ~190 µs against ~12 µs of useful work per call, so probing
/// every 64th call at full width put back a 41% resident regression even
/// though 97% of passes were being skipped. Residency only needs a sample.
pub const PROBE_IDS: usize = 64;

/// Is the gate currently letting prefetch through? Callers use this only to
/// decide how much work a *probe* should do; use [`admit`] for the decision.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Whether to do prefetch work at all, and whether this call is a probe (which
/// must sample residency before advising anything). Returns `(do_work, is_probe)`.
///
/// Call this **once per batch, before building anything**, and pass `is_probe`
/// down to [`crate::reader::ShardReader::prefetch_nodes`].
pub fn admit() -> (bool, bool) {
    if !enabled() {
        return (false, false);
    }
    let probe = CALLS.with(|c| {
        let n = c.get();
        c.set(n + 1);
        n % probe_every() == 0
    });
    let run = probe || ACTIVE.load(Ordering::Relaxed);
    if run {
        RAN.fetch_add(1, Ordering::Relaxed);
    }
    (run, probe)
}

/// Record what a probe found: stay on only while the pages are missing.
/// Call BEFORE advising the ranges, or the advice makes them look resident.
pub fn observe(map: &[u8], ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let step = (ranges.len() / PROBE_SAMPLES).max(1);
    let mut sampled = 0u32;
    let mut resident = 0u32;
    for &(start, _) in ranges.iter().step_by(step).take(PROBE_SAMPLES) {
        match page_resident(map, start) {
            None => return, // probe unavailable — leave the decision alone
            Some(true) => resident += 1,
            Some(false) => {}
        }
        sampled += 1;
    }
    if sampled == 0 {
        return;
    }

    let sample_pct = 100 * resident / sampled;
    let prev = HIT_RATE.load(Ordering::Relaxed);
    let smoothed = prev + (sample_pct >> EWMA_SHIFT) - (prev >> EWMA_SHIFT);
    HIT_RATE.store(smoothed, Ordering::Relaxed);

    // Hysteresis: only the crossings move the switch, so a hit rate sitting
    // near a threshold does not toggle prefetch on and off every probe.
    if smoothed >= HIT_OFF {
        ACTIVE.store(false, Ordering::Relaxed);
    } else if smoothed < HIT_ON {
        ACTIVE.store(true, Ordering::Relaxed);
    }
}

/// Is the page containing byte `offset` of `map` in the page cache?
/// `None` when `mincore` is unavailable.
fn page_resident(map: &[u8], offset: usize) -> Option<bool> {
    if offset >= map.len() {
        return None;
    }
    let page = page_size();
    let addr = unsafe { map.as_ptr().add(offset & !(page - 1)) };
    let mut vec = [0u8; 1];
    #[cfg(target_os = "macos")]
    let vp = vec.as_mut_ptr() as *mut libc::c_char;
    #[cfg(not(target_os = "macos"))]
    let vp = vec.as_mut_ptr();
    let ret = unsafe { libc::mincore(addr as *mut libc::c_void, page, vp) };
    if ret != 0 {
        return None;
    }
    Some(vec[0] & 1 == 1)
}

/// Ask the kernel to begin reading every page covered by `ranges` (byte offsets
/// into the mapping `map`) into the page cache, without blocking.
///
/// `ranges` is sorted and coalesced in place, so callers can reuse the buffer.
/// Errors are silently ignored: a failed hint costs a fault later, nothing more.
pub fn willneed(map: &[u8], ranges: &mut Vec<(usize, usize)>) {
    if ranges.is_empty() || map.is_empty() {
        return;
    }
    let page = page_size();
    let map_len = map.len();

    // Page-align outward, drop empties, clamp to the mapping.
    for r in ranges.iter_mut() {
        let start = r.0.min(map_len) & !(page - 1);
        let end = (r.1.min(map_len) + page - 1) & !(page - 1);
        *r = (start, end.min(map_len).max(start));
    }
    ranges.retain(|&(s, e)| e > s);
    if ranges.is_empty() {
        return;
    }

    // Sort + coalesce so a hub node's neighbours (and repeat visits to the same
    // node within a frontier) collapse into one iovec entry instead of many.
    ranges.sort_unstable();
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let (s, e) = ranges[read];
        let prev_end = ranges[write].1;
        if s <= prev_end.saturating_add(COALESCE_GAP) {
            ranges[write].1 = prev_end.max(e);
        } else {
            write += 1;
            ranges[write] = (s, e);
        }
    }
    ranges.truncate(write + 1);

    let base = map.as_ptr();
    for batch in ranges.chunks(MAX_IOV) {
        advise_batch(base, batch);
    }
}

/// One `MADV_WILLNEED` for a batch of ranges.
///
/// `process_madvise` advises many ranges per syscall, which matters here: at
/// the plateau the fault path is already burning ~44% system time, so a
/// per-range syscall would hand back much of what the deeper queue wins. Where
/// it is unavailable (pre-5.14 kernels, restricted seccomp, non-Linux) we fall
/// back to per-range `madvise` — still asynchronous, just not batched.
#[cfg(target_os = "linux")]
fn advise_batch(base: *const u8, ranges: &[(usize, usize)]) {
    // 0 = untried, 1 = process_madvise works, 2 = fall back to madvise.
    static MODE: AtomicU8 = AtomicU8::new(0);

    if MODE.load(Ordering::Relaxed) != 2 {
        if let Some(pidfd) = self_pidfd() {
            let iov: Vec<libc::iovec> = ranges
                .iter()
                .map(|&(s, e)| libc::iovec {
                    iov_base: unsafe { base.add(s) } as *mut libc::c_void,
                    iov_len: e - s,
                })
                .collect();
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_process_madvise,
                    pidfd,
                    iov.as_ptr(),
                    iov.len(),
                    libc::MADV_WILLNEED,
                    0u32,
                )
            };
            if ret >= 0 {
                MODE.store(1, Ordering::Relaxed);
                return;
            }
            // A short count is still progress; only a hard error demotes us.
            MODE.store(2, Ordering::Relaxed);
        } else {
            MODE.store(2, Ordering::Relaxed);
        }
    }
    advise_each(base, ranges);
}

#[cfg(not(target_os = "linux"))]
fn advise_batch(base: *const u8, ranges: &[(usize, usize)]) {
    advise_each(base, ranges);
}

fn advise_each(base: *const u8, ranges: &[(usize, usize)]) {
    for &(s, e) in ranges {
        unsafe {
            libc::madvise(
                base.add(s) as *mut libc::c_void,
                e - s,
                libc::MADV_WILLNEED,
            );
        }
    }
}

/// A pidfd referring to this process, opened once. `process_madvise` needs one
/// even for self.
#[cfg(target_os = "linux")]
fn self_pidfd() -> Option<libc::c_int> {
    static PIDFD: OnceLock<Option<i32>> = OnceLock::new();
    *PIDFD.get_or_init(|| {
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0u32) };
        if fd < 0 {
            None
        } else {
            Some(fd as i32)
        }
    })
}

fn page_size() -> usize {
    static PAGE: OnceLock<usize> = OnceLock::new();
    *PAGE.get_or_init(|| {
        let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if sz > 0 {
            sz as usize
        } else {
            4096
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges are page-aligned outward, sorted, and merged across sub-page
    /// gaps — one iovec per contiguous run rather than one per node.
    #[test]
    fn coalesces_and_aligns() {
        let page = page_size();
        let map = vec![0u8; page * 16];
        // Out of order; the first two land in the same page, the third is far.
        let mut ranges = vec![(page * 10, page * 10 + 8), (100, 200), (300, 400)];
        willneed(&map, &mut ranges);
        assert_eq!(ranges, vec![(0, page), (page * 10, page * 11)]);
    }

    /// A range past the end of the mapping must be clamped, never advised
    /// out of bounds.
    #[test]
    fn clamps_to_mapping() {
        let page = page_size();
        let map = vec![0u8; page * 2];
        let mut ranges = vec![(page, page * 100)];
        willneed(&map, &mut ranges);
        assert_eq!(ranges, vec![(page, page * 2)]);
    }

    #[test]
    fn empty_inputs_are_noops() {
        let mut ranges: Vec<(usize, usize)> = vec![];
        willneed(&[0u8; 4096], &mut ranges);
        assert!(ranges.is_empty());

        let mut ranges = vec![(0, 4096)];
        willneed(&[], &mut ranges);
    }
}
