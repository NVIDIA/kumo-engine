// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reservation ownership regressions.

use super::{DirectCache, PageKey, CACHE_LOADING_BIT};
use std::sync::atomic::Ordering;

#[test]
fn failed_plan_preserves_replacement() {
    let cache = Box::leak(Box::new(DirectCache::new(1, 1).unwrap()));
    let mut failed = cache.plan_read(101, &[0]);
    let slot = failed.reads[0].slot.unwrap();
    failed.fail("injected read failure");
    failed.fail("repeated failure");
    let replacement = cache.plan_read(101, &[0]);
    assert_eq!(replacement.reads[0].slot, Some(slot));
    drop(failed);
    assert_eq!(
        cache.tags[slot].load(Ordering::Acquire),
        PageKey {
            file_id: 101,
            page_number: 0,
        }
        .packed()
            | CACHE_LOADING_BIT,
        "dropping the failed plan canceled a new owner's reservation"
    );
    drop(replacement);
}

#[test]
fn failed_read_releases_reservation() {
    let cache = Box::leak(Box::new(DirectCache::new(1, 1).unwrap()));
    let failed = cache.plan_read(102, &[0]);
    let slot = failed.reads[0].slot.unwrap();
    assert!(failed.finish(Vec::new()).is_err());
    let replacement = cache.plan_read(102, &[0]);
    assert_eq!(replacement.reads[0].slot, Some(slot));
    drop(replacement);
}

#[cfg(target_os = "linux")]
fn child_passes(action: impl FnOnce() -> bool) -> bool {
    use std::time::{Duration, Instant};
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        unsafe { libc::_exit(if action() { 0 } else { 1 }) };
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut status = 0;
    loop {
        let waited = unsafe { libc::waitpid(child, &mut status, libc::WNOHANG) };
        assert!(waited >= 0);
        if waited == child {
            return libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
        }
        if Instant::now() >= deadline {
            unsafe {
                libc::kill(child, libc::SIGKILL);
                libc::waitpid(child, &mut status, 0);
            }
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "run alone: process initialization fork regression"]
fn fork_during_initialization() {
    let state = super::process_state();
    assert!(state.cache_gib.get().is_none());
    let (started, ready) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let initializing = std::thread::spawn(move || {
        state.cache_gib.get_or_init(|| {
            state.coalesce_pages.get_or_init(|| {
                state.runtime.get_or_init(|| {
                    state.cache.get_or_init(|| {
                        started.send(()).unwrap();
                        released.recv().unwrap();
                        DirectCache::new(1, 1).map_err(|error| error.to_string())
                    });
                    super::PageRuntime::new()
                });
                16
            });
            0
        });
    });
    ready.recv().unwrap();
    let passed = child_passes(|| {
        super::configured_direct_cache_gib();
        super::configured_coalesce_pages();
        super::page_runtime();
        super::direct_cache().is_ok()
    });
    release.send(()).unwrap();
    initializing.join().unwrap();
    assert!(passed, "child entered an inherited initializing OnceLock");
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "run alone: locked cache fork regression"]
fn fork_with_locked_cache() {
    let state = super::process_state();
    let cache = state.cache.get_or_init(|| {
        DirectCache::new(1, 1).map_err(|error| error.to_string())
    }).as_ref().unwrap();
    let held = cache.miss_locks[0].0.lock().unwrap();
    let passed = child_passes(|| {
        let cache = match super::direct_cache() {
            Ok(cache) => cache,
            Err(_) => return false,
        };
        let plan = cache.plan_read(103, &[0]);
        plan.reads[0].slot.is_some()
    });
    drop(held);
    assert!(passed, "child entered an inherited cache miss lock");
}
