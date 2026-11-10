// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime initialization ownership regression.

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
    assert!(state.runtime.get().is_none());
    let (started, ready) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let initializing = std::thread::spawn(move || {
        state.coalesce_pages.get_or_init(|| {
            state.runtime.get_or_init(|| {
                started.send(()).unwrap();
                released.recv().unwrap();
                super::PageRuntime::new()
            });
            16
        });
    });
    ready.recv().unwrap();
    let passed = child_passes(|| {
        super::configured_coalesce_pages();
        super::page_runtime();
        true
    });
    release.send(()).unwrap();
    initializing.join().unwrap();
    assert!(passed, "child entered an inherited initializing OnceLock");
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "requires native Linux io_uring; injects failure after real submissions"]
fn submission_failure_retains_resources() {
    use super::{AlignedPage, PageData, PageReadSpec, PAGE_SIZE};
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use std::sync::{mpsc, Arc};

    for (pages, panic) in [
        ([0u64, 2], false),
        ([0, 1], false),
        ([0, 2], true),
        ([0, 1], true),
    ] {
        std::thread::spawn(move || {
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            let expected = vec![73; PAGE_SIZE * 3];
            tmp.write_all(&expected).unwrap();
            tmp.as_file().sync_all().unwrap();
            let file = Arc::new(tmp.reopen().unwrap());
            let weak = Arc::downgrade(&file);
            let specs = pages.map(|page_number| PageReadSpec {
                fd: file.as_raw_fd(),
                file_len: expected.len() as u64,
                page_number,
            });
            let (recycler, recycled) = mpsc::sync_channel(2);
            let recycler = Box::leak(Box::new(recycler));
            let buffers = pages
                .iter()
                .map(|_| PageData::recycled(Box::new(AlignedPage([0; PAGE_SIZE])), recycler))
                .collect();
            super::FAIL_AFTER_SUBMIT.with(|fail| fail.set(true));
            super::PANIC_AFTER_SUBMIT.with(|fail| fail.set(panic));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::read_specs_local_with_buffers(&specs, 2, buffers, vec![Arc::clone(&file)])
            }));
            if panic {
                assert!(result.is_err(), "injected panic must unwind");
            } else {
                let error = result
                    .unwrap()
                    .err()
                    .expect("injected failure must be returned");
                assert!(error.to_string().contains("injected terminal failure"));
                assert!(error.to_string().contains("owner retired"));
            }
            let ring_fd = super::SUBMITTED_RING_FD.with(std::cell::Cell::get);
            assert!(
                unsafe { libc::fcntl(ring_fd, libc::F_GETFD) } >= 0,
                "pending ring was closed before CQEs were drained"
            );
            drop(file);
            assert_eq!(weak.strong_count(), 1, "kernel file owner was released");
            assert!(
                matches!(recycled.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "in-flight buffers returned to the recycle pool"
            );
            super::RING.with(|ring| assert!(ring.borrow().is_none(), "failed ring was cached"));

            // A retired owner must not submit or quarantine another batch.
            super::FAIL_AFTER_SUBMIT.with(|fail| fail.set(true));
            let error = super::read_specs_local_with_buffers(
                &specs,
                2,
                Vec::new(),
                vec![weak.upgrade().unwrap()],
            )
            .err()
            .expect("retired owner must reject later work");
            assert!(error.to_string().contains("owner retired"));
            assert!(
                super::FAIL_AFTER_SUBMIT.with(|fail| fail.replace(false)),
                "retired owner attempted another submission"
            );
            assert_eq!(
                weak.strong_count(),
                1,
                "later work retained another file owner"
            );

            // A separate owner has a fresh ring and sees only its own CQEs.
            let fresh = Arc::new(tmp.reopen().unwrap());
            std::thread::spawn(move || {
                let read =
                    super::read_pages_local(fresh, expected.len() as u64, &pages, 2).unwrap();
                for page in read {
                    assert_eq!(page.valid(), &[73; PAGE_SIZE]);
                }
            })
            .join()
            .unwrap();
        })
        .join()
        .unwrap();
    }
}
