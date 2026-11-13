// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks. Run on Linux with io_uring permitted; no EPERM skip.
//! Forked workers submit new reads without inheriting parent runtime state.

#![cfg(target_os = "linux")]

use diskgraph_index::page_io::{
    page_read_request, read_prepared_pages, stats, submit_prepared_page_batch, EdgeIoBackend,
    PAGE_SIZE,
};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn fixture() -> (tempfile::NamedTempFile, Vec<u8>, Arc<File>) {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    let expected = (0..(PAGE_SIZE * 70 + 1137))
        .map(|i| ((i * 7 + i / PAGE_SIZE) % 251) as u8)
        .collect::<Vec<_>>();
    tmp.write_all(&expected).unwrap();
    tmp.as_file().sync_all().unwrap();
    let file = Arc::new(tmp.reopen().unwrap());
    (tmp, expected, file)
}

#[test]
#[ignore = "requires native Linux io_uring"]
fn native_roundtrip() {
    let (_tmp, expected, file) = fixture();
    let before = stats();
    for iteration in 0..8 {
        let pages = (0..71)
            .filter(|page| page % 3 != iteration % 3)
            .collect::<Vec<_>>();
        let prepared = read_prepared_pages(
            &file,
            expected.len() as u64,
            EdgeIoBackend::Buffered,
            &pages,
        )
        .unwrap();
        for page in pages {
            let start = page * PAGE_SIZE as u64;
            let end = ((page + 1) * PAGE_SIZE as u64).min(expected.len() as u64);
            assert_eq!(
                prepared.copy_range(start, end).unwrap(),
                expected[start as usize..end as usize]
            );
        }
    }
    let after = stats();
    assert!(
        after.submitted > before.submitted,
        "native reads must execute"
    );
    assert!(after.multi_page_ops > before.multi_page_ops);
}

#[test]
#[ignore = "requires native Linux io_uring"]
fn canceled_batches_allow_new_reads() {
    let (_tmp, expected, file) = fixture();
    std::thread::scope(|scope| {
        for worker in 0..12u64 {
            let file = Arc::clone(&file);
            let expected = &expected;
            scope.spawn(move || {
                for iteration in 0..32u64 {
                    let page = (worker + iteration) % 70;
                    let request = page_read_request(
                        &file,
                        expected.len() as u64,
                        EdgeIoBackend::Buffered,
                        &[page, page + 1],
                    )
                    .unwrap();
                    drop(submit_prepared_page_batch(vec![request]).unwrap());
                    let prepared = read_prepared_pages(
                        &file,
                        expected.len() as u64,
                        EdgeIoBackend::Buffered,
                        &[page],
                    )
                    .unwrap();
                    let offset = page * PAGE_SIZE as u64;
                    assert_eq!(
                        prepared.copy_range(offset, offset + 17).unwrap(),
                        expected[offset as usize..offset as usize + 17]
                    );
                }
            });
        }
    });
}

fn fork_child(action: impl FnOnce() -> bool, timeout: Duration) -> bool {
    // Each diagnostic is selected alone in its own test process. The child
    // uses _exit to avoid running inherited Rust test/allocator destructors.
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        let code = if action() { 0 } else { 2 };
        unsafe { libc::_exit(code) };
    }
    let deadline = Instant::now() + timeout;
    let mut status = 0;
    loop {
        let waited = unsafe { libc::waitpid(child, &mut status, libc::WNOHANG) };
        assert!(waited >= 0);
        if waited == child {
            if libc::WIFSIGNALED(status) {
                eprintln!("child terminated by signal {}", libc::WTERMSIG(status));
            }
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

fn fork_read(file: &Arc<File>, len: u64) -> bool {
    fork_child(
        || {
            read_prepared_pages(file, len, EdgeIoBackend::Buffered, &[0])
                .and_then(|pages| pages.copy_range(0, 1))
                .is_ok_and(|bytes| bytes == [0])
        },
        Duration::from_secs(3),
    )
}

#[test]
#[ignore = "run alone: native io_uring process lifecycle diagnostic"]
fn fork_before_first_read() {
    let (_tmp, expected, file) = fixture();
    assert!(fork_read(&file, expected.len() as u64));
}

#[test]
#[ignore = "run alone: requires native Linux io_uring"]
fn fork_after_first_read() {
    let (_tmp, expected, file) = fixture();
    read_prepared_pages(&file, expected.len() as u64, EdgeIoBackend::Buffered, &[0]).unwrap();
    assert!(
        fork_read(&file, expected.len() as u64),
        "child read timed out or failed after inheriting initialized page runtime"
    );
}

#[test]
#[ignore = "requires native Linux io_uring"]
fn read_error_allows_new_reads() {
    let (tmp, expected, file) = fixture();
    let write_only = Arc::new(OpenOptions::new().write(true).open(tmp.path()).unwrap());
    for _ in 0..12 {
        assert!(read_prepared_pages(
            &write_only,
            expected.len() as u64,
            EdgeIoBackend::Buffered,
            &[0, 70]
        )
        .is_err());
        let prepared = read_prepared_pages(
            &file,
            expected.len() as u64,
            EdgeIoBackend::Buffered,
            &[0, 70],
        )
        .unwrap();
        assert_eq!(prepared.copy_range(0, 71).unwrap(), expected[..71]);
        assert_eq!(
            prepared
                .copy_range((PAGE_SIZE * 70) as u64, expected.len() as u64)
                .unwrap(),
            expected[PAGE_SIZE * 70..]
        );
    }
}

#[test]
#[ignore = "run alone: requires native Linux io_uring"]
fn fork_handles_inherited_work() {
    let (_tmp, expected, file) = fixture();
    let prepared =
        read_prepared_pages(&file, expected.len() as u64, EdgeIoBackend::Buffered, &[0]).unwrap();
    let make_pending = || {
        let request = page_read_request(
            &file,
            expected.len() as u64,
            EdgeIoBackend::Buffered,
            &[0, 1],
        )
        .unwrap();
        submit_prepared_page_batch(vec![request]).unwrap()
    };
    let mut pending_finish = Some(make_pending());
    let mut pending_drop = Some(make_pending());
    assert!(fork_child(
        || {
            drop(pending_drop.take());
            if pending_finish
                .take()
                .unwrap()
                .finish()
                .err()
                .map_or(true, |error| {
                    error.kind() != std::io::ErrorKind::InvalidInput
                })
            {
                return false;
            }
            // Completed Buffered frames are ordinary process-owned allocations;
            // unlike in-flight responses, their bytes remain readable after fork.
            if prepared.copy_range(0, 71).ok().as_deref() != Some(&expected[..71]) {
                return false;
            }
            read_prepared_pages(&file, expected.len() as u64, EdgeIoBackend::Buffered, &[0])
                .and_then(|pages| pages.copy_range(0, 71))
                .is_ok_and(|bytes| bytes == expected[..71])
        },
        Duration::from_secs(5)
    ));
    assert!(pending_finish.take().unwrap().finish().is_ok());
    assert!(pending_drop.take().unwrap().finish().is_ok());
    assert_eq!(prepared.copy_range(0, 71).unwrap(), expected[..71]);
}
