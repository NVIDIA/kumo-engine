// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks. Run on Linux with io_uring permitted; no EPERM skip.
//! The fork regression is a diagnostic for a known PR424 readiness blocker.

use diskgraph_index::page_io::{
    page_read_request, read_prepared_pages, stats, submit_prepared_page_batch, EdgeIoBackend,
    PAGE_SIZE,
};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn fixture() -> (tempfile::NamedTempFile, Vec<u8>, Arc<File>, Arc<File>) {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    let expected = (0..(PAGE_SIZE * 70 + 1137))
        .map(|i| ((i * 7 + i / PAGE_SIZE) % 251) as u8)
        .collect::<Vec<_>>();
    tmp.write_all(&expected).unwrap();
    tmp.as_file().sync_all().unwrap();
    let file = Arc::new(tmp.reopen().unwrap());
    let direct = Arc::new(
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
            .open(tmp.path())
            .unwrap(),
    );
    (tmp, expected, file, direct)
}

#[test]
#[ignore = "requires native Linux io_uring and O_DIRECT; set DG_DIRECT_CACHE_GB=1 for cache coverage"]
fn native_roundtrip() {
    let (_tmp, expected, file, direct) = fixture();
    let before = stats();
    for backend in [EdgeIoBackend::Buffered, EdgeIoBackend::Direct] {
        for iteration in 0..8 {
            let pages = (0..71)
                .filter(|page| page % 3 != iteration % 3)
                .collect::<Vec<_>>();
            let prepared = read_prepared_pages(
                &file,
                Some(&direct),
                1001,
                expected.len() as u64,
                backend,
                &pages,
            )
            .unwrap();
            for page in pages {
                let start = page * PAGE_SIZE as u64;
                let end = ((page + 1) * PAGE_SIZE as u64).min(expected.len() as u64);
                assert_eq!(
                    prepared.copy_range(start, end).unwrap(),
                    expected[start as usize..end as usize],
                    "backend={backend:?}, page={page}, iteration={iteration}"
                );
            }
        }
    }
    let after = stats();
    assert!(
        after.submitted > before.submitted,
        "native reads must execute"
    );
    assert!(after.multi_page_ops > before.multi_page_ops);
    if std::env::var("DG_DIRECT_CACHE_GB").as_deref() == Ok("1") {
        assert!(after.cache_hits > before.cache_hits);
    }
}

#[test]
#[ignore = "requires native Linux io_uring and O_DIRECT"]
fn canceled_batches_release_reservations() {
    let (_tmp, expected, file, direct) = fixture();
    std::thread::scope(|scope| {
        for worker in 0..12u64 {
            let file = Arc::clone(&file);
            let direct = Arc::clone(&direct);
            let expected = &expected;
            scope.spawn(move || {
                for iteration in 0..32u64 {
                    let page = (worker + iteration) % 70;
                    let request = page_read_request(
                        &file,
                        Some(&direct),
                        1002,
                        expected.len() as u64,
                        EdgeIoBackend::Direct,
                        &[page, page + 1],
                    )
                    .unwrap();
                    drop(submit_prepared_page_batch(vec![request]).unwrap());
                    let prepared = read_prepared_pages(
                        &file,
                        Some(&direct),
                        1002,
                        expected.len() as u64,
                        EdgeIoBackend::Direct,
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

fn fork_read(
    file: &Arc<File>,
    direct: Option<&Arc<File>>,
    backend: EdgeIoBackend,
    len: u64,
    timeout: Duration,
) -> bool {
    // Each diagnostic is selected alone in its own test process. The child
    // uses _exit to avoid running inherited Rust test/allocator destructors.
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        let result = read_prepared_pages(file, direct, 1003, len, backend, &[0]);
        let code = match result {
            Ok(pages) if pages.copy_range(0, 1).unwrap() == [0] => 0,
            _ => 2,
        };
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

#[test]
#[ignore = "run alone: native io_uring process lifecycle diagnostic"]
fn fork_before_first_read() {
    let (_tmp, expected, file, _direct) = fixture();
    assert!(fork_read(
        &file,
        None,
        EdgeIoBackend::Buffered,
        expected.len() as u64,
        Duration::from_secs(3)
    ));
}

#[test]
#[ignore = "run alone: known PR424 blocker, inherited owner threads do not survive fork"]
fn fork_after_first_read() {
    let (_tmp, expected, file, _direct) = fixture();
    read_prepared_pages(
        &file,
        None,
        1003,
        expected.len() as u64,
        EdgeIoBackend::Buffered,
        &[0],
    )
    .unwrap();
    assert!(
        fork_read(
            &file,
            None,
            EdgeIoBackend::Buffered,
            expected.len() as u64,
            Duration::from_secs(3)
        ),
        "child read timed out or failed after inheriting initialized page runtime"
    );
}

#[test]
#[ignore = "run alone with DG_DIRECT_CACHE_GB=1: inherited direct-cache arena is MADV_DONTFORK"]
fn fork_with_populated_cache() {
    assert_eq!(std::env::var("DG_DIRECT_CACHE_GB").as_deref(), Ok("1"));
    let (_tmp, expected, file, direct) = fixture();
    read_prepared_pages(
        &file,
        Some(&direct),
        1003,
        expected.len() as u64,
        EdgeIoBackend::Direct,
        &[0],
    )
    .unwrap();
    assert!(
        fork_read(
            &file,
            Some(&direct),
            EdgeIoBackend::Direct,
            expected.len() as u64,
            Duration::from_secs(3)
        ),
        "child cannot read an inherited cache hit"
    );
}
