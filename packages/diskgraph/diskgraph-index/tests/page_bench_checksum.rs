// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::Command;

fn checksum(file: &std::path::Path, backend: &str) -> u64 {
    let result = Command::new(env!("CARGO_BIN_EXE_dg-page-io-bench"))
        .args(["--file", file.to_str().unwrap(), "--backend", backend])
        .args(["--pages", "128", "--iterations", "4", "--threads", "1"])
        .args(["--rings", "1", "--qd", "4"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "backend={backend}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout)
        .unwrap()
        .split_whitespace()
        .find_map(|field| field.strip_prefix("checksum="))
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
#[ignore = "requires native Linux io_uring"]
fn backends_checksum_identical_bytes() {
    let mut fixture = tempfile::NamedTempFile::new().unwrap();
    for page in 1..=8u8 {
        fixture.write_all(&vec![page; 4096]).unwrap();
    }
    fixture.as_file().sync_all().unwrap();
    let reference = checksum(fixture.path(), "mmap");
    assert!(
        reference > 512,
        "checksum must depend on actual byte values"
    );
    assert_eq!(checksum(fixture.path(), "buffered"), reference);
}
