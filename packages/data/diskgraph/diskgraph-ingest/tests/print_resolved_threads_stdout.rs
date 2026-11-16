// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression for the runner mistaking an informational RLIMIT_NOFILE log
//! line for part of the `--print-resolved-threads` numeric result.
//! `--print-resolved-threads` must return before tracing/rlimit setup runs,
//! so stdout can never carry anything but the number, regardless of the
//! process's starting file-descriptor limit.

use std::process::Command;

#[cfg(unix)]
#[test]
fn stdout_is_only_the_number_under_a_low_nofile_limit() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let output = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "ulimit -S -n 512 && exec {bin} --print-resolved-threads"
        ))
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "stdout: {stdout:?}");
    stdout.trim().parse::<u32>().unwrap();
    assert!(output.stderr.is_empty(), "stderr: {:?}", output.stderr);
}
