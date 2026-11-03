// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod native_cpu_guard {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../build-support/native_cpu_guard.rs"
    ));
}

fn build_commit() -> String {
    println!("cargo:rerun-if-env-changed=DISKGRAPH_BUILD_COMMIT");
    if let Ok(commit) = std::env::var("DISKGRAPH_BUILD_COMMIT") {
        return commit;
    }
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            .filter(|out| !out.is_empty())
    };
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(head) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &head]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=DISKGRAPH_BUILD_COMMIT={}", build_commit());

    if std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some() {
        native_cpu_guard::reject_native_cpu_build("diskgraph Python extension");
        return;
    }

    // Rust unit tests embed Python and need to find the same libpython that PyO3
    // linked. Python extension builds are loaded by Python, so leave them alone.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        if let Some(lib_dir) = &pyo3_build_config::get().lib_dir {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
    }
}
