mod native_cpu_guard {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../build-support/native_cpu_guard.rs"
    ));
}

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

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
