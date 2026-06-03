mod native_cpu_guard {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../build-support/native_cpu_guard.rs"
    ));
}

fn main() {
    native_cpu_guard::reject_native_cpu_build("diskgraph-ingest");
}
