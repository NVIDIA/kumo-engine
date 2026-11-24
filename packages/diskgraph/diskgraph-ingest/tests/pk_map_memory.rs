// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{Field, Schema};
use diskgraph_ingest::progress::{ProgressEvent, ProgressKind, ProgressPhase};
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use serde_json::json;
use tracing::field::{Field as EventField, Visit};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

#[global_allocator]
static HEAP: CountingAllocator = CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

const OWN_KEYS: i64 = 200_000;

struct ReservationLayer(Arc<AtomicU64>);

struct ReservationVisitor<'a>(&'a AtomicU64);

impl Visit for ReservationVisitor<'_> {
    fn record_u64(&mut self, field: &EventField, value: u64) {
        if field.name() == "estimate_sum_bytes" {
            self.0.store(value, Ordering::Relaxed);
        }
    }

    fn record_debug(&mut self, _field: &EventField, _value: &dyn std::fmt::Debug) {}
}

impl<S: tracing::Subscriber> Layer<S> for ReservationLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        event.record(&mut ReservationVisitor(&self.0));
    }
}

struct PkMapRun {
    reserved: u64,
    peak: u64,
}

fn write_column(path: &Path, values: ArrayRef) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "user_id",
        values.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).unwrap();
    let mut writer = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn run_pk_map(root: &Path, dtype: &str, users: ArrayRef, references: Option<ArrayRef>) -> PkMapRun {
    fs::create_dir_all(root).unwrap();
    let users_path = root.join("users.parquet");
    write_column(&users_path, users);
    let mut tables = vec![json!({
        "name": "users",
        "columns": [{"name": "user_id", "ingested_dtype": dtype}],
        "primary_key": "user_id",
        "source": {"type": "parquet", "paths": [users_path]},
    })];
    if let Some(references) = references {
        let events_path = root.join("events.parquet");
        write_column(&events_path, references);
        tables.push(json!({
            "name": "events",
            "columns": [{"name": "user_id", "ingested_dtype": dtype}],
            "fkeys": [{"src_column": "user_id", "dst_table": "users"}],
            "source": {"type": "parquet", "paths": [events_path]},
        }));
    }
    let config = json!({"dataset_name": "pk_map_memory", "num_shards": 1, "tables": tables});
    let config_path = root.join("graph_config.json");
    fs::write(&config_path, config.to_string()).unwrap();

    let base = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let progress_base = Arc::clone(&base);
    let progress_peak = Arc::clone(&peak);
    let options = IngestOptions {
        parallel: false,
        pk_map_threads: Some(1),
        progress: Some(Arc::new(move |event: ProgressEvent| {
            if event.phase != ProgressPhase::PkMap {
                return;
            }
            match event.kind {
                ProgressKind::Start => {
                    let live = LIVE.load(Ordering::Relaxed);
                    progress_base.store(live, Ordering::Relaxed);
                    PEAK.store(live, Ordering::Relaxed);
                }
                ProgressKind::Finish => {
                    let top = PEAK.load(Ordering::Relaxed);
                    progress_peak.store(
                        top - progress_base.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                }
                _ => {}
            }
        })),
        ..Default::default()
    };
    let reserved = Arc::new(AtomicU64::new(0));
    let subscriber = tracing_subscriber::registry().with(ReservationLayer(Arc::clone(&reserved)));
    let index = root.join("index");
    tracing::subscriber::with_default(subscriber, || {
        ingest(&config_path, index.to_str().unwrap(), Dest::Local, options).unwrap();
    });
    let run = PkMapRun {
        reserved: reserved.load(Ordering::Relaxed),
        peak: peak.load(Ordering::Relaxed) as u64,
    };
    assert!(run.reserved > 0 && run.peak > 0, "pk_map was not measured");
    run
}

fn int_keys(range: std::ops::Range<i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(range))
}

fn present_int_references(count: i64) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..count).map(|i| i % OWN_KEYS),
    ))
}

fn random_string(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    let mut out = String::with_capacity(32);
    for _ in 0..2 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        out.push_str(&format!("{x:016x}"));
    }
    out
}

fn string_keys(seeds: impl Iterator<Item = u64>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(seeds.map(random_string)))
}

#[test]
fn pk_map_reservation_covers_the_real_build_and_ignores_present_references() {
    let dir = tempfile::tempdir().unwrap();
    let own = OWN_KEYS as u64;

    let alone = run_pk_map(
        &dir.path().join("int-alone"),
        "int64",
        int_keys(0..OWN_KEYS),
        None,
    );
    let present = run_pk_map(
        &dir.path().join("int-present"),
        "int64",
        int_keys(0..OWN_KEYS),
        Some(present_int_references(5 * OWN_KEYS)),
    );
    let missing = run_pk_map(
        &dir.path().join("int-missing"),
        "int64",
        int_keys(0..OWN_KEYS),
        Some(int_keys(OWN_KEYS..2 * OWN_KEYS)),
    );
    let strings = run_pk_map(
        &dir.path().join("string-mixed"),
        "string",
        string_keys(0..own),
        Some(string_keys((0..own / 2).chain(own..own + own / 2))),
    );

    for (case, run) in [
        ("int alone", &alone),
        ("int present references", &present),
        ("int missing references", &missing),
        ("string present and missing references", &strings),
    ] {
        eprintln!("{case}: reserved {} B, peak {} B", run.reserved, run.peak);
        assert!(
            run.peak <= run.reserved,
            "{case}: peak {} B above reservation {} B",
            run.peak,
            run.reserved
        );
    }
    assert_eq!(present.reserved, alone.reserved);
    assert!(missing.reserved > alone.reserved);
}
