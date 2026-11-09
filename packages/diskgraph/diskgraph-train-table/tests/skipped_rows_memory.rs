// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peak heap while writing skipped rows does not grow with their count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_core::time::Seconds;
use diskgraph_core::tt_builder::DiskGraphTTPlan;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::io::Dest;
use diskgraph_train_table::custom_tt_row_errors::IngestOnError;
use diskgraph_train_table::ttgen_executor::{execute_tt_pass_with_source, CustomTtSource};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::json;

#[global_allocator]
static HEAP: PeakAllocator = PeakAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct PeakAllocator;

unsafe impl GlobalAlloc for PeakAllocator {
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

/// Two of the writer's 64k-row flushes per source row group.
const ROWS_PER_GROUP: usize = 128 * 1024;

fn users_graph(root: &Path) -> PathBuf {
    let data = root.join("data");
    fs::create_dir_all(&data).unwrap();
    let users = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("user_id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from_iter_values(0..5))],
    )
    .unwrap();
    let path = data.join("users.parquet");
    let mut writer = ArrowWriter::try_new(fs::File::create(&path).unwrap(), users.schema(), None)
        .unwrap();
    writer.write(&users).unwrap();
    writer.close().unwrap();
    let config = json!({
        "dataset_name": "skipped_rows_memory",
        "num_shards": 1,
        "tables": [{
            "name": "users",
            "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
            "primary_key": "user_id",
            "source": {"type": "parquet", "paths": [path]},
        }],
    });
    let config_path = data.join("graph_config.json");
    fs::write(&config_path, config.to_string()).unwrap();
    let index = root.join("index");
    let options = IngestOptions {
        parallel: false,
        ..Default::default()
    };
    diskgraph_ingest::ingest(&config_path, index.to_str().unwrap(), Dest::Local, options)
        .unwrap();
    index
}

/// `rows` source rows, all for entities absent from the graph, in row groups
/// of `ROWS_PER_GROUP`.
fn unknown_rows(path: &Path, rows: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
        .build();
    let mut writer =
        ArrowWriter::try_new(fs::File::create(path).unwrap(), schema.clone(), Some(props)).unwrap();
    for start in (0..rows).step_by(ROWS_PER_GROUP) {
        let ids: Vec<i64> = (start..start + ROWS_PER_GROUP).map(|i| 1_000 + i as i64).collect();
        let times = vec![1_700_000_000_000_000_000i64; ROWS_PER_GROUP];
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(TimestampNanosecondArray::from(times)),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
}

fn plan() -> DiskGraphTTPlan<Seconds> {
    serde_json::from_value(json!({
        "entity": {"table": "users", "column": "user_id"},
        "prediction_type": "node",
        "is_prediction": true,
        "stages": [],
        "target_stype": "numerical",
    }))
    .unwrap()
}

/// Peak heap above the start of the pass, and the skipped rows written.
fn peak_while_skipping(index: &Path, root: &Path, rows: usize) -> (usize, usize) {
    let source = root.join(format!("rows-{rows}.parquet"));
    unknown_rows(&source, rows);
    let out = root.join(format!("out-{rows}"));
    fs::create_dir_all(&out).unwrap();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let output = pool
        .install(|| {
            execute_tt_pass_with_source(
                &plan(),
                index,
                &out,
                Some(CustomTtSource {
                    target_window_end: None,
                    parquet_path: source.clone(),
                    input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
                    on_error: IngestOnError::SkipRowOnError,
                    allow_negative_weight: false,
                    timestamp_reference: None,
                }),
            )
        })
        .unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    assert_eq!(output.rows_written, 0);
    assert_eq!(output.row_failures.unwrap().skipped_rows, rows as u64);
    let written = fs::read_dir(out.join("skipped"))
        .unwrap()
        .map(|part| {
            let reader = SerializedFileReader::new(fs::File::open(part.unwrap().path()).unwrap());
            reader.unwrap().metadata().file_metadata().num_rows() as usize
        })
        .sum();
    (peak, written)
}

#[test]
fn test_skipped_rows_memory_does_not_grow_with_the_skipped_count() {
    let dir = tempfile::tempdir().unwrap();
    let index = users_graph(dir.path());
    let (small, small_rows) = peak_while_skipping(&index, dir.path(), 2 * ROWS_PER_GROUP);
    let (large, large_rows) = peak_while_skipping(&index, dir.path(), 16 * ROWS_PER_GROUP);
    eprintln!("peak heap: {small} bytes at {small_rows} skipped, {large} bytes at {large_rows} skipped");
    assert_eq!((small_rows, large_rows), (2 * ROWS_PER_GROUP, 16 * ROWS_PER_GROUP));
    // Holding the skipped rows would add well over 16 bytes per row (~28 MiB
    // more at 2M); a bounded writer stays within a small fixed margin.
    assert!(large < small + 4 * 1024 * 1024, "{small} -> {large}");
}
