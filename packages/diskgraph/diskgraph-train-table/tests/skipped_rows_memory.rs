// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writing a prediction source's skipped rows keeps memory bounded: the peak
//! heap of a run that skips 2M rows is no more than one that skips 256k.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_core::time::Seconds;
use diskgraph_core::tt_builder::*;
use diskgraph_ingest::coercion::CoercionMode;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::*;
use diskgraph_ingest::io::Dest;
use diskgraph_train_table::custom_tt_row_errors::IngestOnError;
use diskgraph_train_table::ttgen_executor::{execute_tt_pass_with_source, CustomTtSource};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

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
        vec![Arc::new(Int64Array::from((0..5).collect::<Vec<i64>>()))],
    )
    .unwrap();
    let path = data.join("users.parquet");
    let mut writer = ArrowWriter::try_new(fs::File::create(&path).unwrap(), users.schema(), None)
        .unwrap();
    writer.write(&users).unwrap();
    writer.close().unwrap();
    let config = GraphConfig {
        dataset_name: "skipped_rows_memory".to_string(),
        tables: vec![TableSpec {
            name: "users".to_string(),
            columns: vec![ColumnSpec {
                name: "user_id".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Int64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("user_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![path.to_str().unwrap().to_string()],
            },
            source_size: None,
            fkeys: vec![],
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        }],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: CoercionMode::Lossless,
    };
    let config_path = data.join("graph_config.json");
    fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();
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
    let per_group = ROWS_PER_GROUP;
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(per_group))
        .build();
    let mut writer =
        ArrowWriter::try_new(fs::File::create(path).unwrap(), schema.clone(), Some(props)).unwrap();
    for start in (0..rows).step_by(per_group) {
        let ids: Vec<i64> = (start..start + per_group).map(|i| 1_000 + i as i64).collect();
        let times = vec![1_700_000_000_000_000_000i64; per_group];
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
    DiskGraphTTPlan::<Seconds> {
        entity: EntitySpec::<Seconds> {
            table: "users".to_string(),
            column: "user_id".to_string(),
            output_time_col_name: None,
            default_time_value: None,
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Node,
        is_prediction: true,
        split: None,
        stages: vec![],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    }
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
            parquet::file::reader::FileReader::metadata(
                &parquet::file::reader::SerializedFileReader::new(
                    fs::File::open(part.unwrap().path()).unwrap(),
                )
                .unwrap(),
            )
            .file_metadata()
            .num_rows() as usize
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
