// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the custom-TT executor path (TT-NG Phase 6).
//!
//! These tests cover three cross-cutting aspects:
//!
//! 1. **Phase 0d fixtures**: deterministic synthesis of customer-supplied
//!    parquet files (ENTITY1, TIMESTAMP, TARGET1, autoregression, yoy) into
//!    a tempdir at test time. No binary parquets committed to git.
//! 2. **Phase 0e plan helpers**: hand-crafted `DiskGraphTTPlan` shapes that
//!    correspond to the canonical custom-TT scenarios (complete /
//!    target-only / missing-{autoreg,yoy}). The plan compiler will eventually
//!    emit these shapes; for now they stand in.
//! 3. **Phase 6 plumbing**: `custom_tt_chunk_processor` + source-kind dispatch
//!    in `execute_tt_pass_with_source`. Custom-TT custom splits ride as a SPLIT
//!    COLUMN in the customer parquet (`ManualSplit` on `split_col`), routed via
//!    `ColumnDeciderManual` and dropped from the seed output — there is no
//!    `split.bin` sidecar.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::*;
use arrow_array::{
    Array, Float64Array, Int64Array, ListArray, RecordBatch, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_core::time::{Abs, Dur, Seconds};
use diskgraph_core::tt_builder::*;
use diskgraph_ingest::coercion::CoercionMode;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::*;
use diskgraph_ingest::io::Dest;
use diskgraph_train_table::ttgen_executor::{
    execute_tt_pass_with_customizer, execute_tt_pass_with_source, CustomTtSource,
};
use diskgraph_train_table::ttgen_resolve::ResolvedPlan;
use diskgraph_train_table::ttgen_resolve::{AUTOREGRESSION_OUTPUT_ALIAS, YOY_OUTPUT_ALIAS};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

// ─── Test infrastructure ─────────────────────────────────────────────────────

fn write_parquet(path: &Path, batch: &RecordBatch) {
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn mk_col(name: &str, dtype: IngestedType) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source_dtype: None,
        ingested_dtype: dtype,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    }
}

fn run_ingest(config_path: &Path, index_path: &Path) {
    let options = IngestOptions {
        parallel: false,
        ..Default::default()
    };
    diskgraph_ingest::ingest(
        config_path,
        index_path.to_str().unwrap(),
        Dest::Local,
        options,
    )
    .unwrap();
}

// ─── Phase 0d: synthesize customer-supplied parquets ─────────────────────────
//
// We reuse the existing PK domain {0..N} of a small synthesized graph index
// (a single users table, no edges needed). Each customer parquet has a
// subset of {ENTITY1, TIMESTAMP, TARGET1, autoregression, yoy}. The
// timestamps are encoded as nanoseconds (matching tt.parquet's TIMESTAMP
// column) but the values themselves are deterministic seeded.
//
// The deterministic seed is a simple hash of (entity, ts, kind); no `rand`
// crate dependency is needed.

fn det_target(entity: i64, ts_secs: i64) -> f64 {
    // Deterministic linear function of entity & ts for reproducible tests.
    (entity as f64) * 100.0 + (ts_secs as f64)
}

fn det_autoreg(entity: i64, ts_secs: i64) -> Vec<f64> {
    // 3 lag values, deterministic.
    (0..3)
        .map(|k| det_target(entity, ts_secs - (k + 1) * 86400))
        .collect()
}

fn det_yoy(entity: i64, ts_secs: i64) -> Vec<f64> {
    (0..2)
        .map(|k| det_target(entity, ts_secs - (k + 1) * 365 * 86400))
        .collect()
}

/// Configuration for synthesizing one customer parquet.
struct CustomTtFixtureSpec {
    /// Number of (entity, anchor_ts) rows to emit.
    n_rows: usize,
    /// Entity domain — must match the index's PK range. We cycle entity_ids
    /// across n_rows.
    entity_pks: Vec<i64>,
    /// Anchor timestamps (epoch seconds). One per row, cycled if shorter.
    anchor_ts_secs: Vec<i64>,
    /// Whether to emit TARGET1, autoregression, yoy columns.
    has_target: bool,
    has_autoreg: bool,
    has_yoy: bool,
}

fn synth_custom_tt_parquet(out_path: &Path, spec: &CustomTtFixtureSpec) {
    let mut entity = Int64Builder::new();
    let mut ts = Int64Builder::new(); // nanoseconds
    let mut target = Float64Builder::new();
    let mut autoreg = ListBuilder::new(Float64Builder::new());
    let mut yoy = ListBuilder::new(Float64Builder::new());
    for i in 0..spec.n_rows {
        let e = spec.entity_pks[i % spec.entity_pks.len()];
        let t_secs = spec.anchor_ts_secs[i % spec.anchor_ts_secs.len()];
        entity.append_value(e);
        ts.append_value(t_secs * 1_000_000_000);
        if spec.has_target {
            target.append_value(det_target(e, t_secs));
        }
        if spec.has_autoreg {
            for v in det_autoreg(e, t_secs) {
                autoreg.values().append_value(v);
            }
            autoreg.append(true);
        }
        if spec.has_yoy {
            for v in det_yoy(e, t_secs) {
                yoy.values().append_value(v);
            }
            yoy.append(true);
        }
    }

    let mut fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ];
    let mut columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(
            (0..spec.n_rows)
                .map(|i| spec.anchor_ts_secs[i % spec.anchor_ts_secs.len()] * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
    ];
    drop(ts); // ts builder unused beyond ts_arr above
    if spec.has_target {
        fields.push(Field::new("TARGET1", DataType::Float64, true));
        columns.push(Arc::new(target.finish()));
    }
    if spec.has_autoreg {
        fields.push(Field::new(
            AUTOREGRESSION_OUTPUT_ALIAS,
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ));
        columns.push(Arc::new(autoreg.finish()));
    }
    if spec.has_yoy {
        fields.push(Field::new(
            YOY_OUTPUT_ALIAS,
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ));
        columns.push(Arc::new(yoy.finish()));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize a Node-shaped custom-TT parquet WITH a customer-provided split
/// column. One row per `(entity_pk, split)` pair: ENTITY1, TIMESTAMP, TARGET1,
/// and `split_col` (Int64, 0=train/1=val/2=test — the kumo baseline
/// convention). Mirrors what kumo feeds CTT for a custom split.
fn synth_custom_tt_parquet_with_split_col(
    out_path: &Path,
    entity_pks: &[i64],
    ts_secs: &[i64],
    split_values: &[i64],
) {
    assert_eq!(entity_pks.len(), ts_secs.len());
    assert_eq!(entity_pks.len(), split_values.len());
    let mut entity = Int64Builder::new();
    let mut target = Float64Builder::new();
    let mut split = Int64Builder::new();
    for i in 0..entity_pks.len() {
        entity.append_value(entity_pks[i]);
        target.append_value(det_target(entity_pks[i], ts_secs[i]));
        split.append_value(split_values[i]);
    }
    let fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        // The kumo split convention: column `split_col`, 0/1/2.
        Field::new("split_col", DataType::Int64, true),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(
            ts_secs
                .iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
        Arc::new(target.finish()),
        Arc::new(split.finish()),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// A `ManualSplit` spec on a customer-provided split column (`split_col`),
/// matching the kumo CTT custom-split contract: `manual_*_values` are the
/// stringified stage constants 0=train / 1=val / 2=test.
fn manual_split_col_spec() -> SplitSpec<Seconds> {
    SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("ManualSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: Some("split_col".to_string()),
        split_output_column: None,
        manual_train_values: Some(vec!["0".to_string()]),
        manual_val_values: Some(vec!["1".to_string()]),
        manual_test_values: Some(vec!["2".to_string()]),
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    }
}

// ─── Build a tiny graph index for entity-PK reconciliation ───────────────────
//
// We use the simplest possible index: one table `users` with PKs {0..n-1},
// no edges. Custom-TT tests only need PK lookup, not graph traversal.

fn build_users_only_graph(root: &Path, n_users: i64) -> PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    let mut user_id = Int64Builder::new();
    for i in 0..n_users {
        user_id.append_value(i);
    }
    let users_schema = Schema::new(vec![Field::new("user_id", DataType::Int64, false)]);
    let users_batch =
        RecordBatch::try_new(Arc::new(users_schema), vec![Arc::new(user_id.finish())]).unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    let config = GraphConfig {
        timestamp_unit: Default::default(),
        dataset_name: "test_custom_tt".to_string(),
        tables: vec![TableSpec {
            name: "users".to_string(),
            columns: vec![mk_col("user_id", IngestedType::Int64)],
            primary_key: Some("user_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![data_dir.join("users.parquet").to_str().unwrap().to_string()],
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

    let config_path = data_dir.join("graph_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    run_ingest(&config_path, &index_dir);
    index_dir
}

/// Build a two-table graph: `users` (PKs 0..n_users) and `products`
/// (PKs 0..n_products). Both tables carry a primary key map. Used to
/// exercise the custom-TT RHS-target orphan filter, where TARGET1 holds a
/// product_id that must be validated against the `products` PK map.
fn build_users_and_products_graph(root: &Path, n_users: i64, n_products: i64) -> PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    let mut user_id = Int64Builder::new();
    for i in 0..n_users {
        user_id.append_value(i);
    }
    let users_schema = Schema::new(vec![Field::new("user_id", DataType::Int64, false)]);
    let users_batch =
        RecordBatch::try_new(Arc::new(users_schema), vec![Arc::new(user_id.finish())]).unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    let mut product_id = Int64Builder::new();
    for i in 0..n_products {
        product_id.append_value(i);
    }
    let products_schema = Schema::new(vec![Field::new("product_id", DataType::Int64, false)]);
    let products_batch = RecordBatch::try_new(
        Arc::new(products_schema),
        vec![Arc::new(product_id.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &products_batch);

    let mk_table = |name: &str, pk: &str, file: &str| TableSpec {
        name: name.to_string(),
        columns: vec![mk_col(pk, IngestedType::Int64)],
        primary_key: Some(pk.to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![data_dir.join(file).to_str().unwrap().to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    };

    let config = GraphConfig {
        timestamp_unit: Default::default(),
        dataset_name: "test_custom_tt_lp".to_string(),
        tables: vec![
            mk_table("users", "user_id", "users.parquet"),
            mk_table("products", "product_id", "products.parquet"),
        ],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: CoercionMode::Lossless,
    };

    let config_path = data_dir.join("graph_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    run_ingest(&config_path, &index_dir);
    index_dir
}

/// Synthesize a custom-TT LP parquet: ENTITY1 (user_id) + TIMESTAMP +
/// TARGET1 (product_id, an Int64 FK into the products table). Each row's
/// TARGET1 is taken directly from `target_product_ids`.
fn synth_custom_tt_lp_parquet(
    out_path: &Path,
    entity_pks: &[i64],
    ts_secs: &[i64],
    target_product_ids: &[i64],
) {
    assert_eq!(entity_pks.len(), ts_secs.len());
    assert_eq!(entity_pks.len(), target_product_ids.len());
    let mut entity = Int64Builder::new();
    let mut target = Int64Builder::new();
    for i in 0..entity_pks.len() {
        entity.append_value(entity_pks[i]);
        target.append_value(target_product_ids[i]);
    }
    let fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Int64, true),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(
            ts_secs
                .iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
        Arc::new(target.finish()),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize an array-target custom-TT LP parquet: ENTITY1 + TIMESTAMP +
/// TARGET1 List<Int64>. This is the shape Kumo feeds to TT-NG CTT for
/// link-prediction training.
fn synth_custom_tt_lp_array_parquet(
    out_path: &Path,
    entity_pks: &[i64],
    ts_secs: &[i64],
    target_product_ids: &[Vec<i64>],
) {
    synth_custom_tt_lp_array_parquet_inner(out_path, entity_pks, ts_secs, target_product_ids, None)
}

fn synth_custom_tt_lp_array_parquet_inner(
    out_path: &Path,
    entity_pks: &[i64],
    ts_secs: &[i64],
    target_product_ids: &[Vec<i64>],
    weights: Option<&[Vec<f64>]>,
) {
    assert_eq!(entity_pks.len(), ts_secs.len());
    assert_eq!(entity_pks.len(), target_product_ids.len());
    let mut entity = Int64Builder::new();
    let mut target = ListBuilder::new(Int64Builder::new());
    let mut frequency = ListBuilder::new(Int64Builder::new());
    let mut weight = ListBuilder::new(Float64Builder::new());
    for i in 0..entity_pks.len() {
        entity.append_value(entity_pks[i]);
        for &product_id in &target_product_ids[i] {
            target.values().append_value(product_id);
            frequency.values().append_value(1);
        }
        if let Some(w) = weights {
            for &v in &w[i] {
                weight.values().append_value(v);
            }
            weight.append(true);
        }
        target.append(true);
        frequency.append(true);
    }
    let mut fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new(
            "TARGET1",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "__kumo_internal_frequency_column__",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ];
    let mut columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(
            ts_secs
                .iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
        Arc::new(target.finish()),
        Arc::new(frequency.finish()),
    ];
    if weights.is_some() {
        fields.push(Field::new(
            "WEIGHT",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ));
        columns.push(Arc::new(weight.finish()));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Read the TARGET1 (Int64) column of an LP custom-TT tt.parquet output.
/// Returns (ENTITY1, TARGET1) tuples.
fn read_parquet_batches(path: &Path) -> Vec<RecordBatch> {
    let paths = if path.is_dir() {
        diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(path).unwrap()
    } else {
        vec![path.to_path_buf()]
    };
    paths
        .into_iter()
        .flat_map(|part| {
            ParquetRecordBatchReaderBuilder::try_new(fs::File::open(part).unwrap())
                .unwrap()
                .build()
                .unwrap()
                .map(|batch| batch.unwrap())
        })
        .collect()
}

fn read_tt_parquet_lp_targets(tt_path: &Path) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(tt_path) {
        let schema = batch.schema();
        let e1 = batch
            .column(schema.index_of("ENTITY1").unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let t1 = batch
            .column(schema.index_of("TARGET1").unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((e1.value(i), t1.value(i)));
        }
    }
    out.sort();
    out
}

/// Plan for a custom-TT LP-shaped Node prediction: parquet has
/// ENTITY1 + TIMESTAMP + TARGET1, where TARGET1 is an FK into
/// `target_entity_table`. The fkey_mapping carries both the ENTITY1 → users
/// mapping and the TARGET1 → products RHS mapping (mirrors what
/// `build_train_table_artifacts_for_existing_table` constructs).
fn make_custom_tt_lp_node_plan(target_entity_table: Option<&str>) -> DiskGraphTTPlan<Seconds> {
    let mut fkey_mapping = vec![TTFKeySpec {
        src_col: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
    }];
    if let Some(t) = target_entity_table {
        fkey_mapping.push(TTFKeySpec {
            src_col: "TARGET1".to_string(),
            dst_table: t.to_string(),
        });
    }
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
        is_prediction: false,
        split: None,
        stages: vec![],
        target_stype: Some("ID".to_string()),
        weight_col: None,
        fkey_mapping: Some(fkey_mapping),
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    }
}

// ─── Phase 0e: hand-crafted plan helpers ─────────────────────────────────────

/// Plan for "custom-TT complete" forecasting: parquet has
/// ENTITY1+TIMESTAMP+autoregression+yoy. Empty stage list — executor runs no
/// ops; the parquet provides everything.
///
/// Note: we use `PredictionType::Forecasting` so the output schema is
/// (ENTITY, TIMESTAMP, AUTOREGRESSION_COL, YOY_COL).
fn make_custom_tt_complete_forecasting_plan() -> DiskGraphTTPlan<Seconds> {
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
        prediction_type: PredictionType::Forecasting,
        is_prediction: false,
        split: None,
        // Empty stages: nothing to compute, every column comes from parquet.
        stages: vec![],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    }
}

fn date_offset_split_spec() -> SplitSpec<Seconds> {
    SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("DateOffsetSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
        offsets: Some(vec![Dur::from_raw(-200), Dur::from_raw(-100)]),
        offset_unit: Some("seconds".to_string()),
        unit: Some("seconds".to_string()),
        timeframe: Some(100),
        anchor_start_offset: Some(0),
        ranges: None,
        backtesting_timestamp: None,
        split_column: None,
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: Some(false),
        timeframe_step: None,
    }
}

/// Plan for "custom-TT target-only" Node prediction: parquet has only
/// ENTITY1+TIMESTAMP+TARGET1. No autoregression / yoy ops. Empty stages.
fn make_custom_tt_target_only_node_plan() -> DiskGraphTTPlan<Seconds> {
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
        is_prediction: false,
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

// ─── Helpers to read the produced tt.parquet ─────────────────────────────────

/// Return the produced tt.parquet's column names (schema-shape assertions) and
/// its row count. Used by the predict tests to prove NO TARGET column exists.
fn read_tt_parquet_columns(tt_path: &Path) -> (Vec<String>, usize) {
    let batches = read_parquet_batches(tt_path);
    let names: Vec<String> = batches
        .first()
        .expect("training table dataset has a batch")
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    (names, rows)
}

fn read_tt_parquet_node(tt_path: &Path) -> Vec<(i64, i64, f64)> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(tt_path) {
        let entities = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(1)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let target = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((
                entities.value(i),
                ts.value(i) / 1_000_000_000,
                target.value(i),
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

#[derive(Debug, Clone)]
struct ForecastingTtRow {
    entity: i64,
    ts_secs: i64,
    autoreg: Vec<f64>,
    yoy: Option<Vec<f64>>,
}

fn read_tt_parquet_forecasting(tt_path: &Path) -> Vec<ForecastingTtRow> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(tt_path) {
        let entities = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(1)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let schema = batch.schema();
        let ar_idx = schema.index_of(AUTOREGRESSION_OUTPUT_ALIAS).unwrap();
        let ar = batch
            .column(ar_idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let yoy_opt = if let Ok(yoy_idx) = schema.index_of(YOY_OUTPUT_ALIAS) {
            Some(
                batch
                    .column(yoy_idx)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap(),
            )
        } else {
            None
        };
        for i in 0..batch.num_rows() {
            let ar_list = ar.value(i);
            let ar_f = ar_list.as_any().downcast_ref::<Float64Array>().unwrap();
            let ar_vec: Vec<f64> = (0..ar_f.len()).map(|j| ar_f.value(j)).collect();
            let yoy_vec = yoy_opt.map(|y| {
                let l = y.value(i);
                let f = l.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..f.len()).map(|j| f.value(j)).collect()
            });
            out.push(ForecastingTtRow {
                entity: entities.value(i),
                ts_secs: ts.value(i) / 1_000_000_000,
                autoreg: ar_vec,
                yoy: yoy_vec,
            });
        }
    }
    out.sort_by(|a, b| a.entity.cmp(&b.entity).then(a.ts_secs.cmp(&b.ts_secs)));
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Phase 0d sanity check — fixtures synthesize correctly.
#[test]
fn test_fixture_synthesis_target_only() {
    let dir = tempfile::tempdir().unwrap();
    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 20,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000, 3_000_000, 4_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);
    let file = fs::File::open(&pq).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut total = 0;
    for batch_res in reader {
        let b = batch_res.unwrap();
        total += b.num_rows();
        assert_eq!(b.schema().fields().len(), 3); // ENTITY1, TIMESTAMP, TARGET1
    }
    assert_eq!(total, 20);
}

#[test]
fn test_fixture_synthesis_forecasting_complete() {
    let dir = tempfile::tempdir().unwrap();
    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 10,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000],
        has_target: true,
        has_autoreg: true,
        has_yoy: true,
    };
    synth_custom_tt_parquet(&pq, &spec);
    let file = fs::File::open(&pq).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut total = 0;
    for batch_res in reader {
        let b = batch_res.unwrap();
        total += b.num_rows();
        assert_eq!(b.schema().fields().len(), 5); // ENTITY1, TIMESTAMP, TARGET1, autoreg, yoy
    }
    assert_eq!(total, 10);
}

/// Phase 6 main scenario A: "custom-TT target-only" — Node prediction with
/// empty plan stages. Output tt.parquet has (ENTITY, TIMESTAMP, TARGET) and
/// matches the input rows.
#[test]
fn test_custom_tt_target_only_node() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 20,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000, 3_000_000, 4_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 20);

    let rows = read_tt_parquet_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 20);
    // Every row's TARGET should match det_target.
    for (e, ts, t) in &rows {
        let expected = det_target(*e, *ts);
        assert!(
            (t - expected).abs() < 1e-9,
            "row ({e},{ts}): {t} != {expected}"
        );
    }
}

/// Synthesize a PREDICT seed parquet: ENTITY1 + TIMESTAMP, NO target column.
/// Every row shares `ts_secs` (nanoseconds on disk). Models an LP/node predict
/// seed table (entities + anchor time, no positives).
fn synth_static_custom_tt_parquet_with_ts(out_path: &Path, entity_pks: &[i64], ts_secs: i64) {
    let fields = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(Int64Array::from(entity_pks.to_vec())),
        Arc::new(TimestampNanosecondArray::from(
            entity_pks
                .iter()
                .map(|_| ts_secs * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize a STATIC custom-TT node parquet: ENTITY1 (+ optional TARGET1),
/// and deliberately NO TIMESTAMP column. Used to prove DG synthesizes the
/// native-static sentinel instead of erroring / forcing kumo to rewrite the
/// parquet. `entity_pks` is written verbatim (one row each).
fn synth_static_custom_tt_parquet(out_path: &Path, entity_pks: &[i64], has_target: bool) {
    let mut fields = vec![Field::new("ENTITY1", DataType::Int64, false)];
    let mut columns: Vec<Arc<dyn Array>> = vec![Arc::new(Int64Array::from(entity_pks.to_vec()))];
    if has_target {
        fields.push(Field::new("TARGET1", DataType::Float64, true));
        // Deterministic target keyed off the sentinel ts (0) so the assertion
        // can recompute it.
        let targets: Vec<f64> = entity_pks.iter().map(|e| det_target(*e, 0)).collect();
        columns.push(Arc::new(Float64Array::from(targets)));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// STATIC custom TRAIN: a parquet with no TIMESTAMP column must NOT error;
/// DG synthesizes the native-static sentinel (raw 0 → epoch 0) for every row.
#[test]
fn test_custom_tt_static_train_synthesizes_sentinel_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("static_train.parquet");
    let entity_pks: Vec<i64> = (0..5).collect();
    synth_static_custom_tt_parquet(&pq, &entity_pks, /* has_target */ true);

    let plan = make_custom_tt_target_only_node_plan(); // is_prediction = false
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 5);

    let rows = read_tt_parquet_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 5);
    for (e, ts, t) in &rows {
        // Sentinel: every synthesized anchor is epoch 0.
        assert_eq!(*ts, 0, "static train row {e} must carry the sentinel ts 0");
        let expected = det_target(*e, 0);
        assert!((t - expected).abs() < 1e-9, "row {e}: {t} != {expected}");
    }
}

/// STATIC custom PREDICT: no TIMESTAMP and no TARGET1 column. Composes with the
/// LP/node predict fix — DG synthesizes the sentinel and emits one seed row per
/// entity (targets absent for predict).
#[test]
fn test_custom_tt_static_predict_synthesizes_sentinel_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("static_predict.parquet");
    let entity_pks: Vec<i64> = (0..5).collect();
    // Prediction seed table: ENTITY only, no TIMESTAMP, no TARGET1.
    synth_static_custom_tt_parquet(&pq, &entity_pks, /* has_target */ false);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.is_prediction = true;
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 5);

    // Predict bundle: `[ENTITY1, TIMESTAMP]` only — NO TARGET column.
    let (cols, n) = read_tt_parquet_columns(&output.tt_parquet_path);
    assert_eq!(n, 5);
    assert_eq!(cols, vec!["ENTITY1".to_string(), "TIMESTAMP".to_string()]);
    assert!(cols.iter().all(|c| !c.starts_with("TARGET")));
}

/// A parquet WITH a TIMESTAMP column is unaffected by the sentinel path — it
/// uses the real per-row values (guards against a regression where the
/// synthesized sentinel leaks into the temporal case).
#[test]
fn test_custom_tt_present_timestamp_uses_real_values() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("temporal.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 20,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000, 3_000_000, 4_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 20);

    let rows = read_tt_parquet_node(&output.tt_parquet_path);
    // No row may collapse to the sentinel — every anchor is a real value.
    assert!(rows.iter().all(|(_, ts, _)| *ts != 0));
    for (_, ts, _) in &rows {
        assert!(spec.anchor_ts_secs.contains(ts), "unexpected ts {ts}");
    }
}

/// Phase 6 main scenario B: "custom-TT complete forecasting" — parquet has
/// every output column. Plan has empty stages. Output tt.parquet should
/// equal input parquet (modulo column-name renaming: input "autoregression"
/// → output "autoregression_labels", input "yoy" → output "year_over_year_labels").
#[test]
fn test_custom_tt_complete_forecasting_passthrough() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 12,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000, 3_000_000],
        has_target: true,
        has_autoreg: true,
        has_yoy: true,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_complete_forecasting_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 12);

    let rows = read_tt_parquet_forecasting(&output.tt_parquet_path);
    assert_eq!(rows.len(), 12);
    for r in &rows {
        let expected_ar = det_autoreg(r.entity, r.ts_secs);
        assert_eq!(r.autoreg.len(), expected_ar.len());
        for (a, b) in r.autoreg.iter().zip(expected_ar.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
        let expected_yoy = det_yoy(r.entity, r.ts_secs);
        let yoy_actual = r.yoy.as_ref().expect("yoy column missing in output");
        assert_eq!(yoy_actual.len(), expected_yoy.len());
        for (a, b) in yoy_actual.iter().zip(expected_yoy.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    let stats_payload = output.stats.as_ref().unwrap();
    let outer = stats_payload.parsed_train_table_col_stats.as_ref().unwrap();
    let stats = outer.get("stats").unwrap().as_object().unwrap();
    let has_target_stats = stats.keys().any(|k| k.starts_with("TARGET1__"));
    assert!(
        has_target_stats,
        "TARGET1 stats must be present in parsed_train_table_col_stats"
    );
}

#[test]
fn test_custom_tt_forecasting_date_offset_split_routes_all_stages() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt_forecasting_date_offset.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 5,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![0, 100, 200, 300, 400],
        has_target: true,
        has_autoreg: true,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let mut plan = make_custom_tt_complete_forecasting_plan();
    plan.split = Some(date_offset_split_spec());
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq,
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    assert_eq!(output.rows_written, 5);
    let infos = output
        .stage_gen_infos
        .as_ref()
        .expect("DateOffsetSplit must produce stage seeds for forecasting CTT rows");
    let count = |key: &str| infos.iter().find(|info| info.key == key).unwrap().row_count;
    assert_eq!(count("train"), 2);
    assert_eq!(count("val"), 1);
    assert_eq!(count("test"), 2);
}

/// Unknown entity PK → rows skipped (documented behavior). Customer parquet
/// contains an entity_pk that doesn't exist in the graph index; that row is
/// silently dropped from output.
#[test]
fn test_custom_tt_unknown_entity_pk_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5); // PKs 0..5

    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 10,
        // Half the entity_pks are unknown (5..10).
        entity_pks: (0..10).collect(),
        anchor_ts_secs: vec![1_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    // Only entity_pks {0,1,2,3,4} exist in the index → 5 rows kept; the other
    // 5 should be silently skipped.
    assert_eq!(output.rows_written, 5, "unknown PKs should be skipped");
    let rows = read_tt_parquet_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 5);
    let entities: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let mut sorted = entities.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

/// Read a stage seed part's ENTITY1 column AND assert `split_col` is absent
/// from its schema. Returns the entity ids.
fn read_seed_entities_assert_no_split(out_dir: &Path, part: &str) -> Vec<i64> {
    let file = fs::File::open(out_dir.join(part)).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut out = Vec::new();
    for batch_res in reader {
        let batch = batch_res.unwrap();
        let schema = batch.schema();
        assert!(
            schema.index_of("split_col").is_err(),
            "split_col must be dropped from seed part {part}; schema = {schema:?}"
        );
        let e1 = batch
            .column(schema.index_of("ENTITY1").unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push(e1.value(i));
        }
    }
    out.sort();
    out
}

/// CORE CTT custom-split test (replaces the old split.bin/UseBin path):
/// a customer parquet carrying a `split_col` (0/1/2) + a `ManualSplit` spec on
/// that column routes rows to the right stage AND the split column is DROPPED
/// from every per-stage seed parquet (it must never become a feature). No
/// `split.bin` is read or written.
#[test]
fn test_custom_tt_manual_split_column_routes_and_drops() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt_split.parquet");
    // 7 rows; split values {0,1,2,0,1,2,0} → train={e0,e3,e0}, val={e1,e1},
    // test={e2,e2}. Use distinct entity pks so the routing is checkable.
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4, 0, 1];
    let ts_secs: Vec<i64> = vec![1_000_000; 7];
    let split_values: Vec<i64> = vec![0, 1, 2, 0, 1, 2, 0];
    synth_custom_tt_parquet_with_split_col(&pq, &entity_pks, &ts_secs, &split_values);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.split = Some(manual_split_col_spec());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 7);

    // Per-stage row counts from the split values: train=3, val=2, test=2.
    let infos = output
        .stage_gen_infos
        .as_ref()
        .expect("ManualSplit on a column must produce stage_gen_infos");
    let info = |key: &str| infos.iter().find(|s| s.key == key).unwrap();
    assert_eq!(info("train").row_count, 3);
    assert_eq!(info("val").row_count, 2);
    assert_eq!(info("test").row_count, 2);

    // Routing correctness + split-column drop, read off the on-disk seeds.
    let stage_entities = |key: &str| -> Vec<i64> {
        let mut out = Vec::new();
        for part in &info(key).parts {
            out.extend(read_seed_entities_assert_no_split(out_dir.path(), part));
        }
        out.sort();
        out
    };
    // train rows are split-idx 0,3,6 → entity pks 0,3,1.
    assert_eq!(stage_entities("train"), vec![0, 1, 3]);
    // val rows are split-idx 1,4 → entity pks 1,4.
    assert_eq!(stage_entities("val"), vec![1, 4]);
    // test rows are split-idx 2,5 → entity pks 2,0.
    assert_eq!(stage_entities("test"), vec![0, 2]);

    // tt.parquet (the monolith) still carries split_col as the projection; the
    // DROP guarantee is at the SEED level (the samplable rows). No split.bin.
    assert!(!out_dir.path().join("split.bin").exists());
}

/// A customer-provided split via a COLUMN works with NO `split.bin` sidecar
/// (the deleted path): building with a ManualSplit column succeeds and never
/// looks for a sidecar.
#[test]
fn test_custom_tt_manual_split_column_no_sidecar_needed() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt_split2.parquet");
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3];
    let ts_secs: Vec<i64> = vec![1_000_000; 4];
    let split_values: Vec<i64> = vec![0, 0, 1, 2];
    synth_custom_tt_parquet_with_split_col(&pq, &entity_pks, &ts_secs, &split_values);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.split = Some(manual_split_col_spec());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 4);
    let infos = output.stage_gen_infos.as_ref().unwrap();
    let count = |key: &str| infos.iter().find(|s| s.key == key).unwrap().row_count;
    assert_eq!(count("train"), 2);
    assert_eq!(count("val"), 1);
    assert_eq!(count("test"), 1);
    assert!(!out_dir.path().join("split.bin").exists());
}

/// Cross-cutting: dispatch on source kind. The same `execute_tt_pass_with_source`
/// works with `None` (PQ path) and `Some(...)` (custom-TT path). For `None`
/// the existing PQ behavior holds.
#[test]
fn test_dispatch_pq_source_via_with_source_api() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    // PQ-style plan with no stages → PassThrough Node prediction with
    // default_time_value (no timestamps, no temporal). Each user produces
    // one row.
    let plan = DiskGraphTTPlan::<Seconds> {
        entity: EntitySpec::<Seconds> {
            table: "users".to_string(),
            column: "user_id".to_string(),
            output_time_col_name: None,
            default_time_value: Some(Abs::<Seconds>::from_raw(0)),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::PassThrough,
        is_prediction: false,
        split: None,
        stages: vec![],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    };
    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), None).unwrap();
    // PassThrough yields one row per user.
    assert_eq!(output.rows_written, 5);
    // No split → no stage_gen_infos.
    assert!(output.stage_gen_infos.is_none());
}

/// Sanity: the helper read_tt_parquet_node reproduces row counts.
#[test]
fn test_tt_parquet_reader_helper_works() {
    let dir = tempfile::tempdir().unwrap();
    let pq = dir.path().join("p.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 4,
        entity_pks: vec![0, 1],
        anchor_ts_secs: vec![1, 2],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let file = fs::File::open(&pq).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut total = 0;
    for b in reader {
        total += b.unwrap().num_rows();
    }
    assert_eq!(total, 4);
}

/// Pre-existing PQ tests should still pass through the new
/// `execute_tt_pass_with_source` API: identical signature to
/// `execute_tt_pass(plan, index, output_dir)` with `None` for source.
/// (We also verify `execute_tt_pass` itself still works in the existing
/// integration test file.)
#[test]
fn test_dispatch_summary_keys_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 5,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 5);
    // No split → no stage_gen_infos.
    assert!(output.stage_gen_infos.is_none());
    // Stats payload should be Some.
    assert!(output.stats.is_some());
    // The bounded HLL-backed diagnostic reports the single timeframe without
    // retaining per-timestamp counts.
    let stats = output.stats.unwrap();
    assert_eq!(stats.kumo_contract.num_timeframes, 1);
}

// ─── Deferred: executor-fills-missing-columns scenarios ───────────────────
//
// These tests describe the "customer supplies partial training table; the
// executor fills the missing columns" flow. Implementing them requires:
//   1. populate_aliases_from_record wired into custom_tt_chunk_processor's
//      per-row loop (already written + #[allow(dead_code)] today).
//   2. Plan-compiler-emitted column-name → alias-index map so the helper
//      knows which slot to fill from each customer column.
//   3. execute_stages running only the ops for slots NOT pre-filled.
//
// (1) is orchestration. (2) and (3) live in execute_stages / plan compiler
// (forecasting engineer's domain). Until that lands, these are #[ignore]d
// stubs so they're discoverable but not blocking CI.

#[test]
#[ignore = "needs plan-compiler-emitted alias map (forecasting engineer's work)"]
fn test_custom_tt_missing_autoreg_filled_by_executor() {
    // Customer parquet has ENTITY1, TIMESTAMP, TARGET1, yoy.
    // Plan has Op::Autoregression in its stages list.
    // Expected: output tt.parquet has autoregression column populated by
    // execute_stages; values match what PQ executor would have produced
    // for the same (entity, anchor) pairs.
}

#[test]
#[ignore = "needs plan-compiler-emitted alias map (forecasting engineer's work)"]
fn test_custom_tt_missing_yoy_filled_by_executor() {
    // Customer parquet has ENTITY1, TIMESTAMP, TARGET1, autoregression.
    // Plan has Op::Yoy in its stages list.
    // Expected: output tt.parquet has yoy column populated.
}

#[test]
#[ignore = "needs plan-compiler-emitted alias map (forecasting engineer's work)"]
fn test_custom_tt_target_only_both_filled_by_executor() {
    // Customer parquet has ENTITY1, TIMESTAMP, TARGET1.
    // Plan has both Op::Autoregression and Op::Yoy.
    // Expected: output tt.parquet has both lag columns populated.
}

/// Phase 9: custom-TT path emits a manifest with `source_mode = "custom_tt"`.
#[test]
fn test_custom_tt_manifest_records_source_mode() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 8,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    assert!(output.manifest_path.exists());
    let manifest = diskgraph_train_table::manifest::read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.source_mode, "custom_tt");
    assert_eq!(manifest.split_mode, "none");
    assert!(manifest.split.is_none());
    assert_eq!(manifest.row_count, output.rows_written);
}

// ─── Phase 11: multi-entity / multi-target wide-format tests ─────────────────

/// Synthesize a parquet with two ENTITY columns (ENTITY1, ENTITY2),
/// TIMESTAMP, and one TARGET1.
fn synth_multi_entity_parquet(
    out_path: &Path,
    e1_pks: &[i64],
    e2_pks: &[i64],
    ts_secs: &[i64],
    has_target: bool,
) {
    assert_eq!(e1_pks.len(), e2_pks.len());
    assert_eq!(e1_pks.len(), ts_secs.len());
    let mut e1 = Int64Builder::new();
    let mut e2 = Int64Builder::new();
    let mut target = Float64Builder::new();
    for i in 0..e1_pks.len() {
        e1.append_value(e1_pks[i]);
        e2.append_value(e2_pks[i]);
        if has_target {
            target.append_value(det_target(e1_pks[i], ts_secs[i]) + e2_pks[i] as f64);
        }
    }
    let mut fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("ENTITY2", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ];
    let mut columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(e1.finish()),
        Arc::new(e2.finish()),
        Arc::new(TimestampNanosecondArray::from(
            ts_secs
                .iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
    ];
    if has_target {
        fields.push(Field::new("TARGET1", DataType::Float64, true));
        columns.push(Arc::new(target.finish()));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize a parquet with one ENTITY1, TIMESTAMP, and two TARGETk columns.
fn synth_multi_target_parquet(
    out_path: &Path,
    e1_pks: &[i64],
    ts_secs: &[i64],
    targets1: &[f64],
    targets2: &[f64],
) {
    assert_eq!(e1_pks.len(), ts_secs.len());
    assert_eq!(e1_pks.len(), targets1.len());
    assert_eq!(e1_pks.len(), targets2.len());
    let mut e1 = Int64Builder::new();
    let mut t1 = Float64Builder::new();
    let mut t2 = Float64Builder::new();
    for i in 0..e1_pks.len() {
        e1.append_value(e1_pks[i]);
        t1.append_value(targets1[i]);
        t2.append_value(targets2[i]);
    }
    let fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        Field::new("TARGET2", DataType::Float64, true),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(e1.finish()),
        Arc::new(TimestampNanosecondArray::from(
            ts_secs
                .iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        )),
        Arc::new(t1.finish()),
        Arc::new(t2.finish()),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Read tt.parquet for a multi-entity Node output. Returns one tuple per row:
/// (ENTITY1, ENTITY2, TIMESTAMP_secs, TARGET1).
fn read_tt_parquet_multi_entity_node(tt_path: &Path) -> Vec<(i64, i64, i64, f64)> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(tt_path) {
        let schema = batch.schema();
        // Find columns by name (the executor controls slot order).
        let e1_idx = schema.index_of("ENTITY1").unwrap();
        let e2_idx = schema.index_of("ENTITY2").unwrap();
        let ts_idx = schema.index_of("TIMESTAMP").unwrap();
        let t1_idx = schema.index_of("TARGET1").unwrap();
        let e1 = batch
            .column(e1_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let e2 = batch
            .column(e2_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(ts_idx)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let t1 = batch
            .column(t1_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((
                e1.value(i),
                e2.value(i),
                ts.value(i) / 1_000_000_000,
                t1.value(i),
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
    out
}

/// Read tt.parquet for a multi-target Node output. Returns one tuple per row:
/// (ENTITY1, TIMESTAMP_secs, TARGET1, TARGET2).
fn read_tt_parquet_multi_target_node(tt_path: &Path) -> Vec<(i64, i64, f64, f64)> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(tt_path) {
        let schema = batch.schema();
        let e1_idx = schema.index_of("ENTITY1").unwrap();
        let ts_idx = schema.index_of("TIMESTAMP").unwrap();
        let t1_idx = schema.index_of("TARGET1").unwrap();
        let t2_idx = schema.index_of("TARGET2").unwrap();
        let e1 = batch
            .column(e1_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(ts_idx)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let t1 = batch
            .column(t1_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let t2 = batch
            .column(t2_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((
                e1.value(i),
                ts.value(i) / 1_000_000_000,
                t1.value(i),
                t2.value(i),
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// Phase 11: a custom-TT parquet with ENTITY1+ENTITY2+TIMESTAMP+TARGET1
/// flows through the executor, and the output tt.parquet contains both
/// ENTITY1 and ENTITY2 columns. ENTITY1 is used for the index lookup;
/// ENTITY2 is carried through as auxiliary data.
#[test]
fn test_custom_tt_multi_entity_reads_all_entities() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let e1_pks: Vec<i64> = vec![0, 1, 2, 3, 4];
    let e2_pks: Vec<i64> = vec![100, 200, 300, 400, 500];
    let ts_secs: Vec<i64> = vec![1_000_000; 5];
    synth_multi_entity_parquet(&pq, &e1_pks, &e2_pks, &ts_secs, true);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };

    // Inject a second entity slot via the customizer hook. The plan
    // compiler hasn't been updated yet (forecasting eng's domain) so we
    // patch the resolved plan directly to declare two entity slots.
    let customizer = |resolved: &mut ResolvedPlan<Seconds>| {
        // Slot 0 already populated by the resolver. Add a synthetic slot 1
        // referring to the same rowpack column (we don't read from rowpack
        // for slot 1 in the custom-TT path — ENTITY2 comes straight from
        // the parquet — but the field has to be non-empty so the writer
        // sizes its buffers correctly).
        resolved
            .entity_pk_col_indices
            .push(resolved.entity_pk_col_indices[0]);
        resolved
            .entity_dtypes
            .push(resolved.entity_dtypes[0].clone());
    };

    let output = execute_tt_pass_with_customizer(
        &plan,
        &index,
        out_dir.path(),
        Some(custom),
        Some(customizer),
    )
    .unwrap();
    assert_eq!(output.rows_written, 5);

    let rows = read_tt_parquet_multi_entity_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 5);
    for ((e1, e2, _, t), (exp_e1, exp_e2)) in rows.iter().zip(e1_pks.iter().zip(e2_pks.iter())) {
        assert_eq!(e1, exp_e1, "ENTITY1 should match parquet input");
        assert_eq!(e2, exp_e2, "ENTITY2 should match parquet input");
        let expected = det_target(*exp_e1, ts_secs[0]) + (*exp_e2 as f64);
        assert!(
            (t - expected).abs() < 1e-9,
            "TARGET1 unexpected: {t} vs {expected}"
        );
    }
}

/// Phase 11: a custom-TT parquet with two TARGETk columns produces a
/// tt.parquet with both TARGET1 and TARGET2 columns populated.
#[test]
fn test_custom_tt_multi_target_emits_all_targets() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let e1_pks: Vec<i64> = vec![0, 1, 2, 3, 4];
    let ts_secs: Vec<i64> = vec![1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000];
    let targets1: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let targets2: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    synth_multi_target_parquet(&pq, &e1_pks, &ts_secs, &targets1, &targets2);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };

    // Inject a second target slot. Slot 0 already exists from the resolver
    // (target_alias_indices = vec![0], target_dtypes = vec![Float64]); we
    // append slot 1 with Float64 dtype to mirror the TARGET2 column.
    let customizer = |resolved: &mut ResolvedPlan<Seconds>| {
        // The plan has empty stages so num_aliases == 0. We allocate slot 1
        // by appending a fresh alias index. The custom-TT executor reads
        // TARGET2 directly from the parquet (slot 1) so the alias index
        // value is unused — we just need the Vec lengths to agree.
        let next_idx = resolved.target_alias_indices.last().copied().unwrap_or(0) + 1;
        resolved.target_alias_indices.push(next_idx);
        resolved.target_dtypes.push(arrow_schema::DataType::Float64);
    };

    let output = execute_tt_pass_with_customizer(
        &plan,
        &index,
        out_dir.path(),
        Some(custom),
        Some(customizer),
    )
    .unwrap();
    assert_eq!(output.rows_written, 5);

    let rows = read_tt_parquet_multi_target_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 5);
    for (i, (e1, ts, t1, t2)) in rows.iter().enumerate() {
        assert_eq!(*e1, e1_pks[i]);
        assert_eq!(*ts, ts_secs[i]);
        assert!(
            (t1 - targets1[i]).abs() < 1e-9,
            "TARGET1 mismatch at row {i}"
        );
        assert!(
            (t2 - targets2[i]).abs() < 1e-9,
            "TARGET2 mismatch at row {i}"
        );
    }
}

/// Phase 11: behavior for unknown ENTITY1 in a multi-entity row. Documented
/// behavior: ENTITY1 is the index lookup key; when it doesn't resolve, the
/// row is silently skipped (matches single-entity unknown-PK behavior).
/// ENTITY2 is opaque to the executor — it's just carried through unchanged.
#[test]
fn test_custom_tt_multi_entity_unknown_pk_in_one_skipped() {
    let dir = tempfile::tempdir().unwrap();
    // Index has PKs {0..4}; parquet has ENTITY1 values 0..9 (5 unknown).
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("custom_tt.parquet");
    let e1_pks: Vec<i64> = (0..10).collect();
    let e2_pks: Vec<i64> = (100..110).collect();
    let ts_secs: Vec<i64> = vec![1_000_000; 10];
    synth_multi_entity_parquet(&pq, &e1_pks, &e2_pks, &ts_secs, true);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };

    let customizer = |resolved: &mut ResolvedPlan<Seconds>| {
        resolved
            .entity_pk_col_indices
            .push(resolved.entity_pk_col_indices[0]);
        resolved
            .entity_dtypes
            .push(resolved.entity_dtypes[0].clone());
    };

    let output = execute_tt_pass_with_customizer(
        &plan,
        &index,
        out_dir.path(),
        Some(custom),
        Some(customizer),
    )
    .unwrap();
    // Only ENTITY1 in {0..4} resolve; the other 5 rows are skipped. ENTITY2
    // is opaque and never affects the lookup decision.
    assert_eq!(
        output.rows_written, 5,
        "rows with unknown ENTITY1 must be skipped"
    );

    let rows = read_tt_parquet_multi_entity_node(&output.tt_parquet_path);
    assert_eq!(rows.len(), 5);
    let kept_e1: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let mut sorted = kept_e1.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    // Verify ENTITY2 came through correctly for kept rows.
    for (e1, e2, _, _) in &rows {
        assert_eq!(*e2, *e1 + 100, "ENTITY2 should match parquet for kept rows");
    }
}

/// Phase 11: when the custom-TT parquet is missing an ENTITYk column the
/// plan declared, the executor should error out (not silently skip).
#[test]
fn test_custom_tt_multi_entity_missing_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    // Single-entity parquet (ENTITY1 only), but we declare 2 entity slots.
    let pq = dir.path().join("custom_tt.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 5,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan();
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };

    let customizer = |resolved: &mut ResolvedPlan<Seconds>| {
        resolved
            .entity_pk_col_indices
            .push(resolved.entity_pk_col_indices[0]);
        resolved
            .entity_dtypes
            .push(resolved.entity_dtypes[0].clone());
    };

    let result = execute_tt_pass_with_customizer(
        &plan,
        &index,
        out_dir.path(),
        Some(custom),
        Some(customizer),
    );
    assert!(result.is_err(), "missing ENTITY2 should error");
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("ENTITY2"),
        "error should mention ENTITY2: {msg}"
    );
}

// ─── Weight column tests ─────────────────────────────────────────────────────
//
// Weight rides each row as a reserved `WEIGHT` (Float64) column into the
// per-stage seed parquet (no `weight.bin` sidecar). These tests verify:
// 1. weight_col=Some → `tt.parquet` carries a per-row `WEIGHT` column, manifest
//    references no weight sidecar.
// 2. weight_col=None → no `WEIGHT` column.
// 3. Latent-bug fix: under an interleaved split each stage's seed parquet
//    carries its OWN row's weight, not the global-position-k weight the old
//    sidecar mis-assigned (engine read it by stage-local row id).

/// Synthesize a customer parquet with ENTITY1 + TIMESTAMP + TARGET1 +
/// `weight` (Float64). Weight value = `(entity + 1) * 0.25`.
fn synth_custom_tt_parquet_with_weight(out_path: &Path, n_rows: usize, entity_pks: &[i64]) {
    let weights: Vec<f64> = (0..n_rows)
        .map(|i| ((entity_pks[i % entity_pks.len()] + 1) as f64) * 0.25)
        .collect();
    synth_custom_tt_parquet_with_explicit_weights(out_path, entity_pks, &weights);
}

/// Synthesize a customer parquet with ENTITY1 + TIMESTAMP + TARGET1 +
/// `weight` (Float64) where each row's weight is supplied explicitly.
/// `weights.len()` is the row count; entity PKs cycle through `entity_pks`.
fn synth_custom_tt_parquet_with_explicit_weights(
    out_path: &Path,
    entity_pks: &[i64],
    weights: &[f64],
) {
    let n_rows = weights.len();
    let mut entity = Int64Builder::new();
    let mut target = Float64Builder::new();
    let mut weight = Float64Builder::new();
    let anchor_ts_secs: Vec<i64> = vec![1_000_000, 2_000_000, 3_000_000];
    let mut ts_values = Vec::with_capacity(n_rows);
    for (i, &w) in weights.iter().enumerate() {
        let e = entity_pks[i % entity_pks.len()];
        let t_secs = anchor_ts_secs[i % anchor_ts_secs.len()];
        entity.append_value(e);
        ts_values.push(t_secs * 1_000_000_000);
        target.append_value(det_target(e, t_secs));
        weight.append_value(w);
    }
    let fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        Field::new("weight", DataType::Float64, true),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(ts_values)),
        Arc::new(target.finish()),
        Arc::new(weight.finish()),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize a customer parquet with ENTITY1 + TIMESTAMP + TARGET1 +
/// `weight` (Float64) + `split_col` (Int64, 0/1/2). One row per supplied
/// weight; `split_values[i]` is the stage of row `i`.
fn synth_custom_tt_parquet_with_weights_and_split(
    out_path: &Path,
    entity_pks: &[i64],
    weights: &[f64],
    split_values: &[i64],
) {
    let n_rows = weights.len();
    assert_eq!(n_rows, split_values.len());
    let mut entity = Int64Builder::new();
    let mut target = Float64Builder::new();
    let mut weight = Float64Builder::new();
    let mut split = Int64Builder::new();
    let anchor_ts_secs: Vec<i64> = vec![1_000_000, 2_000_000, 3_000_000];
    let mut ts_values = Vec::with_capacity(n_rows);
    for (i, &w) in weights.iter().enumerate() {
        let e = entity_pks[i % entity_pks.len()];
        let t_secs = anchor_ts_secs[i % anchor_ts_secs.len()];
        entity.append_value(e);
        ts_values.push(t_secs * 1_000_000_000);
        target.append_value(det_target(e, t_secs));
        weight.append_value(w);
        split.append_value(split_values[i]);
    }
    let fields: Vec<Field> = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        Field::new("weight", DataType::Float64, true),
        Field::new("split_col", DataType::Int64, true),
    ];
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(entity.finish()),
        Arc::new(TimestampNanosecondArray::from(ts_values)),
        Arc::new(target.finish()),
        Arc::new(weight.finish()),
        Arc::new(split.finish()),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    write_parquet(out_path, &batch);
}

/// Read the `WEIGHT` (Float64) column off a parquet file in row order.
/// Panics if the column is absent.
fn read_weight_column(path: &Path) -> Vec<f64> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(path) {
        let idx = batch
            .schema()
            .index_of("WEIGHT")
            .expect("parquet must carry a WEIGHT column");
        let col = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("WEIGHT column must be Float64");
        for i in 0..col.len() {
            out.push(col.value(i));
        }
    }
    out
}

/// Whether a parquet file's schema declares a `WEIGHT` column.
fn parquet_has_weight_column(path: &Path) -> bool {
    read_parquet_batches(path)
        .first()
        .expect("parquet input has a batch")
        .schema()
        .index_of("WEIGHT")
        .is_ok()
}

#[test]
fn test_weight_column_emitted_when_weight_col_set() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("custom_tt_with_weight.parquet");
    let entity_pks: Vec<i64> = (0..5).collect();
    synth_custom_tt_parquet_with_weight(&pq, 12, &entity_pks);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.weight_col = Some("weight".to_string());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 12);

    // No weight.bin sidecar — weight rides as a column now.
    assert!(
        !out_dir.path().join("weight.bin").exists(),
        "weight.bin must NOT be emitted (weight is a column now)"
    );

    // tt.parquet carries a WEIGHT column with the per-row payload.
    let weights = read_weight_column(&output.tt_parquet_path);
    assert_eq!(weights.len(), 12);
    let expected: Vec<f64> = (0..12)
        .map(|i| ((entity_pks[i % entity_pks.len()] + 1) as f64) * 0.25)
        .collect();
    assert_eq!(weights, expected, "WEIGHT column must match input order");

    // Manifest no longer references a weight sidecar.
    let manifest = diskgraph_train_table::manifest::read_manifest(&output.manifest_path).unwrap();
    assert!(
        manifest.weight_sidecar.is_none(),
        "manifest.weight_sidecar must be None (no sidecar)"
    );
    assert!(
        manifest.checksums.weight_sidecar_sha256.is_none(),
        "checksums.weight_sidecar_sha256 must be None (no sidecar)"
    );
}

#[test]
fn test_weight_column_absent_when_weight_col_unset() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("custom_tt_no_weight_field.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 8,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan(); // weight_col = None
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 8);

    assert!(
        !out_dir.path().join("weight.bin").exists(),
        "weight.bin must NOT exist when plan.weight_col is None"
    );
    assert!(
        !parquet_has_weight_column(&output.tt_parquet_path),
        "tt.parquet must NOT carry a WEIGHT column when plan.weight_col is None"
    );

    let manifest = diskgraph_train_table::manifest::read_manifest(&output.manifest_path).unwrap();
    assert!(
        manifest.weight_sidecar.is_none(),
        "manifest.weight_sidecar must be None when plan.weight_col is None"
    );
    assert!(
        manifest.checksums.weight_sidecar_sha256.is_none(),
        "checksums.weight_sidecar_sha256 must be None when plan.weight_col is None"
    );
}

#[test]
fn test_weight_column_skips_orphan_rows() {
    // Verify the WEIGHT column row count equals the writer's emitted rows,
    // NOT the customer parquet's input row count. Customer parquet has 10
    // rows with entity PKs 0..10; index only has PKs 0..5. Half the rows
    // should be skipped — the WEIGHT column has 5 entries (one per emitted
    // row), filtered automatically with the row.
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5); // PKs 0..5
    let pq = dir.path().join("custom_tt_with_orphans.parquet");
    let entity_pks: Vec<i64> = (0..10).collect(); // 5 known + 5 orphans
    synth_custom_tt_parquet_with_weight(&pq, 10, &entity_pks);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.weight_col = Some("weight".to_string());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 5, "5 orphan rows must be skipped");

    let weights = read_weight_column(&output.tt_parquet_path);
    assert_eq!(
        weights.len(),
        5,
        "WEIGHT column must have one value per emitted row, not input row"
    );
    // Surviving rows are entity PKs 0..5 → weights (e+1)*0.25.
    let expected: Vec<f64> = (0..5).map(|e| ((e + 1) as f64) * 0.25).collect();
    assert_eq!(weights, expected);
}

/// LATENT-BUG REGRESSION: under an interleaved split each stage's seed parquet
/// must carry its OWN row's weight, not the global-position-k weight.
///
/// The old sidecar stored weights in global order but the engine read them by
/// stage-LOCAL row id, so train-local row 1 (global row 3) read global weight 1.
/// The column-based fix carries the weight with the row into the stage part.
/// Now driven by a `ManualSplit` on a customer-provided `split_col` (the
/// split.bin path is gone), exercising the same interleaved routing.
#[test]
fn test_weight_column_correct_under_interleaved_split() {
    let dir = tempfile::tempdir().unwrap();
    // 9 rows, entity PKs cycling 0..3 (all in-graph), distinct per-row
    // weights = global index as f64 so any mis-assignment is detectable.
    let n_rows = 9usize;
    let index = build_users_only_graph(dir.path(), 3);
    let pq = dir.path().join("custom_tt_interleaved.parquet");
    let entity_pks: Vec<i64> = (0..3).collect();
    let row_weights: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
    // Interleaved split: row i → stage i % 3 (0=train,1=val,2=test).
    // train = global rows {0,3,6} (weights 0,3,6)
    // val   = global rows {1,4,7} (weights 1,4,7)
    // test  = global rows {2,5,8} (weights 2,5,8)
    let split_values: Vec<i64> = (0..n_rows as i64).map(|i| i % 3).collect();
    synth_custom_tt_parquet_with_weights_and_split(&pq, &entity_pks, &row_weights, &split_values);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.weight_col = Some("weight".to_string());
    plan.split = Some(manual_split_col_spec());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, n_rows as u64);

    let infos = output
        .stage_gen_infos
        .as_ref()
        .expect("interleaved split must produce stage_gen_infos");

    // Expected per-stage weights (= the global-index weights of the rows
    // routed to that stage).
    let expected = |stage: usize| -> Vec<f64> {
        (0..n_rows)
            .filter(|i| i % 3 == stage)
            .map(|i| i as f64)
            .collect()
    };
    for (stage, key) in ["train", "val", "test"].iter().enumerate() {
        let info = infos.iter().find(|s| &s.key == key).unwrap();
        // Collect the stage's WEIGHT column across all its parts (single
        // part here; iterate for robustness).
        let mut stage_weights = Vec::new();
        for part in &info.parts {
            stage_weights.extend(read_weight_column(&out_dir.path().join(part)));
        }
        assert_eq!(
            stage_weights,
            expected(stage),
            "stage `{key}` must carry its own rows' weights, not global-position-k weights"
        );
    }
}

// ─── Phase A: custom-TT LP RHS-target orphan filter ──────────────────────────
//
// For a custom LP table, TARGET1 holds an RHS node id read straight from the
// customer parquet. When the plan declares `target_entity_table` (via the
// fkey_mapping TARGET1 entry), rows whose TARGET1 is not a member of that
// table's PK map must be dropped — mirroring the PQL ListDistinct RHS-target
// orphan filter (`ttgen_executor.rs:2836-2849`). Otherwise orphan RHS ids
// silently mis-resolve at ingest (the external-parent MPH does not verify).

#[test]
fn test_custom_tt_lp_drops_orphan_target_rows() {
    let dir = tempfile::tempdir().unwrap();
    // users PKs 0..5, products PKs 0..3 (valid product_ids: 0,1,2).
    let index = build_users_and_products_graph(dir.path(), 5, 3);

    let pq = dir.path().join("custom_tt_lp.parquet");
    // 5 rows: TARGET1 values {0,1,2} are valid; {99,100} are orphans.
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4];
    let ts_secs: Vec<i64> = vec![1_000_000; 5];
    let target_product_ids: Vec<i64> = vec![0, 1, 2, 99, 100];
    synth_custom_tt_lp_parquet(&pq, &entity_pks, &ts_secs, &target_product_ids);

    let plan = make_custom_tt_lp_node_plan(Some("products"));
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    // Only the 3 valid-target rows survive; the 2 orphan-target rows drop.
    assert_eq!(output.rows_written, 3, "orphan-target rows must be dropped");
    let rows = read_tt_parquet_lp_targets(&output.tt_parquet_path);
    let targets: Vec<i64> = rows.iter().map(|r| r.1).collect();
    assert_eq!(
        targets,
        vec![0, 1, 2],
        "only valid product_ids should remain; got {targets:?}"
    );
}

#[test]
fn test_custom_tt_lp_keeps_all_targets_without_target_entity_table() {
    // Control: when no target_entity_table is declared, the orphan filter is
    // disabled and every row (including orphan targets) is kept.
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 5, 3);

    let pq = dir.path().join("custom_tt_lp_no_filter.parquet");
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4];
    let ts_secs: Vec<i64> = vec![1_000_000; 5];
    let target_product_ids: Vec<i64> = vec![0, 1, 2, 99, 100];
    synth_custom_tt_lp_parquet(&pq, &entity_pks, &ts_secs, &target_product_ids);

    let plan = make_custom_tt_lp_node_plan(None);
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    assert_eq!(output.rows_written, 5, "no filter → all rows kept");
    let rows = read_tt_parquet_lp_targets(&output.tt_parquet_path);
    let mut targets: Vec<i64> = rows.iter().map(|r| r.1).collect();
    targets.sort();
    assert_eq!(targets, vec![0, 1, 2, 99, 100]);
}

// ─── Phase A2: entity orphan filter in PassThrough mode ──────────────────────
//
// Verify the ENTITY orphan skip (`lookup_entity_node` → continue) fires for
// `PredictionType::PassThrough`, matching Node/Forecasting behavior. A custom
// PassThrough source with orphan ENTITY1 values must drop those rows.

#[test]
fn test_custom_tt_passthrough_skips_orphan_entity() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5); // PKs 0..5

    let pq = dir.path().join("custom_tt_passthrough.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 10,
        // Half the entity_pks are unknown (5..10).
        entity_pks: (0..10).collect(),
        anchor_ts_secs: vec![1_000_000],
        has_target: false,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.prediction_type = PredictionType::PassThrough;

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    // Only entity_pks {0..4} resolve → 5 rows kept; orphans skipped.
    assert_eq!(
        output.rows_written, 5,
        "PassThrough must skip orphan ENTITY rows"
    );
}

// ─── Phase B: custom path ingests a samplable extension via the seam ─────────
//
// After `execute_tt_pass_with_source` produces tt.parquet (+ per-stage seed
// parquets for split plans), the custom path must ingest a DG extension (mirroring
// the PQL `execute_tt_plan` seam) so the materialized seeds are samplable. We
// open the published extension via `open_with_extension_subdir` and assert the
// ENTITY (and TARGET, for LP) FKs resolve into `seed.edges.bin`.

/// Collect every FK dst_id (across all relation segments) for every seed row
/// of a stage table under a namespaced extension. Returns the flat list of
/// resolved parent global node ids.
fn collect_stage_dst_ids(
    reader: &diskgraph_index::graph::GraphIndexReader,
    stage_table: &str,
) -> Vec<u64> {
    let seed = reader
        .table(stage_table)
        .unwrap_or_else(|| panic!("stage table `{stage_table}` missing"));
    let mut dst_ids = Vec::new();
    for seed_local in 0..seed.node_count() {
        let view = seed.edges_for_node(seed_local).unwrap();
        for seg in view.relation_segments().unwrap() {
            if seg.count == 0 {
                continue;
            }
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            // Each entry is `entry_bytes` wide; iterate all entries in the seg.
            for e in 0..seg.count as usize {
                let start = e * entry_bytes;
                let mut buf = [0u8; 16];
                buf[..entry_bytes].copy_from_slice(&seg.data[start..start + entry_bytes]);
                dst_ids.push((u128::from_le_bytes(buf) & dst_mask) as u64);
            }
        }
    }
    dst_ids
}

fn read_stage_node_json(
    reader: &diskgraph_index::graph::GraphIndexReader,
    stage_table: &str,
    node_id: u64,
) -> serde_json::Value {
    let table = reader
        .table(stage_table)
        .unwrap_or_else(|| panic!("stage table `{stage_table}` missing"));
    let view = table.edges_for_node(node_id).unwrap();
    table
        .decode_node_data_json(&view)
        .unwrap()
        .unwrap_or_else(|| panic!("stage table `{stage_table}` node {node_id} has no rowpack"))
}

#[test]
fn test_custom_tt_lp_ingests_samplable_extension() {
    use diskgraph_train_table::tt_stage_ingest::ingest_and_publish_tt_extension;

    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 8, 8);

    let pq = dir.path().join("custom_tt_lp_ingest.parquet");
    // All targets valid (0..4) so every row survives the orphan filter and
    // becomes a samplable seed.
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4, 5];
    let ts_secs: Vec<i64> = vec![1_000_000; 6];
    let target_product_ids: Vec<Vec<i64>> =
        vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]];
    synth_custom_tt_lp_array_parquet(&pq, &entity_pks, &ts_secs, &target_product_ids);

    // Random split so the pass emits per-stage seed parquets (train/val/test).
    let mut plan = make_custom_tt_lp_node_plan(Some("products"));
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    plan.lp_resolution = Some(vec![
        RefResolveSpec {
            src_col: "ENTITY1".to_string(),
            dst_table: "users".to_string(),
            kind: RefResolveKind::Scalar,
            node_id_col: "entity_node_id".to_string(),
        },
        RefResolveSpec {
            src_col: "TARGET1".to_string(),
            dst_table: "products".to_string(),
            kind: RefResolveKind::List,
            node_id_col: "target_node_ids".to_string(),
        },
    ]);
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: Some(vec![0.5, 0.25, 0.25]),
        kind: Some("RandomSplit".to_string()),
        rnd_seed: Some(42),
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: None,
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    });

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let pass = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(pass.rows_written, 6);
    assert!(
        pass.stage_gen_infos.is_some(),
        "split plan must produce per-stage seed parquets (stage_gen_infos)"
    );

    // Ingest + publish under train_tables/<id>.
    let tt_id = "custom_lp_tt";
    let tt_root = index.join("train_tables").join(tt_id);
    let staging = out_dir.path().join(".staging");
    let ingest_root = out_dir.path().join(".ingested");
    std::fs::create_dir_all(&staging).unwrap();
    let res = ingest_and_publish_tt_extension(
        &format!("tt_seeds_{tt_id}"),
        &pass,
        &plan,
        &index,
        &staging,
        &ingest_root,
        &tt_root,
    )
    .unwrap();
    assert_eq!(res.extension_path, tt_root);
    let total: u64 = res.stage_counts.iter().map(|(_, n)| *n).sum();
    assert_eq!(total, 6, "all 6 seeds ingested across train/val/test");

    // Open the published extension and assert LP resolution-mode writes the
    // resolved ids into rowpack columns, not seed FK edges.
    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        &index,
        &format!("train_tables/{tt_id}"),
        tt_id,
    )
    .unwrap();

    let mut resolved_entities = Vec::new();
    let mut resolved_targets = Vec::new();
    for stage in ["train_seeds", "val_seeds", "test_seeds"] {
        let stage_table = format!("{tt_id}/{stage}");
        assert!(
            collect_stage_dst_ids(&reader, &stage_table).is_empty(),
            "resolution-mode stage `{stage_table}` must not emit seed FK edges"
        );
        let table = reader
            .table(&stage_table)
            .unwrap_or_else(|| panic!("stage table `{stage_table}` missing"));
        for node_id in 0..table.node_count() {
            let json = read_stage_node_json(&reader, &stage_table, node_id);
            let entity = json
                .get("entity_node_id")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("entity_node_id missing/not u64 in {json}"));
            resolved_entities.push(entity as i64);
            let targets = json
                .get("target_node_ids")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("target_node_ids missing/not array in {json}"));
            for target in targets {
                resolved_targets.push(
                    target
                        .as_i64()
                        .unwrap_or_else(|| panic!("target_node_ids element not i64 in {json}")),
                );
            }
        }
    }
    resolved_entities.sort();
    resolved_targets.sort();
    assert_eq!(resolved_entities, entity_pks);
    assert_eq!(resolved_targets, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_custom_tt_pred_passthrough_ingests_predict_stage() {
    use diskgraph_train_table::tt_stage_ingest::ingest_and_publish_tt_extension;

    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 8);

    let pq = dir.path().join("custom_tt_pred.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 6,
        entity_pks: (0..6).collect(),
        anchor_ts_secs: vec![1_000_000],
        has_target: false,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    // Pred-shaped: PassThrough, no target, no split.
    let mut plan = make_custom_tt_target_only_node_plan();
    plan.prediction_type = PredictionType::PassThrough;
    plan.is_prediction = true;
    plan.split = None;
    plan.fkey_mapping = Some(vec![TTFKeySpec {
        src_col: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
    }]);

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let pass = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(pass.rows_written, 6);
    assert!(
        pass.stage_gen_infos.is_none(),
        "pred plan (no split) must NOT produce stage_gen_infos"
    );

    let tt_id = "custom_pred_tt";
    let tt_root = index.join("pred_tables").join(tt_id);
    let staging = out_dir.path().join(".staging");
    let ingest_root = out_dir.path().join(".ingested");
    std::fs::create_dir_all(&staging).unwrap();
    let res = ingest_and_publish_tt_extension(
        &format!("pred_seeds_{tt_id}"),
        &pass,
        &plan,
        &index,
        &staging,
        &ingest_root,
        &tt_root,
    )
    .unwrap();

    // Exactly one `predict` stage, no split stages.
    assert_eq!(res.stage_counts.len(), 1);
    assert_eq!(res.stage_counts[0].0, "predict");
    assert_eq!(res.stage_counts[0].1, 6);

    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        &index,
        &format!("pred_tables/{tt_id}"),
        tt_id,
    )
    .unwrap();
    // Resolution mode: the predict stage emits NO seed FK edges — each ENTITY1
    // is resolved into the `entity_node_id` rowpack column instead.
    let stage_table = format!("{tt_id}/predict");
    assert!(
        collect_stage_dst_ids(&reader, &stage_table).is_empty(),
        "resolution-mode predict stage must not emit seed FK edges"
    );
    let predict = reader.table(&stage_table).unwrap();
    let mut dst = Vec::new();
    for node_id in 0..predict.node_count() {
        let json = read_stage_node_json(&reader, &stage_table, node_id);
        dst.push(
            json.get("entity_node_id")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("entity_node_id missing/not u64 in {json}")),
        );
    }
    assert_eq!(
        dst.len(),
        6,
        "6 predict seeds, each ENTITY1 → entity_node_id"
    );
    let n_users = reader.table("users").unwrap().node_count();
    assert!(
        dst.iter().all(|&d| d < n_users),
        "every predict-stage ENTITY resolves to a valid user node; got {dst:?}"
    );
}

// ─── Predict bundles carry NO target column (node-pred + LP) ─────────────────
//
// Contract: a PREDICTION bundle (is_prediction=true) carries NO target column
// for EVERY task type — the tt.parquet is `[ENTITY1, TIMESTAMP]` (+ features),
// matching the native predict bundle. Train must still emit TARGET/TARGET1.

/// NODE-PRED PREDICT: with a real TIMESTAMP present, the predict bundle emits
/// `[ENTITY1, TIMESTAMP]` and NO TARGET1 (previously emitted an all-NaN
/// TARGET1, which broke the categorical-target encoder downstream).
#[test]
fn test_custom_tt_node_predict_emits_no_target_column() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    // Predict seed table: ENTITY1 + TIMESTAMP, NO TARGET1.
    let pq = dir.path().join("node_predict.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 6,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000],
        has_target: false,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let mut plan = make_custom_tt_target_only_node_plan();
    plan.is_prediction = true;
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 6);

    let (cols, n) = read_tt_parquet_columns(&output.tt_parquet_path);
    assert_eq!(n, 6);
    assert_eq!(cols, vec!["ENTITY1".to_string(), "TIMESTAMP".to_string()]);
    assert!(
        cols.iter().all(|c| !c.starts_with("TARGET")),
        "node predict bundle must carry no TARGET column; got {cols:?}"
    );
}

/// NODE-PRED TRAIN: control — a train bundle still emits TARGET1.
#[test]
fn test_custom_tt_node_train_still_emits_target_column() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    let pq = dir.path().join("node_train.parquet");
    let spec = CustomTtFixtureSpec {
        n_rows: 6,
        entity_pks: (0..5).collect(),
        anchor_ts_secs: vec![1_000_000, 2_000_000],
        has_target: true,
        has_autoreg: false,
        has_yoy: false,
    };
    synth_custom_tt_parquet(&pq, &spec);

    let plan = make_custom_tt_target_only_node_plan(); // is_prediction = false
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 6);

    let (cols, _n) = read_tt_parquet_columns(&output.tt_parquet_path);
    assert!(
        cols.contains(&"TARGET1".to_string()),
        "node train bundle must still carry TARGET1; got {cols:?}"
    );
}

/// LP (LinkArray) PREDICT: the seed table has no TARGET1 list column; the
/// predict bundle emits `[ENTITY1, TIMESTAMP]` — no TARGET1 and no frequency
/// column.
#[test]
fn test_custom_tt_lp_predict_emits_no_target_column() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 5, 3);

    // Predict LP seed table: ENTITY1 + TIMESTAMP only (no TARGET1 list).
    let pq = dir.path().join("lp_predict.parquet");
    synth_static_custom_tt_parquet_with_ts(&pq, &[0, 1, 2, 3, 4], 1_000_000);

    let mut plan = make_custom_tt_lp_node_plan(Some("products"));
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    plan.is_prediction = true;
    // Predict LP resolution: ENTITY1 only (mirrors the pybind predict entry).
    plan.lp_resolution = Some(vec![RefResolveSpec {
        src_col: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
        kind: RefResolveKind::Scalar,
        node_id_col: "entity_node_id".to_string(),
    }]);
    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 5);

    let (cols, n) = read_tt_parquet_columns(&output.tt_parquet_path);
    assert_eq!(n, 5);
    assert_eq!(cols, vec!["ENTITY1".to_string(), "TIMESTAMP".to_string()]);
    assert!(
        cols.iter()
            .all(|c| !c.starts_with("TARGET") && c != "__kumo_internal_frequency_column__"),
        "LP predict bundle must carry no TARGET/frequency column; got {cols:?}"
    );
}

fn read_weight_lists(path: &Path) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for batch in read_parquet_batches(path) {
        let idx = batch.schema().index_of("WEIGHT").expect("WEIGHT column");
        let col = batch
            .column(idx)
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .expect("WEIGHT must be a ListArray beside an array target");
        for i in 0..col.len() {
            let cell = col.value(i);
            let vals = cell.as_any().downcast_ref::<Float64Array>().unwrap();
            out.push((0..vals.len()).map(|j| vals.value(j)).collect());
        }
    }
    out
}

#[test]
fn test_lp_weight_column_is_per_element() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 8, 8);

    let pq = dir.path().join("custom_tt_lp_weighted.parquet");
    let entity_pks: Vec<i64> = vec![0, 1, 2];
    let ts_secs: Vec<i64> = vec![1_000_000; 3];
    let targets: Vec<Vec<i64>> = vec![vec![0, 1, 2], vec![3], vec![4, 5]];
    let weights: Vec<Vec<f64>> = vec![vec![0.5, 1.5, 2.5], vec![3.5], vec![4.5, 5.5]];
    synth_custom_tt_lp_array_parquet_inner(&pq, &entity_pks, &ts_secs, &targets, Some(&weights));

    let mut plan = make_custom_tt_lp_node_plan(None);
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    plan.weight_col = Some("WEIGHT".to_string());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let pass = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(pass.rows_written, 3);

    let tt = out_dir.path().join("tt.parquet");
    assert_eq!(read_weight_lists(&tt), weights);
}

#[test]
fn test_lp_weight_length_mismatch_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 8, 8);

    let pq = dir.path().join("custom_tt_lp_bad_weight.parquet");
    let targets: Vec<Vec<i64>> = vec![vec![0, 1, 2]];
    let weights: Vec<Vec<f64>> = vec![vec![1.0]];
    synth_custom_tt_lp_array_parquet_inner(&pq, &[0], &[1_000_000], &targets, Some(&weights));

    let mut plan = make_custom_tt_lp_node_plan(None);
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    plan.weight_col = Some("WEIGHT".to_string());

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let err = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom))
        .expect_err("a weight list shorter than its target must not be accepted");
    assert!(
        format!("{err}").contains("weight list length differs"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_lp_weights_follow_their_link_across_a_split() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 8, 8);

    // weight == target * 10, so any link that loses its weight is visible.
    let pq = dir.path().join("custom_tt_lp_split_weights.parquet");
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3];
    let ts_secs: Vec<i64> = vec![1_000_000; 4];
    let targets: Vec<Vec<i64>> = vec![
        vec![0, 1, 2, 3],
        vec![4, 5, 6],
        vec![7, 0, 1],
        vec![2, 3, 4, 5],
    ];
    let weights: Vec<Vec<f64>> = targets
        .iter()
        .map(|row| row.iter().map(|t| *t as f64 * 10.0).collect())
        .collect();
    synth_custom_tt_lp_array_parquet_inner(&pq, &entity_pks, &ts_secs, &targets, Some(&weights));

    let mut plan = make_custom_tt_lp_node_plan(None);
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    plan.weight_col = Some("WEIGHT".to_string());
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: Some(vec![0.5, 0.25, 0.25]),
        kind: Some("RandomLinkSplit".to_string()),
        rnd_seed: Some(7),
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: None,
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    });

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error: diskgraph_train_table::custom_tt_row_errors::IngestOnError::Abort,
        allow_negative_weight: false,
        timestamp_reference: None,
    };
    let pass = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    let infos = pass
        .stage_gen_infos
        .expect("a link split must emit per-stage seed parquets");

    let mut seen = 0usize;
    for info in &infos {
        for part in &info.parts {
            let path = out_dir.path().join(part);
            let stage_targets = read_target_list_column(&path);
            let stage_weights = read_weight_lists(&path);
            assert_eq!(stage_targets.len(), stage_weights.len());
            for (ts, ws) in stage_targets.iter().zip(stage_weights.iter()) {
                assert_eq!(ts.len(), ws.len(), "stage {} lost a weight", info.key);
                for (target, weight) in ts.iter().zip(ws.iter()) {
                    assert_eq!(
                        *weight,
                        *target as f64 * 10.0,
                        "stage {} paired target {target} with weight {weight}",
                        info.key
                    );
                    seen += 1;
                }
            }
        }
    }
    assert_eq!(seen, targets.iter().map(|r| r.len()).sum::<usize>());
}

fn read_target_list_column(path: &Path) -> Vec<Vec<i64>> {
    let file = fs::File::open(path).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let idx = batch.schema().index_of("TARGET1").unwrap();
        let col = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for i in 0..col.len() {
            let cell = col.value(i);
            let vals = cell.as_any().downcast_ref::<Int64Array>().unwrap();
            out.push((0..vals.len()).map(|j| vals.value(j)).collect());
        }
    }
    out
}

// ─── Row validation: `ingest_on_error` ───────────────────────────────────────
//
// Every row-level condition the reader checks, in both modes: `Abort` names
// the condition, column and source row; `SkipRowOnError` drops the row and
// counts it in `row_failures`.

use diskgraph_train_table::custom_tt_row_errors::{IngestOnError, RowFailureKind};
use diskgraph_train_table::errors::TTGenResult;
use diskgraph_train_table::ttgen_executor::TTPassOutput;

/// Five rows of a Node custom TT; `bad` rewrites row 2 into the case under test.
struct NodeRows {
    entity: Vec<Option<i64>>,
    ts_secs: Vec<Option<i64>>,
    target: Vec<Option<f64>>,
    weight: Vec<Option<f64>>,
    split: Vec<Option<i64>>,
}

impl NodeRows {
    fn good() -> Self {
        Self {
            entity: (0..5).map(Some).collect(),
            ts_secs: vec![Some(1_000_000); 5],
            target: (0..5).map(|e| Some(e as f64)).collect(),
            weight: vec![Some(1.0); 5],
            split: vec![Some(0), Some(1), Some(2), Some(0), Some(1)],
        }
    }

    fn write(&self, path: &Path, weight: bool, split: bool) {
        let mut fields = vec![
            Field::new("ENTITY1", DataType::Int64, true),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new("TARGET1", DataType::Float64, true),
        ];
        let mut columns: Vec<Arc<dyn Array>> = vec![
            Arc::new(Int64Array::from(self.entity.clone())),
            Arc::new(TimestampNanosecondArray::from(
                self.ts_secs
                    .iter()
                    .map(|t| t.map(|s| s * 1_000_000_000))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(self.target.clone())),
        ];
        if weight {
            fields.push(Field::new("weight", DataType::Float64, true));
            columns.push(Arc::new(Float64Array::from(self.weight.clone())));
        }
        if split {
            fields.push(Field::new("split_col", DataType::Int64, true));
            columns.push(Arc::new(Int64Array::from(self.split.clone())));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        write_parquet(path, &batch);
    }
}

fn source(pq: &Path, on_error: IngestOnError) -> CustomTtSource {
    CustomTtSource {
        parquet_path: pq.to_path_buf(),
        input_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        on_error,
        allow_negative_weight: false,
        timestamp_reference: None,
    }
}

fn run_node_rows(
    rows: &NodeRows,
    weight: bool,
    split: bool,
    on_error: IngestOnError,
) -> TTGenResult<TTPassOutput> {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("rows.parquet");
    rows.write(&pq, weight, split);
    let mut plan = make_custom_tt_target_only_node_plan();
    if weight {
        plan.weight_col = Some("weight".to_string());
    }
    if split {
        plan.split = Some(manual_split_col_spec());
    }
    // Kept alive (and left behind in the OS temp dir) so callers can read the output.
    let out_dir = tempfile::tempdir().unwrap().keep();
    execute_tt_pass_with_source(&plan, &index, &out_dir, Some(source(&pq, on_error)))
}

/// One bad row (index 2) per case: (name, condition, column, weight?, split?, edit).
#[allow(clippy::type_complexity)]
fn node_cases() -> Vec<(
    &'static str,
    RowFailureKind,
    &'static str,
    bool,
    bool,
    fn(&mut NodeRows),
)> {
    vec![
        (
            "null entity",
            RowFailureKind::NullEntity,
            "ENTITY1",
            false,
            false,
            |r| r.entity[2] = None,
        ),
        (
            "null target",
            RowFailureKind::NullTarget,
            "TARGET1",
            false,
            false,
            |r| r.target[2] = None,
        ),
        (
            "null anchor",
            RowFailureKind::NullTimestamp,
            "TIMESTAMP",
            false,
            false,
            |r| r.ts_secs[2] = None,
        ),
        (
            "null weight",
            RowFailureKind::NullWeight,
            "weight",
            true,
            false,
            |r| r.weight[2] = None,
        ),
        (
            "nan weight",
            RowFailureKind::NonFiniteWeight,
            "weight",
            true,
            false,
            |r| r.weight[2] = Some(f64::NAN),
        ),
        (
            "inf weight",
            RowFailureKind::NonFiniteWeight,
            "weight",
            true,
            false,
            |r| r.weight[2] = Some(f64::INFINITY),
        ),
        (
            "negative weight",
            RowFailureKind::NegativeWeight,
            "weight",
            true,
            false,
            |r| r.weight[2] = Some(-1.0),
        ),
        (
            "null split",
            RowFailureKind::NullSplit,
            "split_col",
            false,
            true,
            |r| r.split[2] = None,
        ),
        (
            "unknown split value",
            RowFailureKind::InvalidSplit,
            "split_col",
            false,
            true,
            |r| r.split[2] = Some(7),
        ),
    ]
}

fn kind_name(kind: RowFailureKind) -> String {
    serde_json::to_value(kind)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn test_row_validation_abort_names_condition_column_and_row() {
    for (name, kind, column, weight, split, edit) in node_cases() {
        let mut rows = NodeRows::good();
        edit(&mut rows);
        let err = run_node_rows(&rows, weight, split, IngestOnError::Abort)
            .err()
            .unwrap_or_else(|| panic!("{name}: abort must fail the run"))
            .to_string();
        assert!(err.contains(kind.describe()), "{name}: {err}");
        assert!(err.contains(column), "{name}: {err}");
        assert!(err.contains("row 2"), "{name}: {err}");
    }
}

#[test]
fn test_row_validation_skip_drops_and_counts_the_row() {
    for (name, kind, column, weight, split, edit) in node_cases() {
        let mut rows = NodeRows::good();
        edit(&mut rows);
        let output = run_node_rows(&rows, weight, split, IngestOnError::SkipRowOnError)
            .unwrap_or_else(|e| panic!("{name}: skip must not fail: {e}"));
        assert_eq!(output.rows_written, 4, "{name}");
        let report = output.row_failures.expect("custom-TT reports row_failures");
        assert_eq!(report.on_error, "skip_row_on_error", "{name}");
        assert_eq!((report.rows_read, report.skipped_rows), (5, 1), "{name}");
        assert_eq!(report.by_condition.len(), 1, "{name}");
        let entry = &report.by_condition[0];
        assert_eq!(entry.condition, kind, "{name}");
        assert_eq!((entry.column.as_str(), entry.count), (column, 1), "{name}");
        assert!(
            entry.samples[0].contains("row 2"),
            "{name}: {:?}",
            entry.samples
        );
        assert_eq!(kind_name(kind), kind_name(entry.condition), "{name}");
    }
}

#[test]
fn test_row_validation_clean_source_reports_zero_skips_in_both_modes() {
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let output = run_node_rows(&NodeRows::good(), true, true, mode).unwrap();
        assert_eq!(output.rows_written, 5);
        let report = output.row_failures.unwrap();
        assert_eq!((report.rows_read, report.skipped_rows), (5, 0));
        assert!(report.by_condition.is_empty());
    }
}

#[test]
fn test_row_validation_skipped_rows_keep_weights_aligned() {
    let mut rows = NodeRows::good();
    rows.weight = (0..5).map(|e| Some(e as f64 + 1.0)).collect();
    rows.entity[1] = None;
    let output = run_node_rows(&rows, true, false, IngestOnError::SkipRowOnError).unwrap();
    assert_eq!(
        read_weight_column(&output.tt_parquet_path),
        vec![1.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn test_row_validation_manifest_records_row_failures() {
    let mut rows = NodeRows::good();
    rows.target[3] = None;
    let output = run_node_rows(&rows, false, false, IngestOnError::SkipRowOnError).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&output.manifest_path).unwrap()).unwrap();
    let failures = &manifest["row_failures"];
    assert_eq!(failures["on_error"], "skip_row_on_error");
    assert_eq!(failures["skipped_rows"], 1);
    assert_eq!(failures["by_condition"][0]["condition"], "null_target");
    assert_eq!(failures["by_condition"][0]["column"], "TARGET1");
}

#[test]
fn test_row_validation_counts_each_condition_but_skips_the_row_once() {
    let mut rows = NodeRows::good();
    rows.entity[1] = None; // also has a null target below
    rows.target[1] = None;
    let output = run_node_rows(&rows, false, false, IngestOnError::SkipRowOnError).unwrap();
    let report = output.row_failures.unwrap();
    assert_eq!(report.skipped_rows, 1);
    assert_eq!(report.by_condition.len(), 2);
    assert_eq!(report.by_condition[0].condition, RowFailureKind::NullEntity);
    assert_eq!(report.by_condition[1].condition, RowFailureKind::NullTarget);
}

#[test]
fn test_row_validation_unknown_entity_key_is_counted_and_skipped_in_both_modes() {
    // Entity 9 is absent: skip it in both modes and include the reason in the report.
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let mut rows = NodeRows::good();
        rows.entity[2] = Some(9);
        let output = run_node_rows(&rows, false, false, mode).unwrap();
        assert_eq!(output.rows_written, 4);
        let report = output.row_failures.unwrap();
        assert_eq!(report.skipped_rows, 1);
        assert_eq!(
            report.by_condition[0].condition,
            RowFailureKind::UnknownEntity
        );
    }
}

#[test]
fn test_row_validation_source_that_keeps_no_rows_fails_in_both_modes() {
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        // Every row's entity is unknown to the graph.
        let mut rows = NodeRows::good();
        rows.entity = (10..15).map(Some).collect();
        let err = run_node_rows(&rows, false, false, mode)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("kept no rows"), "{err}");
    }
    // Every row invalid, skipped one by one.
    let mut rows = NodeRows::good();
    rows.target = vec![None; 5];
    let err = run_node_rows(&rows, false, false, IngestOnError::SkipRowOnError)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("5 read, 5 skipped"), "{err}");
}

#[test]
fn test_row_validation_weight_with_no_positive_value_fails_in_both_modes() {
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let mut rows = NodeRows::good();
        rows.weight = vec![Some(0.0); 5];
        let err = run_node_rows(&rows, true, false, mode)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("no positive value"), "{err}");
    }
}

#[test]
fn test_row_validation_split_values_come_from_the_plan() {
    // The plan names its own split values; 7 is valid when the plan says so.
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("rows.parquet");
    let mut rows = NodeRows::good();
    rows.split[2] = Some(7);
    rows.write(&pq, false, true);
    let mut plan = make_custom_tt_target_only_node_plan();
    let mut spec = manual_split_col_spec();
    spec.manual_test_values = Some(vec!["2".to_string(), "7".to_string()]);
    plan.split = Some(spec);
    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass_with_source(
        &plan,
        &index,
        out_dir.path(),
        Some(source(&pq, IngestOnError::Abort)),
    )
    .unwrap();
    assert_eq!(output.rows_written, 5);
}

// ── Link prediction (list targets, per-element weights) ──────────────────────

fn lp_plan(weight: bool) -> DiskGraphTTPlan<Seconds> {
    let mut plan = make_custom_tt_lp_node_plan(None);
    plan.prediction_type = PredictionType::Link {
        include_frequency: true,
    };
    if weight {
        plan.weight_col = Some("WEIGHT".to_string());
    }
    plan
}

/// Write an LP parquet whose TARGET1 (and optionally WEIGHT) cells may be null.
fn write_lp_rows(
    path: &Path,
    targets: Vec<Option<Vec<Option<i64>>>>,
    weights: Option<Vec<Option<Vec<f64>>>>,
) {
    let n = targets.len();
    let mut target = ListBuilder::new(Int64Builder::new());
    let mut frequency = ListBuilder::new(Int64Builder::new());
    for cell in &targets {
        match cell {
            Some(items) => {
                for item in items {
                    target.values().append_option(*item);
                    frequency.values().append_value(1);
                }
                target.append(true);
                frequency.append(true);
            }
            None => {
                target.append(false);
                frequency.append(false);
            }
        }
    }
    let list_type = |t| DataType::List(Arc::new(Field::new("item", t, true)));
    let mut fields = vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", list_type(DataType::Int64), true),
        Field::new(
            "__kumo_internal_frequency_column__",
            list_type(DataType::Int64),
            true,
        ),
    ];
    let mut columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
        Arc::new(TimestampNanosecondArray::from(vec![
            1_000_000_000_000_i64;
            n
        ])),
        Arc::new(target.finish()),
        Arc::new(frequency.finish()),
    ];
    if let Some(weights) = weights {
        let mut weight = ListBuilder::new(Float64Builder::new());
        for cell in &weights {
            match cell {
                Some(items) => {
                    items.iter().for_each(|w| weight.values().append_value(*w));
                    weight.append(true);
                }
                None => weight.append(false),
            }
        }
        fields.push(Field::new("WEIGHT", list_type(DataType::Float64), true));
        columns.push(Arc::new(weight.finish()));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    write_parquet(path, &batch);
}

fn run_lp_rows(
    targets: Vec<Option<Vec<Option<i64>>>>,
    weights: Option<Vec<Option<Vec<f64>>>>,
    allow_negative_weight: bool,
    on_error: IngestOnError,
) -> TTGenResult<TTPassOutput> {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_and_products_graph(dir.path(), 8, 8);
    let pq = dir.path().join("lp.parquet");
    let weighted = weights.is_some();
    write_lp_rows(&pq, targets, weights);
    let out_dir = tempfile::tempdir().unwrap();
    let mut src = source(&pq, on_error);
    src.allow_negative_weight = allow_negative_weight;
    execute_tt_pass_with_source(&lp_plan(weighted), &index, out_dir.path(), Some(src))
}

fn lp_good() -> Vec<Option<Vec<Option<i64>>>> {
    (0..4).map(|i| Some(vec![Some(i), Some(i + 1)])).collect()
}

fn lp_good_weights() -> Vec<Option<Vec<f64>>> {
    (0..4).map(|_| Some(vec![1.0, 2.0])).collect()
}

#[test]
fn test_row_validation_lp_conditions_in_both_modes() {
    type Case = (
        &'static str,
        RowFailureKind,
        &'static str,
        Box<dyn Fn() -> (Vec<Option<Vec<Option<i64>>>>, Option<Vec<Option<Vec<f64>>>>)>,
    );
    let cases: Vec<Case> = vec![
        (
            "null list",
            RowFailureKind::NullTarget,
            "TARGET1",
            Box::new(|| {
                let mut t = lp_good();
                t[2] = None;
                (t, None)
            }),
        ),
        (
            "null element",
            RowFailureKind::NullTargetItem,
            "TARGET1",
            Box::new(|| {
                let mut t = lp_good();
                t[2] = Some(vec![Some(1), None]);
                (t, None)
            }),
        ),
        (
            "null weight list",
            RowFailureKind::NullWeight,
            "WEIGHT",
            Box::new(|| {
                let mut w = lp_good_weights();
                w[2] = None;
                (lp_good(), Some(w))
            }),
        ),
        (
            "weight length mismatch",
            RowFailureKind::WeightLengthMismatch,
            "WEIGHT",
            Box::new(|| {
                let mut w = lp_good_weights();
                w[2] = Some(vec![1.0]);
                (lp_good(), Some(w))
            }),
        ),
        (
            "negative weight element",
            RowFailureKind::NegativeWeight,
            "WEIGHT",
            Box::new(|| {
                let mut w = lp_good_weights();
                w[2] = Some(vec![1.0, -2.0]);
                (lp_good(), Some(w))
            }),
        ),
        (
            "infinite weight element",
            RowFailureKind::NonFiniteWeight,
            "WEIGHT",
            Box::new(|| {
                let mut w = lp_good_weights();
                w[2] = Some(vec![f64::INFINITY, 2.0]);
                (lp_good(), Some(w))
            }),
        ),
    ];
    for (name, kind, column, build) in cases {
        let (targets, weights) = build();
        let err = run_lp_rows(
            targets.clone(),
            weights.clone(),
            false,
            IngestOnError::Abort,
        )
        .err()
        .unwrap_or_else(|| panic!("{name}: abort must fail"))
        .to_string();
        assert!(
            err.contains(kind.describe()) && err.contains(column) && err.contains("row 2"),
            "{name}: {err}"
        );

        let output = run_lp_rows(targets, weights, false, IngestOnError::SkipRowOnError)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(output.rows_written, 3, "{name}");
        let report = output.row_failures.unwrap();
        assert_eq!(
            (report.skipped_rows, report.by_condition[0].condition),
            (1, kind),
            "{name}"
        );
    }
}

#[test]
fn test_row_validation_lp_negative_weight_is_allowed_when_the_task_allows_it() {
    let mut weights = lp_good_weights();
    weights[2] = Some(vec![1.0, -2.0]);
    let output = run_lp_rows(lp_good(), Some(weights), true, IngestOnError::Abort).unwrap();
    assert_eq!(output.rows_written, 4);
}

#[test]
fn test_row_validation_lp_empty_target_list_is_counted_and_skipped_in_both_modes() {
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let mut targets = lp_good();
        targets[2] = Some(vec![]);
        let output = run_lp_rows(targets, None, false, mode).unwrap();
        assert_eq!(output.rows_written, 3);
        let report = output.row_failures.unwrap();
        assert_eq!(
            report.by_condition[0].condition,
            RowFailureKind::EmptyTargetList
        );
        assert_eq!(report.skipped_rows, 1);
    }
}

#[test]
fn test_row_validation_lp_every_list_empty_fails_in_both_modes() {
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let targets = (0..4).map(|_| Some(vec![])).collect();
        let err = run_lp_rows(targets, None, false, mode)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("kept no rows"), "{err}");
    }
}

fn run_timestamp_reference_case(
    timestamp: Arc<dyn Array>,
    declared: diskgraph_core::time::TimeUnit,
    reference: Option<diskgraph_train_table::custom_tt_row_errors::TimestampReference>,
    mode: IngestOnError,
) -> TTGenResult<TTPassOutput> {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let parts = dir.path().join("source");
    fs::create_dir(&parts).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", timestamp.data_type().clone(), false),
            Field::new("TARGET1", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4])),
            timestamp,
            Arc::new(Float64Array::from(vec![1.0; 5])),
        ],
    )
    .unwrap();
    write_parquet(&parts.join("part-0.parquet"), &batch.slice(0, 2));
    write_parquet(&parts.join("part-1.parquet"), &batch.slice(2, 3));
    let mut source = source(&parts, mode);
    source.input_timestamp_unit = declared;
    source.timestamp_reference = reference;
    let out = dir.path().join("out");
    fs::create_dir(&out).unwrap();
    execute_tt_pass_with_source(
        &make_custom_tt_target_only_node_plan(),
        &index,
        &out,
        Some(source),
    )
}

#[test]
fn test_timestamp_reference_rejects_wrong_integer_units_in_both_modes() {
    use diskgraph_core::time::TimeUnit as Unit;
    use diskgraph_train_table::custom_tt_row_errors::TimestampReference;
    let nanos = 1_700_000_004_000_000_000i64;
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        for (divisor, name) in [
            (1_000_000_000, "seconds"),
            (1_000_000, "milliseconds"),
            (1_000, "microseconds"),
        ] {
            let timestamps: Vec<i64> = (0..5)
                .map(|i| (nanos - (4 - i) * 1_000_000_000) / divisor)
                .collect();
            let err = run_timestamp_reference_case(
                Arc::new(Int64Array::from(timestamps)),
                Unit::Nanoseconds,
                Some(TimestampReference {
                    rows: 5,
                    max_ns: nanos,
                }),
                mode,
            )
            .err()
            .unwrap()
            .to_string();
            assert!(
                err.contains(&format!("looks like {name}, not nanoseconds")),
                "{err}"
            );
        }
    }
}

#[test]
fn test_timestamp_reference_accepts_correct_units_and_self_describing_timestamps() {
    use diskgraph_core::time::TimeUnit as Unit;
    use diskgraph_train_table::custom_tt_row_errors::TimestampReference;
    let nanos = 1_700_000_000_000_000_000i64;
    for (divisor, unit) in [
        (1, Unit::Nanoseconds),
        (1_000, Unit::Microseconds),
        (1_000_000, Unit::Milliseconds),
        (1_000_000_000, Unit::Seconds),
    ] {
        let out = run_timestamp_reference_case(
            Arc::new(Int64Array::from(vec![nanos / divisor; 5])),
            unit,
            Some(TimestampReference {
                rows: 5,
                max_ns: nanos,
            }),
            IngestOnError::Abort,
        )
        .unwrap();
        assert_eq!(out.rows_written, 5);
    }
    let out = run_timestamp_reference_case(
        Arc::new(TimestampNanosecondArray::from(vec![nanos; 5])),
        Unit::Seconds,
        Some(TimestampReference {
            rows: 5,
            max_ns: nanos,
        }),
        IngestOnError::Abort,
    )
    .unwrap();
    assert_eq!(out.rows_written, 5);
}

#[test]
fn test_timestamp_reference_does_not_guess_without_correspondence() {
    use diskgraph_core::time::TimeUnit as Unit;
    use diskgraph_train_table::custom_tt_row_errors::TimestampReference;
    for reference in [
        None,
        Some(TimestampReference {
            rows: 6,
            max_ns: 1_700_000_000_000_000_000,
        }),
    ] {
        let out = run_timestamp_reference_case(
            Arc::new(Int64Array::from(vec![1_700_000_000; 5])),
            Unit::Nanoseconds,
            reference,
            IngestOnError::Abort,
        )
        .unwrap();
        assert_eq!(out.rows_written, 5);
    }
}

#[test]
fn test_unknown_entity_does_not_hide_invalid_content() {
    let mut rows = NodeRows::good();
    rows.entity[2] = Some(9);
    rows.target[2] = None;
    rows.weight[2] = None;
    let error = run_node_rows(&rows, true, false, IngestOnError::Abort)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("target is null"), "{error}");
    assert!(error.contains("weight is null"), "{error}");
    let output = run_node_rows(&rows, true, false, IngestOnError::SkipRowOnError).unwrap();
    let report = output.row_failures.unwrap();
    assert_eq!(report.skipped_rows, 1);
    assert_eq!(report.by_condition.len(), 2);
    assert!(report
        .by_condition
        .iter()
        .all(|condition| condition.count == 1));
}

#[test]
fn test_empty_target_does_not_hide_invalid_weight() {
    let mut targets = lp_good();
    targets[2] = Some(vec![]);
    let mut weights = lp_good_weights();
    weights[2] = None;
    let error = run_lp_rows(
        targets.clone(),
        Some(weights.clone()),
        false,
        IngestOnError::Abort,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("target list is empty"), "{error}");
    assert!(error.contains("weight is null"), "{error}");
    let output = run_lp_rows(targets, Some(weights), false, IngestOnError::SkipRowOnError).unwrap();
    let report = output.row_failures.unwrap();
    assert_eq!(report.skipped_rows, 1);
    assert_eq!(report.by_condition.len(), 2);
}

#[test]
fn test_default_skip_policy_keeps_even_a_small_valid_remainder() {
    let mut rows = NodeRows::good();
    rows.target[..4].fill(None);
    let output = run_node_rows(&rows, false, false, IngestOnError::default()).unwrap();
    assert_eq!(output.rows_written, 1);
    assert_eq!(output.row_failures.unwrap().skipped_rows, 4);
}

#[test]
fn test_filtered_empty_link_targets_are_counted_and_all_filtered_rows_fail() {
    let run = |targets, mode| {
        let dir = tempfile::tempdir().unwrap();
        let index = build_users_and_products_graph(dir.path(), 8, 8);
        let parquet = dir.path().join("links.parquet");
        write_lp_rows(&parquet, targets, None);
        fs::create_dir_all(dir.path().join("out")).unwrap();
        let mut plan = make_custom_tt_lp_node_plan(Some("products"));
        plan.prediction_type = PredictionType::Link {
            include_frequency: false,
        };
        execute_tt_pass_with_source(
            &plan,
            &index,
            &dir.path().join("out"),
            Some(source(&parquet, mode)),
        )
    };
    for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
        let mut targets = lp_good();
        targets[2] = Some(vec![Some(999)]);
        let output = run(targets, mode).unwrap();
        assert_eq!(output.rows_written, 3);
        let report = output.row_failures.unwrap();
        assert_eq!(report.skipped_rows, 1);
        assert_eq!(
            report.by_condition[0].condition,
            RowFailureKind::UnavailableTarget
        );
        let error = run(vec![Some(vec![Some(999)]); 4], mode)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("kept no rows"), "{error}");
    }
}
