//! TT-NG comprehensive end-to-end tests.
//!
//! Each test exercises the FULL pipeline (pass A → fused pass-B routing →
//! manifest writer) and asserts cross-phase invariants — per-stage parquet
//! presence, row counts, manifest schema, and seed-time correctness.
//!
//! The previous phases each tested their isolated piece. The tests in
//! this file verify the whole pipeline still agrees with itself.
//!
//! Conventions:
//! - Every test runs against a tiny synthesized graph (typically 5
//!   users / 4 products / 10 orders) with a `tempfile::TempDir`
//!   for output. No fixture lives outside the test binary.
//! - Greenfield TT-NG: gen writes no `split.bin` / `arange.bin`. A split plan
//!   produces per-stage seed parquets and records their row counts + seed-times
//!   in `TTPassOutput.stage_gen_infos`; the no-split (prediction) path leaves
//!   `stage_gen_infos = None`. Helpers below read these the way ingest does, so
//!   an assertion failure here is a real cross-component contract violation.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::*;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, ListArray, RecordBatch, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
use diskgraph_core::time::{Abs, Seconds, TimeUnit as DGTimeUnit};
use diskgraph_core::tt_builder::*;
use diskgraph_ingest::coercion::CoercionMode;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::*;
use diskgraph_ingest::io::Dest;
use diskgraph_train_table::manifest::{
    read_manifest, sha256_file, ArtifactRef, SidecarRef, TTManifest, SPLIT_STATS_FORMAT,
    STATS_FORMAT, TT_PARQUET_FORMAT,
};
use diskgraph_train_table::ttgen_executor::{
    execute_tt_pass, execute_tt_pass_with_source, CustomTtSource, TTPassOutput,
    STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

// ─── Test infrastructure ─────────────────────────────────────────────────────
//
// Self-contained graph + plan builders. Reuses the *shape* of
// `tests/ttgen_integration.rs::build_graph1` (a 5-user, 4-product,
// 10-order graph) but trimmed to the fields these e2e tests actually
// touch. Inlining keeps the test file readable on its own.

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

/// 5 users / 4 products / 10 orders, with a `users.user_id` ↔ `orders.user_id`
/// fkey carrying timestamps. This mirrors `build_graph1` from the existing
/// integration suite (just enough columns for the temporal-anchor test plans
/// below).
fn build_e2e_graph(root: &Path) -> PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    // Users: 5 PKs, all `active=true` so include_mask doesn't kick out anyone
    // by default. The lag-timesteps test below flips that.
    let mut user_id = Int64Builder::new();
    let mut active = BooleanBuilder::new();
    for (uid, a) in [(0i64, true), (1, true), (2, false), (3, true), (4, true)] {
        user_id.append_value(uid);
        active.append_value(a);
    }
    let users_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("active", DataType::Boolean, false),
    ]);
    let users_batch = RecordBatch::try_new(
        Arc::new(users_schema),
        vec![Arc::new(user_id.finish()), Arc::new(active.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    // Products: 4 PKs.
    let mut product_id = Int64Builder::new();
    for pid in [10i64, 20, 30, 40] {
        product_id.append_value(pid);
    }
    let products_schema = Schema::new(vec![Field::new("product_id", DataType::Int64, false)]);
    let products_batch = RecordBatch::try_new(
        Arc::new(products_schema),
        vec![Arc::new(product_id.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &products_batch);

    // Orders: each user buys 2 products at distinct timestamps.
    let mut order_id = Int64Builder::new();
    let mut order_user_id = Int64Builder::new();
    let mut order_product_id = Int64Builder::new();
    let mut price = Float64Builder::new();
    let mut ts = Int64Builder::new();
    let mut oid = 0i64;
    for uid in 0..5i64 {
        for (j, pid) in [10i64, 20].iter().enumerate() {
            order_id.append_value(oid);
            order_user_id.append_value(uid);
            order_product_id.append_value(*pid);
            price.append_value(10.0 * (uid as f64 + 1.0) + j as f64);
            ts.append_value(1000 + uid * 500 + j as i64 * 200);
            oid += 1;
        }
    }
    let orders_schema = Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("ts", DataType::Int64, false),
    ]);
    let orders_batch = RecordBatch::try_new(
        Arc::new(orders_schema),
        vec![
            Arc::new(order_id.finish()),
            Arc::new(order_user_id.finish()),
            Arc::new(order_product_id.finish()),
            Arc::new(price.finish()),
            Arc::new(ts.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("orders.parquet"), &orders_batch);

    let orders_time = TimeSpec {
        table: "orders".to_string(),
        column: "ts".to_string(),
        bitwidth: Some(16),
        anchor_time: Some(0),
        unit: "s".to_string(),
    };
    let config = GraphConfig {
        dataset_name: "test_e2e".to_string(),
        tables: vec![
            TableSpec {
                name: "users".to_string(),
                columns: vec![
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("active", IngestedType::Bool),
                ],
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
            },
            TableSpec {
                name: "products".to_string(),
                columns: vec![mk_col("product_id", IngestedType::Int64)],
                primary_key: Some("product_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir
                        .join("products.parquet")
                        .to_str()
                        .unwrap()
                        .to_string()],
                },
                source_size: None,
                fkeys: vec![],
                data_version: None,
                kumo_connector_id: None,
                kumo_source_table_name: None,
                group_by_timestamp: false,
            },
            TableSpec {
                name: "orders".to_string(),
                columns: vec![
                    mk_col("order_id", IngestedType::Int64),
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("price", IngestedType::Float64),
                    mk_col("ts", IngestedType::Int64),
                ],
                primary_key: Some("order_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir
                        .join("orders.parquet")
                        .to_str()
                        .unwrap()
                        .to_string()],
                },
                source_size: None,
                fkeys: vec![
                    FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(orders_time.clone()),
                            weight: None,
                        }),
                        ..Default::default()
                    },
                    FKeySpec {
                        src_column: "product_id".to_string(),
                        dst_table: "products".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(orders_time.clone()),
                            weight: None,
                        }),
                        ..Default::default()
                    },
                ],
                data_version: None,
                kumo_connector_id: None,
                kumo_source_table_name: None,
                group_by_timestamp: false,
            },
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

/// Single-table users-only graph (5 PKs). Used by the custom-TT tests —
/// they don't need edges.
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
        dataset_name: "test_e2e_users_only".to_string(),
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

/// Build a custom-TT parquet with ENTITY1, TIMESTAMP, TARGET1.
fn synth_target_only_parquet(out_path: &Path, entity_pks: &[i64], anchor_ts_secs: &[i64]) {
    let n_rows = entity_pks.len();
    assert_eq!(anchor_ts_secs.len(), n_rows);
    let mut entity = Int64Builder::new();
    let mut target = Float64Builder::new();
    for i in 0..n_rows {
        entity.append_value(entity_pks[i]);
        target.append_value((entity_pks[i] as f64) * 100.0 + (anchor_ts_secs[i] as f64));
    }
    let ts_arr = TimestampNanosecondArray::from(
        anchor_ts_secs
            .iter()
            .map(|t| t * 1_000_000_000)
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(entity.finish()),
            Arc::new(ts_arr),
            Arc::new(target.finish()),
        ],
    )
    .unwrap();
    write_parquet(out_path, &batch);
}

/// Synthesize a Node custom-TT parquet WITH a customer-provided `split_col`
/// (Int64, 0/1/2 per the kumo baseline convention).
fn synth_target_only_parquet_with_split(
    out_path: &Path,
    entity_pks: &[i64],
    anchor_ts_secs: &[i64],
    split_values: &[i64],
) {
    let n_rows = entity_pks.len();
    assert_eq!(anchor_ts_secs.len(), n_rows);
    assert_eq!(split_values.len(), n_rows);
    let mut entity = Int64Builder::new();
    let mut target = Float64Builder::new();
    let mut split = Int64Builder::new();
    for i in 0..n_rows {
        entity.append_value(entity_pks[i]);
        target.append_value((entity_pks[i] as f64) * 100.0 + (anchor_ts_secs[i] as f64));
        split.append_value(split_values[i]);
    }
    let ts_arr = TimestampNanosecondArray::from(
        anchor_ts_secs
            .iter()
            .map(|t| t * 1_000_000_000)
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        Field::new("split_col", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(entity.finish()),
            Arc::new(ts_arr),
            Arc::new(target.finish()),
            Arc::new(split.finish()),
        ],
    )
    .unwrap();
    write_parquet(out_path, &batch);
}

/// A `ManualSplit` spec on a customer-provided `split_col` (the kumo CTT
/// custom-split contract): values "0"/"1"/"2" = train/val/test.
fn manual_split_col_spec() -> SplitSpec<Seconds> {
    let mut s = empty_split_spec("ManualSplit");
    s.split_column = Some("split_col".to_string());
    s.manual_train_values = Some(vec!["0".to_string()]);
    s.manual_val_values = Some(vec!["1".to_string()]);
    s.manual_test_values = Some(vec!["2".to_string()]);
    s.is_temporal = Some(false);
    s
}

// ─── Plan builders ───────────────────────────────────────────────────────────

fn count_orders_op(alias: &str, window: Option<(i64, i64)>) -> Op<Seconds> {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![EdgeRef {
            src_table: "users".to_string(),
            src_col: "user_id".to_string(),
            dst_table: "orders".to_string(),
            dst_col: "user_id".to_string(),
        }],
        func: AggFunc::Count,
        target_col: None,
        time_col: None,
        window,
        window_unit: Some(DGTimeUnit::Seconds),
        row_filter: None,
        limit: None,
        target_entity_table: None,
    }
}

fn list_distinct_product_id_op(alias: &str) -> Op<Seconds> {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![EdgeRef {
            src_table: "users".to_string(),
            src_col: "user_id".to_string(),
            dst_table: "orders".to_string(),
            dst_col: "user_id".to_string(),
        }],
        func: AggFunc::ListDistinct,
        target_col: Some("product_id".to_string()),
        time_col: None,
        window: None,
        window_unit: None,
        row_filter: None,
        limit: None,
        target_entity_table: None,
    }
}

fn anchored_node_plan(timestamps: Vec<i64>) -> DiskGraphTTPlan<Seconds> {
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
        temporal: Some(TemporalSpec::<Seconds> {
            timestamps: timestamps.into_iter().map(Abs::from_raw).collect(),
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        is_prediction: false,
        split: None,
        stages: vec![Stage::<Seconds> {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: None,
        }],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    }
}

fn array_lp_plan() -> DiskGraphTTPlan<Seconds> {
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
        prediction_type: PredictionType::Link {
            include_frequency: true,
        },
        is_prediction: false,
        split: None,
        stages: vec![Stage::<Seconds> {
            ops: vec![list_distinct_product_id_op("ld")],
            filter: None,
        }],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    }
}

fn time_range_split_spec(ranges: Vec<(i64, i64)>) -> SplitSpec<Seconds> {
    SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("TimeRangeSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: Some(
            ranges
                .into_iter()
                .map(|(a, b)| (Abs::<Seconds>::from_raw(a), Abs::<Seconds>::from_raw(b)))
                .collect(),
        ),
        backtesting_timestamp: None,
        split_column: None,
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    }
}

fn empty_split_spec(kind: &str) -> SplitSpec<Seconds> {
    SplitSpec::<Seconds> {
        ratio: None,
        kind: Some(kind.to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
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
    }
}

// ─── Stage-gen-info helpers (greenfield TT-NG) ───────────────────────────────
//
// A split plan emits per-stage seed parquets and records them in
// `TTPassOutput.stage_gen_infos` (no split.bin / arange.bin). Contract
// assertions flow through these helpers.

/// Per-stage row count from `stage_gen_infos` for a given key (0 if absent).
fn stage_row_count(output: &TTPassOutput, key: &str) -> u64 {
    output
        .stage_gen_infos
        .as_ref()
        .and_then(|v| v.iter().find(|s| s.key == key))
        .map(|s| s.row_count)
        .unwrap_or(0)
}

/// Cross-phase invariant: `stage_gen_infos` carries exactly train/val/test;
/// each stage's `parts` exist in `output_dir` with Σ part rows == `row_count`.
/// A row in `unused` appears in no seed parquet, so the tier sum can be
/// < `rows_written`; callers that know the exact unused count assert it.
fn assert_stage_infos_consistent(output: &TTPassOutput, output_dir: &Path) {
    let infos = output
        .stage_gen_infos
        .as_ref()
        .expect("split plan must produce stage_gen_infos");
    let keys: Vec<&str> = infos.iter().map(|s| s.key.as_str()).collect();
    assert_eq!(
        keys,
        vec!["train", "val", "test"],
        "stage_gen_infos must carry train/val/test in order"
    );
    for s in infos {
        assert!(
            !s.parts.is_empty(),
            "stage `{}` must record at least one part",
            s.key
        );
        let inventory = s
            .artifact_inventory
            .as_ref()
            .expect("new stage output must carry an artifact inventory");
        assert_eq!(
            inventory.schema_version,
            STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION
        );
        assert_eq!(inventory.parts.len(), s.parts.len());
        let mut part_sum = 0u64;
        for (part_index, part) in s.parts.iter().enumerate() {
            let seed_path = output_dir.join(part);
            assert!(
                seed_path.exists(),
                "stage `{}` part {} must exist on disk",
                s.key,
                seed_path.display()
            );
            let rows = parquet_row_count(&seed_path);
            let artifact = &inventory.parts[part_index];
            assert_eq!(artifact.path, *part);
            assert_eq!(artifact.size_bytes, fs::metadata(&seed_path).unwrap().len());
            assert_eq!(artifact.row_count, rows);
            part_sum += rows;
        }
        assert_eq!(
            part_sum, s.row_count,
            "stage `{}` Σ part rows ({part_sum}) disagrees with \
             stage_gen_infos.row_count ({})",
            s.key, s.row_count
        );
    }
    // No `tier_sum <= rows_written` bound: per-link array splits explode each
    // anchor into ≤3 (anchor, stage) rows, so the tier sum can exceed the
    // monolith anchor count. Per-anchor callers assert the relationship.
    let _ = output.rows_written;
}

// ─── Test 1: PQ end-to-end with TimeRangeSplit ──────────────────────────────

#[test]
fn e2e_pq_time_range_split_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(time_range_split_spec(vec![
        (0, 1500),
        (1501, 2500),
        (2501, 4000),
    ]));

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // 1. tt.parquet
    assert!(output.tt_parquet_path.exists());
    assert_eq!(output.tt_parquet_path, out_dir.path().join("tt.parquet"));
    let parquet_rows = parquet_row_count(&output.tt_parquet_path);
    assert_eq!(parquet_rows, output.rows_written);
    assert!(output.rows_written > 0);

    // 2. Split reflected in stage_gen_infos: three seed parquets on disk, row
    //    counts agree.
    assert_stage_infos_consistent(&output, out_dir.path());

    // 4. stats.json parses + envelope is correct
    let stats_path = out_dir.path().join("stats.json");
    assert!(stats_path.exists());
    let stats_raw = fs::read_to_string(&stats_path).unwrap();
    let stats_v: serde_json::Value = serde_json::from_str(&stats_raw).unwrap();
    assert_eq!(stats_v["version"].as_u64(), Some(2));
    assert_eq!(stats_v["kind"].as_str(), Some("tt_stats"));
    assert_eq!(
        stats_v["num_total_rows"].as_u64(),
        Some(output.rows_written)
    );

    // 5. manifest.json parses + every ArtifactRef.path exists. Flat-manifest
    //    split fields stay None (split lives in stage_gen_infos).
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.row_count, output.rows_written);
    assert_eq!(manifest.source_mode, "pquery");
    assert_eq!(manifest.split_mode, "none");
    assert_eq!(manifest.split_kind.as_deref(), Some("TimeRangeSplit"));
    assert!(manifest.split.is_none());
    assert!(manifest.stage_counts.is_none());
    assert_artifact_paths_exist(&manifest, out_dir.path());

    // 6. stage row counts sum to row_count (TimeRangeSplit over a range that
    //    covers all anchors → no unused rows here).
    let tier_sum =
        stage_row_count(&output, "train") + stage_row_count(&output, "val") + stage_row_count(&output, "test");
    assert_eq!(tier_sum, output.rows_written);
}

// ─── Test 2: PQ with no split (SplitMode::None) ─────────────────────────────

#[test]
fn e2e_pq_no_split_omits_split_bin() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let plan = anchored_node_plan(vec![1000, 2000, 3000]);

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // tt.parquet present.
    assert!(output.tt_parquet_path.exists());
    assert!(output.rows_written > 0);
    // No-split (prediction) path: no per-stage seed parquets and no
    // stage_gen_infos. The single stage is `tt.parquet` itself.
    assert!(output.stage_gen_infos.is_none());
    assert!(!out_dir.path().join("split.bin").exists());
    assert!(!out_dir.path().join("arange.bin").exists());
    assert!(!out_dir.path().join("train_seeds.parquet").exists());
    assert!(!out_dir.path().join("val_seeds.parquet").exists());
    assert!(!out_dir.path().join("test_seeds.parquet").exists());

    // Manifest: split absent, stage_counts absent, split_mode = "none".
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert!(manifest.split.is_none(), "manifest.split should be absent");
    assert!(
        manifest.stage_counts.is_none(),
        "manifest.stage_counts should be absent"
    );
    assert_eq!(manifest.split_mode, "none");
    assert!(manifest.split_kind.is_none());
    // The on-disk JSON shouldn't even mention `"split"` as a key (it
    // elides via `skip_serializing_if`). Spot-check that.
    let raw = fs::read_to_string(&output.manifest_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let obj = v.as_object().unwrap();
    assert!(!obj.contains_key("split"), "split key should be elided");
    assert!(!obj.contains_key("stage_counts"));
}

// ─── Test 3: Custom-TT passthrough end-to-end ───────────────────────────────

#[test]
fn e2e_custom_tt_passthrough_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("custom_tt.parquet");
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4, 0, 1, 2];
    let ts_secs: Vec<i64> = vec![
        1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 2_000_000,
    ];
    synth_target_only_parquet(&pq, &entity_pks, &ts_secs);

    let plan = DiskGraphTTPlan::<Seconds> {
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
    };

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: DGTimeUnit::Nanoseconds,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    // All 8 customer rows preserved (PKs 0..4 all exist in the index).
    assert_eq!(output.rows_written, 8);

    // tt.parquet exists; row count agrees.
    let parquet_rows = parquet_row_count(&output.tt_parquet_path);
    assert_eq!(parquet_rows, 8);

    // Verify the customer's column values made it through. The output
    // tt.parquet may reorder columns; we just check ENTITY/TARGET pairs
    // match the deterministic synthesis function.
    let rows = read_node_rows(&output.tt_parquet_path);
    assert_eq!(rows.len(), 8);
    let mut sorted_input: Vec<(i64, i64, f64)> = entity_pks
        .iter()
        .zip(ts_secs.iter())
        .map(|(&e, &t)| (e, t, (e as f64) * 100.0 + t as f64))
        .collect();
    sorted_input.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut sorted_output = rows.clone();
    sorted_output.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(sorted_input, sorted_output);

    // Stats sidecar emitted; no split → no per-stage seed parquets and no
    // stage_gen_infos (arange.bin / split.bin are gone in TT-NG).
    assert!(out_dir.path().join("stats.json").exists());
    assert!(output.stage_gen_infos.is_none());
    assert!(!out_dir.path().join("split.bin").exists());
    assert!(!out_dir.path().join("arange.bin").exists());

    // Manifest source_mode = "custom_tt"
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.source_mode, "custom_tt");
    assert_eq!(manifest.split_mode, "none");
    assert_eq!(manifest.row_count, output.rows_written);
    assert_artifact_paths_exist(&manifest, out_dir.path());
}

// ─── Test 4: LP+array+RandomLinkSplit regroup pipeline ──────────────────────

#[test]
fn e2e_lp_array_random_link_split_regroup() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = array_lp_plan();
    let mut spec = empty_split_spec("RandomLinkSplit");
    spec.is_temporal = Some(false);
    spec.ratio = Some(vec![0.5, 0.3, 0.2]);
    spec.rnd_seed = Some(13);
    plan.split = Some(spec);

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // 5 users × 2 link-cells. Per-link pass-B assigns each link a stage
    // independently and emits one row per (user, stage) non-empty cell. The
    // monolith `tt.parquet` is NOT regrouped — no inline stage column.
    assert!(output.rows_written > 0);
    let parquet = read_tt_parquet_columns(&output.tt_parquet_path);
    assert!(
        !parquet
            .column_names
            .iter()
            .any(|n| n == "__internal_kumo_split_id_col__"),
        "fused per-link pass-B must NOT add an inline stage column to the \
         monolith; cols={:?}",
        parquet.column_names
    );

    // 1. Per-stage seed parts exist + Σ part rows == row_count.
    assert_stage_infos_consistent(&output, out_dir.path());
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert!(manifest.stage_counts.is_none());
    assert_eq!(manifest.split_kind.as_deref(), Some("RandomLinkSplit"));
    assert!(
        manifest.train_table_col_stats.is_none(),
        "LP has no train-table target encoder and must omit the col-stats sidecar"
    );
    assert!(
        !out_dir.path().join("train_table_col_stats.json").exists(),
        "LP must not publish an unreferenced col-stats sidecar"
    );

    // 2. All 10 link-cells (5 users × 2 products) are accounted for across the
    //    three stages: count the TOTAL TARGET items in every seed part. Ratio
    //    components are all > 0, so no link is dropped to UNUSED.
    let infos = output.stage_gen_infos.as_ref().unwrap();
    let mut total_links: usize = 0;
    let mut stage_rows: u64 = 0;
    for s in infos {
        for part in &s.parts {
            let file = fs::File::open(out_dir.path().join(part)).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap();
            for batch_res in reader {
                let batch = batch_res.unwrap();
                let ti = batch.schema().index_of("TARGET1").unwrap();
                let tl = batch
                    .column(ti)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    let a = tl.value_offsets()[i] as usize;
                    let b = tl.value_offsets()[i + 1] as usize;
                    total_links += b - a;
                    stage_rows += 1;
                }
            }
        }
    }
    assert_eq!(total_links, 10, "all 5×2 link-cells must be routed (none dropped)");
    // Each user contributes 1+ stage rows; 5 users → 5..=10 output rows.
    assert!(
        (5..=10).contains(&stage_rows),
        "per-link output rows must be in 5..=10; got {stage_rows}"
    );
    let tier_sum: u64 = infos.iter().map(|s| s.row_count).sum();
    assert_eq!(tier_sum, stage_rows);
}

// ─── Test 4b: LP+array+DateOffsetSplit drops empty-target anchor rows ───────
//
// Regression for the ttgen empty-target bug: array LP (`explode: false`) with
// a TEMPORAL split (`DateOffsetSplit`) used to fall to the per-row pass-A/B
// path (NOT the per-link regroup), so empty `LIST_DISTINCT` anchor rows were
// written as-is instead of being dropped. This asserts that the emitted
// tt.parquet has NO row with an empty TARGET1 list, and that the manifest
// row_count + stage_gen_infos stay row-aligned to the surviving rows.
#[test]
fn e2e_lp_array_date_offset_split_drops_empty_targets() {
    use chrono::NaiveDate;

    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());

    // Windowed LIST_DISTINCT relative to anchor time. With anchors spread
    // across 2024 and a SHORT backward window, anchors whose window precedes
    // all of a user's orders produce an EMPTY product list → these are the
    // anchor rows the fix must drop.
    let mut plan = array_lp_plan();
    let ld_windowed = Op::<Seconds>::Aggregate {
        alias: "ld".to_string(),
        path: vec![EdgeRef {
            src_table: "users".to_string(),
            src_col: "user_id".to_string(),
            dst_table: "orders".to_string(),
            dst_col: "user_id".to_string(),
        }],
        func: AggFunc::ListDistinct,
        target_col: Some("product_id".to_string()),
        time_col: None,
        // Backward window [−3600s, 0]: only orders within the last hour of the
        // anchor count. The synthetic orders sit at small epoch seconds
        // (~1000-3200s from epoch), so only anchors near epoch-zero capture
        // any orders; the 2024 anchors capture none → empty lists.
        window: Some((-3600, 0)),
        window_unit: Some(DGTimeUnit::Seconds),
        row_filter: None,
        limit: None,
        target_entity_table: None,
    };
    plan.stages = vec![Stage::<Seconds> {
        ops: vec![ld_windowed],
        filter: None,
    }];

    // Multiple anchor timestamps per entity: an early one (near the orders,
    // → non-empty) and several 2024 ones (→ empty).
    let ymd = |y, m, d| {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp()
    };
    plan.temporal = Some(TemporalSpec::<Seconds> {
        timestamps: vec![
            Abs::from_raw(3600), // near the synthetic orders → some non-empty
            Abs::from_raw(ymd(2024, 1, 20)),
            Abs::from_raw(ymd(2024, 2, 20)),
            Abs::from_raw(ymd(2024, 3, 20)),
        ]
        .into_iter()
        .collect(),
        forecast_length: 1,
        lag_timesteps: None,
    });
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("DateOffsetSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
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
        month_offsets: Some(vec![-2, -1]),
        flexible_boundaries: None,
        timeframe_step: None,
    });

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // The empty-target drop is folded into the per-anchor route kernel: the
    // monolith retains its rows, but no seed-part row carries an empty TARGET1
    // list — empty-target anchors are routed to no stage.
    let infos = output
        .stage_gen_infos
        .as_ref()
        .expect("split plan must produce stage_gen_infos");
    let mut seed_rows: u64 = 0;
    for s in infos {
        for part in &s.parts {
            let file = fs::File::open(out_dir.path().join(part)).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap();
            for batch_res in reader {
                let batch = batch_res.unwrap();
                let ti = batch.schema().index_of("TARGET1").unwrap();
                let tl = batch
                    .column(ti)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    let a = tl.value_offsets()[i] as usize;
                    let b = tl.value_offsets()[i + 1] as usize;
                    assert!(
                        !tl.is_null(i) && (b - a) > 0,
                        "seed part {part} row {i} has an EMPTY TARGET1 list — \
                         empty-target anchor was not dropped from the stage"
                    );
                    seed_rows += 1;
                }
            }
        }
    }

    // Empties genuinely produced and dropped: 4 users × 4 anchors = 16
    // candidates; the short window leaves most with empty product lists, so
    // seed rows < 16. (`rows_written` is the monolith, which keeps the empties.)
    assert!(output.rows_written > 0);
    assert!(
        seed_rows > 0 && seed_rows < 16,
        "expected some empty-target anchors to be dropped from stages \
         (0 < seed_rows < 16), got seed_rows={seed_rows}"
    );

    // Per-stage seed parquets are internally consistent (parts exist, Σ part
    // rows == row_count). Tier sum (= seed_rows) <= rows_written.
    assert_stage_infos_consistent(&output, out_dir.path());
    let tier_sum: u64 = infos.iter().map(|s| s.row_count).sum();
    assert_eq!(tier_sum, seed_rows);
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.row_count, output.rows_written);
    assert_eq!(manifest.split_kind.as_deref(), Some("DateOffsetSplit"));
    assert!(manifest.stage_counts.is_none());
}

// ─── Test 5: Manifest round-trip through kumo's reader contract ─────────────

#[test]
fn e2e_manifest_kumo_reader_contract() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(time_range_split_spec(vec![
        (0, 1500),
        (1501, 2500),
        (2501, 4000),
    ]));

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // Required fields: version, row_count, source_mode, train_table, stats.
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.version, 3);
    assert_eq!(manifest.row_count, output.rows_written);
    assert_eq!(manifest.source_mode, "pquery");
    // The train-table reference is bundle-relative and records the exact
    // closed-file size without requiring a second content hash pass.
    let train_table_path = out_dir.path().join(&manifest.train_table.path);
    assert!(train_table_path.exists());
    let parts = diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(&train_table_path)
        .unwrap();
    assert_eq!(manifest.train_table.parts.len(), parts.len());
    assert_eq!(
        manifest.train_table.size_bytes,
        parts.iter().map(|part| std::fs::metadata(part).unwrap().len()).sum::<u64>()
    );
    for (declared, part) in manifest.train_table.parts.iter().zip(parts) {
        assert_eq!(
            declared.path,
            format!("tt.parquet/{}", part.file_name().unwrap().to_string_lossy())
    );
    }
    // `train_table.format` identifies the directory-dataset contract.
    assert_eq!(manifest.train_table.format, TT_PARQUET_FORMAT);
    // stats artifact ref.
    let stats_path = out_dir.path().join(&manifest.stats.path);
    assert!(stats_path.exists());
    assert_eq!(manifest.stats.format, STATS_FORMAT);
    // Flat-manifest split / arange fields stay absent (split lives in
    // stage_gen_infos, asserted below); split_sha256 empty.
    assert!(manifest.split.is_none(), "split ArtifactRef must be absent");
    assert!(
        manifest.arange_sidecar.is_none(),
        "arange_sidecar must be absent"
    );
    assert!(manifest.checksums.split_sha256.is_empty());
    // The split itself is reflected in the per-stage seed parquets.
    assert_stage_infos_consistent(&output, out_dir.path());

    // Bounded sidecar checksums remain content-validated. The large parquet
    // intentionally has no default SHA field in manifest v3.
    let stats_sha = sha256_file(&stats_path).unwrap();
    assert_eq!(stats_sha, manifest.checksums.stats_sha256);
    let sha = &manifest.checksums.stats_sha256;
    assert_eq!(sha.len(), 64, "sha is {} chars: {sha}", sha.len());
    assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));

    // Round-trip the parsed manifest back to JSON and re-parse — must be
    // bit-equal.
    let serde_round_trip: TTManifest =
        serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
    assert_eq!(manifest, serde_round_trip);
}

// ─── Test 6: include_mask cross-phase invariant ─────────────────────────────

#[test]
fn e2e_include_mask_forces_unused_across_phases() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());

    // `lag_timesteps > 0` plus a filter excluding the `active=false` user.
    // Rows with ENTITY_INCLUDE_BIT=false (pre-history + filter-failing) must be
    // routed to UNUSED (written to no stage).
    let plan = DiskGraphTTPlan::<Seconds> {
        entity: EntitySpec::<Seconds> {
            table: "users".to_string(),
            column: "user_id".to_string(),
            output_time_col_name: None,
            default_time_value: None,
            time_col: None,
            valid_to_col: None,
            filter: Some(RowFilter::Compare {
                lhs: RowFilterLhs::Column("active".to_string()),
                op: CompareOp::Eq,
                rhs: RowFilterRhs::Constant(Value::Bool(true)),
            }),
        },
        temporal: Some(TemporalSpec::<Seconds> {
            timestamps: vec![Abs::<Seconds>::from_raw(2000)],
            forecast_length: 1,
            lag_timesteps: Some(1),
        }),
        prediction_type: PredictionType::Node,
        is_prediction: false,
        split: Some(SplitSpec::<Seconds> {
            ratio: None,
            kind: Some("TimeRangeSplit".to_string()),
            rnd_seed: None,
            is_temporal: None,
            offsets: None,
            // Range covers ALL conceivable timestamps so the decider
            // would assign every row to train (0). The only way bytes
            // become 255 is via the include_mask hook.
            offset_unit: None,
            unit: None,
            timeframe: None,
            anchor_start_offset: None,
            ranges: Some(vec![(
                Abs::<Seconds>::from_raw(-1_000_000),
                Abs::<Seconds>::from_raw(1_000_000),
            )]),
            backtesting_timestamp: None,
            split_column: None,
            split_output_column: None,
            manual_train_values: None,
            manual_val_values: None,
            manual_test_values: None,
            month_offsets: None,
            flexible_boundaries: None,
            timeframe_step: None,
        }),
        stages: vec![Stage::<Seconds> {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: None,
        }],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    };

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // tt.parquet must carry the include-mask column (because lag>0).
    let parquet = read_tt_parquet_columns(&output.tt_parquet_path);
    assert!(
        parquet
            .column_names
            .iter()
            .any(|n| n == "ENTITY_INCLUDE_BIT"),
        "tt.parquet must emit ENTITY_INCLUDE_BIT when lag_timesteps>0; cols={:?}",
        parquet.column_names
    );
    let include_bits = parquet
        .include_bits
        .expect("include_bits must be readable when column exists");
    assert_eq!(
        include_bits.len() as u64,
        output.rows_written,
        "one mask byte per row"
    );
    // Count include_mask=false rows — forced to UNUSED by the include-mask hook
    // (the split range covers all timestamps, so the decider alone → all train).
    let false_count: u64 = include_bits.iter().filter(|m| !**m).count() as u64;
    assert!(
        false_count > 0,
        "fixture should have produced at least one include_mask=false row"
    );

    // UNUSED = rows_written - tier_sum (rows in no seed parquet). Must be
    // >= the include_mask=false count, which is all forced unused.
    assert_stage_infos_consistent(&output, out_dir.path());
    let tier_sum = stage_row_count(&output, "train")
        + stage_row_count(&output, "val")
        + stage_row_count(&output, "test");
    let unused = output.rows_written - tier_sum;
    assert!(
        unused >= false_count,
        "unused rows ({unused}) < false-mask rows ({false_count})"
    );
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert!(manifest.stage_counts.is_none());
}

// ─── Test 7: Custom-TT custom split via a SPLIT COLUMN ──────────────────────

/// Full gen→artifacts pipeline for a customer-provided custom split delivered
/// as a `split_col` column + a `ManualSplit` spec (the split.bin/UseBin path is
/// deleted). Asserts per-stage routing AND that `split_col` is dropped from the
/// seed parquets (never a feature), with no `split.bin` anywhere.
#[test]
fn e2e_custom_tt_manual_split_column_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);
    let pq = dir.path().join("custom_tt.parquet");
    // 6 rows: 5 known PKs + a duplicate.
    let entity_pks: Vec<i64> = vec![0, 1, 2, 3, 4, 0];
    let ts_secs: Vec<i64> = vec![
        1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000,
    ];
    // split_col: 0=train,1=val,2=test, plus a 255 "unused" value (no value set
    // matches it → the row is dropped from all stages).
    let split_values: Vec<i64> = vec![0, 1, 2, 0, 1, 255];
    synth_target_only_parquet_with_split(&pq, &entity_pks, &ts_secs, &split_values);

    let plan = DiskGraphTTPlan::<Seconds> {
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
        split: Some(manual_split_col_spec()),
        stages: vec![],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    };

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: DGTimeUnit::Nanoseconds,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();
    assert_eq!(output.rows_written, 6);

    // split_col routing: [0,1,2,0,1,255] → train=2, val=2, test=1, unused=1.
    assert_stage_infos_consistent(&output, out_dir.path());
    assert_eq!(stage_row_count(&output, "train"), 2);
    assert_eq!(stage_row_count(&output, "val"), 2);
    assert_eq!(stage_row_count(&output, "test"), 1);
    let tier_sum = stage_row_count(&output, "train")
        + stage_row_count(&output, "val")
        + stage_row_count(&output, "test");
    // The 255-valued row matched no value set → dropped from all stages.
    assert_eq!(output.rows_written - tier_sum, 1);

    // split_col must be DROPPED from every seed part (never a feature).
    let infos = output.stage_gen_infos.as_ref().unwrap();
    for info in infos {
        for part in &info.parts {
            let file = fs::File::open(out_dir.path().join(part)).unwrap();
            let schema = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .schema()
                .clone();
            assert!(
                schema.index_of("split_col").is_err(),
                "split_col must be dropped from seed part {part}"
            );
        }
    }

    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.split_mode, "none");
    assert_eq!(manifest.source_mode, "custom_tt");
    assert!(manifest.stage_counts.is_none());
    assert!(!out_dir.path().join("split.bin").exists());
}

/// Phase 2 (CTT rework): MULTI-PART custom-TT ingest. kumo passes the
/// train-table DIRECTORY containing ≥2 `part-*.parquet`, each carrying
/// `split_col`. DG enumerates/globs the parts and routes per-row via the
/// `split_col` column (Phase 1) — so routing must be correct ACROSS parts,
/// `split_col` dropped from all seed parts, and row counts == sum across parts.
#[test]
fn e2e_custom_tt_manual_split_column_multi_part_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    // The custom-TT "source" is now a DIRECTORY of parquet parts. Split the
    // same 6 logical rows as the single-part test across TWO parts so we
    // exercise multi-part enumeration + cross-part split routing.
    let src_dir = dir.path().join("custom_tt_dir");
    fs::create_dir_all(&src_dir).unwrap();

    // Part 0: 4 rows. split_col [0,1,2,0] → train=2, val=1, test=1.
    synth_target_only_parquet_with_split(
        &src_dir.join("part-00000.parquet"),
        &[0, 1, 2, 3],
        &[1_000_000, 1_000_000, 1_000_000, 1_000_000],
        &[0, 1, 2, 0],
    );
    // Part 1: 2 rows. split_col [1,255] → val=1, plus a 255 "unused" row
    // (matches no value set → dropped from all stages).
    synth_target_only_parquet_with_split(
        &src_dir.join("part-00001.parquet"),
        &[4, 0],
        &[1_000_000, 2_000_000],
        &[1, 255],
    );

    let plan = DiskGraphTTPlan::<Seconds> {
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
        split: Some(manual_split_col_spec()),
        stages: vec![],
        target_stype: Some("numerical".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: None,
        lp_rhs_is_temporal: false,
        fact_pseudo_time: None,
    };

    let out_dir = tempfile::tempdir().unwrap();
    // Pass the DIRECTORY (not a single part) as the custom-TT source.
    let custom = CustomTtSource {
        parquet_path: src_dir.clone(),
        input_timestamp_unit: DGTimeUnit::Nanoseconds,
    };
    let output = execute_tt_pass_with_source(&plan, &index, out_dir.path(), Some(custom)).unwrap();

    // Row count == sum across both parts (4 + 2 = 6).
    assert_eq!(output.rows_written, 6);

    // Cross-part split routing: combined split_col [0,1,2,0] + [1,255] →
    // train=2, val=2 (1 from each part), test=1, unused=1.
    assert_stage_infos_consistent(&output, out_dir.path());
    assert_eq!(stage_row_count(&output, "train"), 2);
    assert_eq!(stage_row_count(&output, "val"), 2);
    assert_eq!(stage_row_count(&output, "test"), 1);
    let tier_sum = stage_row_count(&output, "train")
        + stage_row_count(&output, "val")
        + stage_row_count(&output, "test");
    // The 255-valued row matched no value set → dropped from all stages.
    assert_eq!(output.rows_written - tier_sum, 1);

    // split_col must be DROPPED from every seed part across all stages.
    let infos = output.stage_gen_infos.as_ref().unwrap();
    for info in infos {
        for part in &info.parts {
            let file = fs::File::open(out_dir.path().join(part)).unwrap();
            let schema = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .schema()
                .clone();
            assert!(
                schema.index_of("split_col").is_err(),
                "split_col must be dropped from seed part {part}"
            );
        }
    }

    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.split_mode, "none");
    assert_eq!(manifest.source_mode, "custom_tt");
    assert!(manifest.stage_counts.is_none());
    assert!(!out_dir.path().join("split.bin").exists());
}

// ─── Phase 11: multi-entity / multi-target full-pipeline e2e ────────────────

#[test]
fn e2e_multi_entity_multi_target_pipeline() {
    use diskgraph_train_table::ttgen_executor::execute_tt_pass_with_customizer;
    use diskgraph_train_table::ttgen_resolve::ResolvedPlan;

    let dir = tempfile::tempdir().unwrap();
    let index = build_users_only_graph(dir.path(), 5);

    // Synthesize a parquet with ENTITY1, ENTITY2, TIMESTAMP, TARGET1, TARGET2.
    let pq = dir.path().join("custom_tt.parquet");
    let n_rows = 6;
    let e1_pks: Vec<i64> = (0..n_rows as i64).map(|i| i % 5).collect();
    let e2_pks: Vec<i64> = (0..n_rows as i64).map(|i| 100 + i).collect();
    let ts_secs: Vec<i64> = (0..n_rows as i64).map(|i| 1_000_000 + i * 1000).collect();
    let targets1: Vec<f64> = (0..n_rows as i64).map(|i| 1.0 + i as f64).collect();
    let targets2: Vec<f64> = (0..n_rows as i64).map(|i| 100.0 + i as f64).collect();

    let mut e1 = Int64Builder::new();
    let mut e2 = Int64Builder::new();
    let mut t1 = Float64Builder::new();
    let mut t2 = Float64Builder::new();
    for i in 0..n_rows {
        e1.append_value(e1_pks[i]);
        e2.append_value(e2_pks[i]);
        t1.append_value(targets1[i]);
        t2.append_value(targets2[i]);
    }
    let ts_arr = TimestampNanosecondArray::from(
        ts_secs
            .iter()
            .map(|s| s * 1_000_000_000)
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("ENTITY2", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, true),
        Field::new("TARGET2", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(e1.finish()),
            Arc::new(e2.finish()),
            Arc::new(ts_arr),
            Arc::new(t1.finish()),
            Arc::new(t2.finish()),
        ],
    )
    .unwrap();
    write_parquet(&pq, &batch);

    let plan = DiskGraphTTPlan::<Seconds> {
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
    };

    let out_dir = tempfile::tempdir().unwrap();
    let custom = CustomTtSource {
        parquet_path: pq.clone(),
        input_timestamp_unit: DGTimeUnit::Nanoseconds,
    };
    let customizer = |resolved: &mut ResolvedPlan<Seconds>| {
        // Add ENTITY2 slot.
        resolved
            .entity_pk_col_indices
            .push(resolved.entity_pk_col_indices[0]);
        resolved
            .entity_dtypes
            .push(resolved.entity_dtypes[0].clone());
        // Add TARGET2 slot.
        let next = resolved.target_alias_indices.last().copied().unwrap_or(0) + 1;
        resolved.target_alias_indices.push(next);
        resolved.target_dtypes.push(DataType::Float64);
    };
    let output = execute_tt_pass_with_customizer(
        &plan,
        &index,
        out_dir.path(),
        Some(custom),
        Some(customizer),
    )
    .unwrap();

    assert_eq!(output.rows_written, n_rows as u64);

    // Read tt.parquet — verify all five payload columns are present:
    // ENTITY1, ENTITY2, TIMESTAMP, TARGET1, TARGET2.
    let parquet_rows = parquet_row_count(&output.tt_parquet_path);
    assert_eq!(parquet_rows, n_rows as u64);
    let first_part = diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(
        &output.tt_parquet_path,
    )
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let file = fs::File::open(first_part).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    assert!(!batches.is_empty());
    let schema = batches[0].schema();
    assert!(schema.index_of("ENTITY1").is_ok());
    assert!(schema.index_of("ENTITY2").is_ok());
    assert!(schema.index_of("TIMESTAMP").is_ok());
    assert!(schema.index_of("TARGET1").is_ok());
    assert!(schema.index_of("TARGET2").is_ok());

    // Verify per-row payload integrity.
    let e1_idx = schema.index_of("ENTITY1").unwrap();
    let e2_idx = schema.index_of("ENTITY2").unwrap();
    let t1_idx = schema.index_of("TARGET1").unwrap();
    let t2_idx = schema.index_of("TARGET2").unwrap();
    let mut got_rows: Vec<(i64, i64, f64, f64)> = Vec::new();
    for batch in &batches {
        let e1c = batch
            .column(e1_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let e2c = batch
            .column(e2_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let t1c = batch
            .column(t1_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let t2c = batch
            .column(t2_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            got_rows.push((e1c.value(i), e2c.value(i), t1c.value(i), t2c.value(i)));
        }
    }
    got_rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut expected: Vec<(i64, i64, f64, f64)> = (0..n_rows)
        .map(|i| (e1_pks[i], e2_pks[i], targets1[i], targets2[i]))
        .collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(got_rows.len(), expected.len());
    for ((g_e1, g_e2, g_t1, g_t2), (e_e1, e_e2, e_t1, e_t2)) in got_rows.iter().zip(expected.iter())
    {
        assert_eq!(g_e1, e_e1);
        assert_eq!(g_e2, e_e2);
        assert!((g_t1 - e_t1).abs() < 1e-9);
        assert!((g_t2 - e_t2).abs() < 1e-9);
    }

    // Manifest is well-formed.
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.source_mode, "custom_tt");
    assert_eq!(manifest.split_mode, "none");
    assert_eq!(manifest.row_count, output.rows_written);
    assert_artifact_paths_exist(&manifest, out_dir.path());

    // No split spec → no stage_gen_infos and no per-stage seed parquets.
    assert!(output.stage_gen_infos.is_none());
    assert!(!out_dir.path().join("split.bin").exists());
}

// ─── Phase 12: ColumnSplit end-to-end ──────────────────────────────────────

#[test]
fn e2e_column_split_manual_full_pipeline() {
    // Phase 12 (α): ColumnSplit + manual_*_values on the users.user_id
    // column (Int64, but matched by string repr). Asserts the full
    // pipeline produces per-stage seed parquets with row counts consistent
    // with the empirical histogram.
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("ColumnSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: Some("user_id".to_string()),
        split_output_column: None,
        manual_train_values: Some(vec!["0".to_string(), "1".to_string()]),
        manual_val_values: Some(vec!["2".to_string()]),
        manual_test_values: Some(vec!["3".to_string()]),
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
        // user_id=4 (eve) should land UNUSED.
    });

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();
    assert_stage_infos_consistent(&output, out_dir.path());

    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.split_kind.as_deref(), Some("ColumnSplit"));
    assert_eq!(manifest.split_mode, "none");
    assert!(manifest.stage_counts.is_none());

    // Each user contributes 3 anchors (1000/2000/3000). With manual
    // assignments user 0,1 → train; user 2 → val; user 3 → test;
    // user 4 → unused. Three rows per user = 6 train, 3 val, 3 test,
    // 3 unused — total 15. The unused rows land in no seed parquet.
    assert_eq!(stage_row_count(&output, "train"), 6, "expected 6 train rows");
    assert_eq!(stage_row_count(&output, "val"), 3, "expected 3 val rows");
    assert_eq!(stage_row_count(&output, "test"), 3, "expected 3 test rows");
    let tier_sum = stage_row_count(&output, "train")
        + stage_row_count(&output, "val")
        + stage_row_count(&output, "test");
    assert_eq!(
        output.rows_written - tier_sum,
        3,
        "expected 3 unused rows (in no seed parquet)"
    );
}

#[test]
fn e2e_column_split_ratio_numeric_full_pipeline() {
    // Phase 12 (γ): ColumnSplit + ratio on the users.user_id Int64 column.
    // Pass A's ColumnDeciderRatioNumeric accumulates a t-digest over the
    // projected user_id values and `derive_cutoffs` quantizes to the
    // ratio thresholds. Asserts per-stage seed parquets emerge with row
    // counts consistent with the empirical histogram.
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: Some(vec![0.6, 0.4]),
        kind: Some("ColumnSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: Some("user_id".to_string()),
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    });

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();
    assert_stage_infos_consistent(&output, out_dir.path());

    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.split_kind.as_deref(), Some("ColumnSplit"));
    assert_eq!(manifest.split_mode, "none");
    assert!(manifest.stage_counts.is_none());

    // Total rows = 5 users × 3 anchors = 15. Ratio 0.6/0.4 should put
    // most rows in train (60%) and the rest in val. The t-digest's
    // approximation tolerance + the quantile-via-public-api shim's
    // anchor approximation make the exact count fuzzy on a 5-user
    // fixture, so we just require: train is non-empty and the tier sum
    // plus unused covers everything (asserted by the consistency helper).
    assert!(
        stage_row_count(&output, "train") > 0,
        "train must be non-empty"
    );
}

// ─── Test 8: Stage count consistency across split families ──────────────────

#[test]
fn e2e_stage_count_consistency_across_families() {
    // For each SplitMode::Compute family, assert stage_gen_infos row counts
    // agree with the per-stage seed parquets on disk.
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());

    struct Case<'a> {
        name: &'a str,
        spec: SplitSpec<Seconds>,
    }
    let cases: Vec<Case> = vec![
        Case {
            name: "TimeRangeSplit",
            spec: time_range_split_spec(vec![(0, 1500), (1501, 2500), (2501, 4000)]),
        },
        Case {
            name: "BacktestingSplit",
            spec: SplitSpec::<Seconds> {
                ratio: None,
                kind: Some("BacktestingSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
                offsets: None,
                offset_unit: None,
                unit: None,
                timeframe: None,
                anchor_start_offset: None,
                ranges: None,
                backtesting_timestamp: Some(Abs::<Seconds>::from_raw(2000)),
                split_column: None,
                split_output_column: None,
                manual_train_values: None,
                manual_val_values: None,
                manual_test_values: None,
                month_offsets: None,
                flexible_boundaries: None,
                timeframe_step: None,
            },
        },
        Case {
            name: "ManualSplit",
            spec: SplitSpec::<Seconds> {
                ratio: None,
                kind: Some("ManualSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
                offsets: None,
                offset_unit: None,
                unit: None,
                timeframe: None,
                anchor_start_offset: None,
                ranges: None,
                backtesting_timestamp: None,
                split_column: Some("TIMESTAMP".to_string()),
                split_output_column: None,
                manual_train_values: Some(vec!["1000".to_string()]),
                manual_val_values: Some(vec!["2000".to_string()]),
                manual_test_values: Some(vec!["3000".to_string()]),
                month_offsets: None,
                flexible_boundaries: None,
                timeframe_step: None,
            },
        },
        Case {
            name: "RandomSplit",
            spec: SplitSpec::<Seconds> {
                ratio: Some(vec![0.7, 0.2, 0.1]),
                kind: Some("RandomSplit".to_string()),
                rnd_seed: Some(42),
                is_temporal: Some(true),
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
            },
        },
        Case {
            name: "FixedSplit",
            spec: SplitSpec::<Seconds> {
                ratio: Some(vec![0.7, 0.2, 0.1]),
                kind: Some("FixedSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
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
            },
        },
        Case {
            name: "TemporalSplit",
            spec: SplitSpec::<Seconds> {
                ratio: Some(vec![0.5, 0.5]),
                kind: Some("TemporalSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
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
            },
        },
        Case {
            name: "DateOffsetSplit",
            spec: SplitSpec::<Seconds> {
                ratio: None,
                kind: Some("DateOffsetSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
                offsets: Some(vec![
                    diskgraph_core::time::Dur::<Seconds>::from_raw(-1000),
                    diskgraph_core::time::Dur::<Seconds>::from_raw(0),
                ]),
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
            },
        },
        Case {
            name: "ApproxDateOffsetSplit",
            spec: SplitSpec::<Seconds> {
                ratio: Some(vec![0.5, 0.3, 0.2]),
                kind: Some("ApproxDateOffsetSplit".to_string()),
                rnd_seed: None,
                is_temporal: Some(true),
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
            },
        },
    ];

    for case in cases {
        let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
        plan.split = Some(case.spec);
        let out_dir = tempfile::tempdir().unwrap();
        let output = execute_tt_pass(&plan, &index, out_dir.path())
            .unwrap_or_else(|e| panic!("[{}] execute_tt_pass failed: {e}", case.name));
        let infos = output
            .stage_gen_infos
            .as_ref()
            .unwrap_or_else(|| panic!("[{}] stage_gen_infos not produced", case.name));
        // Three stages train/val/test, each seed parquet on disk with a row
        // count matching its StageGenInfo, and the tier sum <= rows_written
        // (the remainder, if any, is `unused`).
        assert_stage_infos_consistent(&output, out_dir.path());
        let keys: Vec<&str> = infos.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["train", "val", "test"],
            "[{}] stage keys must be train/val/test",
            case.name
        );
        for stage in infos.iter().filter(|stage| stage.row_count > 0) {
            let bounds = stage.seed_time.unwrap_or_else(|| {
                panic!(
                    "[{}] non-empty stage {} must carry authoritative time bounds",
                    case.name, stage.key
                )
            });
            assert!(bounds[0] <= bounds[1]);
        }
        let stats: serde_json::Value =
            serde_json::from_slice(&fs::read(out_dir.path().join("stats.json")).unwrap()).unwrap();
        let stats = stats.as_object().unwrap();
        for removed in [
            "payload",
            "timestamp_counts",
            "stats_per_timeframe",
            "timestamp_stage_assignments",
            "kumo_contract",
        ] {
            assert!(
                !stats.contains_key(removed),
                "[{}] bounded stats must omit {removed}",
                case.name
            );
        }
        let manifest = read_manifest(&output.manifest_path).unwrap();
        // Flat-manifest split fields stay None on the TT-NG path.
        assert!(
            manifest.stage_counts.is_none(),
            "[{}] manifest.stage_counts must be None",
            case.name
        );
    }
}

// ─── Reusable readers for tt.parquet (used by multiple tests) ──────────────

struct TTParquetCols {
    column_names: Vec<String>,
    /// `ENTITY_INCLUDE_BIT` column, if present.
    include_bits: Option<Vec<bool>>,
}

fn read_tt_parquet_columns(path: &Path) -> TTParquetCols {
    let mut column_names: Vec<String> = Vec::new();
    let mut include_bits: Option<Vec<bool>> = None;
    for part in diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(path).unwrap() {
        let reader = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(part).unwrap())
        .unwrap()
        .build()
        .unwrap();
    for batch_res in reader {
        let batch = batch_res.unwrap();
        if column_names.is_empty() {
            column_names = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect();
        }
        if let Ok(idx) = batch.schema().index_of("ENTITY_INCLUDE_BIT") {
            let arr = batch
                .column(idx)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            let mut v = include_bits.unwrap_or_default();
            for i in 0..arr.len() {
                v.push(arr.value(i));
            }
            include_bits = Some(v);
        }
    }
    }
    TTParquetCols {
        column_names,
        include_bits,
    }
}

fn parquet_row_count(path: &Path) -> u64 {
    let parts = if path.is_dir() {
        diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(path).unwrap()
    } else {
        vec![path.to_path_buf()]
    };
    parts
        .into_iter()
        .map(|part| {
            ParquetRecordBatchReaderBuilder::try_new(fs::File::open(part).unwrap())
        .unwrap()
                .metadata()
                .file_metadata()
                .num_rows() as u64
        })
        .sum()
}

// ─── Kumo-contract sidecar tests ─────────────────────────────────────────────

#[test]
fn e2e_train_table_col_stats_emitted_when_payload_carries_data() {
    // Phase 17 also re-enables train_table_col_stats.json via the
    // bridge. The pass-A inline col-stats accumulator already populates
    // `parsed_train_table_col_stats` for any plan with a TARGET1
    // column; the bridge forwards it to disk in the kumo-expected
    // envelope.
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(time_range_split_spec(vec![
        (0, 1500),
        (1501, 2500),
        (2501, 4000),
    ]));

    let out_dir = tempfile::tempdir().unwrap();
    let output =
        diskgraph_train_table::ttgen_executor::execute_tt_pass(&plan, &index, out_dir.path())
            .unwrap();

    let cs_path = out_dir.path().join("train_table_col_stats.json");
    if !cs_path.exists() {
        // The synthetic e2e graph may not produce non-empty col-stats
        // (e.g. PassThrough mode); skip silently — manifest field will
        // be absent in that case.
        let manifest = read_manifest(&output.manifest_path).unwrap();
        assert!(manifest.train_table_col_stats.is_none());
        return;
    }

    let raw = fs::read_to_string(&cs_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

    // Kumo's `parse_train_table_col_stats_artifact` validates these:
    assert_eq!(v["version"].as_u64(), Some(1));
    assert_eq!(v["kind"].as_str(), Some("tt_train_table_col_stats"));
    assert_eq!(v["stats_format"].as_str(), Some("kumo_col_stats_v1"));
    let stats = v["stats"].as_object().expect("stats object");
    for (key, value) in stats {
        let parts: Vec<&str> = key.rsplitn(4, "__").collect();
        assert_eq!(parts.len(), 4, "stat key {key:?} must split 4-way on __");
        let entry = value
            .as_object()
            .unwrap_or_else(|| panic!("entry {key:?} must be object"));
        assert!(entry.contains_key("value"), "{key:?} missing `value`");
    }

    // Manifest carries the ArtifactRef + checksum.
    let manifest = read_manifest(&output.manifest_path).unwrap();
    let cs_ref = manifest.train_table_col_stats.as_ref().expect("set");
    assert_eq!(cs_ref.format, "tt_train_table_col_stats_json");
    let actual = sha256_file(&cs_path).unwrap();
    assert_eq!(
        manifest.checksums.train_table_col_stats_sha256.as_ref(),
        Some(&actual)
    );
}

#[test]
fn e2e_split_stats_emitted_when_split_computed() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let mut plan = anchored_node_plan(vec![1000, 2000, 3000]);
    plan.split = Some(time_range_split_spec(vec![
        (0, 1500),
        (1501, 2500),
        (2501, 4000),
    ]));

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();
    assert_stage_infos_consistent(&output, out_dir.path());

    let split_stats_path = out_dir.path().join("split_stats.json");
    assert!(
        split_stats_path.exists(),
        "split_stats.json should be written when split stage parts are produced"
    );

    let raw = fs::read_to_string(&split_stats_path).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(payload["version"].as_u64(), Some(1));
    assert_eq!(payload["kind"].as_str(), Some("tt_split_stats"));

    let train = payload["train_table_stats"]
        .as_object()
        .expect("train_table_stats object");
    assert_eq!(
        train.get("num_train_rows").and_then(|v| v.as_u64()),
        Some(stage_row_count(&output, "train")),
    );
    assert!(train.get("num_entities").and_then(|v| v.as_u64()).is_some());
    assert!(train.get("num_targets").and_then(|v| v.as_u64()).is_some());
    assert!(
        train
            .get("target_distribution")
            .and_then(|v| v.get("histogram"))
            .is_some(),
        "numerical split stats should include a histogram"
    );

    for (stage_key, payload_key) in [("val", "val_table_stats"), ("test", "test_table_stats")] {
        let row_count = stage_row_count(&output, stage_key);
        if row_count == 0 {
            assert!(
                payload[payload_key].is_null(),
                "{payload_key} should be null"
            );
        } else {
            assert_eq!(
                payload[payload_key]["num_train_rows"].as_u64(),
                Some(row_count),
                "{payload_key} row count should match stage_gen_infos"
            );
        }
    }

    let manifest = read_manifest(&output.manifest_path).unwrap();
    let split_ref = manifest.split_stats.as_ref().expect("split_stats set");
    assert_eq!(split_ref.format, SPLIT_STATS_FORMAT);
    assert_eq!(out_dir.path().join(&split_ref.path), split_stats_path);
    let actual = sha256_file(&split_stats_path).unwrap();
    assert_eq!(
        manifest.checksums.split_stats_sha256.as_ref(),
        Some(&actual)
    );
}

#[test]
fn e2e_split_stats_absent_when_no_split() {
    // Plan with no split → SplitMode::None → no fused pass-B → no
    // split_stats.json on disk and manifest.split_stats == None.
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());
    let plan = anchored_node_plan(vec![1000, 2000, 3000]);

    let out_dir = tempfile::tempdir().unwrap();
    let output =
        diskgraph_train_table::ttgen_executor::execute_tt_pass(&plan, &index, out_dir.path())
            .unwrap();

    // No split → no stage_gen_infos, no split_stats.json.
    assert!(output.stage_gen_infos.is_none());
    let split_stats_path = out_dir.path().join("split_stats.json");
    assert!(
        !split_stats_path.exists(),
        "split_stats.json should NOT be written when no split is requested"
    );

    // Manifest: split_stats absent.
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert!(
        manifest.split_stats.is_none(),
        "manifest.split_stats must be None when no split"
    );
    assert!(manifest.checksums.split_stats_sha256.is_none());

    // The on-disk JSON shouldn't even mention split_stats key.
    let raw = fs::read_to_string(&output.manifest_path).unwrap();
    assert!(
        !raw.contains("\"split_stats\""),
        "manifest JSON should elide split_stats key"
    );
}

// ─── Phase 13: tighter quantile + calendar-month e2e ───────────────────────

/// Phase 13 — TemporalDecider now queries `TDigestEstimator::quantile_at`
/// directly (no more workaround interpolating between q25/q50/q75). The
/// resulting cutoff for a uniform distribution should be much tighter than
/// the legacy ~1-2% absolute error. We assert ≤0.5% absolute error of the
/// empirical median across the full TT pipeline.
#[test]
fn e2e_temporal_split_uses_tighter_quantiles() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());

    // Anchor 200 timestamps spread over a day. With 200 samples uniform
    // over the range, q=0.5 should land near the empirical median to
    // <1% relative error using direct `quantile_at` queries.
    let timestamps: Vec<i64> = (0..200).map(|i| 1000 + i * 100).collect();
    let mut plan = anchored_node_plan(timestamps.clone());
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: Some(vec![0.5, 0.5]),
        kind: Some("TemporalSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
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
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();
    assert_stage_infos_consistent(&output, out_dir.path());

    // Roughly half train, half val — from the per-stage seed parquet counts.
    let total = output.rows_written as i64;
    let stage0 = stage_row_count(&output, "train") as i64;
    // |stage0 - total/2| ≤ 0.5% of total.
    let drift = (stage0 - total / 2).abs();
    let allowed = (total as f64 * 0.005).ceil() as i64;
    assert!(
        drift <= allowed.max(2),
        "TemporalSplit train-stage drift {drift} > allowed {allowed} (total={total})"
    );
}

/// Full pipeline with `DateOffsetSplit` + calendar-month offsets: the
/// SplitSpec.month_offsets path through executor → DateOffsetDecider → stages.
#[test]
fn e2e_date_offset_calendar_months_e2e() {
    use chrono::NaiveDate;
    let dir = tempfile::tempdir().unwrap();
    let index = build_e2e_graph(dir.path());

    // Anchor timestamps over Dec-Mar. With month_offsets = [-2, -1]
    // and max_ts in mid-March, boundaries land at mid-January and mid-
    // February. The December history leaves the snapped train window at
    // least one inferred monthly timeframe wide, matching Kumo's
    // flexible-boundary validity requirement.
    let ymd = |y, m, d| {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp()
    };
    let timestamps = vec![
        ymd(2023, 12, 5),
        ymd(2024, 1, 5),
        ymd(2024, 1, 20),
        ymd(2024, 2, 5),
        ymd(2024, 2, 20),
        ymd(2024, 3, 5),
        ymd(2024, 3, 20),
    ];
    let mut plan = anchored_node_plan(timestamps);
    plan.split = Some(SplitSpec::<Seconds> {
        ratio: None,
        kind: Some("DateOffsetSplit".to_string()),
        rnd_seed: None,
        is_temporal: Some(true),
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
        month_offsets: Some(vec![-2, -1]),
        flexible_boundaries: None,
        timeframe_step: None,
    });

    let out_dir = tempfile::tempdir().unwrap();
    let output = execute_tt_pass(&plan, &index, out_dir.path()).unwrap();

    // 1) Per-stage seed parquets emitted; row counts agree on disk.
    assert_stage_infos_consistent(&output, out_dir.path());

    // 2) Manifest reports DateOffsetSplit; flat split fields stay None.
    let manifest = read_manifest(&output.manifest_path).unwrap();
    assert_eq!(manifest.split_kind.as_deref(), Some("DateOffsetSplit"));
    assert!(manifest.stage_counts.is_none());

    // 3) Calendar arithmetic engaged: train+val+test > 0 (some rows may be
    // `unused` outside the boundary window — expected for DateOffsetSplit).
    let train = stage_row_count(&output, "train");
    let val = stage_row_count(&output, "val");
    let test = stage_row_count(&output, "test");
    assert!(
        train + val + test > 0,
        "[calendar months] no rows assigned to any stage: train={train} val={val} test={test}"
    );

    // 5) At least one of {val, test} is populated, proving calendar arithmetic
    // moved boundaries off the all-fixed fallback (which would collapse to
    // all-train at Dur::zero()).
    assert!(
        val + test > 0,
        "[calendar months] expected non-empty val or test stage; got val={val} test={test}"
    );
}

fn read_node_rows(path: &Path) -> Vec<(i64, i64, f64)> {
    let mut out = Vec::new();
    for part in diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(path).unwrap() {
        let reader = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(part).unwrap())
        .unwrap()
        .build()
        .unwrap();
    for batch_res in reader {
        let batch = batch_res.unwrap();
        let schema = batch.schema();
        let entity_idx = schema.index_of("ENTITY1").unwrap_or(0);
        let ts_idx = schema.index_of("TIMESTAMP").unwrap_or(1);
        let target_idx = schema.index_of("TARGET1").unwrap_or(2);
        let entities = batch
            .column(entity_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(ts_idx)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let targets = batch
            .column(target_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((
                entities.value(i),
                ts.value(i) / 1_000_000_000,
                targets.value(i),
            ));
        }
    }
    }
    out
}

/// Asserts every `ArtifactRef` / `SidecarRef` path in the manifest
/// exists on disk.
fn assert_artifact_paths_exist(m: &TTManifest, bundle_dir: &Path) {
    let check = |label: &str, p: &str| {
        assert!(
            bundle_dir.join(p).exists(),
            "manifest.{label} path does not exist: {p}"
        );
    };
    check("train_table", &m.train_table.path);
    check("stats", &m.stats.path);
    if let Some(s) = m.split.as_ref() {
        check("split", &s.path);
    }
    if let Some(s) = m.split_stats.as_ref() {
        check("split_stats", &s.path);
    }
    if let Some(s) = m.train_table_col_stats.as_ref() {
        check("train_table_col_stats", &s.path);
    }
    if let Some(s) = m.arange_sidecar.as_ref() {
        check("arange_sidecar", &s.path);
    }
}

// ─── Coverage notes ─────────────────────────────────────────────────────────
//
// We deliberately exercise the eight cases the Phase 10 spec calls out:
//
//   Test 1: PQ + TimeRangeSplit (`e2e_pq_time_range_split_full_pipeline`).
//   Test 2: PQ no split          (`e2e_pq_no_split_omits_split_bin`).
//   Test 3: Custom-TT passthrough (`e2e_custom_tt_passthrough_full_pipeline`).
//   Test 4: LP+array+RandomLink   (`e2e_lp_array_random_link_split_regroup`).
//   Test 5: Manifest contract     (`e2e_manifest_kumo_reader_contract`).
//   Test 6: include_mask invariant (`e2e_include_mask_forces_unused_across_phases`).
//   Test 7: Custom-TT split column (`e2e_custom_tt_manual_split_column_full_pipeline`).
//   Test 8: stage count drift     (`e2e_stage_count_consistency_across_families`).
//
// Test 8 is itself a parameterized sweep over the eight families that
// engage SplitMode::Compute, so the file effectively tests
// {TimeRangeSplit, BacktestingSplit, ManualSplit, RandomSplit, FixedSplit,
// TemporalSplit, DateOffsetSplit, ApproxDateOffsetSplit} on a single
// invariant — the pass-B accumulator agreeing with the file.
//
// What we don't test here:
// - `split_stats` shape (currently `None` in execute_tt_pass).
// - Edge artifact bundle (separate `export_tt_artifacts` path).
// - CustomLinkSplit regroup (covered by an existing integration test;
//   uses a different fixture shape).
//
// `_` to silence the unused-import warning when this file is compiled
// in isolation (some imports above are conditional on which tests run).
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = ArtifactRef::new("", "");
    let _ = SidecarRef { path: "".into() };
    let _ = ListArray::from(arrow_array::Array::to_data(&Int64Array::from(
        Vec::<i64>::new(),
    )));
    let _: TTPassOutput;
}
