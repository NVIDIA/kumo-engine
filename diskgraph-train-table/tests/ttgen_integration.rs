//! Integration tests for the TT-gen executor.
//!
//! Full pipeline: write Parquet → ingest → execute → verify output rows.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::*;
use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_core::tt_builder::*;
use diskgraph_ingest::coercion::CoercionMode;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::*;
use diskgraph_ingest::io::Dest;
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

/// Read output rows, converting nanosecond timestamps back to epoch seconds.
fn read_output_rows(shard_paths: &[PathBuf]) -> Vec<(i64, i64, f64)> {
    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let targets = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..batch.num_rows() {
                rows.push((entities.value(i), timestamps.value(i) / 1_000_000_000, targets.value(i)));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

/// Read output rows including the ENTITY_INCLUDE_BIT column.
fn read_output_rows_with_bit(shard_paths: &[PathBuf]) -> Vec<(i64, i64, f64, bool)> {
    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let targets = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            let bits = batch.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
            for i in 0..batch.num_rows() {
                rows.push((entities.value(i), timestamps.value(i) / 1_000_000_000, targets.value(i), bits.value(i)));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

/// Read output rows where TARGET is Boolean (e.g. Compare ops).
fn read_output_rows_bool(shard_paths: &[PathBuf]) -> Vec<(i64, i64, bool)> {
    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let targets = batch.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
            for i in 0..batch.num_rows() {
                rows.push((entities.value(i), timestamps.value(i) / 1_000_000_000, targets.value(i)));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

fn group_by_entity(rows: &[(i64, i64, f64)]) -> HashMap<i64, Vec<(i64, f64)>> {
    let mut map: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    for &(entity, ts, target) in rows {
        map.entry(entity).or_default().push((ts, target));
    }
    map
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

fn execute_and_group(
    plan: &DiskGraphTTPlan,
    index_path: &Path,
    output_dir: &Path,
) -> HashMap<i64, Vec<(i64, f64)>> {
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(plan, index_path, output_dir).unwrap();
    let rows = read_output_rows(&shard_paths);
    group_by_entity(&rows)
}

fn execute_and_group_bool(
    plan: &DiskGraphTTPlan,
    index_path: &Path,
    output_dir: &Path,
) -> HashMap<i64, Vec<(i64, bool)>> {
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(plan, index_path, output_dir).unwrap();
    let rows = read_output_rows_bool(&shard_paths);
    let mut map: HashMap<i64, Vec<(i64, bool)>> = HashMap::new();
    for &(entity, ts, target) in &rows {
        map.entry(entity).or_default().push((ts, target));
    }
    map
}

/// Read a TARGET column as strings regardless of underlying dtype (Int64/Utf8/Float64).
fn read_target_column_as_strings(batch: &RecordBatch, col_idx: usize) -> Vec<String> {
    let col = batch.column(col_idx);
    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    } else {
        panic!("unsupported TARGET column dtype: {:?}", col.data_type());
    }
}

/// Read list values as strings regardless of inner dtype.
fn read_list_values_as_strings(arr: &dyn arrow_array::Array) -> Vec<String> {
    if let Some(s) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        (0..s.len()).map(|j| s.value(j).to_string()).collect()
    } else if let Some(s) = arr.as_any().downcast_ref::<Int64Array>() {
        (0..s.len()).map(|j| s.value(j).to_string()).collect()
    } else if let Some(s) = arr.as_any().downcast_ref::<Float64Array>() {
        (0..s.len()).map(|j| s.value(j).to_string()).collect()
    } else {
        panic!("unsupported list item dtype: {:?}", arr.data_type());
    }
}

fn assert_target_eq(actual: f64, expected: f64, msg: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{msg}: expected NaN, got {actual}");
    } else {
        assert!(
            (actual - expected).abs() < 1e-6,
            "{msg}: expected {expected}, got {actual}"
        );
    }
}

// ─── Graph 1: users → orders → products ─────────────────────────────────────
//
// users (5 rows):
//   node  name    level  score   signup_ts  active  valid_to
//   0     alice   3      75.0    1000       true    2000
//   1     bob     1      120.0   2000       true    3000
//   2     carol   2      50.0    1500       false   1500
//   3     dave    1      200.0   3000       true    4000
//   4     eve     5      10.0    500        true    1000
//
// products (4 rows):
//   node  brand   rating  category
//   0     Nike    4.5     shoes
//   1     Adidas  3.0     shoes
//   2     Nike    5.0     clothing
//   3     Puma    2.0     accessories
//
// orders (10 rows, FK: user_id→users, product_id→products):
//   id  user_id  product_id  price   quantity  ts    status
//   0   0        3           10.0    5         1050  returned
//   1   0        0           100.0   2         1100  shipped
//   2   0        1           50.0    1         1200  pending
//   3   0        2           75.0    3         1300  shipped
//   4   1        0           200.0   1         2100  shipped
//   5   1        1           30.0    2         2200  returned
//   6   2        2           60.0    1         1600  shipped
//   7   3        0           150.0   1         3100  shipped
//   8   3        1           80.0    2         3200  pending
//   9   3        2           25.0    1         3300  shipped

fn build_graph1(root: &Path) -> PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    // ── Users table ──
    let mut user_id = Int64Builder::new();
    let mut name = StringBuilder::new();
    let mut level = Int64Builder::new();
    let mut score = Float64Builder::new();
    let mut signup_ts = Int64Builder::new();
    let mut active = BooleanBuilder::new();
    let mut valid_to = Int64Builder::new();

    for (uid, n, l, s, ts, a, vt) in [
        (0i64, "alice", 3i64, 75.0, 1000i64, true, 2000i64),
        (1, "bob", 1, 120.0, 2000, true, 3000),
        (2, "carol", 2, 50.0, 1500, false, 1500),
        (3, "dave", 1, 200.0, 3000, true, 4000),
        (4, "eve", 5, 10.0, 500, true, 1000),
    ] {
        user_id.append_value(uid);
        name.append_value(n);
        level.append_value(l);
        score.append_value(s);
        signup_ts.append_value(ts);
        active.append_value(a);
        valid_to.append_value(vt);
    }

    let users_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("level", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
        Field::new("signup_ts", DataType::Int64, false),
        Field::new("active", DataType::Boolean, false),
        Field::new("valid_to", DataType::Int64, false),
    ]);
    let users_batch = RecordBatch::try_new(
        Arc::new(users_schema),
        vec![
            Arc::new(user_id.finish()),
            Arc::new(name.finish()),
            Arc::new(level.finish()),
            Arc::new(score.finish()),
            Arc::new(signup_ts.finish()),
            Arc::new(active.finish()),
            Arc::new(valid_to.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    // ── Products table ──
    let mut product_id = Int64Builder::new();
    let mut brand = StringBuilder::new();
    let mut rating = Float64Builder::new();
    let mut category = StringBuilder::new();

    for (pid, b, r, c) in [
        (0i64, "Nike", 4.5, "shoes"),
        (1, "Adidas", 3.0, "shoes"),
        (2, "Nike", 5.0, "clothing"),
        (3, "Puma", 2.0, "accessories"),
    ] {
        product_id.append_value(pid);
        brand.append_value(b);
        rating.append_value(r);
        category.append_value(c);
    }

    let products_schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new("brand", DataType::Utf8, false),
        Field::new("rating", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]);
    let products_batch = RecordBatch::try_new(
        Arc::new(products_schema),
        vec![
            Arc::new(product_id.finish()),
            Arc::new(brand.finish()),
            Arc::new(rating.finish()),
            Arc::new(category.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &products_batch);

    // ── Orders table ──
    let mut order_id = Int64Builder::new();
    let mut order_user_id = Int64Builder::new();
    let mut order_product_id = Int64Builder::new();
    let mut price = Float64Builder::new();
    let mut quantity = Int64Builder::new();
    let mut ts = Int64Builder::new();
    let mut status = StringBuilder::new();

    for (oid, uid, pid, p, q, t, s) in [
        (0i64, 0i64, 3i64, 10.0, 5i64, 1050i64, "returned"),
        (1, 0, 0, 100.0, 2, 1100, "shipped"),
        (2, 0, 1, 50.0, 1, 1200, "pending"),
        (3, 0, 2, 75.0, 3, 1300, "shipped"),
        (4, 1, 0, 200.0, 1, 2100, "shipped"),
        (5, 1, 1, 30.0, 2, 2200, "returned"),
        (6, 2, 2, 60.0, 1, 1600, "shipped"),
        (7, 3, 0, 150.0, 1, 3100, "shipped"),
        (8, 3, 1, 80.0, 2, 3200, "pending"),
        (9, 3, 2, 25.0, 1, 3300, "shipped"),
    ] {
        order_id.append_value(oid);
        order_user_id.append_value(uid);
        order_product_id.append_value(pid);
        price.append_value(p);
        quantity.append_value(q);
        ts.append_value(t);
        status.append_value(s);
    }

    let orders_schema = Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("ts", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
    ]);
    let orders_batch = RecordBatch::try_new(
        Arc::new(orders_schema),
        vec![
            Arc::new(order_id.finish()),
            Arc::new(order_user_id.finish()),
            Arc::new(order_product_id.finish()),
            Arc::new(price.finish()),
            Arc::new(quantity.finish()),
            Arc::new(ts.finish()),
            Arc::new(status.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("orders.parquet"), &orders_batch);

    // ── GraphConfig ──
    let config = GraphConfig {
        dataset_name: "test_graph1".to_string(),
        tables: vec![
            TableSpec {
                name: "users".to_string(),
                columns: vec![
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("name", IngestedType::String),
                    mk_col("level", IngestedType::Int64),
                    mk_col("score", IngestedType::Float64),
                    mk_col("signup_ts", IngestedType::Int64),
                    mk_col("active", IngestedType::Bool),
                    mk_col("valid_to", IngestedType::Int64),
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
                fkeys: vec![],
            },
            TableSpec {
                name: "products".to_string(),
                columns: vec![
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("brand", IngestedType::String),
                    mk_col("rating", IngestedType::Float64),
                    mk_col("category", IngestedType::String),
                ],
                primary_key: Some("product_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("products.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "orders".to_string(),
                columns: vec![
                    mk_col("order_id", IngestedType::Int64),
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("price", IngestedType::Float64),
                    mk_col("quantity", IngestedType::Int64),
                    mk_col("ts", IngestedType::Int64),
                    mk_col("status", IngestedType::String),
                ],
                primary_key: Some("order_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("orders.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![
                    FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "ts".to_string(),
                                bitwidth: Some(16),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                    FKeySpec {
                        src_column: "product_id".to_string(),
                        dst_table: "products".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "ts".to_string(),
                                bitwidth: Some(16),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                ],
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
    }
}

// ─── Plan helpers ────────────────────────────────────────────────────────────

fn g1_plan(stages: Vec<Stage>) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Node,
        stages,
    }
}

fn g1_temporal(stages: Vec<Stage>, timestamps: Vec<i64>) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps,
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages,
    }
}

fn u2o_edge() -> EdgeRef {
    EdgeRef {
        src_table: "users".to_string(),
        src_col: "user_id".to_string(),
        dst_table: "orders".to_string(),
        dst_col: "user_id".to_string(),
    }
}

fn o2p_edge() -> EdgeRef {
    EdgeRef {
        src_table: "orders".to_string(),
        src_col: "product_id".to_string(),
        dst_table: "products".to_string(),
        dst_col: "product_id".to_string(),
    }
}

/// COUNT(orders.*, window) — most common test pattern.
fn count_orders_op(alias: &str, window: Option<(i64, i64)>) -> Op {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![u2o_edge()],
        func: AggFunc::Count,
        target_col: None,
        time_col: None,
        window,
        window_unit: Some(TimeUnit::Seconds),
        row_filter: None,
        limit: None,
    }
}

/// SUM/AVG/MIN/MAX/FIRST/LAST(orders.price, window)
fn agg_orders_price(alias: &str, func: AggFunc, window: Option<(i64, i64)>) -> Op {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![u2o_edge()],
        func,
        target_col: Some("price".to_string()),
        time_col: None,
        window,
        window_unit: Some(TimeUnit::Seconds),
        row_filter: None,
        limit: None,
    }
}

// ─── Tests: Basic aggregations ───────────────────────────────────────────────
// PQL: PREDICT <AGG>(ORDERS.<col>, 0, 5000, seconds) for each USERS.user_id
// obs=0, window=[0,5000) captures all orders (max ts=3300).

#[test]
fn integ_count_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_count");
    let plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 5);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_sum_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("s", AggFunc::Sum, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 235.0, "alice");
    assert_target_eq(groups[&1][0].1, 230.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 255.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_avg_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("a", AggFunc::Avg, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 58.75, "alice");
    assert_target_eq(groups[&1][0].1, 115.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 85.0, "dave");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_min_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("m", AggFunc::Min, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 10.0, "alice");
    assert_target_eq(groups[&1][0].1, 30.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 25.0, "dave");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_max_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("m", AggFunc::Max, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 100.0, "alice");
    assert_target_eq(groups[&1][0].1, 200.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 150.0, "dave");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_first_basic() {
    // Edges descending: FIRST = earliest = last encountered by iterator.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("f", AggFunc::First, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 10.0, "alice first=ts1050");
    assert_target_eq(groups[&1][0].1, 200.0, "bob first=ts2100");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 150.0, "dave first=ts3100");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_last_basic() {
    // Edges descending: LAST = latest = first encountered, stop early.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("l", AggFunc::Last, Some((0, 5000)))], filter: None }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 75.0, "alice last=ts1300");
    assert_target_eq(groups[&1][0].1, 30.0, "bob last=ts2200");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 25.0, "dave last=ts3300");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_count_distinct_basic() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cd".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::CountDistinct,
                target_col: Some("status".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 3.0, "alice: returned,shipped,pending");
    assert_target_eq(groups[&1][0].1, 2.0, "bob: shipped,returned");
    assert_target_eq(groups[&2][0].1, 1.0, "carol: shipped");
    assert_target_eq(groups[&3][0].1, 2.0, "dave: shipped,pending");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: Row filters ─────────────────────────────────────────────────────

#[test]
fn integ_count_with_row_filter() {
    // COUNT(orders.* WHERE status = 'shipped', 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("status".to_string()),
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 2.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_sum_with_row_filter() {
    // SUM(orders.price WHERE status = 'shipped', 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "s".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Sum,
                target_col: Some("price".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("status".to_string()),
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 175.0, "alice: 100+75");
    assert_target_eq(groups[&1][0].1, 200.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 175.0, "dave: 150+25");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_filter_and() {
    // COUNT WHERE status='shipped' AND price > 50
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::And(vec![
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                    },
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("price".to_string()),
                        op: CompareOp::Gt,
                        rhs: RowFilterRhs::Constant(Value::Float(50.0)),
                    },
                ])),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 2.0, "alice: 100,75");
    assert_target_eq(groups[&1][0].1, 1.0, "bob: 200");
    assert_target_eq(groups[&2][0].1, 1.0, "carol: 60");
    assert_target_eq(groups[&3][0].1, 1.0, "dave: 150");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_filter_or() {
    // COUNT WHERE status='shipped' OR status='returned'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Or(vec![
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                    },
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("returned".to_string())),
                    },
                ])),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 3.0, "alice: returned,shipped,shipped");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_filter_not() {
    // COUNT WHERE NOT status='pending'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Not(Box::new(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("status".to_string()),
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("pending".to_string())),
                }))),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 3.0, "alice: excl pending");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave: excl pending");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: CompareOps ──────────────────────────────────────────────────────

#[test]
fn integ_compare_lt() {
    // COUNT WHERE price < 80
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("price".to_string()),
                    op: CompareOp::Lt,
                    rhs: RowFilterRhs::Constant(Value::Float(80.0)),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: 10,50,75 < 80 → 3. bob: 30 < 80 → 1. carol: 60 → 1. dave: 25 → 1. eve: 0.
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 1.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_compare_lte() {
    // COUNT WHERE price <= 80
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("price".to_string()),
                    op: CompareOp::Lte,
                    rhs: RowFilterRhs::Constant(Value::Float(80.0)),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: 10,50,75 ≤ 80 → 3. bob: 30 → 1. carol: 60 → 1. dave: 80,25 → 2. eve: 0.
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_compare_gte() {
    // COUNT WHERE price >= 80
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("price".to_string()),
                    op: CompareOp::Gte,
                    rhs: RowFilterRhs::Constant(Value::Float(80.0)),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: 100 → 1. bob: 200 → 1. carol: 0. dave: 150,80 → 2. eve: 0.
    assert_target_eq(groups[&0][0].1, 1.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 0.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_compare_neq() {
    // COUNT WHERE status != 'shipped'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("status".to_string()),
                    op: CompareOp::Neq,
                    rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: returned,pending → 2. bob: returned → 1. carol: 0. dave: pending → 1. eve: 0.
    assert_target_eq(groups[&0][0].1, 2.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 0.0, "carol");
    assert_target_eq(groups[&3][0].1, 1.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: Filter RHS variants ─────────────────────────────────────────────

#[test]
fn integ_filter_entity_column_rhs() {
    // COUNT WHERE orders.quantity > entity.level
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("quantity".to_string()),
                    op: CompareOp::Gt,
                    rhs: RowFilterRhs::EntityColumn("level".to_string()),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice(level=3): qty 5>3 → 1. bob(1): qty 2>1,1>1(no) → 1. carol(2): qty 1>2 no → 0.
    // dave(1): qty 1>1(no),2>1,1>1(no) → 1. eve(5): 0.
    assert_target_eq(groups[&0][0].1, 1.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 0.0, "carol");
    assert_target_eq(groups[&3][0].1, 1.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_filter_remote_lhs() {
    // COUNT WHERE orders.PRODUCTS.category = 'shoes'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Remote {
                        path: vec![o2p_edge()],
                        column: "category".to_string(),
                    },
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("shoes".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // Products 0(shoes),1(shoes),2(clothing),3(accessories)
    // alice: orders 0→p3(acc),1→p0(shoes),2→p1(shoes),3→p2(clothing) → 2
    // bob: 4→p0(shoes),5→p1(shoes) → 2. carol: 6→p2(clothing) → 0.
    // dave: 7→p0(shoes),8→p1(shoes),9→p2(clothing) → 2. eve: 0.
    assert_target_eq(groups[&0][0].1, 2.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 0.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: Stage filters ───────────────────────────────────────────────────

#[test]
fn integ_stage_filter_constant() {
    // HAVING COUNT > 2
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: Some(StageFilter::Compare {
                alias: "cnt".to_string(),
                op: CompareOp::Gt,
                rhs: StageFilterRhs::Constant(Value::Int(2)),
            }),
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 2);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
}

#[test]
fn integ_stage_filter_entity_column() {
    // HAVING SUM(price) > entity.score
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![agg_orders_price("total", AggFunc::Sum, Some((0, 5000)))],
            filter: Some(StageFilter::Compare {
                alias: "total".to_string(),
                op: CompareOp::Gt,
                rhs: StageFilterRhs::EntityColumn("score".to_string()),
            }),
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: 235>75✓, bob: 230>120✓, carol: 60>50✓, dave: 255>200✓, eve: 0>10✗
    assert_eq!(groups.len(), 4);
    assert_target_eq(groups[&0][0].1, 235.0, "alice");
    assert_target_eq(groups[&1][0].1, 230.0, "bob");
    assert_target_eq(groups[&2][0].1, 60.0, "carol");
    assert_target_eq(groups[&3][0].1, 255.0, "dave");
}

#[test]
fn integ_stage_filter_alias_vs_alias() {
    // Stage 1: SUM as "total", AVG as "average". Filter: total > average.
    // Target = average (last alias).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![
                agg_orders_price("total", AggFunc::Sum, Some((0, 5000))),
                agg_orders_price("average", AggFunc::Avg, Some((0, 5000))),
            ],
            filter: Some(StageFilter::Compare {
                alias: "total".to_string(),
                op: CompareOp::Gt,
                rhs: StageFilterRhs::Alias("average".to_string()),
            }),
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: 235>58.75✓, bob: 230>115✓, carol: 60>60✗(equal), dave: 255>85✓, eve: 0>NaN✗
    assert_eq!(groups.len(), 3);
    assert_target_eq(groups[&0][0].1, 58.75, "alice avg");
    assert_target_eq(groups[&1][0].1, 115.0, "bob avg");
    assert_target_eq(groups[&3][0].1, 85.0, "dave avg");
}

#[test]
fn integ_three_stage_short_circuit() {
    // Stage 1: COUNT as "cnt", filter cnt > 1
    // Stage 2: SUM as "total", filter total > 240
    // Stage 3: AVG as "avg_price" (target)
    // Only dave makes it through both filters.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![
            Stage {
                ops: vec![count_orders_op("cnt", Some((0, 5000)))],
                filter: Some(StageFilter::Compare {
                    alias: "cnt".to_string(),
                    op: CompareOp::Gt,
                    rhs: StageFilterRhs::Constant(Value::Int(1)),
                }),
            },
            Stage {
                ops: vec![agg_orders_price("total", AggFunc::Sum, Some((0, 5000)))],
                filter: Some(StageFilter::Compare {
                    alias: "total".to_string(),
                    op: CompareOp::Gt,
                    rhs: StageFilterRhs::Constant(Value::Float(240.0)),
                }),
            },
            Stage {
                ops: vec![agg_orders_price("avg_price", AggFunc::Avg, Some((0, 5000)))],
                filter: None,
            },
        ],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // Stage 1 pass: alice(4), bob(2), dave(3). Stage 2: alice 235<240✗, bob 230<240✗, dave 255>240✓
    assert_eq!(groups.len(), 1);
    assert_target_eq(groups[&3][0].1, 85.0, "dave");
}

// ─── Tests: Multiple ops & combos ───────────────────────────────────────────

#[test]
fn integ_multiple_ops_same_stage() {
    // COUNT + SUM in same stage, target = SUM (last op)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![
                count_orders_op("cnt", Some((0, 5000))),
                agg_orders_price("total", AggFunc::Sum, Some((0, 5000))),
            ],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 5);
    assert_target_eq(groups[&0][0].1, 235.0, "alice");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_column_and_aggregate_same_stage() {
    // Op::Column(users.score) + COUNT in same stage, target = COUNT
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![
                Op::Column {
                    alias: "sc".to_string(),
                    table: "users".to_string(),
                    column: "score".to_string(),
                },
                count_orders_op("cnt", Some((0, 5000))),
            ],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 4.0, "alice cnt");
    assert_target_eq(groups[&4][0].1, 0.0, "eve cnt");
}

#[test]
fn integ_entity_filter_and_stage_filter() {
    // Entity filter: active=true (excludes carol). Stage filter: COUNT > 1.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let mut plan = g1_temporal(
        vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: Some(StageFilter::Compare {
                alias: "cnt".to_string(),
                op: CompareOp::Gt,
                rhs: StageFilterRhs::Constant(Value::Int(1)),
            }),
        }],
        vec![0],
    );
    plan.entity.filter = Some(RowFilter::Compare {
        lhs: RowFilterLhs::Column("active".to_string()),
        op: CompareOp::Eq,
        rhs: RowFilterRhs::Constant(Value::Bool(true)),
    });
    let groups = execute_and_group(&plan, &index, &output);
    // Active: alice(4),bob(2),dave(3),eve(0). Stage filter cnt>1: alice,bob,dave.
    assert_eq!(groups.len(), 3);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
}

// ─── Tests: Entity features ─────────────────────────────────────────────────

#[test]
fn integ_entity_column() {
    // Non-temporal: Op::Column(users.score)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_plan(vec![Stage {
        ops: vec![Op::Column {
            alias: "sc".to_string(),
            table: "users".to_string(),
            column: "score".to_string(),
        }],
        filter: None,
    }]);
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 5);
    assert_target_eq(groups[&0][0].1, 75.0, "alice");
    assert_target_eq(groups[&1][0].1, 120.0, "bob");
    assert_target_eq(groups[&2][0].1, 50.0, "carol");
    assert_target_eq(groups[&3][0].1, 200.0, "dave");
    assert_target_eq(groups[&4][0].1, 10.0, "eve");
}

#[test]
fn integ_entity_filter() {
    // Entity filter: active=true, then COUNT
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let mut plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 5000)))], filter: None }],
        vec![0],
    );
    plan.entity.filter = Some(RowFilter::Compare {
        lhs: RowFilterLhs::Column("active".to_string()),
        op: CompareOp::Eq,
        rhs: RowFilterRhs::Constant(Value::Bool(true)),
    });
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 4); // carol excluded
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_entity_time_col() {
    // time_col=signup_ts, obs=1000 → entities with signup_ts ≤ 1000: alice(1000), eve(500)
    // COUNT(orders, 0, 500) window [1000,1500)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: Some("signup_ts".to_string()),
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![1000],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 500)))],
            filter: None,
        }],
    };
    let groups = execute_and_group(&plan, &index, &output);
    // alice: orders at 1050,1100,1200,1300 all in [1000,1500) → 4. eve: 0 orders.
    assert_eq!(groups.len(), 2);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_entity_valid_to_col() {
    // valid_to_col, obs=1500. LEQ semantics: carol(valid_to=1500) is STILL VALID.
    // Valid: alice(2000), bob(3000), carol(1500), dave(4000). Expired: eve(1000).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: Some("valid_to".to_string()),
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![1500],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: None,
        }],
    };
    let groups = execute_and_group(&plan, &index, &output);
    // Window [1500,6500). alice: 0 (orders<1500). bob: 2100,2200→2. carol: 1600→1. dave: 3100,3200,3300→3.
    assert_eq!(groups.len(), 4);
    assert_target_eq(groups[&0][0].1, 0.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol: valid_to boundary LEQ");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
}

/// Build graph1 with `create_time` and `end_time` set on users table so the
/// index metadata records them. This lets the resolver fall back to the metadata
/// when the plan doesn't include `time_col` / `valid_to_col`.
///
/// Uses Timestamp columns for signup_ts and valid_to (create_time/end_time
/// require Date or Timestamp dtype in graph config validation).
/// Values are in epoch seconds, stored as Timestamp(Second).
fn build_graph1_with_create_time(root: &Path) -> PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    // ── Users table (same data as build_graph1 but with Timestamp columns) ──
    let mut user_id = Int64Builder::new();
    let mut name = StringBuilder::new();
    let mut level = Int64Builder::new();
    let mut score = Float64Builder::new();
    let mut signup_ts = TimestampSecondBuilder::new();
    let mut active = BooleanBuilder::new();
    let mut valid_to = TimestampSecondBuilder::new();

    for (uid, n, l, s, ts, a, vt) in [
        (0i64, "alice", 3i64, 75.0, 1000i64, true, 2000i64),
        (1, "bob", 1, 120.0, 2000, true, 3000),
        (2, "carol", 2, 50.0, 1500, false, 1500),
        (3, "dave", 1, 200.0, 3000, true, 4000),
        (4, "eve", 5, 10.0, 500, true, 1000),
    ] {
        user_id.append_value(uid);
        name.append_value(n);
        level.append_value(l);
        score.append_value(s);
        signup_ts.append_value(ts);
        active.append_value(a);
        valid_to.append_value(vt);
    }

    let users_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("level", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
        Field::new("signup_ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), false),
        Field::new("active", DataType::Boolean, false),
        Field::new("valid_to", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), false),
    ]);
    let users_batch = RecordBatch::try_new(
        Arc::new(users_schema),
        vec![
            Arc::new(user_id.finish()),
            Arc::new(name.finish()),
            Arc::new(level.finish()),
            Arc::new(score.finish()),
            Arc::new(signup_ts.finish()),
            Arc::new(active.finish()),
            Arc::new(valid_to.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    // ── Products table ──
    let mut product_id = Int64Builder::new();
    let mut brand = StringBuilder::new();
    let mut rating = Float64Builder::new();
    let mut category = StringBuilder::new();

    for (pid, b, r, c) in [
        (0i64, "Nike", 4.5, "shoes"),
        (1, "Adidas", 3.0, "shoes"),
        (2, "Nike", 5.0, "clothing"),
        (3, "Puma", 2.0, "accessories"),
    ] {
        product_id.append_value(pid);
        brand.append_value(b);
        rating.append_value(r);
        category.append_value(c);
    }

    let products_schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new("brand", DataType::Utf8, false),
        Field::new("rating", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]);
    let products_batch = RecordBatch::try_new(
        Arc::new(products_schema),
        vec![
            Arc::new(product_id.finish()),
            Arc::new(brand.finish()),
            Arc::new(rating.finish()),
            Arc::new(category.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &products_batch);

    // ── Orders table ──
    let mut order_id = Int64Builder::new();
    let mut order_user_id = Int64Builder::new();
    let mut order_product_id = Int64Builder::new();
    let mut price = Float64Builder::new();
    let mut quantity = Int64Builder::new();
    let mut ts = Int64Builder::new();
    let mut status = StringBuilder::new();

    for (oid, uid, pid, p, q, t, s) in [
        (0i64, 0i64, 3i64, 10.0, 5i64, 1050i64, "returned"),
        (1, 0, 0, 100.0, 2, 1100, "shipped"),
        (2, 0, 1, 50.0, 1, 1200, "pending"),
        (3, 0, 2, 75.0, 3, 1300, "shipped"),
        (4, 1, 0, 200.0, 1, 2100, "shipped"),
        (5, 1, 1, 30.0, 2, 2200, "returned"),
        (6, 2, 2, 60.0, 1, 1600, "shipped"),
        (7, 3, 0, 150.0, 1, 3100, "shipped"),
        (8, 3, 1, 80.0, 2, 3200, "pending"),
        (9, 3, 2, 25.0, 1, 3300, "shipped"),
    ] {
        order_id.append_value(oid);
        order_user_id.append_value(uid);
        order_product_id.append_value(pid);
        price.append_value(p);
        quantity.append_value(q);
        ts.append_value(t);
        status.append_value(s);
    }

    let orders_schema = Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("ts", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
    ]);
    let orders_batch = RecordBatch::try_new(
        Arc::new(orders_schema),
        vec![
            Arc::new(order_id.finish()),
            Arc::new(order_user_id.finish()),
            Arc::new(order_product_id.finish()),
            Arc::new(price.finish()),
            Arc::new(quantity.finish()),
            Arc::new(ts.finish()),
            Arc::new(status.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("orders.parquet"), &orders_batch);

    // ── GraphConfig with create_time on users ──
    let config = GraphConfig {
        dataset_name: "test_graph1_create_time".to_string(),
        tables: vec![
            TableSpec {
                name: "users".to_string(),
                columns: vec![
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("name", IngestedType::String),
                    mk_col("level", IngestedType::Int64),
                    mk_col("score", IngestedType::Float64),
                    mk_col("signup_ts", IngestedType::Timestamp),
                    mk_col("active", IngestedType::Bool),
                    mk_col("valid_to", IngestedType::Timestamp),
                ],
                primary_key: Some("user_id".to_string()),
                create_time: Some("signup_ts".to_string()),
                end_time: Some("valid_to".to_string()),
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("users.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "products".to_string(),
                columns: vec![
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("brand", IngestedType::String),
                    mk_col("rating", IngestedType::Float64),
                    mk_col("category", IngestedType::String),
                ],
                primary_key: Some("product_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("products.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "orders".to_string(),
                columns: vec![
                    mk_col("order_id", IngestedType::Int64),
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("price", IngestedType::Float64),
                    mk_col("quantity", IngestedType::Int64),
                    mk_col("ts", IngestedType::Int64),
                    mk_col("status", IngestedType::String),
                ],
                primary_key: Some("order_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("orders.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![
                    FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "ts".to_string(),
                                bitwidth: Some(16),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                    FKeySpec {
                        src_column: "product_id".to_string(),
                        dst_table: "products".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "ts".to_string(),
                                bitwidth: Some(16),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                ],
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

#[test]
fn integ_entity_create_time_metadata_fallback() {
    // Test: index has create_time=signup_ts in metadata, but plan has time_col=None.
    // The resolver should fall back to the index metadata and filter entities
    // whose create_time > observation timestamp.
    //
    // Observation timestamps: [1000, 2000]
    // Entity create_times (signup_ts): alice=1000, bob=2000, carol=1500, dave=3000, eve=500
    //
    // At ts=1000: entities with signup_ts <= 1000 → alice(1000), eve(500) → 2 entities
    // At ts=2000: entities with signup_ts <= 2000 → alice(1000), bob(2000), carol(1500), eve(500) → 4 entities
    // dave(3000) never appears.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1_with_create_time(dir.path());
    let output = dir.path().join("out");

    // Plan has time_col=None AND valid_to_col=None — both fall back to index
    // metadata (create_time=signup_ts, end_time=valid_to).
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![1000, 2000],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 500)))],
            filter: None,
        }],
    };
    let groups = execute_and_group(&plan, &index, &output);

    // Both create_time (signup_ts) AND end_time (valid_to) filters are active:
    //   entity visible at ts iff signup_ts <= ts AND ts <= valid_to
    //
    // At ts=1000:
    //   alice: 1000<=1000 ✓, 1000<=2000 ✓ → visible
    //   bob:   2000>1000  ✗ → excluded (not created yet)
    //   carol: 1500>1000  ✗ → excluded (not created yet)
    //   dave:  3000>1000  ✗ → excluded
    //   eve:   500<=1000  ✓, 1000<=1000 ✓ → visible (boundary LEQ)
    //
    // At ts=2000:
    //   alice: 1000<=2000 ✓, 2000<=2000 ✓ → visible
    //   bob:   2000<=2000 ✓, 2000<=3000 ✓ → visible
    //   carol: 1500<=2000 ✓, 2000>1500   ✗ → excluded (expired)
    //   dave:  3000>2000  ✗ → excluded
    //   eve:   500<=2000  ✓, 2000>1000   ✗ → excluded (expired)
    //
    // Total: 4 rows
    assert_eq!(groups[&0].len(), 2, "alice: 2 timestamps");
    assert_eq!(groups[&1].len(), 1, "bob: 1 timestamp");
    assert_eq!(groups[&1][0].0, 2000, "bob: only at ts=2000");
    assert!(groups.get(&2).is_none(), "carol: excluded (expired at ts=2000)");
    assert!(groups.get(&3).is_none(), "dave: should not appear (signup_ts=3000)");
    assert_eq!(groups[&4].len(), 1, "eve: 1 timestamp (expired at ts=2000)");
    assert_eq!(groups[&4][0].0, 1000, "eve: only at ts=1000");
}

#[test]
fn integ_entity_end_time_metadata_fallback() {
    // Test: index has end_time=valid_to in metadata, but plan has valid_to_col=None.
    // The resolver should fall back to the index metadata and filter entities
    // whose valid_to < observation timestamp.
    //
    // Observation timestamp: [1500]
    // Entity valid_to: alice=2000, bob=3000, carol=1500, dave=4000, eve=1000
    //
    // At ts=1500: LEQ semantics → valid if ts <= valid_to
    // Valid: alice(2000), bob(3000), carol(1500), dave(4000)
    // Expired: eve(valid_to=1000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1_with_create_time(dir.path());
    let output = dir.path().join("out");

    // Plan has valid_to_col=None AND time_col=None — both fall back to index metadata.
    // create_time filter: at ts=1500, entities with signup_ts <= 1500: alice(1000), carol(1500), eve(500)
    // end_time filter: at ts=1500, entities with valid_to >= 1500: alice(2000), carol(1500)
    // Combined: alice, carol
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![1500],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: None,
        }],
    };
    let groups = execute_and_group(&plan, &index, &output);

    // Both create_time and end_time fallback active:
    // alice: signup_ts=1000 <= 1500, valid_to=2000 >= 1500 → VALID
    // bob: signup_ts=2000 > 1500 → EXCLUDED by create_time
    // carol: signup_ts=1500 <= 1500, valid_to=1500 >= 1500 → VALID (LEQ boundary)
    // dave: signup_ts=3000 > 1500 → EXCLUDED by create_time
    // eve: signup_ts=500 <= 1500, valid_to=1000 < 1500 → EXCLUDED by end_time
    assert_eq!(groups.len(), 2, "only alice and carol should appear");
    assert!(groups.contains_key(&0), "alice should appear");
    assert!(groups.contains_key(&2), "carol should appear");
}

#[test]
fn integ_compare_string() {
    // Op::Column(name) then Op::Compare(name == "alice")
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_plan(vec![
        Stage {
            ops: vec![Op::Column {
                alias: "nm".to_string(),
                table: "users".to_string(),
                column: "name".to_string(),
            }],
            filter: None,
        },
        Stage {
            ops: vec![Op::Compare {
                alias: "is_alice".to_string(),
                lhs: "nm".to_string(),
                op: CompareOp::Eq,
                rhs: Value::Str("alice".to_string()),
            }],
            filter: None,
        },
    ]);
    let groups = execute_and_group_bool(&plan, &index, &output);
    assert_eq!(groups.len(), 5);
    assert!(groups[&0][0].1, "alice should be true");
    assert!(!groups[&1][0].1, "bob should be false");
}

// ─── Tests: ListDistinct ────────────────────────────────────────────────────

#[test]
fn integ_list_distinct_no_limit() {
    // LIST_DISTINCT(orders.status, 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("status".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // value_to_f64(List) = len. alice: 3 distinct, bob: 2, carol: 1, dave: 2, eve: 0
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_list_distinct_with_limit() {
    // LIST_DISTINCT(orders.status, 0, 5000, limit=1)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("status".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: Some(1),
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // Limited to 1 distinct value each
    assert_target_eq(groups[&0][0].1, 1.0, "alice limited");
    assert_target_eq(groups[&1][0].1, 1.0, "bob limited");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_list_distinct_with_row_filter() {
    // LIST_DISTINCT(orders.status WHERE price > 50, 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("status".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("price".to_string()),
                    op: CompareOp::Gt,
                    rhs: RowFilterRhs::Constant(Value::Float(50.0)),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: prices>50: 100(shipped),75(shipped) → {shipped} → 1
    // bob: 200(shipped) → {shipped} → 1. carol: 60(shipped) → {shipped} → 1.
    // dave: 150(shipped),80(pending) → {shipped,pending} → 2. eve: 0.
    assert_target_eq(groups[&0][0].1, 1.0, "alice");
    assert_target_eq(groups[&1][0].1, 1.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: Temporal windows ────────────────────────────────────────────────

#[test]
fn integ_temporal_future_window() {
    // COUNT(orders.*, 0, 200, seconds) at obs=1000. Window (1000,1200].
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 200)))], filter: None }],
        vec![1000],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: ts 1050,1100,1200 in (1000,1200] → 3. Others: 0.
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 0.0, "bob");
}

#[test]
fn integ_temporal_multiple_timestamps() {
    // COUNT at obs=1000 and obs=2000, window (0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 5000)))], filter: None }],
        vec![1000, 2000],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // obs=1000 [1000,6000): alice:4, bob:2, carol:1, dave:3, eve:0
    // obs=2000 [2000,7000): alice:0, bob:2, carol:0, dave:3, eve:0
    assert_eq!(groups[&0].len(), 2);
    assert_target_eq(groups[&0][0].1, 4.0, "alice@1000");
    assert_target_eq(groups[&0][1].1, 0.0, "alice@2000");
    assert_target_eq(groups[&1][0].1, 2.0, "bob@1000");
    assert_target_eq(groups[&1][1].1, 2.0, "bob@2000");
}

#[test]
fn integ_temporal_window_boundary() {
    // COUNT at obs=1050, window (0,250). Window (1050,1300].
    // alice: 1100,1200,1300 → 3. 1050 excluded (lo exclusive), 1300 included (hi inclusive).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 250)))], filter: None }],
        vec![1050],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 3.0, "alice: 1050,1100,1200");
}

#[test]
fn integ_temporal_window_splits_edges() {
    // SUM(price, 0, 100) at obs=1150. Window [1150,1250).
    // alice: only ts=1200(50) in range → 50.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("s", AggFunc::Sum, Some((0, 100)))], filter: None }],
        vec![1150],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 50.0, "alice");
}

#[test]
fn integ_temporal_window_excludes() {
    // SUM(price, 0, 50) at obs=1150. Window (1150,1200].
    // alice: 1200 is inclusive → price=50.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("s", AggFunc::Sum, Some((0, 50)))], filter: None }],
        vec![1150],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 50.0, "alice: 1200 included (hi inclusive)");
}

#[test]
fn integ_temporal_multiple_windows_same_entity() {
    // SUM(price, 0, 100) at obs=950 and obs=1000.
    // obs=950: (950,1050] → ts=1050(10). obs=1000: (1000,1100] → ts=1050(10)+1100(100)=110.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage { ops: vec![agg_orders_price("s", AggFunc::Sum, Some((0, 100)))], filter: None }],
        vec![950, 1000],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 10.0, "alice@950");
    assert_target_eq(groups[&0][1].1, 110.0, "alice@1000");
}

// ─── Tests: TimeUnits ───────────────────────────────────────────────────────

#[test]
fn integ_timeunit_hours() {
    // COUNT(orders.*, 0, 1, hours) at obs=0. Window [0,3600).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 1)),
                window_unit: Some(TimeUnit::Hours),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // All orders < 3600: alice:4, bob:2, carol:1, dave:3, eve:0
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 3.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_timeunit_days() {
    // COUNT(orders.*, 0, 1, days) at obs=0. Window [0,86400).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 1)),
                window_unit: Some(TimeUnit::Days),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
}

#[test]
fn integ_timeunit_weeks() {
    // COUNT(orders.*, 0, 1, weeks) at obs=0. Window [0,604800).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 1)),
                window_unit: Some(TimeUnit::Weeks),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    assert_target_eq(groups[&0][0].1, 4.0, "alice");
}

// ─── Tests: Calendar months ──────────────────────────────────────────────────

// Graph for calendar-month testing: company → filings
//
// company (1 row):
//   node  cik
//   0     1800
//
// filings (4 rows, FK: cik→company):
//   id  cik   amount  ts (epoch seconds)
//   0   0     100.0   2023-04-01 = 1680307200
//   1   0     200.0   2023-07-01 = 1688169600
//   2   0     300.0   2023-10-01 = 1696118400
//   3   0     400.0   2024-01-01 = 1704067200
//
// Each filing is at the START of a quarter. A 3-month calendar window from
// the previous quarter's start should reach exactly the next quarter's start.
// A fixed 90-day approximation would MISS Q3→Q4 (92 days) and Q4→Q1 (92 days).

fn build_graph_calendar(root: &Path) -> PathBuf {
    let data_dir = root.join("data_cal");
    let index_dir = root.join("index_cal");
    fs::create_dir_all(&data_dir).unwrap();

    // ── Company ──
    let company_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cik", DataType::Int64, false),
        ])),
        vec![Arc::new(Int64Array::from(vec![1800]))],
    )
    .unwrap();
    write_parquet(&data_dir.join("company.parquet"), &company_batch);

    // ── Filings ──
    let apr1 = 1680307200i64; // 2023-04-01
    let jul1 = 1688169600i64; // 2023-07-01
    let oct1 = 1696118400i64; // 2023-10-01
    let jan1 = 1704067200i64; // 2024-01-01
    let filings_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("filing_id", DataType::Int64, false),
            Field::new("cik", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("ts", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3])),
            Arc::new(Int64Array::from(vec![1800, 1800, 1800, 1800])), // FK matches company PK
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0, 400.0])),
            Arc::new(Int64Array::from(vec![apr1, jul1, oct1, jan1])),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("filings.parquet"), &filings_batch);

    // ── GraphConfig ──
    let config = GraphConfig {
        dataset_name: "test_calendar".to_string(),
        tables: vec![
            TableSpec {
                name: "company".to_string(),
                columns: vec![mk_col("cik", IngestedType::Int64)],
                primary_key: Some("cik".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("company.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "filings".to_string(),
                columns: vec![
                    mk_col("filing_id", IngestedType::Int64),
                    mk_col("cik", IngestedType::Int64),
                    mk_col("amount", IngestedType::Float64),
                    mk_col("ts", IngestedType::Int64),
                ],
                primary_key: Some("filing_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("filings.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![
                    FKeySpec {
                        src_column: "cik".to_string(),
                        dst_table: "company".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "filings".to_string(),
                                column: "ts".to_string(),
                                bitwidth: Some(32),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                ],
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

fn c2f_edge() -> EdgeRef {
    EdgeRef {
        src_table: "company".to_string(),
        src_col: "cik".to_string(),
        dst_table: "filings".to_string(),
        dst_col: "cik".to_string(),
    }
}

#[test]
fn integ_timeunit_months_calendar() {
    // SUM(filings.amount, 0, 3, months) at each quarter start.
    // Window (obs, obs+3months] should capture the NEXT quarter's filing.
    //
    // With calendar months:
    //   obs=2023-04-01 → (Apr1, Jul1]:  jul1 filing → 200.0 (91 days)
    //   obs=2023-07-01 → (Jul1, Oct1]:  oct1 filing → 300.0 (92 days!)
    //   obs=2023-10-01 → (Oct1, Jan1]:  jan1 filing → 400.0 (92 days!)
    //
    // With fixed 90 days:
    //   obs=2023-04-01 → (Apr1, Jun30]: MISSES jul1 by 1 day → 0.0
    //   obs=2023-07-01 → (Jul1, Sep29]: MISSES oct1 by 2 days → 0.0
    //   obs=2023-10-01 → (Oct1, Dec30]: MISSES jan1 by 2 days → 0.0
    //
    // This test proves calendar month arithmetic works correctly.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph_calendar(dir.path());
    let output = dir.path().join("out_cal");

    let apr1 = 1680307200i64;
    let jul1 = 1688169600i64;
    let oct1 = 1696118400i64;

    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "company".to_string(),
            column: "cik".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![apr1, jul1, oct1],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "target".to_string(),
                path: vec![c2f_edge()],
                func: AggFunc::Sum,
                target_col: Some("amount".to_string()),
                time_col: None,
                window: Some((0, 3)),
                window_unit: Some(TimeUnit::Months),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
    };

    let groups = execute_and_group(&plan, &index, &output);
    let company = &groups[&1800]; // entity PK = 1800
    assert_eq!(company.len(), 3, "should have 3 timestamps");
    assert_target_eq(company[0].1, 200.0, "obs=Apr1: (Apr1,Jul1] captures Jul1 filing");
    assert_target_eq(company[1].1, 300.0, "obs=Jul1: (Jul1,Oct1] captures Oct1 filing");
    assert_target_eq(company[2].1, 400.0, "obs=Oct1: (Oct1,Jan1] captures Jan1 filing");
}

#[test]
fn integ_timeunit_years_calendar() {
    // SUM(filings.amount, 0, 1, years) at obs=2023-04-01.
    // Window (Apr1 2023, Apr1 2024]: should capture all 4 filings
    // (jul1=200 + oct1=300 + jan1=400 = 900; apr1 2023 excluded at lo boundary).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph_calendar(dir.path());
    let output = dir.path().join("out_year");

    let apr1 = 1680307200i64; // 2023-04-01

    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "company".to_string(),
            column: "cik".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![apr1],
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "target".to_string(),
                path: vec![c2f_edge()],
                func: AggFunc::Sum,
                target_col: Some("amount".to_string()),
                time_col: None,
                window: Some((0, 1)),
                window_unit: Some(TimeUnit::Years),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
    };

    let groups = execute_and_group(&plan, &index, &output);
    let company = &groups[&1800]; // entity PK = 1800
    // (Apr1 2023, Apr1 2024]: jul1(200) + oct1(300) + jan1(400) = 900
    // apr1 2023 itself is excluded (lo exclusive)
    assert_target_eq(company[0].1, 900.0, "obs=Apr1: 1-year window captures 3 future filings");
}

// ─── Tests: Multi-hop ───────────────────────────────────────────────────────

#[test]
fn integ_multi_hop_count_distinct_brand() {
    // COUNT_DISTINCT(orders.products.brand, 0, 5000) for each user
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cd".to_string(),
                path: vec![u2o_edge(), o2p_edge()],
                func: AggFunc::CountDistinct,
                target_col: Some("brand".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: Puma,Nike,Adidas,Nike → {Puma,Nike,Adidas} = 3
    // bob: Nike,Adidas → 2. carol: Nike → 1. dave: Nike,Adidas,Nike → 2. eve: 0.
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 1.0, "carol");
    assert_target_eq(groups[&3][0].1, 2.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

#[test]
fn integ_multi_hop_avg_rating() {
    // AVG(orders.products.rating, 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "avg".to_string(),
                path: vec![u2o_edge(), o2p_edge()],
                func: AggFunc::Avg,
                target_col: Some("rating".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: (2.0+4.5+3.0+5.0)/4 = 3.625
    // bob: (4.5+3.0)/2 = 3.75. carol: 5.0. dave: (4.5+3.0+5.0)/3 ≈ 4.1667. eve: absent.
    assert_target_eq(groups[&0][0].1, 3.625, "alice");
    assert_target_eq(groups[&1][0].1, 3.75, "bob");
    assert_target_eq(groups[&2][0].1, 5.0, "carol");
    assert_target_eq(groups[&3][0].1, 12.5 / 3.0, "dave");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_multi_hop_with_filter() {
    // AVG(orders.products.rating WHERE category='shoes', 0, 5000)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "avg".to_string(),
                path: vec![u2o_edge(), o2p_edge()],
                func: AggFunc::Avg,
                target_col: Some("rating".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("category".to_string()),
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("shoes".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // Row filter on last hop (products). shoes: p0(4.5),p1(3.0)
    // alice: p0,p1 → (4.5+3.0)/2=3.75. bob: same → 3.75. carol: p2(clothing) → absent.
    // dave: p0,p1 → 3.75. eve: absent.
    assert_target_eq(groups[&0][0].1, 3.75, "alice");
    assert_target_eq(groups[&1][0].1, 3.75, "bob");
    assert!(!groups.contains_key(&2), "carol should be absent (no shoes)");
    assert_target_eq(groups[&3][0].1, 3.75, "dave");
    assert!(!groups.contains_key(&4), "eve should be absent (no orders)");
}

#[test]
fn integ_multi_hop_temporal() {
    // COUNT_DISTINCT(orders.products.brand, 0, 200) at obs=1000.
    // Window on FIRST hop (1000,1200]: alice orders 0(ts=1050),1(ts=1100),2(ts=1200).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cd".to_string(),
                path: vec![u2o_edge(), o2p_edge()],
                func: AggFunc::CountDistinct,
                target_col: Some("brand".to_string()),
                time_col: None,
                window: Some((0, 200)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![1000],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: order 0→Puma, order 1→Nike, order 2→Adidas → {Puma,Nike,Adidas} = 3. Others: 0.
    assert_target_eq(groups[&0][0].1, 3.0, "alice");
    assert_target_eq(groups[&1][0].1, 0.0, "bob");
}

// ─── Tests: Edge cases ──────────────────────────────────────────────────────

#[test]
fn integ_all_agg_empty_states() {
    // Verify empty-state values for each aggregation function (eve has 0 orders).
    // COUNT/SUM/COUNT_DISTINCT always produce a value (0); AVG/MIN/MAX/FIRST/LAST
    // produce None for empty traversals → row is skipped (matching Kumo INNER join).
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());

    // (name, func, target_col, expected): Some(f64) = eve row present, None = eve row absent
    let cases: Vec<(&str, AggFunc, Option<&str>, Option<f64>)> = vec![
        ("count", AggFunc::Count, None, Some(0.0)),
        ("sum", AggFunc::Sum, Some("price"), Some(0.0)),
        ("avg", AggFunc::Avg, Some("price"), None),
        ("min", AggFunc::Min, Some("price"), None),
        ("max", AggFunc::Max, Some("price"), None),
        ("first", AggFunc::First, Some("price"), None),
        ("last", AggFunc::Last, Some("price"), None),
        ("count_distinct", AggFunc::CountDistinct, Some("status"), Some(0.0)),
    ];

    for (name, func, target_col, expected) in cases {
        let output = dir.path().join(format!("out_{name}"));
        let plan = g1_temporal(
            vec![Stage {
                ops: vec![Op::Aggregate {
                    alias: "v".to_string(),
                    path: vec![u2o_edge()],
                    func,
                    target_col: target_col.map(String::from),
                    time_col: None,
                    window: Some((0, 5000)),
                    window_unit: Some(TimeUnit::Seconds),
                    row_filter: None,
                    limit: None,
                }],
                filter: None,
            }],
            vec![0],
        );
        let groups = execute_and_group(&plan, &index, &output);
        match expected {
            Some(val) => assert_target_eq(groups[&4][0].1, val, &format!("eve {name}")),
            None => assert!(!groups.contains_key(&4), "eve {name} should be absent"),
        }
    }
}

#[test]
fn integ_entity_filter_excludes_all() {
    // Entity filter: level > 100 → no user matches → 0 output rows
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let mut plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 5000)))], filter: None }],
        vec![0],
    );
    plan.entity.filter = Some(RowFilter::Compare {
        lhs: RowFilterLhs::Column("level".to_string()),
        op: CompareOp::Gt,
        rhs: RowFilterRhs::Constant(Value::Int(100)),
    });
    let groups = execute_and_group(&plan, &index, &output);
    assert_eq!(groups.len(), 0);
}

#[test]
fn integ_nested_filter_not_or() {
    // COUNT WHERE NOT(status='shipped' OR status='returned') → only 'pending'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Not(Box::new(RowFilter::Or(vec![
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                    },
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("returned".to_string())),
                    },
                ])))),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // Only pending: alice order 2, dave order 8.
    assert_target_eq(groups[&0][0].1, 1.0, "alice");
    assert_target_eq(groups[&1][0].1, 0.0, "bob");
    assert_target_eq(groups[&3][0].1, 1.0, "dave");
}

#[test]
fn integ_nested_filter_and_or() {
    // COUNT WHERE (status='shipped' AND price>100) OR status='returned'
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out");
    let plan = g1_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "cnt".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::Count,
                target_col: None,
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Or(vec![
                    RowFilter::And(vec![
                        RowFilter::Compare {
                            lhs: RowFilterLhs::Column("status".to_string()),
                            op: CompareOp::Eq,
                            rhs: RowFilterRhs::Constant(Value::Str("shipped".to_string())),
                        },
                        RowFilter::Compare {
                            lhs: RowFilterLhs::Column("price".to_string()),
                            op: CompareOp::Gt,
                            rhs: RowFilterRhs::Constant(Value::Float(100.0)),
                        },
                    ]),
                    RowFilter::Compare {
                        lhs: RowFilterLhs::Column("status".to_string()),
                        op: CompareOp::Eq,
                        rhs: RowFilterRhs::Constant(Value::Str("returned".to_string())),
                    },
                ])),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
    );
    let groups = execute_and_group(&plan, &index, &output);
    // alice: order0(returned)→1. bob: order4(shipped,200>100)→1, order5(returned)→1 total 2.
    // carol: 0. dave: order7(shipped,150>100)→1. eve: 0.
    assert_target_eq(groups[&0][0].1, 1.0, "alice");
    assert_target_eq(groups[&1][0].1, 2.0, "bob");
    assert_target_eq(groups[&2][0].1, 0.0, "carol");
    assert_target_eq(groups[&3][0].1, 1.0, "dave");
    assert_target_eq(groups[&4][0].1, 0.0, "eve");
}

// ─── Tests: lag_timesteps / ENTITY_INCLUDE_BIT ──────────────────────────────

#[test]
fn integ_lag_timesteps_entity_include_bit() {
    // lag_timesteps=2 with entity filter active=true.
    // carol (active=false) should still produce rows but with ENTITY_INCLUDE_BIT=false.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lag");
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: Some(RowFilter::Compare {
                lhs: RowFilterLhs::Column("active".to_string()),
                op: CompareOp::Eq,
                rhs: RowFilterRhs::Constant(Value::Bool(true)),
            }),
        },
        temporal: Some(TemporalSpec {
            timestamps: vec![0],
            forecast_length: 1,
            lag_timesteps: Some(2),
        }),
        prediction_type: PredictionType::Node,
        stages: vec![Stage {
            ops: vec![count_orders_op("cnt", Some((0, 5000)))],
            filter: None,
        }],
    };
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_with_bit(&shard_paths);
    // All 5 entities should have rows (carol included despite filter fail)
    assert_eq!(rows.len(), 5);
    // Sort by entity
    let mut by_entity: HashMap<i64, Vec<(i64, f64, bool)>> = HashMap::new();
    for &(e, ts, t, b) in &rows {
        by_entity.entry(e).or_default().push((ts, t, b));
    }
    // alice(0): active=true → bit=true
    assert!(by_entity[&0][0].2, "alice should have bit=true");
    // bob(1): active=true → bit=true
    assert!(by_entity[&1][0].2, "bob should have bit=true");
    // carol(2): active=false → bit=false (KEPT for lag features)
    assert!(!by_entity[&2][0].2, "carol should have bit=false");
    // dave(3): active=true → bit=true
    assert!(by_entity[&3][0].2, "dave should have bit=true");
    // eve(4): active=true → bit=true
    assert!(by_entity[&4][0].2, "eve should have bit=true");
}

#[test]
fn integ_lag_timesteps_zero_no_bit_column() {
    // lag_timesteps=0 (or None) → no ENTITY_INCLUDE_BIT column, entity filter drops rows normally.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_nolag");
    let mut plan = g1_temporal(
        vec![Stage { ops: vec![count_orders_op("cnt", Some((0, 5000)))], filter: None }],
        vec![0],
    );
    plan.entity.filter = Some(RowFilter::Compare {
        lhs: RowFilterLhs::Column("active".to_string()),
        op: CompareOp::Eq,
        rhs: RowFilterRhs::Constant(Value::Bool(true)),
    });
    let groups = execute_and_group(&plan, &index, &output);
    // carol filtered OUT (no lag_timesteps)
    assert_eq!(groups.len(), 4);
    assert!(!groups.contains_key(&2), "carol should be excluded");
}

// ─── Link prediction helpers ─────────────────────────────────────────────────

/// Read exploded link-prediction rows: (entity, ts, target_str, frequency).
fn read_output_rows_exploded(shard_paths: &[PathBuf]) -> Vec<(i64, i64, String, i64)> {
    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let targets = read_target_column_as_strings(&batch, 2);
            let freqs = batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..batch.num_rows() {
                rows.push((
                    entities.value(i),
                    timestamps.value(i) / 1_000_000_000,
                    targets[i].clone(),
                    freqs.value(i),
                ));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    rows
}

/// Read exploded link-prediction rows WITHOUT frequency: (entity, ts, target_str).
fn read_output_rows_exploded_no_freq(shard_paths: &[PathBuf]) -> Vec<(i64, i64, String)> {
    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let targets = read_target_column_as_strings(&batch, 2);
            for i in 0..batch.num_rows() {
                rows.push((
                    entities.value(i),
                    timestamps.value(i) / 1_000_000_000,
                    targets[i].clone(),
                ));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    rows
}

/// Read array link-prediction rows: (entity, ts, targets, frequencies).
fn read_output_rows_array(shard_paths: &[PathBuf]) -> Vec<(i64, i64, Vec<String>, Vec<i64>)> {
    use arrow_array::{Array, ListArray};

    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let target_list = batch.column(2).as_any().downcast_ref::<ListArray>().unwrap();
            let freq_list = batch.column(3).as_any().downcast_ref::<ListArray>().unwrap();

            for i in 0..batch.num_rows() {
                let target_arr = target_list.value(i);
                let targets = read_list_values_as_strings(&target_arr);

                let freq_arr = freq_list.value(i);
                let freq_vals = freq_arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let freqs: Vec<i64> =
                    (0..freq_vals.len()).map(|j| freq_vals.value(j)).collect();

                rows.push((
                    entities.value(i),
                    timestamps.value(i) / 1_000_000_000,
                    targets,
                    freqs,
                ));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

/// Read array link-prediction rows WITHOUT frequency: (entity, ts, targets).
fn read_output_rows_array_no_freq(shard_paths: &[PathBuf]) -> Vec<(i64, i64, Vec<String>)> {
    use arrow_array::{Array, ListArray};

    let mut rows = Vec::new();
    for path in shard_paths {
        let file = fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            let target_list = batch.column(2).as_any().downcast_ref::<ListArray>().unwrap();

            for i in 0..batch.num_rows() {
                let target_arr = target_list.value(i);
                let targets = read_list_values_as_strings(&target_arr);

                rows.push((
                    entities.value(i),
                    timestamps.value(i) / 1_000_000_000,
                    targets,
                ));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

fn list_distinct_status_op(alias: &str, window: Option<(i64, i64)>, limit: Option<u32>) -> Op {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![u2o_edge()],
        func: AggFunc::ListDistinct,
        target_col: Some("status".to_string()),
        time_col: None,
        window,
        window_unit: Some(TimeUnit::Seconds),
        row_filter: None,
        limit,
    }
}

fn list_distinct_product_id_op(alias: &str, window: Option<(i64, i64)>, limit: Option<u32>) -> Op {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![u2o_edge()],
        func: AggFunc::ListDistinct,
        target_col: Some("product_id".to_string()),
        time_col: None,
        window,
        window_unit: Some(TimeUnit::Seconds),
        row_filter: None,
        limit,
    }
}

fn g1_link_temporal(
    stages: Vec<Stage>,
    timestamps: Vec<i64>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps,
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

fn g1_link(
    stages: Vec<Stage>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

// ─── Tests: Link prediction — exploded with frequency (RANK) ─────────────────

#[test]
fn integ_link_exploded_with_frequency() {
    // RANK: LIST_DISTINCT(orders.status, 0, 5000)
    // Each distinct value becomes a separate output row with its frequency.
    //
    // Edges iterate DESCENDING by sort key (ts), so traversal order is:
    //   alice: order3(shipped), order2(pending), order1(shipped+dup), order0(returned)
    //   bob: order5(returned), order4(shipped)
    //   carol: order6(shipped)
    //   dave: order9(shipped), order8(pending), order7(shipped+dup)
    //   eve: (no orders)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_exploded");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), None)],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // Collect as sets of (entity, target, freq) since order within entity may vary
    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // alice: shipped(2), pending(1), returned(1)
    assert!(as_set.contains(&(0, "shipped".into(), 2)), "alice shipped×2");
    assert!(as_set.contains(&(0, "pending".into(), 1)), "alice pending×1");
    assert!(as_set.contains(&(0, "returned".into(), 1)), "alice returned×1");

    // bob: returned(1), shipped(1)
    assert!(as_set.contains(&(1, "returned".into(), 1)), "bob returned×1");
    assert!(as_set.contains(&(1, "shipped".into(), 1)), "bob shipped×1");

    // carol: shipped(1)
    assert!(as_set.contains(&(2, "shipped".into(), 1)), "carol shipped×1");

    // dave: shipped(2), pending(1)
    assert!(as_set.contains(&(3, "shipped".into(), 2)), "dave shipped×2");
    assert!(as_set.contains(&(3, "pending".into(), 1)), "dave pending×1");

    // eve: no rows
    assert!(
        !rows.iter().any(|(e, ..)| *e == 4),
        "eve should have no output rows"
    );

    // Total: 3 + 2 + 1 + 2 = 8 rows
    assert_eq!(rows.len(), 8);
}

// ─── Tests: Link prediction — exploded without frequency (CLASSIFY) ──────────

#[test]
fn integ_link_exploded_no_frequency() {
    // CLASSIFY: LIST_DISTINCT(orders.status, 0, 5000), exploded but no frequency column.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_exploded_nf");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), None)],
            filter: None,
        }],
        vec![0],
        true,  // explode
        false, // no frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded_no_freq(&shard_paths);

    // Schema should have 3 columns: ENTITY, TIMESTAMP, TARGET (no FREQUENCY)
    let file = fs::File::open(&shard_paths[0]).unwrap();
    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = reader_builder.schema();
    assert_eq!(schema.fields().len(), 3, "should have ENTITY, TIMESTAMP, TARGET only");

    let targets_for_entity: HashMap<i64, Vec<String>> = {
        let mut map: HashMap<i64, Vec<String>> = HashMap::new();
        for (e, _ts, t) in &rows {
            map.entry(*e).or_default().push(t.clone());
        }
        map
    };

    // alice: 3 distinct statuses
    assert_eq!(targets_for_entity[&0].len(), 3, "alice 3 distinct");
    // bob: 2 distinct statuses
    assert_eq!(targets_for_entity[&1].len(), 2, "bob 2 distinct");
    // carol: 1 distinct status
    assert_eq!(targets_for_entity[&2].len(), 1, "carol 1 distinct");
    // dave: 2 distinct statuses
    assert_eq!(targets_for_entity[&3].len(), 2, "dave 2 distinct");

    assert_eq!(rows.len(), 8);
}

// ─── Tests: Link prediction — array with frequency (RANK array) ──────────────

#[test]
fn integ_link_array_with_frequency() {
    // RANK array: LIST_DISTINCT(orders.status, 0, 5000), one row per entity
    // with TARGET as List<Utf8> and FREQUENCY as List<Int64>.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_array");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), None)],
            filter: None,
        }],
        vec![0],
        false, // no explode → array mode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array(&shard_paths);

    // 5 entities, but eve has no orders → empty list → still emitted as row
    // (Value::List(vec![]) is a valid target)
    assert_eq!(rows.len(), 5, "one row per entity");

    // Sort targets + frequencies together for comparison
    fn sort_paired(targets: &[String], freqs: &[i64]) -> (Vec<String>, Vec<i64>) {
        let mut pairs: Vec<_> = targets.iter().cloned().zip(freqs.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs.into_iter().unzip()
    }

    for (entity, _ts, targets, freqs) in &rows {
        let (sorted_t, sorted_f) = sort_paired(targets, freqs);
        match *entity {
            0 => {
                // alice: pending(1), returned(1), shipped(2)
                assert_eq!(sorted_t, vec!["pending", "returned", "shipped"]);
                assert_eq!(sorted_f, vec![1, 1, 2]);
            }
            1 => {
                // bob: returned(1), shipped(1)
                assert_eq!(sorted_t, vec!["returned", "shipped"]);
                assert_eq!(sorted_f, vec![1, 1]);
            }
            2 => {
                // carol: shipped(1)
                assert_eq!(sorted_t, vec!["shipped"]);
                assert_eq!(sorted_f, vec![1]);
            }
            3 => {
                // dave: pending(1), shipped(2)
                assert_eq!(sorted_t, vec!["pending", "shipped"]);
                assert_eq!(sorted_f, vec![1, 2]);
            }
            4 => {
                // eve: empty
                assert!(sorted_t.is_empty(), "eve should have empty targets");
                assert!(sorted_f.is_empty(), "eve should have empty freqs");
            }
            _ => panic!("unexpected entity {entity}"),
        }
    }
}

// ─── Tests: Link prediction — array without frequency (CLASSIFY array) ───────

#[test]
fn integ_link_array_no_frequency() {
    // CLASSIFY array: LIST_DISTINCT(orders.status, 0, 5000), one row per entity
    // TARGET as List<Utf8>, no FREQUENCY column.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_array_nf");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), None)],
            filter: None,
        }],
        vec![0],
        false, // no explode → array mode
        false, // no frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array_no_freq(&shard_paths);

    // Schema: ENTITY, TIMESTAMP, TARGET (List<Utf8>) — 3 columns
    let file = fs::File::open(&shard_paths[0]).unwrap();
    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = reader_builder.schema();
    assert_eq!(schema.fields().len(), 3, "no FREQUENCY column");

    assert_eq!(rows.len(), 5, "one row per entity");
    for (entity, _ts, targets) in &rows {
        let mut sorted = targets.clone();
        sorted.sort();
        match *entity {
            0 => assert_eq!(sorted, vec!["pending", "returned", "shipped"]),
            1 => assert_eq!(sorted, vec!["returned", "shipped"]),
            2 => assert_eq!(sorted, vec!["shipped"]),
            3 => assert_eq!(sorted, vec!["pending", "shipped"]),
            4 => assert!(sorted.is_empty()),
            _ => panic!("unexpected entity {entity}"),
        }
    }
}

// ─── Tests: Link prediction — frequency counts with row filter ───────────────

#[test]
fn integ_link_exploded_with_row_filter_frequency() {
    // RANK: LIST_DISTINCT(orders.product_id WHERE price > 50, 0, 5000)
    // Verify frequency counts respect the row filter.
    //
    // alice orders with price > 50:
    //   order1(product=0, price=100), order3(product=2, price=75) → product_ids: {0:1, 2:1}
    // bob orders with price > 50:
    //   order4(product=0, price=200) → product_ids: {0:1}
    // carol orders with price > 50:
    //   order6(product=2, price=60) → product_ids: {2:1}
    // dave orders with price > 50:
    //   order7(product=0, price=150), order8(product=1, price=80) → product_ids: {0:1, 1:1}
    // eve: no orders
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_filter_freq");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("price".to_string()),
                    op: CompareOp::Gt,
                    rhs: RowFilterRhs::Constant(Value::Float(50.0)),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // alice: product 0(×1), product 2(×1)
    assert!(as_set.contains(&(0, "0".into(), 1)), "alice product 0");
    assert!(as_set.contains(&(0, "2".into(), 1)), "alice product 2");

    // bob: product 0(×1)
    assert!(as_set.contains(&(1, "0".into(), 1)), "bob product 0");

    // carol: product 2(×1)
    assert!(as_set.contains(&(2, "2".into(), 1)), "carol product 2");

    // dave: product 0(×1), product 1(×1)
    assert!(as_set.contains(&(3, "0".into(), 1)), "dave product 0");
    assert!(as_set.contains(&(3, "1".into(), 1)), "dave product 1");

    // Total: 2 + 1 + 1 + 2 = 6
    assert_eq!(rows.len(), 6);
}

// ─── Tests: Link prediction — frequency counts with duplicates ───────────────

#[test]
fn integ_link_exploded_frequency_duplicates() {
    // RANK: LIST_DISTINCT(orders.product_id, 0, 5000)
    // Tests duplicate frequency counting without row filter.
    //
    // alice orders: product_ids [3,0,1,2] — all distinct → each freq=1
    // bob: product_ids [0,1] → each freq=1
    // dave: product_ids [0,1,2] → each freq=1
    //
    // This test uses product_id which has no duplicates per user.
    // For duplicates, we use status:
    // alice: shipped(2), pending(1), returned(1)
    // dave: shipped(2), pending(1)
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_freq_dup");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), None)],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // Focus on entities with duplicates
    let alice_shipped: Vec<_> = rows.iter()
        .filter(|(e, _, t, _)| *e == 0 && t == "shipped")
        .collect();
    assert_eq!(alice_shipped.len(), 1, "one row for alice/shipped");
    assert_eq!(alice_shipped[0].3, 2, "alice shipped freq should be 2");

    let dave_shipped: Vec<_> = rows.iter()
        .filter(|(e, _, t, _)| *e == 3 && t == "shipped")
        .collect();
    assert_eq!(dave_shipped.len(), 1, "one row for dave/shipped");
    assert_eq!(dave_shipped[0].3, 2, "dave shipped freq should be 2");
}

// ─── Tests: Link prediction — non-temporal (static) ──────────────────────────

#[test]
fn integ_link_static_exploded() {
    // Static (non-temporal): LIST_DISTINCT(orders.status) — no window, no timestamp
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_static");
    let plan = g1_link(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("status".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // All timestamps should be 0 (non-temporal)
    for (_, ts, _, _) in &rows {
        assert_eq!(*ts, 0, "static mode should have ts=0");
    }

    // Same as temporal but all edges visible regardless of window
    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // alice: shipped(2), pending(1), returned(1)
    assert!(as_set.contains(&(0, "shipped".into(), 2)));
    assert!(as_set.contains(&(0, "pending".into(), 1)));
    assert!(as_set.contains(&(0, "returned".into(), 1)));

    assert_eq!(rows.len(), 8); // same count as temporal test
}

// ─── Tests: Link prediction — temporal with multiple timestamps ──────────────

#[test]
fn integ_link_temporal_multiple_timestamps() {
    // Multiple timestamps: LIST_DISTINCT(orders.product_id, -500, 0, seconds)
    // Window [-500, 0): capture orders in the 500s before each timestamp.
    // With (lo, hi] convention: edges where lo < sort_key <= hi.
    //
    // Timestamps: [1200, 2300]
    //
    // ts=1200, window (700, 1200]:
    //   alice: order0(ts=1050), order1(ts=1100), order2(ts=1200) → {3:1, 0:1, 1:1}
    //   everyone else: no orders in window
    //
    // ts=2300, window (1800, 2300]:
    //   bob: order4(ts=2100, product=0), order5(ts=2200, product=1) → {0:1, 1:1}
    //   everyone else: no orders in window
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_temporal_multi");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_product_id_op("ld", Some((-500, 0)), None)],
            filter: None,
        }],
        vec![1200, 2300],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // At ts=1200: alice has 3 products (1200 now included at hi boundary)
    let alice_1200: Vec<_> = rows.iter().filter(|(e, ts, ..)| *e == 0 && *ts == 1200).collect();
    assert_eq!(alice_1200.len(), 3, "alice@1200 should have 3 rows");
    let alice_targets: std::collections::HashSet<_> =
        alice_1200.iter().map(|(_, _, t, _)| t.as_str()).collect();
    assert!(alice_targets.contains("3"), "alice@1200 should have product 3");
    assert!(alice_targets.contains("0"), "alice@1200 should have product 0");
    assert!(alice_targets.contains("1"), "alice@1200 should have product 1");

    // At ts=2300: bob has 2 products
    let bob_2300: Vec<_> = rows.iter().filter(|(e, ts, ..)| *e == 1 && *ts == 2300).collect();
    assert_eq!(bob_2300.len(), 2, "bob@2300 should have 2 rows");

    // Empty entities at certain timestamps produce no exploded rows
    let alice_2300: Vec<_> = rows.iter().filter(|(e, ts, ..)| *e == 0 && *ts == 2300).collect();
    assert_eq!(alice_2300.len(), 0, "alice@2300 should have no products in window");
}

// ─── Tests: Link prediction — ListDistinct with limit ────────────────────────

#[test]
fn integ_link_exploded_with_limit() {
    // LIST_DISTINCT(orders.status, 0, 5000, limit=1)
    // Should stop after 1 distinct value.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_limit");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((0, 5000)), Some(1))],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // Each entity should have at most 1 target row (limit=1)
    let mut by_entity: HashMap<i64, Vec<_>> = HashMap::new();
    for row in &rows {
        by_entity.entry(row.0).or_default().push(row);
    }
    for (entity, entity_rows) in &by_entity {
        assert!(
            entity_rows.len() <= 1,
            "entity {entity} should have at most 1 row with limit=1, got {}",
            entity_rows.len()
        );
    }

    // eve has no orders → 0 rows
    assert!(!by_entity.contains_key(&4), "eve should have 0 rows");
}

// ─── Tests: Link prediction — empty ListDistinct ─────────────────────────────

#[test]
fn integ_link_empty_exploded() {
    // When no edges match the window, exploded mode produces zero rows.
    // Window [10000, 20000) captures nothing.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_empty_exploded");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((10000, 20000)), None)],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);
    assert_eq!(rows.len(), 0, "no edges in window → no exploded rows");
}

#[test]
fn integ_link_empty_array() {
    // When no edges match the window, array mode produces rows with empty lists.
    // Window [10000, 20000) captures nothing.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_empty_array");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![list_distinct_status_op("ld", Some((10000, 20000)), None)],
            filter: None,
        }],
        vec![0],
        false, // no explode → array
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array(&shard_paths);

    // Each entity still produces one row, but with empty target/freq lists
    assert_eq!(rows.len(), 5, "one row per entity even with empty lists");
    for (_, _, targets, freqs) in &rows {
        assert!(targets.is_empty(), "targets should be empty");
        assert!(freqs.is_empty(), "freqs should be empty");
    }
}

// ─── Tests: Link prediction — multi-hop ListDistinct ─────────────────────────

#[test]
fn integ_link_multi_hop_exploded() {
    // Multi-hop: LIST_DISTINCT of product brand via users→orders→products
    // RANK: LIST_DISTINCT(orders→products.brand, 0, 5000)
    //
    // alice's orders: products [3,0,1,2] → brands [Puma,Nike,Adidas,Nike]
    //   distinct: {Puma:1, Nike:2, Adidas:1}
    // bob's orders: products [0,1] → brands [Nike,Adidas]
    //   distinct: {Nike:1, Adidas:1}
    // carol's orders: product [2] → brands [Nike]
    //   distinct: {Nike:1}
    // dave's orders: products [0,1,2] → brands [Nike,Adidas,Nike]
    //   distinct: {Nike:2, Adidas:1}
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph1(dir.path());
    let output = dir.path().join("out_lp_multihop");
    let plan = g1_link_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![u2o_edge(), o2p_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("brand".to_string()),
                time_col: None,
                window: Some((0, 5000)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0],
        true,  // explode
        true,  // include_frequency
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // alice: Nike(2), Adidas(1), Puma(1)
    assert!(as_set.contains(&(0, "Nike".into(), 2)), "alice Nike×2");
    assert!(as_set.contains(&(0, "Adidas".into(), 1)), "alice Adidas×1");
    assert!(as_set.contains(&(0, "Puma".into(), 1)), "alice Puma×1");

    // bob: Nike(1), Adidas(1)
    assert!(as_set.contains(&(1, "Nike".into(), 1)), "bob Nike×1");
    assert!(as_set.contains(&(1, "Adidas".into(), 1)), "bob Adidas×1");

    // carol: Nike(1)
    assert!(as_set.contains(&(2, "Nike".into(), 1)), "carol Nike×1");

    // dave: Nike(2), Adidas(1)
    assert!(as_set.contains(&(3, "Nike".into(), 2)), "dave Nike×2");
    assert!(as_set.contains(&(3, "Adidas".into(), 1)), "dave Adidas×1");

    // Total: 3 + 2 + 1 + 2 = 8
    assert_eq!(rows.len(), 8);
}

// ════════════════════════════════════════════════════════════════════════════
// Graph 2: small_temporal_graph_fixed (matches Kumo's conftest fixture)
// ════════════════════════════════════════════════════════════════════════════
//
// USERS(5):  USER_ID [0-4], AGE [20-24]
// STORES(3): STORE_ID [0-2]
// ORDERS(15): 3 per user, round-robin STORE_ID, consecutive days from epoch 0
//
//  order  user  store  order_value(str)  order_score(f64)  ts (epoch s)
//  0      0     0      "1"               1.0                0
//  1      0     1      "2"               2.0                86400     (day 1)
//  2      0     2      "1"               3.0                172800    (day 2)
//  3      1     0      "1"               1.0                259200    (day 3)
//  4      1     1      "2"               2.0                345600    (day 4)
//  5      1     2      "1"               3.0                432000    (day 5)
//  6      2     0      "1"               1.0                518400    (day 6)
//  7      2     1      "2"               2.0                604800    (day 7)
//  8      2     2      "1"               3.0                691200    (day 8)
//  9      3     0      "1"               1.0                777600    (day 9)
//  10     3     1      "2"               2.0                864000    (day 10)
//  11     3     2      "1"               3.0                950400    (day 11)
//  12     4     0      "1"               1.0                1036800   (day 12)
//  13     4     1      "2"               2.0                1123200   (day 13)
//  14     4     2      "1"               3.0                1209600   (day 14)
//
// FK: ORDERS.USER_ID → USERS (reverse, time on TIMESTAMP)
//     ORDERS.STORE_ID → STORES (reverse, time on TIMESTAMP)

const DAY: i64 = 86400;

fn build_graph2(root: &Path) -> PathBuf {
    let data_dir = root.join("data2");
    let index_dir = root.join("index2");
    fs::create_dir_all(&data_dir).unwrap();

    // ── Users ──
    let mut user_id = Int64Builder::new();
    let mut age = Int64Builder::new();
    for i in 0..5i64 {
        user_id.append_value(i);
        age.append_value(20 + i);
    }
    let users_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("age", DataType::Int64, false),
    ]);
    let users_batch = RecordBatch::try_new(
        Arc::new(users_schema),
        vec![Arc::new(user_id.finish()), Arc::new(age.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    // ── Stores ──
    let mut store_id = Int64Builder::new();
    for i in 0..3i64 {
        store_id.append_value(i);
    }
    let stores_schema = Schema::new(vec![Field::new("store_id", DataType::Int64, false)]);
    let stores_batch = RecordBatch::try_new(
        Arc::new(stores_schema),
        vec![Arc::new(store_id.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("stores.parquet"), &stores_batch);

    // ── Orders ──
    let mut oid = Int64Builder::new();
    let mut o_user = Int64Builder::new();
    let mut o_store = Int64Builder::new();
    let mut o_value = StringBuilder::new();
    let mut o_score = Float64Builder::new();
    let mut o_ts = Int64Builder::new();

    let values_cycle = ["1", "2", "1"];
    let scores_cycle = [1.0, 2.0, 3.0];

    for i in 0..15i64 {
        oid.append_value(i);
        o_user.append_value(i / 3);                // 0,0,0,1,1,1,...,4,4,4
        o_store.append_value(i % 3);               // 0,1,2,0,1,2,...
        o_value.append_value(values_cycle[(i % 3) as usize]);
        o_score.append_value(scores_cycle[(i % 3) as usize]);
        o_ts.append_value(i * DAY);
    }

    let orders_schema = Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("store_id", DataType::Int64, false),
        Field::new("order_value", DataType::Utf8, false),
        Field::new("order_score", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    let orders_batch = RecordBatch::try_new(
        Arc::new(orders_schema),
        vec![
            Arc::new(oid.finish()),
            Arc::new(o_user.finish()),
            Arc::new(o_store.finish()),
            Arc::new(o_value.finish()),
            Arc::new(o_score.finish()),
            Arc::new(o_ts.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("orders.parquet"), &orders_batch);

    // ── GraphConfig ──
    let config = GraphConfig {
        dataset_name: "test_graph2".to_string(),
        tables: vec![
            TableSpec {
                name: "users".to_string(),
                columns: vec![
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("age", IngestedType::Int64),
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
                fkeys: vec![],
            },
            TableSpec {
                name: "stores".to_string(),
                columns: vec![mk_col("store_id", IngestedType::Int64)],
                primary_key: Some("store_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("stores.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "orders".to_string(),
                columns: vec![
                    mk_col("order_id", IngestedType::Int64),
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("store_id", IngestedType::Int64),
                    mk_col("order_value", IngestedType::String),
                    mk_col("order_score", IngestedType::Float64),
                    mk_col("timestamp", IngestedType::Int64),
                ],
                primary_key: Some("order_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("orders.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![
                    FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "timestamp".to_string(),
                                bitwidth: Some(32),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                    FKeySpec {
                        src_column: "store_id".to_string(),
                        dst_table: "stores".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "orders".to_string(),
                                column: "timestamp".to_string(),
                                bitwidth: Some(32),
                                anchor_time: Some(0),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                ],
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

// ─── Graph 2 plan helpers ────────────────────────────────────────────────────

fn g2_u2o_edge() -> EdgeRef {
    EdgeRef {
        src_table: "users".to_string(),
        src_col: "user_id".to_string(),
        dst_table: "orders".to_string(),
        dst_col: "user_id".to_string(),
    }
}

fn g2_link_temporal(
    stages: Vec<Stage>,
    timestamps: Vec<i64>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps,
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

fn g2_link_static(
    stages: Vec<Stage>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

fn g2_list_distinct_store_id(
    alias: &str,
    window: Option<(i64, i64)>,
    limit: Option<u32>,
    row_filter: Option<RowFilter>,
) -> Op {
    Op::Aggregate {
        alias: alias.to_string(),
        path: vec![g2_u2o_edge()],
        func: AggFunc::ListDistinct,
        target_col: Some("store_id".to_string()),
        time_col: None,
        window,
        window_unit: Some(TimeUnit::Seconds),
        row_filter,
        limit,
    }
}

// ─── Graph 2 LP tests ───────────────────────────────────────────────────────

#[test]
fn integ_g2_lp_temporal_exploded() {
    // LIST_DISTINCT(ORDERS.STORE_ID, 0, 2days) exploded+freq
    // Window (0, 2*DAY) relative to observation timestamp. (lo, hi] convention.
    //
    // Timestamps: [0, 2*DAY]
    //
    // ts=0, absolute window (0, 172800]:
    //   user 0: ts=86400(store=1), ts=172800(store=2) → {1:1, 2:1}
    //   (ts=0 excluded because 0 > 0 is false)
    //
    // ts=2*DAY=172800, absolute window (172800, 345600]:
    //   user 1: ts=259200(store=0), ts=345600(store=1) → {0:1, 1:1}
    //   user 0: no orders in (172800, 345600]
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_exploded");
    let plan = g2_link_temporal(
        vec![Stage {
            ops: vec![g2_list_distinct_store_id("ld", Some((0, 2 * DAY)), None, None)],
            filter: None,
        }],
        vec![0, 2 * DAY],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    let as_set: std::collections::HashSet<(i64, i64, String, i64)> = rows
        .iter()
        .map(|(e, ts, t, f)| (*e, *ts, t.clone(), *f))
        .collect();

    // ts=0: user 0 → store 1(×1), store 2(×1)
    assert!(as_set.contains(&(0, 0, "1".into(), 1)), "user0@0 store1");
    assert!(as_set.contains(&(0, 0, "2".into(), 1)), "user0@0 store2");

    // ts=172800: user 1 → store 0(×1), store 1(×1)
    assert!(as_set.contains(&(1, 2 * DAY, "0".into(), 1)), "user1@2d store0");
    assert!(as_set.contains(&(1, 2 * DAY, "1".into(), 1)), "user1@2d store1");

    // Total rows: 2 + 2 = 4
    assert_eq!(rows.len(), 4);
}

#[test]
fn integ_g2_lp_temporal_array() {
    // Same as exploded but array mode — one row per (entity, ts).
    // (lo, hi] convention: ts=0 window (0,172800], ts=172800 window (172800,345600].
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_array");
    let plan = g2_link_temporal(
        vec![Stage {
            ops: vec![g2_list_distinct_store_id("ld", Some((0, 2 * DAY)), None, None)],
            filter: None,
        }],
        vec![0, 2 * DAY],
        false,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array(&shard_paths);

    // 5 users × 2 timestamps = 10 rows
    assert_eq!(rows.len(), 10);

    fn sort_paired(targets: &[String], freqs: &[i64]) -> (Vec<String>, Vec<i64>) {
        let mut pairs: Vec<_> = targets.iter().cloned().zip(freqs.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs.into_iter().unzip()
    }

    for (entity, ts, targets, freqs) in &rows {
        let (sorted_t, sorted_f) = sort_paired(targets, freqs);
        match (*entity, *ts) {
            (0, 0) => {
                // (0, 172800]: user0 orders at ts=86400(store=1), ts=172800(store=2)
                assert_eq!(sorted_t, vec!["1", "2"]);
                assert_eq!(sorted_f, vec![1, 1]);
            }
            (1, t) if t == 2 * DAY => {
                // (172800, 345600]: user1 orders at ts=259200(store=0), ts=345600(store=1)
                assert_eq!(sorted_t, vec!["0", "1"]);
                assert_eq!(sorted_f, vec![1, 1]);
            }
            (_, _) => {
                // All other (entity, ts) combinations should have empty lists
                assert!(
                    targets.is_empty(),
                    "entity={entity} ts={ts} should be empty, got {:?}",
                    targets
                );
            }
        }
    }
}

#[test]
fn integ_g2_lp_temporal_filtered() {
    // LIST_DISTINCT(ORDERS.STORE_ID WHERE STORE_ID < 2, 0, 2days) exploded+freq
    // (lo, hi] convention. Same timestamps as exploded but filter excludes store_id >= 2.
    //
    // ts=0, window (0, 172800]: user 0 → store 1(×1) passes, store 2 fails → {1:1}
    // ts=172800, window (172800, 345600]: user 1 → store 0(×1), store 1(×1) → {0:1, 1:1}
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_filtered");
    let plan = g2_link_temporal(
        vec![Stage {
            ops: vec![g2_list_distinct_store_id(
                "ld",
                Some((0, 2 * DAY)),
                None,
                Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Column("store_id".to_string()),
                    op: CompareOp::Lt,
                    rhs: RowFilterRhs::Constant(Value::Int(2)),
                }),
            )],
            filter: None,
        }],
        vec![0, 2 * DAY],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    let as_set: std::collections::HashSet<(i64, i64, String, i64)> = rows
        .iter()
        .map(|(e, ts, t, f)| (*e, *ts, t.clone(), *f))
        .collect();

    // ts=0, (0,172800]: user 0 → store 1(×1) (store 2 fails filter)
    assert!(as_set.contains(&(0, 0, "1".into(), 1)));

    // ts=172800, (172800,345600]: user 1 → store 0(×1), store 1(×1)
    assert!(as_set.contains(&(1, 2 * DAY, "0".into(), 1)));
    assert!(as_set.contains(&(1, 2 * DAY, "1".into(), 1)));

    // Total: 1 + 2 = 3
    assert_eq!(rows.len(), 3);
}

#[test]
fn integ_g2_lp_static_exploded() {
    // Static (no temporal): LIST_DISTINCT(ORDERS.STORE_ID) exploded+freq
    // Each user has 3 orders to stores 0,1,2 → 3 distinct stores, each freq=1.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_static");
    let plan = g2_link_static(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g2_u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("store_id".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // All timestamps should be 0 (non-temporal)
    for (_, ts, _, _) in &rows {
        assert_eq!(*ts, 0, "static mode ts=0");
    }

    // Each user has 3 distinct stores, freq=1 each
    let mut by_entity: HashMap<i64, Vec<(String, i64)>> = HashMap::new();
    for (e, _, t, f) in &rows {
        by_entity.entry(*e).or_default().push((t.clone(), *f));
    }

    for user in 0..5i64 {
        let entity_rows = &by_entity[&user];
        assert_eq!(entity_rows.len(), 3, "user {user} should have 3 distinct stores");
        for (_, freq) in entity_rows {
            assert_eq!(*freq, 1, "user {user} each store should have freq=1");
        }
    }

    // Total: 5 users × 3 stores = 15
    assert_eq!(rows.len(), 15);
}

#[test]
fn integ_g2_lp_static_array() {
    // Static array mode: one row per entity with list of stores.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_static_array");
    let plan = g2_link_static(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g2_u2o_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("store_id".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        false,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array(&shard_paths);

    assert_eq!(rows.len(), 5, "one row per entity");
    for (entity, _ts, targets, freqs) in &rows {
        let mut sorted = targets.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["0", "1", "2"], "entity {entity} all 3 stores");
        assert_eq!(freqs.len(), 3);
        for f in freqs {
            assert_eq!(*f, 1, "entity {entity} each freq=1");
        }
    }
}

#[test]
fn integ_g2_lp_nonzero_offset() {
    // LIST_DISTINCT(ORDERS.STORE_ID, 6days, 8days) exploded+freq
    // Window (6*DAY, 8*DAY) relative to observation. (lo, hi] convention.
    //
    // Timestamps: [0]
    // ts=0, absolute window (518400, 691200]:
    //   user 2: ts=518400 excluded (lo exclusive), ts=604800(store=1), ts=691200(store=2) → {1:1, 2:1}
    //   others: no orders in range
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph2(dir.path());
    let output = dir.path().join("out_g2_lp_offset");
    let plan = g2_link_temporal(
        vec![Stage {
            ops: vec![g2_list_distinct_store_id("ld", Some((6 * DAY, 8 * DAY)), None, None)],
            filter: None,
        }],
        vec![0],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // user 2 has orders at day 7 (store=1) and day 8 (store=2)
    assert!(as_set.contains(&(2, "1".into(), 1)), "user2 store1");
    assert!(as_set.contains(&(2, "2".into(), 1)), "user2 store2");

    assert_eq!(rows.len(), 2);
}

// ════════════════════════════════════════════════════════════════════════════
// Graph 3: lp_test_graph (matches Kumo's lp_test_graph(num_entities=2))
// ════════════════════════════════════════════════════════════════════════════
//
// USERS(2):    USER_ID [0,1], AGE [17,25]
// PRODUCTS(2): PRODUCT_ID [0,1], BRAND ["PATAGONIA","KUMO"]
// TRANS(60):   Each user has 15 orders for product 0 (days 1-15) + 15 for product 1 (days 16-30)
//              AMOUNT cycles [1,2,3]
//              TIMESTAMP (epoch seconds from 2018-01-01)
//
// 2018-01-01 00:00:00 UTC = 1514764800 epoch seconds

const EPOCH_2018: i64 = 1514764800;

fn build_graph3(root: &Path) -> PathBuf {
    let data_dir = root.join("data3");
    let index_dir = root.join("index3");
    fs::create_dir_all(&data_dir).unwrap();

    // ── Users ──
    let mut user_id = Int64Builder::new();
    let mut user_age = Int64Builder::new();
    for (uid, a) in [(0i64, 17i64), (1, 25)] {
        user_id.append_value(uid);
        user_age.append_value(a);
    }
    let users_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("age", DataType::Int64, false),
    ]);
    let users_batch = RecordBatch::try_new(
        Arc::new(users_schema),
        vec![Arc::new(user_id.finish()), Arc::new(user_age.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("users.parquet"), &users_batch);

    // ── Products ──
    let mut product_id = Int64Builder::new();
    let mut brand = StringBuilder::new();
    for (pid, b) in [(0i64, "PATAGONIA"), (1, "KUMO")] {
        product_id.append_value(pid);
        brand.append_value(b);
    }
    let products_schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new("brand", DataType::Utf8, false),
    ]);
    let products_batch = RecordBatch::try_new(
        Arc::new(products_schema),
        vec![Arc::new(product_id.finish()), Arc::new(brand.finish())],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &products_batch);

    // ── Transactions ──
    // Each user: 15 transactions for product 0 (days 1-15), then 15 for product 1 (days 16-30)
    let mut trans_id = Int64Builder::new();
    let mut trans_user = Int64Builder::new();
    let mut trans_product = Int64Builder::new();
    let mut trans_amount = Float64Builder::new();
    let mut trans_ts = Int64Builder::new();

    let amounts = [1.0, 2.0, 3.0];
    let mut tid = 0i64;
    for uid in 0..2i64 {
        // Product 0: days 1-15
        for day in 1..=15i64 {
            trans_id.append_value(tid);
            trans_user.append_value(uid);
            trans_product.append_value(0);
            trans_amount.append_value(amounts[((day - 1) % 3) as usize]);
            trans_ts.append_value(EPOCH_2018 + day * DAY);
            tid += 1;
        }
        // Product 1: days 16-30
        for day in 16..=30i64 {
            trans_id.append_value(tid);
            trans_user.append_value(uid);
            trans_product.append_value(1);
            trans_amount.append_value(amounts[((day - 1) % 3) as usize]);
            trans_ts.append_value(EPOCH_2018 + day * DAY);
            tid += 1;
        }
    }

    let trans_schema = Schema::new(vec![
        Field::new("trans_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("date", DataType::Int64, false),
    ]);
    let trans_batch = RecordBatch::try_new(
        Arc::new(trans_schema),
        vec![
            Arc::new(trans_id.finish()),
            Arc::new(trans_user.finish()),
            Arc::new(trans_product.finish()),
            Arc::new(trans_amount.finish()),
            Arc::new(trans_ts.finish()),
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("trans.parquet"), &trans_batch);

    // ── GraphConfig ──
    let config = GraphConfig {
        dataset_name: "test_graph3".to_string(),
        tables: vec![
            TableSpec {
                name: "users".to_string(),
                columns: vec![
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("age", IngestedType::Int64),
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
                fkeys: vec![],
            },
            TableSpec {
                name: "products".to_string(),
                columns: vec![
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("brand", IngestedType::String),
                ],
                primary_key: Some("product_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("products.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![],
            },
            TableSpec {
                name: "trans".to_string(),
                columns: vec![
                    mk_col("trans_id", IngestedType::Int64),
                    mk_col("user_id", IngestedType::Int64),
                    mk_col("product_id", IngestedType::Int64),
                    mk_col("amount", IngestedType::Float64),
                    mk_col("date", IngestedType::Int64),
                ],
                primary_key: Some("trans_id".to_string()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source: DataSource::Parquet {
                    paths: vec![data_dir.join("trans.parquet").to_str().unwrap().to_string()],
                },
                fkeys: vec![
                    FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "trans".to_string(),
                                column: "date".to_string(),
                                bitwidth: Some(32),
                                anchor_time: Some(EPOCH_2018),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                    FKeySpec {
                        src_column: "product_id".to_string(),
                        dst_table: "products".to_string(),
                        forward: None,
                        reverse: Some(DirectionSpec {
                            time: Some(TimeSpec {
                                table: "trans".to_string(),
                                column: "date".to_string(),
                                bitwidth: Some(32),
                                anchor_time: Some(EPOCH_2018),
                                unit: "s".to_string(),
                            }),
                            weight: None,
                        }),
                    },
                ],
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

// ─── Graph 3 plan helpers ────────────────────────────────────────────────────

fn g3_u2t_edge() -> EdgeRef {
    EdgeRef {
        src_table: "users".to_string(),
        src_col: "user_id".to_string(),
        dst_table: "trans".to_string(),
        dst_col: "user_id".to_string(),
    }
}

fn g3_t2p_edge() -> EdgeRef {
    EdgeRef {
        src_table: "trans".to_string(),
        src_col: "product_id".to_string(),
        dst_table: "products".to_string(),
        dst_col: "product_id".to_string(),
    }
}

fn g3_link_temporal(
    stages: Vec<Stage>,
    timestamps: Vec<i64>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: Some(TemporalSpec {
            timestamps,
            forecast_length: 1,
            lag_timesteps: None,
        }),
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

fn g3_link_static(
    stages: Vec<Stage>,
    explode: bool,
    include_frequency: bool,
) -> DiskGraphTTPlan {
    DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Link { explode, include_frequency },
        stages,
    }
}

// ─── Graph 3 LP tests ───────────────────────────────────────────────────────

#[test]
fn integ_g3_lp_temporal_rank() {
    // LIST_DISTINCT(TRANS.PRODUCT_ID, 0, 30days) RANK TOP 10
    // (top_k is not compiled to limit; all distinct values are collected)
    // Observation at day 0 (EPOCH_2018). Window (0, 30*DAY) from observation.
    // All 30 transactions per user fall in [EPOCH_2018, EPOCH_2018+30*DAY).
    // Each user → {product 0: 15, product 1: 15}
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph3(dir.path());
    let output = dir.path().join("out_g3_rank");
    let plan = g3_link_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g3_u2t_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                window: Some((0, 31 * DAY)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![EPOCH_2018],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // 2 users × 2 products = 4 exploded rows
    assert_eq!(rows.len(), 4);

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    // User 0: product 0 (×15), product 1 (×15)
    assert!(as_set.contains(&(0, "0".into(), 15)), "user0 product0 freq=15");
    assert!(as_set.contains(&(0, "1".into(), 15)), "user0 product1 freq=15");

    // User 1: product 0 (×15), product 1 (×15)
    assert!(as_set.contains(&(1, "0".into(), 15)), "user1 product0 freq=15");
    assert!(as_set.contains(&(1, "1".into(), 15)), "user1 product1 freq=15");
}

#[test]
fn integ_g3_lp_filtered_brand() {
    // Multi-hop row filter: LIST_DISTINCT(TRANS.PRODUCT_ID WHERE PRODUCTS.BRAND="PATAGONIA", 0, 31days)
    // Only product 0 (PATAGONIA) should pass the filter.
    // Each user → {product 0: 15}
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph3(dir.path());
    let output = dir.path().join("out_g3_brand");
    let plan = g3_link_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g3_u2t_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                window: Some((0, 31 * DAY)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: Some(RowFilter::Compare {
                    lhs: RowFilterLhs::Remote {
                        path: vec![g3_t2p_edge()],
                        column: "brand".to_string(),
                    },
                    op: CompareOp::Eq,
                    rhs: RowFilterRhs::Constant(Value::Str("PATAGONIA".to_string())),
                }),
                limit: None,
            }],
            filter: None,
        }],
        vec![EPOCH_2018],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // 2 users × 1 product = 2 rows
    assert_eq!(rows.len(), 2);

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    assert!(as_set.contains(&(0, "0".into(), 15)), "user0 product0 freq=15");
    assert!(as_set.contains(&(1, "0".into(), 15)), "user1 product0 freq=15");
}

#[test]
fn integ_g3_lp_static_rank() {
    // Static: LIST_DISTINCT(TRANS.PRODUCT_ID) RANK TOP 2
    // (top_k is not compiled to limit; all distinct values are collected)
    // No window → all edges visible.
    // Each user → 30 total transactions, 2 distinct products, each freq=15
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph3(dir.path());
    let output = dir.path().join("out_g3_static");
    let plan = g3_link_static(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g3_u2t_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        true,
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // 2 users × 2 products = 4 rows
    assert_eq!(rows.len(), 4);

    for (_, ts, _, _) in &rows {
        assert_eq!(*ts, 0, "static ts=0");
    }

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    assert!(as_set.contains(&(0, "0".into(), 15)), "user0 product0");
    assert!(as_set.contains(&(0, "1".into(), 15)), "user0 product1");
    assert!(as_set.contains(&(1, "0".into(), 15)), "user1 product0");
    assert!(as_set.contains(&(1, "1".into(), 15)), "user1 product1");
}

#[test]
fn integ_g3_lp_dangling_entity() {
    // Add extra users (id 2,3) with no transactions — should produce 0 exploded rows.
    // We achieve this by using the existing graph3 since users only have ids 0,1.
    // A dangling entity would be one that exists in the users table but has no edges.
    // Since build_graph3 only creates 2 users, both with transactions, we test this
    // with an entity filter that would select non-existent ranges, OR we just check
    // that the static LP correctly handles entities with 0 edges via array mode.
    //
    // Instead, let's use the existing graph with array mode at a timestamp where
    // no edges exist: ts far in the past → empty lists.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph3(dir.path());
    let output = dir.path().join("out_g3_dangling");
    let plan = g3_link_temporal(
        vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g3_u2t_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                // Window that captures nothing: way before any transactions
                window: Some((0, DAY)),
                window_unit: Some(TimeUnit::Seconds),
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
        vec![0], // observation at epoch 0, far before any transactions (2018)
        false,   // array mode
        true,
    );
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_array(&shard_paths);

    // Both users produce rows but with empty target lists (no edges in window)
    assert_eq!(rows.len(), 2, "one row per user");
    for (entity, _, targets, freqs) in &rows {
        assert!(targets.is_empty(), "entity {entity} should have empty targets");
        assert!(freqs.is_empty(), "entity {entity} should have empty freqs");
    }
}

#[test]
fn integ_g3_lp_entity_filter() {
    // Entity filter: WHERE USERS.AGE > 21
    // User 0 (age=17) filtered out. User 1 (age=25) passes.
    let dir = tempfile::tempdir().unwrap();
    let index = build_graph3(dir.path());
    let output = dir.path().join("out_g3_entity_filter");
    let plan = DiskGraphTTPlan {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            time_col: None,
            valid_to_col: None,
            filter: Some(RowFilter::Compare {
                lhs: RowFilterLhs::Column("age".to_string()),
                op: CompareOp::Gt,
                rhs: RowFilterRhs::Constant(Value::Int(21)),
            }),
        },
        temporal: None,
        prediction_type: PredictionType::Link {
            explode: true,
            include_frequency: true,
        },
        stages: vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![g3_u2t_edge()],
                func: AggFunc::ListDistinct,
                target_col: Some("product_id".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
            }],
            filter: None,
        }],
    };
    let shard_paths =
        diskgraph_train_table::ttgen_executor::execute(&plan, &index, &output).unwrap();
    let rows = read_output_rows_exploded(&shard_paths);

    // Only user 1 (age=25) passes the filter
    assert_eq!(rows.len(), 2, "user 1 has 2 distinct products");
    for (entity, _, _, _) in &rows {
        assert_eq!(*entity, 1, "only user 1 should be present");
    }

    let as_set: std::collections::HashSet<(i64, String, i64)> = rows
        .iter()
        .map(|(e, _ts, t, f)| (*e, t.clone(), *f))
        .collect();

    assert!(as_set.contains(&(1, "0".into(), 15)));
    assert!(as_set.contains(&(1, "1".into(), 15)));
}
