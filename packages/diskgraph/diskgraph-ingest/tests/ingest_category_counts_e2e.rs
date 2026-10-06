// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage that single (non-multi) categorical columns emit a
//! populated `CATEGORY_COUNTS` vocab through the real ingest pipeline — the
//! encoder builds its vocab from this, so an empty list silently drops every
//! categorical feature to OOV.

use anyhow::Result;
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::col_stats::CategoryCountStrategy;
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, batch: &RecordBatch) {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).unwrap();
    }
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn read_kumo_stats(
    index_dir: &Path,
    table: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let path = index_dir
        .join("kumo_col_stats")
        .join("col_stats_full")
        .join(format!("{}.json", table));
    let bytes = fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    Ok(v.as_object().cloned().expect("kumo stats is a JSON object"))
}

fn category_counts(
    stats: &serde_json::Map<String, serde_json::Value>,
    col: &str,
) -> HashMap<String, u64> {
    let key = format!("{col}__kumo.categorical__CATEGORY_COUNTS__False");
    let entry = stats.get(&key).unwrap_or_else(|| {
        panic!(
            "missing {key}; present keys: {:?}",
            stats.keys().collect::<Vec<_>>()
        )
    });
    entry["value"]
        .as_array()
        .expect("CATEGORY_COUNTS value is an array")
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            let k = match p.first()? {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Some((k, p.get(1)?.as_u64()?))
        })
        .collect()
}

fn assert_no_public_category_cap_stat(stats: &serde_json::Map<String, serde_json::Value>) {
    assert!(stats
        .keys()
        .all(|key| !key.contains("CATEGORY_COUNTS_CAPPED")));
}

#[test]
fn ingest_single_categoricals_emit_populated_category_counts() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let ids: Vec<i64> = (0..6).collect();
    let color = vec!["red", "blue", "red", "green", "red", "blue"];
    let region = Int64Array::from(vec![1_i64, 2, 1, 3, 1, 2]);
    let flag = BooleanArray::from(vec![true, false, true, true, false, true]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("color", DataType::Utf8, true),
        Field::new("region", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(color)) as ArrayRef,
            Arc::new(region) as ArrayRef,
            Arc::new(flag) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("rows.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "cat_counts_test",
        "num_shards": 1,
        "tables": [{
            "name": "rows",
            "columns": [
                {"name": "row_id", "ingested_dtype": "int64"},
                {"name": "color", "ingested_dtype": "string", "semantic_type": "categorical"},
                {"name": "region", "ingested_dtype": "int64", "semantic_type": "categorical"},
                {"name": "flag", "ingested_dtype": "bool", "semantic_type": "categorical"}
            ],
            "primary_key": "row_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("rows.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    ingest_with_stats(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            col_stats: true,
            cat_count_strategy: CategoryCountStrategy::Hybrid,
            ..IngestOptions::default()
        },
    )?;

    let stats = read_kumo_stats(&index_dir, "rows")?;

    assert_eq!(
        category_counts(&stats, "color"),
        HashMap::from([("red".into(), 3), ("blue".into(), 2), ("green".into(), 1)]),
    );
    assert_eq!(
        category_counts(&stats, "region"),
        HashMap::from([("1".into(), 3), ("2".into(), 2), ("3".into(), 1)]),
    );
    assert_eq!(
        category_counts(&stats, "flag"),
        HashMap::from([("true".into(), 4), ("false".into(), 2)]),
    );
    assert_no_public_category_cap_stat(&stats);
    Ok(())
}

#[test]
fn ingest_scalar_float_signed_zero_serializes_one_category() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.0, -0.0, 1.5])) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("rows.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "cat_float_signed_zero_test",
        "num_shards": 1,
        "tables": [{
            "name": "rows",
            "columns": [
                {"name": "row_id", "ingested_dtype": "int64"},
                {"name": "value", "ingested_dtype": "float64", "semantic_type": "categorical"}
            ],
            "primary_key": "row_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("rows.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    ingest_with_stats(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            col_stats: true,
            cat_count_strategy: CategoryCountStrategy::Hybrid,
            ..IngestOptions::default()
        },
    )?;

    let stats = read_kumo_stats(&index_dir, "rows")?;
    assert_eq!(
        category_counts(&stats, "value"),
        HashMap::from([("0.0".into(), 2), ("1.5".into(), 1)]),
    );
    assert_no_public_category_cap_stat(&stats);
    assert_eq!(
        stats["value__kumo.categorical__NUM_UNIQUE__False"]["value"],
        serde_json::json!(2)
    );
    Ok(())
}

/// Through the ingest binary: long gxhash twins now hash apart, and short
/// strings and integers whose 64-bit hashes still collide are counted apart.
#[test]
#[cfg(target_arch = "x86_64")]
fn ingest_binary_counts_colliding_categories_apart() -> Result<()> {
    use diskgraph_core::accumulators::HyperLogLog;

    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let output_dir = dir.path().join("output");
    fs::create_dir_all(&data_dir)?;

    let (left, right) = ptr_hash::hash::gxhash_twins();
    let (code_a, code_b) = ("c6e7edf31ad42508", "1226478a63bc70d3");
    let (id_a, id_b) = (
        -6_792_358_849_790_161_234_i64,
        -3_036_548_811_846_647_200_i64,
    );
    assert_eq!(HyperLogLog::hash_str(code_a), HyperLogLog::hash_str(code_b));
    assert_eq!(HyperLogLog::hash_i64(id_a), HyperLogLog::hash_i64(id_b));

    let pick = |i: usize| [0, 1, 0, 1, 1, 2][i];
    let emails = [left.as_str(), right.as_str(), "short@example.com"];
    let codes = [code_a, code_b, "other"];
    let ids = [id_a, id_b, 7];
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("code", DataType::Utf8, false),
        Field::new("account", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..6)) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                (0..6).map(|i| emails[pick(i)]),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                (0..6).map(|i| codes[pick(i)]),
            )) as ArrayRef,
            Arc::new(Int64Array::from_iter_values((0..6).map(|i| ids[pick(i)]))) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("rows.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "cat_colliding",
        "num_shards": 1,
        "tables": [{
            "name": "rows",
            "columns": [
                {"name": "row_id", "ingested_dtype": "int64"},
                {"name": "email", "ingested_dtype": "string", "semantic_type": "categorical"},
                {"name": "code", "ingested_dtype": "string", "semantic_type": "categorical"},
                {"name": "account", "ingested_dtype": "int64", "semantic_type": "categorical"}
            ],
            "primary_key": "row_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("rows.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    let process = std::process::Command::new(env!("CARGO_BIN_EXE_ingest"))
        .env("NO_COLOR", "1")
        .arg("--source")
        .arg(&config_path)
        .arg("--output")
        .arg(&output_dir)
        .args([
            "--dest",
            "local",
            "--col-stats",
            "--cat-count-strategy",
            "hybrid",
        ])
        .output()?;
    assert!(
        process.status.success(),
        "{}",
        String::from_utf8_lossy(&process.stderr)
    );

    let log = String::from_utf8_lossy(&process.stdout);
    let warnings = log
        .lines()
        .filter(|line| line.contains("share a 64-bit hash"))
        .collect::<Vec<_>>();
    assert_eq!(warnings.len(), 2, "{log}");
    for column in ["code", "account"] {
        let field = format!("column=\"{column}\" colliding_values=1");
        assert!(warnings.iter().any(|w| w.contains(&field)), "{log}");
    }
    assert!(warnings
        .iter()
        .all(|w| !w.contains(code_b) && !w.contains(&id_b.to_string())));

    let stats = read_kumo_stats(&output_dir, "rows")?;
    let expected = |values: [String; 3]| {
        HashMap::from([
            (values[0].clone(), 2),
            (values[1].clone(), 3),
            (values[2].clone(), 1),
        ])
    };
    assert_eq!(
        category_counts(&stats, "email"),
        expected(emails.map(str::to_owned))
    );
    assert_eq!(
        category_counts(&stats, "code"),
        expected(codes.map(str::to_owned))
    );
    assert_eq!(
        category_counts(&stats, "account"),
        expected(ids.map(|id| id.to_string()))
    );
    Ok(())
}
