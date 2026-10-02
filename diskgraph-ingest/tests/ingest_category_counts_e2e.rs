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
