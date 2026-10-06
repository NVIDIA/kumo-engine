// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for multicategorical column stats.
//!
//! Two layers are exercised:
//!
//! 1. **Runtime**: build a parquet fixture + diskgraph config that declares
//!    a `multicategorical` column (comma-delimited string, list<utf8>,
//!    list<int64>, list<float64>), run `ingest_with_stats`, read the emitted
//!    per-table kumo stats JSON, assert the three `MCAT__...` keys tt_executor
//!    also emits are present with the expected shape and byte-parity key
//!    format.
//!
//! 2. **Convert-time delimiter detection**: build a parquet fixture + a
//!    Kumo-side config declaring `stype=multicategorical` without an explicit
//!    delimiter. Call `convert_kumo_to_diskgraph`. Assert the resulting
//!    `ColumnSpec.multicategorical_delimiter` is set correctly for the clean
//!    case and for sparse mixed-candidate data.

use anyhow::Result;
use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, LargeStringArray as ArrowLargeStringArray, RecordBatch,
    StringArray as ArrowStringArray,
};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::col_stats::CategoryCountStrategy;
use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
};
use diskgraph_ingest::errors::ConvertError;
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, batch: &RecordBatch) {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).unwrap();
    }
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

/// Read the kumo-format per-table stats JSON written by ingest to
/// `<root>/kumo_col_stats/col_stats_full/<table>.json`.
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

// =============================================================================
// Layer 1 — runtime ingest with multicategorical columns
// =============================================================================

#[test]
fn ingest_multicategorical_string_scalar_emits_three_multi_keys() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Parquet with a string column containing comma-separated tags.
    let item_ids: Vec<i64> = vec![0, 1, 2, 3];
    let tags = vec!["kids,outdoor,sale", "audio,pro", "kids", ""];
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    // DiskGraph config declares the multicategorical column with an explicit
    // delimiter — the convert-time detector isn't exercised here (layer 2
    // covers that path).
    let config = serde_json::json!({
        "dataset_name": "mc_str_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical",
                    "multicategorical_delimiter": ","
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    let stats = read_kumo_stats(&index_dir, "items")?;

    // Three keys matching tt_executor's multicat emit format.
    let sep = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False")
        .expect("separator key present");
    assert_eq!(sep["value"], ",");

    let counts = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
        .expect("category-counts key present");
    let arr = counts["value"].as_array().expect("array");
    // Expected: kids:2 (rows 0 and 2), outdoor:1, sale:1, audio:1, pro:1.
    // Row 3 is the empty string `""`; under GraphEngine-parity drop-empty
    // semantics it produces zero tokens (matches `pyspark.sql.functions.split`
    // followed by `filter(elem != '')` in `kumo/data/data_processing_backend/
    // pyspark_processing_backend.py:267`). Pre-#28478 this row contributed an
    // empty `""` token to category counts.
    let map: std::collections::HashMap<String, u64> = arr
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            let k = p.first()?.as_str()?.to_string();
            let v = p.get(1)?.as_u64()?;
            Some((k, v))
        })
        .collect();
    assert_eq!(map.get("kids").copied(), Some(2));
    assert_eq!(map.get("outdoor").copied(), Some(1));
    assert_eq!(map.get("sale").copied(), Some(1));
    assert_eq!(map.get("audio").copied(), Some(1));
    assert_eq!(map.get("pro").copied(), Some(1));
    assert_eq!(
        map.get("").copied(),
        None,
        "empty-string row should produce zero tokens under GE-parity drop-empty"
    );

    let uniq = stats
        .get("tags__kumo.multicategorical__NUM_UNIQUE_MULTI__True")
        .expect("num-unique key present");
    assert_eq!(uniq["approx"], true);
    assert!(uniq["value"].as_u64().unwrap() >= 1);

    // Universal stats emitted under the same stype key.
    assert!(stats.contains_key("tags__kumo.multicategorical__COUNT__False"));
    assert!(stats.contains_key("tags__kumo.multicategorical__NUM_NA__False"));

    Ok(())
}

#[test]
fn ingest_multicategorical_stringarray_emits_null_separator() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // List<Utf8> column: ingest should emit MULTI_CATEGORIES_SEPARATOR = null
    // and drop null list elements while preserving legitimate empty strings.
    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(StringBuilder::new());

    id_b.append_value(0);
    list_b.values().append_value("a");
    list_b.values().append_null();
    list_b.values().append_value("b");
    list_b.append(true);

    id_b.append_value(1);
    list_b.values().append_value("b");
    list_b.values().append_value("c");
    list_b.values().append_value("");
    list_b.append(true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mc_list_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string_array",
                    "semantic_type": "multicategorical"
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    let stats = read_kumo_stats(&index_dir, "items")?;
    let sep = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False")
        .expect("separator key present");
    assert_eq!(sep["value"], serde_json::Value::Null);

    let counts = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
        .expect("category-counts key present");
    let arr = counts["value"].as_array().expect("array");
    // Expected: a:1, b:2, c:1, "":1. The native Arrow null element is dropped.
    let map: std::collections::HashMap<String, u64> = arr
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            Some((p.first()?.as_str()?.to_string(), p.get(1)?.as_u64()?))
        })
        .collect();
    assert_eq!(map.get("a").copied(), Some(1));
    assert_eq!(map.get("b").copied(), Some(2));
    assert_eq!(map.get("c").copied(), Some(1));
    assert_eq!(map.get("").copied(), Some(1));
    Ok(())
}

#[test]
fn ingest_multicategorical_int64array_stringifies_tokens() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(Int64Builder::new());
    // row 0: [10, null, 20]; row 1: [10, -3].
    id_b.append_value(0);
    list_b.values().append_value(10);
    list_b.values().append_null();
    list_b.values().append_value(20);
    list_b.append(true);
    id_b.append_value(1);
    list_b.values().append_value(10);
    list_b.values().append_value(-3);
    list_b.append(true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mc_intlist_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "ids",
                    "ingested_dtype": "int64_array",
                    "semantic_type": "multicategorical"
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    let stats = read_kumo_stats(&index_dir, "items")?;
    let counts = stats
        .get("ids__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
        .expect("category-counts key present");
    let arr = counts["value"].as_array().expect("array");
    let map: std::collections::HashMap<String, u64> = arr
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            Some((p.first()?.as_str()?.to_owned(), p.get(1)?.as_u64()?))
        })
        .collect();
    assert_eq!(map.get("10").copied(), Some(2));
    assert_eq!(map.get("20").copied(), Some(1));
    assert_eq!(map.get("-3").copied(), Some(1));
    assert!(!map.contains_key("0"));
    Ok(())
}

#[test]
fn ingest_multicategorical_float64array_drops_nan() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(Float64Builder::new());

    id_b.append_value(0);
    list_b.values().append_value(f64::NAN);
    list_b.values().append_value(1.0);
    list_b.values().append_null();
    list_b.values().append_value(2.5);
    list_b.append(true);

    id_b.append_value(1);
    list_b.values().append_value(1.0);
    list_b.values().append_value(f64::NAN);
    list_b.append(true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "vals",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mc_floatlist_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "vals",
                    "ingested_dtype": "float64_array",
                    "semantic_type": "multicategorical"
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    let stats = read_kumo_stats(&index_dir, "items")?;
    let counts = stats
        .get("vals__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
        .expect("category-counts key present");
    let arr = counts["value"].as_array().expect("array");
    // String keys now (core helper always emits strings). NaN and native Arrow
    // null elements were dropped; row 0 contributes {1.0, 2.5}, row 1 contributes {1.0}.
    // Expected: 1.0×2, 2.5×1.
    let map: std::collections::HashMap<String, u64> = arr
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            let k = p.first()?.as_str()?.to_owned();
            Some((k, p.get(1)?.as_u64()?))
        })
        .collect();
    assert_eq!(map.get("1").copied(), Some(2));
    assert_eq!(map.get("2.5").copied(), Some(1));
    assert!(!map.contains_key("0"));
    Ok(())
}

// =============================================================================
// Layer 2 — convert-time delimiter detection
// =============================================================================

/// Build a minimal Kumo-side config with a table that has an `item_id` PK and a
/// `tags` string column declared as `multicategorical` with no explicit
/// delimiter — the converter is supposed to sample the parquet and detect it.
///
/// Field names match `types::KumoGraphConfig` / `KumoTable`: `cols` (not
/// `columns`), `data_source` (the connector id used for fpv3 path building —
/// not actually hit here because `source_paths` is supplied as an override).
fn build_kumo_config(_data_dir: &Path, tag_delim: Option<&str>) -> serde_json::Value {
    let mut tags_col = serde_json::json!({
        "name": "tags",
        "dtype": "string",
        "stype": "multicategorical"
    });
    if let Some(d) = tag_delim {
        tags_col["multicategorical_delimiter"] = serde_json::Value::String(d.to_string());
    }
    serde_json::json!({
        "tables": [{
            "table_name": "items",
            "data_source": "test_ds",
            "data_version": "v1",
            "pkey": {"name": "item_id", "dtype": "int64"},
            "cols": [tags_col]
        }]
    })
}

fn write_tags_parquet(path: &Path, rows: &[&str]) {
    let ids: Vec<i64> = (0..rows.len() as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(rows.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(path, &batch);
}

fn write_tags_parquet_nullable(path: &Path, rows: &[Option<&str>]) {
    let ids: Vec<i64> = (0..rows.len() as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(rows.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(path, &batch);
}

fn write_tags_large_parquet(path: &Path, rows: &[&str]) {
    let ids: Vec<i64> = (0..rows.len() as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::LargeUtf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(ids)) as ArrayRef,
            Arc::new(ArrowLargeStringArray::from(rows.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(path, &batch);
}

#[test]
fn convert_multicat_string_detects_comma() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let rows: Vec<&str> = (0..100).map(|_| "kids,outdoor,sale").collect();
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_comma".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some(","));
    Ok(())
}

#[test]
fn convert_multicat_string_detects_sparse_comma() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    // Sparse multicategorical data: most rows have one token, but comma is
    // still the only observed delimiter.
    let mut rows: Vec<&str> = Vec::new();
    rows.extend(std::iter::repeat_n("solo", 990));
    rows.extend(std::iter::repeat_n("kids,outdoor", 10));
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_sparse_comma".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some(","));
    assert_eq!(
        col.ingested_dtype,
        diskgraph_ingest::graph_config::IngestedType::StringArray,
    );
    Ok(())
}

#[test]
fn convert_multicat_string_rejects_bracketed_json_rows() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let rows: Vec<&str> = (0..100).map(|_| r#"["kids","outdoor"]"#).collect();
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_bracketed".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter, None);
    Ok(())
}

#[test]
fn convert_multicat_string_detects_pipe() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let rows: Vec<&str> = (0..100).map(|_| "a|b|c").collect();
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_pipe".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some("|"));
    Ok(())
}

#[test]
fn convert_multicat_string_scans_past_first_file() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let null_rows: Vec<Option<&str>> = vec![None; 100];
    write_tags_parquet_nullable(&data_dir.join("part_00000.parquet"), &null_rows);
    let rows: Vec<&str> = (0..100).map(|_| "a,b,c").collect();
    write_tags_parquet(&data_dir.join("part_00001.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_second_file".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some(","));
    Ok(())
}

#[test]
fn convert_multicat_string_detects_large_utf8() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let rows: Vec<&str> = (0..100).map(|_| "a|b|c").collect();
    write_tags_large_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_large_utf8".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some("|"));
    Ok(())
}

#[test]
fn convert_multicat_string_picks_most_common_delimiter() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    // Auto-detection picks the delimiter that appears in the most sampled
    // rows. It does not fail the conversion for competing candidates.
    let mut rows: Vec<&str> = Vec::new();
    rows.extend(std::iter::repeat_n("a,b,c", 40));
    rows.extend(std::iter::repeat_n("x|y|z", 60));
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_most_common".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some("|"));
    Ok(())
}

#[test]
fn convert_multicat_string_explicit_delimiter_respected() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    // Content that would also trip the auto-detector (commas + pipes both
    // present), but using `\t` as the actual separator so the explicit
    // declaration matches the source. Detection should be skipped because
    // the user set an explicit delimiter, and post-#28478 the
    // delim-vs-data validation should pass because the declared `\t`
    // really does appear in every row. Pre-#28478 the data could be
    // arbitrary and detection-skipping was the only assertion; post-flip
    // a wrong delim would bake into the rowpack on disk, so the data
    // fixture is tightened to match the declared delimiter.
    let mut rows: Vec<&str> = Vec::new();
    rows.extend(std::iter::repeat_n("a,b\tc|d", 50));
    rows.extend(std::iter::repeat_n("x|y\tz,w", 50));
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, Some("\t"));
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_explicit_delim".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some("\t"));
    Ok(())
}

// =============================================================================
// kumo#28478 — pseudo-list normalize-at-ingest tests
// =============================================================================

/// Drive each GraphEngine drop-empty edge case through ingest and verify the
/// emitted category counts. Reference:
/// `kumo/data/data_processing_backend/pyspark_processing_backend.py:267-279`.
#[test]
fn ingest_multicat_delim_string_edge_rows_match_ge_drop_empty() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Each fixture row exercises one edge case.
    //   row 0: "a,b"   → ["a", "b"]
    //   row 1: ""      → []                   (empty input → empty list)
    //   row 2: "a,"    → ["a"]                (trailing empty dropped)
    //   row 3: "a,,b"  → ["a", "b"]           (middle empty dropped)
    //   row 4: ",a,"   → ["a"]                (leading + trailing empty dropped)
    //   row 5: ","     → []                   (delim-only row → empty list)
    //   row 6: "x"     → ["x"]                (single token, no delim)
    let item_ids: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6];
    let tags = vec!["a,b", "", "a,", "a,,b", ",a,", ",", "x"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mc_edge_rows",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical",
                    "multicategorical_delimiter": ","
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    let stats = read_kumo_stats(&index_dir, "items")?;
    let counts = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False")
        .expect("category-counts key present");
    let arr = counts["value"].as_array().expect("array");
    let map: std::collections::HashMap<String, u64> = arr
        .iter()
        .filter_map(|pair| {
            let p = pair.as_array()?;
            let k = p.first()?.as_str()?.to_string();
            let v = p.get(1)?.as_u64()?;
            Some((k, v))
        })
        .collect();

    // Expected per-token tallies (after drop-empty):
    //   a: 4 (rows 0, 2, 3, 4)
    //   b: 2 (rows 0, 3)
    //   x: 1 (row 6)
    //   "": 0
    assert_eq!(map.get("a").copied(), Some(4));
    assert_eq!(map.get("b").copied(), Some(2));
    assert_eq!(map.get("x").copied(), Some(1));
    assert_eq!(
        map.get("").copied(),
        None,
        "GE-parity drop-empty: empty tokens never appear in counts",
    );

    // Separator stat is preserved post-flip so downstream stat consumers
    // continue to see the same schema regardless of the pre-split.
    let sep = stats
        .get("tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False")
        .expect("separator key present");
    assert_eq!(sep["value"], ",");
    Ok(())
}

/// `convert_kumo_to_diskgraph` is the canonical Kumo→DiskGraph conversion entry.
/// After it returns, a `(stype=multicategorical, dtype=string, delim)` column
/// must already be normalized to `IngestedType::StringArray` so callers that
/// inspect the converted config see the post-flip shape (the same shape is
/// re-applied unconditionally at ingest entry as a defense-in-depth guarantee).
#[test]
fn convert_normalizes_multicat_delim_string_to_stringarray() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;

    // Parquet with tags column whose token universe is unambiguously
    // comma-delimited so the auto-detector picks `,`.
    let item_ids: Vec<i64> = vec![0, 1, 2];
    let tags = vec!["a,b,c", "d,e", "a,c"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "items",
            "data_source": "test",
            "data_version": "v1",
            "pkey": {"name": "item_id", "dtype": "int", "stype": "ID"},
            "fkeys": [],
            "cols": [{
                "name": "tags",
                "dtype": "string",
                "stype": "multicategorical"
            }],
            "time_col": null,
            "end_time_col": null,
            "edge_table": null
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let opts = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_normalize".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &opts)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    // The flip lives in `convert_kumo_to_diskgraph` post-detect; verify it ran.
    assert_eq!(
        col.ingested_dtype,
        diskgraph_ingest::graph_config::IngestedType::StringArray,
        "convert should leave the multicat column in post-normalize StringArray shape",
    );
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some(","));
    Ok(())
}

/// Sparse multi-token data is *valid* under an explicit delimiter: rows with
/// a single tag tokenize as a one-element list under GE-parity drop-empty.
/// The validator should not reject a config just because most rows have
/// only one token — only total mismatch (zero rows containing the delim)
/// is the real error.
#[test]
fn convert_accepts_explicit_delim_with_mostly_single_token_rows() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;

    // 90% single-token rows, 10% multi-token. Comma appears in only 10% of
    // rows, well below the 50% auto-detect winner threshold but the user
    // declared it explicitly so the validator should accept.
    let mut tags: Vec<&str> = Vec::new();
    tags.extend(std::iter::repeat_n("solo", 90));
    tags.extend(std::iter::repeat_n("a,b", 10));
    let item_ids: Vec<i64> = (0..tags.len() as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "items",
            "data_source": "test",
            "data_version": "v1",
            "pkey": {"name": "item_id", "dtype": "int", "stype": "ID"},
            "fkeys": [],
            "cols": [{
                "name": "tags",
                "dtype": "string",
                "stype": "multicategorical",
                "multicategorical_delimiter": ","
            }],
            "time_col": null,
            "end_time_col": null,
            "edge_table": null
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let opts = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_sparse_delim".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    // Should succeed — explicit declarations get the floor of "delim must
    // appear at least once", not the 50% auto-detect-winner threshold.
    let cfg = convert_kumo_to_diskgraph(&kumo, &opts)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(
        col.ingested_dtype,
        diskgraph_ingest::graph_config::IngestedType::StringArray,
    );
    Ok(())
}

/// Explicit validation uses a larger budget than automatic detection so a
/// sparse declared delimiter can appear after the first 1,000 values.
#[test]
fn convert_and_ingest_accept_explicit_delim_with_hit_after_auto_sample_cap() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut tags = vec!["solo"; 1_000];
    tags.push("a,b");
    let item_ids: Vec<i64> = (0..tags.len() as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "items",
            "data_source": "test",
            "data_version": "v1",
            "pkey": {"name": "item_id", "dtype": "int", "stype": "ID"},
            "fkeys": [],
            "cols": [{
                "name": "tags",
                "dtype": "string",
                "stype": "multicategorical",
                "multicategorical_delimiter": ","
            }],
            "time_col": null,
            "end_time_col": null,
            "edge_table": null
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let opts = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_delim_after_sample_cap".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };

    let cfg = convert_kumo_to_diskgraph(&kumo, &opts)?;
    let col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "tags")
        .expect("tags col");
    assert_eq!(
        col.ingested_dtype,
        diskgraph_ingest::graph_config::IngestedType::StringArray,
    );
    assert_eq!(col.multicategorical_delimiter.as_deref(), Some(","));

    let config_path = data_dir.join("diskgraph_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&cfg)?)?;
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
    Ok(())
}

/// Direct DiskGraph configs (i.e. configs hand-authored or emitted by
/// non-kumo tooling, fed straight to `ingest_with_stats`) must also see
/// the wrong-delimiter validation. Pre-#28478 the convert path was the
/// only entry; post-flip we run the validation universally at ingest entry
/// so a typo'd delimiter on a direct config does not silently bake into the
/// rowpack on disk.
#[test]
fn ingest_errors_on_direct_config_explicit_delim_mismatch() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Source uses ';'. User declares ',' below. Direct-config flow,
    // bypassing convert_kumo_to_diskgraph entirely.
    let item_ids: Vec<i64> = vec![0, 1, 2, 3];
    let tags = vec!["a;b;c", "d;e", "a;c", "x;y;z"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mc_direct_wrong_delim",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical",
                    "multicategorical_delimiter": ","
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    let res = ingest_with_stats(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            col_stats: true,
            cat_count_strategy: CategoryCountStrategy::Hybrid,
            ..IngestOptions::default()
        },
    );
    let err = res.expect_err("ingest should error on declared-vs-data mismatch");
    let s = err.to_string();
    assert!(
        s.contains("does not match the source data") && s.contains("tags"),
        "expected delim-mismatch error pointing at column `tags`, got: {}",
        s,
    );
    assert!(
        s.contains("Sample row that did not contain the delimiter: \"<omitted>\""),
        "expected mismatch diagnostics not to retain source data, got: {}",
        s,
    );
    Ok(())
}

/// User declares `multicategorical_delimiter=","` on a column whose source
/// data uses `;`. Today (pre-#28478) the mistake silently produces
/// one-token-per-row tokenization at every read site; post-flip it bakes
/// into the rowpack on disk because ingest pre-splits, so the wrong choice
/// is irreversible without re-ingest. `convert_kumo_to_diskgraph` must
/// surface this loudly with column + sample-row context.
#[test]
fn convert_errors_on_explicit_delim_mismatch_with_data() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;

    let item_ids: Vec<i64> = vec![0, 1, 2, 3];
    // Source data uses ';'. User will declare ',' below.
    let tags = vec!["a;b;c", "d;e", "a;c", "x;y;z"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(ArrowStringArray::from(tags)) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "items",
            "data_source": "test",
            "data_version": "v1",
            "pkey": {"name": "item_id", "dtype": "int", "stype": "ID"},
            "fkeys": [],
            "cols": [{
                "name": "tags",
                "dtype": "string",
                "stype": "multicategorical",
                "multicategorical_delimiter": ","
            }],
            "time_col": null,
            "end_time_col": null,
            "edge_table": null
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let opts = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_wrong_delim".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let err = convert_kumo_to_diskgraph(&kumo, &opts)
        .expect_err("convert should error on delim mismatch");
    match err {
        ConvertError::MulticategoricalDelimiterMismatch {
            table,
            column,
            declared,
            hits,
            total,
            sample,
            ..
        } => {
            assert_eq!(table, "items");
            assert_eq!(column, "tags");
            assert_eq!(declared, ",");
            assert_eq!(hits, 0, "comma absent from every sampled row");
            assert!(total >= 1, "scan should have sampled at least one row");
            assert!(
                sample.contains(';'),
                "sample row should be a real source row containing the actual delimiter; got {:?}",
                sample,
            );
        }
        other => panic!("unexpected error variant: {:?}", other),
    }
    Ok(())
}

// =============================================================================
// Layer 3 — diskgraph_config.json persistence at the index root
// =============================================================================

#[test]
fn ingest_persists_diskgraph_config_with_semantic_type() -> Result<()> {
    use arrow_array::Int64Array;

    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Minimal parquet: one int64 PK, one delimited multicat string column.
    let item_ids = Int64Array::from(vec![0_i64, 1, 2]);
    let tags = ArrowStringArray::from(vec!["a,b", "c", "a,c"]);
    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(item_ids) as ArrayRef, Arc::new(tags) as ArrayRef],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "persist_config_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical",
                    "multicategorical_delimiter": ","
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
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

    // 1. File is on disk at the expected path.
    let persisted = index_dir.join("diskgraph_config.json");
    assert!(
        persisted.is_file(),
        "diskgraph_config.json should be written to index root, but is missing at {}",
        persisted.display(),
    );

    // 2. Contents are parseable JSON with the per-column semantic_type wired
    //    through. We don't pin the exact GraphConfig shape — just the field
    //    that downstream consumers (TT-gen resolver) care about.
    let bytes = fs::read(&persisted)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let columns = v["tables"][0]["columns"]
        .as_array()
        .expect("tables[0].columns is an array");
    let tags_col = columns
        .iter()
        .find(|c| c["name"].as_str() == Some("tags"))
        .expect("tags column entry present");
    assert_eq!(
        tags_col["semantic_type"].as_str(),
        Some("multicategorical"),
        "persisted tags column must carry semantic_type=multicategorical; got {tags_col}",
    );

    // 3. Roundtrip: GraphIndexReader::open must load the persisted config
    //    and expose the semantic_type via column_semantic_type(). This is
    //    the read path TT-gen's resolver relies on.
    use diskgraph_index::graph::GraphIndexReader;
    let metadata_path = index_dir.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &index_dir)?;
    assert_eq!(
        reader.column_semantic_type("items", "tags"),
        Some("multicategorical"),
        "GraphIndexReader.column_semantic_type must round-trip the persisted stype",
    );
    assert_eq!(
        reader.column_semantic_type("items", "nonexistent"),
        None,
        "missing column must return None, not a default stype",
    );

    Ok(())
}

#[test]
fn ingest_rejects_multicategorical_string_without_delimiter() -> Result<()> {
    use arrow_array::Int64Array;

    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Single-token-per-row content. The user declares stype=multicategorical
    // on a scalar string column without a delimiter — degenerate shape that
    // is equivalent to stype=categorical. graph_config::validate must reject.
    let item_ids = Int64Array::from(vec![0_i64, 1, 2]);
    let tags = ArrowStringArray::from(vec!["kids", "audio", "kids"]);
    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("tags", DataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(item_ids) as ArrayRef, Arc::new(tags) as ArrayRef],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "multicat_no_delim_reject",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical"
                    // NOTE: no multicategorical_delimiter — this is the
                    // degenerate combo that must be rejected.
                }
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    let result = ingest_with_stats(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            col_stats: true,
            cat_count_strategy: CategoryCountStrategy::Hybrid,
            ..IngestOptions::default()
        },
    );

    let err =
        result.expect_err("ingest must reject (multicat, string, no-delim) as a degenerate shape");
    let msg = err.to_string();
    assert!(
        msg.contains("stype=multicategorical")
            && msg.contains("list-shaped")
            && msg.contains("categorical"),
        "error message must explain the contract and suggest categorical; got: {msg}",
    );
    // No on-disk artifacts should have been written when validation fails.
    assert!(
        !index_dir.join("diskgraph_config.json").exists(),
        "rejected config must not produce a persisted diskgraph_config.json",
    );

    Ok(())
}
