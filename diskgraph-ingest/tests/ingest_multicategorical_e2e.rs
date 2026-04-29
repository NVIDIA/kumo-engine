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
//!    case, and that an ambiguous fixture raises the typed
//!    `ConvertError::MulticategoricalDelimiterAmbiguous`.

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
    // Expected: kids:2 (rows 0 and 2), outdoor:1, sale:1, audio:1, pro:1, "":1 (empty row 3).
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
    assert_eq!(map.get("").copied(), Some(1));

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
fn convert_multicat_string_ambiguous_errors() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    // 50% rows contain comma, 50% contain pipe. Both ≥ 20%, neither ≥ 50%
    // alone — should trip the "ambiguous" rule.
    let mut rows: Vec<&str> = Vec::new();
    rows.extend(std::iter::repeat_n("a,b,c", 50));
    rows.extend(std::iter::repeat_n("x|y|z", 50));
    write_tags_parquet(&data_dir.join("items.parquet"), &rows);

    let kumo_json = build_kumo_config(&data_dir, None);
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "mc_detect_ambiguous".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("items".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
    };
    let err = convert_kumo_to_diskgraph(&kumo, &options).unwrap_err();
    match err {
        ConvertError::MulticategoricalDelimiterAmbiguous {
            column, candidates, ..
        } => {
            assert_eq!(column, "tags");
            let has_comma = candidates.iter().any(|(c, _)| c == ",");
            let has_pipe = candidates.iter().any(|(c, _)| c == "|");
            assert!(
                has_comma && has_pipe,
                "expected both candidates in error payload; got {:?}",
                candidates
            );
        }
        other => panic!(
            "expected MulticategoricalDelimiterAmbiguous, got {:?}",
            other
        ),
    }
    Ok(())
}

#[test]
fn convert_multicat_string_explicit_delimiter_respected() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;
    // Ambiguous content — if detection were run, it'd error. But the user
    // set an explicit delimiter, so detection should be skipped and the
    // declared value passed through.
    let mut rows: Vec<&str> = Vec::new();
    rows.extend(std::iter::repeat_n("a,b,c", 50));
    rows.extend(std::iter::repeat_n("x|y|z", 50));
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
