//! End-to-end round-trip tests for intlist and stringlist feature columns.
//!
//! Pipeline: write Parquet with List<Int64> / List<Utf8> columns
//!           → ingest into DiskGraph index
//!           → open index with GraphIndexReader
//!           → decode node data JSON
//!           → assert values match source data exactly.
//!
//! This is the critical gap not covered by rowpack unit tests: those test
//! encode/decode in memory. These tests exercise the full
//! Parquet → ingest pipeline → on-disk index → RowPack decode path.

use anyhow::Result;
use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

// ── helpers ──────────────────────────────────────────────────────────────────

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

/// Open the index at `index_dir`, look up `node_id` in `table_name`, and
/// return the decoded feature JSON for that node.
fn decode_node_json(index_dir: &Path, table_name: &str, node_id: u64) -> Result<serde_json::Value> {
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), index_dir)?;
    let table = reader
        .table(table_name)
        .ok_or_else(|| anyhow::anyhow!("table {table_name} not found"))?;
    let view = table.edges_for_node(node_id)?;
    let json = table
        .decode_node_data_json(&view)?
        .ok_or_else(|| anyhow::anyhow!("no node data for node {node_id}"))?;
    Ok(json)
}

// ── test: Int64Array (intlist) ────────────────────────────────────────────────

/// Build a minimal graph with one table that has an intlist feature column.
///
/// Table schema:
///   item_id   Int64 PK
///   tag_ids   List<Int64>    ← intlist feature
///
/// Data:
///   item 0 → tag_ids = [10, 20, 30]
///   item 1 → tag_ids = [40]
///   item 2 → tag_ids = []            ← empty list
///   item 3 → tag_ids = [1, 2, 3, 4, 5, 6, 7]  ← longer list
fn build_intlist_graph(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    // ── build Parquet ──
    let mut item_id_builder = Int64Builder::new();
    let mut tag_ids_builder = ListBuilder::new(Int64Builder::new());

    // item 0: [10, 20, 30]
    item_id_builder.append_value(0);
    tag_ids_builder.values().append_value(10);
    tag_ids_builder.values().append_value(20);
    tag_ids_builder.values().append_value(30);
    tag_ids_builder.append(true);

    // item 1: [40]
    item_id_builder.append_value(1);
    tag_ids_builder.values().append_value(40);
    tag_ids_builder.append(true);

    // item 2: []
    item_id_builder.append_value(2);
    tag_ids_builder.append(true);

    // item 3: [1,2,3,4,5,6,7]
    item_id_builder.append_value(3);
    for v in 1i64..=7 {
        tag_ids_builder.values().append_value(v);
    }
    tag_ids_builder.append(true);

    let item_field = Field::new("item_id", DataType::Int64, false);
    let list_field = Field::new(
        "tag_ids",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        true,
    );
    let schema = Schema::new(vec![item_field, list_field]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(item_id_builder.finish()) as ArrayRef,
            Arc::new(tag_ids_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("items.parquet"), &batch);

    // ── write config JSON ──
    let config = serde_json::json!({
        "dataset_name": "intlist_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id",  "ingested_dtype": "int64"},
                {"name": "tag_ids",  "ingested_dtype": "int64_array"}
            ],
            "primary_key": "item_id",
            "source": {
                "type": "parquet",
                "paths": [data_dir.join("items.parquet").to_str().unwrap()]
            }
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

    // ── ingest ──
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )
    .unwrap();

    (data_dir, index_dir)
}

#[test]
fn integ_intlist_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let (_, index_dir) = build_intlist_graph(dir.path());

    // item 0 → [10, 20, 30]
    let json = decode_node_json(&index_dir, "items", 0)?;
    let list = json["tag_ids"].as_array().expect("tag_ids should be array");
    assert_eq!(list.len(), 3, "item 0: expected 3 elements");
    assert_eq!(list[0].as_i64().unwrap(), 10);
    assert_eq!(list[1].as_i64().unwrap(), 20);
    assert_eq!(list[2].as_i64().unwrap(), 30);

    // item 1 → [40]
    let json = decode_node_json(&index_dir, "items", 1)?;
    let list = json["tag_ids"].as_array().expect("tag_ids should be array");
    assert_eq!(list.len(), 1, "item 1: expected 1 element");
    assert_eq!(list[0].as_i64().unwrap(), 40);

    // item 2 → []
    let json = decode_node_json(&index_dir, "items", 2)?;
    let list = json["tag_ids"].as_array().expect("tag_ids should be array");
    assert!(list.is_empty(), "item 2: expected empty list");

    // item 3 → [1,2,3,4,5,6,7]
    let json = decode_node_json(&index_dir, "items", 3)?;
    let list = json["tag_ids"].as_array().expect("tag_ids should be array");
    assert_eq!(list.len(), 7, "item 3: expected 7 elements");
    for (i, v) in list.iter().enumerate() {
        assert_eq!(v.as_i64().unwrap(), (i + 1) as i64);
    }

    Ok(())
}

#[test]
fn integ_intlist_negative_values() -> Result<()> {
    // Separate small fixture: negative and large integers round-trip correctly.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(Int64Builder::new());

    // item 0: [-1, -128, i64::MIN/2, 0, i64::MAX/2]
    let vals: Vec<i64> = vec![-1, -128, i64::MIN / 2, 0, i64::MAX / 2];
    id_b.append_value(0);
    for &v in &vals {
        list_b.values().append_value(v);
    }
    list_b.append(true);

    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "vals",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "neg_int_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {"name": "vals",    "ingested_dtype": "int64_array"}
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    let json = decode_node_json(&index_dir, "items", 0)?;
    let list = json["vals"].as_array().expect("vals should be array");
    assert_eq!(list.len(), vals.len());
    for (i, &expected) in vals.iter().enumerate() {
        assert_eq!(list[i].as_i64().unwrap(), expected, "element {i}");
    }

    Ok(())
}

#[test]
fn integ_intlist_variable_length_no_drift() -> Result<()> {
    // Multiple rows with different list lengths. If the byte pointer drifts
    // between rows, subsequent nodes will decode garbage.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // rows: lengths 1, 3, 0, 7, 2
    let rows: Vec<Vec<i64>> = vec![
        vec![100],
        vec![1, 2, 3],
        vec![],
        vec![10, 20, 30, 40, 50, 60, 70],
        vec![999, 888],
    ];

    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(Int64Builder::new());
    for (i, row) in rows.iter().enumerate() {
        id_b.append_value(i as i64);
        for &v in row {
            list_b.values().append_value(v);
        }
        list_b.append(true);
    }

    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "tag_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "drift_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {"name": "tag_ids", "ingested_dtype": "int64_array"}
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet", "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    for (node_id, expected) in rows.iter().enumerate() {
        let json = decode_node_json(&index_dir, "items", node_id as u64)?;
        let list = json["tag_ids"].as_array().expect("tag_ids should be array");
        assert_eq!(
            list.len(),
            expected.len(),
            "node {node_id}: length mismatch"
        );
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(list[i].as_i64().unwrap(), v, "node {node_id} element {i}");
        }
    }

    Ok(())
}

// ── test: StringArray (stringlist) ────────────────────────────────────────────

/// Build a minimal graph with a stringlist feature column.
///
/// Table schema:
///   product_id  Int64 PK
///   tags        List<Utf8>   ← stringlist feature
///
/// Data:
///   product 0 → ["sports", "sale", "outdoor"]
///   product 1 → ["kids"]
///   product 2 → []
///   product 3 → ["pro", "audio", "electronics"]
fn build_stringlist_graph(root: &Path) -> std::path::PathBuf {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    let mut id_b = Int64Builder::new();
    let mut tags_b = ListBuilder::new(StringBuilder::new());

    let rows: &[&[&str]] = &[
        &["sports", "sale", "outdoor"],
        &["kids"],
        &[],
        &["pro", "audio", "electronics"],
    ];
    for (i, row) in rows.iter().enumerate() {
        id_b.append_value(i as i64);
        for &t in *row {
            tags_b.values().append_value(t);
        }
        tags_b.append(true);
    }

    let schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(tags_b.finish()) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&data_dir.join("products.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "stringlist_test",
        "num_shards": 1,
        "tables": [{
            "name": "products",
            "columns": [
                {"name": "product_id", "ingested_dtype": "int64"},
                {"name": "tags",       "ingested_dtype": "string_array"}
            ],
            "primary_key": "product_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("products.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )
    .unwrap();

    index_dir
}

#[test]
fn integ_stringlist_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let index_dir = build_stringlist_graph(dir.path());

    let expected: &[&[&str]] = &[
        &["sports", "sale", "outdoor"],
        &["kids"],
        &[],
        &["pro", "audio", "electronics"],
    ];

    for (node_id, tokens) in expected.iter().enumerate() {
        let json = decode_node_json(&index_dir, "products", node_id as u64)?;
        let list = json["tags"]
            .as_array()
            .unwrap_or_else(|| panic!("product {node_id}: tags should be array, got {json}"));
        assert_eq!(
            list.len(),
            tokens.len(),
            "product {node_id}: length mismatch (expected {:?})",
            tokens
        );
        for (i, &tok) in tokens.iter().enumerate() {
            assert_eq!(
                list[i].as_str().unwrap(),
                tok,
                "product {node_id} token {i}"
            );
        }
    }

    Ok(())
}

#[test]
fn integ_stringlist_oov_tokens_survive() -> Result<()> {
    // Inject a token that won't be in the context-phase sample (by using a
    // very small context sample size and a unique token in the last row).
    // The tag-2 raw fallback must encode and decode it correctly.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // 20 rows of common tokens to drive dict selection, then one row with a
    // unique token that is very unlikely to appear in a 10-sample context pass.
    let mut id_b = Int64Builder::new();
    let mut tags_b = ListBuilder::new(StringBuilder::new());

    for i in 0i64..20 {
        id_b.append_value(i);
        tags_b.values().append_value("common_a");
        tags_b.values().append_value("common_b");
        tags_b.append(true);
    }
    // Unique token: highly unlikely to be sampled in context phase.
    id_b.append_value(20);
    tags_b
        .values()
        .append_value("token_that_is_unique_xyzzy_42");
    tags_b.append(true);

    let schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(tags_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("products.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "oov_test",
        "num_shards": 1,
        "tables": [{
            "name": "products",
            "columns": [
                {"name": "product_id", "ingested_dtype": "int64"},
                {"name": "tags",       "ingested_dtype": "string_array"}
            ],
            "primary_key": "product_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("products.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    // All 20 common-token rows must decode correctly.
    for i in 0u64..20 {
        let json = decode_node_json(&index_dir, "products", i)?;
        let list = json["tags"].as_array().expect("tags array");
        assert_eq!(list.len(), 2, "row {i}: expected 2 tokens");
        assert_eq!(list[0].as_str().unwrap(), "common_a");
        assert_eq!(list[1].as_str().unwrap(), "common_b");
    }

    // The OOV row must survive via tag-2 raw encoding.
    let json = decode_node_json(&index_dir, "products", 20)?;
    let list = json["tags"].as_array().expect("tags array for oov row");
    assert_eq!(list.len(), 1, "oov row: expected 1 token");
    assert_eq!(list[0].as_str().unwrap(), "token_that_is_unique_xyzzy_42");

    Ok(())
}

#[test]
fn integ_stringlist_variable_length_no_drift() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let rows: Vec<Vec<&str>> = vec![
        vec!["a"],
        vec!["x", "y", "z"],
        vec![],
        vec!["p", "q", "r", "s", "t", "u", "v"],
        vec!["m", "n"],
    ];

    let mut id_b = Int64Builder::new();
    let mut tags_b = ListBuilder::new(StringBuilder::new());
    for (i, row) in rows.iter().enumerate() {
        id_b.append_value(i as i64);
        for &t in row {
            tags_b.values().append_value(t);
        }
        tags_b.append(true);
    }

    let schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(tags_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("products.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "string_drift_test",
        "num_shards": 1,
        "tables": [{
            "name": "products",
            "columns": [
                {"name": "product_id", "ingested_dtype": "int64"},
                {"name": "tags",       "ingested_dtype": "string_array"}
            ],
            "primary_key": "product_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("products.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    for (node_id, expected) in rows.iter().enumerate() {
        let json = decode_node_json(&index_dir, "products", node_id as u64)?;
        let list = json["tags"].as_array().expect("tags array");
        assert_eq!(list.len(), expected.len(), "node {node_id}: length");
        for (i, &t) in expected.iter().enumerate() {
            assert_eq!(list[i].as_str().unwrap(), t, "node {node_id} token {i}");
        }
    }

    Ok(())
}

// ── test: coexistence with other feature columns ──────────────────────────────

#[test]
fn integ_list_features_alongside_scalars() -> Result<()> {
    // A table with both scalar and list columns: verifies that the RowPack
    // byte layout is correct when list columns are interspersed with scalars.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut score_b = arrow_array::builder::Float64Builder::new();
    let mut tag_ids_b = ListBuilder::new(Int64Builder::new());
    let mut tags_b = ListBuilder::new(StringBuilder::new());

    // row 0: score=1.5, tag_ids=[1,2], tags=["a","b"]
    id_b.append_value(0);
    score_b.append_value(1.5);
    tag_ids_b.values().append_value(1);
    tag_ids_b.values().append_value(2);
    tag_ids_b.append(true);
    tags_b.values().append_value("a");
    tags_b.values().append_value("b");
    tags_b.append(true);

    // row 1: score=2.5, tag_ids=[10], tags=["x"]
    id_b.append_value(1);
    score_b.append_value(2.5);
    tag_ids_b.values().append_value(10);
    tag_ids_b.append(true);
    tags_b.values().append_value("x");
    tags_b.append(true);

    // row 2: score=3.5, tag_ids=[], tags=[]
    id_b.append_value(2);
    score_b.append_value(3.5);
    tag_ids_b.append(true);
    tags_b.append(true);

    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
        Field::new(
            "tag_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(score_b.finish()) as ArrayRef,
            Arc::new(tag_ids_b.finish()) as ArrayRef,
            Arc::new(tags_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "mixed_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id",  "ingested_dtype": "int64"},
                {"name": "score",    "ingested_dtype": "float64"},
                {"name": "tag_ids",  "ingested_dtype": "int64_array"},
                {"name": "tags",     "ingested_dtype": "string_array"}
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    // row 0
    let j = decode_node_json(&index_dir, "items", 0)?;
    assert!((j["score"].as_f64().unwrap() - 1.5).abs() < 1e-9);
    let tids = j["tag_ids"].as_array().unwrap();
    assert_eq!(tids.len(), 2);
    assert_eq!(tids[0].as_i64().unwrap(), 1);
    assert_eq!(tids[1].as_i64().unwrap(), 2);
    let ts = j["tags"].as_array().unwrap();
    assert_eq!(ts.len(), 2);
    assert_eq!(ts[0].as_str().unwrap(), "a");
    assert_eq!(ts[1].as_str().unwrap(), "b");

    // row 1
    let j = decode_node_json(&index_dir, "items", 1)?;
    assert!((j["score"].as_f64().unwrap() - 2.5).abs() < 1e-9);
    assert_eq!(j["tag_ids"].as_array().unwrap().len(), 1);
    assert_eq!(j["tag_ids"][0].as_i64().unwrap(), 10);
    assert_eq!(j["tags"][0].as_str().unwrap(), "x");

    // row 2: both lists empty, score intact
    let j = decode_node_json(&index_dir, "items", 2)?;
    assert!((j["score"].as_f64().unwrap() - 3.5).abs() < 1e-9);
    assert!(j["tag_ids"].as_array().unwrap().is_empty());
    assert!(j["tags"].as_array().unwrap().is_empty());

    Ok(())
}

// =============================================================================
// String-source fallback: Parquet Utf8 column → declared as intlist/stringlist
// =============================================================================
//
// When a Parquet column is physically Utf8 but declared ingested_dtype=int64_array
// or string_array, DiskGraph should parse the JSON-encoded string at ingest time,
// matching the existing behavior for float32_array.

#[test]
fn integ_intlist_from_string_column_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    // Physical column is Utf8, values are JSON-encoded int lists
    use arrow_array::StringArray as ArrowStringArray;
    let ids = Arc::new(arrow_array::Int64Array::from(vec![0i64, 1, 2])) as ArrayRef;
    let lists = Arc::new(ArrowStringArray::from(vec![
        Some("[10, 20, 30]"),
        Some("[40]"),
        Some("[]"),
    ])) as ArrayRef;

    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("item_id", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("tag_ids", arrow_schema::DataType::Utf8, true),
    ]);
    let batch = arrow_array::RecordBatch::try_new(Arc::new(schema), vec![ids, lists])?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "intlist_from_string_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {"name": "tag_ids", "ingested_dtype": "int64_array"}
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    // item 0 → [10, 20, 30]
    let j = decode_node_json(&index_dir, "items", 0)?;
    let list = j["tag_ids"].as_array().expect("tag_ids should be array");
    assert_eq!(list.len(), 3);
    assert_eq!(list[0].as_i64().unwrap(), 10);
    assert_eq!(list[1].as_i64().unwrap(), 20);
    assert_eq!(list[2].as_i64().unwrap(), 30);

    // item 1 → [40]
    let j = decode_node_json(&index_dir, "items", 1)?;
    assert_eq!(j["tag_ids"].as_array().unwrap().len(), 1);
    assert_eq!(j["tag_ids"][0].as_i64().unwrap(), 40);

    // item 2 → []
    let j = decode_node_json(&index_dir, "items", 2)?;
    assert!(j["tag_ids"].as_array().unwrap().is_empty());

    Ok(())
}

#[test]
fn integ_stringlist_from_string_column_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    use arrow_array::StringArray as ArrowStringArray;
    let ids = Arc::new(arrow_array::Int64Array::from(vec![0i64, 1, 2])) as ArrayRef;
    let lists = Arc::new(ArrowStringArray::from(vec![
        Some(r#"["sports", "sale"]"#),
        Some(r#"["kids"]"#),
        Some("[]"),
    ])) as ArrayRef;

    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("product_id", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("tags", arrow_schema::DataType::Utf8, true),
    ]);
    let batch = arrow_array::RecordBatch::try_new(Arc::new(schema), vec![ids, lists])?;
    write_parquet(&data_dir.join("products.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "stringlist_from_string_test",
        "num_shards": 1,
        "tables": [{
            "name": "products",
            "columns": [
                {"name": "product_id", "ingested_dtype": "int64"},
                {"name": "tags",       "ingested_dtype": "string_array"}
            ],
            "primary_key": "product_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("products.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    let j = decode_node_json(&index_dir, "products", 0)?;
    let list = j["tags"].as_array().expect("tags should be array");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].as_str().unwrap(), "sports");
    assert_eq!(list[1].as_str().unwrap(), "sale");

    let j = decode_node_json(&index_dir, "products", 1)?;
    assert_eq!(j["tags"][0].as_str().unwrap(), "kids");

    let j = decode_node_json(&index_dir, "products", 2)?;
    assert!(j["tags"].as_array().unwrap().is_empty());

    Ok(())
}

// ── tests: inner-null drop-null contract ────────────────────────────────────

#[test]
fn integ_intlist_inner_nulls_dropped() -> Result<()> {
    // Full ingest-to-decode: null elements inside an intlist column must be
    // dropped at encode time. Decoded lists are shorter than the source
    // Arrow lists; remaining values preserve their source order.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut list_b = ListBuilder::new(Int64Builder::new());

    // node 0: [1, null, 3] → expect decoded [1, 3]
    id_b.append_value(0);
    list_b.values().append_value(1);
    list_b.values().append_null();
    list_b.values().append_value(3);
    list_b.append(true);

    // node 1: [null] → expect decoded []
    id_b.append_value(1);
    list_b.values().append_null();
    list_b.append(true);

    // node 2: [10, 20] dense → expect decoded [10, 20]
    id_b.append_value(2);
    list_b.values().append_value(10);
    list_b.values().append_value(20);
    list_b.append(true);

    // node 3: [null, 0, null, 99, null] → expect decoded [0, 99]
    // (legitimate zero must survive; only the nulls are dropped).
    id_b.append_value(3);
    list_b.values().append_null();
    list_b.values().append_value(0);
    list_b.values().append_null();
    list_b.values().append_value(99);
    list_b.values().append_null();
    list_b.append(true);

    let schema = Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "tag_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(list_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("items.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "intlist_inner_null_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {"name": "tag_ids", "ingested_dtype": "int64_array"}
            ],
            "primary_key": "item_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("items.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    let expected: &[&[i64]] = &[
        &[1, 3],
        &[],
        &[10, 20],
        &[0, 99],
    ];
    for (node_id, want) in expected.iter().enumerate() {
        let json = decode_node_json(&index_dir, "items", node_id as u64)?;
        let list = json["tag_ids"].as_array().expect("tag_ids should be array");
        assert_eq!(
            list.len(),
            want.len(),
            "node {node_id}: length mismatch, decoded={:?}, expected={:?}",
            list,
            want
        );
        for (i, &v) in want.iter().enumerate() {
            assert_eq!(
                list[i].as_i64().unwrap(),
                v,
                "node {node_id} element {i}"
            );
        }
    }
    Ok(())
}

#[test]
fn integ_stringlist_inner_nulls_dropped() -> Result<()> {
    // Full ingest-to-decode: null elements are dropped; legitimate empty
    // strings are preserved.
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&data_dir)?;

    let mut id_b = Int64Builder::new();
    let mut tags_b = ListBuilder::new(StringBuilder::new());

    // product 0: ["a", null, "b"] → ["a", "b"]
    id_b.append_value(0);
    tags_b.values().append_value("a");
    tags_b.values().append_null();
    tags_b.values().append_value("b");
    tags_b.append(true);

    // product 1: [null, null] → []
    id_b.append_value(1);
    tags_b.values().append_null();
    tags_b.values().append_null();
    tags_b.append(true);

    // product 2: ["x", "", "y"] → ["x", "", "y"] (empty string preserved)
    id_b.append_value(2);
    tags_b.values().append_value("x");
    tags_b.values().append_value("");
    tags_b.values().append_value("y");
    tags_b.append(true);

    // product 3: ["p", "", null, "q"] → ["p", "", "q"]
    id_b.append_value(3);
    tags_b.values().append_value("p");
    tags_b.values().append_value("");
    tags_b.values().append_null();
    tags_b.values().append_value("q");
    tags_b.append(true);

    let schema = Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(tags_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet(&data_dir.join("products.parquet"), &batch);

    let config = serde_json::json!({
        "dataset_name": "stringlist_inner_null_test",
        "num_shards": 1,
        "tables": [{
            "name": "products",
            "columns": [
                {"name": "product_id", "ingested_dtype": "int64"},
                {"name": "tags",       "ingested_dtype": "string_array"}
            ],
            "primary_key": "product_id",
            "source": {"type": "parquet",
                       "paths": [data_dir.join("products.parquet").to_str().unwrap()]}
        }]
    });
    let config_path = data_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )?;

    let expected: &[&[&str]] = &[
        &["a", "b"],
        &[],
        &["x", "", "y"],
        &["p", "", "q"],
    ];
    for (node_id, want) in expected.iter().enumerate() {
        let json = decode_node_json(&index_dir, "products", node_id as u64)?;
        let list = json["tags"].as_array().expect("tags should be array");
        assert_eq!(
            list.len(),
            want.len(),
            "product {node_id}: length mismatch, decoded={:?}, expected={:?}",
            list,
            want
        );
        for (i, &tok) in want.iter().enumerate() {
            assert_eq!(
                list[i].as_str().unwrap(),
                tok,
                "product {node_id} token {i}"
            );
        }
    }
    Ok(())
}
