use anyhow::Result;
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_ingest::{ingest, Dest, IngestOptions, MappingKind};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

use arrow_array::{
    ArrayRef, BinaryArray, Float32Array, Int64Array, RecordBatch, StringArray,
    TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema};

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    writeln!(file, "{}", headers.join(","))?;
    for row in rows {
        writeln!(file, "{}", row.join(","))?;
    }
    Ok(())
}

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder().set_max_row_group_size(1024).build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn count_parquet_rows(path: &Path) -> Result<usize> {
    let file = fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(1024)
        .build()?;
    let mut rows = 0usize;
    for batch in reader {
        rows += batch?.num_rows();
    }
    Ok(rows)
}

fn read_first_node_value(path: &Path, dict: Option<&[u8]>) -> Result<serde_json::Value> {
    let file = fs::File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(1)
        .build()?;
    let batch = reader.next().unwrap()?;
    let idx = batch.schema().index_of("data")?;
    let array = batch.column(idx);
    let bin = array
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| anyhow::anyhow!("data column not binary"))?;
    let raw = bin.value(0);
    let decoded = if let Some(dict) = dict {
        let mut decoder = zstd::stream::Decoder::with_dictionary(std::io::Cursor::new(raw), dict)?;
        let mut out = Vec::new();
        decoder.read_to_end(&mut out)?;
        out
    } else {
        raw.to_vec()
    };
    Ok(rmp_serde::from_slice(&decoded)?)
}

#[test]
fn ingest_csv_with_edges_and_dict() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("users.csv"),
        &["id", "name", "weight"],
        &[
            vec!["0".into(), "user0001".into(), "10.0".into()],
            vec!["1".into(), "user0002".into(), "20.0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("orders.csv"),
        &["id", "user_id"],
        &[
            vec!["0".into(), "1".into()],
            vec!["1".into(), "0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("products.csv"),
        &["id", "weight"],
        &[
            vec!["0".into(), "5.0".into()],
            vec!["1".into(), "7.0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("order_items.csv"),
        &["order_id", "product_id", "weight"],
        &[
            vec!["0".into(), "1".into(), "3".into()],
            vec!["1".into(), "0".into(), "2".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "csv_test",
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [input_root.join("users.csv").to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string"},
                    {"name": "weight", "dtype": "float32"}
                ]
            },
            {
                "name": "Order",
                "format": "csv",
                "paths": [input_root.join("orders.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "user_id", "dtype": "int64"}
                ]
            },
            {
                "name": "Product",
                "format": "csv",
                "paths": [input_root.join("products.csv").to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "weight", "dtype": "float32"}
                ]
            },
            {
                "name": "OrderItem",
                "format": "csv",
                "paths": [input_root.join("order_items.csv").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "order_id", "ref_table": "Order", "ref_column": "id"},
                    "dst_fk": {"column": "product_id", "ref_table": "Product", "ref_column": "id"}
                },
                "weight": "weight",
                "columns": [
                    {"name": "order_id", "dtype": "int64"},
                    {"name": "product_id", "dtype": "int64"},
                    {"name": "weight", "dtype": "int32"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "order_user",
                "src_table": "Order",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, IngestOptions::default())?;
    let index_path = output_root.join("index_metadata.json");
    let disk_metadata: DatasetMetadata = serde_json::from_slice(&fs::read(index_path)?)?;
    assert_eq!(metadata.node_types.len(), 3);
    assert_eq!(disk_metadata.node_types.len(), 3);

    let user = disk_metadata
        .node_types
        .iter()
        .find(|n| n.name == "User")
        .unwrap();
    let dict_path = output_root.join("nodes/user/node_data.dict");
    if user.node_data_compression.is_some() {
        assert!(dict_path.exists());
    }
    let dict = dict_path.exists().then(|| fs::read(&dict_path)).transpose()?;
    let node_data = output_root.join("nodes/user/part-00000.parquet");
    let value = read_first_node_value(&node_data, dict.as_deref())?;
    assert_eq!(value["name"], "user0001");

    let order_user = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "order_user")
        .unwrap();
    let order_user_rev = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "order_user_reverse")
        .unwrap();
    assert_eq!(order_user.dest_sort_key_bits, Some(32));
    assert_eq!(order_user_rev.dest_sort_key_bits, None);

    let edge_src = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_OrderItem_src_to_dst")
        .unwrap();
    let edge_dst = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_OrderItem_dst_to_src")
        .unwrap();
    assert_eq!(edge_src.dest_sort_key_bits, Some(32));
    assert_eq!(edge_dst.dest_sort_key_bits, Some(32));

    let order_user_rows = count_parquet_rows(
        &output_root.join("edges/order_user/part-00000.parquet"),
    )?;
    let order_user_rev_rows = count_parquet_rows(
        &output_root.join("edges/order_user_reverse/part-00000.parquet"),
    )?;
    assert_eq!(order_user_rows, 2);
    assert_eq!(order_user_rev_rows, 2);

    let edge_src_rows = count_parquet_rows(
        &output_root.join("edges/edge_orderitem_src_to_dst/part-00000.parquet"),
    )?;
    let edge_dst_rows = count_parquet_rows(
        &output_root.join("edges/edge_orderitem_dst_to_src/part-00000.parquet"),
    )?;
    assert_eq!(edge_src_rows, 2);
    assert_eq!(edge_dst_rows, 2);

    Ok(())
}

#[test]
fn ingest_parquet_with_sorted_mapping() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    let table_path = input_root.join("sku.parquet");

    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
    ]);
    let ids = Arc::new(StringArray::from(vec!["SKU00001", "SKU00002"])) as ArrayRef;
    let weights = Arc::new(Float32Array::from(vec![1.0_f32, 2.0_f32])) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids, weights])?;

    let source = serde_json::json!({
        "dataset_name": "parquet_test",
        "tables": [
            {
                "name": "Sku",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "dtype": "string", "fixed_len": 8},
                    {"name": "weight", "dtype": "float32"}
                ]
            }
        ],
        "relationships": []
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        mapping_kind: MappingKind::Sorted,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    assert_eq!(metadata.node_types.len(), 1);
    assert!(metadata.edge_relations.is_empty());

    let sku = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Sku")
        .unwrap();
    let pk = sku.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "sorted");
    assert_eq!(pk.fixed_len, Some(8));
    let map_path = output_root.join("nodes/sku/pk_map.bin");
    assert!(map_path.exists());
    assert!(fs::metadata(map_path)?.len() > 0);

    Ok(())
}

#[test]
fn ingest_csv_with_bytes_pk_mapping() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("assets.parquet");
    let schema = Schema::new(vec![
        Field::new("id", DataType::Binary, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let ids = Arc::new(BinaryArray::from(vec![b"ABCD".as_ref(), b"EFGH".as_ref()])) as ArrayRef;
    let names = Arc::new(StringArray::from(vec!["asset-a", "asset-b"])) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids, names])?;

    let source = serde_json::json!({
        "dataset_name": "bytes_pk_test",
        "tables": [
            {
                "name": "Asset",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "dtype": "bytes", "fixed_len": 4},
                    {"name": "name", "dtype": "string"}
                ]
            }
        ],
        "relationships": []
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        mapping_kind: MappingKind::Hash,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    let asset = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Asset")
        .unwrap();
    let pk = asset.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "hash");
    assert_eq!(pk.fixed_len, Some(4));
    let map_path = output_root.join("nodes/asset/pk_map.bin");
    assert!(map_path.exists());
    assert!(fs::metadata(map_path)?.len() > 0);

    Ok(())
}

#[test]
fn ingest_relationship_uses_dest_timestamp_sort_key() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let event_path = input_root.join("event.parquet");
    let event_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("session_id", DataType::Int64, false),
    ]);
    let event_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    let session_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    write_parquet(&event_path, event_schema, vec![event_ids, session_ids])?;

    let session_path = input_root.join("session.parquet");
    let session_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            false,
        ),
    ]);
    let session_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    let session_ts = Arc::new(TimestampSecondArray::from(vec![1_700_000_000, 1_700_001_000]))
        as ArrayRef;
    write_parquet(&session_path, session_schema, vec![session_ids, session_ts])?;

    let source = serde_json::json!({
        "dataset_name": "timestamp_sort_test",
        "tables": [
            {
                "name": "Event",
                "format": "parquet",
                "paths": [event_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "session_id", "dtype": "int64"}
                ]
            },
            {
                "name": "Session",
                "format": "parquet",
                "paths": [session_path.to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "created_at", "dtype": "timestamp"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "event_session",
                "src_table": "Event",
                "src_column": "session_id",
                "dst_table": "Session",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, IngestOptions::default())?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "event_session")
        .unwrap();
    let reverse = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "event_session_reverse")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(32));
    assert_eq!(reverse.dest_sort_key_bits, None);
    let forward_rows = count_parquet_rows(
        &output_root.join("edges/event_session/part-00000.parquet"),
    )?;
    let reverse_rows = count_parquet_rows(
        &output_root.join("edges/event_session_reverse/part-00000.parquet"),
    )?;
    assert_eq!(forward_rows, 2);
    assert_eq!(reverse_rows, 2);

    Ok(())
}
