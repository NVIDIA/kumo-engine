use anyhow::Result;
use arrow_array::{
    ArrayRef, BinaryArray, Int64Array, RecordBatch, StringArray, TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::pkmap::Key;
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::{ingest, IngestOptions};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    let file = fs::File::create(path)?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[test]
fn test_int64_pk_extraction() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&input_root)?;

    // 1. Table with Timestamp (Mixed Validity)
    let path_ts = input_root.join("int64_ts.parquet");
    let schema_ts = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), true),
    ]);
    let ids_ts = Arc::new(Int64Array::from(vec![1, 2, 5])) as ArrayRef;
    // Row 2 (id=2) has NULL timestamp -> Should be dropped
    let tss_ts = Arc::new(TimestampSecondArray::from(vec![Some(100), None, Some(200)])) as ArrayRef;
    write_parquet(&path_ts, schema_ts, vec![ids_ts, tss_ts])?;

    // 2. Table without Timestamp (Implicit Validity)
    let path_nots = input_root.join("int64_nots.parquet");
    let schema_nots = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let ids_nots = Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef;
    write_parquet(&path_nots, schema_nots, vec![ids_nots])?;

    let source = serde_json::json!({
        "dataset_name": "int64_extraction_test",
        "tables": [
            {
                "name": "WithTS",
                "format": "parquet",
                "paths": [path_ts.to_string_lossy()],
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "NoTS",
                "format": "parquet",
                "paths": [path_nots.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            }
        ]
    });
    
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;

    // Check WithTS
    let table = reader.table("WithTS").unwrap();
    // Valid Positive
    assert!(table.node_id_from_pk(&Key::Int(1)).unwrap().is_some(), "id=1 should exist");
    // Null TS -> Dropped
    assert!(table.node_id_from_pk(&Key::Int(2)).unwrap().is_none(), "id=2 should be dropped (null ts)");
    // Valid Negative -> Two's complement cast
    assert!(table.node_id_from_pk(&Key::Int(5)).unwrap().is_some(), "id=5 should exist");

    // Check NoTS
    let table = reader.table("NoTS").unwrap();
    assert!(table.node_id_from_pk(&Key::Int(10)).unwrap().is_some(), "id=10 should exist");
    assert!(table.node_id_from_pk(&Key::Int(20)).unwrap().is_some(), "id=20 should exist");

    Ok(())
}

#[test]
fn test_varbytes_pk_extraction() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&input_root)?;

    // 1. String Table
    let path_str = input_root.join("str.parquet");
    let schema_str = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), true),
    ]);
    let ids_str = Arc::new(StringArray::from(vec!["valid", "invalid_null", ""])) as ArrayRef;
    let tss_str = Arc::new(TimestampSecondArray::from(vec![Some(100), None, Some(200)])) as ArrayRef;
    write_parquet(&path_str, schema_str, vec![ids_str, tss_str])?;

    // 2. Bytes Table
    let path_bin = input_root.join("bin.parquet");
    let schema_bin = Schema::new(vec![
        Field::new("id", DataType::Binary, false),
        Field::new("ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), true),
    ]);
    let ids_bin = Arc::new(BinaryArray::from(vec![b"AAAA".as_ref(), b"BBBB".as_ref()])) as ArrayRef;
    // Row 2 (BBBB) has NULL timestamp
    let tss_bin = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&path_bin, schema_bin, vec![ids_bin, tss_bin])?;

    let source = serde_json::json!({
        "dataset_name": "varbytes_extraction_test",
        "tables": [
            {
                "name": "StrTable",
                "format": "parquet",
                "paths": [path_str.to_string_lossy()],
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "BinTable",
                "format": "parquet",
                "paths": [path_bin.to_string_lossy()],
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "bytes"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            }
        ]
    });
    
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;

    // Check StrTable
    let table = reader.table("StrTable").unwrap();
    assert!(table.node_id_from_pk(&Key::Bytes(b"valid".to_vec())).unwrap().is_some(), "id='valid' should exist");
    // This assertion implicitly checks key corruption: if validity byte was appended, lookup "valid" would fail.
    assert!(table.node_id_from_pk(&Key::Bytes(b"".to_vec())).unwrap().is_some(), "id='' should exist");
    assert!(table.node_id_from_pk(&Key::Bytes(b"invalid_null".to_vec())).unwrap().is_none(), "id='invalid_null' should be dropped");

    // Check BinTable
    let table = reader.table("BinTable").unwrap();
    assert!(table.node_id_from_pk(&Key::Bytes(b"AAAA".to_vec())).unwrap().is_some(), "id=AAAA should exist");
    assert!(table.node_id_from_pk(&Key::Bytes(b"BBBB".to_vec())).unwrap().is_none(), "id=BBBB should be dropped");

    Ok(())
}

#[test]
fn test_uuid_pk_extraction() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&input_root)?;

    // 1. UUID (16)
    let path_uuid = input_root.join("uuid.parquet");
    let schema_uuid = Schema::new(vec![
        Field::new("id", DataType::FixedSizeBinary(16), false),
        Field::new("ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), true),
    ]);
    let uuid1 = [1u8; 16];
    let uuid2 = [2u8; 16]; // Null TS
    let mut builder = arrow_array::builder::FixedSizeBinaryBuilder::new(16);
    builder.append_value(&uuid1).unwrap();
    builder.append_value(&uuid2).unwrap();
    let ids_uuid = Arc::new(builder.finish()) as ArrayRef;
    let tss_uuid = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&path_uuid, schema_uuid, vec![ids_uuid, tss_uuid])?;

    // 2. UUID256 (32)
    let path_uuid256 = input_root.join("uuid256.parquet");
    let schema_uuid256 = Schema::new(vec![
        Field::new("id", DataType::FixedSizeBinary(32), false),
        Field::new("ts", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), true),
    ]);
    let uuid256_1 = [3u8; 32];
    let uuid256_2 = [4u8; 32]; // Null TS
    let mut builder = arrow_array::builder::FixedSizeBinaryBuilder::new(32);
    builder.append_value(&uuid256_1).unwrap();
    builder.append_value(&uuid256_2).unwrap();
    let ids_uuid256 = Arc::new(builder.finish()) as ArrayRef;
    let tss_uuid256 = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&path_uuid256, schema_uuid256, vec![ids_uuid256, tss_uuid256])?;

    let source = serde_json::json!({
        "dataset_name": "uuid_extraction_test",
        "tables": [
            {
                "name": "UuidTable",
                "format": "parquet",
                "paths": [path_uuid.to_string_lossy()],
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "uuid"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "Uuid256Table",
                "format": "parquet",
                "paths": [path_uuid256.to_string_lossy()],
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "uuid256"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            }
        ]
    });
    
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;

    // Check UuidTable
    let table = reader.table("UuidTable").unwrap();
    assert!(table.node_id_from_pk(&Key::Uuid(uuid1)).unwrap().is_some(), "uuid1 should exist");
    assert!(table.node_id_from_pk(&Key::Uuid(uuid2)).unwrap().is_none(), "uuid2 should be dropped");

    // Check Uuid256Table
    let table = reader.table("Uuid256Table").unwrap();
    assert!(table.node_id_from_pk(&Key::Uuid256(uuid256_1)).unwrap().is_some(), "uuid256_1 should exist");
    assert!(table.node_id_from_pk(&Key::Uuid256(uuid256_2)).unwrap().is_none(), "uuid256_2 should be dropped");

    Ok(())
}