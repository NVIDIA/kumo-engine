// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

use arrow_array::{
    ArrayRef, FixedSizeBinaryArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1024))
        .build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn sum_relation_edges(table: &diskgraph_index::graph::TableReader, rel_name: &str) -> Result<u64> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| {
            let names: Vec<_> = table.relations().iter().map(|r| &r.name).collect();
            anyhow!("missing relation {rel_name}. Available: {:?}", names)
        })?;
    let mut total = 0u64;
    for node_id in 0..table.node_count() {
        let view = table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;
        total = total.saturating_add(segments[rel_idx].count as u64);
    }
    Ok(total)
}

#[test]
fn test_fkey_optimization_vectors() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // 1. Create Source Data

    // Table: users_int
    let schema_users_int = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let users_int_ids = Int32Array::from(vec![100, 200, 300]);
    write_parquet(
        &raw_dir.join("users_int/part-0.parquet"),
        schema_users_int,
        vec![Arc::new(users_int_ids)],
    )?;

    // Table: users_uuid
    let schema_users_uuid =
        Schema::new(vec![Field::new("id", DataType::FixedSizeBinary(16), false)]);
    let mut uuid1 = [0u8; 16];
    uuid1[15] = 1;
    let mut uuid2 = [0u8; 16];
    uuid2[15] = 2;
    let users_uuid_ids = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        vec![Some(uuid1), Some(uuid2)].into_iter(),
        16,
    )
    .unwrap();
    write_parquet(
        &raw_dir.join("users_uuid/part-0.parquet"),
        schema_users_uuid,
        vec![Arc::new(users_uuid_ids)],
    )?;

    // Table: users_direct
    let schema_users_direct = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_direct_ids = Int64Array::from(vec![10, 20, 100, 200]);
    write_parquet(
        &raw_dir.join("users_direct/part-0.parquet"),
        schema_users_direct,
        vec![Arc::new(users_direct_ids)],
    )?;

    // Table: events
    let schema_events = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("fk_cast", DataType::Int64, true),
        Field::new("fk_parse_int", DataType::Utf8, true),
        Field::new("fk_parse_uuid", DataType::Utf8, true),
        Field::new("fk_direct", DataType::Int64, true),
    ]);

    let event_ids = Int64Array::from(vec![1, 2, 3, 4]);

    let fk_cast = Int64Array::from(vec![Some(100), Some(200), None, Some(100)]);

    let fk_parse_int =
        StringArray::from(vec![Some("100"), Some("200"), None, Some("not_a_number")]);

    let uuid1_str = "00000000-0000-0000-0000-000000000001";
    let uuid2_str = "00000000-0000-0000-0000-000000000002";
    let fk_parse_uuid = StringArray::from(vec![
        Some(uuid1_str),
        Some(uuid2_str),
        None,
        Some("invalid-uuid"),
    ]);

    let fk_direct = Int64Array::from(vec![Some(10), Some(20), None, Some(10)]);

    write_parquet(
        &raw_dir.join("events/part-0.parquet"),
        schema_events,
        vec![
            Arc::new(event_ids),
            Arc::new(fk_cast),
            Arc::new(fk_parse_int),
            Arc::new(fk_parse_uuid),
            Arc::new(fk_direct),
        ],
    )?;

    // 2. Create Config
    let config = serde_json::json!({
        "dataset_name": "test_dataset",
        "tables": [
            {
                "name": "users_int",
                "source": {"type": "parquet", "paths": [raw_dir.join("users_int/part-0.parquet").to_str().unwrap()]},
                "columns": [{"name": "id", "ingested_dtype": "int32"}],
                "primary_key": "id"
            },
            {
                "name": "users_uuid",
                "source": {"type": "parquet", "paths": [raw_dir.join("users_uuid/part-0.parquet").to_str().unwrap()]},
                "columns": [{"name": "id", "ingested_dtype": "uuid"}],
                "primary_key": "id"
            },
            {
                "name": "users_direct",
                "source": {"type": "parquet", "paths": [raw_dir.join("users_direct/part-0.parquet").to_str().unwrap()]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "events",
                "source": {"type": "parquet", "paths": [raw_dir.join("events/part-0.parquet").to_str().unwrap()]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_cast", "ingested_dtype": "int32"},
                    {"name": "fk_parse_int", "ingested_dtype": "int64"},
                    {"name": "fk_parse_uuid", "ingested_dtype": "uuid"},
                    {"name": "fk_direct", "ingested_dtype": "int64"}
                ],
                "primary_key": "id",
                "fkeys": [
                    {"src_column": "fk_cast", "dst_table": "users_int"},
                    {"src_column": "fk_parse_int", "dst_table": "users_direct"},
                    {"src_column": "fk_parse_uuid", "dst_table": "users_uuid"},
                    {"src_column": "fk_direct", "dst_table": "users_direct"}
                ]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // 3. Run Ingest
    let opts = IngestOptions::default();

    ingest(&config_path, index_dir.to_str().unwrap(), Dest::Local, opts)?;

    // 4. Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;
    let events_table = reader.table("events").unwrap();

    // Check counts
    assert_eq!(
        sum_relation_edges(&events_table, "events_fk_cast_to_users_int_id")?,
        3,
        "fk_cast mismatch (Int64->Int32)"
    );
    assert_eq!(
        sum_relation_edges(&events_table, "events_fk_parse_int_to_users_direct_id")?,
        2,
        "fk_parse_int mismatch (String->Int64)"
    );
    assert_eq!(
        sum_relation_edges(&events_table, "events_fk_parse_uuid_to_users_uuid_id")?,
        2,
        "fk_parse_uuid mismatch (String->Uuid)"
    );
    assert_eq!(
        sum_relation_edges(&events_table, "events_fk_direct_to_users_direct_id")?,
        3,
        "fk_direct mismatch (Int64->Int64)"
    );

    Ok(())
}
