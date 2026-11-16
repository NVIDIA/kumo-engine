// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::{ingest_with_stats, IngestOptions};
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
fn test_null_timestamp_metrics_and_filtering() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&input_root)?;

    // 1. TableWithPK: Has PK and TS.
    // Row 1: Valid
    // Row 2: Null TS (Should drop)
    let path_pk = input_root.join("with_pk.parquet");
    let schema_pk = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            true,
        ),
    ]);
    let ids_pk = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
    let tss_pk = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&path_pk, schema_pk, vec![ids_pk, tss_pk])?;

    // 2. TableNoPK: No PK, has TS and FK to TableWithPK.
    // Row 1: Valid TS, FK=1 (Valid)
    // Row 2: Null TS, FK=1 (Should drop, no edge)
    // Row 3: Valid TS, FK=1 (Valid)
    let path_nopk = input_root.join("no_pk.parquet");
    let schema_nopk = Schema::new(vec![
        Field::new("target_id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            true,
        ),
    ]);
    let target_ids = Arc::new(Int64Array::from(vec![1, 1, 1])) as ArrayRef;
    let tss_nopk =
        Arc::new(TimestampSecondArray::from(vec![Some(100), None, Some(200)])) as ArrayRef;
    write_parquet(&path_nopk, schema_nopk, vec![target_ids, tss_nopk])?;

    let source = serde_json::json!({
        "dataset_name": "null_ts_metrics_test",
        "tables": [
            {
                "name": "TableWithPK",
                "source": {
                    "type": "parquet",
                    "paths": [path_pk.to_string_lossy()]
                },
                "primary_key": "id",
                "create_time": "ts",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "TableNoPK",
                "source": {
                    "type": "parquet",
                    "paths": [path_nopk.to_string_lossy()]
                },
                // No primary_key
                "create_time": "ts",
                "columns": [
                    {"name": "target_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ],
                "fkeys": [
                    {
                        "src_column": "target_id",
                        "dst_table": "TableWithPK",
                        "forward": {"time": {"table": "TableNoPK", "column": "ts", "unit": "s"}}
                    }
                ]
            }
        ]
    });

    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Check Metrics
    println!("Dropped rows counts: {:?}", stats.dropped_rows_counts);

    // 1. TableWithPK should have 1 drop.
    assert_eq!(
        *stats.dropped_rows_counts.get("TableWithPK").unwrap_or(&0),
        1,
        "TableWithPK should have 1 drop"
    );

    // 2. TableNoPK should have 1 drop.
    assert_eq!(
        *stats.dropped_rows_counts.get("TableNoPK").unwrap_or(&0),
        1,
        "TableNoPK should have 1 drop"
    );

    // Check Graph Structure (Validation)
    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;

    // Verify edges from TableNoPK -> TableWithPK
    // TableWithPK id=1 is valid.
    // TableNoPK has 3 rows: 1 (valid), 2 (null ts), 3 (valid).
    // So node 1 should have 2 incoming edges (from rows 1 and 3).
    let table = reader.table("TableWithPK").unwrap();
    let _node_id = table
        .node_id_from_pk(&diskgraph_index::pkmap::Key::Int(1))?
        .unwrap();

    // Check edge counts from stats
    let mut edge_count = 0;
    for (name, count) in &stats.edge_counts {
        // Look for the forward relationship name
        if name.contains("TableNoPK") && name.contains("TableWithPK") && !name.contains("reverse") {
            println!("Edge count for {}: {}", name, count);
            edge_count = *count;
        }
    }
    assert_eq!(
        edge_count, 2,
        "Should have 2 edges (1 dropped due to null TS)"
    );

    Ok(())
}
