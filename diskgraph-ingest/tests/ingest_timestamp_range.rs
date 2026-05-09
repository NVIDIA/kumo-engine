//! Tests for timestamp range handling during ingest.
//!
//! Test 1: String-encoded timestamps where some rows have unparseable dates.
//! Test 2: Date32 column with out-of-range values (year 3193).
//! Test 3: Edge table with Date32 out-of-range time_col.
//! Test 4: Non-time-col timestamp with out-of-range value (should NOT drop row, just null the feature).

use anyhow::Result;
use arrow_array::{ArrayRef, Date32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::{ingest_with_stats, IngestOptions};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Test 1: String time_col with unparseable values → rows dropped.
#[test]
fn test_string_timestamp_parse_failure_drops_rows() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    // Hub table: just a PK
    let hub_path = input_root.join("hub.parquet");
    write_parquet(
        &hub_path,
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![1i64])) as ArrayRef],
    )?;

    // Event table: string time_col, some invalid
    let event_path = input_root.join("events.parquet");
    write_parquet(
        &event_path,
        Schema::new(vec![
            Field::new("hub_id", DataType::Int64, false),
            Field::new("ts", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 1, 1, 1, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "2024-01-15",  // valid
                "not-a-date",  // invalid
                "2024-03-20",  // valid
                "garbage",     // invalid
                "2024-06-01",  // valid
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
        ],
    )?;

    let accounts_path = input_root.join("accounts.parquet");
    write_parquet(
        &accounts_path,
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("acct_open_date", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Sep-02", "Apr-14", "Jul-03"])) as ArrayRef,
        ],
    )?;

    let config = serde_json::json!({
        "dataset_name": "string_ts_test",
        "tables": [
            {
                "name": "Hub",
                "source": {"type": "parquet", "paths": [hub_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}]
            },
            {
                "name": "Events",
                "source": {"type": "parquet", "paths": [event_path.to_string_lossy()]},
                "create_time": "ts",
                "columns": [
                    {"name": "hub_id", "ingested_dtype": "int64"},
                    {"name": "ts", "source_dtype": "Utf8", "ingested_dtype": "timestamp"},
                    {"name": "value", "ingested_dtype": "int64"}
                ],
                "fkeys": [{
                    "src_column": "hub_id",
                    "dst_table": "Hub",
                    "forward": {"time": {"table": "Events", "column": "ts", "unit": "s"}}
                }]
            },
            {
                "name": "Accounts",
                "source": {"type": "parquet", "paths": [accounts_path.to_string_lossy()]},
                "primary_key": "id",
                "create_time": "acct_open_date",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {
                        "name": "acct_open_date",
                        "source_dtype": "Utf8",
                        "ingested_dtype": "date"
                    }
                ]
            }
        ]
    });

    let config_path = dir.path().join("config.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &config_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            col_stats: true,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            ..IngestOptions::default()
        },
    )?;

    println!("dropped_rows_counts: {:?}", stats.dropped_rows_counts);
    println!("edge_counts: {:?}", stats.edge_counts);
    println!("node_counts: {:?}", stats.node_counts);

    let events_dropped = *stats.dropped_rows_counts.get("Events").unwrap_or(&0);
    assert_eq!(events_dropped, 2, "Should drop 2 rows with unparseable timestamps, got {events_dropped}");

    let accounts_dropped = *stats.dropped_rows_counts.get("Accounts").unwrap_or(&0);
    assert_eq!(
        accounts_dropped, 3,
        "Should drop all rows with unparseable date create_time, got {accounts_dropped}"
    );

    let edge_count = stats.edge_counts.iter()
        .find(|(k, _)| k.contains("Events") && k.contains("Hub") && !k.contains("reverse"))
        .map(|(_, v)| *v).unwrap_or(0);
    assert_eq!(edge_count, 3, "Should have 3 edges (2 dropped), got {edge_count}");

    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;
    assert!(reader.table("Hub").is_some());
    assert!(reader.table("Events").is_some());

    Ok(())
}

/// Test 2: Date32 time_col with out-of-range values → rows dropped.
#[test]
fn test_date32_out_of_range_drops_rows() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let hub_path = input_root.join("hub.parquet");
    write_parquet(
        &hub_path,
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![1i64])) as ArrayRef],
    )?;

    let event_path = input_root.join("events.parquet");
    write_parquet(
        &event_path,
        Schema::new(vec![
            Field::new("hub_id", DataType::Int64, false),
            Field::new("ts", DataType::Date32, false),
            Field::new("value", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 1, 1, 1, 1])) as ArrayRef,
            Arc::new(Date32Array::from(vec![
                19738,   // 2024-01-15 (valid)
                446824,  // 3193-05-13 (out of range)
                19875,   // 2024-06-01 (valid)
                385314,  // 3024-12-15 (out of range)
                20453,   // 2025-12-31 (valid)
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
        ],
    )?;

    let config = serde_json::json!({
        "dataset_name": "date32_range_test",
        "tables": [
            {
                "name": "Hub",
                "source": {"type": "parquet", "paths": [hub_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}]
            },
            {
                "name": "Events",
                "source": {"type": "parquet", "paths": [event_path.to_string_lossy()]},
                "create_time": "ts",
                "columns": [
                    {"name": "hub_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"},
                    {"name": "value", "ingested_dtype": "int64"}
                ],
                "fkeys": [{
                    "src_column": "hub_id",
                    "dst_table": "Hub",
                    "forward": {"time": {"table": "Events", "column": "ts", "unit": "days"}}
                }]
            }
        ]
    });

    let config_path = dir.path().join("config.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &config_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    println!("dropped_rows_counts: {:?}", stats.dropped_rows_counts);
    println!("edge_counts: {:?}", stats.edge_counts);
    println!("node_counts: {:?}", stats.node_counts);

    let events_dropped = *stats.dropped_rows_counts.get("Events").unwrap_or(&0);
    assert_eq!(events_dropped, 2, "Should drop 2 rows with out-of-range dates, got {events_dropped}");

    let edge_count = stats.edge_counts.iter()
        .find(|(k, _)| k.contains("Events") && k.contains("Hub") && !k.contains("reverse"))
        .map(|(_, v)| *v).unwrap_or(0);
    assert_eq!(edge_count, 3, "Should have 3 edges (2 dropped), got {edge_count}");

    Ok(())
}

/// Test 3: Edge table with Date32 out-of-range time_col → rows dropped.
#[test]
fn test_edge_table_date32_out_of_range_drops_rows() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    // Src table (PK)
    let src_path = input_root.join("src.parquet");
    write_parquet(
        &src_path,
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![1i64, 2])) as ArrayRef],
    )?;

    // Dst table (PK)
    let dst_path = input_root.join("dst.parquet");
    write_parquet(
        &dst_path,
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![100i64, 200])) as ArrayRef],
    )?;

    // Edge table with Date32 time_col (only fkey cols + time col, no other cols)
    let edge_path = input_root.join("edges.parquet");
    write_parquet(
        &edge_path,
        Schema::new(vec![
            Field::new("src_id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
            Field::new("ts", DataType::Date32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![100i64, 200, 100, 200])) as ArrayRef,
            Arc::new(Date32Array::from(vec![
                19738,   // 2024-01-15 (valid)
                446824,  // 3193-05-13 (out of range)
                19875,   // 2024-06-01 (valid)
                385314,  // 3024-12-15 (out of range)
            ])) as ArrayRef,
        ],
    )?;

    let config = serde_json::json!({
        "dataset_name": "edge_date32_test",
        "tables": [
            {
                "name": "Src",
                "source": {"type": "parquet", "paths": [src_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}]
            },
            {
                "name": "Dst",
                "source": {"type": "parquet", "paths": [dst_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}]
            },
            {
                "name": "Edges",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "create_time": "ts",
                "edge_table": {
                    "src_fk": {"edge_column": "src_id", "ref_table": "Src"},
                    "dst_fk": {"edge_column": "dst_id", "ref_table": "Dst"}
                },
                "columns": [
                    {"name": "src_id", "ingested_dtype": "int64"},
                    {"name": "dst_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ]
            }
        ]
    });

    let config_path = dir.path().join("config.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &config_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    println!("dropped_rows_counts: {:?}", stats.dropped_rows_counts);
    println!("edge_counts: {:?}", stats.edge_counts);

    let edges_dropped = *stats.dropped_rows_counts.get("Edges").unwrap_or(&0);
    assert_eq!(edges_dropped, 2, "Should drop 2 edge rows with out-of-range dates, got {edges_dropped}");

    // Each valid edge row creates edges in both directions
    let fwd_edge_count = stats.edge_counts.iter()
        .find(|(k, _)| k.contains("Edges") && k.contains("src_to_dst"))
        .map(|(_, v)| *v).unwrap_or(0);
    assert_eq!(fwd_edge_count, 2, "Should have 2 forward edges, got {fwd_edge_count}");

    Ok(())
}

/// Test 4: Non-time-col Date32 with out-of-range → null feature, row NOT dropped.
#[test]
fn test_non_timecol_out_of_range_keeps_row() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("data.parquet");
    write_parquet(
        &table_path,
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("valid_ts", DataType::Date32, false),
            Field::new("other_ts", DataType::Date32, false),
            Field::new("value", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])) as ArrayRef,
            // time_col: all valid
            Arc::new(Date32Array::from(vec![19738, 19875, 20453])) as ArrayRef,
            // non-time-col: one out of range
            Arc::new(Date32Array::from(vec![19738, 446824, 19875])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
        ],
    )?;

    let config = serde_json::json!({
        "dataset_name": "non_timecol_range_test",
        "tables": [{
            "name": "Data",
            "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
            "primary_key": "id",
            "create_time": "valid_ts",
            "columns": [
                {"name": "id", "ingested_dtype": "int64"},
                {"name": "valid_ts", "ingested_dtype": "timestamp"},
                {"name": "other_ts", "ingested_dtype": "timestamp"},
                {"name": "value", "ingested_dtype": "int64"}
            ]
        }]
    });

    let config_path = dir.path().join("config.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &config_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    println!("dropped_rows_counts: {:?}", stats.dropped_rows_counts);
    println!("node_counts: {:?}", stats.node_counts);

    // No rows should be dropped — the out-of-range value is in a non-time-col
    let dropped = *stats.dropped_rows_counts.get("Data").unwrap_or(&0);
    assert_eq!(dropped, 0, "Should not drop rows when non-time-col has out-of-range dates, got {dropped}");

    // All 3 rows should be present
    let node_count = *stats.node_counts.get("Data").unwrap_or(&0);
    assert_eq!(node_count, 3, "All 3 rows should be present, got {node_count}");

    Ok(())
}
