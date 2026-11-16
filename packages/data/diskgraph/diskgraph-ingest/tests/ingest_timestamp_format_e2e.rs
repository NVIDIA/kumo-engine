// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for the `timestamp_format` plumbing path: a hint on the
//! Kumo config (kumo's `metadata.TS_FORMAT`) must flow through
//! `convert_kumo_to_diskgraph` into `ColumnSpec.timestamp_format`, drive
//! `RowAccessHints.timestamp_format` in the rowpack parser, and parse
//! correctly on readback.
//!
//! Mirrors the production wire shape: `issued` appears in `cols[]` (with
//! the hint set) and `time_col` (hint omitted). The format used here
//! (`%y%m%d %H:%M:%S`) is not in `diskgraph_rowpack::TIMESTAMP_FORMATS`, so
//! the explicit-hint path is the only way it can parse.

use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
};
use diskgraph_ingest::graph_config::{IngestedType, TimestampUnit};
use diskgraph_ingest::{ingest_with_stats_from_source, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
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
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[test]
fn ingest_with_explicit_timestamp_format_parses_2digit_year() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&data_dir)?;

    // Compact 2-digit-year rows — no entry in
    // `diskgraph_rowpack::TIMESTAMP_FORMATS` matches, so only the
    // explicit-hint path can parse these correctly.
    let path = data_dir.join("cards.parquet");
    write_parquet(
        &path,
        Schema::new(vec![
            Field::new("card_id", DataType::Int64, false),
            Field::new("issued", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "931107 00:00:00",
                "010315 12:30:45",
                "991231 23:59:59",
            ])) as ArrayRef,
        ],
    )?;

    // Production wire shape: `issued` in `cols[]` carries the hint, the
    // `time_col` reference is bare. The consistency check accepts this
    // one-side-`None` pair; the `cols[]` value drives the parser.
    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "Card",
            "data_source": "test_ds",
            "data_version": "v1",
            "pkey": {"name": "card_id", "dtype": "int64"},
            "cols": [
                {
                    "name": "issued",
                    "dtype": "time",
                    "stype": "timestamp",
                    "timestamp_format": "%y%m%d %H:%M:%S"
                }
            ],
            // Same column name as above with `timestamp_format` omitted;
            // dedup is first-wins so the `cols[]` hint reaches ColumnSpec.
            "time_col": {
                "name": "issued",
                "dtype": "time",
                "stype": "timestamp"
            }
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;

    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "ts_format_e2e".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("Card".to_string(), data_dir.to_string_lossy().to_string());
            m
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;

    // Sanity check the conversion result before ingest.
    let issued_col = cfg.tables[0]
        .columns
        .iter()
        .find(|c| c.name == "issued")
        .ok_or_else(|| anyhow!("missing issued column"))?;
    assert_eq!(
        issued_col.timestamp_format.as_deref(),
        Some("%y%m%d %H:%M:%S"),
        "explicit timestamp_format from `cols[]` must reach ColumnSpec",
    );
    assert_eq!(
        cfg.tables[0].create_time.as_deref(),
        Some("issued"),
        "time_col reference must still be honored as create_time",
    );

    // Drive ingest end-to-end.
    let (_metadata, _stats) = ingest_with_stats_from_source(
        cfg,
        None,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )?;

    // Read back via JSON; the timestamp column should land as a
    // chrono-parseable epoch value (int or RFC3339 string).
    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;
    let table = reader
        .table("Card")
        .ok_or_else(|| anyhow!("missing Card table"))?;

    // 1993-11-07 00:00:00 UTC = 752630400 epoch seconds
    let expected_1993 = 752_630_400_i64;
    // 2001-03-15 12:30:45 UTC = 984659445 epoch seconds
    let expected_2001 = 984_659_445_i64;
    // 1999-12-31 23:59:59 UTC = 946684799 epoch seconds
    let expected_1999 = 946_684_799_i64;

    let mut found = std::collections::HashMap::<i64, i64>::new();
    for node_id in 0..table.node_count() {
        let view = table.edges_for_node(node_id)?;
        if let Some(row) = table.decode_node_data_json(&view)? {
            let card_id = row
                .get("card_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow!("missing card_id in row {row}"))?;
            // Timestamp comes back as either an integer (seconds since
            // epoch) or a string; accept either shape.
            let issued = row
                .get("issued")
                .ok_or_else(|| anyhow!("missing issued in row {row}"))?
                .clone();
            let issued_epoch = match issued {
                serde_json::Value::Number(n) => n
                    .as_i64()
                    .ok_or_else(|| anyhow!("issued not i64: {n}"))?,
                serde_json::Value::String(s) => {
                    // RFC3339 / ISO8601 readback — strip the time and parse.
                    chrono::DateTime::parse_from_rfc3339(&s)
                        .map(|d| d.timestamp())
                        .or_else(|_| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .map(|d| d.and_utc().timestamp())
                        })
                        .map_err(|e| anyhow!("could not parse issued string '{s}': {e}"))?
                }
                other => return Err(anyhow!("unexpected issued shape: {other}")),
            };
            found.insert(card_id, issued_epoch);
        }
    }

    assert_eq!(
        found.get(&1).copied(),
        Some(expected_1993),
        "card_id=1 ('931107 00:00:00') should parse to 1993-11-07 epoch",
    );
    assert_eq!(
        found.get(&2).copied(),
        Some(expected_2001),
        "card_id=2 ('010315 12:30:45') should parse to 2001-03-15 epoch",
    );
    assert_eq!(
        found.get(&3).copied(),
        Some(expected_1999),
        "card_id=3 ('991231 23:59:59') should parse to 1999-12-31 epoch",
    );

    Ok(())
}

#[test]
fn ingest_int64_unix_seconds_time_role_as_timestamp() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    let output_root = dir.path().join("output");
    fs::create_dir_all(&data_dir)?;

    let row_count = 40_i64;
    let path = data_dir.join("events.parquet");
    write_parquet(
        &path,
        Schema::new(vec![
            Field::new("EventID", DataType::Int64, false),
            Field::new("EventTime", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from_iter_values(0..row_count)) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(
                (0..row_count).map(|offset| 1_700_000_000 + offset),
            )) as ArrayRef,
        ],
    )?;

    // This matches the Kumo graph shape from kumo-ai/kumo#31171: Unix
    // seconds remain physically int64 while the authored role is timestamp.
    let kumo_json = serde_json::json!({
        "tables": [{
            "table_name": "events",
            "data_source": "test_ds",
            "data_version": "v1",
            "pkey": {"name": "EventID", "dtype": "int64"},
            "cols": [],
            "time_col": {
                "name": "EventTime",
                "dtype": "int64",
                "stype": "timestamp",
                "timestamp_format": "s"
            }
        }]
    });
    let kumo: KumoGraphConfig = serde_json::from_value(kumo_json)?;
    let options = ConvertOptions {
        data_persist_dir: data_dir.to_string_lossy().to_string(),
        dataset_name: "int64_time_role_e2e".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some({
            let mut paths = std::collections::HashMap::new();
            paths.insert(
                "events".to_string(),
                data_dir.to_string_lossy().to_string(),
            );
            paths
        }),
        source_specs: None,
    };
    let cfg = convert_kumo_to_diskgraph(&kumo, &options)?;

    let event_time = cfg.tables[0]
        .columns
        .iter()
        .find(|column| column.name == "EventTime")
        .ok_or_else(|| anyhow!("missing EventTime column"))?;
    assert_eq!(event_time.source_dtype.as_deref(), Some("int64"));
    assert_eq!(event_time.ingested_dtype, IngestedType::Timestamp);
    assert_eq!(event_time.timestamp_unit, Some(TimestampUnit::Seconds));

    let (_metadata, _stats) = ingest_with_stats_from_source(
        cfg,
        None,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )?;

    let metadata_path = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, &output_root)?;
    let table = reader
        .table("events")
        .ok_or_else(|| anyhow!("missing events table"))?;
    assert_eq!(table.node_count(), row_count as u64);

    Ok(())
}
