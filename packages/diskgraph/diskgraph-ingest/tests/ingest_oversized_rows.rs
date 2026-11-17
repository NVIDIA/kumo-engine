// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use arrow_array::{
    Array, ArrayRef, BinaryArray, Int64Array, RecordBatch, StringArray, TimestampSecondArray,
};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::InMemoryOutput;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::stats::IngestStats;
use diskgraph_ingest::{ingest_with_stats, IngestOptions};
use diskgraph_rowpack::RowPackDecodeScratch;
use std::path::Path;
use std::sync::{Arc, Mutex};

const BIG: usize = 33 << 20;

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn write_parquet(path: &Path, columns: Vec<(&str, ArrayRef)>) -> Result<()> {
    let batch = RecordBatch::try_from_iter(columns)?;
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(std::fs::File::create(path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Ingests `Owners(owner_id)` and `Items(item_id, created, owner_id, blob_a,
/// blob_b)`, with `owner_id` a string FK, and returns the stats, warnings and
/// reader.
fn ingest_items(
    dir: &Path,
    owner_ids: &[&str],
    blobs: &[&[u8]],
) -> Result<(IngestStats, String, Arc<GraphIndexReader>)> {
    let owners = dir.join("owners.parquet");
    let items = dir.join("items.parquet");
    write_parquet(
        &owners,
        vec![(
            "owner_id",
            Arc::new(StringArray::from(vec!["o1"])) as ArrayRef,
        )],
    )?;
    let ids: Vec<i64> = (1..=owner_ids.len() as i64).collect();
    let created: Vec<i64> = ids.iter().map(|id| 1000 * id).collect();
    write_parquet(
        &items,
        vec![
            ("item_id", Arc::new(Int64Array::from(ids)) as ArrayRef),
            ("created", Arc::new(TimestampSecondArray::from(created))),
            ("owner_id", Arc::new(StringArray::from(owner_ids.to_vec()))),
            ("blob_a", Arc::new(BinaryArray::from(blobs.to_vec()))),
            ("blob_b", Arc::new(BinaryArray::from(blobs.to_vec()))),
        ],
    )?;
    let config = serde_json::json!({
        "dataset_name": "oversized_rows",
        "tables": [
            {
                "name": "Owners",
                "source": {"type": "parquet", "paths": [owners]},
                "primary_key": "owner_id",
                "columns": [{"name": "owner_id", "ingested_dtype": "string"}]
            },
            {
                "name": "Items",
                "source": {"type": "parquet", "paths": [items]},
                "primary_key": "item_id",
                "create_time": "created",
                "columns": [
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "created", "ingested_dtype": "timestamp"},
                    {"name": "owner_id", "ingested_dtype": "string"},
                    {"name": "blob_a", "ingested_dtype": "bytes"},
                    {"name": "blob_b", "ingested_dtype": "bytes"}
                ],
                "fkeys": [{"src_column": "owner_id", "dst_table": "Owners"}]
            }
        ]
    });
    let config_path = dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_vec(&config)?)?;
    let output = dir.join("index");
    let logs = Logs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer({
            let logs = logs.clone();
            move || logs.clone()
        })
        .with_ansi(false)
        .finish();
    let (_, stats) = tracing::subscriber::with_default(subscriber, || {
        ingest_with_stats(
            &config_path,
            output.to_str().unwrap(),
            Dest::Local,
            IngestOptions::default(),
        )
    })?;
    let logs = String::from_utf8(logs.0.lock().unwrap().clone())?;
    let reader = GraphIndexReader::open(&output.join("index_metadata.json"), &output)?;
    Ok((stats, logs, Arc::new(reader)))
}

fn item_node(items: &Arc<TableReader>, item_id: &str) -> u64 {
    let key = items.parse_pk_key(item_id).unwrap();
    items.node_id_from_pk(&key).unwrap().unwrap()
}

fn fetch_items(reader: &Arc<GraphIndexReader>, node: u64) -> RecordBatch {
    let engine = DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1)).unwrap();
    let type_id = engine
        .get_type_name_mapping()
        .node_types
        .iter()
        .position(|name| name == "Items")
        .unwrap();
    let mut output = InMemoryOutput::new();
    engine
        .fetch_features_for_type(type_id, &[node], &[], &mut output)
        .unwrap();
    output.feature_batches.remove(0)
}

fn is_valid_at(items: &Arc<TableReader>, node: u64, time: i64) -> bool {
    let (create, end) = items.temporal_col_indices();
    let mut scratch = RowPackDecodeScratch::default();
    items
        .is_node_valid_between(node, time, time, create, end, &mut scratch)
        .unwrap()
}

#[test]
fn oversized_row_keeps_key_time_and_fk_with_null_features() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let big = vec![7u8; BIG];
    let (stats, logs, reader) =
        ingest_items(dir.path(), &["o1", "o1"], &[b"small", big.as_slice()])?;

    assert_eq!(stats.oversized_rows_counts.get("Items"), Some(&1));
    assert_eq!(stats.dropped_rows_counts.get("Items"), None);
    assert!(logs.contains(
        "1 rows of table Items exceeded the 64 MiB node-data limit; \
         their features are stored as null"
    ));

    let items = reader.table("Items").unwrap();
    let node = item_node(&items, "2");
    let batch = fetch_items(&reader, node);
    let item = batch.column_by_name("item_id").unwrap();
    let item = item.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(item.value(0), 2);
    let owner = batch.column_by_name("owner_id").unwrap();
    let owner = owner.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(owner.value(0), "o1");
    let created = batch.column_by_name("created").unwrap();
    assert!(created.is_valid(0));
    assert!(batch.column_by_name("blob_a").unwrap().is_null(0));
    assert!(batch.column_by_name("blob_b").unwrap().is_null(0));

    assert!(!is_valid_at(&items, node, 1999));
    assert!(is_valid_at(&items, node, 2000));
    let edges = items.edges_for_node(node)?.relation_segments()?;
    assert_eq!(edges.iter().map(|s| s.count).sum::<u32>(), 1);
    Ok(())
}

#[test]
fn row_over_the_limit_without_features_has_no_node_data() -> Result<()> {
    let dir = tempfile::tempdir()?;
    // An FK column is kept, and this value alone is over the limit.
    let mut state = 1u64;
    let owner: String = (0..65 << 20)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            char::from(b'a' + (state % 26) as u8)
        })
        .collect();
    let (stats, _, reader) = ingest_items(dir.path(), &[&owner], &[b"small"])?;

    assert_eq!(stats.oversized_rows_counts.get("Items"), Some(&1));
    let items = reader.table("Items").unwrap();
    let node = item_node(&items, "1");
    let raw = items.edges_for_node(node)?.node_data()?;
    assert_eq!(raw, Some(&[][..]));
    assert_eq!(
        items.validate_pk_at_node(node, &items.parse_pk_key("1")?)?,
        None
    );

    let batch = fetch_items(&reader, node);
    assert!(batch.columns().iter().all(|column| column.is_null(0)));
    assert!(is_valid_at(&items, node, 0));
    Ok(())
}
