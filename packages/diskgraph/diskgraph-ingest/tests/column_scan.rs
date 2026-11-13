// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::column_scan::{node_chunks, scan_chunk, ScanBudget};
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;

#[test]
fn dictionary_expanded_strings_decode_in_batches_within_one_budget() -> Result<()> {
    let dir = tempfile::tempdir()?;
    // Three distinct 1 KiB values over 70,000 rows, two chunks: a small index, 68 MiB decoded.
    let values = ["a", "b", "c"].map(|c| c.repeat(KIB as usize));
    let rows = 70_000;
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef,
        ),
        (
            "tier",
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|i| &values[i % 3]),
            )),
        ),
    ])?;
    let parquet = dir.path().join("t.parquet");
    let mut writer = ArrowWriter::try_new(fs::File::create(&parquet)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    let config = dir.path().join("config.json");
    fs::write(
        &config,
        serde_json::to_vec(&serde_json::json!({"dataset_name": "wide", "tables": [{
            "name": "t", "primary_key": "id",
            "columns": [{"name": "id", "ingested_dtype": "int64"},
                        {"name": "tier", "ingested_dtype": "string"}],
            "source": {"type": "parquet", "paths": [parquet]}}]}))?,
    )?;
    let index = dir.path().join("index");
    let options = IngestOptions {
        col_stats: false,
        ..Default::default()
    };
    ingest(&config, index.to_str().unwrap(), Dest::Local, options)?;
    let reader = GraphIndexReader::open(&index.join("index_metadata.json"), &index)?;
    let table = &reader.table("t").unwrap();

    let budget = ScanBudget::with_limits(3 * MIB / 2, MIB)?;
    let seen = Mutex::new(vec![0u64; 3]);
    let (budget_ref, seen_ref, values_ref) = (&budget, &seen, &values);
    std::thread::scope(|scope| -> Result<()> {
        let workers: Vec<_> = node_chunks(table)
            .into_iter()
            .map(|chunk| {
                scope.spawn(move || {
                    scan_chunk(table, chunk, &[1], budget_ref, |batch, nodes| {
                        let tiers = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        assert!(tiers.value_data().len() as u64 <= MIB + KIB);
                        assert_eq!(nodes.len(), batch.num_rows());
                        let mut seen = seen_ref.lock().unwrap();
                        for tier in tiers.iter().flatten() {
                            seen[values_ref.iter().position(|v| v == tier).unwrap()] += 1;
                        }
                        Ok(())
                    })
                })
            })
            .collect();
        assert!(workers.len() > 1);
        for worker in workers {
            worker.join().unwrap()?;
        }
        Ok(())
    })?;
    assert_eq!(*seen.lock().unwrap(), [23_334, 23_333, 23_333]);
    assert!(budget.high_water_bytes() <= 3 * MIB / 2);
    Ok(())
}
