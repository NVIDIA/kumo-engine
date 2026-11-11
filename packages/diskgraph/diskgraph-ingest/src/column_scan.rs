// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One stored string column, streamed in shard then node order over the rows
//! the column statistics count at a cutoff. Chunks of nodes decode ahead in
//! parallel and reach the reader in order.

use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use arrow_array::builder::{ArrayBuilder, LargeStringBuilder};
use arrow_array::LargeStringArray;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_rowpack::{IngestedType, RowPackDecodeScratch, ValueRef};
use rayon::prelude::*;

use crate::cutoff_stats::{column_index, node_chunks, stored_row, CreateTime, SCAN_CHUNK_NODES};
use crate::index_metadata::INDEX_METADATA_FILENAME;

/// The column's non-null values, `chunk_rows` per array except the last.
pub struct ColumnScan {
    chunks: Receiver<Result<LargeStringArray>>,
}

impl Iterator for ColumnScan {
    type Item = Result<LargeStringArray>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.recv().ok()
    }
}

struct Scan {
    table: Arc<TableReader>,
    column: usize,
    mask: Vec<bool>,
    create_time: CreateTime,
    cutoff: Option<i64>,
}

/// Opens the scan; decoding runs on `num_threads` threads (all cores when
/// `None`), at most one chunk of nodes each plus as many finished arrays ahead.
pub fn scan_column(
    index_dir: &Path,
    table: &str,
    column: &str,
    cutoff: Option<i64>,
    chunk_rows: usize,
    num_threads: Option<usize>,
) -> Result<ColumnScan> {
    ensure!(chunk_rows > 0, "chunk_rows must be positive");
    let reader = GraphIndexReader::open_subset(
        &index_dir.join(INDEX_METADATA_FILENAME),
        index_dir,
        &HashSet::from([table.to_string()]),
    )?;
    let table_reader = reader
        .table(table)
        .ok_or_else(|| anyhow!("index has no node table {table}"))?;
    let decoder = table_reader
        .node_data_decoder()
        .ok_or_else(|| anyhow!("table {table} has no node data"))?;
    let columns: Vec<_> = decoder
        .context()
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c.logical_type))
        .collect();
    let index = column_index(&columns, column).with_context(|| format!("table {table}"))?;
    ensure!(
        columns[index].1 == IngestedType::String,
        "column {table}.{column} holds {:?} values, not strings",
        columns[index].1
    );
    let create_time = CreateTime::new(
        &columns,
        table_reader.create_time_col(),
        reader.metadata().timestamp_unit,
    )
    .with_context(|| format!("table {table}"))?;
    let mask = (0..columns.len())
        .map(|idx| idx == index || create_time.is(idx))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(0))
        .build()?;
    let (sender, chunks) = sync_channel(pool.current_num_threads());
    let scan = Scan {
        table: table_reader,
        column: index,
        mask,
        create_time,
        cutoff,
    };
    std::thread::spawn(move || {
        if let Err(err) = scan.produce(&pool, chunk_rows, &sender) {
            let _ = sender.send(Err(err));
        }
    });
    Ok(ColumnScan { chunks })
}

impl Scan {
    /// Decodes a window of node chunks at a time and re-chunks the values in
    /// order; stops once the reader is gone.
    fn produce(
        &self,
        pool: &rayon::ThreadPool,
        chunk_rows: usize,
        sender: &SyncSender<Result<LargeStringArray>>,
    ) -> Result<()> {
        let nodes = (chunk_rows as u64).min(SCAN_CHUNK_NODES);
        let mut out = LargeStringBuilder::new();
        for window in node_chunks(&self.table, nodes).chunks(pool.current_num_threads()) {
            let decoded: Vec<LargeStringArray> = pool.install(|| {
                window
                    .par_iter()
                    .map(|&chunk| self.decode(chunk))
                    .collect::<Result<_>>()
            })?;
            for value in decoded.iter().flat_map(|array| array.iter().flatten()) {
                out.append_value(value);
                if out.len() == chunk_rows && sender.send(Ok(out.finish())).is_err() {
                    return Ok(());
                }
            }
        }
        if !out.is_empty() {
            let _ = sender.send(Ok(out.finish()));
        }
        Ok(())
    }

    fn decode(&self, (shard, start, end): (usize, u64, u64)) -> Result<LargeStringArray> {
        let shard = self.table.shard(shard).expect("shard in range");
        let decoder = self.table.node_data_decoder().expect("checked at open");
        let mut scratch = RowPackDecodeScratch::default();
        let mut text = String::new();
        let mut values = LargeStringBuilder::new();
        for node in start..end {
            let Some(row) = stored_row(shard, node)? else {
                continue;
            };
            let (mut present, mut time) = (false, None);
            decoder.for_each_value(row, &self.mask, &mut scratch, |idx, value| {
                if idx != self.column {
                    time = self.create_time.ticks(value.into());
                } else if let ValueRef::String(value) = value {
                    text.clear();
                    text.push_str(value);
                    present = true;
                }
            })?;
            if present && self.create_time.includes(self.cutoff, time) {
                values.append_value(&text);
            }
        }
        Ok(values.finish())
    }
}
