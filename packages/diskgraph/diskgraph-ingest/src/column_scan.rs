// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One stored string column, streamed in shard then node order over the rows
//! the column statistics count at a cutoff. Chunks of nodes decode ahead in
//! parallel and reach the reader in order.

use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::Duration;

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
    stop: Arc<AtomicBool>,
}

pub enum Next {
    Chunk(Result<LargeStringArray>),
    End,
    Pending,
}

impl ColumnScan {
    /// The next chunk, or `Pending` if none is ready within `timeout`.
    pub fn next_within(&self, timeout: Duration) -> Next {
        match self.chunks.recv_timeout(timeout) {
            Ok(chunk) => Next::Chunk(chunk),
            Err(RecvTimeoutError::Timeout) => Next::Pending,
            Err(RecvTimeoutError::Disconnected) => Next::End,
        }
    }
}

impl Drop for ColumnScan {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

struct Scan {
    table: Arc<TableReader>,
    column: usize,
    mask: Vec<bool>,
    create_time: CreateTime,
    cutoff: Option<i64>,
    stop: Arc<AtomicBool>,
}

/// Opens the scan; decoding runs on `num_threads` threads (all cores when
/// `None`), one chunk of nodes each, with two finished arrays queued ahead.
pub fn scan_column(
    index_dir: &Path,
    table: &str,
    column: &str,
    cutoff: Option<i64>,
    chunk_rows: usize,
    num_threads: Option<usize>,
) -> Result<ColumnScan> {
    ensure!(chunk_rows > 0, "chunk_rows must be positive");
    ensure!(num_threads != Some(0), "num_threads must be positive");
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
    let (sender, chunks) = sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let scan = Scan {
        table: table_reader,
        column: index,
        mask,
        create_time,
        cutoff,
        stop: Arc::clone(&stop),
    };
    let panicked = format!("column scan of {table}.{column} panicked");
    std::thread::spawn(move || {
        let produced = panic::catch_unwind(AssertUnwindSafe(|| {
            scan.produce(&pool, chunk_rows, &sender)
        }))
        .unwrap_or_else(|_| Err(anyhow!(panicked)));
        if let Err(err) = produced {
            let _ = sender.send(Err(err));
        }
    });
    Ok(ColumnScan { chunks, stop })
}

impl Scan {
    /// Decodes a window of node chunks at a time and re-chunks the values in
    /// order; stops once the scan is dropped.
    fn produce(
        &self,
        pool: &rayon::ThreadPool,
        chunk_rows: usize,
        sender: &SyncSender<Result<LargeStringArray>>,
    ) -> Result<()> {
        let mut out = LargeStringBuilder::new();
        for window in node_chunks(&self.table, SCAN_CHUNK_NODES).chunks(pool.current_num_threads())
        {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
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
