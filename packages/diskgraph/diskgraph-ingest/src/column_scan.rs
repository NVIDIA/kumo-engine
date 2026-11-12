// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stored rows of an index table, decoded to Arrow a chunk of nodes at a time.

use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, ensure, Result};
use arrow_array::builder::{ArrayBuilder, LargeStringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, LargeStringArray, RecordBatch};
use diskgraph_core::time::TimeUnit;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_rowpack::{IngestedType, RowAccess, RowAccessHints, RowAccessStats};

use crate::index_metadata::INDEX_METADATA_FILENAME;

const CHUNK_NODES: u64 = 1 << 16;

/// `(shard, first node, end node)`, in node order.
pub fn node_chunks(table: &TableReader) -> Vec<(usize, u64, u64)> {
    (0..table.shard_count())
        .filter_map(|shard| {
            let (min, max) = table.shard_id_range(shard)?;
            (min <= max).then_some((shard, max - min + 1))
        })
        .flat_map(|(shard, nodes)| {
            (0..nodes)
                .step_by(CHUNK_NODES as usize)
                .map(move |start| (shard, start, nodes.min(start + CHUNK_NODES)))
        })
        .collect()
}

/// The `columns` of a chunk's rows stored with node data, in node order.
pub fn decode_chunk(
    table: &TableReader,
    (shard, start, end): (usize, u64, u64),
    columns: &[usize],
) -> Result<RecordBatch> {
    let shard = table
        .shard(shard)
        .ok_or_else(|| anyhow!("no shard {shard}"))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("table has no node data"))?;
    let mut rows = Vec::new();
    for node in start..end {
        // Rows stored without node data, as oversized ones are, hold no values.
        if let Some(row) = shard.edges_for_node(node)?.node_data()? {
            if !row.is_empty() {
                rows.push(row);
            }
        }
    }
    decoder.decode_batch_to_arrow(&rows, columns)
}

/// Each create time in ticks of `clock`, read as ingest reads it.
pub fn create_times(array: &ArrayRef, clock: TimeUnit) -> Vec<Option<i64>> {
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::new(None))
        .with_clock(clock);
    access.set_cols(std::slice::from_ref(array));
    (0..array.len())
        .map(|row| {
            access.set_row(row);
            access.get_timestamp(0).ok().flatten()
        })
        .collect()
}

/// Whether each row counts at `cutoff`: created at or before it; every row without a create time.
pub fn at_cutoff(times: Option<&[Option<i64>]>, rows: usize, cutoff: Option<i64>) -> BooleanArray {
    match (times, cutoff) {
        (Some(times), Some(cutoff)) => times
            .iter()
            .map(|t| Some(t.is_some_and(|t| t <= cutoff)))
            .collect(),
        _ => BooleanArray::from(vec![true; rows]),
    }
}

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
    /// The string column, then the create time if the table has one.
    columns: Vec<usize>,
    cutoff: Option<i64>,
    clock: TimeUnit,
    stop: Arc<AtomicBool>,
}

/// A string column's non-null values over the rows counted at `cutoff`, in node order:
/// `chunk_rows` per array, decoded on `num_threads` threads, two arrays ahead.
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
    let stored = &table_reader
        .node_data_decoder()
        .ok_or_else(|| anyhow!("table {table} has no node data"))?
        .context()
        .columns;
    let position = |name: &str| stored.iter().position(|c| c.name == name);
    let index = position(column).ok_or_else(|| anyhow!("table {table} has no column {column}"))?;
    ensure!(
        stored[index].logical_type == IngestedType::String,
        "column {table}.{column} holds {:?} values, not strings",
        stored[index].logical_type
    );
    let columns = std::iter::once(index)
        .chain(table_reader.create_time_col().and_then(position))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(0))
        .build()?;
    let (sender, chunks) = sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let scan = Scan {
        table: table_reader,
        columns,
        cutoff,
        clock: reader.metadata().timestamp_unit,
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
    fn produce(
        &self,
        pool: &rayon::ThreadPool,
        chunk_rows: usize,
        sender: &SyncSender<Result<LargeStringArray>>,
    ) -> Result<()> {
        use rayon::prelude::*;

        let mut out = LargeStringBuilder::new();
        for window in node_chunks(&self.table).chunks(pool.current_num_threads()) {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let decoded: Vec<RecordBatch> = pool.install(|| {
                window
                    .par_iter()
                    .map(|&chunk| decode_chunk(&self.table, chunk, &self.columns))
                    .collect::<Result<_>>()
            })?;
            for batch in &decoded {
                let times =
                    (batch.num_columns() > 1).then(|| create_times(batch.column(1), self.clock));
                let keep = at_cutoff(times.as_deref(), batch.num_rows(), self.cutoff);
                let values = batch.column(0).as_string::<i32>();
                for row in
                    (0..batch.num_rows()).filter(|&row| keep.value(row) && values.is_valid(row))
                {
                    out.append_value(values.value(row));
                    if out.len() == chunk_rows && sender.send(Ok(out.finish())).is_err() {
                        return Ok(());
                    }
                }
            }
        }
        if !out.is_empty() {
            let _ = sender.send(Ok(out.finish()));
        }
        Ok(())
    }
}
