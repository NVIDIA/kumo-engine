// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stored rows of an index table, decoded to Arrow a chunk of nodes at a time.

use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use diskgraph_core::time::TimeUnit;
use diskgraph_index::graph::TableReader;
use diskgraph_rowpack::{RowAccess, RowAccessHints, RowAccessStats};

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
