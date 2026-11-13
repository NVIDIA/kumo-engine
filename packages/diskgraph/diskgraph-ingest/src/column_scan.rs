// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stored rows of an index table, decoded to Arrow a chunk of nodes at a time.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use diskgraph_core::time::TimeUnit;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_rowpack::{IngestedType, RowAccess, RowAccessHints, RowAccessStats};

use crate::index_metadata::INDEX_METADATA_FILENAME;

/// `(shard, first node, end node)` of up to `size` nodes, in node order.
pub fn node_chunks(table: &TableReader, size: u64) -> impl Iterator<Item = (usize, u64, u64)> + '_ {
    (0..table.shard_count())
        .filter_map(|shard| {
            let (min, max) = table.shard_id_range(shard)?;
            (min <= max).then(|| (shard, max - min + 1))
        })
        .flat_map(move |(shard, nodes)| {
            (0..nodes)
                .step_by(size as usize)
                .map(move |start| (shard, start, nodes.min(start + size)))
        })
}

/// The `columns` of a chunk's rows stored with node data, in node order, strings and bytes
/// with 64-bit offsets.
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
    // A chunk's strings can pass the 2 GiB that 32-bit offsets hold.
    decoder.decode_batch_to_arrow_with(&rows, columns, true)
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

pub struct StoredTable {
    pub reader: Arc<TableReader>,
    pub columns: Vec<(String, IngestedType)>,
    pub create_time: Option<usize>,
}

pub fn stored_tables(
    index_dir: &Path,
    names: &HashSet<String>,
) -> Result<(HashMap<String, StoredTable>, TimeUnit)> {
    let reader =
        GraphIndexReader::open_subset(&index_dir.join(INDEX_METADATA_FILENAME), index_dir, names)?;
    let tables = names
        .iter()
        .map(|name| {
            let table = reader
                .table(name)
                .ok_or_else(|| anyhow!("index has no node table {name}"))?;
            let columns: Vec<(String, IngestedType)> = table
                .node_data_decoder()
                .ok_or_else(|| anyhow!("table {name} has no node data"))?
                .context()
                .columns
                .iter()
                .map(|c| (c.name.clone(), c.logical_type))
                .collect();
            let create_time = table
                .create_time_col()
                .and_then(|time| columns.iter().position(|(c, _)| c == time));
            let stored = StoredTable {
                reader: table,
                columns,
                create_time,
            };
            Ok((name.clone(), stored))
        })
        .collect::<Result<_>>()?;
    Ok((tables, reader.metadata().timestamp_unit))
}
