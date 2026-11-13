// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stored rows of an index table, decoded to Arrow in batches under a shared byte budget.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::Schema;
use diskgraph_core::time::TimeUnit;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_limits::total_memory_bytes;
use diskgraph_rowpack::{
    builder_value_bytes, make_arrow_builder, IngestedType, RowAccess, RowAccessHints,
    RowAccessStats, RowPackDecodeScratch,
};

use crate::source_memory::ByteBudget;

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

/// Decoded value bytes per batch. A chunk of narrow rows ends a batch first; wide or
/// dictionary-expanded rows end it here, well below Arrow's 2 GiB string offsets, with a
/// few batches per worker inside a budget of a third of memory.
const BATCH_BYTES: u64 = 128 << 20;

/// The decoded bytes the workers of one scan hold at once.
pub struct ScanBudget {
    budget: Arc<ByteBudget>,
    batch_bytes: u64,
}

impl ScanBudget {
    /// Ingest's Encode budget for the memory of this machine or cgroup.
    pub fn new() -> Result<Self> {
        Self::with_limits(
            crate::encode_memory_budget_bytes(true, total_memory_bytes()),
            BATCH_BYTES,
        )
    }

    pub fn with_limits(total_bytes: u64, batch_bytes: u64) -> Result<Self> {
        Ok(Self {
            budget: ByteBudget::with_oversized_limit(
                total_bytes,
                crate::ENCODE_MAX_OVERSIZED_CHUNK_BYTES,
            )?,
            batch_bytes,
        })
    }

    pub fn high_water_bytes(&self) -> u64 {
        self.budget.high_water_bytes()
    }
}

/// Visits the `columns` of a chunk's rows stored with node data, with each row's node, in node
/// order and in batches of about the budget's batch bytes; a larger row is a batch alone.
pub fn scan_chunk(
    table: &TableReader,
    (shard, start, end): (usize, u64, u64),
    columns: &[usize],
    budget: &ScanBudget,
    mut visit: impl FnMut(&RecordBatch, &[u64]) -> Result<()>,
) -> Result<()> {
    let shard = table
        .shard(shard)
        .ok_or_else(|| anyhow!("no shard {shard}"))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("table has no node data"))?;
    let context = &decoder.context().columns;
    let mask: Vec<bool> = (0..context.len()).map(|c| columns.contains(&c)).collect();
    let decoded: Vec<usize> = (0..context.len()).filter(|&c| mask[c]).collect();
    let projection: Vec<usize> = columns
        .iter()
        .map(|c| decoded.iter().position(|d| d == c).expect("decoded column"))
        .collect();
    let mut scratch = RowPackDecodeScratch::default();
    let mut node = start;
    loop {
        let mut permit = budget.budget.acquire_input_chunk(budget.batch_bytes)?;
        let (mut fields, mut builders) = (Vec::new(), Vec::new());
        for &c in &decoded {
            let (field, builder) = make_arrow_builder(
                &context[c].name,
                context[c].logical_type,
                1024,
                decoder.clock(),
            )?;
            fields.push(field);
            builders.push(builder);
        }
        let (mut nodes, mut bytes) = (Vec::new(), 0);
        while node < end && bytes < budget.batch_bytes {
            // Rows stored without node data, as oversized ones are, hold no values.
            if let Some(row) = shard.edges_for_node(node)?.node_data()? {
                if !row.is_empty() {
                    decoder.decode_row_to_builders(row, &mut builders, &mask, &mut scratch)?;
                    nodes.push(node);
                    bytes = builders
                        .iter()
                        .map(|b| builder_value_bytes(b.as_ref()))
                        .sum::<usize>() as u64;
                }
            }
            node += 1;
        }
        if bytes > budget.batch_bytes {
            drop(permit);
            permit = budget.budget.acquire_input_chunk(bytes)?;
        } else {
            permit.shrink_to(bytes);
        }
        let arrays = builders.iter_mut().map(|b| b.finish()).collect();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        visit(&batch.project(&projection)?, &nodes)?;
        drop(permit);
        if node >= end {
            return Ok(());
        }
    }
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
