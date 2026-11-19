// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The canonical node IDs of a table, optionally only rows passing a filter.

use std::sync::Arc;

use diskgraph_core::tt_builder::RowFilter;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_rowpack::RowPackDecodeScratch;
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::errors::{TTGenError, TTGenResult};
use crate::ttgen_filter::eval_row_filter_on_row;
use crate::ttgen_resolve::{resolve_table_row_filter, ResolvedRowFilter};

const CHUNK_NODES: u64 = 1 << 16;

/// A table's canonical rows (one per primary key) that pass a filter, one
/// bit per node.
pub struct RhsCatalog {
    chunks: Vec<(u64, Vec<u64>)>,
}

impl RhsCatalog {
    pub fn len(&self) -> usize {
        self.chunks
            .iter()
            .flat_map(|(_, words)| words)
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The node IDs, ascending.
    pub fn node_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.chunks.iter().flat_map(|(start, words)| {
            words.iter().enumerate().flat_map(move |(index, &word)| {
                let base = start + 64 * index as u64;
                std::iter::successors((word != 0).then_some(word), |bits| {
                    let rest = bits & (bits - 1);
                    (rest != 0).then_some(rest)
                })
                .map(move |bits| base + u64::from(bits.trailing_zeros()))
            })
        })
    }
}

/// The canonical rows of `node_type` that pass `filter`, evaluated on each
/// row's own columns. Chunks of the node range run on `pool`.
pub fn rhs_catalog(
    reader: &GraphIndexReader,
    node_type: &str,
    filter: Option<&RowFilter>,
    pool: &ThreadPool,
) -> TTGenResult<RhsCatalog> {
    let table = reader
        .table(node_type)
        .ok_or_else(|| TTGenError::InvalidPlan(format!("unknown node type: {node_type}")))?;
    let filter = filter
        .map(|filter| resolve_table_row_filter(filter, reader, &table))
        .transpose()?;
    let ranges: Vec<(u64, u64)> = (0..table.shard_count())
        .filter_map(|shard| table.shard_id_range(shard))
        .filter(|(min_id, max_id)| max_id >= min_id)
        .flat_map(|(min_id, max_id)| {
            (min_id..=max_id)
                .step_by(CHUNK_NODES as usize)
                .map(move |start| (start, (start + CHUNK_NODES - 1).min(max_id)))
        })
        .collect();
    let chunks = pool.install(|| {
        ranges
            .par_iter()
            .map(|&(start, end)| catalog_chunk(&table, filter.as_ref(), start, end))
            .collect::<TTGenResult<_>>()
    })?;
    Ok(RhsCatalog { chunks })
}

fn catalog_chunk(
    table: &Arc<TableReader>,
    filter: Option<&ResolvedRowFilter>,
    start: u64,
    end: u64,
) -> TTGenResult<(u64, Vec<u64>)> {
    let mut scratch = RowPackDecodeScratch::default();
    let mut words = vec![0u64; ((end - start) / 64 + 1) as usize];
    for node_id in start..=end {
        let row = table
            .edges_for_node(node_id)
            .map_err(data_error)?
            .node_data()
            .map_err(data_error)?
            .ok_or_else(|| {
                TTGenError::SchemaError(format!(
                    "{} node {node_id} has no feature data",
                    table.node_type()
                ))
            })?;
        if table
            .is_canonical_row(node_id, row, &mut scratch)
            .map_err(data_error)?
            && filter
                .is_none_or(|filter| eval_row_filter_on_row(filter, node_id, table, Some(row), &[]))
        {
            let offset = node_id - start;
            words[(offset / 64) as usize] |= 1 << (offset % 64);
        }
    }
    Ok((start, words))
}

fn data_error(error: impl std::fmt::Display) -> TTGenError {
    TTGenError::SchemaError(format!("{error:#}"))
}
