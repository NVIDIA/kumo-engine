// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, ensure, Result};
use arrow::compute::filter_record_batch;
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::column_scan::{at_cutoff, create_times, node_chunks, scan_chunk, ScanBudget};
use diskgraph_ingest::cutoff_col_stats::CutoffColStatsRequest;
use diskgraph_ingest::graph_config::{read_persisted_config, ColumnSpec};
use diskgraph_ingest::index_metadata::INDEX_METADATA_FILENAME;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bp_col_stats::drive_from_arrow_column;
use crate::{map_anyhow_err, DiskgraphConfigError};

/// Recomputes `request` from an index's stored rows.
fn from_index(index_dir: &Path, request: &CutoffColStatsRequest) -> Result<()> {
    let config = read_persisted_config(index_dir)?;
    request.validate(&config.tables)?;
    let names: HashSet<String> = request.columns.keys().cloned().collect();
    let reader =
        GraphIndexReader::open_subset(&index_dir.join(INDEX_METADATA_FILENAME), index_dir, &names)?;
    let clock = reader.metadata().timestamp_unit;
    let budget = ScanBudget::new()?;
    let mut tables = Vec::new();
    for spec in &config.tables {
        let (Some(mut table), Some(wanted)) =
            (request.table(spec), request.columns.get(&spec.name))
        else {
            continue;
        };
        let table_reader = reader
            .table(&spec.name)
            .ok_or_else(|| anyhow!("index has no node table {}", spec.name))?;
        let stored = &table_reader
            .node_data_decoder()
            .ok_or_else(|| anyhow!("table {} has no node data", spec.name))?
            .context()
            .columns;
        ensure!(
            stored
                .iter()
                .map(|c| &c.name)
                .eq(spec.columns.iter().map(|c| &c.name)),
            "table {} stores other columns than its config lists",
            spec.name
        );
        let requested: Vec<usize> = (0..spec.columns.len())
            .filter(|&idx| wanted.contains_key(&spec.columns[idx].name))
            .collect();
        let create_time = spec
            .create_time
            .as_ref()
            .and_then(|name| spec.columns.iter().position(|c| &c.name == name));
        let decoded: Vec<usize> = requested.iter().copied().chain(create_time).collect();
        // Windows of chunks keep a few sinks per thread alive and merge in chunk order.
        for window in node_chunks(&table_reader).chunks(4 * rayon::current_num_threads()) {
            let parts = window
                .par_iter()
                .map(|&chunk| {
                    let mut sinks = table.sinks()?;
                    scan_chunk(&table_reader, chunk, &decoded, &budget, |batch, _| {
                        let times =
                            create_time.map(|_| create_times(batch.column(requested.len()), clock));
                        for (stats, cutoff) in sinks.stats.iter_mut().zip(&request.cutoffs) {
                            let keep = at_cutoff(times.as_deref(), batch.num_rows(), *cutoff);
                            let rows = filter_record_batch(batch, &keep)?;
                            for (position, &col_idx) in requested.iter().enumerate() {
                                // Stored values are observed by their type, as ingest observes them.
                                let column = ColumnSpec {
                                    semantic_type: None,
                                    ..spec.columns[col_idx].clone()
                                };
                                let values = rows.column(position);
                                drive_from_arrow_column(
                                    col_idx, values, &column, stats, None, clock,
                                )?;
                            }
                        }
                        Ok(())
                    })?;
                    Ok(sinks)
                })
                .collect::<Result<Vec<_>>>()?;
            for sinks in parts {
                table.add(sinks);
            }
        }
        tables.push(table);
    }
    request.publish(index_dir, clock, tables)
}

/// Recomputes column statistics at cutoffs from an index's stored rows, into `col_stats/<dir>/`.
#[pyfunction]
#[pyo3(signature = (index_dir, request_json, num_threads=None))]
pub fn cutoff_col_stats(
    py: Python<'_>,
    index_dir: &str,
    request_json: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    let request: CutoffColStatsRequest = serde_json::from_str(request_json)
        .map_err(|e| DiskgraphConfigError::new_err(format!("invalid request_json: {e}")))?;
    py.detach(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(0))
            .build()?;
        pool.install(|| from_index(Path::new(index_dir), &request))
    })
    .map_err(map_anyhow_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cutoff_col_stats, m)?)
}
