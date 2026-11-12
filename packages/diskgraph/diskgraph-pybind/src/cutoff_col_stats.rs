// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use arrow::compute::filter_record_batch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, Int64Array, TimestampSecondArray};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::column_scan::{at_cutoff, create_times, decode_chunk, node_chunks};
use diskgraph_ingest::cutoff_col_stats::CutoffColStatsRequest;
use diskgraph_ingest::graph_config::{read_persisted_config, ColumnSpec, IngestedType};
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
        let parts = node_chunks(&table_reader)
            .into_par_iter()
            .map(|chunk| {
                let batch = decode_chunk(&table_reader, chunk, &decoded)?;
                let times = create_time.map(|_| create_times(batch.column(requested.len()), clock));
                let mut sinks = table.sinks()?;
                for (stats, cutoff) in sinks.stats.iter_mut().zip(&request.cutoffs) {
                    let keep = at_cutoff(times.as_deref(), batch.num_rows(), *cutoff);
                    let rows = filter_record_batch(&batch, &keep)?;
                    for (position, &col_idx) in requested.iter().enumerate() {
                        // Stored values are observed by their type, as ingest observes them.
                        let column = ColumnSpec {
                            semantic_type: None,
                            ..spec.columns[col_idx].clone()
                        };
                        let (values, column) = match column.ingested_dtype {
                            // Ingest observes a u64's bits as an i64.
                            IngestedType::UInt64 => (
                                Arc::new(Int64Array::from_iter(
                                    rows.column(position)
                                        .as_primitive::<UInt64Type>()
                                        .iter()
                                        .map(|value| value.map(|value| value as i64)),
                                )) as ArrayRef,
                                ColumnSpec {
                                    ingested_dtype: IngestedType::Int64,
                                    ..column
                                },
                            ),
                            _ => (rows.column(position).clone(), column),
                        };
                        drive_from_arrow_column(col_idx, &values, &column, stats, None, clock)?;
                    }
                }
                Ok(sinks)
            })
            .collect::<Result<Vec<_>>>()?;
        for sinks in parts {
            table.add(sinks);
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

/// Epoch `seconds` in the index clock `unit`, rounded as ingest rounds a row's create time.
#[pyfunction]
pub fn cutoff_ticks(seconds: i64, unit: &str) -> PyResult<i64> {
    let clock = crate::tt_builder::parse_index_clock(unit)?;
    let array: ArrayRef = Arc::new(TimestampSecondArray::from(vec![seconds]));
    create_times(&array, clock)[0].ok_or_else(|| {
        DiskgraphConfigError::new_err(format!("cutoff {seconds} s is outside the {unit} clock"))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cutoff_col_stats, m)?)?;
    m.add_function(wrap_pyfunction!(cutoff_ticks, m)?)
}
