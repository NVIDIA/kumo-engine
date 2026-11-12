// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Result};
use arrow::array::Array;
use arrow::compute::filter_record_batch;
use arrow::pyarrow::ToPyArrow;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, Int64Array, TimestampSecondArray};
use diskgraph_ingest::column_scan::{
    self, at_cutoff, create_times, decode_chunk, node_chunks, stored_tables,
};
use diskgraph_ingest::cutoff_col_stats::CutoffColStatsRequest;
use diskgraph_ingest::graph_config::{
    read_persisted_config, ColumnSpec, IngestedType, SemanticType,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bp_col_stats::drive_from_arrow_column;
use crate::{map_anyhow_err, DiskgraphConfigError};

const CHUNK_NODES: u64 = 1 << 16;

/// Recomputes `request` from an index's stored rows.
fn from_index(index_dir: &Path, request: &CutoffColStatsRequest) -> Result<()> {
    let config = read_persisted_config(index_dir)?;
    request.validate(&config.tables)?;
    let names: HashSet<String> = request.columns.keys().cloned().collect();
    let (stored_tables, clock) = stored_tables(index_dir, &names)?;
    let mut tables = Vec::new();
    for spec in &config.tables {
        let Some(mut table) = request.table(spec) else {
            continue;
        };
        let stored = &stored_tables[&spec.name];
        ensure!(
            stored
                .columns
                .iter()
                .map(|(name, _)| name)
                .eq(spec.columns.iter().map(|c| &c.name)),
            "table {} stores other columns than its config lists",
            spec.name
        );
        let requested: Vec<usize> = (0..table.columns.len())
            .filter(|&idx| table.columns[idx].semantic_type != Some(SemanticType::Unsupported))
            .collect();
        let create_time = stored.create_time;
        let decoded: Vec<usize> = requested.iter().copied().chain(create_time).collect();
        let parts = node_chunks(&stored.reader, CHUNK_NODES)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| {
                let batch = decode_chunk(&stored.reader, chunk, &decoded)?;
                let times = create_time.map(|_| create_times(batch.column(requested.len()), clock));
                let mut sinks = table.sinks();
                for (stats, cutoff) in sinks.stats.iter_mut().zip(&request.cutoffs) {
                    let keep = at_cutoff(times.as_deref(), batch.num_rows(), *cutoff);
                    let rows = filter_record_batch(&batch, &keep)?;
                    for (position, &col_idx) in requested.iter().enumerate() {
                        let column = &table.columns[col_idx];
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
                                    ..column.clone()
                                },
                            ),
                            _ => (rows.column(position).clone(), column.clone()),
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

#[pyclass(module = "diskgraph._c")]
pub struct ColumnScan {
    scan: column_scan::ColumnScan,
}

#[pymethods]
impl ColumnScan {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        loop {
            match py.detach(|| self.scan.chunks.recv_timeout(Duration::from_millis(100))) {
                Ok(chunk) => {
                    let chunk = chunk.map_err(map_anyhow_err)?;
                    return Ok(Some(chunk.to_data().to_pyarrow(py)?.unbind()));
                }
                Err(err) if err.is_timeout() => py.check_signals()?,
                Err(_) => return Ok(None),
            }
        }
    }
}

/// A stored string column's non-null values over the rows counted at `cutoff`, as `large_string`.
#[pyfunction]
#[pyo3(signature = (index_dir, table, column, cutoff, *, chunk_rows=65536, num_threads=None))]
pub fn scan_column(
    py: Python<'_>,
    index_dir: &str,
    table: &str,
    column: &str,
    cutoff: Option<i64>,
    chunk_rows: usize,
    num_threads: Option<usize>,
) -> PyResult<ColumnScan> {
    let scan = py
        .detach(|| {
            column_scan::scan_column(
                Path::new(index_dir),
                table,
                column,
                cutoff,
                chunk_rows,
                num_threads,
            )
        })
        .map_err(map_anyhow_err)?;
    Ok(ColumnScan { scan })
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
    m.add_function(wrap_pyfunction!(cutoff_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(scan_column, m)?)?;
    m.add_class::<ColumnScan>()
}
