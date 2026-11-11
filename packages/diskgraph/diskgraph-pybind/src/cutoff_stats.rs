// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact column statistics at a cutoff; see `diskgraph_ingest::cutoff_stats`.

use std::path::Path;

use arrow::array::{make_array, ArrayData};
use arrow::pyarrow::FromPyArrow;
use diskgraph_ingest::cutoff_stats::{self, Stat};
use diskgraph_ingest::index_metadata::{index_id, read_dataset_metadata};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{map_anyhow_err, map_py_err, DiskgraphConfigError};

fn parse_stat(name: &str) -> PyResult<Stat> {
    Stat::parse(name)
        .ok_or_else(|| DiskgraphConfigError::new_err(format!("unknown column statistic {name:?}")))
}

fn json_loads(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

/// The id that names this index's column statistics.
#[pyfunction]
pub fn col_stats_index_id(py: Python<'_>, index_dir: &str) -> PyResult<String> {
    let index_dir = Path::new(index_dir);
    py.detach(|| index_id(index_dir, &read_dataset_metadata(index_dir)?))
        .map_err(map_anyhow_err)
}

/// Reuses or recomputes each `(table, column, stat)` file at `cutoff` and
/// returns `{"cutoff_id", "index_id", "unit", "files", "seconds"}`.
#[pyfunction]
#[pyo3(signature = (index_dir, cutoff, requests, num_threads=None))]
pub fn ensure_col_stats(
    py: Python<'_>,
    index_dir: &str,
    cutoff: Option<i64>,
    requests: Vec<[String; 3]>,
    num_threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let requests = requests
        .into_iter()
        .map(|[table, column, stat]| Ok((table, column, parse_stat(&stat)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let ensured = py
        .detach(|| {
            cutoff_stats::ensure_col_stats(Path::new(index_dir), cutoff, &requests, num_threads)
        })
        .map_err(map_anyhow_err)?;
    json_loads(py, &serde_json::to_string(&ensured).map_err(map_py_err)?)
}

/// `{stat: data}` of `values` over the rows whose `create_time` is at or
/// before `cutoff`, exactly as the statistics files hold it.
#[pyfunction]
#[pyo3(signature = (values, create_time, cutoff, stats, unit="seconds"))]
pub fn col_stats_from_arrow(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    create_time: Option<&Bound<'_, PyAny>>,
    cutoff: Option<i64>,
    stats: Vec<String>,
    unit: &str,
) -> PyResult<Py<PyAny>> {
    let values = make_array(ArrayData::from_pyarrow_bound(values)?);
    let create_time = create_time
        .map(|array| ArrayData::from_pyarrow_bound(array).map(make_array))
        .transpose()?;
    let stats = stats
        .iter()
        .map(|stat| parse_stat(stat))
        .collect::<PyResult<Vec<_>>>()?;
    let clock = crate::tt_builder::parse_index_clock(unit)?;
    let data = py
        .detach(|| {
            cutoff_stats::stats_from_arrow(&values, create_time.as_ref(), cutoff, &stats, clock)
        })
        .map_err(map_anyhow_err)?;
    let out = PyDict::new(py);
    for (stat, json) in data {
        out.set_item(stat.name(), json_loads(py, &json)?)?;
    }
    Ok(out.into_any().unbind())
}

/// Epoch `seconds` in ticks of the index clock `unit`, converted as ingest
/// converts a row's `timestamp[s]` create time.
#[pyfunction]
pub fn cutoff_ticks(seconds: i64, unit: &str) -> PyResult<i64> {
    let clock = crate::tt_builder::parse_index_clock(unit)?;
    cutoff_stats::cutoff_ticks(seconds, clock).map_err(map_anyhow_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cutoff_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(col_stats_index_id, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_col_stats, m)?)?;
    m.add_function(wrap_pyfunction!(col_stats_from_arrow, m)?)
}
