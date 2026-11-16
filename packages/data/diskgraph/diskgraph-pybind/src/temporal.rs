// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The temporal surface Python calls instead of re-implementing it.
//!
//! A format string is an opaque token on the Python side: it is produced here,
//! stored, and handed back here, never parsed by Python.

use arrow::array::{make_array, ArrayData};
use arrow::pyarrow::FromPyArrow;
use diskgraph_ingest::config_converter::inference;
use diskgraph_ingest::config_converter::map_timestamp_unit;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// One epoch value per row, plus the format and unit tokens that produced them.
type ResolvedColumn = (Vec<Option<i64>>, Option<String>, Option<String>);

/// Detect (unless declared) and apply one pyarrow column's temporal hint in a
/// single call, so the resolved format never round-trips through Python.
/// Returns epochs in `timestamp_unit` ticks — epoch days when `as_days` —
/// aligned with the input, `None` for a null or unparseable value, plus the
/// hint that was resolved.
#[pyfunction]
#[pyo3(signature = (
    values,
    declared_format=None,
    declared_unit=None,
    as_days=false,
    timestamp_unit="seconds",
))]
pub fn resolve_temporal(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    declared_format: Option<String>,
    declared_unit: Option<String>,
    as_days: bool,
    timestamp_unit: &str,
) -> PyResult<ResolvedColumn> {
    let array = make_array(ArrayData::from_pyarrow_bound(values)?);
    let unit = declared_unit.as_deref().and_then(map_timestamp_unit);
    let clock = crate::tt_builder::parse_index_clock(timestamp_unit)?;
    let resolved = py.detach(|| {
        inference::resolve_temporal(&array, declared_format.as_deref(), unit, as_days, clock)
    });
    Ok((
        resolved.epochs,
        resolved.format,
        resolved.unit.map(|u| u.token().to_string()),
    ))
}

#[pyfunction]
pub fn detect_multicat_delimiter(rows: Vec<String>) -> Option<&'static str> {
    inference::detect_multicat_delimiter(rows.iter().map(String::as_str))
}

#[pyfunction]
pub fn multicat_delimiter_is_proven(rows: Vec<String>, delimiter: &str) -> bool {
    inference::multicat_delimiter_is_proven(rows.iter().map(String::as_str), delimiter)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resolve_temporal, m)?)?;
    m.add_function(wrap_pyfunction!(detect_multicat_delimiter, m)?)?;
    m.add_function(wrap_pyfunction!(multicat_delimiter_is_proven, m)?)?;
    m.add(
        "TIMESTAMP_FORMATS",
        PyList::new(m.py(), diskgraph_rowpack::TIMESTAMP_FORMATS)?,
    )?;
    m.add(
        "MULTICAT_DELIMITERS",
        PyList::new(m.py(), inference::MULTICAT_DELIMITER_CANDIDATES)?,
    )
}
