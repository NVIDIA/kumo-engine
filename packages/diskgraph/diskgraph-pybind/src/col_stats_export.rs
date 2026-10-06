// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError};
use pyo3::prelude::*;
use std::path::PathBuf;

fn map_col_stats_err(err: diskgraph_ingest::col_stats_export::ColStatsReadError) -> PyErr {
    use diskgraph_ingest::col_stats_export::ColStatsReadError::*;
    match err {
        InvalidVariant(value) => DiskgraphConfigError::new_err(format!(
            "invalid col stats variant: {value:?} (expected \"full\" or \"min\")"
        )),
        Io(path, err) => {
            DiskgraphSystemError::new_err(format!("IO error reading {path}: {err}"))
        }
        InvalidJson(path, err) => {
            DiskgraphDataError::new_err(format!("invalid JSON in {path}: {err}"))
        }
    }
}

/// Return the contents of
/// `{index_path}/kumo_col_stats/{variant}/{table_name}.json` as a JSON
/// string, or None if the file doesn't exist.
///
/// `variant` is either `"full"` or `"min"`; any other value raises
/// `DiskgraphConfigError`.
#[pyfunction]
pub fn read_col_stats(
    py: Python<'_>,
    index_path: &str,
    table_name: &str,
    variant: &str,
) -> PyResult<Option<String>> {
    let variant = diskgraph_ingest::col_stats_export::ColStatsVariant::from_str_input(variant)
        .map_err(map_col_stats_err)?;
    let index_path = PathBuf::from(index_path);
    let table_name = table_name.to_string();
    py.detach(|| {
        diskgraph_ingest::col_stats_export::read_col_stats(&index_path, &table_name, variant)
    })
    .map_err(map_col_stats_err)
}

/// Return the contents of `{index_path}/edge_stats.json` as a JSON
/// string, or None if the file doesn't exist.
#[pyfunction]
pub fn read_edge_stats(py: Python<'_>, index_path: &str) -> PyResult<Option<String>> {
    let index_path = PathBuf::from(index_path);
    py.detach(|| diskgraph_ingest::col_stats_export::read_edge_stats(&index_path))
        .map_err(map_col_stats_err)
}
