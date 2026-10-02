// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DiskgraphConfigError, DiskgraphSystemError};
use pyo3::prelude::*;
use std::path::PathBuf;

fn map_index_metadata_err(err: diskgraph_ingest::index_metadata::IndexMetadataError) -> PyErr {
    match err {
        diskgraph_ingest::index_metadata::IndexMetadataError::InvalidPayload(err) => {
            DiskgraphConfigError::new_err(err.to_string())
        }
        diskgraph_ingest::index_metadata::IndexMetadataError::Io(err) => {
            DiskgraphSystemError::new_err(err.to_string())
        }
    }
}

#[pyfunction]
pub fn read_index_metadata(py: Python<'_>, index_path: &str) -> PyResult<String> {
    let index_path = PathBuf::from(index_path);
    py.detach(|| diskgraph_ingest::index_metadata::read_index_metadata(&index_path))
        .map_err(map_index_metadata_err)
}
