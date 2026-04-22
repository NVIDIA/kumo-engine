use crate::{DiskgraphConfigError, DiskgraphSystemError};
use pyo3::prelude::*;
use std::path::PathBuf;

fn map_success_marker_err(err: diskgraph_ingest::success_marker::SuccessMarkerError) -> PyErr {
    match err {
        diskgraph_ingest::success_marker::SuccessMarkerError::InvalidPayload(err) => {
            DiskgraphConfigError::new_err(err.to_string())
        }
        diskgraph_ingest::success_marker::SuccessMarkerError::Io(err) => {
            DiskgraphSystemError::new_err(err.to_string())
        }
    }
}

#[pyfunction]
pub fn read_success_marker(py: Python<'_>, output_dir: &str) -> PyResult<Option<String>> {
    let output_dir = PathBuf::from(output_dir);
    py.detach(|| diskgraph_ingest::success_marker::read_success_marker(&output_dir))
        .map_err(map_success_marker_err)
}

#[pyfunction]
pub fn write_success_marker(py: Python<'_>, output_dir: &str, payload_json: &str) -> PyResult<()> {
    let output_dir = PathBuf::from(output_dir);
    py.detach(|| diskgraph_ingest::success_marker::write_success_marker(&output_dir, payload_json))
        .map_err(map_success_marker_err)
}

#[pyfunction]
pub fn touch_success_marker(py: Python<'_>, output_dir: &str) -> PyResult<bool> {
    let output_dir = PathBuf::from(output_dir);
    py.detach(|| diskgraph_ingest::success_marker::touch_success_marker(&output_dir))
        .map_err(map_success_marker_err)
}
