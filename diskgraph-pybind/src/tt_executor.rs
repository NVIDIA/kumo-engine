//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` so Python can drive
//! training-table generation end-to-end:
//!
//! ```python
//! from diskgraph._c import execute_tt_plan
//!
//! shard_paths = execute_tt_plan(plan_json, "/path/to/index", "/path/to/output")
//! ```

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use diskgraph_core::tt_builder::DiskGraphTTPlan;

/// Execute a TT-gen plan against a DiskGraph index, producing Parquet shards.
///
/// Args:
///     plan_json: JSON-serialised `DiskGraphTTPlan` (output of `DiskGraphTTBuilder.build()`).
///     index_path: Path to the DiskGraph index directory.
///     output_dir: Directory where output Parquet shards will be written.
///
/// Returns:
///     List of output shard file paths (one per thread).
#[pyfunction]
pub fn execute_tt_plan(
    plan_json: &str,
    index_path: &str,
    output_dir: &str,
) -> PyResult<Vec<String>> {
    let plan: DiskGraphTTPlan = serde_json::from_str(plan_json)
        .map_err(|e| PyValueError::new_err(format!("invalid plan JSON: {e}")))?;

    let shard_paths = diskgraph_train_table::ttgen_executor::execute(
        &plan,
        Path::new(index_path),
        Path::new(output_dir),
    )
    .map_err(|e| PyValueError::new_err(format!("execute failed: {e}")))?;

    Ok(shard_paths
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}
