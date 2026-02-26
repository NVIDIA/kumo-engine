//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` that returns a
//! `TTPlanIterator` yielding `pyarrow.RecordBatch` objects one at a time.
//!
//! ```python
//! from diskgraph._c import execute_tt_plan
//!
//! for batch in execute_tt_plan(plan_json, "/path/to/index"):
//!     # batch is a pyarrow.RecordBatch
//!     ...
//! ```

use std::path::Path;
use std::thread::JoinHandle;

use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
use crossbeam_channel::Receiver;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::Py;

use diskgraph_core::tt_builder::DiskGraphTTPlan;

/// Python iterator that yields `pyarrow.RecordBatch` objects from the
/// TT-gen executor. Batches are streamed via a crossbeam channel from
/// background rayon workers.
#[pyclass]
pub struct TTPlanIterator {
    receiver: Option<Receiver<RecordBatch>>,
    handle: Option<JoinHandle<anyhow::Result<u64>>>,
}

#[pymethods]
impl TTPlanIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let receiver = match &self.receiver {
            Some(r) => r.clone(),
            None => return Err(PyStopIteration::new_err(())),
        };

        // Release the GIL while blocking on recv so rayon threads can run.
        let result = py.detach(|| receiver.recv());

        match result {
            Ok(batch) => batch.to_pyarrow(py).map(|b| b.unbind()),
            Err(_) => {
                // Channel closed — check if the background thread errored.
                self.receiver = None;
                if let Some(handle) = self.handle.take() {
                    match handle.join() {
                        Ok(Ok(_)) => {}
                        Ok(Err(e)) => {
                            return Err(PyRuntimeError::new_err(format!(
                                "TT-gen executor failed: {e}"
                            )));
                        }
                        Err(_) => {
                            return Err(PyRuntimeError::new_err(
                                "TT-gen executor thread panicked",
                            ));
                        }
                    }
                }
                Err(PyStopIteration::new_err(()))
            }
        }
    }
}

/// Execute a TT-gen plan against a DiskGraph index, returning a streaming
/// iterator of `pyarrow.RecordBatch` objects.
///
/// Args:
///     plan_json: JSON-serialised `DiskGraphTTPlan` (output of `DiskGraphTTBuilder.build()`).
///     index_path: Path to the DiskGraph index directory.
///
/// Returns:
///     A `TTPlanIterator` that yields `pyarrow.RecordBatch` objects.
#[pyfunction]
pub fn execute_tt_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
) -> PyResult<TTPlanIterator> {
    let plan: DiskGraphTTPlan = serde_json::from_str(plan_json)
        .map_err(|e| PyValueError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();

    // Start executor — opens index and spawns rayon workers on main thread
    // (before returning to Python), so errors are reported immediately.
    let (receiver, handle) = py
        .detach(|| {
            diskgraph_train_table::ttgen_executor::execute(&plan, Path::new(&idx))
        })
        .map_err(|e| PyValueError::new_err(format!("execute failed: {e}")))?;

    Ok(TTPlanIterator {
        receiver: Some(receiver),
        handle: Some(handle),
    })
}
