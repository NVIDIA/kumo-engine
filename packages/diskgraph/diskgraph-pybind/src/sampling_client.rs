// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-aware native Flight client handles and Arrow C Data Interface export.
use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
use diskgraph_gateway::{resolve_reference, ArrowValue, SamplingClient, SamplingStream};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use serde_json::Value;

fn client_error(py: Python<'_>, error: anyhow::Error) -> PyErr {
    if let Some(status) = error.downcast_ref::<tonic::Status>() {
        let (module, name) = match status.code() {
            tonic::Code::Cancelled => ("pyarrow.flight", "FlightCancelledError"),
            tonic::Code::DeadlineExceeded => ("pyarrow.flight", "FlightTimedOutError"),
            tonic::Code::Unavailable => ("pyarrow.flight", "FlightUnavailableError"),
            tonic::Code::InvalidArgument
            | tonic::Code::FailedPrecondition
            | tonic::Code::ResourceExhausted => ("pyarrow", "ArrowInvalid"),
            _ => ("pyarrow.flight", "FlightServerError"),
        };
        return match py
            .import(module)
            .and_then(|module| module.getattr(name))
            .and_then(|class| class.call1((status.message(),)))
        {
            Ok(value) => PyErr::from_value(value),
            Err(error) => error,
        };
    }
    PyValueError::new_err(format!("{error:#}"))
}

fn export_layout<'py>(
    py: Python<'py>,
    value: &Value,
    components: &[RecordBatch],
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(value) = resolve_reference(value, components)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))?
    {
        return match value {
            ArrowValue::Array(array) => array.to_data().to_pyarrow(py),
            ArrowValue::RecordBatch(batch) => batch.to_pyarrow(py),
        };
    }
    match value {
        Value::Object(values) => {
            let result = PyDict::new(py);
            for (key, value) in values {
                result.set_item(key, export_layout(py, value, components)?)?;
            }
            Ok(result.into_any())
        }
        Value::Array(values) => {
            let result = PyList::empty(py);
            for value in values {
                result.append(export_layout(py, value, components)?)?;
            }
            Ok(result.into_any())
        }
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any()),
        Value::String(value) => Ok(value.into_pyobject(py)?.into_any()),
        Value::Number(value) => {
            if let Some(value) = value.as_u64() {
                return Ok(value.into_pyobject(py)?.into_any());
            }
            if let Some(value) = value.as_i64() {
                return Ok(value.into_pyobject(py)?.into_any());
            }
            Ok(value.as_f64().unwrap().into_pyobject(py)?.into_any())
        }
    }
}

/// Reuse a native Flight connection across epoch shards with different Arrow schemas.
///
/// Args:
///     endpoint: Sampling server's grpc:// or http:// URI.
/// Raises:
///     ValueError: If the endpoint or native runtime cannot be initialized.
#[pyclass(name = "SamplingClient", module = "diskgraph._c")]
pub(crate) struct PySamplingClient {
    client: SamplingClient,
}
#[pymethods]
impl PySamplingClient {
    #[new]
    fn new(py: Python<'_>, endpoint: &str) -> PyResult<Self> {
        py.detach(|| SamplingClient::new(endpoint))
            .map(|client| Self { client })
            .map_err(|error| client_error(py, error))
    }
    /// Open a JSON DoGet epoch ticket; return a native component stream.
    ///
    /// Args:
    ///     request: UTF-8 JSON bytes describing the epoch and shard.
    /// Returns:
    ///     SamplingStream with native sample and batch counts.
    /// Raises:
    ///     ValueError: If stream schema metadata is invalid.
    ///     pyarrow.flight.FlightError: If the server rejects or interrupts the RPC.
    fn open(&self, py: Python<'_>, request: Vec<u8>) -> PyResult<PySamplingStream> {
        py.detach(|| self.client.open(request))
            .map(|stream| PySamplingStream { stream })
            .map_err(|error| client_error(py, error))
    }
    /// Cancel all streams and release the cached native connection.
    fn close(&self, py: Python<'_>) {
        py.detach(|| self.client.close());
    }
}

/// One cancel-safe epoch shard assembled from native Arrow component batches.
#[pyclass(name = "SamplingStream", module = "diskgraph._c")]
pub(crate) struct PySamplingStream {
    stream: SamplingStream,
}
#[pymethods]
impl PySamplingStream {
    /// Number of selected seed examples in this shard.
    #[getter]
    fn num_samples(&self) -> u64 {
        self.stream.num_samples
    }
    /// Number of logical minibatches advertised by the native iterator.
    #[getter]
    fn num_batches(&self) -> u64 {
        self.stream.num_batches
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let result = py
            .detach(|| self.stream.next_batch())
            .map_err(|error| client_error(py, error))?;
        let Some(batch) = result else {
            return Ok(None);
        };
        match export_layout(py, &batch.layout, &batch.components) {
            Ok(value) => Ok(Some(value)),
            Err(error) => {
                self.stream.close();
                Err(error)
            }
        }
    }
    /// Cancel this RPC and wake a concurrent blocked next() call.
    fn close(&self) {
        self.stream.close();
    }
}
