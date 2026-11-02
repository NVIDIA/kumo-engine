// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-aware native Flight client handles and Arrow C Data Interface export.
use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
use diskgraph_gateway::{
    resolve_reference, ArrowValue, SamplingClient, SamplingStream, TaskOptions, TaskRequest,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyList},
};
use serde_json::Value;
use std::collections::HashMap;

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

/// Immutable native epoch request; the extension owns Flight ticket serialization.
///
/// Args:
///     extension_ref: Worker-relative training-table path below the graph.
///     stage_table: Seed table within the training-table extension.
///     sampling_plan_json: Existing DiskGraph sampling-plan JSON.
///     batch_size: Maximum seeds per logical minibatch.
///     engine_seed: Optional unsigned 64-bit engine seed; booleans are rejected.
///     num_threads: Positive native sampling thread count.
///     num_workers: Number of epoch shards.
///     worker_id: Zero-based shard index.
///     shuffle: Shuffle selected seed rows.
///     rnd_seed: Optional shuffle seed.
///     epoch: Epoch number for deterministic sampling.
///     drop_last: Drop the final partial minibatch.
///     fused: Optional native graph-fusion setting.
///     swap_edge_direction: Reverse emitted edge directions.
///     neg_rhs_node_type: Node type for generated negatives.
///     neg_amount_m: Number of negatives per positive seed.
///     neg_rnd_seed: Optional negative-sampling seed.
///     replace: Sample neighbors with replacement.
///     disjoint: Keep per-seed sampled graphs disjoint.
///     adaptive: Enable native adaptive sampling.
///     entity_node_type: Seed entity node type.
///     target_node_type: Link-prediction target node type.
///     rhs_seed_node_type: Optional node type for ordered RHS catalog scans.
///     rhs_seed_time: Fixed RHS catalog-scan timestamp.
///     group_by_timestamp: Group seed rows by timestamp.
///     weighted_seed_sampling_json: Native weighted-seed bucket settings.
///     selected_stage_node_ids: Optional explicit unsigned stage-row IDs.
///     use_seed_timestamps: Use seed timestamps as sampling anchors.
///     pseudo_seed_time: Optional static link-prediction cutoff.
///     disable_pseudo_time: Disable the static cutoff.
///     projections: Node types mapped to selected feature columns.
/// Raises:
///     TypeError: If an argument has an incompatible type or name.
///     OverflowError: If an integer does not fit its native type.
///     ValueError: If batch, thread or shard settings are invalid.
#[pyclass(name = "SamplingRequest", module = "diskgraph._c", frozen)]
pub(crate) struct PySamplingRequest {
    request: TaskRequest,
}
#[pymethods]
impl PySamplingRequest {
    #[new]
    #[pyo3(signature = (extension_ref, *,
        stage_table,
        sampling_plan_json,
        batch_size,
        num_workers=1,
        worker_id=0,
        shuffle=false,
        rnd_seed=None,
        epoch=0,
        drop_last=false,
        fused=None,
        swap_edge_direction=true,
        neg_rhs_node_type=None,
        neg_amount_m=None,
        neg_rnd_seed=None,
        replace=false,
        disjoint=true,
        adaptive=false,
        entity_node_type=None,
        target_node_type=None,
        rhs_seed_node_type=None,
        rhs_seed_time=None,
        group_by_timestamp=false,
        weighted_seed_sampling_json=None,
        selected_stage_node_ids=None,
        use_seed_timestamps=true,
        pseudo_seed_time=None,
        disable_pseudo_time=false,
        projections=None,
        engine_seed=None, num_threads=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        extension_ref: String,
        stage_table: String,
        sampling_plan_json: String,
        batch_size: usize,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        drop_last: bool,
        fused: Option<bool>,
        swap_edge_direction: bool,
        neg_rhs_node_type: Option<String>,
        neg_amount_m: Option<usize>,
        neg_rnd_seed: Option<u64>,
        replace: bool,
        disjoint: bool,
        adaptive: bool,
        entity_node_type: Option<String>,
        target_node_type: Option<String>,
        rhs_seed_node_type: Option<String>,
        rhs_seed_time: Option<i64>,
        group_by_timestamp: bool,
        weighted_seed_sampling_json: Option<String>,
        selected_stage_node_ids: Option<Vec<u64>>,
        use_seed_timestamps: bool,
        pseudo_seed_time: Option<i64>,
        disable_pseudo_time: bool,
        projections: Option<HashMap<String, Vec<String>>>,
        engine_seed: Option<Bound<'_, PyAny>>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let engine_seed = engine_seed
            .map(|value| {
                if value.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err(
                        "engine_seed must be an unsigned 64-bit integer",
                    ));
                }
                value.extract::<u64>()
            })
            .transpose()?;
        let options = TaskOptions {
            stage_table,
            sampling_plan_json,
            num_workers,
            worker_id,
            shuffle,
            rnd_seed,
            epoch,
            batch_size,
            drop_last,
            fused,
            swap_edge_direction,
            neg_rhs_node_type,
            neg_amount_m,
            neg_rnd_seed,
            replace,
            disjoint,
            adaptive,
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            rhs_seed_time,
            group_by_timestamp,
            weighted_seed_sampling_json,
            selected_stage_node_ids,
            use_seed_timestamps,
            pseudo_seed_time,
            disable_pseudo_time,
            projections,
        };
        let request = TaskRequest::new(extension_ref, options, engine_seed, num_threads)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { request })
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
    /// Serialize a native request into a DoGet ticket and open its stream.
    ///
    /// Args:
    ///     request: SamplingRequest describing the epoch and shard.
    /// Returns:
    ///     SamplingStream with native sample and batch counts.
    /// Raises:
    ///     ValueError: If stream schema metadata is invalid.
    ///     pyarrow.flight.FlightError: If the server rejects or interrupts the RPC.
    fn open(&self, py: Python<'_>, request: &PySamplingRequest) -> PyResult<PySamplingStream> {
        py.detach(|| self.client.open(&request.request))
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
