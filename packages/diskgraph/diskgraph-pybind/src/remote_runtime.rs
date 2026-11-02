// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python handles for native replica orchestration and epoch lifetime.
use crate::batch::export_batch;
use crate::sampling_client::{client_error, PySamplingRequest};
use diskgraph_gateway::{RemoteBatchIterator, RemoteRuntimeIndex};
use pyo3::prelude::*;
use std::{path::Path, sync::Arc};

/// Shared remote connections; construct RuntimeIndex(sampling_endpoints=...) for manifest normalization.
///
/// Args:
///     endpoints: Nonempty list of distinct Flight replica URIs.
/// Raises:
///     ValueError: If replica identities are invalid.
#[pyclass(name = "RemoteRuntimeIndex", module = "diskgraph._c", frozen)]
pub(crate) struct PyRemoteRuntimeIndex {
    index: RemoteRuntimeIndex,
}
#[pymethods]
impl PyRemoteRuntimeIndex {
    #[new]
    fn new(py: Python<'_>, endpoints: Vec<String>) -> PyResult<Self> {
        RemoteRuntimeIndex::new(endpoints)
            .map(|index| Self { index })
            .map_err(|error| client_error(py, error))
    }
    /// Open native epoch shards and start one bounded prefetch per replica.
    ///
    /// Args:
    ///     request: Immutable SamplingRequest with native iterator settings.
    ///     graph_root: Trainer-side graph path used only to relativize weighted sidecars.
    /// Returns:
    ///     RemoteBatchIterator exposing complete batches and their sample count.
    /// Raises:
    ///     ValueError: If sharding or sidecar paths are invalid.
    ///     pyarrow.flight.FlightError: If a replica rejects or interrupts the epoch.
    fn iter(
        &self,
        py: Python<'_>,
        request: &PySamplingRequest,
        graph_root: &str,
    ) -> PyResult<PyRemoteBatchIterator> {
        py.detach(|| self.index.iter(&request.request, Path::new(graph_root)))
            .map(|iterator| PyRemoteBatchIterator { iterator })
            .map_err(|error| client_error(py, error))
    }
    /// Cancel all epochs, join prefetch threads and release cached connections.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.index.close())
            .map_err(|error| client_error(py, error))
    }
}

/// Complete minibatches merged from native replica streams in round-robin order.
#[pyclass(name = "RemoteBatchIterator", module = "diskgraph._c", frozen)]
pub(crate) struct PyRemoteBatchIterator {
    iterator: Arc<RemoteBatchIterator>,
}
#[pymethods]
impl PyRemoteBatchIterator {
    /// Return the total selected seed count across this epoch's replica shards.
    fn num_samples(&self) -> u64 {
        self.iterator.num_samples
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let result = py
            .detach(|| self.iterator.next_batch())
            .map_err(|error| client_error(py, error))?;
        let Some(batch) = result else { return Ok(None) };
        match export_batch(py, &batch) {
            Ok(value) => Ok(Some(value)),
            Err(error) => {
                self.iterator
                    .close()
                    .map_err(|error| client_error(py, error))?;
                Err(error)
            }
        }
    }
    /// Cancel this epoch and join its native prefetch threads.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.iterator.close())
            .map_err(|error| client_error(py, error))
    }
}
