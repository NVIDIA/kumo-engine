// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python lifecycle handle; Flight requests and batches remain entirely native.
use crate::{DiskgraphConfigError, DiskgraphSystemError};
use diskgraph_gateway::{SamplingServer, ServerConfig};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::Value;
use std::{net::IpAddr, path::PathBuf};

/// Serve native sampling in this process, on Rust-owned background threads.
///
/// Args:
///     index_path: Completed worker-local graph index directory.
///     host: IP address to bind; use 0.0.0.0 for remote connections.
///     port: TCP port; zero atomically selects an available port.
///     max_streams: Positive limit on concurrent native epoch producers.
///     max_threads: Positive per-index sampling thread budget.
///
/// Raises:
///     ValueError: If the host is not an IP address.
///     DiskgraphConfigError: If graph metadata or sampling budgets are invalid.
///     DiskgraphSystemError: If the socket or native runtime cannot start.
#[pyclass(name = "SamplingServer", module = "diskgraph._c")]
pub(crate) struct PySamplingServer {
    server: SamplingServer,
}

#[pymethods]
impl PySamplingServer {
    #[new]
    #[pyo3(signature = (index_path, *, host="127.0.0.1", port=0, max_streams=1, max_threads=2))]
    fn new(
        py: Python<'_>,
        index_path: PathBuf,
        host: &str,
        port: u16,
        max_streams: usize,
        max_threads: usize,
    ) -> PyResult<Self> {
        if let Some(logger) = crate::PYTHON_LOGGER.get() {
            logger.reset();
        }
        let address = host
            .parse::<IpAddr>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let server = py
            .detach(|| {
                SamplingServer::start(ServerConfig {
                    index_path,
                    address: (address, port).into(),
                    max_streams,
                    max_threads,
                })
            })
            .map_err(|error| {
                if error
                    .downcast_ref::<tonic::Status>()
                    .is_some_and(|status| status.code() == tonic::Code::InvalidArgument)
                {
                    DiskgraphConfigError::new_err(error.to_string())
                } else {
                    DiskgraphSystemError::new_err(format!("{error:#}"))
                }
            })?;
        Ok(Self { server })
    }

    /// The actual bound TCP port, including when zero was requested.
    #[getter]
    fn port(&self) -> u16 {
        self.server.address().port()
    }

    /// Return the bound Flight URI; advertise a routable IP for remote clients.
    ///
    /// Args:
    ///     advertised_host: Optional worker IP to advertise instead of the bind address.
    ///
    /// Returns:
    ///     A Flight URI with the listener's actual bound port.
    ///
    /// Raises:
    ///     ValueError: If the advertised host is not an IP address.
    #[pyo3(signature = (advertised_host=None))]
    fn endpoint(&self, advertised_host: Option<&str>) -> PyResult<String> {
        let host = advertised_host
            .map(|host| host.parse::<IpAddr>())
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .unwrap_or_else(|| {
                let host = self.server.address().ip();
                if host.is_unspecified() {
                    if host.is_ipv4() {
                        IpAddr::from([127, 0, 0, 1])
                    } else {
                        IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
                    }
                } else {
                    host
                }
            });
        Ok(format!(
            "grpc://{}",
            std::net::SocketAddr::new(host, self.port())
        ))
    }

    /// Return readiness, process identity, and native request/batch counters.
    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (key, value) in self.server.health().as_object().expect("health object") {
            match value {
                Value::Bool(value) => result.set_item(key, *value)?,
                Value::Number(value) => {
                    result.set_item(key, value.as_u64().expect("health counter"))?
                }
                _ => unreachable!("health values are booleans or unsigned counters"),
            }
        }
        Ok(result)
    }

    /// Cancel streams and join the native listener without holding the GIL.
    ///
    /// Raises:
    ///     DiskgraphSystemError: If the native listener failed or panicked.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.server.close())
            .map_err(|error| DiskgraphSystemError::new_err(format!("{error:#}")))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}
