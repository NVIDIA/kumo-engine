// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use diskgraph_index::page_io::{self, EdgeIoBackend, PageIoStats};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::OnceLock;

static IMPORT_PID: OnceLock<u32> = OnceLock::new();

/// Immutable process-local counters. Counts exclude DataLoader child processes.
#[pyclass(frozen, module = "diskgraph._c")]
pub struct PageIoSnapshot {
    #[pyo3(get)]
    pub pid: u32,
    #[pyo3(get)]
    pub configured_backend: String,
    #[pyo3(get)]
    pub runtime_initialized: bool,
    #[pyo3(get)]
    pub cache_initialized: bool,
    #[pyo3(get)]
    pub runtime_rings: usize,
    stats: PageIoStats,
}

#[pymethods]
impl PageIoSnapshot {
    /// JSON-compatible snapshot of metadata, cumulative counters, and gauges.
    ///
    /// Reads are relaxed and not an atomic transaction. Subtract cumulative
    /// counters between quiescent boundaries; capacity/entries are gauges and
    /// max_batch_pages is a high-water mark. Frame recycling is batched.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("pid", self.pid)?;
        result.set_item("configured_backend", &self.configured_backend)?;
        result.set_item("runtime_initialized", self.runtime_initialized)?;
        result.set_item("cache_initialized", self.cache_initialized)?;
        result.set_item("runtime_rings", self.runtime_rings)?;
        result.set_item("submitted", self.stats.submitted)?;
        result.set_item("io_ops", self.stats.io_ops)?;
        result.set_item("multi_page_ops", self.stats.multi_page_ops)?;
        result.set_item("multi_page_pages", self.stats.multi_page_pages)?;
        result.set_item("runtime_jobs", self.stats.runtime_jobs)?;
        result.set_item("runtime_envelopes", self.stats.runtime_envelopes)?;
        result.set_item("runtime_batches", self.stats.runtime_batches)?;
        result.set_item("max_batch_pages", self.stats.max_batch_pages)?;
        result.set_item("cache_hits", self.stats.cache_hits)?;
        result.set_item("cache_misses", self.stats.cache_misses)?;
        result.set_item("coalesced", self.stats.coalesced)?;
        result.set_item("evictions", self.stats.evictions)?;
        result.set_item("bytes", self.stats.bytes)?;
        result.set_item("frames_allocated", self.stats.frames_allocated)?;
        result.set_item("frames_reused", self.stats.frames_reused)?;
        result.set_item("frames_recycled", self.stats.frames_recycled)?;
        result.set_item("frames_dropped", self.stats.frames_dropped)?;
        result.set_item("cache_capacity_pages", self.stats.cache_capacity_pages)?;
        result.set_item("cache_entries", self.stats.cache_entries)?;
        Ok(result)
    }
}

/// Read native page-I/O counters without resetting counters or starting I/O.
///
/// mmap uses the kernel page cache and does not increment these counters.
/// Buffered io_uring bypasses the native cache; only Direct uses that cache.
/// Counters cover this process, not DataLoader workers. After fork this query
/// rejects inherited counters; use a fresh spawned interpreter for diagnostics.
/// Ordinary native index and worker behavior is unaffected by this restriction.
#[pyfunction]
pub fn page_io_stats() -> PyResult<PageIoSnapshot> {
    let pid = std::process::id();
    if IMPORT_PID.get().copied() != Some(pid) {
        return Err(PyRuntimeError::new_err(
            "native page-I/O counters were inherited from another process; use a fresh spawned interpreter for diagnostics",
        ));
    }
    let backend = EdgeIoBackend::from_env()
        .map_err(|error| crate::DiskgraphConfigError::new_err(error.to_string()))?;
    let state = page_io::runtime_state();
    Ok(PageIoSnapshot {
        pid,
        configured_backend: backend.name().to_owned(),
        runtime_initialized: state.runtime_initialized,
        cache_initialized: state.cache_initialized,
        runtime_rings: state.runtime_rings,
        stats: page_io::stats(),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    IMPORT_PID.get_or_init(std::process::id);
    m.add_class::<PageIoSnapshot>()?;
    m.add_function(wrap_pyfunction!(page_io_stats, m)?)?;
    Ok(())
}
