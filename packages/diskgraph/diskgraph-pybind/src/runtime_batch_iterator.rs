// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::batch::export_batch;
use crate::{DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError};
use diskgraph_runtime::ArrowBatch;
#[cfg(test)]
pub(crate) use diskgraph_runtime::PseudoTimeSeed;
use pyo3::prelude::*;

#[pyclass]
pub struct RuntimeIndex {
    inner: diskgraph_runtime::RuntimeIndex,
}
#[pyclass]
pub struct RuntimeBatchIterator {
    inner: diskgraph_runtime::RuntimeBatchIterator,
}
fn map_runtime_err(error: diskgraph_runtime::RuntimeError) -> PyErr {
    use diskgraph_runtime::RuntimeError;
    match error {
        RuntimeError::Config(msg) => DiskgraphConfigError::new_err(msg),
        RuntimeError::Data(msg) => DiskgraphDataError::new_err(msg),
        RuntimeError::System(msg) => DiskgraphSystemError::new_err(msg),
        RuntimeError::TargetCategoryLimit(msg) => {
            crate::DiskgraphTargetCategoryLimitError::new_err(msg)
        }
    }
}
#[pymethods]
impl RuntimeIndex {
    #[new]
    #[pyo3(signature = (parent_root, tt_id, rnd_seed=None, num_threads=None))]
    pub fn new(
        parent_root: String,
        tt_id: String,
        rnd_seed: Option<u64>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: diskgraph_runtime::RuntimeIndex::new(parent_root, tt_id, rnd_seed, num_threads)
                .map_err(map_runtime_err)?,
        })
    }

    /// Build an iterator over one (stage, epoch, shard) of this index.
    ///
    /// Cheap: no mmap, no decoders, no thread pool. Call it once per epoch, or
    /// once per chunk when work is handed out dynamically.
    #[pyo3(signature = (
        stage_table,
        sampling_plan_json,
        num_workers,
        worker_id,
        shuffle,
        rnd_seed,
        epoch,
        batch_size,
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
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn iter(
        &self,
        stage_table: String,
        sampling_plan_json: String,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        batch_size: usize,
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
        projections: Option<std::collections::HashMap<String, Vec<String>>>,
    ) -> PyResult<RuntimeBatchIterator> {
        Ok(RuntimeBatchIterator {
            inner: self
                .inner
                .iter(
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
                )
                .map_err(map_runtime_err)?,
        })
    }
}
#[pymethods]
impl RuntimeBatchIterator {
    /// Backwards-compatible constructor: open an index and immediately
    /// iterate it. Equivalent to `RuntimeIndex(...).iter(...)`, and kept so
    /// existing callers need no change. Callers that build many iterators
    /// over one index should hold a `RuntimeIndex` instead -- that is the
    /// whole point of the split, and this shim cannot do it for them.
    #[new]
    #[pyo3(signature = (
        parent_root,
        tt_id,
        stage_table,
        sampling_plan_json,
        num_workers,
        worker_id,
        shuffle,
        rnd_seed,
        epoch,
        batch_size,
        drop_last=false,
        num_threads=None,
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
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        parent_root: String,
        tt_id: String,
        stage_table: String,
        sampling_plan_json: String,
        num_workers: u64,
        worker_id: u64,
        shuffle: bool,
        rnd_seed: Option<u64>,
        epoch: u64,
        batch_size: usize,
        drop_last: bool,
        num_threads: Option<usize>,
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
        projections: Option<std::collections::HashMap<String, Vec<String>>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: diskgraph_runtime::RuntimeIndex::new(parent_root, tt_id, rnd_seed, num_threads)
                .map_err(map_runtime_err)?
                .iter(
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
                )
                .map_err(map_runtime_err)?,
        })
    }

    /// No-op lifecycle hook for synchronous local iteration.
    ///
    /// There is no background reader to cancel; resources follow normal
    /// ownership of the iterator and its shared index.
    pub fn close(&self) {}

    pub fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }
    pub fn worker_batch_count(&self) -> usize {
        self.inner.worker_batch_count()
    }
    pub fn stage_size(&self) -> u64 {
        self.inner.stage_size()
    }
    pub fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let batch = py
            .detach(|| {
                self.inner
                    .next_batch()?
                    .map(ArrowBatch::try_from)
                    .transpose()
                    .map_err(|error| diskgraph_runtime::RuntimeError::data(error.to_string()))
            })
            .map_err(map_runtime_err)?;
        batch
            .map(|batch| export_batch(py, &batch).map(|value| value.unbind()))
            .transpose()
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.next_batch(py)
    }

    /// Number of batches this worker yields in one epoch. For the grouped path
    /// this is the count of owned global batch indices; for the flat path it is
    /// `ceil(num_samples / batch_size)` (or floor when `drop_last`).
    fn __len__(&self) -> usize {
        self.worker_batch_count()
    }
}
/// Free function: list finalized tt_ids under `<parent_root>/train_tables/`.
/// Mirrors `diskgraph_index::graph::discover_tts`; surfaces the admin API
/// so Python can ask "what tt_ids are available?" without opening the
/// reader first.
#[pyfunction]
fn discover_tts(parent_root: String) -> PyResult<Vec<String>> {
    let path = std::path::Path::new(&parent_root);
    diskgraph_index::graph::discover_tts(path)
        .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RuntimeIndex>()?;
    m.add_class::<RuntimeBatchIterator>()?;
    m.add_function(wrap_pyfunction!(discover_tts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::map_runtime_err;
    use crate::{
        DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError,
        DiskgraphTargetCategoryLimitError,
    };
    use diskgraph_runtime::RuntimeError;
    use pyo3::Python;

    #[test]
    fn runtime_errors_preserve_python_exception_categories_and_messages() {
        Python::initialize();
        Python::attach(|py| {
            let config = map_runtime_err(RuntimeError::config("invalid batch size"));
            assert!(config.is_instance_of::<DiskgraphConfigError>(py));
            assert_eq!(config.value(py).to_string(), "invalid batch size");

            let data = map_runtime_err(RuntimeError::data("invalid seed sidecar"));
            assert!(data.is_instance_of::<DiskgraphDataError>(py));
            assert_eq!(data.value(py).to_string(), "invalid seed sidecar");

            let system = map_runtime_err(RuntimeError::system("index unavailable"));
            assert!(system.is_instance_of::<DiskgraphSystemError>(py));
            assert_eq!(system.value(py).to_string(), "index unavailable");

            let limit =
                map_runtime_err(RuntimeError::TargetCategoryLimit("too many targets".into()));
            assert!(limit.is_instance_of::<DiskgraphTargetCategoryLimitError>(py));
            assert_eq!(limit.value(py).to_string(), "too many targets");
        });
    }
}
