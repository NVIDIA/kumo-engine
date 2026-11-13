// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::temporal_query::QueryPython as _;
use crate::{DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError};
use arrow::array::{make_array, ArrayData};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, UInt32Array, UInt64Array};
#[cfg(test)]
pub(crate) use diskgraph_runtime::PseudoTimeSeed;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
fn py_record_batch_list<'py>(
    py: Python<'py>,
    batches: &[RecordBatch],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for batch in batches {
        out.append(batch.to_pyarrow(py)?)?;
    }
    Ok(out)
}

fn py_array<'py>(py: Python<'py>, array: &dyn Array) -> PyResult<Bound<'py, PyAny>> {
    array.to_data().to_pyarrow(py)
}

fn py_u64_array<'py>(py: Python<'py>, values: Vec<u64>) -> PyResult<Bound<'py, PyAny>> {
    UInt64Array::from(values).to_data().to_pyarrow(py)
}

fn py_u32_array<'py>(py: Python<'py>, values: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
    UInt32Array::from(values).to_data().to_pyarrow(py)
}

fn py_bool_array<'py>(py: Python<'py>, values: Vec<bool>) -> PyResult<Bound<'py, PyAny>> {
    arrow_array::BooleanArray::from(values)
        .to_data()
        .to_pyarrow(py)
}

// ---------------------------------------------------------------------------
// RuntimeBatchIterator — merged-reader path (replaces the legacy
// dual-engine `TaskBatchIterator`).
//
// Iterator over runtime batches matching PR-28765's
// `runtime_batch_to_hetero` contract. One engine over the merged
// (parent + seed-table extension) catalog; per-seed parent-row resolution is
// done by Phase-1 ingest, so this hot loop is just:
//
//   1. Pick the next `batch_size` row offsets for this worker shard.
//   2. `engine.sample_runtime_batch(...)` — walks `seed.edges.bin` to
//      recover parent row ids, reads seed rowpack, calls `engine.sample`,
//      and returns a `RuntimeBatch`.
//   3. `runtime_batch_to_pydict(...)` — converts the Rust struct to the
//      PyDict shape the trainer adapter expects, zero-copy where possible
//      (via `to_data().to_pyarrow` and `RecordBatch.to_pyarrow`).
// ---------------------------------------------------------------------------

use diskgraph_engine::output::{GraphChannel, RawFeatureBank, RuntimeBatch, TaskBag};

/// Convert our Rust `RuntimeBatch` to PR-28765's PyDict shape. All Arrow
/// payloads cross the FFI boundary via Arrow C Data Interface (zero-copy);
/// the only allocations are the Python dict scaffolding itself.
fn runtime_batch_to_pydict<'py>(
    py: Python<'py>,
    batch: &RuntimeBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // task ----
    let task = PyDict::new(py);
    let TaskBag {
        input_ids,
        tt_stage_node_ids,
        seed_node_ids,
        seed_timestamps,
        target,
        weight,
        amount_m,
        num_positives,
        rhs_seed_type,
        neg_mask,
        lp_pos_counts,
        lp_pos_rhs_global,
        lp_pos_mask,
        lp_pos_weight,
        lp_pos_local,
        lp_exclude_counts,
        lp_exclude_rhs_global,
    } = &batch.task;
    task.set_item("input_ids", py_u32_array(py, input_ids.clone())?)?;
    task.set_item(
        "tt_stage_node_ids",
        py_u64_array(py, tt_stage_node_ids.clone())?,
    )?;
    task.set_item("seed_node_ids", py_u64_array(py, seed_node_ids.clone())?)?;
    if let Some(ts) = seed_timestamps {
        let arr = Int64Array::from(ts.clone());
        task.set_item("seed_timestamps", py_array(py, &arr)?)?;
    }
    if let Some(tb) = target {
        task.set_item("target", tb.to_pyarrow(py)?)?;
    }
    if let Some(w) = weight {
        let arr = arrow_array::Float32Array::from(w.clone());
        task.set_item("weight", py_array(py, &arr)?)?;
    }
    // Negative-sampling shape (M, B, rhs type). Scalars convert directly.
    task.set_item("amount_m", *amount_m)?;
    task.set_item("num_positives", *num_positives)?;
    match rhs_seed_type {
        Some(t) => task.set_item("rhs_seed_type", t)?,
        None => task.set_item("rhs_seed_type", py.None())?,
    }
    // `neg_mask` is a bool array of length `B*amount_m`, anchor-major and
    // parallel to the `[B:B+B*M]` rhs negative slots in
    // graphs.main.node_ids[rhs_seed_type]. `true` = use the slot in the loss;
    // `false` = the slot holds a dense placeholder id (temporal-resample budget
    // exhausted) that the loss must ignore. Because masked slots keep a dense
    // placeholder, the `[B, M]` layout / `dst_neg_index` reshape is unaffected —
    // the loss applies the mask. Empty when no negative sampling was requested.
    task.set_item("neg_mask", py_bool_array(py, neg_mask.clone())?)?;
    // Ragged multi-positive LP label payload (FK-free resolution, design §S5).
    // Anchor-major. `lp_pos_counts` is `[B]` (per-anchor positive count);
    // `lp_pos_rhs_global` / `lp_pos_mask` are `[ΣP]`. Emitted only when non-empty
    // — the LP label path populates them; every non-LP path leaves the keys
    // absent. `lp_pos_weight` / `lp_pos_local` are emitted only when Some.
    if !lp_pos_counts.is_empty() {
        task.set_item("lp_pos_counts", py_u32_array(py, lp_pos_counts.clone())?)?;
        let rhs = Int64Array::from(lp_pos_rhs_global.clone());
        task.set_item("lp_pos_rhs_global", py_array(py, &rhs)?)?;
        task.set_item("lp_pos_mask", py_bool_array(py, lp_pos_mask.clone())?)?;
    }
    if let Some(w) = lp_pos_weight {
        let arr = arrow_array::Float32Array::from(w.clone());
        task.set_item("lp_pos_weight", py_array(py, &arr)?)?;
    }
    if let Some(local) = lp_pos_local {
        task.set_item("lp_pos_local", py_u32_array(py, local.clone())?)?;
    }
    // STATIC LP past-visit exclusion (eval-only). Emitted only when non-empty,
    // so train / predict batches leave the keys absent (the adapter's contract).
    if !lp_exclude_counts.is_empty() {
        task.set_item(
            "lp_exclude_counts",
            py_u32_array(py, lp_exclude_counts.clone())?,
        )?;
        let exclude_rhs = Int64Array::from(lp_exclude_rhs_global.clone());
        task.set_item("lp_exclude_rhs_global", py_array(py, &exclude_rhs)?)?;
    }
    dict.set_item("task", task)?;

    // graphs ----
    let graphs = PyDict::new(py);
    for (channel_name, channel) in &batch.graphs {
        graphs.set_item(channel_name, graph_channel_to_pydict(py, channel)?)?;
    }
    dict.set_item("graphs", graphs)?;

    // raw_features ----
    let raw_features = PyDict::new(py);
    for (ns, bank) in &batch.raw_features {
        raw_features.set_item(ns, raw_feature_bank_to_pydict(py, ns, bank)?)?;
    }
    dict.set_item("raw_features", raw_features)?;

    Ok(dict)
}

fn graph_channel_to_pydict<'py>(
    py: Python<'py>,
    channel: &GraphChannel,
) -> PyResult<Bound<'py, PyDict>> {
    let ch = PyDict::new(py);

    // node_ids (per type)
    let node_ids = PyDict::new(py);
    for (nt, ids) in &channel.node_ids {
        node_ids.set_item(nt, py_u64_array(py, ids.clone())?)?;
    }
    ch.set_item("node_ids", node_ids)?;

    // batch_dict (per type)
    let batch_dict = PyDict::new(py);
    for (nt, idxs) in &channel.batch_dict {
        batch_dict.set_item(nt, py_u32_array(py, idxs.clone())?)?;
    }
    ch.set_item("batch_dict", batch_dict)?;

    // edge_index_dict — grouped by (src, rel, dst). Each entry: {edge_type:{src,rel,dst}, src:[u32], dst:[u32]}.
    let edge_index_dict = PyList::empty(py);
    for (rel_idx, (srcs, dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (src, rel, dst) = channel
            .rel_id_to_names
            .get(&rel_id)
            .cloned()
            .unwrap_or_else(|| ("?".to_string(), format!("rel_{rel_id}"), "?".to_string()));
        let key = PyDict::new(py);
        key.set_item("src", src)?;
        key.set_item("rel", rel)?;
        key.set_item("dst", dst)?;
        let entry = PyDict::new(py);
        entry.set_item("edge_type", key)?;
        entry.set_item("src", py_u32_array(py, srcs.clone())?)?;
        entry.set_item("dst", py_u32_array(py, dsts.clone())?)?;
        edge_index_dict.append(entry)?;
    }
    ch.set_item("edge_index_dict", edge_index_dict)?;

    // num_sampled_nodes_dict — per-type per-hop counts: {name: [hop0, hop1, ...]}.
    let num_sampled_nodes = PyDict::new(py);
    for (nt, counts) in &channel.num_sampled_nodes {
        num_sampled_nodes.set_item(nt, py_u64_array(py, counts.clone())?)?;
    }
    ch.set_item("num_sampled_nodes_dict", num_sampled_nodes)?;

    // num_sampled_edges_dict — per-edge-type per-hop counts. Mirrors the
    // edge_index_dict entry shape ({edge_type:{src,rel,dst}, counts:[...]}) but
    // emits EVERY edge type the plan can reach, INCLUDING all-zero ones (no
    // `is_empty` skip), so the adapter can materialize every schema edge store
    // for layer-wise / explainer indexing — matching PyG NeighborLoader.
    let num_sampled_edges = PyList::empty(py);
    for (names, counts) in &channel.num_sampled_edges {
        let (src, rel, dst) = names;
        let key = PyDict::new(py);
        key.set_item("src", src)?;
        key.set_item("rel", rel)?;
        key.set_item("dst", dst)?;
        let entry = PyDict::new(py);
        entry.set_item("edge_type", key)?;
        entry.set_item("counts", py_u64_array(py, counts.clone())?)?;
        num_sampled_edges.append(entry)?;
    }
    ch.set_item("num_sampled_edges_dict", num_sampled_edges)?;

    Ok(ch)
}

fn raw_feature_bank_to_pydict<'py>(
    py: Python<'py>,
    namespace: &str,
    bank: &RawFeatureBank,
) -> PyResult<Bound<'py, PyDict>> {
    let block = PyDict::new(py);
    block.set_item("key", namespace)?;
    let num_rows: usize = bank.batches.iter().map(|b| b.num_rows()).sum();
    block.set_item("num_rows", num_rows)?;
    block.set_item("record_batches", py_record_batch_list(py, &bank.batches)?)?;
    Ok(block)
}

#[pyclass]
pub struct RuntimeIndex {
    inner: diskgraph_runtime::RuntimeIndex,
}
#[pyclass]
pub struct RuntimeBatchIterator {
    inner: diskgraph_runtime::RuntimeBatchIterator,
    history: Option<HistoryIteration>,
    planning: Option<LengthPlanning>,
}

/// Task-owned cache. The one-shard anchor extension has dense, stable stage IDs.
/// Encoders/length metadata are frozen for its lifetime; Python owns one cache
/// per (stage, prepared length contract), never a process-global graph cache.
#[pyclass]
pub struct RuntimeLengthCache {
    values: Arc<Mutex<diskgraph_runtime::MappedLengthCache>>,
}
#[pymethods]
impl RuntimeLengthCache {
    #[new]
    #[pyo3(signature = (stage_size, path, fields=3))]
    fn new(stage_size: usize, path: String, fields: usize) -> PyResult<Self> {
        Ok(Self {
            values: Arc::new(Mutex::new(
                diskgraph_runtime::MappedLengthCache::new(
                    std::path::Path::new(&path),
                    stage_size,
                    fields,
                )
                .map_err(map_runtime_err)?,
            )),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LengthColumn {
    column: String,
    #[serde(default)]
    metadata: Option<String>,
    #[serde(default)]
    missing_token: bool,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    validity_only: bool,
    #[serde(default)]
    missing_only: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlattenLengths {
    maximum: usize,
    keep_last: bool,
    event_boundary: bool,
    separator: bool,
    between: bool,
    before: bool,
    after: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LengthContract {
    query: String,
    merge: String,
    columns: HashMap<String, Vec<LengthColumn>>,
    #[serde(default)]
    history_only: bool,
    #[serde(default)]
    fixed: HashMap<String, usize>,
    row_maximum: Option<usize>,
    flatten: Option<FlattenLengths>,
    max_samples: usize,
    max_tokens: Option<usize>,
    packed: bool,
    #[serde(default)]
    nested_padded: bool,
    bucket: bool,
    drop_last: bool,
}
struct LengthMetadata {
    chunks: Vec<ArrayRef>,
    ends: Vec<usize>,
}
impl LengthMetadata {
    fn get(&self, id: u64) -> PyResult<Option<u64>> {
        let id =
            usize::try_from(id).map_err(|_| length_error("token metadata row ID exceeds usize"))?;
        let chunk = self.ends.partition_point(|&end| end <= id);
        let values = self.chunks.get(chunk).ok_or_else(|| {
            length_error("post-fit token metadata does not cover this graph row ID")
        })?;
        let row = id - if chunk == 0 { 0 } else { self.ends[chunk - 1] };
        if values.is_null(row) {
            return Ok(None);
        }
        let value = if let Some(values) = values.as_any().downcast_ref::<arrow_array::UInt16Array>()
        {
            u64::from(values.value(row))
        } else if let Some(values) = values.as_any().downcast_ref::<UInt64Array>() {
            values.value(row)
        } else {
            u64::try_from(
                values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(row),
            )
            .map_err(|_| length_error("post-fit token lengths cannot be negative"))?
        };
        Ok(Some(value))
    }
}
struct LengthPlanning {
    query: diskgraph_train_table::batch_aggregate::PreparedTemporalPlan,
    contract: LengthContract,
    token_lengths: HashMap<String, LengthMetadata>,
    cache: Arc<Mutex<diskgraph_runtime::MappedLengthCache>>,
    batches: VecDeque<Vec<u64>>,
    consumed: usize,
    computed: usize,
    hits: usize,
}

fn length_error(message: impl Into<String>) -> PyErr {
    DiskgraphConfigError::new_err(message.into())
}

impl LengthPlanning {
    fn enqueue(
        &mut self,
        batch: &RuntimeBatch,
        history: &HistoryIteration,
        final_pool: bool,
    ) -> PyResult<()> {
        let ids = &batch.task.tt_stage_node_ids;
        let cache = self
            .cache
            .lock()
            .map_err(|_| length_error("length cache poisoned"))?;
        if ids.iter().any(|&id| id as usize >= cache.len()) {
            return Err(length_error(
                "length cache does not match the native anchor stage",
            ));
        }
        let missing: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| cache.get(id as usize).is_none().then_some(i))
            .collect();
        self.hits += ids.len() - missing.len();
        drop(cache);
        if !missing.is_empty() {
            let roots = missing
                .iter()
                .map(|&i| batch.task.seed_node_ids[i])
                .collect::<Vec<_>>();
            let cutoffs = missing
                .iter()
                .map(|&i| batch.task.seed_timestamps.as_ref().map_or(0, |v| v[i]))
                .collect::<Vec<_>>();
            // Only clock/identity and length metadata are projected. Actual feature
            // decoding and encoding happen once when the final batch is delivered.
            let result = self.query.execute(
                &history.reader, &roots, &cutoffs, &history.engine.thread_pool,
            )
            .map_err(length_error)?;
            let merges = result
                .merged_event_orders(&history.merges)
                .map_err(length_error)?;
            let (_, bounds, order) = merges
                .iter()
                .find(|(name, _, _)| name == &self.contract.merge)
                .ok_or_else(|| length_error("unknown length input merge"))?;
            // Merge offsets refer to sequences in merge declaration order.
            let selection = history.merges.iter()
                .find(|plan| plan.name == self.contract.merge)
                .ok_or_else(|| length_error("unknown length input merge"))?;
            let mut costs = Vec::new();
            for name in &selection.sequences {
                let name = name.as_str();
                let sequence = result
                    .sequences
                    .iter()
                    .find(|s| s.name == name)
                    .ok_or_else(|| length_error("unknown length sequence"))?;
                let arrays =
                    self.contract
                        .columns
                        .get(name)
                        .into_iter()
                        .flatten()
                        .map(|column| {
                            if let Some(key) = &column.metadata {
                                let metadata = self.token_lengths.get(key).ok_or_else(|| {
                                    length_error("missing post-fit token metadata")
                                })?;
                                return Ok((column, None, Some(metadata)));
                            }
                            let array =
                                sequence
                                    .batch
                                    .column_by_name(&column.column)
                                    .ok_or_else(|| {
                                        length_error(format!(
                                            "missing token length column {:?}",
                                            column.column
                                        ))
                                    })?;
                            Ok((column, Some(array), None))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                for row in 0..sequence.batch.num_rows() {
                    let mut cost = usize::from(self.contract.history_only)
                        + self.contract.fixed.get(name).copied().unwrap_or(0);
                    for (column, array, metadata) in &arrays {
                        let value = if let Some(metadata) = metadata {
                            metadata.get(sequence.node_ids[row])?
                        } else if array.unwrap().is_null(row) {
                            None
                        } else {
                            Some(1)
                        };
                        if metadata.is_some_and(|m| {
                            m.chunks
                                .first()
                                .is_some_and(|a| a.data_type() == &arrow_schema::DataType::UInt16)
                        }) && value == Some(u16::MAX as u64)
                            && self
                                .contract
                                .row_maximum
                                .is_none_or(|max| max > u16::MAX as usize)
                        {
                            return Err(length_error("saturated event token metadata requires a row token cap no greater than 65535"));
                        }
                        if column.required && value.is_none() {
                            cost = usize::MAX;
                            break;
                        }
                        if column.validity_only {
                            continue;
                        }
                        let value = match value {
                            Some(_) if column.missing_only => 0,
                            Some(value) => usize::try_from(value)
                                .map_err(|_| length_error("token length exceeds usize"))?,
                            None => usize::from(column.missing_token),
                        };
                        cost = cost
                            .checked_add(value)
                            .ok_or_else(|| length_error("token length overflow"))?;
                    }
                    costs.push(if cost == usize::MAX {
                        cost
                    } else {
                        self.contract.row_maximum.map_or(cost, |max| cost.min(max))
                    });
                }
            }
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| length_error("length cache poisoned"))?;
            for (sample, &index) in missing.iter().enumerate() {
                let events = order[bounds[sample] as usize..bounds[sample + 1] as usize]
                    .iter()
                    .map(|&i| costs[i as usize])
                    .filter(|&cost| cost != usize::MAX)
                    .collect::<Vec<_>>();
                let history_length = events.len();
                let tokens = if let Some(flatten) = &self.contract.flatten {
                    crate::token_stream::packed_length(
                        &events,
                        flatten.maximum,
                        flatten.keep_last,
                        flatten.event_boundary,
                        flatten.separator,
                        flatten.between,
                        flatten.before,
                        flatten.after,
                    )
                    .map_err(length_error)?
                } else {
                    events
                        .iter()
                        .try_fold(0usize, |sum, &v| sum.checked_add(v))
                        .ok_or_else(|| length_error("token length overflow"))?
                };
                cache
                    .set(
                        ids[index] as usize,
                        [
                            history_length,
                            tokens,
                            events.iter().copied().max().unwrap_or(0),
                        ],
                    )
                    .map_err(map_runtime_err)?;
            }
            self.computed += missing.len();
        }
        let cache = self
            .cache
            .lock()
            .map_err(|_| length_error("length cache poisoned"))?;
        let mut ordered = ids.clone();
        if self.contract.bucket {
            // Model length is event count for row layouts and token count for token layouts.
            ordered.sort_by_key(|&id| cache.get(id as usize).expect("planned length")[1]);
        }
        let (mut current, mut sum, mut maximum, mut history_sum) =
            (Vec::new(), 0usize, 0usize, 0usize);
        for id in ordered {
            let [history_length, length, row_maximum] = cache.get(id as usize).expect("planned length");
            let single_cost = if self.contract.nested_padded {
                history_length
                    .checked_mul(row_maximum)
                    .ok_or_else(|| length_error("batch token length overflow"))?
            } else {
                length
            };
            if self
                .contract
                .max_tokens
                .is_some_and(|budget| single_cost > budget)
            {
                return Err(length_error("one sample exceeds Batching.max_tokens"));
            }
            let cost = if self.contract.nested_padded {
                history_sum
                    .checked_add(history_length)
                    .and_then(|n| n.checked_mul(maximum.max(row_maximum)))
            } else if self.contract.packed {
                sum.checked_add(length)
            } else {
                (current.len() + 1).checked_mul(maximum.max(length))
            }
            .ok_or_else(|| length_error("batch token length overflow"))?;
            if !current.is_empty()
                && (current.len() >= self.contract.max_samples
                    || self.contract.max_tokens.is_some_and(|budget| cost > budget))
            {
                self.batches.push_back(std::mem::take(&mut current));
                sum = 0;
                maximum = 0;
                history_sum = 0;
            }
            current.push(id);
            sum += length;
            history_sum += history_length;
            maximum = maximum.max(if self.contract.nested_padded {
                row_maximum
            } else {
                length
            });
        }
        if !(current.is_empty()
            || final_pool && self.contract.drop_last && current.len() < self.contract.max_samples)
        {
            self.batches.push_back(current);
        }
        Ok(())
    }
}

struct HistoryIteration {
    root_type: String,
    reader: std::sync::Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: std::sync::Arc<diskgraph_engine::engine::DiskGraphEngine>,
    plan: diskgraph_train_table::batch_aggregate::PreparedTemporalPlan,
    merges: Vec<crate::temporal_query::EventMerge>,
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

    /// A sampler view over this runtime's combined reader; does not open files.
    pub fn sampler(&self) -> crate::SamplerEngine {
        let (reader, engine) = self.inner.shared_context();
        crate::SamplerEngine {
            reader,
            engine,
            pool: diskgraph_engine::state::StatePool::new(),
            sample_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
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
            history: None,
            planning: None,
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
            history: None,
            planning: None,
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

    pub fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }
    pub fn worker_batch_count(&self) -> PyResult<usize> {
        if self.planning.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "length-aware batch count is known only after planning",
            ));
        }
        Ok(self.inner.worker_batch_count())
    }
    pub fn stage_size(&self) -> u64 {
        self.inner.stage_size()
    }

    /// Attach named neighborhoods to the SAME native anchor schedule.
    pub fn with_neighborhoods(&mut self, specs: String) -> PyResult<()> {
        self.inner
            .with_neighborhoods(&specs)
            .map_err(map_runtime_err)
    }

    /// Attach history work once per epoch; next_batch remains the only batch call.
    pub fn with_history(
        &mut self,
        sampler: &crate::SamplerEngine,
        root_type: String,
        plan: String,
        merges: String,
    ) -> PyResult<()> {
        let plan = diskgraph_train_table::batch_aggregate::PreparedTemporalPlan::prepare_history(
            &sampler.reader, &root_type, &plan,
        ).map_err(DiskgraphConfigError::new_err)?;
        let merges: Vec<crate::temporal_query::EventMerge> = serde_json::from_str(&merges)
            .map_err(|e| DiskgraphConfigError::new_err(format!("invalid history merges: {e}")))?;
        for merge in &merges {
            if merge.sequences.len() != merge.ascending.len() {
                return Err(DiskgraphConfigError::new_err("merge sequence/order counts differ"));
            }
            for sequence in &merge.sequences {
                plan.require_timed_sequence(sequence).map_err(DiskgraphConfigError::new_err)?;
            }
        }
        self.history = Some(HistoryIteration {
            root_type,
            reader: std::sync::Arc::clone(&sampler.reader),
            engine: std::sync::Arc::clone(&sampler.engine),
            plan,
            merges,
        });
        Ok(())
    }

    /// Bounded shuffled pools; sample composition and epoch cache stay native.
    pub fn with_length_planning(
        &mut self,
        cache: &RuntimeLengthCache,
        contract: String,
        token_lengths: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let contract: LengthContract =
            serde_json::from_str(&contract).map_err(|e| length_error(e.to_string()))?;
        if self.history.is_none() || contract.max_samples == 0 || contract.max_tokens == Some(0) {
            return Err(length_error(
                "length planning requires attached history and positive limits",
            ));
        }
        let mut metadata = HashMap::new();
        for (name, values) in token_lengths.iter() {
            let arrays = values.cast::<PyList>()?;
            let mut chunks = Vec::with_capacity(arrays.len());
            let mut ends = Vec::with_capacity(arrays.len());
            let mut end = 0usize;
            for value in arrays.iter() {
                let array = make_array(ArrayData::from_pyarrow_bound(&value)?);
                if !matches!(
                    array.data_type(),
                    arrow_schema::DataType::UInt16
                        | arrow_schema::DataType::Int64
                        | arrow_schema::DataType::UInt64
                ) {
                    return Err(length_error(
                        "post-fit token lengths must be uint16, int64 or uint64",
                    ));
                }
                let values = array;
                end = end
                    .checked_add(values.len())
                    .ok_or_else(|| length_error("token metadata size overflow"))?;
                ends.push(end);
                chunks.push(values);
            }
            metadata.insert(name.extract::<String>()?, LengthMetadata { chunks, ends });
        }
        let history = self.history.as_ref().expect("checked attached history");
        let query = diskgraph_train_table::batch_aggregate::PreparedTemporalPlan::prepare_history(
            &history.reader, &history.root_type, &contract.query,
        ).map_err(length_error)?;
        self.planning = Some(LengthPlanning {
            query,
            contract,
            token_lengths: metadata,
            cache: Arc::clone(&cache.values),
            batches: VecDeque::new(),
            consumed: 0,
            computed: 0,
            hits: 0,
        });
        Ok(())
    }

    pub fn length_planning_stats(&self) -> (usize, usize) {
        self.planning
            .as_ref()
            .map_or((0, 0), |v| (v.computed, v.hits))
    }

    pub fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let next = py.detach(|| {
            let batch = if let Some(planning) = self.planning.as_mut() {
                while planning.batches.is_empty() {
                    let Some(pool) = self.inner.next_batch().map_err(map_runtime_err)? else {
                        return Ok(None);
                    };
                    planning.consumed += pool.task.tt_stage_node_ids.len();
                    planning.enqueue(
                        &pool,
                        self.history.as_ref().unwrap(),
                        planning.consumed == self.inner.num_samples(),
                    )?;
                }
                Some(
                    self.inner
                        .sample_stage_ids(&planning.batches.pop_front().unwrap())
                        .map_err(map_runtime_err)?,
                )
            } else {
                self.inner.next_batch().map_err(map_runtime_err)?
            };
            let Some(mut batch) = batch else {
                return Ok(None);
            };
            self.inner
                .add_neighborhoods(&mut batch)
                .map_err(map_runtime_err)?;
            let history = self
                .history
                .as_ref()
                .map(|config| {
                    let cutoffs = batch
                        .task
                        .seed_timestamps
                        .clone()
                        .unwrap_or_else(|| vec![0; batch.task.seed_node_ids.len()]);
                    config.plan.execute(
                        &config.reader,
                        &batch.task.seed_node_ids,
                        &cutoffs,
                        &config.engine.thread_pool,
                    )
                    .and_then(|result| {
                        let orders = result.merged_event_orders(&config.merges)?;
                        Ok((result, orders))
                    })
                    .map_err(DiskgraphConfigError::new_err)
                })
                .transpose()?;
            Ok::<_, PyErr>(Some((batch, history)))
        })?;
        let Some((batch, history)) = next else {
            return Ok(None);
        };
        let output = runtime_batch_to_pydict(py, &batch)?;
        if let Some((result, orders)) = history {
            output.set_item("history", result.into_python(py, true)?)?;
            let merged = PyDict::new(py);
            for (name, bounds, order) in orders {
                merged.set_item(
                    name,
                    (
                        Int64Array::from(bounds).to_data().to_pyarrow(py)?,
                        Int64Array::from(order).to_data().to_pyarrow(py)?,
                    ),
                )?;
            }
            output.set_item("history_merges", merged)?;
        }
        Ok(Some(output.into()))
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
    fn __len__(&self) -> PyResult<usize> {
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
    m.add_class::<RuntimeLengthCache>()?;
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
