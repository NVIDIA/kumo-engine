use arrow::pyarrow::ToPyArrow;
use arrow_array::{Array, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use diskgraph_core::sampling::{NegativeSamplingSpec, SamplingFlags, SamplingPlan};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_index::graph::TableReader;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

use crate::{map_engine_err, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn epoch_shuffle_seed(base_seed: u64, epoch: u64, stage_key: &str) -> u64 {
    let mut h = splitmix64(base_seed);
    h ^= splitmix64(epoch);
    for byte in stage_key.as_bytes() {
        h = splitmix64(h ^ u64::from(*byte));
    }
    h
}

fn make_worker_offsets(
    input_count: u64,
    num_workers: u64,
    worker_id: u64,
    shuffle: bool,
    rnd_seed: Option<u64>,
    epoch: u64,
    stage_key: &str,
) -> PyResult<Vec<u64>> {
    if num_workers == 0 {
        return Err(DiskgraphConfigError::new_err("num_workers must be >= 1"));
    }
    if worker_id >= num_workers {
        return Err(DiskgraphConfigError::new_err(format!(
            "worker_id {worker_id} out of range for num_workers {num_workers}"
        )));
    }
    if shuffle && rnd_seed.is_none() {
        return Err(DiskgraphConfigError::new_err(
            "rnd_seed is required when shuffle=True",
        ));
    }
    let mut order: Vec<u64> = (0..input_count).collect();
    if shuffle {
        let mut rng =
            StdRng::seed_from_u64(epoch_shuffle_seed(rnd_seed.unwrap_or(0), epoch, stage_key));
        order.shuffle(&mut rng);
    }
    Ok(order
        .into_iter()
        .enumerate()
        .filter_map(|(pos, offset)| {
            if (pos as u64) % num_workers == worker_id {
                Some(offset)
            } else {
                None
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// TT-NG temporal-LP timestamp grouping — LOAD side.
//
// When `group_by_timestamp` is set, the iterator yields SINGLE-TIMESTAMP
// batches by consuming the per-stage `bucket_index.bin` + bucket list the
// ingest side recorded in `diskgraph_manifest.json` (see
// `diskgraph-ingest/src/ts_grouping.rs` + `tt_executor.rs`). The bin is a flat
// little-endian `u64` array of FINAL node-ids; bucket `b` occupies
// `[offset_b, offset_b + count_b)` (units = node-ids, byte range ×8).
//
// The schedule is LAZY and O(D)+O(one bucket) RAM (D = #buckets = distinct
// timestamps): we hold only the ordered bucket list + a prefix-sum of
// per-bucket batch counts + the bin path, and read/permute one bucket at a time
// as batches are pulled. Determinism: bucket ORDER and WITHIN-bucket order
// depend only on `(base_seed, epoch, stage_key[, ts])`, never on `worker_id`,
// so every worker computes the identical global order and then keeps its own
// residue class of global batch indices (`i % num_workers == worker_id`) — a
// true partition of all node-ids, single-timestamp by construction.
// ---------------------------------------------------------------------------

/// One bucket in the (epoch-shuffled) schedule: its timestamp (shuffle-seed
/// input + metadata, units = SECONDS — never compared to `seed_time` ns), its
/// `[offset, offset + count)` slice of `bucket_index.bin` (units = node-ids),
/// and the global batch index of its FIRST batch (`batch_start`). The bucket
/// owns global batch indices `[batch_start, batch_start + ceil(count/bs))`.
struct ScheduledBucket {
    ts: i64,
    offset: u64,
    count: u64,
    batch_start: u64,
}

/// Lazy single-timestamp batch schedule. Holds O(D) state (the ordered buckets,
/// bin path, and params) plus a cache of at most ONE materialized (permuted)
/// bucket's node-ids. Owned by the iterator when `group_by_timestamp` is set.
struct BucketSchedule {
    /// Buckets in epoch-shuffled order, each tagged with its global batch range
    /// start. `batch_start` is the running prefix-sum of `ceil(count/bs)`.
    buckets: Vec<ScheduledBucket>,
    /// Total batches across all buckets (`Σ ceil(count_b / batch_size)`).
    total_batches: u64,
    /// Absolute path of `bucket_index.bin`.
    bin_path: std::path::PathBuf,
    /// Shuffle base seed and epoch (within-bucket permutations key off these
    /// plus the bucket `ts`).
    base_seed: u64,
    epoch: u64,
    stage_key: String,
    batch_size: usize,
    num_workers: u64,
    worker_id: u64,
    /// Next owned global batch index to emit. Owned indices are
    /// `{ worker_id, worker_id + num_workers, ... } ∩ [0, total_batches)`.
    next_global: u64,
    /// Cache of the currently-materialized bucket: its index into `buckets` and
    /// its permuted node-ids. `None` until the first read; replaced whenever a
    /// batch lands in a different bucket. Because owned global indices are
    /// monotonic, bucket reads are monotonic → each bucket is read ~once.
    cached: Option<(usize, Vec<u64>)>,
}

impl BucketSchedule {
    /// Build the schedule from the manifest bucket list. `raw_buckets` are the
    /// `(ts, offset, count)` triples from `diskgraph_manifest.json` (ingest
    /// order = ascending ts). The schedule shuffles the BUCKET ORDER by
    /// `epoch_shuffle_seed(base, epoch, stage_key)` and prefix-sums the
    /// per-bucket batch counts. O(D) RAM.
    #[allow(clippy::too_many_arguments)]
    fn build(
        raw_buckets: Vec<RawBucket>,
        bin_path: std::path::PathBuf,
        base_seed: u64,
        epoch: u64,
        stage_key: String,
        batch_size: usize,
        num_workers: u64,
        worker_id: u64,
    ) -> Self {
        // Shuffle the bucket order deterministically by (base, epoch, stage_key)
        // — identical across workers.
        let mut order: Vec<usize> = (0..raw_buckets.len()).collect();
        let mut rng =
            StdRng::seed_from_u64(epoch_shuffle_seed(base_seed, epoch, &stage_key));
        order.shuffle(&mut rng);

        let mut buckets = Vec::with_capacity(raw_buckets.len());
        let mut acc: u64 = 0;
        for idx in order {
            let (ts, offset, count) = raw_buckets[idx];
            // Empty buckets contribute zero batches; skip so binary search and
            // the prefix-sum stay tight (an empty bucket would share a
            // batch_start with its successor and never be selected anyway).
            if count == 0 {
                continue;
            }
            let nbatches = count.div_ceil(batch_size as u64);
            buckets.push(ScheduledBucket {
                ts,
                offset,
                count,
                batch_start: acc,
            });
            acc += nbatches;
        }
        let total_batches = acc;
        BucketSchedule {
            buckets,
            total_batches,
            bin_path,
            base_seed,
            epoch,
            stage_key,
            batch_size,
            num_workers,
            worker_id,
            next_global: worker_id,
            cached: None,
        }
    }

    /// This worker's batch count = number of owned global indices in
    /// `[0, total_batches)`.
    fn worker_batch_count(&self) -> u64 {
        worker_batch_count(self.total_batches, self.num_workers, self.worker_id)
    }

    /// Resolve a global batch index to `(bucket index, chunk index within the
    /// bucket)` via binary search over the per-bucket batch ranges. `g` must be
    /// `< total_batches`.
    fn locate(&self, g: u64) -> (usize, u64) {
        // partition_point: first bucket whose batch_start > g; the owning bucket
        // is the one before it.
        let after = self
            .buckets
            .partition_point(|b| b.batch_start <= g);
        debug_assert!(after >= 1, "g < total_batches implies a non-empty prefix");
        let b = after - 1;
        let chunk = g - self.buckets[b].batch_start;
        (b, chunk)
    }

    /// Read bucket `b`'s node-id slice from `bucket_index.bin` and apply the
    /// within-bucket shuffle keyed by `(base, epoch, "{stage_key}:{ts}")`.
    /// O(count_b) RAM. Caller caches the result.
    fn materialize_bucket(&self, b: usize) -> PyResult<Vec<u64>> {
        use std::os::unix::fs::FileExt;
        let bucket = &self.buckets[b];
        let f = std::fs::File::open(&self.bin_path).map_err(|e| {
            DiskgraphSystemError::new_err(format!(
                "open bucket_index.bin {}: {e}",
                self.bin_path.display()
            ))
        })?;
        let mut bytes = vec![0u8; (bucket.count as usize) * 8];
        f.read_exact_at(&mut bytes, bucket.offset * 8).map_err(|e| {
            DiskgraphDataError::new_err(format!(
                "read bucket [{}, {}) from {}: {e}",
                bucket.offset,
                bucket.offset + bucket.count,
                self.bin_path.display()
            ))
        })?;
        let mut ids: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let key = format!("{}:{}", self.stage_key, bucket.ts);
        let mut rng =
            StdRng::seed_from_u64(epoch_shuffle_seed(self.base_seed, self.epoch, &key));
        ids.shuffle(&mut rng);
        Ok(ids)
    }

    /// Pull this worker's next single-timestamp batch of FINAL node-ids, or
    /// `None` when the worker's owned global indices are exhausted.
    fn next_node_ids(&mut self) -> PyResult<Option<Vec<u64>>> {
        if self.next_global >= self.total_batches {
            return Ok(None);
        }
        let g = self.next_global;
        self.next_global += self.num_workers;

        let (b, chunk) = self.locate(g);
        // (Re)materialize the bucket if it isn't the cached one. Owned indices
        // are monotonic ⇒ bucket index is non-decreasing ⇒ ~one read per bucket.
        if self.cached.as_ref().map(|(idx, _)| *idx) != Some(b) {
            let ids = self.materialize_bucket(b)?;
            self.cached = Some((b, ids));
        }
        let ids = &self.cached.as_ref().unwrap().1;
        let bs = self.batch_size as u64;
        let start = (chunk * bs) as usize;
        let end = (((chunk + 1) * bs).min(self.buckets[b].count)) as usize;
        Ok(Some(ids[start..end].to_vec()))
    }
}

/// A bucket as recorded in the manifest: `(ts, offset, count)` (matches the
/// `tt_executor` manifest encoding). Units: `ts` seconds, `offset`/`count`
/// node-ids.
type RawBucket = (i64, u64, u64);

/// Locate the `bucket_index` entry for the stage whose `table_name` matches
/// `stage_table` in `<ext_root>/diskgraph_manifest.json`, returning
/// `(bin_abs_path, raw_buckets)`. The bin `path` is recorded relative to the
/// extension root. FAILS LOUD if the manifest, the stage, or its `bucket_index`
/// is missing — `group_by_timestamp` was requested, so a missing artifact is an
/// error, never a silent fallback.
fn read_bucket_index(
    ext_root: &std::path::Path,
    stage_table: &str,
) -> PyResult<(std::path::PathBuf, Vec<RawBucket>)> {
    let manifest_path = ext_root.join("diskgraph_manifest.json");
    let bytes = std::fs::read(&manifest_path).map_err(|e| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but cannot read {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        DiskgraphConfigError::new_err(format!(
            "invalid diskgraph_manifest.json at {}: {e}",
            manifest_path.display()
        ))
    })?;
    let stages = manifest
        .get("task_dataset")
        .and_then(|td| td.get("stages"))
        .and_then(|s| s.as_object())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "diskgraph_manifest.json has no task_dataset.stages: {}",
                manifest_path.display()
            ))
        })?;
    // Stages are keyed by stage_key (train/val/test); the iterator only knows
    // the table_name, so match on the stage's `table_name`. table_names are
    // guaranteed distinct across stages (each stage emits its own table), but
    // assert it here so a manifest bug surfaces loudly instead of silently
    // picking an arbitrary first match.
    let mut matches = stages
        .values()
        .filter(|v| v.get("table_name").and_then(|t| t.as_str()) == Some(stage_table));
    let stage_entry = matches.next().ok_or_else(|| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but no stage with table_name {stage_table:?} \
             in {}",
            manifest_path.display()
        ))
    })?;
    if matches.next().is_some() {
        return Err(DiskgraphConfigError::new_err(format!(
            "diskgraph_manifest.json has multiple stages with table_name {stage_table:?} \
             in {} (stage table_names must be distinct)",
            manifest_path.display()
        )));
    }
    let bucket_index = stage_entry.get("bucket_index").ok_or_else(|| {
        DiskgraphConfigError::new_err(format!(
            "group_by_timestamp=True but stage {stage_table:?} has no bucket_index \
             in {} (ingest did not emit timestamp grouping for this stage)",
            manifest_path.display()
        ))
    })?;
    let rel_path = bucket_index
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "bucket_index for stage {stage_table:?} has no string `path`"
            ))
        })?;
    let buckets_json = bucket_index
        .get("buckets")
        .and_then(|b| b.as_array())
        .ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "bucket_index for stage {stage_table:?} has no `buckets` array"
            ))
        })?;
    let mut raw = Vec::with_capacity(buckets_json.len());
    for b in buckets_json {
        let ts = b.get("ts").and_then(|v| v.as_i64());
        let offset = b.get("offset").and_then(|v| v.as_u64());
        let count = b.get("count").and_then(|v| v.as_u64());
        match (ts, offset, count) {
            (Some(ts), Some(offset), Some(count)) => raw.push((ts, offset, count)),
            _ => {
                return Err(DiskgraphConfigError::new_err(format!(
                    "malformed bucket entry {b} for stage {stage_table:?}"
                )))
            }
        }
    }
    Ok((ext_root.join(rel_path), raw))
}

/// Number of global batch indices in `[0, total_batches)` that fall in worker
/// `worker_id`'s residue class `i % num_workers == worker_id`.
fn worker_batch_count(total_batches: u64, num_workers: u64, worker_id: u64) -> u64 {
    if worker_id >= total_batches {
        0
    } else {
        (total_batches - worker_id).div_ceil(num_workers)
    }
}

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

use diskgraph_engine::output::{
    GraphChannel, RawFeatureBank, RuntimeBatch, TaskBag,
};

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
    // absent. `lp_pos_weight` / `lp_pos_local` are emitted only when Some (S4).
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
        let (src, rel, dst) = channel.rel_id_to_names.get(&rel_id).cloned().unwrap_or_else(|| {
            ("?".to_string(), format!("rel_{rel_id}"), "?".to_string())
        });
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

/// Worker-sharded iterator over `RuntimeBatch`es of a seed-table stage.
/// Consumed via the merged-reader path (single engine over parent +
/// extension).
#[pyclass]
pub struct RuntimeBatchIterator {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    stage_table_catalog_key: String,
    stage_table: Arc<TableReader>,
    plan: SamplingPlan,
    flags: SamplingFlags,
    offsets: Vec<u64>,
    /// Worker-local explicit stage-table node ids to emit in caller order.
    /// These are final stage node ids (`task.tt_stage_node_ids`), not dense
    /// offsets.
    selected_stage_node_ids: Option<Vec<u64>>,
    cursor: usize,
    batch_size: usize,
    drop_last: bool,
    /// Optional engine-side global negative RHS sampling. `None` => no
    /// negatives (byte-identical to the pre-negatives path).
    negatives: Option<NegativeSamplingSpec>,
    /// FK-free LP resolution (design §S5). When set, the engine reads the seed
    /// set from the `entity_node_id` rowpack column instead of walking
    /// `seed.edges.bin`. `None` => legacy FK-edge path.
    entity_node_type: Option<String>,
    /// FK-free LP resolution: when set, the engine emits the ragged `lp_pos_*`
    /// labels from the `target_node_ids` rowpack column. `None` => raw target
    /// passthrough (non-LP / pred path).
    target_node_type: Option<String>,
    /// NodeRange seed mode (two-tower MIPS RHS pass). When set, the iterator
    /// seeds the FULL `[0, num_rhs)` catalog of this node type directly (the
    /// `offsets` ARE the rhs global node-ids, no stage-table map) at the fixed
    /// `rhs_seed_time`, bypassing stage / entity-target resolution and negatives.
    /// `None` => stage-table seeding (the default path).
    rhs_seed_node_type: Option<String>,
    /// Fixed anchor timestamp applied to every NodeRange seed. Required when
    /// `rhs_seed_node_type` is set.
    rhs_seed_time: Option<i64>,
    /// TT-NG temporal-LP timestamp grouping (LOAD side). When `Some`, the
    /// iterator yields single-timestamp batches from this lazy schedule instead
    /// of the flat `offsets` path; `offsets` is left empty in that mode. `None`
    /// => the existing flat `make_worker_offsets` path (byte-identical).
    bucket_schedule: Option<BucketSchedule>,
    /// Whether a TIMESTAMP column in the seed rowpack should be used as the
    /// temporal sampler anchor. Static task datasets may still carry a pseudo
    /// TIMESTAMP column for split/schema compatibility; those must not anchor
    /// temporal-most-recent sampling.
    use_seed_timestamps: bool,
}

#[pymethods]
impl RuntimeBatchIterator {
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
        selected_stage_node_ids=None,
        use_seed_timestamps=true,
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
        // Sampler dispatch: fused interleaves feature fetching with the edge
        // walk (cache-locality win for small per-batch feature working sets);
        // non-fused walks first + fetches features in a parallel map-reduce
        // (amortizes memory latency for large working sets). Both produce
        // the same RuntimeBatch shape.
        //
        // `None` (default) → DG picks via `engine::choose_fused_mode` over
        // (reader, plan, batch_size). Explicit `True`/`False` overrides for
        // callers that have profiled their workload. Long-term the explicit
        // override goes away and DG fully owns the decision (Phase 5 legacy
        // deletion — kumo's sampler abstraction is gone by then anyway).
        fused: Option<bool>,
        // PyG CSC convention: swap edge src/dst (and rel id via reverse_rel_map)
        // when recording edges. Default true matches PR 269 + what PR 28765's
        // adapter expects.
        swap_edge_direction: bool,
        // Engine-side global negative RHS sampling (LP training). Passed as 3
        // optional scalars (cleaner than a dict given the existing
        // signature-based arg list). `neg_rhs_node_type=None` (default)
        // disables negatives; when set, `neg_amount_m` is required.
        // `neg_rnd_seed=None` (default) ⇒ `thread_rng` (random negatives, the
        // training/prod default); `Some(x)` ⇒ deterministic draw (tests).
        neg_rhs_node_type: Option<String>,
        neg_amount_m: Option<usize>,
        neg_rnd_seed: Option<u64>,
        // Sampler flags exposed for callers (#307): with-replacement neighbor
        // draw, per-seed disjoint neighborhoods, and adaptive sampling.
        replace: bool,
        disjoint: bool,
        adaptive: bool,
        // FK-free LP resolution (design §S5). `entity_node_type` => read the seed
        // set from the `entity_node_id` rowpack column (no `seed.edges.bin`
        // walk). `target_node_type` => emit ragged `lp_pos_*` labels from the
        // `target_node_ids` column (positives are LABELS, never rooted). Both
        // `None` (default) preserves the legacy FK-edge / raw-target path.
        entity_node_type: Option<String>,
        target_node_type: Option<String>,
        // NodeRange seed mode (two-tower MIPS RHS pass). When `rhs_seed_node_type`
        // is set, the iterator seeds the FULL `[0, num_rhs)` catalog of that node
        // type directly at the fixed `rhs_seed_time` (required), bypassing
        // stage-table seeding / entity-target resolution / negatives. `None`
        // (default) preserves the stage-table seeding path byte-for-byte.
        rhs_seed_node_type: Option<String>,
        rhs_seed_time: Option<i64>,
        // TT-NG temporal-LP timestamp grouping (LOAD side). `false` (default) =>
        // the flat `make_worker_offsets` path, byte-identical. `true` => yield
        // single-timestamp batches from the per-stage `bucket_index.bin` the
        // ingest side recorded; FAILS LOUD if that artifact is missing.
        // Mutually exclusive with NodeRange mode (the RHS catalog pass has no
        // per-row timestamp to group on).
        group_by_timestamp: bool,
        // XAI / targeted replay mode: explicit stage-table node ids, in caller
        // order. These are the same ids emitted as task.tt_stage_node_ids.
        selected_stage_node_ids: Option<Vec<u64>>,
        use_seed_timestamps: bool,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(DiskgraphConfigError::new_err("batch_size must be >= 1"));
        }
        if selected_stage_node_ids.is_some() {
            if shuffle {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with shuffle=True",
                ));
            }
            if drop_last {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with drop_last=True",
                ));
            }
            if group_by_timestamp {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with group_by_timestamp=True",
                ));
            }
            if rhs_seed_node_type.is_some() {
                return Err(DiskgraphConfigError::new_err(
                    "selected_stage_node_ids is incompatible with rhs_seed_node_type (NodeRange mode)",
                ));
            }
        }
        if group_by_timestamp && rhs_seed_node_type.is_some() {
            return Err(DiskgraphConfigError::new_err(
                "group_by_timestamp is incompatible with rhs_seed_node_type (NodeRange mode)",
            ));
        }
        if group_by_timestamp && shuffle && rnd_seed.is_none() {
            return Err(DiskgraphConfigError::new_err(
                "rnd_seed is required when group_by_timestamp=True (used for the \
                 bucket-order + within-bucket shuffle)",
            ));
        }
        if num_workers == 0 {
            return Err(DiskgraphConfigError::new_err("num_workers must be >= 1"));
        }
        if worker_id >= num_workers {
            return Err(DiskgraphConfigError::new_err(format!(
                "worker_id {worker_id} out of range for num_workers {num_workers}"
            )));
        }
        if rhs_seed_node_type.is_some() && rhs_seed_time.is_none() {
            return Err(DiskgraphConfigError::new_err(
                "rhs_seed_time is required when rhs_seed_node_type is set (NodeRange mode)",
            ));
        }
        let negatives = match neg_rhs_node_type {
            Some(rhs_node_type) => {
                let amount_m = neg_amount_m.ok_or_else(|| {
                    DiskgraphConfigError::new_err(
                        "neg_amount_m is required when neg_rhs_node_type is set",
                    )
                })?;
                Some(NegativeSamplingSpec {
                    rhs_node_type,
                    amount_m,
                    rnd_seed: neg_rnd_seed,
                })
            }
            None => None,
        };
        let parent_path = std::path::Path::new(&parent_root);
        let reader = diskgraph_index::graph::GraphIndexReader::open_with_tt(parent_path, &tt_id)
            .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))?;
        let namespace = tt_id.rsplit('/').next().unwrap_or(&tt_id).to_string();
        let reader = Arc::new(reader);
        let engine = Arc::new(
            DiskGraphEngine::from_reader(reader.clone(), rnd_seed, num_threads)
                .map_err(map_engine_err)?,
        );
        // Stage table in the merged catalog is namespaced as `<namespace>/<bare>`;
        // accept either form from the caller.
        let stage_catalog_key = if stage_table.contains('/') {
            stage_table.clone()
        } else {
            format!("{namespace}/{stage_table}")
        };
        let stage_table_arc = reader.table(&stage_catalog_key).ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "unknown stage table in merged catalog: {stage_catalog_key}"
            ))
        })?;
        let selected_stage_node_ids = if let Some(ids) = selected_stage_node_ids {
            let mut local_ids = Vec::new();
            for (idx, node_id) in ids.into_iter().enumerate() {
                if (idx as u64) % num_workers == worker_id {
                    if stage_table_arc.shard_for_node(node_id).is_none() {
                        return Err(DiskgraphDataError::new_err(format!(
                            "selected_stage_node_ids[{idx}]={node_id} is not a valid node id for stage table `{stage_catalog_key}`"
                        )));
                    }
                    local_ids.push(node_id);
                }
            }
            Some(local_ids)
        } else {
            None
        };
        let sampling_plan: SamplingPlan = serde_json::from_str(&sampling_plan_json)
            .map_err(|e| DiskgraphConfigError::new_err(format!("Invalid plan JSON: {e}")))?;
        // Grouped (single-timestamp) path: build the lazy bucket schedule from
        // the per-stage `bucket_index.bin` the ingest side recorded. `offsets`
        // is unused in this mode (left empty); FAILS LOUD on a missing artifact.
        let bucket_schedule = if group_by_timestamp {
            let ext_root = reader.tt_extension_root().ok_or_else(|| {
                DiskgraphConfigError::new_err(
                    "group_by_timestamp=True but reader has no TT extension root",
                )
            })?;
            let bare_table = stage_catalog_key
                .rsplit('/')
                .next()
                .unwrap_or(&stage_catalog_key);
            let (bin_path, raw_buckets) = read_bucket_index(ext_root, bare_table)?;
            Some(BucketSchedule::build(
                raw_buckets,
                bin_path,
                rnd_seed.unwrap_or(0),
                epoch,
                stage_catalog_key.clone(),
                batch_size,
                num_workers,
                worker_id,
            ))
        } else {
            None
        };
        // NodeRange mode: seeds are the RHS catalog node-ids `[lo, hi]` (dense by
        // ingest guarantee), NOT stage-table row offsets. Build the offsets over
        // the dense range and add `lo` so each offset IS the global node-id. The
        // stage table is ignored for seeding in this mode.
        let offsets = if selected_stage_node_ids.is_some() {
            Vec::new()
        } else if group_by_timestamp {
            // Grouped path drives iteration from `bucket_schedule`, not offsets.
            Vec::new()
        } else if let Some(rhs_type) = rhs_seed_node_type.as_deref() {
            let (lo, hi) = engine
                .node_type_id_range(rhs_type)
                .map_err(map_engine_err)?;
            let num_rhs = hi - lo + 1;
            let mut offs = make_worker_offsets(
                num_rhs,
                num_workers,
                worker_id,
                shuffle,
                rnd_seed,
                epoch,
                rhs_type,
            )?;
            if lo != 0 {
                for o in offs.iter_mut() {
                    *o += lo;
                }
            }
            offs
        } else {
            make_worker_offsets(
                stage_table_arc.node_count(),
                num_workers,
                worker_id,
                shuffle,
                rnd_seed,
                epoch,
                &stage_catalog_key,
            )?
        };
        // `fused=None` → DG picks via heuristic. Explicit override wins.
        // Telemetry so trainers can see what was chosen and why.
        let chosen_fused = match fused {
            Some(v) => v,
            None => {
                let h = diskgraph_engine::engine::choose_fused_mode(
                    reader.as_ref(),
                    &sampling_plan,
                    batch_size,
                );
                tracing::info!(
                    chosen_fused = h,
                    batch_size = batch_size,
                    stage_table = %stage_catalog_key,
                    "DG-side fused-mode heuristic chose mode for this iterator"
                );
                h
            }
        };
        Ok(Self {
            reader,
            engine,
            stage_table_catalog_key: stage_catalog_key,
            stage_table: stage_table_arc,
            plan: sampling_plan,
            flags: SamplingFlags {
                replace,
                directed: true,
                disjoint,
                return_edge_id: false,
                adaptive,
                fused_feature_fetching: chosen_fused,
                swap_edge_direction,
            },
            offsets,
            selected_stage_node_ids,
            cursor: 0,
            batch_size,
            drop_last,
            negatives,
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            rhs_seed_time,
            bucket_schedule,
            use_seed_timestamps,
        })
    }

    pub fn num_samples(&self) -> usize {
        if let Some(ids) = self.selected_stage_node_ids.as_ref() {
            return ids.len();
        }
        // Grouped path: report the worker's batch count (offsets is empty).
        match &self.bucket_schedule {
            Some(sched) => sched.worker_batch_count() as usize,
            None => self.offsets.len(),
        }
    }

    /// This worker's batch count for one epoch. For the grouped path this is the
    /// number of owned global batch indices; Python's `__len__` calls it.
    pub fn worker_batch_count(&self) -> usize {
        if let Some(ids) = self.selected_stage_node_ids.as_ref() {
            return ids.len().div_ceil(self.batch_size);
        }
        match &self.bucket_schedule {
            Some(sched) => sched.worker_batch_count() as usize,
            None => {
                let remaining = self.offsets.len();
                if self.drop_last {
                    remaining / self.batch_size
                } else {
                    remaining.div_ceil(self.batch_size)
                }
            }
        }
    }

    pub fn stage_size(&self) -> u64 {
        self.stage_table.node_count()
    }

    pub fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        // Grouped (single-timestamp) path: pull the next bucket-chunk of FINAL
        // node-ids and feed them DIRECTLY to the engine (they are already final
        // node-ids — no stage `node_id_from_offset` map, no NodeRange seed).
        //
        // INVARIANT: ingest writes `bucket_index.bin` entries as `shard.min_id +
        // local` (see ts_grouping::merge_stage) — the SAME formula
        // `node_id_from_offset` uses to map a dense offset to a final node-id.
        // So a grouped id IS a valid final node-id of the stage table, and the
        // flat path's offset→node-id remap would be the identity here. This holds
        // for any shard layout (the +min_id term carries the shard base). The
        // `debug_assert!` below catches a future ingest/load skew (e.g. a stage
        // sub-index whose shard ranges don't match what ingest wrote) in dev
        // builds; it is compiled out of the release .so.
        //
        // Disjoint-field borrows: `sched` borrows ONLY `self.bucket_schedule`
        // mutably, leaving every other field free to borrow immutably. This lets
        // the `detach` (GIL-release) closure capture only the specific Rust
        // fields the sample needs (no `&self`, no `Py<…>` field) so it stays
        // `Send`. `next_node_ids` (which triggers `materialize_bucket`'s blocking
        // file IO) touches NO Python, so we fold it INTO the closure: both the
        // per-bucket IO and the multi-threaded rayon sample run with the GIL
        // released. Only `runtime_batch_to_pydict` (Python construction) stays
        // under the GIL.
        if let Some(sched) = self.bucket_schedule.as_mut() {
            let engine = &self.engine;
            let reader = &self.reader;
            let stage_table = &self.stage_table;
            let stage_table_catalog_key = &self.stage_table_catalog_key;
            let plan = &self.plan;
            let flags = &self.flags;
            let negatives = self.negatives.as_ref();
            let entity_node_type = self.entity_node_type.as_deref();
            let target_node_type = self.target_node_type.as_deref();
            let runtime_batch = py.detach(|| -> PyResult<Option<RuntimeBatch>> {
                let node_ids = match sched.next_node_ids()? {
                    Some(ids) => ids,
                    None => return Ok(None),
                };
                debug_assert!(
                    node_ids
                        .iter()
                        .all(|id| stage_table.shard_for_node(*id).is_some()),
                    "grouped seed id not a valid stage-table node-id — ingest \
                     bucket_index / load stage sub-index shard layouts disagree",
                );
                let rb = engine
                    .sample_runtime_batch(
                        reader.as_ref(),
                        stage_table_catalog_key,
                        &node_ids,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        negatives,
                        entity_node_type,
                        target_node_type,
                        self.use_seed_timestamps,
                        None,
                    )
                    .map_err(map_engine_err)?;
                Ok(Some(rb))
            })?;
            let runtime_batch = match runtime_batch {
                Some(rb) => rb,
                None => return Ok(None),
            };
            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        if let Some(selected_ids) = self.selected_stage_node_ids.as_ref() {
            let remaining = selected_ids.len().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(None);
            }
            let take = self.batch_size.min(remaining);
            let stage_node_ids = selected_ids[self.cursor..self.cursor + take].to_vec();
            self.cursor += take;

            let engine = &self.engine;
            let reader = &self.reader;
            let stage_table_catalog_key = &self.stage_table_catalog_key;
            let plan = &self.plan;
            let flags = &self.flags;
            let negatives = self.negatives.as_ref();
            let entity_node_type = self.entity_node_type.as_deref();
            let target_node_type = self.target_node_type.as_deref();
            let runtime_batch = py
                .detach(|| {
                    engine.sample_runtime_batch(
                        reader.as_ref(),
                        stage_table_catalog_key,
                        &stage_node_ids,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        negatives,
                        entity_node_type,
                        target_node_type,
                        self.use_seed_timestamps,
                        None,
                    )
                })
                .map_err(map_engine_err)?;

            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        let remaining = self.offsets.len().saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(None);
        }
        let take = self.batch_size.min(remaining);
        if take < self.batch_size && self.drop_last {
            return Ok(None);
        }
        let batch_offsets = self.offsets[self.cursor..self.cursor + take].to_vec();
        self.cursor += take;

        // NodeRange mode: the offsets ARE the rhs global node-ids (built over the
        // dense `[lo, hi]` catalog range), so there is no stage-table
        // `node_id_from_offset` map — pass them straight to the NodeRange engine
        // path with the fixed seed_time.
        if let Some(rhs_type) = self.rhs_seed_node_type.as_deref() {
            let seed_time = self.rhs_seed_time.expect(
                "rhs_seed_time present when rhs_seed_node_type is set (validated in new)",
            );
            // Release the GIL around the multi-threaded rayon sample (the engine
            // is GIL-free pure Rust); re-acquire only to build the PyDict.
            // Capture only the Rust fields needed so the closure stays `Send`.
            let engine = &self.engine;
            let reader = &self.reader;
            let plan = &self.plan;
            let flags = &self.flags;
            let runtime_batch = py
                .detach(|| {
                    engine.sample_runtime_batch(
                        reader.as_ref(),
                        // stage_table is unused in NodeRange mode; pass the rhs type.
                        rhs_type,
                        &batch_offsets,
                        diskgraph_core::dataset::DatasetId::nil(),
                        plan,
                        flags,
                        None,
                        None,
                        None,
                        true,
                        Some(diskgraph_engine::engine::NodeRangeSeed {
                            rhs_node_type: rhs_type,
                            seed_time,
                        }),
                    )
                })
                .map_err(map_engine_err)?;
            let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
            return Ok(Some(dict.into()));
        }

        // Map offsets → stage table node ids. With a single-shard stage
        // sub-index (Phase-1 invariant) these are equal, but `node_id_from_offset`
        // does the right thing regardless.
        let mut stage_node_ids = Vec::with_capacity(batch_offsets.len());
        for offset in &batch_offsets {
            let node_id = self
                .stage_table
                .node_id_from_offset(*offset)
                .ok_or_else(|| {
                    DiskgraphDataError::new_err(format!(
                        "offset {offset} out of range for stage table {}",
                        self.stage_table_catalog_key
                    ))
                })?;
            stage_node_ids.push(node_id);
        }

        // Sidecars (weight.bin, etc.) are auto-detected inside the engine
        // via `reader.tt_extension_root()` — no path threading at this layer.
        // Release the GIL around the multi-threaded rayon sample (the engine is
        // GIL-free pure Rust); re-acquire only to build the PyDict. Capture only
        // the Rust fields needed so the closure stays `Send` (no `&self`).
        let engine = &self.engine;
        let reader = &self.reader;
        let stage_table_catalog_key = &self.stage_table_catalog_key;
        let plan = &self.plan;
        let flags = &self.flags;
        let negatives = self.negatives.as_ref();
        let entity_node_type = self.entity_node_type.as_deref();
        let target_node_type = self.target_node_type.as_deref();
        let runtime_batch = py
            .detach(|| {
                engine.sample_runtime_batch(
                    reader.as_ref(),
                    stage_table_catalog_key,
                    &stage_node_ids,
                    diskgraph_core::dataset::DatasetId::nil(),
                    plan,
                    flags,
                    negatives,
                    entity_node_type,
                    target_node_type,
                    self.use_seed_timestamps,
                    None, // node_range off (stage-table path)
                )
            })
            .map_err(map_engine_err)?;

        let dict = runtime_batch_to_pydict(py, &runtime_batch)?;
        Ok(Some(dict.into()))
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
    m.add_class::<RuntimeBatchIterator>()?;
    m.add_function(wrap_pyfunction!(discover_tts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::make_worker_offsets;
    use super::{read_bucket_index, worker_batch_count, BucketSchedule};
    use pyo3::Python;
    use std::collections::HashSet;
    use std::io::Write;

    // -----------------------------------------------------------------------
    // TT-NG temporal-LP timestamp grouping (LOAD side) — schedule tests.
    // -----------------------------------------------------------------------

    /// Write a synthetic `bucket_index.bin` from a list of `(ts, [node_ids])`
    /// buckets laid out back-to-back (the ingest contract), returning the path,
    /// the raw `(ts, offset, count)` triples, and the full node-id set.
    fn write_bin(
        dir: &std::path::Path,
        buckets: &[(i64, Vec<u64>)],
    ) -> (std::path::PathBuf, Vec<(i64, u64, u64)>, HashSet<u64>) {
        let path = dir.join("bucket_index.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        let mut raw = Vec::new();
        let mut all = HashSet::new();
        let mut offset: u64 = 0;
        for (ts, ids) in buckets {
            for id in ids {
                f.write_all(&id.to_le_bytes()).unwrap();
                assert!(all.insert(*id), "duplicate node-id {id} in fixture");
            }
            raw.push((*ts, offset, ids.len() as u64));
            offset += ids.len() as u64;
        }
        (path, raw, all)
    }

    /// Drain a worker's schedule into the list of its per-batch node-id vecs.
    fn drain(sched: &mut BucketSchedule) -> Vec<Vec<u64>> {
        let mut out = Vec::new();
        while let Some(ids) = sched.next_node_ids().unwrap() {
            out.push(ids);
        }
        out
    }

    /// THE partition-invariant test. For a sweep of worker counts, batch sizes,
    /// and bucket-size distributions (buckets smaller/larger than batch_size, a
    /// partial tail, single-node buckets), collect every batch from every
    /// worker for one epoch and assert: the union equals the full node-id set
    /// with no duplicates (a true partition), and every batch is single-ts.
    #[test]
    fn bucket_schedule_is_a_single_ts_partition() {
        let tmp = tempfile::tempdir().unwrap();
        // ts -> node-ids for that timestamp. Distributions chosen to exercise
        // buckets < / == / > batch_size and partial tails. Node-ids are
        // arbitrary (not 0..n) to confirm we feed FINAL ids, not offsets.
        let bucket_dists: Vec<Vec<(i64, Vec<u64>)>> = vec![
            // Mixed sizes incl. a single-node bucket and a big one.
            vec![
                (100, vec![7]),
                (50, vec![10, 11, 12, 13, 14]),
                (200, vec![20, 21]),
                (75, (1000..1017).collect()),
                (10, vec![900, 901, 902]),
            ],
            // All buckets smaller than any tested batch_size.
            vec![(1, vec![1, 2]), (2, vec![3]), (3, vec![4, 5]), (4, vec![6])],
            // One large bucket → forces multi-chunk within a single ts.
            vec![(9, (0..23).collect())],
            // Two equal buckets exactly a multiple of batch_size 2.
            vec![(5, vec![100, 101, 102, 103]), (6, vec![200, 201])],
            // INTERIOR empty bucket: a `(ts, [])` entry between non-empty ones
            // exercises the `count == 0` skip in `build` (the empty bucket must
            // contribute zero batches and never be selected, yet the partition of
            // the remaining ids must stay exact and single-ts).
            vec![
                (11, vec![300, 301, 302]),
                (12, vec![]),
                (13, vec![400, 401]),
            ],
        ];

        for (di, dist) in bucket_dists.iter().enumerate() {
            let sub = tmp.path().join(format!("dist{di}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bin_path, raw, all_ids) = write_bin(&sub, dist);
            // Map node-id -> its ts, to assert single-ts batches.
            let ts_of: std::collections::HashMap<u64, i64> = dist
                .iter()
                .flat_map(|(ts, ids)| ids.iter().map(move |id| (*id, *ts)))
                .collect();

            for &num_workers in &[1u64, 2, 3, 4] {
                for &batch_size in &[1usize, 2, 3, 5, 8] {
                    let mut union: Vec<u64> = Vec::new();
                    let mut total_batches_seen = 0u64;
                    for worker_id in 0..num_workers {
                        let mut sched = BucketSchedule::build(
                            raw.clone(),
                            bin_path.clone(),
                            42,
                            0,
                            "train".to_string(),
                            batch_size,
                            num_workers,
                            worker_id,
                        );
                        // __len__ must match the batches actually yielded.
                        let reported = sched.worker_batch_count();
                        let batches = drain(&mut sched);
                        assert_eq!(
                            batches.len() as u64,
                            reported,
                            "dist{di} nw={num_workers} bs={batch_size} w{worker_id}: \
                             reported len != yielded"
                        );
                        total_batches_seen += batches.len() as u64;
                        for batch in &batches {
                            assert!(
                                !batch.is_empty(),
                                "no empty batch should be emitted"
                            );
                            assert!(
                                batch.len() <= batch_size,
                                "batch exceeds batch_size"
                            );
                            // I1: single timestamp per batch.
                            let first_ts = ts_of[&batch[0]];
                            for id in batch {
                                assert_eq!(
                                    ts_of[id], first_ts,
                                    "dist{di} bs={batch_size}: batch mixes timestamps"
                                );
                            }
                            union.extend_from_slice(batch);
                        }
                    }
                    // I2 no-dup + I3 complete: exact partition of all node-ids.
                    let uniq: HashSet<u64> = union.iter().copied().collect();
                    assert_eq!(
                        uniq.len(),
                        union.len(),
                        "dist{di} nw={num_workers} bs={batch_size}: node-id emitted twice"
                    );
                    assert_eq!(
                        uniq, all_ids,
                        "dist{di} nw={num_workers} bs={batch_size}: union != full set"
                    );
                    // Σ over workers of len == B (total global batches).
                    let one = BucketSchedule::build(
                        raw.clone(),
                        bin_path.clone(),
                        42,
                        0,
                        "train".to_string(),
                        batch_size,
                        1,
                        0,
                    );
                    assert_eq!(
                        total_batches_seen, one.total_batches,
                        "dist{di} nw={num_workers} bs={batch_size}: Σ len != B"
                    );
                }
            }
        }
    }

    /// An all-empty / zero-bucket schedule (`total_batches == 0`) drained
    /// end-to-end: every worker must yield nothing, report a batch count of 0,
    /// and `total_batches` must be 0. Covers both an empty bucket list and a
    /// list of only-empty buckets.
    #[test]
    fn bucket_schedule_zero_batches_drains_empty() {
        let tmp = tempfile::tempdir().unwrap();
        // Two zero-bucket cases: no buckets at all, and only-empty buckets.
        let cases: Vec<Vec<(i64, Vec<u64>)>> =
            vec![vec![], vec![(1, vec![]), (2, vec![]), (3, vec![])]];
        for (ci, dist) in cases.iter().enumerate() {
            let sub = tmp.path().join(format!("zero{ci}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bin_path, raw, all_ids) = write_bin(&sub, dist);
            assert!(all_ids.is_empty(), "zero-bucket fixture has no ids");
            for &num_workers in &[1u64, 2, 4] {
                for worker_id in 0..num_workers {
                    let mut sched = BucketSchedule::build(
                        raw.clone(),
                        bin_path.clone(),
                        42,
                        0,
                        "train".to_string(),
                        4,
                        num_workers,
                        worker_id,
                    );
                    assert_eq!(sched.total_batches, 0, "case{ci}: total_batches != 0");
                    assert_eq!(
                        sched.worker_batch_count(),
                        0,
                        "case{ci} w{worker_id}: reported a batch"
                    );
                    let batches = drain(&mut sched);
                    assert!(
                        batches.is_empty(),
                        "case{ci} w{worker_id}: yielded a batch from a zero-bucket schedule"
                    );
                }
            }
        }
    }

    /// `num_workers > total_batches`: some workers own ZERO global batch indices.
    /// Drive the full drain (not just the arithmetic) and assert every worker
    /// yields exactly its reported count, the high-id workers yield nothing, and
    /// the union across all workers is still an exact, no-dup, single-ts
    /// partition with Σ len == total_batches.
    #[test]
    fn bucket_schedule_more_workers_than_batches() {
        let tmp = tempfile::tempdir().unwrap();
        // 3 ids in 3 single-id buckets at batch_size 1 => exactly 3 batches.
        let dist: Vec<(i64, Vec<u64>)> =
            vec![(10, vec![500]), (20, vec![600]), (30, vec![700])];
        let (bin_path, raw, all_ids) = write_bin(tmp.path(), &dist);
        let ts_of: std::collections::HashMap<u64, i64> = dist
            .iter()
            .flat_map(|(ts, ids)| ids.iter().map(move |id| (*id, *ts)))
            .collect();

        let total = BucketSchedule::build(
            raw.clone(),
            bin_path.clone(),
            42,
            0,
            "train".to_string(),
            1,
            1,
            0,
        )
        .total_batches;
        assert_eq!(total, 3, "expected 3 total batches");

        let num_workers = 5u64; // > total_batches (3): workers 3 and 4 own nothing.
        let mut union: Vec<u64> = Vec::new();
        let mut total_seen = 0u64;
        let mut workers_with_zero = 0u64;
        for worker_id in 0..num_workers {
            let mut sched = BucketSchedule::build(
                raw.clone(),
                bin_path.clone(),
                42,
                0,
                "train".to_string(),
                1,
                num_workers,
                worker_id,
            );
            let reported = sched.worker_batch_count();
            let batches = drain(&mut sched);
            assert_eq!(
                batches.len() as u64,
                reported,
                "w{worker_id}: reported len != yielded"
            );
            if reported == 0 {
                workers_with_zero += 1;
            }
            total_seen += batches.len() as u64;
            for batch in &batches {
                assert!(!batch.is_empty(), "no empty batch should be emitted");
                let first_ts = ts_of[&batch[0]];
                for id in batch {
                    assert_eq!(ts_of[id], first_ts, "batch mixes timestamps");
                }
                union.extend_from_slice(batch);
            }
        }
        assert_eq!(
            workers_with_zero, 2,
            "expected exactly 2 workers (3,4) to own zero batches"
        );
        let uniq: HashSet<u64> = union.iter().copied().collect();
        assert_eq!(uniq.len(), union.len(), "node-id emitted twice");
        assert_eq!(uniq, all_ids, "union != full set");
        assert_eq!(total_seen, total, "Σ len != total_batches");
    }

    /// Determinism: same `(base, epoch)` => identical schedule independent of a
    /// second construction; different `epoch` => different global order but
    /// still a valid partition; the order is independent of `worker_id`.
    #[test]
    fn bucket_schedule_is_deterministic_and_epoch_varying() {
        let tmp = tempfile::tempdir().unwrap();
        let dist: Vec<(i64, Vec<u64>)> = (0..12)
            .map(|t| (t as i64, vec![(t * 10) as u64, (t * 10 + 1) as u64]))
            .collect();
        let (bin_path, raw, all_ids) = write_bin(tmp.path(), &dist);

        let build = |epoch: u64, worker_id: u64, num_workers: u64| {
            let mut s = BucketSchedule::build(
                raw.clone(),
                bin_path.clone(),
                7,
                epoch,
                "train".to_string(),
                3,
                num_workers,
                worker_id,
            );
            drain(&mut s)
        };

        // Same (base, epoch) replays byte-identically.
        assert_eq!(build(0, 0, 1), build(0, 0, 1));
        // Different epoch => different order (probabilistic but safe at 12 ts).
        assert_ne!(build(0, 0, 1), build(1, 0, 1));

        // The GLOBAL order (num_workers=1) is independent of worker_id: a
        // worker's batches are exactly the global batches at its residue class.
        let global = build(3, 0, 1);
        let nw = 3u64;
        let mut reassembled: Vec<Vec<u64>> = vec![Vec::new(); global.len()];
        for w in 0..nw {
            let wb = build(3, w, nw);
            // worker w owns global indices w, w+nw, ...
            for (k, batch) in wb.into_iter().enumerate() {
                let g = w as usize + k * nw as usize;
                reassembled[g] = batch;
            }
        }
        assert_eq!(reassembled, global, "per-worker residue classes != global order");

        // Different epoch still a full partition.
        let mut u = HashSet::new();
        for w in 0..nw {
            for b in build(5, w, nw) {
                for id in b {
                    assert!(u.insert(id), "dup across workers at epoch 5");
                }
            }
        }
        assert_eq!(u, all_ids);
    }

    /// `worker_batch_count` matches the residue-class arithmetic and Σ == B.
    #[test]
    fn worker_batch_count_sums_to_total() {
        for total in [0u64, 1, 5, 7, 16] {
            for nw in [1u64, 2, 3, 4] {
                let sum: u64 = (0..nw).map(|w| worker_batch_count(total, nw, w)).sum();
                assert_eq!(sum, total, "Σ worker_batch_count != total ({total},{nw})");
            }
        }
    }

    /// Fail-loud: `read_bucket_index` errors clearly when the manifest is
    /// missing, the stage is absent, or it has no `bucket_index`. Formatting a
    /// `PyErr` requires the interpreter, so the message assertion runs under the
    /// GIL.
    #[test]
    fn read_bucket_index_fails_loud() {
        Python::initialize();
        let tmp = tempfile::tempdir().unwrap();
        // No manifest at all.
        assert!(read_bucket_index(tmp.path(), "stage").is_err());

        // Manifest present but stage has no bucket_index.
        let manifest = serde_json::json!({
            "task_dataset": {
                "stages": {
                    "train": { "table_name": "stage", "row_count": 3 }
                }
            }
        });
        std::fs::write(
            tmp.path().join("diskgraph_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let err = read_bucket_index(tmp.path(), "stage").unwrap_err();
        Python::attach(|py| {
            assert!(
                err.value(py).to_string().contains("no bucket_index"),
                "expected a 'no bucket_index' error, got: {}",
                err.value(py)
            );
        });
        // Unknown stage table.
        assert!(read_bucket_index(tmp.path(), "nope").is_err());

        // Well-formed bucket_index round-trips through read_bucket_index.
        let manifest = serde_json::json!({
            "task_dataset": {
                "stages": {
                    "train": {
                        "table_name": "stage",
                        "bucket_index": {
                            "path": "stage/bucket_index.bin",
                            "buckets": [
                                { "ts": 10, "offset": 0, "count": 2 },
                                { "ts": 20, "offset": 2, "count": 1 }
                            ]
                        }
                    }
                }
            }
        });
        std::fs::write(
            tmp.path().join("diskgraph_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let (bin_path, raw) = read_bucket_index(tmp.path(), "stage").unwrap();
        assert_eq!(bin_path, tmp.path().join("stage/bucket_index.bin"));
        assert_eq!(raw, vec![(10, 0, 2), (20, 2, 1)]);
    }

    #[test]
    fn make_worker_offsets_shards_without_shuffle() {
        assert_eq!(
            make_worker_offsets(10, 3, 0, false, None, 0, "train").unwrap(),
            vec![0, 3, 6, 9]
        );
        assert_eq!(
            make_worker_offsets(10, 3, 1, false, None, 0, "train").unwrap(),
            vec![1, 4, 7]
        );
        assert_eq!(
            make_worker_offsets(10, 3, 2, false, None, 0, "train").unwrap(),
            vec![2, 5, 8]
        );
    }

    #[test]
    fn make_worker_offsets_handles_empty_and_more_workers_than_rows() {
        assert_eq!(
            make_worker_offsets(0, 4, 0, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
        assert_eq!(
            make_worker_offsets(2, 4, 0, false, None, 0, "train").unwrap(),
            vec![0]
        );
        assert_eq!(
            make_worker_offsets(2, 4, 1, false, None, 0, "train").unwrap(),
            vec![1]
        );
        assert_eq!(
            make_worker_offsets(2, 4, 2, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
        assert_eq!(
            make_worker_offsets(2, 4, 3, false, None, 0, "train").unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn make_worker_offsets_shuffle_is_deterministic_and_covers_all_offsets() {
        let a0 = make_worker_offsets(20, 3, 0, true, Some(42), 0, "train").unwrap();
        let a1 = make_worker_offsets(20, 3, 1, true, Some(42), 0, "train").unwrap();
        let a2 = make_worker_offsets(20, 3, 2, true, Some(42), 0, "train").unwrap();
        assert_eq!(
            a0,
            make_worker_offsets(20, 3, 0, true, Some(42), 0, "train").unwrap()
        );
        assert_ne!(
            a0,
            make_worker_offsets(20, 3, 0, false, None, 0, "train").unwrap()
        );

        let mut all = [a0, a1, a2].concat();
        all.sort_unstable();
        assert_eq!(all, (0..20).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_shuffle_varies_by_epoch_and_stage() {
        let epoch0 = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();
        let epoch1 = make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap();
        let val0 = make_worker_offsets(50, 2, 0, true, Some(42), 0, "val").unwrap();

        assert_ne!(epoch0, epoch1);
        assert_ne!(epoch0, val0);
        assert_eq!(
            epoch1,
            make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap()
        );

        let mut all = [
            make_worker_offsets(50, 2, 0, true, Some(42), 1, "train").unwrap(),
            make_worker_offsets(50, 2, 1, true, Some(42), 1, "train").unwrap(),
        ]
        .concat();
        all.sort_unstable();
        assert_eq!(all, (0..50).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_rejects_invalid_worker_config() {
        assert!(make_worker_offsets(10, 0, 0, false, None, 0, "train").is_err());
        assert!(make_worker_offsets(10, 3, 3, false, None, 0, "train").is_err());
        assert!(make_worker_offsets(10, 3, 0, true, None, 0, "train").is_err());
    }

    // Ported from PR 268 (tt-ng) — these locked invariants on the shuffle
    // sharder added in PR 266. We preserve `make_worker_offsets` verbatim
    // under the new module name, so the tests come along with it.

    #[test]
    fn make_worker_offsets_shuffle_varies_by_rnd_seed() {
        let seed_a = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();
        let seed_b = make_worker_offsets(50, 2, 0, true, Some(43), 0, "train").unwrap();
        let seed_a_replay = make_worker_offsets(50, 2, 0, true, Some(42), 0, "train").unwrap();

        assert_ne!(
            seed_a, seed_b,
            "different rnd_seed at same (epoch, stage, worker) should differ"
        );
        assert_eq!(
            seed_a, seed_a_replay,
            "same (rnd_seed, epoch, stage, worker) must replay byte-identical"
        );
    }

    #[test]
    fn make_worker_offsets_disjoint_across_workers_at_fixed_epoch() {
        let num_workers: u64 = 4;
        let input_count: u64 = 50;
        let mut shards: Vec<Vec<u64>> = Vec::with_capacity(num_workers as usize);
        for w in 0..num_workers {
            shards.push(
                make_worker_offsets(input_count, num_workers, w, true, Some(42), 3, "train")
                    .unwrap(),
            );
        }

        // Pairwise disjoint (not just sorted union — every pair has zero overlap).
        for i in 0..shards.len() {
            for j in (i + 1)..shards.len() {
                let a: std::collections::HashSet<_> = shards[i].iter().copied().collect();
                for offset in &shards[j] {
                    assert!(
                        !a.contains(offset),
                        "worker {} and worker {} overlap on offset {}",
                        i,
                        j,
                        offset
                    );
                }
            }
        }

        // Union covers exactly [0, input_count).
        let mut union: Vec<u64> = shards.into_iter().flatten().collect();
        union.sort_unstable();
        assert_eq!(union, (0..input_count).collect::<Vec<u64>>());
    }

    #[test]
    fn make_worker_offsets_multi_epoch_sweep_all_distinct_and_replayable() {
        let epochs = 5u64;
        let mut orderings: Vec<Vec<u64>> = Vec::with_capacity(epochs as usize);
        for e in 0..epochs {
            orderings.push(make_worker_offsets(64, 2, 0, true, Some(42), e, "train").unwrap());
        }

        // All epochs produce pairwise-distinct orderings on the same worker.
        // (Probabilistic for tiny input sizes; 64 rows is safe.)
        for i in 0..orderings.len() {
            for j in (i + 1)..orderings.len() {
                assert_ne!(
                    orderings[i], orderings[j],
                    "epoch {} and epoch {} produced identical ordering on worker 0",
                    i, j
                );
            }
        }

        // Each epoch is byte-identical on replay.
        for (e, ordering) in orderings.iter().enumerate() {
            let replay =
                make_worker_offsets(64, 2, 0, true, Some(42), e as u64, "train").unwrap();
            assert_eq!(*ordering, replay, "epoch {} replay diverged", e);
        }
    }
}
