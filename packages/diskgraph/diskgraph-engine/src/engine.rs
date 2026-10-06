// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::edge_unpacker::EdgeUnpacker;
use crate::errors::{EngineError, Result};
use crate::output::OutputHandler;
use crate::state::{LookupKey, NodeRecord, SamplingState};
use diskgraph_core::sampling::*;
use diskgraph_core::sortkey::SortKeyDecoder;
use diskgraph_core::tt_builder::{Value, WEIGHT_COL};
use diskgraph_core::time::{abs_from_unit, Seconds, TimeUnit};
use diskgraph_index::layout::{EdgeListView, RelationSegment};
use diskgraph_index::reader::ShardReader;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

const DETERMINISTIC_UNIFORM_SEED_V1: u64 = 41;
const FREQUENCY_COL: &str = "__kumo_internal_frequency_column__";
/// The customer-provided custom-TT split column (kumo baseline convention).
/// It is dropped from the per-stage seed parquets at routing time (the primary
/// guarantee), so it should never reach a seed rowpack; this is a cheap
/// belt-and-suspenders so it can never be materialized as a feature.
const SPLIT_COL: &str = "split_col";

/// True when `name` is a reserved TT seed/stage rowpack column that is
/// consumed for a task purpose and must NEVER be materialized as a model
/// feature. Single source of truth for the generic feature paths. Matches
/// `WEIGHT`/`entity_node_id`/`target_node_ids` exactly and the indexed
/// `ENTITY<n>`/`TARGET<n>` families precisely (`TARGET_AUDIENCE`, bare
/// `ENTITY` do NOT match).
///
/// `TIMESTAMP` is intentionally not filtered here: it is reserved in seed
/// rowpacks, but graph tables may also have a real timestamp feature with that
/// name. Seed-rowpack projection filters it at the task boundary.
#[inline]
pub(crate) fn is_reserved_feature_column(name: &str) -> bool {
    matches!(
        name,
        WEIGHT_COL
            | FREQUENCY_COL
            | SPLIT_COL
            | "entity_node_id"
            | "target_node_ids"
            | "exclude_node_ids"
    ) || is_indexed_reserved(name, "ENTITY")
        || is_indexed_reserved(name, "TARGET")
}

/// True when `name` is `<prefix><digits>` with at least one digit
/// (e.g. `ENTITY1`, `TARGET12`) — the indexed reserved families. Rejects the
/// bare prefix and any non-digit suffix (`ENTITY`, `TARGET_AUDIENCE`).
#[inline]
fn is_indexed_reserved(name: &str, prefix: &str) -> bool {
    match name.strip_prefix(prefix) {
        Some(rest) => !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// Metadata and reader for a single graph shard.
pub struct ShardHandle {
    pub reader: Arc<ShardReader>,
    pub min_id: u64,
    pub max_id: u64,
}

/// NodeRange seed mode for [`DiskGraphEngine::sample_runtime_batch`] (two-tower
/// MIPS RHS pass). When supplied, the iterator seeds RHS/target catalog node-ids
/// directly (the `batch_row_ids` ARE those ids) at a single fixed `seed_time`,
/// bypassing stage-table / entity-target resolution and negatives.
#[derive(Debug, Clone, Copy)]
pub struct NodeRangeSeed<'a> {
    /// The RHS/target node type to seed (e.g. the item catalog).
    pub rhs_node_type: &'a str,
    /// The fixed anchor timestamp applied to every seed in the batch.
    pub seed_time: i64,
}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct EdgeRelationInfo {
    pub name: String,
    pub src_type: String,
    pub dst_type: String,
}

#[derive(Serialize, Deserialize)]
pub struct TypeNameMapping {
    pub node_types: Vec<String>,
    pub edge_relations: Vec<EdgeRelationInfo>,
}

use diskgraph_rowpack::{RowPackDecodeScratch, RowPackDecoder};

/// DG-side heuristic picking fused vs non-fused dispatch for this
/// (reader, plan, batch_size), so kumo passes `fused=None` and lets DG decide
/// (explicit `Some(..)` overrides still honored). Fused interleaves feature
/// fetch with the edge walk (wins when the per-batch working set fits L2);
/// non-fused walks edges then fetches in a parallel pass (wins when it blows
/// L2, where map-reduce hides memory latency better).
///
/// v1 heuristic: `working_set ≈ batch_size × Πlayer_fanout × ~128 B/row` vs the
/// L2 budget (`DG_FUSED_L2_TARGET`, default 512 KB); below → fused. Missing
/// fanouts count as 0 (a layer's true expansion is the relation's natural
/// fanout, unknown here); bias toward non-fused on uncertainty.
pub fn choose_fused_mode(
    _reader: &diskgraph_index::graph::GraphIndexReader,
    plan: &SamplingPlan,
    batch_size: usize,
) -> bool {
    let nodes_per_seed: usize = plan
        .layers
        .iter()
        .map(|layer| {
            let from_default = layer
                .default_fanout
                .as_ref()
                .map(|f| f.fanout as usize)
                .unwrap_or(0);
            let from_explicit: usize = layer.fanouts.values().map(|f| f.fanout as usize).sum();
            from_default.max(from_explicit).max(1)
        })
        .product();
    // ~128 B/row is the rough cost of a typical sampled row's feature footprint
    // (rowpack-decoded fields + small embedding columns). Coarse but stable.
    const BYTES_PER_ROW: usize = 128;
    let working_set = batch_size.saturating_mul(nodes_per_seed).saturating_mul(BYTES_PER_ROW);
    let threshold = std::env::var("DG_FUSED_L2_TARGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512_000);
    working_set <= threshold
}

fn get_feature_batch_bytes() -> usize {
    std::env::var("FEATURE_BATCH_SIZE_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64 * 1024 * 1024)
}

fn get_feature_batch_rows() -> usize {
    std::env::var("FEATURE_BATCH_SIZE_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64_000)
}

/// Bounded resample budget for temporally-gated negative draws. If no
/// temporally-valid candidate is found within K extra attempts, the slot is
/// **masked** (see [`append_global_negatives`]): the last-drawn id is kept as a
/// dense placeholder but its `neg_mask` entry is set `false` so the loss ignores
/// it, rather than emitting a temporally-invalid negative.
const NEG_RESAMPLE_BUDGET: usize = 5;

/// Append `amount_m` global negative RHS ids per positive to
/// `seed_nodes[rhs_node_type]`, keeping `seed_ts[rhs_node_type]` row-aligned.
///
/// This is the pure core of engine-side global negative RHS sampling, factored
/// out of [`DiskGraphEngine::sample_runtime_batch`] so it can be unit-tested
/// without a seed-table-on-disk. Temporal validity is supplied by the caller as
/// a **validator-factory** `make_validator: Fn() -> V` returning a fresh
/// per-row validator `V: FnMut(candidate_id, anchor_ts) -> Result<bool>` — the
/// engine wires `V` to `TableReader::is_node_temporally_valid` (each validator
/// owns its own `RowPackDecodeScratch`); tests stub it.
///
/// ## Parallelism
/// The positive rows are fully independent — per-`(row, m)` RNG seeds, per-row
/// anchor, per-row validity memo — so the outer row loop is parallelized across
/// `thread_pool` (the engine's per-instance rayon pool, NOT the global pool, to
/// stay DataLoader-fork safe). Each row-task calls `make_validator()` to obtain
/// its OWN validator (and thus its own scratch), keeping the work
/// contention-free. The inner `m` loop and the resample sequence stay
/// sequential (the draw→check→redraw chain is inherently ordered). Per-`(row,m)`
/// seeding is independent of execution order, so when `rnd_seed` is `Some`, the
/// output is **bit-identical** to a sequential run for that base seed; row-order
/// `collect` preserves the anchor-major layout. `None` ⇒ `thread_rng` (random).
///
/// Contract:
/// - `seed_nodes[rhs_node_type]` must already hold exactly `b` positives
///   (asserted) and ends length `b + b * amount_m`.
/// - Negatives are appended **anchor-major**: outer loop over positive rows
///   `i`, inner over `m`, i.e. `[row0's M, row1's M, ...]`.
/// - When `seed_ts` is `Some`, each row's anchor is replicated `amount_m` times
///   into `seed_ts[rhs_node_type]` in lockstep so the two stay row-aligned.
/// - When `rnd_seed` is `Some`, the draw is deterministic: a per-`(row, m)`
///   `ChaCha8Rng` is seeded from it, so the same base reproduces the same
///   negatives. `None` draws from `thread_rng` (random per run).
/// - Other seed types (lhs) are left untouched.
/// - `amount_m == 0` is a no-op (returns an empty mask).
///
/// Returns a `neg_mask: Vec<bool>` of length `b * amount_m`, anchor-major and
/// parallel to the appended negatives:
/// - a candidate that was accepted (temporally valid, or `!do_temporal` so no
///   check ran) → `true`;
/// - resample budget exhausted without finding a valid candidate → `false`,
///   while the last-drawn id is still pushed as a **dense placeholder** (a real
///   RHS id, GNN-encoded downstream) so the `[B, M]` layout stays dense. The
///   mask lets the loss ignore the slot.
///
/// When `!do_temporal` (no anchors / non-temporal RHS), the validity loop never
/// runs and every slot is `true`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_global_negatives<MV, V>(
    seed_nodes: &mut HashMap<String, Vec<u64>>,
    seed_ts: &mut Option<HashMap<String, Vec<i64>>>,
    rhs_node_type: &str,
    amount_m: usize,
    rnd_seed: Option<u64>,
    lo: u64,
    hi: u64,
    b: usize,
    thread_pool: &rayon::ThreadPool,
    make_validator: MV,
) -> Result<Vec<bool>>
where
    MV: Fn() -> V + Sync,
    V: FnMut(u64, i64) -> Result<bool>,
{
    if amount_m == 0 {
        return Ok(Vec::new());
    }

    let rhs_ids = seed_nodes.get_mut(rhs_node_type).ok_or_else(|| {
        EngineError::InvalidRequest(format!(
            "negative sampling: rhs seed type `{rhs_node_type}` absent from seed_nodes"
        ))
    })?;
    if rhs_ids.len() != b {
        return Err(EngineError::InvalidRequest(format!(
            "negative sampling: seed_nodes[`{rhs_node_type}`] has {} positives, expected B={b}",
            rhs_ids.len()
        )));
    }

    // The rhs anchor array (one per positive) if this is a temporal task.
    let rhs_anchors: Option<Vec<i64>> = seed_ts
        .as_ref()
        .and_then(|m| m.get(rhs_node_type))
        .cloned();
    if let Some(anchors) = rhs_anchors.as_ref() {
        if anchors.len() != b {
            return Err(EngineError::InvalidRequest(format!(
                "negative sampling: seed_ts[`{rhs_node_type}`] has {} entries, expected B={b}",
                anchors.len()
            )));
        }
    }

    // Parallelize across positive rows on the engine's per-instance pool. Each
    // row is independent (own RNG seeds, anchor, validity memo, validator). The
    // `map(..).collect()` preserves row index order, so the flattened output is
    // anchor-major and bit-identical to a sequential run for a given rnd_seed.
    let rhs_anchors_ref = rhs_anchors.as_ref();
    let per_row: Vec<Result<(Vec<u64>, Vec<bool>)>> = thread_pool.install(|| {
        (0..b)
            .into_par_iter()
            .map(|i| {
                let anchor_i = rhs_anchors_ref.map(|a| a[i]);
                append_global_negatives_row(
                    i,
                    anchor_i,
                    amount_m,
                    rnd_seed,
                    lo,
                    hi,
                    make_validator(),
                )
            })
            .collect()
    });

    let mut new_negs: Vec<u64> = Vec::with_capacity(b * amount_m);
    let mut new_neg_ts: Vec<i64> = Vec::with_capacity(b * amount_m);
    // Anchor-major mask, parallel to `new_negs`: true = accepted (use in loss),
    // false = budget exhausted (dense placeholder, ignore in loss).
    let mut neg_mask: Vec<bool> = Vec::with_capacity(b * amount_m);

    for (i, row) in per_row.into_iter().enumerate() {
        let (row_negs, row_mask) = row?;
        if let Some(anchors) = rhs_anchors_ref {
            new_neg_ts.extend(std::iter::repeat_n(anchors[i], row_negs.len()));
        }
        new_negs.extend(row_negs);
        neg_mask.extend(row_mask);
    }

    // Re-borrow mutably to extend (the parallel section only needed `&rhs_anchors`).
    let rhs_ids = seed_nodes
        .get_mut(rhs_node_type)
        .expect("rhs seed type present (checked above)");
    rhs_ids.reserve(b * amount_m);
    rhs_ids.extend_from_slice(&new_negs);
    if rhs_anchors.is_some() {
        if let Some(map) = seed_ts.as_mut() {
            if let Some(ts) = map.get_mut(rhs_node_type) {
                ts.extend_from_slice(&new_neg_ts);
            }
        }
    }

    Ok(neg_mask)
}

/// Draw the `amount_m` negatives for a single positive row `i`. Pure and
/// self-contained: owns its `validity_memo` and `validator`, so it is the unit
/// of parallelism in [`append_global_negatives`]. Returns `(negs, mask)`, both
/// length `amount_m`. When `rnd_seed` is `Some`, the RNG is seeded per-`(row, m)`
/// so the result is independent of which thread runs it; `None` ⇒ `thread_rng`.
fn append_global_negatives_row<V>(
    i: usize,
    anchor_i: Option<i64>,
    amount_m: usize,
    rnd_seed: Option<u64>,
    lo: u64,
    hi: u64,
    mut validator: V,
) -> Result<(Vec<u64>, Vec<bool>)>
where
    V: FnMut(u64, i64) -> Result<bool>,
{
    use rand::{Rng, SeedableRng};

    let mut negs: Vec<u64> = Vec::with_capacity(amount_m);
    let mut mask: Vec<bool> = Vec::with_capacity(amount_m);
    // Memoize validity by candidate id so resampled repeats are free. The anchor
    // is fixed within a row, so a per-row memo is sound (no clearing needed).
    let mut validity_memo: HashMap<u64, bool> = HashMap::new();

    for m in 0..amount_m {
        // `Some(base)` ⇒ deterministic per-(row, m) RNG: a stable mix that varies
        // per draw and is independent of execution order. `None` ⇒ `thread_rng`.
        let mut chacha;
        let mut thread;
        let rng: &mut dyn rand::RngCore = match rnd_seed {
            Some(base) => {
                let rng_seed = base
                    .wrapping_add(i as u64)
                    .wrapping_add((m as u64) << 32);
                chacha = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
                &mut chacha
            }
            None => {
                thread = rand::thread_rng();
                &mut thread
            }
        };

        let mut cand = rng.gen_range(lo..=hi);
        // No temporal gating (non-temporal RHS) → always accepted.
        let mut accepted = true;
        if let Some(anchor) = anchor_i {
            // Bounded resampling: try the initial draw + up to K more. On
            // exhaustion the slot is masked (accepted=false) and the
            // last-drawn id is kept as a dense placeholder.
            let mut attempts = 0;
            loop {
                let valid = match validity_memo.get(&cand) {
                    Some(&v) => v,
                    None => {
                        let v = validator(cand, anchor)?;
                        validity_memo.insert(cand, v);
                        v
                    }
                };
                if valid {
                    accepted = true;
                    break;
                }
                if attempts >= NEG_RESAMPLE_BUDGET {
                    // Budget exhausted with no valid candidate: mask the
                    // slot but keep the last draw as a dense placeholder.
                    accepted = false;
                    break;
                }
                attempts += 1;
                cand = rng.gen_range(lo..=hi);
            }
        }
        negs.push(cand);
        mask.push(accepted);
    }

    Ok((negs, mask))
}

/// Hides `node_type` nodes whose pseudo time (`times[node_id]`) exceeds
/// `seed_time`, on every hop of the request.
#[derive(Clone, Copy, Debug)]
pub struct PseudoTimeFilter<'a> {
    pub node_type: &'a str,
    pub times: &'a [u8],
    pub seed_time: u8,
}

#[derive(Clone, Copy)]
struct NodeCutoff<'a> {
    type_id: usize,
    times: &'a [u8],
    seed_time: u8,
}

impl NodeCutoff<'_> {
    #[inline]
    fn visible(&self, node_id: u64) -> bool {
        self.times
            .get(node_id as usize)
            .is_some_and(|t| *t <= self.seed_time)
    }
}

/// The core engine responsible for executing graph sampling requests.
pub struct DiskGraphEngine {
    /// Shard readers mapped by dense Node Type ID.
    pub readers: Vec<Vec<ShardHandle>>,
    /// Map from dense Relation ID to DstTypeID.
    pub relation_info: Vec<usize>,
    /// Sort key decoders for each relation.
    pub decoders: Vec<Option<SortKeyDecoder>>,
    /// Node feature decoders for each type.
    pub node_decoders: Vec<Option<Arc<RowPackDecoder>>>,
    /// Precomputed Arrow schemas for each node type.
    pub node_schemas: Vec<Option<Arc<arrow_schema::Schema>>>,
    /// Map from node type name to node type ID.
    pub node_type_to_id: HashMap<String, usize>,
    /// Map from relation name to edge type ID.
    pub rel_name_to_id: HashMap<String, usize>,
    /// Map from dense ID to node type name.
    pub id_to_node_type: Vec<String>,
    /// Map from dense ID to relation name.
    pub id_to_rel_name: Vec<String>,
    /// Map from dense Relation ID to SrcTypeID.
    pub relation_src_type: Vec<usize>,
    /// Forward↔reverse relation pairs: rel_id -> reverse_rel_id.
    /// Built from the `*_src_to_dst` ↔ `*_dst_to_src` naming convention.
    pub reverse_rel_map: Vec<Option<usize>>,
    /// Feature batch target size in bytes.
    pub target_batch_bytes: usize,
    /// Feature batch target size in num rows.
    pub target_batch_rows: usize,
    /// Global timestamp unit for the index.
    pub timestamp_unit: diskgraph_core::time::TimeUnit,
    /// Per-source-table mapping from a shard segment's local `relation_id`
    /// to the merged catalog's `rel_id`. Indexed by
    /// `[src_type_id][local_rel_id] = merged_rel_id`. Required because each
    /// table's `RelationLayout.relation_id` is assigned positionally within
    /// the SUB-INDEX that produced it — parent and extension start their
    /// counts at 0 independently. Without translation, a seed sub-index's
    /// `local rel 0 = seed→users` collides with the parent's
    /// `local rel 0 = orders→users` and the engine reads `relation_info[0]`
    /// for the WRONG relation. `u16::MAX` is the "no mapping" sentinel.
    pub local_to_merged_rel: Vec<Vec<u16>>,
    /// Optional global random seed for deterministic neighbor sampling.
    pub random_seed: Option<u64>,
    /// Local rayon thread pool for parallel sampling.
    /// Uses a per-instance pool (not the global pool) to avoid
    /// deadlocks when PyTorch DataLoader forks worker processes.
    pub thread_pool: rayon::ThreadPool,
}

struct CompiledSamplingPlan {
    layers: Vec<CompiledLayerSpec>,
}

struct CompiledLayerSpec {
    fanouts: HashMap<usize, FanoutSpec>,
    default_fanout: Option<FanoutSpec>,
}

/// A discovered edge before it is formally committed to the SamplingState.
struct RawEdge {
    src_local_id: u32,
    dst_global_id: u64,
    dst_type_id: usize,
    rel_id: u16,
    seed_index: u32,
    /// Root seed type id, propagated from the source node so the destination
    /// node can resolve its timestamp anchor against the correct per-type array.
    seed_type_id: u32,
    swap_edge_direction: bool,
}

impl DiskGraphEngine {
    /// Build an engine directly from an already-opened `GraphIndexReader`.
    /// This is the canonical constructor; `load(path, ...)` is a thin
    /// wrapper that opens the reader first and forwards here.
    ///
    /// The engine adopts the reader's `Arc<ShardReader>` handles
    /// (cheap clone), uses the reader's `merged_edge_relations()` to
    /// build its relation table, and derives `node_type_to_id`
    /// deterministically from the sorted iteration of `reader.tables()`.
    /// No re-discovery, no duplicate mmap. Works identically for a
    /// single-index reader (parent only) and a merged reader (parent + tt).
    pub fn from_reader(
        reader: Arc<diskgraph_index::graph::GraphIndexReader>,
        random_seed: Option<u64>,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        let sorted_names: Vec<String> = reader.table_names();
        let mut readers_map: HashMap<String, Vec<ShardHandle>> = HashMap::new();
        let mut node_decoders_map: HashMap<String, RowPackDecoder> = HashMap::new();

        for name in &sorted_names {
            let table = reader.table(name).ok_or_else(|| {
                EngineError::Internal(format!("reader missing table {name}"))
            })?;
            let mut shards = Vec::with_capacity(table.shard_count());
            for s in 0..table.shard_count() {
                let arc = table.shard_arc(s).unwrap();
                let (min_id, max_id) = table.shard_id_range(s).unwrap();
                if min_id > max_id {
                    continue;
                }
                shards.push(ShardHandle {
                    reader: arc,
                    min_id,
                    max_id,
                });
            }
            readers_map.insert(name.clone(), shards);

            if let Some(decoder) = table.node_data_decoder() {
                node_decoders_map.insert(name.clone(), decoder.clone());
            }
        }

        let merged = reader.merged_edge_relations();
        let mut relation_info_map: HashMap<usize, (String, String, String)> = HashMap::new();
        let mut decoders_map: HashMap<usize, SortKeyDecoder> = HashMap::new();
        let mut reverse_pairs: HashMap<usize, usize> = HashMap::new();

        for (rel_id, rel) in merged.iter().enumerate() {
            relation_info_map.insert(
                rel_id,
                (rel.name.clone(), rel.src_type.clone(), rel.dst_type.clone())
            );
            if let Some(dest_sort_key) = rel.dest_sort_key.as_ref() {
                decoders_map.insert(rel_id, SortKeyDecoder::new(dest_sort_key));
            }
            if let Some(partner) = rel.reverse_of.as_deref() {
                if let Some(partner_id) = merged.iter().position(|r| r.name == partner) {
                    reverse_pairs.insert(rel_id, partner_id);
                }
            }
        }

        let timestamp_unit = reader.metadata().timestamp_unit;

        Ok(Self::new(
            readers_map,
            relation_info_map,
            decoders_map,
            node_decoders_map,
            if reverse_pairs.is_empty() { None } else { Some(reverse_pairs) },
            timestamp_unit,
            random_seed,
            num_threads,
        ))
    }

    pub fn load(path: &Path, random_seed: Option<u64>, num_threads: Option<usize>) -> Result<Self> {

        let metadata_path = path.join("index_metadata.json");
        let metadata_bytes = std::fs::read(&metadata_path)
            .map_err(|e| EngineError::Internal(format!("failed to read metadata: {}", e)))?;
        let metadata: diskgraph_core::metadata::DatasetMetadata =
            serde_json::from_slice(&metadata_bytes)
                .map_err(|e| EngineError::Internal(format!("failed to parse metadata: {}", e)))?;
        println!(
            "Loading graph metadata for {} node types...",
            metadata.node_types.len()
        );

        // Per-table layouts with LOCAL relation_ids: each table's outgoing
        // relations get positional ids 0..n in the filtered list. This
        // matches the on-disk encoding (ingest writes segments in this same
        // filtered order) AND the merged-reader path's semantics, so
        // `local_to_merged_rel` translation works uniformly across both
        // entry points. The legacy path used to compute unfiltered global
        // ids here, which broke for any sampler that traversed more than
        // one node type's relations.
        let mut readers = HashMap::new();
        let mut node_decoders_map = HashMap::new();

        for node_type in &metadata.node_types {
            let per_table_relations: Vec<_> = metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.src_type == node_type.name)
                .cloned()
                .collect();
            let type_layouts =
                diskgraph_index::builder::compute_relation_layouts(
                    &per_table_relations,
                    &metadata.node_types,
                )
                .map_err(|e| {
                    EngineError::Internal(format!("failed to compute layouts: {}", e))
                })?;
            let type_layouts = Arc::new(type_layouts);

            // Ingest currently writes node shard directories in lowercase
            // (`nodes/<table_name_lower>/...`), while metadata preserves the
            // original table name case. Support both to keep runtime robust.
            let node_dir_exact = path.join("nodes").join(&node_type.name);
            let node_dir_lower = path.join("nodes").join(node_type.name.to_lowercase());
            let node_dir = if node_dir_exact.exists() {
                node_dir_exact
            } else if node_dir_lower.exists() {
                node_dir_lower
            } else {
                // Keep old behavior for error surfaces if neither exists.
                path.join("nodes").join(&node_type.name)
            };

            let mut shards = Vec::new();
            for (i, shard_meta) in node_type.shards.iter().enumerate() {
                // Skip empty shards. Ingest writes (min_id=1, max_id=0) as
                // a sentinel for shards that received no data. Including
                // them in the shard vec breaks binary search in find_shard.
                if shard_meta.min_id > shard_meta.max_id {
                    continue;
                }
                let shard_dir = node_dir.join(format!("shard-{:05}", i));
                if shard_dir.exists() {
                    let reader = ShardReader::open(
                        &shard_dir,
                        type_layouts.clone(),
                        node_type.has_node_data,
                    )
                    .map_err(|e| EngineError::Internal(format!("failed to open shard: {}", e)))?;

                    shards.push(ShardHandle {
                        reader: Arc::new(reader),
                        min_id: shard_meta.min_id,
                        max_id: shard_meta.max_id,
                    });
                }
            }
            readers.insert(node_type.name.clone(), shards);

            // Load RowPack Decoder if present
            if let Some(config) = &node_type.node_data_compression {
                let ctx_path = path.join(&config.context_path);
                let ctx_bytes = std::fs::read(&ctx_path).map_err(|e| {
                    EngineError::Internal(format!("failed to read rowpack ctx: {}", e))
                })?;
                let ctx =
                    diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes).map_err(|e| {
                        EngineError::Internal(format!("failed to parse rowpack ctx: {}", e))
                    })?;
                let decoder = RowPackDecoder::new(ctx).map_err(|e| {
                    EngineError::Internal(format!("failed to create decoder: {}", e))
                })?;
                node_decoders_map.insert(node_type.name.clone(), decoder);
            }
        }

        let mut relation_info = HashMap::new();
        let mut decoders = HashMap::new();
        for (rel_id, rel) in metadata.edge_relations.iter().enumerate() {
            relation_info.insert(
                rel_id,
                (rel.name.clone(), rel.src_type.clone(), rel.dst_type.clone()),
            );

            if let Some(meta) = &rel.dest_sort_key {
                let decoder = SortKeyDecoder::new(meta);
                decoders.insert(rel_id, decoder);
            }
        }

        // Build explicit reverse pairs from metadata.
        let mut reverse_pairs: HashMap<usize, usize> = HashMap::new();
        for (rel_id, rel) in metadata.edge_relations.iter().enumerate() {
            if let Some(ref partner_name) = rel.reverse_of {
                if let Some(partner_id) = metadata
                    .edge_relations
                    .iter()
                    .position(|r| r.name == *partner_name)
                {
                    reverse_pairs.insert(rel_id, partner_id);
                }
            }
        }
        let reverse_pairs_opt = if reverse_pairs.is_empty() {
            None
        } else {
            Some(reverse_pairs)
        };

        println!("Graph loaded. Ready to serve.");
        Ok(Self::new(
            readers,
            relation_info,
            decoders,
            node_decoders_map,
            reverse_pairs_opt,
            metadata.timestamp_unit,
            random_seed,
            num_threads,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        readers_map: HashMap<String, Vec<ShardHandle>>,
        relation_info_map: HashMap<usize, (String, String, String)>,
        decoders_map: HashMap<usize, SortKeyDecoder>,
        node_decoders_map: HashMap<String, RowPackDecoder>,
        reverse_pairs: Option<HashMap<usize, usize>>,
        timestamp_unit: diskgraph_core::time::TimeUnit,
        random_seed: Option<u64>,
        num_threads: Option<usize>,
    ) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1)
        });
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to create rayon thread pool");
        let mut node_type_to_id = HashMap::new();
        let mut id_to_node_type = Vec::new();
        let mut readers = Vec::new();
        let mut node_decoders = Vec::new();
        let mut node_schemas = Vec::new();

        // Sort keys to ensure deterministic ID assignment
        let mut sorted_types: Vec<_> = readers_map.keys().cloned().collect();
        sorted_types.sort();

        for (idx, name) in sorted_types.iter().enumerate() {
            node_type_to_id.insert(name.clone(), idx);
            id_to_node_type.push(name.clone());

            // Populate decoders/arrow schemas aligned with ID
            if let Some(decoder) = node_decoders_map.get(name) {
                let decoder_arc = Arc::new(decoder.clone());
                node_decoders.push(Some(decoder_arc.clone()));

                let context = decoder_arc.context();
                let col_indices: Vec<usize> = (0..context.columns.len()).collect();
                let dummy_batch = decoder_arc
                    .decode_batch_to_arrow::<&[u8]>(&[], &col_indices)
                    .expect("failed to generate precomputed schema");
                node_schemas.push(Some(dummy_batch.schema()));
            } else {
                node_decoders.push(None);
                node_schemas.push(None);
            }
        }

        // Populate readers vector
        readers.resize_with(sorted_types.len(), Vec::new);
        for (name, shards) in readers_map {
            if let Some(&idx) = node_type_to_id.get(&name) {
                readers[idx] = shards;
            }
        }

        // Remap relation info to use Type IDs
        let mut rel_name_to_id = HashMap::new();
        let mut id_to_rel_name = Vec::new();

        let max_rel_id = relation_info_map.keys().max().copied().unwrap_or(0);
        id_to_rel_name.resize(max_rel_id + 1, String::new());
        // Use usize::MAX as sentinel for invalid/unmapped relations
        let mut relation_info = vec![usize::MAX; max_rel_id + 1];
        let mut relation_src_type = vec![usize::MAX; max_rel_id + 1];
        let mut decoders: Vec<Option<SortKeyDecoder>> = vec![None; max_rel_id + 1];

        for (rel_id, (rel_name, src_type, dst_type)) in &relation_info_map {
            if let (Some(&src_id), Some(&dst_id)) =
                (node_type_to_id.get(src_type), node_type_to_id.get(dst_type))
            {
                rel_name_to_id.insert(rel_name.clone(), *rel_id);
                id_to_rel_name[*rel_id] = rel_name.clone();
                relation_info[*rel_id] = dst_id;
                relation_src_type[*rel_id] = src_id;

                if let Some(decoder) = decoders_map.get(rel_id) {
                    decoders[*rel_id] = Some(decoder.clone());
                }
            }
        }

        // Per-src-table local→merged rel-id mapping. Legacy path: relations
        // are filtered per-table when their layouts get computed at ingest
        // time, and per-table local rel_id is the position in that filtered
        // list. To reconstruct that mapping here we iterate the global
        // rel_ids in sorted order (which matches the original metadata
        // ordering since `relation_info_map` was populated from
        // `metadata.edge_relations.iter().enumerate()`) and group by
        // src_type. The Nth global rel_id seen for src_type T becomes
        // local rel_id N for table T.
        let mut sorted_rel_ids: Vec<usize> = relation_info_map.keys().copied().collect();
        sorted_rel_ids.sort();
        let mut local_to_merged_rel: Vec<Vec<u16>> = vec![Vec::new(); sorted_types.len()];
        for &rel_id in &sorted_rel_ids {
            let (_name, src_type, _dst_type) = &relation_info_map[&rel_id];
            if let Some(&src_id) = node_type_to_id.get(src_type) {
                local_to_merged_rel[src_id].push(rel_id as u16);
            }
        }

        // Build forward↔reverse relation pairs.
        // Prefer explicit pairs from metadata; fall back to naming convention.
        let mut reverse_rel_map: Vec<Option<usize>> = vec![None; id_to_rel_name.len()];
        if let Some(pairs) = reverse_pairs {
            for (rel_id, partner_id) in pairs {
                if rel_id < reverse_rel_map.len() && partner_id < reverse_rel_map.len() {
                    reverse_rel_map[rel_id] = Some(partner_id);
                }
            }
        } else {
            // Fallback: naming convention for old indexes without reverse_of.
            for (i, name_i) in id_to_rel_name.iter().enumerate() {
                if name_i.is_empty() || reverse_rel_map[i].is_some() {
                    continue;
                }
                let partner = if let Some(base) = name_i.strip_suffix("_src_to_dst") {
                    format!("{}_dst_to_src", base)
                } else if let Some(base) = name_i.strip_suffix("_dst_to_src") {
                    format!("{}_src_to_dst", base)
                } else {
                    continue;
                };
                if let Some(&j) = rel_name_to_id.get(&partner) {
                    reverse_rel_map[i] = Some(j);
                    reverse_rel_map[j] = Some(i);
                }
            }
        }

        Self {
            readers,
            relation_info,
            decoders,
            node_decoders,
            node_schemas,
            node_type_to_id,
            rel_name_to_id,
            id_to_node_type,
            id_to_rel_name,
            relation_src_type,
            reverse_rel_map,
            target_batch_bytes: get_feature_batch_bytes(),
            target_batch_rows: get_feature_batch_rows(),
            timestamp_unit,
            local_to_merged_rel,
            random_seed,
            thread_pool,
        }
    }

    pub fn get_type_name_mapping(&self) -> TypeNameMapping {
        let edge_relations = self
            .id_to_rel_name
            .iter()
            .enumerate()
            .map(|(rel_id, name)| {
                let src_type = self
                    .relation_src_type
                    .get(rel_id)
                    .and_then(|&id| self.id_to_node_type.get(id))
                    .cloned()
                    .unwrap_or_default();
                let dst_type = self
                    .relation_info
                    .get(rel_id)
                    .and_then(|&id| self.id_to_node_type.get(id))
                    .cloned()
                    .unwrap_or_default();
                EdgeRelationInfo {
                    name: name.clone(),
                    src_type,
                    dst_type,
                }
            })
            .collect();
        TypeNameMapping {
            node_types: self.id_to_node_type.clone(),
            edge_relations,
        }
    }

    pub fn fetch_features_for_type(
        &self,
        node_type_id: usize,
        node_ids: &[u64],
        feature_names: &[String],
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }

        let decoder = self
            .node_decoders
            .get(node_type_id)
            .and_then(|d| d.as_ref())
            .ok_or_else(|| {
                EngineError::InvalidRequest(format!(
                    "no feature data for node type {}",
                    node_type_id
                ))
            })?;

        let context = decoder.context();
        let mut col_indices = Vec::new();
        let select_all = feature_names.is_empty() || feature_names.iter().any(|n| n == "*");

        if select_all {
            // This path materializes ALL columns, so filter reserved columns
            // (e.g. WEIGHT) out of features here too. See
            // `is_reserved_feature_column`.
            col_indices = (0..context.columns.len())
                .filter(|&i| !is_reserved_feature_column(&context.columns[i].name))
                .collect();
        } else {
            for name in feature_names {
                // Defense-in-depth: even if a caller explicitly names the
                // reserved column, never materialize it as a feature.
                if is_reserved_feature_column(name) {
                    continue;
                }
                let idx = context
                    .columns
                    .iter()
                    .position(|c| &c.name == name)
                    .ok_or_else(|| {
                        EngineError::InvalidRequest(format!("unknown feature: {}", name))
                    })?;
                col_indices.push(idx);
            }
        }

        // Send Schema based on selection
        if let Some(full_schema) = self.node_schemas.get(node_type_id).and_then(|s| s.as_ref()) {
            let projected = full_schema
                .project(&col_indices)
                .map_err(|e| EngineError::Internal(format!("schema projection failed: {}", e)))?;
            output.handle_schema(Arc::new(projected))?;
        } else {
            return Err(EngineError::Internal(format!(
                "no schema for type {}",
                node_type_id
            )));
        }

        // Use Bulk Fetch helper
        self.accumulate_features_from_ids_chunked(
            node_type_id,
            node_ids,
            &col_indices,
            decoder,
            output,
        )?;

        Ok(())
    }

    /// Fetch the full, unfiltered rowpack for the seed/stage type.
    ///
    /// Unlike `fetch_features_for_type` (which strips reserved columns), the
    /// seed batch is CONSUMED: `entity_node_id`/`target_node_ids` build the
    /// seed set + labels, `WEIGHT` feeds `task.weight`, and
    /// `project_seed_rowpack` routes `TARGET*` → labels. All columns must be
    /// present.
    fn fetch_seed_rowpack_full(
        &self,
        node_type_id: usize,
        node_ids: &[u64],
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }
        let decoder = self
            .node_decoders
            .get(node_type_id)
            .and_then(|d| d.as_ref())
            .ok_or_else(|| {
                EngineError::InvalidRequest(format!("no feature data for node type {}", node_type_id))
            })?;
        let context = decoder.context();
        // Full rowpack: every column, NO reserved filter.
        let col_indices: Vec<usize> = (0..context.columns.len()).collect();

        if let Some(full_schema) = self.node_schemas.get(node_type_id).and_then(|s| s.as_ref()) {
            let projected = full_schema
                .project(&col_indices)
                .map_err(|e| EngineError::Internal(format!("schema projection failed: {}", e)))?;
            output.handle_schema(Arc::new(projected))?;
        } else {
            return Err(EngineError::Internal(format!(
                "no schema for type {}",
                node_type_id
            )));
        }

        self.accumulate_features_from_ids_chunked(
            node_type_id,
            node_ids,
            &col_indices,
            decoder,
            output,
        )?;

        Ok(())
    }

    pub fn find_shard(&self, shards: &[ShardHandle], global_id: u64) -> Result<usize> {
        shards
            .binary_search_by(|s| {
                if global_id < s.min_id {
                    std::cmp::Ordering::Greater
                } else if global_id > s.max_id {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map_err(|_| {
                EngineError::DataCorruption(format!("node id {} not found in any shard", global_id))
            })
    }

    /// Phase 3 convenience: turn a batch of seed-table row ids into the
    /// `RuntimeBatch` shape PR-28765's `runtime_batch_to_hetero` consumes.
    ///
    /// Pipeline (single engine over a merged parent + tt reader):
    ///   1. Walk the seed sub-index's `edges.bin` for each row id to recover
    ///      the parent entity row ids that the FK pre-resolved at Phase 1
    ///      ingest. These become `task.seed_node_ids` and the actual sampling
    ///      seeds. Seed-namespace edges are never traversed at runtime (we
    ///      seed directly at the parent entity type), so the seed namespace
    ///      never appears as topology in the emitted dict.
    ///   2. Read the seed rowpack for those rows once → `task.target`,
    ///      `task.seed_timestamps` (if a TIMESTAMP column exists), etc.
    ///   3. Call `sample()` with the resolved parent ids + seed timestamps
    ///      as the sampling anchors. RuntimeBatchOutput collates edges,
    ///      node_ids, and batch_dict per channel.
    ///   4. Stitch task fields onto the result.
    ///
    /// Seed-src rel ids are still registered with the output handler as a
    /// belt-and-suspenders filter: even if someone later changes the entry
    /// point to seed at the seed namespace, seed-src edges still get
    /// dropped at the output layer.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_runtime_batch(
        &self,
        reader: &diskgraph_index::graph::GraphIndexReader,
        stage_table: &str,
        batch_row_ids: &[u64],
        dataset_id: diskgraph_core::dataset::DatasetId,
        plan: &SamplingPlan,
        flags: &SamplingFlags,
        negatives: Option<&diskgraph_core::sampling::NegativeSamplingSpec>,
        // Seed set from the `entity_node_id` column; required (`None` errors).
        entity_node_type: Option<&str>,
        // LP positives from `target_node_ids`, emitted as labels only — never
        // rooted as seeds (no message-passing leak).
        target_node_type: Option<&str>,
        use_seed_timestamps: bool,
        // NodeRange seed mode (two-tower MIPS RHS pass). When `Some`, the engine
        // does NOT seed from a stage table: `batch_row_ids` ARE the RHS/target
        // catalog global node-ids directly, and every seed is anchored at the
        // FIXED `seed_time`. The entity/target FK-free resolution AND the
        // negatives blocks are BYPASSED entirely (NodeRange emits embeddings for
        // the RHS tower — there are no positives, labels, or negatives), then the
        // request falls through to the SAME `self.sample()` / `finalize()` path.
        // `None` (default) preserves the stage-table seeding behavior byte-for-byte.
        node_range: Option<NodeRangeSeed<'_>>,
        pseudo_time: Option<PseudoTimeFilter<'_>>,
    ) -> Result<crate::output::RuntimeBatch> {
        use crate::output::{InMemoryOutput, OutputHandler, RuntimeBatchOutput};
        use std::collections::HashSet;

        let cutoff = pseudo_time.map(|f| self.node_cutoff(f)).transpose()?;

        // NodeRange seed mode: bypass stage/entity/target resolution AND
        // negatives, seed `[0, num_rhs)`-style ids at a fixed seed_time, then
        // reuse the shared sample/finalize path.
        if let Some(nr) = node_range {
            return self.sample_node_range_batch(
                reader,
                batch_row_ids,
                dataset_id,
                plan,
                flags,
                nr,
                cutoff,
            );
        }

        // 1. Resolve seed type id from the merged catalog.
        let seed_type_id = *self.node_type_to_id.get(stage_table).ok_or_else(|| {
            EngineError::InvalidRequest(format!(
                "stage_table `{stage_table}` not in engine catalog"
            ))
        })?;

        // The stage table is namespaced as `<tt_id>/<bare_table_name>`.
        let tt_id = stage_table.split('/').next().unwrap_or("");

        // 1b. Identify ALL of this tt's stage seed-table node types (train/val/
        //     test/predict), not just the stage currently being sampled. The
        //     merged TT catalog contains one node type + one `seed→ENTITY`
        //     relation per stage; the pybind emits every catalog edge type
        //     (incl. all-zero ones), so the sibling stages would leak their
        //     encoder-less seed namespace into the kumo adapter unless excluded
        //     from emitted topology here too.
        //
        //     Source of the set (structural): `merge_extension_into`
        //     (diskgraph-index) namespaces ONLY the loaded extension's tables
        //     and edge src_types under `<tt_id>/`; base-graph node types
        //     (users/products/...) are never namespaced, and exactly one
        //     extension is layered per reader. So every node type whose name
        //     starts with `"<tt_id>/"` is a stage seed table of this tt — a
        //     scoped, structurally-grounded allowlist (no loose `_seeds`
        //     substring matching, no hardcoded stage-suffix enumeration).
        let seed_prefix = format!("{tt_id}/");
        let seed_type_ids: HashSet<usize> = self
            .id_to_node_type
            .iter()
            .enumerate()
            .filter(|(_, name)| name.starts_with(&seed_prefix))
            .map(|(id, _)| id)
            .collect();

        // Anchors come from the `entity_node_id` column, so the type is required.
        let entity_node_type = entity_node_type.ok_or_else(|| {
            EngineError::InvalidRequest(format!(
                "stage table `{stage_table}`: `entity_node_type` is required \
                 (resolution mode) — FK-edge seed mode was removed"
            ))
        })?;

        // Keep seed-table edges out of emitted topology (normally empty in
        // resolution mode; kept for robustness).
        let mut seed_src_rel_ids: HashSet<u16> = HashSet::new();
        for (rel_id_usize, src_id) in self.relation_src_type.iter().enumerate() {
            if !seed_type_ids.contains(src_id) {
                continue;
            }
            seed_src_rel_ids.insert(rel_id_usize as u16);
        }

        // 3. Build rel_id_to_names for the collator.
        let mut rel_id_to_names: HashMap<u16, (String, String, String)> = HashMap::new();
        for (rel_id_usize, name) in self.id_to_rel_name.iter().enumerate() {
            let src_id = self.relation_src_type[rel_id_usize];
            let dst_id = self.relation_info[rel_id_usize];
            if src_id >= self.id_to_node_type.len() || dst_id >= self.id_to_node_type.len() {
                continue;
            }
            rel_id_to_names.insert(
                rel_id_usize as u16,
                (
                    self.id_to_node_type[src_id].clone(),
                    name.clone(),
                    self.id_to_node_type[dst_id].clone(),
                ),
            );
        }

        // 4. Read seed rowpack for the batch → seed RecordBatch + (optional)
        //    seed_timestamps. We collect into an in-memory output, then concat
        //    the per-chunk batches into one. This is fetched FIRST (it only
        //    depends on `seed_type_id` + `batch_row_ids`) so the FK-free
        //    resolution path can read the `entity_node_id` / `target_node_ids`
        //    columns off it to build the seed set + labels.
        // Unfiltered fetch: the consumed seed batch must retain reserved columns
        // (see `fetch_seed_rowpack_full`).
        let mut seed_feat_out = InMemoryOutput::new();
        self.fetch_seed_rowpack_full(seed_type_id, batch_row_ids, &mut seed_feat_out)?;
        let seed_target_batch = if seed_feat_out.feature_batches.is_empty() {
            None
        } else {
            let schema = seed_feat_out.schema.clone().ok_or_else(|| {
                EngineError::Internal("seed feature schema missing after fetch".into())
            })?;
            let combined = arrow_select::concat::concat_batches(
                &schema,
                seed_feat_out.feature_batches.iter(),
            )
            .map_err(|e| EngineError::Internal(format!("concat seed batches: {e}")))?;
            Some(combined)
        };

        // 4b. Orphan invariant: orphans are dropped at ingest, so `entity_node_id`
        //     is never null here. O(1) guard (arrow precomputes `null_count`).
        if let Some(rb) = seed_target_batch.as_ref() {
            if let Ok(idx) = rb.schema().index_of("entity_node_id") {
                if rb.column(idx).null_count() > 0 {
                    return Err(EngineError::DataCorruption(format!(
                        "seed table `{stage_table}`: null `entity_node_id` survived \
                         ingest (stale/corrupt extension)"
                    )));
                }
            }
        }

        // 5. Build the seed set from the `entity_node_id` column.
        let mut seed_nodes: HashMap<String, Vec<u64>> = HashMap::new();

        let rb = seed_target_batch.as_ref().ok_or_else(|| {
            EngineError::DataCorruption(format!(
                "resolution mode: seed table `{stage_table}` produced no rowpack batch \
                 (need `entity_node_id` column)"
            ))
        })?;
        let idx = rb
            .schema()
            .fields()
            .iter()
            .position(|f| f.name() == "entity_node_id")
            .ok_or_else(|| {
                EngineError::DataCorruption(format!(
                    "resolution mode: seed table `{stage_table}` rowpack has no \
                     `entity_node_id` column"
                ))
            })?;
        let col = rb.column(idx);
        let ids: Vec<u64> = if let Some(a) =
            col.as_any().downcast_ref::<arrow_array::UInt64Array>()
        {
            (0..a.len()).map(|i| a.value(i)).collect()
        } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
            (0..a.len()).map(|i| a.value(i) as u64).collect()
        } else {
            return Err(EngineError::DataCorruption(format!(
                "resolution mode: `entity_node_id` column has unsupported dtype {:?} \
                 (expected UInt64/Int64)",
                col.data_type()
            )));
        };
        seed_nodes.insert(entity_node_type.to_string(), ids);

        if seed_nodes.is_empty() {
             return Err(EngineError::DataCorruption(format!(
                 "seed table `{stage_table}` batch resolved no seed nodes"
             )));
        }

        let seed_timestamps: Option<Vec<i64>> = if use_seed_timestamps {
            seed_target_batch.as_ref().and_then(|rb| {
                let idx = rb.schema().fields().iter().position(|f| f.name() == "TIMESTAMP")?;
                let col = rb.column(idx);
                col.as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                    .or_else(|| {
                        col.as_any()
                            .downcast_ref::<arrow_array::TimestampSecondArray>()
                            .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                    })
            })
        } else {
            None
        };

        // 6. Build the sampling request: seed at the parent entity type(s)
        //    with the resolved ids, anchored at the seed timestamps when
        //    present (non-temporal tasks just pass None).
        let mut request_seed_ts: Option<HashMap<String, Vec<i64>>> = seed_timestamps.as_ref().map(|ts| {
            let mut m = HashMap::new();
            for entity_type_name in seed_nodes.keys() {
                m.insert(entity_type_name.clone(), ts.clone());
            }
            m
        });

        // 6b. Global negative RHS injection. Snapshot the primary-entity ids
        //     BEFORE mutating `seed_nodes` so `task.seed_node_ids` stays the
        //     positives-only view (negatives are extra sampling seeds, not
        //     task inputs). When `negatives` is None the snapshot equals the
        //     pre-injection value, so behavior is byte-identical to before.
        let primary_ids_snapshot = seed_nodes.values().next().cloned();
        // Mask over the appended negatives (anchor-major, parallel to the
        // `[B:B+B*M]` rhs slots). Empty when no negative sampling was requested.
        let mut neg_mask: Vec<bool> = Vec::new();

        // Two-tower positive rooting (FK-free design §S4). When engine negatives
        // are requested (`negatives.is_some()`) AND a target type is set, the
        // two-tower model needs ONE positive per anchor ROOTED as an rhs seed so
        // the RHS tower can GNN-encode it. The FK-free label path left
        // `seed_nodes[rhs]` empty (positives are labels, never edges), so root the
        // chosen positives as the FIRST B entries of `seed_nodes[rhs]` BEFORE
        // `append_global_negatives` runs — that gives `seed_nodes[rhs] = [B pos]`
        // so the `len == B` assert holds, then negatives append B*M, ending
        // `[B pos, B*M neg]` anchor-major. The positives are read off the resolved
        // `target_node_ids: list<i64>` column (already node-ids → IDENTITY
        // resolve closure, no MPH). `pair_from_array` picks one per row from
        // `spec.rnd_seed` (`None` ⇒ `thread_rng`, fresh per epoch — the
        // training/prod default; `Some` ⇒ deterministic for tests). This is the
        // ONLY model that roots + uses engine negatives; shallow/id/hybrid stay on
        // the label-only path (`target_node_type` set, `negatives = None`).
        let mut pair_from_array_mask: Option<Vec<bool>> = None;
        let mut pair_from_array_offsets: Vec<Option<usize>> = Vec::new();
        if let (Some(spec), Some(_tn_type)) = (negatives, target_node_type) {
            // Only root when the rhs slot is empty (the FK-free label path always
            // leaves it empty; a future FK-edge path that pre-seeds rhs is left
            // untouched, byte-for-byte).
            let rhs_already_seeded = seed_nodes
                .get(&spec.rhs_node_type)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if !rhs_already_seeded {
                let rb = seed_target_batch.as_ref().ok_or_else(|| {
                    EngineError::DataCorruption(format!(
                        "two-tower resolution mode: seed table `{stage_table}` produced no \
                         rowpack batch (need `target_node_ids` column to root positives)"
                    ))
                })?;
                let idx = rb
                    .schema()
                    .fields()
                    .iter()
                    .position(|f| f.name() == "target_node_ids")
                    .ok_or_else(|| {
                        EngineError::DataCorruption(format!(
                            "two-tower resolution mode: seed table `{stage_table}` rowpack has \
                             no `target_node_ids` column"
                        ))
                    })?;
                let col = rb.column(idx);
                // The column already holds GLOBAL rhs node-ids → identity closure.
                let pair = pair_from_array(col.as_ref(), spec.rnd_seed, |v| match v {
                    Value::Int(i) => Some(*i as u64),
                    _ => None,
                });
                // Root the B chosen positives as the FIRST B rhs seeds so
                // `len == B` for `append_global_negatives`.
                seed_nodes.insert(spec.rhs_node_type.clone(), pair.chosen_rhs);
                // Align the rhs anchor timestamps (length B) so negatives can be
                // temporally validated against the right anchor per row.
                if let (Some(map), Some(ts)) =
                    (request_seed_ts.as_mut(), seed_timestamps.as_ref())
                {
                    map.insert(spec.rhs_node_type.clone(), ts.clone());
                }
                pair_from_array_mask = Some(pair.mask);
                pair_from_array_offsets = pair.chosen_offset;
            }
        }

        if let Some(spec) = negatives {
            // Resolve the RHS dense type id from the merged catalog.
            let rhs_type_id = *self.node_type_to_id.get(&spec.rhs_node_type).ok_or_else(|| {
                EngineError::InvalidRequest(format!(
                    "negative sampling: rhs_node_type `{}` not in engine catalog",
                    spec.rhs_node_type
                ))
            })?;
            // Dense global-id range across the type's concatenated shards.
            let shards = self.readers.get(rhs_type_id).ok_or_else(|| {
                EngineError::InvalidRequest(format!(
                    "negative sampling: no readers for rhs_node_type `{}`",
                    spec.rhs_node_type
                ))
            })?;
            let lo = shards
                .first()
                .ok_or_else(|| {
                    EngineError::InvalidRequest(format!(
                        "negative sampling: rhs_node_type `{}` has no shards",
                        spec.rhs_node_type
                    ))
                })?
                .min_id;
            let hi = shards
                .last()
                .expect("non-empty shards (checked via first())")
                .max_id;

            let b = batch_row_ids.len();

            // Resolve temporal gating once. `do_temporal` is true only when we
            // have anchors AND the RHS type carries temporal columns.
            let table = reader.table(&spec.rhs_node_type);
            let (c_idx, e_idx) = table
                .as_ref()
                .map(|t| t.temporal_col_indices())
                .unwrap_or((None, None));
            let do_temporal =
                request_seed_ts.is_some() && (c_idx.is_some() || e_idx.is_some());

            // Validator-factory: each parallel row-task calls this to get its
            // OWN validator owning a freshly-allocated `RowPackDecodeScratch`,
            // so scratch never crosses threads. `TableReader` mmap reads are
            // `Send + Sync`, so the captured `Arc<TableReader>` (cheap clone)
            // and `(c_idx, e_idx)` are shared read-only. The `Fn() -> V` is
            // `Sync` as required by `append_global_negatives`.
            let table_for_check = table.clone();
            let make_validator = || {
                let table = table_for_check.clone();
                let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();
                move |cand: u64, anchor: i64| -> Result<bool> {
                    if !do_temporal {
                        return Ok(true);
                    }
                    match table.as_ref() {
                        Some(t) => t
                            .is_node_temporally_valid(cand, anchor, c_idx, e_idx, &mut scratch)
                            .map_err(|e| {
                                EngineError::Internal(format!(
                                    "negative sampling temporal check: {e}"
                                ))
                            }),
                        None => Ok(true),
                    }
                }
            };

            neg_mask = append_global_negatives(
                &mut seed_nodes,
                &mut request_seed_ts,
                &spec.rhs_node_type,
                spec.amount_m,
                spec.rnd_seed,
                lo,
                hi,
                b,
                &self.thread_pool,
                make_validator,
            )?;
        }

        let request = BatchSamplingRequest {
            dataset_id,
            seed_nodes: seed_nodes.clone(),
            seed_timestamps: request_seed_ts,
            seed_timestamp_unit: self.timestamp_unit,
            plan: plan.clone(),
            flags: flags.clone(),
        };

        // 7. Sample + collate.
        let mut state = SamplingState::default();
        let mut output = RuntimeBatchOutput::new(
            "main",
            seed_src_rel_ids,
            rel_id_to_names,
            self.id_to_node_type.clone(),
            // Exclude ALL stage seed namespaces (current + sibling stages) from
            // emitted topology (they're storage types, not model node types) —
            // symmetric with the seed-src edge filtering above.
            seed_type_ids,
        );
        self.sample_with_cutoff(&request, &mut state, &mut output, cutoff.as_ref())?;
        output.finalize(&state)?;

        // 7b. Non-fused path: the sampler doesn't fetch features inline, so
        // `raw_features` would be empty. Fetch them post-sampling per visited
        // type — produces the same dict shape as the fused path. The fused
        // path already populated `raw_features` via `handle_fused_features`
        // during sampling; we skip the post-fetch in that case to avoid
        // duplicate batches.
        if !flags.fused_feature_fetching {
            for (type_id, type_state) in &state.type_states {
                if self.node_decoders.get(*type_id).and_then(|d| d.as_ref()).is_none() {
                    continue;
                }
                let node_ids: Vec<u64> =
                    type_state.nodes.iter().map(|n| n.global_id).collect();
                if node_ids.is_empty() {
                    continue;
                }
                let type_name = match self.id_to_node_type.get(*type_id) {
                    Some(n) => n.clone(),
                    None => continue,
                };
                output.set_namespace(type_name);
                self.fetch_features_for_type(*type_id, &node_ids, &[], &mut output)?;
            }
        }

        // 8. Stitch task fields. Project the seed rowpack RecordBatch into:
        //   - `task.target` ← `TARGET*` columns only. Adapter consumes this
        //     as the supervised signal; ENTITY*/TIMESTAMP/feature columns
        //     would be miscount-bug-prone if included.
        //   - `raw_features[<stage_table>]` ← non-identity columns
        //     (anything that isn't `ENTITY*`, `TARGET*`, or `TIMESTAMP`).
        //     This is the slot PR-28765's adapter reads to populate
        //     `hetero[train_table_name].tf`. Empty today for PQ plans (no
        //     extra cols in the schema) but the bucket is correctly
        //     addressed for the moment tt-ng emits FEAT*/LAG_FEAT*.
        let mut runtime_batch = output.into_runtime_batch();
        runtime_batch.task.input_ids = (0..batch_row_ids.len() as u32).collect();
        runtime_batch.task.tt_stage_node_ids = batch_row_ids.to_vec();
        // We populate seed_node_ids with the primary entity IDs to satisfy the
        // struct. Use the pre-injection snapshot so appended negatives never
        // leak into task inputs (when negatives is None this equals today's
        // `seed_nodes.values().next()`).
        let primary_ids = primary_ids_snapshot.unwrap_or_default();
        runtime_batch.task.seed_node_ids = primary_ids;
        runtime_batch.task.seed_timestamps = seed_timestamps;
        // Negative-sampling shape surfaced through the task bag (FFI contract).
        // `num_positives` is always the batch's positive count; `amount_m` /
        // `rhs_seed_type` describe the appended global negatives (None => 0 /
        // None, byte-identical to pre-negatives behavior).
        runtime_batch.task.num_positives = batch_row_ids.len() as u32;
        match negatives {
            Some(spec) => {
                runtime_batch.task.amount_m = spec.amount_m as u32;
                runtime_batch.task.rhs_seed_type = Some(spec.rhs_node_type.clone());
            }
            None => {
                runtime_batch.task.amount_m = 0;
                runtime_batch.task.rhs_seed_type = None;
            }
        }
        // Mask over the appended negatives (empty when no negatives requested).
        runtime_batch.task.neg_mask = neg_mask;
        if let Some(rb) = seed_target_batch.as_ref() {
            // FK-free LP label path (design §S5): when `target_node_type` is set,
            // read the resolved `target_node_ids: list<i64>` column off the seed
            // rowpack and emit the ragged `lp_pos_*` label fields. The column
            // already holds GLOBAL rhs node-ids (resolved once at ingest), so the
            // resolve closure is the IDENTITY (`Value::Int(i) => Some(i as u64)`)
            // — no MPH, no `reader.table()` lookup. The positives are LABELS only:
            // `seed_nodes[rhs]` was never populated above, so the rhs positive is
            // NEVER rooted as a seed → no message-passing leak (the leak-free
            // property for shallow_rhs / ID-GNN / hybrid). Rooting + two-tower
            // pair selection + negatives are S4 (out of scope here).
            if let Some(_rhs_type) = target_node_type {
                let idx = rb
                    .schema()
                    .fields()
                    .iter()
                    .position(|f| f.name() == "target_node_ids")
                    .ok_or_else(|| {
                        EngineError::DataCorruption(format!(
                            "resolution mode: seed table `{stage_table}` rowpack has no \
                             `target_node_ids` column"
                        ))
                    })?;
                let col = rb.column(idx);
                let ragged = hydrate_ragged_lp_target(col.as_ref(), |v| match v {
                    Value::Int(i) => Some(*i as u64),
                    _ => None,
                });
                runtime_batch.task.lp_pos_counts = ragged.counts;
                runtime_batch.task.lp_pos_rhs_global = ragged.rhs_global;
                runtime_batch.task.lp_pos_mask = ragged.mask;
                if pair_from_array_mask.is_none() {
                    runtime_batch.task.lp_pos_weight =
                        read_lp_list_weights(rb, &runtime_batch.task.lp_pos_counts)?;
                }
                // `lp_pos_local` stays None on the label-only path (positives are
                // never rooted).

                // Two-tower stitch (FK-free design §S4): when rooting fired,
                // exactly ONE positive per anchor was rooted as the FIRST B rhs
                // seeds. Overwrite the multi-positive ragged label fields above
                // with the single-positive-per-anchor payload that matches the
                // rooted pair seeds: `lp_pos_counts = [1; B]`, `lp_pos_rhs_global
                // = chosen` (the first B rhs seeds), `lp_pos_mask = pf_mask`, and
                // `lp_pos_local = arange(B)`. The kumo adapter reads
                // `lp_pos_local` as `dst_pos_index = arange(B)`; `dst_neg_index`
                // then indexes the appended negatives at `arange(B, B+B*M)`.
                if let Some(pf_mask) = pair_from_array_mask.take() {
                    let b = pf_mask.len();
                    let chosen: Vec<i64> = seed_nodes
                        .get(target_node_type.expect("target_node_type set on this path"))
                        .map(|v| v.iter().take(b).map(|&x| x as i64).collect())
                        .unwrap_or_default();
                    runtime_batch.task.lp_pos_counts = vec![1u32; b];
                    runtime_batch.task.lp_pos_rhs_global = chosen;
                    runtime_batch.task.lp_pos_mask = pf_mask;
                    runtime_batch.task.lp_pos_local = Some((0..b as u32).collect());
                    runtime_batch.task.lp_pos_weight =
                        read_lp_rooted_weights(rb, &pair_from_array_offsets, b)?;
                }
            }
            // STATIC LP past-visit exclusion (eval-only). The resolved
            // `exclude_node_ids: list<i64>` column carries the prior-stage RHS
            // ids (already global, hence the identity closure). Reuse
            // `hydrate_ragged_lp_target`, keeping only counts + rhs_global.
            if let Ok(idx) = rb.schema().index_of("exclude_node_ids") {
                let ragged = hydrate_ragged_lp_target(rb.column(idx).as_ref(), |v| match v {
                    Value::Int(i) => Some(*i as u64),
                    _ => None,
                });
                // Train seed tables carry all-empty lists; leave the task fields
                // empty so pybind omits the keys (the "absent on train" contract).
                if !ragged.rhs_global.is_empty() {
                    runtime_batch.task.lp_exclude_counts = ragged.counts;
                    runtime_batch.task.lp_exclude_rhs_global = ragged.rhs_global;
                }
            }
            let (target_proj, feature_proj) = project_seed_rowpack(rb)?;
            runtime_batch.task.target = target_proj;
            if let Some(feature_rb) = feature_proj {
                let bank = runtime_batch
                    .raw_features
                    .entry(stage_table.to_string())
                    .or_default();
                bank.schema = Some(feature_rb.schema());
                bank.batches.push(feature_rb);
            }
        }
        // Weight: read from the `WEIGHT` column of the seed rowpack (present when
        // `weight_col` was set in the plan). Absent column → `None`.
        runtime_batch.task.weight = match seed_target_batch.as_ref() {
            Some(rb) => read_weight_column(rb)?,
            None => None,
        };

        Ok(runtime_batch)
    }

    /// Dense global-id range `(min_id, max_id)` (inclusive) for a node type
    /// across its concatenated shards. Mirrors the negatives `lo..=hi` derivation
    /// (`sample_runtime_batch` §6b). Used by the pybind NodeRange iterator to
    /// build the `[0, num_rhs)` seed range; the catalog is dense by ingest
    /// guarantee (`max_id == num_rows - 1`).
    pub fn node_type_id_range(&self, node_type: &str) -> Result<(u64, u64)> {
        let type_id = *self.node_type_to_id.get(node_type).ok_or_else(|| {
            EngineError::InvalidRequest(format!("node type `{node_type}` not in engine catalog"))
        })?;
        let shards = self.readers.get(type_id).ok_or_else(|| {
            EngineError::InvalidRequest(format!("no readers for node type `{node_type}`"))
        })?;
        let lo = shards
            .first()
            .ok_or_else(|| {
                EngineError::InvalidRequest(format!("node type `{node_type}` has no shards"))
            })?
            .min_id;
        let hi = shards
            .last()
            .expect("non-empty shards (checked via first())")
            .max_id;
        Ok((lo, hi))
    }

    /// NodeRange seed path (two-tower MIPS RHS embeddings). Seeds the supplied
    /// RHS/target catalog node-ids (`batch_row_ids` ARE the global node-ids of
    /// `nr.rhs_node_type`) at the single fixed `nr.seed_time`, samples each
    /// seed's neighborhood via the shared plan, and emits the same per-type
    /// HeteroData batch as the stage-table path.
    ///
    /// Deliberately BYPASSES the stage-table seed-rowpack read, the entity/target
    /// FK-free resolution, the two-tower positive rooting, AND the negatives
    /// blocks of [`Self::sample_runtime_batch`]: NodeRange is an RHS-tower forward
    /// pass over the catalog, so there are no positives, labels, or negatives to
    /// produce. It reuses the SAME `self.sample()` / `RuntimeBatchOutput::finalize`
    /// path so topology / feature collation stays identical.
    ///
    /// `batch_row_ids` are the seeds for this batch; the pybind iterator drives
    /// the full dense `[0, num_rhs)` range in `batch_size` chunks so the seeds
    /// cover the catalog exactly once, in order, deterministically.
    #[allow(clippy::too_many_arguments)]
    fn sample_node_range_batch(
        &self,
        reader: &diskgraph_index::graph::GraphIndexReader,
        batch_row_ids: &[u64],
        dataset_id: diskgraph_core::dataset::DatasetId,
        plan: &SamplingPlan,
        flags: &SamplingFlags,
        nr: NodeRangeSeed<'_>,
        cutoff: Option<NodeCutoff<'_>>,
    ) -> Result<crate::output::RuntimeBatch> {
        use crate::output::{OutputHandler, RuntimeBatchOutput};
        use std::collections::HashSet;

        // The RHS type must be in the merged catalog (sanity — the pybind layer
        // already validated it to derive the id range).
        if !self.node_type_to_id.contains_key(nr.rhs_node_type) {
            return Err(EngineError::InvalidRequest(format!(
                "NodeRange: rhs_node_type `{}` not in engine catalog",
                nr.rhs_node_type
            )));
        }

        // Seed set = the supplied RHS catalog ids, anchored at the fixed time.
        let mut seed_nodes: HashMap<String, Vec<u64>> = HashMap::new();
        seed_nodes.insert(nr.rhs_node_type.to_string(), batch_row_ids.to_vec());
        let mut seed_ts_map: HashMap<String, Vec<i64>> = HashMap::new();
        seed_ts_map.insert(
            nr.rhs_node_type.to_string(),
            vec![nr.seed_time; batch_row_ids.len()],
        );

        // rel_id → (src, name, dst) for the collator. Mirrors the stage-table
        // path; no seed-src relations are dropped (NodeRange seeds a real catalog
        // type, not a synthetic stage table).
        let mut rel_id_to_names: HashMap<u16, (String, String, String)> = HashMap::new();
        for (rel_id_usize, name) in self.id_to_rel_name.iter().enumerate() {
            let src_id = self.relation_src_type[rel_id_usize];
            let dst_id = self.relation_info[rel_id_usize];
            rel_id_to_names.insert(
                rel_id_usize as u16,
                (
                    self.id_to_node_type[src_id].clone(),
                    name.clone(),
                    self.id_to_node_type[dst_id].clone(),
                ),
            );
        }

        let request = BatchSamplingRequest {
            dataset_id,
            seed_nodes,
            seed_timestamps: Some(seed_ts_map),
            seed_timestamp_unit: self.timestamp_unit,
            plan: plan.clone(),
            flags: flags.clone(),
        };

        // Sample + collate via the shared path. No seed-src relation ids to
        // drop, and no stage seed namespaces in this NodeRange path — pass an
        // empty seed_type_ids set.
        let mut state = SamplingState::default();
        let mut output = RuntimeBatchOutput::new(
            "main",
            HashSet::new(),
            rel_id_to_names,
            self.id_to_node_type.clone(),
            HashSet::new(),
        );
        self.sample_with_cutoff(&request, &mut state, &mut output, cutoff.as_ref())?;
        output.finalize(&state)?;

        // Non-fused path: fetch features post-sampling per visited type (same as
        // the stage-table path) so the dict shape matches.
        if !flags.fused_feature_fetching {
            for (type_id, type_state) in &state.type_states {
                if self
                    .node_decoders
                    .get(*type_id)
                    .and_then(|d| d.as_ref())
                    .is_none()
                {
                    continue;
                }
                let node_ids: Vec<u64> =
                    type_state.nodes.iter().map(|n| n.global_id).collect();
                if node_ids.is_empty() {
                    continue;
                }
                let type_name = match self.id_to_node_type.get(*type_id) {
                    Some(n) => n.clone(),
                    None => continue,
                };
                output.set_namespace(type_name);
                self.fetch_features_for_type(*type_id, &node_ids, &[], &mut output)?;
            }
        }

        let mut runtime_batch = output.into_runtime_batch();
        // Task bag: only the seed identity / count / timestamps are meaningful in
        // NodeRange mode. All LP / negative / target fields stay at their Default
        // (empty / None / 0) — the two-tower MIPS reader uses the RHS embeddings
        // off `graphs["main"]`, not these fields.
        runtime_batch.task.input_ids = (0..batch_row_ids.len() as u32).collect();
        runtime_batch.task.tt_stage_node_ids = batch_row_ids.to_vec();
        runtime_batch.task.seed_node_ids = batch_row_ids.to_vec();
        runtime_batch.task.seed_timestamps = Some(vec![nr.seed_time; batch_row_ids.len()]);
        runtime_batch.task.num_positives = batch_row_ids.len() as u32;

        // `reader` is unused for sidecars in NodeRange mode (no stage extension /
        // weight.bin); kept in the signature for symmetry with the stage path and
        // future per-type sidecars.
        let _ = reader;

        Ok(runtime_batch)
    }

    fn compile_plan(
        &self,
        plan: &diskgraph_core::sampling::SamplingPlan,
    ) -> Result<CompiledSamplingPlan> {
        let mut layers = Vec::with_capacity(plan.layers.len());

        for layer in &plan.layers {
            let mut fanouts = HashMap::new();
            for (name, spec) in &layer.fanouts {
                if let Some(&id) = self.rel_name_to_id.get(name) {
                    fanouts.insert(id, spec.clone());
                }
            }
            layers.push(CompiledLayerSpec {
                fanouts,
                default_fanout: layer.default_fanout.clone(),
            });
        }

        Ok(CompiledSamplingPlan { layers })
    }

    fn node_cutoff<'a>(&self, filter: PseudoTimeFilter<'a>) -> Result<NodeCutoff<'a>> {
        let type_id = *self.node_type_to_id.get(filter.node_type).ok_or_else(|| {
            EngineError::InvalidRequest(format!(
                "pseudo time node type `{}` not in engine catalog",
                filter.node_type
            ))
        })?;
        let required = self.readers[type_id]
            .iter()
            .filter(|shard| shard.min_id <= shard.max_id)
            .map(|shard| shard.max_id + 1)
            .max()
            .unwrap_or(0);
        if (filter.times.len() as u64) < required {
            return Err(EngineError::InvalidRequest(format!(
                "pseudo times cover {} `{}` nodes, graph has {}",
                filter.times.len(),
                filter.node_type,
                required
            )));
        }
        Ok(NodeCutoff {
            type_id,
            times: filter.times,
            seed_time: filter.seed_time,
        })
    }

    pub fn sample(
        &self,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        self.sample_with_cutoff(request, state, output, None)
    }

    fn sample_with_cutoff(
        &self,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
        cutoff: Option<&NodeCutoff<'_>>,
    ) -> Result<()> {
        let sample_start = Instant::now();

        // Convert seed timestamps from the caller's unit to the index unit (Seconds).
        // The index always stores timestamps in seconds today; assert to catch if
        // this assumption changes (at which point the engine should be made generic
        // over UnitSystem).
        assert_eq!(
            self.timestamp_unit,
            TimeUnit::Seconds,
            "engine assumes seconds index; make engine generic over UnitSystem if this changes"
        );
        let mut request_cow = std::borrow::Cow::Borrowed(request);
        if request.seed_timestamp_unit != TimeUnit::Seconds {
            if let Some(mut ts_map) = request.seed_timestamps.clone() {
                for ts_vec in ts_map.values_mut() {
                    for ts in ts_vec.iter_mut() {
                        *ts = abs_from_unit::<Seconds>(*ts, request.seed_timestamp_unit).raw();
                    }
                }
                let mut owned = request.clone();
                owned.seed_timestamps = Some(ts_map);
                owned.seed_timestamp_unit = TimeUnit::Seconds;
                request_cow = std::borrow::Cow::Owned(owned);
            }
        }
        let request = request_cow.as_ref();

        // 1. Initial Validation & Initialization
        if request.seed_nodes.is_empty() {
            return Err(EngineError::InvalidRequest("no seed nodes provided".to_string()));
        }

        debug!(
            dataset_id = %request.dataset_id,
            seed_types = ?request.seed_nodes.keys().collect::<Vec<_>>(),
            hops = request.plan.layers.len(),
            fused_feature_fetching = request.flags.fused_feature_fetching,
            swap_edge_direction = request.flags.swap_edge_direction,
            "Sampling engine run started"
        );

        // Pre-allocate stats
        let num_types = self.readers.len();
        let num_rels = self.relation_info.len();
        let num_hops = request.plan.layers.len();
        state.stats.resize(num_types, num_rels, num_hops);

        let fused = request.flags.fused_feature_fetching;

        // 2. Initialize Seed Layer (Hop 0)
        // This ensures seed nodes get Local IDs 0..K and are preserved in order.
        for (seed_type, seed_ids) in &request.seed_nodes {
            let seed_type_id = *self.node_type_to_id.get(seed_type).ok_or_else(|| {
                EngineError::InvalidRequest(format!("seed type `{seed_type}` not found in catalog"))
            })?;
            let seed_state = state.type_states.entry(seed_type_id).or_default();
            for (i, &global_id) in seed_ids.iter().enumerate() {
                let seed_index = i as u32;
                let key = LookupKey {
                    seed_type_id: seed_type_id as u32,
                    seed_index,
                    global_id,
                };

                // Assign Local ID for seeds
                if !seed_state.index_map.contains_key(&key) {
                    let local_id = seed_state.nodes.len() as u32;
                    seed_state.index_map.insert(key, local_id);

                    // Cache Node (Hop 0). A seed is its own root, so its
                    // root seed type is its own type id.
                    seed_state.nodes.push(NodeRecord {
                        global_id,
                        seed_index,
                        local_id,
                        seed_type_id: seed_type_id as u32,
                    });
                    state.stats.record_node(0, seed_type_id);

                    // Add to frontier for Layer 0
                    seed_state.next_frontier.push(NodeRecord {
                        global_id,
                        seed_index,
                        local_id,
                        seed_type_id: seed_type_id as u32,
                    });
                }
            }
        }

        // 3. Process Layers (BFS)
        let compiled_plan = self.compile_plan(&request.plan)?;
        for layer_idx in 0..request.plan.layers.len() {
            let hop_start = Instant::now();
            let edges_before = state.stats.total_edges;
            let nodes_before = state.stats.total_nodes;

            self.process_layer(
                layer_idx,
                &compiled_plan.layers[layer_idx],
                request,
                state,
                output,
                cutoff,
            )?;

            debug!(
                hop = layer_idx,
                hop_edges = state.stats.total_edges.saturating_sub(edges_before),
                hop_new_nodes = state.stats.total_nodes.saturating_sub(nodes_before),
                elapsed_ms = hop_start.elapsed().as_millis() as u64,
                "Completed sampling hop"
            );
        }

        // 4. Finalize fused fetching by fetching leaf node features
        // and flush remaining features.
        if fused {
            let type_ids: Vec<usize> = state.type_states.keys().cloned().collect();
            for type_id in type_ids {
                let scratch = &mut state.decode_scratch;
                if let Some(ts) = state.type_states.get_mut(&type_id) {
                    if !ts.next_frontier.is_empty() {
                        let node_ids: Vec<u64> =
                            ts.next_frontier.iter().map(|n| n.global_id).collect();
                        self.accumulate_features_from_ids_builder(
                            type_id, &node_ids, ts, scratch, output,
                        )?;
                        ts.next_frontier.clear();
                    }
                    self.flush_fused_features(type_id, ts, output)?;
                }
            }
        }

        // 5. Finalize node batch and stats
        output.finalize(state)?;

        debug!(
            total_nodes = state.stats.total_nodes,
            total_edges = state.stats.total_edges,
            elapsed_ms = sample_start.elapsed().as_millis() as u64,
            "Sampling engine run finished"
        );

        Ok(())
    }

    /// Feature fetching from an EdgeListView, used in fused mode
    fn accumulate_features_from_view(
        &self,
        type_id: usize,
        type_state: &mut crate::state::TypeState,
        scratch: &mut RowPackDecodeScratch,
        edges_view: &EdgeListView,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let decoder = self
            .node_decoders
            .get(type_id)
            .and_then(|d| d.as_ref())
            .ok_or_else(|| EngineError::Internal(format!("no decoder for type {}", type_id)))?;

        self.accumulate_features_from_view_internal(
            type_id, type_state, scratch, decoder, edges_view, output,
        )
    }

    /// Reused for fused mode feature fetching for both non-leaf and leaf nodes.
    /// Non-leaf nodes EdgeListView is accessed as part of sampling. Leaf nodes
    /// requires a separate access to EdgeListView given their IDs.
    fn accumulate_features_from_view_internal(
        &self,
        type_id: usize,
        type_state: &mut crate::state::TypeState,
        scratch: &mut RowPackDecodeScratch,
        decoder: &RowPackDecoder,
        edges_view: &EdgeListView,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        if let Some(data) = edges_view
            .node_data()
            .map_err(|e| EngineError::DataCorruption(e.to_string()))?
        {
            let context = decoder.context();
            // Drop reserved columns (e.g. WEIGHT) from features via a per-column
            // mask; `flush_fused_features` projects the flush schema to the same
            // indices to keep them aligned.
            let mask: Vec<bool> = context
                .columns
                .iter()
                .map(|c| !is_reserved_feature_column(&c.name))
                .collect();
            if type_state.builders.is_none() {
                let mut builders = Vec::with_capacity(context.columns.len());
                for col in &context.columns {
                    if is_reserved_feature_column(&col.name) {
                        continue;
                    }
                    let (_, builder) =
                        diskgraph_rowpack::make_arrow_builder(&col.name, col.logical_type, 1024)
                            .map_err(|e| EngineError::Internal(e.to_string()))?;
                    builders.push(builder);
                }
                type_state.builders = Some(builders);
            }

            if let Some(builders) = &mut type_state.builders {
                decoder
                    .decode_row_to_builders(data, builders, &mask, scratch)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;

                type_state.builder_rows += 1;
                type_state.builder_bytes += data.len();
            }

            if type_state.builder_bytes >= self.target_batch_bytes
                || type_state.builder_rows >= self.target_batch_rows
            {
                self.flush_fused_features(type_id, type_state, output)?;
            }
        }
        Ok(())
    }

    /// Feature fetching for leaf nodes (current frontier after last hop) in
    /// fused mode. Sampling no longer access edge list for leaf nodes. An
    /// additional access is needed specifically for feature fetching.
    pub(crate) fn accumulate_features_from_ids_builder(
        &self,
        type_id: usize,
        node_ids: &[u64],
        type_state: &mut crate::state::TypeState,
        scratch: &mut RowPackDecodeScratch,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let shards = self
            .readers
            .get(type_id)
            .ok_or_else(|| EngineError::InvalidRequest("invalid node type id".to_string()))?;

        let decoder = self
            .node_decoders
            .get(type_id)
            .and_then(|d| d.as_ref())
            .ok_or_else(|| EngineError::Internal(format!("no decoder for type {}", type_id)))?;

        // No shard tracking/sorting for fused fetching to strictly preserve order
        for &global_id in node_ids {
            let shard_idx = self.find_shard(shards, global_id)?;
            let shard = &shards[shard_idx];
            let local_id = global_id - shard.min_id;
            let edges_view = shard
                .reader
                .edges_for_node(local_id)
                .map_err(|e| EngineError::DataCorruption(e.to_string()))?;

            self.accumulate_features_from_view_internal(
                type_id,
                type_state,
                scratch,
                decoder,
                &edges_view,
                output,
            )?;
        }
        Ok(())
    }

    /// Feature fetching in standalone mode. Multi-threaded to speedup feature
    /// fetching for cases when sampled subgraph size is large.
    fn accumulate_features_from_ids_chunked(
        &self,
        type_id: usize,
        node_ids: &[u64],
        col_indices: &[usize],
        decoder: &RowPackDecoder,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }

        let shards = &self.readers[type_id];

        let num_nodes = node_ids.len();
        let num_cores = self.thread_pool.current_num_threads();
        let ideal_chunks = num_cores * 2;
        let chunk_size = (num_nodes / ideal_chunks).clamp(2048, 64_000);
        let chunks: Vec<_> = node_ids.chunks(chunk_size).collect();

        let results: Vec<Result<arrow_array::RecordBatch>> = self.thread_pool.install(|| chunks
            .into_par_iter()
            .map(|chunk_ids| {
                // 1. Deduplicate requested IDs to minimize Disk I/O and Decoding
                let mut unique_ids_unsorted = Vec::new();
                let mut dedup_map = HashMap::new();
                let mut take_indices = vec![0u32; chunk_ids.len()];

                for (pos, &global_id) in chunk_ids.iter().enumerate() {
                    let idx = *dedup_map.entry(global_id).or_insert_with(|| {
                        let new_idx = unique_ids_unsorted.len() as u32;
                        unique_ids_unsorted.push(global_id);
                        new_idx
                    });
                    take_indices[pos] = idx;
                }

                // 2. Sort Unique IDs for Sequential Disk I/O
                let mut sorted_pairs: Vec<(u64, u32)> = unique_ids_unsorted
                    .into_iter()
                    .enumerate()
                    .map(|(i, id)| (id, i as u32))
                    .collect();
                sorted_pairs.sort_unstable_by_key(|x| x.0);

                // 3. Create reverse mapping from old unique index to new sorted index
                let mut old_to_new = vec![0u32; sorted_pairs.len()];
                for (new_idx, &(_, old_idx)) in sorted_pairs.iter().enumerate() {
                    old_to_new[old_idx as usize] = new_idx as u32;
                }

                // 4. Update take_indices to point to the new sorted positions
                for idx in &mut take_indices {
                    *idx = old_to_new[*idx as usize];
                }
                let take_array = arrow_array::UInt32Array::from(take_indices);

                let unique_ids: Vec<u64> = sorted_pairs.into_iter().map(|x| x.0).collect();

                // 5. Fetch and Decode ONLY unique IDs
                let mut raw_rows = Vec::with_capacity(unique_ids.len());
                let mut last_shard_idx = 0;

                for &global_id in &unique_ids {
                    let shard = if global_id >= shards[last_shard_idx].min_id
                        && global_id <= shards[last_shard_idx].max_id
                    {
                        &shards[last_shard_idx]
                    } else {
                        last_shard_idx = self.find_shard(shards, global_id)?;
                        &shards[last_shard_idx]
                    };

                    let local_id = global_id - shard.min_id;
                    let edges_view = shard
                        .reader
                        .edges_for_node(local_id)
                        .map_err(|e| EngineError::DataCorruption(e.to_string()))?;

                    let data = edges_view
                        .node_data()
                        .map_err(|e| EngineError::DataCorruption(e.to_string()))?
                        .ok_or_else(|| {
                            EngineError::DataCorruption(format!(
                                "node {} has no feature data",
                                global_id
                            ))
                        })?;
                    raw_rows.push(data);
                }

                let unique_batch = decoder
                    .decode_batch_to_arrow::<&[u8]>(&raw_rows, col_indices)
                    .map_err(|e| EngineError::Internal(format!("decoding failed: {}", e)))?;

                // 6. Broadcast (take) unique rows back to the requested redundant order
                let expanded_columns: std::result::Result<Vec<_>, arrow::error::ArrowError> = unique_batch
                    .columns()
                    .iter()
                    .map(|col| {
                        arrow::compute::take(col, &take_array, None)
                    })
                    .collect();

                let expanded_columns = expanded_columns
                    .map_err(|e| EngineError::Internal(format!("arrow take failed: {}", e)))?;

                arrow_array::RecordBatch::try_new(unique_batch.schema(), expanded_columns)
                    .map_err(|e| EngineError::Internal(format!("batch creation failed: {}", e)))
            })
            .collect()
        );

        for res in results {
            let batch = res?;
            output.handle_feature_batch(batch)?;
        }

        Ok(())
    }

    pub(crate) fn flush_fused_features(
        &self,
        type_id: usize,
        type_state: &mut crate::state::TypeState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        if type_state.builder_rows == 0 {
            return Ok(());
        }

        let mut arrays = Vec::new();

        if let Some(builders) = &mut type_state.builders {
            for builder in builders {
                arrays.push(builder.finish());
            }
        }

        let full_schema = self.node_schemas[type_id].as_ref().ok_or_else(|| {
            EngineError::Internal(format!("no precomputed schema for type {}", type_id))
        })?;

        // Reserved columns (WEIGHT) were dropped from the builders/mask in
        // `accumulate_features_from_view_internal`, so project the schema to
        // the same non-reserved columns to keep field count == array count.
        let keep: Vec<usize> = full_schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| !is_reserved_feature_column(f.name()))
            .map(|(i, _)| i)
            .collect();
        let schema = if keep.len() == full_schema.fields().len() {
            full_schema.clone()
        } else {
            Arc::new(full_schema.project(&keep).map_err(|e| {
                EngineError::Internal(format!("fused schema projection failed: {e}"))
            })?)
        };

        let batch = arrow_array::RecordBatch::try_new(schema, arrays)
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        output.handle_fused_features(type_id, batch)?;
        type_state.builder_rows = 0;
        type_state.builder_bytes = 0;

        Ok(())
    }


    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn sample_segment_edges(
        &self,
        node: &NodeRecord,
        rel_id: usize,
        segment: &RelationSegment<'_>,
        fanout_spec: &diskgraph_core::sampling::FanoutSpec,
        request: &BatchSamplingRequest,
        dst_scratch: &mut Vec<u64>,
        key_scratch: &mut Vec<u64>,
        cutoff: Option<&NodeCutoff<'_>>,
        mut on_edge: impl FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        let edge_count = segment.count as usize;
        if edge_count == 0 {
            return Ok(());
        }

        let mut unpacker = EdgeUnpacker::new(segment.data, &segment.layout);
        let k = fanout_spec.fanout as usize;
        if k == 0
            && matches!(
                fanout_spec.mode,
                SamplingMode::Uniform
                    | SamplingMode::DeterministicUniform
                    | SamplingMode::TemporalMostRecent
            )
        {
            return Ok(());
        }
        let cutoff = cutoff.filter(|c| self.relation_info.get(rel_id) == Some(&c.type_id));

        match fanout_spec.mode {
            SamplingMode::TemporalMostRecent => {
                let decoder = self
                    .decoders
                    .get(rel_id)
                    .and_then(|d: &Option<SortKeyDecoder>| d.as_ref());
                // Resolve the anchor against the array of the node's *own*
                // root seed type — not an arbitrary `keys().next()` type, which
                // is wrong (and out-of-bounds) once seed arrays differ in length
                // across types (e.g. LP lhs vs rhs + negatives).
                let anchor = self
                    .id_to_node_type
                    .get(node.seed_type_id as usize)
                    .and_then(|seed_type| {
                        request
                            .seed_timestamps
                            .as_ref()
                            .and_then(|m| m.get(seed_type))
                            .and_then(|v| v.get(node.seed_index as usize).copied())
                    });

                let mut n = 0;
                let batch_size = k.min(128);

                loop {
                    let take = (k - n).min(batch_size);
                    dst_scratch.resize(take, 0);
                    key_scratch.resize(take, 0);

                    let unpacked = unpacker.unpack_batch(dst_scratch, key_scratch);
                    if unpacked == 0 {
                        break;
                    }

                    for (&dst_global_id, &sort_key) in
                        dst_scratch.iter().zip(key_scratch.iter()).take(unpacked)
                    {
                        if cutoff.is_some_and(|c| !c.visible(dst_global_id)) {
                            continue;
                        }
                        if let (Some(dec), Some(anch)) = (decoder, anchor) {
                            let (edge_ts, _) = dec.decode(sort_key);
                            if let Some(ts) = edge_ts {
                                if ts > anch {
                                    continue;
                                }
                            }
                        }

                        on_edge(dst_global_id)?;
                        n += 1;
                        if n >= k {
                            break;
                        }
                    }

                    if unpacked < take || n >= k {
                        break;
                    }
                }
            }
            SamplingMode::Uniform | SamplingMode::DeterministicUniform if cutoff.is_some() => {
                let c = cutoff.expect("guarded");
                if k >= edge_count {
                    loop {
                        dst_scratch.resize(128, 0);
                        key_scratch.resize(128, 0);
                        let count = unpacker.unpack_batch(dst_scratch, key_scratch);
                        if count == 0 {
                            break;
                        }
                        for &dst in &dst_scratch[..count] {
                            if c.visible(dst) {
                                on_edge(dst)?;
                            }
                        }
                    }
                    return Ok(());
                }
                use rand::SeedableRng;
                let mut rng = match self.sampling_seed(node, rel_id, &fanout_spec.mode) {
                    Some(seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed),
                    None => rand_chacha::ChaCha8Rng::seed_from_u64(rand::random()),
                };
                let budget = k.saturating_mul(16).saturating_add(32);
                if budget < edge_count / 2 {
                    // Shuffled order makes the first k visible candidates uniform.
                    let candidates =
                        rand::seq::index::sample(&mut rng, edge_count, budget).into_vec();
                    let mut picked = Vec::new();
                    for index in candidates {
                        let mut candidate = EdgeUnpacker::new(segment.data, &segment.layout);
                        candidate.advance(index);
                        let mut dst = [0];
                        let mut key = [0];
                        if candidate.unpack_batch(&mut dst, &mut key) == 1 && c.visible(dst[0]) {
                            picked.push((index, dst[0]));
                            if picked.len() == k {
                                break;
                            }
                        }
                    }
                    if picked.len() == k {
                        picked.sort_unstable_by_key(|&(index, _)| index);
                        for (_, dst) in picked {
                            on_edge(dst)?;
                        }
                        return Ok(());
                    }
                }
                let collect = edge_count <= budget.saturating_mul(2);
                let mut visible = Vec::new();
                let mut count = 0;
                loop {
                    dst_scratch.resize(128, 0);
                    key_scratch.resize(128, 0);
                    let unpacked = unpacker.unpack_batch(dst_scratch, key_scratch);
                    if unpacked == 0 {
                        break;
                    }
                    let batch = dst_scratch[..unpacked].iter().filter(|&&dst| c.visible(dst));
                    if collect {
                        visible.extend(batch);
                    } else {
                        count += batch.count();
                    }
                }
                if collect {
                    if k >= visible.len() {
                        for dst in visible {
                            on_edge(dst)?;
                        }
                    } else {
                        let mut indices =
                            rand::seq::index::sample(&mut rng, visible.len(), k).into_vec();
                        indices.sort_unstable();
                        for index in indices {
                            on_edge(visible[index])?;
                        }
                    }
                    return Ok(());
                }
                let mut picked = rand::seq::index::sample(&mut rng, count, k.min(count)).into_vec();
                picked.sort_unstable();
                let mut picked = picked.into_iter().peekable();
                let mut position = 0;
                unpacker = EdgeUnpacker::new(segment.data, &segment.layout);
                while picked.peek().is_some() {
                    let count = unpacker.unpack_batch(dst_scratch, key_scratch);
                    if count == 0 {
                        break;
                    }
                    for &dst in &dst_scratch[..count] {
                        if !c.visible(dst) {
                            continue;
                        }
                        if picked.peek() == Some(&position) {
                            on_edge(dst)?;
                            picked.next();
                        }
                        position += 1;
                    }
                }
            }
            SamplingMode::Uniform | SamplingMode::DeterministicUniform => {
                if k >= edge_count {
                    loop {
                        dst_scratch.resize(128, 0);
                        key_scratch.resize(128, 0);
                        let unpacked = unpacker.unpack_batch(dst_scratch, key_scratch);
                        if unpacked == 0 {
                            break;
                        }
                        for &dst_global_id in dst_scratch.iter().take(unpacked) {
                            on_edge(dst_global_id)?;
                        }
                    }
                } else {
                    let mut sorted_indices =
                        self.uniform_indices(node, rel_id, &fanout_spec.mode, edge_count, k);
                    sorted_indices.sort_unstable();

                    let mut current_pos = 0;
                    for target_idx in sorted_indices {
                        let to_skip = target_idx - current_pos;
                        if to_skip > 0 {
                            unpacker.advance(to_skip);
                        }
                        dst_scratch.resize(1, 0);
                        key_scratch.resize(1, 0);
                        if unpacker.unpack_batch(dst_scratch, key_scratch) > 0 {
                            on_edge(dst_scratch[0])?;
                            current_pos = target_idx + 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            _ => {
                warn!(
                    relation_id = rel_id,
                    mode = ?fanout_spec.mode,
                    "Unsupported sampling mode requested"
                );
                return Err(EngineError::InvalidRequest(format!(
                    "Sampling mode {:?} not supported",
                    fanout_spec.mode
                )));
            }
        }
        Ok(())
    }

    fn sampling_seed(&self, node: &NodeRecord, rel_id: usize, mode: &SamplingMode) -> Option<u64> {
        let seed = match mode {
            SamplingMode::DeterministicUniform => Some(DETERMINISTIC_UNIFORM_SEED_V1),
            SamplingMode::Uniform => self.random_seed,
            _ => None,
        };
        seed.map(|seed| {
            seed.wrapping_add(node.global_id)
                .wrapping_add(rel_id as u64)
        })
    }

    fn uniform_indices(
        &self,
        node: &NodeRecord,
        rel_id: usize,
        mode: &SamplingMode,
        n: usize,
        k: usize,
    ) -> Vec<usize> {
        use rand::seq::index::sample;
        if let Some(rng_seed) = self.sampling_seed(node, rel_id, mode) {
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
            sample(&mut rng, n, k).into_vec()
        } else {
            sample(&mut rand::thread_rng(), n, k).into_vec()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_layer(
        &self,
        layer_idx: usize,
        layer_spec: &CompiledLayerSpec,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
        cutoff: Option<&NodeCutoff<'_>>,
    ) -> Result<()> {
        let fused = request.flags.fused_feature_fetching;
        let mut work_queue = Vec::new();
        for (node_type_id, type_state) in state.type_states.iter_mut() {
            if !type_state.next_frontier.is_empty() {
                let frontier = std::mem::take(&mut type_state.next_frontier);
                work_queue.push((*node_type_id, frontier));
            }
        }

        // Fused fetching path
        if fused {
            for (src_type_id, frontier) in work_queue {
                if src_type_id >= self.readers.len() {
                    continue;
                }
                let shards = &self.readers[src_type_id];
                if shards.is_empty() {
                    continue;
                }

                for node in &frontier {
                    let shard_idx = self.find_shard(shards, node.global_id)?;
                    let shard = &shards[shard_idx];
                    let local_id = node.global_id - shard.min_id;

                    let edges_view = shard
                        .reader
                        .edges_for_node(local_id)
                        .map_err(|e| EngineError::DataCorruption(e.to_string()))?;

                    // FETCH SOURCE FEATURES (Locality)
                    {
                        let scratch = &mut state.decode_scratch;
                        let type_state = state.type_states.get_mut(&src_type_id).unwrap();
                        self.accumulate_features_from_view(
                            src_type_id,
                            type_state,
                            scratch,
                            &edges_view,
                            output,
                        )?;
                    }

                    let segments = edges_view.relation_segments().unwrap_or_default();

                    self.process_node_relations(
                        layer_idx, src_type_id, &segments, node, layer_spec, request, state, output,
                        cutoff,
                    )?;
                }
            }
            return Ok(());
        }

        // Standalone sampling path (Multi-threaded Map-Reduce)
        for (src_type_id, frontier) in work_queue {
            if src_type_id >= self.readers.len() {
                continue;
            }
            let shards = &self.readers[src_type_id];
            if shards.is_empty() {
                continue;
            }

            // group nodes by shard
            let mut shard_groups: Vec<Vec<&NodeRecord>> = vec![Vec::new(); shards.len()];
            for item in &frontier {
                let idx = self.find_shard(shards, item.global_id)?;
                shard_groups[idx].push(item);
            }

            // Map Phase: Parallel Shard Traversal
            // Each thread processes one shard to emit all sampled RawEdge
            let raw_edges_results: Vec<Vec<RawEdge>> = self.thread_pool.install(|| shard_groups
                .into_par_iter()
                .enumerate()
                .filter(|(_, nodes)| !nodes.is_empty())
                .map(|(shard_idx, mut shard_nodes)| {
                    let shard = &shards[shard_idx];

                    // STABLE SORT: Sort by global_id, then seed_index to ensure
                    // deterministic visitation order even for hub nodes reached via multiple paths.
                    shard_nodes.sort_by(|a, b| {
                        a.global_id.cmp(&b.global_id).then(a.seed_index.cmp(&b.seed_index))
                    });

                    let mut local_raw_edges = Vec::new();
                    let mut current_segments = Vec::new();
                    let mut last_id = None;

                    // Thread-local scratch
                    let mut dst_scratch = Vec::with_capacity(128);
                    let mut key_scratch = Vec::with_capacity(128);

                    for node in shard_nodes {
                        // HUB DEDUPING
                        if last_id != Some(node.global_id) {
                            let local_id = node.global_id - shard.min_id;
                            if let Ok(edges_view) = shard.reader.edges_for_node(local_id) {
                                current_segments = edges_view.relation_segments().unwrap_or_default();
                            }
                            last_id = Some(node.global_id);
                        }

                        let type_local_map = self
                            .local_to_merged_rel
                            .get(src_type_id)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]);
                        for segment in &current_segments {
                            let local_rel_id = segment.layout.relation_id as usize;
                            // Translate this shard segment's LOCAL rel id to
                            // the merged catalog's id before any downstream
                            // indexing — `relation_info`, fanout lookups, and
                            // the recorded edge's `rel_id` field are all
                            // keyed on merged ids.
                            let merged_rel_u16 = type_local_map
                                .get(local_rel_id)
                                .copied()
                                .unwrap_or(u16::MAX);
                            if merged_rel_u16 == u16::MAX {
                                continue;
                            }
                            let rel_id = merged_rel_u16 as usize;
                            let fanout_spec = layer_spec.fanouts.get(&rel_id).or(layer_spec.default_fanout.as_ref());

                            if let Some(fanout_spec) = fanout_spec {
                                let dst_type_id = *self.relation_info.get(rel_id).unwrap_or(&usize::MAX);
                                if dst_type_id != usize::MAX {
                                    let _ = self.sample_segment_edges(
                                        node,
                                        rel_id,
                                        segment,
                                        fanout_spec,
                                        request,
                                        &mut dst_scratch,
                                        &mut key_scratch,
                                        cutoff,
                                        |dst_global_id| {
                                            local_raw_edges.push(RawEdge {
                                                src_local_id: node.local_id,
                                                dst_global_id,
                                                dst_type_id,
                                                rel_id: rel_id as u16,
                                                seed_index: node.seed_index,
                                                seed_type_id: node.seed_type_id,
                                                swap_edge_direction: request.flags.swap_edge_direction,
                                            });
                                            Ok(())
                                        }
                                    );
                                }
                            }
                        }
                    }
                    local_raw_edges
                })
                .collect()
            );

            // Reduce Phase: Sequential Commitment
            for raw_edges in raw_edges_results {
                for edge in raw_edges {
                    self.record_edge(
                        layer_idx,
                        edge.src_local_id,
                        edge.seed_index,
                        edge.seed_type_id,
                        edge.dst_global_id,
                        edge.dst_type_id,
                        edge.rel_id,
                        edge.swap_edge_direction,
                        state,
                        output,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_node_relations(
        &self,
        layer_idx: usize,
        src_type_id: usize,
        segments: &[RelationSegment],
        node: &NodeRecord,
        layer_spec: &CompiledLayerSpec,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
        cutoff: Option<&NodeCutoff<'_>>,
    ) -> Result<()> {
        let mut dst_scratch = std::mem::take(&mut state.dst_scratch);
        let mut key_scratch = std::mem::take(&mut state.key_scratch);

        let type_local_map = self
            .local_to_merged_rel
            .get(src_type_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for segment in segments {
            let local_rel_id = segment.layout.relation_id as usize;
            // Translate LOCAL → merged before any downstream indexing.
            let merged_rel_u16 = type_local_map
                .get(local_rel_id)
                .copied()
                .unwrap_or(u16::MAX);
            if merged_rel_u16 == u16::MAX {
                continue;
            }
            let rel_id = merged_rel_u16 as usize;
            let fanout_spec = layer_spec
                .fanouts
                .get(&rel_id)
                .or(layer_spec.default_fanout.as_ref());

            if let Some(fanout_spec) = fanout_spec {
                let dst_type_id = *self.relation_info.get(rel_id).unwrap_or(&usize::MAX);
                if dst_type_id != usize::MAX {
                    self.sample_segment_edges(
                        node,
                        rel_id,
                        segment,
                        fanout_spec,
                        request,
                        &mut dst_scratch,
                        &mut key_scratch,
                        cutoff,
                        |dst_global_id| {
                            self.record_edge(
                                layer_idx, node.local_id, node.seed_index, node.seed_type_id,
                                dst_global_id, dst_type_id, rel_id as u16,
                                request.flags.swap_edge_direction, state, output)
                        }
                    )?;
                }
            }
        }
        state.dst_scratch = dst_scratch;
        state.key_scratch = key_scratch;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_edge(
        &self,
        layer_idx: usize,
        src_local_id: u32,
        seed_index: u32,
        seed_type_id: u32,
        dst_global_id: u64,
        dst_type_id: usize,
        rel_id: u16,
        swap_edge_direction: bool,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let dst_type_state = state.type_states.entry(dst_type_id).or_default();
        let key = LookupKey {
            seed_type_id,
            seed_index,
            global_id: dst_global_id,
        };

        let dst_local_id = if let Some(&id) = dst_type_state.index_map.get(&key) {
            id
        } else {
            let id = dst_type_state.nodes.len() as u32;
            dst_type_state.index_map.insert(key, id);

            dst_type_state.nodes.push(NodeRecord {
                global_id: dst_global_id,
                seed_index,
                local_id: id,
                seed_type_id,
            });
            state.stats.record_node(layer_idx + 1, dst_type_id);

            dst_type_state.next_frontier.push(NodeRecord {
                global_id: dst_global_id,
                seed_index,
                local_id: id,
                seed_type_id,
            });
            id
        };

        // Apply edge direction reversal if requested (PyG CSC convention).
        let (actual_rel, actual_src, actual_dst) = if swap_edge_direction {
            match self.reverse_rel_map.get(rel_id as usize) {
                Some(Some(rev_id)) => (*rev_id as u16, dst_local_id, src_local_id),
                _ => (rel_id, src_local_id, dst_local_id),
            }
        } else {
            (rel_id, src_local_id, dst_local_id)
        };

        output.handle_edge(layer_idx, actual_rel, actual_src, actual_dst)?;
        state.stats.record_edge(layer_idx, actual_rel as usize);

        Ok(())
    }
}

/// Project a seed-rowpack `RecordBatch` into the two consumer buckets
/// `sample_runtime_batch` needs:
///
///   - `task.target` — `TARGET*` columns only. The supervised signal
///     PR-28765's adapter reads (and turns into `hetero[...].y` etc).
///     Returns `None` when the seed rowpack has no TARGET columns
///     (legitimate for PassThrough-mode pred batches).
///
///   - `raw_features[<stage_table>]` — every column that is neither task
///     identity (`ENTITY*`) nor temporal anchor (`TIMESTAMP`) nor target.
///     This is what the adapter routes into `hetero[train_table_name].tf`.
///     Returns `None` when the seed rowpack has no feature columns (the
///     usual case for PQ plans today; pre-wires the bucket for when tt-ng
///     starts emitting `FEAT*` / `LAG_FEAT*` per the ML team contract).
///
/// Column class is decided by prefix match. We deliberately use prefix
/// rather than reading the manifest because (a) the manifest isn't in
/// engine scope, (b) the kumo-side and tt-ng-side both use these
/// conventions (`PRIMARY_ENTITY_COL = "ENTITY1"`, `PRIMARY_TARGET_COL =
/// "TARGET1"`, `TIMESTAMP_COL = "TIMESTAMP"`), and (c) the prefix-based
/// rule generalizes for free to multi-entity / multi-target / autoregression
/// outputs.
/// Read one Arrow cell as a [`Value`], covering the node-id-eligible dtypes
/// (Int64 / Float64 / Utf8). Returns `None` for nulls and unsupported dtypes
/// (caller treats `None` as an unresolved/masked row). Mirrors the helper of
/// the same name in `diskgraph-train-table::custom_tt_chunk_processor`.
fn cell_to_value(arr: &dyn arrow_array::Array, idx: usize) -> Option<Value> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Some(Value::Int(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        return Some(Value::Int(a.value(idx) as i64));
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Some(Value::Float(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Some(Value::Str(a.value(idx).to_string()));
    }
    None
}

/// Read one element of an Arrow `List` cell's child array as a [`Value`].
/// `child` is the list's flat values array; `pos` is the absolute index into
/// it. Mirrors [`cell_to_value`] but for an element of a list child.
fn list_child_to_value(child: &dyn arrow_array::Array, pos: usize) -> Option<Value> {
    cell_to_value(child, pos)
}

/// Result of rooting ONE positive per anchor for the two-tower runtime mode
/// (FK-free design §S4). Anchor-major, length `B`.
pub(crate) struct PairFromArray {
    /// Chosen positive's global rhs node id per anchor, length `B`. `0` for an
    /// anchor with no positives (an empty `target_node_ids` list) or an
    /// unresolved value — the slot is a dense placeholder the loss must ignore
    /// via `mask`.
    pub chosen_rhs: Vec<u64>,
    /// Per-anchor validity mask, length `B`. `true` == this anchor has no usable
    /// positive (empty list / unresolved); the rooted id is a placeholder.
    /// Same convention as `lp_pos_mask` / `neg_mask` (`true` == drop).
    pub mask: Vec<bool>,
    /// Offset of the rooted positive, so a per-element column can be sampled
    /// at the same place. `None` when masked.
    pub chosen_offset: Vec<Option<usize>>,
}

/// `pair_from_array` (FK-free design §S4): explode a two-tower anchor's
/// `target_node_ids` list to **one rooted positive per anchor**, reproducing
/// legacy pair-mode + inverse-degree edge sampling without exploding the table.
///
/// For each of the `B` seed rows, read its positive list; if non-empty, sample
/// one offset `j ~ Uniform(0, len)` with a deterministic per-row `ChaCha8Rng`
/// (a per-`row` seed folded from `rnd_seed`, mirroring
/// [`append_global_negatives_row`]), then map that one element to a global rhs
/// node id via `resolve`. An empty list or an unresolved element yields id `0`
/// + `mask=true`. The result is anchor-major (length `B`).
///
/// On the FK-free resolution path the column is already `target_node_ids:
/// list<i64>` whose values ARE graph node-ids, so the caller passes the IDENTITY
/// closure (`Value::Int(i) => Some(i as u64)`) — no MPH / `reader.table()`.
///
/// **Epoch coverage:** `rnd_seed` is the `NegativeSamplingSpec` seed (shared with
/// the negative draw). `None` (the training/prod default) ⇒ `thread_rng`, so each
/// row picks a *fresh* positive every epoch — covering a multi-positive row's
/// positives over training. `Some(x)` ⇒ deterministic per-`(row)` pick (tests).
pub(crate) fn pair_from_array(
    col: &dyn arrow_array::Array,
    rnd_seed: Option<u64>,
    mut resolve: impl FnMut(&Value) -> Option<u64>,
) -> PairFromArray {
    use arrow_array::Array;
    use rand::{Rng, SeedableRng};

    let n = col.len();
    let mut out = PairFromArray {
        chosen_rhs: Vec::with_capacity(n),
        mask: Vec::with_capacity(n),
        chosen_offset: Vec::with_capacity(n),
    };

    // Sample one offset within a row's [0, len) list. Mirrors the negative
    // sampler (`append_global_negatives_row`): `Some(base)` ⇒ deterministic
    // per-row `base.wrapping_add(i)` ChaCha8 (reproducible); `None` ⇒
    // `thread_rng` (random, fresh per epoch — the training/prod default).
    let sample_offset = |i: usize, len: usize| -> usize {
        debug_assert!(len > 0);
        match rnd_seed {
            Some(base) => {
                let mut rng =
                    rand_chacha::ChaCha8Rng::seed_from_u64(base.wrapping_add(i as u64));
                rng.gen_range(0..len)
            }
            None => rand::thread_rng().gen_range(0..len),
        }
    };

    // Resolve the chosen element of row `i`'s list (or the scalar fallback) into
    // (id, mask) and push it. Shared by the List / LargeList / scalar arms.
    let mut push_choice = |out: &mut PairFromArray, value: Option<Value>, offset: Option<usize>| {
        match value.and_then(|v| resolve(&v)) {
            Some(id) => {
                out.chosen_rhs.push(id);
                out.mask.push(false);
                out.chosen_offset.push(offset);
            }
            None => {
                out.chosen_rhs.push(0);
                out.mask.push(true);
                out.chosen_offset.push(None);
            }
        }
    };

    if let Some(list) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        let child = list.values();
        for i in 0..n {
            if list.is_null(i) {
                out.chosen_rhs.push(0);
                out.mask.push(true);
                out.chosen_offset.push(None);
                continue;
            }
            let start = list.value_offsets()[i] as usize;
            let end = list.value_offsets()[i + 1] as usize;
            let len = end - start;
            if len == 0 {
                out.chosen_rhs.push(0);
                out.mask.push(true);
                out.chosen_offset.push(None);
                continue;
            }
            let off = sample_offset(i, len);
            push_choice(&mut out, list_child_to_value(child.as_ref(), start + off), Some(off));
        }
        return out;
    }
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
        let child = list.values();
        for i in 0..n {
            if list.is_null(i) {
                out.chosen_rhs.push(0);
                out.mask.push(true);
                out.chosen_offset.push(None);
                continue;
            }
            let start = list.value_offsets()[i] as usize;
            let end = list.value_offsets()[i + 1] as usize;
            let len = end - start;
            if len == 0 {
                out.chosen_rhs.push(0);
                out.mask.push(true);
                out.chosen_offset.push(None);
                continue;
            }
            let off = sample_offset(i, len);
            push_choice(&mut out, list_child_to_value(child.as_ref(), start + off), Some(off));
        }
        return out;
    }

    // Scalar single-positive fallback: one positive per row, no offset sampling.
    for i in 0..n {
        push_choice(&mut out, cell_to_value(col, i), Some(0));
    }
    out
}

/// Ragged multi-positive label payload produced from the resolved
/// `target_node_ids` column (FK-free design §S5). All vectors are anchor-major.
/// `counts` has length `B` (number of seed rows); `rhs_global` / `mask` have
/// length `ΣP` (sum of counts).
#[derive(Debug, Default, Clone)]
pub(crate) struct RaggedLpTarget {
    /// Per-anchor positive count, length `B`.
    pub counts: Vec<u32>,
    /// Global rhs node id per positive, length `ΣP` (`0` if unresolved).
    pub rhs_global: Vec<i64>,
    /// Per-positive validity mask, length `ΣP`. `true` = unresolved/drop.
    pub mask: Vec<bool>,
}

/// Resolve one positive value and append (id, mask) to the ragged payload.
fn push_resolved(
    out: &mut RaggedLpTarget,
    value: Option<Value>,
    resolve: &mut impl FnMut(&Value) -> Option<u64>,
) {
    match value.and_then(|v| resolve(&v)) {
        Some(id) => {
            out.rhs_global.push(id as i64);
            out.mask.push(false);
        }
        None => {
            out.rhs_global.push(0);
            out.mask.push(true);
        }
    }
}

/// Build the anchor-major ragged LP label payload from a resolved target
/// column. `resolve` maps each positive [`Value`] to a global rhs node id.
///
/// On the FK-free resolution path the column is `target_node_ids: list<i64>`
/// whose values are ALREADY graph node-ids, so the caller passes an identity
/// closure (`Value::Int(i) => Some(i as u64)`) — no MPH / `reader.table()`
/// lookup. Both `List` (i32 offsets) and `LargeList` (i64 offsets) are handled,
/// plus a scalar single-positive fallback. Unresolved / null values push id `0`
/// with mask `true` (never aborts a batch).
pub(crate) fn hydrate_ragged_lp_target(
    col: &dyn arrow_array::Array,
    mut resolve: impl FnMut(&Value) -> Option<u64>,
) -> RaggedLpTarget {
    use arrow_array::Array;
    let n = col.len();
    let mut out = RaggedLpTarget {
        counts: Vec::with_capacity(n),
        rhs_global: Vec::new(),
        mask: Vec::new(),
    };

    if let Some(list) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        let child = list.values();
        for row in 0..n {
            if list.is_null(row) {
                out.counts.push(0);
                continue;
            }
            let start = list.value_offsets()[row] as usize;
            let end = list.value_offsets()[row + 1] as usize;
            out.counts.push((end - start) as u32);
            for pos in start..end {
                push_resolved(&mut out, list_child_to_value(child.as_ref(), pos), &mut resolve);
            }
        }
        return out;
    }
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
        let child = list.values();
        for row in 0..n {
            if list.is_null(row) {
                out.counts.push(0);
                continue;
            }
            let start = list.value_offsets()[row] as usize;
            let end = list.value_offsets()[row + 1] as usize;
            out.counts.push((end - start) as u32);
            for pos in start..end {
                push_resolved(&mut out, list_child_to_value(child.as_ref(), pos), &mut resolve);
            }
        }
        return out;
    }

    // Scalar single-positive fallback: one positive per row.
    for i in 0..n {
        out.counts.push(1);
        push_resolved(&mut out, cell_to_value(col, i), &mut resolve);
    }
    out
}

pub(crate) fn project_seed_rowpack(
    rb: &arrow_array::RecordBatch,
) -> Result<(Option<arrow_array::RecordBatch>, Option<arrow_array::RecordBatch>)> {
    let mut target_indices = Vec::new();
    let mut feature_indices = Vec::new();
    // Routes TARGET → labels rather than skipping it, so it deliberately does
    // NOT use `is_reserved_feature_column` (which treats TARGET<n> as never-a-
    // feature). `starts_with` is safe here — only runs on the controlled
    // seed/stage type. Order matters: TARGET first.
    for (idx, field) in rb.schema().fields().iter().enumerate() {
        let name = field.name();
        if name.starts_with("TARGET") {
            target_indices.push(idx);
        } else if name.starts_with("ENTITY")
            || name == "TIMESTAMP"
            || name == "entity_node_id"
            || name == "target_node_ids"
            || name == "exclude_node_ids"
            || name == WEIGHT_COL
            || name == FREQUENCY_COL
            || name == SPLIT_COL
        {
            // identity / anchor / FK-free resolution / exclude / supervision
            // weight columns: already lifted into task fields, never routed into
            // raw features. `split_col` (custom-TT split signal) is dropped at
            // routing, so it should never appear here; skipped belt-and-suspenders.
            continue;
        } else {
            feature_indices.push(idx);
        }
    }
    let project = |indices: &[usize]| -> Result<Option<arrow_array::RecordBatch>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let projected = rb
            .project(indices)
            .map_err(|e| EngineError::Internal(format!("project seed rowpack: {e}")))?;
        Ok(Some(projected))
    };
    Ok((project(&target_indices)?, project(&feature_indices)?))
}

fn read_lp_rooted_weights(
    rb: &arrow_array::RecordBatch,
    offsets: &[Option<usize>],
    b: usize,
) -> Result<Option<Vec<f32>>> {
    use arrow_array::Array;

    if !lp_list_weight_present(rb) {
        return Ok(None);
    }
    let idx = rb.schema().index_of(WEIGHT_COL).expect("checked above");
    let col = rb.column(idx);
    let (values, offs): (arrow_array::ArrayRef, Vec<usize>) =
        if let Some(l) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
            (
                l.values().clone(),
                l.value_offsets().iter().map(|&o| o as usize).collect(),
            )
        } else if let Some(l) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            (
                l.values().clone(),
                l.value_offsets().iter().map(|&o| o as usize).collect(),
            )
        } else {
            return Ok(None);
        };
    (0..b)
        .map(
            |i| match (offsets.get(i).copied().flatten(), offs.get(i), offs.get(i + 1)) {
                (Some(off), Some(&start), Some(&end)) if start + off < end => {
                    lp_weight_value(values.as_ref(), start + off)
                }
                _ => Ok(1.0),
            },
        )
        .collect::<Result<Vec<f32>>>()
        .map(Some)
}

fn lp_list_weight_present(rb: &arrow_array::RecordBatch) -> bool {
    rb.schema().index_of(WEIGHT_COL).is_ok_and(|idx| {
        matches!(
            rb.column(idx).data_type(),
            arrow_schema::DataType::List(_) | arrow_schema::DataType::LargeList(_)
        )
    })
}

fn read_lp_list_weights(
    rb: &arrow_array::RecordBatch,
    counts: &[u32],
) -> Result<Option<Vec<f32>>> {
    if !lp_list_weight_present(rb) {
        return Ok(None);
    }
    let idx = rb.schema().index_of(WEIGHT_COL).expect("checked above");
    let col = rb.column(idx);
    let weights = hydrate_ragged_lp_weights(col.as_ref())?;
    let expected = counts.iter().map(|&c| c as usize).sum::<usize>();
    if weights.len() != expected {
        return Err(EngineError::DataCorruption(format!(
            "seed rowpack `{WEIGHT_COL}` length mismatch: got {} weights for {} positives",
            weights.len(),
            expected
        )));
    }
    Ok(Some(weights))
}

pub(crate) fn hydrate_ragged_lp_weights(col: &dyn arrow_array::Array) -> Result<Vec<f32>> {
    use arrow_array::Array;

    let mut out = Vec::new();
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        let child = list.values();
        for row in 0..list.len() {
            if list.is_null(row) {
                continue;
            }
            let start = list.value_offsets()[row] as usize;
            let end = list.value_offsets()[row + 1] as usize;
            for pos in start..end {
                out.push(lp_weight_value(child.as_ref(), pos)?);
            }
        }
        return Ok(out);
    }
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
        let child = list.values();
        for row in 0..list.len() {
            if list.is_null(row) {
                continue;
            }
            let start = list.value_offsets()[row] as usize;
            let end = list.value_offsets()[row + 1] as usize;
            for pos in start..end {
                out.push(lp_weight_value(child.as_ref(), pos)?);
            }
        }
        return Ok(out);
    }

    for row in 0..col.len() {
        out.push(lp_weight_value(col, row)?);
    }
    Ok(out)
}

fn lp_weight_value(col: &dyn arrow_array::Array, idx: usize) -> Result<f32> {
    if col.is_null(idx) {
        return Ok(1.0);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Ok(a.value(idx) as f32);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        return Ok(a.value(idx) as f32);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Ok(a.value(idx) as f32);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow_array::Float32Array>() {
        return Ok(a.value(idx));
    }
    Err(EngineError::DataCorruption(format!(
        "seed rowpack ragged weight column must contain numeric weights, got {:?}",
        col.data_type()
    )))
}

/// Read the reserved `WEIGHT` (Float64) column off the seed rowpack batch as
/// one `f32` per row, in `batch_row_ids` order (so weights align 1:1 with
/// `task` rows). `Ok(None)` when absent (PQ-path / no-weight). The weight rides
/// the row through gen → pass-B → stage parquet, so it can't drift — unlike the
/// old `weight.bin` sidecar (global/local mismatch under interleaved splits).
fn read_weight_column(rb: &arrow_array::RecordBatch) -> Result<Option<Vec<f32>>> {
    use arrow_array::Array;
    let Ok(idx) = rb.schema().index_of(WEIGHT_COL) else {
        return Ok(None);
    };
    let col = rb.column(idx);
    if lp_list_weight_present(rb) {
        return Ok(None);
    }
    let arr = col
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .ok_or_else(|| {
            EngineError::DataCorruption(format!(
                "seed rowpack `{WEIGHT_COL}` column must be Float64, got {:?}",
                col.data_type()
            ))
        })?;
    let mut out = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        // Null weight → 1.0 (neutral). Custom-TT gen writes a finite weight
        // per emitted row, so nulls are not expected, but be defensive.
        out.push(if arr.is_null(i) { 1.0 } else { arr.value(i) as f32 });
    }
    Ok(Some(out))
}

#[cfg(test)]
#[path = "pseudo_time_tests.rs"]
mod pseudo_time_tests;
