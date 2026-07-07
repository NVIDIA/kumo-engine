use crate::errors::Result;
use crate::state::SamplingState;

/// Trait for handling the output stream of the sampler.
pub trait OutputHandler {
    /// Called when an edge is traversed.
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<()>;

    /// Called before sending feature batches to define schema.
    fn handle_schema(&mut self, schema: arrow_schema::SchemaRef) -> Result<()>;

    /// Called when a batch of features is decoded.
    fn handle_feature_batch(&mut self, batch: arrow_array::RecordBatch) -> Result<()>;

    /// Called during sampling when a batch of fused features is ready for a type.
    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: arrow_array::RecordBatch,
    ) -> Result<()>;

    /// Called at the end of sampling to flush nodes and stats.
    fn finalize(&mut self, state: &SamplingState) -> Result<()>;
}

/// A simple in-memory accumulator for testing/benchmarking.
pub struct InMemoryOutput {
    pub edges: Vec<(usize, u16, u32, u32)>, // (hop, rel, src, dst)
    pub feature_batches: Vec<arrow_array::RecordBatch>,
    pub schema: Option<arrow_schema::SchemaRef>,
    pub fused_batches: Vec<(usize, arrow_array::RecordBatch)>,
}

impl Default for InMemoryOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryOutput {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            feature_batches: Vec::new(),
            schema: None,
            fused_batches: Vec::new(),
        }
    }
}

impl OutputHandler for InMemoryOutput {
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<()> {
        self.edges.push((hop, rel_id, src_local_id, dst_local_id));
        Ok(())
    }

    fn handle_schema(&mut self, schema: arrow_schema::SchemaRef) -> Result<()> {
        self.schema = Some(schema);
        Ok(())
    }

    fn handle_feature_batch(&mut self, batch: arrow_array::RecordBatch) -> Result<()> {
        self.feature_batches.push(batch);
        Ok(())
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: arrow_array::RecordBatch,
    ) -> Result<()> {
        self.fused_batches.push((type_id, batch));
        Ok(())
    }

    fn finalize(&mut self, _state: &SamplingState) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RuntimeBatch — the shape PR-28765's `runtime_batch_to_hetero` consumes.
// Engine emits via this collator; pybind converts to PyDict.
//
// Mostly Arrow data already produced by the engine. Three additions over
// what InMemoryOutput captures:
//   - per-channel batch_dict (which seed each emitted node descends from)
//   - per-channel node_ids (the deduplicated set per type)
//   - the task bag (seed rowpack slice for the batch — input_ids, target,
//     timestamps, weight, seed_node_ids resolved via seed.edges.bin)
// ---------------------------------------------------------------------------

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use std::collections::HashMap;

/// The task bag — per-row data from the seed table for this batch. Maps onto
/// `task.*` in PR-28765's runtime batch dict.
#[derive(Debug, Default, Clone)]
pub struct TaskBag {
    /// 0..batch_size, one per seed row. Position within this batch.
    pub input_ids: Vec<u32>,
    /// Seed table row ids in the stage sub-index (local to the stage table).
    pub tt_stage_node_ids: Vec<u64>,
    /// Parent entity row ids — one per seed row. Resolved via walking
    /// `seed.edges.bin` at hop 0 (the cross-boundary FK pre-resolved this
    /// at Phase 1 ingest time, so no PK lookup needed here).
    pub seed_node_ids: Vec<u64>,
    /// Per-row TIMESTAMP from seed rowpack; `None` for non-temporal tasks
    /// where the seed table has no TIMESTAMP column.
    pub seed_timestamps: Option<Vec<i64>>,
    /// Per-row TARGET1 values from seed rowpack, as a RecordBatch so we
    /// preserve whatever dtype TT-NG emitted (Bool / Float64 / Int64 / etc.).
    pub target: Option<RecordBatch>,
    /// Per-row weight from the weight sidecar (custom-TT path only).
    pub weight: Option<Vec<f32>>,
    /// Engine-side global negatives drawn per positive (M). `0` when no
    /// negative sampling was requested. The rhs node set in
    /// `graphs[..].node_ids[rhs_seed_type]` is then `[B positives,
    /// B*amount_m negatives]` anchor-major.
    pub amount_m: u32,
    /// The positive count B for this batch (always the batch's row count, with
    /// or without negative sampling). Python uses this to split the rhs node
    /// set into positives/negatives without re-deriving the batch size.
    pub num_positives: u32,
    /// The RHS node type negatives were drawn from. `None` when no negative
    /// sampling was requested.
    pub rhs_seed_type: Option<String>,
    /// Per-negative validity mask, length `B*amount_m`, anchor-major and
    /// parallel to the `[B:B+B*M]` rhs negative slots in
    /// `graphs[..].node_ids[rhs_seed_type]`. `true` = use the slot in the loss;
    /// `false` = the temporal-resample budget was exhausted, so the slot holds a
    /// dense placeholder id (a real RHS id, still GNN-encoded) that the loss
    /// must ignore. Empty when no negative sampling was requested.
    pub neg_mask: Vec<bool>,
    // ── Ragged multi-positive LP label payload (FK-free resolution, design §S5)
    // Populated only on the LP label path (shallow_rhs / ID-GNN / hybrid) when a
    // `target_node_ids: list<i64>` resolution column is present on the seed
    // rowpack. The positives are LABELS, never rooted — `seed_nodes[rhs]` stays
    // empty, so there is no message-passing leak. Positives are ANCHOR-MAJOR:
    // anchor 0's positives, then anchor 1's, … . With `B` anchors and `ΣP` total
    // positives:
    //
    /// Per-anchor positive count, length `B` (rowptr deltas; cumsum → offsets).
    pub lp_pos_counts: Vec<u32>,
    /// Global rhs node id per positive, length `ΣP` (`0` if unresolved). On the
    /// resolution path the ids come straight off the `target_node_ids` column
    /// (already graph node-ids), so this is an identity copy.
    pub lp_pos_rhs_global: Vec<i64>,
    /// Per-positive validity mask, length `ΣP`. `true` = unresolved/drop; the
    /// loss must ignore masked positives.
    pub lp_pos_mask: Vec<bool>,
    /// Per-positive edge weight, length `ΣP`. `None` when no weights are present.
    /// Unused on the label-only S3 path (left `None`); reserved for S4.
    pub lp_pos_weight: Option<Vec<f32>>,
    /// Index into `node_ids[rhs]` of each rooted positive; length `ΣP`. Only set
    /// when rhs positives are rooted as seeds (two-tower / hybrid rooting, S4).
    /// `None` on the label-only S3 path (positives are never rooted).
    pub lp_pos_local: Option<Vec<u32>>,
    // STATIC LP past-visit exclusion (eval-only): anchor-major, cumsum(counts)
    // is the rowptr. Hydrated from the `exclude_node_ids` rowpack column (gen
    // writes it on val/test seed rows only) — see `ttng-past-visit-design.md`.
    /// Per-anchor exclude count, length `B`. Empty when the column is absent
    /// (train / predict seed tables).
    pub lp_exclude_counts: Vec<u32>,
    /// Global rhs node id per excluded target, length `ΣE`. Empty when absent.
    pub lp_exclude_rhs_global: Vec<i64>,
}

/// Per-channel graph topology, one channel for node prediction ("main").
#[derive(Debug, Default, Clone)]
pub struct GraphChannel {
    pub node_ids: HashMap<String, Vec<u64>>,
    pub edges: Vec<(Vec<u32>, Vec<u32>)>,
    pub batch_dict: HashMap<String, Vec<u32>>,
    pub rel_id_to_names: HashMap<u16, (String, String, String)>,
    /// Per-hop sampled node counts per node-type name, PyG-style:
    /// `name -> [hop0, hop1, ...]` (hop0 = seeds). Mirrors PyG
    /// NeighborLoader's `num_sampled_nodes`. Emitted for every node type
    /// the plan can reach, including all-zero ones, so the adapter can
    /// reconstruct the full per-hop structure.
    pub num_sampled_nodes: HashMap<String, Vec<u64>>,
    /// Per-hop sampled edge counts per edge-type triple, PyG-style:
    /// `(src, rel, dst) -> [hop0, hop1, ...]`. Mirrors PyG
    /// `num_sampled_edges`. Seed-src relations (dropped from topology) are
    /// excluded so counts match the emitted edges.
    pub num_sampled_edges: HashMap<(String, String, String), Vec<u64>>,
}

/// The full runtime batch — shape PR-28765's `runtime_batch_to_hetero` adapter
/// consumes. Pybind translates this struct into a PyDict at the FFI boundary.
#[derive(Debug, Default, Clone)]
pub struct RuntimeBatch {
    pub task: TaskBag,
    pub graphs: HashMap<String, GraphChannel>,
    /// raw_features by namespace name. Each namespace's record batches are
    /// concatenated to match its node_ids order. The seed namespace
    /// (e.g. "tt_id/train_seeds") is included here only when the seed has
    /// feature columns; otherwise `hetero[train_table_name].tf` stays empty.
    pub raw_features: HashMap<String, RawFeatureBank>,
}

#[derive(Debug, Default, Clone)]
pub struct RawFeatureBank {
    pub schema: Option<SchemaRef>,
    pub batches: Vec<RecordBatch>,
}

/// An OutputHandler that collates engine events into a `RuntimeBatch`.
///
/// The seed namespaces (the stage seed tables, e.g. `<tt_id>/train_seeds`,
/// `<tt_id>/val_seeds`, `<tt_id>/test_seeds`, `<tt_id>/predict_seeds`) are
/// storage/bookkeeping types, NOT model graph node types — they have no encoder
/// on the kumo side. The merged TT catalog contains ALL of a tt's stage seed
/// tables as node types (with a `seed→ENTITY` relation each), not just the
/// stage currently being sampled, so EVERY stage's seed namespace must be
/// excluded from the emitted GRAPH TOPOLOGY:
///   - EDGES: relation ids whose src is any seed table (`seed_src_rel_ids`) are
///     dropped at `handle_edge` time. `num_sampled_edges` additionally drops
///     any relation with a seed-type endpoint (sibling stages don't traverse
///     edges, but the pybind emits every catalog edge type — including the
///     sibling stages' all-zero `seed→ENTITY` relations — so they must be
///     filtered here or they leak into the kumo adapter's node-type back-fill).
///   - NODES: any seed type (`seed_type_ids`) is skipped when populating
///     `node_ids` / `batch_dict` / `num_sampled_nodes` in `finalize`.
///
/// The current stage's seed `raw_features` entry is NOT topology and is
/// preserved — it carries the train-table / seed feature columns consumed by
/// the kumo adapter.
pub struct RuntimeBatchOutput {
    /// Channel name to write into (we only use "main" for node prediction).
    pub channel: String,
    /// Relation ids whose src_type is any seed namespace — these edges are
    /// dropped before being captured.
    pub seed_src_rel_ids: std::collections::HashSet<u16>,
    /// Map from numeric rel_id → (src_type_name, rel_name, dst_type_name).
    /// Provided at construction by the engine so the collator can produce
    /// stringly-typed edge_type tuples for the dict.
    pub rel_id_to_names: HashMap<u16, (String, String, String)>,
    /// Map from numeric node type id → name. Same purpose for node_ids.
    pub type_id_to_name: Vec<String>,
    /// Numeric type ids of ALL of the tt's stage seed tables (train/val/test/
    /// predict). Excluded from the emitted topology (`node_ids` / `batch_dict`
    /// / `num_sampled_nodes`, and any `num_sampled_edges` relation touching one
    /// as src or dst), symmetric with the seed-src EDGE drop. Empty means "no
    /// seed namespaces to exclude" (e.g. the NodeRange path).
    pub seed_type_ids: std::collections::HashSet<usize>,
    /// Namespace name for the currently-flushing feature batches (set by
    /// pybind orchestrator before each `fetch_features_for_type` call).
    pub current_features_namespace: Option<String>,
    pub batch: RuntimeBatch,
}

impl RuntimeBatchOutput {
    pub fn new(
        channel: impl Into<String>,
        seed_src_rel_ids: std::collections::HashSet<u16>,
        rel_id_to_names: HashMap<u16, (String, String, String)>,
        type_id_to_name: Vec<String>,
        seed_type_ids: std::collections::HashSet<usize>,
    ) -> Self {
        Self {
            channel: channel.into(),
            seed_src_rel_ids,
            rel_id_to_names,
            type_id_to_name,
            seed_type_ids,
            current_features_namespace: None,
            batch: RuntimeBatch::default(),
        }
    }

    /// Set the namespace that subsequent `handle_feature_batch` / `handle_schema`
    /// calls belong to. Use before invoking `engine.fetch_features_for_type(...)`
    /// for that namespace's type.
    pub fn set_namespace(&mut self, namespace: impl Into<String>) {
        self.current_features_namespace = Some(namespace.into());
    }

    pub fn into_runtime_batch(self) -> RuntimeBatch {
        self.batch
    }
}

impl OutputHandler for RuntimeBatchOutput {
    fn handle_edge(
        &mut self,
        _hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> Result<()> {
        if self.seed_src_rel_ids.contains(&rel_id) {
            // Seed-src edges are dropped — the seed is task data, never
            // appears as topology in the emitted dict.
            return Ok(());
        }
        let channel = self
            .batch
            .graphs
            .entry(self.channel.clone())
            .or_default();
        let rel_idx = rel_id as usize;
        if rel_idx >= channel.edges.len() {
            channel.edges.resize(rel_idx + 1, (Vec::new(), Vec::new()));
        }
        let entry = &mut channel.edges[rel_idx];
        entry.0.push(src_local_id);
        entry.1.push(dst_local_id);
        Ok(())
    }

    fn handle_schema(&mut self, schema: SchemaRef) -> Result<()> {
        if let Some(ns) = &self.current_features_namespace {
            self.batch
                .raw_features
                .entry(ns.clone())
                .or_default()
                .schema = Some(schema);
        }
        Ok(())
    }

    fn handle_feature_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if let Some(ns) = &self.current_features_namespace {
            self.batch
                .raw_features
                .entry(ns.clone())
                .or_default()
                .batches
                .push(batch);
        }
        Ok(())
    }

    fn handle_fused_features(&mut self, type_id: usize, batch: RecordBatch) -> Result<()> {
        // Fused features are tagged by type id (no namespace ambiguity). Map
        // to the type name and use that as the namespace.
        let name = self
            .type_id_to_name
            .get(type_id)
            .cloned()
            .unwrap_or_else(|| format!("type_{type_id}"));
        self.batch
            .raw_features
            .entry(name)
            .or_default()
            .batches
            .push(batch);
        Ok(())
    }

    fn finalize(&mut self, state: &SamplingState) -> Result<()> {
        // Populate node_ids + batch_dict from final SamplingState. Seed
        // namespaces (all stages of the tt) are storage types, not model node
        // types, so they are excluded from topology — symmetric with the
        // seed-src EDGE exclusion below (and in `handle_edge`). The current
        // stage's seed `raw_features` entry is untouched.
        let channel = self
            .batch
            .graphs
            .entry(self.channel.clone())
            .or_default();
        channel.rel_id_to_names = self.rel_id_to_names.clone();
        for (type_id, type_state) in state.type_states.iter() {
            if self.seed_type_ids.contains(type_id) {
                continue;
            }
            let name = match self.type_id_to_name.get(*type_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            // SamplingState.nodes is the dedup'd, ordered set per type.
            let node_ids: Vec<u64> = type_state.nodes.iter().map(|n| n.global_id).collect();
            let batch_idx: Vec<u32> = type_state.nodes.iter().map(|n| n.seed_index).collect();
            channel.node_ids.insert(name.clone(), node_ids);
            channel.batch_dict.insert(name, batch_idx);
        }

        // Per-hop sampled counts (PyG `num_sampled_nodes` / `num_sampled_edges`).
        // Copied verbatim from the engine's per-hop accounting (`SamplingStats`)
        // so the adapter can reconstruct the layer structure rather than
        // flattening to a single total. Emit for EVERY node type / relation the
        // plan can reach (including all-zero ones) to match PyG NeighborLoader,
        // which lets the trainer/explainer index any schema edge type safely.
        for (type_id, name) in self.type_id_to_name.iter().enumerate() {
            // Seed namespaces are excluded from emitted topology (see node_ids
            // / batch_dict above), so omit them here too — keeps the count map
            // consistent with the node types actually present in the batch.
            if self.seed_type_ids.contains(&type_id) {
                continue;
            }
            let per_hop = state
                .stats
                .nodes_per_type
                .get(type_id)
                .cloned()
                .unwrap_or_default();
            channel.num_sampled_nodes.insert(name.clone(), per_hop);
        }
        // Names of the seed node types, to drop any edge relation that touches
        // one as src OR dst. The pybind emits every catalog edge type (incl.
        // all-zero ones), so sibling stages' `seed→ENTITY` relations would
        // otherwise surface their seed namespace via the adapter's edge-endpoint
        // node-type back-fill.
        let seed_type_names: std::collections::HashSet<&str> = self
            .seed_type_ids
            .iter()
            .filter_map(|id| self.type_id_to_name.get(*id).map(|s| s.as_str()))
            .collect();
        for (rel_id, names) in self.rel_id_to_names.iter() {
            // Drop seed-src relations (current stage; dropped in `handle_edge`)
            // and any relation whose src/dst is a seed namespace (sibling
            // stages), keeping counts consistent with emitted topology.
            let (src, _rel, dst) = names;
            if self.seed_src_rel_ids.contains(rel_id)
                || seed_type_names.contains(src.as_str())
                || seed_type_names.contains(dst.as_str())
            {
                continue;
            }
            let per_hop = state
                .stats
                .edges_per_type
                .get(*rel_id as usize)
                .cloned()
                .unwrap_or_default();
            channel.num_sampled_edges.insert(names.clone(), per_hop);
        }
        Ok(())
    }
}
