// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gxhash::{HashMap as GxHashMap, HashMapExt};
use parking_lot::Mutex;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Aggregated statistics for the sampling request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingStats {
    /// Total nodes sampled per type ID, indexed by hop: TypeID -> [Hop0Count, Hop1Count, ...]
    pub nodes_per_type: Vec<Vec<u64>>,
    /// Total edges sampled per relation ID, indexed by hop: RelID -> [Hop0Count, Hop1Count, ...]
    pub edges_per_type: Vec<Vec<u64>>,
    /// Total unique nodes sampled.
    pub total_nodes: u64,
    /// Total edges sampled.
    pub total_edges: u64,
}

impl SamplingStats {
    pub fn new() -> Self {
        Self {
            nodes_per_type: Vec::new(),
            edges_per_type: Vec::new(),
            total_nodes: 0,
            total_edges: 0,
        }
    }

    pub fn reset(&mut self) {
        for v in &mut self.nodes_per_type {
            v.clear();
        }
        for v in &mut self.edges_per_type {
            v.clear();
        }
        self.total_nodes = 0;
        self.total_edges = 0;
    }

    pub fn resize(&mut self, num_types: usize, num_rels: usize, num_hops: usize) {
        if self.nodes_per_type.len() < num_types {
            self.nodes_per_type.resize(num_types, Vec::new());
        }
        for i in 0..num_types {
            self.nodes_per_type[i].resize(num_hops + 1, 0);
            // Ensure values are zero (since we reused vectors from pool)
            self.nodes_per_type[i].fill(0);
        }

        if self.edges_per_type.len() < num_rels {
            self.edges_per_type.resize(num_rels, Vec::new());
        }
        for i in 0..num_rels {
            self.edges_per_type[i].resize(num_hops, 0);
            self.edges_per_type[i].fill(0);
        }
    }

    pub fn record_node(&mut self, hop: usize, type_id: usize) {
        if self.nodes_per_type.len() <= type_id {
            self.nodes_per_type.resize(type_id + 1, Vec::new());
        }
        let counts = &mut self.nodes_per_type[type_id];
        if counts.len() <= hop {
            counts.resize(hop + 1, 0);
        }
        counts[hop] += 1;
        self.total_nodes += 1;
    }

    pub fn record_edge(&mut self, hop: usize, rel_id: usize) {
        if self.edges_per_type.len() <= rel_id {
            self.edges_per_type.resize(rel_id + 1, Vec::new());
        }
        let counts = &mut self.edges_per_type[rel_id];
        if counts.len() <= hop {
            counts.resize(hop + 1, 0);
        }
        counts[hop] += 1;
        self.total_edges += 1;
    }
}

/// Unique identifier for a node instance in the traversal.
///
/// Disjoint sampling keeps the same global node reached from different seeds as
/// distinct instances. `seed_index` is the seed's position *within its own seed
/// type*, so it is NOT unique across types on its own — `seed_type_id` (the root
/// seed type) is required to disambiguate, e.g. LP lhs#0 from rhs#0. Without it,
/// a node reachable from both an lhs and an rhs seed at the same index would
/// wrongly merge into one instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LookupKey {
    pub seed_type_id: u32,
    pub seed_index: u32,
    pub global_id: u64,
}

/// Metadata recorded for every sampled node.
#[derive(Debug, Clone, Copy)]
pub struct NodeRecord {
    pub global_id: u64,
    pub seed_index: u32,
    pub local_id: u32,
    /// Node-type id of the *root seed* this node descends from. Used to look
    /// up the seed's timestamp anchor in the correct per-type array. Seeds set
    /// this to their own type id; children inherit it from their parent.
    pub seed_type_id: u32,
}

/// State specific to a single Node Type (table).
pub struct TypeState {
    /// Maps (Seed, Global) -> LocalID. Holds only the seeds when edges are
    /// recorded through `SamplingState::seed_maps`.
    pub index_map: GxHashMap<LookupKey, u32>,

    /// Ordered list of nodes discovered (Index = LocalID)
    pub nodes: Vec<NodeRecord>,

    /// Nodes to be processed in the next hop for this type
    pub next_frontier: Vec<NodeRecord>,

    /// Pooled Arrow Builders for fused feature fetching.
    pub builders: Option<Vec<Box<dyn arrow_array::builder::ArrayBuilder>>>,
    /// Number of rows currently accumulated in builders.
    pub builder_rows: usize,
    /// Approximate size of accumulated data in bytes.
    pub builder_bytes: usize,
}

impl Default for TypeState {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeState {
    pub fn new() -> Self {
        Self {
            index_map: GxHashMap::with_capacity(1024),
            nodes: Vec::with_capacity(1024),
            next_frontier: Vec::with_capacity(1024),
            builders: None,
            builder_rows: 0,
            builder_bytes: 0,
        }
    }

    /// Resets the state, releasing memory if capacity exceeds the threshold.
    /// Threshold is number of items.
    pub fn adaptive_reset(&mut self, threshold: usize) {
        self.nodes.clear();
        if self.nodes.capacity() > threshold {
            self.nodes.shrink_to(threshold);
        }

        self.next_frontier.clear();
        if self.next_frontier.capacity() > threshold {
            self.next_frontier.shrink_to(threshold);
        }

        self.index_map.clear();
        if self.index_map.capacity() > threshold {
            self.index_map.shrink_to(threshold);
        }

        self.builder_rows = 0;
        self.builder_bytes = 0;
    }
}

/// The feature projection for ONE plan, resolved against ONE graph, indexed by
/// engine `type_id`.
///
/// Resolving means turning the caller's `HashMap<node type name, Vec<column
/// name>>` into a per-type answer the non-fused `fetch_features_for_type` path
/// can index into. This is a pure function of (graph, request), so it is
/// computed ONCE per iterator and shared by `Arc` across every batch, rather
/// than being re-derived from strings at four sites per batch.
///
/// Fused-mode dispatch does not consult this at all: projection pushdown is
/// non-fused-only for now. `RuntimeBatchIterator::build` ignores projections
/// for explicit fused mode and selects non-fused mode for projected automatic
/// dispatch. It is not
/// carried on `SamplingState`: it is resolved once per iterator (in
/// `RuntimeBatchIterator::build`) and threaded explicitly into
/// `sample_runtime_batch` / `sample_node_range_batch`, the only two callers
/// that need it -- `sample()` itself never sees one.
#[derive(Debug, Default)]
pub struct ResolvedProjection {
    /// Per-`type_id` column names for the non-fused path, as a TRI-STATE:
    ///
    /// * `None` -- the request named no entry for this type: SELECT ALL, the
    ///   pre-pushdown behavior.
    /// * `Some([])` -- the caller explicitly asked for NOTHING. The type is
    ///   SKIPPED: no rowpack decode, no `raw_features` block emitted at all.
    /// * `Some([..])` -- materialize exactly these.
    ///
    /// The optionality lives here rather than on the wire type because this
    /// `Vec` is DENSE over `type_id` and so has no "absent" state of its own.
    /// The caller-facing map expresses "select all" by omitting the key.
    pub names: Vec<Option<Vec<String>>>,
}

impl ResolvedProjection {
    /// Column names for the non-fused path, or `None` for SELECT ALL.
    /// Out-of-range ids yield `None`, matching the old per-call helper's
    /// tolerance.
    pub fn names_for(&self, type_id: usize) -> Option<&[String]> {
        self.names.get(type_id).and_then(|n| n.as_deref())
    }

    /// `true` when the caller consumes NO column of this type, so the whole
    /// type can be skipped: no decode, no emitted block.
    ///
    /// DERIVED from `names` rather than stored alongside it, so the skip set
    /// and the name list cannot drift apart.
    pub fn skips(&self, type_id: usize) -> bool {
        matches!(self.names.get(type_id), Some(Some(names)) if names.is_empty())
    }

    /// `true` when this projection is the select-all identity for every type
    /// (equivalent to resolving an empty caller map). Used to detect when a
    /// caller handed a non-trivial projection directly into a fused-dispatch
    /// request, bypassing the iterator's handling of unsupported projections.
    pub fn is_trivial(&self) -> bool {
        self.names.iter().all(|n| n.is_none())
    }
}

/// The main context for a single sampling request.
/// Pooled to avoid allocation churn.
pub struct SamplingState {
    /// Registry of state per node type ID.
    pub type_states: GxHashMap<usize, TypeState>,

    // Scratch buffers for EdgeUnpacker (u64 fast path)
    pub dst_scratch: Vec<u64>,
    pub key_scratch: Vec<u64>,

    // Scratch buffer for Reservoir Sampling
    pub reservoir_scratch: Vec<u64>,

    /// Scratch space for RowPack decoding (shared across types).
    pub decode_scratch: diskgraph_rowpack::RowPackDecodeScratch,

    /// Statistics for the current run.
    pub stats: SamplingStats,

    /// slot = seed_slot_base[seed_type_id] + seed_index
    pub seed_slot_base: Vec<u32>,
    pub seed_slots: usize,
    pub seed_maps: Vec<GxHashMap<u64, u32>>,
    pub commit_scratch: CommitScratch,
    pub raw_edges: Vec<Vec<RawEdge>>,
}

/// A discovered edge before it is formally committed to the SamplingState.
pub struct RawEdge {
    pub src_local_id: u32,
    pub dst_global_id: u64,
    pub dst_type_id: usize,
    pub rel_id: u16,
    pub seed_index: u32,
    /// Root seed type id, propagated from the source node so the destination
    /// node can resolve its timestamp anchor against the correct per-type array.
    pub seed_type_id: u32,
    pub swap_edge_direction: bool,
}

#[derive(Default)]
pub struct CommitScratch {
    pub bounds: Vec<u32>,
    pub cursor: Vec<u32>,
    pub order: Vec<(u64, u32)>,
    pub res: Vec<u32>,
    pub new_edges: Vec<(u64, u32)>,
    pub new_bounds: Vec<u32>,
}

fn clear_and_cap<T>(v: &mut Vec<T>, cap: usize) {
    v.clear();
    if v.capacity() > cap {
        v.shrink_to(cap);
    }
}

const SEED_MAPS_KEEP: usize = 1 << 10;

const RESET_THRESHOLD: usize = 1 << 18; // ~262k items

impl Default for SamplingState {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingState {
    pub fn new() -> Self {
        Self {
            type_states: GxHashMap::default(),
            dst_scratch: Vec::with_capacity(1024),
            key_scratch: Vec::with_capacity(1024),
            reservoir_scratch: Vec::with_capacity(128),
            decode_scratch: diskgraph_rowpack::RowPackDecodeScratch::default(),
            stats: SamplingStats::new(),
            seed_slot_base: Vec::new(),
            seed_slots: 0,
            seed_maps: Vec::new(),
            commit_scratch: CommitScratch::default(),
            raw_edges: Vec::new(),
        }
    }

    fn reset(&mut self) {
        // Recurse into type states
        for state in self.type_states.values_mut() {
            state.adaptive_reset(RESET_THRESHOLD);
        }
        self.dst_scratch.clear();
        self.key_scratch.clear();
        self.reservoir_scratch.clear();
        self.stats.reset();
        self.seed_slot_base.clear();
        self.seed_slots = 0;
        let per_buffer = RESET_THRESHOLD / self.raw_edges.len().max(1);
        for v in &mut self.raw_edges {
            clear_and_cap(v, per_buffer);
        }
        let cs = &mut self.commit_scratch;
        clear_and_cap(&mut cs.bounds, RESET_THRESHOLD);
        clear_and_cap(&mut cs.cursor, RESET_THRESHOLD);
        clear_and_cap(&mut cs.order, RESET_THRESHOLD);
        clear_and_cap(&mut cs.res, RESET_THRESHOLD);
        clear_and_cap(&mut cs.new_edges, RESET_THRESHOLD);
        clear_and_cap(&mut cs.new_bounds, RESET_THRESHOLD);
        self.seed_maps.truncate(SEED_MAPS_KEEP);
        let mut budget = RESET_THRESHOLD;
        for m in &mut self.seed_maps {
            m.clear();
            if m.capacity() > budget {
                m.shrink_to(0);
            }
            budget -= m.capacity();
        }
        // Scratch buffer is inherently reusable
    }
    /// Access or create the state for a specific node type.
    pub fn get_type_state_mut(&mut self, node_type_id: usize) -> &mut TypeState {
        self.type_states.entry(node_type_id).or_default()
    }
}

/// A thread-safe pool of SamplingState objects.
#[derive(Clone)]
#[allow(clippy::vec_box)]
pub struct StatePool {
    pool: Arc<Mutex<Vec<Box<SamplingState>>>>,
}

impl Default for StatePool {
    fn default() -> Self {
        Self::new()
    }
}

impl StatePool {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn acquire(&self) -> PooledState {
        let mut inner = self.pool.lock();
        let state = if let Some(s) = inner.pop() {
            s
        } else {
            Box::new(SamplingState::new())
        };
        PooledState {
            state: Some(state),
            pool: self.clone(),
        }
    }
}

/// RAII Guard for a SamplingState. Returns the state to the pool on Drop.
pub struct PooledState {
    state: Option<Box<SamplingState>>,
    pool: StatePool,
}

impl std::ops::Deref for PooledState {
    type Target = SamplingState;
    fn deref(&self) -> &Self::Target {
        self.state.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state.as_mut().unwrap()
    }
}

impl Drop for PooledState {
    fn drop(&mut self) {
        if let Some(mut state) = self.state.take() {
            // Reset state before returning to pool
            state.reset();
            let mut inner = self.pool.pool.lock();
            inner.push(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_caps_edge_buffers_and_commit_scratch() {
        let pool = StatePool::new();
        {
            let mut state = pool.acquire();
            state.raw_edges = (0..4).map(|_| Vec::with_capacity(RESET_THRESHOLD * 2)).collect();
            let cs = &mut state.commit_scratch;
            cs.order.reserve(RESET_THRESHOLD * 4);
            cs.res.reserve(RESET_THRESHOLD * 4);
            cs.new_edges.reserve(RESET_THRESHOLD * 4);
            state.seed_maps = (0..64).map(|_| GxHashMap::with_capacity(RESET_THRESHOLD / 16)).collect();
        }
        let state = pool.acquire();
        let edge_capacity: usize = state.raw_edges.iter().map(Vec::capacity).sum();
        assert!(edge_capacity <= RESET_THRESHOLD);
        let cs = &state.commit_scratch;
        assert!(cs.order.capacity() <= RESET_THRESHOLD);
        assert!(cs.res.capacity() <= RESET_THRESHOLD);
        assert!(cs.new_edges.capacity() <= RESET_THRESHOLD);
        let map_capacity: usize = state.seed_maps.iter().map(GxHashMap::capacity).sum();
        assert!(map_capacity <= RESET_THRESHOLD);
        assert!(map_capacity > 0);
    }

    #[test]
    fn test_pool_recycling() {
        let pool = StatePool::new();

        // 1. Acquire and Modify
        {
            let mut state = pool.acquire();
            let ts = state.get_type_state_mut(0);

            // Push some data to grow capacity
            ts.nodes.reserve(1000);
            ts.nodes.push(NodeRecord {
                global_id: 1,
                seed_index: 0,
                local_id: 0,
                seed_type_id: 0,
            });
            ts.index_map.insert(
                LookupKey {
                    seed_type_id: 0,
                    seed_index: 0,
                    global_id: 1,
                },
                0,
            );

            assert!(ts.nodes.capacity() >= 1000);
            assert_eq!(ts.nodes.len(), 1);
            assert_eq!(ts.index_map.len(), 1);
        } // state dropped here

        // 2. Re-Acquire
        {
            let mut state = pool.acquire();
            let ts = state.get_type_state_mut(0);

            // Should be logically empty
            assert_eq!(ts.nodes.len(), 0);
            assert_eq!(ts.index_map.len(), 0);

            // Should retain capacity (Zero Allocation check)
            assert!(ts.nodes.capacity() >= 1000);

            // 0 entry should still exist in type_states (map not cleared, values reset)
            assert!(state.type_states.contains_key(&0));
        }
    }

    #[test]
    fn test_pool_concurrency() {
        let pool = StatePool::new();
        let mut guards = Vec::new();

        // Acquire 10 states
        for _ in 0..10 {
            guards.push(pool.acquire());
        }

        // Drop them all
        drop(guards);

        // Acquire 1 (should reuse one of the 10)
        let _s = pool.acquire();
        let inner = pool.pool.lock();
        assert_eq!(inner.len(), 9); // 1 taken, 9 remaining
    }
}
