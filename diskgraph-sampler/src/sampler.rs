use diskgraph_core::sampling::{BatchSamplingRequest, SamplingMode, FanoutSpec};
use crate::errors::{Result, SamplerError};

/// A sampling plan where relation names have been resolved to IDs.
pub struct CompiledSamplingPlan {
    pub layers: Vec<CompiledLayerSpec>,
}

pub struct CompiledLayerSpec {
    /// Fanouts keyed by Relation ID.
    pub fanouts: HashMap<usize, FanoutSpec>,
    pub default_fanout: Option<FanoutSpec>,
}
use diskgraph_index::reader::ShardReader;
use diskgraph_index::layout::RelationSegment;
use diskgraph_core::sortkey::SortKeyDecoder;
use crate::edge_unpacker::EdgeUnpacker;
use crate::state::{SamplingState, LookupKey, NodeRecord};
use crate::output::OutputHandler;
use std::collections::HashMap;
use std::sync::Arc;

/// Metadata and reader for a single graph shard.
pub struct ShardHandle {
    pub reader: Arc<ShardReader>,
    pub min_id: u64,
    pub max_id: u64,
}

use serde::Serialize;

#[derive(Serialize)]
pub struct SchemaMapping {
    pub node_types: Vec<String>,
    pub edge_relations: Vec<String>,
}

/// The core engine responsible for executing graph sampling requests.
pub struct Sampler {
    /// Shard readers mapped by Node Type ID.
    pub readers: Vec<Vec<ShardHandle>>,
    /// Map from Relation ID to DstTypeID.
    pub relation_info: Vec<usize>,
    /// Sort key decoders for each relation.
    pub decoders: HashMap<usize, SortKeyDecoder>,
    /// Map from node type name to dense ID.
    pub node_type_to_id: HashMap<String, usize>,
    /// Map from relation name to ID.
    pub rel_name_to_id: HashMap<String, usize>,
    /// Map from dense ID to node type name.
    pub id_to_node_type: Vec<String>,
    /// Map from dense ID to relation name (reconstructed from relation_info).
    pub id_to_rel_name: Vec<String>,
}

impl Sampler {
    pub fn new(
        readers_map: HashMap<String, Vec<ShardHandle>>,
        relation_info_map: HashMap<usize, (String, String)>,
        decoders: HashMap<usize, SortKeyDecoder>,
    ) -> Self {
        let mut node_type_to_id = HashMap::new();
        let mut id_to_node_type = Vec::new();
        let mut readers = Vec::new();

        // Sort keys to ensure deterministic ID assignment
        let mut sorted_types: Vec<_> = readers_map.keys().cloned().collect();
        sorted_types.sort();

        for (idx, name) in sorted_types.iter().enumerate() {
            node_type_to_id.insert(name.clone(), idx);
            id_to_node_type.push(name.clone());
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

        for (rel_id, (rel_name, dst_type)) in relation_info_map {
            if let Some(&dst_id) = node_type_to_id.get(&dst_type) {
                rel_name_to_id.insert(rel_name.clone(), rel_id);
                id_to_rel_name[rel_id] = rel_name.clone();
                relation_info[rel_id] = dst_id;
            }
        }

        Self { readers, relation_info, decoders, node_type_to_id, rel_name_to_id, id_to_node_type, id_to_rel_name }
    }

    pub fn get_schema_mapping(&self) -> SchemaMapping {
        SchemaMapping {
            node_types: self.id_to_node_type.clone(),
            edge_relations: self.id_to_rel_name.clone(),
        }
    }

    fn compile_plan(&self, plan: &diskgraph_core::sampling::SamplingPlan) -> Result<CompiledSamplingPlan> {
        let mut layers = Vec::with_capacity(plan.layers.len());

        for layer in &plan.layers {
            let mut fanouts = HashMap::new();
            for (name, spec) in &layer.fanouts {
                if let Some(&id) = self.rel_name_to_id.get(name) {
                    fanouts.insert(id, spec.clone());
                } else {
                    // Warn or error? Design says ignore unknown relations in plan?
                    // Or maybe the plan references a relation not in this subgraph?
                    // For safety, we can ignore, but logging would be good.
                }
            }
            layers.push(CompiledLayerSpec {
                fanouts,
                default_fanout: layer.default_fanout.clone(),
            });
        }

        Ok(CompiledSamplingPlan { layers })
    }

    /// Perform a sampling pass using the provided request and state.
    /// This implementation follows the "Gather-Fill-Finish" pattern.
    pub fn sample(
        &self,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        // 1. Initial Validation & Initialization
        let (seed_type, seed_ids) = request.seed_nodes.iter().next()
            .ok_or_else(|| SamplerError::InvalidRequest("request has no seed nodes".to_string()))?;

        let seed_type_id = *self.node_type_to_id.get(seed_type)
            .ok_or_else(|| SamplerError::InvalidRequest(format!("unknown seed type: {}", seed_type)))?;

        // Pre-allocate stats
        let num_types = self.readers.len();
        let num_rels = self.relation_info.len();
        let num_hops = request.plan.layers.len();
        state.stats.resize(num_types, num_rels, num_hops);

        if request.seed_nodes.len() > 1 {
            return Err(SamplerError::InvalidRequest("multiple seed node types not supported in initial scope".to_string()));
        }

        let seed_ts_vec = request.seed_timestamps.as_ref()
            .and_then(|m| m.get(seed_type));

        // 2. Initialize Seed Layer (Hop 0)
        // This ensures seed nodes get Local IDs 0..K and are preserved in order.
        let seed_state = state.type_states.entry(seed_type_id).or_insert_with(crate::state::TypeState::new);
        for (i, &global_id) in seed_ids.iter().enumerate() {
            let seed_index = i as u32;
            let key = LookupKey { seed_index, global_id };

            let _anchor_ts = seed_ts_vec.and_then(|v| v.get(i).copied());

            // Assign Local ID
            if !seed_state.index_map.contains_key(&key) {
                let local_id = seed_state.nodes.len() as u32;
                seed_state.index_map.insert(key, local_id);

                // Cache Node (Hop 0)
                seed_state.nodes.push(NodeRecord { global_id, seed_index, local_id });
                state.stats.record_node(0, seed_type_id);

                // Add to frontier for Layer 0
                seed_state.next_frontier.push(NodeRecord { global_id, seed_index, local_id });
            }
        }

        // 3. Process Layers (BFS)
        let compiled_plan = self.compile_plan(&request.plan)?;
        for layer_idx in 0..request.plan.layers.len() {
            self.process_layer(layer_idx, &compiled_plan.layers[layer_idx], request, state, output)?;
        }

        // 4. Finalize request by outputting node batch and stats
        output.finalize(state)?;

        Ok(())
    }

    fn process_layer(
        &self,
        layer_idx: usize,
        layer_spec: &CompiledLayerSpec,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        // 1. Drain frontiers
        let mut work_queue = Vec::new();
        for (node_type_id, type_state) in state.type_states.iter_mut() {
            if !type_state.next_frontier.is_empty() {
                let frontier = std::mem::take(&mut type_state.next_frontier);
                work_queue.push((*node_type_id, frontier));
            }
        }

        // 2. Process each Node Type
        for (src_type_id, frontier) in work_queue {
            if src_type_id >= self.readers.len() { continue; }
            let shards = &self.readers[src_type_id];
            if shards.is_empty() { continue; }

            // 3. Group nodes by Shard (Perform Binary Search ONCE per node)
            let mut shard_groups: Vec<Vec<&NodeRecord>> = vec![Vec::new(); shards.len()];
            for item in &frontier {
                let shard_idx = shards.binary_search_by(|s| {
                    if item.global_id < s.min_id {
                        std::cmp::Ordering::Greater
                    } else if item.global_id > s.max_id {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                });
                if let Ok(idx) = shard_idx {
                    shard_groups[idx].push(item);
                }
            }

            // 4. Iterate Shards
            for (shard_idx, mut nodes) in shard_groups.into_iter().enumerate() {
                if nodes.is_empty() { continue; }
                let shard = &shards[shard_idx];

                // HUB DEDUPING: Sort by local_idx to process same physical nodes together
                nodes.sort_by_key(|n| n.global_id);

                let mut current_segments: Vec<RelationSegment<'_>> = Vec::new();
                let mut last_id = None;

                // 5. Iterate Nodes in Shard
                for node in nodes {
                    // Only open view if node id changed
                    if last_id != Some(node.global_id) {
                        let local_idx = node.global_id - shard.min_id;
                        let edges_view = shard.reader.edges_for_node(local_idx)
                            .map_err(|e| SamplerError::DataCorruption(e.to_string()))?;

                        // Parse segments once per unique node
                        current_segments = match edges_view.relation_segments() {
                            Ok(s) => s,
                            Err(_) => Vec::new(),
                        };
                        last_id = Some(node.global_id);
                    }

                    // 6. Iterate Relations present in this node's edge list
                    for segment in &current_segments {
                        let rel_id = segment.layout.relation_id as usize;

                        // Check if this relation is in the Sampling Plan, or use default
                        let fanout_spec = layer_spec.fanouts.get(&rel_id)
                            .or(layer_spec.default_fanout.as_ref());

                        if let Some(fanout_spec) = fanout_spec {
                            // Resolve metadata (only needed for dst_type_id now)
                            let dst_type_id = *self.relation_info.get(rel_id).unwrap_or(&usize::MAX);
                            if dst_type_id != usize::MAX {
                                // 7. Sample (from memory-mapped segment)
                                self.sample_from_view(
                                    layer_idx,
                                    &segment,
                                    node,
                                    dst_type_id,
                                    rel_id,
                                    fanout_spec.fanout,
                                    &fanout_spec.mode,
                                    request,
                                    state,
                                    output,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn sample_from_view(
        &self,
        layer_idx: usize,
        segment: &RelationSegment<'_>,
        src_item: &NodeRecord,
        dst_type_id: usize,
        rel_id: usize,
        fanout: u32,
        mode: &SamplingMode,
        _request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let edge_count = segment.count as usize;
        if edge_count == 0 {
            return Ok(());
        }

        let mut unpacker = EdgeUnpacker::new(segment.data, &segment.layout);
        let decoder = self.decoders.get(&rel_id);

        match mode {
            SamplingMode::TemporalMostRecent => {
                let mut n = 0;
                let batch_size = (fanout as usize).min(128);

                loop {
                    let take = (fanout as usize - n).min(batch_size);
                    state.dst_scratch.resize(take, 0);
                    state.key_scratch.resize(take, 0);

                    let unpacked = unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                    if unpacked == 0 { break; }

                    for i in 0..unpacked {
                        let dst_id = state.dst_scratch[i];
                        let packed_key = state.key_scratch[i];

                        // Temporal filtering
                        let seed_type = _request.seed_nodes.keys().next().unwrap();
                        let anchor = _request.seed_timestamps.as_ref()
                            .and_then(|m| m.get(seed_type))
                            .and_then(|v| v.get(src_item.seed_index as usize).copied());

                        if let (Some(decoder), Some(anchor)) = (decoder, anchor) {
                            let (edge_ts, _): (Option<i64>, Option<f32>) = decoder.decode(packed_key);
                            if let Some(ts) = edge_ts {
                                if ts >= anchor {
                                    // Edge is in the future, skip it
                                    continue;
                                }
                            }
                        }

                        self.record_edge(layer_idx, src_item, dst_id, dst_type_id, rel_id as u16, state, output)?;
                        n += 1;
                        if n >= fanout as usize { break; }
                    }
                    if unpacked < take { break; }
                }
            }
            SamplingMode::Uniform => {
                // Optimized Uniform Sampling for known N (edge_count)
                // O(fanout) instead of O(edge_count)
                let k = fanout as usize;

                if k >= edge_count {
                    // Take all
                    loop {
                        state.dst_scratch.resize(128, 0);
                        state.key_scratch.resize(128, 0);
                        let unpacked = unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                        if unpacked == 0 { break; }
                        for i in 0..unpacked {
                            self.record_edge(layer_idx, src_item, state.dst_scratch[i], dst_type_id, rel_id as u16, state, output)?;
                        }
                    }
                } else {
                    // Select k unique indices
                    // Uses Floyd's algorithm or simple rejection depending on sparsity
                    // For simplicity and speed with small k, simple rejection or a pool shuffle is fine.
                    // But sorting a random sample is easiest.
                    let mut rng = rand::thread_rng();
                    use rand::seq::index::sample;

                    // rand::seq::index::sample is efficient (floyd's for small k/n, etc)
                    let indices = sample(&mut rng, edge_count, k);
                    let mut sorted_indices = indices.into_vec();
                    sorted_indices.sort_unstable();

                    let mut current_pos = 0;
                    for target_idx in sorted_indices {
                        let to_skip = target_idx - current_pos;
                        if to_skip > 0 {
                            unpacker.advance(to_skip);
                        }

                        // Unpack 1
                        state.dst_scratch.resize(1, 0);
                        state.key_scratch.resize(1, 0);
                        let n = unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                        if n > 0 {
                            self.record_edge(layer_idx, src_item, state.dst_scratch[0], dst_type_id, rel_id as u16, state, output)?;
                        }

                        current_pos = target_idx + 1;
                    }
                }
            }
            _ => return Err(SamplerError::InvalidRequest(format!("Sampling mode {:?} not supported in initial scope", mode))),
        }

        Ok(())
    }

    fn record_edge(
        &self,
        layer_idx: usize,
        src: &NodeRecord,
        dst_global_id: u64,
        dst_type_id: usize,
        rel_id: u16,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let dst_type_state = state.type_states.entry(dst_type_id).or_insert_with(crate::state::TypeState::new);
        let key = LookupKey { seed_index: src.seed_index, global_id: dst_global_id };

        let dst_local_id = if let Some(&id) = dst_type_state.index_map.get(&key) {
            id
        } else {
            let id = dst_type_state.nodes.len() as u32;
            dst_type_state.index_map.insert(key, id);

            // Cache Node (Hop K)
            dst_type_state.nodes.push(NodeRecord {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            state.stats.record_node(layer_idx + 1, dst_type_id);

            // Add to frontier for the next hop
            dst_type_state.next_frontier.push(NodeRecord {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            id
        };

        // Emit Edge Event
        output.handle_edge(layer_idx, rel_id, src.local_id, dst_local_id)?;

        state.stats.record_edge(layer_idx, rel_id as usize);

        Ok(())
    }
}
