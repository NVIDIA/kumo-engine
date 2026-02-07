use anyhow::{anyhow, Result};
use diskgraph_core::sampling::{BatchSamplingRequest, SamplingMode};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::layout::EdgeListView;
use crate::edge_unpacker::EdgeUnpacker;
use crate::state::{SamplingState, LookupKey, FrontierItem, NodeRecord};
use std::collections::HashMap;
use std::sync::Arc;
// use rand::seq::SliceRandom;
// use rand::Rng;

/// The core engine responsible for executing graph sampling requests.
pub struct Sampler {
    /// Shard readers mapped by node type.
    pub readers: HashMap<String, Vec<Arc<ShardReader>>>,
}

impl Sampler {
    pub fn new(readers: HashMap<String, Vec<Arc<ShardReader>>>) -> Self {
        Self { readers }
    }

    /// Perform a sampling pass using the provided request and state.
    /// This implementation follows the "Gather-Fill-Finish" pattern.
    pub fn sample(
        &self,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
    ) -> Result<()> {
        // 1. Initial Validation & Initialization
        let (seed_type, seed_ids) = request.seed_nodes.iter().next()
            .ok_or_else(|| anyhow!("request has no seed nodes"))?;
        if request.seed_nodes.len() > 1 {
            return Err(anyhow!("multiple seed node types not supported in initial scope"));
        }
        
        let _seed_ts = request.seed_timestamps.as_ref()
            .and_then(|m| m.get(seed_type));

        // 2. Initialize Seed Layer (Hop 0)
        // This ensures seed nodes get Local IDs 0..K and are preserved in order.
        let seed_state = state.get_type_state_mut(seed_type);
        for (i, &global_id) in seed_ids.iter().enumerate() {
            let seed_index = i as u32;
            let key = LookupKey { seed_index, global_id };
            
            // Assign Local ID
            let local_id = seed_state.nodes.len() as u32;
            seed_state.index_map.insert(key, local_id);
            seed_state.nodes.push(NodeRecord { global_id, seed_index, local_id });
            
            // Add to frontier for Layer 0
            seed_state.next_frontier.push(FrontierItem { global_id, seed_index, local_id });
        }

        // 3. Process Layers (BFS)
        for layer_idx in 0..request.plan.layers.len() {
            self.process_layer(layer_idx, request, state)?;
        }

        Ok(())
    }

    fn process_layer(
        &self,
        layer_idx: usize,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
    ) -> Result<()> {
        let layer_spec = &request.plan.layers[layer_idx];
        
        // We need to move the items from 'next_frontier' to 'active_frontier'
        // Since we have multiple node types, we must handle each.
        // For simplicity in this scope, we'll collect all current frontiers.
        let mut work_queue = Vec::new();
        for (node_type, type_state) in state.type_states.iter_mut() {
            if !type_state.next_frontier.is_empty() {
                let frontier = std::mem::take(&mut type_state.next_frontier);
                work_queue.push((node_type.clone(), frontier));
            }
        }

        for (src_type, frontier) in work_queue {
            // For each edge type starting from this src_type, perform sampling
            for (edge_type_name, fanout_spec) in &layer_spec.fanouts {
                // Parse edge type name: "SrcType__RelName__DstType"
                let parts: Vec<&str> = edge_type_name.split("__").collect();
                if parts.len() != 3 || parts[0] != src_type {
                    continue;
                }
                let _rel_name = parts[1];
                let dst_type = parts[2];

                self.sample_edges(
                    &src_type,
                    dst_type,
                    edge_type_name,
                    &frontier,
                    fanout_spec.fanout,
                    &fanout_spec.mode,
                    request,
                    state,
                )?;
            }
        }

        Ok(())
    }

    fn sample_edges(
        &self,
        src_type: &str,
        dst_type: &str,
        relation_name: &str,
        frontier: &[FrontierItem],
        fanout: u32,
        mode: &SamplingMode,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
    ) -> Result<()> {
        let shard_readers = self.readers.get(src_type)
            .ok_or_else(|| anyhow!("missing readers for node type {}", src_type))?;
        let num_shards = shard_readers.len();

        // 1. Group frontier items by shard
        // (This is the "Deduplication and Locality" win discussed earlier)
        let mut shard_groups: Vec<Vec<&FrontierItem>> = vec![Vec::new(); num_shards];
        for item in frontier {
            let shard_idx = (item.global_id % num_shards as u64) as usize;
            shard_groups[shard_idx].push(item);
        }

        for (shard_idx, items) in shard_groups.into_iter().enumerate() {
            if items.is_empty() { continue; }
            let reader = &shard_readers[shard_idx];

            for item in items {
                // TODO: Optimization - Batch index lookup if possible, 
                // but currently we use edges_for_node (simple path).
                let edges_view = reader.edges_for_node(item.global_id)?;
                
                self.sample_from_view(
                    edges_view,
                    item,
                    dst_type,
                    relation_name,
                    fanout,
                    mode,
                    request,
                    state,
                )?;
            }
        }

        Ok(())
    }

    fn sample_from_view(
        &self,
        view: EdgeListView<'_>,
        src_item: &FrontierItem,
        dst_type: &str,
        relation_name: &str,
        fanout: u32,
        mode: &SamplingMode,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
    ) -> Result<()> {
        // Resolve relation index within the shard
        let rel_idx = view.layouts.iter().position(|l| {
             // In diskgraph, relation names in layouts are usually indices or IDs.
             // We need a mapping from 'relation_name' string to ID.
             // For now, assume relation_name matches or use a lookup.
             // TODO: Fix relation resolution logic.
             true 
        }).unwrap_or(0);

        // Parse raw bytes into segments
        let segments = view.relation_segments().map_err(|e| anyhow!(e))?;
        
        let segment = segments.get(rel_idx)
            .ok_or_else(|| anyhow!("relation index {} out of bounds", rel_idx))?;
            
        let edge_count = segment.count as usize;
        if edge_count == 0 { return Ok(()); }
        
        let mut unpacker = EdgeUnpacker::new(segment.data, &segment.layout);

        // let mut rng = rand::thread_rng();

        match mode {
            SamplingMode::TemporalMostRecent => {
                // Temporal Most Recent: Read first 'fanout' valid edges.
                // Edges are physically sorted by time on disk.
                let mut n = 0;
                let batch_size = (fanout as usize).min(128); // Process in chunks
                
                while n < fanout as usize {
                    let take = (fanout as usize - n).min(batch_size);
                    state.dst_scratch.resize(take, 0);
                    state.key_scratch.resize(take, 0);
                    
                    let unpacked = unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                    if unpacked == 0 { break; }

                    for i in 0..unpacked {
                        let dst_id = state.dst_scratch[i];
                        // let packed_key = state.key_scratch[i];
                        
                        // TODO: Temporal filtering (if seed_timestamps present)
                        // if packed_key > packed_anchor { ... }

                        self.record_edge(src_item, dst_id, dst_type, state);
                        n += 1;
                        if n >= fanout as usize { break; }
                    }
                    if unpacked < take { break; }
                }
            }
            SamplingMode::Uniform => {
                // Uniform: Reservoir sampling or simple random selection.
                // For simplicity: Reservoir sampling over the whole list.
                // TODO: Optimize with skip-distance if edge_count is large.
                let mut n = 0;
                let mut selected = Vec::with_capacity(fanout as usize);
                
                // Scalar loop for now, optimize with unpack_batch later
                // Actually, let's just take the first N for the skeleton.
                let take = (fanout as usize).min(edge_count);
                state.dst_scratch.resize(take, 0);
                state.key_scratch.resize(take, 0);
                let unpacked = unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                
                for i in 0..unpacked {
                    selected.push(state.dst_scratch[i]);
                }
                
                for dst_id in selected {
                    self.record_edge(src_item, dst_id, dst_type, state);
                }
            }
            _ => return Err(anyhow!("Sampling mode {:?} not supported in initial scope", mode)),
        }

        Ok(())
    }

    fn record_edge(
        &self,
        src: &FrontierItem,
        dst_global_id: u64,
        dst_type: &str,
        state: &mut SamplingState,
    ) {
        let dst_type_state = state.get_type_state_mut(dst_type);
        let key = LookupKey { seed_index: src.seed_index, global_id: dst_global_id };

        let _dst_local_id = if let Some(&id) = dst_type_state.index_map.get(&key) {
            id
        } else {
            let id = dst_type_state.nodes.len() as u32;
            dst_type_state.index_map.insert(key, id);
            dst_type_state.nodes.push(NodeRecord {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            // Add to frontier for the next hop
            dst_type_state.next_frontier.push(FrontierItem {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            id
        };

        // Emit edge record (src_local_id, dst_local_id)
        // TODO: Buffered emission to Arrow Flight stream
    }
}
