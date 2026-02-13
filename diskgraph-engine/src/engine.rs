use crate::edge_unpacker::EdgeUnpacker;
use crate::errors::{EngineError, Result};
use crate::output::OutputHandler;
use crate::state::{LookupKey, NodeRecord, SamplingState};
use diskgraph_core::sampling::*;
use diskgraph_core::sortkey::SortKeyDecoder;
use diskgraph_index::layout::{EdgeListView, RelationSegment};
use diskgraph_index::reader::ShardReader;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Metadata and reader for a single graph shard.
pub struct ShardHandle {
    pub reader: Arc<ShardReader>,
    pub min_id: u64,
    pub max_id: u64,
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
}

struct CompiledSamplingPlan {
    layers: Vec<CompiledLayerSpec>,
}

struct CompiledLayerSpec {
    fanouts: HashMap<usize, FanoutSpec>,
    default_fanout: Option<FanoutSpec>,
}

impl DiskGraphEngine {
    pub fn load(path: &Path) -> Result<Self> {
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

        // Compute layouts from metadata
        let all_layouts = diskgraph_index::builder::compute_relation_layouts(
            &metadata.edge_relations,
            &metadata.node_types,
        ).map_err(|e| EngineError::Internal(format!("failed to compute layouts: {}", e)))?;

        let mut readers = HashMap::new();
        let mut node_decoders_map = HashMap::new();

        for node_type in &metadata.node_types {
            let mut type_layouts = Vec::new();
            for layout in &all_layouts {
                let rel_meta = &metadata.edge_relations[layout.relation_id as usize];
                if rel_meta.src_type == node_type.name {
                    type_layouts.push(layout.clone());
                }
            }
            let type_layouts = Arc::new(type_layouts);

            let mut shards = Vec::new();
            for (i, shard_meta) in node_type.shards.iter().enumerate() {
                let shard_dir = path
                    .join("nodes")
                    .join(&node_type.name)
                    .join(format!("shard-{:05}", i));

                if shard_dir.exists() {
                    let reader =
                        ShardReader::open(&shard_dir, type_layouts.clone(), node_type.has_node_data)
                            .map_err(|e| {
                                EngineError::Internal(format!("failed to open shard: {}", e))
                            })?;

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

        println!("Graph loaded. Ready to serve.");
        Ok(Self::new(
            readers,
            relation_info,
            decoders,
            node_decoders_map,
        ))
    }

    pub fn new(
        readers_map: HashMap<String, Vec<ShardHandle>>,
        relation_info_map: HashMap<usize, (String, String, String)>,
        decoders_map: HashMap<usize, SortKeyDecoder>,
        node_decoders_map: HashMap<String, RowPackDecoder>,
    ) -> Self {
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
            if let (Some(&src_id), Some(&dst_id)) = (
                node_type_to_id.get(src_type),
                node_type_to_id.get(dst_type),
            ) {
                rel_name_to_id.insert(rel_name.clone(), *rel_id);
                id_to_rel_name[*rel_id] = rel_name.clone();
                relation_info[*rel_id] = dst_id;
                relation_src_type[*rel_id] = src_id;

                if let Some(decoder) = decoders_map.get(rel_id) {
                    decoders[*rel_id] = Some(decoder.clone());
                }
            }
        }

        // Build forward↔reverse relation pairs from naming convention.
        let mut reverse_rel_map: Vec<Option<usize>> = vec![None; id_to_rel_name.len()];
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

        // Generate schema based on selection
        // Create a dummy empty batch to get schema
        if select_all {
            col_indices = (0..context.columns.len()).collect();
        } else {
            for name in feature_names {
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
             return Err(EngineError::Internal(format!("no schema for type {}", node_type_id)));
        }

        // Use Bulk Fetch helper
        self.accumulate_features_from_ids_chunked(node_type_id, node_ids, &col_indices, decoder, output)?;

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

    pub fn sample(
        &self,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        // 1. Initial Validation & Initialization
        let (seed_type, seed_ids) =
            request.seed_nodes.iter().next().ok_or_else(|| {
                EngineError::InvalidRequest("request has no seed nodes".to_string())
            })?;

        let seed_type_id = *self.node_type_to_id.get(seed_type).ok_or_else(|| {
            EngineError::InvalidRequest(format!("unknown seed type: {}", seed_type))
        })?;

        // Pre-allocate stats
        let num_types = self.readers.len();
        let num_rels = self.relation_info.len();
        let num_hops = request.plan.layers.len();
        state.stats.resize(num_types, num_rels, num_hops);

        if request.seed_nodes.len() > 1 {
            return Err(EngineError::InvalidRequest(
                "multiple seed node types not supported in initial scope".to_string(),
            ));
        }

        let seed_ts_vec = request
            .seed_timestamps
            .as_ref()
            .and_then(|m| m.get(seed_type));

        let fused = request.flags.fused_feature_fetching;

        // 2. Initialize Seed Layer (Hop 0)
        // This ensures seed nodes get Local IDs 0..K and are preserved in order.
        let seed_state = state.type_states.entry(seed_type_id).or_default();
        for (i, &global_id) in seed_ids.iter().enumerate() {
            let seed_index = i as u32;
            let key = LookupKey {
                seed_index,
                global_id,
            };

            let _anchor_ts = seed_ts_vec.and_then(|v| v.get(i).copied());

            // Assign Local ID for seeds
            if !seed_state.index_map.contains_key(&key) {
                let local_id = seed_state.nodes.len() as u32;
                seed_state.index_map.insert(key, local_id);

                // Cache Node (Hop 0)
                seed_state.nodes.push(NodeRecord {
                    global_id,
                    seed_index,
                    local_id,
                });
                state.stats.record_node(0, seed_type_id);

                // Add to frontier for Layer 0
                seed_state.next_frontier.push(NodeRecord {
                    global_id,
                    seed_index,
                    local_id,
                });
            }
        }

        // 3. Process Layers (BFS)
        let compiled_plan = self.compile_plan(&request.plan)?;
        for layer_idx in 0..request.plan.layers.len() {
            self.process_layer(
                layer_idx,
                &compiled_plan.layers[layer_idx],
                request,
                state,
                output,
            )?;
        }

        // 4. Finalize fused fetching by fetching leaf node features
        // and flush remaining features.
        if fused {
            let type_ids: Vec<usize> = state.type_states.keys().cloned().collect();
            for type_id in type_ids {
                let scratch = &mut state.decode_scratch;
                if let Some(ts) = state.type_states.get_mut(&type_id) {
                    if !ts.next_frontier.is_empty() {
                         let node_ids: Vec<u64> = ts.next_frontier.iter().map(|n| n.global_id).collect();
                         self.accumulate_features_from_ids_builder(type_id, &node_ids, ts, scratch, output)?;
                         ts.next_frontier.clear();
                    }
                    self.flush_fused_features(type_id, ts, output)?;
                }
            }
        }

        // 5. Finalize node batch and stats
        output.finalize(state)?;

        Ok(())
    }

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

        self.accumulate_features_from_view_internal(type_id, type_state, scratch, decoder, edges_view, output)
    }

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
            if type_state.builders.is_none() {
                let mut builders = Vec::with_capacity(context.columns.len());
                for col in &context.columns {
                    let (_, builder) =
                        diskgraph_rowpack::make_arrow_builder(&col.name, col.logical_type, 1024)
                            .map_err(|e| EngineError::Internal(e.to_string()))?;
                    builders.push(builder);
                }
                type_state.builders = Some(builders);
            }

            let mask = vec![true; context.columns.len()];

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

    fn accumulate_features_from_ids_builder(
        &self,
        type_id: usize,
        node_ids: &[u64],
        type_state: &mut crate::state::TypeState,
        scratch: &mut RowPackDecodeScratch,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let shards = self.readers.get(type_id).ok_or_else(|| {
            EngineError::InvalidRequest("invalid node type id".to_string())
        })?;

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

            self.accumulate_features_from_view_internal(type_id, type_state, scratch, decoder, &edges_view, output)?;
        }
        Ok(())
    }

    fn accumulate_features_from_ids_chunked(
        &self,
        type_id: usize,
        node_ids: &[u64],
        col_indices: &[usize],
        decoder: &RowPackDecoder,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        let shards = self.readers.get(type_id).ok_or_else(|| {
            EngineError::InvalidRequest("invalid node type id".to_string())
        })?;

        let mut last_shard_idx = 0;
        let mut batch_size = 128;

        let mut ptr = 0;
        while ptr < node_ids.len() {
            let end = (ptr + batch_size).min(node_ids.len());
            let chunk_ids = &node_ids[ptr..end];

            let mut raw_rows = Vec::with_capacity(chunk_ids.len());
            let mut chunk_bytes = 0;

            for &global_id in chunk_ids {
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

                chunk_bytes += data.len();
                raw_rows.push(data);
            }

            let batch = decoder
                .decode_batch_to_arrow::<&[u8]>(&raw_rows, col_indices)
                .map_err(|e| EngineError::Internal(format!("decoding failed: {}", e)))?;

            output.handle_feature_batch(batch)?;

            if chunk_bytes > 0 {
                let avg_bytes = chunk_bytes / chunk_ids.len();
                if avg_bytes > 0 {
                    batch_size = (self.target_batch_bytes / avg_bytes).clamp(1, self.target_batch_rows);
                }
            }

            ptr = end;
        }
        Ok(())
    }

    fn flush_fused_features(
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

        let schema = self.node_schemas[type_id]
            .as_ref()
            .ok_or_else(|| EngineError::Internal(format!("no precomputed schema for type {}", type_id)))?;

        let batch = arrow_array::RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        output.handle_fused_features(type_id, batch)?;
        type_state.builder_rows = 0;
        type_state.builder_bytes = 0;

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
                if src_type_id >= self.readers.len() { continue; }
                let shards = &self.readers[src_type_id];
                if shards.is_empty() { continue; }

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

                    let segments = match edges_view.relation_segments() {
                        Ok(s) => s,
                        Err(_) => Vec::new(),
                    };

                    self.process_node_relations(layer_idx, &segments, node, layer_spec, request, state, output)?;
                }
            }
            return Ok(());
        }

        // Standalone sampling path
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

            for (shard_idx, mut shard_nodes) in shard_groups.into_iter().enumerate() {
                if shard_nodes.is_empty() {
                    continue;
                }
                let shard = &shards[shard_idx];

                // HUB DEDUPING: Sort by local_idx to process same physical nodes together
                shard_nodes.sort_by_key(|n| n.global_id);

                let mut current_segments = Vec::new();
                let mut last_id = None;

                for node in shard_nodes {
                    if last_id != Some(node.global_id) {
                        let local_id = node.global_id - shard.min_id;
                        let edges_view = shard
                            .reader
                            .edges_for_node(local_id)
                            .map_err(|e| EngineError::DataCorruption(e.to_string()))?;

                        last_id = Some(node.global_id);
                        current_segments = match edges_view.relation_segments() {
                            Ok(s) => s,
                            Err(_) => Vec::new(),
                        };
                    }

                    self.process_node_relations(layer_idx, &current_segments, node, layer_spec, request, state, output)?;
                }
            }
        }

        Ok(())
    }

    fn process_node_relations(
        &self,
        layer_idx: usize,
        segments: &[RelationSegment],
        node: &NodeRecord,
        layer_spec: &CompiledLayerSpec,
        request: &BatchSamplingRequest,
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
    ) -> Result<()> {
        for segment in segments {
            let rel_id = segment.layout.relation_id as usize;
            let fanout_spec = layer_spec
                .fanouts
                .get(&rel_id)
                .or(layer_spec.default_fanout.as_ref());

            if let Some(fanout_spec) = fanout_spec {
                let dst_type_id = *self.relation_info.get(rel_id).unwrap_or(&usize::MAX);
                if dst_type_id != usize::MAX {
                    self.sample_relation(
                        layer_idx,
                        segment,
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
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_relation(
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
        let decoder = self
            .decoders
            .get(rel_id)
            .and_then(|d: &Option<SortKeyDecoder>| d.as_ref());

        match mode {
            SamplingMode::TemporalMostRecent => {
                let mut n = 0;
                let batch_size = (fanout as usize).min(128);

                loop {
                    let take = (fanout as usize - n).min(batch_size);
                    state.dst_scratch.resize(take, 0);
                    state.key_scratch.resize(take, 0);

                    let unpacked =
                        unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                    if unpacked == 0 {
                        break;
                    }

                    for i in 0..unpacked {
                        let dst_id = state.dst_scratch[i];
                        let packed_key = state.key_scratch[i];

                        let seed_type = _request.seed_nodes.keys().next().unwrap();
                        let anchor = _request
                            .seed_timestamps
                            .as_ref()
                            .and_then(|m| m.get(seed_type))
                            .and_then(|v| v.get(src_item.seed_index as usize).copied());

                        if let (Some(decoder), Some(anchor)) = (decoder, anchor) {
                            let (edge_ts, _): (Option<i64>, Option<f32>) =
                                decoder.decode(packed_key);
                            if let Some(ts) = edge_ts {
                                if ts >= anchor {
                                    continue;
                                }
                            }
                        }

                        self.record_edge(
                            layer_idx,
                            src_item,
                            dst_id,
                            dst_type_id,
                            rel_id as u16,
                            state,
                            output,
                        )?;
                        n += 1;
                        if n >= fanout as usize {
                            break;
                        }
                    }
                    if unpacked < take {
                        break;
                    }
                }
            }
            SamplingMode::Uniform => {
                let k = fanout as usize;

                if k >= edge_count {
                    loop {
                        state.dst_scratch.resize(128, 0);
                        state.key_scratch.resize(128, 0);
                        let unpacked =
                            unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                        if unpacked == 0 {
                            break;
                        }
                        for i in 0..unpacked {
                            self.record_edge(
                                layer_idx,
                                src_item,
                                state.dst_scratch[i],
                                dst_type_id,
                                rel_id as u16,
                                state,
                                output,
                            )?;
                        }
                    }
                } else {
                    let mut rng = rand::thread_rng();
                    use rand::seq::index::sample;

                    let indices = sample(&mut rng, edge_count, k);
                    let mut sorted_indices = indices.into_vec();
                    sorted_indices.sort_unstable();

                    let mut current_pos = 0;
                    for target_idx in sorted_indices {
                        let to_skip = target_idx - current_pos;
                        if to_skip > 0 {
                            unpacker.advance(to_skip);
                        }

                        state.dst_scratch.resize(1, 0);
                        state.key_scratch.resize(1, 0);
                        let n =
                            unpacker.unpack_batch(&mut state.dst_scratch, &mut state.key_scratch);
                        if n > 0 {
                            self.record_edge(
                                layer_idx,
                                src_item,
                                state.dst_scratch[0],
                                dst_type_id,
                                rel_id as u16,
                                state,
                                output,
                            )?;
                        }

                        current_pos = target_idx + 1;
                    }
                }
            }
            _ => {
                return Err(EngineError::InvalidRequest(format!(
                    "Sampling mode {:?} not supported in initial scope",
                    mode
                )))
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
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
        let dst_type_state = state.type_states.entry(dst_type_id).or_default();
        let key = LookupKey {
            seed_index: src.seed_index,
            global_id: dst_global_id,
        };

        let dst_local_id = if let Some(&id) = dst_type_state.index_map.get(&key) {
            id
        } else {
            let id = dst_type_state.nodes.len() as u32;
            dst_type_state.index_map.insert(key, id);

            dst_type_state.nodes.push(NodeRecord {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            state.stats.record_node(layer_idx + 1, dst_type_id);

            dst_type_state.next_frontier.push(NodeRecord {
                global_id: dst_global_id,
                seed_index: src.seed_index,
                local_id: id,
            });
            id
        };

        output.handle_edge(layer_idx, rel_id, src.local_id, dst_local_id)?;
        state.stats.record_edge(layer_idx, rel_id as usize);

        Ok(())
    }
}
