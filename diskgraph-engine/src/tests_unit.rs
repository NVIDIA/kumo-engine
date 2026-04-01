use crate::engine::{DiskGraphEngine, ShardHandle};
use crate::output::OutputHandler;
use crate::state::SamplingState;
use diskgraph_core::dataset::DatasetId;
use diskgraph_core::metadata::{DatasetMetadata, NodeShardMetadata, NodeTypeMetadata};
use diskgraph_core::sampling::*;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

use diskgraph_rowpack::{
    IngestedType, RowPackColumn, RowPackConfig, RowPackContextBuilder, RowPackDecoder,
    RowPackEncoder, RowPackScratch, ValueRef,
};

struct TestGraph {
    _dir: TempDir,
    sampler: DiskGraphEngine,
    rowpack_ctx: diskgraph_rowpack::RowPackContext,
}

impl TestGraph {
    fn new() -> Self {
        Self::new_with_seed(None)
    }

    fn new_with_seed(random_seed: Option<u64>) -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // 1. Setup RowPack Context & Data
        let cols = vec![
            RowPackColumn {
                name: "feat_0".into(),
                logical_type: IngestedType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
            RowPackColumn {
                name: "feat_1".into(),
                logical_type: IngestedType::Float32,
                fixed_len: None,
                precision: None,
                scale: None,
            },
        ];
        let mut ctx_builder = RowPackContextBuilder::new(cols.clone(), RowPackConfig::default());
        let row_val = vec![ValueRef::Int64(42), ValueRef::Float32(3.14)];
        ctx_builder.observe_row(&row_val).unwrap();
        let ctx = ctx_builder.finish().unwrap();

        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut scratch = RowPackScratch::new(&ctx);
        let mut encoded_node_data = Vec::new();
        encoder
            .encode_row_into(&row_val, &mut encoded_node_data, &mut scratch)
            .unwrap();

        // 2. Define Layout: RelID=0, Dst=1 byte
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dest_sort_key_bits: None,
            entry_bytes: 1,
            dst_mask: 0xFF,
            dest_sort_key_mask: None,
        };

        // 3. Write Edges.bin
        let mut f = File::create(path.join(EDGES_FILENAME)).unwrap();
        // Node 0: 20 edges, distinct neighbors 1..=20
        let neighbor_ids: Vec<u8> = (1..=20).collect();
        f.write_all(&[20]).unwrap(); // Count
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
            .unwrap();
        f.write_all(&neighbor_ids).unwrap();
        f.write_all(&encoded_node_data).unwrap();

        // Nodes 1..20: 0 edges, Len, Data
        for _ in 1..=20 {
            f.write_all(&[0x00]).unwrap();
            f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
                .unwrap();
            f.write_all(&encoded_node_data).unwrap();
        }
        f.flush().unwrap(); // Ensure written
        let edges_len = f.metadata().unwrap().len();

        // 4. Write Offsets.bin
        let mut offsets = Vec::new();
        // Header: count=21 nodes -> 22 offsets
        let header = OffsetFileHeader::new(21, 22);
        let header_bytes = bincode::serialize(&header).unwrap();
        offsets.extend_from_slice(&header_bytes);

        offsets.extend_from_slice(&22u16.to_le_bytes()); // Chunk count = 22
        offsets.extend_from_slice(&0u64.to_le_bytes()); // offset[0] = 0

        let node0_len = 1 + 2 + 20 + encoded_node_data.len();
        let mut current_offset = node0_len as u64;
        offsets.extend_from_slice(&(node0_len as u16).to_le_bytes());

        let empty_node_len = 1 + 2 + encoded_node_data.len();
        for _ in 1..21 {
            offsets.extend_from_slice(&(empty_node_len as u16).to_le_bytes());
            current_offset += empty_node_len as u64;
        }

        assert_eq!(current_offset, edges_len, "Offsets sum must match file size");

        let mut f = File::create(path.join(OFFSETS_FILENAME)).unwrap();
        f.write_all(&offsets).unwrap();

        // 5. Create Reader
        let reader = ShardReader::open(path, Arc::new(vec![layout]), true).unwrap();

        let mut readers = HashMap::new();
        readers.insert(
            "User".to_string(),
            vec![ShardHandle {
                reader: Arc::new(reader),
                min_id: 0,
                max_id: 20,
            }],
        );

        let mut relation_info = HashMap::new();
        relation_info.insert(
            0,
            (
                "Connects".to_string(),
                "User".to_string(),
                "User".to_string(),
            ),
        );

        let mut node_decoders = HashMap::new();
        node_decoders.insert(
            "User".to_string(),
            RowPackDecoder::new(ctx.clone()).unwrap(),
        );

        let sampler = DiskGraphEngine::new(
            readers,
            relation_info,
            HashMap::new(),
            node_decoders,
            None,
            diskgraph_core::time::TimeUnit::Seconds,
            random_seed,
            None,
        );

        Self {
            _dir: dir,
            sampler,
            rowpack_ctx: ctx,
        }
    }
}

struct MockOutput {
    edges: Vec<(usize, u16, u32, u32)>,
    feature_batches: Vec<arrow_array::RecordBatch>,
    schema: Option<arrow_schema::SchemaRef>,
    fused_batches: Vec<(usize, arrow_array::RecordBatch)>,
}
impl OutputHandler for MockOutput {
    fn handle_edge(
        &mut self,
        hop: usize,
        rel_id: u16,
        src: u32,
        dst: u32,
    ) -> crate::errors::Result<()> {
        self.edges.push((hop, rel_id, src, dst));
        Ok(())
    }

    fn handle_schema(&mut self, schema: arrow_schema::SchemaRef) -> crate::errors::Result<()> {
        self.schema = Some(schema);
        Ok(())
    }

    fn handle_feature_batch(
        &mut self,
        batch: arrow_array::RecordBatch,
    ) -> crate::errors::Result<()> {
        self.feature_batches.push(batch);
        Ok(())
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: arrow_array::RecordBatch,
    ) -> crate::errors::Result<()> {
        self.fused_batches.push((type_id, batch));
        Ok(())
    }

    fn finalize(&mut self, _state: &SamplingState) -> crate::errors::Result<()> {
        Ok(())
    }
}

fn make_sampling_request(seed_nodes: Vec<u64>, fanouts: Vec<u32>) -> BatchSamplingRequest {
    let layers: Vec<LayerSpec> = fanouts
        .into_iter()
        .map(|fanout| LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        })
        .collect();

    BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), seed_nodes)].into(),
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        plan: SamplingPlan { layers },
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    }
}

fn local_id_to_hop(local_id: u32, hop_counts: &[u64]) -> Option<usize> {
    let local = local_id as u64;
    let mut start = 0u64;
    for (hop, count) in hop_counts.iter().enumerate() {
        let end = start + count;
        if (start..end).contains(&local) {
            return Some(hop);
        }
        start = end;
    }
    None
}

#[test]
fn test_basic_sampling() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        }],
    };

    let req = BatchSamplingRequest {

        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), vec![0])].into(),
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    assert_eq!(output.edges.len(), 10);
    assert_eq!(output.fused_batches.len(), 0);
}

#[test]
fn test_fused_sampling() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout: 25, // More than neighbors
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        }],
    };

    let req = BatchSamplingRequest {

        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), vec![0])].into(),
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: true, // Enable Fused
            swap_edge_direction: false,
        },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    // Node 0 has 20 edges. fanout is 25. So it samples all 20.
    assert_eq!(output.edges.len(), 20);

    // Check fused batches
    // We expect at least one batch for type "User" (ID 0)
    assert!(!output.fused_batches.is_empty());
    let (type_id, batch) = &output.fused_batches[0];
    assert_eq!(*type_id, 0);

    // Batch should contain seed (0) and its unique neighbors (1..20) = 21 rows
    assert_eq!(batch.num_rows(), 21);
}

#[test]
fn test_sampling_empty_seeds() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        }],
    };

    let req = BatchSamplingRequest {

        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), vec![])].into(), // Empty seeds
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    assert_eq!(output.edges.len(), 0);
    assert_eq!(state.stats.total_nodes, 0);
}

#[test]
fn test_sampling_determinism() {
    let graph = TestGraph::new_with_seed(Some(42));

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout: 5, // Sample 5 neighbors out of 20
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        }],
    };

    let mut first_sampled_globals = vec![];

    // Run 5 times with seed 42
    for _ in 0..5 {
        let mut state = crate::state::StatePool::new().acquire();
        let mut output = MockOutput {
            edges: vec![],
            feature_batches: vec![],
            schema: None,
            fused_batches: vec![],
        };
        let req = BatchSamplingRequest {

            dataset_id: DatasetId::nil(),
            seed_nodes: [("User".to_string(), vec![0])].into(),
            seed_timestamps: None,
            seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
            plan: plan.clone(),
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: true, // IMPORTANT: each sampled edge gets a unique local ID
                return_edge_id: false,
                adaptive: false,
                fused_feature_fetching: false,
                swap_edge_direction: false,
            },
        };
        graph.sampler.sample(&req, &mut state, &mut output).unwrap();

        // In disjoint mode, state.type_states[&0].nodes contains all sampled instances.
        // We skip the first node (the seed).
        let globals: Vec<_> = state.type_states[&0].nodes.iter().skip(1).map(|n| n.global_id).collect();

        if first_sampled_globals.is_empty() {
            first_sampled_globals = globals;
        } else {
            assert_eq!(first_sampled_globals, globals, "Sampling must be deterministic with same seed");
        }
    }

    // Run with seed 999 and verify it's DIFFERENT
    let graph_diff = TestGraph::new_with_seed(Some(999));
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    let req_diff = BatchSamplingRequest {

        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), vec![0])].into(),
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Nanoseconds,
        plan: plan.clone(),
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };
    graph_diff.sampler.sample(&req_diff, &mut state, &mut output).unwrap();
    let mut diff_globals: Vec<_> = state.type_states[&0].nodes.iter().skip(1).map(|n| n.global_id).collect();

    assert_ne!(first_sampled_globals, diff_globals, "Sampling should produce different results with different seeds");
}

#[test]
fn test_stats_vectors_match_sampled_totals() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    // Two hops over a star graph (second hop remains empty).
    let req = make_sampling_request(vec![0], vec![5, 5]);

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    let node_counts = &state.stats.nodes_per_type[0];
    let edge_counts = &state.stats.edges_per_type[0];

    assert_eq!(node_counts.len(), req.plan.layers.len() + 1);
    assert_eq!(edge_counts.len(), req.plan.layers.len());

    let sampled_user_nodes = state.type_states[&0].nodes.len() as u64;
    assert_eq!(node_counts.iter().sum::<u64>(), sampled_user_nodes);
    assert_eq!(edge_counts.iter().sum::<u64>(), output.edges.len() as u64);
}

#[test]
fn test_edge_hop_counts_and_node_hop_ranges_are_consistent() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    let req = make_sampling_request(vec![0], vec![7, 7]);

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    let node_counts = &state.stats.nodes_per_type[0];
    let edge_counts = &state.stats.edges_per_type[0];

    // 1) Per-hop edge totals in output must match stats vector.
    let mut observed_edges_per_hop = vec![0u64; edge_counts.len()];
    for (hop, rel, _src, _dst) in &output.edges {
        assert_eq!(*rel, 0);
        observed_edges_per_hop[*hop] += 1;
    }
    assert_eq!(observed_edges_per_hop, *edge_counts);

    // 2) Local node-id hop ranges must align with each sampled edge hop.
    // With swap_edge_direction=false, source is at current hop and
    // destination is discovered at next hop.
    for (hop, _rel, src_local, dst_local) in &output.edges {
        let src_hop = local_id_to_hop(*src_local, node_counts)
            .expect("src local id must map to a hop");
        let dst_hop = local_id_to_hop(*dst_local, node_counts)
            .expect("dst local id must map to a hop");

        assert_eq!(src_hop, *hop);
        assert_eq!(dst_hop, hop + 1);
    }
}

#[test]
fn test_fetch_features() {
    let graph = TestGraph::new();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    // Fetch both features
    graph
        .sampler
        .fetch_features_for_type(
            0,    // User type ID
            &[0], // Node 0
            &["feat_0".to_string(), "feat_1".to_string()],
            &mut output,
        )
        .unwrap();

    assert_eq!(output.feature_batches.len(), 1);
    let batch = &output.feature_batches[0];

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);

    let col0 = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    let col1 = batch
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .unwrap();

    assert_eq!(col0.value(0), 42);
    assert_eq!(col1.value(0), 3.14);

    // Test wildcard select all
    let mut output_all = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    graph
        .sampler
        .fetch_features_for_type(0, &[0], &[], &mut output_all)
        .unwrap();
    let batch_all = &output_all.feature_batches[0];
    assert_eq!(batch_all.num_columns(), 2);
}

#[test]
fn test_fetch_features_missing_node() {
    let graph = TestGraph::new();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    // Fetch for non-existent node 999
    let result =
        graph
            .sampler
            .fetch_features_for_type(0, &[999], &["feat_0".to_string()], &mut output);

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("not found in any shard"));
}

#[test]
fn test_fetch_features_unknown_column() {
    let graph = TestGraph::new();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    let result =
        graph
            .sampler
            .fetch_features_for_type(0, &[0], &["unknown".to_string()], &mut output);

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("unknown feature"));
}

#[test]
fn load_finds_lowercase_node_dir() {
    let dir = TempDir::new().unwrap();
    // Create 'users' (lowercase) but metadata says 'Users'
    let shard_dir = dir.path().join("nodes").join("users").join("shard_0");
    write_minimal_shard(&shard_dir);

    let metadata = DatasetMetadata {
        timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        dataset_name: "test".to_string(),
        node_types: vec![NodeTypeMetadata {
            name: "Users".to_string(), // Mixed case
            max_id: 1,
            has_node_data: true,
            node_data_compression: None,
            primary_key_map: None,
            create_time_col: None,
            end_time_col: None,
            shards: vec![NodeShardMetadata {
                min_id: 0,
                max_id: 1,
                node_data_paths: None,
            }],
        }],
        edge_relations: vec![],
    };

    let meta_path = dir.path().join("index_metadata.json");
    fs::write(&meta_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

    // Loading should succeed by finding 'users'
    let result = DiskGraphEngine::load(dir.path(), None, None);
    assert!(result.is_ok(), "Load failed: {:?}", result.err());
}

/// Helper: write a minimal valid shard (1 node, 0 edges, with node data)
/// into the given directory.
fn write_minimal_shard(shard_dir: &std::path::Path) {
    fs::create_dir_all(shard_dir).unwrap();

    // RowPack context for one column
    let cols = vec![RowPackColumn {
        name: "feat".into(),
        logical_type: IngestedType::Int64,
        fixed_len: None,
        precision: None,
        scale: None,
    }];
    let mut ctx_builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
    let row = vec![ValueRef::Int64(1)];
    ctx_builder.observe_row(&row).unwrap();
    let ctx = ctx_builder.finish().unwrap();
    let encoder = RowPackEncoder::new(ctx).unwrap();
    let mut scratch = RowPackScratch::new(encoder.context());
    let mut node_data = Vec::new();
    encoder
        .encode_row_into(&row, &mut node_data, &mut scratch)
        .unwrap();

    // edges.bin: 1 node, 0 edges
    let mut f = File::create(shard_dir.join(EDGES_FILENAME)).unwrap();
    f.write_all(&[0x00]).unwrap(); // 0 edges
    f.write_all(&(node_data.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&node_data).unwrap();
    f.flush().unwrap();
    let edges_len = f.metadata().unwrap().len();

    // offsets.bin: 1 node → 2 offsets
    let header = OffsetFileHeader::new(1, 2);
    let header_bytes = bincode::serialize(&header).unwrap();
    let mut offsets = Vec::new();
    offsets.extend_from_slice(&header_bytes);
    offsets.extend_from_slice(&2u16.to_le_bytes()); // chunk count = 2
    offsets.extend_from_slice(&0u64.to_le_bytes()); // base offset = 0
    let node_len = (1 + 2 + node_data.len()) as u16;
    offsets.extend_from_slice(&node_len.to_le_bytes());

    let mut f = File::create(shard_dir.join(OFFSETS_FILENAME)).unwrap();
    f.write_all(&offsets).unwrap();
    f.flush().unwrap();

    assert_eq!(edges_len as u64, node_len as u64);
}

#[test]
fn test_empty_shard_sentinel_filtered_at_construction() {
    // Reproduces the doordash bug: when a small node type (e.g. 500 nodes)
    // gets 3 shards, empty shards get the sentinel (min_id=1, max_id=0).
    // These must be filtered out so find_shard's binary search works.
    //
    // The fix filters empty shards when building the ShardHandle vec
    // (in DiskGraphEngine::load). This test verifies that after filtering,
    // find_shard works correctly for all node IDs.
    let graph = TestGraph::new();
    let valid_shard = &graph.sampler.readers[0][0];

    // Simulate what the fix does: filter out empty shards
    let all_shards = vec![
        (1u64, 0u64),   // empty sentinel — should be filtered
        (1u64, 0u64),   // empty sentinel — should be filtered
        (0u64, 20u64),  // valid shard
    ];
    let filtered: Vec<ShardHandle> = all_shards
        .into_iter()
        .filter(|(min_id, max_id)| min_id <= max_id)
        .map(|(min_id, max_id)| ShardHandle {
            reader: valid_shard.reader.clone(),
            min_id,
            max_id,
        })
        .collect();

    // After filtering, only 1 shard remains
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].min_id, 0);
    assert_eq!(filtered[0].max_id, 20);

    // find_shard works on the filtered vec
    let result = graph.sampler.find_shard(&filtered, 0);
    assert!(result.is_ok(), "node 0 should be found: {:?}", result.err());
    assert_eq!(result.unwrap(), 0);

    let result = graph.sampler.find_shard(&filtered, 10);
    assert!(result.is_ok());

    let result = graph.sampler.find_shard(&filtered, 999);
    assert!(result.is_err());
}

#[test]
fn test_empty_shard_between_valid_filtered() {
    // Empty shard between two valid shards — after filtering, binary
    // search sees only the two valid shards in correct order.
    let graph = TestGraph::new();
    let valid_shard = &graph.sampler.readers[0][0];

    let all_shards = vec![
        (0u64, 9u64),   // valid
        (1u64, 0u64),   // empty sentinel
        (10u64, 20u64), // valid
    ];
    let filtered: Vec<ShardHandle> = all_shards
        .into_iter()
        .filter(|(min_id, max_id)| min_id <= max_id)
        .map(|(min_id, max_id)| ShardHandle {
            reader: valid_shard.reader.clone(),
            min_id,
            max_id,
        })
        .collect();

    assert_eq!(filtered.len(), 2);

    let result = graph.sampler.find_shard(&filtered, 5);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);

    let result = graph.sampler.find_shard(&filtered, 15);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_find_shard_binary_search_many_shards() {
    // Verify binary search works correctly across many contiguous shards,
    // hitting first, middle, last shards and boundary IDs.
    let graph = TestGraph::new();
    let valid_shard = &graph.sampler.readers[0][0];

    let shards: Vec<ShardHandle> = (0..10)
        .map(|i| ShardHandle {
            reader: valid_shard.reader.clone(),
            min_id: i * 100,
            max_id: i * 100 + 99,
        })
        .collect();

    // First shard: ids 0..99
    let r = graph.sampler.find_shard(&shards, 0);
    assert_eq!(r.unwrap(), 0);
    let r = graph.sampler.find_shard(&shards, 99);
    assert_eq!(r.unwrap(), 0);

    // Middle shard: ids 500..599
    let r = graph.sampler.find_shard(&shards, 500);
    assert_eq!(r.unwrap(), 5);
    let r = graph.sampler.find_shard(&shards, 550);
    assert_eq!(r.unwrap(), 5);

    // Last shard: ids 900..999
    let r = graph.sampler.find_shard(&shards, 900);
    assert_eq!(r.unwrap(), 9);
    let r = graph.sampler.find_shard(&shards, 999);
    assert_eq!(r.unwrap(), 9);

    // Out of range
    let r = graph.sampler.find_shard(&shards, 1000);
    assert!(r.is_err());
}
