// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
        Self::new_with_seed_and_weight(random_seed, false)
    }

    /// Like `new_with_seed`, but when `include_reserved` is set the rowpack adds
    /// the reserved task-column set (`WEIGHT`, `ENTITY1`, `TARGET1`,
    /// `entity_node_id`, `target_node_ids`, frequency) plus real `TIMESTAMP` and
    /// `TARGET_AUDIENCE` features. Used to prove the select_all and fused
    /// feature paths strip task-only columns while keeping real ones.
    fn new_with_seed_and_weight(random_seed: Option<u64>, include_reserved: bool) -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // 1. Setup RowPack Context & Data
        let mut cols = vec![
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
        let mut row_val = vec![ValueRef::Int64(42), ValueRef::Float32(3.14)];
        if include_reserved {
            // Reserved task columns must be stripped from graph features.
            let reserved: [(&str, IngestedType, ValueRef); 7] = [
                ("WEIGHT", IngestedType::Float64, ValueRef::Float64(0.75)),
                (
                    "__kumo_internal_frequency_column__",
                    IngestedType::Int64,
                    ValueRef::Int64(1),
                ),
                ("ENTITY1", IngestedType::Int64, ValueRef::Int64(7)),
                ("TARGET1", IngestedType::Int64, ValueRef::Int64(9)),
                ("TIMESTAMP", IngestedType::Int64, ValueRef::Int64(123)),
                ("entity_node_id", IngestedType::Int64, ValueRef::Int64(7)),
                ("target_node_ids", IngestedType::Int64, ValueRef::Int64(9)),
            ];
            for (name, ty, val) in reserved {
                cols.push(RowPackColumn {
                    name: name.into(),
                    logical_type: ty,
                    fixed_len: None,
                    precision: None,
                    scale: None,
                });
                row_val.push(val);
            }
            // Precise-match guard: `TIMESTAMP` and `TARGET_AUDIENCE` are REAL
            // graph features; the generic reserved predicate must not match
            // them. Seed-rowpack projection has its own TIMESTAMP exclusion.
            cols.push(RowPackColumn {
                name: "TARGET_AUDIENCE".into(),
                logical_type: IngestedType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            });
            row_val.push(ValueRef::Int64(5));
        }
        let mut ctx_builder = RowPackContextBuilder::new(cols.clone(), RowPackConfig::default());
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
    make_sampling_request_with_mode(seed_nodes, fanouts, SamplingMode::Uniform)
}

fn make_sampling_request_with_mode(
    seed_nodes: Vec<u64>,
    fanouts: Vec<u32>,
    mode: SamplingMode,
) -> BatchSamplingRequest {
    let layers: Vec<LayerSpec> = fanouts
        .into_iter()
        .map(|fanout| LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout,
                    mode: mode.clone(),
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
        plan: SamplingPlan {
            layers,
        },
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

fn sample_user_globals(graph: &TestGraph, req: &BatchSamplingRequest) -> Vec<u64> {
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };

    graph.sampler.sample(req, &mut state, &mut output).unwrap();

    state.type_states[&0]
        .nodes
        .iter()
        .skip(1)
        .map(|n| n.global_id)
        .collect()
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
    let diff_globals: Vec<_> = state.type_states[&0].nodes.iter().skip(1).map(|n| n.global_id).collect();

    assert_ne!(first_sampled_globals, diff_globals, "Sampling should produce different results with different seeds");
}

#[test]
fn test_deterministic_uniform_mode_is_stable_on_unseeded_engine() {
    let graph = TestGraph::new();
    let req = make_sampling_request_with_mode(
        vec![0],
        vec![5],
        SamplingMode::DeterministicUniform,
    );

    let first_sampled_globals = sample_user_globals(&graph, &req);
    for _ in 0..5 {
        assert_eq!(
            first_sampled_globals,
            sample_user_globals(&graph, &req),
            "deterministic_uniform should be stable without an engine seed",
        );
    }

    let seeded_graph = TestGraph::new_with_seed(Some(999));
    assert_eq!(
        first_sampled_globals,
        sample_user_globals(&seeded_graph, &req),
        "deterministic_uniform should use the mode seed, not the engine seed",
    );
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

/// select_all / neighbor path on a reserved-bearing rowpack must surface no
/// task-only reserved column in features while keeping real graph features
/// (`TIMESTAMP`, `TARGET_AUDIENCE`).
#[test]
fn test_fetch_features_select_all_excludes_weight() {
    let graph = TestGraph::new_with_seed_and_weight(None, true);

    const TASK_RESERVED: [&str; 6] = [
        "WEIGHT",
        "__kumo_internal_frequency_column__",
        "ENTITY1",
        "TARGET1",
        "entity_node_id",
        "target_node_ids",
    ];

    // Sanity: the rowpack genuinely carries every reserved column.
    for r in TASK_RESERVED.iter().copied().chain(["TIMESTAMP"]) {
        assert!(
            graph.rowpack_ctx.columns.iter().any(|c| c.name == r),
            "test setup should include column {r}"
        );
    }

    // select_all path (empty feature_names) — this is the neighbor /
    // post-sampling materialization site (engine.rs ~1379/1631).
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

    // Schema (sent on the projected selection) must exclude all reserved cols
    // and keep ONLY the real features (incl. the TARGET_AUDIENCE foot-gun).
    let schema = output_all.schema.expect("schema should be emitted");
    let schema_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    for r in TASK_RESERVED {
        assert!(
            !schema_names.contains(&r),
            "{r} must not appear in the select_all schema: {schema_names:?}"
        );
    }
    assert_eq!(schema_names, vec!["feat_0", "feat_1", "TIMESTAMP", "TARGET_AUDIENCE"]);

    // The materialized batch must carry only the real features.
    let batch_all = &output_all.feature_batches[0];
    let batch_all_schema = batch_all.schema();
    let batch_names: Vec<&str> = batch_all_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(batch_names, vec!["feat_0", "feat_1", "TIMESTAMP", "TARGET_AUDIENCE"]);

    // Wildcard "*" path must behave identically.
    let mut output_star = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    graph
        .sampler
        .fetch_features_for_type(0, &[0], &["*".to_string()], &mut output_star)
        .unwrap();
    assert_eq!(output_star.feature_batches[0].num_columns(), 4);
}

/// Fused path: `accumulate_features_from_ids_builder` → `flush_fused_features`
/// must also drop every task-only reserved column while keeping real graph
/// features.
#[test]
fn test_fetch_features_fused_excludes_weight() {
    let graph = TestGraph::new_with_seed_and_weight(None, true);

    const TASK_RESERVED: [&str; 6] = [
        "WEIGHT",
        "__kumo_internal_frequency_column__",
        "ENTITY1",
        "TARGET1",
        "entity_node_id",
        "target_node_ids",
    ];

    // Sanity: the rowpack genuinely carries every reserved column.
    for r in TASK_RESERVED.iter().copied().chain(["TIMESTAMP"]) {
        assert!(
            graph.rowpack_ctx.columns.iter().any(|c| c.name == r),
            "test setup should include column {r}"
        );
    }

    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    let mut type_state = crate::state::TypeState::new();
    let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();

    // Drive the fused leaf-fetch path for node 0, then flush. Fused mode
    // always fetches every non-reserved column; it does not consult a
    // `ResolvedProjection`.
    graph
        .sampler
        .accumulate_features_from_ids_builder(0, &[0], &mut type_state, &mut scratch, &mut output)
        .unwrap();
    graph
        .sampler
        .flush_fused_features(0, &mut type_state, &mut output)
        .unwrap();

    assert_eq!(output.fused_batches.len(), 1, "one fused batch expected");
    let (_type_id, batch) = &output.fused_batches[0];
    let batch_schema = batch.schema();
    let names: Vec<&str> = batch_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    for r in TASK_RESERVED {
        assert!(
            !names.contains(&r),
            "{r} must not appear in fused features: {names:?}"
        );
    }
    // field count == array count is enforced by RecordBatch::try_new; assert
    // the surviving columns are exactly the real features.
    assert_eq!(names, vec!["feat_0", "feat_1", "TIMESTAMP", "TARGET_AUDIENCE"]);
}

/// `resolve_projection` is a TRI-STATE, and each state is reachable only
/// through a distinct spelling of the caller-facing map. Collapsing any two of
/// them is the bug class this whole parameter exists to prevent, so pin all
/// three plus the skew fallback.
#[test]
fn test_resolve_projection_tri_state() {
    let graph = TestGraph::new();
    let type_name = graph.sampler.id_to_node_type[0].clone();
    let stored = graph.rowpack_ctx.columns.len();
    assert!(stored >= 2, "fixture must store >= 2 columns to prove narrowing");

    // 1. Empty map => select all, everywhere.
    let all = graph.sampler.resolve_projection(&Default::default());
    assert_eq!(all.names_for(0), None, "empty map must mean select-all");
    assert!(!all.skips(0));

    // 2. Key absent => select all for that type, even when OTHER keys are set.
    let other = HashMap::from([("not_a_node_type".to_string(), vec!["x".to_string()])]);
    let other = graph.sampler.resolve_projection(&other);
    assert_eq!(other.names_for(0), None, "absent key must mean select-all");
    assert!(!other.skips(0));

    // 3. Non-empty list => exactly those.
    let narrow = HashMap::from([(type_name.clone(), vec!["feat_0".to_string()])]);
    let narrow = graph.sampler.resolve_projection(&narrow);
    assert_eq!(narrow.names_for(0), Some(&["feat_0".to_string()][..]));
    assert!(!narrow.skips(0));

    // 4. Empty list => select NOTHING. Distinct from 1 and 2 -- that
    //    distinction is the entire point of the `projections` rename.
    let skip = HashMap::from([(type_name.clone(), Vec::<String>::new())]);
    let skip = graph.sampler.resolve_projection(&skip);
    assert_eq!(skip.names_for(0), Some(&[][..]));
    assert!(skip.skips(0), "an empty request must skip the type");

    // 5. A NON-empty request that intersects nothing is schema skew, not a
    //    skip: fall back to select-all rather than starving the encoders.
    let skew = HashMap::from([(type_name.clone(), vec!["no_such_column".to_string()])]);
    let skew = graph.sampler.resolve_projection(&skew);
    assert_eq!(skew.names_for(0), None, "skew must degrade to select-all");
    assert!(!skew.skips(0), "skew must NOT be mistaken for a skip request");

    // 6. A PARTIAL miss (one requested name stored, one not) must still
    //    intersect down to the surviving name(s) rather than degrading to
    //    select-all -- that fallback is reserved for a TOTAL miss (case 5).
    let stored_name = graph.rowpack_ctx.columns[0].name.clone();
    let partial = HashMap::from([(
        type_name.clone(),
        vec![stored_name.clone(), "no_such_column".to_string()],
    )]);
    let partial = graph.sampler.resolve_projection(&partial);
    assert_eq!(
        partial.names_for(0),
        Some(&[stored_name.clone()][..]),
        "a partial miss keeps the surviving column(s), not select-all"
    );
    assert!(!partial.skips(0));

    // 7. A DUPLICATE requested name must be deduped, not carried through
    // twice: a duplicate field name in the resolved list becomes a duplicate
    // field in the projected Arrow schema downstream, which corrupts any
    // column lookup by name.
    let duplicated = HashMap::from([(type_name, vec![stored_name.clone(), stored_name.clone()])]);
    let duplicated = graph.sampler.resolve_projection(&duplicated);
    assert_eq!(
        duplicated.names_for(0),
        Some(&[stored_name][..]),
        "a duplicated requested column must appear exactly once in the resolved list"
    );
}

/// `is_reserved_feature_column` matches the reserved set exactly (with
/// `ENTITY<n>`/`TARGET<n>` requiring a digit suffix). Prefix-sharing or
/// bare-keyword columns (`TARGET_AUDIENCE`, `TARGET`, `ENTITY`) must NOT match.
#[test]
fn test_is_reserved_feature_column_precise_match() {
    use crate::engine::is_reserved_feature_column as r;

    // Reserved — must match.
    for name in [
        "WEIGHT",
        "__kumo_internal_frequency_column__",
        "entity_node_id",
        "target_node_ids",
        "ENTITY1",
        "ENTITY2",
        "TARGET1",
        "TARGET12",
    ] {
        assert!(r(name), "{name} should be reserved");
    }

    // NOT reserved — real features / foot-guns.
    for name in [
        "TARGET_AUDIENCE",
        "TIMESTAMP",
        "TARGETED",
        "TARGET",
        "ENTITY",
        "ENTITY_NAME",
        "ENTITY1A",
        "feat_0",
        "weight",
        "Timestamp",
    ] {
        assert!(!r(name), "{name} must NOT be reserved");
    }
}

/// `project_seed_rowpack` routing (uses `starts_with` since it only runs on the
/// controlled seed type): TARGET* → target projection; task/resolution/weight
/// columns (ENTITY1, TIMESTAMP, entity_node_id, target_node_ids, WEIGHT,
/// frequency) are excluded from both; remaining → feature projection.
#[test]
fn test_project_seed_rowpack_routing() {
    use arrow_array::{Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
        Field::new("feat_0", DataType::Int64, false),
        Field::new("TARGET1", DataType::Int64, false),
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("TIMESTAMP", DataType::Int64, false),
        Field::new("entity_node_id", DataType::Int64, false),
        Field::new("target_node_ids", DataType::Int64, false),
        Field::new("WEIGHT", DataType::Float64, false),
        Field::new("__kumo_internal_frequency_column__", DataType::Int64, false),
    ]));
    let i = |v: i64| Arc::new(Int64Array::from(vec![v])) as arrow_array::ArrayRef;
    let rb = RecordBatch::try_new(
        schema,
        vec![
            i(1),
            i(2),
            i(3),
            i(4),
            i(5),
            i(6),
            Arc::new(Float64Array::from(vec![0.5])),
            i(1),
        ],
    )
    .unwrap();

    let (target_proj, feature_proj) = crate::engine::project_seed_rowpack(&rb).unwrap();

    // TARGET1 → target projection.
    let target_proj = target_proj.expect("target projection");
    let target_schema = target_proj.schema();
    let target_names: Vec<&str> = target_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(target_names, vec!["TARGET1"]);

    // Features: only the real column. ENTITY1/TIMESTAMP/entity_node_id/
    // target_node_ids/WEIGHT/frequency are all excluded.
    let feature_proj = feature_proj.expect("feature projection");
    let feature_schema = feature_proj.schema();
    let feature_names: Vec<&str> = feature_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(feature_names, vec!["feat_0"]);
    for excluded in [
        "WEIGHT",
        "ENTITY1",
        "TIMESTAMP",
        "entity_node_id",
        "target_node_ids",
        "__kumo_internal_frequency_column__",
    ] {
        assert!(
            !feature_names.contains(&excluded),
            "{excluded} must not be a seed feature: {feature_names:?}"
        );
    }
}

#[test]
fn test_hydrate_ragged_lp_weights_flattens_a_ragged_list() {
    use arrow_array::builder::{Int64Builder, ListBuilder};

    let mut weight = ListBuilder::new(Int64Builder::new());
    weight.values().append_value(2);
    weight.values().append_value(5);
    weight.append(true);
    weight.values().append_value(7);
    weight.append(true);
    weight.append(true);

    let weights =
        crate::engine::hydrate_ragged_lp_weights(&weight.finish()).expect("weights hydrate");
    assert_eq!(weights, vec![2.0, 5.0, 7.0]);
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

    assert_eq!(edges_len, node_len as u64);
}

#[test]
fn test_empty_shard_sentinel_filtered_at_construction() {
    // When a small node type (e.g. 500 nodes)
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

// ---------------------------------------------------------------------------
// Per-type temporal anchor regression test.
//
// Pins the fix for the `seed_nodes.keys().next()` bug: the
// `TemporalMostRecent` sampler used to resolve EVERY node's timestamp anchor
// against an arbitrary HashMap-ordered seed type instead of the node's own
// root seed type. With multiple seed types (LP lhs + rhs) this reads the
// wrong per-type `seed_timestamps` array — and once the arrays differ in
// length, indexes out of bounds, silently dropping the temporal filter
// (future-edge leakage). The fix threads `seed_type_id` through the
// traversal so the anchor comes from the node's OWN type array.
// ---------------------------------------------------------------------------

use diskgraph_core::metadata::{
    EdgeSortKeyMetadata, EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampKind,
    EdgeSortKeyTimestampMetadata,
};
use diskgraph_core::sortkey::{SortKeyDecoder, SortKeyEncoder};

/// 16-bit timestamp sort-key metadata: anchor 0, step 1. Suitable for small
/// integer timestamps used in the temporal anchor test below.
fn temporal_sortkey_meta() -> EdgeSortKeyMetadata {
    EdgeSortKeyMetadata {
        width_bits: 16,
        timestamp: Some(EdgeSortKeyTimestampMetadata {
            table: "t".into(),
            column: "ts".into(),
            encoding: EdgeSortKeyTimestampEncoding {
                kind: EdgeSortKeyTimestampKind::Int16,
                anchor: 0,
                step: 1,
            },
        }),
        weight: None,
    }
}

/// Build a single-shard reader for a node type whose node 0 has the given
/// `(neighbor_id, timestamp)` edges encoded with a 16-bit timestamp sort key.
/// All nodes 0..=max_neighbor exist (so destinations are addressable). Returns
/// the reader plus the RelationLayout used (rel_id 0).
///
/// Edge entry packing (see EdgeUnpacker): `dst | (sort_key << dst_bits)`, with
/// dst_bits = 8 and a 16-bit sort key => 3 bytes per entry.
fn build_temporal_shard(
    dir: &std::path::Path,
    edges: &[(u8, i64)],
    max_node: u8,
) -> (ShardReader, RelationLayout, diskgraph_rowpack::RowPackContext) {
    let sk_encoder = SortKeyEncoder::new(&temporal_sortkey_meta());
    let entries: Vec<(u8, u64)> =
        edges.iter().map(|&(dst, ts)| (dst, sk_encoder.encode_lower(ts))).collect();
    build_sort_key_shard(dir, &entries, max_node, 16)
}

fn build_sort_key_shard(
    dir: &std::path::Path,
    entries: &[(u8, u64)],
    max_node: u8,
    key_bits: u8,
) -> (ShardReader, RelationLayout, diskgraph_rowpack::RowPackContext) {
    let (ctx, node_data) = trivial_node_rowpack();
    let mut adjacency = vec![Vec::new(); max_node as usize + 1];
    adjacency[0] = entries.to_vec();
    let (reader, layout) = write_adjacency_shard(dir, &adjacency, key_bits, &node_data);
    (reader, layout, ctx)
}

/// Node feature context (one trivial column), mirroring TestGraph.
fn trivial_node_rowpack() -> (diskgraph_rowpack::RowPackContext, Vec<u8>) {
    let cols = vec![RowPackColumn {
        name: "feat_0".into(),
        logical_type: IngestedType::Int64,
        fixed_len: None,
        precision: None,
        scale: None,
    }];
    let mut ctx_builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
    let row_val = vec![ValueRef::Int64(7)];
    ctx_builder.observe_row(&row_val).unwrap();
    let ctx = ctx_builder.finish().unwrap();
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let mut scratch = RowPackScratch::new(&ctx);
    let mut node_data = Vec::new();
    encoder
        .encode_row_into(&row_val, &mut node_data, &mut scratch)
        .unwrap();
    (ctx, node_data)
}

/// Writes one shard whose node `i` has the `(dst, sort_key)` entries in
/// `adjacency[i]` (rel 0, 8-bit dst ids, at most 63 nodes).
fn write_adjacency_shard(
    dir: &std::path::Path,
    adjacency: &[Vec<(u8, u64)>],
    key_bits: u8,
    node_data: &[u8],
) -> (ShardReader, RelationLayout) {
    let entry_bytes = 1 + key_bits as usize / 8;
    let layout = RelationLayout {
        relation_id: 0,
        dst_bits: 8,
        dest_sort_key_bits: Some(key_bits),
        entry_bytes,
        dst_mask: 0xFF,
        dest_sort_key_mask: Some(u64::MAX >> (64 - key_bits)),
    };

    // Each node block: [count u8][len u16][entry_bytes*count edge bytes][node_data].
    let mut f = File::create(dir.join(EDGES_FILENAME)).unwrap();
    let mut lens = Vec::new();
    for entries in adjacency {
        let edge_bytes: Vec<u8> = entries
            .iter()
            .flat_map(|&(dst, sort_key)| {
                let entry = (dst as u128) | ((sort_key as u128) << 8);
                entry.to_le_bytes()[..entry_bytes].to_vec()
            })
            .collect();
        f.write_all(&[entries.len() as u8]).unwrap();
        f.write_all(&(node_data.len() as u16).to_le_bytes()).unwrap();
        f.write_all(&edge_bytes).unwrap();
        f.write_all(node_data).unwrap();
        lens.push((1 + 2 + edge_bytes.len() + node_data.len()) as u64);
    }
    f.flush().unwrap();
    let edges_len = f.metadata().unwrap().len();

    let num_nodes = adjacency.len() as u64;
    let header = OffsetFileHeader::new(num_nodes, num_nodes + 1);
    let mut offsets = bincode::serialize(&header).unwrap();
    offsets.extend_from_slice(&((num_nodes + 1) as u16).to_le_bytes());
    offsets.extend_from_slice(&0u64.to_le_bytes()); // base offset
    for &len in &lens {
        offsets.extend_from_slice(&(len as u16).to_le_bytes());
    }
    assert_eq!(
        lens.iter().sum::<u64>(),
        edges_len,
        "offsets must sum to edge file size"
    );

    let mut fo = File::create(dir.join(OFFSETS_FILENAME)).unwrap();
    fo.write_all(&offsets).unwrap();
    fo.flush().unwrap();

    let reader = ShardReader::open(dir, Arc::new(vec![layout.clone()]), true).unwrap();
    (reader, layout)
}

/// Run a `TemporalMostRecent` sample over a two-seed-type graph and return the
/// number of edges emitted from the `rhs` seed. `lhs_anchor` / `rhs_anchor`
/// are the per-type seed timestamps; the rhs seed is sampled and its neighbors
/// (all timestamped 50) are kept only if `edge_ts <= anchor`.
///
/// Two node types ("lhs" id 0, "rhs" id 1, sorted) each get their own shard.
/// Only the rhs->rhs relation (rel 0) carries temporal edges. The rhs seed
/// sits at seed_index 0; lhs is seeded too so `seed_nodes` has two keys (the
/// arbitrary `keys().next()` of the old code could pick either).
fn run_two_type_temporal(lhs_anchor: i64, rhs_anchor: i64, fused: bool) -> usize {
    let dir_rhs = TempDir::new().unwrap();
    let dir_lhs = TempDir::new().unwrap();

    // rhs node 0: 4 neighbors (1..=4), all at timestamp 50.
    let rhs_edges = [(1u8, 50i64), (2, 50), (3, 50), (4, 50)];
    let (rhs_reader, _layout, rhs_ctx) =
        build_temporal_shard(dir_rhs.path(), &rhs_edges, 4);
    // lhs has no edges (just node 0 so it's a valid seed).
    let (lhs_reader, _l2, lhs_ctx) = build_temporal_shard(dir_lhs.path(), &[], 0);

    let mut readers: HashMap<String, Vec<ShardHandle>> = HashMap::new();
    readers.insert(
        "rhs".to_string(),
        vec![ShardHandle {
            reader: Arc::new(rhs_reader),
            min_id: 0,
            max_id: 4,
        }],
    );
    readers.insert(
        "lhs".to_string(),
        vec![ShardHandle {
            reader: Arc::new(lhs_reader),
            min_id: 0,
            max_id: 0,
        }],
    );

    // Single relation: rhs -> rhs (rel_id 0) with a temporal sort key.
    let mut relation_info = HashMap::new();
    relation_info.insert(0, ("Connects".to_string(), "rhs".to_string(), "rhs".to_string()));

    let mut decoders_map = HashMap::new();
    decoders_map.insert(0usize, SortKeyDecoder::new(&temporal_sortkey_meta()));

    let mut node_decoders = HashMap::new();
    node_decoders.insert("rhs".to_string(), RowPackDecoder::new(rhs_ctx).unwrap());
    node_decoders.insert("lhs".to_string(), RowPackDecoder::new(lhs_ctx).unwrap());

    let sampler = DiskGraphEngine::new(
        readers,
        relation_info,
        decoders_map,
        node_decoders,
        None,
        diskgraph_core::time::TimeUnit::Seconds,
        Some(42),
        Some(1),
    );

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [(
                "Connects".to_string(),
                FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::TemporalMostRecent,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into(),
            default_fanout: None,
        }],
    };

    let mut seed_timestamps: HashMap<String, Vec<i64>> = HashMap::new();
    seed_timestamps.insert("lhs".to_string(), vec![lhs_anchor]);
    seed_timestamps.insert("rhs".to_string(), vec![rhs_anchor]);

    let mut seed_nodes: HashMap<String, Vec<u64>> = HashMap::new();
    seed_nodes.insert("lhs".to_string(), vec![0]);
    seed_nodes.insert("rhs".to_string(), vec![0]);

    let req = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes,
        seed_timestamps: Some(seed_timestamps),
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: fused,
            swap_edge_direction: false,
        },
    };

    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    sampler.sample(&req, &mut state, &mut output).unwrap();
    output.edges.len()
}

#[test]
fn test_temporal_anchor_uses_own_seed_type() {
    // The rhs seed's neighbors are all at timestamp 50. The rhs anchor is held
    // fixed at 1000 (>= 50), so under correct per-type resolution ALL 4 rhs
    // neighbors are kept regardless of the lhs anchor — the lhs timestamp array
    // must NEVER influence the rhs neighborhood.
    //
    // Under the OLD `seed_nodes.keys().next()` logic the anchor for the rhs
    // node was taken from whichever seed type the HashMap happened to yield
    // first. When that was "lhs", the rhs neighbors were filtered against the
    // lhs anchor — so a small lhs anchor (e.g. 0 < 50) would wrongly DROP every
    // rhs neighbor. `keys().next()` ordering is randomized per process (std
    // HashMap RandomState), so we run many independent requests, each with a
    // freshly seeded HashMap, and assert the result is invariant. The new code
    // always reads the rhs node's own anchor, so every iteration keeps all 4
    // edges. The old code fails an iteration with prob ~1/2, so 64 iterations
    // catch the regression with prob ~1 - 2^-64.
    // The anchor fix lives in shared `sample_segment_edges` used by BOTH
    // `process_layer` branches, so we pin both: `fused = false` exercises the
    // multi-threaded map-reduce path, `fused = true` the inline fused path.
    for fused in [false, true] {
        for _ in 0..64 {
            // lhs anchor 0 would drop all rhs edges if (wrongly) applied to them;
            // rhs anchor 1000 keeps them. Correct behavior: always 4 edges.
            let kept = run_two_type_temporal(/* lhs_anchor */ 0, /* rhs_anchor */ 1000, fused);
            assert_eq!(
                kept, 4,
                "rhs neighborhood must be filtered by rhs's own anchor (1000), \
                 not the lhs anchor (0); got {kept} edges (fused={fused})"
            );
        }

        // Sanity: when the rhs's OWN anchor is below the edge timestamps, the
        // temporal filter genuinely drops them — proving the filter is active and
        // the count above is not trivially 4 for unrelated reasons.
        let dropped = run_two_type_temporal(/* lhs_anchor */ 1000, /* rhs_anchor */ 0, fused);
        assert_eq!(
            dropped, 0,
            "with rhs anchor 0 < edge ts 50, all rhs neighbors must be filtered out (fused={fused})"
        );
    }
}

fn most_recent_neighbors(
    meta: &EdgeSortKeyMetadata,
    entries: &[(u8, u64)],
    anchor: i64,
    fanout: u32,
) -> Vec<u64> {
    let dir = TempDir::new().unwrap();
    let (reader, _, ctx) = build_sort_key_shard(dir.path(), entries, 255, meta.width_bits);
    let readers: HashMap<String, Vec<ShardHandle>> = [(
        "n".to_string(),
        vec![ShardHandle {
            reader: Arc::new(reader),
            min_id: 0,
            max_id: 255,
        }],
    )]
    .into();
    let relation_info = [(0, ("E".to_string(), "n".to_string(), "n".to_string()))].into();
    let decoders = [(0usize, SortKeyDecoder::new(meta))].into();
    let node_decoders = [("n".to_string(), RowPackDecoder::new(ctx).unwrap())].into();
    let sampler = DiskGraphEngine::new(
        readers,
        relation_info,
        decoders,
        node_decoders,
        None,
        diskgraph_core::time::TimeUnit::Seconds,
        Some(42),
        Some(1),
    );
    let req = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes: [("n".to_string(), vec![0])].into(),
        seed_timestamps: Some([("n".to_string(), vec![anchor])].into()),
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan: SamplingPlan {
            layers: vec![LayerSpec {
                fanouts: [(
                    "E".to_string(),
                    FanoutSpec {
                        fanout,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    },
                )]
                .into(),
                default_fanout: None,
            }],
        },
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
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    sampler.sample(&req, &mut state, &mut output).unwrap();
    state.type_states[&0].nodes[1..].iter().map(|n| n.global_id).collect()
}

#[test]
fn test_temporal_most_recent_matches_linear_scan() {
    use diskgraph_core::metadata::{
        EdgeSortKeyWeightEncoding, EdgeSortKeyWeightKind, EdgeSortKeyWeightMetadata,
    };
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    let meta = |kind, bits, anchor, step, weight: Option<EdgeSortKeyWeightKind>| {
        EdgeSortKeyMetadata {
            width_bits: bits,
            timestamp: Some(EdgeSortKeyTimestampMetadata {
                table: "t".into(),
                column: "ts".into(),
                encoding: EdgeSortKeyTimestampEncoding { kind, anchor, step },
            }),
            weight: weight.map(|dtype| EdgeSortKeyWeightMetadata {
                table: "t".into(),
                expr: "w".into(),
                encoding: EdgeSortKeyWeightEncoding { dtype },
            }),
        }
    };
    use EdgeSortKeyTimestampKind::{Int16, Int32};
    let cases = [
        (meta(Int16, 16, 0, 1, None), 0),
        (meta(Int16, 24, 1_000, 60, Some(EdgeSortKeyWeightKind::Byte)), 8),
        (meta(Int32, 64, -5, 7, Some(EdgeSortKeyWeightKind::Float32)), 32),
    ];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    for (meta, weight_bits) in &cases {
        let encoder = SortKeyEncoder::new(meta);
        let decoder = SortKeyDecoder::new(meta);
        for _ in 0..200 {
            let len = rng.gen_range(0..=40);
            let mut dsts: Vec<u8> = (1..=255).collect();
            dsts.shuffle(&mut rng);
            let mut entries: Vec<(u8, u64)> = dsts[..len]
                .iter()
                .map(|&dst| {
                    let ts = if rng.gen_bool(0.2) {
                        0
                    } else {
                        encoder.encode_lower(rng.gen_range(0..2_000)) >> weight_bits
                    };
                    let weight = rng.gen::<u64>() & ((1u64 << weight_bits) - 1);
                    (dst, (ts << weight_bits) | weight)
                })
                .collect();
            entries.sort_by_key(|&(_, key)| std::cmp::Reverse(key));
            let anchor = rng.gen_range(-100..2_100);
            let fanout = rng.gen_range(1..=50);
            let expected: Vec<u64> = entries
                .iter()
                .filter(|&&(_, key)| decoder.decode(key).0.is_none_or(|ts| ts <= anchor))
                .take(fanout as usize)
                .map(|&(dst, _)| dst as u64)
                .collect();
            assert_eq!(most_recent_neighbors(meta, &entries, anchor, fanout), expected);
        }
    }
}

// ---------------------------------------------------------------------------
// Regression: disjoint dedup must distinguish nodes by ROOT SEED TYPE.
//
// `seed_index` is the seed's position *within its own seed type* and resets to
// 0 per type (engine.rs: `for (i, &global_id) in seed_ids.iter().enumerate()`,
// per `(seed_type, seed_ids)`). So an lhs seed #0 and an rhs seed #0 BOTH carry
// seed_index 0. Before this fix `LookupKey` was `{seed_index, global_id}` with
// no type, so a single global node reachable from BOTH an lhs-seed subtree and
// an rhs-seed subtree at the same seed_index collided on one key and wrongly
// MERGED into a single `NodeRecord` — breaking disjoint semantics (the merged
// node carries edges from both subtrees). Adding `seed_type_id` to the key
// keeps them as two distinct instances.
//
// This test builds exactly that collision: two node types lhs (id 0) and rhs
// (id 1); a `lhs -> rhs` relation and a `rhs -> rhs` relation, both pointing at
// the SAME rhs node (global id 2). Seeds {lhs:[0], rhs:[0]} (both seed_index 0).
// After one hop the shared rhs node must appear as TWO records in the rhs
// TypeState — one rooted at lhs (seed_type_id 0), one at rhs (seed_type_id 1).
// ---------------------------------------------------------------------------

#[test]
fn test_disjoint_dedup_distinguishes_root_seed_type() {
    // The dedup-key fix lives in shared `record_edge`, used by BOTH
    // `process_layer` branches. Pin both: `fused = false` is the multi-threaded
    // map-reduce path, `fused = true` the inline fused path. The two-instance
    // outcome must be identical regardless of feature-fetching mode.
    for fused in [false, true] {
        run_disjoint_dedup_check(fused);
    }
}

fn run_disjoint_dedup_check(fused: bool) {
    let dir_lhs = TempDir::new().unwrap();
    let dir_rhs = TempDir::new().unwrap();

    // The shared rhs node both seeds reach is global id 2. Timestamps are
    // irrelevant here (Uniform mode applies no temporal filter), but the shard
    // builder always encodes a sort key, so we pass an arbitrary value.
    let shared: u8 = 2;

    // lhs node 0 -> rhs node `shared`. lhs has a single node (id 0).
    let (lhs_reader, _l_layout, lhs_ctx) =
        build_temporal_shard(dir_lhs.path(), &[(shared, 1i64)], 0);
    // rhs node 0 -> rhs node `shared`. rhs needs nodes 0..=shared addressable.
    let (rhs_reader, _r_layout, rhs_ctx) =
        build_temporal_shard(dir_rhs.path(), &[(shared, 1i64)], shared);

    let mut readers: HashMap<String, Vec<ShardHandle>> = HashMap::new();
    readers.insert(
        "lhs".to_string(),
        vec![ShardHandle {
            reader: Arc::new(lhs_reader),
            min_id: 0,
            max_id: 0,
        }],
    );
    readers.insert(
        "rhs".to_string(),
        vec![ShardHandle {
            reader: Arc::new(rhs_reader),
            min_id: 0,
            max_id: shared as u64,
        }],
    );

    // Two relations, one per source type. Each type therefore has exactly one
    // relation at local position 0 (its shard segment's `relation_id: 0`), which
    // `local_to_merged_rel` translates to these merged ids:
    //   rel 0 = LhsToRhs (src lhs, dst rhs)
    //   rel 1 = RhsToRhs (src rhs, dst rhs)
    let mut relation_info = HashMap::new();
    relation_info.insert(
        0,
        ("LhsToRhs".to_string(), "lhs".to_string(), "rhs".to_string()),
    );
    relation_info.insert(
        1,
        ("RhsToRhs".to_string(), "rhs".to_string(), "rhs".to_string()),
    );

    let mut decoders_map = HashMap::new();
    decoders_map.insert(0usize, SortKeyDecoder::new(&temporal_sortkey_meta()));
    decoders_map.insert(1usize, SortKeyDecoder::new(&temporal_sortkey_meta()));

    let mut node_decoders = HashMap::new();
    node_decoders.insert("lhs".to_string(), RowPackDecoder::new(lhs_ctx).unwrap());
    node_decoders.insert("rhs".to_string(), RowPackDecoder::new(rhs_ctx).unwrap());

    let sampler = DiskGraphEngine::new(
        readers,
        relation_info,
        decoders_map,
        node_decoders,
        None,
        diskgraph_core::time::TimeUnit::Seconds,
        Some(42),
        Some(1),
    );

    // One hop, Uniform fanout on both relations (fanout 10 > 1 neighbor, so the
    // single edge from each seed is always kept).
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: [
                (
                    "LhsToRhs".to_string(),
                    FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::Uniform,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    },
                ),
                (
                    "RhsToRhs".to_string(),
                    FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::Uniform,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    },
                ),
            ]
            .into(),
            default_fanout: None,
        }],
    };

    let mut seed_nodes: HashMap<String, Vec<u64>> = HashMap::new();
    seed_nodes.insert("lhs".to_string(), vec![0]); // lhs seed #0 -> seed_index 0
    seed_nodes.insert("rhs".to_string(), vec![0]); // rhs seed #0 -> seed_index 0

    let req = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes,
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: fused,
            swap_edge_direction: false,
        },
    };

    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
        fused_batches: vec![],
    };
    sampler.sample(&req, &mut state, &mut output).unwrap();

    // Resolve type ids (sorted-name order: lhs=0, rhs=1).
    let lhs_id = sampler.node_type_to_id["lhs"];
    let rhs_id = sampler.node_type_to_id["rhs"];

    // The shared rhs node (global id 2) must appear as TWO distinct records in
    // the rhs TypeState: one rooted at the lhs seed, one at the rhs seed. The
    // rhs seed itself (global id 0) is also in this TypeState, so we filter to
    // the shared node only.
    let shared_records: Vec<_> = state.type_states[&rhs_id]
        .nodes
        .iter()
        .filter(|n| n.global_id == shared as u64)
        .collect();

    assert_eq!(
        shared_records.len(),
        2,
        "shared rhs node reached from both an lhs and an rhs seed (both at \
         seed_index 0) must stay as TWO instances, not merge into one \
         (fused={fused}); got {:?}",
        shared_records
    );

    // One instance is rooted at lhs, the other at rhs — proving the dedup key
    // disambiguates by root seed type, not just seed_index.
    let mut roots: Vec<u32> = shared_records.iter().map(|n| n.seed_type_id).collect();
    roots.sort_unstable();
    assert_eq!(
        roots,
        vec![lhs_id as u32, rhs_id as u32],
        "the two shared-node instances must be rooted at the lhs and rhs seed \
         types respectively"
    );

    // Both instances were reached at seed_index 0 (the per-type seed position),
    // confirming this is the exact lhs#0 / rhs#0 collision the key now resolves.
    assert!(
        shared_records.iter().all(|n| n.seed_index == 0),
        "both instances are reached from seed_index 0; seed_type_id is what \
         keeps them distinct"
    );
}

// ---------------------------------------------------------------------------
// Global negative RHS sampling (append_global_negatives helper).
//
// These tests pin the pure-helper contract that `sample_runtime_batch` relies
// on to append M global negative RHS ids per positive as additional sampling
// seeds. The helper is pure (no I/O): temporal validity is supplied as a
// closure so the contract can be exercised without a seed-table-on-disk.
// ---------------------------------------------------------------------------

use crate::engine::append_global_negatives;

/// A validator that always accepts — i.e. non-temporal RHS type.
fn always_valid_validator(_id: u64, _anchor: i64) -> crate::errors::Result<bool> {
    Ok(true)
}

/// A validator-FACTORY that always accepts. Each call returns a fresh validator
/// (the factory shape `append_global_negatives` requires so every parallel
/// row-task owns its own scratch/state). Returns a concrete fn-pointer factory
/// to sidestep the no-nested-`impl Trait` rule.
fn always_valid() -> impl Fn() -> fn(u64, i64) -> crate::errors::Result<bool> + Sync {
    || always_valid_validator
}

/// A single-threaded rayon pool for the helper tests (the real engine passes
/// its per-instance `self.thread_pool`). One thread keeps tests deterministic
/// and cheap while still exercising the parallel code path.
fn test_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap()
}

/// (seed_nodes, seed_ts) pair as `append_global_negatives` consumes it.
type SeedState = (HashMap<String, Vec<u64>>, Option<HashMap<String, Vec<i64>>>);

/// Build a fresh (seed_nodes, seed_ts) pair with B positives for lhs+rhs.
fn make_seed_state(positives_rhs: Vec<u64>, anchors: Option<Vec<i64>>) -> SeedState {
    let b = positives_rhs.len();
    let mut seed_nodes: HashMap<String, Vec<u64>> = HashMap::new();
    seed_nodes.insert("lhs".to_string(), (0..b as u64).collect());
    seed_nodes.insert("rhs".to_string(), positives_rhs);
    let seed_ts = anchors.map(|a| {
        let mut m = HashMap::new();
        m.insert("lhs".to_string(), a.clone());
        m.insert("rhs".to_string(), a);
        m
    });
    (seed_nodes, seed_ts)
}

#[test]
fn test_negatives_lengths_and_positives_preserved() {
    // Pins (a) seed_nodes[rhs] grows to B + B*M and (b) the first B entries
    // (the positives) are left byte-identical. lhs is untouched.
    let b = 3;
    let m = 4;
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![100, 101, 102], Some(vec![10, 20, 30]));
    let positives = seed_nodes["rhs"].clone();
    let lhs_before = seed_nodes["lhs"].clone();
    let pool = test_pool();

    append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(7), 1000, 2000, b, &pool, always_valid(),
    )
    .unwrap();

    assert_eq!(seed_nodes["rhs"].len(), b + b * m);
    assert_eq!(&seed_nodes["rhs"][0..b], positives.as_slice());
    assert_eq!(seed_nodes["lhs"], lhs_before, "lhs must be untouched");
}

#[test]
fn test_negatives_within_range() {
    // Pins (c): every appended negative id lies in [lo, hi].
    let b = 5;
    let m = 3;
    let (lo, hi) = (1000u64, 1010u64);
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![1, 2, 3, 4, 5], Some(vec![10, 10, 10, 10, 10]));
    let pool = test_pool();

    append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(99), lo, hi, b, &pool, always_valid(),
    )
    .unwrap();

    for &id in &seed_nodes["rhs"][b..] {
        assert!((lo..=hi).contains(&id), "negative {id} out of range [{lo},{hi}]");
    }
}

#[test]
fn test_negatives_timestamps_anchor_major() {
    // Pins (d): seed_ts[rhs] stays row-aligned with seed_nodes[rhs], and the
    // appended timestamps replicate each positive's anchor in anchor-major
    // order: [row0's M anchors, row1's M anchors, ...].
    let b = 3;
    let m = 2;
    let anchors = vec![10i64, 20, 30];
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![100, 101, 102], Some(anchors.clone()));
    let pool = test_pool();

    append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(7), 1000, 2000, b, &pool, always_valid(),
    )
    .unwrap();

    let ts = &seed_ts.as_ref().unwrap()["rhs"];
    assert_eq!(ts.len(), seed_nodes["rhs"].len(), "ts must stay 1:1 with ids");
    assert_eq!(&ts[0..b], anchors.as_slice(), "positive anchors preserved");
    // Anchor-major: row i contributes M copies of anchors[i].
    let expected_neg_ts: Vec<i64> =
        (0..b).flat_map(|i| std::iter::repeat_n(anchors[i], m)).collect();
    assert_eq!(&ts[b..], expected_neg_ts.as_slice());
}

#[test]
fn test_negatives_determinism() {
    // Pins (e): same rnd_seed => identical draws; different seed => (very
    // likely) different draws.
    let b = 4;
    let m = 5;
    let pool = test_pool();
    let run = |seed: u64| {
        let (mut sn, mut st) =
            make_seed_state(vec![1, 2, 3, 4], Some(vec![10, 10, 10, 10]));
        append_global_negatives(&mut sn, &mut st, "rhs", m, Some(seed), 0, 1_000_000, b, &pool, always_valid())
            .unwrap();
        sn["rhs"][b..].to_vec()
    };
    assert_eq!(run(42), run(42), "same seed must reproduce draws");
    assert_ne!(run(42), run(43), "different seed should change draws");
}

#[test]
fn test_negatives_none_seed_is_nondeterministic_but_in_range() {
    // Pins the training/prod path: `rnd_seed = None` draws from `thread_rng`, so
    // two runs (very likely) differ — fresh negatives each epoch — while every
    // id stays in the valid [lo, hi] range. A wide range makes a collision
    // between two independent runs astronomically unlikely.
    let b = 8;
    let m = 8;
    let (lo, hi) = (0u64, 1_000_000u64);
    let pool = test_pool();
    let run = || {
        let (mut sn, mut st) = make_seed_state((0..b as u64).collect(), None);
        append_global_negatives(&mut sn, &mut st, "rhs", m, None, lo, hi, b, &pool, always_valid())
            .unwrap();
        sn["rhs"][b..].to_vec()
    };
    let a = run();
    let c = run();
    assert_eq!(a.len(), b * m);
    assert!(a.iter().all(|&id| (lo..=hi).contains(&id)), "negatives in range");
    assert_ne!(a, c, "None seed must draw fresh (non-deterministic) negatives");
}

#[test]
fn test_negatives_noop_when_m_zero() {
    // Pins (f, partial): amount_m == 0 is a no-op.
    let b = 3;
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![100, 101, 102], Some(vec![10, 20, 30]));
    let before_nodes = seed_nodes.clone();
    let before_ts = seed_ts.clone();
    let pool = test_pool();

    append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", 0, Some(7), 1000, 2000, b, &pool, always_valid(),
    )
    .unwrap();

    assert_eq!(seed_nodes, before_nodes);
    assert_eq!(seed_ts, before_ts);
}

#[test]
fn test_negatives_no_timestamps() {
    // When seed_ts is None (non-temporal task), ids still get appended and
    // seed_ts stays None.
    let b = 2;
    let m = 3;
    let (mut seed_nodes, mut seed_ts) = make_seed_state(vec![5, 6], None);
    let pool = test_pool();

    append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(1), 100, 200, b, &pool, always_valid(),
    )
    .unwrap();

    assert_eq!(seed_nodes["rhs"].len(), b + b * m);
    assert!(seed_ts.is_none());
}

#[test]
fn test_negatives_temporal_rejection_via_closure() {
    // Pins the temporal-gating contract: candidates the closure rejects must
    // not be accepted while a valid alternative is reachable within the
    // bounded resample budget. Here we forbid a single id (lo) and assert it
    // never appears among the negatives (range lo..=hi has many valid ids).
    let b = 4;
    let m = 4;
    let (lo, hi) = (1000u64, 1020u64);
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![1, 2, 3, 4], Some(vec![10, 10, 10, 10]));

    let forbidden = lo;
    let pool = test_pool();
    append_global_negatives(
        &mut seed_nodes,
        &mut seed_ts,
        "rhs",
        m,
        Some(7),
        lo,
        hi,
        b,
        &pool,
        || move |id, _anchor| Ok(id != forbidden),
    )
    .unwrap();

    assert!(
        seed_nodes["rhs"][b..].iter().all(|&id| id != forbidden),
        "forbidden candidate must be resampled away"
    );
}

#[test]
fn test_negatives_exhaustion_masks_slot() {
    // When NOTHING is valid, the bounded-resample budget is exhausted: rather
    // than emitting a temporally-invalid negative, every slot is MASKED
    // (neg_mask=false) while still keeping the last-drawn id as a dense
    // placeholder. Pins: (a) the tensor stays dense (length B + B*M, all ids
    // in range); (b) the returned mask has length B*M and is all-false.
    let b = 2;
    let m = 3;
    let (lo, hi) = (1000u64, 1005u64);
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![1, 2], Some(vec![10, 10]));

    let pool = test_pool();
    let mask = append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(7), lo, hi, b, &pool, || |_id, _anchor| Ok(false),
    )
    .unwrap();

    // Dense: ids still present, in range — masked slots keep a placeholder id.
    assert_eq!(seed_nodes["rhs"].len(), b + b * m);
    for &id in &seed_nodes["rhs"][b..] {
        assert!((lo..=hi).contains(&id));
    }
    // Mask: parallel to the appended negatives, all false (none ever valid).
    assert_eq!(mask.len(), b * m);
    assert!(mask.iter().all(|&v| !v), "exhausted slots must be masked false");
}

#[test]
fn test_negatives_mask_all_true_when_all_valid() {
    // Temporal task where every candidate is valid → no slot is ever masked.
    let b = 3;
    let m = 4;
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![100, 101, 102], Some(vec![10, 20, 30]));
    let pool = test_pool();

    let mask = append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(7), 1000, 2000, b, &pool, always_valid(),
    )
    .unwrap();

    assert_eq!(mask.len(), b * m);
    assert!(mask.iter().all(|&v| v), "all-valid draws must all be unmasked");
}

#[test]
fn test_negatives_mask_all_true_when_non_temporal() {
    // Non-temporal RHS (seed_ts == None): the validity loop never runs, so
    // every slot is unconditionally accepted (mask all true).
    let b = 2;
    let m = 3;
    let (mut seed_nodes, mut seed_ts) = make_seed_state(vec![5, 6], None);
    let pool = test_pool();

    let mask = append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", m, Some(1), 100, 200, b, &pool,
        // Even a closure that would reject everything must be ignored when
        // !do_temporal, because the loop is gated on having an anchor.
        || |_id, _anchor| Ok(false),
    )
    .unwrap();

    assert_eq!(mask.len(), b * m);
    assert!(mask.iter().all(|&v| v), "non-temporal slots must all be unmasked");
}

#[test]
fn test_negatives_mask_empty_when_m_zero() {
    // amount_m == 0 → no-op, empty mask.
    let b = 3;
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![100, 101, 102], Some(vec![10, 20, 30]));
    let pool = test_pool();

    let mask = append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", 0, Some(7), 1000, 2000, b, &pool, always_valid(),
    )
    .unwrap();

    assert!(mask.is_empty(), "M=0 must produce an empty mask");
}

#[test]
fn test_negatives_errors_on_b_mismatch() {
    // Defensive: the helper asserts seed_nodes[rhs].len() == B before append.
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state(vec![1, 2, 3], Some(vec![10, 10, 10]));
    let pool = test_pool();
    let res = append_global_negatives(
        &mut seed_nodes, &mut seed_ts, "rhs", 2, Some(7), 0, 100, 99, &pool, always_valid(),
    );
    assert!(res.is_err(), "B mismatch must be rejected");
}

/// Reference sequential implementation of the negative draw, mirroring the
/// pre-parallelization logic exactly (per-(row,m) ChaCha8 seed, per-row memo,
/// bounded resample, mask-on-exhaustion). The parallel helper MUST reproduce
/// this byte-for-byte for a given rnd_seed. Kept local to the test so it can't
/// drift with the production helper signature.
fn sequential_negatives_reference(
    b: usize,
    amount_m: usize,
    rnd_seed: u64,
    lo: u64,
    hi: u64,
    anchors: Option<&[i64]>,
    mut is_valid: impl FnMut(u64, i64) -> crate::errors::Result<bool>,
) -> (Vec<u64>, Vec<bool>) {
    use rand::{Rng, SeedableRng};
    const NEG_RESAMPLE_BUDGET: usize = 5;
    let mut negs = Vec::new();
    let mut mask = Vec::new();
    let mut memo: HashMap<u64, bool> = HashMap::new();
    for i in 0..b {
        let anchor_i = anchors.map(|a| a[i]);
        memo.clear();
        for m in 0..amount_m {
            let rng_seed = rnd_seed
                .wrapping_add(i as u64)
                .wrapping_add((m as u64) << 32);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
            let mut cand = rng.gen_range(lo..=hi);
            let mut accepted = true;
            if let Some(anchor) = anchor_i {
                let mut attempts = 0;
                loop {
                    let valid = *memo
                        .entry(cand)
                        .or_insert_with(|| is_valid(cand, anchor).unwrap());
                    if valid {
                        accepted = true;
                        break;
                    }
                    if attempts >= NEG_RESAMPLE_BUDGET {
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
    }
    (negs, mask)
}

#[test]
fn test_negatives_parallel_matches_sequential_reference() {
    // Equivalence pin: the parallelized helper produces the EXACT same negatives
    // and mask as the (pre-change) sequential logic for a fixed rnd_seed. This
    // is the guard that parallelism didn't change results. We exercise a
    // temporal task with a non-trivial validity predicate that forces resamples
    // (so the resample/memo path is covered), and run the helper on a
    // MULTI-threaded pool to catch any ordering/scratch issues.
    //
    // B is large (128) so the rayon pool actually fans the row loop across all 4
    // worker threads — at small B work-stealing can keep everything on one
    // thread and never exercise the cross-thread collect/ordering. This is the
    // Rust-side determinism + ordering guard at scale called for by the plan.
    let b = 128;
    let m = 6;
    let (lo, hi) = (1000u64, 1050u64);
    let rnd_seed = 12345u64;
    let anchors: Vec<i64> = (0..b as i64).map(|i| 100 + i).collect();

    // A predicate that rejects roughly half the id space (and varies with the
    // anchor) so the resample loop runs and memoization matters.
    let predicate = |id: u64, anchor: i64| -> crate::errors::Result<bool> {
        Ok(!(id.wrapping_add(anchor as u64)).is_multiple_of(3))
    };

    // Sequential reference.
    let (ref_negs, ref_mask) =
        sequential_negatives_reference(b, m, rnd_seed, lo, hi, Some(&anchors), predicate);

    // Parallel helper (multi-threaded pool).
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state((0..b as u64).map(|i| 500 + i).collect(), Some(anchors.clone()));
    let par_mask = append_global_negatives(
        &mut seed_nodes,
        &mut seed_ts,
        "rhs",
        m,
        Some(rnd_seed),
        lo,
        hi,
        b,
        &pool,
        || move |id, anchor| predicate(id, anchor),
    )
    .unwrap();

    let par_negs = seed_nodes["rhs"][b..].to_vec();
    assert_eq!(
        par_negs, ref_negs,
        "parallel negatives must be bit-identical to the sequential reference"
    );
    assert_eq!(
        par_mask, ref_mask,
        "parallel mask must be bit-identical to the sequential reference"
    );
    // And the appended timestamps stay anchor-major / row-aligned.
    let ts = &seed_ts.as_ref().unwrap()["rhs"];
    let expected_neg_ts: Vec<i64> =
        (0..b).flat_map(|i| std::iter::repeat_n(anchors[i], m)).collect();
    assert_eq!(&ts[b..], expected_neg_ts.as_slice());
}

#[test]
fn test_negatives_multithread_pool_no_deadlock_and_ordered() {
    // Exercises the real self.thread_pool shape (multiple threads, many rows,
    // temporal stub) to ensure no deadlock and that row-order collect preserves
    // the anchor-major layout. Each row's M negatives are a contiguous block;
    // we assert the per-row blocks land in row order by re-deriving the expected
    // draws independently and comparing the full flattened output.
    let b = 64;
    let m = 8;
    let (lo, hi) = (10u64, 10u64); // single-id range → every draw is `lo`.
    let rnd_seed = 999u64;
    let anchors: Vec<i64> = vec![42; b];

    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (mut seed_nodes, mut seed_ts) =
        make_seed_state((0..b as u64).collect(), Some(anchors.clone()));
    let mask = append_global_negatives(
        &mut seed_nodes,
        &mut seed_ts,
        "rhs",
        m,
        Some(rnd_seed),
        lo,
        hi,
        b,
        &pool,
        always_valid(),
    )
    .unwrap();

    // With a single-id range, every negative is `lo` and every slot accepted.
    assert_eq!(seed_nodes["rhs"].len(), b + b * m);
    assert!(seed_nodes["rhs"][b..].iter().all(|&id| id == lo));
    assert_eq!(mask.len(), b * m);
    assert!(mask.iter().all(|&v| v));
}

#[test]
fn test_finalize_preserves_anchor_major_seed_prefix() {
    // FFI-contract pin (Step 4): after sampling, `node_ids[rhs]` must be
    // `[B positives, B*M negatives]` anchor-major — the exact slice layout the
    // pybind dict (and Python e2e) rely on to split positives from negatives.
    //
    // Seeds are inserted into `TypeState.nodes` first, in seed order, at hop 0
    // (see RuntimeBatchOutput::finalize, which maps `type_state.nodes` →
    // `global_id` in order). Here we drive `finalize` directly over a
    // SamplingState whose rhs TypeState holds the post-`append_global_negatives`
    // seed set, then assert the prefix ordering survives finalize verbatim.
    use crate::output::{OutputHandler, RuntimeBatchOutput};
    use crate::state::{NodeRecord, SamplingState, TypeState};
    use std::collections::HashSet;

    let b = 3usize;
    let m = 2usize;
    let positives: Vec<u64> = vec![100, 101, 102];
    // Anchor-major negatives: row0's M, then row1's M, then row2's M.
    let negatives: Vec<u64> = vec![900, 901, /*row0*/ 910, 911, /*row1*/ 920, 921 /*row2*/];

    // rhs type id = 1, lhs type id = 0 (lhs left as a single positive set).
    let lhs_type_id = 0usize;
    let rhs_type_id = 1usize;
    let type_id_to_name = vec!["lhs".to_string(), "rhs".to_string()];

    // Build the rhs TypeState: nodes in seed order = positives ++ negatives.
    // seed_index/local_id values are irrelevant to the ordering assertion; we
    // only assert on global_id order, which is what node_ids carries.
    let mk_record = |gid: u64, idx: u32| NodeRecord {
        global_id: gid,
        seed_index: idx,
        local_id: idx,
        seed_type_id: rhs_type_id as u32,
    };
    let mut rhs_state = TypeState::new();
    for (i, &gid) in positives.iter().chain(negatives.iter()).enumerate() {
        rhs_state.nodes.push(mk_record(gid, i as u32));
    }
    let mut lhs_state = TypeState::new();
    for i in 0..b {
        lhs_state.nodes.push(NodeRecord {
            global_id: i as u64,
            seed_index: i as u32,
            local_id: i as u32,
            seed_type_id: lhs_type_id as u32,
        });
    }

    let mut state = SamplingState::new();
    state.type_states.insert(lhs_type_id, lhs_state);
    state.type_states.insert(rhs_type_id, rhs_state);

    let mut output = RuntimeBatchOutput::new(
        "main",
        HashSet::new(),
        HashMap::new(),
        type_id_to_name,
        // No seed namespaces in this fixture — empty set excludes nothing.
        HashSet::new(),
    );
    output.finalize(&state).unwrap();
    let batch = output.into_runtime_batch();
    let rhs_node_ids = &batch.graphs["main"].node_ids["rhs"];

    assert_eq!(
        rhs_node_ids.len(),
        b + b * m,
        "rhs node set must be B + B*M long"
    );
    assert_eq!(
        &rhs_node_ids[0..b],
        positives.as_slice(),
        "first B node_ids must be the positives in seed order"
    );
    assert_eq!(
        &rhs_node_ids[b..b + b * m],
        negatives.as_slice(),
        "next B*M node_ids must be the negatives, anchor-major, in order"
    );
}

type SampleSnapshot = (
    Vec<(usize, u16, u32, u32)>,
    Vec<Vec<(u64, u32, u32, u32)>>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
);

fn random_adjacency(rng: &mut rand_chacha::ChaCha8Rng, n: u8) -> Vec<Vec<(u8, u64)>> {
    use rand::Rng;
    let sk = SortKeyEncoder::new(&temporal_sortkey_meta());
    (0..n)
        .map(|_| {
            let mut e: Vec<(u8, u64)> = (0..rng.gen_range(0..30))
                .map(|_| {
                    (
                        rng.gen_range(0..n),
                        sk.encode_lower(rng.gen_range(0..1_000)),
                    )
                })
                .collect();
            e.sort_by_key(|&(_, k)| std::cmp::Reverse(k));
            e
        })
        .collect()
}

fn per_seed_request(
    rng: &mut rand_chacha::ChaCha8Rng,
    types: &[&str],
    n: u8,
    seeds: usize,
    i: usize,
) -> BatchSamplingRequest {
    use rand::Rng;
    let mode = if i.is_multiple_of(2) {
        SamplingMode::TemporalMostRecent
    } else {
        SamplingMode::Uniform
    };
    let mut req = make_sampling_request_with_mode(vec![], vec![6, 4, 3, 5], mode);
    for layer in &mut req.plan.layers {
        layer.default_fanout = layer.fanouts.drain().next().map(|(_, f)| f);
    }
    req.seed_nodes = types
        .iter()
        .map(|t| {
            (
                t.to_string(),
                (0..seeds).map(|_| rng.gen_range(0..n as u64)).collect(),
            )
        })
        .collect();
    req.seed_timestamps = Some(
        types
            .iter()
            .map(|t| {
                (
                    t.to_string(),
                    (0..seeds).map(|_| rng.gen_range(0..1_000)).collect(),
                )
            })
            .collect(),
    );
    req.seed_timestamp_unit = diskgraph_core::time::TimeUnit::Seconds;
    req.flags.swap_edge_direction = i.is_multiple_of(3);
    req
}

/// Samples every request on both recording paths and asserts identical
/// edges, node records and stats.
fn assert_per_seed_matches_legacy(
    sampler: &mut DiskGraphEngine,
    requests: &[BatchSamplingRequest],
) -> Vec<SampleSnapshot> {
    let run = |sampler: &DiskGraphEngine| -> Vec<SampleSnapshot> {
        let pool = crate::state::StatePool::new();
        requests
            .iter()
            .map(|req| {
                let mut state = pool.acquire();
                let mut output = MockOutput {
                    edges: vec![],
                    feature_batches: vec![],
                    schema: None,
                    fused_batches: vec![],
                };
                sampler.sample(req, &mut state, &mut output).unwrap();
                let mut types: Vec<_> = state.type_states.iter().collect();
                types.sort_by_key(|(t, _)| **t);
                let nodes = types
                    .iter()
                    .map(|(_, ts)| {
                        ts.nodes
                            .iter()
                            .map(|r| (r.global_id, r.seed_index, r.local_id, r.seed_type_id))
                            .collect()
                    })
                    .collect();
                (
                    output.edges,
                    nodes,
                    state.stats.nodes_per_type.clone(),
                    state.stats.edges_per_type.clone(),
                )
            })
            .collect()
    };
    sampler.legacy_recording = false;
    let per_seed = run(sampler);
    sampler.legacy_recording = true;
    let legacy = run(sampler);
    assert_eq!(per_seed, legacy);
    per_seed
}

#[test]
fn per_seed_recording_matches_legacy_recording() {
    use rand::SeedableRng;
    let dir = TempDir::new().unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let n = 41u8;
    let adjacency = random_adjacency(&mut rng, n);
    let (ctx, node_data) = trivial_node_rowpack();
    let (reader, _) = write_adjacency_shard(dir.path(), &adjacency, 16, &node_data);
    let reader = Arc::new(reader);
    let mut sampler = DiskGraphEngine::new(
        [(
            "User".to_string(),
            vec![ShardHandle {
                reader,
                min_id: 0,
                max_id: n as u64 - 1,
            }],
        )]
        .into(),
        [(
            0,
            (
                "Connects".to_string(),
                "User".to_string(),
                "User".to_string(),
            ),
        )]
        .into(),
        [(0usize, SortKeyDecoder::new(&temporal_sortkey_meta()))].into(),
        [("User".to_string(), RowPackDecoder::new(ctx).unwrap())].into(),
        None,
        diskgraph_core::time::TimeUnit::Seconds,
        Some(42),
        Some(4),
    );

    let requests: Vec<BatchSamplingRequest> = (0..6)
        .map(|i| per_seed_request(&mut rng, &["User"], n, [1, 40, 700][i % 3], i))
        .collect();
    let per_seed = assert_per_seed_matches_legacy(&mut sampler, &requests);
    let (edges, nodes) = per_seed
        .iter()
        .fold((0, 0), |(e, v), s| (e + s.0.len(), v + s.1[0].len()));
    assert!(edges > nodes, "the graph must revisit nodes within a seed");
}

#[test]
fn per_seed_recording_matches_legacy_recording_across_seed_types() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12);
    let n = 41u8;
    let (ctx, node_data) = trivial_node_rowpack();
    let dirs = [TempDir::new().unwrap(), TempDir::new().unwrap()];
    let readers: Vec<Arc<ShardReader>> = dirs
        .iter()
        .map(|dir| {
            Arc::new(
                write_adjacency_shard(dir.path(), &random_adjacency(&mut rng, n), 16, &node_data).0,
            )
        })
        .collect();
    let requests: Vec<BatchSamplingRequest> = (0..6)
        .map(|i| per_seed_request(&mut rng, &["A", "B"], n, [3, 40, 700][i % 3], i))
        .collect();
    for threads in [1, 4] {
        let shard = |t: usize| {
            vec![ShardHandle {
                reader: readers[t].clone(),
                min_id: 0,
                max_id: n as u64 - 1,
            }]
        };
        let mut sampler = DiskGraphEngine::new(
            [("A".to_string(), shard(0)), ("B".to_string(), shard(1))].into(),
            [
                (0, ("AToB".to_string(), "A".to_string(), "B".to_string())),
                (1, ("BToA".to_string(), "B".to_string(), "A".to_string())),
            ]
            .into(),
            [
                (0usize, SortKeyDecoder::new(&temporal_sortkey_meta())),
                (1usize, SortKeyDecoder::new(&temporal_sortkey_meta())),
            ]
            .into(),
            [
                ("A".to_string(), RowPackDecoder::new(ctx.clone()).unwrap()),
                ("B".to_string(), RowPackDecoder::new(ctx.clone()).unwrap()),
            ]
            .into(),
            None,
            diskgraph_core::time::TimeUnit::Seconds,
            Some(42),
            Some(threads),
        );
        let per_seed = assert_per_seed_matches_legacy(&mut sampler, &requests);
        assert!(per_seed
            .iter()
            .all(|s| s.1.len() == 2 && s.0.iter().any(|e| e.1 == 1)));
    }
}

#[test]
fn repeated_runs_give_identical_output() {
    let (dir_lhs, dir_rhs) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let (lhs, _, lhs_ctx) = build_temporal_shard(dir_lhs.path(), &[(1, 1), (2, 1)], 2);
    let (rhs, _, rhs_ctx) = build_temporal_shard(dir_rhs.path(), &[(1, 1), (2, 1)], 2);
    let shard = |reader| vec![ShardHandle { reader: Arc::new(reader), min_id: 0, max_id: 2 }];
    let sampler = DiskGraphEngine::new(
        [("lhs".to_string(), shard(lhs)), ("rhs".to_string(), shard(rhs))].into(),
        [
            (0, ("LhsToRhs".to_string(), "lhs".to_string(), "rhs".to_string())),
            (1, ("RhsToRhs".to_string(), "rhs".to_string(), "rhs".to_string())),
        ]
        .into(),
        [
            (0usize, SortKeyDecoder::new(&temporal_sortkey_meta())),
            (1usize, SortKeyDecoder::new(&temporal_sortkey_meta())),
        ]
        .into(),
        [
            ("lhs".to_string(), RowPackDecoder::new(lhs_ctx).unwrap()),
            ("rhs".to_string(), RowPackDecoder::new(rhs_ctx).unwrap()),
        ]
        .into(),
        None,
        diskgraph_core::time::TimeUnit::Seconds,
        Some(42),
        Some(1),
    );
    for fused in [false, true] {
        let mut req = make_sampling_request_with_mode(vec![], vec![10], SamplingMode::Uniform);
        let layer = &mut req.plan.layers[0];
        layer.default_fanout = layer.fanouts.drain().next().map(|(_, f)| f);
        req.seed_nodes = [("lhs".to_string(), vec![0]), ("rhs".to_string(), vec![0])].into();
        req.flags.fused_feature_fetching = fused;
        let runs: Vec<_> = (0..16)
            .map(|_| {
                let mut state = crate::state::StatePool::new().acquire();
                let mut output = MockOutput {
                    edges: vec![],
                    feature_batches: vec![],
                    schema: None,
                    fused_batches: vec![],
                };
                sampler.sample(&req, &mut state, &mut output).unwrap();
                let rhs: Vec<_> = state.type_states[&1].nodes.iter().map(|n| (n.global_id, n.seed_type_id)).collect();
                (output.edges, rhs, output.fused_batches)
            })
            .collect();
        assert_eq!(runs[0].0.len(), 4);
        assert_eq!(runs[0].2.len(), if fused { 2 } else { 0 });
        assert!(runs.iter().all(|r| *r == runs[0]), "fused={fused}");
    }
}
