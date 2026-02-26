use crate::engine::{DiskGraphEngine, ShardHandle};
use crate::output::OutputHandler;
use crate::state::SamplingState;
use diskgraph_core::dataset::DatasetId;
use diskgraph_core::sampling::*;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use std::collections::HashMap;
use std::fs::File;
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
        // Node 0: 2 edges (0x02), Len, Edges[1, 2], Data
        f.write_all(&[0x02]).unwrap();
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
            .unwrap();
        f.write_all(&[0x01, 0x02]).unwrap();
        f.write_all(&encoded_node_data).unwrap();

        // Node 1: 0 edges, Len, Data
        f.write_all(&[0x00]).unwrap();
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
            .unwrap();
        f.write_all(&encoded_node_data).unwrap();

        // Node 2: 0 edges, Len, Data
        f.write_all(&[0x00]).unwrap();
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
            .unwrap();
        f.write_all(&encoded_node_data).unwrap();
        f.flush().unwrap(); // Ensure written
        let edges_len = f.metadata().unwrap().len();

        // 4. Write Offsets.bin
        let mut offsets = Vec::new();
        // Header: count=3 nodes -> 4 offsets
        let header = OffsetFileHeader::new(3, 4);
        let header_bytes = bincode::serialize(&header).unwrap();
        offsets.extend_from_slice(&header_bytes);

        offsets.extend_from_slice(&4u16.to_le_bytes()); // Chunk count = 4
        offsets.extend_from_slice(&0u64.to_le_bytes()); // offset[0] = 0

        // offset[0] = 0 (Base)
        // offset[1] = 0 + node0_len
        let node0_len = 1 + 2 + 2 + encoded_node_data.len();
        let delta1 = node0_len as u16;
        offsets.extend_from_slice(&delta1.to_le_bytes());

        // offset[2] = offset[1] + node1_len
        let node1_len = 1 + 2 + encoded_node_data.len();
        let delta2 = node1_len as u16;
        offsets.extend_from_slice(&delta2.to_le_bytes());

        // offset[3] = offset[2] + node2_len
        let node2_len = 1 + 2 + encoded_node_data.len();
        let delta3 = node2_len as u16;
        offsets.extend_from_slice(&delta3.to_le_bytes());

        let expected_size = (node0_len + node1_len + node2_len) as u64;
        assert_eq!(expected_size, edges_len, "Offsets sum must match file size");

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
                max_id: 2,
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

    // Fetch unknown column
    let result =
        graph
            .sampler
            .fetch_features_for_type(0, &[0], &["unknown_col".to_string()], &mut output);

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("unknown feature"));
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

    assert_eq!(output.edges.len(), 2);
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
            fused_feature_fetching: true, // Enable Fused
            swap_edge_direction: false,
        },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    assert_eq!(output.edges.len(), 2);

    // Check fused batches
    // We expect at least one batch for type "User" (ID 0)
    assert!(!output.fused_batches.is_empty());
    let (type_id, batch) = &output.fused_batches[0];
    assert_eq!(*type_id, 0);

    // Batch should contain seed (0) and its neighbors (1, 2) = 3 rows
    assert_eq!(batch.num_rows(), 3);
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
