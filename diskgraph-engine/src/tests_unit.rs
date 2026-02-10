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

        // 3. Write Edges.bin (Count 2 + u16 Data Len + 2 Edges + Node Data)
        let mut f = File::create(path.join(EDGES_FILENAME)).unwrap();
        // Node 0: 2 edges (0x02)
        f.write_all(&[0x02]).unwrap();
        // Node data len (u16 LE)
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes())
            .unwrap();
        // Edges: dst=1, dst=2
        f.write_all(&[0x01, 0x02]).unwrap();
        f.write_all(&encoded_node_data).unwrap();

        // 4. Write Offsets.bin
        let mut offsets = Vec::new();
        // total_offsets = 2 (start of node 0, start of node 1/end of shard)
        let header = OffsetFileHeader::new(1, 2);
        let header_bytes = bincode::serialize(&header).unwrap();
        offsets.extend_from_slice(&header_bytes);

        offsets.extend_from_slice(&2u16.to_le_bytes()); // 2 offsets in this chunk
        offsets.extend_from_slice(&0u64.to_le_bytes()); // offset[0] = 0
                                                        // offset[1] = 1 (count byte) + 2 (len bytes) + 2 (edge bytes) + data len
        let end_offset = (1 + 2 + 2 + encoded_node_data.len()) as u16;
        offsets.extend_from_slice(&end_offset.to_le_bytes());

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
                max_id: 0,
            }],
        );

        let mut relation_info = HashMap::new();
        relation_info.insert(0, ("Connects".to_string(), "User".to_string()));

        let mut node_decoders = HashMap::new();
        node_decoders.insert(
            "User".to_string(),
            RowPackDecoder::new(ctx.clone()).unwrap(),
        );

        let sampler = DiskGraphEngine::new(readers, relation_info, HashMap::new(), node_decoders);

        Self {
            _dir: dir,
            sampler,
            rowpack_ctx: ctx,
        }
    }
}

#[test]
fn test_fetch_features() {
    let graph = TestGraph::new();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
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
    };
    graph
        .sampler
        .fetch_features_for_type(0, &[0], &[], &mut output_all)
        .unwrap();
    let batch_all = &output_all.feature_batches[0];
    assert_eq!(batch_all.num_columns(), 2);
}

struct MockOutput {
    edges: Vec<(usize, u16, u32, u32)>,
    feature_batches: Vec<arrow_array::RecordBatch>,
    schema: Option<arrow_schema::SchemaRef>,
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

    fn finalize(&mut self, _state: &SamplingState) -> crate::errors::Result<()> {
        Ok(())
    }
}

#[test]
fn test_basic_sampling() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput {
        edges: vec![],
        feature_batches: vec![],
        schema: None,
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
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
        },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    assert_eq!(output.edges.len(), 2);
    // Node 0 (Local 0) -> 1 (Local 1) and 2 (Local 2)
}
