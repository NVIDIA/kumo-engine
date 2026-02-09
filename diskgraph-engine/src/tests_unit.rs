use crate::engine::{DiskGraphEngine, ShardHandle};
use crate::state::SamplingState;
use crate::output::OutputHandler;
use diskgraph_core::sampling::*;
use diskgraph_core::dataset::DatasetId;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

struct TestGraph {
    _dir: TempDir,
    sampler: DiskGraphEngine,
}

impl TestGraph {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // layout: rel_id=0, dst_bits=8 (1 byte), no sort key, entry=1 byte
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dest_sort_key_bits: None,
            entry_bytes: 1,
            dst_mask: 0xFF,
            dest_sort_key_mask: None,
        };

        // Create artifacts for 1 node with 2 edges
        // Node 0 -> [1, 2]
        // Edges: dst=1 (0x01), dst=2 (0x02)
        let mut f = File::create(path.join(EDGES_FILENAME)).unwrap();
        // Node 0 edges: count=2 (0x02), data=[0x01, 0x02]
        f.write_all(&[0x02, 0x01, 0x02]).unwrap();
        
        // Offsets:
        // Header (32 bytes)
        // Chunk 0 (count=2, base=0)
        let mut offsets = Vec::new();
        let header = OffsetFileHeader::new(1, 2);
        let header_bytes = bincode::serialize(&header).unwrap();
        offsets.extend_from_slice(&header_bytes);
        
        // Chunk: count=2 (u16), base=0 (u64)
        offsets.extend_from_slice(&2u16.to_le_bytes());
        offsets.extend_from_slice(&0u64.to_le_bytes()); // Base offset = 0
        
        // Offset 1: delta=3 (relative to base) -> 3
        offsets.extend_from_slice(&3u16.to_le_bytes()); 

        let mut f = File::create(path.join(OFFSETS_FILENAME)).unwrap();
        f.write_all(&offsets).unwrap();

        let reader = ShardReader::open(path, Arc::new(vec![layout]), false).unwrap();
        
        let mut readers = HashMap::new();
        readers.insert("User".to_string(), vec![ShardHandle {
            reader: Arc::new(reader),
            min_id: 0,
            max_id: 10,
        }]);

        let mut relation_info = HashMap::new();
        relation_info.insert(0, ("Connects".to_string(), "User".to_string()));

        let sampler = DiskGraphEngine::new(readers, relation_info, HashMap::new(), HashMap::new());
        
        Self { _dir: dir, sampler }
    }
}

struct MockOutput {
    edges: Vec<(usize, u16, u32, u32)>,
}
impl OutputHandler for MockOutput {
    fn handle_edge(&mut self, hop: usize, rel_id: u16, src: u32, dst: u32) -> crate::errors::Result<()> {
        self.edges.push((hop, rel_id, src, dst));
        Ok(())
    }
    fn finalize(&mut self, _state: &SamplingState) -> crate::errors::Result<()> { Ok(()) }
}

#[test]
fn test_basic_sampling() {
    let graph = TestGraph::new();
    let mut state = crate::state::StatePool::new().acquire();
    let mut output = MockOutput { edges: vec![] };

    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: [("Connects".to_string(), FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                })].into(),
                default_fanout: None,
            }
        ]
    };

    let req = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes: [("User".to_string(), vec![0])].into(),
        seed_timestamps: None,
        plan,
        flags: SamplingFlags { replace: false, directed: true, disjoint: true, return_edge_id: false, adaptive: false },
    };

    graph.sampler.sample(&req, &mut state, &mut output).unwrap();

    assert_eq!(output.edges.len(), 2);
    // Node 0 (Local 0) -> 1 (Local 1) and 2 (Local 2)
    // Order depends on reservoir/scan, but for small k, likely sequential if Uniform logic falls back to take-all.
}