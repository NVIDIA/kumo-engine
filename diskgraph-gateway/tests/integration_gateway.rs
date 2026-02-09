use diskgraph_gateway::SamplerGateway;
use diskgraph_sampler::sampler::{Sampler, ShardHandle};
use diskgraph_sampler::state::StatePool;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use arrow_flight::{Ticket, flight_service_server::FlightService};
use std::sync::Arc;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tokio_stream::StreamExt;
use tempfile::TempDir;

struct TestGraph {
    _dir: TempDir,
    sampler: Sampler,
}

impl TestGraph {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // 1. Define Layout: RelID=0, Dst=1 byte
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dest_sort_key_bits: None,
            entry_bytes: 1,
            dst_mask: 0xFF,
            dest_sort_key_mask: None,
        };

        // 2. Write Edges.bin
        // Node 0 has 2 neighbors: 1 (0x01) and 2 (0x02)
        // Count 2 encoded as 0x02
        let mut f = File::create(path.join(EDGES_FILENAME)).unwrap();
        f.write_all(&[0x02, 0x01, 0x02]).unwrap();
        
        // 3. Write Offsets.bin
        let mut offsets = Vec::new();
        let header = OffsetFileHeader::new(1, 2); 
        let header_bytes = bincode::serialize(&header).unwrap();
        offsets.extend_from_slice(&header_bytes);
        
        offsets.extend_from_slice(&2u16.to_le_bytes());
        offsets.extend_from_slice(&0u64.to_le_bytes()); 
        offsets.extend_from_slice(&3u16.to_le_bytes()); 

        let mut f = File::create(path.join(OFFSETS_FILENAME)).unwrap();
        f.write_all(&offsets).unwrap();

        // 4. Create Reader
        let reader = ShardReader::open(path, Arc::new(vec![layout]), false).unwrap();
        
        let mut readers = HashMap::new();
        readers.insert("users".to_string(), vec![ShardHandle {
            reader: Arc::new(reader),
            min_id: 0,
            max_id: 10,
        }]);

        let mut relation_info = HashMap::new();
        relation_info.insert(0, ("connects".to_string(), "users".to_string()));

        let sampler = Sampler::new(readers, relation_info, HashMap::new());
        
        Self { _dir: dir, sampler }
    }
}

#[tokio::test]
async fn test_gateway_end_to_end_flow() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);

    let req_json = r#"{
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "seed_nodes": { "users": [0] },
        "seed_timestamps": null,
        "plan": {
            "layers": [
                {
                    "fanouts": {}, 
                    "default_fanout": { "fanout": 10, "mode": "uniform", "absent_key_policy": "uniform_fallback" }
                }
            ]
        },
        "flags": {
            "replace": false, "directed": true, "disjoint": true, "return_edge_id": false, "adaptive": false
        }
    }"#;

    let ticket = Ticket { ticket: req_json.into() };

    let response = service.do_get(tonic::Request::new(ticket)).await.expect("do_get failed");
    let mut stream = response.into_inner();

    // 1. Schema
    let _schema_msg = stream.next().await.expect("Stream empty").expect("Error in stream");
    // Define the schema used by ArrowFlightOutput
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
        arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("meta_val", arrow_schema::DataType::UInt64, true),
    ]));
    
    // 2. Data Collection
    let mut stats: Option<diskgraph_sampler::state::SamplingStats> = None;
    let mut captured_edges = Vec::new();
    let mut captured_nodes = Vec::new();

    let dictionaries_by_id = HashMap::new();

    while let Some(msg_result) = stream.next().await {
        let msg = msg_result.expect("Stream error");
        
        if !msg.app_metadata.is_empty() {
            stats = Some(serde_json::from_slice(&msg.app_metadata).expect("Failed to deserialize stats"));
        } else if !msg.data_body.is_empty() {
            let batch = arrow_flight::utils::flight_data_to_arrow_batch(
                &msg,
                schema.clone(),
                &dictionaries_by_id
            ).expect("Failed to decode batch");

            let row_type_arr = batch.column(0).as_any().downcast_ref::<arrow_array::Int8Array>().expect("row_type missing");
            let id_a_arr = batch.column(1).as_any().downcast_ref::<arrow_array::UInt64Array>().expect("id_a missing");
            let id_b_arr = batch.column(2).as_any().downcast_ref::<arrow_array::UInt64Array>().expect("id_b missing");

            for i in 0..batch.num_rows() {
                if row_type_arr.value(i) == 0 {
                    captured_edges.push((id_a_arr.value(i), id_b_arr.value(i)));
                } else {
                    captured_nodes.push(id_b_arr.value(i));
                }
            }
        }
    }

    // 3. Final Assertions
    let stats = stats.expect("Should receive stats metadata");
    assert_eq!(stats.total_edges, 2);
    assert_eq!(stats.total_nodes, 3); // 0 (seed) + 1, 2 (sampled)

    captured_edges.sort();
    assert_eq!(captured_edges, vec![(0, 1), (0, 2)], "Edge content mismatch");

    captured_nodes.sort();
    assert_eq!(captured_nodes, vec![0, 1, 2], "Node content mismatch");
}
