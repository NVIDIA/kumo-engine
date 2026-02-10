use diskgraph_gateway::SamplerGateway;
use diskgraph_engine::engine::{DiskGraphEngine, ShardHandle};
use diskgraph_engine::state::StatePool;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use diskgraph_rowpack::{RowPackContextBuilder, RowPackConfig, RowPackColumn, IngestedType, RowPackEncoder, RowPackScratch, ValueRef, RowPackDecoder};
use arrow_flight::{Ticket, flight_service_server::FlightService};
use std::sync::Arc;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tokio_stream::StreamExt;
use tempfile::TempDir;

struct TestGraph {
    _dir: TempDir,
    sampler: DiskGraphEngine,
}

impl TestGraph {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // 1. Setup RowPack Context & Data
        let cols = vec![
            RowPackColumn { name: "feat_0".into(), logical_type: IngestedType::Int64, fixed_len: None, precision: None, scale: None },
        ];
        let mut ctx_builder = RowPackContextBuilder::new(cols.clone(), RowPackConfig::default());
        let row_val = vec![ValueRef::Int64(123)];
        ctx_builder.observe_row(&row_val).unwrap();
        let ctx = ctx_builder.finish().unwrap();

        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut scratch = RowPackScratch::new(&ctx);
        let mut encoded_node_data = Vec::new();
        encoder.encode_row_into(&row_val, &mut encoded_node_data, &mut scratch).unwrap();

        // 2. Define Layout
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dest_sort_key_bits: None,
            entry_bytes: 1,
            dst_mask: 0xFF,
            dest_sort_key_mask: None,
        };

        // 3. Write Edges.bin (Count 0 + u16 Data Len + Node Data)
        let mut f = File::create(path.join(EDGES_FILENAME)).unwrap();
        // Node 0: 0 edges (0x00)
        f.write_all(&[0x00]).unwrap();
        // Node data len (u16 LE)
        f.write_all(&(encoded_node_data.len() as u16).to_le_bytes()).unwrap();
        f.write_all(&encoded_node_data).unwrap();

        // 4. Write Offsets.bin
        let mut offsets = Vec::new();
        let header = OffsetFileHeader::new(1, 2); // 1 node, 2 offsets
        offsets.extend_from_slice(&bincode::serialize(&header).unwrap());
        offsets.extend_from_slice(&2u16.to_le_bytes()); // Chunk count = 2
        offsets.extend_from_slice(&0u64.to_le_bytes()); // offset[0] = 0
        let end_offset = (1 + 2 + encoded_node_data.len()) as u16;
        offsets.extend_from_slice(&end_offset.to_le_bytes());
        let mut f = File::create(path.join(OFFSETS_FILENAME)).unwrap();
        f.write_all(&offsets).unwrap();

        // 5. Setup Engine
        let reader = ShardReader::open(path, Arc::new(vec![layout]), true).unwrap();

        let mut readers = HashMap::new();
        readers.insert("users".to_string(), vec![ShardHandle {
            reader: Arc::new(reader),
            min_id: 0,
            max_id: 0,
        }]);

        let mut node_decoders = HashMap::new();
        node_decoders.insert("users".to_string(), RowPackDecoder::new(ctx.clone()).unwrap());

        let sampler = DiskGraphEngine::new(readers, HashMap::new(), HashMap::new(), node_decoders);

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

    // 2. Data
    while let Some(msg_result) = stream.next().await {
        let _msg = msg_result.expect("Stream error");
        // Sampling verification already tested, here we just ensure it finishes.
    }
}

#[tokio::test]
async fn test_feature_fetching_flow() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);

    let req_json = r#"{
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "batch": [
            {
                "node_type": "users",
                "node_ids": [0],
                "feature_names": ["feat_0"]
            }
        ]
    }"#;

    let ticket = Ticket { ticket: req_json.into() };
    let response = service.do_get(tonic::Request::new(ticket)).await.expect("do_get failed");
    let mut stream = response.into_inner();

    // 1. Schema
    let schema_msg = stream.next().await.expect("Stream empty").expect("Error in stream");
    assert!(!schema_msg.data_header.is_empty());
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("feat_0", arrow_schema::DataType::Int64, true),
    ]));

    // 2. Data Batch
    let data_msg = stream.next().await.expect("No data batch").expect("Error in stream");
    let dict_tracker = HashMap::new();
    let batch = arrow_flight::utils::flight_data_to_arrow_batch(
        &data_msg,
        schema,
        &dict_tracker
    ).expect("Failed to decode batch");

    assert_eq!(batch.num_rows(), 1);
    let col = batch.column(0).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
    assert_eq!(col.value(0), 123);
}

#[tokio::test]
async fn test_get_schema_mapping_action() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);

    let action = arrow_flight::Action {
        r#type: "get_schema_mapping".to_string(),
        body: vec![].into(),
    };
    let response = service.do_action(tonic::Request::new(action)).await.expect("do_action failed");
    let mut stream = response.into_inner();
    let res = stream.next().await.expect("No result").expect("Action error");

    let mapping: diskgraph_engine::engine::SchemaMapping = serde_json::from_slice(&res.body).expect("Failed to deserialize mapping");
    assert_eq!(mapping.node_types, vec!["users"]);
}
