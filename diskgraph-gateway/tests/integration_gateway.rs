use arrow_flight::{flight_service_server::FlightService, Ticket};
use diskgraph_engine::engine::{DiskGraphEngine, ShardHandle};
use diskgraph_engine::state::StatePool;
use diskgraph_gateway::SamplerGateway;
use diskgraph_index::layout::{OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_index::reader::ShardReader;
use diskgraph_rowpack::{
    IngestedType, RowPackColumn, RowPackConfig, RowPackContextBuilder, RowPackDecoder,
    RowPackEncoder, RowPackScratch, ValueRef,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;
use tokio_stream::StreamExt;

struct TestGraph {
    _dir: TempDir,
    sampler: DiskGraphEngine,
}

impl TestGraph {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path();

        // --- Node Type 1: Users ---
        let cols_user = vec![RowPackColumn {
            name: "feat_user".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        }];
        let (ctx_user, encoded_user) =
            Self::create_node_data(cols_user, vec![ValueRef::Int64(100)]);
        Self::write_shard_files(path, "users_shard", &encoded_user);

        // --- Node Type 2: Items ---
        let cols_item = vec![RowPackColumn {
            name: "feat_item".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        }];
        let (ctx_item, encoded_item) =
            Self::create_node_data(cols_item, vec![ValueRef::Float32(99.9)]);
        Self::write_shard_files(path, "items_shard", &encoded_item);

        // --- Layout ---
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dest_sort_key_bits: None,
            entry_bytes: 1,
            dst_mask: 0xFF,
            dest_sort_key_mask: None,
        };

        // --- Readers ---
        let mut readers = HashMap::new();

        let reader_user = ShardReader::open(
            &path.join("users_shard"),
            Arc::new(vec![layout.clone()]),
            true,
        )
        .unwrap();
        readers.insert(
            "users".to_string(),
            vec![ShardHandle {
                reader: Arc::new(reader_user),
                min_id: 0,
                max_id: 0,
            }],
        );

        let reader_item =
            ShardReader::open(&path.join("items_shard"), Arc::new(vec![layout]), true).unwrap();
        readers.insert(
            "items".to_string(),
            vec![ShardHandle {
                reader: Arc::new(reader_item),
                min_id: 0,
                max_id: 0,
            }],
        );

        // --- Decoders ---
        let mut node_decoders = HashMap::new();
        node_decoders.insert("users".to_string(), RowPackDecoder::new(ctx_user).unwrap());
        node_decoders.insert("items".to_string(), RowPackDecoder::new(ctx_item).unwrap());

        let sampler = DiskGraphEngine::new(readers, HashMap::new(), HashMap::new(), node_decoders);

        Self { _dir: dir, sampler }
    }

    fn create_node_data(
        cols: Vec<RowPackColumn>,
        values: Vec<ValueRef>,
    ) -> (diskgraph_rowpack::RowPackContext, Vec<u8>) {
        let mut ctx_builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
        ctx_builder.observe_row(&values).unwrap();
        let ctx = ctx_builder.finish().unwrap();

        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut scratch = RowPackScratch::new(&ctx);
        let mut encoded = Vec::new();
        encoder
            .encode_row_into(&values, &mut encoded, &mut scratch)
            .unwrap();
        (ctx, encoded)
    }

    fn write_shard_files(base_path: &std::path::Path, shard_name: &str, node_data: &[u8]) {
        let shard_path = base_path.join(shard_name);
        std::fs::create_dir_all(&shard_path).unwrap();

        // Edges.bin
        let mut f = File::create(shard_path.join(EDGES_FILENAME)).unwrap();
        f.write_all(&[0x00]).unwrap(); // 0 edges
        f.write_all(&(node_data.len() as u16).to_le_bytes())
            .unwrap(); // len
        f.write_all(node_data).unwrap();

        // Offsets.bin
        let mut offsets = Vec::new();
        let header = OffsetFileHeader::new(1, 2);
        offsets.extend_from_slice(&bincode::serialize(&header).unwrap());
        offsets.extend_from_slice(&2u16.to_le_bytes()); // Count 2
        offsets.extend_from_slice(&0u64.to_le_bytes());
        let end_offset = (1 + 2 + node_data.len()) as u16;
        offsets.extend_from_slice(&end_offset.to_le_bytes());
        let mut f = File::create(shard_path.join(OFFSETS_FILENAME)).unwrap();
        f.write_all(&offsets).unwrap();
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
                "feature_names": ["feat_user"]
            },
            {
                "node_type": "items",
                "node_ids": [0],
                "feature_names": ["feat_item"]
            }
        ]
    }"#;

    let ticket = Ticket {
        ticket: req_json.into(),
    };
    let response = service
        .do_get(tonic::Request::new(ticket))
        .await
        .expect("do_get failed");
    let mut stream = response.into_inner();

    // --- Message 1: Schema for Users ---
    let schema_msg_1 = stream
        .next()
        .await
        .expect("Stream empty")
        .expect("Error in stream");
    assert!(!schema_msg_1.data_header.is_empty());
    let schema_1 = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "feat_user",
        arrow_schema::DataType::Int64,
        true,
    )]));

    // --- Message 2: Data Batch for Users ---
    let data_msg_1 = stream
        .next()
        .await
        .expect("No data batch")
        .expect("Error in stream");
    let dict_tracker = HashMap::new();
    let batch_1 =
        arrow_flight::utils::flight_data_to_arrow_batch(&data_msg_1, schema_1, &dict_tracker)
            .expect("Failed to decode user batch");

    assert_eq!(batch_1.num_rows(), 1);
    let col = batch_1
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 100);

    // --- Message 3: Schema for Items ---
    let schema_msg_2 = stream
        .next()
        .await
        .expect("Stream empty")
        .expect("Error in stream");
    assert!(!schema_msg_2.data_header.is_empty());
    let schema_2 = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "feat_item",
        arrow_schema::DataType::Float32,
        true,
    )]));

    // --- Message 4: Data Batch for Items ---
    let data_msg_2 = stream
        .next()
        .await
        .expect("No data batch")
        .expect("Error in stream");
    let batch_2 =
        arrow_flight::utils::flight_data_to_arrow_batch(&data_msg_2, schema_2, &dict_tracker)
            .expect("Failed to decode item batch");

    assert_eq!(batch_2.num_rows(), 1);
    let col = batch_2
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .unwrap();
    assert_eq!(col.value(0), 99.9);
}

#[tokio::test]
async fn test_gateway_end_to_end_flow() {
    let graph = TestGraph::new();
    // ... rest of test logic remains similar but simplified since TestGraph changed ...
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

    let ticket = Ticket {
        ticket: req_json.into(),
    };
    let response = service
        .do_get(tonic::Request::new(ticket))
        .await
        .expect("do_get failed");
    let mut stream = response.into_inner();

    // 1. Schema
    let _schema_msg = stream
        .next()
        .await
        .expect("Stream empty")
        .expect("Error in stream");

    // 2. Data
    while let Some(msg_result) = stream.next().await {
        let _msg = msg_result.expect("Stream error");
    }
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
    let response = service
        .do_action(tonic::Request::new(action))
        .await
        .expect("do_action failed");
    let mut stream = response.into_inner();
    let res = stream
        .next()
        .await
        .expect("No result")
        .expect("Action error");

    let mapping: diskgraph_engine::engine::SchemaMapping =
        serde_json::from_slice(&res.body).expect("Failed to deserialize mapping");
    assert!(mapping.node_types.contains(&"users".to_string()));
    assert!(mapping.node_types.contains(&"items".to_string()));
}

#[tokio::test]
async fn test_invalid_requests() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);

    // 1. Unknown Node Type
    let req_unknown_type = r#"{
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "batch": [
            {
                "node_type": "unknown_table",
                "node_ids": [0],
                "feature_names": ["*"]
            }
        ]
    }"#;
    let ticket = Ticket {
        ticket: req_unknown_type.into(),
    };
    let response = service.do_get(tonic::Request::new(ticket)).await;

    // Gateway streams results, so do_get succeeds, but the stream should contain an error.
    let mut stream = response.expect("do_get failed").into_inner();
    let msg = stream.next().await;

    // The error is sent via the stream if possible, or do_get returns error if immediate.
    // In handle_features, we send Err via channel.
    // So `msg` should be `Some(Err(Status))`
    assert!(msg.is_some());
    let err = msg.unwrap().unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(err.message().contains("Unknown node type"));

    // 2. Malformed JSON
    let req_malformed = r#"{ "dataset_id": "bad json"#;
    let ticket_bad = Ticket {
        ticket: req_malformed.into(),
    };
    let response_bad = service.do_get(tonic::Request::new(ticket_bad)).await;
    // This should fail deserialization immediately
    assert!(response_bad.is_err());
    assert_eq!(
        response_bad.err().unwrap().code(),
        tonic::Code::InvalidArgument
    );
}
