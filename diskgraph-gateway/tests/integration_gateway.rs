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

        // --- Node Type 1: Users (Global 0..69) ---
        let cols_user = vec![RowPackColumn {
            name: "feat_user".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        }];
        let (ctx_user, _) = Self::create_node_data(cols_user.clone(), vec![ValueRef::Int64(0)]);
        let u_encoder = RowPackEncoder::new(ctx_user.clone()).unwrap();
        let mut u_scratch = RowPackScratch::new(&ctx_user);

        let mut user_edges_blob = Vec::new();
        let mut user_offsets = Vec::new();
        let header_u = OffsetFileHeader::new(70, 71);
        user_offsets.extend_from_slice(&bincode::serialize(&header_u).unwrap());

        let mut u_curr = 0u64;
        let mut u_chunk = Vec::new();
        let mut u_base = 0u64;

        for i in 0..70 {
            let mut encoded = Vec::new();
            u_encoder.encode_row_into(&vec![ValueRef::Int64((i as i64) * 100)], &mut encoded, &mut u_scratch).unwrap();

            let mut record_buf = Vec::new();
            let edge_count = if i == 0 { 5 } else { 0 };
            record_buf.push(edge_count as u8); // Rel 0 (purchased)

            record_buf.extend_from_slice(&(encoded.len() as u16).to_le_bytes());
            if edge_count > 0 {
                for k in 0..5 {
                    let target = 5_000_000_000u64 + k as u64;
                    record_buf.extend_from_slice(&target.to_le_bytes()[..5]);
                }
            }
            record_buf.extend_from_slice(&encoded);

            user_edges_blob.extend_from_slice(&record_buf);
            u_chunk.push(u_curr);
            u_curr += record_buf.len() as u64;
            if u_chunk.len() == 64 {
                Self::write_chunk(&mut user_offsets, 64, u_base, &u_chunk);
                u_chunk.clear(); u_base += 64;
            }
        }
        u_chunk.push(u_curr);
        Self::write_chunk(&mut user_offsets, u_chunk.len() as u16, u_base, &u_chunk);
        Self::write_raw_file(path, "users_shard", EDGES_FILENAME, &user_edges_blob);
        Self::write_raw_file(path, "users_shard", OFFSETS_FILENAME, &user_offsets);

        // --- Node Type 2: Items (Global 5B..5B+69) ---
        let cols_item = vec![RowPackColumn {
            name: "feat_item".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        }];
        let (ctx_item, _) = Self::create_node_data(cols_item.clone(), vec![ValueRef::Float32(0.0)]);
        let i_encoder = RowPackEncoder::new(ctx_item.clone()).unwrap();
        let mut i_scratch = RowPackScratch::new(&ctx_item);

        let mut item_edges_blob = Vec::new();
        let mut item_offsets = Vec::new();
        let header_i = OffsetFileHeader::new(70, 71);
        item_offsets.extend_from_slice(&bincode::serialize(&header_i).unwrap());

        let mut i_curr = 0u64;
        let mut i_chunk = Vec::new();
        let mut i_base = 0u64;

        for i in 0..70 {
            let mut encoded = Vec::new();
            i_encoder.encode_row_into(&vec![ValueRef::Float32(10.0 + i as f32)], &mut encoded, &mut i_scratch).unwrap();

            let mut record_buf = Vec::new();
            let edge_count = if i == 0 { 1 } else { 0 };
            record_buf.push(edge_count as u8); // Rel 1 (viewed)

            record_buf.extend_from_slice(&(encoded.len() as u16).to_le_bytes());
            if edge_count > 0 {
                let target = 1u64; // User 1
                record_buf.extend_from_slice(&target.to_le_bytes()[..5]);
            }
            record_buf.extend_from_slice(&encoded);

            item_edges_blob.extend_from_slice(&record_buf);
            i_chunk.push(i_curr);
            i_curr += record_buf.len() as u64;
            if i_chunk.len() == 64 {
                Self::write_chunk(&mut item_offsets, 64, i_base, &i_chunk);
                i_chunk.clear(); i_base += 64;
            }
        }
        i_chunk.push(i_curr);
        Self::write_chunk(&mut item_offsets, i_chunk.len() as u16, i_base, &i_chunk);
        Self::write_raw_file(path, "items_shard", EDGES_FILENAME, &item_edges_blob);
        Self::write_raw_file(path, "items_shard", OFFSETS_FILENAME, &item_offsets);

        // --- Layouts ---
        let layout_u = RelationLayout {
            relation_id: 0,
            dst_bits: 33, entry_bytes: 5, dst_mask: (1 << 33) - 1,
            dest_sort_key_bits: None, dest_sort_key_mask: None,
        };
        let layout_i = RelationLayout {
            relation_id: 1,
            dst_bits: 33, entry_bytes: 5, dst_mask: (1 << 33) - 1,
            dest_sort_key_bits: None, dest_sort_key_mask: None,
        };

        // --- Readers ---
        let mut readers = HashMap::new();
        readers.insert("users".to_string(), vec![ShardHandle {
            reader: Arc::new(ShardReader::open(&path.join("users_shard"), Arc::new(vec![layout_u]), true).unwrap()),
            min_id: 0, max_id: 69,
        }]);
        readers.insert("items".to_string(), vec![ShardHandle {
            reader: Arc::new(ShardReader::open(&path.join("items_shard"), Arc::new(vec![layout_i]), true).unwrap()),
            min_id: 5_000_000_000, max_id: 5_000_000_069,
        }]);

        // --- Relations ---
        let mut relation_info = HashMap::new();
        relation_info.insert(0, ("purchased".to_string(), "items".to_string()));
        relation_info.insert(1, ("viewed".to_string(), "users".to_string()));

        let mut node_decoders = HashMap::new();
        node_decoders.insert("users".to_string(), RowPackDecoder::new(ctx_user).unwrap());
        node_decoders.insert("items".to_string(), RowPackDecoder::new(ctx_item).unwrap());

        let sampler = DiskGraphEngine::new(readers, relation_info, HashMap::new(), node_decoders);
        Self { _dir: dir, sampler }
    }

    fn create_node_data(cols: Vec<RowPackColumn>, values: Vec<ValueRef>) -> (diskgraph_rowpack::RowPackContext, Vec<u8>) {
        let mut ctx_builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
        ctx_builder.observe_row(&values).unwrap();
        let ctx = ctx_builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut scratch = RowPackScratch::new(&ctx);
        let mut encoded = Vec::new();
        encoder.encode_row_into(&values, &mut encoded, &mut scratch).unwrap();
        (ctx, encoded)
    }

    fn write_chunk(out: &mut Vec<u8>, count: u16, _base_idx: u64, offsets: &[u64]) {
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&offsets[0].to_le_bytes());
        let mut prev = offsets[0];
        for i in 1..offsets.len() {
            out.extend_from_slice(&((offsets[i] - prev) as u16).to_le_bytes());
            prev = offsets[i];
        }
    }

    fn write_raw_file(base: &std::path::Path, shard: &str, file: &str, data: &[u8]) {
        let p = base.join(shard);
        std::fs::create_dir_all(&p).unwrap();
        File::create(p.join(file)).unwrap().write_all(data).unwrap();
    }
}

#[tokio::test]
async fn test_fused_sampling_flow() {
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
                    "fanouts": { "purchased": { "fanout": 5, "mode": "uniform", "absent_key_policy": "uniform_fallback" } },
                    "default_fanout": null
                },
                {
                    "fanouts": { "viewed": { "fanout": 1, "mode": "uniform", "absent_key_policy": "uniform_fallback" } },
                    "default_fanout": null
                }
            ]
        },
        "flags": {
            "replace": false, "directed": true, "disjoint": true, "return_edge_id": false, "adaptive": false,
            "fused_feature_fetching": true
        }
    }"#;

    let response = service.do_get(tonic::Request::new(Ticket { ticket: req_json.into() })).await.unwrap();
    let mut stream = response.into_inner();
    let _schema_msg = stream.next().await.unwrap().unwrap(); // Schema

    let mut tickets = HashMap::new();
    let mut stats_msg = None;
    let mut node_batches = Vec::new();
    let mut edge_batches = Vec::new();

    let main_schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
        arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("meta_val", arrow_schema::DataType::UInt64, true),
    ]));
    let dict_tracker = HashMap::new();

    while let Some(msg) = stream.next().await {
        let msg = msg.expect("main stream error");
        if !msg.app_metadata.is_empty() {
             let meta: HashMap<String, serde_json::Value> = serde_json::from_slice(&msg.app_metadata).unwrap();
             if meta.contains_key("total_nodes") {
                 stats_msg = Some(msg);
                 break;
             } else {
                 // Tickets
                 tickets = meta.into_iter().map(|(k, v)| (k, v.as_str().unwrap().to_string())).collect();
             }
        }

        // Handle Data (Skip empty ticket batch)
        if msg.data_header.is_empty() { continue; }
        let batch = arrow_flight::utils::flight_data_to_arrow_batch(&msg, main_schema.clone(), &dict_tracker)
            .expect("failed to decode main batch");

        if batch.num_rows() == 0 { continue; }

        let row_type = batch.column(0).as_any().downcast_ref::<arrow_array::Int8Array>().unwrap();
        if row_type.value(0) == 1 {
            node_batches.push(batch);
        } else {
            edge_batches.push(batch);
        }
    }

    assert!(!tickets.is_empty(), "Tickets not found");
    assert!(!node_batches.is_empty(), "No nodes returned");
    assert!(!edge_batches.is_empty(), "No edges returned");

    // 1. Check Users (2 nodes: Seed 0, Leaf 1)
    let u_ticket = tickets.get("1").unwrap();
    let u_res = service.do_get(tonic::Request::new(Ticket { ticket: u_ticket.clone().into() })).await.unwrap();
    let mut u_stream = u_res.into_inner();
    let _ = u_stream.next().await; // Schema
    let u_data = u_stream.next().await.unwrap().unwrap();
    let u_batch = arrow_flight::utils::flight_data_to_arrow_batch(&u_data, Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("feat_user", arrow_schema::DataType::Int64, true)])), &HashMap::new()).unwrap();

    assert_eq!(u_batch.num_rows(), 2);
    let u_col = u_batch.column(0).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
    assert_eq!(u_col.value(0), 0);   // User 0 (Seed)
    assert_eq!(u_col.value(1), 100); // User 1 (Leaf)

    // 2. Check Items (5 nodes: Middle Items 0..4)
    let i_ticket = tickets.get("0").unwrap();
    let i_res = service.do_get(tonic::Request::new(Ticket { ticket: i_ticket.clone().into() })).await.unwrap();
    let mut i_stream = i_res.into_inner();
    let _ = i_stream.next().await; // Schema
    let i_data = i_stream.next().await.unwrap().unwrap();
    let i_batch = arrow_flight::utils::flight_data_to_arrow_batch(&i_data, Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("feat_item", arrow_schema::DataType::Float32, true)])), &HashMap::new()).unwrap();

    assert_eq!(i_batch.num_rows(), 5);
    let i_col = i_batch.column(0).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
    for k in 0..5 {
        let val = i_col.value(k);
        let expected = 10.0 + (k as f32);
        assert!((val - expected).abs() < 0.001, "Item {} mismatch", k);
    }

    // 3. Stats
    let stats: diskgraph_engine::state::SamplingStats = serde_json::from_slice(&stats_msg.unwrap().app_metadata).unwrap();
    assert_eq!(stats.total_nodes, 7);
}

#[tokio::test]
async fn test_feature_fetching_flow() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);

    let req_json = r#"{
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "batch": [{ "node_type": "users", "node_ids": [0], "feature_names": ["feat_user"] }]
    }"#;

    let response = service.do_get(tonic::Request::new(Ticket { ticket: req_json.into() })).await.unwrap();
    let mut stream = response.into_inner();
    let _ = stream.next().await; // Schema
    let data = stream.next().await.unwrap().unwrap();
    let batch = arrow_flight::utils::flight_data_to_arrow_batch(&data, Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("feat_user", arrow_schema::DataType::Int64, true)])), &HashMap::new()).unwrap();
    assert_eq!(batch.num_rows(), 1);
}

#[tokio::test]
async fn test_invalid_requests() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);
    assert!(service.do_get(tonic::Request::new(Ticket { ticket: "not json".into() })).await.is_err());
}

#[tokio::test]
async fn test_get_type_name_mapping_action() {
    let graph = TestGraph::new();
    let sampler = Arc::new(graph.sampler);
    let pool = StatePool::new();
    let service = SamplerGateway::new(sampler, pool);
    let action = arrow_flight::Action { r#type: "get_type_name_mapping".to_string(), body: b"".to_vec().into() };
    let response = service.do_action(tonic::Request::new(action)).await.unwrap();
    let result = response.into_inner().next().await.unwrap().unwrap();
    let mapping: diskgraph_engine::engine::TypeNameMapping = serde_json::from_slice(&result.body).unwrap();
    assert!(mapping.node_types.contains(&"users".to_string()));
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
        "plan": { "layers": [] },
        "flags": { "replace": false, "directed": true, "disjoint": false, "return_edge_id": false, "adaptive": false, "fused_feature_fetching": false }
    }"#;
    let response = service.do_get(tonic::Request::new(Ticket { ticket: req_json.into() })).await.unwrap();
    let mut stream = response.into_inner();
    let mut stats_found = false;
    while let Some(msg) = stream.next().await {
        if !msg.unwrap().app_metadata.is_empty() { stats_found = true; break; }
    }
    assert!(stats_found);
}
