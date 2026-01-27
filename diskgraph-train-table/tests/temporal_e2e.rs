use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeSortKeyMetadata, EdgeSortKeyTimestampEncoding,
    EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata, NodeShardMetadata, NodeTypeMetadata,
};
use diskgraph_index::builder::compute_relation_layouts;
use diskgraph_index::layout::OffsetFileHeader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::tempdir;

use diskgraph_train_table::config::{JobConfig, QueryOptions, SplitConfig, SplitType, TemporalSampling};
use diskgraph_train_table::pipeline::run_job;
use diskgraph_rowpack::{ColumnContext, ColumnEncoding, FloatStorage, IngestedType, RowPackContext, RowPackEncoder, RowPackScratch, ValueRef};

#[test]
fn temporal_count_train_table_e2e() {
    let dir = tempdir().unwrap();
    let index_root = dir.path().join("index");
    fs::create_dir_all(&index_root).unwrap();

    let metadata = build_metadata_count();
    fs::write(
        index_root.join("index_metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();

    write_edge_shard(
        &index_root,
        "users",
        0,
        &metadata,
        vec![
            vec![(1u64, 0i64), (2u64, 100i64), (1u64, 200i64)],
            vec![],
        ],
    );
    write_empty_shard(&index_root, "articles", 0, 2);

    {
        let graph = diskgraph_index::graph::GraphIndexReader::open(
            &index_root.join("index_metadata.json"),
            &index_root,
        )
        .unwrap();
        let table = graph.table("users").unwrap();
        let view = table.edges_for_node(0).unwrap();
        let segs = view.relation_segments().unwrap();
        assert_eq!(segs[0].count, 3);
    }

    let output_path = dir.path().join("train.parquet");
    let config = JobConfig {
        job_name: None,
        index_path: index_root.to_string_lossy().to_string(),
        query: "PREDICT COUNT(transactions.*, 0, 100, seconds) FOR EACH users.user_id".to_string(),
        output_parquet: output_path.to_string_lossy().to_string(),
        split: SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![1.0, 0.0, 0.0],
            seed: Some(0),
            time: None,
            range: None,
        },
        options: QueryOptions {
            temporal_sampling: TemporalSampling::Timeframe,
            ..Default::default()
        },
        validation_output: None,
    };

    run_job(&config).unwrap();

    let file = File::open(&output_path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(batch.num_rows() > 0);

    let entity = batch.column(0).as_any().downcast_ref::<arrow_array::UInt64Array>().unwrap();
    let target = batch.column(1).as_any().downcast_ref::<arrow_array::UInt64Array>().unwrap();
    let timestamp = batch.column(2).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();

    for row in 0..batch.num_rows() {
        let entity_id = entity.value(row);
        let ts = timestamp.value(row);
        let count = target.value(row);
        let expected = count_edges_in_window(entity_id, ts, 0, 100);
        assert_eq!(count as i64, expected);
    }
}

#[test]
fn temporal_sum_with_filter_e2e() {
    let dir = tempdir().unwrap();
    let index_root = dir.path().join("index");
    fs::create_dir_all(&index_root).unwrap();

    let metadata = build_metadata_sum();
    fs::write(
        index_root.join("index_metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();

    write_edge_shard(
        &index_root,
        "users",
        0,
        &metadata,
        vec![vec![(0u64, 0i64), (1u64, 100i64), (2u64, 200i64)]],
    );

    let txn_ctx = build_transactions_ctx();
    write_rowpack_ctx(&index_root.join("nodes/transactions/node_data.ctx"), &txn_ctx);
    write_node_data_shard(
        &index_root,
        "transactions",
        0,
        &txn_ctx,
        vec![
            vec![ValueRef::Float64(5.0), ValueRef::Int64(1)],
            vec![ValueRef::Float64(7.0), ValueRef::Int64(0)],
            vec![ValueRef::Float64(11.0), ValueRef::Int64(1)],
        ],
    );

    {
        let graph = diskgraph_index::graph::GraphIndexReader::open(
            &index_root.join("index_metadata.json"),
            &index_root,
        )
        .unwrap();
        let table = graph.table("users").unwrap();
        let view = table.edges_for_node(0).unwrap();
        let segs = view.relation_segments().unwrap();
        assert_eq!(segs[0].count, 3);
        let txn_table = graph.table("transactions").unwrap();
        let view = txn_table.edges_for_node(0).unwrap();
        let data = view.node_data().unwrap().unwrap();
        let decoder = txn_table.node_data_decoder().unwrap();
        let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();
        let val = decoder.decode_column_value(&data, 1, &mut scratch).unwrap();
        let encoder = RowPackEncoder::new(txn_ctx.clone()).unwrap();
        let mut enc_scratch = RowPackScratch::new(encoder.context());
        let mut expected = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Float64(5.0), ValueRef::Int64(1)],
                &mut expected,
                &mut enc_scratch,
            )
            .unwrap();
        assert_eq!(data, expected);
        assert!(
            matches!(
                val,
                ValueRef::Int64(1) | ValueRef::Int32(1) | ValueRef::UInt64(1) | ValueRef::UInt32(1)
            )
        );
    }

    let output_path = dir.path().join("train.parquet");
    let config = JobConfig {
        job_name: None,
        index_path: index_root.to_string_lossy().to_string(),
        query: "PREDICT SUM(transactions.price WHERE transactions.type = 1, 0, 100, seconds) FOR EACH users.user_id".to_string(),
        output_parquet: output_path.to_string_lossy().to_string(),
        split: SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![1.0, 0.0, 0.0],
            seed: Some(0),
            time: None,
            range: None,
        },
        options: QueryOptions {
            temporal_sampling: TemporalSampling::Timeframe,
            ..Default::default()
        },
        validation_output: None,
    };

    run_job(&config).unwrap();

    let file = File::open(&output_path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(batch.num_rows() > 0);

    let entity = batch.column(0).as_any().downcast_ref::<arrow_array::UInt64Array>().unwrap();
    let target = batch.column(1).as_any().downcast_ref::<arrow_array::Float64Array>().unwrap();
    let timestamp = batch.column(2).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();

    for row in 0..batch.num_rows() {
        let entity_id = entity.value(row);
        let ts = timestamp.value(row);
        let sum = target.value(row);
        let expected = sum_edges_in_window(entity_id, ts, 0, 100);
        assert!((sum - expected).abs() < 1e-6);
    }
}

#[test]
fn temporal_list_distinct_e2e() {
    let dir = tempdir().unwrap();
    let index_root = dir.path().join("index");
    fs::create_dir_all(&index_root).unwrap();

    let metadata = build_metadata_sum();
    fs::write(
        index_root.join("index_metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();

    write_edge_shard(
        &index_root,
        "users",
        0,
        &metadata,
        vec![vec![(0u64, 0i64), (1u64, 100i64), (2u64, 200i64)]],
    );

    let txn_ctx = build_transactions_ctx();
    write_rowpack_ctx(&index_root.join("nodes/transactions/node_data.ctx"), &txn_ctx);
    write_node_data_shard(
        &index_root,
        "transactions",
        0,
        &txn_ctx,
        vec![
            vec![ValueRef::Float64(5.0), ValueRef::Int64(1)],
            vec![ValueRef::Float64(7.0), ValueRef::Int64(0)],
            vec![ValueRef::Float64(11.0), ValueRef::Int64(1)],
        ],
    );

    let output_path = dir.path().join("train.parquet");
    let config = JobConfig {
        job_name: None,
        index_path: index_root.to_string_lossy().to_string(),
        query: "PREDICT LIST_DISTINCT(transactions.type, 0, 100, seconds) FOR EACH users.user_id".to_string(),
        output_parquet: output_path.to_string_lossy().to_string(),
        split: SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![1.0, 0.0, 0.0],
            seed: Some(0),
            time: None,
            range: None,
        },
        options: QueryOptions {
            temporal_sampling: TemporalSampling::Timeframe,
            ..Default::default()
        },
        validation_output: None,
    };

    run_job(&config).unwrap();

    let file = File::open(&output_path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(batch.num_rows() > 0);

    let entity = batch.column(0).as_any().downcast_ref::<arrow_array::UInt64Array>().unwrap();
    let target = batch.column(1).as_any().downcast_ref::<arrow_array::ListArray>().unwrap();
    let timestamp = batch.column(2).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();

    for row in 0..batch.num_rows() {
        let entity_id = entity.value(row);
        let ts = timestamp.value(row);
        let mut expected = list_distinct_in_window(entity_id, ts, 0, 100);
        let values = target.value(row);
        let list = values.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        let mut actual = Vec::new();
        for i in 0..list.len() {
            actual.push(list.value(i));
        }
        expected.sort();
        actual.sort();
        assert_eq!(expected, actual);
    }
}

#[test]
fn temporal_assuming_filter_e2e() {
    let dir = tempdir().unwrap();
    let index_root = dir.path().join("index");
    fs::create_dir_all(&index_root).unwrap();

    let metadata = build_metadata_sum();
    fs::write(
        index_root.join("index_metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();

    write_edge_shard(
        &index_root,
        "users",
        0,
        &metadata,
        vec![vec![(0u64, 0i64), (1u64, 50i64), (2u64, 80i64)]],
    );

    let txn_ctx = build_transactions_ctx();
    write_rowpack_ctx(&index_root.join("nodes/transactions/node_data.ctx"), &txn_ctx);
    write_node_data_shard(
        &index_root,
        "transactions",
        0,
        &txn_ctx,
        vec![
            vec![ValueRef::Float64(5.0), ValueRef::Int64(1)],
            vec![ValueRef::Float64(7.0), ValueRef::Int64(1)],
            vec![ValueRef::Float64(11.0), ValueRef::Int64(0)],
        ],
    );

    let output_path = dir.path().join("train.parquet");
    let config = JobConfig {
        job_name: None,
        index_path: index_root.to_string_lossy().to_string(),
        query: "PREDICT SUM(transactions.price WHERE transactions.type = 1, 0, 100, seconds) FOR EACH users.user_id ASSUMING COUNT(transactions.*, 0, 100, seconds) > 1".to_string(),
        output_parquet: output_path.to_string_lossy().to_string(),
        split: SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![1.0, 0.0, 0.0],
            seed: Some(0),
            time: None,
            range: None,
        },
        options: QueryOptions {
            temporal_sampling: TemporalSampling::Timeframe,
            ..Default::default()
        },
        validation_output: None,
    };

    run_job(&config).unwrap();

    let file = File::open(&output_path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(batch.num_rows() > 0);

    let target = batch.column(1).as_any().downcast_ref::<arrow_array::Float64Array>().unwrap();
    let timestamp = batch.column(2).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();

    for row in 0..batch.num_rows() {
        let ts = timestamp.value(row);
        assert_eq!(ts, 0);
        let sum = target.value(row);
        assert!((sum - 12.0).abs() < 1e-6);
    }
}

fn build_metadata_count() -> DatasetMetadata {
    DatasetMetadata {
        dataset_name: "temporal_test".to_string(),
        node_types: vec![
            NodeTypeMetadata {
                name: "users".to_string(),
                max_id: 1,
                has_node_data: false,
                node_data_compression: None,
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 1,
                    node_data_paths: None,
                }],
            },
            NodeTypeMetadata {
                name: "articles".to_string(),
                max_id: 2,
                has_node_data: false,
                node_data_compression: None,
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            },
        ],
        edge_relations: vec![EdgeRelationMetadata {
            name: "transactions".to_string(),
            src_type: "users".to_string(),
            dst_type: "articles".to_string(),
            shards: vec![],
            dest_sort_key_bits: Some(16),
            dest_sort_key: Some(EdgeSortKeyMetadata {
                width_bits: 16,
                timestamp: Some(EdgeSortKeyTimestampMetadata {
                    table: "transactions".to_string(),
                    column: "timestamp".to_string(),
                    encoding: EdgeSortKeyTimestampEncoding {
                        kind: EdgeSortKeyTimestampKind::Int16,
                        anchor: 0,
                        step: 1,
                    },
                }),
                weight: None,
            }),
            format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
        }],
    }
}

fn build_metadata_sum() -> DatasetMetadata {
    DatasetMetadata {
        dataset_name: "temporal_sum_test".to_string(),
        node_types: vec![
            NodeTypeMetadata {
                name: "users".to_string(),
                max_id: 0,
                has_node_data: false,
                node_data_compression: None,
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 0,
                    node_data_paths: None,
                }],
            },
            NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(diskgraph_core::metadata::NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            },
        ],
        edge_relations: vec![EdgeRelationMetadata {
            name: "transactions".to_string(),
            src_type: "users".to_string(),
            dst_type: "transactions".to_string(),
            shards: vec![],
            dest_sort_key_bits: Some(16),
            dest_sort_key: Some(EdgeSortKeyMetadata {
                width_bits: 16,
                timestamp: Some(EdgeSortKeyTimestampMetadata {
                    table: "transactions".to_string(),
                    column: "timestamp".to_string(),
                    encoding: EdgeSortKeyTimestampEncoding {
                        kind: EdgeSortKeyTimestampKind::Int16,
                        anchor: 0,
                        step: 1,
                    },
                }),
                weight: None,
            }),
            format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
        }],
    }
}

fn write_edge_shard(
    root: &Path,
    table: &str,
    shard_idx: usize,
    metadata: &DatasetMetadata,
    edges_per_node: Vec<Vec<(u64, i64)>>,
) {
    let dir = root.join("nodes").join(table).join(format!("shard-{:05}", shard_idx));
    fs::create_dir_all(&dir).unwrap();
    let rels = metadata.edge_relations.clone();
    let layouts = compute_relation_layouts(&rels, &metadata.node_types).unwrap();
    let layout = &layouts[0];

    let mut offsets = Vec::new();
    offsets.push(0u64);
    let mut edge_bytes = Vec::new();
    let mut edge_count = 0u64;

    for node_edges in edges_per_node {
        let count = node_edges.len() as u32;
        edge_bytes.extend_from_slice(&encode_varint(count));
        for (dst, ts) in node_edges {
            let sort_key = encode_timestamp(ts, EdgeSortKeyTimestampKind::Int16, 0, 1);
            let entry = encode_edge_entry(dst, sort_key, layout);
            edge_bytes.extend_from_slice(&entry);
            edge_count += 1;
        }
        offsets.push(edge_bytes.len() as u64);
    }

    let edges_path = dir.join("edges.bin");
    fs::write(&edges_path, &edge_bytes).unwrap();
    write_offsets_file(&dir.join("offsets.bin"), offsets, edge_count);
}

fn write_empty_shard(root: &Path, table: &str, shard_idx: usize, nodes: u64) {
    let dir = root.join("nodes").join(table).join(format!("shard-{:05}", shard_idx));
    fs::create_dir_all(&dir).unwrap();
    let mut offsets = Vec::with_capacity(nodes as usize + 1);
    for _ in 0..=nodes {
        offsets.push(0u64);
    }
    fs::write(dir.join("edges.bin"), &[]).unwrap();
    write_offsets_file(&dir.join("offsets.bin"), offsets, 0);
}

fn build_transactions_ctx() -> RowPackContext {
    RowPackContext {
        version: 3,
        columns: vec![
            ColumnContext {
                name: "price".to_string(),
                logical_type: IngestedType::Float64,
                encoding: ColumnEncoding::Float { storage: FloatStorage::F64 },
            },
            ColumnContext {
                name: "type".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            },
        ],
        estimated_row_bytes: 24,
    }
}

fn write_rowpack_ctx(path: &Path, ctx: &RowPackContext) {
    let bytes = ctx.to_bytes().unwrap();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_node_data_shard(
    root: &Path,
    table: &str,
    shard_idx: usize,
    ctx: &RowPackContext,
    rows: Vec<Vec<ValueRef<'_>>>,
) {
    let dir = root.join("nodes").join(table).join(format!("shard-{:05}", shard_idx));
    fs::create_dir_all(&dir).unwrap();
    let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
    let mut scratch = RowPackScratch::new(encoder.context());
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    offsets.push(0u64);
    for row in rows {
        let mut encoded = Vec::new();
        encoder.encode_row_into(&row, &mut encoded, &mut scratch).unwrap();
        let len = encoded.len() as u16;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(&encoded);
        offsets.push(data.len() as u64);
    }
    fs::write(dir.join("edges.bin"), &data).unwrap();
    write_offsets_file(&dir.join("offsets.bin"), offsets, 0);
}

fn write_offsets_file(path: &Path, offsets: Vec<u64>, edge_count: u64) {
    let mut file = File::create(path).unwrap();
    let mut header = OffsetFileHeader::new(offsets.len() as u64 - 1, offsets.len() as u64);
    header.delta_bytes = 2;
    header.node_data_len_bytes = 2;
    header.edge_count = edge_count;
    bincode::serialize_into(&mut file, &header).unwrap();
    let mut scratch = Vec::new();
    delta_encode_offsets_into(&offsets, &mut scratch, 2);
    file.write_all(&scratch).unwrap();
    file.flush().unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    bincode::serialize_into(&mut file, &header).unwrap();
}

fn delta_encode_offsets_into(offsets: &[u64], out: &mut Vec<u8>, delta_bytes: u8) {
    let mut idx = 0;
    let chunk = 1024;
    while idx < offsets.len() {
        let chunk_end = (idx + chunk).min(offsets.len());
        let base = offsets[idx];
        let count = (chunk_end - idx) as u16;
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&base.to_le_bytes());
        let mut prev = base;
        for &off in &offsets[idx + 1..chunk_end] {
            let delta = off - prev;
            match delta_bytes {
                2 => out.extend_from_slice(&(delta as u16).to_le_bytes()),
                _ => panic!("unsupported delta bytes"),
            }
            prev = off;
        }
        idx = chunk_end;
    }
}

fn encode_varint(count: u32) -> Vec<u8> {
    if count <= 0x3F {
        return vec![(count as u8) & 0x3F];
    }
    let hi = ((count >> 8) & 0x3F) as u8;
    let lo = (count & 0xFF) as u8;
    vec![0x40 | hi, lo]
}

fn encode_edge_entry(dst: u64, sort_key: u64, layout: &diskgraph_index::layout::RelationLayout) -> Vec<u8> {
    let mut acc: u128 = 0;
    let mut bit_cursor = 0usize;
    let dst_val = dst & layout.dst_mask;
    acc |= (dst_val as u128) << bit_cursor;
    bit_cursor += layout.dst_bits as usize;
    if layout.dest_sort_key_bits.is_some() {
        acc |= (sort_key as u128) << bit_cursor;
    }
    let bytes_needed = layout.entry_bytes.min(16);
    let bytes = acc.to_le_bytes();
    bytes[..bytes_needed].to_vec()
}

fn encode_timestamp(value: i64, kind: EdgeSortKeyTimestampKind, anchor: i64, step: i64) -> u64 {
    let step = step.max(1);
    let delta = value.saturating_sub(anchor).div_euclid(step);
    match kind {
        EdgeSortKeyTimestampKind::Int8 => ordered_signed_to_u64(delta.clamp(i8::MIN as i64, i8::MAX as i64), 8),
        EdgeSortKeyTimestampKind::Int16 => ordered_signed_to_u64(delta.clamp(i16::MIN as i64, i16::MAX as i64), 16),
        EdgeSortKeyTimestampKind::Int32 => ordered_signed_to_u64(delta.clamp(i32::MIN as i64, i32::MAX as i64), 32),
    }
}

fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
    let mask = 1u64 << (bits - 1);
    let raw = match bits {
        8 => (value as i8 as u8) as u64,
        16 => (value as i16 as u16) as u64,
        32 => (value as i32 as u32) as u64,
        _ => value as u64,
    };
    raw ^ mask
}

fn count_edges_in_window(entity: u64, ts: i64, start: i64, end: i64) -> i64 {
    let edges = match entity {
        0 => vec![0i64, 100, 200],
        _ => vec![],
    };
    let window_start = ts + start;
    let window_end = ts + end;
    edges.into_iter().filter(|t| *t >= window_start && *t < window_end).count() as i64
}

fn sum_edges_in_window(entity: u64, ts: i64, start: i64, end: i64) -> f64 {
    let edges = match entity {
        0 => vec![(0i64, 5.0, 1), (100i64, 7.0, 0), (200i64, 11.0, 1)],
        _ => vec![],
    };
    let window_start = ts + start;
    let window_end = ts + end;
    edges
        .into_iter()
        .filter(|(t, _, ty)| *t >= window_start && *t < window_end && *ty == 1)
        .map(|(_, val, _)| val)
        .sum()
}

fn list_distinct_in_window(entity: u64, ts: i64, start: i64, end: i64) -> Vec<i64> {
    let edges = match entity {
        0 => vec![(0i64, 1), (100i64, 0), (200i64, 1)],
        _ => vec![],
    };
    let window_start = ts + start;
    let window_end = ts + end;
    let mut out = Vec::new();
    for (t, ty) in edges {
        if t >= window_start && t < window_end && !out.contains(&ty) {
            out.push(ty);
        }
    }
    out
}
