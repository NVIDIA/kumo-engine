use anyhow::{anyhow, Result};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_ingest::{ingest, ingest_with_stats, Dest, IngestOptions};
use diskgraph_index::graph::GraphIndexReader;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;
use ahash::RandomState;

use arrow_array::{
    ArrayRef, BinaryArray, Float32Array, Int64Array, RecordBatch, StringArray,
    TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema};

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    writeln!(file, "{}", headers.join(","))?;
    for row in rows {
        writeln!(file, "{}", row.join(","))?;
    }
    Ok(())
}

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder().set_max_row_group_size(1024).build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn build_global_ids(sharded_files: &[Vec<u64>], num_shards: usize) -> std::collections::HashMap<u64, u64> {
    const AHASH_SEED_1: u64 = 0x9e3779b185ebca87;
    const AHASH_SEED_2: u64 = 0xc2b2ae3d27d4eb4f;
    const AHASH_SEED_3: u64 = 0x165667b19e3779f9;
    const AHASH_SEED_4: u64 = 0xd6e8feb86659fd93;
    let state = RandomState::with_seeds(
        AHASH_SEED_1,
        AHASH_SEED_2,
        AHASH_SEED_3,
        AHASH_SEED_4,
    );
    let mut shard_counts = vec![0u64; num_shards];
    let mut shard_maps: Vec<std::collections::HashMap<u64, u64>> =
        (0..num_shards).map(|_| std::collections::HashMap::new()).collect();
    for file in sharded_files {
        for &key in file {
            let mut hasher = state.build_hasher();
            hasher.write(&key.to_le_bytes());
            let shard = (hasher.finish() as usize) % num_shards;
            let local = shard_counts[shard];
            shard_counts[shard] = shard_counts[shard].saturating_add(1);
            shard_maps[shard].insert(key, local);
        }
    }
    let mut offsets = vec![0u64; num_shards];
    let mut acc = 0u64;
    for (idx, count) in shard_counts.iter().enumerate() {
        offsets[idx] = acc;
        acc = acc.saturating_add(*count);
    }
    let mut out = std::collections::HashMap::new();
    for shard in 0..num_shards {
        for (key, local) in shard_maps[shard].iter() {
            out.insert(*key, offsets[shard].saturating_add(*local));
        }
    }
    out
}

fn sum_relation_edges(
    table: &diskgraph_index::graph::TableReader,
    rel_name: &str,
) -> Result<u64> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| anyhow!("missing relation {rel_name}"))?;
    let mut total = 0u64;
    for node_id in 0..table.node_count() {
        let view = table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;
        total = total.saturating_add(segments[rel_idx].count as u64);
    }
    Ok(total)
}

#[test]
fn ingest_csv_with_edges_and_rowpack() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("users.csv"),
        &["id", "name", "weight"],
        &[
            vec!["0".into(), "user0001".into(), "10.0".into()],
            vec!["1".into(), "user0002".into(), "20.0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("orders.csv"),
        &["id", "user_id"],
        &[
            vec!["0".into(), "1".into()],
            vec!["1".into(), "0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("products.csv"),
        &["id", "weight"],
        &[
            vec!["0".into(), "5.0".into()],
            vec!["1".into(), "7.0".into()],
        ],
    )?;
    write_csv(
        &input_root.join("order_items.csv"),
        &["order_id", "product_id", "weight"],
        &[
            vec!["0".into(), "1".into(), "3".into()],
            vec!["1".into(), "0".into(), "2".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "csv_test",
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [input_root.join("users.csv").to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "name", "source_dtype": "string", "ingested_dtype": "string"},
                    {"name": "weight", "source_dtype": "float32", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "Order",
                "format": "csv",
                "paths": [input_root.join("orders.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "user_id", "source_dtype": "int64", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Product",
                "format": "csv",
                "paths": [input_root.join("products.csv").to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "weight", "source_dtype": "float32", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "OrderItem",
                "format": "csv",
                "paths": [input_root.join("order_items.csv").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "order_id", "ref_table": "Order", "ref_column": "id"},
                    "dst_fk": {"column": "product_id", "ref_table": "Product", "ref_column": "id"}
                },
                "weight": "weight",
                "columns": [
                    {"name": "order_id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "product_id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "weight", "source_dtype": "int32", "ingested_dtype": "int32"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "order_user",
                "src_table": "Order",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
        },
    )?;
    let index_path = output_root.join("index_metadata.json");
    let disk_metadata: DatasetMetadata = serde_json::from_slice(&fs::read(index_path)?)?;
    assert_eq!(metadata.node_types.len(), 3);
    assert_eq!(disk_metadata.node_types.len(), 3);

    let user = disk_metadata
        .node_types
        .iter()
        .find(|n| n.name == "User")
        .unwrap();
    let compression = user.node_data_compression.as_ref().unwrap();
    assert_eq!(compression.codec, "rowpack");
    let ctx_path = output_root.join(&compression.context_path);
    assert!(ctx_path.exists());
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let view = user_table.edges_for_node(0)?;
    let value = user_table.decode_node_data_json(&view)?.unwrap();
    assert_eq!(value["name"], "user0001");

    let order_user = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "order_user")
        .unwrap();
    let order_user_rev = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "order_user_reverse")
        .unwrap();
    assert_eq!(order_user.dest_sort_key_bits, Some(32));
    assert_eq!(order_user_rev.dest_sort_key_bits, None);

    let edge_src = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_OrderItem_src_to_dst")
        .unwrap();
    let edge_dst = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_OrderItem_dst_to_src")
        .unwrap();
    assert_eq!(edge_src.dest_sort_key_bits, Some(32));
    assert_eq!(edge_dst.dest_sort_key_bits, Some(32));

    let order_table = reader.table("Order").unwrap();
    let user_table = reader.table("User").unwrap();
    let order_user_rows = sum_relation_edges(&order_table, "order_user")?;
    let order_user_rev_rows = sum_relation_edges(&user_table, "order_user_reverse")?;
    assert_eq!(order_user_rows, 2);
    assert_eq!(order_user_rev_rows, 2);

    let product_table = reader.table("Product").unwrap();
    let edge_src_rows = sum_relation_edges(&order_table, "edge_OrderItem_src_to_dst")?;
    let edge_dst_rows = sum_relation_edges(&product_table, "edge_OrderItem_dst_to_src")?;
    assert_eq!(edge_src_rows, 2);
    assert_eq!(edge_dst_rows, 2);

    Ok(())
}

#[test]
fn ingest_csv_multi_file_table_with_two_shards() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let mut user_rows_1 = Vec::new();
    let mut user_rows_2 = Vec::new();
    for id in 0..10 {
        user_rows_1.push(vec![id.to_string(), format!("user{:04}", id)]);
    }
    for id in 10..20 {
        user_rows_2.push(vec![id.to_string(), format!("user{:04}", id)]);
    }
    let mut order_rows_1 = Vec::new();
    let mut order_rows_2 = Vec::new();
    for id in 0..10 {
        let user_id = (id + 1) % 20;
        order_rows_1.push(vec![id.to_string(), user_id.to_string()]);
    }
    for id in 10..20 {
        let user_id = (id + 3) % 20;
        order_rows_2.push(vec![id.to_string(), user_id.to_string()]);
    }
    let mut edge_rows_1 = Vec::new();
    let mut edge_rows_2 = Vec::new();
    for id in 0..10 {
        let user_id = (id * 2) % 20;
        let order_id = (id + 5) % 20;
        edge_rows_1.push(vec![user_id.to_string(), order_id.to_string()]);
    }
    for id in 10..20 {
        let user_id = (id * 3) % 20;
        let order_id = (id + 7) % 20;
        edge_rows_2.push(vec![user_id.to_string(), order_id.to_string()]);
    }

    write_csv(
        &input_root.join("users_part1.csv"),
        &["id", "name"],
        &user_rows_1,
    )?;
    write_csv(
        &input_root.join("users_part2.csv"),
        &["id", "name"],
        &user_rows_2,
    )?;
    write_csv(
        &input_root.join("orders_part1.csv"),
        &["id", "user_id"],
        &order_rows_1,
    )?;
    write_csv(
        &input_root.join("orders_part2.csv"),
        &["id", "user_id"],
        &order_rows_2,
    )?;
    write_csv(
        &input_root.join("order_links_part1.csv"),
        &["user_id", "order_id"],
        &edge_rows_1,
    )?;
    write_csv(
        &input_root.join("order_links_part2.csv"),
        &["user_id", "order_id"],
        &edge_rows_2,
    )?;

    let source = serde_json::json!({
        "dataset_name": "csv_shards_multi_file",
        "num_shards": 2,
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [
                    input_root.join("users_part1.csv").to_string_lossy(),
                    input_root.join("users_part2.csv").to_string_lossy()
                ],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "name", "source_dtype": "string", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Order",
                "format": "csv",
                "paths": [
                    input_root.join("orders_part1.csv").to_string_lossy(),
                    input_root.join("orders_part2.csv").to_string_lossy()
                ],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "user_id", "source_dtype": "int64", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "OrderLink",
                "format": "csv",
                "paths": [
                    input_root.join("order_links_part1.csv").to_string_lossy(),
                    input_root.join("order_links_part2.csv").to_string_lossy()
                ],
                "edge_table": {
                    "src_fk": {"column": "user_id", "ref_table": "User", "ref_column": "id"},
                    "dst_fk": {"column": "order_id", "ref_table": "Order", "ref_column": "id"}
                },
                "columns": [
                    {"name": "user_id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "order_id", "source_dtype": "int64", "ingested_dtype": "int64"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "order_user",
                "src_table": "Order",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
        },
    )?;
    assert_eq!(metadata.node_types.len(), 2);
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let order_table = reader.table("Order").unwrap();
    assert_eq!(user_table.node_count(), 20);
    assert_eq!(order_table.node_count(), 20);
    let user_ids = build_global_ids(&[user_rows_1.iter().map(|r| r[0].parse().unwrap()).collect(),
        user_rows_2.iter().map(|r| r[0].parse().unwrap()).collect()], 2);
    let order_ids = build_global_ids(&[order_rows_1.iter().map(|r| r[0].parse().unwrap()).collect(),
        order_rows_2.iter().map(|r| r[0].parse().unwrap()).collect()], 2);
    let mut expected_edges = 0u64;
    let rel_idx = user_table
        .relations()
        .iter()
        .position(|rel| rel.name == "edge_OrderLink_src_to_dst")
        .ok_or_else(|| anyhow!("missing edge relation"))?;
    for row in edge_rows_1.iter().chain(edge_rows_2.iter()) {
        let user_key: u64 = row[0].parse().unwrap();
        let order_key: u64 = row[1].parse().unwrap();
        let user_id = *user_ids.get(&user_key).unwrap();
        let order_id = *order_ids.get(&order_key).unwrap();
        let view = user_table.edges_for_node(user_id)?;
        let segments = view.relation_segments()?;
        let segment = &segments[rel_idx];
        let mut found = false;
        let entry_bytes = segment.layout.entry_bytes;
        let mask = segment.layout.dst_mask as u128;
        for chunk in segment.data.chunks_exact(entry_bytes) {
            let mut acc: u128 = 0;
            for (idx, byte) in chunk.iter().enumerate() {
                acc |= (*byte as u128) << (8 * idx);
            }
            let dst_id = (acc & mask) as u64;
            if dst_id == order_id {
                found = true;
                break;
            }
        }
        assert!(found, "missing edge user_id={} order_id={}", user_id, order_id);
        expected_edges += 1;
    }
    let total = sum_relation_edges(&user_table, "edge_OrderLink_src_to_dst")?;
    assert_eq!(total, expected_edges);
    Ok(())
}

#[test]
fn ingest_parquet_with_sorted_mapping() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");
    let table_path = input_root.join("sku.parquet");

    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
    ]);
    let ids = Arc::new(StringArray::from(vec!["SKU00001", "SKU00002"])) as ArrayRef;
    let weights = Arc::new(Float32Array::from(vec![1.0_f32, 2.0_f32])) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids, weights])?;

    let source = serde_json::json!({
        "dataset_name": "parquet_test",
        "tables": [
            {
                "name": "Sku",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "source_dtype": "string", "ingested_dtype": "string", "fixed_len": 8},
                    {"name": "weight", "source_dtype": "float32", "ingested_dtype": "float32"}
                ]
            }
        ],
        "relationships": []
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        profile_dir: None,
        progress: None,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    assert_eq!(metadata.node_types.len(), 1);
    assert!(metadata.edge_relations.is_empty());

    let sku = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Sku")
        .unwrap();
    let pk = sku.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "sharded");
    assert_eq!(pk.fixed_len, Some(8));
    let map_path = output_root.join("nodes/sku/pk_map-00000.bin");
    assert!(map_path.exists());
    assert!(fs::metadata(map_path)?.len() > 0);

    Ok(())
}

#[test]
fn ingest_csv_with_bytes_pk_mapping() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("assets.parquet");
    let schema = Schema::new(vec![
        Field::new("id", DataType::Binary, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let ids = Arc::new(BinaryArray::from(vec![b"ABCD".as_ref(), b"EFGH".as_ref()])) as ArrayRef;
    let names = Arc::new(StringArray::from(vec!["asset-a", "asset-b"])) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids, names])?;

    let source = serde_json::json!({
        "dataset_name": "bytes_pk_test",
        "tables": [
            {
                "name": "Asset",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "source_dtype": "bytes", "ingested_dtype": "bytes", "fixed_len": 4},
                    {"name": "name", "source_dtype": "string", "ingested_dtype": "string"}
                ]
            }
        ],
        "relationships": []
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        profile_dir: None,
        progress: None,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    let asset = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Asset")
        .unwrap();
    let pk = asset.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "sharded");
    assert_eq!(pk.fixed_len, Some(4));
    let map_path = output_root.join("nodes/asset/pk_map-00000.bin");
    assert!(map_path.exists());
    assert!(fs::metadata(map_path)?.len() > 0);

    Ok(())
}

#[test]
fn ingest_relationship_uses_dest_timestamp_sort_key() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let event_path = input_root.join("event.parquet");
    let event_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("session_id", DataType::Int64, false),
    ]);
    let event_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    let session_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    write_parquet(&event_path, event_schema, vec![event_ids, session_ids])?;

    let session_path = input_root.join("session.parquet");
    let session_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            false,
        ),
    ]);
    let session_ids = Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef;
    let session_ts = Arc::new(TimestampSecondArray::from(vec![1_700_000_000, 1_700_001_000]))
        as ArrayRef;
    write_parquet(&session_path, session_schema, vec![session_ids, session_ts])?;

    let source = serde_json::json!({
        "dataset_name": "timestamp_sort_test",
        "tables": [
            {
                "name": "Event",
                "format": "parquet",
                "paths": [event_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "session_id", "source_dtype": "int64", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Session",
                "format": "parquet",
                "paths": [session_path.to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "source_dtype": "int64", "ingested_dtype": "int64"},
                    {"name": "created_at", "source_dtype": "timestamp", "ingested_dtype": "timestamp"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "event_session",
                "src_table": "Event",
                "src_column": "session_id",
                "dst_table": "Session",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
        },
    )?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "event_session")
        .unwrap();
    let reverse = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "event_session_reverse")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(32));
    assert_eq!(reverse.dest_sort_key_bits, None);
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let event_table = reader.table("Event").unwrap();
    let session_table = reader.table("Session").unwrap();
    let forward_rows = sum_relation_edges(&event_table, "event_session")?;
    let reverse_rows = sum_relation_edges(&session_table, "event_session_reverse")?;
    assert_eq!(forward_rows, 2);
    assert_eq!(reverse_rows, 2);

    Ok(())
}

#[test]
fn ingest_invalid_fields_records_failures_and_builds() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("users.csv"),
        &["id", "name", "created_at"],
        &[
            vec!["1".into(), "alice".into(), "2020-01-01T00:00:00Z".into()],
            vec!["2".into(), "bob".into(), "not-a-time".into()],
            vec!["bad".into(), "carol".into(), "2020-01-03T00:00:00Z".into()],
        ],
    )?;
    write_csv(
        &input_root.join("visits.csv"),
        &["id", "user_id", "created_at"],
        &[
            vec!["10".into(), "1".into(), "2020-02-01T00:00:00Z".into()],
            vec!["11".into(), "oops".into(), "2020-02-02T00:00:00Z".into()],
            vec!["12".into(), "2".into(), "not-a-time".into()],
        ],
    )?;
    write_csv(
        &input_root.join("user_visit_edges.csv"),
        &["src_user_id", "dst_visit_id"],
        &[
            vec!["1".into(), "10".into()],
            vec!["nope".into(), "11".into()],
            vec!["2".into(), "bad".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "invalid_fields_test",
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [input_root.join("users.csv").to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "source_dtype": "string", "ingested_dtype": "int64"},
                    {"name": "name", "source_dtype": "string", "ingested_dtype": "string"},
                    {"name": "created_at", "source_dtype": "string", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "Visit",
                "format": "csv",
                "paths": [input_root.join("visits.csv").to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "source_dtype": "string", "ingested_dtype": "int64"},
                    {"name": "user_id", "source_dtype": "string", "ingested_dtype": "int64"},
                    {"name": "created_at", "source_dtype": "string", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "UserVisitEdge",
                "format": "csv",
                "paths": [input_root.join("user_visit_edges.csv").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "src_user_id", "ref_table": "User", "ref_column": "id"},
                    "dst_fk": {"column": "dst_visit_id", "ref_table": "Visit", "ref_column": "id"}
                },
                "columns": [
                    {"name": "src_user_id", "source_dtype": "string", "ingested_dtype": "int64"},
                    {"name": "dst_visit_id", "source_dtype": "string", "ingested_dtype": "int64"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "visit_user",
                "src_table": "Visit",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            }
        ]
    });
    let source_path = dir.path().join("source_metadata.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let visit_table = reader.table("Visit").unwrap();
    assert_eq!(user_table.node_count(), 2);
    assert_eq!(visit_table.node_count(), 3);

    assert!(stats.key_parse_failures >= 4);
    for context in [
        "pkey:User.id",
        "fkey:Visit.user_id",
        "edge_fk:UserVisitEdge.src_user_id",
        "edge_fk:UserVisitEdge.dst_visit_id",
    ] {
        let count = stats
            .key_failures_by_context
            .get(context)
            .map(|c| c.parse_failures)
            .unwrap_or(0);
        assert!(count > 0);
    }
    assert!(!stats.key_failure_samples.is_empty());

    Ok(())
}
