use anyhow::{anyhow, Result};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_ingest::{ingest, ingest_with_stats, Dest, IngestOptions};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::pkmap::{shard_for_key, Key};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

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

fn assert_pkey_lookups(reader: &GraphIndexReader, table_name: &str, keys: &[&str]) -> Result<()> {
    let table = reader
        .table(table_name)
        .ok_or_else(|| anyhow!("missing table {table_name}"))?;
    let pk = table
        .primary_key_map()
        .ok_or_else(|| anyhow!("missing pk metadata for {table_name}"))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("missing rowpack decoder for {table_name}"))?;
    for key_str in keys {
        let key = table.parse_pk_key(key_str)?;
        let node_id = table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("missing pk lookup for {table_name}:{key_str}"))?;
        let view = table.edges_for_node(node_id)?;
        let matches = view
            .pk_matches(pk, decoder, &key)?
            .ok_or_else(|| anyhow!("missing node data for {table_name}:{key_str}"))?;
        if !matches {
            return Err(anyhow!("pk mismatch for {table_name}:{key_str}"));
        }
    }
    Ok(())
}

fn assert_pkey_missing(reader: &GraphIndexReader, table_name: &str, keys: &[&str]) -> Result<()> {
    let table = reader
        .table(table_name)
        .ok_or_else(|| anyhow!("missing table {table_name}"))?;
    for key_str in keys {
        let key = table.parse_pk_key(key_str)?;
        if table.node_id_from_pk(&key)?.is_some() {
            return Err(anyhow!("expected missing pk for {table_name}:{key_str}"));
        }
    }
    Ok(())
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

fn extract_relation_sort_keys(
    table: &diskgraph_index::graph::TableReader,
    rel_name: &str,
    node_id: u64,
) -> Result<Vec<u64>> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| anyhow!("missing relation {rel_name}"))?;
    let view = table.edges_for_node(node_id)?;
    let segments = view.relation_segments()?;
    let seg = &segments[rel_idx];
    let mut buf = [0u8; 16];
    let mut out = Vec::new();
    for chunk in seg.data.chunks(seg.layout.entry_bytes) {
        buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        if let Some(bits) = seg.layout.dest_sort_key_bits {
            let sort = (val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1);
            out.push(sort as u64);
        }
        buf.fill(0);
    }
    Ok(out)
}

fn extract_relation_entries(
    table: &diskgraph_index::graph::TableReader,
    rel_name: &str,
    node_id: u64,
) -> Result<Vec<(u64, u64)>> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| anyhow!("missing relation {rel_name}"))?;
    let view = table.edges_for_node(node_id)?;
    let segments = view.relation_segments()?;
    let seg = &segments[rel_idx];
    let mut buf = [0u8; 16];
    let mut out = Vec::new();
    for chunk in seg.data.chunks(seg.layout.entry_bytes) {
        buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        let dst_mask = seg.layout.dst_mask as u128;
        let dst = (val & dst_mask) as u64;
        let sort = if let Some(bits) = seg.layout.dest_sort_key_bits {
            (val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1)
        } else {
            0
        };
        out.push((dst, sort as u64));
        buf.fill(0);
    }
    Ok(out)
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn encode_ts_i32(value: i64, anchor: i64, step: i64) -> u64 {
    let delta = value.saturating_sub(anchor).div_euclid(step.max(1));
    let clamped = delta.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let raw = clamped as u32 as u64;
    raw ^ 0x8000_0000
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
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"},
                    {"name": "weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "Order",
                "format": "csv",
                "paths": [input_root.join("orders.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User",
                        "forward": {
                            "weight": {
                                "table": "User",
                                "expr": "weight",
                                "dtype": "float32"
                            }
                        }
                    }
                ]
            },
            {
                "name": "Product",
                "format": "csv",
                "paths": [input_root.join("products.csv").to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "OrderItem",
                "format": "csv",
                "paths": [input_root.join("order_items.csv").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "order_id", "ref_table": "Order", "ref_column": "id"},
                    "dst_fk": {"column": "product_id", "ref_table": "Product", "ref_column": "id"},
                    "forward": {
                        "weight": {
                            "table": "OrderItem",
                            "expr": "weight",
                            "dtype": "float32"
                        }
                    },
                    "reverse": {
                        "weight": {
                            "table": "OrderItem",
                            "expr": "weight",
                            "dtype": "float32"
                        }
                    }
                },
                "weight": "weight",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "int64"},
                    {"name": "product_id", "ingested_dtype": "int64"},
                    {"name": "weight", "ingested_dtype": "int32"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
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
        .find(|r| r.name == "Order_user_id_to_User_id")
        .unwrap();
    let order_user_rev = disk_metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "Order_user_id_to_User_id_reverse")
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
    let order_user_rows = sum_relation_edges(&order_table, "Order_user_id_to_User_id")?;
    let order_user_rev_rows = sum_relation_edges(&user_table, "Order_user_id_to_User_id_reverse")?;
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
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
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
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User"
                    }
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
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "order_id", "ingested_dtype": "int64"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;
    assert_eq!(metadata.node_types.len(), 2);
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let order_table = reader.table("Order").unwrap();
    assert_eq!(user_table.node_count(), 20);
    assert_eq!(order_table.node_count(), 20);
    let mut expected_edges = 0u64;
    let rel_idx = user_table
        .relations()
        .iter()
        .position(|rel| rel.name == "edge_OrderLink_src_to_dst")
        .ok_or_else(|| anyhow!("missing edge relation"))?;
    for row in edge_rows_1.iter().chain(edge_rows_2.iter()) {
        let user_key: u64 = row[0].parse().unwrap();
        let order_key: u64 = row[1].parse().unwrap();
        let user_key = user_table.parse_pk_key(&user_key.to_string())?;
        let order_key = order_table.parse_pk_key(&order_key.to_string())?;
        let user_id = user_table.node_id_from_pk(&user_key)?.unwrap();
        let order_id = order_table.node_id_from_pk(&order_key)?.unwrap();
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
    let edge_path = input_root.join("sku_edges.parquet");

    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
    ]);
    let ids = Arc::new(StringArray::from(vec![
        "SKU00001",
        "SKU00002",
        "SKU00003",
        "SKU00004",
        "SKU00005",
        "SKU00006",
    ])) as ArrayRef;
    let weights = Arc::new(Float32Array::from(vec![
        1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32,
    ])) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids, weights])?;
    write_parquet(
        &edge_path,
        Schema::new(vec![
            Field::new("src_id", DataType::Utf8, false),
            Field::new("dst_id", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec!["SKU00001", "SKU00002", "SKU00003"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["SKU00002", "SKU00003", "SKU99999"])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "parquet_test",
        "num_shards": 2,
        "tables": [
            {
                "name": "Sku",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "ingested_dtype": "string", "fixed_len": 8},
                    {"name": "weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "SkuEdge",
                "format": "parquet",
                "paths": [edge_path.to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "src_id", "ref_table": "Sku", "ref_column": "id"},
                    "dst_fk": {"column": "dst_id", "ref_table": "Sku", "ref_column": "id"}
                },
                "columns": [
                    {"name": "src_id", "ingested_dtype": "string", "fixed_len": 8},
                    {"name": "dst_id", "ingested_dtype": "string", "fixed_len": 8}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        profile_dir: None,
        progress: None,
        parallel: true,
        index_mem_budget_bytes: None,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    assert_eq!(metadata.node_types.len(), 1);
    assert_eq!(metadata.edge_relations.len(), 2);

    let sku = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Sku")
        .unwrap();
    let pk = sku.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "ptrhash");
    assert_eq!(pk.fixed_len, Some(8));
    let data_path = output_root.join("nodes/sku/pk_data-00000.bin");
    let hash_path = output_root.join("nodes/sku/pk_hash-00000.bin");
    assert!(data_path.exists());
    assert!(hash_path.exists());
    assert!(fs::metadata(data_path)?.len() > 0);
    assert!(fs::metadata(hash_path)?.len() > 0);

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let table = reader.table("Sku").ok_or_else(|| anyhow!("missing Sku table"))?;
    let mut shard_seen = vec![false; table.shard_count()];
    let pk = table
        .primary_key_map()
        .ok_or_else(|| anyhow!("missing pk metadata"))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("missing rowpack decoder"))?;
    for expected in [
        "SKU00001",
        "SKU00002",
        "SKU00003",
        "SKU00004",
        "SKU00005",
        "SKU00006",
    ] {
        let key = table.parse_pk_key(expected)?;
        let node_id = table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("missing pk lookup for {expected}"))?;
        let shard_idx = table
            .shard_for_node(node_id)
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("missing shard for node {node_id}"))?;
        shard_seen[shard_idx] = true;
        let view = table.edges_for_node(node_id)?;
        let matches = view
            .pk_matches(pk, decoder, &key)?
            .ok_or_else(|| anyhow!("missing node data for {expected}"))?;
        if !matches {
            let actual = table
                .decode_node_data_json(&view)?
                .and_then(|v| v.get("id").cloned())
                .unwrap_or(serde_json::Value::Null);
            panic!("pk mismatch for {expected}, node id value={actual}");
        }
    }
    assert!(
        shard_seen.iter().all(|seen| *seen),
        "expected keys to cover all shards, got {shard_seen:?}"
    );
    assert_pkey_missing(&reader, "Sku", &["SKU99999"])?;
    let total_edges = sum_relation_edges(&table, "edge_SkuEdge_src_to_dst")?;
    assert_eq!(total_edges, 2);

    Ok(())
}

#[test]
fn ingest_parquet_with_int_pk_lookup() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("items.parquet");
    let edge_path = input_root.join("item_edges.parquet");
    let mut ids: Vec<i64> = Vec::new();
    let mut seen = vec![false; 2];
    let mut cursor = 1_i64;
    while (!seen.iter().all(|v| *v) || ids.len() < 3) && ids.len() < 8 {
        let key = Key::Int(cursor as u64);
        let shard = shard_for_key(&key, 2);
        if !seen[shard] {
            seen[shard] = true;
        }
        ids.push(cursor);
        cursor += 1;
    }
    let names: Vec<String> = ids.iter().map(|id| format!("name-{id}")).collect();
    write_parquet(
        &table_path,
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(ids.clone())) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &edge_path,
        Schema::new(vec![
            Field::new("src_id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![ids[0], ids[1], ids[2]])) as ArrayRef,
            Arc::new(Int64Array::from(vec![ids[1], ids[2], 9999_i64])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "int_pk_test",
        "num_shards": 2,
        "tables": [
            {
                "name": "Item",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "ItemEdge",
                "format": "parquet",
                "paths": [edge_path.to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "src_id", "ref_table": "Item", "ref_column": "id"},
                    "dst_fk": {"column": "dst_id", "ref_table": "Item", "ref_column": "id"}
                },
                "columns": [
                    {"name": "src_id", "ingested_dtype": "int64"},
                    {"name": "dst_id", "ingested_dtype": "int64"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let keys: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    assert_pkey_lookups(&reader, "Item", &key_refs)?;
    assert_pkey_missing(&reader, "Item", &["9999"])?;
    let table = reader.table("Item").ok_or_else(|| anyhow!("missing Item table"))?;
    let total_edges = sum_relation_edges(&table, "edge_ItemEdge_src_to_dst")?;
    assert_eq!(total_edges, 2);

    Ok(())
}

#[test]
fn ingest_csv_with_bytes_pk_mapping() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("assets.parquet");
    let edge_path = input_root.join("asset_edges.parquet");
    let schema = Schema::new(vec![
        Field::new("id", DataType::Binary, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let mut ids: Vec<[u8; 4]> = Vec::new();
    let mut seen = vec![false; 2];
    let mut cursor = 0u8;
    while (!seen.iter().all(|v| *v) || ids.len() < 3) && ids.len() < 16 {
        let key = [b'A', b'A', b'A', b'A' + cursor];
        let shard = shard_for_key(&Key::Bytes(key.to_vec()), 2);
        if !seen[shard] {
            seen[shard] = true;
        }
        ids.push(key);
        cursor = cursor.wrapping_add(1);
    }
    let ids_bytes: Vec<&[u8]> = ids.iter().map(|id| id.as_ref()).collect();
    let ids_array = Arc::new(BinaryArray::from(ids_bytes)) as ArrayRef;
    let names: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(idx, _)| format!("asset-{idx}"))
        .collect();
    let names = Arc::new(StringArray::from(names)) as ArrayRef;
    write_parquet(&table_path, schema, vec![ids_array, names])?;
    let edge_ids_src = Arc::new(BinaryArray::from(vec![
        ids[0].as_ref(),
        ids[1].as_ref(),
        ids[2].as_ref(),
    ])) as ArrayRef;
    let edge_ids_dst = Arc::new(BinaryArray::from(vec![
        ids[1].as_ref(),
        ids[0].as_ref(),
        b"IJKL".as_ref(),
    ])) as ArrayRef;
    write_parquet(
        &edge_path,
        Schema::new(vec![
            Field::new("src_id", DataType::Binary, false),
            Field::new("dst_id", DataType::Binary, false),
        ]),
        vec![edge_ids_src, edge_ids_dst],
    )?;

    let source = serde_json::json!({
        "dataset_name": "bytes_pk_test",
        "num_shards": 2,
        "tables": [
            {
                "name": "Asset",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "bytes", "fixed_len": 4},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "AssetEdge",
                "format": "parquet",
                "paths": [edge_path.to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "src_id", "ref_table": "Asset", "ref_column": "id"},
                    "dst_fk": {"column": "dst_id", "ref_table": "Asset", "ref_column": "id"}
                },
                "columns": [
                    {"name": "src_id", "ingested_dtype": "bytes", "fixed_len": 4},
                    {"name": "dst_id", "ingested_dtype": "bytes", "fixed_len": 4}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        profile_dir: None,
        progress: None,
        parallel: true,
        index_mem_budget_bytes: None,
    };
    let metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    let asset = metadata
        .node_types
        .iter()
        .find(|n| n.name == "Asset")
        .unwrap();
    let pk = asset.primary_key_map.as_ref().unwrap();
    assert_eq!(pk.mapping_kind, "ptrhash");
    assert_eq!(pk.fixed_len, Some(4));
    let data_path = output_root.join("nodes/asset/pk_data-00000.bin");
    let hash_path = output_root.join("nodes/asset/pk_hash-00000.bin");
    assert!(data_path.exists());
    assert!(hash_path.exists());
    assert!(fs::metadata(data_path)?.len() > 0);
    assert!(fs::metadata(hash_path)?.len() > 0);

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let keys: Vec<String> = ids
        .iter()
        .map(|id| String::from_utf8_lossy(id).to_string())
        .collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    assert_pkey_lookups(&reader, "Asset", &key_refs)?;
    assert_pkey_missing(&reader, "Asset", &["IJKL"])?;
    let table = reader.table("Asset").ok_or_else(|| anyhow!("missing Asset table"))?;
    let total_edges = sum_relation_edges(&table, "edge_AssetEdge_src_to_dst")?;
    assert_eq!(total_edges, 2);

    Ok(())
}

#[test]
fn ingest_pk_map_header_has_expected_format() -> Result<()> {
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
        "dataset_name": "pk_header_test",
        "tables": [
            {
                "name": "Asset",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "bytes", "fixed_len": 4},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions {
        profile_dir: None,
        progress: None,
        parallel: true,
        index_mem_budget_bytes: None,
    };
    let _metadata = ingest(&source_path, output_root.to_str().unwrap(), Dest::Local, options)?;
    let data_path = output_root.join("nodes/asset/pk_data-00000.bin");
    let bytes = fs::read(&data_path)?;
    assert!(bytes.starts_with(b"DGPKDAT\0"));
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(version, 3);
    let key_type_tag = bytes[12];
    assert_eq!(key_type_tag, 2);
    let key_len = u32::from_le_bytes(bytes[13..17].try_into().unwrap());
    assert_eq!(key_len, 0);
    assert!(bytes.len() > 0);

    Ok(())
}

#[test]
fn ingest_missing_fkeys_do_not_create_edges() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("assets.csv"),
        &["id", "name"],
        &[
            vec!["0".into(), "asset-a".into()],
            vec!["1".into(), "asset-b".into()],
        ],
    )?;
    write_csv(
        &input_root.join("events.csv"),
        &["id", "asset_id"],
        &[
            vec!["0".into(), "0".into()],
            vec!["1".into(), "999".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "zero_fp_fkey_test",
        "tables": [
            {
                "name": "Asset",
                "format": "csv",
                "paths": [input_root.join("assets.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Event",
                "format": "csv",
                "paths": [input_root.join("events.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "asset_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "asset_id",
                        "dst_table": "Asset"
                    }
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let options = IngestOptions::default();
    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        options,
    )?;

    let index_metadata = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&index_metadata, &output_root)?;
    let event_table = reader.table("Event").ok_or_else(|| anyhow!("missing Event"))?;
    let asset_table = reader.table("Asset").ok_or_else(|| anyhow!("missing Asset"))?;
    let forward_rows = sum_relation_edges(&event_table, "Event_asset_id_to_Asset_id")?;
    let reverse_rows = sum_relation_edges(&asset_table, "Event_asset_id_to_Asset_id_reverse")?;
    assert_eq!(forward_rows, 1);
    assert_eq!(reverse_rows, 1);

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
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "session_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "session_id",
                        "dst_table": "Session",
                        "forward": {
                            "time": {
                                "table": "Session",
                                "column": "created_at",
                                "bitwidth": 32,
                                "anchor_time": 0,
                                "unit": "s"
                            }
                        }
                    }
                ]
            },
            {
                "name": "Session",
                "format": "parquet",
                "paths": [session_path.to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "created_at", "ingested_dtype": "timestamp"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "Event_session_id_to_Session_id")
        .unwrap();
    let reverse = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "Event_session_id_to_Session_id_reverse")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(32));
    assert_eq!(reverse.dest_sort_key_bits, None);
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let event_table = reader.table("Event").unwrap();
    let session_table = reader.table("Session").unwrap();
    let forward_rows = sum_relation_edges(&event_table, "Event_session_id_to_Session_id")?;
    let reverse_rows = sum_relation_edges(&session_table, "Event_session_id_to_Session_id_reverse")?;
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
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"},
                    {"name": "created_at", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "Visit",
                "format": "csv",
                "paths": [input_root.join("visits.csv").to_string_lossy()],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "created_at", "ingested_dtype": "timestamp"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User"
                    }
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
                    {"name": "src_user_id", "ingested_dtype": "int64"},
                    {"name": "dst_visit_id", "ingested_dtype": "int64"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
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

#[test]
fn ingest_edge_table_mixed_sort_keys() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("users.csv"),
        &["id", "user_weight"],
        &[
            vec!["0".into(), "0.25".into()],
            vec!["1".into(), "0.75".into()],
        ],
    )?;
    write_csv(
        &input_root.join("items.csv"),
        &["id", "rank", "created_at"],
        &[
            vec!["0".into(), "1.5".into(), "1000".into()],
            vec!["1".into(), "2.5".into(), "2000".into()],
        ],
    )?;
    write_csv(
        &input_root.join("clicks.csv"),
        &["user_id", "item_id", "ts", "edge_weight"],
        &[
            vec!["0".into(), "0".into(), "10".into(), "0.5".into()],
            vec!["0".into(), "1".into(), "20".into(), "1.5".into()],
            vec!["1".into(), "0".into(), "30".into(), "2.5".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "edge_table_mixed_sort_keys",
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [input_root.join("users.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "Item",
                "format": "csv",
                "paths": [input_root.join("items.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "rank", "ingested_dtype": "float32"},
                    {"name": "created_at", "ingested_dtype": "timestamp", "timestamp_unit": "seconds"}
                ]
            },
            {
                "name": "Click",
                "format": "csv",
                "paths": [input_root.join("clicks.csv").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "user_id", "ref_table": "User", "ref_column": "id"},
                    "dst_fk": {"column": "item_id", "ref_table": "Item", "ref_column": "id"},
                    "forward": {
                        "time": {
                            "table": "Click",
                            "column": "ts",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        },
                        "weight": {
                            "table": "Item",
                            "expr": "rank",
                            "dtype": "float32"
                        }
                    },
                    "reverse": {
                        "time": {
                            "table": "Click",
                            "column": "ts",
                            "bitwidth": 16,
                            "anchor_time": 0,
                            "unit": "s"
                        },
                        "weight": {
                            "table": "User",
                            "expr": "user_weight",
                            "dtype": "float32"
                        }
                    }
                },
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp", "timestamp_unit": "seconds"},
                    {"name": "edge_weight", "ingested_dtype": "float32"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_Click_src_to_dst")
        .unwrap();
    let reverse = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_Click_dst_to_src")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(64));
    assert_eq!(reverse.dest_sort_key_bits, Some(48));

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let rel_idx = user_table
        .relations()
        .iter()
        .position(|rel| rel.name == "edge_Click_src_to_dst")
        .ok_or_else(|| anyhow!("missing edge_Click_src_to_dst relation"))?;
    let segments = user_table.edges_for_node(0)?.relation_segments()?;
    let layout = &segments[rel_idx].layout;
    let dst_bytes = ((layout.dst_bits as usize) + 7) / 8;
    assert!(
        layout.entry_bytes > dst_bytes,
        "expected edge payload bytes for edge_Click_src_to_dst"
    );
    let sort_keys = extract_relation_sort_keys(&user_table, "edge_Click_src_to_dst", 0)?;
    let ts0 = encode_ts_i32(10, 0, 1);
    let ts1 = encode_ts_i32(20, 0, 1);
    let weight0 = ordered_f32_bits(1.5) as u64;
    let weight1 = ordered_f32_bits(2.5) as u64;
    let expected0 = (ts1 << 32) | weight1;
    let expected1 = (ts0 << 32) | weight0;
    assert_eq!(sort_keys, vec![expected0, expected1]);
    let entries = extract_relation_entries(&user_table, "edge_Click_src_to_dst", 0)?;
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].1, expected0);

    Ok(())
}

#[test]
fn ingest_parquet_with_uuid_ip_types() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let user_ids = vec![
        "00112233445566778899aabbccddeeff",
        "112233445566778899aabbccddeeff00",
    ];
    let token_ids = vec![
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
        "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100",
    ];
    let ip_ids = vec!["192.168.1.2", "2001:db8::1"];

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(user_ids.clone())),
            Arc::new(StringArray::from(vec!["alice", "bob"])),
        ],
    )?;

    write_parquet(
        &input_root.join("device_tokens.parquet"),
        Schema::new(vec![
            Field::new("token_id", DataType::Utf8, false),
            Field::new("user_id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(token_ids.clone())),
            Arc::new(StringArray::from(user_ids.clone())),
            Arc::new(StringArray::from(vec!["t1", "t2"])),
        ],
    )?;

    write_parquet(
        &input_root.join("token_events.parquet"),
        Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("token_id", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec![
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ])),
            Arc::new(StringArray::from(token_ids.clone())),
        ],
    )?;

    write_parquet(
        &input_root.join("ip_addresses.parquet"),
        Schema::new(vec![
            Field::new("ip_id", DataType::Utf8, false),
            Field::new("region", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(ip_ids.clone())),
            Arc::new(StringArray::from(vec!["na", "eu"])),
        ],
    )?;

    write_parquet(
        &input_root.join("user_ip_edges.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("ip_id", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec![
                user_ids[0].to_string(),
                user_ids[1].to_string(),
                user_ids[0].to_string(),
            ])),
            Arc::new(StringArray::from(vec![
                ip_ids[0].to_string(),
                ip_ids[1].to_string(),
                "203.0.113.99".to_string(),
            ])),
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "uuid_ip_parquet",
        "tables": [
            {
                "name": "User",
                "format": "parquet",
                "paths": [input_root.join("users.parquet").to_string_lossy()],
                "primary_key": "user_id",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "uuid"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "DeviceToken",
                "format": "parquet",
                "paths": [input_root.join("device_tokens.parquet").to_string_lossy()],
                "primary_key": "token_id",
                "columns": [
                    {"name": "token_id", "ingested_dtype": "uuid256"},
                    {"name": "user_id", "ingested_dtype": "uuid"},
                    {"name": "label", "ingested_dtype": "string"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User"
                    }
                ]
            },
            {
                "name": "TokenEvent",
                "format": "parquet",
                "paths": [input_root.join("token_events.parquet").to_string_lossy()],
                "primary_key": "event_id",
                "columns": [
                    {"name": "event_id", "ingested_dtype": "uuid"},
                    {"name": "token_id", "ingested_dtype": "uuid256"}
                ],
                "fkeys": [
                    {
                        "src_column": "token_id",
                        "dst_table": "DeviceToken"
                    }
                ]
            },
            {
                "name": "IpAddress",
                "format": "parquet",
                "paths": [input_root.join("ip_addresses.parquet").to_string_lossy()],
                "primary_key": "ip_id",
                "columns": [
                    {"name": "ip_id", "ingested_dtype": "ip"},
                    {"name": "region", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "UserIpEdge",
                "format": "parquet",
                "paths": [input_root.join("user_ip_edges.parquet").to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "user_id", "ref_table": "User", "ref_column": "user_id"},
                    "dst_fk": {"column": "ip_id", "ref_table": "IpAddress", "ref_column": "ip_id"}
                },
                "columns": [
                    {"name": "user_id", "ingested_dtype": "uuid"},
                    {"name": "ip_id", "ingested_dtype": "ip"}
                ]
            }
        ]
    });

    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0);
    assert!(
        stats.key_lookup_failures <= 1,
        "expected at most one missing edge lookup"
    );

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    assert_eq!(reader.table("User").unwrap().node_count(), 2);
    assert_eq!(reader.table("DeviceToken").unwrap().node_count(), 2);
    assert_eq!(reader.table("IpAddress").unwrap().node_count(), 2);
    assert_pkey_lookups(&reader, "User", &user_ids)?;
    assert_pkey_lookups(&reader, "DeviceToken", &token_ids)?;
    assert_pkey_lookups(&reader, "IpAddress", &ip_ids)?;
    assert_pkey_lookups(
        &reader,
        "TokenEvent",
        &[
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ],
    )?;
    assert_pkey_missing(
        &reader,
        "User",
        &["ffffffffffffffffffffffffffffffff"],
    )?;
    assert_pkey_missing(
        &reader,
        "DeviceToken",
        &[ "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff" ],
    )?;
    assert_pkey_missing(&reader, "IpAddress", &["203.0.113.99"])?;
    let user_table = reader.table("User").unwrap();
    let total_edges = sum_relation_edges(&user_table, "edge_UserIpEdge_src_to_dst")?;
    assert_eq!(total_edges, 2);

    Ok(())
}

#[test]
fn ingest_parquet_with_utf8_string_pk_lookup() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let table_path = input_root.join("words.parquet");
    let edge_path = input_root.join("word_edges.parquet");
    let mut ids: Vec<String> = Vec::new();
    let mut seen = vec![false; 2];
    let candidates = [
        "café-01",
        "niño-02",
        "résumé",
        "jalapeño",
        "über-05",
        "façade",
    ];
    for candidate in candidates.iter() {
        let key = Key::Bytes(candidate.as_bytes().to_vec());
        let shard = shard_for_key(&key, 2);
        if !seen[shard] {
            seen[shard] = true;
        }
        ids.push(candidate.to_string());
        if seen.iter().all(|v| *v) {
            break;
        }
    }
    if !seen.iter().all(|v| *v) {
        return Err(anyhow!("failed to cover all shards with utf8 keys"));
    }
    write_parquet(
        &table_path,
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(ids.clone())) as ArrayRef,
            Arc::new(StringArray::from(
                ids.iter().map(|id| format!("{id}-name")).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    write_parquet(
        &edge_path,
        Schema::new(vec![
            Field::new("src_id", DataType::Utf8, false),
            Field::new("dst_id", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec![ids[0].clone(), ids[1].clone(), ids[0].clone()]))
                as ArrayRef,
            Arc::new(StringArray::from(vec![ids[1].clone(), ids[0].clone(), "不存在".to_string()]))
                as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "utf8_string_pk_test",
        "num_shards": 2,
        "tables": [
            {
                "name": "Word",
                "format": "parquet",
                "paths": [table_path.to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "WordEdge",
                "format": "parquet",
                "paths": [edge_path.to_string_lossy()],
                "edge_table": {
                    "src_fk": {"column": "src_id", "ref_table": "Word", "ref_column": "id"},
                    "dst_fk": {"column": "dst_id", "ref_table": "Word", "ref_column": "id"}
                },
                "columns": [
                    {"name": "src_id", "ingested_dtype": "string"},
                    {"name": "dst_id", "ingested_dtype": "string"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let keys: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    assert_pkey_lookups(&reader, "Word", &keys)?;
    assert_pkey_missing(&reader, "Word", &["不存在"])?;
    let table = reader.table("Word").ok_or_else(|| anyhow!("missing Word table"))?;
    let total_edges = sum_relation_edges(&table, "edge_WordEdge_src_to_dst")?;
    assert_eq!(total_edges, 2);

    Ok(())
}

#[test]
fn ingest_parquet_to_index_reader_e2e() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let user_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let user_ids = Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef;
    let user_names = Arc::new(StringArray::from(vec!["alice", "bob", "cora"])) as ArrayRef;
    write_parquet(
        &input_root.join("users.parquet"),
        user_schema,
        vec![user_ids, user_names],
    )?;

    let order_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
    ]);
    let order_ids = Arc::new(Int64Array::from(vec![100, 101, 102])) as ArrayRef;
    let order_user_ids = Arc::new(Int64Array::from(vec![0, 1, 1])) as ArrayRef;
    write_parquet(
        &input_root.join("orders.parquet"),
        order_schema,
        vec![order_ids, order_user_ids],
    )?;

    let metadata = serde_json::json!({
        "dataset_name": "parquet_e2e_index",
        "tables": [
            {
                "name": "User",
                "format": "parquet",
                "paths": [input_root.join("users.parquet").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Order",
                "format": "parquet",
                "paths": [input_root.join("orders.parquet").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User"
                    }
                ]
            }
        ]
    });
    let source_path = input_root.join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&metadata)?)?;

    let options = IngestOptions::default();
    ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        options,
    )?;

    let index_metadata = output_root.join("index_metadata.json");
    let reader = GraphIndexReader::open(&index_metadata, &output_root)?;
    let order_table = reader.table("Order").ok_or_else(|| anyhow!("missing Order"))?;
    let total_edges = sum_relation_edges(&order_table, "Order_user_id_to_User_id")?;
    assert_eq!(total_edges, 3);
    Ok(())
}

#[test]
fn ingest_weight_expr_with_backticks() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("src.csv"),
        &["id", "dst_id"],
        &[vec!["0".into(), "0".into()]],
    )?;
    write_csv(
        &input_root.join("dst.csv"),
        &["id", "weight-kg"],
        &[vec!["0".into(), "3.5".into()]],
    )?;

    let source = serde_json::json!({
        "dataset_name": "weight_expr_backticks",
        "tables": [
            {
                "name": "Src",
                "format": "csv",
                "paths": [input_root.join("src.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "dst_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "dst_id",
                        "dst_table": "Dst",
                        "forward": {
                            "weight": {
                                "table": "Dst",
                                "expr": "`weight-kg`",
                                "dtype": "float32"
                            }
                        }
                    }
                ]
            },
            {
                "name": "Dst",
                "format": "csv",
                "paths": [input_root.join("dst.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "weight-kg", "ingested_dtype": "float32"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "Src_dst_id_to_Dst_id")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(32));

    Ok(())
}

#[test]
fn ingest_weight_expr_with_log_and_abs() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("src.csv"),
        &["id", "dst_id"],
        &[vec!["0".into(), "0".into()]],
    )?;
    write_csv(
        &input_root.join("dst.csv"),
        &["id", "weight-kg", "offset"],
        &[vec!["0".into(), "-3.5".into(), "-2.0".into()]],
    )?;

    let source = serde_json::json!({
        "dataset_name": "weight_expr_log_abs",
        "tables": [
            {
                "name": "Src",
                "format": "csv",
                "paths": [input_root.join("src.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "dst_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "dst_id",
                        "dst_table": "Dst",
                        "forward": {
                            "weight": {
                                "table": "Dst",
                                "expr": "log(abs(`weight-kg`) + 1) + abs(offset)",
                                "dtype": "float32"
                            }
                        }
                    }
                ]
            },
            {
                "name": "Dst",
                "format": "csv",
                "paths": [input_root.join("dst.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "weight-kg", "ingested_dtype": "float32"},
                    {"name": "offset", "ingested_dtype": "float32"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        },
    )?;
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "Src_dst_id_to_Dst_id")
        .unwrap();
    assert_eq!(forward.dest_sort_key_bits, Some(32));

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let src_table = reader.table("Src").unwrap();
    let sort_keys = extract_relation_sort_keys(&src_table, "Src_dst_id_to_Dst_id", 0)?;
    let expected = (f64::log10(3.5_f64 + 1.0) + 2.0) as f32;
    let expected_bits = ordered_f32_bits(expected) as u64;
    assert_eq!(sort_keys, vec![expected_bits]);

    Ok(())
}

#[test]
fn ingest_missing_dst_key_records_lookup() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_csv(
        &input_root.join("users.csv"),
        &["id"],
        &[
            vec!["1".into()],
            vec!["2".into()],
        ],
    )?;
    write_csv(
        &input_root.join("visits.csv"),
        &["id", "user_id"],
        &[
            vec!["10".into(), "1".into()],
            vec!["11".into(), "2".into()],
            vec!["12".into(), "999".into()],
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "missing_dst_key_test",
        "tables": [
            {
                "name": "User",
                "format": "csv",
                "paths": [input_root.join("users.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Visit",
                "format": "csv",
                "paths": [input_root.join("visits.csv").to_string_lossy()],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "User"
                    }
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let visit_table = reader.table("Visit").unwrap();
    let forward_rows = sum_relation_edges(&visit_table, "Visit_user_id_to_User_id")?;
    assert_eq!(forward_rows, 2);

    assert!(stats.key_lookup_failures >= 1);
    let lookup = stats
        .key_failures_by_context
        .get("dst_lookup:Visit_user_id_to_User_id")
        .map(|c| c.lookup_failures)
        .unwrap_or(0);
    assert!(lookup > 0);

    Ok(())
}
