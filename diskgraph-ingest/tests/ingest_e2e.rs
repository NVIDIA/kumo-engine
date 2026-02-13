use anyhow::{anyhow, Result};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::pkmap::{shard_for_key, Key};
use diskgraph_ingest::{ingest, ingest_with_stats, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

use arrow_array::{
    ArrayRef, BinaryArray, FixedSizeBinaryArray, Float32Array, Int32Array, Int64Array, NullArray,
    RecordBatch, StringArray, TimestampSecondArray,
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
    let props = WriterProperties::builder()
        .set_max_row_group_size(1024)
        .build();
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

fn sum_relation_edges(table: &diskgraph_index::graph::TableReader, rel_name: &str) -> Result<u64> {
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
fn ingest_parquet_with_edges_and_rowpack() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("weight", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["user0001", "user0002"])) as ArrayRef,
            Arc::new(Float32Array::from(vec![10.0_f32, 20.0_f32])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("orders.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("user_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("products.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("weight", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Float32Array::from(vec![5.0_f32, 7.0_f32])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("order_items.parquet"),
        Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("product_id", DataType::Int64, false),
            Field::new("weight", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 2])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "parquet_test",
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("orders.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("products.parquet").to_string_lossy()]},
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "OrderItem",
                "source": {"type": "parquet", "paths": [input_root.join("order_items.parquet").to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "order_id", "ref_table": "Order"},
                    "dst_fk": {"edge_column": "product_id", "ref_table": "Product"},
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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
fn ingest_parquet_multi_file_table_with_two_shards() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let user_ids_1: Vec<i64> = (0..10).collect();
    let user_names_1: Vec<String> = (0..10).map(|id| format!("user{:04}", id)).collect();
    let user_ids_2: Vec<i64> = (10..20).collect();
    let user_names_2: Vec<String> = (10..20).map(|id| format!("user{:04}", id)).collect();

    let order_ids_1: Vec<i64> = (0..10).collect();
    let order_user_ids_1: Vec<i64> = (0..10).map(|id| (id + 1) % 20).collect();
    let order_ids_2: Vec<i64> = (10..20).collect();
    let order_user_ids_2: Vec<i64> = (10..20).map(|id| (id + 3) % 20).collect();

    let mut edge_rows_1 = Vec::new();
    let mut edge_rows_2 = Vec::new();
    let mut edge_user_ids_1 = Vec::new();
    let mut edge_order_ids_1 = Vec::new();
    let mut edge_user_ids_2 = Vec::new();
    let mut edge_order_ids_2 = Vec::new();
    for id in 0i64..10 {
        let user_id = (id * 2) % 20;
        let order_id = (id + 5) % 20;
        edge_rows_1.push(vec![user_id.to_string(), order_id.to_string()]);
        edge_user_ids_1.push(user_id);
        edge_order_ids_1.push(order_id);
    }
    for id in 10i64..20 {
        let user_id = (id * 3) % 20;
        let order_id = (id + 7) % 20;
        edge_rows_2.push(vec![user_id.to_string(), order_id.to_string()]);
        edge_user_ids_2.push(user_id);
        edge_order_ids_2.push(order_id);
    }

    let user_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let order_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
    ]);
    let edge_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("order_id", DataType::Int64, false),
    ]);

    write_parquet(
        &input_root.join("users_part1.parquet"),
        user_schema.clone(),
        vec![
            Arc::new(Int64Array::from(user_ids_1)) as ArrayRef,
            Arc::new(StringArray::from(user_names_1)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("users_part2.parquet"),
        user_schema,
        vec![
            Arc::new(Int64Array::from(user_ids_2)) as ArrayRef,
            Arc::new(StringArray::from(user_names_2)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("orders_part1.parquet"),
        order_schema.clone(),
        vec![
            Arc::new(Int64Array::from(order_ids_1)) as ArrayRef,
            Arc::new(Int64Array::from(order_user_ids_1)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("orders_part2.parquet"),
        order_schema,
        vec![
            Arc::new(Int64Array::from(order_ids_2)) as ArrayRef,
            Arc::new(Int64Array::from(order_user_ids_2)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("order_links_part1.parquet"),
        edge_schema.clone(),
        vec![
            Arc::new(Int64Array::from(edge_user_ids_1)) as ArrayRef,
            Arc::new(Int64Array::from(edge_order_ids_1)) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("order_links_part2.parquet"),
        edge_schema,
        vec![
            Arc::new(Int64Array::from(edge_user_ids_2)) as ArrayRef,
            Arc::new(Int64Array::from(edge_order_ids_2)) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "parquet_shards_multi_file",
        "num_shards": 2,
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [
                    input_root.join("users_part1.parquet").to_string_lossy(),
                    input_root.join("users_part2.parquet").to_string_lossy()
                ]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Order",
                "source": {"type": "parquet", "paths": [
                    input_root.join("orders_part1.parquet").to_string_lossy(),
                    input_root.join("orders_part2.parquet").to_string_lossy()
                ]},
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
                "source": {"type": "parquet", "paths": [
                    input_root.join("order_links_part1.parquet").to_string_lossy(),
                    input_root.join("order_links_part2.parquet").to_string_lossy()
                ]},
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "order_id", "ref_table": "Order"}
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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
        assert!(
            found,
            "missing edge user_id={} order_id={}",
            user_id, order_id
        );
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
        "SKU00001", "SKU00002", "SKU00003", "SKU00004", "SKU00005", "SKU00006",
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
                "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
                "primary_key": "id",
                "weight": "weight",
                "columns": [
                    {"name": "id", "ingested_dtype": "string", "fixed_len": 8},
                    {"name": "weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "SkuEdge",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "src_id", "ref_table": "Sku"},
                    "dst_fk": {"edge_column": "dst_id", "ref_table": "Sku"}
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
        peak_mem_stats: None,
        col_stats: false,
        cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
        row_subsample: false,
    };
    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        options,
    )?;
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
    let table = reader
        .table("Sku")
        .ok_or_else(|| anyhow!("missing Sku table"))?;
    let mut shard_seen = vec![false; table.shard_count()];
    let pk = table
        .primary_key_map()
        .ok_or_else(|| anyhow!("missing pk metadata"))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("missing rowpack decoder"))?;
    for expected in [
        "SKU00001", "SKU00002", "SKU00003", "SKU00004", "SKU00005", "SKU00006",
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
    let mut seen = [false; 2];
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
                "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "ItemEdge",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "src_id", "ref_table": "Item"},
                    "dst_fk": {"edge_column": "dst_id", "ref_table": "Item"}
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
        },
    )?;

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let keys: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    assert_pkey_lookups(&reader, "Item", &key_refs)?;
    assert_pkey_missing(&reader, "Item", &["9999"])?;
    let table = reader
        .table("Item")
        .ok_or_else(|| anyhow!("missing Item table"))?;
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
    let mut seen = [false; 2];
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
                "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "bytes", "fixed_len": 4},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "AssetEdge",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "src_id", "ref_table": "Asset"},
                    "dst_fk": {"edge_column": "dst_id", "ref_table": "Asset"}
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
        peak_mem_stats: None,
        col_stats: false,
        cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
        row_subsample: false,
    };
    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        options,
    )?;
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
    let table = reader
        .table("Asset")
        .ok_or_else(|| anyhow!("missing Asset table"))?;
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
                "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
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
        peak_mem_stats: None,
        col_stats: false,
        cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
        row_subsample: false,
    };
    let _metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        options,
    )?;
    let data_path = output_root.join("nodes/asset/pk_data-00000.bin");
    let bytes = fs::read(&data_path)?;
    assert!(bytes.starts_with(b"DGPKDAT\0"));
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(version, 3);
    let key_type_tag = bytes[12];
    assert_eq!(key_type_tag, 2);
    let key_len = u32::from_le_bytes(bytes[13..17].try_into().unwrap());
    assert_eq!(key_len, 0);
    assert!(!bytes.is_empty());

    Ok(())
}

#[test]
fn ingest_missing_fkeys_do_not_create_edges() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("assets.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec!["asset-a", "asset-b"])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("events.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("asset_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 999])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "zero_fp_fkey_test",
        "tables": [
            {
                "name": "Asset",
                "source": {"type": "parquet", "paths": [input_root.join("assets.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Event",
                "source": {"type": "parquet", "paths": [input_root.join("events.parquet").to_string_lossy()]},
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
    let event_table = reader
        .table("Event")
        .ok_or_else(|| anyhow!("missing Event"))?;
    let asset_table = reader
        .table("Asset")
        .ok_or_else(|| anyhow!("missing Asset"))?;
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
    let session_ts = Arc::new(TimestampSecondArray::from(vec![
        1_700_000_000,
        1_700_001_000,
    ])) as ArrayRef;
    write_parquet(&session_path, session_schema, vec![session_ids, session_ts])?;

    let source = serde_json::json!({
        "dataset_name": "timestamp_sort_test",
        "tables": [
            {
                "name": "Event",
                "source": {"type": "parquet", "paths": [event_path.to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [session_path.to_string_lossy()]},
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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
    let reverse_rows =
        sum_relation_edges(&session_table, "Event_session_id_to_Session_id_reverse")?;
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
                "source": {"type": "csv", "paths": [input_root.join("users.csv").to_string_lossy()]},
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
                "source": {"type": "csv", "paths": [input_root.join("visits.csv").to_string_lossy()]},
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
                "source": {"type": "csv", "paths": [input_root.join("user_visit_edges.csv").to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "src_user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "dst_visit_id", "ref_table": "Visit"}
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

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("user_weight", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.25_f32, 0.75_f32])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("items.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("rank", DataType::Float32, false),
            Field::new(
                "created_at",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            ),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(Float32Array::from(vec![1.5_f32, 2.5_f32])) as ArrayRef,
            Arc::new(TimestampSecondArray::from(vec![1000, 2000])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("clicks.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("item_id", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            ),
            Field::new("edge_weight", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1, 0])) as ArrayRef,
            Arc::new(TimestampSecondArray::from(vec![10, 20, 30])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.5_f32, 1.5_f32, 2.5_f32])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "edge_table_mixed_sort_keys",
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_weight", "ingested_dtype": "float32"}
                ]
            },
            {
                "name": "Item",
                "source": {"type": "parquet", "paths": [input_root.join("items.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "rank", "ingested_dtype": "float32"},
                    {"name": "created_at", "ingested_dtype": "timestamp", "timestamp_unit": "seconds"}
                ]
            },
            {
                "name": "Click",
                "source": {"type": "parquet", "paths": [input_root.join("clicks.parquet").to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "item_id", "ref_table": "Item"},
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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
    let dst_bytes = (layout.dst_bits as usize).div_ceil(8);
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
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "user_id",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "uuid"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "DeviceToken",
                "source": {"type": "parquet", "paths": [input_root.join("device_tokens.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("token_events.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("ip_addresses.parquet").to_string_lossy()]},
                "primary_key": "ip_id",
                "columns": [
                    {"name": "ip_id", "ingested_dtype": "ip"},
                    {"name": "region", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "UserIpEdge",
                "source": {"type": "parquet", "paths": [input_root.join("user_ip_edges.parquet").to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "ip_id", "ref_table": "IpAddress"}
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
    assert_pkey_missing(&reader, "User", &["ffffffffffffffffffffffffffffffff"])?;
    assert_pkey_missing(
        &reader,
        "DeviceToken",
        &["ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"],
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
    let mut seen = [false; 2];
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
                ids.iter()
                    .map(|id| format!("{id}-name"))
                    .collect::<Vec<_>>(),
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
            Arc::new(StringArray::from(vec![
                ids[0].clone(),
                ids[1].clone(),
                ids[0].clone(),
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                ids[1].clone(),
                ids[0].clone(),
                "不存在".to_string(),
            ])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "utf8_string_pk_test",
        "num_shards": 2,
        "tables": [
            {
                "name": "Word",
                "source": {"type": "parquet", "paths": [table_path.to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "WordEdge",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "edge_table": {
                    "src_fk": {"edge_column": "src_id", "ref_table": "Word"},
                    "dst_fk": {"edge_column": "dst_id", "ref_table": "Word"}
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
    let table = reader
        .table("Word")
        .ok_or_else(|| anyhow!("missing Word table"))?;
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
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Order",
                "source": {"type": "parquet", "paths": [input_root.join("orders.parquet").to_string_lossy()]},
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
    let order_table = reader
        .table("Order")
        .ok_or_else(|| anyhow!("missing Order"))?;
    let total_edges = sum_relation_edges(&order_table, "Order_user_id_to_User_id")?;
    assert_eq!(total_edges, 3);
    Ok(())
}

#[test]
fn ingest_weight_expr_with_backticks() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("src.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("dst.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("weight-kg", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Float32Array::from(vec![3.5_f32])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "weight_expr_backticks",
        "tables": [
            {
                "name": "Src",
                "source": {"type": "parquet", "paths": [input_root.join("src.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("dst.parquet").to_string_lossy()]},
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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

    write_parquet(
        &input_root.join("src.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("dst.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("weight-kg", DataType::Float32, false),
            Field::new("offset", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![0])) as ArrayRef,
            Arc::new(Float32Array::from(vec![-3.5_f32])) as ArrayRef,
            Arc::new(Float32Array::from(vec![-2.0_f32])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "weight_expr_log_abs",
        "tables": [
            {
                "name": "Src",
                "source": {"type": "parquet", "paths": [input_root.join("src.parquet").to_string_lossy()]},
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
                "source": {"type": "parquet", "paths": [input_root.join("dst.parquet").to_string_lossy()]},
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
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
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

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
    )?;
    write_parquet(
        &input_root.join("visits.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("user_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![10, 11, 12])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 999])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "missing_dst_key_test",
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Visit",
                "source": {"type": "parquet", "paths": [input_root.join("visits.parquet").to_string_lossy()]},
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

#[test]
fn ingest_drops_edges_from_duplicate_source_pks_embedded_fkey() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("in");
    let output_root = dir.path().join("out");

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("friend_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("targets.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef],
    )?;

    let source = serde_json::json!({
        "dataset_name": "dup_test",
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "friend_id", "ingested_dtype": "int64"}
                ],
                "fkeys": [
                    {
                        "src_column": "friend_id",
                        "dst_table": "Target"
                    }
                ]
            },
            {
                "name": "Target",
                "source": {"type": "parquet", "paths": [input_root.join("targets.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let dest = Dest::Local;
    let (_, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        dest,
        IngestOptions::default(),
    )?;

    let dup_key = "User:shard-00000";
    let user_dup_count = stats.pk_duplicate_counts.get(dup_key).copied().unwrap_or(0);
    assert_eq!(user_dup_count, 1, "Should detect 1 duplicate User row");

    let rel_name = "User_friend_id_to_Target_id";
    let dropped = stats.pk_duplicate_drops.get(rel_name).copied().unwrap_or(0);
    assert_eq!(dropped, 1, "Should drop 1 edge from duplicate User row");

    let total_edges = stats.edge_counts.get(rel_name).copied().unwrap_or(0);
    assert_eq!(total_edges, 2, "Should have 2 valid edges");

    Ok(())
}

#[test]
fn ingest_parquet_drops_rows_with_null_create_time() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    // Node Table
    let node_path = input_root.join("nodes.parquet");
    let node_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            true,
        ), // Nullable
    ]);
    let node_ids = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
    let node_ts = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&node_path, node_schema, vec![node_ids, node_ts])?;

    // Edge Table
    let edge_path = input_root.join("edges.parquet");
    let edge_schema = Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            true,
        ), // Nullable
    ]);
    let edge_src = Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef;
    let edge_dst = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
    let edge_ts = Arc::new(TimestampSecondArray::from(vec![Some(100), None])) as ArrayRef;
    write_parquet(&edge_path, edge_schema, vec![edge_src, edge_dst, edge_ts])?;

    let source = serde_json::json!({
        "dataset_name": "dropped_rows_e2e",
        "tables": [
            {
                "name": "Node",
                "source": {"type": "parquet", "paths": [node_path.to_string_lossy()]},
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "created_at", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "Edge",
                "source": {"type": "parquet", "paths": [edge_path.to_string_lossy()]},
                "create_time": "created_at",
                "edge_table": {
                    "src_fk": {"edge_column": "src", "ref_table": "Node"},
                    "dst_fk": {"edge_column": "dst", "ref_table": "Node"}
                },
                "columns": [
                    {"name": "src", "ingested_dtype": "int64"},
                    {"name": "dst", "ingested_dtype": "int64"},
                    {"name": "created_at", "ingested_dtype": "timestamp"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Check Node Drops
    let node_dropped = stats.dropped_rows_counts.get("Node").copied().unwrap_or(0);
    assert_eq!(node_dropped, 1, "Expected 1 dropped node row");

    // Check Edge Drops
    let edge_dropped = stats.dropped_rows_counts.get("Edge").copied().unwrap_or(0);
    assert_eq!(edge_dropped, 1, "Expected 1 dropped edge row");

    // Check Samples
    let node_sample_key = "node_data_dropped:Node";
    let edge_sample_key = "node_data_dropped:Edge";

    // Note: key_failures_by_context stores counts
    let node_failures = stats
        .key_failures_by_context
        .get(node_sample_key)
        .map(|c| c.parse_failures)
        .unwrap_or(0);
    let edge_failures = stats
        .key_failures_by_context
        .get(edge_sample_key)
        .map(|c| c.parse_failures)
        .unwrap_or(0);

    assert_eq!(node_failures, 1, "Expected 1 recorded failure for Node");
    assert_eq!(edge_failures, 1, "Expected 1 recorded failure for Edge");

    // Check that we have samples
    let has_node_sample = stats
        .key_failure_samples
        .iter()
        .any(|s| s.contains(node_sample_key));
    let has_edge_sample = stats
        .key_failure_samples
        .iter()
        .any(|s| s.contains(edge_sample_key));

    assert!(has_node_sample, "Expected sample for dropped node");
    assert!(has_edge_sample, "Expected sample for dropped edge");

    Ok(())
}

// =============================================================================
// FK Type Resolution E2E Tests
// =============================================================================

/// Test that String FK correctly resolves to Int64 PK type and edges are created.
///
/// Scenario:
/// - customers table: id (Int64 PK) with values [1, 2, 3]
/// - orders table: customer_id (String FK) with values ["1", "2", "3"]
///
/// Expected:
/// - All 4 edges should be created (String "1" joins with Int64 1)
#[test]
fn ingest_string_fk_to_int64_pk_resolution() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("customers.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef,
        ],
    )?;

    // Orders: customer_id is String FK pointing to Int64 PK
    write_parquet(
        &input_root.join("orders.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("customer_id", DataType::Utf8, false),
            Field::new("amount", DataType::Float32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![100, 101, 102, 103])) as ArrayRef,
            Arc::new(StringArray::from(vec!["1", "2", "3", "1"])) as ArrayRef,
            Arc::new(Float32Array::from(vec![
                50.0_f32, 75.0_f32, 100.0_f32, 25.0_f32,
            ])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "fk_resolution_test",
        "tables": [
            {
                "name": "Customer",
                "source": {"type": "parquet", "paths": [input_root.join("customers.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Order",
                "source": {"type": "parquet", "paths": [input_root.join("orders.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "customer_id", "ingested_dtype": "string"},
                    {"name": "amount", "ingested_dtype": "float32"}
                ],
                "fkeys": [
                    {
                        "src_column": "customer_id",
                        "dst_table": "Customer"
                    }
                ]
            }
        ]
    });

    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    // Run ingest
    let (metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify metadata has correct table counts
    assert_eq!(metadata.node_types.len(), 2);

    // Verify edge counts
    let rel_name = "Order_customer_id_to_Customer_id";
    let edge_count = stats.edge_counts.get(rel_name).copied().unwrap_or(0);
    assert_eq!(edge_count, 4, "Should have 4 edges (one per order)");

    // Open index and verify edges
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;

    let order_table_reader = reader.table("Order").unwrap();
    let customer_table = reader.table("Customer").unwrap();

    // Verify forward edges from Order → Customer
    let forward_edges = sum_relation_edges(&order_table_reader, rel_name)?;
    assert_eq!(forward_edges, 4, "Should have 4 forward edges");

    // Verify reverse edges from Customer → Order
    let reverse_rel_name = "Order_customer_id_to_Customer_id_reverse";
    let reverse_edges = sum_relation_edges(&customer_table, reverse_rel_name)?;
    assert_eq!(reverse_edges, 4, "Should have 4 reverse edges");

    // Verify Alice (customer 1) has 2 orders
    let alice_node_id = customer_table
        .node_id_from_pk(&Key::Int(1))?
        .expect("Alice should exist");
    let alice_view = customer_table.edges_for_node(alice_node_id)?;
    let alice_segments = alice_view.relation_segments()?;
    let rev_rel_idx = customer_table
        .relations()
        .iter()
        .position(|r| r.name == reverse_rel_name)
        .expect("reverse relation should exist");
    assert_eq!(
        alice_segments[rev_rel_idx].count, 2,
        "Alice should have 2 orders"
    );

    Ok(())
}

/// Test that Safe coercion (Int32 FK → Int64 PK) works and resolves the type.
#[test]
fn ingest_int32_fk_to_int64_pk_safe_coercion() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("parents.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
    )?;
    write_parquet(
        &input_root.join("children.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("parent_id", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![100, 101])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "safe_coercion_test",
        "tables": [
            {
                "name": "Parent",
                "source": {"type": "parquet", "paths": [input_root.join("parents.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Child",
                "source": {"type": "parquet", "paths": [input_root.join("children.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "parent_id", "ingested_dtype": "int32"}
                ],
                "fkeys": [
                    {
                        "src_column": "parent_id",
                        "dst_table": "Parent"
                    }
                ]
            }
        ]
    });

    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // NOTE: config_resolved.json is no longer written. FK type resolution
    // happens internally via fk_edge_targets without modifying the user config.
    // The key verification is that edges are created correctly.

    // Verify edges created
    let rel_name = "Child_parent_id_to_Parent_id";
    let edge_count = stats.edge_counts.get(rel_name).copied().unwrap_or(0);
    assert_eq!(edge_count, 2, "Should have 2 edges");

    Ok(())
}

/// Test edge table FK resolution - both src_fk and dst_fk are resolved independently.
///
/// Scenario:
/// - users table: id (Int64 PK)
/// - products table: id (Int64 PK)
/// - reviews edge table: user_id (String) → users, product_id (String) → products
///
/// Expected:
/// - Both user_id and product_id should be resolved to int64
/// - Edges should be created correctly
#[test]
fn ingest_edge_table_fk_resolution() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("products.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(vec![100, 101])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Widget", "Gadget"])) as ArrayRef,
        ],
    )?;
    // Edge table with String FKs pointing to Int64 PKs
    write_parquet(
        &input_root.join("reviews.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("product_id", DataType::Utf8, false),
            Field::new("rating", DataType::Int32, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec!["1", "1", "2"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["100", "101", "100"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![5, 4, 3])) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "edge_table_fk_test",
        "tables": [
            {
                "name": "User",
                "source": {"type": "parquet", "paths": [input_root.join("users.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Product",
                "source": {"type": "parquet", "paths": [input_root.join("products.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Review",
                "source": {"type": "parquet", "paths": [input_root.join("reviews.parquet").to_string_lossy()]},
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "product_id", "ingested_dtype": "string"},
                    {"name": "rating", "ingested_dtype": "int32"}
                ],
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "product_id", "ref_table": "Product"}
                }
            }
        ]
    });

    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    // Run ingest
    let (metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify metadata
    assert_eq!(
        metadata.node_types.len(),
        2,
        "Should have User and Product node types"
    );

    // Verify edge counts
    let forward_rel = "edge_Review_src_to_dst";
    let reverse_rel = "edge_Review_dst_to_src";

    let forward_count = stats.edge_counts.get(forward_rel).copied().unwrap_or(0);
    let reverse_count = stats.edge_counts.get(reverse_rel).copied().unwrap_or(0);

    assert_eq!(
        forward_count, 3,
        "Should have 3 forward edges (User → Product)"
    );
    assert_eq!(
        reverse_count, 3,
        "Should have 3 reverse edges (Product → User)"
    );

    // Open index and verify edges
    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;

    let user_table = reader.table("User").unwrap();
    let product_table = reader.table("Product").unwrap();

    // Verify Alice (user 1) exists and total forward edges
    let _alice_node_id = user_table
        .node_id_from_pk(&Key::Int(1))?
        .expect("Alice should exist");
    let total_user_edges = sum_relation_edges(&user_table, forward_rel)?;
    assert_eq!(
        total_user_edges, 3,
        "Should have 3 total forward edges from users"
    );

    // Verify Widget (product 100) has 2 reviews (edges from 2 users)
    let widget_node_id = product_table
        .node_id_from_pk(&Key::Int(100))?
        .expect("Widget should exist");
    let widget_view = product_table.edges_for_node(widget_node_id)?;
    let widget_segments = widget_view.relation_segments()?;
    let rev_rel_idx = product_table
        .relations()
        .iter()
        .position(|r| r.name == reverse_rel)
        .expect("reverse relation should exist");
    assert_eq!(
        widget_segments[rev_rel_idx].count, 2,
        "Widget should have 2 reviews"
    );

    Ok(())
}

// =============================================================================
// Benchmark Scenario Tests — one per FK→PK type combination
// =============================================================================

/// UUID FK → UUID PK (same type, fast path)
#[test]
fn ingest_parquet_uuid_fk_to_uuid_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_uuids = vec![
        "00112233445566778899aabbccddeef0",
        "00112233445566778899aabbccddeef1",
        "00112233445566778899aabbccddeef2",
        "00112233445566778899aabbccddeef3",
        "00112233445566778899aabbccddeef4",
    ];
    // 10 FK rows, each PK referenced 2x
    let fk_uuids: Vec<&str> = pk_uuids
        .iter()
        .copied()
        .chain(pk_uuids.iter().copied())
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(StringArray::from(pk_uuids.clone())) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((0..10).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(StringArray::from(fk_uuids)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "uuid_fk_to_uuid_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "uuid"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "uuid"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// Int64 FK → Int64 PK (same type, fast path)
#[test]
fn ingest_parquet_int64_fk_to_int64_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_ids: Vec<i64> = (1..=5).collect();
    // 10 FK rows, each PK referenced 2x
    let fk_ids: Vec<i64> = pk_ids
        .iter()
        .copied()
        .chain(pk_ids.iter().copied())
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Int64, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(pk_ids)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Int64, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(Int64Array::from(fk_ids)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "int64_fk_to_int64_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "int64"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "int64"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// Int32 FK → Int64 PK (safe coercion)
#[test]
fn ingest_parquet_int32_fk_to_int64_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_ids: Vec<i64> = (1..=5).collect();
    let fk_ids: Vec<i32> = pk_ids
        .iter()
        .copied()
        .chain(pk_ids.iter().copied())
        .map(|v| v as i32)
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Int64, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(pk_ids)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Int32, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(Int32Array::from(fk_ids)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "int32_fk_to_int64_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "int64"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "int32"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// String FK → Int64 PK (parse coercion)
#[test]
fn ingest_parquet_string_fk_to_int64_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_ids: Vec<i64> = (1..=5).collect();
    let fk_strs: Vec<String> = pk_ids
        .iter()
        .copied()
        .chain(pk_ids.iter().copied())
        .map(|v| v.to_string())
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Int64, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from(pk_ids)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(StringArray::from(fk_strs)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "string_fk_to_int64_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "int64"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "string"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// String FK → UUID PK (String→UUID coercion)
#[test]
fn ingest_parquet_string_fk_to_uuid_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_uuids = vec![
        "00112233445566778899aabbccddeef0",
        "00112233445566778899aabbccddeef1",
        "00112233445566778899aabbccddeef2",
        "00112233445566778899aabbccddeef3",
        "00112233445566778899aabbccddeef4",
    ];
    // 10 FK rows as plain strings, each PK referenced 2x
    let fk_strs: Vec<&str> = pk_uuids
        .iter()
        .copied()
        .chain(pk_uuids.iter().copied())
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(StringArray::from(pk_uuids)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(StringArray::from(fk_strs)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "string_fk_to_uuid_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "uuid"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "string"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// String-encoded IP FK → String PK (Utf8 arrow data with ingested_dtype "ip", fast path string pass-through)
#[test]
fn ingest_parquet_string_ip_fk_to_string_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_ips = vec![
        "192.168.1.1",
        "192.168.1.2",
        "192.168.1.3",
        "192.168.1.4",
        "192.168.1.5",
    ];
    // 10 FK rows, each PK referenced 2x
    let fk_ips: Vec<&str> = pk_ips
        .iter()
        .copied()
        .chain(pk_ips.iter().copied())
        .collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(StringArray::from(pk_ips)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(StringArray::from(fk_ips)) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "ip_fk_to_string_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "string"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "ip"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}

/// Parquet column with DataType::Null is rejected with a clear error.
#[test]
fn ingest_parquet_null_column_type_rejected() -> Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join("data");
    fs::create_dir_all(&data_dir)?;

    // Write a parquet file where one column has DataType::Null (simulates Pandas
    // writing a shard where the column is entirely None).
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Null, true),
    ]);
    let id_array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
    let null_array: ArrayRef = Arc::new(NullArray::new(3));
    write_parquet(
        &data_dir.join("nodes.parquet"),
        schema,
        vec![id_array, null_array],
    )?;

    let config = serde_json::json!({
        "dataset_name": "null_test",
        "tables": [{
            "name": "Nodes",
            "columns": [
                {"name": "id", "ingested_dtype": "int64"},
                {"name": "value", "ingested_dtype": "int64"}
            ],
            "primary_key": "id",
            "source": {
                "type": "parquet",
                "paths": ["data/nodes.parquet"]
            }
        }],
        "relationships": []
    });
    let config_path = dir.path().join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

    let output_root = dir.path().join("output");
    let result = ingest(
        &config_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    );

    assert!(result.is_err(), "ingest should fail for Null-typed column");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Null data type"),
        "error should mention Null data type, got: {err_msg}"
    );
    assert!(
        err_msg.contains("value"),
        "error should mention the column name, got: {err_msg}"
    );

    Ok(())
}

/// Binary IP FK → String PK (FixedSizeBinary(4) arrow data, BinaryStringHint::Ip formats back to IPv4 string)
#[test]
fn ingest_parquet_binary_ip_fk_to_string_pk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    let pk_ips = vec![
        "192.168.1.1",
        "192.168.1.2",
        "192.168.1.3",
        "192.168.1.4",
        "192.168.1.5",
    ];

    // 4-byte IPv4 binary representations, each PK referenced 2x
    let ip_bytes: Vec<[u8; 4]> = vec![
        [192, 168, 1, 1],
        [192, 168, 1, 2],
        [192, 168, 1, 3],
        [192, 168, 1, 4],
        [192, 168, 1, 5],
        [192, 168, 1, 1],
        [192, 168, 1, 2],
        [192, 168, 1, 3],
        [192, 168, 1, 4],
        [192, 168, 1, 5],
    ];
    let fk_binary_values: Vec<&[u8]> = ip_bytes.iter().map(|b| b.as_ref()).collect();

    write_parquet(
        &input_root.join("pk_nodes.parquet"),
        Schema::new(vec![
            Field::new("pk_col", DataType::Utf8, false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(StringArray::from(pk_ips)) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )?;
    write_parquet(
        &input_root.join("fk_nodes.parquet"),
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("fk_col", DataType::FixedSizeBinary(4), false),
            Field::new("attr", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110).collect::<Vec<i64>>())) as ArrayRef,
            Arc::new(FixedSizeBinaryArray::try_from_iter(
                fk_binary_values.into_iter(),
            )?) as ArrayRef,
            Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
        ],
    )?;

    let source = serde_json::json!({
        "dataset_name": "binary_ip_fk_to_string_pk",
        "tables": [
            {
                "name": "PkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("pk_nodes.parquet").to_string_lossy()]},
                "primary_key": "pk_col",
                "columns": [
                    {"name": "pk_col", "ingested_dtype": "string"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ]
            },
            {
                "name": "FkNodes",
                "source": {"type": "parquet", "paths": [input_root.join("fk_nodes.parquet").to_string_lossy()]},
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "fk_col", "ingested_dtype": "ip"},
                    {"name": "attr", "ingested_dtype": "int32"}
                ],
                "fkeys": [{"src_column": "fk_col", "dst_table": "PkNodes"}]
            }
        ]
    });
    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let (_metadata, stats) = ingest_with_stats(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    assert_eq!(stats.key_parse_failures, 0, "no parse failures");
    assert_eq!(stats.key_lookup_failures, 0, "no lookup failures");

    let reader = GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let fk_table = reader.table("FkNodes").unwrap();
    let pk_table = reader.table("PkNodes").unwrap();
    let forward = sum_relation_edges(&fk_table, "FkNodes_fk_col_to_PkNodes_pk_col")?;
    let reverse = sum_relation_edges(&pk_table, "FkNodes_fk_col_to_PkNodes_pk_col_reverse")?;
    assert_eq!(forward, 10, "10 forward edges");
    assert_eq!(reverse, 10, "10 reverse edges");

    Ok(())
}
