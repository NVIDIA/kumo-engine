//! FK Key Config E2E Tests
//!
//! Tests that verify the FkKeyConfig system works correctly:
//! 1. FK columns are stored in their original `ingested_dtype` (rowpack)
//! 2. Edges are built correctly using `target_type` from FkKeyConfig
//! 3. Various FK→PK type combinations work as expected
//!
//! The key insight: rowpack storage uses `ingested_dtype`, but edge building
//! uses `target_type` from FkKeyConfig for key extraction and hash matching.

use anyhow::{anyhow, Result};
use arrow_array::{
    ArrayRef, Decimal128Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_rowpack::IngestedType;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

// ============================================================================
// Helper Functions
// ============================================================================

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

fn sum_relation_edges(
    table: &diskgraph_index::graph::TableReader,
    rel_name: &str,
) -> Result<u64> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| {
            let names: Vec<_> = table.relations().iter().map(|r| &r.name).collect();
            anyhow!("missing relation {rel_name}. Available: {:?}", names)
        })?;
    let mut total = 0u64;
    for node_id in 0..table.node_count() {
        let view = table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;
        total = total.saturating_add(segments[rel_idx].count as u64);
    }
    Ok(total)
}

/// Verify a column in rowpack is stored as the expected type.
fn assert_rowpack_column_type(
    reader: &GraphIndexReader,
    table_name: &str,
    column_name: &str,
    expected_type: IngestedType,
) -> Result<()> {
    let table = reader
        .table(table_name)
        .ok_or_else(|| anyhow!("table {} not found", table_name))?;
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("no rowpack decoder for {}", table_name))?;
    let ctx = decoder.context();

    let col = ctx
        .columns
        .iter()
        .find(|c| c.name == column_name)
        .ok_or_else(|| anyhow!("column {} not found in {}", column_name, table_name))?;

    assert_eq!(
        col.logical_type, expected_type,
        "Column {}.{} should be stored as {:?}, got {:?}",
        table_name, column_name, expected_type, col.logical_type
    );
    Ok(())
}

// ============================================================================
// Test: Int64 FK → Decimal128 PK
// ============================================================================
//
// This is the primary motivating case for FkKeyConfig:
// - FK: Int64(42) should be stored as Int64 (8 bytes) in rowpack
// - PK: Decimal128(42.00, scale=2) stored as Decimal128 (16 bytes)
// - Edge building: Int64(42) * 10^scale = 4200, then 4200/10^scale = 42
//   which matches PK's key extraction: Decimal128(4200) / 10^2 = 42
//
// Without FkKeyConfig: FK would be stored as Decimal128 (16 bytes) - wasteful!
// With FkKeyConfig: FK stays Int64 (8 bytes), edge keys still match.

#[test]
fn test_int64_fk_to_decimal128_pk() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // PK table: users with Decimal128 id (scale=2)
    // Values: 42.00, 43.00, 44.00 stored as 4200, 4300, 4400
    let users_schema = Schema::new(vec![Field::new(
        "id",
        DataType::Decimal128(38, 2),
        false,
    )]);
    let users_ids = Decimal128Array::from(vec![4200i128, 4300, 4400])
        .with_precision_and_scale(38, 2)?;
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // FK table: orders with Int64 user_id
    // Values: 42, 43, 99 (99 doesn't exist in users)
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3]);
    let user_fks = Int64Array::from(vec![Some(42i64), Some(43), Some(99)]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(user_fks)],
    )?;

    // Config
    let config = serde_json::json!({
        "dataset_name": "test_int64_to_decimal",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "decimal128", "scale": 2, "precision": 38}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "primary_key": "id",
                "fkeys": [{"src_column": "user_id", "dst_table": "users"}]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // KEY ASSERTION 1: FK column stored as Int64 (not Decimal128)
    // This is the whole point - storage savings!
    assert_rowpack_column_type(&reader, "orders", "user_id", IngestedType::Int64)?;

    // KEY ASSERTION 2: Edges created correctly
    // 42 → 42.00 (match), 43 → 43.00 (match), 99 → no match
    let orders_table = reader.table("orders").unwrap();
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_user_id_to_users_id")?,
        2,
        "Expected 2 edges (42→42.00, 43→43.00), 99 has no match"
    );

    Ok(())
}

// ============================================================================
// Test: String FK → Int64 PK (Cross-KeyType)
// ============================================================================
//
// String and Int64 have different KeyTypes (VarBytes vs Int64).
// FkKeyConfig enables parsing string as int64 for key extraction.
// - FK: String "42" stored as String in rowpack
// - PK: Int64(42)
// - Edge building: parse "42" → Int64(42) → Key::Int(42)

#[test]
fn test_string_fk_to_int64_pk() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // PK table: users with Int64 id
    let users_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_ids = Int64Array::from(vec![1i64, 2, 3]);
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // FK table: orders with String user_id
    // "1", "2", "invalid" (invalid won't parse)
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Utf8, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3, 4]);
    let user_fks = StringArray::from(vec![Some("1"), Some("2"), Some("invalid"), None]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(user_fks)],
    )?;

    // Config - note FK is String, PK is Int64
    let config = serde_json::json!({
        "dataset_name": "test_string_to_int64",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "string"}
                ],
                "primary_key": "id",
                "fkeys": [{"src_column": "user_id", "dst_table": "users"}]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // KEY ASSERTION 1: FK column stored as String (not Int64)
    assert_rowpack_column_type(&reader, "orders", "user_id", IngestedType::String)?;

    // KEY ASSERTION 2: Edges created correctly
    // "1" → 1 (match), "2" → 2 (match), "invalid" → parse failure (no edge), null → no edge
    let orders_table = reader.table("orders").unwrap();
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_user_id_to_users_id")?,
        2,
        "Expected 2 edges (\"1\"→1, \"2\"→2); \"invalid\" and null have no edges"
    );

    Ok(())
}

// ============================================================================
// Test: Int32 FK → Int64 PK (Same KeyType, Safe Coercion)
// ============================================================================
//
// Both Int32 and Int64 map to KeyType::Int64.
// Currently this changes FK to Int64. With FkKeyConfig, FK stays Int32.

#[test]
fn test_int32_fk_to_int64_pk() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // PK table: users with Int64 id
    let users_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_ids = Int64Array::from(vec![100i64, 200, 300]);
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // FK table: orders with Int32 user_id
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int32, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3]);
    let user_fks = Int32Array::from(vec![Some(100i32), Some(200), Some(999)]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(user_fks)],
    )?;

    // Config
    let config = serde_json::json!({
        "dataset_name": "test_int32_to_int64",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int32"}
                ],
                "primary_key": "id",
                "fkeys": [{"src_column": "user_id", "dst_table": "users"}]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // KEY ASSERTION 1: FK column stored as Int32 (not Int64)
    // This saves 4 bytes per row (8 bytes Int64 → 4 bytes Int32)
    assert_rowpack_column_type(&reader, "orders", "user_id", IngestedType::Int32)?;

    // KEY ASSERTION 2: Edges created correctly
    let orders_table = reader.table("orders").unwrap();
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_user_id_to_users_id")?,
        2,
        "Expected 2 edges (100, 200 match); 999 has no match"
    );

    Ok(())
}

// ============================================================================
// Test: Same Type FK/PK (Baseline)
// ============================================================================
//
// When FK and PK have the same type, FkKeyConfig is still created
// (for consistency) but no conversion is needed.

#[test]
fn test_same_type_fk_pk() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // PK table: users with Int64 id
    let users_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_ids = Int64Array::from(vec![10i64, 20, 30]);
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // FK table: orders with Int64 user_id (same type)
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3, 4]);
    let user_fks = Int64Array::from(vec![Some(10i64), Some(20), Some(999), None]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(user_fks)],
    )?;

    // Config
    let config = serde_json::json!({
        "dataset_name": "test_same_type",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"}
                ],
                "primary_key": "id",
                "fkeys": [{"src_column": "user_id", "dst_table": "users"}]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // FK column stays Int64 (same as PK)
    assert_rowpack_column_type(&reader, "orders", "user_id", IngestedType::Int64)?;

    // Edges created correctly
    let orders_table = reader.table("orders").unwrap();
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_user_id_to_users_id")?,
        2,
        "Expected 2 edges (10, 20 match); 999 and null have no edges"
    );

    Ok(())
}

// ============================================================================
// Test: Float64 FK → Decimal128 PK
// ============================================================================
//
// Float64 can be coerced to Decimal128 for key matching.
// - FK: Float64(42.0) stored as Float64 in rowpack
// - PK: Decimal128(42.00, scale=2)
// - Edge building: Float64(42.0) → convert to Decimal128 representation → key

#[test]
fn test_float64_fk_to_decimal128_pk() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // PK table: prices with Decimal128 id (scale=2)
    let prices_schema = Schema::new(vec![Field::new(
        "id",
        DataType::Decimal128(38, 2),
        false,
    )]);
    // 10.00, 20.50, 30.99 stored as 1000, 2050, 3099
    let prices_ids = Decimal128Array::from(vec![1000i128, 2050, 3099])
        .with_precision_and_scale(38, 2)?;
    write_parquet(
        &raw_dir.join("prices/part.parquet"),
        prices_schema,
        vec![Arc::new(prices_ids)],
    )?;

    // FK table: orders with Float64 price_id
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("price_id", DataType::Float64, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3, 4]);
    // 10.0 should match 10.00, 20.5 should match 20.50, 99.99 no match
    let price_fks = Float64Array::from(vec![Some(10.0f64), Some(20.5), Some(99.99), None]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(price_fks)],
    )?;

    // Config
    let config = serde_json::json!({
        "dataset_name": "test_float64_to_decimal",
        "tables": [
            {
                "name": "prices",
                "source": {"type": "parquet", "paths": [raw_dir.join("prices/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "decimal128", "scale": 2, "precision": 38}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "price_id", "ingested_dtype": "float64"}
                ],
                "primary_key": "id",
                "fkeys": [{"src_column": "price_id", "dst_table": "prices"}]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest should FAIL: Float64→Decimal128 is Lossy coercion (rejected by default)
    let result = ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    );

    assert!(result.is_err(), "Float64→Decimal128 should be rejected as Lossy coercion");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("lossy") || err.contains("Lossy"),
        "Error should mention lossy coercion, got: {}",
        err
    );

    Ok(())
}

// ============================================================================
// Test: Multiple FK Columns with Different Target Types
// ============================================================================
//
// A single table can have multiple FK columns pointing to different PK types.
// Each FK column gets its own FkKeyConfig entry.

#[test]
fn test_multiple_fk_columns() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // users: Int64 PK
    let users_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_ids = Int64Array::from(vec![1i64, 2, 3]);
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // products: Decimal128 PK (scale=2)
    let products_schema = Schema::new(vec![Field::new(
        "id",
        DataType::Decimal128(38, 2),
        false,
    )]);
    let products_ids = Decimal128Array::from(vec![100i128, 200, 300])
        .with_precision_and_scale(38, 2)?;
    write_parquet(
        &raw_dir.join("products/part.parquet"),
        products_schema,
        vec![Arc::new(products_ids)],
    )?;

    // orders: String user_id FK, Int64 product_id FK
    let orders_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Utf8, true),
        Field::new("product_id", DataType::Int64, true),
    ]);
    let order_ids = Int64Array::from(vec![1i64, 2, 3]);
    let user_fks = StringArray::from(vec![Some("1"), Some("2"), Some("invalid")]);
    // product_id 1, 2 should match Decimal128 1.00, 2.00
    let product_fks = Int64Array::from(vec![Some(1i64), Some(2), Some(999)]);
    write_parquet(
        &raw_dir.join("orders/part.parquet"),
        orders_schema,
        vec![Arc::new(order_ids), Arc::new(user_fks), Arc::new(product_fks)],
    )?;

    // Config
    let config = serde_json::json!({
        "dataset_name": "test_multiple_fks",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "products",
                "source": {"type": "parquet", "paths": [raw_dir.join("products/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "decimal128", "scale": 2, "precision": 38}],
                "primary_key": "id"
            },
            {
                "name": "orders",
                "source": {"type": "parquet", "paths": [raw_dir.join("orders/part.parquet")]},
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "product_id", "ingested_dtype": "int64"}
                ],
                "primary_key": "id",
                "fkeys": [
                    {"src_column": "user_id", "dst_table": "users"},
                    {"src_column": "product_id", "dst_table": "products"}
                ]
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // Both FK columns stored in original types
    assert_rowpack_column_type(&reader, "orders", "user_id", IngestedType::String)?;
    assert_rowpack_column_type(&reader, "orders", "product_id", IngestedType::Int64)?;

    // Edges created correctly for both relations
    let orders_table = reader.table("orders").unwrap();
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_user_id_to_users_id")?,
        2,
        "Expected 2 user edges"
    );
    assert_eq!(
        sum_relation_edges(&orders_table, "orders_product_id_to_products_id")?,
        2,
        "Expected 2 product edges"
    );

    Ok(())
}

// ============================================================================
// Test: Edge Table with FKs to Different PK Types
// ============================================================================
//
// Edge tables have src_fk and dst_fk, each can have different types.

#[test]
fn test_edge_table_fk_types() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    let raw_dir = root.join("raw");
    let index_dir = root.join("index");
    let config_path = root.join("config.json");

    // users: Int64 PK
    let users_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let users_ids = Int64Array::from(vec![1i64, 2, 3]);
    write_parquet(
        &raw_dir.join("users/part.parquet"),
        users_schema,
        vec![Arc::new(users_ids)],
    )?;

    // products: Decimal128 PK (scale=2)
    let products_schema = Schema::new(vec![Field::new(
        "id",
        DataType::Decimal128(38, 2),
        false,
    )]);
    let products_ids = Decimal128Array::from(vec![100i128, 200, 300])
        .with_precision_and_scale(38, 2)?;
    write_parquet(
        &raw_dir.join("products/part.parquet"),
        products_schema,
        vec![Arc::new(products_ids)],
    )?;

    // reviews: edge table user→product
    // user_id: String FK → Int64 PK
    // product_id: Int64 FK → Decimal128 PK
    let reviews_schema = Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("rating", DataType::Int32, false),
    ]);
    let user_fks = StringArray::from(vec!["1", "2", "1"]);
    // 1, 2 should match Decimal128 1.00, 2.00
    let product_fks = Int64Array::from(vec![1i64, 2, 1]);
    let ratings = Int32Array::from(vec![5i32, 4, 3]);
    write_parquet(
        &raw_dir.join("reviews/part.parquet"),
        reviews_schema,
        vec![Arc::new(user_fks), Arc::new(product_fks), Arc::new(ratings)],
    )?;

    // Config with edge table
    let config = serde_json::json!({
        "dataset_name": "test_edge_table",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [raw_dir.join("users/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "primary_key": "id"
            },
            {
                "name": "products",
                "source": {"type": "parquet", "paths": [raw_dir.join("products/part.parquet")]},
                "columns": [{"name": "id", "ingested_dtype": "decimal128", "scale": 2, "precision": 38}],
                "primary_key": "id"
            },
            {
                "name": "reviews",
                "source": {"type": "parquet", "paths": [raw_dir.join("reviews/part.parquet")]},
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "product_id", "ingested_dtype": "int64"},
                    {"name": "rating", "ingested_dtype": "int32"}
                ],
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "users"},
                    "dst_fk": {"edge_column": "product_id", "ref_table": "products"}
                }
            }
        ]
    });

    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    // Ingest
    ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    // Verify
    let reader = GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir)?;

    // Edge tables don't have rowpack storage - they just create edges between nodes.
    // Verify edges exist on user and product tables instead.
    let users_table = reader.table("users").unwrap();
    let products_table = reader.table("products").unwrap();

    // Check that relations exist (edge count verification)
    // The edge table creates bidirectional edges between users and products
    let user_relations: Vec<_> = users_table.relations().iter().map(|r| &r.name).collect();
    let product_relations: Vec<_> = products_table.relations().iter().map(|r| &r.name).collect();

    assert!(
        user_relations.iter().any(|r| r.contains("reviews")),
        "users should have reviews relation, got: {:?}",
        user_relations
    );
    assert!(
        product_relations.iter().any(|r| r.contains("reviews")),
        "products should have reviews relation, got: {:?}",
        product_relations
    );

    Ok(())
}
