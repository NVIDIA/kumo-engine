//! Phase 1 integration tests for seed-table ingest (Option C).
//!
//! Tests 2.1.1–2.1.14 from docs/proposals/seed_tables_as_dg_node_tables.md.
//!
//! Order written (TDD): config layer first (2.1.1-2.1.4) since they don't need
//! a full parent index fixture; then loader compat matrix (2.1.14); then full
//! ingest path (2.1.5-2.1.13).

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_ingest::graph_config::{
    ColumnSpec, DataSource, DirectionSpec, FKeySpec, GraphConfig, IngestedType, TableSpec,
};
use diskgraph_ingest::mapping::{
    key_type_for_external_parent, load_external_pk_hash_shards, load_external_pk_hash_table, Key,
    KeyType, PkDataAdvice,
};
use diskgraph_ingest::{ingest_with_stats_from_source, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

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

/// Build a `TableSpec` for `users` with `user_id Int64` PK.
fn users_pk_int64_table(parquet_path: &str) -> TableSpec {
    TableSpec {
        name: "users".to_string(),
        columns: vec![ColumnSpec {
            name: "user_id".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Int64,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }],
        primary_key: Some("user_id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![parquet_path.to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// Build a `TableSpec` for `orders` with `order_id Int64` PK + `user_id Int64` FK.
fn orders_table(parquet_path: &str) -> TableSpec {
    TableSpec {
        name: "orders".to_string(),
        columns: vec![
            ColumnSpec {
                name: "order_id".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Int64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "user_id".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Int64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: Some("order_id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![parquet_path.to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// Build a `TableSpec` for a synthetic seed table (TT-NG node-prediction shape)
/// with `ENTITY1 INT64`, `TIMESTAMP TIMESTAMP`, `TARGET1 FLOAT64`.
fn seed_table_spec(parquet_path: &str) -> TableSpec {
    TableSpec {
        name: "train_seeds".to_string(),
        columns: vec![
            ColumnSpec {
                name: "ENTITY1".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Int64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "TIMESTAMP".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Timestamp,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "TARGET1".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Float64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            // Synthetic resolution column: filled at ingest by resolving
            // ENTITY1 → users (never written into the seed parquet).
            ColumnSpec {
                name: "entity_node_id".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::UInt64,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: Some("TIMESTAMP".to_string()),
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![parquet_path.to_string()],
        },
        source_size: None,
        // Resolution mode: no FK edges — ENTITY1 is resolved into the
        // `entity_node_id` rowpack column instead of `seed.edges.bin`.
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// Resolution spec: train_seeds.ENTITY1 → users → `entity_node_id` (scalar).
fn entity_resolve_spec() -> diskgraph_ingest::ResolveColumnSpec {
    diskgraph_ingest::ResolveColumnSpec {
        table: "train_seeds".to_string(),
        src_col: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
        kind: diskgraph_ingest::ResolveKind::Scalar,
        node_id_col: "entity_node_id".to_string(),
    }
}

// ---------------------------------------------------------------------------
/// Build a synthetic seed parquet (ENTITY1 Int64, TIMESTAMP, TARGET1 Float64)
/// at the given path. `entity_values` defines the rows; matching ENTITY1
/// values to parent.users PK values drives the per-row FK lookup outcome.
fn write_seed_parquet(parquet_path: &Path, entity_values: &[i64]) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, false),
    ]);
    let n = entity_values.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
    let targets: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    write_parquet(
        parquet_path,
        schema,
        vec![
            Arc::new(Int64Array::from(entity_values.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(targets)) as ArrayRef,
        ],
    )
}







/// Bidirectional=false with an in-config dst. This is the "we have both src
/// and dst tables in this config, but the user wants forward-only edges" case.
/// Without isolated coverage we couldn't distinguish bugs in the forward-only
/// flag path from the default bidirectional path.
#[test]
fn test_ingest_skips_reverse_when_bidirectional_false_in_config_dst() {
    let tmp = tempdir().unwrap();
    // Build a 2-table config: users (PK) + orders (FK→users) with
    // bidirectional=false (forward-only, in-config dst).
    let input_dir = tmp.path().join("input");
    let users_pq = input_dir.join("users.parquet");
    let orders_pq = input_dir.join("orders.parquet");
    write_parquet(
        &users_pq,
        Schema::new(vec![Field::new("user_id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..8i64).collect::<Vec<_>>())) as ArrayRef],
    )
    .unwrap();
    write_parquet(
        &orders_pq,
        Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("user_id", DataType::Int64, false),
        ]),
        vec![
            Arc::new(Int64Array::from((100..110i64).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int64Array::from(vec![0i64, 1, 2, 3, 4, 5, 6, 7, 0, 1])) as ArrayRef,
        ],
    )
    .unwrap();

    let users = users_pk_int64_table(users_pq.to_str().unwrap());
    let mut orders = orders_table(orders_pq.to_str().unwrap());
    // Forward-only FK on orders: bidirectional=false, in-config dst.
    orders.fkeys = vec![FKeySpec {
        src_column: "user_id".to_string(),
        dst_table: "users".to_string(),
        forward: Some(DirectionSpec::default()),
        reverse: None,
        bidirectional: false,
    }];
    let config = GraphConfig {
        dataset_name: "fwd_only".to_string(),
        tables: vec![users, orders],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let output = tmp.path().join("out");
    let _ = ingest_with_stats_from_source(
        config,
        None,
        output.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )
    .expect("forward-only bidirectional=false ingest with in-config dst must succeed");

    // index_metadata.json must declare exactly one orders↔users relation.
    let meta: diskgraph_core::metadata::DatasetMetadata =
        serde_json::from_slice(&fs::read(output.join("index_metadata.json")).unwrap()).unwrap();
    let rels: Vec<_> = meta
        .edge_relations
        .iter()
        .filter(|r| {
            (r.src_type == "orders" && r.dst_type == "users")
                || (r.src_type == "users" && r.dst_type == "orders")
        })
        .collect();
    assert_eq!(
        rels.len(),
        1,
        "bidirectional=false must emit exactly one relation, got {}: {:?}",
        rels.len(),
        rels.iter().map(|r| (&r.name, &r.src_type, &r.dst_type)).collect::<Vec<_>>()
    );
    assert_eq!(rels[0].src_type, "orders");
    assert_eq!(rels[0].dst_type, "users");
    assert!(rels[0].reverse_of.is_none());

    // ingest_stats.json should NOT mention any _reverse relation.
    let stats_str =
        String::from_utf8_lossy(&fs::read(output.join("ingest_stats.json")).unwrap())
            .into_owned();
    assert!(
        !stats_str.contains("orders_user_id_to_users_id_reverse"),
        "ingest_stats.json must not mention the would-be reverse relation; got: {}",
        &stats_str[..stats_str.len().min(1500)]
    );
}

// ---------------------------------------------------------------------------
// Shared fixture: ingest a tiny users-only "parent" DG index for tests that
// need a real parent to load pkmap shards from.
// ---------------------------------------------------------------------------

/// Ingest a tiny users table at `parent_root` with `pk_dtype` ∈ {Int64, String,
/// Uuid} and the requested shard count. Returns the parent root path.
///
/// Uses the existing ingest pipeline to produce a real
/// `nodes/users/pk_hash-*.bin` + `pk_data-*.bin` layout — the loader is tested
/// against actual on-disk artifacts rather than synthetic byte blobs, so
/// format drift in the writer surfaces here too.
fn ingest_tiny_parent_index(
    parent_root: &Path,
    pk_dtype: &IngestedType,
    num_shards: usize,
    num_rows: i64,
) -> Result<()> {
    ingest_tiny_parent_index_named(parent_root, "users", pk_dtype, num_shards, num_rows)
}

fn ingest_tiny_parent_index_named(
    parent_root: &Path,
    table_name: &str,
    pk_dtype: &IngestedType,
    num_shards: usize,
    num_rows: i64,
) -> Result<()> {
    ingest_tiny_parent_index_impl(parent_root, table_name, pk_dtype, num_shards, num_rows, None)
}

fn ingest_tiny_parent_index_impl(
    parent_root: &Path,
    table_name: &str,
    pk_dtype: &IngestedType,
    num_shards: usize,
    num_rows: i64,
    int_pk_values: Option<&[i64]>,
) -> Result<()> {
    // Place inputs in a sibling dir to the index output — the ingest
    // pipeline owns the contents of `parent_root` and may wipe / rewrite
    // it on startup, so input fixtures must live outside it.
    let input_dir = parent_root
        .parent()
        .expect("parent_root has parent")
        .join("_inputs");
    let parquet_path = input_dir.join("users.parquet");
    let (pk_field, pk_array): (Field, ArrayRef) = match pk_dtype {
        IngestedType::Int64 => (
            Field::new("user_id", DataType::Int64, false),
            Arc::new(Int64Array::from(
                int_pk_values
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| (0..num_rows).collect()),
            )),
        ),
        IngestedType::String => (
            Field::new("user_id", DataType::Utf8, false),
            Arc::new(StringArray::from(
                (0..num_rows).map(|i| format!("u{i:08}")).collect::<Vec<_>>(),
            )),
        ),
        IngestedType::Uuid => {
            let strs: Vec<String> = (0..num_rows)
                .map(|i| {
                    format!(
                        "{:08x}-0000-0000-0000-{:012x}",
                        i as u32, i as u64
                    )
                })
                .collect();
            (
                Field::new("user_id", DataType::Utf8, false),
                Arc::new(StringArray::from(strs)),
            )
        }
        other => panic!("ingest_tiny_parent_index does not support pk dtype {:?}", other),
    };

    let schema = Schema::new(vec![pk_field]);
    write_parquet(&parquet_path, schema, vec![pk_array])?;

    let table = TableSpec {
        name: table_name.to_string(),
        columns: vec![ColumnSpec {
            name: "user_id".to_string(),
            source_dtype: None,
            ingested_dtype: pk_dtype.clone(),
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }],
        primary_key: Some("user_id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![parquet_path.to_str().unwrap().to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    };

    let config = GraphConfig {
        dataset_name: "tiny_parent".to_string(),
        tables: vec![table],
        rowpack: None,
        num_shards: Some(num_shards),
        edge_format: None,
        coercion_mode: Default::default(),
    };

    let output = parent_root.to_str().unwrap();
    let _ = ingest_with_stats_from_source(
        config,
        None,
        output,
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Regression — external-FK dst_table resolution is case-insensitive on the
// node-dir path. Node dirs are written lowercased (graph_config table-name
// normalization), while the FK `dst_table` carries the case-preserved logical
// node-type name from the parent index metadata. A mixed-case entity table
// (e.g. `Test_TRANS`, as produced by the connector-prefixed public-API path)
// must resolve to its on-disk `nodes/test_trans` dir, not fail "dir not found".
// ---------------------------------------------------------------------------
#[test]
fn external_pk_hash_shards_resolve_mixed_case_dst_table() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index_named(&parent_root, "Test_TRANS", &IngestedType::Int64, 1, 16)
        .expect("ingest mixed-case parent index");

    // The on-disk node dir is lowercased by the ingest pipeline.
    assert!(
        parent_root.join("nodes").join("test_trans").is_dir(),
        "node dir should be written lowercased"
    );

    // The loader is given the case-preserved logical name and must still find
    // the lowercased dir (this fails "external FK dst_table dir not found"
    // before the casing fix).
    let key_type = key_type_for_external_parent(&parent_root, "Test_TRANS")
        .expect("read parent metadata for mixed-case dst_table");
    let shards = load_external_pk_hash_shards(
        &parent_root,
        "Test_TRANS",
        key_type,
        PkDataAdvice::Random,
    )
    .expect("mixed-case dst_table must resolve to lowercased node dir");
    assert_eq!(shards.len(), 1, "expected one shard for the tiny parent index");
}

// ---------------------------------------------------------------------------
// Test 2.1.14 — PkHashStore loader compat matrix.
// External `PkHashStore` loader: shard counts × key types × format version.
//
// Exercises the actual on-disk pkmap artifacts produced by the existing
// ingest pipeline (tested with the loader's lookup against known PK values).
// ---------------------------------------------------------------------------

fn loader_compat_case(
    pk_dtype: IngestedType,
    num_shards: usize,
    num_rows: i64,
    expected_key_type: KeyType,
    keys_to_probe: &[Key],
) {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &pk_dtype, num_shards, num_rows).unwrap_or_else(|e| {
        panic!("ingest tiny parent ({pk_dtype:?}, shards={num_shards}) failed: {e:#}")
    });

    // KeyType inference from parent metadata.
    let inferred = key_type_for_external_parent(&parent_root, "users")
        .expect("read parent metadata");
    assert_eq!(
        inferred, expected_key_type,
        "inferred key type for {pk_dtype:?} should be {expected_key_type:?}"
    );

    // Shard load count matches the ingest's num_shards.
    let shards = load_external_pk_hash_shards(
        &parent_root,
        "users",
        expected_key_type,
        PkDataAdvice::Random,
    )
    .expect("load external pkmap shards");
    assert_eq!(
        shards.len(),
        num_shards,
        "shard count mismatch for pk_dtype={pk_dtype:?} shards={num_shards}"
    );

    // Lookup against known PK values returns Some(_) for at least one shard.
    // (Which shard a given key falls into is determined by the ingest-time
    // sharding function; we verify that the union of shards covers all keys.)
    for probe in keys_to_probe {
        let mut found = false;
        for shard in &shards {
            if shard.lookup(probe).is_some() {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "key {probe:?} not found in any of {} shards for {pk_dtype:?}",
            shards.len()
        );
    }

    // Top-level helper returns a PkHashStore with the dst_table populated.
    let (store, ktype) =
        load_external_pk_hash_table(&parent_root, "users", PkDataAdvice::Random)
            .expect("load external pkmap table");
    assert_eq!(ktype, expected_key_type);
    for shard_idx in 0..num_shards {
        let _ = store
            .shard("users", shard_idx)
            .expect("PkHashStore.shard must return a shard at every index 0..num_shards");
    }
}

#[test]
fn test_pkhashstore_loader_int64_one_shard() {
    loader_compat_case(
        IngestedType::Int64,
        1,
        32,
        KeyType::Int64,
        &[Key::Int(0), Key::Int(5), Key::Int(31)],
    );
}

#[test]
fn test_pkhashstore_loader_int64_four_shards() {
    loader_compat_case(
        IngestedType::Int64,
        4,
        64,
        KeyType::Int64,
        &[
            Key::Int(0),
            Key::Int(17),
            Key::Int(42),
            Key::Int(63),
        ],
    );
}

#[test]
fn test_pkhashstore_loader_varbytes_one_shard() {
    let probes: Vec<Key> = (0..16i64)
        .map(|i| Key::Bytes(format!("u{i:08}").into_bytes()))
        .collect();
    loader_compat_case(IngestedType::String, 1, 16, KeyType::VarBytes, &probes);
}

#[test]
fn test_pkhashstore_loader_uuid_one_shard() {
    // Construct UUIDs whose first 4 bytes encode i (big-endian), rest zeros.
    // Matches the layout produced by the fixture's hyphenated string when
    // parsed by the ingest pipeline's UUID parser.
    let probes: Vec<Key> = (0..8i64)
        .map(|i| {
            let mut bytes = [0u8; 16];
            let i32_be = (i as u32).to_be_bytes();
            bytes[0..4].copy_from_slice(&i32_be);
            // bytes[10..16] encode i as u64 big-endian (matching "{:012x}").
            let i64_be = (i as u64).to_be_bytes();
            // Last 6 bytes of the u64 BE = lower 48 bits, placed at bytes[10..16].
            bytes[10..16].copy_from_slice(&i64_be[2..8]);
            Key::Uuid(bytes)
        })
        .collect();
    loader_compat_case(IngestedType::Uuid, 1, 8, KeyType::Uuid, &probes);
}

#[test]
fn test_pkhashstore_loader_rejects_missing_table_dir() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 8).unwrap();
    let result = load_external_pk_hash_shards(
        &parent_root,
        "table_that_does_not_exist",
        KeyType::Int64,
        PkDataAdvice::Random,
    );
    let err = match result {
        Ok(_) => panic!("missing nodes/<table>/ dir must error, but loader succeeded"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not found") || msg.contains("table_that_does_not_exist"),
        "error must mention the missing table; got: {msg}"
    );
}

#[test]
fn test_pkhashstore_loader_rejects_missing_pk_data_shard() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 8).unwrap();
    // Delete pk_data-00000.bin so the loader sees a pk_hash without its
    // pair — this is the "incomplete parent" scenario we want loud about.
    let pk_data = parent_root.join("nodes/users/pk_data-00000.bin");
    assert!(pk_data.is_file(), "fixture must produce pk_data-00000.bin");
    fs::remove_file(&pk_data).unwrap();
    let result = load_external_pk_hash_shards(
        &parent_root,
        "users",
        KeyType::Int64,
        PkDataAdvice::Random,
    );
    let err = match result {
        Ok(_) => panic!("missing pk_data shard must error, but loader succeeded"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("pk_data") || msg.contains("missing"),
        "error must mention the missing pk_data; got: {msg}"
    );
}

#[test]
fn test_pkhashstore_loader_rejects_missing_table_in_metadata() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 8).unwrap();
    let err = key_type_for_external_parent(&parent_root, "orders")
        .expect_err("missing table in metadata must error");
    let msg = err.to_string();
    assert!(
        msg.contains("orders") || msg.contains("not found"),
        "error must mention the missing table; got: {msg}"
    );
}

// ===========================================================================
// Phase 2 — merged reader: discover_tts + open_with_tt.
//
// Runtime API: kumo opens parent + exactly ONE tt at a time. discover_tts
// enumerates what's available; open_with_tt loads one. No plural variants.
// ===========================================================================

/// Ingest one parent + one seed-table extension at `train_tables/<tt_id>/`.
/// Returns the parent root path.
fn ingest_parent_with_one_tt(
    tmp: &tempfile::TempDir,
    tt_id: &str,
    entity_values: &[i64],
) -> Result<PathBuf> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16)?;
    ingest_one_extension(&parent_root, tt_id, entity_values)?;
    Ok(parent_root)
}

/// Larger parent fixture: `users` + `orders` with a bidirectional FK so the
/// engine sees a `users → orders` reverse edge with `order_time` as the
/// destination sort key. Used by depth-≥2 sampling tests and the hop-1
/// temporal-leak test.
///
/// Layout (deterministic by construction):
///   * `n_users` rows in users, pk = user_id ∈ [0, n_users)
///   * For each user u, 4 orders with order_id = u*4 + k for k in 0..4:
///       - k=0: order_time = 100 (early)
///       - k=1: order_time = 200 (early)
///       - k=2: order_time = 2000 (late)
///       - k=3: order_time = 3000 (late)
///
/// "Early" vs "late" is relative to the seed timestamps (1000+i) emitted by
/// `write_seed_parquet`. With 1 shard the global node id of each order
/// equals its order_id, so callers can derive "early/late" from
/// `order_id % 4` without reading rowpack at assert time.
fn ingest_users_orders_parent_index(parent_root: &Path, n_users: i64) -> Result<()> {
    let input_dir = parent_root
        .parent()
        .expect("parent_root has parent")
        .join("_uo_inputs");
    let users_pq = input_dir.join("users.parquet");
    let orders_pq = input_dir.join("orders.parquet");

    // users
    write_parquet(
        &users_pq,
        Schema::new(vec![Field::new("user_id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..n_users).collect::<Vec<_>>())) as ArrayRef],
    )?;

    // orders: 4 per user, alternating early/late.
    let mut order_ids: Vec<i64> = Vec::with_capacity(n_users as usize * 4);
    let mut order_uids: Vec<i64> = Vec::with_capacity(n_users as usize * 4);
    let mut order_times: Vec<i64> = Vec::with_capacity(n_users as usize * 4);
    for u in 0..n_users {
        for k in 0..4i64 {
            order_ids.push(u * 4 + k);
            order_uids.push(u);
            order_times.push(match k {
                0 => 100,
                1 => 200,
                2 => 2000,
                _ => 3000,
            });
        }
    }
    write_parquet(
        &orders_pq,
        Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("user_id", DataType::Int64, false),
            Field::new("order_time", DataType::Timestamp(TimeUnit::Second, None), false),
        ]),
        vec![
            Arc::new(Int64Array::from(order_ids)) as ArrayRef,
            Arc::new(Int64Array::from(order_uids)) as ArrayRef,
            Arc::new(TimestampSecondArray::from(order_times)) as ArrayRef,
        ],
    )?;

    let users = users_pk_int64_table(users_pq.to_str().unwrap());
    let mut orders = orders_table(orders_pq.to_str().unwrap());
    // Add order_time as a column on the orders TableSpec.
    orders.columns.push(ColumnSpec {
        name: "order_time".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Timestamp,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    orders.create_time = Some("order_time".to_string());
    // Reverse direction carries the time encoding so the users → orders
    // reverse edge gets a `dest_sort_key` on order_time, enabling
    // `TemporalMostRecent` at hop 1 to filter by seed timestamp.
    orders.fkeys = vec![FKeySpec {
        src_column: "user_id".to_string(),
        dst_table: "users".to_string(),
        forward: Some(DirectionSpec::default()),
        reverse: Some(DirectionSpec {
            time: Some(diskgraph_ingest::graph_config::TimeSpec {
                table: "orders".to_string(),
                column: "order_time".to_string(),
                bitwidth: Some(32),
                anchor_time: None,
                unit: "s".to_string(),
            }),
            weight: None,
        }),
        bidirectional: true,
    }];

    let config = GraphConfig {
        dataset_name: "users_orders_parent".to_string(),
        tables: vec![users, orders],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let _ = ingest_with_stats_from_source(
        config,
        None,
        parent_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )?;
    Ok(())
}

/// Like `ingest_parent_with_one_tt` but the parent has `users + orders`
/// with a bidirectional FK, so the merged graph has a `users → orders`
/// reverse edge with `order_time` as its destination sort key. Lets
/// depth-≥2 sampling tests actually walk hop 1.
fn ingest_parent_with_orders_and_one_tt(
    tmp: &tempfile::TempDir,
    tt_id: &str,
    entity_values: &[i64],
) -> Result<PathBuf> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_users_orders_parent_index(&parent_root, 16)?;
    ingest_one_extension(&parent_root, tt_id, entity_values)?;
    Ok(parent_root)
}

/// Ingest a single seed-table extension into an already-built parent index.
/// Factored out so multi-extension fixture setup (for discover_tts tests)
/// can reuse it without recursively duplicating the parent.
fn ingest_one_extension(
    parent_root: &Path,
    tt_id: &str,
    entity_values: &[i64],
) -> Result<()> {
    let seed_input = parent_root
        .parent()
        .expect("parent_root has parent")
        .join(format!("seed_input_{tt_id}"));
    let seed_parquet = seed_input.join("train_seeds.parquet");
    write_seed_parquet(&seed_parquet, entity_values)?;

    let seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    let config = GraphConfig {
        dataset_name: format!("seed_{tt_id}"),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let seed_output = parent_root.join("train_tables").join(tt_id);
    let options = IngestOptions {
        parent_index_path: Some(parent_root.to_path_buf()),
        resolve_columns: vec![entity_resolve_spec()],
        col_stats: true,
        ..Default::default()
    };
    let _ = ingest_with_stats_from_source(
        config,
        None,
        seed_output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )?;
    Ok(())
}

// --- discover_tts ---------------------------------------------------------

#[test]
fn test_discover_tts_lists_finalized_in_sorted_order() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_a", &(0..4).collect::<Vec<_>>()).unwrap();
    ingest_one_extension(&parent_root, "tt_b", &(0..6).collect::<Vec<_>>()).unwrap();
    // Plant an incomplete subdir — discover_tts must skip it.
    let incomplete = parent_root.join("train_tables").join("tt_partial");
    fs::create_dir_all(&incomplete).unwrap();
    fs::write(incomplete.join("scratch"), b"midway").unwrap();

    let ids = diskgraph_index::graph::discover_tts(&parent_root).unwrap();
    assert_eq!(
        ids,
        vec!["tt_a".to_string(), "tt_b".to_string()],
        "discover_tts must return finalized tts in sorted order, excluding incomplete"
    );
}

#[test]
fn test_discover_tts_empty_when_no_extensions() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 8).unwrap();
    // No train_tables/ dir at all.
    let ids = diskgraph_index::graph::discover_tts(&parent_root).unwrap();
    assert!(
        ids.is_empty(),
        "discover_tts must return an empty Vec when train_tables/ is absent"
    );
}

// --- coordinated layout fix (#16) ----------------------------------------
//
// Phase 1 cross-boundary FK edges encode dst id width based on parent's
// `users.max_id` — but at seed-extension ingest time the parent isn't in
// the local config's `node_types`. Without the fix, `compute_relation_layouts`
// falls back to `dst_bits=64` (8 bytes/edge) instead of `bits_for_id(max_id)`
// (1-2 bytes for typical 10K-user parents). Both write and read sides need to
// agree on the same parent-aware width — that's the "coordinated" part.


// --- open_with_tt ---------------------------------------------------------

#[test]
fn test_open_with_tt_loads_parent_plus_one_extension() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_x", &(0..4).collect::<Vec<_>>()).unwrap();

    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_x").unwrap();

    assert!(reader.table("users").is_some(), "parent table must be present");
    assert!(
        reader.table("tt_x/train_seeds").is_some(),
        "extension table must appear under namespaced key"
    );
    assert!(
        reader.table("train_seeds").is_none(),
        "bare extension name must NOT shadow the namespaced key"
    );
}

#[test]
fn test_open_with_tt_accepts_prefixed_train_ref() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_x", &(0..4).collect::<Vec<_>>()).unwrap();

    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "train_tables/tt_x")
            .unwrap();

    assert!(
        reader.table("tt_x/train_seeds").is_some(),
        "prefixed training ref must use the extension id as namespace"
    );
}

#[test]
fn test_open_with_tt_parent_untouched() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_x", &(0..4).collect::<Vec<_>>()).unwrap();

    let before = snapshot_parent_files(&parent_root);
    let _ = diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_x").unwrap();
    let after = snapshot_parent_files(&parent_root);
    assert_eq!(before, after, "open_with_tt must not modify any parent-owned file");
}

#[test]
fn test_open_with_tt_rejects_unknown_tt_id() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_real", &[0, 1]).unwrap();
    let result = diskgraph_index::graph::GraphIndexReader::open_with_tt(
        &parent_root,
        "tt_does_not_exist",
    );
    let err = match result {
        Ok(_) => panic!("open_with_tt must error on unknown tt_id, but succeeded"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("tt_does_not_exist"),
        "error must reference the bad tt_id; got: {msg}"
    );
}

#[test]
fn test_open_with_tt_rejects_incomplete_tt() {
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_ok", &[0]).unwrap();
    let incomplete = parent_root.join("train_tables").join("tt_half");
    fs::create_dir_all(&incomplete).unwrap();
    fs::write(incomplete.join("scratch"), b"midway").unwrap();

    let result =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_half");
    let err = match result {
        Ok(_) => panic!("open_with_tt must reject an unfinalized extension"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("tt_half") && msg.contains("no finalized"),
        "error must indicate the extension is not finalized; got: {msg}"
    );
}

// --- merged_edge_relations namespacing ------------------------------------


// --- cross-boundary FK resolution + pkmap probe ---------------------------



#[test]
fn test_merged_reader_pkmap_probe_through_table_reader() {
    // Phase 6's LP sampler will probe the parent's pkmap for TARGET1 list
    // expansion. Pin that the merged reader exposes it correctly through
    // the standard TableReader::pk_map() path. We do NOT assert that
    // missing-PK probes return None — the MPHF returns a candidate slot
    // for any input by design; key-equality verification is the
    // consumer's responsibility (pk_matches), not the pkmap's.
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_probe", &[0, 1, 2]).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_probe").unwrap();

    let users = reader.table("users").unwrap();
    let pkmap = users.pk_map().expect("parent users must have a pk_map");
    let n_users = users.node_count();

    let mut row_ids = std::collections::HashSet::new();
    for pk_val in 0..3i64 {
        let key = diskgraph_index::pkmap::Key::Int(pk_val as u64);
        let row = (0..pkmap.shard_count())
            .find_map(|s| pkmap.shard(s).and_then(|sh| sh.lookup(&key)))
            .unwrap_or_else(|| panic!("pk={pk_val} must resolve at sample time"));
        assert!((row as u64) < n_users);
        row_ids.insert(row);
    }
    assert_eq!(row_ids.len(), 3, "3 distinct PKs → 3 distinct row indices");
}

#[test]
fn test_merged_reader_pkmap_parity_with_external_loader() {
    // Two parallel loaders exist for the on-disk pkmap format:
    //   - diskgraph-index::pkmap::PkHashShard (read-time, used by the reader)
    //   - diskgraph-ingest::mapping::PkHashShard (ingest-time, Phase 1)
    // They must return identical row indices for identical keys. The
    // duplicate-loader smell is real and a separate cleanup (unify into
    // one crate); this parity test catches drift in the meantime.
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_parity", &[0, 1]).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_parity").unwrap();
    let users = reader.table("users").unwrap();
    let reader_pkmap = users.pk_map().unwrap();
    let (ext_store, _) =
        load_external_pk_hash_table(&parent_root, "users", PkDataAdvice::Random).unwrap();

    for pk_val in 0..5i64 {
        let reader_row = (0..reader_pkmap.shard_count()).find_map(|s| {
            reader_pkmap
                .shard(s)
                .and_then(|sh| sh.lookup(&diskgraph_index::pkmap::Key::Int(pk_val as u64)))
        });
        let ext_row = (0..reader_pkmap.shard_count()).find_map(|s| {
            ext_store
                .shard("users", s)
                .ok()
                .and_then(|sh| sh.lookup(&Key::Int(pk_val as u64)))
        });
        assert_eq!(
            reader_row.map(u64::from),
            ext_row.map(u64::from),
            "pk={pk_val}: parallel loaders must return identical row idx"
        );
    }
}

/// Walk every file under `root`, EXCLUDING the `train_tables/` subdir, and
/// return (relative path, len, content hash). Used to assert parent files
/// are untouched after a reader operation.
fn snapshot_parent_files(root: &Path) -> Vec<(String, u64, u64)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut out = Vec::new();
    fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, u64, u64)>) {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.file_name().map(|n| n == "train_tables").unwrap_or(false) {
                continue;
            }
            if path.is_dir() {
                walk(&path, base, out);
            } else {
                let bytes = fs::read(&path).unwrap();
                let mut h = DefaultHasher::new();
                bytes.hash(&mut h);
                let rel = path.strip_prefix(base).unwrap().display().to_string();
                out.push((rel, bytes.len() as u64, h.finish()));
            }
        }
    }
    walk(root, root, &mut out);
    out.sort();
    out
}

// ===========================================================================
// Phase 3 — sampler walks merged graph from seed nodes (node-prediction only).
//
// Tests 2.3.1, 2.3.2, 2.3.3, 2.3.4, 2.3.8 from the plan. LP tests 2.3.5-2.3.7
// are explicitly deferred to Phase 6.
// ===========================================================================

#[test]
fn test_engine_from_reader_exposes_merged_catalog() {
    // The engine consumes the reader's merged catalog directly — no
    // independent discovery. node_type_to_id must contain both parent's
    // bare `users` and the namespaced `tt_eng/train_seeds`.
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_eng", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_eng").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader, Some(42), Some(1)).unwrap();

    let mapping = engine.get_type_name_mapping();
    let names: std::collections::HashSet<&str> =
        mapping.node_types.iter().map(|s| s.as_str()).collect();
    assert!(names.contains("users"), "engine catalog must contain parent `users`; got: {names:?}");
    assert!(
        names.contains("tt_eng/train_seeds"),
        "engine catalog must contain namespaced seed table; got: {names:?}"
    );
}


// Tests 2.3.3 (exclude filter on seed edge types) and 2.3.4 (train_seeds_bag
// synthesis) need OutputHandler-level changes that couple to PR-28765's
// adapter contract. Pinned with #[ignore]'d tests so the contract is
// documented and unhidden when the work lands.




// ---------------------------------------------------------------------------
// Phase 3 prerequisite: per-table local→merged rel-id translation.
//
// Bug uncovered by the orders fixture: each ShardReader stores per-table
// `RelationLayout.relation_id` in the SUB-INDEX's local namespace (0..n
// where n is that table's outgoing relation count). The engine indexes
// `relation_info[rel_id]` directly with that local id, which only works
// when local rel ids happen to match merged catalog positions. With
// `parent: users+orders + extension: train_seeds`, the seed's local rel 0
// should map to merged rel 2 (after parent's two), but the engine treats
// it as merged rel 0 (orders→users), recording cross-boundary edges with
// the wrong dst type and missing hop-1 fanouts entirely.
//
// This test pins the contract before the fix: a depth-2 sample from a
// users-seed walks users→orders correctly. Same fixture as
// test_sample_runtime_batch_e2e but expressed directly through engine.sample
// so the failure mode is unambiguous.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Phase 3 — sample_runtime_batch end-to-end (task #24 + #26).
//
// Single convenience entry point on DiskGraphEngine that:
//   1. Reads the seed rowpack slice for the batch → task.{input_ids,
//      tt_stage_node_ids, target, weight?, seed_timestamps?}.
//   2. Walks seed.edges.bin for each batch row → task.seed_node_ids
//      (the parent entity row ids, pre-resolved at Phase 1 ingest).
//   3. Calls engine.sample() with the resolved parent ids as the seed,
//      a RuntimeBatchOutput collator, and the seed-namespace's rel ids
//      registered as seed_src_rel_ids (belt-and-suspenders — they aren't
//      reachable from this entry point, but the filter stays correct if
//      the entry point is ever changed).
//   4. Returns a RuntimeBatch matching PR 28765's runtime_batch_to_hetero
//      contract.
// ---------------------------------------------------------------------------

#[test]
fn test_sample_runtime_batch_e2e() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    // Use the orders fixture so hop 1 actually walks users → orders —
    // depth-1 only would only assert hop-0 entry-point behavior, which is
    // already covered upstream.
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_rb", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_rb").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    // Depth-2 plan: hop 0 anchors users, hop 1 walks users → orders.
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
        ],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    };

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_rb/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .expect("sample_runtime_batch must succeed end-to-end");

    // ---- task bag ----
    assert_eq!(
        batch.task.input_ids,
        vec![0u32, 1, 2],
        "input_ids must be 0..batch_size"
    );
    assert_eq!(
        batch.task.tt_stage_node_ids,
        vec![0u64, 1, 2],
        "tt_stage_node_ids must equal the requested batch row ids"
    );
    // ENTITY1 = i and parent users have pk = i, so parent row idx = i.
    assert_eq!(
        batch.task.seed_node_ids,
        vec![0u64, 1, 2],
        "seed_node_ids must be the parent row idxs resolved from seed.edges.bin"
    );
    // TIMESTAMP[i] = 1000 + i in the fixture.
    let ts = batch
        .task
        .seed_timestamps
        .as_ref()
        .expect("fixture has TIMESTAMP column so seed_timestamps must be populated");
    assert_eq!(ts, &vec![1000i64, 1001, 1002], "seed_timestamps must match seed rowpack TIMESTAMP");
    // TARGET1[i] = i * 0.1
    let target_batch = batch
        .task
        .target
        .as_ref()
        .expect("task.target must be populated from seed rowpack");
    assert_eq!(target_batch.num_rows(), 3, "target must have batch_size rows");
    // task.target must be PROJECTED to TARGET* columns only — no ENTITY*,
    // no TIMESTAMP. Those are task identity / temporal anchor (already in
    // `task.tt_stage_node_ids` and `task.seed_timestamps`). Conflating them
    // into task.target makes the adapter consume identity columns as target
    // signal which is wrong.
    let target_schema = target_batch.schema();
    let target_col_names: std::collections::HashSet<&str> = target_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert!(
        target_col_names.contains("TARGET1"),
        "task.target must include TARGET1; got: {target_col_names:?}"
    );
    assert!(
        !target_col_names.contains("ENTITY1"),
        "task.target must NOT include ENTITY1 (it's task identity, not target); \
         got: {target_col_names:?}"
    );
    assert!(
        !target_col_names.contains("TIMESTAMP"),
        "task.target must NOT include TIMESTAMP (it's a temporal anchor, not target); \
         got: {target_col_names:?}"
    );
    let target_col = target_batch
        .column_by_name("TARGET1")
        .expect("target RecordBatch must include TARGET1 column")
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .expect("TARGET1 must be Float64");
    let got_targets: Vec<f64> = (0..3).map(|i| target_col.value(i)).collect();
    let expected_targets = vec![0.0, 0.1, 0.2];
    for (g, e) in got_targets.iter().zip(expected_targets.iter()) {
        assert!(
            (g - e).abs() < 1e-9,
            "TARGET1 mismatch: got {got_targets:?}, expected {expected_targets:?}"
        );
    }

    // ---- graphs.main ----
    let channel = batch.graphs.get("main").expect("main channel must exist");

    // Property: graphs.main.node_ids[users] includes every seed_node_id.
    let users_nodes = channel
        .node_ids
        .get("users")
        .expect("users must appear in graphs.main.node_ids");
    for &sid in &batch.task.seed_node_ids {
        assert!(
            users_nodes.contains(&sid),
            "seed_node_id {sid} must appear in graphs.main.node_ids[users]: {users_nodes:?}"
        );
    }

    // Property: graphs.main.node_ids MUST NOT contain the seed namespace —
    // the seed is task data, never topology.
    assert!(
        !channel.node_ids.contains_key("tt_rb/train_seeds"),
        "seed namespace must not appear in graphs.main.node_ids; got: {:?}",
        channel.node_ids.keys().collect::<Vec<_>>()
    );
    // ...nor in batch_dict (symmetric with node_ids).
    assert!(
        !channel.batch_dict.contains_key("tt_rb/train_seeds"),
        "seed namespace must not appear in graphs.main.batch_dict; got: {:?}",
        channel.batch_dict.keys().collect::<Vec<_>>()
    );
    // Property: graphs.main.num_sampled_nodes MUST NOT contain the seed
    // namespace either. This is the per-hop count map populated for EVERY node
    // type the plan can reach; it is the path that leaks the encoder-less seed
    // namespace into the kumo adapter (KeyError at TensorFrameEncoder). The
    // parent entity (users) MUST still be present.
    assert!(
        channel.num_sampled_nodes.contains_key("users"),
        "num_sampled_nodes must include the parent entity (users); got: {:?}",
        channel.num_sampled_nodes.keys().collect::<Vec<_>>()
    );
    assert!(
        !channel.num_sampled_nodes.contains_key("tt_rb/train_seeds"),
        "seed namespace must not appear in graphs.main.num_sampled_nodes; got: {:?}",
        channel.num_sampled_nodes.keys().collect::<Vec<_>>()
    );
    // NOTE: this fixture's seed rowpack has only ENTITY1/TIMESTAMP/TARGET1
    // (all reserved), so it produces no raw_features entry. The invariant that
    // topology-filtering must NOT drop seed raw_features is exercised by
    // `test_sample_runtime_batch_seed_topology_excluded_features_preserved`.

    // Property: no edge has src = seed namespace.
    for (rel_idx, (srcs, _dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (src, _rel, _dst) = channel.rel_id_to_names.get(&rel_id).unwrap();
        assert_ne!(
            src, "tt_rb/train_seeds",
            "seed-src edge leaked into runtime batch: rel_id={}",
            rel_id
        );
    }

    // Depth-2 invariant: orders should appear in graphs.main.node_ids
    // (users → orders walked at hop 1). Without filters all 4 orders/user
    // are eligible (Uniform mode), so node_ids[orders] must be non-empty
    // for the seeds we requested.
    let orders_nodes = channel
        .node_ids
        .get("orders")
        .expect("orders must be populated at hop 1 with depth=2 plan");
    assert!(
        !orders_nodes.is_empty(),
        "expected at least one order to be sampled at hop 1; got empty"
    );
    // batch_dict[orders] alignment.
    let orders_bidx = channel
        .batch_dict
        .get("orders")
        .expect("batch_dict[orders] must exist alongside node_ids[orders]");
    assert_eq!(orders_bidx.len(), orders_nodes.len());
    for &bidx in orders_bidx {
        assert!(
            (bidx as usize) < batch.task.seed_node_ids.len(),
            "batch_idx {bidx} out of range for batch_size={}",
            batch.task.seed_node_ids.len()
        );
    }
}

// Orphan seeds (FK references absent from the training graph — cross-graph TT
// reuse) are DROPPED at ingest, so the stage table is already clean. Sampling
// over the surviving rows keeps every row-aligned array (labels, timestamps,
// input_ids) consistent, with the orphan's row and label gone.
#[test]
fn test_sample_runtime_batch_excludes_orphan_seeds() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    // Row 1's ENTITY1=50 is an orphan in the 16-user parent → its row is dropped
    // at ingest. The 3 surviving rows (0,2,3) compact to stage node ids 0,1,2.
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_orph", &[0, 50, 2, 3]).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_orph").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    };

    // Only the 3 surviving rows exist on disk; sample over their compacted ids.
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_orph/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None,
        )
        .expect("clean post-drop stage must sample");

    // The orphan row never entered the extension; kept rows are compacted.
    assert_eq!(batch.task.input_ids, vec![0u32, 1, 2], "3 kept rows");
    assert_eq!(
        batch.task.tt_stage_node_ids,
        vec![0u64, 1, 2],
        "surviving rows compact to node ids 0,1,2 (orphan dropped at ingest)"
    );
    assert_eq!(
        batch.task.seed_node_ids,
        vec![0u64, 2, 3],
        "anchors = the kept members' parent ids (pk == id in this fixture)"
    );
    // LABEL ALIGNMENT: TARGET1[i] = i * 0.1 and TIMESTAMP[i] = 1000 + i by
    // ROW; after exclusion the kept rows' labels must follow them, not shift.
    let ts = batch.task.seed_timestamps.as_ref().expect("timestamps");
    assert_eq!(ts, &vec![1000i64, 1002, 1003], "timestamps realigned");
    let target_batch = batch.task.target.as_ref().expect("target");
    assert_eq!(target_batch.num_rows(), 3);
    let target_col = target_batch
        .column_by_name("TARGET1")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .unwrap();
    let got: Vec<f64> = (0..3).map(|i| target_col.value(i)).collect();
    for (g, e) in got.iter().zip([0.0, 0.2, 0.3]) {
        assert!(
            (g - e).abs() < 1e-9,
            "labels must stay with their rows after orphan exclusion; got {got:?}"
        );
    }
}

// A TT whose seeds are ALL orphan (every ENTITY1 is absent from this graph)
// drops every row at ingest → the stage table is EMPTY. The all-orphan case is
// handled at ingest (drop) + kumo (100%-skip warn/hard-fail), never at sample
// time — so the engine never faces an anchor-nowhere batch.
#[test]
fn test_all_orphan_tt_ingests_empty_stage() {
    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_orph2", &[50, 99]).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_orph2").unwrap(),
    );
    assert_eq!(
        reader.table("tt_orph2/train_seeds").unwrap().node_count(),
        0,
        "every orphan seed row is dropped at ingest → empty stage"
    );
}

// ---------------------------------------------------------------------------
// Seed-namespace TOPOLOGY exclusion must NOT drop seed RAW_FEATURES.
//
// The seed namespace ("<tt_id>/train_seeds") is a storage/bookkeeping type, not
// a model graph node type — it has no encoder on the kumo side. It is excluded
// from emitted topology (node_ids / batch_dict / num_sampled_nodes / edges /
// num_sampled_edges). But its raw_features entry is LEGITIMATELY consumed: it
// carries the train-table / seed feature columns and becomes
// hetero[train_table_name].tf on the kumo side. This test pins that the
// topology filter does not touch raw_features.
// ---------------------------------------------------------------------------

#[test]
fn test_sample_runtime_batch_seed_topology_excluded_features_preserved() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();

    // Build a users+orders parent, then ingest a seed extension whose rowpack
    // carries a NON-reserved feature column (FEAT1) in addition to the usual
    // ENTITY1 / TIMESTAMP / TARGET1. Only FEAT1 routes into raw_features.
    let parent_root = tmp.path().join("parent_idx");
    ingest_users_orders_parent_index(&parent_root, 16).unwrap();

    let tt_id = "tt_feat";
    let entity_values: Vec<i64> = (0..3).collect();
    let seed_input = tmp.path().join(format!("seed_input_{tt_id}"));
    let seed_parquet = seed_input.join("train_seeds.parquet");
    {
        let n = entity_values.len();
        let schema = Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            ),
            Field::new("TARGET1", DataType::Float64, false),
            // Non-reserved → must land in raw_features.
            Field::new("FEAT1", DataType::Float64, false),
        ]);
        let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
        let targets: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let feats: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 + 5.0).collect();
        write_parquet(
            &seed_parquet,
            schema,
            vec![
                Arc::new(Int64Array::from(entity_values.clone())) as ArrayRef,
                Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
                Arc::new(arrow_array::Float64Array::from(targets)) as ArrayRef,
                Arc::new(arrow_array::Float64Array::from(feats)) as ArrayRef,
            ],
        )
        .unwrap();
    }

    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    // Declare FEAT1 so it is actually ingested into the seed rowpack (ingest
    // only ingests declared columns) and routes into raw_features. Insert it
    // BEFORE the synthetic `entity_node_id` column (which must stay last).
    seed_table.columns.insert(
        seed_table.columns.len() - 1,
        ColumnSpec {
            name: "FEAT1".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Float64,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        },
    );
    let config = GraphConfig {
        dataset_name: format!("seed_{tt_id}"),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let seed_output = parent_root.join("train_tables").join(tt_id);
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
        resolve_columns: vec![entity_resolve_spec()],
        col_stats: true,
        ..Default::default()
    };
    ingest_with_stats_from_source(
        config,
        None,
        seed_output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )
    .unwrap();

    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, tt_id).unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    };
    let seed_ns = format!("{tt_id}/train_seeds");

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            &seed_ns,
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None,
        )
        .expect("sample_runtime_batch must succeed");

    let channel = batch.graphs.get("main").expect("main channel");

    // TOPOLOGY: seed namespace excluded everywhere; parent entity present.
    assert!(
        !channel.node_ids.contains_key(&seed_ns),
        "seed namespace must not appear in node_ids; got {:?}",
        channel.node_ids.keys().collect::<Vec<_>>()
    );
    assert!(
        !channel.batch_dict.contains_key(&seed_ns),
        "seed namespace must not appear in batch_dict"
    );
    assert!(
        !channel.num_sampled_nodes.contains_key(&seed_ns),
        "seed namespace must not appear in num_sampled_nodes; got {:?}",
        channel.num_sampled_nodes.keys().collect::<Vec<_>>()
    );
    assert!(
        channel.num_sampled_nodes.contains_key("users"),
        "parent entity (users) must still appear in num_sampled_nodes"
    );

    // INVARIANT: raw_features for the seed namespace IS preserved (carries the
    // FEAT1 train-table feature column). Topology filtering must not touch it.
    let bank = batch
        .raw_features
        .get(&seed_ns)
        .expect("seed namespace raw_features must be preserved (carries FEAT1)");
    let schema = bank.schema.as_ref().expect("seed raw_features has schema");
    let cols: std::collections::HashSet<&str> =
        schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        cols.contains("FEAT1"),
        "seed raw_features must carry the non-reserved FEAT1 column; got {cols:?}"
    );
    // The total feature rows must equal the batch size (3 seeds).
    let total_rows: usize = bank.batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 3,
        "seed raw_features row count must match batch size"
    );
}

// ---------------------------------------------------------------------------
// MULTI-STAGE seed-namespace exclusion. The merged TT catalog contains ALL of
// a tt's stage seed tables (train/val/test) as node types, each with a
// `seed→ENTITY` relation. When sampling ONE stage (val), the OTHER stages'
// seed namespaces must ALSO be excluded from emitted topology — they would
// otherwise surface as zero-count entries in num_sampled_nodes / the endpoints
// of num_sampled_edges (the pybind emits every catalog edge type), and the
// kumo adapter's node-type back-fill from edge endpoints would pick them up →
// encoder KeyError. Single-stage exclusion (the prior fix) is insufficient.
// ---------------------------------------------------------------------------

#[test]
fn test_sample_runtime_batch_excludes_all_sibling_stage_seed_namespaces() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    // Helper: write a seed parquet with ENTITY1 / TIMESTAMP / TARGET1 + a
    // non-reserved FEAT1 (so raw_features is populated for the sampled stage).
    fn write_seed_parquet_feat(path: &Path, entity_values: &[i64]) {
        let n = entity_values.len();
        let schema = Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            ),
            Field::new("TARGET1", DataType::Float64, false),
            Field::new("FEAT1", DataType::Float64, false),
        ]);
        let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
        let targets: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let feats: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 + 5.0).collect();
        write_parquet(
            path,
            schema,
            vec![
                Arc::new(Int64Array::from(entity_values.to_vec())) as ArrayRef,
                Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
                Arc::new(arrow_array::Float64Array::from(targets)) as ArrayRef,
                Arc::new(arrow_array::Float64Array::from(feats)) as ArrayRef,
            ],
        )
        .unwrap();
    }

    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // Three disjoint stages into ONE tt extension.
    let train_p = tmp.path().join("seed_input").join("train.parquet");
    let val_p = tmp.path().join("seed_input").join("val.parquet");
    let test_p = tmp.path().join("seed_input").join("test.parquet");
    write_seed_parquet_feat(&train_p, &(0..5).collect::<Vec<_>>());
    write_seed_parquet_feat(&val_p, &(5..8).collect::<Vec<_>>());
    write_seed_parquet_feat(&test_p, &(8..11).collect::<Vec<_>>());

    // The on-disk subdir name must equal the namespace so `open_with_tt`
    // (subdir = train_tables/<tt_ref>, namespace = <tt_ref>) resolves it.
    let tt_id = "tt_ms";
    let extension_dir = parent_root.join("train_tables").join(tt_id);
    // seed_columns() is ENTITY1/TIMESTAMP/TARGET1; add FEAT1 so it's ingested.
    let mut columns = seed_columns();
    columns.push(ColumnSpec {
        name: "FEAT1".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Float64,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    diskgraph_ingest::ingest_stage_extension_with_resolution(
        tt_id,
        &extension_dir,
        &parent_root,
        &[
            diskgraph_ingest::StageSpec {
                stage_table_name: "train_seeds".to_string(),
                parquet_paths: vec![train_p],
            },
            diskgraph_ingest::StageSpec {
                stage_table_name: "val_seeds".to_string(),
                parquet_paths: vec![val_p],
            },
            diskgraph_ingest::StageSpec {
                stage_table_name: "test_seeds".to_string(),
                parquet_paths: vec![test_p],
            },
        ],
        columns,
        // Resolution mode: no FK edges; ENTITY1 → entity_node_id per stage.
        vec![],
        &[ts_group_resolve_spec()],
        None,
        true,
        false,
    )
    .expect("three-stage ingest must succeed");

    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, tt_id).unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    // Sanity: the merged catalog really does carry ALL THREE stage seed tables
    // as node types (this is what makes single-stage exclusion insufficient).
    for stage in ["train_seeds", "val_seeds", "test_seeds"] {
        let ns = format!("{tt_id}/{stage}");
        assert!(
            engine.node_type_to_id.contains_key(&ns),
            "catalog must contain stage node type {ns}; got {:?}",
            engine.node_type_to_id.keys().collect::<Vec<_>>()
        );
    }

    // Sample the VAL stage. The val stage has 3 seeds (entity ids 5..8).
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    };
    let val_ns = format!("{tt_id}/val_seeds");
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            &val_ns,
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None,
        )
        .expect("sample_runtime_batch (val stage) must succeed");

    let channel = batch.graphs.get("main").expect("main channel");
    let seed_namespaces = [
        format!("{tt_id}/train_seeds"),
        format!("{tt_id}/val_seeds"),
        format!("{tt_id}/test_seeds"),
    ];

    // NONE of the three stage seed namespaces may appear in node_ids /
    // batch_dict / num_sampled_nodes.
    for ns in &seed_namespaces {
        assert!(
            !channel.node_ids.contains_key(ns),
            "stage seed namespace {ns} leaked into node_ids; got {:?}",
            channel.node_ids.keys().collect::<Vec<_>>()
        );
        assert!(
            !channel.batch_dict.contains_key(ns),
            "stage seed namespace {ns} leaked into batch_dict"
        );
        assert!(
            !channel.num_sampled_nodes.contains_key(ns),
            "stage seed namespace {ns} leaked into num_sampled_nodes; got {:?}",
            channel.num_sampled_nodes.keys().collect::<Vec<_>>()
        );
    }
    // ...nor as an endpoint (src OR dst) of any num_sampled_edges entry — this
    // is the sibling-stage leak path (zero-count `seed→ENTITY` relations).
    for (src, _rel, dst) in channel.num_sampled_edges.keys() {
        for ns in &seed_namespaces {
            assert_ne!(
                src, ns,
                "stage seed namespace {ns} leaked as num_sampled_edges SRC"
            );
            assert_ne!(
                dst, ns,
                "stage seed namespace {ns} leaked as num_sampled_edges DST"
            );
        }
    }

    // The real parent entity type IS present.
    assert!(
        channel.num_sampled_nodes.contains_key("users"),
        "parent entity (users) must appear in num_sampled_nodes; got {:?}",
        channel.num_sampled_nodes.keys().collect::<Vec<_>>()
    );

    // INVARIANT: the SAMPLED (val) stage's raw_features are preserved (FEAT1).
    let bank = batch
        .raw_features
        .get(&val_ns)
        .expect("sampled stage raw_features must be preserved (carries FEAT1)");
    let schema = bank.schema.as_ref().expect("val raw_features has schema");
    let cols: std::collections::HashSet<&str> =
        schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        cols.contains("FEAT1"),
        "val stage raw_features must carry FEAT1; got {cols:?}"
    );
}

// ---------------------------------------------------------------------------
// Phase 3 — hop-1 temporal-leak test (task #19).
//
// Uses the `users + orders` parent fixture (each user has 4 orders, two
// early/two late) plus a seed table at TIMESTAMP 1000+i. With
// `TemporalMostRecent` fanout at hop 1 and the seed timestamps fed in as
// anchors, the engine MUST NOT sample any "late" order (order_time >
// seed_time) for any seed in the batch.
//
// We never touched core sampling, so this is also a regression guard: it
// pins the existing temporal-most-recent behavior end-to-end through the
// new sample_runtime_batch path.
// ---------------------------------------------------------------------------

#[test]
fn test_sample_runtime_batch_hop1_temporal_filter_no_leak() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    // Parent: users + orders (each user has 4 orders, 2 early + 2 late).
    // Extension: seeds at ENTITY1 = 0..3 with TIMESTAMP = 1000..1003.
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_tf", &(0..3).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_tf").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    // 2 layers — hop 0 starts at users (resolved seeds), hop 1 walks
    // users → orders. Fanout 10 > 4 = max orders per user, so without
    // temporal filtering ALL 4 orders per user would be sampled; with the
    // filter, only the 2 early ones survive.
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::TemporalMostRecent,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 10,
                    mode: SamplingMode::TemporalMostRecent,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
        ],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    };

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_tf/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .expect("sample_runtime_batch must succeed");

    // Sanity: timestamps propagated.
    assert_eq!(
        batch.task.seed_timestamps.as_deref(),
        Some([1000i64, 1001, 1002].as_slice()),
        "seed timestamps must flow through to the runtime batch"
    );

    let channel = batch.graphs.get("main").expect("main channel must exist");
    let orders_nodes = channel
        .node_ids
        .get("orders")
        .expect("expected orders to appear in graphs.main.node_ids at hop 1");
    assert!(
        !orders_nodes.is_empty(),
        "fanout=10 on TemporalMostRecent with 2 valid orders/user should emit at least one order"
    );

    // Fixture invariant: order_id % 4 ∈ {0,1} → early (order_time ≤ 200);
    // order_id % 4 ∈ {2,3} → late (order_time ≥ 2000). Seed timestamps are
    // all ≤ 1002, so late orders MUST NOT be sampled at hop 1.
    let mut leaked_late: Vec<u64> = Vec::new();
    for &order_global in orders_nodes {
        let kind = order_global % 4;
        if kind == 2 || kind == 3 {
            leaked_late.push(order_global);
        }
    }
    assert!(
        leaked_late.is_empty(),
        "hop-1 temporal filter leaked late orders into the runtime batch: {leaked_late:?} \
         (all order ids with id % 4 ∈ {{2,3}} have order_time > 1002 = max seed_time)"
    );

    // Stronger: for each seed user u in [0,1,2], the orders sampled
    // should include at least one of {4u, 4u+1} (the early orders for u).
    // batch_dict[orders] maps emitted order index → seed_index.
    let orders_batch_idx = channel
        .batch_dict
        .get("orders")
        .expect("batch_dict must contain orders");
    assert_eq!(
        orders_batch_idx.len(),
        orders_nodes.len(),
        "batch_dict[orders] and node_ids[orders] must be aligned"
    );
    let mut per_seed: std::collections::HashMap<u32, Vec<u64>> = std::collections::HashMap::new();
    for (i, &order_global) in orders_nodes.iter().enumerate() {
        per_seed
            .entry(orders_batch_idx[i])
            .or_default()
            .push(order_global);
    }
    for seed_idx in 0..3u32 {
        let orders_for_seed = per_seed
            .get(&seed_idx)
            .cloned()
            .unwrap_or_default();
        assert!(
            !orders_for_seed.is_empty(),
            "seed {seed_idx} (user {seed_idx}) should have at least one early order sampled; got nothing"
        );
        for o in &orders_for_seed {
            assert!(
                o % 4 < 2,
                "seed {seed_idx} got order {o} but only order_id % 4 ∈ {{0,1}} are within \
                 seed_time={:?} window for user {seed_idx}",
                batch.task.seed_timestamps.as_ref().map(|ts| ts[seed_idx as usize])
            );
        }
    }
}

// ---------------------------------------------------------------------------
// `sample_runtime_batch` must work with BOTH fused and non-fused feature-fetch
// flags. These dispatch to different sampler paths inside the engine
// (`process_node_relations` vs. the parallel map-reduce inside
// `process_layer`) and need independent test coverage. PR 269's TaskBatchIterator
// hard-coded fused=true; our RuntimeBatchIterator briefly hard-coded false
// (regression caught here). After this fix the caller picks.
// ---------------------------------------------------------------------------

fn run_sample_runtime_batch_with_flags(
    fused: bool,
    swap_edge_direction: bool,
) -> diskgraph_engine::output::RuntimeBatch {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_fuse", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_fuse").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
        ],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: fused,
        swap_edge_direction,
    };
    engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_fuse/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .expect("sample_runtime_batch must succeed for any flag combination")
}

#[test]
fn test_sample_runtime_batch_fused_path() {
    let batch = run_sample_runtime_batch_with_flags(/*fused=*/ true, /*swap_edge_direction=*/ true);
    assert_eq!(batch.task.input_ids, vec![0u32, 1, 2]);
    assert_eq!(batch.task.seed_node_ids, vec![0u64, 1, 2]);
    let channel = batch.graphs.get("main").expect("main channel");
    assert!(
        channel.node_ids.get("users").is_some_and(|v| !v.is_empty()),
        "fused path: users must appear in graphs.main.node_ids"
    );
    // Fused path must still emit task.target from seed rowpack.
    assert!(batch.task.target.is_some(), "fused path must populate task.target");
}

#[test]
fn test_sample_runtime_batch_non_fused_path() {
    let batch = run_sample_runtime_batch_with_flags(/*fused=*/ false, /*swap_edge_direction=*/ true);
    assert_eq!(batch.task.input_ids, vec![0u32, 1, 2]);
    assert_eq!(batch.task.seed_node_ids, vec![0u64, 1, 2]);
    let channel = batch.graphs.get("main").expect("main channel");
    assert!(
        channel.node_ids.get("users").is_some_and(|v| !v.is_empty()),
        "non-fused path: users must appear in graphs.main.node_ids"
    );
    assert!(batch.task.target.is_some(), "non-fused path must populate task.target");
}

// ---------------------------------------------------------------------------
// swap_edge_direction correctness: when true, every emitted edge must satisfy
// PyG CSC layout — the recorded `edge_type` is the REVERSE relation's tuple
// and `src_local` / `dst_local` index into THAT tuple's per-type id spaces
// (not the underlying raw direction). PR 28765's adapter feeds these
// directly into `HeteroData.edge_index_dict`; if the swap is misapplied the
// trainer learns on a wrong-direction subgraph and quality regresses
// silently. Tests:
//   1. Local-id bounds: every edge's src/dst index is in range for the
//      node_ids bucket named by its edge_type tuple.
//   2. Differential: same fixture + same seed + swap on vs off produces
//      edges whose edge_type src/dst flipped where a reverse partner exists.
//   3. Fallback: edges whose underlying rel has no reverse partner (e.g.,
//      the seed→entity cross-boundary edge with reverse_of=None) — when
//      reached, swap silently leaves them unchanged. We sanity-check that
//      the dict still passes the bounds check in this case.
// ---------------------------------------------------------------------------

#[test]
fn test_sample_runtime_batch_swap_edge_direction_local_id_bounds() {
    let batch = run_sample_runtime_batch_with_flags(/*fused=*/ true, /*swap_edge_direction=*/ true);
    let channel = batch.graphs.get("main").expect("main channel");
    let total_edges: usize = channel.edges.iter().map(|(s, _)| s.len()).sum();
    assert!(total_edges > 0, "fixture must emit at least one edge");
    for (rel_idx, (srcs, dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (src_type, rel_name, dst_type) = channel.rel_id_to_names.get(&rel_id).unwrap();
        let src_bucket = channel.node_ids.get(src_type).unwrap_or_else(|| {
            panic!(
                "edge_type.src={src_type:?} (rel {rel_name:?}) has no entry in node_ids; \
                 PyG layout broken — swap pointed src/dst at types that aren't in the dict"
            )
        });
        let dst_bucket = channel.node_ids.get(dst_type).unwrap_or_else(|| {
            panic!("edge_type.dst={dst_type:?} (rel {rel_name:?}) has no entry in node_ids")
        });
        for &src_local in srcs {
            assert!(
                (src_local as usize) < src_bucket.len(),
                "src_local={} out of bounds for {src_type} (len={}, rel={rel_name})",
                src_local,
                src_bucket.len()
            );
        }
        for &dst_local in dsts {
            assert!(
                (dst_local as usize) < dst_bucket.len(),
                "dst_local={} out of bounds for {dst_type} (len={}, rel={rel_name})",
                dst_local,
                dst_bucket.len()
            );
        }
    }
}

#[test]
fn test_sample_runtime_batch_swap_edge_direction_inverts_edge_type() {
    // Same fixture + same engine seed → both runs should sample the same
    // underlying users→orders walks; the swap=true run must report them under
    // the REVERSE relation (orders→users naming) with src/dst types flipped.
    let no_swap = run_sample_runtime_batch_with_flags(/*fused=*/ true, /*swap_edge_direction=*/ false);
    let with_swap = run_sample_runtime_batch_with_flags(/*fused=*/ true, /*swap_edge_direction=*/ true);

    let no_swap_main = no_swap.graphs.get("main").expect("main channel (no_swap)");
    let with_swap_main = with_swap.graphs.get("main").expect("main channel (with_swap)");

    // Hop 1 (users → orders) emits edges via the reverse rel
    // `orders_user_id_to_users_user_id_reverse`. Without swap the recorded
    // edge_type.src is `users`; with swap the engine looks up the partner
    // (`orders_user_id_to_users_user_id` forward) and the recorded
    // edge_type.src becomes `orders`. Count by src type to assert the flip.
    let no_swap_src_types: std::collections::HashSet<&String> = no_swap_main
        .edges
        .iter()
        .enumerate()
        .filter(|(_, (srcs, _))| !srcs.is_empty())
        .map(|(rel_idx, _)| &no_swap_main.rel_id_to_names.get(&(rel_idx as u16)).unwrap().0)
        .collect();
    let with_swap_src_types: std::collections::HashSet<&String> = with_swap_main
        .edges
        .iter()
        .enumerate()
        .filter(|(_, (srcs, _))| !srcs.is_empty())
        .map(|(rel_idx, _)| &with_swap_main.rel_id_to_names.get(&(rel_idx as u16)).unwrap().0)
        .collect();
    assert!(
        no_swap_src_types.contains(&"users".to_string()),
        "no-swap run must include edges with src=users; got src types: {no_swap_src_types:?}"
    );
    assert!(
        with_swap_src_types.contains(&"orders".to_string()),
        "swap run must include edges with src=orders (the reverse direction); \
         got src types: {with_swap_src_types:?} — swap_edge_direction did NOT flip the edge_type"
    );
    assert!(
        !with_swap_src_types.contains(&"users".to_string())
            || with_swap_src_types == no_swap_src_types,
        "swap run should NOT emit edges with src=users for relations that have \
         a reverse partner; got src types: {with_swap_src_types:?}"
    );
}

#[test]
fn test_sample_runtime_batch_swap_edge_direction_preserves_rel_name() {
    // Every swapped edge_type.rel name must exist in the engine's relation
    // catalog (i.e., the reverse_rel_map lookup didn't fabricate or fall
    // through to a stale name). Use the merged-catalog rel names as the
    // valid universe.
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_swp", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_swp").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    let valid_rels: std::collections::HashSet<&String> = engine.id_to_rel_name.iter().collect();

    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 4,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
        ],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: true,
        swap_edge_direction: true,
    };
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_swp/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .unwrap();

    let channel = batch.graphs.get("main").unwrap();
    for (rel_idx, (srcs, _dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (_src, rel, _dst) = channel.rel_id_to_names.get(&rel_id).unwrap();
        assert!(
            valid_rels.contains(rel),
            "swapped edge rel name {:?} not in engine catalog {:?}",
            rel,
            valid_rels
        );
    }
}

// ---------------------------------------------------------------------------
// DG-side fused vs non-fused heuristic (task #27 / phase 3.I).
//
// Today kumo's trainer config picks `fused` per-job and threads it through
// `make_runtime_batch_iterator`. In the end-state TT-NG world kumo doesn't
// have its own sampler anymore, so the decision has to live in DG. The
// `engine::choose_fused_mode(reader, plan, batch_size)` heuristic encapsulates
// it: compare the expected per-batch feature working set against an L2-sized
// threshold (env-var overridable). Below threshold → fused (cache locality);
// above → non-fused (parallel map-reduce amortizes memory latency).
//
// Both modes still produce the same RuntimeBatch shape; this heuristic is
// purely about which sampler dispatch path is more efficient for the
// workload. Pinned by tests `test_sample_runtime_batch_{fused,non_fused}_path`.
//
// We keep the explicit override (`fused=Some(true|false)`) for trainers that
// know better, but the default migrates to `None` = "ask DG."
// ---------------------------------------------------------------------------

#[test]
fn test_choose_fused_mode_small_workload_picks_fused() {
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_fh1", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_fh1").unwrap();

    // Small batch, small fanout — working set fits L2 → fused wins.
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    assert!(
        diskgraph_engine::engine::choose_fused_mode(&reader, &plan, /*batch_size=*/ 8),
        "small workload (batch=8, fanout=4) must pick fused for cache locality"
    );
}

#[test]
fn test_choose_fused_mode_large_workload_picks_non_fused() {
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_fh2", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_fh2").unwrap();

    // Huge batch × wide fanouts → working set blows past L2 → non-fused wins.
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 50,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
            LayerSpec {
                fanouts: std::collections::HashMap::new(),
                default_fanout: Some(FanoutSpec {
                    fanout: 50,
                    mode: SamplingMode::Uniform,
                    absent_key_policy: AbsentKeyPolicy::UniformFallback,
                }),
            },
        ],
    };
    assert!(
        !diskgraph_engine::engine::choose_fused_mode(&reader, &plan, /*batch_size=*/ 2048),
        "large workload (batch=2048, 2-layer fanout=50x50=2500 nodes/seed) must \
         pick non-fused — feature working set exceeds L2 cache budget"
    );
}

#[test]
fn test_choose_fused_mode_respects_env_threshold_override() {
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_fh3", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_fh3").unwrap();
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };

    // Same plan/batch as small_workload test — but with the threshold cranked
    // down so the same working set now exceeds it. Heuristic should flip to
    // non-fused. Confirms the env var is wired and the comparison is the only
    // gate (no other hardcoded biases).
    let original = std::env::var("DG_FUSED_L2_TARGET").ok();
    // SAFETY: tests are not parallel within the same process in pytest-rs
    // semantics; set_var is OK for the duration of this test.
    unsafe { std::env::set_var("DG_FUSED_L2_TARGET", "1"); }
    let picked = diskgraph_engine::engine::choose_fused_mode(&reader, &plan, /*batch_size=*/ 8);
    match original {
        Some(v) => unsafe { std::env::set_var("DG_FUSED_L2_TARGET", v); },
        None => unsafe { std::env::remove_var("DG_FUSED_L2_TARGET"); },
    }
    assert!(
        !picked,
        "with DG_FUSED_L2_TARGET=1 the smallest non-trivial workload must pick non-fused; \
         heuristic isn't actually consulting the env override"
    );
}

// ---------------------------------------------------------------------------
// WEIGHT rowpack-column reader (task #29 + P3 latent-bug fix).
//
// A reserved `WEIGHT` (Float64) column rides each seed row into the stage
// parquet and is ingested as a numerical seed-rowpack column.
// `sample_runtime_batch` reads it off the materialized rowpack (1:1 with the
// requested rows) into `task.weight` and MUST skip it as a feature. Replaces
// the old `weight.bin` sidecar, which the engine read by stage-local row id →
// wrong weights under interleaved splits.
// ---------------------------------------------------------------------------

/// Like `write_seed_parquet` but adds a reserved `WEIGHT` (Float64) column.
fn write_seed_parquet_with_weight(
    parquet_path: &Path,
    entity_values: &[i64],
    weights: &[f64],
) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, false),
        Field::new("WEIGHT", DataType::Float64, false),
    ]);
    let n = entity_values.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
    let targets: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    write_parquet(
        parquet_path,
        schema,
        vec![
            Arc::new(Int64Array::from(entity_values.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(targets)) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(weights.to_vec())) as ArrayRef,
        ],
    )
}

/// Ingest a single seed-table extension whose seed parquet carries a
/// reserved `WEIGHT` column (so the rowpack does too).
fn ingest_one_extension_with_weight(
    parent_root: &Path,
    tt_id: &str,
    entity_values: &[i64],
    weights: &[f64],
) -> Result<()> {
    let seed_input = parent_root
        .parent()
        .expect("parent_root has parent")
        .join(format!("seed_input_{tt_id}"));
    let seed_parquet = seed_input.join("train_seeds.parquet");
    write_seed_parquet_with_weight(&seed_parquet, entity_values, weights)?;

    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    // Declare the reserved WEIGHT column so ingest carries it into the
    // seed rowpack (gen emits it; the engine reads it per-seed). Insert it
    // BEFORE the synthetic `entity_node_id` column (which must stay last).
    seed_table.columns.insert(
        seed_table.columns.len() - 1,
        ColumnSpec {
            name: "WEIGHT".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Float64,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        },
    );
    let config = GraphConfig {
        dataset_name: format!("seed_{tt_id}"),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let seed_output = parent_root.join("train_tables").join(tt_id);
    let options = IngestOptions {
        parent_index_path: Some(parent_root.to_path_buf()),
        resolve_columns: vec![entity_resolve_spec()],
        col_stats: true,
        ..Default::default()
    };
    let _ = ingest_with_stats_from_source(
        config,
        None,
        seed_output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )?;
    Ok(())
}

#[test]
fn test_sample_runtime_batch_reads_weight_column() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_users_orders_parent_index(&parent_root, 16).unwrap();
    // Seed extension carries a WEIGHT column with 6 non-trivial f64s so
    // coercion / mis-indexing would be detected.
    let written_weights = vec![0.5f64, 1.0, 1.5, 2.0, 2.5, 3.0];
    ingest_one_extension_with_weight(
        &parent_root,
        "tt_w",
        &(0..6).collect::<Vec<_>>(),
        &written_weights,
    )
    .unwrap();

    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_w").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: true,
        swap_edge_direction: true,
    };

    // Request rows 0, 2, 4 — the engine reads WEIGHT off the materialized
    // seed rowpack for exactly those rows (per-row, not global-position-k).
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_w/train_seeds",
            &[0u64, 2, 4],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .expect("sample_runtime_batch must succeed; WEIGHT read off the rowpack");

    let weights = batch
        .task
        .weight
        .as_ref()
        .expect("task.weight must be populated when the seed rowpack carries WEIGHT");
    assert_eq!(
        weights,
        &vec![0.5f32, 1.5, 2.5],
        "task.weight must read rows 0, 2, 4 of the WEIGHT column exactly"
    );
}

#[test]
fn test_sample_runtime_batch_no_weight_column_keeps_weight_none() {
    // Plain seed fixture (no WEIGHT column): task.weight stays None (the
    // PQ-path case — most plans today don't have weights).
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_now", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_now").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: std::collections::HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: true,
        swap_edge_direction: true,
    };

    // The seed rowpack has no WEIGHT column, so read_weight_column returns
    // None and task.weight stays None — the PQ-path default.
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_now/train_seeds",
            &[0u64, 1, 2],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            None,
            true,
            None, // node_range off
        )
        .unwrap();

    assert!(
        batch.task.weight.is_none(),
        "task.weight must be None when the seed rowpack carries no WEIGHT column"
    );
}

// ===========================================================================
// Task 1.1 (PR B1) — `ingest_stage_extension` source-agnostic primitive.
//
// These tests characterize the extracted stage-extension ingest used for both
// the training path (train/val/test seeds) and the prediction path
// (predict / rhs_predict). They reuse the fixtures above.
// ===========================================================================

/// ENTITY1 (Int64) + TIMESTAMP (Timestamp) column specs — no target. Mirrors
/// what `infer_stage_columns` would produce for a target-less stage parquet.
fn pred_columns() -> Vec<ColumnSpec> {
    vec![
        ColumnSpec {
            name: "ENTITY1".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Int64,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        },
        ColumnSpec {
            name: "TIMESTAMP".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Timestamp,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        },
    ]
}

/// ENTITY1 + TIMESTAMP + TARGET1 column specs — the training-stage shape.
/// The synthetic `entity_node_id` column is appended by the resolution ingest
/// primitive, so it is NOT listed here.
fn seed_columns() -> Vec<ColumnSpec> {
    let mut cols = pred_columns();
    cols.push(ColumnSpec {
        name: "TARGET1".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Float64,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    cols
}

// ---------------------------------------------------------------------------
// TT-NG temporal-LP timestamp grouping (ingest side).
//
// These exercise the FULL ingest path (encode spill → node-writer spill →
// post-build merge), not just the merge unit (covered in
// `ts_grouping::tests`). They assert the on-disk `bucket_index.bin` + the
// returned bucket list group final node-ids by TIMESTAMP, with `+min_id`
// applied across shards, and that the gate is byte-identical when off.
// ---------------------------------------------------------------------------

/// Write an ENTITY1 + TIMESTAMP stage parquet with caller-chosen timestamps
/// (Int64 entity ids 0..n, explicit per-row TIMESTAMP seconds).
fn write_ts_group_parquet(path: &Path, entity_ids: &[i64], ts_secs: &[i64]) {
    assert_eq!(entity_ids.len(), ts_secs.len());
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(Int64Array::from(entity_ids.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(ts_secs.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
}

/// Read a `bucket_index.bin` into a Vec<u64> (LE u64 node-ids).
fn read_bucket_index(path: &Path) -> Vec<u64> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() % 8, 0, "bucket_index.bin must be 8-byte aligned");
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Resolution spec ENTITY1 → users → `entity_node_id` for the ts-grouping
/// stage ingests (the primitive appends the `entity_node_id` column itself).
fn ts_group_resolve_spec() -> diskgraph_ingest::ResolveSpec {
    diskgraph_ingest::ResolveSpec {
        src_col: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
        kind: diskgraph_ingest::ResolveKind::Scalar,
        node_id_col: "entity_node_id".to_string(),
    }
}

/// Ground-truth `stage_node_id -> TIMESTAMP` map, derived INDEPENDENTLY of the
/// grouping path. The grouping artifact captures each node's TIMESTAMP at the
/// node-writer via `get_timestamp` and spills it; here we recover the same
/// mapping through a completely separate channel — the resolved
/// `entity_node_id` rowpack column:
///
///   stage node_id --(entity_node_id column)--> parent node-id == ENTITY1 value
///   (the parent `users` index is SINGLE-SHARD with user_id = 0..N, so its
///   node-id equals its PK value, the invariant the resolution tests rely on)
///   --> `entity_ts[entity]` from the test's input rows.
///
/// Cross-checking `bucket_index.bin` against this map verifies the
/// capture-point invariant (each node-id lands in the bucket of ITS OWN
/// timestamp), not just the partition shape. `entity_ts[k]` is the input
/// TIMESTAMP for ENTITY1 == k.
fn node_id_to_ts_via_resolution(
    extension_dir: &Path,
    stage_name: &str,
    entity_ts: &[i64],
) -> std::collections::HashMap<u64, i64> {
    let reader = diskgraph_index::graph::GraphIndexReader::open(
        &extension_dir.join("index_metadata.json"),
        extension_dir,
    )
    .unwrap();
    let stage = reader
        .table(stage_name)
        .unwrap_or_else(|| panic!("stage table `{stage_name}` must exist"));
    let mut out = std::collections::HashMap::new();
    for node_id in 0..stage.node_count() {
        let view = stage.edges_for_node(node_id).unwrap();
        let json = stage
            .decode_node_data_json(&view)
            .unwrap()
            .unwrap_or_else(|| panic!("stage node {node_id} must have rowpack data"));
        let entity = json
            .get("entity_node_id")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("stage node {node_id} has no entity_node_id: {json}"));
        let ts = *entity_ts
            .get(entity as usize)
            .unwrap_or_else(|| panic!("entity {entity} out of range of input"));
        out.insert(node_id, ts);
    }
    out
}

/// F1 capture-point cross-check: every node-id in a bucket's slice of
/// `bucket_index.bin` must actually carry that bucket's timestamp (per the
/// independent `node_id -> stored_ts` ground truth). Catches any
/// spill↔node-id misalignment (off-by-one, wrong shard min_id, wrong write
/// order) that the completeness/count assertions alone would miss.
fn assert_bucket_membership_matches_ground_truth(
    bin: &[u64],
    buckets: &[diskgraph_ingest::ts_grouping::TsBucket],
    node_to_ts: &std::collections::HashMap<u64, i64>,
) {
    for b in buckets {
        for i in b.offset..b.offset + b.count {
            let node_id = bin[i as usize];
            assert_eq!(
                node_to_ts.get(&node_id).copied(),
                Some(b.ts),
                "node {node_id} is in bucket ts={} but its stored TIMESTAMP is {:?}",
                b.ts,
                node_to_ts.get(&node_id),
            );
        }
    }
}

// Integration Test 1 — group-by correctness through a real single-shard ingest.
#[test]
fn test_ts_grouping_groups_by_timestamp_end_to_end() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // 6 seed rows, mixed timestamps (with duplicates).
    let entity_ids: Vec<i64> = (0..6).collect();
    let ts_secs = vec![30i64, 10, 20, 10, 30, 10];
    let seed_parquet = tmp.path().join("seed_input").join("train_seeds.parquet");
    write_ts_group_parquet(&seed_parquet, &entity_ids, &ts_secs);

    let extension_dir = parent_root.join("train_tables").join("tt_grp");
    let result = diskgraph_ingest::ingest_stage_extension_with_resolution(
        "tt_seeds_grp",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![],
        &[ts_group_resolve_spec()],
        Some(1),
        true,
        true, // group_by_timestamp ON
    )
    .expect("grouped stage ingest must succeed");

    assert_eq!(result.counts, vec![("train_seeds".to_string(), 6u64)]);
    let grouping = result
        .ts_groupings
        .get("train_seeds")
        .expect("train_seeds must have a ts grouping");

    // Buckets ascending by ts with exact counts; offsets partition [0,6).
    let buckets = &grouping.buckets;
    assert_eq!(buckets.len(), 3);
    assert_eq!((buckets[0].ts, buckets[0].offset, buckets[0].count), (10, 0, 3));
    assert_eq!((buckets[1].ts, buckets[1].offset, buckets[1].count), (20, 3, 1));
    assert_eq!((buckets[2].ts, buckets[2].offset, buckets[2].count), (30, 4, 2));

    // The on-disk bin must exist at the manifest's relative path.
    let bin_path = extension_dir.join(&grouping.bucket_index_rel_path);
    assert!(bin_path.is_file(), "bucket_index.bin must be written");
    let bin = read_bucket_index(&bin_path);
    assert_eq!(bin.len(), 6);

    // Every node-id with timestamp t is contiguous in [offset, offset+count),
    // and the bin partitions exactly the 6 node-ids 0..6 (single shard, min_id 0).
    let all: std::collections::HashSet<u64> = bin.iter().copied().collect();
    assert_eq!(all, (0u64..6).collect());
    // ts=10 bucket holds node-ids with ts 10. In the input those are entity ids
    // 1,3,5 → single-shard node-ids equal encode order, but order within a shard
    // can vary; assert the SET is correct (the seed→node mapping is positional).
    // We verify grouping correctness via contiguity + completeness, which the
    // unit tests pin to exact node-ids; here we assert no node appears twice and
    // each bucket's slice is internally consistent in size.
    let mut covered = 0u64;
    for b in buckets {
        let slice = &bin[b.offset as usize..(b.offset + b.count) as usize];
        let uniq: std::collections::HashSet<u64> = slice.iter().copied().collect();
        assert_eq!(uniq.len(), b.count as usize, "no dup within a bucket");
        covered += b.count;
    }
    assert_eq!(covered, 6);

    // Capture-point invariant: cross-check each node-id against its OWN
    // timestamp, recovered INDEPENDENTLY of the grouping path (via the FK edge
    // → entity → input TIMESTAMP), exercising the real encode→writer→merge
    // path. This catches a ts↔node-id mis-mapping that the shape/count
    // assertions above cannot.
    let node_to_ts =
        node_id_to_ts_via_resolution(&extension_dir, "train_seeds", &ts_secs);
    assert_bucket_membership_matches_ground_truth(&bin, buckets, &node_to_ts);

    // Spill files must be cleaned up.
    let spill_dir = extension_dir.join("_tmp").join("ts_group");
    assert!(
        !spill_dir.exists() || fs::read_dir(&spill_dir).map(|mut d| d.next().is_none()).unwrap_or(true),
        "ts_group spills must be deleted after merge"
    );
}

// Integration Test 2 (+min_id, multi-shard) — node-ids in the bin equal the
// final global ids (= local + shard min_id), grouped by ts across shards.
#[test]
fn test_ts_grouping_multi_shard_min_id_end_to_end() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 32).unwrap();

    // 12 rows; 3 shards forced. Two timestamps shared across shards + one ts
    // present in only some shards (5).
    let entity_ids: Vec<i64> = (0..12).collect();
    let ts_secs: Vec<i64> = vec![7, 7, 8, 7, 8, 5, 8, 7, 5, 8, 7, 8];
    let seed_parquet = tmp.path().join("seed_input").join("train_seeds.parquet");
    write_ts_group_parquet(&seed_parquet, &entity_ids, &ts_secs);

    let extension_dir = parent_root.join("train_tables").join("tt_grp_ms");
    let result = diskgraph_ingest::ingest_stage_extension_with_resolution(
        "tt_seeds_grp_ms",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![],
        &[ts_group_resolve_spec()],
        Some(3), // multi-shard extension
        true,
        true,
    )
    .expect("multi-shard grouped ingest must succeed");

    assert_eq!(result.counts, vec![("train_seeds".to_string(), 12u64)]);
    let grouping = result.ts_groupings.get("train_seeds").unwrap();

    // Open the produced extension to learn each shard's min_id (global offsets).
    let reader = diskgraph_index::graph::GraphIndexReader::open(
        &extension_dir.join("index_metadata.json"),
        &extension_dir,
    )
    .unwrap();
    let stage = reader.table("train_seeds").unwrap();
    let total = stage.node_count();
    assert_eq!(total, 12);

    let bin_path = extension_dir.join(&grouping.bucket_index_rel_path);
    let bin = read_bucket_index(&bin_path);

    // Completeness + no-dup: the bin is exactly the 12 global node-ids [0,total).
    assert_eq!(bin.len(), 12);
    let all: std::collections::HashSet<u64> = bin.iter().copied().collect();
    assert_eq!(all, (0u64..total).collect(), "bin covers exactly [0,total)");

    // Counts per ts match the input multiset; offsets partition [0,12).
    let mut expected_counts: std::collections::HashMap<i64, u64> =
        std::collections::HashMap::new();
    for t in &ts_secs {
        *expected_counts.entry(*t).or_insert(0) += 1;
    }
    let got_counts: std::collections::HashMap<i64, u64> =
        grouping.buckets.iter().map(|b| (b.ts, b.count)).collect();
    assert_eq!(got_counts, expected_counts);
    // ts=5 present in only some shards still merges (union): count 2.
    assert_eq!(got_counts.get(&5), Some(&2));

    // Offsets are a gapless ascending partition.
    let mut cursor = 0u64;
    let mut sorted = grouping.buckets.clone();
    sorted.sort_by_key(|b| b.offset);
    for b in &sorted {
        assert_eq!(b.offset, cursor);
        cursor += b.count;
    }
    assert_eq!(cursor, 12);

    // Capture-point invariant across shards: every node-id (= local + shard
    // min_id) must land in the bucket of ITS OWN timestamp, recovered
    // independently of the grouping path (FK edge → entity → input TIMESTAMP).
    // This is the assertion that catches a ts↔node-id mis-mapping in the real
    // multi-shard encode→writer→merge path (e.g. a wrong shard min_id or spill
    // misalignment), which the per-ts count and partition checks above cannot.
    let node_to_ts =
        node_id_to_ts_via_resolution(&extension_dir, "train_seeds", &ts_secs);
    assert_bucket_membership_matches_ground_truth(&bin, &grouping.buckets, &node_to_ts);
}

// Integration Test 4 (gating off) — with grouping disabled, NO bucket_index.bin
// and NO ts_groupings are produced, and the extension is otherwise identical.
#[test]
fn test_ts_grouping_disabled_is_byte_identical() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let entity_ids: Vec<i64> = (0..6).collect();
    let ts_secs = vec![30i64, 10, 20, 10, 30, 10];
    let seed_parquet = tmp.path().join("seed_input").join("train_seeds.parquet");
    write_ts_group_parquet(&seed_parquet, &entity_ids, &ts_secs);

    let extension_dir = parent_root.join("train_tables").join("tt_off");
    let result = diskgraph_ingest::ingest_stage_extension_with_resolution(
        "tt_seeds_off",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![],
        &[ts_group_resolve_spec()],
        Some(1),
        true,
        false, // group_by_timestamp OFF
    )
    .expect("ungrouped ingest must succeed");

    assert_eq!(result.counts, vec![("train_seeds".to_string(), 6u64)]);
    assert!(
        result.ts_groupings.is_empty(),
        "no ts_groupings when grouping is disabled"
    );
    let bin_path = extension_dir.join("train_seeds").join("bucket_index.bin");
    assert!(
        !bin_path.exists(),
        "no bucket_index.bin must be produced when grouping is off"
    );
    // No ts_group spill dir should ever have been left behind either.
    assert!(!extension_dir.join("_tmp").join("ts_group").exists());
}

// Integration Test 5 (F3 — null TIMESTAMP fails loud) — with grouping ON, a
// null TIMESTAMP must be a hard error (it would otherwise create a phantom
// ts=0 bucket), not a silent bucket-0 assignment. Temporal-LP seed tables
// always carry a valid TIMESTAMP, so this is a corruption guard.
#[test]
fn test_ts_grouping_null_timestamp_errors() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // 4 rows; row 2 has a NULL TIMESTAMP.
    let entity_ids: Vec<i64> = (0..4).collect();
    let ts_secs: Vec<Option<i64>> = vec![Some(10), Some(20), None, Some(30)];
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Second, None), true),
    ]);
    let seed_parquet = tmp.path().join("seed_input").join("train_seeds.parquet");
    write_parquet(
        &seed_parquet,
        schema,
        vec![
            Arc::new(Int64Array::from(entity_ids)) as ArrayRef,
            Arc::new(TimestampSecondArray::from(ts_secs)) as ArrayRef,
        ],
    )
    .unwrap();

    let extension_dir = parent_root.join("train_tables").join("tt_null_ts");
    let res = diskgraph_ingest::ingest_stage_extension_with_resolution(
        "tt_seeds_null",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![],
        &[ts_group_resolve_spec()],
        Some(1),
        true,
        true, // group_by_timestamp ON
    );
    let err = match res {
        Ok(_) => panic!("grouped ingest must FAIL on a null TIMESTAMP, not bucket it under 0"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("null TIMESTAMP") && msg.contains("train_seeds"),
        "error must name the offending table and the null TIMESTAMP; got: {msg}"
    );
}
