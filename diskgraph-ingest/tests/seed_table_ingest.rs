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
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// Build an FKeySpec from train_seeds.ENTITY1 → users (external parent FK,
/// one-directional forward only). This is the canonical shape Phase 1 enables.
fn external_one_directional_fk() -> FKeySpec {
    FKeySpec {
        src_column: "ENTITY1".to_string(),
        dst_table: "users".to_string(),
        forward: Some(DirectionSpec::default()),
        reverse: None,
        external_parent: true,
        bidirectional: false,
    }
}

// ---------------------------------------------------------------------------
// Test 2.1.1 — FKeySpec serde roundtrip for new fields.
// ---------------------------------------------------------------------------

#[test]
fn test_fkeyspec_serde_external_parent_roundtrip() {
    let spec = external_one_directional_fk();
    let json = serde_json::to_string(&spec).expect("serialize");
    let parsed: FKeySpec = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.src_column, "ENTITY1");
    assert_eq!(parsed.dst_table, "users");
    assert!(
        parsed.external_parent,
        "external_parent must round-trip as true"
    );
    assert!(
        !parsed.bidirectional,
        "bidirectional must round-trip as false"
    );

    // Defaults preserve existing behavior when both fields are absent in the JSON.
    let legacy_json = r#"{"src_column":"ENTITY1","dst_table":"users"}"#;
    let legacy: FKeySpec = serde_json::from_str(legacy_json).expect("legacy deserialize");
    assert!(
        !legacy.external_parent,
        "external_parent must default to false (legacy behavior)"
    );
    assert!(
        legacy.bidirectional,
        "bidirectional must default to true (legacy behavior)"
    );
}

// ---------------------------------------------------------------------------
// Test 2.1.2 — validate() accepts external_parent FK whose dst_table is
// absent from the same config; rejects when external_parent=false.
// ---------------------------------------------------------------------------

#[test]
fn test_graphconfig_validate_accepts_external_fk() {
    let tmp = tempdir().unwrap();
    let parquet = tmp.path().join("seeds.parquet");
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, false),
    ]);
    write_parquet(
        &parquet,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])) as ArrayRef,
            Arc::new(TimestampSecondArray::from(vec![100, 200, 300])) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(vec![0.1, 0.2, 0.3])) as ArrayRef,
        ],
    )
    .unwrap();

    let mut seed = seed_table_spec(parquet.to_str().unwrap());
    seed.fkeys = vec![external_one_directional_fk()];

    let config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![seed],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    assert!(
        config.validate().is_ok(),
        "validate() must accept external_parent FK whose dst_table is not in this config: {:?}",
        config.validate()
    );

    // Flip external_parent=false on the same config — must error.
    let mut config_legacy = config.clone();
    config_legacy.tables[0].fkeys[0].external_parent = false;
    let err = config_legacy
        .validate()
        .expect_err("must error when external_parent=false and dst_table is missing");
    assert!(
        err.to_string().contains("missing table"),
        "error must mention missing table; got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Test 2.1.3 — validate() still errors on missing src_column even with
// external_parent=true.
// ---------------------------------------------------------------------------

#[test]
fn test_graphconfig_validate_rejects_external_fk_missing_src_column() {
    let tmp = tempdir().unwrap();
    let parquet = tmp.path().join("seeds.parquet");
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new("TARGET1", DataType::Float64, false),
    ]);
    write_parquet(
        &parquet,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            Arc::new(TimestampSecondArray::from(vec![100])) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(vec![0.1])) as ArrayRef,
        ],
    )
    .unwrap();

    let mut seed = seed_table_spec(parquet.to_str().unwrap());
    let mut bad_fk = external_one_directional_fk();
    bad_fk.src_column = "DOES_NOT_EXIST".to_string();
    seed.fkeys = vec![bad_fk];

    let config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![seed],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let err = config
        .validate()
        .expect_err("must error when src_column is missing even under external_parent=true");
    let s = err.to_string();
    assert!(
        s.contains("DOES_NOT_EXIST") || s.contains("fkey column"),
        "error must reference the missing column; got: {s}"
    );
}

// ---------------------------------------------------------------------------
// Test 2.1.4 — single-table config with one external one-directional FK
// ingests cleanly, producing only forward edges (no reverse RelationDef, no
// reverse temp files).
//
// This test is gated on Phase 1 impl landing the `parent_index_path` parameter
// + loader; until then, it asserts the precondition exists and is wired.
// ---------------------------------------------------------------------------

#[test]
fn test_ingest_skips_reverse_batcher_when_bidirectional_false() {
    let tmp = tempdir().unwrap();
    let entity_values: Vec<i64> = (0..6).collect();
    let seed_output = ingest_seed_into_parent(&tmp, &entity_values)
        .expect("seed ingest with bidirectional=false must succeed");

    let meta_bytes = fs::read(seed_output.join("index_metadata.json")).unwrap();
    let meta: diskgraph_core::metadata::DatasetMetadata =
        serde_json::from_slice(&meta_bytes).unwrap();

    // Exactly one edge relation should be present (the forward one).
    // The legacy bidirectional path would emit two (forward + reverse).
    let train_seeds_rels: Vec<_> = meta
        .edge_relations
        .iter()
        .filter(|r| r.src_type == "train_seeds" || r.dst_type == "train_seeds")
        .collect();
    assert_eq!(
        train_seeds_rels.len(),
        1,
        "expected exactly one edge relation touching train_seeds (forward only), got {}: {:?}",
        train_seeds_rels.len(),
        train_seeds_rels
            .iter()
            .map(|r| (&r.name, &r.src_type, &r.dst_type, &r.reverse_of))
            .collect::<Vec<_>>()
    );
    let forward = train_seeds_rels[0];
    assert_eq!(forward.src_type, "train_seeds");
    assert_eq!(forward.dst_type, "users");
    // Forward relation must have no reverse pair declared.
    assert!(
        forward.reverse_of.is_none(),
        "forward relation must not declare reverse_of when bidirectional=false; got {:?}",
        forward.reverse_of
    );

    // No reverse edge shard files should have been created. The edge phase
    // writes to `edges/<relation_name>/shard-*.bin`; with bidirectional=false
    // the reverse batcher is never constructed so neither its directory nor
    // any of its shard files should exist anywhere under the output. Walk
    // recursively to catch reverse data wherever the batcher might have
    // landed it — metadata-only checks would miss orphaned writes.
    fn collect_paths(p: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(read_dir) = fs::read_dir(p) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_paths(&path, out);
                } else {
                    out.push(path);
                }
            }
        }
    }
    let mut all_files = Vec::new();
    collect_paths(&seed_output, &mut all_files);
    for p in &all_files {
        let s = p.to_string_lossy();
        assert!(
            !s.contains("_reverse"),
            "no path containing `_reverse` should exist anywhere under seed_output \
             when bidirectional=false; found: {s}"
        );
    }

    // ingest_stats.json must report 0 edges in any reverse-named relation.
    // (If reverse_batcher ran silently, reverse_count would be nonzero
    //  in edge_counts even though no RelationDef references it — that's
    //  the orphaned-work signature.)
    let stats_bytes = fs::read(seed_output.join("ingest_stats.json")).unwrap();
    let stats_str = String::from_utf8_lossy(&stats_bytes);
    assert!(
        !stats_str.contains("_reverse"),
        "ingest_stats.json must not mention any `_reverse` relation when \
         bidirectional=false (would indicate the reverse_batcher ran and \
         emitted edge counts); got: {}",
        &stats_str[..stats_str.len().min(2000)]
    );
}

// ---------------------------------------------------------------------------
// Tests 2.1.5–2.1.13 — full ingest path with external pkmap.
// Stubbed pending Phase 1 impl. Each test is `#[ignore]`d with a `panic!` so
// it shows in the failing test count when the ignore is removed.
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

/// End-to-end helper: ingest a seed table that points at the parent index
/// via an external one-directional FK. Returns the seed-table output dir.
///
/// `entity_values` are written into the seed parquet's ENTITY1 column.
fn ingest_seed_into_parent(
    tmp: &tempfile::TempDir,
    entity_values: &[i64],
) -> Result<PathBuf> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16)?;

    let seed_input_dir = tmp.path().join("seed_input");
    let seed_parquet = seed_input_dir.join("train_seeds.parquet");
    write_seed_parquet(&seed_parquet, entity_values)?;

    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    seed_table.fkeys = vec![external_one_directional_fk()];

    let config = GraphConfig {
        dataset_name: "seed_run".to_string(),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };

    let seed_output = parent_root.join("train_tables").join("tt_001");
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
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
    Ok(seed_output)
}

#[test]
fn test_ingest_mmaps_external_pkmap_for_resolution() {
    let tmp = tempdir().unwrap();
    // ENTITY1 values 0..8 — all exist in the parent's 0..16 users.
    let entity_values: Vec<i64> = (0..8).collect();
    let seed_output = ingest_seed_into_parent(&tmp, &entity_values)
        .expect("seed ingest with external_parent FK must succeed");

    // After ingest, index_metadata.json must be present — proves the ingest
    // ran end-to-end. (_SUCCESS is written by the caller, not by ingest
    // itself when success_marker_content is None — so we don't assert on it.)
    assert!(
        seed_output.join("index_metadata.json").is_file(),
        "index_metadata.json must be written under {}",
        seed_output.display()
    );
    // Forward-edge file exists under nodes/<seed_table>/shard-NNNNN/edges.bin.
    let forward_edges = seed_output.join("nodes/train_seeds/shard-00000/edges.bin");
    assert!(
        forward_edges.is_file(),
        "forward edges.bin must exist at {}",
        forward_edges.display()
    );
    // Read index_metadata.json and assert at least one forward edge
    // relation was emitted from train_seeds.
    let meta_bytes = fs::read(seed_output.join("index_metadata.json")).unwrap();
    let meta: diskgraph_core::metadata::DatasetMetadata =
        serde_json::from_slice(&meta_bytes).unwrap();
    let forward_rel = meta
        .edge_relations
        .iter()
        .find(|r| r.src_type == "train_seeds" && r.dst_type == "users");
    assert!(
        forward_rel.is_some(),
        "expected a forward edge relation (train_seeds -> users) in metadata, got: {:?}",
        meta.edge_relations.iter().map(|r| (&r.name, &r.src_type, &r.dst_type)).collect::<Vec<_>>()
    );
}

#[test]
fn test_ingest_external_fk_key_failure_recording() {
    let tmp = tempdir().unwrap();
    // Mix valid (0..4) and invalid (1000..1003) ENTITY1 values. Valid ones
    // match parent.users PK; invalid ones don't.
    let entity_values: Vec<i64> = vec![0, 1, 2, 3, 1000, 1001, 1002, 1003];
    let seed_output = ingest_seed_into_parent(&tmp, &entity_values)
        .expect("seed ingest must succeed even with some bogus FKs");

    // ingest_stats.json should record key lookup failures. Don't assert the
    // exact mechanism (KeyFailureState is internal); assert observable
    // outcomes: the ingest completed (index_metadata.json exists) AND
    // forward edges.bin is present (proves the valid rows produced edges).
    assert!(
        seed_output.join("index_metadata.json").is_file(),
        "ingest must complete even when some FK keys miss the parent pkmap"
    );
    assert!(
        seed_output.join("nodes/train_seeds/shard-00000/edges.bin").is_file(),
        "forward edges.bin must exist for the rows whose FKs resolved"
    );
    // ingest_stats.json must exist and mention key_lookup_failures or
    // key_failures_by_context (the existing reporting mechanism).
    let stats_bytes = fs::read(seed_output.join("ingest_stats.json")).unwrap();
    let stats_str = String::from_utf8_lossy(&stats_bytes);
    assert!(
        stats_str.contains("key_lookup_failures") || stats_str.contains("key_failures"),
        "ingest_stats.json must report key lookup failures; got: {}",
        &stats_str[..stats_str.len().min(2000)]
    );
}

#[test]
fn test_ingest_parent_untouched() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // Snapshot parent files (relative path + content len + content hash).
    fn snapshot(root: &Path) -> Vec<(String, u64, u64)> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut out = Vec::new();
        fn walk(root: &Path, base: &Path, out: &mut Vec<(String, u64, u64)>) {
            for entry in fs::read_dir(root).unwrap().flatten() {
                let path = entry.path();
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
    let before = snapshot(&parent_root);

    // Now ingest a seed table that uses the parent's pkmap.
    let seed_input_dir = tmp.path().join("seed_input");
    let seed_parquet = seed_input_dir.join("train_seeds.parquet");
    write_seed_parquet(&seed_parquet, &(0..4).collect::<Vec<_>>()).unwrap();
    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    seed_table.fkeys = vec![external_one_directional_fk()];
    let config = GraphConfig {
        dataset_name: "seed_run".to_string(),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    // Output ELSEWHERE — not inside parent_root — to confirm Phase 1
    // physically cannot touch the parent. (When we later support
    // train_tables/<tt_id>/ under parent_root, that's a separate concern
    // about subdir placement; the bytes invariant must hold either way.)
    let seed_output = tmp.path().join("seed_output");
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
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
    )
    .unwrap();

    let after = snapshot(&parent_root);
    assert_eq!(
        before, after,
        "parent index files must be byte-identical before and after seed ingest"
    );
}

#[test]
fn test_ingest_col_stats_observer_fires_for_seed_table() {
    let tmp = tempdir().unwrap();
    let entity_values: Vec<i64> = (0..8).collect();
    let seed_output = ingest_seed_into_parent(&tmp, &entity_values)
        .expect("seed ingest must succeed");
    let col_stats = seed_output.join("col_stats.json");
    assert!(
        col_stats.is_file(),
        "col_stats.json must be produced when IngestOptions.col_stats=true"
    );
    let stats_bytes = fs::read(&col_stats).unwrap();
    let stats_str = String::from_utf8_lossy(&stats_bytes);
    // Per-column stats for ENTITY1 + TARGET1 must appear.
    for col in &["ENTITY1", "TARGET1"] {
        assert!(
            stats_str.contains(col),
            "col_stats.json must contain entry for `{col}`; got: {}",
            &stats_str[..stats_str.len().min(1500)]
        );
    }
}

#[test]
fn test_ingest_sort_keys_on_observation_ts() {
    let tmp = tempdir().unwrap();
    // Build a seed with TIMESTAMP-bearing FK (DirectionSpec.time set).
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let seed_input_dir = tmp.path().join("seed_input");
    let seed_parquet = seed_input_dir.join("train_seeds.parquet");
    write_seed_parquet(&seed_parquet, &(0..8).collect::<Vec<_>>()).unwrap();

    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    // Forward DirectionSpec with TIMESTAMP encoding — engages sort_keys.
    let mut fk = external_one_directional_fk();
    fk.forward = Some(DirectionSpec {
        time: Some(diskgraph_ingest::graph_config::TimeSpec {
            table: "train_seeds".to_string(),
            column: "TIMESTAMP".to_string(),
            bitwidth: Some(32),
            anchor_time: None,
            unit: "s".to_string(),
        }),
        weight: None,
    });
    seed_table.fkeys = vec![fk];

    let config = GraphConfig {
        dataset_name: "seed_run".to_string(),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let seed_output = parent_root.join("train_tables").join("tt_sort");
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
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
    )
    .unwrap();

    // Metadata's forward relation must declare dest_sort_key with TIMESTAMP.
    let meta: diskgraph_core::metadata::DatasetMetadata =
        serde_json::from_slice(&fs::read(seed_output.join("index_metadata.json")).unwrap())
            .unwrap();
    let fwd = meta
        .edge_relations
        .iter()
        .find(|r| r.src_type == "train_seeds")
        .expect("forward relation must be present");
    assert!(
        fwd.dest_sort_key_bits.is_some(),
        "forward relation must declare dest_sort_key_bits when DirectionSpec.time is set; got {:?}",
        fwd.dest_sort_key_bits
    );
}

/// Bidirectional=false with an in-config dst (external_parent=false).
/// This is the "we have both src and dst tables in this config, but the
/// user wants forward-only edges" case — orthogonal to external_parent.
/// Without isolated coverage we couldn't distinguish bugs in the two flag
/// paths.
#[test]
fn test_ingest_skips_reverse_when_bidirectional_false_in_config_dst() {
    let tmp = tempdir().unwrap();
    // Build a 2-table config: users (PK) + orders (FK→users) with
    // bidirectional=false but external_parent=false.
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
    // Forward-only FK on orders: bidirectional=false, external_parent=false.
    orders.fkeys = vec![FKeySpec {
        src_column: "user_id".to_string(),
        dst_table: "users".to_string(),
        forward: Some(DirectionSpec::default()),
        reverse: None,
        external_parent: false,
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

/// Bidirectional=true with external_parent=true. This combination would mean
/// "emit both forward and reverse edges, where dst lives in the parent".
/// The current architecture doesn't support it (we have no way to materialize
/// reverse edges into the parent's index — that would mutate the parent).
/// The ingest should reject this combination loudly rather than silently
/// producing broken output.
///
/// NOTE: today this combination *might* not error explicitly — we don't have
/// an explicit reject path. The test is written first; if it fails, we add
/// the reject. If it passes, we know we have validation. Either way the
/// combination is locked down by test.
#[test]
fn test_ingest_rejects_bidirectional_with_external_parent() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 8).unwrap();

    let seed_input = tmp.path().join("seed_input");
    let seed_pq = seed_input.join("train_seeds.parquet");
    write_seed_parquet(&seed_pq, &(0..4).collect::<Vec<_>>()).unwrap();

    let mut seed_table = seed_table_spec(seed_pq.to_str().unwrap());
    let mut fk = external_one_directional_fk();
    fk.bidirectional = true; // ← invalid combination with external_parent=true
    seed_table.fkeys = vec![fk];
    let config = GraphConfig {
        dataset_name: "invalid_combo".to_string(),
        tables: vec![seed_table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let seed_output = tmp.path().join("seed_out");
    let options = IngestOptions {
        parent_index_path: Some(parent_root),
        ..Default::default()
    };

    let result = ingest_with_stats_from_source(
        config,
        None,
        seed_output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    );
    // bidirectional=true with external_parent=true must fail loudly. If it
    // succeeds, the ingest silently produced inconsistent output (claims to
    // emit reverse edges but has no parent-side place to put them).
    assert!(
        result.is_err(),
        "bidirectional=true + external_parent=true is unsupported and must \
         be rejected at config validation, not silently accepted; ingest \
         returned Ok(()) which is a correctness bug"
    );
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("bidirectional") || msg.contains("external_parent") || msg.contains("reverse"),
        "error must clearly indicate the invalid flag combination; got: {msg}"
    );
}

#[test]
fn test_edge_relations_consumers_tolerate_asymmetric_relations() {
    let tmp = tempdir().unwrap();
    let entity_values: Vec<i64> = (0..6).collect();
    let seed_output = ingest_seed_into_parent(&tmp, &entity_values)
        .expect("seed ingest must succeed");
    // GraphIndexReader::open must succeed on the produced seed index even
    // though the train_seeds → users relation has no reverse pair.
    let metadata_path = seed_output.join("index_metadata.json");
    let reader = diskgraph_index::graph::GraphIndexReader::open(&metadata_path, &seed_output)
        .expect("merged-reader's existing open() must tolerate asymmetric edge_relations");
    // The reader's node-type catalog must contain the seed's `train_seeds` table.
    assert!(
        reader.table("train_seeds").is_some(),
        "reader must expose `train_seeds` table"
    );
    // The reader does NOT contain `users` directly — `users` lives in the
    // parent index, not in this sub-index. (Phase 2 builds the merged
    // reader that resolves across the boundary.) Verify that's the case.
    assert!(
        reader.table("users").is_none(),
        "seed-only reader must NOT contain `users` (parent table)"
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

/// Int64-PK parent with caller-chosen PK values (e.g. negative ids).
fn ingest_tiny_parent_index_int_values(
    parent_root: &Path,
    num_shards: usize,
    pk_values: &[i64],
) -> Result<()> {
    ingest_tiny_parent_index_impl(
        parent_root,
        "users",
        &IngestedType::Int64,
        num_shards,
        pk_values.len() as i64,
        Some(pk_values),
    )
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
        external_parent: false,
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

    let mut seed_table = seed_table_spec(seed_parquet.to_str().unwrap());
    seed_table.fkeys = vec![external_one_directional_fk()];
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

#[test]
fn test_seed_edges_dst_bits_use_parent_max_id() {
    // Parent users table has 16 rows (ids 0..15 inclusive).
    // bits_for_id(15) = 4 bits → entry_bytes = ceil(4/8) = 1.
    // Without the fix: dst_bits=64 → entry_bytes=8.
    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_one_tt(&tmp, "tt_layout", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_layout").unwrap();
    let seed = reader
        .table("tt_layout/train_seeds")
        .expect("seed extension table");

    // Walk a representative seed row's cross-boundary segment.
    let view = seed
        .edges_for_node(0)
        .expect("seed row 0 must have an edges view");
    let segments = view
        .relation_segments()
        .expect("segments parseable")
        .into_iter()
        .filter(|seg| seg.count > 0)
        .collect::<Vec<_>>();
    assert!(
        !segments.is_empty(),
        "seed row 0 must have at least one cross-boundary edge segment; without it \
         this test can't observe the dst_bits choice"
    );
    let seg = &segments[0];
    assert!(
        seg.layout.entry_bytes <= 2,
        "expected dst_bits to be sized by parent's users.max_id=15 (4 bits → 1 byte), \
         not the 64-bit fallback. Got entry_bytes={} (≈{}x bloat per edge). \
         Means coordinated layout fix is not in place at ingest+read sides.",
        seg.layout.entry_bytes,
        seg.layout.entry_bytes,
    );
}

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

#[test]
fn test_merged_edge_relations_namespaces_extension_src_type() {
    // The reader's merged_edge_relations() is what the engine consumes to
    // build its node_type_to_id-keyed relation table. The extension's
    // src_type and relation name must be rewritten to the namespaced form
    // so they resolve against the merged catalog. dst_type stays bare
    // because the destination is a parent table.
    let tmp = tempdir().unwrap();
    let parent_root = ingest_parent_with_one_tt(&tmp, "tt_rel", &[0, 1, 2]).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_rel").unwrap();

    let merged = reader.merged_edge_relations();
    let seed_rel = merged
        .iter()
        .find(|r| r.dst_type == "users" && r.src_type.contains("train_seeds"))
        .expect("merged_edge_relations must include the seed→users forward edge");
    assert_eq!(
        seed_rel.src_type, "tt_rel/train_seeds",
        "extension src_type must be namespaced as <tt_id>/<bare_name>"
    );
    assert_eq!(
        seed_rel.dst_type, "users",
        "dst_type must remain bare (it is a parent table)"
    );
    assert!(
        seed_rel.name.starts_with("tt_rel/"),
        "relation name must be namespaced too; got: {}",
        seed_rel.name
    );
    assert!(
        seed_rel.reverse_of.is_none(),
        "seed forward edge must have no reverse pair"
    );
}

// --- cross-boundary FK resolution + pkmap probe ---------------------------

#[test]
fn test_merged_reader_fk_resolution_across_boundary() {
    // Walking the seed's edges.bin yields dst node_ids that index into
    // the parent's users rowpack. "Cross-boundary" = src in this sub-index
    // (train_tables/tt_xb/), dst in the parent. Phase 1 resolved PK values
    // at ingest time; the merged reader (Phase 2) just exposes both sides.
    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_one_tt(&tmp, "tt_xb", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_xb").unwrap();

    let seed = reader.table("tt_xb/train_seeds").expect("seed table present");
    let users = reader.table("users").expect("parent users present");
    assert!(users.node_count() >= 6);

    let users_node_count = users.node_count();
    let mut dst_ids_seen = std::collections::HashSet::new();
    let mut total_edges = 0u64;
    for seed_local in 0..seed.node_count() {
        let view = seed.edges_for_node(seed_local).unwrap();
        for seg in view.relation_segments().unwrap() {
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            for chunk in seg.data.chunks(entry_bytes).take(seg.count as usize) {
                let mut buf = [0u8; 16];
                buf[..entry_bytes].copy_from_slice(chunk);
                let dst_id = (u128::from_le_bytes(buf) & dst_mask) as u64;
                assert!(
                    dst_id < users_node_count,
                    "dst_id {dst_id} out of range for users (n={users_node_count})"
                );
                dst_ids_seen.insert(dst_id);
                total_edges += 1;
            }
        }
    }
    assert!(total_edges >= 6, "expected >= 6 forward edges, got {total_edges}");
    assert_eq!(
        dst_ids_seen.len(),
        6,
        "6 ENTITY1 values must resolve to 6 distinct parent rows; got {:?}",
        dst_ids_seen
    );
}

#[test]
fn test_seed_edges_dst_id_is_parent_global_id_multi_shard() {
    // Cross-boundary FK pre-resolution at ingest must write the parent's
    // GLOBAL node id into seed.edges.bin — not the per-shard local id.
    // With a single-shard parent the two coincide (shard 0 offset == 0),
    // so existing coverage misses the bug; this test forces a multi-shard
    // parent so the dst_offsets vector is non-trivial and any
    // "missing parent shard_offsets" regression collapses dst_ids to
    // shard-local ids — exactly the failure mode of kumo-diskgraph#279.
    //
    // Verification strategy: with 4 shards × 16 rows = 64 parent users,
    // shard layout is [0..15] [16..31] [32..47] [48..63]. ptrhash
    // distributes 32 distinct PKs across all 4 shards with overwhelming
    // probability. After the fix at least one dst_id must land outside
    // shard 0's range (i.e. >= 16). Before the fix, ALL dst_ids collapse
    // to shard-local ids in [0..15] regardless of which shard the key
    // actually lives in.
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 4, 64).unwrap();
    let entity_values: Vec<i64> = (0..32).collect();
    ingest_one_extension(&parent_root, "tt_multi_shard", &entity_values).unwrap();

    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_multi_shard")
            .unwrap();
    let users = reader.table("users").unwrap();
    assert_eq!(users.node_count(), 64, "parent users node count");

    // Collect dst_ids from seed.edges.bin.
    let seed = reader.table("tt_multi_shard/train_seeds").unwrap();
    let mut dst_ids: Vec<u64> = Vec::with_capacity(entity_values.len());
    for seed_local in 0..seed.node_count() {
        let view = seed.edges_for_node(seed_local).unwrap();
        let mut found = false;
        for seg in view.relation_segments().unwrap() {
            if seg.count == 0 {
                continue;
            }
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            let mut buf = [0u8; 16];
            buf[..entry_bytes].copy_from_slice(&seg.data[..entry_bytes]);
            dst_ids.push((u128::from_le_bytes(buf) & dst_mask) as u64);
            found = true;
            break;
        }
        assert!(found, "seed row {seed_local} has no FK edge");
    }

    // Bug signature: dst_offsets fallback ([0; num_shards]) makes every
    // dst_id = 0 + dst_local, so all dst_ids land in shard 0's range
    // ([0..15]). After fix, dst_ids span [0..63] following ptrhash
    // distribution across shards.
    let max_dst = *dst_ids.iter().max().unwrap();
    let min_dst = *dst_ids.iter().min().unwrap();
    assert!(
        max_dst >= 16,
        "with a 4-shard parent (shards at offsets 0/16/32/48), ptrhash distributes 32 PKs \
         across shards; seed.edges.bin dst_ids must span > shard 0's range. \
         Got dst_ids in [{min_dst}, {max_dst}] — all collapsed into shard 0. \
         Bug signature of kumo-diskgraph#279: dst_offsets fallback `vec![0; num_shards]` \
         in the edge phase wrote `0 + dst_local` instead of `parent_shard.min_id + dst_local`."
    );
    // Sanity: every dst_id is a valid parent global id.
    assert!(
        dst_ids.iter().all(|&d| d < 64),
        "every dst_id must lie within parent node range [0, 64); got dst_ids={:?}",
        dst_ids
    );
}

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

#[test]
fn test_sampler_walks_hop_0_from_seed_to_parent() {
    // Hop 0: sampler starts at a seed-table node and follows the forward
    // edge into the parent's `users` table. End-to-end proof that the
    // engine (built from a merged reader) walks across the index
    // boundary correctly.
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, BatchSamplingRequest, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode,
        SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_one_tt(&tmp, "tt_walk", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_walk").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader, Some(42), Some(1)).unwrap();

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
    let mut seed_nodes = std::collections::HashMap::new();
    seed_nodes.insert("tt_walk/train_seeds".to_string(), vec![0u64, 1, 2]);
    let request = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes,
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };

    let mut state = diskgraph_engine::state::SamplingState::default();
    let mut output = diskgraph_engine::output::InMemoryOutput::new();
    engine
        .sample(&request, &mut state, &mut output)
        .expect("hop-0 sample must succeed against merged graph");

    // InMemoryOutput records every traversed edge as (hop, rel_id, src, dst).
    assert!(
        output.edges.len() >= 3,
        "sampler must emit at least 3 forward edges for 3 seed rows; got {}",
        output.edges.len()
    );
    // Every emitted edge should land at hop 0 (only one hop in this plan).
    assert!(
        output.edges.iter().all(|(hop, _, _, _)| *hop == 0),
        "all edges must be at hop 0; got: {:?}",
        output.edges
    );
}

// Tests 2.3.3 (exclude filter on seed edge types) and 2.3.4 (train_seeds_bag
// synthesis) need OutputHandler-level changes that couple to PR-28765's
// adapter contract. Pinned with #[ignore]'d tests so the contract is
// documented and unhidden when the work lands.

#[test]
#[ignore = "Phase 3 follow-up: needs OutputHandler exclude filter on (<seed>, *, *)"]
fn test_sampler_excludes_seed_edge_types_in_emitted_output() {
    panic!("Phase 3 follow-up: implement exclude filter for seed edge types");
}

#[test]
#[ignore = "Phase 3 follow-up: needs train_seeds_bag synthesis in output handler"]
fn test_sampler_synthesizes_train_seeds_bag() {
    panic!("Phase 3 follow-up: implement train_seeds_bag synthesis");
}

#[test]
fn test_runtime_batch_output_excludes_seed_src_edges_and_populates_graph() {
    // RuntimeBatchOutput must:
    //   - DROP edges whose rel_id is in seed_src_rel_ids (the seed namespace
    //     never appears as topology in the emitted dict).
    //   - POPULATE graphs[channel].node_ids per node type from final
    //     SamplingState, and batch_dict (seed_index per emitted node).
    //   - NOT populate any seed-namespace entry in node_ids since the seed
    //     doesn't actually act as a sampling start for parent neighbors —
    //     seed nodes are filtered out / aren't a node type in topology.
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, BatchSamplingRequest, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode,
        SamplingPlan,
    };
    use diskgraph_engine::output::{OutputHandler, RuntimeBatchOutput};
    use std::collections::HashMap as StdHashMap;

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_one_tt(&tmp, "tt_rb", &(0..6).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_rb").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    // Identify the rel ids whose src_type is the seed namespace — we
    // configure RuntimeBatchOutput to drop those.
    let seed_namespace = "tt_rb/train_seeds";
    let mut seed_src_rel_ids: std::collections::HashSet<u16> = std::collections::HashSet::new();
    let mut rel_id_to_names: StdHashMap<u16, (String, String, String)> = StdHashMap::new();
    for (rel_id, name) in engine.id_to_rel_name.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        // Look up the src/dst type ids for this rel.
        let src_id = engine.relation_src_type.get(rel_id).copied().unwrap_or(usize::MAX);
        let dst_id = engine.relation_info.get(rel_id).copied().unwrap_or(usize::MAX);
        if src_id == usize::MAX || dst_id == usize::MAX {
            continue;
        }
        let src_name = engine.id_to_node_type[src_id].clone();
        let dst_name = engine.id_to_node_type[dst_id].clone();
        if src_name == seed_namespace {
            seed_src_rel_ids.insert(rel_id as u16);
        }
        rel_id_to_names.insert(rel_id as u16, (src_name, name.clone(), dst_name));
    }
    assert!(
        !seed_src_rel_ids.is_empty(),
        "fixture must produce at least one rel with src_type = seed namespace"
    );

    // Build the request: seed at the SEED node type, depth 1 — this walks
    // seed → users via the cross-boundary edge.
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: StdHashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 4,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };
    let mut seed_nodes = StdHashMap::new();
    seed_nodes.insert(seed_namespace.to_string(), vec![0u64, 1, 2]);
    let request = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes,
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };

    let seed_type_id = engine.node_type_to_id[seed_namespace];
    let seed_type_ids: std::collections::HashSet<usize> = [seed_type_id].into_iter().collect();
    let mut state = diskgraph_engine::state::SamplingState::default();
    let mut output = RuntimeBatchOutput::new(
        "main",
        seed_src_rel_ids,
        rel_id_to_names,
        engine.id_to_node_type.clone(),
        seed_type_ids,
    );
    engine.sample(&request, &mut state, &mut output).unwrap();
    output.finalize(&state).unwrap();

    let batch = output.into_runtime_batch();
    let channel = batch.graphs.get("main").expect("main channel must exist");

    // Property 1: graphs.main.node_ids must contain `users` (parent entity)
    // and MUST NOT contain the seed namespace (seed is task data, not topology).
    assert!(
        channel.node_ids.contains_key("users"),
        "graphs.main.node_ids must include the parent entity type (users); got: {:?}",
        channel.node_ids.keys().collect::<Vec<_>>()
    );
    // The seed namespace must NOT appear in node_ids / batch_dict: it's a
    // storage namespace, not a model node type (no encoder on the kumo side).
    // RuntimeBatchOutput excludes it from topology by its numeric seed_type_id,
    // symmetric with the seed-src EDGE exclusion below.
    assert!(
        !channel.node_ids.contains_key(seed_namespace),
        "seed namespace must NOT appear in node_ids; got: {:?}",
        channel.node_ids.keys().collect::<Vec<_>>()
    );
    assert!(
        !channel.batch_dict.contains_key(seed_namespace),
        "seed namespace must NOT appear in batch_dict; got: {:?}",
        channel.batch_dict.keys().collect::<Vec<_>>()
    );

    // Property 2: edge_index_dict (channel.edges) must NOT contain any edge
    // whose src is the seed namespace. The exclude filter does its job.
    for (rel_idx, (srcs, _dsts)) in channel.edges.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let rel_id = rel_idx as u16;
        let (src, _rel, _dst) = channel.rel_id_to_names.get(&rel_id).unwrap();
        assert_ne!(
            src, seed_namespace,
            "seed-src edge leaked through exclude filter: rel_id={}",
            rel_id
        );
    }

    // Property 3: edge count > 0 (we should have seen seed→users walked,
    // even though the emitted edges are entity↔entity at hops 1+; hop 0
    // edges are seed-src and excluded). With depth=1 and only seed-src
    // relation, this will be 0 emitted edges. Bump depth to 2 to see hop 1
    // edges if available.
    // For our fixture (users has no outgoing edges), depth=1 from seed →
    // users → STOP (no hop 1). The edges from hop 0 are filtered out.
    // So edge count = 0 is correct.
    // What we DO assert: batch_dict for users has 3 entries (one per seed).
    let users_batch = channel
        .batch_dict
        .get("users")
        .expect("batch_dict must have entry for `users`");
    assert!(
        users_batch.len() >= 3,
        "expected >= 3 entries in users batch_dict (one per seed); got {}",
        users_batch.len()
    );
    for &b in users_batch {
        assert!(
            (b as usize) < 3,
            "batch index {b} out of range for batch_size=3"
        );
    }
}

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

#[test]
fn test_engine_translates_extension_local_rel_id_to_merged() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, BatchSamplingRequest, FanoutSpec, LayerSpec, SamplingFlags,
        SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_rid", &(0..3).collect::<Vec<_>>()).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_rid").unwrap(),
    );
    let engine = diskgraph_engine::engine::DiskGraphEngine::from_reader(
        reader.clone(),
        Some(42),
        Some(1),
    )
    .unwrap();

    // Sanity: merged catalog has the expected 3 relations.
    assert_eq!(
        engine.id_to_rel_name.len(),
        3,
        "expected 3 merged relations (orders→users, users→orders, seed→users); got {:?}",
        engine.id_to_rel_name
    );

    // Seed the SEED namespace, depth 2. With the bug, hop 0 (seed→users)
    // gets mis-recorded as orders→users — dst type set to orders — and hop 1
    // never reaches the real users→orders walk because users is never the
    // src type at hop 1.
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
    let mut seed_nodes = std::collections::HashMap::new();
    seed_nodes.insert("tt_rid/train_seeds".to_string(), vec![0u64, 1, 2]);
    let request = BatchSamplingRequest {
        dataset_id: DatasetId::nil(),
        seed_nodes,
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: false,
        },
    };

    let mut state = diskgraph_engine::state::SamplingState::default();
    let mut output = diskgraph_engine::output::InMemoryOutput::new();
    engine.sample(&request, &mut state, &mut output).unwrap();

    // Look up type ids by name for clarity.
    let users_id = engine.node_type_to_id["users"];
    let orders_id = engine.node_type_to_id["orders"];
    let seed_id = engine.node_type_to_id["tt_rid/train_seeds"];

    // The seed namespace appears at hop 0 in state.type_states (3 seed nodes).
    let seed_state = state.type_states.get(&seed_id).expect("seed type state");
    assert_eq!(seed_state.nodes.len(), 3, "3 seed nodes at hop 0");

    // Hop 0: seed→users — users type must have at least 3 nodes (one per seed).
    let users_state = state.type_states.get(&users_id).expect(
        "users type state must be populated after hop 0 seed→users walk; if missing, \
         the engine is mis-translating the seed's local rel id to a different merged \
         relation and routing its dsts to the wrong type",
    );
    assert!(
        users_state.nodes.len() >= 3,
        "users should have ≥3 nodes after hop 0; got {} — local→merged rel id translation likely broken",
        users_state.nodes.len()
    );

    // Hop 1: users→orders — orders type must have at least one node.
    let orders_state = state.type_states.get(&orders_id).expect(
        "orders type state must be populated after hop 1 users→orders walk; if missing, \
         the engine is mis-routing hop-0 dsts to the wrong type so hop 1 never sees \
         users in the frontier",
    );
    assert!(
        !orders_state.nodes.is_empty(),
        "orders should have at least one node after hop 1; got 0"
    );
}

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
            None,
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

// Orphan seeds (FK references skipped at ingest — cross-graph TT reuse) must
// be EXCLUDED from the sampling batch entirely. An orphan seed row has no
// seed edge; the anchor walk emits one anchor per EDGE while task.target /
// input_ids / timestamps are per ROW, so keeping the row would shift every
// later row onto the wrong label. The exclusion must drop the row from ALL
// row-aligned arrays consistently.
#[test]
fn test_sample_runtime_batch_excludes_orphan_seeds() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    // Row 1's ENTITY1=50 is an orphan in the 16-user parent → its edge is
    // skipped at ingest (membership verification). Rows 0,2,3 are members.
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

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_orph/train_seeds",
            &[0u64, 1, 2, 3],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("batch with a partial orphan must still sample");

    // The orphan row (stage node id 1) is gone from every row-aligned field.
    assert_eq!(batch.task.input_ids, vec![0u32, 1, 2], "3 kept rows");
    assert_eq!(
        batch.task.tt_stage_node_ids,
        vec![0u64, 2, 3],
        "orphan stage row id 1 must be excluded"
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

// A batch consisting ONLY of orphan seeds cannot be sampled — fail with a
// message pointing at the cross-graph cause instead of anchoring nowhere.
#[test]
fn test_sample_runtime_batch_all_orphan_batch_errors() {
    use diskgraph_core::dataset::DatasetId;
    use diskgraph_core::sampling::{
        AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
    };

    let tmp = tempdir().unwrap();
    let parent_root =
        ingest_parent_with_orders_and_one_tt(&tmp, "tt_orph2", &[50, 99]).unwrap();
    let reader = std::sync::Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_orph2").unwrap(),
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

    let err = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_orph2/train_seeds",
            &[0u64, 1],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            None,
            None,
            true,
            None,
        )
        .expect_err("an all-orphan batch must fail loudly");
    let msg = format!("{err}");
    assert!(
        msg.contains("orphan"),
        "error must explain the orphan cause: {msg}"
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
    seed_table.fkeys = vec![external_one_directional_fk()];
    // Declare FEAT1 so it is actually ingested into the seed rowpack (ingest
    // only ingests declared columns) and routes into raw_features.
    seed_table.columns.push(ColumnSpec {
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
            None,
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
    diskgraph_ingest::ingest_stage_extension(
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
        vec![ent_to("users")],
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
            None,
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
            None,
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
            None,
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
            None,
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
    seed_table.fkeys = vec![external_one_directional_fk()];
    // Declare the reserved WEIGHT column so ingest carries it into the
    // seed rowpack (gen emits it; the engine reads it per-seed).
    seed_table.columns.push(ColumnSpec {
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
    });
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
            None,
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
            None,
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

/// Write a prediction-stage parquet: ENTITY1 (Int64) + TIMESTAMP (Timestamp
/// Second), NO target column. `entity_values` defines the rows.
fn write_pred_parquet_str(parquet_path: &Path, entity_values: &[&str]) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Utf8, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);
    let n = entity_values.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
    write_parquet(
        parquet_path,
        schema,
        vec![
            Arc::new(StringArray::from(entity_values.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
        ],
    )
}

fn write_pred_parquet(parquet_path: &Path, entity_values: &[i64]) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);
    let n = entity_values.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
    write_parquet(
        parquet_path,
        schema,
        vec![
            Arc::new(Int64Array::from(entity_values.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
        ],
    )
}

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

/// [`pred_columns`] with a String ENTITY1 (string-PK parent).
fn pred_columns_str() -> Vec<ColumnSpec> {
    let mut cols = pred_columns();
    cols[0].ingested_dtype = IngestedType::String;
    cols
}

/// ENTITY1 + TIMESTAMP + TARGET1 column specs — the training-stage shape.
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

/// Forward-only external_parent FK ENTITY1 → `dst_table`.
fn ent_to(dst_table: &str) -> FKeySpec {
    FKeySpec {
        src_column: "ENTITY1".to_string(),
        dst_table: dst_table.to_string(),
        forward: Some(DirectionSpec::default()),
        reverse: None,
        external_parent: true,
        bidirectional: false,
    }
}

/// Walk a stage table's seed.edges.bin (via a standalone-opened extension) and
/// return the set of distinct dst node-ids resolved by the external FK.
fn distinct_dst_ids_in_stage(
    parent_root: &Path,
    subdir_rel: &str,
    stage_name: &str,
) -> std::collections::HashSet<u64> {
    // Open the extension merged with the parent so its cross-boundary edges
    // (seed.edges.bin → parent users) are decoded against the parent's node
    // types — the same recipe the merged-reader FK-resolution tests use.
    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        parent_root,
        subdir_rel,
        "ext",
    )
    .unwrap();
    let stage = reader
        .table(&format!("ext/{stage_name}"))
        .unwrap_or_else(|| panic!("stage table `ext/{stage_name}` present"));
    let mut dst_ids_seen = std::collections::HashSet::new();
    for local in 0..stage.node_count() {
        let view = stage.edges_for_node(local).unwrap();
        for seg in view.relation_segments().unwrap() {
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            for chunk in seg.data.chunks(entry_bytes).take(seg.count as usize) {
                let mut buf = [0u8; 16];
                buf[..entry_bytes].copy_from_slice(chunk);
                let dst_id = (u128::from_le_bytes(buf) & dst_mask) as u64;
                dst_ids_seen.insert(dst_id);
            }
        }
    }
    dst_ids_seen
}

#[test]
fn test_ingest_prediction_lhs() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();
    let users_node_count = {
        let metadata_path = parent_root.join("index_metadata.json");
        diskgraph_index::graph::GraphIndexReader::open(&metadata_path, &parent_root)
            .unwrap()
            .table("users")
            .unwrap()
            .node_count()
    };

    let pred_parquet = tmp.path().join("pred_input").join("pred.parquet");
    write_pred_parquet(&pred_parquet, &(0..5).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job1");
    let counts = diskgraph_ingest::ingest_stage_extension(
        "pred_job1",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet.clone()],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("prediction LHS ingest must succeed").counts;

    assert_eq!(counts, vec![("predict".to_string(), 5u64)]);
    assert!(extension_dir.is_dir(), "extension dir must exist");
    let reader =
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "pred_tables/job1")
            .unwrap();
    assert!(
        reader.table("job1/predict").is_some(),
        "prefixed prediction ref must use the job id as namespace"
    );

    let dst_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job1", "predict");
    assert_eq!(dst_ids.len(), 5, "5 distinct resolved dst ids, got {dst_ids:?}");
    for id in &dst_ids {
        assert!(*id < users_node_count, "dst_id {id} out of range");
    }
}

/// GATE (kumo-diskgraph#... multi-shard extensions): prove a multi-shard
/// *extension* ingests, opens merged with the parent, and resolves every
/// cross-boundary FK to a correct in-range parent global id — across ALL
/// extension shards. Extensions have always been single-shard until now; this
/// is the test that establishes the engine/reader actually support `num_shards
/// > 1` for an extension before we ship size-derived sharding.
#[test]
fn test_ingest_prediction_lhs_multi_shard() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    // 16-row single-shard parent — the extension is what we force multi-shard.
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();
    let users_node_count = diskgraph_index::graph::GraphIndexReader::open(
        &parent_root.join("index_metadata.json"),
        &parent_root,
    )
    .unwrap()
    .table("users")
    .unwrap()
    .node_count();

    // 12 distinct entity ids, all real members of the 16-row parent.
    let entity_values: Vec<i64> = (0..12).collect();
    let pred_parquet = tmp.path().join("pred_input").join("pred.parquet");
    write_pred_parquet(&pred_parquet, &entity_values).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_multi");
    let counts = diskgraph_ingest::ingest_stage_extension(
        "pred_job_multi",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        Some(3), // force a multi-shard EXTENSION
        true,
        false,
    )
    .expect("multi-shard prediction-extension ingest must succeed").counts;

    // 1) The stage ingests without error and node_count covers all input rows.
    assert_eq!(
        counts,
        vec![("predict".to_string(), entity_values.len() as u64)],
        "emitted node_count must match all input rows"
    );

    // 2) Open merged with the parent and confirm the extension is genuinely
    //    multi-shard (otherwise this test wouldn't be exercising the path).
    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        &parent_root,
        "pred_tables/job_multi",
        "ext",
    )
    .unwrap();
    let stage = reader
        .table("ext/predict")
        .expect("stage table `ext/predict` present");
    assert!(
        stage.shard_count() > 1,
        "extension must actually be multi-shard; got {} shard(s)",
        stage.shard_count()
    );
    assert_eq!(
        stage.node_count(),
        entity_values.len() as u64,
        "stage node_count across all shards must equal input row count"
    );

    // 3) EVERY seed node (across all shards) must resolve its FK to exactly one
    //    in-range parent global id — none dropped, none corrupted. Collect one
    //    dst_id per seed node and check the full multiset.
    let mut per_node_dst: Vec<u64> = Vec::with_capacity(entity_values.len());
    for local in 0..stage.node_count() {
        let view = stage.edges_for_node(local).unwrap();
        let mut found = None;
        for seg in view.relation_segments().unwrap() {
            if seg.count == 0 {
                continue;
            }
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            let mut buf = [0u8; 16];
            buf[..entry_bytes].copy_from_slice(&seg.data[..entry_bytes]);
            found = Some((u128::from_le_bytes(buf) & dst_mask) as u64);
            break;
        }
        let dst = found.unwrap_or_else(|| {
            panic!("seed node {local} has no resolved FK edge (FK dropped)")
        });
        assert!(
            dst < users_node_count,
            "seed node {local} resolved to out-of-range parent id {dst} (>= {users_node_count})"
        );
        per_node_dst.push(dst);
    }
    // Exactly one resolved edge per input row — none dropped or duplicated.
    assert_eq!(
        per_node_dst.len(),
        entity_values.len(),
        "must have exactly one resolved FK per input row across all shards"
    );
    // 12 distinct real members → 12 distinct parent ids (no collisions, since
    // every entity is a true parent member).
    let distinct: std::collections::HashSet<u64> = per_node_dst.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        entity_values.len(),
        "all {} member entities must resolve to distinct parent ids; got {distinct:?}",
        entity_values.len()
    );
}

/// Step 2: a `None` (size-derived) shard count on tiny fixtures must collapse
/// to a single shard — the parquet bytes are well below the per-shard target,
/// so `estimate_num_shards` returns 1. Guards against the size-derivation path
/// accidentally over-sharding small inputs (the common training/prediction
/// case).
#[test]
fn test_ingest_stage_extension_none_tiny_yields_single_shard() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let pred_parquet = tmp.path().join("pred_input").join("pred.parquet");
    write_pred_parquet(&pred_parquet, &(0..5).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_none");
    let _ = diskgraph_ingest::ingest_stage_extension(
        "pred_job_none",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None, // derive from size — tiny fixture must yield 1 shard
        true,
        false,
    )
    .expect("size-derived single-stage ingest must succeed")
    .counts;

    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        &parent_root,
        "pred_tables/job_none",
        "ext",
    )
    .unwrap();
    let stage = reader.table("ext/predict").unwrap();
    assert_eq!(
        stage.shard_count(),
        1,
        "tiny fixture (bytes << per-shard target) must derive a single shard; got {}",
        stage.shard_count()
    );
}

#[test]
fn test_ingest_prediction_rhs() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let pred_parquet = tmp.path().join("pred_input").join("rhs.parquet");
    write_pred_parquet(&pred_parquet, &(0..5).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job1_rhs");
    let counts = diskgraph_ingest::ingest_stage_extension(
        "pred_job1_rhs",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "rhs_predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("prediction RHS ingest must succeed").counts;

    assert_eq!(counts, vec![("rhs_predict".to_string(), 5u64)]);
    assert!(
        extension_dir.is_dir() && extension_dir.ends_with("job1_rhs"),
        "_rhs extension dir layout must exist"
    );
    let dst_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job1_rhs", "rhs_predict");
    assert_eq!(dst_ids.len(), 5, "5 distinct resolved dst ids, got {dst_ids:?}");
}

#[test]
fn test_no_split_no_target() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let pred_parquet = tmp.path().join("pred_input").join("pred.parquet");
    write_pred_parquet(&pred_parquet, &(0..5).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_nosplit");
    let _ = diskgraph_ingest::ingest_stage_extension(
        "pred_nosplit",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("target-less single-stage ingest must succeed")
    .counts;

    // No split.bin anywhere under the prediction extension dir.
    fn find_split(p: &Path) -> bool {
        if let Ok(rd) = fs::read_dir(p) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if find_split(&path) {
                        return true;
                    }
                } else if path.file_name().and_then(|n| n.to_str()) == Some("split.bin") {
                    return true;
                }
            }
        }
        false
    }
    assert!(
        !find_split(&extension_dir),
        "no split.bin must exist under a prediction extension"
    );
}

// Orphan FK values (keys absent from the parent table) must be skipped and
// counted, never mis-resolved. The external pkmap is a ptrhash minimal-
// perfect-hash, so an orphan key does not miss — it collides onto an
// arbitrary in-range parent slot (concretely 50 -> 6 and 99 -> 12 in this
// fixture, which is how cross-graph TT reuse silently trained on wrong
// parents before membership verification was added at ingest).
#[test]
fn test_orphan_rows_skipped() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // 50, 99 do not exist in the 16-row parent -> orphans.
    let pred_parquet = tmp.path().join("pred_input").join("orphan.parquet");
    write_pred_parquet(&pred_parquet, &[0, 1, 2, 50, 99]).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_orphan");
    let result = diskgraph_ingest::ingest_stage_extension(
        "pred_orphan",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("ingest with orphan rows must succeed");

    // Input rows are all kept as seed NODES (node_count == 5); only the
    // orphan EDGES are dropped — the seed row simply has no resolved parent,
    // exactly like a genuine pkmap miss.
    assert_eq!(
        result.counts,
        vec![("predict".to_string(), 5u64)],
        "all input rows are kept as seed nodes"
    );
    // Membership verification rejects both MPH false positives.
    assert_eq!(
        result.key_failures.membership_failures, 2,
        "both orphan keys must be rejected by membership verification: {:?}",
        result.key_failures
    );
    assert_eq!(
        result.key_failures.lookup_failures, 0,
        "member keys must not be dropped: {:?}",
        result.key_failures
    );
    let by_context = &result.key_failures.by_context;
    assert!(
        by_context
            .get("dst_lookup:predict_ENTITY1_to_users_id")
            .map(|c| c.membership_failures == 2)
            .unwrap_or(false),
        "per-context counters must attribute the skips to the ENTITY1 FK: {by_context:?}"
    );
    assert!(
        !result.key_failures.samples.is_empty(),
        "skipped keys must be sampled for diagnostics"
    );

    // Only the 3 member keys (0, 1, 2) produce edges, resolved to their true
    // parent node ids — no orphan edge survives.
    let dst_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job_orphan", "predict");
    assert_eq!(
        dst_ids.len(),
        3,
        "only member keys may produce edges: {dst_ids:?}"
    );
    assert!(
        dst_ids.iter().all(|id| *id <= 2),
        "member keys 0,1,2 resolve to their own node ids: {dst_ids:?}"
    );
}

// Same-graph ingest (every key a member) must not drop anything and must
// report zero skips — membership verification never over-drops.
#[test]
fn test_member_rows_not_skipped_zero_counters() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let pred_parquet = tmp.path().join("pred_input").join("members.parquet");
    write_pred_parquet(&pred_parquet, &(0..16).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_members");
    let result = diskgraph_ingest::ingest_stage_extension(
        "pred_members",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("all-member ingest must succeed");

    assert_eq!(result.counts, vec![("predict".to_string(), 16u64)]);
    assert_eq!(result.key_failures.membership_failures, 0);
    assert_eq!(result.key_failures.lookup_failures, 0);
    let dst_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job_members", "predict");
    assert_eq!(dst_ids.len(), 16, "every member key keeps its edge");
}

// Orphan skipping must be correct when the PARENT is multi-shard: membership
// verification runs against the parent's global node-id space, and member
// keys in every parent shard keep their edges.
#[test]
fn test_orphan_rows_skipped_multi_shard_parent() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 4, 64).unwrap();

    // Members spread across parent shards + orphans well outside [0, 64).
    let pred_parquet = tmp.path().join("pred_input").join("orphan_ms.parquet");
    write_pred_parquet(&pred_parquet, &[0, 17, 34, 51, 63, 100, 200, 999]).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_orphan_ms");
    let result = diskgraph_ingest::ingest_stage_extension(
        "pred_orphan_ms",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("multi-shard orphan ingest must succeed");

    assert_eq!(result.counts, vec![("predict".to_string(), 8u64)]);
    // Every orphan is rejected (membership mismatch or genuine pkmap miss —
    // either way skipped + counted), every member survives.
    let skipped =
        result.key_failures.membership_failures + result.key_failures.lookup_failures;
    assert_eq!(
        skipped, 3,
        "exactly the 3 orphan keys are skipped: {:?}",
        result.key_failures
    );
    let dst_ids =
        distinct_dst_ids_in_stage(&parent_root, "pred_tables/job_orphan_ms", "predict");
    assert_eq!(dst_ids.len(), 5, "5 member keys keep edges: {dst_ids:?}");
    // Golden check: each member edge points at exactly the node the parent's
    // own verifying lookup resolves for that key.
    let users = diskgraph_index::graph::GraphIndexReader::open(
        &parent_root.join("index_metadata.json"),
        &parent_root,
    )
    .unwrap()
    .table("users")
    .unwrap();
    let expected: std::collections::HashSet<u64> = [0u64, 17, 34, 51, 63]
        .iter()
        .map(|k| {
            users
                .node_id_from_pk(&diskgraph_index::pkmap::Key::Int(*k))
                .unwrap()
                .expect("member key must resolve in parent")
        })
        .collect();
    let got: std::collections::HashSet<u64> = dst_ids.iter().copied().collect();
    assert_eq!(got, expected, "member edges resolve to their true parent nodes");
}

// Negative integer PKs are stored as two's-complement u64 keys; membership
// verification must match them (regression: an earlier `pk_value_matches`
// guard rejected any negative stored PK, which would have shed every
// negative-id member's edge as a spurious membership failure).
#[test]
fn test_negative_int_member_keys_not_skipped() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index_int_values(&parent_root, 1, &[-8, -5, -3, -1, 0, 2, 4, 7]).unwrap();

    // Three negative members + one genuine orphan.
    let pred_parquet = tmp.path().join("pred_input").join("neg.parquet");
    write_pred_parquet(&pred_parquet, &[-8, -1, 7, 100]).unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_neg");
    let result = diskgraph_ingest::ingest_stage_extension(
        "pred_neg",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("negative-pk ingest must succeed");

    let skipped =
        result.key_failures.membership_failures + result.key_failures.lookup_failures;
    assert_eq!(
        skipped, 1,
        "only the genuine orphan is skipped; negative members verify: {:?}",
        result.key_failures
    );
    let dst_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job_neg", "predict");
    assert_eq!(dst_ids.len(), 3, "all 3 members keep edges: {dst_ids:?}");
}

// String-PK parent: orphan string keys are skipped + counted, members keep
// their edges (VarBytes MPH domain + rowpack string compare).
#[test]
fn test_orphan_rows_skipped_string_pk() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::String, 1, 16).unwrap();

    // Parent PKs are "u00000000".."u00000015"; two orphans.
    let pred_parquet = tmp.path().join("pred_input").join("orphan_str.parquet");
    write_pred_parquet_str(
        &pred_parquet,
        &["u00000000", "u00000002", "not-a-user", "u99999999"],
    )
    .unwrap();

    let extension_dir = parent_root.join("pred_tables").join("job_orphan_str");
    let result = diskgraph_ingest::ingest_stage_extension(
        "pred_orphan_str",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![pred_parquet],
        }],
        pred_columns_str(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("string-pk orphan ingest must succeed");

    assert_eq!(result.counts, vec![("predict".to_string(), 4u64)]);
    let skipped =
        result.key_failures.membership_failures + result.key_failures.lookup_failures;
    assert_eq!(
        skipped, 2,
        "both string orphans are skipped: {:?}",
        result.key_failures
    );
    let dst_ids =
        distinct_dst_ids_in_stage(&parent_root, "pred_tables/job_orphan_str", "predict");
    assert_eq!(dst_ids.len(), 2, "member strings keep edges: {dst_ids:?}");
}

#[test]
fn test_lhs_rhs_no_namespace_collision() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let lhs_parquet = tmp.path().join("pred_input").join("lhs.parquet");
    write_pred_parquet(&lhs_parquet, &(0..5).collect::<Vec<_>>()).unwrap();
    let rhs_parquet = tmp.path().join("pred_input").join("rhs.parquet");
    write_pred_parquet(&rhs_parquet, &(5..10).collect::<Vec<_>>()).unwrap();

    let lhs_dir = parent_root.join("pred_tables").join("job1");
    let rhs_dir = parent_root.join("pred_tables").join("job1_rhs");

    let _ = diskgraph_ingest::ingest_stage_extension(
        "pred_job1",
        &lhs_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![lhs_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .unwrap()
    .counts;
    let _ = diskgraph_ingest::ingest_stage_extension(
        "pred_job1_rhs",
        &rhs_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "rhs_predict".to_string(),
            parquet_paths: vec![rhs_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .unwrap()
    .counts;

    assert!(lhs_dir.is_dir() && rhs_dir.is_dir(), "both extensions exist");
    let lhs_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job1", "predict");
    let rhs_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/job1_rhs", "rhs_predict");
    assert_eq!(lhs_ids.len(), 5, "lhs resolves its own 5 ids");
    assert_eq!(rhs_ids.len(), 5, "rhs resolves its own 5 ids");
}

#[test]
fn test_ingest_stage_extension_training_shape_regression() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    let train_p = tmp.path().join("seed_input").join("train.parquet");
    let val_p = tmp.path().join("seed_input").join("val.parquet");
    let test_p = tmp.path().join("seed_input").join("test.parquet");
    write_seed_parquet(&train_p, &(0..5).collect::<Vec<_>>()).unwrap();
    write_seed_parquet(&val_p, &(5..9).collect::<Vec<_>>()).unwrap();
    write_seed_parquet(&test_p, &(9..12).collect::<Vec<_>>()).unwrap();

    let extension_dir = parent_root.join("train_tables").join("tt_regress");
    let counts = diskgraph_ingest::ingest_stage_extension(
        "tt_seeds_regress",
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
        seed_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("three-stage training-shape ingest must succeed").counts;

    assert_eq!(
        counts,
        vec![
            ("train_seeds".to_string(), 5u64),
            ("val_seeds".to_string(), 4u64),
            ("test_seeds".to_string(), 3u64),
        ]
    );
    assert_eq!(
        distinct_dst_ids_in_stage(&parent_root, "train_tables/tt_regress", "train_seeds").len(),
        5
    );
    assert_eq!(
        distinct_dst_ids_in_stage(&parent_root, "train_tables/tt_regress", "val_seeds").len(),
        4
    );
    assert_eq!(
        distinct_dst_ids_in_stage(&parent_root, "train_tables/tt_regress", "test_seeds").len(),
        3
    );
}

/// U3 — multi-part stage ingest. A stage sourced from N disjoint
/// `part-*.parquet` files must produce a sub-index identical to ingesting one
/// concatenated parquet over their union: `node_count == sum(part rows)` and
/// the same resolved parent dst-id set. Exercises `StageSpec.parquet_paths`.
#[test]
fn test_ingest_stage_extension_multi_part_equals_concatenated() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, &IngestedType::Int64, 1, 16).unwrap();

    // Three disjoint parts: 0..5, 5..9, 9..12 → 12 distinct entity ids total,
    // all real members of the 16-row parent.
    let part0 = tmp.path().join("stage").join("part-00000.parquet");
    let part1 = tmp.path().join("stage").join("part-00001.parquet");
    let part2 = tmp.path().join("stage").join("part-00002.parquet");
    write_pred_parquet(&part0, &(0..5).collect::<Vec<_>>()).unwrap();
    write_pred_parquet(&part1, &(5..9).collect::<Vec<_>>()).unwrap();
    write_pred_parquet(&part2, &(9..12).collect::<Vec<_>>()).unwrap();

    // The equivalent single concatenated parquet (same union of rows).
    let concat = tmp.path().join("concat").join("all.parquet");
    write_pred_parquet(&concat, &(0..12).collect::<Vec<_>>()).unwrap();

    // ── Multi-part ingest ────────────────────────────────────────────────
    let multi_dir = parent_root.join("pred_tables").join("multi_part");
    let multi_counts = diskgraph_ingest::ingest_stage_extension(
        "pred_multi_part",
        &multi_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![part0.clone(), part1.clone(), part2.clone()],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("multi-part stage ingest must succeed").counts;

    // ── Single concatenated ingest (reference) ───────────────────────────
    let concat_dir = parent_root.join("pred_tables").join("concat");
    let concat_counts = diskgraph_ingest::ingest_stage_extension(
        "pred_concat",
        &concat_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths: vec![concat.clone()],
        }],
        pred_columns(),
        vec![ent_to("users")],
        None,
        true,
        false,
    )
    .expect("single concatenated stage ingest must succeed").counts;

    // node_count == sum of part rows == concatenated count.
    assert_eq!(
        multi_counts,
        vec![("predict".to_string(), 12u64)],
        "multi-part node_count must equal the union of all part rows"
    );
    assert_eq!(
        multi_counts, concat_counts,
        "multi-part ingest must produce the same per-stage counts as the concatenated ingest"
    );

    // Same resolved parent dst-id set as the concatenated ingest.
    let multi_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/multi_part", "predict");
    let concat_ids = distinct_dst_ids_in_stage(&parent_root, "pred_tables/concat", "predict");
    assert_eq!(
        multi_ids.len(),
        12,
        "12 distinct resolved dst ids across all parts, got {multi_ids:?}"
    );
    assert_eq!(
        multi_ids, concat_ids,
        "multi-part resolved dst-id set must be identical to the concatenated ingest"
    );
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

/// Ground-truth `stage_node_id -> TIMESTAMP` map, derived INDEPENDENTLY of the
/// grouping path. The grouping artifact captures each node's TIMESTAMP at the
/// node-writer via `get_timestamp` and spills it; here we recover the same
/// mapping through a completely separate channel — the stage table's FK edge:
///
///   stage node_id --(seed.edges.bin)--> parent dst_id == ENTITY1 value (the
///   parent `users` index is SINGLE-SHARD with user_id = 0..N, so its node-id
///   equals its PK value, the invariant the resolution tests rely on) -->
///   `entity_ts[entity]` from the test's input rows.
///
/// Cross-checking `bucket_index.bin` against this map verifies the
/// capture-point invariant (each node-id lands in the bucket of ITS OWN
/// timestamp), not just the partition shape. `entity_ts[k]` is the input
/// TIMESTAMP for ENTITY1 == k.
fn node_id_to_ts_via_fk(
    parent_root: &Path,
    subdir_rel: &str,
    stage_name: &str,
    entity_ts: &[i64],
) -> std::collections::HashMap<u64, i64> {
    let reader = diskgraph_index::graph::GraphIndexReader::open_with_extension_subdir(
        parent_root,
        subdir_rel,
        "ext",
    )
    .unwrap();
    let stage = reader
        .table(&format!("ext/{stage_name}"))
        .unwrap_or_else(|| panic!("stage table `ext/{stage_name}` must exist"));
    let mut out = std::collections::HashMap::new();
    for local in 0..stage.node_count() {
        let view = stage.edges_for_node(local).unwrap();
        let mut dst: Option<u64> = None;
        for seg in view.relation_segments().unwrap() {
            let entry_bytes = seg.layout.entry_bytes;
            let dst_mask = seg.layout.dst_mask as u128;
            for chunk in seg.data.chunks(entry_bytes).take(seg.count as usize) {
                let mut buf = [0u8; 16];
                buf[..entry_bytes].copy_from_slice(chunk);
                let dst_id = (u128::from_le_bytes(buf) & dst_mask) as u64;
                assert!(dst.is_none(), "stage node {local} must have exactly one FK edge");
                dst = Some(dst_id);
            }
        }
        let entity = dst.unwrap_or_else(|| panic!("stage node {local} has no resolved FK edge"));
        let ts = *entity_ts
            .get(entity as usize)
            .unwrap_or_else(|| panic!("entity {entity} out of range of input"));
        // `local` here is the stage's local node-id; the extension is opened in
        // isolation so its node-ids are 0-based and equal the global stage ids
        // recorded in bucket_index.bin.
        out.insert(local, ts);
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
    let result = diskgraph_ingest::ingest_stage_extension(
        "tt_seeds_grp",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
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
        node_id_to_ts_via_fk(&parent_root, "train_tables/tt_grp", "train_seeds", &ts_secs);
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
    let result = diskgraph_ingest::ingest_stage_extension(
        "tt_seeds_grp_ms",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
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
        node_id_to_ts_via_fk(&parent_root, "train_tables/tt_grp_ms", "train_seeds", &ts_secs);
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
    let result = diskgraph_ingest::ingest_stage_extension(
        "tt_seeds_off",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
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
    let res = diskgraph_ingest::ingest_stage_extension(
        "tt_seeds_null",
        &extension_dir,
        &parent_root,
        &[diskgraph_ingest::StageSpec {
            stage_table_name: "train_seeds".to_string(),
            parquet_paths: vec![seed_parquet],
        }],
        pred_columns(),
        vec![ent_to("users")],
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
