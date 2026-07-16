//! FK-free LP "resolution mode" ingest tests.
//!
//! Spec: `ttng-lp-fkfree-resolution-design.md` §S3. The special tt-ingest
//! resolves the train-table reference columns `ENTITY1` (scalar PK ref) and
//! `TARGET1` (list of PK refs) — which point at PKs in a parent node table —
//! to graph node-ids ONCE at ingest, and writes them as rowpack columns
//! (`entity_node_id: u64` scalar, `target_node_ids: list<i64>` ragged) with
//! NO FK edge / NO `seed.edges.bin`. The engine later reads these columns
//! directly.
//!
//! These tests build a tiny parent index (users PK = 0..16) and a stage
//! parquet whose ENTITY1/TARGET1 reference it, run the resolution-mode ingest,
//! and assert: (a) `entity_node_id` resolves correctly; (b) `target_node_ids`
//! lists resolve correctly (order + values); (c) NO edge / no `seed.edges.bin`
//! for these columns; (d) round-trip read returns the expected ids.

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, ListArray, RecordBatch, TimestampSecondArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_ingest::graph_config::{ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec};
use diskgraph_ingest::{
    ingest_with_stats_from_source, Dest, IngestOptions, ResolveColumnSpec, ResolveKind,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

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

fn int64_col(name: &str) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Int64,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    }
}

/// Ingest a tiny `users` parent index (PK = user_id Int64, 0..num_rows).
fn ingest_tiny_parent_index(parent_root: &Path, num_rows: i64) -> Result<()> {
    let input_dir = parent_root
        .parent()
        .expect("parent_root has parent")
        .join("_inputs");
    let parquet_path = input_dir.join("users.parquet");
    write_parquet(
        &parquet_path,
        Schema::new(vec![Field::new("user_id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..num_rows).collect::<Vec<i64>>())) as ArrayRef],
    )?;
    let table = TableSpec {
        name: "users".to_string(),
        columns: vec![int64_col("user_id")],
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
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    ingest_with_stats_from_source(
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

/// Write a stage parquet: `ENTITY1 Int64` (scalar PK ref), `TIMESTAMP`,
/// `TARGET1 List<Int64>` (list of PK refs). One row per `entity`, with the
/// matching `targets` list.
fn write_stage_parquet(
    path: &Path,
    entities: &[i64],
    targets: &[Vec<i64>],
) -> Result<()> {
    assert_eq!(entities.len(), targets.len());
    let n = entities.len();
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new(
            "TIMESTAMP",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        Field::new(
            "TARGET1",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);

    // Build a ListArray<Int64> from `targets`.
    let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
    let mut values: Vec<i64> = Vec::new();
    offsets.push(0);
    for t in targets {
        values.extend_from_slice(t);
        offsets.push(values.len() as i32);
    }
    let value_array = Arc::new(Int64Array::from(values)) as ArrayRef;
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        OffsetBuffer::new(offsets.into()),
        value_array,
        None,
    );

    let timestamps: Vec<i64> = (0..n as i64).map(|i| 1_000 + i).collect();
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(Int64Array::from(entities.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
            Arc::new(list) as ArrayRef,
        ],
    )
}

/// Stage TableSpec carrying ENTITY1/TIMESTAMP/TARGET1 plus the two NEW
/// resolution-target columns (`entity_node_id` UInt64, `target_node_ids`
/// Int64Array). No fkeys — the no-edge guarantee.
fn stage_table_spec(parquet_path: &str) -> TableSpec {
    TableSpec {
        name: "train_seeds".to_string(),
        columns: vec![
            int64_col("ENTITY1"),
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
                ingested_dtype: IngestedType::Int64Array,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
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
            ColumnSpec {
                name: "target_node_ids".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Int64Array,
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

fn resolve_specs() -> Vec<ResolveColumnSpec> {
    vec![
        ResolveColumnSpec {
            table: "train_seeds".to_string(),
            src_col: "ENTITY1".to_string(),
            dst_table: "users".to_string(),
            kind: ResolveKind::Scalar,
            node_id_col: "entity_node_id".to_string(),
        },
        ResolveColumnSpec {
            table: "train_seeds".to_string(),
            src_col: "TARGET1".to_string(),
            dst_table: "users".to_string(),
            kind: ResolveKind::List,
            node_id_col: "target_node_ids".to_string(),
        },
    ]
}

/// End-to-end: ingest a stage table in resolution mode against the parent.
/// Returns the stage extension output dir plus the ingest stats (which carry
/// the orphan skip counters).
fn ingest_resolution_with_stats(
    tmp: &tempfile::TempDir,
    entities: &[i64],
    targets: &[Vec<i64>],
) -> Result<(std::path::PathBuf, diskgraph_ingest::IngestStats)> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, 16)?;

    let stage_parquet = tmp.path().join("stage_input").join("train_seeds.parquet");
    write_stage_parquet(&stage_parquet, entities, targets)?;

    let mut table = stage_table_spec(stage_parquet.to_str().unwrap());
    table.fkeys = vec![]; // explicit: no FK edges.
    let config = GraphConfig {
        dataset_name: "resolve_run".to_string(),
        tables: vec![table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let output = parent_root.join("train_tables").join("tt_resolve");
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
        resolve_columns: resolve_specs(),
        col_stats: false,
        ..Default::default()
    };
    let (_, stats) = ingest_with_stats_from_source(
        config,
        None,
        output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )?;
    Ok((output, stats))
}

/// End-to-end: ingest a stage table in resolution mode against the parent.
/// Returns the stage extension output dir.
fn ingest_resolution(
    tmp: &tempfile::TempDir,
    entities: &[i64],
    targets: &[Vec<i64>],
) -> Result<std::path::PathBuf> {
    ingest_resolution_with_stats(tmp, entities, targets).map(|(output, _)| output)
}

/// Read one `train_seeds` node's rowpack columns back as JSON.
fn read_node_json(output: &Path, node_id: u64) -> serde_json::Value {
    let metadata_path = output.join("index_metadata.json");
    let reader = diskgraph_index::graph::GraphIndexReader::open(&metadata_path, output)
        .expect("open produced extension");
    let table = reader.table("train_seeds").expect("train_seeds table");
    let view = table.edges_for_node(node_id).expect("edges_for_node");
    table
        .decode_node_data_json(&view)
        .expect("decode_node_data_json")
        .expect("node data present")
}

#[test]
fn test_resolution_scalar_entity_node_id() {
    let tmp = tempdir().unwrap();
    // ENTITY1 = parent PK i → node id i (single shard, contiguous).
    let entities: Vec<i64> = vec![0, 3, 7, 15];
    let targets: Vec<Vec<i64>> = vec![vec![1], vec![2], vec![4], vec![8]];
    let output = ingest_resolution(&tmp, &entities, &targets).expect("resolution ingest");

    for (row, &ent) in entities.iter().enumerate() {
        let json = read_node_json(&output, row as u64);
        let got = json
            .get("entity_node_id")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("row {row}: entity_node_id missing/not u64 in {json}"));
        assert_eq!(
            got, ent as u64,
            "row {row}: entity_node_id must resolve ENTITY1={ent} to node id {ent}"
        );
    }
}

// Orphan reference values (absent from the parent) must NEVER produce a
// node-id. The pkmap is a blind MPH, so an orphan key collides onto an
// arbitrary in-range parent slot; membership verification at resolution turns
// that false positive into a null (scalar) / dropped element (list), counted
// in the ingest key-failure stats. Rows are kept — only the resolved ids are
// affected — mirroring how a null reference behaves.
#[test]
fn test_resolution_orphans_nulled_dropped_and_counted() {
    let tmp = tempdir().unwrap();
    // Row 0: member entity, target list with one orphan element (99) between
    //        two members (order of survivors must hold).
    // Row 1: ORPHAN entity (50), member target.
    let entities: Vec<i64> = vec![0, 50];
    let targets: Vec<Vec<i64>> = vec![vec![1, 99, 2], vec![3]];
    let (output, stats) =
        ingest_resolution_with_stats(&tmp, &entities, &targets).expect("resolution ingest");

    let row0 = read_node_json(&output, 0);
    assert_eq!(
        row0.get("entity_node_id").and_then(|v| v.as_u64()),
        Some(0),
        "member entity resolves: {row0}"
    );
    let row0_targets: Vec<i64> = row0
        .get("target_node_ids")
        .and_then(|v| v.as_array())
        .expect("target_node_ids array")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(
        row0_targets,
        vec![1, 2],
        "orphan list element dropped, member order preserved"
    );

    let row1 = read_node_json(&output, 1);
    assert!(
        row1.get("entity_node_id")
            .map(|v| v.is_null())
            .unwrap_or(true),
        "orphan entity must resolve to null, never a collided node id: {row1}"
    );
    let row1_targets: Vec<i64> = row1
        .get("target_node_ids")
        .and_then(|v| v.as_array())
        .expect("target_node_ids array")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(row1_targets, vec![3], "member target survives");

    // Both orphans are counted, attributed to their resolve contexts.
    assert_eq!(
        stats.key_membership_failures + stats.key_lookup_failures,
        2,
        "exactly the 2 orphan references are counted: {:?}",
        stats.key_failures_by_context
    );
    let ctx = &stats.key_failures_by_context;
    let ctx_total = |name: &str| {
        ctx.get(name)
            .map(|c| c.lookup_failures + c.membership_failures)
            .unwrap_or(0)
    };
    assert_eq!(
        ctx_total("resolve:train_seeds.ENTITY1_to_users"),
        1,
        "entity orphan attributed to the ENTITY1 context: {ctx:?}"
    );
    assert_eq!(
        ctx_total("resolve:train_seeds.TARGET1_to_users"),
        1,
        "target orphan attributed to the TARGET1 context: {ctx:?}"
    );
}

// Same-graph resolution (all members) must report zero skips.
#[test]
fn test_resolution_members_zero_counters() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 3, 7, 15];
    let targets: Vec<Vec<i64>> = vec![vec![1], vec![2, 5], vec![4], vec![8, 9, 10]];
    let (_, stats) =
        ingest_resolution_with_stats(&tmp, &entities, &targets).expect("resolution ingest");
    assert_eq!(stats.key_membership_failures, 0);
    assert_eq!(stats.key_lookup_failures, 0);
}

#[test]
fn test_resolution_list_target_node_ids() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 1, 2];
    // Lists of varying length, including order-sensitivity.
    let targets: Vec<Vec<i64>> = vec![vec![5, 2, 9], vec![10], vec![0, 15, 7, 3]];
    let output = ingest_resolution(&tmp, &entities, &targets).expect("resolution ingest");

    for (row, expected) in targets.iter().enumerate() {
        let json = read_node_json(&output, row as u64);
        let arr = json
            .get("target_node_ids")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("row {row}: target_node_ids missing/not array in {json}"));
        let got: Vec<i64> = arr
            .iter()
            .map(|v| v.as_i64().expect("list element must be int"))
            .collect();
        // In a 1-shard parent, node id == PK value, so resolved == reference.
        assert_eq!(
            &got, expected,
            "row {row}: target_node_ids must resolve TARGET1 list in order"
        );
    }
}

// ── STATIC LP past-visit exclude resolution (design `ttng-past-visit-design.md`)
//
// gen emits a second List key-column `exclude_target_keys` (TRAIN targets on
// val_seeds; TRAIN∪VAL on test_seeds; empty on train_seeds). Ingest resolves it
// through the SAME RefResolver List path as `target_node_ids`, into a sibling
// `exclude_node_ids: list<i64>` rowpack column. This test asserts the resolved
// exclude lists match the expected per-row key lists (incl. empty → empty).

/// Stage parquet with ENTITY1/TIMESTAMP/TARGET1 + the past-visit
/// `exclude_target_keys` List<Int64> column.
fn write_stage_parquet_with_exclude(
    path: &Path,
    entities: &[i64],
    targets: &[Vec<i64>],
    excludes: &[Vec<i64>],
) -> Result<()> {
    assert_eq!(entities.len(), targets.len());
    assert_eq!(entities.len(), excludes.len());
    let n = entities.len();
    let build_list = |lists: &[Vec<i64>]| -> ArrayRef {
        let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
        let mut values: Vec<i64> = Vec::new();
        offsets.push(0);
        for t in lists {
            values.extend_from_slice(t);
            offsets.push(values.len() as i32);
        }
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int64, true)),
            OffsetBuffer::new(offsets.into()),
            Arc::new(Int64Array::from(values)) as ArrayRef,
            None,
        )) as ArrayRef
    };
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Second, None), false),
        Field::new(
            "TARGET1",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "exclude_target_keys",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 2_000 + i).collect();
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(Int64Array::from(entities.to_vec())) as ArrayRef,
            Arc::new(TimestampSecondArray::from(timestamps)) as ArrayRef,
            build_list(targets),
            build_list(excludes),
        ],
    )
}

#[test]
fn test_resolution_exclude_node_ids() {
    let tmp = tempdir().unwrap();
    let parent_root = tmp.path().join("parent_idx");
    ingest_tiny_parent_index(&parent_root, 16).unwrap();

    let entities: Vec<i64> = vec![0, 1, 2];
    let targets: Vec<Vec<i64>> = vec![vec![5], vec![6], vec![7]];
    // val-style exclude lists: row0 empty, row1 one prior, row2 multi (order
    // preserved; node id == PK in 1-shard parent).
    let excludes: Vec<Vec<i64>> = vec![vec![], vec![3], vec![1, 8, 4]];

    let stage_parquet = tmp.path().join("stage_input").join("val_seeds.parquet");
    write_stage_parquet_with_exclude(&stage_parquet, &entities, &targets, &excludes).unwrap();

    // Stage spec: ENTITY1/TIMESTAMP/TARGET1/exclude_target_keys (source) +
    // entity_node_id/target_node_ids/exclude_node_ids (resolved, Int64Array).
    let mut table = stage_table_spec(stage_parquet.to_str().unwrap());
    table.name = "val_seeds".to_string();
    table.columns.push(ColumnSpec {
        name: "exclude_target_keys".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Int64Array,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    table.columns.push(ColumnSpec {
        name: "exclude_node_ids".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Int64Array,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });

    let config = GraphConfig {
        dataset_name: "resolve_exclude_run".to_string(),
        tables: vec![table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let output = parent_root.join("train_tables").join("tt_exclude");
    let mut specs = vec![
        ResolveColumnSpec {
            table: "val_seeds".to_string(),
            src_col: "ENTITY1".to_string(),
            dst_table: "users".to_string(),
            kind: ResolveKind::Scalar,
            node_id_col: "entity_node_id".to_string(),
        },
        ResolveColumnSpec {
            table: "val_seeds".to_string(),
            src_col: "TARGET1".to_string(),
            dst_table: "users".to_string(),
            kind: ResolveKind::List,
            node_id_col: "target_node_ids".to_string(),
        },
    ];
    // The exclude spec rides the SAME List resolution path as positives.
    specs.push(ResolveColumnSpec {
        table: "val_seeds".to_string(),
        src_col: "exclude_target_keys".to_string(),
        dst_table: "users".to_string(),
        kind: ResolveKind::List,
        node_id_col: "exclude_node_ids".to_string(),
    });
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
        resolve_columns: specs,
        col_stats: false,
        ..Default::default()
    };
    ingest_with_stats_from_source(
        config,
        None,
        output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )
    .expect("exclude resolution ingest");

    for (row, expected) in excludes.iter().enumerate() {
        let metadata_path = output.join("index_metadata.json");
        let reader = diskgraph_index::graph::GraphIndexReader::open(&metadata_path, &output)
            .expect("open produced extension");
        let table = reader.table("val_seeds").expect("val_seeds table");
        let view = table.edges_for_node(row as u64).expect("edges_for_node");
        let json = table
            .decode_node_data_json(&view)
            .expect("decode")
            .expect("node data present");
        let arr = json
            .get("exclude_node_ids")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("row {row}: exclude_node_ids missing/not array in {json}"));
        let got: Vec<i64> = arr.iter().map(|v| v.as_i64().expect("int")).collect();
        // 1-shard parent → node id == PK, so resolved == reference (order kept).
        assert_eq!(
            &got, expected,
            "row {row}: exclude_node_ids must resolve exclude_target_keys in order"
        );
    }
}

#[test]
fn test_resolution_emits_no_seed_edges() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 1, 2, 3];
    let targets: Vec<Vec<i64>> = vec![vec![1], vec![2], vec![3], vec![4]];
    let output = ingest_resolution(&tmp, &entities, &targets).expect("resolution ingest");

    // (c) NO edge relation for the resolved columns: index_metadata must have
    // no edge_relation touching train_seeds, and no edges.bin file.
    let meta: diskgraph_core::metadata::DatasetMetadata =
        serde_json::from_slice(&fs::read(output.join("index_metadata.json")).unwrap()).unwrap();
    let touching: Vec<_> = meta
        .edge_relations
        .iter()
        .filter(|r| r.src_type == "train_seeds" || r.dst_type == "train_seeds")
        .collect();
    assert!(
        touching.is_empty(),
        "resolution mode must emit NO edge relation touching train_seeds; got: {:?}",
        touching
            .iter()
            .map(|r| (&r.name, &r.src_type, &r.dst_type))
            .collect::<Vec<_>>()
    );

    // The per-node `edges.bin` is the standard node-table adjacency CONTAINER
    // (it also holds node_data) and is written for every node table — its mere
    // presence is not a leak. The real no-edge guarantee is that it contains
    // ZERO edges for the resolved columns: assert every `train_seeds` node has
    // no relation segments / zero edge count. (Combined with the empty
    // `edge_relations` check above, this is the "no FK edge / no leak"
    // guarantee; the FK-mode `seed.edges.bin` payload would instead carry the
    // resolved parent edges here.)
    let reader = diskgraph_index::graph::GraphIndexReader::open(
        &output.join("index_metadata.json"),
        &output,
    )
    .expect("open produced extension");
    let table = reader.table("train_seeds").expect("train_seeds table");
    for node_id in 0..(entities.len() as u64) {
        let view = table.edges_for_node(node_id).expect("edges_for_node");
        let segs = view.relation_segments().expect("relation_segments");
        let total_edges: u64 = segs.iter().map(|s| s.count as u64).sum();
        assert_eq!(
            total_edges, 0,
            "resolution mode must emit ZERO edges for train_seeds node {node_id}; got {total_edges}"
        );
    }
}
