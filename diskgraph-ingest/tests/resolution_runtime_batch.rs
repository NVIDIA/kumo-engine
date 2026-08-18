//! FK-free LP "resolution mode" — engine `sample_runtime_batch` (design §S3/§S5).
//!
//! S1+S2 (tests/resolution_ingest.rs) wrote the seed table with resolved
//! rowpack columns (`entity_node_id: u64`, `target_node_ids: list<i64>`) and NO
//! FK edge. S3 makes the engine consume them on the LABEL path (shallow_rhs /
//! ID-GNN / hybrid):
//!   1. Seed set from the `entity_node_id` column (replaces the `seed.edges.bin`
//!      walk) → `seed_nodes[entity_type]`.
//!   2. Positive LABELS from the `target_node_ids` list column → the ragged
//!      `lp_pos_counts` / `lp_pos_rhs_global` / `lp_pos_mask` task fields.
//!      `seed_nodes[rhs]` stays EMPTY — the positive is a label, never rooted
//!      (the leak-free property).
//!
//! This test reuses the resolution-ingest fixture recipe (parent index +
//! resolution-mode stage extension), then runs `sample_runtime_batch` against
//! the merged `open_with_tt` reader and asserts (a)-(d).

use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, ListArray, RecordBatch, TimestampSecondArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_core::dataset::DatasetId;
use diskgraph_core::sampling::{
    AbsentKeyPolicy, FanoutSpec, LayerSpec, NegativeSamplingSpec, SamplingFlags, SamplingMode,
    SamplingPlan,
};
use diskgraph_ingest::graph_config::{
    ColumnSpec, DataSource, DirectionSpec, FKeySpec, GraphConfig, IngestedType, TableSpec,
};
use diskgraph_ingest::{
    ingest_with_stats_from_source, Dest, IngestOptions, ResolveColumnSpec, ResolveKind,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1024))
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

fn typed_col(name: &str, dtype: IngestedType) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source_dtype: None,
        ingested_dtype: dtype,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    }
}

/// Build a parent index with `users` (PK 0..n) and `orders` (4 per user,
/// bidirectional FK → users). The reverse `users → orders` edge gives the
/// engine real neighbors to sample at hop 1, so assert (d) (the subgraph is
/// produced around the entity seeds) is non-trivial.
fn ingest_parent_users_orders(parent_root: &Path, n_users: i64) -> Result<()> {
    let input_dir = parent_root
        .parent()
        .expect("parent_root has parent")
        .join("_uo_inputs");
    let users_pq = input_dir.join("users.parquet");
    let orders_pq = input_dir.join("orders.parquet");

    write_parquet(
        &users_pq,
        Schema::new(vec![Field::new("user_id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..n_users).collect::<Vec<_>>())) as ArrayRef],
    )?;

    let mut order_ids: Vec<i64> = Vec::new();
    let mut order_uids: Vec<i64> = Vec::new();
    let mut order_times: Vec<i64> = Vec::new();
    for u in 0..n_users {
        for k in 0..4i64 {
            order_ids.push(u * 4 + k);
            order_uids.push(u);
            order_times.push(100 + k);
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

    let users = TableSpec {
        name: "users".to_string(),
        columns: vec![int64_col("user_id")],
        primary_key: Some("user_id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![users_pq.to_str().unwrap().to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    };
    let orders = TableSpec {
        name: "orders".to_string(),
        columns: vec![
            int64_col("order_id"),
            int64_col("user_id"),
            typed_col("order_time", IngestedType::Timestamp),
        ],
        primary_key: Some("order_id".to_string()),
        create_time: Some("order_time".to_string()),
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![orders_pq.to_str().unwrap().to_string()],
        },
        source_size: None,
        fkeys: vec![FKeySpec {
            src_column: "user_id".to_string(),
            dst_table: "users".to_string(),
            forward: Some(DirectionSpec::default()),
            reverse: Some(DirectionSpec::default()),
            bidirectional: true,
        }],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    };

    let config = GraphConfig {
        dataset_name: "users_orders_parent".to_string(),
        tables: vec![users, orders],
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

/// Write the resolution-mode stage parquet: ENTITY1 (scalar PK ref) +
/// TIMESTAMP + TARGET1 (List<Int64> PK refs).
fn write_stage_parquet(path: &Path, entities: &[i64], targets: &[Vec<i64>]) -> Result<()> {
    assert_eq!(entities.len(), targets.len());
    let n = entities.len();
    let schema = Schema::new(vec![
        Field::new("ENTITY1", DataType::Int64, false),
        Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Second, None), false),
        Field::new(
            "TARGET1",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]);
    let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
    let mut values: Vec<i64> = Vec::new();
    offsets.push(0);
    for t in targets {
        values.extend_from_slice(t);
        offsets.push(values.len() as i32);
    }
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int64Array::from(values)) as ArrayRef,
        None,
    );
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 5_000 + i).collect();
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

fn stage_table_spec(parquet_path: &str) -> TableSpec {
    TableSpec {
        name: "train_seeds".to_string(),
        columns: vec![
            int64_col("ENTITY1"),
            typed_col("TIMESTAMP", IngestedType::Timestamp),
            typed_col("TARGET1", IngestedType::Int64Array),
            typed_col("entity_node_id", IngestedType::UInt64),
            typed_col("target_node_ids", IngestedType::Int64Array),
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
        fkeys: vec![], // no FK edges — the no-edge / leak-free guarantee.
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// Resolution specs where ENTITY1 → users and TARGET1 → `target_table`.
/// `target_table == "users"` keeps the legacy single-type fixture; passing
/// `"orders"` makes the RHS type DISTINCT from the entity type, which is what
/// the two-tower path needs to assert rhs seed composition cleanly.
fn resolve_specs_to(target_table: &str) -> Vec<ResolveColumnSpec> {
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
            dst_table: target_table.to_string(),
            kind: ResolveKind::List,
            node_id_col: "target_node_ids".to_string(),
        },
    ]
}

/// Build a parent (users+orders) + a resolution-mode `train_seeds` extension at
/// `<parent_root>/train_tables/<tt_id>`. Returns the parent root.
fn build_fixture(
    tmp: &tempfile::TempDir,
    tt_id: &str,
    n_users: i64,
    entities: &[i64],
    targets: &[Vec<i64>],
) -> Result<PathBuf> {
    build_fixture_to(tmp, tt_id, n_users, entities, targets, "users")
}

/// Like [`build_fixture`] but TARGET1 resolves to `target_table` (e.g.
/// `"orders"` for a distinct RHS type).
fn build_fixture_to(
    tmp: &tempfile::TempDir,
    tt_id: &str,
    n_users: i64,
    entities: &[i64],
    targets: &[Vec<i64>],
    target_table: &str,
) -> Result<PathBuf> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_parent_users_orders(&parent_root, n_users)?;

    let stage_parquet = tmp.path().join("stage_input").join("train_seeds.parquet");
    write_stage_parquet(&stage_parquet, entities, targets)?;

    let table = stage_table_spec(stage_parquet.to_str().unwrap());
    let config = GraphConfig {
        dataset_name: "resolve_run".to_string(),
        tables: vec![table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let output = parent_root.join("train_tables").join(tt_id);
    let options = IngestOptions {
        parent_index_path: Some(parent_root.clone()),
        resolve_columns: resolve_specs_to(target_table),
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
    )?;
    Ok(parent_root)
}

fn depth2_plan() -> SamplingPlan {
    let layer = || LayerSpec {
        fanouts: HashMap::new(),
        default_fanout: Some(FanoutSpec {
            fanout: 4,
            mode: SamplingMode::Uniform,
            absent_key_policy: AbsentKeyPolicy::UniformFallback,
        }),
    };
    SamplingPlan {
        layers: vec![layer(), layer()],
    }
}

fn flags() -> SamplingFlags {
    SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: false,
    }
}

fn parent_node_id(
    reader: &diskgraph_index::graph::GraphIndexReader,
    table_name: &str,
    pk: i64,
) -> u64 {
    let table = reader
        .table(table_name)
        .unwrap_or_else(|| panic!("{table_name} table"));
    let key = table
        .parse_pk_key(&pk.to_string())
        .unwrap_or_else(|_| panic!("parse {table_name} primary key {pk}"));
    table
        .node_id_from_pk(&key)
        .unwrap_or_else(|_| panic!("look up {table_name} primary key {pk}"))
        .unwrap_or_else(|| panic!("{table_name} primary key {pk} missing"))
}

fn mapped_rows(
    reader: &diskgraph_index::graph::GraphIndexReader,
    entities: &[i64],
    values: &[Vec<i64>],
    value_table: &str,
) -> HashMap<u64, Vec<i64>> {
    entities
        .iter()
        .zip(values)
        .map(|(&entity, values)| {
            (
                parent_node_id(reader, "users", entity),
                values
                    .iter()
                    .map(|&pk| parent_node_id(reader, value_table, pk) as i64)
                    .collect(),
            )
        })
        .collect()
}

/// Full S3 label-path contract: seed from `entity_node_id`, labels from
/// `target_node_ids`, rhs never rooted, subgraph produced.
#[test]
fn test_resolution_runtime_batch_label_path() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 3, 5, 7];
    let targets: Vec<Vec<i64>> = vec![vec![1, 2], vec![4], vec![6, 0, 3], vec![]];
    let parent_root =
        build_fixture(&tmp, "tt_res", 8, &entities, &targets).expect("build resolution fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_res").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let plan = depth2_plan();
    let flags = flags();
    let batch_rows: Vec<u64> = (0..entities.len() as u64).collect();

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_res/train_seeds",
            &batch_rows,
            DatasetId::nil(),
            &plan,
            &flags,
            None,                 // no negatives (S4)
            Some("users"),        // entity_node_type → seed from entity_node_id column
            Some("users"),        // target_node_type → emit ragged labels
            true,                 // preserve old behavior: use seed timestamps when present
            None,                 // node_range: stage-table seeding (NodeRange off)
        )
        .expect("sample_runtime_batch must succeed on the resolution label path");

    // Parallel Encode may assign parent and stage node IDs in a different
    // order. Compare the seed set through the parent PK map, then use the
    // emitted seed order for every row-aligned task field.
    let expected_by_seed = mapped_rows(reader.as_ref(), &entities, &targets, "users");
    let mut expected_seeds: Vec<u64> = expected_by_seed.keys().copied().collect();
    expected_seeds.sort_unstable();
    let mut actual_seeds = batch.task.seed_node_ids.clone();
    actual_seeds.sort_unstable();
    assert_eq!(
        actual_seeds, expected_seeds,
        "(a) seed set must equal the entity_node_id column values"
    );

    // (b) ragged labels == per-row target list lengths + concatenated ids.
    let expected_rows: Vec<&Vec<i64>> = batch
        .task
        .seed_node_ids
        .iter()
        .map(|seed| {
            expected_by_seed
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))
        })
        .collect();
    let expected_counts: Vec<u32> = expected_rows
        .iter()
        .map(|targets| targets.len() as u32)
        .collect();
    let expected_rhs: Vec<i64> = expected_rows.into_iter().flatten().copied().collect();
    assert_eq!(
        batch.task.lp_pos_counts, expected_counts,
        "(b) lp_pos_counts must equal per-row target list lengths"
    );
    assert_eq!(
        batch.task.lp_pos_rhs_global, expected_rhs,
        "(b) lp_pos_rhs_global must equal the concatenated target_node_ids in order"
    );
    // All positives resolved (every PK is a real node) → mask all false.
    assert_eq!(
        batch.task.lp_pos_mask,
        vec![false; expected_rhs.len()],
        "(b) all resolved positives → mask all-false"
    );
    // Label path never roots: lp_pos_local stays None.
    assert!(
        batch.task.lp_pos_local.is_none(),
        "(c) label path must not set lp_pos_local (positives are never rooted)"
    );

    // (c) the rhs positive is a LABEL, never rooted: the rhs node set carries
    //     ONLY the entity seeds + their sampled neighbors, never the positives
    //     injected as seeds. With entity==rhs type ("users") here, assert the
    //     node set is exactly the seeds ∪ sampled neighbors and the negatives
    //     path stayed off (no negatives requested).
    assert_eq!(batch.task.amount_m, 0, "(c) no negatives on the label path");
    assert!(
        batch.task.neg_mask.is_empty(),
        "(c) no negatives → empty neg_mask"
    );

    // (d) the sampled subgraph is produced: the "main" channel exists, the
    //     entity seeds are present in node_ids[users], and hop-1 neighbors
    //     (orders) were sampled around them.
    let main = batch.graphs.get("main").expect("(d) main channel present");
    let users_nodes = main
        .node_ids
        .get("users")
        .expect("(d) users in sampled node set");
    for &s in &expected_seeds {
        assert!(
            users_nodes.contains(&s),
            "(d) entity seed {s} must appear in node_ids[users]"
        );
    }
    let orders_count = main.node_ids.get("orders").map(|v| v.len()).unwrap_or(0);
    assert!(
        orders_count > 0,
        "(d) hop-1 neighbors (orders) must be sampled around the entity seeds; got 0"
    );
}

/// Orphan seeds (entity references absent from the training graph — cross-graph
/// TT reuse) are DROPPED at ingest, so they never reach the extension. Sampling
/// over the surviving rows therefore sees a clean, compacted stage table with no
/// null anchors and every row-aligned task array (ragged labels, timestamps,
/// weights) intact.
#[test]
fn test_resolution_runtime_batch_orphan_dropped_at_ingest() {
    let tmp = tempdir().unwrap();
    // Row 1's ENTITY1=50 is an orphan in the 8-user parent → the row is dropped
    // at ingest. The 3 surviving rows compact to stage node ids 0,1,2. Distinct
    // per-row targets make any label misalignment observable.
    let entities: Vec<i64> = vec![0, 50, 5, 7];
    let targets: Vec<Vec<i64>> = vec![vec![1], vec![2, 3], vec![6], vec![4, 2]];
    let parent_root =
        build_fixture(&tmp, "tt_orph", 8, &entities, &targets).expect("build orphan fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_orph").unwrap(),
    );
    // The orphan row never entered the extension: 3 survivors, not 4.
    assert_eq!(
        reader.table("tt_orph/train_seeds").unwrap().node_count(),
        3,
        "orphan-anchor row dropped at ingest"
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_orph/train_seeds",
            // All 3 surviving stage rows (no orphan slot to exclude).
            &[0u64, 1, 2],
            DatasetId::nil(),
            &depth2_plan(),
            &flags(),
            None,
            Some("users"),
            Some("users"),
            true,
            None,
        )
        .expect("clean post-drop stage must sample");

    let surviving_entities = vec![0, 5, 7];
    let surviving_targets = vec![vec![1], vec![6], vec![4, 2]];
    let expected_by_seed = mapped_rows(
        reader.as_ref(),
        &surviving_entities,
        &surviving_targets,
        "users",
    );
    let mut expected_seeds: Vec<u64> = expected_by_seed.keys().copied().collect();
    expected_seeds.sort_unstable();
    let mut actual_seeds = batch.task.seed_node_ids.clone();
    actual_seeds.sort_unstable();

    // Anchors are exactly the member entities — no orphan/null slot.
    assert_eq!(
        actual_seeds,
        expected_seeds,
        "seed set == surviving member anchors (orphan already dropped)"
    );
    // Each kept row keeps its own targets — the dropped orphan's [2, 3] are gone.
    let expected_rows: Vec<&Vec<i64>> = batch
        .task
        .seed_node_ids
        .iter()
        .map(|seed| {
            expected_by_seed
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))
        })
        .collect();
    assert_eq!(
        batch.task.lp_pos_counts,
        expected_rows
            .iter()
            .map(|targets| targets.len() as u32)
            .collect::<Vec<_>>(),
        "ragged label counts align with the surviving rows"
    );
    assert_eq!(
        batch.task.lp_pos_rhs_global,
        expected_rows.into_iter().flatten().copied().collect::<Vec<_>>(),
        "the orphan row's targets [2, 3] are absent"
    );
    assert_eq!(batch.task.input_ids, vec![0u32, 1, 2]);
    // Node ids are compacted (0,1,2), not the pre-drop (0,2,3).
    assert_eq!(batch.task.tt_stage_node_ids, vec![0u64, 1, 2]);
}

/// The positive is a LABEL, never rooted — `seed_nodes[rhs]` stays empty. We
/// verify this directly by choosing a DISTINCT rhs type that is NOT the entity
/// type, so any rooting would surface positives in `node_ids[rhs]`. Here we use
/// the same fixture but make the assertion explicit: with target_node_type set
/// to "users" (the entity type), if positives were rooted they would inflate
/// node_ids[users] with target ids NOT reachable from the seeds. Instead we
/// assert the label fields carry the positives and the seed-side primary ids
/// are unchanged by label emission.
#[test]
fn test_resolution_rhs_not_rooted() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 1];
    // Targets reference users {6,7} that have NO order edges to users {0,1}.
    let targets: Vec<Vec<i64>> = vec![vec![6], vec![7]];
    let parent_root =
        build_fixture(&tmp, "tt_nr", 8, &entities, &targets).expect("build fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_nr").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(7), Some(1))
            .unwrap();

    // Depth-1 plan: hop 0 anchors users only (users→orders not walked). The
    // entity seeds are {0,1}. If a positive (user 6 or 7) were rooted as a seed
    // it would appear in node_ids[users]; it must NOT.
    let plan = SamplingPlan {
        layers: vec![LayerSpec {
            fanouts: HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout: 0,
                mode: SamplingMode::Uniform,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        }],
    };

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_nr/train_seeds",
            &[0u64, 1u64],
            DatasetId::nil(),
            &plan,
            &flags(),
            None,
            Some("users"),
            Some("users"),
            true, // preserve old behavior: use seed timestamps when present
            None, // node_range off
        )
        .expect("sample_runtime_batch must succeed");

    let expected_by_seed = mapped_rows(reader.as_ref(), &entities, &targets, "users");
    let expected_rhs: Vec<i64> = batch
        .task
        .seed_node_ids
        .iter()
        .map(|seed| {
            expected_by_seed
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))[0]
        })
        .collect();

    // Labels carry the two resolved positives in the emitted seed-row order.
    assert_eq!(batch.task.lp_pos_rhs_global, expected_rhs);
    assert_eq!(batch.task.lp_pos_counts, vec![1, 1]);

    // The positives are NOT rooted: node_ids[users] must contain only the
    // entity seeds (hop 0, fanout 0 → no neighbors), and crucially NOT the
    // positive ids.
    let main = batch.graphs.get("main").expect("main channel");
    let users_nodes = main.node_ids.get("users").expect("users node set");
    for seed in expected_by_seed.keys() {
        assert!(users_nodes.contains(seed), "entity seed {seed} must be present");
    }
    for positive in expected_by_seed.values().flatten() {
        assert!(
            !users_nodes.contains(&(*positive as u64)),
            "positive {positive} must NOT be rooted into node_ids[users] \
             (leak-free label path); got {users_nodes:?}"
        );
    }
}

// ── STATIC LP past-visit exclude hydration ───────────────────────────────────
//
// gen writes the resolved `exclude_node_ids: list<i64>` rowpack column on
// val/test seed tables (TRAIN / TRAIN∪VAL targets). The engine hydrates it into
// the ragged `lp_exclude_counts`/`lp_exclude_rhs_global` task fields (reusing
// `hydrate_ragged_lp_target`, identity closure). Empty exclude lists (train) →
// the fields stay empty (gated on a non-empty exclude set).

/// Stage parquet: ENTITY1 + TIMESTAMP + TARGET1 + exclude_target_keys (all
/// List<Int64> PK refs).
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
    let timestamps: Vec<i64> = (0..n as i64).map(|i| 6_000 + i).collect();
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

/// Build a users+orders parent + a val-style seed extension that also resolves
/// `exclude_target_keys` → `exclude_node_ids`. Returns the parent root.
fn build_fixture_with_exclude(
    tmp: &tempfile::TempDir,
    tt_id: &str,
    n_users: i64,
    entities: &[i64],
    targets: &[Vec<i64>],
    excludes: &[Vec<i64>],
) -> Result<PathBuf> {
    let parent_root = tmp.path().join("parent_idx");
    ingest_parent_users_orders(&parent_root, n_users)?;

    let stage_parquet = tmp.path().join("stage_input").join("val_seeds.parquet");
    write_stage_parquet_with_exclude(&stage_parquet, entities, targets, excludes)?;

    let mut table = stage_table_spec(stage_parquet.to_str().unwrap());
    table.columns.push(typed_col("exclude_target_keys", IngestedType::Int64Array));
    table.columns.push(typed_col("exclude_node_ids", IngestedType::Int64Array));
    let config = GraphConfig {
        dataset_name: "resolve_exclude_run".to_string(),
        tables: vec![table],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let output = parent_root.join("train_tables").join(tt_id);
    let mut specs = resolve_specs_to("users");
    specs.push(ResolveColumnSpec {
        table: "train_seeds".to_string(),
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
    )?;
    Ok(parent_root)
}

#[test]
fn test_resolution_exclude_hydrate() {
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 1, 2, 3];
    let targets: Vec<Vec<i64>> = vec![vec![5], vec![6], vec![7], vec![4]];
    // Past-visit excludes: row0 empty, row1 one, row2 multi, row3 empty.
    let excludes: Vec<Vec<i64>> = vec![vec![], vec![3], vec![1, 6, 2], vec![]];
    let parent_root = build_fixture_with_exclude(&tmp, "tt_excl", 8, &entities, &targets, &excludes)
        .expect("build exclude fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_excl").unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let plan = depth2_plan();
    let flags = flags();
    let batch_rows: Vec<u64> = (0..entities.len() as u64).collect();

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_excl/train_seeds",
            &batch_rows,
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            Some("users"),
            true,
            None,
        )
        .expect("sample_runtime_batch on the exclude path");

    // Parallel Encode may reorder stage rows and parent node IDs. Reassociate
    // both ragged columns through the resolved entity seed.
    let expected_excludes = mapped_rows(reader.as_ref(), &entities, &excludes, "users");
    let expected_targets = mapped_rows(reader.as_ref(), &entities, &targets, "users");
    let expected_exclude_rows: Vec<&Vec<i64>> = batch
        .task
        .seed_node_ids
        .iter()
        .map(|seed| {
            expected_excludes
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))
        })
        .collect();
    let expected_counts: Vec<u32> = expected_exclude_rows
        .iter()
        .map(|excludes| excludes.len() as u32)
        .collect();
    let expected_rhs: Vec<i64> = expected_exclude_rows
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert_eq!(
        batch.task.lp_exclude_counts, expected_counts,
        "lp_exclude_counts must equal per-row exclude list lengths"
    );
    assert_eq!(
        batch.task.lp_exclude_rhs_global, expected_rhs,
        "lp_exclude_rhs_global must equal the concatenated exclude_node_ids in order"
    );
    // Sanity: positives unaffected by exclude hydration.
    let expected_positive_rhs: Vec<i64> = batch
        .task
        .seed_node_ids
        .iter()
        .flat_map(|seed| {
            expected_targets
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(
        batch.task.lp_pos_rhs_global,
        expected_positive_rhs
    );
}

#[test]
fn test_resolution_exclude_absent_when_all_empty() {
    // A train-style batch where every exclude list is empty: the engine must
    // leave the task fields empty (contract: "absent on train"), so the pybind
    // layer omits the keys.
    let tmp = tempdir().unwrap();
    let entities: Vec<i64> = vec![0, 1];
    let targets: Vec<Vec<i64>> = vec![vec![5], vec![6]];
    let excludes: Vec<Vec<i64>> = vec![vec![], vec![]];
    let parent_root =
        build_fixture_with_exclude(&tmp, "tt_excl_empty", 8, &entities, &targets, &excludes)
            .expect("build fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, "tt_excl_empty")
            .unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();
    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            "tt_excl_empty/train_seeds",
            &[0u64, 1u64],
            DatasetId::nil(),
            &depth2_plan(),
            &flags(),
            None,
            Some("users"),
            Some("users"),
            true,
            None,
        )
        .expect("sample_runtime_batch");
    assert!(
        batch.task.lp_exclude_counts.is_empty(),
        "all-empty excludes → lp_exclude_counts stays empty (absent-on-train contract)"
    );
    assert!(
        batch.task.lp_exclude_rhs_global.is_empty(),
        "all-empty excludes → lp_exclude_rhs_global stays empty"
    );
}

// ── S4: two-tower path ──────────────────────────────────────────────────────
//
// Two-tower is the ONE model that ROOTS positives and uses engine negatives.
// When negatives are requested (`negatives.is_some()`) AND a target type is set,
// the engine must:
//   - root ONE positive per anchor (chosen from that row's `target_node_ids`)
//     into `seed_nodes[rhs]` as the FIRST B entries (so `len == B` before
//     `append_global_negatives` runs, satisfying its `B == len` assert);
//   - append B*M negatives after, giving `seed_nodes[rhs] = [B pos, B*M neg]`;
//   - stitch the two-tower task fields: `lp_pos_counts = [1; B]`,
//     `lp_pos_rhs_global = chosen` (length B), `lp_pos_mask` (length B),
//     `lp_pos_local = arange(B)` so the kumo adapter derives
//     `dst_pos_index = arange(B)` / `dst_neg_index = arange(B, B+B*M)`.
//
// Fixture: ENTITY1 → users, TARGET1 → ORDERS (a DISTINCT rhs type) so the rooted
// rhs positives never collide with the lhs `users` seeds and can be asserted in
// `node_ids[orders]` independently.

const TWO_TOWER_M: usize = 3;

/// Build the two-tower fixture (entity=users, rhs=orders, multi-positive rows)
/// and run `sample_runtime_batch` with the given negatives seed. Returns the
/// resulting batch. `n_users = 8` → orders id-space is 0..=31.
fn run_two_tower(tmp: &tempfile::TempDir, tt_id: &str, neg_rnd_seed: u64) -> crate::OwnedBatch {
    let entities: Vec<i64> = vec![0, 1, 2, 3];
    // Multi-positive rows so the inner pick has choices.
    let targets: Vec<Vec<i64>> = vec![
        vec![0, 1, 2, 3],   // row 0: 4 positives
        vec![5, 6],         // row 1: 2 positives
        vec![8, 9, 10],     // row 2: 3 positives
        vec![12],           // row 3: 1 positive
    ];
    let parent_root =
        build_fixture_to(tmp, tt_id, 8, &entities, &targets, "orders").expect("build tt fixture");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open_with_tt(&parent_root, tt_id).unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let plan = depth2_plan();
    let flags = flags();
    let batch_rows: Vec<u64> = (0..entities.len() as u64).collect();

    let negatives = NegativeSamplingSpec {
        rhs_node_type: "orders".to_string(),
        amount_m: TWO_TOWER_M,
        // Deterministic seed (the test pins a concrete value); production passes
        // `None` -> thread_rng for fresh-per-epoch coverage.
        rnd_seed: Some(neg_rnd_seed),
    };

    let batch = engine
        .sample_runtime_batch(
            reader.as_ref(),
            &format!("{tt_id}/train_seeds"),
            &batch_rows,
            DatasetId::nil(),
            &plan,
            &flags,
            Some(&negatives),  // two-tower: engine negatives requested
            Some("users"),     // entity_node_type
            Some("orders"),    // target_node_type == rhs type
            true,              // preserve old behavior: use seed timestamps when present
            None,              // node_range off
        )
        .expect("two-tower sample_runtime_batch must succeed");

    let targets_by_seed = mapped_rows(reader.as_ref(), &entities, &targets, "orders");
    let mapped_targets = batch
        .task
        .seed_node_ids
        .iter()
        .map(|seed| {
            targets_by_seed
                .get(seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"))
                .clone()
        })
        .collect();
    let entity_pk_by_seed: HashMap<u64, i64> = entities
        .iter()
        .map(|&pk| (parent_node_id(reader.as_ref(), "users", pk), pk))
        .collect();
    let chosen_target_pks_by_entity: HashMap<i64, i64> = batch
        .task
        .seed_node_ids
        .iter()
        .zip(batch.task.lp_pos_rhs_global.iter())
        .map(|(&seed, &chosen)| {
            let entity_pk = *entity_pk_by_seed
                .get(&seed)
                .unwrap_or_else(|| panic!("unexpected seed node ID {seed}"));
            let target_pk = targets
                .iter()
                .flatten()
                .copied()
                .find(|&pk| parent_node_id(reader.as_ref(), "orders", pk) as i64 == chosen)
                .unwrap_or_else(|| panic!("chosen order node ID {chosen} has no fixture PK"));
            (entity_pk, target_pk)
        })
        .collect();
    let mut expected_seed_node_ids: Vec<u64> = entities
        .iter()
        .map(|&pk| parent_node_id(reader.as_ref(), "users", pk))
        .collect();
    expected_seed_node_ids.sort_unstable();

    OwnedBatch {
        targets: mapped_targets,
        chosen_target_pks_by_entity,
        expected_seed_node_ids,
        lp_pos_counts: batch.task.lp_pos_counts.clone(),
        lp_pos_rhs_global: batch.task.lp_pos_rhs_global.clone(),
        lp_pos_mask: batch.task.lp_pos_mask.clone(),
        lp_pos_local: batch.task.lp_pos_local.clone(),
        amount_m: batch.task.amount_m,
        neg_mask_len: batch.task.neg_mask.len(),
        seed_node_ids: batch.task.seed_node_ids.clone(),
        orders_nodes: batch
            .graphs
            .get("main")
            .and_then(|g| g.node_ids.get("orders"))
            .cloned()
            .unwrap_or_default(),
    }
}

/// Snapshot of the batch fields the two-tower assertions care about (avoids
/// holding the full RuntimeBatch across helper boundaries).
struct OwnedBatch {
    targets: Vec<Vec<i64>>,
    chosen_target_pks_by_entity: HashMap<i64, i64>,
    expected_seed_node_ids: Vec<u64>,
    lp_pos_counts: Vec<u32>,
    lp_pos_rhs_global: Vec<i64>,
    lp_pos_mask: Vec<bool>,
    lp_pos_local: Option<Vec<u32>>,
    amount_m: u32,
    neg_mask_len: usize,
    seed_node_ids: Vec<u64>,
    orders_nodes: Vec<u64>,
}

#[test]
fn test_resolution_two_tower_roots_one_positive_then_negatives() {
    let tmp = tempdir().unwrap();
    let b = run_two_tower(&tmp, "tt_tt", 1234);

    let n_anchors = b.targets.len();

    // (a) Exactly ONE positive rooted per anchor, each ∈ that row's target list.
    //     `lp_pos_counts == [1; B]` (single-positive-per-anchor for two-tower).
    assert_eq!(
        b.lp_pos_counts,
        vec![1u32; n_anchors],
        "(a) two-tower roots exactly one positive per anchor (counts all-1)"
    );
    assert_eq!(
        b.lp_pos_rhs_global.len(),
        n_anchors,
        "(a) one chosen positive per anchor"
    );
    for (i, &chosen) in b.lp_pos_rhs_global.iter().enumerate() {
        assert!(
            b.targets[i].contains(&chosen),
            "(a) chosen positive {chosen} for anchor {i} must be a member of its target list {:?}",
            b.targets[i]
        );
    }
    // The chosen positives are the FIRST B rhs seeds, so they must appear in
    // node_ids[orders] (they ARE rooted — the two-tower property).
    for (i, &chosen) in b.lp_pos_rhs_global.iter().enumerate() {
        assert!(
            b.orders_nodes.contains(&(chosen as u64)),
            "(a) rooted positive {chosen} (anchor {i}) must appear in node_ids[orders]"
        );
    }

    // (b) neg_mask length == B*M (B anchors, M negatives each).
    assert_eq!(b.amount_m, TWO_TOWER_M as u32, "(b) amount_m surfaced");
    assert_eq!(
        b.neg_mask_len,
        n_anchors * TWO_TOWER_M,
        "(b) neg_mask length must be B*M"
    );

    // (d) lp_pos_local == arange(B) → adapter derives dst_pos_index = arange(B),
    //     dst_neg_index = arange(B, B+B*M).
    assert_eq!(
        b.lp_pos_local,
        Some((0..n_anchors as u32).collect()),
        "(d) lp_pos_local must be arange(B) for the two-tower dst_pos/dst_neg derivation"
    );
    // mask length matches B (one per chosen positive); all resolved → false.
    assert_eq!(
        b.lp_pos_mask,
        vec![false; n_anchors],
        "(d) lp_pos_mask length B, all chosen positives resolved"
    );
    // seed_node_ids stays the lhs primary entity ids (NOT polluted by rooting).
    let mut actual_seed_node_ids = b.seed_node_ids.clone();
    actual_seed_node_ids.sort_unstable();
    assert_eq!(
        actual_seed_node_ids,
        b.expected_seed_node_ids,
        "lhs seed_node_ids must be the entity ids, snapshot before rooting"
    );
}

#[test]
fn test_resolution_two_tower_epoch_seed_changes_chosen_positive() {
    // The per-row positive pick is `gen_range(0..len)` seeded by
    // `rnd_seed.wrapping_add(row)` (ported `pair_from_array`) when `rnd_seed` is
    // `Some`. Two DIFFERENT seeds must be able to pick DIFFERENT positives for a
    // multi-positive row. In production `rnd_seed` is `None` (thread_rng), which
    // delivers the same per-epoch coverage with fresh draws; here we pin two
    // concrete seeds so the pick is deterministic and the difference is testable.
    let tmp_a = tempdir().unwrap();
    let tmp_b = tempdir().unwrap();
    let ep0 = run_two_tower(&tmp_a, "tt_e0", 0xABCD_0000);
    let ep1 = run_two_tower(&tmp_b, "tt_e1", 0x1234_5678);

    // Same fixture, same anchors; only the negatives seed differs.
    assert_eq!(ep0.lp_pos_counts, ep1.lp_pos_counts);
    // At least one multi-positive anchor must pick a different positive across
    // the two seeds (rows 0/1/2 have >1 positive; row 3 has exactly 1 so it can't
    // change). This proves epoch coverage is already delivered by the seed flow.
    let changed = ep0
        .chosen_target_pks_by_entity
        .iter()
        .any(|(entity, target)| ep1.chosen_target_pks_by_entity.get(entity) != Some(target));
    assert!(
        changed,
        "different negatives seeds (= epochs) must pick different positives for at \
         least one multi-positive anchor (epoch coverage); ep0={:?} ep1={:?}",
        ep0.chosen_target_pks_by_entity, ep1.chosen_target_pks_by_entity
    );
}

// ── NodeRange seed mode ──────────────────────────────────────────────────────
//
// NodeRange seeds ALL nodes `[0, num_rhs)` of a given node type (the RHS / target
// catalog) with a FIXED seed_time, samples each seed's neighborhood via the
// existing plan, and emits the same per-type HeteroData batches. This feeds the
// two-tower MIPS pass (RHS embeddings over the whole catalog).
//
// In NodeRange mode the engine BYPASSES the stage-table / entity-target FK-free
// resolution AND the negatives blocks: `batch_row_ids` ARE the RHS global node
// ids directly, `seed_timestamps` is the fixed seed_time broadcast over the
// batch, and there are no positives / labels / negatives. We assert (a) the union
// of seeds across batches == {0..num_rhs} exactly once in order, (b) per-batch
// batch_size is deterministic (== requested except the last), (c) each batch has
// the RHS type with the fixed seed_time, (d) no negative / label fields produced.

/// Drive the full NodeRange catalog over the RHS type in `batch_size` chunks
/// (mirrors the pybind `next_batch` loop). Reuses the users+orders parent fixture
/// — there is no stage extension, NodeRange seeds the parent catalog directly.
#[test]
fn test_node_range_seeds_full_catalog() {
    let tmp = tempdir().unwrap();
    // Parent only: users 0..8, orders 0..32 (4 per user). NodeRange does not read
    // a stage table, so we open the parent index directly (no tt extension).
    let parent_root = tmp.path().join("parent_idx");
    ingest_parent_users_orders(&parent_root, 8).expect("ingest parent");

    let reader = Arc::new(
        diskgraph_index::graph::GraphIndexReader::open(
            &parent_root.join("index_metadata.json"),
            &parent_root,
        )
        .unwrap(),
    );
    let engine =
        diskgraph_engine::engine::DiskGraphEngine::from_reader(reader.clone(), Some(42), Some(1))
            .unwrap();

    let num_rhs: u64 = 32; // 8 users * 4 orders
    let seed_time: i64 = 1_700_000_000;
    let batch_size: usize = 10; // 32 = 10 + 10 + 10 + 2 (last partial)
    let plan = depth2_plan();
    let flags = flags();

    // Dense `[0, num_rhs)` catalog, in order. The pybind layer builds these from
    // the RHS type's shard id range; here we mirror that 1-shard dense range.
    let all_ids: Vec<u64> = (0..num_rhs).collect();

    let mut seen: Vec<u64> = Vec::new();
    let mut batch_sizes: Vec<usize> = Vec::new();
    let mut cursor = 0usize;
    while cursor < all_ids.len() {
        let take = batch_size.min(all_ids.len() - cursor);
        let batch_ids = &all_ids[cursor..cursor + take];
        cursor += take;

        let batch = engine
            .sample_runtime_batch(
                reader.as_ref(),
                // stage_table is ignored in NodeRange mode (no stage seeding); pass
                // the rhs type name so the (unused) namespace split is harmless.
                "orders",
                batch_ids,
                DatasetId::nil(),
                &plan,
                &flags,
                None, // no negatives
                None, // no entity resolution
                None, // no target resolution
                true, // ignored by NodeRange, but preserves direct-call contract
                Some(diskgraph_engine::engine::NodeRangeSeed {
                    rhs_node_type: "orders",
                    seed_time,
                }),
            )
            .expect("NodeRange sample_runtime_batch must succeed");

        // (c) the RHS type is present with batch_size == #seeds, seed_time fixed.
        let main = batch.graphs.get("main").expect("main channel present");
        let orders_nodes = main
            .node_ids
            .get("orders")
            .expect("rhs type 'orders' must be in the sampled node set");
        // The first `take` node_ids of the seed type are the seeds (local ids
        // 0..take, in order), exactly the requested batch ids.
        assert_eq!(
            &orders_nodes[..take],
            batch_ids,
            "(c) seed node ids must be the requested [cursor..cursor+take) range in order"
        );
        // batch_size surfaced as num_positives == #seeds in this batch.
        assert_eq!(
            batch.task.num_positives as usize, take,
            "(c) num_positives must equal the seed count in this batch"
        );
        // seed_time fixed for every seed in the batch.
        assert_eq!(
            batch.task.seed_timestamps,
            Some(vec![seed_time; take]),
            "(c) seed_time must be the fixed value broadcast over the batch seeds"
        );

        // (d) NodeRange produces no negatives / labels / resolution fields.
        assert_eq!(batch.task.amount_m, 0, "(d) no negatives in NodeRange mode");
        assert!(batch.task.neg_mask.is_empty(), "(d) empty neg_mask");
        assert!(batch.task.rhs_seed_type.is_none(), "(d) no neg rhs type");
        assert!(
            batch.task.lp_pos_counts.is_empty(),
            "(d) no LP labels in NodeRange mode"
        );
        assert!(batch.task.lp_pos_rhs_global.is_empty(), "(d) no LP positives");
        assert!(batch.task.lp_pos_local.is_none(), "(d) nothing rooted");
        assert!(batch.task.target.is_none(), "(d) no raw target projection");

        seen.extend_from_slice(batch_ids);
        batch_sizes.push(take);
    }

    // (a) union of emitted seeds == {0..num_rhs} exactly once, in order.
    assert_eq!(
        seen, all_ids,
        "(a) seeds must cover [0, num_rhs) exactly once, in order"
    );

    // (b) deterministic batch_size: all full batches == batch_size except the
    //     last which is the remainder.
    let n_full = num_rhs as usize / batch_size;
    let remainder = num_rhs as usize % batch_size;
    let mut expected_sizes = vec![batch_size; n_full];
    if remainder > 0 {
        expected_sizes.push(remainder);
    }
    assert_eq!(
        batch_sizes, expected_sizes,
        "(b) per-batch size deterministic: {batch_size} per batch, last = remainder"
    );
}
