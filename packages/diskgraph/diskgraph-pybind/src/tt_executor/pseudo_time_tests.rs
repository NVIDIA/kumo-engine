// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{
    Float64Builder, Int64Builder, LargeListBuilder, ListBuilder, StringBuilder,
};
use arrow_array::TimestampNanosecondArray;
use arrow_array::{Array, ArrayRef, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{Field, Schema};
use diskgraph_core::dataset::DatasetId;
use diskgraph_core::sampling::{
    AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode, SamplingPlan,
};
use diskgraph_core::time::Seconds;
use diskgraph_core::tt_builder::{
    AggFunc, DiskGraphTTPlan, EdgeRef, EntitySpec, FactPseudoTimeSpec, Op, PredictionType,
    RefResolveKind, RefResolveSpec, SplitSpec, Stage, Value,
};
use diskgraph_engine::engine::{DiskGraphEngine, NodeRangeSeed};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::graph_config::{
    ColumnSpec, DataSource, FKeySpec, GraphConfig, IngestedType, TableSpec,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use pyo3::types::PyAnyMethods;

use crate::runtime_batch_iterator::PseudoTimeSeed;

const USERS: i64 = 8;
const ITEMS: i64 = 10;
const STAGES: [&str; 3] = ["train_seeds", "val_seeds", "test_seeds"];
const SEED_TIMES: [u8; 3] = [1, 1, 2];

fn write_parquet(path: &Path, columns: Vec<(&str, ArrayRef)>) {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, a)| Field::new(*name, a.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let batch =
        RecordBatch::try_new(schema.clone(), columns.into_iter().map(|c| c.1).collect()).unwrap();
    let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn int_col(name: &str) -> ColumnSpec {
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

fn table(name: &str, cols: &[&str], pk: &str, path: &Path, fkeys: Vec<FKeySpec>) -> TableSpec {
    TableSpec {
        name: name.to_string(),
        columns: cols.iter().map(|c| int_col(c)).collect(),
        primary_key: Some(pk.to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![path.to_string_lossy().to_string()],
        },
        source_size: None,
        fkeys,
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

/// users ← orders → items; order `i` is `orders[i]`, with one duplicate link
/// per user.
fn build_index(root: &Path) -> (PathBuf, Vec<(i64, i64)>) {
    build_index_order(root, false)
}

fn build_index_order(root: &Path, reverse: bool) -> (PathBuf, Vec<(i64, i64)>) {
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut orders = Vec::new();
    for u in 0..USERS {
        for d in [0, 2, 5, 7, 0] {
            orders.push((u, (u + d) % ITEMS));
        }
    }
    if reverse {
        orders.reverse();
    }
    let ids = |n: i64| Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())) as ArrayRef;
    write_parquet(&data.join("users.parquet"), vec![("user_id", ids(USERS))]);
    write_parquet(&data.join("items.parquet"), vec![("item_id", ids(ITEMS))]);
    write_parquet(
        &data.join("orders.parquet"),
        vec![
            (
                "order_id",
                Arc::new(Int64Array::from_iter_values((0..orders.len() as i64).map(
                    |id| {
                        if reverse {
                            orders.len() as i64 - 1 - id
                        } else {
                            id
                        }
                    },
                ))),
            ),
            (
                "unmatched_id",
                Arc::new(Int64Array::from_iter_values((0..orders.len()).map(|i| {
                    (if reverse { orders.len() - 1 - i } else { i }) as i64 + 1000
                }))),
            ),
            (
                "user_id",
                Arc::new(Int64Array::from_iter_values(orders.iter().map(|o| o.0))),
            ),
            (
                "item_id",
                Arc::new(Int64Array::from_iter_values(orders.iter().map(|o| o.1))),
            ),
        ],
    );
    let fk = |col: &str, dst: &str| FKeySpec {
        src_column: col.to_string(),
        dst_table: dst.to_string(),
        ..Default::default()
    };
    let config = GraphConfig {
        timestamp_unit: Default::default(),
        dataset_name: "static_lp".to_string(),
        tables: vec![
            table(
                "users",
                &["user_id"],
                "user_id",
                &data.join("users.parquet"),
                vec![],
            ),
            table(
                "items",
                &["item_id"],
                "item_id",
                &data.join("items.parquet"),
                vec![],
            ),
            table(
                "orders",
                &["order_id", "user_id", "item_id", "unmatched_id"],
                "order_id",
                &data.join("orders.parquet"),
                vec![fk("user_id", "users"), fk("item_id", "items")],
            ),
        ],
        rowpack: None,
        num_shards: Some(1),
        edge_format: None,
        coercion_mode: Default::default(),
    };
    let config_path = data.join("graph_config.json");
    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let index = root.join("index");
    diskgraph_ingest::ingest(
        &config_path,
        index.to_str().unwrap(),
        diskgraph_ingest::io::Dest::Local,
        diskgraph_ingest::config::IngestOptions {
            parallel: false,
            ..Default::default()
        },
    )
    .unwrap();
    if reverse {
        orders.reverse();
    }
    (index, orders)
}

fn split(kind: &str) -> SplitSpec<Seconds> {
    SplitSpec {
        ratio: Some(vec![0.5, 0.25, 0.25]),
        kind: Some(kind.to_string()),
        rnd_seed: Some(7),
        is_temporal: Some(false),
        offsets: None,
        offset_unit: None,
        unit: None,
        timeframe: None,
        anchor_start_offset: None,
        ranges: None,
        backtesting_timestamp: None,
        split_column: None,
        split_output_column: None,
        manual_train_values: None,
        manual_val_values: None,
        manual_test_values: None,
        month_offsets: None,
        flexible_boundaries: None,
        timeframe_step: None,
    }
}

fn manual_split() -> SplitSpec<Seconds> {
    SplitSpec {
        ratio: None,
        rnd_seed: None,
        split_column: Some("SPLIT".to_string()),
        manual_train_values: Some(vec!["0".to_string()]),
        manual_val_values: Some(vec!["1".to_string()]),
        manual_test_values: Some(vec!["2".to_string()]),
        ..split("ManualSplit")
    }
}

fn fact_spec() -> FactPseudoTimeSpec {
    FactPseudoTimeSpec {
        table: "orders".to_string(),
        entity_col: "user_id".to_string(),
        target_col: "item_id".to_string(),
    }
}

fn resolve(src: &str, dst: &str, kind: RefResolveKind, col: &str) -> RefResolveSpec {
    RefResolveSpec {
        src_col: src.to_string(),
        dst_table: dst.to_string(),
        kind,
        node_id_col: col.to_string(),
    }
}

fn generated_plan(split: SplitSpec<Seconds>) -> DiskGraphTTPlan<Seconds> {
    let per_link = split.kind.as_deref() == Some("RandomLinkSplit");
    let mut lp_resolution = vec![
        resolve("ENTITY1", "users", RefResolveKind::Scalar, "entity_node_id"),
        resolve("TARGET1", "items", RefResolveKind::List, "target_node_ids"),
    ];
    if per_link {
        lp_resolution.push(resolve(
            "exclude_target_keys",
            "items",
            RefResolveKind::List,
            "exclude_node_ids",
        ));
    }
    DiskGraphTTPlan::<Seconds> {
        entity: EntitySpec {
            table: "users".to_string(),
            column: "user_id".to_string(),
            output_time_col_name: None,
            default_time_value: None,
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: PredictionType::Link {
            include_frequency: true,
        },
        is_prediction: false,
        split: Some(split),
        stages: vec![Stage {
            ops: vec![Op::Aggregate {
                alias: "ld".to_string(),
                path: vec![EdgeRef {
                    src_table: "users".to_string(),
                    src_col: "user_id".to_string(),
                    dst_table: "orders".to_string(),
                    dst_col: "user_id".to_string(),
                }],
                func: AggFunc::ListDistinct,
                target_col: Some("item_id".to_string()),
                time_col: None,
                window: None,
                window_unit: None,
                row_filter: None,
                limit: None,
                target_entity_table: Some("items".to_string()),
            }],
            filter: None,
        }],
        target_stype: Some("ID".to_string()),
        weight_col: None,
        fkey_mapping: None,
        lp_resolution: Some(lp_resolution),
        lp_rhs_is_temporal: false,
        fact_pseudo_time: Some(fact_spec()),
    }
}

fn generated_tt(index: &Path, tt_id: &str, split: SplitSpec<Seconds>) {
    let plan = generated_plan(split);
    let plan_json = serde_json::to_string(&plan).unwrap();
    pyo3::Python::attach(|py| {
        let gen = super::execute_tt_gen(py, &plan_json, index.to_str().unwrap(), tt_id).unwrap();
        let staging: String = gen
            .bind(py)
            .get_item("staging_dir")
            .unwrap()
            .extract()
            .unwrap();
        super::execute_tt_ingest(py, &staging, index.to_str().unwrap(), tt_id, None).unwrap();
    });
}

/// Custom TT rows `(user, items, split)`; `split` is a ManualSplit stage.
fn custom_tt(
    root: &Path,
    index: &Path,
    tt_id: &str,
    rows: &[(i64, Vec<i64>, i64)],
    split: SplitSpec<Seconds>,
) {
    let mut targets = ListBuilder::new(Int64Builder::new());
    for (_, items, _) in rows {
        targets.values().append_slice(items);
        targets.append(true);
    }
    let path = root.join(format!("{tt_id}.parquet"));
    let mut columns: Vec<(&str, ArrayRef)> = vec![
        (
            "ENTITY1",
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
        ),
        (
            "TIMESTAMP",
            Arc::new(TimestampNanosecondArray::from(vec![0i64; rows.len()])),
        ),
        ("TARGET1", Arc::new(targets.finish())),
    ];
    if split.kind.as_deref() == Some("CustomLinkSplit") {
        let mut stages = ListBuilder::new(StringBuilder::new());
        for (_, items, _) in rows {
            for item in items {
                stages.values().append_value((item % 3).to_string());
            }
            stages.append(true);
        }
        columns.push(("SPLIT", Arc::new(stages.finish())));
    } else if split.split_column.is_some() {
        columns.push((
            "SPLIT",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.2.to_string()),
            )),
        ));
    }
    write_parquet(&path, columns);
    let staging = custom_artifacts(&path, index, tt_id, split).unwrap();
    pyo3::Python::attach(|py| {
        super::execute_tt_ingest(
            py,
            staging.to_str().unwrap(),
            index.to_str().unwrap(),
            tt_id,
            None,
        )
        .unwrap();
    });
}

fn custom_artifacts(
    path: &Path,
    index: &Path,
    tt_id: &str,
    split: SplitSpec<Seconds>,
) -> pyo3::PyResult<PathBuf> {
    custom_weighted_artifacts(path, index, tt_id, split, None)
}

fn custom_weighted_artifacts(
    path: &Path,
    index: &Path,
    tt_id: &str,
    split: SplitSpec<Seconds>,
    weight_col: Option<&str>,
) -> pyo3::PyResult<PathBuf> {
    let split_json = serde_json::to_string(&split).unwrap();
    let spec_json = serde_json::to_string(&fact_spec()).unwrap();
    pyo3::Python::attach(|py| {
        let out = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
            py,
            path.to_str().unwrap(),
            index.to_str().unwrap(),
            tt_id,
            "users",
            "user_id",
            "ID",
            "nanoseconds",
            Some(&split_json),
            weight_col,
            Some("items"),
            None,
            false,
            false,
            Some(&spec_json),
            "abort",
            false,
            None,
        )?;
        let staging: String = out
            .bind(py)
            .get_item("staging_dir")
            .unwrap()
            .extract()
            .unwrap();
        Ok(PathBuf::from(staging))
    })
}

/// `(user, item) -> stage` of every published label link (max over stages)
/// and the seed users of each stage.
type Labels = HashMap<(i64, i64), u8>;

fn label_stages(ext: &Path) -> (Labels, [BTreeSet<i64>; 3]) {
    let mut out = HashMap::new();
    let mut seeds: [BTreeSet<i64>; 3] = Default::default();
    for (stage, table) in STAGES.iter().enumerate() {
        for entry in std::fs::read_dir(ext.join(table)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "parquet") {
                continue;
            }
            let reader =
                ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap())
                    .unwrap()
                    .build()
                    .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                let users = batch.column_by_name("ENTITY1").unwrap();
                let users = users.as_any().downcast_ref::<Int64Array>().unwrap();
                let targets = batch.column_by_name("TARGET1").unwrap();
                let targets = targets.as_any().downcast_ref::<ListArray>().unwrap();
                for row in 0..batch.num_rows() {
                    seeds[stage].insert(users.value(row));
                    let items = targets.value(row);
                    let items = items.as_any().downcast_ref::<Int64Array>().unwrap();
                    for item in items.values() {
                        let slot = out.entry((users.value(row), *item)).or_insert(0);
                        *slot = (*slot).max(stage as u8);
                    }
                }
            }
        }
    }
    (out, seeds)
}

struct Sampled {
    reader: Arc<GraphIndexReader>,
    engine: DiskGraphEngine,
    order_of_node: HashMap<u64, usize>,
}

impl Sampled {
    fn open(index: &Path, tt_id: &str) -> Self {
        let reader = Arc::new(GraphIndexReader::open_with_tt(index, tt_id).unwrap());
        let engine = DiskGraphEngine::from_reader(reader.clone(), Some(3), Some(1)).unwrap();
        let fact = reader.table("orders").unwrap();
        let col = fact
            .node_data_decoder()
            .unwrap()
            .context()
            .columns
            .iter()
            .position(|c| c.name == "order_id")
            .unwrap();
        let order_of_node = (0..fact.node_count())
            .map(|offset| {
                let node = fact.node_id_from_offset(offset).unwrap();
                let Some(Value::Int(order)) =
                    diskgraph_train_table::ttgen_traversal::read_column_value(&fact, node, col)
                else {
                    panic!("order_id missing for node {node}");
                };
                (node, order as usize)
            })
            .collect();
        Self {
            reader,
            engine,
            order_of_node,
        }
    }

    fn plan(fanout: u32, mode: SamplingMode) -> SamplingPlan {
        let layer = LayerSpec {
            fanouts: HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout,
                mode,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        };
        SamplingPlan {
            layers: vec![layer.clone(), layer],
        }
    }

    /// Orders reached by sampling `stage` seeds (or every item when
    /// `stage` is `None`) at `seed_time`.
    fn orders(
        &self,
        stage: Option<&str>,
        seed_time: Option<u8>,
        plan: &SamplingPlan,
    ) -> Vec<usize> {
        let flags = SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: true,
        };
        let pseudo = seed_time.map(|t| PseudoTimeSeed::open(&self.reader, t as i64).unwrap());
        let filter = pseudo.as_ref().map(PseudoTimeSeed::filter);
        let batch = match stage {
            Some(stage) => {
                let table = self.reader.table(&format!("tt/{stage}")).unwrap();
                let rows: Vec<u64> = (0..table.node_count())
                    .map(|o| table.node_id_from_offset(o).unwrap())
                    .collect();
                self.engine.sample_runtime_batch(
                    &self.reader,
                    &format!("tt/{stage}"),
                    &rows,
                    DatasetId::nil(),
                    plan,
                    &flags,
                    None,
                    Some("users"),
                    Some("items"),
                    false,
                    None,
                    filter,
                    None,
                )
            }
            None => self.engine.sample_runtime_batch(
                &self.reader,
                "items",
                &(0..ITEMS as u64).collect::<Vec<_>>(),
                DatasetId::nil(),
                plan,
                &flags,
                None,
                None,
                None,
                true,
                Some(NodeRangeSeed {
                    rhs_node_type: "items",
                    seed_time: 0,
                }),
                filter,
                None,
            ),
        }
        .unwrap();
        batch.graphs["main"]
            .node_ids
            .get("orders")
            .map(|ids| ids.iter().map(|id| self.order_of_node[id]).collect())
            .unwrap_or_default()
    }
}

/// Every split's seeds (user side) and the item catalog (item side) reach
/// exactly the fact rows whose pseudo time is at most their seed time.
fn assert_k2_visibility(index: &Path, orders: &[(i64, i64)]) {
    let ext = index.join("train_tables").join("tt");
    let (labels, seeds) = label_stages(&ext);
    let found: BTreeSet<u8> = labels.values().copied().collect();
    assert_eq!(found, BTreeSet::from([0, 1, 2]), "every split holds labels");
    let pseudo = |o: usize| labels.get(&orders[o]).map_or(0, |s| s + 1);
    let all = 0..orders.len();
    let sampled = Sampled::open(index, "tt");
    let full = Sampled::plan(1000, SamplingMode::Uniform);

    for (i, (stage, t)) in STAGES.iter().zip(SEED_TIMES).enumerate() {
        let own: BTreeSet<usize> = all
            .clone()
            .filter(|&o| seeds[i].contains(&orders[o].0))
            .collect();
        let visible: BTreeSet<usize> = own.iter().copied().filter(|&o| pseudo(o) <= t).collect();
        let labels_of_stage = own.iter().filter(|&&o| pseudo(o) == i as u8 + 1).count();
        if i > 0 {
            assert!(own.len() - visible.len() >= labels_of_stage && labels_of_stage > 0);
        }
        let reached: BTreeSet<usize> = sampled
            .orders(Some(stage), Some(t), &full)
            .into_iter()
            .collect();
        assert_eq!(reached, visible, "{stage} at pseudo time {t}");
        let recent = Sampled::plan(1000, SamplingMode::TemporalMostRecent);
        let reached: BTreeSet<usize> = sampled
            .orders(Some(stage), Some(t), &recent)
            .into_iter()
            .collect();
        assert_eq!(reached, visible, "{stage} most-recent at pseudo time {t}");
        let one = sampled.orders(
            Some(stage),
            Some(t),
            &Sampled::plan(1, SamplingMode::Uniform),
        );
        assert_eq!(one.is_empty(), visible.is_empty(), "{stage} fanout 1");
        assert!(one.iter().all(|o| visible.contains(o)), "{stage} fanout 1");
        for fanout in [0, 1, 3, 1000] {
            let plan = Sampled::plan(fanout, SamplingMode::DeterministicUniform);
            let first = sampled.orders(Some(stage), Some(t), &plan);
            assert_eq!(first, sampled.orders(Some(stage), Some(t), &plan));
            assert!(first.iter().all(|o| visible.contains(o)));
            if fanout == 0 {
                assert!(first.is_empty());
            }
        }
        let unfiltered: BTreeSet<usize> = sampled
            .orders(Some(stage), None, &full)
            .into_iter()
            .collect();
        assert_eq!(
            unfiltered, own,
            "{stage} without pseudo time sees every row"
        );
    }
    for (stage, time) in STAGES.iter().zip(SEED_TIMES) {
        let mut multihop = full.clone();
        multihop.layers.resize(4, multihop.layers[0].clone());
        let reached = sampled.orders(Some(stage), Some(time), &multihop);
        let direct = sampled.orders(Some(stage), Some(time), &full);
        assert!(direct.iter().all(|order| reached.contains(order)));
        assert!(reached.iter().all(|&order| pseudo(order) <= time));
        let unfiltered = sampled.orders(Some(stage), None, &multihop);
        if *stage != "train_seeds" {
            assert!(unfiltered.iter().any(|&order| pseudo(order) > time));
        }
    }
    for t in [1u8, 2] {
        let visible: BTreeSet<usize> = all.clone().filter(|&o| pseudo(o) <= t).collect();
        let reached: BTreeSet<usize> = sampled.orders(None, Some(t), &full).into_iter().collect();
        assert_eq!(reached, visible, "items at pseudo time {t}");
    }
    let reached: BTreeSet<usize> = sampled.orders(None, None, &full).into_iter().collect();
    assert_eq!(reached, all.collect(), "predict sees every fact row");
}

#[test]
fn generated_random_link_split_hides_later_split_links() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, orders) = build_index(dir.path());
    generated_tt(&index, "tt", split("RandomLinkSplit"));
    assert_k2_visibility(&index, &orders);
}

#[test]
fn generated_random_split_hides_later_split_links() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, orders) = build_index(dir.path());
    generated_tt(&index, "tt", split("RandomSplit"));
    assert_k2_visibility(&index, &orders);
}

#[test]
fn custom_per_link_splits_hide_later_split_links() {
    pyo3::Python::initialize();
    for split in [
        split("RandomLinkSplit"),
        SplitSpec {
            kind: Some("CustomLinkSplit".to_string()),
            ..manual_split()
        },
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (index, orders) = build_index(dir.path());
        let rows: Vec<(i64, Vec<i64>, i64)> = (0..USERS)
            .map(|u| {
                let mut items: Vec<_> = orders.iter().filter(|o| o.0 == u).map(|o| o.1).collect();
                items.insert(1, ITEMS + 100);
                (u, items, 0)
            })
            .collect();
        custom_tt(dir.path(), &index, "tt", &rows, split);
        assert_k2_visibility(&index, &orders);
    }
}

#[test]
fn custom_string_labels_do_not_match_integer_facts() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, _) = build_index(dir.path());
    let path = dir.path().join("custom.parquet");
    let mut targets = ListBuilder::new(StringBuilder::new());
    targets.values().append_value("0");
    targets.append(true);
    write_parquet(
        &path,
        vec![
            ("ENTITY1", Arc::new(Int64Array::from(vec![0]))),
            (
                "TIMESTAMP",
                Arc::new(TimestampNanosecondArray::from(vec![0])),
            ),
            ("TARGET1", Arc::new(targets.finish())),
            ("SPLIT", Arc::new(StringArray::from(vec!["0"]))),
        ],
    );
    let error = custom_artifacts(&path, &index, "mismatch", manual_split()).unwrap_err();
    assert!(error.to_string().contains("kept no rows"), "{error}");
}

#[test]
fn custom_target_validation_preserves_filtered_link_alignment() {
    pyo3::Python::initialize();
    for large in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (index, orders) = build_index(dir.path());
        let path = dir.path().join("nullable.parquet");
        let targets = |nullable: bool| -> ArrayRef {
            if large {
                let mut builder = LargeListBuilder::new(Int64Builder::new());
                builder
                    .values()
                    .append_option(if nullable { None } else { Some(ITEMS + 100) });
                builder.values().append_slice(&[0, 2]);
                builder.append(true);
                Arc::new(builder.finish())
            } else {
                let mut builder = ListBuilder::new(Int64Builder::new());
                builder
                    .values()
                    .append_option(if nullable { None } else { Some(ITEMS + 100) });
                builder.values().append_slice(&[0, 2]);
                builder.append(true);
                Arc::new(builder.finish())
            }
        };
        let ints = |values: &[i64]| -> ArrayRef {
            let mut builder = ListBuilder::new(Int64Builder::new());
            builder.values().append_slice(values);
            builder.append(true);
            Arc::new(builder.finish())
        };
        let frequencies = if large {
            let mut builder = LargeListBuilder::new(Int64Builder::new());
            builder.values().append_slice(&[99, 11, 22]);
            builder.append(true);
            Arc::new(builder.finish()) as ArrayRef
        } else {
            ints(&[99, 11, 22])
        };
        let mut weights = ListBuilder::new(Float64Builder::new());
        weights.values().append_slice(&[999.0, 1.5, 2.5]);
        weights.append(true);
        let mut columns = vec![
            ("ENTITY1", Arc::new(Int64Array::from(vec![0])) as ArrayRef),
            (
                "TIMESTAMP",
                Arc::new(TimestampNanosecondArray::from(vec![0])) as ArrayRef,
            ),
            ("TARGET1", targets(true)),
            ("SPLIT", ints(&[2, 0, 1])),
            ("__kumo_internal_frequency_column__", frequencies),
            ("weight", Arc::new(weights.finish()) as ArrayRef),
        ];
        write_parquet(&path, columns.clone());
        let error = custom_weighted_artifacts(
            &path,
            &index,
            "tt",
            SplitSpec {
                kind: Some("CustomLinkSplit".to_string()),
                ..manual_split()
            },
            Some("weight"),
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("target list contains a null element"),
            "{message}"
        );
        assert!(message.contains("TARGET1"), "{message}");
        assert!(message.contains("row 0"), "{message}");

        // Null items reject the row; unknown graph targets are filtered while
        // retaining the corresponding split, frequency and weight positions.
        columns[2].1 = targets(false);
        write_parquet(&path, columns);
        let staging = custom_weighted_artifacts(
            &path,
            &index,
            "tt",
            SplitSpec {
                kind: Some("CustomLinkSplit".to_string()),
                ..manual_split()
            },
            Some("weight"),
        )
        .unwrap();
        let (labels, _) = label_stages(&staging);
        assert_eq!(labels, HashMap::from([((0, 0), 0), ((0, 2), 1)]));
        for (stage, expected_frequency, expected_weight) in
            [("train_seeds", 11, 1.5), ("val_seeds", 22, 2.5)]
        {
            for part in std::fs::read_dir(staging.join(stage)).unwrap() {
                let file = std::fs::File::open(part.unwrap().path()).unwrap();
                for batch in ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .build()
                    .unwrap()
                {
                    let batch = batch.unwrap();
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let frequencies = batch
                        .column_by_name("__kumo_internal_frequency_column__")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .value(0);
                    assert_eq!(
                        frequencies
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .values()
                            .as_ref(),
                        &[expected_frequency]
                    );
                    let weights = batch
                        .column_by_name("WEIGHT")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .value(0);
                    assert_eq!(
                        weights
                            .as_any()
                            .downcast_ref::<arrow_array::Float64Array>()
                            .unwrap()
                            .values()
                            .as_ref(),
                        &[expected_weight]
                    );
                }
            }
        }
        pyo3::Python::attach(|py| {
            super::execute_tt_ingest(
                py,
                staging.to_str().unwrap(),
                index.to_str().unwrap(),
                "tt",
                None,
            )
            .unwrap();
        });
        let sampled = Sampled::open(&index, "tt");
        let bytes = std::fs::read(index.join("train_tables/tt/fact_pseudo_time.bin")).unwrap();
        let times = diskgraph_train_table::pseudo_time::fact_pseudo_time_payload(&bytes).unwrap();
        for (node, order) in sampled.order_of_node {
            assert_eq!(
                times[node as usize],
                labels.get(&orders[order]).map_or(0, |stage| stage + 1)
            );
        }
    }
}

#[test]
fn custom_link_split_rejects_null_stage_values() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, _) = build_index(dir.path());
    let path = dir.path().join("null_split.parquet");
    let mut targets = ListBuilder::new(Int64Builder::new());
    targets.values().append_value(0);
    targets.append(true);
    let mut stages = ListBuilder::new(Int64Builder::new());
    stages.values().append_null();
    stages.append(true);
    write_parquet(
        &path,
        vec![
            ("ENTITY1", Arc::new(Int64Array::from(vec![0]))),
            (
                "TIMESTAMP",
                Arc::new(TimestampNanosecondArray::from(vec![0])),
            ),
            ("TARGET1", Arc::new(targets.finish())),
            ("SPLIT", Arc::new(stages.finish())),
        ],
    );
    let error = custom_artifacts(
        &path,
        &index,
        "null_split",
        SplitSpec {
            kind: Some("CustomLinkSplit".to_string()),
            ..manual_split()
        },
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("split value is null in column \"SPLIT\""),
        "{error}"
    );
}

#[test]
fn custom_manual_split_takes_the_latest_split_of_a_link() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, orders) = build_index(dir.path());
    let items = |u: i64| -> Vec<i64> { orders.iter().filter(|o| o.0 == u).map(|o| o.1).collect() };
    let mut rows: Vec<(i64, Vec<i64>, i64)> = (0..USERS)
        .map(|u| (u, items(u)[..2].to_vec(), u % 3))
        .collect();
    rows.push((0, items(0)[2..].to_vec(), 2));
    rows.push((1, vec![items(1)[0]], 2));
    custom_tt(dir.path(), &index, "tt", &rows, manual_split());
    assert_k2_visibility(&index, &orders);
}

#[test]
fn sidecar_marks_every_duplicate_row_and_rejects_unknown_versions() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, orders) = build_index(dir.path());
    generated_tt(&index, "tt", split("RandomLinkSplit"));
    let ext = index.join("train_tables/tt");
    let (labels, _) = label_stages(&ext);
    let bytes = std::fs::read(ext.join("fact_pseudo_time.bin")).unwrap();
    let times = diskgraph_train_table::pseudo_time::fact_pseudo_time_payload(&bytes)
        .unwrap()
        .to_vec();
    let sampled = Sampled::open(&index, "tt");
    let mut by_order = vec![0u8; orders.len()];
    for (node, order) in &sampled.order_of_node {
        by_order[*order] = times[*node as usize];
    }
    for (o, link) in orders.iter().enumerate() {
        assert_eq!(
            by_order[o],
            labels.get(link).map_or(0, |s| s + 1),
            "order {o} {link:?}"
        );
    }

    for seed_time in [-1, 4, 300] {
        let error = PseudoTimeSeed::open(&sampled.reader, seed_time)
            .err()
            .unwrap();
        assert!(error.to_string().contains("must be between 0 and 3"));
    }

    let mut newer = bytes.clone();
    newer[8] = 2;
    std::fs::write(ext.join("fact_pseudo_time.bin"), &newer).unwrap();
    let err = PseudoTimeSeed::open(&sampled.reader, 1).err().unwrap();
    assert!(
        err.to_string()
            .contains("unsupported fact_pseudo_time version=2"),
        "{err}"
    );

    std::fs::write(ext.join("fact_pseudo_time.bin"), &bytes).unwrap();
    let manifest_path = ext.join("manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    std::fs::write(
        &manifest_path,
        manifest.replace("tt_fact_pseudo_time_v1", "tt_fact_pseudo_time_v2"),
    )
    .unwrap();
    let err = PseudoTimeSeed::open(&sampled.reader, 1).err().unwrap();
    assert!(
        err.to_string()
            .contains("unsupported fact_pseudo_time format"),
        "{err}"
    );
}

#[test]
fn generated_pseudo_time_rejects_unmatched_labels_and_missing_split() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, _) = build_index(dir.path());
    for missing_split in [false, true] {
        let mut plan = generated_plan(split("RandomLinkSplit"));
        if missing_split {
            plan.split = None;
        } else {
            plan.fact_pseudo_time.as_mut().unwrap().target_col = "unmatched_id".to_string();
        }
        let json = serde_json::to_string(&plan).unwrap();
        pyo3::Python::attach(|py| {
            let error = super::execute_tt_gen(py, &json, index.to_str().unwrap(), "invalid")
                .err()
                .unwrap();
            let expected = if missing_split {
                "requires stage parts"
            } else {
                "no generated label link matched"
            };
            assert!(error.to_string().contains(expected), "{error}");
        });
    }
}

#[path = "pseudo_time_cutoff_tests.rs"]
mod cutoff_tests;

#[test]
fn empty_last_shard_does_not_hide_truncated_fact_coverage() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, _) = build_index(dir.path());
    generated_tt(&index, "tt", split("RandomLinkSplit"));
    let mut sampled = Sampled::open(&index, "tt");
    let type_id = sampled.engine.node_type_to_id["orders"];
    let shards = &mut sampled.engine.readers[type_id];
    shards.push(diskgraph_engine::engine::ShardHandle {
        reader: shards[0].reader.clone(),
        min_id: 1,
        max_id: 0,
    });
    let pseudo = PseudoTimeSeed::open(&sampled.reader, 1).unwrap();
    let mut filter = pseudo.filter();
    filter.times = &filter.times[..1];
    let plan = Sampled::plan(1000, SamplingMode::Uniform);
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: false,
        swap_edge_direction: true,
    };
    let error = sampled
        .engine
        .sample_runtime_batch(
            &sampled.reader,
            "tt/train_seeds",
            &[],
            DatasetId::nil(),
            &plan,
            &flags,
            None,
            Some("users"),
            Some("items"),
            false,
            None,
            Some(filter),
            None,
        )
        .unwrap_err();
    assert!(error.to_string().contains("pseudo times cover 1"));
    sampled.engine.readers[type_id].pop();
    let visible = sampled.orders(None, Some(1), &plan);
    assert!(!visible.is_empty());
}

#[path = "pseudo_time_cross_graph_tests.rs"]
mod cross_graph_tests;
