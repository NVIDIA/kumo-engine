// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `rhs_catalog`: canonical rows of a table, optionally filtered.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use diskgraph_core::tt_builder::{CompareOp, RowFilter, RowFilterLhs, RowFilterRhs, Value};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::pkmap::Key;
use diskgraph_ingest::coercion::CoercionMode;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::*;
use diskgraph_ingest::io::Dest;
use diskgraph_train_table::rhs_catalog::rhs_catalog;
use diskgraph_train_table::ttgen_traversal::read_column_value;
use parquet::arrow::ArrowWriter;
use rayon::ThreadPool;

fn column(name: &str, dtype: IngestedType, semantic_type: Option<SemanticType>) -> ColumnSpec {
    ColumnSpec {
        source_name: None,
        name: name.to_string(),
        source_dtype: None,
        ingested_dtype: dtype,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type,
        multicategorical_delimiter: None,
    }
}

fn table(data_dir: &Path, name: &str, batch: RecordBatch, columns: Vec<ColumnSpec>) -> TableSpec {
    let path = data_dir.join(format!("{name}.parquet"));
    let mut writer =
        ArrowWriter::try_new(fs::File::create(&path).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    TableSpec {
        name: name.to_string(),
        columns,
        primary_key: Some("id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec![path.to_str().unwrap().to_string()],
        },
        source_size: None,
        fkeys: vec![],
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn ids(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    let mut builder = Int64Builder::new();
    builder.extend(values.into_iter().map(Some));
    Arc::new(builder.finish())
}

/// `items` (node = row in one shard): ids 1, 2, 2, 3, 4, 5, so node 2
/// repeats node 1's key, and node 3 holds nulls. `many`: 200,000 rows over
/// 150,000 keys. Ingest refuses a table without rows.
fn build_index(root: &Path, num_shards: usize) -> GraphIndexReader {
    let data_dir = root.join("data");
    let index_dir = root.join("index");
    fs::create_dir_all(&data_dir).unwrap();

    let mut active = BooleanBuilder::new();
    active.extend([
        Some(true),
        Some(false),
        Some(true),
        None,
        Some(true),
        Some(false),
    ]);
    let mut rating = Float64Builder::new();
    rating.extend([Some(1.0), Some(2.0), Some(5.0), None, Some(4.0), Some(0.5)]);
    let mut tags = ListBuilder::new(StringBuilder::new());
    for row in [
        Some(vec!["a", "b"]),
        Some(vec!["b"]),
        Some(vec!["a"]),
        None,
        Some(vec![]),
        Some(vec!["c"]),
    ] {
        match row {
            Some(tokens) => {
                for token in tokens {
                    tags.values().append_value(token);
                }
                tags.append(true);
            }
            None => tags.append(false),
        }
    }
    let items = batch(vec![
        ("id", ids([1, 2, 2, 3, 4, 5])),
        ("active", Arc::new(active.finish())),
        ("rating", Arc::new(rating.finish())),
        ("tags", Arc::new(tags.finish())),
    ]);
    let many = batch(vec![
        ("id", ids((0..200_000).map(|row| row % 150_000))),
        ("value", ids(0..200_000)),
    ]);
    let config = GraphConfig {
        timestamp_unit: Default::default(),
        dataset_name: "rhs_catalog".to_string(),
        tables: vec![
            table(
                &data_dir,
                "items",
                items,
                vec![
                    column("id", IngestedType::Int64, None),
                    column("active", IngestedType::Bool, None),
                    column("rating", IngestedType::Float64, None),
                    column(
                        "tags",
                        IngestedType::StringArray,
                        Some(SemanticType::Multicategorical),
                    ),
                ],
            ),
            table(
                &data_dir,
                "many",
                many,
                vec![
                    column("id", IngestedType::Int64, None),
                    column("value", IngestedType::Int64, None),
                ],
            ),
        ],
        rowpack: None,
        num_shards: Some(num_shards),
        edge_format: None,
        coercion_mode: CoercionMode::Lossless,
    };
    let config_path = data_dir.join("graph_config.json");
    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let options = IngestOptions {
        parallel: false,
        ..Default::default()
    };
    diskgraph_ingest::ingest(
        &config_path,
        index_dir.to_str().unwrap(),
        Dest::Local,
        options,
    )
    .unwrap();
    GraphIndexReader::open(&index_dir.join("index_metadata.json"), &index_dir).unwrap()
}

fn pool(threads: usize) -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn compare(column: &str, op: CompareOp, value: Value) -> RowFilter {
    RowFilter::Compare {
        lhs: RowFilterLhs::Column(column.to_string()),
        op,
        rhs: RowFilterRhs::Constant(value),
    }
}

fn catalog(
    reader: &GraphIndexReader,
    node_type: &str,
    filter: Option<&RowFilter>,
    threads: usize,
) -> Vec<u64> {
    let catalog = rhs_catalog(reader, node_type, filter, &pool(threads)).unwrap();
    let node_ids: Vec<u64> = catalog.node_ids().collect();
    assert_eq!(catalog.len(), node_ids.len());
    node_ids
}

#[test]
fn rhs_catalog_keeps_canonical_rows_passing_the_filter() {
    let root = tempfile::tempdir().unwrap();
    let reader = build_index(root.path(), 1);
    let active = || compare("active", CompareOp::Eq, Value::Bool(true));
    let rated = || compare("rating", CompareOp::Gt, Value::Float(1.5));
    let tagged = |op| compare("tags", op, Value::Str("a".to_string()));

    let cases = [
        (None, vec![0, 1, 3, 4, 5]),
        (Some(active()), vec![0, 4]),
        (Some(RowFilter::And(vec![active(), rated()])), vec![4]),
        (Some(RowFilter::Or(vec![active(), rated()])), vec![0, 1, 4]),
        // A comparison with a null value is unknown, so NOT fails it too.
        (Some(RowFilter::Not(Box::new(active()))), vec![1, 5]),
        (
            Some(RowFilter::Not(Box::new(RowFilter::Or(vec![
                active(),
                rated(),
            ])))),
            vec![5],
        ),
        // An unknown operand of OR still passes when another is true.
        (
            Some(RowFilter::Or(vec![
                active(),
                compare("rating", CompareOp::IsNull, Value::Bool(true)),
            ])),
            vec![0, 3, 4],
        ),
        (
            Some(compare("active", CompareOp::IsNull, Value::Bool(true))),
            vec![3],
        ),
        (Some(tagged(CompareOp::Contains)), vec![0]),
        // NOT CONTAINS passes an empty list and fails a null one.
        (Some(tagged(CompareOp::NotContains)), vec![1, 4, 5]),
        (
            Some(RowFilter::Not(Box::new(tagged(CompareOp::Contains)))),
            vec![1, 4, 5],
        ),
    ];
    for (filter, expected) in cases {
        assert_eq!(
            catalog(&reader, "items", filter.as_ref(), 2),
            expected,
            "{filter:?}"
        );
    }
    let none = compare("id", CompareOp::Gt, Value::Int(5));
    assert!(catalog(&reader, "items", Some(&none), 2).is_empty());
    let empty_row = reader
        .table("items")
        .unwrap()
        .is_canonical_row(3, &[], &mut Default::default())
        .unwrap_err();
    assert!(format!("{empty_row:#}").starts_with("items node 3: "));
}

#[test]
fn rhs_catalog_does_not_depend_on_the_thread_count() {
    let root = tempfile::tempdir().unwrap();
    let reader = build_index(root.path(), 3);
    let filter = compare("value", CompareOp::Gte, Value::Int(100_000));
    let single = catalog(&reader, "many", Some(&filter), 1);
    assert_eq!(catalog(&reader, "many", Some(&filter), 5), single);
    assert!(single.windows(2).all(|pair| pair[0] < pair[1]));
    let table = reader.table("many").unwrap();
    let key_at = |node| match read_column_value(&table, node, 0) {
        Some(Value::Int(id)) => Key::Int(id as u64),
        other => panic!("{other:?}"),
    };
    let canonical: Vec<u64> = (0..table.node_count())
        .filter(|&node| table.node_id_from_pk(&key_at(node)).unwrap() == Some(node))
        .collect();
    assert_eq!(canonical.len(), 150_000);
    assert_eq!(catalog(&reader, "many", None, 3), canonical);
    assert_eq!(single.len(), 50_000);
    assert!(single
        .iter()
        .all(|node| canonical.binary_search(node).is_ok()));
}
