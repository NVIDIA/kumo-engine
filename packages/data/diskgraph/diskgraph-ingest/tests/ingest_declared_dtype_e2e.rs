// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A source column whose Parquet type differs from its declared dtype is
//! converted value for value, and a value the declared dtype cannot hold
//! becomes null and is counted as invalid.

use anyhow::Result;
use arrow_array::types::{Float32Type, Int32Type, Int64Type, Int8Type, UInt32Type, UInt64Type};
use arrow_array::{
    ArrayRef, BinaryArray, Decimal128Array, DictionaryArray, Float16Array, Float32Array,
    Float64Array, Int64Array, LargeStringArray, ListArray, RecordBatch, UInt32Array, UInt64Array,
};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
};
use diskgraph_ingest::stats::IngestStats;
use diskgraph_ingest::{ingest_with_stats_from_source, Dest, IngestOptions};
use half::f16;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

fn ingest(root: &Path, tables: Vec<(&str, RecordBatch)>, kumo: Value) -> Result<IngestStats> {
    let mut paths = HashMap::new();
    for (name, batch) in tables {
        let dir = root.join("data").join(name);
        fs::create_dir_all(&dir)?;
        let file = fs::File::create(dir.join("part-0.parquet"))?;
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        paths.insert(name.to_string(), dir.to_string_lossy().to_string());
    }
    let kumo: KumoGraphConfig = serde_json::from_value(kumo)?;
    let options = ConvertOptions {
        data_persist_dir: root.join("data").to_string_lossy().to_string(),
        dataset_name: "declared".to_string(),
        table_data_versions: None,
        source_paths_base: None,
        source_paths: Some(paths),
        source_specs: None,
    };
    let config = convert_kumo_to_diskgraph(&kumo, &options)?;
    let options = IngestOptions {
        col_stats: true,
        ..IngestOptions::default()
    };
    let output = root.join("output");
    let (_, stats) = ingest_with_stats_from_source(
        config,
        None,
        output.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )?;
    Ok(stats)
}

fn ingest_column(column: ArrayRef, dtype: &str, stype: &str) -> Result<(Vec<Value>, Value)> {
    let dir = tempdir()?;
    let ids = Arc::new(Int64Array::from_iter_values(0..column.len() as i64)) as ArrayRef;
    let batch = RecordBatch::try_from_iter([("id", ids), ("c", column)])?;
    let mut col = json!({"name": "c", "dtype": dtype, "stype": stype});
    if stype == "multicategorical" && dtype == "string" {
        col["multicategorical_delimiter"] = json!("|");
    }
    let kumo = json!({"tables": [{
        "table_name": "t", "data_source": "test", "data_version": "v1",
        "pkey": {"name": "id", "dtype": "int64", "stype": "ID"},
        "cols": [col],
    }]});
    ingest(dir.path(), vec![("t", batch)], kumo)?;
    let output = dir.path().join("output");
    let reader = GraphIndexReader::open(&output.join("index_metadata.json"), &output)?;
    let table = reader.table("t").unwrap();
    let mut rows = Vec::new();
    for node in 0..table.node_count() {
        let row = table
            .decode_node_data_json(&table.edges_for_node(node)?)?
            .unwrap();
        rows.push((row["id"].as_i64().unwrap(), row["c"].clone()));
    }
    rows.sort_by_key(|row| row.0);
    let stats: Value = serde_json::from_slice(&fs::read(output.join("col_stats.json"))?)?;
    Ok((
        rows.into_iter().map(|row| row.1).collect(),
        stats["t"]["c"]["parse_fail_count"].clone(),
    ))
}

fn decimal(values: &[Option<i128>]) -> ArrayRef {
    Arc::new(
        Decimal128Array::from(values.to_vec())
            .with_precision_and_scale(18, 2)
            .unwrap(),
    )
}

#[test]
fn declared_dtype_converts_exactly_or_counts_invalid() -> Result<()> {
    let big = 1i64 << 31;
    let cases: Vec<(&str, ArrayRef, &str, &str, Value, u64)> = vec![
        (
            "dictionary",
            Arc::new(DictionaryArray::<Int32Type>::from_iter([
                Some("a"),
                Some("b"),
                None,
                Some("a"),
            ])),
            "string",
            "categorical",
            json!(["a", "b", null, "a"]),
            0,
        ),
        (
            "dictionary int8 large_string",
            Arc::new(DictionaryArray::<Int8Type>::try_new(
                vec![Some(0i8), Some(1), None].into(),
                Arc::new(LargeStringArray::from(vec!["a b", "c"])),
            )?),
            "string",
            "text",
            json!(["a b", "c", null]),
            0,
        ),
        (
            "int64 into int32",
            Arc::new(Int64Array::from(vec![
                Some(-big),
                Some(big),
                None,
                Some(1 << 40),
            ])),
            "int32",
            "numerical",
            json!([-big, null, null, null]),
            2,
        ),
        (
            "uint32 into int32",
            Arc::new(UInt32Array::from(vec![
                Some(7),
                Some(u32::MAX),
                Some(1 << 31),
            ])),
            "int32",
            "categorical",
            json!([7, null, null]),
            2,
        ),
        (
            "uint64 into int32",
            Arc::new(UInt64Array::from(vec![
                Some(7),
                Some(1 << 63),
                Some((1 << 32) + 3),
            ])),
            "int32",
            "categorical",
            json!([7, null, null]),
            2,
        ),
        (
            "float64 into int64",
            Arc::new(Float64Array::from(vec![
                Some(-3.0),
                Some(9007199254740992.0),
                None,
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(1e30),
                Some(1.5),
            ])),
            "int64",
            "numerical",
            json!([-3, 9007199254740992i64, null, null, null, null, null]),
            4,
        ),
        (
            "float32 into int32",
            Arc::new(Float32Array::from(vec![Some(7.0), Some(3e9), Some(-2.5)])),
            "int32",
            "numerical",
            json!([7, null, null]),
            2,
        ),
        (
            "fractional decimal into int64",
            decimal(&[Some(1291), Some(-700), None]),
            "int64",
            "numerical",
            json!([null, -7, null]),
            1,
        ),
        (
            "fractional decimal into int32",
            decimal(&[Some(700), Some(300_000_000_000), Some(1)]),
            "int32",
            "numerical",
            json!([7, null, null]),
            2,
        ),
        (
            "float16",
            Arc::new(Float16Array::from(vec![
                Some(f16::from_f32(1.5)),
                None,
                Some(f16::from_f32(65504.0)),
            ])),
            "float32",
            "numerical",
            json!([1.5, null, 65504.0]),
            0,
        ),
        (
            "unsigned list elements",
            Arc::new(ListArray::from_iter_primitive::<UInt32Type, _, _>([
                Some(vec![Some(1), Some(u32::MAX)]),
                Some(vec![]),
                None,
            ])),
            "intlist",
            "multicategorical",
            json!([[1, u32::MAX], [], null]),
            0,
        ),
        (
            "uint64 list elements",
            Arc::new(ListArray::from_iter_primitive::<UInt64Type, _, _>([
                Some(vec![Some(1)]),
                Some(vec![Some(2), Some(1 << 63)]),
            ])),
            "intlist",
            "multicategorical",
            json!([[1], null]),
            1,
        ),
        (
            "float list elements into intlist",
            Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>([
                Some(vec![Some(1.0), Some(2.0)]),
                Some(vec![Some(1.5)]),
            ])),
            "intlist",
            "multicategorical",
            json!([[1, 2], null]),
            1,
        ),
        (
            "int list elements into floatlist",
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
                Some(vec![Some(1), Some(-2)]),
                Some(vec![]),
                Some(vec![Some(1), None, Some(3)]),
            ])),
            "floatlist",
            "sequence",
            json!([[1.0, -2.0], [], [1.0, 0.0, 3.0]]),
            0,
        ),
        (
            "int list with a null element into latlong",
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
                Some(vec![Some(1), Some(2)]),
                Some(vec![Some(1), None]),
            ])),
            "floatlist",
            "latlong",
            json!([[1.0, 2.0], null]),
            1,
        ),
        (
            "float into string",
            Arc::new(Float64Array::from(vec![1e70, 1e-70, 1.5])),
            "string",
            "categorical",
            json!(["1e70", "1e-70", "1.5"]),
            0,
        ),
        (
            "int into multicategorical string",
            Arc::new(Int64Array::from(vec![Some(1), None])),
            "string",
            "multicategorical",
            json!([["1"], null]),
            0,
        ),
        (
            "binary into multicategorical string",
            Arc::new(BinaryArray::from(vec![b"a|b".as_ref(), b"c"])),
            "string",
            "multicategorical",
            json!([["a", "b"], ["c"]]),
            0,
        ),
    ];
    for (name, column, dtype, stype, expected, invalid) in cases {
        let (values, parse_failures) = ingest_column(column, dtype, stype)?;
        assert_eq!(Value::from(values), expected, "{name}");
        assert_eq!(parse_failures, json!(invalid), "{name}");
    }
    Ok(())
}

#[test]
fn foreign_keys_convert_exactly_or_drop_the_edge() -> Result<()> {
    let dir = tempdir()?;
    let users =
        RecordBatch::try_from_iter([("id", Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef)])?;
    let events = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int64Array::from(vec![0, 1, 2, 3])) as ArrayRef,
        ),
        (
            "by_name",
            Arc::new(DictionaryArray::<Int32Type>::from_iter([
                Some("1"),
                Some("2"),
                None,
                Some("2"),
            ])) as ArrayRef,
        ),
        (
            "by_float",
            Arc::new(Float64Array::from(vec![1.0, 1.5, 2.0, f64::NAN])) as ArrayRef,
        ),
    ])?;
    let fkey =
        |name: &str| json!({"name": name, "dtype": "int64", "dst_table": "users", "dst_key": "id"});
    let kumo = json!({"tables": [
        {
            "table_name": "users", "data_source": "test", "data_version": "v1",
            "pkey": {"name": "id", "dtype": "int64", "stype": "ID"}, "cols": [],
        },
        {
            "table_name": "events", "data_source": "test", "data_version": "v1",
            "pkey": {"name": "id", "dtype": "int64", "stype": "ID"},
            "fkeys": [fkey("by_name"), fkey("by_float")], "cols": [],
        },
    ]});
    let stats = ingest(dir.path(), vec![("users", users), ("events", events)], kumo)?;
    assert_eq!(stats.edge_counts["events_by_name_to_users_id"], 3);
    assert_eq!(stats.edge_counts["events_by_float_to_users_id"], 2);
    Ok(())
}
