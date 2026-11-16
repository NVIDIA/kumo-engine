// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Infer stage-extension [`ColumnSpec`]s from a parquet file's Arrow schema.
//!
//! Shared by the TT-NG stage-ingest callers (the PQL path in
//! `diskgraph-pybind/src/tt_executor.rs` and the source-agnostic path in
//! `diskgraph-train-table/src/tt_stage_ingest.rs`), both of which feed the
//! result into [`crate::ingest_stage_extension_with_resolution`]. Sits next to that primitive
//! and owns the dtype/timestamp-unit/semantic-type mapping so the two callers
//! stay byte-for-byte identical.

use std::path::Path;

use anyhow::{Context, Result};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::graph_config::{ColumnSpec, IngestedType, SemanticType, TimestampUnit};
use diskgraph_core::tt_builder::{PRIMARY_ENTITY_COL, TIMESTAMP_COL, PRIMARY_TARGET_COL, WEIGHT_COL};

/// Infer `ColumnSpec`s for a TT stage parquet from its Arrow schema.
pub fn infer_stage_columns(parquet_path: &Path) -> Result<Vec<ColumnSpec>> {
    let file = std::fs::File::open(parquet_path)
        .with_context(|| format!("opening stage parquet {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet schema {}", parquet_path.display()))?;
    builder
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let ingested_dtype = arrow_dtype_to_ingested(field.data_type()).ok_or_else(|| {
                anyhow::anyhow!(
                    "unsupported TT stage column {} dtype {:?}",
                    field.name(),
                    field.data_type()
                )
            })?;
            Ok(ColumnSpec {
                name: field.name().clone(),
                source_dtype: Some(format!("{:?}", field.data_type())),
                ingested_dtype,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: match field.data_type() {
                    DataType::Timestamp(unit, _) => Some(match unit {
                        arrow_schema::TimeUnit::Second => TimestampUnit::Seconds,
                        arrow_schema::TimeUnit::Millisecond => TimestampUnit::Millis,
                        arrow_schema::TimeUnit::Microsecond => TimestampUnit::Micros,
                        arrow_schema::TimeUnit::Nanosecond => TimestampUnit::Nanos,
                    }),
                    _ => None,
                },
                precision: None,
                scale: None,
                semantic_type: infer_semantic_type(field.name(), field.data_type()),
                multicategorical_delimiter: None,
            })
        })
        .collect()
}

fn arrow_dtype_to_ingested(dtype: &DataType) -> Option<IngestedType> {
    match dtype {
        DataType::Int8 => Some(IngestedType::Int8),
        DataType::Int16 => Some(IngestedType::Int16),
        DataType::Int32 => Some(IngestedType::Int32),
        DataType::Int64 => Some(IngestedType::Int64),
        DataType::UInt8 => Some(IngestedType::UInt8),
        DataType::UInt16 => Some(IngestedType::UInt16),
        DataType::UInt32 => Some(IngestedType::UInt32),
        DataType::UInt64 => Some(IngestedType::UInt64),
        DataType::Float32 => Some(IngestedType::Float32),
        DataType::Float64 => Some(IngestedType::Float64),
        DataType::Boolean => Some(IngestedType::Bool),
        DataType::Utf8 | DataType::LargeUtf8 => Some(IngestedType::String),
        DataType::Binary | DataType::LargeBinary => Some(IngestedType::Bytes),
        DataType::Date32 => Some(IngestedType::Date),
        DataType::Timestamp(_, _) => Some(IngestedType::Timestamp),
        DataType::List(field) | DataType::LargeList(field) => match field.data_type() {
            DataType::Float32 => Some(IngestedType::Float32Array),
            DataType::Float64 => Some(IngestedType::Float64Array),
            DataType::Int64 => Some(IngestedType::Int64Array),
            DataType::Utf8 | DataType::LargeUtf8 => Some(IngestedType::StringArray),
            _ => None,
        },
        _ => None,
    }
}

fn infer_semantic_type(name: &str, dtype: &DataType) -> Option<SemanticType> {
    if name == PRIMARY_ENTITY_COL {
        return Some(SemanticType::Id);
    }
    if name == TIMESTAMP_COL || matches!(dtype, DataType::Timestamp(_, _) | DataType::Date32) {
        return Some(SemanticType::Timestamp);
    }
    if name == PRIMARY_TARGET_COL || name == WEIGHT_COL {
        return Some(SemanticType::Numerical);
    }
    match dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => Some(SemanticType::Numerical),
        DataType::Boolean => Some(SemanticType::Categorical),
        DataType::Utf8 | DataType::LargeUtf8 => Some(SemanticType::Categorical),
        DataType::List(_) | DataType::LargeList(_) => Some(SemanticType::Sequence),
        _ => None,
    }
}
