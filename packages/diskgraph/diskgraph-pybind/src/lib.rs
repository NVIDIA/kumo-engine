// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::manual_inspect,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments
)]

use arrow::array::{
    make_array, ArrayData, BinaryArray, BinaryViewArray, Decimal128Array, Decimal256Array,
    FixedSizeBinaryArray, Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray,
    LargeStringArray, StringArray, StringViewArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::builder::{ArrayBuilder, UInt64Builder};
use arrow_array::{Array, RecordBatch};
use crossbeam_channel::{bounded, Receiver, Sender};
use diskgraph_index::pkmap::{Key, KeyType};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyString, PyTuple};

create_exception!(diskgraph, DiskgraphDataError, PyException);
create_exception!(
    diskgraph,
    DiskgraphSourceValidationError,
    DiskgraphDataError
);
create_exception!(
    diskgraph,
    DiskgraphTargetCategoryLimitError,
    DiskgraphDataError
);
create_exception!(diskgraph, DiskgraphConfigError, PyException);
create_exception!(diskgraph, DiskgraphSystemError, PyException);
// Distinct from the Data/Config/System siblings: corruption is unexpected
// and warrants alerting + falling through to a fresh ingest at the
// caller's level. Raised by ``verify_snapshot_integrity`` after a
// dataplane→local copy when the local snapshot does not match its
// ``_MANIFEST.json``.
create_exception!(diskgraph, DiskgraphCorruptionError, PyException);

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tracing::{error, info};

fn get_current_rss_mb() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = content.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(rss_pages) = parts[1].parse::<usize>() {
                    // Page size is usually 4KB on Linux
                    return (rss_pages * 4096) / 1024 / 1024;
                }
            }
        }
    }
    0
}

pub(crate) fn map_ttgen_err(err: diskgraph_train_table::errors::TTGenError) -> PyErr {
    use diskgraph_train_table::errors::TTGenError;
    match err {
        TTGenError::SchemaError(msg) => DiskgraphDataError::new_err(msg),
        TTGenError::InvalidPlan(msg) => DiskgraphConfigError::new_err(msg),
        TTGenError::TargetCategoryLimit(msg) => DiskgraphTargetCategoryLimitError::new_err(msg),
        TTGenError::SystemError(msg) => DiskgraphSystemError::new_err(msg),
        TTGenError::Io(e) => DiskgraphSystemError::new_err(e.to_string()),
        TTGenError::Parquet(msg) => DiskgraphSystemError::new_err(msg),
    }
}

pub(crate) fn map_engine_err(err: diskgraph_engine::errors::EngineError) -> PyErr {
    use diskgraph_engine::errors::EngineError;
    match err {
        EngineError::InvalidRequest(msg) => DiskgraphConfigError::new_err(msg),
        EngineError::Configuration(msg) => DiskgraphConfigError::new_err(msg),
        EngineError::DataCorruption(msg)
        | EngineError::Output(msg)
        | EngineError::Internal(msg) => DiskgraphSystemError::new_err(msg),
        EngineError::Io(e) => DiskgraphSystemError::new_err(e.to_string()),
    }
}

pub(crate) fn map_disk_manager_err(err: diskgraph_ingest::disk_manager::DiskManagerError) -> PyErr {
    use diskgraph_ingest::disk_manager::DiskManagerError;
    match err {
        DiskManagerError::InvalidPath(msg) => DiskgraphConfigError::new_err(msg),
        DiskManagerError::InsufficientSpace { needed, available } => {
            DiskgraphSystemError::new_err(format!(
                "Insufficient disk space: need {} bytes, only {} available",
                needed, available
            ))
        }
        DiskManagerError::Io(e) => DiskgraphSystemError::new_err(e.to_string()),
    }
}

pub(crate) fn map_anyhow_err(err: anyhow::Error) -> PyErr {
    // Check for disk space errors (ENOSPC/EDQUOT) first — actionable message
    if diskgraph_ingest::is_disk_space_error(&err) {
        return DiskgraphSystemError::new_err(format!(
            "Disk space exhausted during operation (ENOSPC/EDQUOT). \
             The local SSD may be full. Error: {err}"
        ));
    }
    if let Some(e) = err.downcast_ref::<diskgraph_ingest::errors::ConvertError>() {
        return DiskgraphConfigError::new_err(e.to_string());
    }
    if let Some(e) = err.downcast_ref::<diskgraph_ingest::errors::CoercionError>() {
        return DiskgraphDataError::new_err(e.to_string());
    }
    if let Some(e) = err.downcast_ref::<diskgraph_ingest::errors::KeyParseError>() {
        return DiskgraphDataError::new_err(e.to_string());
    }
    if let Some(e) = err.downcast_ref::<diskgraph_ingest::PublicIngestError>() {
        return DiskgraphSourceValidationError::new_err(e.to_string());
    }
    if let Some(e) = err.downcast_ref::<diskgraph_train_table::errors::TTGenError>() {
        match e {
            diskgraph_train_table::errors::TTGenError::SchemaError(_) => {
                return DiskgraphDataError::new_err(e.to_string());
            }
            diskgraph_train_table::errors::TTGenError::InvalidPlan(_) => {
                return DiskgraphConfigError::new_err(e.to_string());
            }
            diskgraph_train_table::errors::TTGenError::TargetCategoryLimit(_) => {
                return DiskgraphTargetCategoryLimitError::new_err(e.to_string());
            }
            _ => return DiskgraphSystemError::new_err(e.to_string()),
        }
    }

    map_py_err(err)
}

pub(crate) fn map_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    // Use alternate Display ({:#}) for anyhow errors to include the full
    // causal chain, not just the outermost context message.
    let msg = format!("{err:#}");
    let msg_lower = msg.to_lowercase();

    // Data Errors (Schema/mismatches bubbling from anyhow in ingest/export)
    if msg_lower.contains("schema error")
        || msg_lower.contains("column not found")
        || msg_lower.contains("table not found")
        || msg_lower.contains("type mismatch")
    {
        return DiskgraphDataError::new_err(msg);
    }

    // Config Errors (bubbling from anyhow in ingest/export)
    if msg_lower.contains("invalid plan json")
        || msg_lower.contains("invalid split config")
        || msg_lower.contains("invalid downsampling config")
        || msg_lower.contains("invalid kumo config json")
        || msg_lower.contains("invalid config json")
        || msg_lower.contains("unknown cat_count_strategy")
        || msg_lower.contains("unsupported time unit")
        || msg_lower.contains("unknown node type")
    {
        return DiskgraphConfigError::new_err(msg);
    }

    // Default to System Error for any unclassified anyhow payloads (panics, IO, memory)
    DiskgraphSystemError::new_err(msg)
}



use diskgraph_core::features::{BatchFeatureRequest, FeatureRequest};
use diskgraph_core::sampling::{BatchSamplingRequest, SamplingFlags, SamplingPlan};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::OutputHandler;
use diskgraph_engine::state::{SamplingState, StatePool};

mod artifact_helpers;
mod bp_col_stats;
mod col_stats_export;
mod custom_tt_entry;
mod index_metadata;
mod success_marker;
mod runtime_batch_iterator;
mod custom_tt_contract;
mod temporal;
mod tt_builder;
mod tt_executor;

#[pyclass(eq, eq_int, from_py_object, name = "TimeUnit", module = "diskgraph._c")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyTimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

#[pymethods]
impl PyTimeUnit {
    fn __hash__(&self) -> u64 {
        *self as u64
    }

    /// Make ``TimeUnit`` picklable across process boundaries.
    ///
    /// Pickle calls ``__reduce__`` to get a ``(callable, args)`` pair such
    /// that ``callable(*args)`` reconstructs the object on unpickle. We
    /// return ``(getattr, (TimeUnit, "<VariantName>"))`` so reconstruction
    /// resolves to the same class attribute (``TimeUnit.Nanoseconds`` etc.)
    /// without leaking integer reprs into the pickle stream.
    fn __reduce__<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let getattr = py.import("builtins")?.getattr("getattr")?;
        let cls = py.get_type::<Self>();
        let name = match *slf {
            PyTimeUnit::Seconds => "Seconds",
            PyTimeUnit::Milliseconds => "Milliseconds",
            PyTimeUnit::Microseconds => "Microseconds",
            PyTimeUnit::Nanoseconds => "Nanoseconds",
            PyTimeUnit::Minutes => "Minutes",
            PyTimeUnit::Hours => "Hours",
            PyTimeUnit::Days => "Days",
            PyTimeUnit::Weeks => "Weeks",
            PyTimeUnit::Months => "Months",
            PyTimeUnit::Years => "Years",
        };
        let args = PyTuple::new(py, [cls.into_any(), PyString::new(py, name).into_any()])?;
        PyTuple::new(py, [getattr, args.into_any()])
    }
}

impl From<PyTimeUnit> for diskgraph_core::time::TimeUnit {
    fn from(unit: PyTimeUnit) -> Self {
        match unit {
            PyTimeUnit::Seconds => diskgraph_core::time::TimeUnit::Seconds,
            PyTimeUnit::Milliseconds => diskgraph_core::time::TimeUnit::Milliseconds,
            PyTimeUnit::Microseconds => diskgraph_core::time::TimeUnit::Microseconds,
            PyTimeUnit::Nanoseconds => diskgraph_core::time::TimeUnit::Nanoseconds,
            PyTimeUnit::Minutes => diskgraph_core::time::TimeUnit::Minutes,
            PyTimeUnit::Hours => diskgraph_core::time::TimeUnit::Hours,
            PyTimeUnit::Days => diskgraph_core::time::TimeUnit::Days,
            PyTimeUnit::Weeks => diskgraph_core::time::TimeUnit::Weeks,
            PyTimeUnit::Months => diskgraph_core::time::TimeUnit::Months,
            PyTimeUnit::Years => diskgraph_core::time::TimeUnit::Years,
        }
    }
}

/// Events yielded by the sampling iterator.
#[derive(Debug)]
enum SampleEvent {
    EdgeBatch { rel_id: usize, batch: RecordBatch },
    NodeBatch { type_id: usize, batch: RecordBatch },
    FeatureBatch { type_id: usize, batch: RecordBatch },
    Stats { stats: String },
    Error { err: diskgraph_engine::errors::EngineError },
}

/// Bridge between Rust Engine and Python Iterator.
struct PyOutputHandler {
    sender: Sender<SampleEvent>,
    edge_builders: HashMap<u16, (UInt64Builder, UInt64Builder)>,
    chunk_size: usize,
    relation_src_type: Vec<usize>,
    relation_dst_type: Vec<usize>,
    edge_max_ids: HashMap<u16, (u32, u32)>,
}

impl PyOutputHandler {
    fn new(
        sender: Sender<SampleEvent>,
        relation_src_type: Vec<usize>,
        relation_dst_type: Vec<usize>,
    ) -> Self {
        let chunk_size = std::env::var("EDGE_BATCH_SIZE_ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64 * 1024);
        Self {
            sender,
            edge_builders: HashMap::new(),
            chunk_size,
            relation_src_type,
            relation_dst_type,
            edge_max_ids: HashMap::new(),
        }
    }

    fn flush_edge_builder(&mut self, rel_id: u16) -> diskgraph_engine::errors::Result<()> {
        let (src_b, dst_b) = if let Some(b) = self.edge_builders.get_mut(&rel_id) {
            b
        } else {
            return Ok(());
        };

        if src_b.is_empty() {
            return Ok(());
        }

        let len = src_b.len();
        let src_array = src_b.finish();
        let dst_array = dst_b.finish();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
            arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        ]));

        let row_type = arrow_array::Int8Array::from(vec![0; len]);
        let meta_type = arrow_array::UInt64Array::from(vec![rel_id as u64; len]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(row_type) as Arc<dyn Array>,
                Arc::new(src_array) as Arc<dyn Array>,
                Arc::new(dst_array) as Arc<dyn Array>,
                Arc::new(meta_type) as Arc<dyn Array>,
            ],
        )
        .map_err(|e| diskgraph_engine::errors::EngineError::Internal(e.to_string()))?;

        let _ = self.sender.send(SampleEvent::EdgeBatch {
            rel_id: rel_id as usize,
            batch,
        });
        Ok(())
    }
}

impl OutputHandler for PyOutputHandler {
    fn handle_edge(
        &mut self,
        _hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> diskgraph_engine::errors::Result<()> {
        let maxes = self.edge_max_ids.entry(rel_id).or_insert((0, 0));
        maxes.0 = maxes.0.max(src_local_id);
        maxes.1 = maxes.1.max(dst_local_id);

        let chunk_size = self.chunk_size;
        let (src_b, dst_b) = self.edge_builders.entry(rel_id).or_insert_with(|| {
            (
                UInt64Builder::with_capacity(chunk_size),
                UInt64Builder::with_capacity(chunk_size),
            )
        });

        src_b.append_value(src_local_id as u64);
        dst_b.append_value(dst_local_id as u64);

        if src_b.len() >= chunk_size {
            self.flush_edge_builder(rel_id)?;
        }
        Ok(())
    }

    fn handle_schema(
        &mut self,
        _schema: arrow_schema::SchemaRef,
    ) -> diskgraph_engine::errors::Result<()> {
        Ok(())
    }

    fn handle_feature_batch(&mut self, batch: RecordBatch) -> diskgraph_engine::errors::Result<()> {
        let _ = self
            .sender
            .send(SampleEvent::FeatureBatch { type_id: 0, batch });
        Ok(())
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: RecordBatch,
    ) -> diskgraph_engine::errors::Result<()> {
        let _ = self
            .sender
            .send(SampleEvent::FeatureBatch { type_id, batch });
        Ok(())
    }

    fn finalize(&mut self, state: &SamplingState) -> diskgraph_engine::errors::Result<()> {
        // Validation logic
        for (&rel_id, &(max_src, max_dst)) in &self.edge_max_ids {
            let rel_idx = rel_id as usize;
            if let (Some(&src_type_id), Some(&dst_type_id)) = (
                self.relation_src_type.get(rel_idx),
                self.relation_dst_type.get(rel_idx),
            ) {
                let src_count = state
                    .type_states
                    .get(&src_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);
                let dst_count = state
                    .type_states
                    .get(&dst_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);

                if src_count > 0 && (max_src as usize) >= src_count {
                    return Err(diskgraph_engine::errors::EngineError::DataCorruption(
                        format!(
                            "invalid src_local_id for rel_id={}: max_src={} >= src_count={}",
                            rel_id, max_src, src_count
                        ),
                    ));
                }
                if dst_count > 0 && (max_dst as usize) >= dst_count {
                    return Err(diskgraph_engine::errors::EngineError::DataCorruption(
                        format!(
                            "invalid dst_local_id for rel_id={}: max_dst={} >= dst_count={}",
                            rel_id, max_dst, dst_count
                        ),
                    ));
                }
            }
        }

        let rel_ids: Vec<u16> = self.edge_builders.keys().cloned().collect();
        for rel_id in rel_ids {
            self.flush_edge_builder(rel_id)?;
        }

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
            arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        ]));

        for (type_id, type_state) in &state.type_states {
            if type_state.nodes.is_empty() {
                continue;
            }
            let len = type_state.nodes.len();
            let row_type = arrow_array::Int8Array::from(vec![1; len]);
            let id_a = arrow_array::UInt64Array::from_iter_values(
                type_state.nodes.iter().map(|n| n.seed_index as u64),
            );
            let id_b = arrow_array::UInt64Array::from_iter_values(
                type_state.nodes.iter().map(|n| n.global_id),
            );
            let meta_type = arrow_array::UInt64Array::from(vec![*type_id as u64; len]);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(row_type) as Arc<dyn Array>,
                    Arc::new(id_a) as Arc<dyn Array>,
                    Arc::new(id_b) as Arc<dyn Array>,
                    Arc::new(meta_type) as Arc<dyn Array>,
                ],
            )
            .map_err(|e| diskgraph_engine::errors::EngineError::Internal(e.to_string()))?;

            let _ = self.sender.send(SampleEvent::NodeBatch {
                type_id: *type_id,
                batch,
            });
        }

        let stats_json = serde_json::to_string(&state.stats).unwrap_or_default();
        let _ = self.sender.send(SampleEvent::Stats { stats: stats_json });
        Ok(())
    }
}

#[pyclass]
pub struct SamplerEngine {
    reader: Arc<diskgraph_index::graph::GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    pool: StatePool,
    sample_count: Arc<std::sync::atomic::AtomicUsize>,
}

fn fixed_primary_key_bytes<const N: usize>(value: &[u8], key_type: &str) -> PyResult<[u8; N]> {
    value.try_into().map_err(|_| {
        DiskgraphDataError::new_err(format!(
            "{key_type} primary keys must contain exactly {N} bytes"
        ))
    })
}

fn primary_key_type_error(key_type: KeyType, data_type: &DataType) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "{key_type:?} primary keys are incompatible with Arrow type {data_type}"
    ))
}

fn canonical_node_mask_with<F>(
    table: &diskgraph_index::graph::TableReader,
    values: &dyn Array,
    node_ids: Vec<u64>,
    mut primary_key_at: F,
) -> PyResult<Vec<bool>>
where
    F: FnMut(usize) -> PyResult<Key>,
{
    node_ids.into_iter().enumerate().map(|(index, node_id)| {
        if values.is_null(index) {
            return Err(DiskgraphDataError::new_err(format!(
                "null primary key at batch offset {index}"
            )));
        }
        let key = primary_key_at(index)?;
        table
            .node_id_from_pk(&key)
            .map(|canonical| canonical == Some(node_id))
            .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))
    }).collect()
}

fn canonical_node_mask_from_arrow(
    table: &diskgraph_index::graph::TableReader,
    values: &dyn Array,
    node_ids: Vec<u64>,
) -> PyResult<Vec<bool>> {
    let key_type = table
        .pk_map()
        .ok_or_else(|| DiskgraphConfigError::new_err("table has no primary-key index"))?
        .key_type();

    macro_rules! mask_array {
        ($array_type:ty, |$array:ident, $index:ident| $key:expr) => {{
            let $array = values.as_any().downcast_ref::<$array_type>().unwrap();
            canonical_node_mask_with(table, $array, node_ids, |$index| $key)
        }};
    }
    macro_rules! mask_uuid_like {
        ($variant:ident, $size:literal, $label:literal) => {{
            match values.data_type() {
                DataType::Utf8 => mask_array!(StringArray, |array, index| table
                    .parse_pk_key(array.value(index))
                    .map_err(|e| DiskgraphDataError::new_err(e.to_string()))),
                DataType::LargeUtf8 => mask_array!(LargeStringArray, |array, index| table
                    .parse_pk_key(array.value(index))
                    .map_err(|e| DiskgraphDataError::new_err(e.to_string()))),
                DataType::Utf8View => mask_array!(StringViewArray, |array, index| table
                    .parse_pk_key(array.value(index))
                    .map_err(|e| DiskgraphDataError::new_err(e.to_string()))),
                DataType::Binary => mask_array!(BinaryArray, |array, index| Ok(Key::$variant(
                    fixed_primary_key_bytes::<$size>(array.value(index), $label)?
                ))),
                DataType::LargeBinary => mask_array!(LargeBinaryArray, |array, index| Ok(
                    Key::$variant(fixed_primary_key_bytes::<$size>(
                        array.value(index), $label
                    )?)
                )),
                DataType::BinaryView => mask_array!(BinaryViewArray, |array, index| Ok(
                    Key::$variant(fixed_primary_key_bytes::<$size>(
                        array.value(index), $label
                    )?)
                )),
                DataType::FixedSizeBinary(_) => mask_array!(FixedSizeBinaryArray,
                    |array, index| Ok(Key::$variant(fixed_primary_key_bytes::<$size>(
                        array.value(index), $label
                    )?))),
                data_type => Err(primary_key_type_error(key_type, data_type)),
            }
        }};
    }

    match key_type {
        KeyType::Int64 => match values.data_type() {
            DataType::Int8 => mask_array!(Int8Array, |array, index| Ok(Key::Int(
                array.value(index) as i64 as u64))),
            DataType::Int16 => mask_array!(Int16Array, |array, index| Ok(Key::Int(
                array.value(index) as i64 as u64))),
            DataType::Int32 => mask_array!(Int32Array, |array, index| Ok(Key::Int(
                array.value(index) as i64 as u64))),
            DataType::Int64 => mask_array!(Int64Array, |array, index| Ok(Key::Int(
                array.value(index) as u64))),
            DataType::UInt8 => mask_array!(UInt8Array, |array, index| Ok(Key::Int(
                array.value(index) as u64))),
            DataType::UInt16 => mask_array!(UInt16Array, |array, index| Ok(Key::Int(
                array.value(index) as u64))),
            DataType::UInt32 => mask_array!(UInt32Array, |array, index| Ok(Key::Int(
                array.value(index) as u64))),
            DataType::UInt64 => mask_array!(UInt64Array, |array, index| Ok(Key::Int(
                array.value(index)))),
            DataType::Decimal128(_, scale) => {
                let scale = *scale;
                mask_array!(Decimal128Array, |array, index| {
                    let mut value = array.value(index);
                    if scale > 0 {
                        if let Some(divisor) = 10i128.checked_pow(scale as u32) {
                            value /= divisor;
                        }
                    }
                    let value = i64::try_from(value).map_err(|_| {
                        DiskgraphDataError::new_err(format!(
                            "decimal128 primary key at batch offset {index} exceeds int64 range"
                        ))
                    })?;
                    Ok(Key::Int(value as u64))
                })
            }
            data_type => Err(primary_key_type_error(key_type, data_type)),
        },
        KeyType::VarBytes => match values.data_type() {
            DataType::Utf8 => mask_array!(StringArray, |array, index| Ok(Key::Bytes(
                array.value(index).as_bytes().to_vec()))),
            DataType::LargeUtf8 => mask_array!(LargeStringArray, |array, index| Ok(Key::Bytes(
                array.value(index).as_bytes().to_vec()))),
            DataType::Utf8View => mask_array!(StringViewArray, |array, index| Ok(Key::Bytes(
                array.value(index).as_bytes().to_vec()))),
            DataType::Binary => mask_array!(BinaryArray, |array, index| Ok(Key::Bytes(
                array.value(index).to_vec()))),
            DataType::LargeBinary => mask_array!(LargeBinaryArray, |array, index| Ok(Key::Bytes(
                array.value(index).to_vec()))),
            DataType::BinaryView => mask_array!(BinaryViewArray, |array, index| Ok(Key::Bytes(
                array.value(index).to_vec()))),
            DataType::FixedSizeBinary(_) => mask_array!(FixedSizeBinaryArray,
                |array, index| Ok(Key::Bytes(array.value(index).to_vec()))),
            DataType::Decimal256(_, _) => mask_array!(Decimal256Array, |array, index| {
                let value = array.value_as_string(index);
                let bytes = diskgraph_core::decimal_key::normalize_base10_key(value.as_bytes())
                    .map_err(|error| {
                        DiskgraphDataError::new_err(format!(
                            "invalid decimal256 primary key at batch offset {index}: {error}"
                        ))
                    })?;
                Ok(Key::Bytes(bytes))
            }),
            data_type => Err(primary_key_type_error(key_type, data_type)),
        },
        KeyType::Uuid => mask_uuid_like!(Uuid, 16, "UUID"),
        KeyType::Uuid256 => mask_uuid_like!(Uuid256, 32, "UUID256"),
        KeyType::Ip => mask_uuid_like!(Ip, 16, "IP"),
    }
}

#[pymethods]
impl SamplerEngine {
    #[new]
    #[pyo3(signature = (path, random_seed=None, num_threads=None))]
    #[tracing::instrument(level = "info", skip(path), fields(index_path = %path))]
    pub fn new(path: String, random_seed: Option<u64>, num_threads: Option<usize>) -> PyResult<Self> {
        let root = std::path::Path::new(&path);
        let reader = Arc::new(
            diskgraph_index::graph::GraphIndexReader::open(&root.join("index_metadata.json"), root)
                .map_err(|e| DiskgraphSystemError::new_err(e.to_string()))?,
        );
        let engine = DiskGraphEngine::from_reader(Arc::clone(&reader), random_seed, num_threads)
            .map_err(map_engine_err)?;
        info!("SamplerEngine initialized successfully");
        Ok(Self {
            reader,
            engine: Arc::new(engine),
            pool: StatePool::new(),
            sample_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Return whether each supplied node ID is the PK map's canonical row.
    /// The caller already fetched the aligned primary-key values, so this is a
    /// bounded lookup-only bridge over `TableReader::node_id_from_pk`.
    pub fn canonical_node_mask(
        &self,
        py: Python<'_>,
        node_type: String,
        node_ids: Vec<u64>,
        primary_keys: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<bool>> {
        let primary_keys = make_array(ArrayData::from_pyarrow_bound(primary_keys)?);
        if node_ids.len() != primary_keys.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node_ids and primary_keys must have equal length: {} != {}",
                node_ids.len(),
                primary_keys.len()
            )));
        }
        let table = self.reader.table(&node_type).ok_or_else(|| {
            DiskgraphConfigError::new_err(format!("unknown node type: {node_type}"))
        })?;
        if table.pk_map().is_none() {
            return Err(DiskgraphConfigError::new_err(format!(
                "node type {node_type} has no primary-key index"
            )));
        }
        py.detach(move || canonical_node_mask_from_arrow(&table, primary_keys.as_ref(), node_ids))
    }

    #[pyo3(signature = (seed_nodes, plan_json, fused, seed_timestamps=None, swap_edge_direction=None, seed_timestamp_unit=None))]
    #[tracing::instrument(level = "debug", skip(self, py, seed_nodes, plan_json))]
    pub fn sample(
        &self,
        py: Python<'_>,
        seed_nodes: Bound<'_, PyDict>,
        plan_json: String,
        fused: bool,
        seed_timestamps: Option<Bound<'_, PyDict>>,
        swap_edge_direction: Option<bool>,
        seed_timestamp_unit: Option<PyTimeUnit>,
    ) -> PyResult<PySampleIterator> {
        let mut seeds = HashMap::new();
        for (key, value) in seed_nodes.into_iter() {
            let type_name: String = key.extract()?;
            let ids: Vec<u64> = value.extract()?;
            seeds.insert(type_name, ids);
        }

        let mut timestamps = None;
        if let Some(st) = seed_timestamps {
            let mut ts_map = HashMap::new();
            for (key, value) in st.into_iter() {
                let type_name: String = key.extract()?;
                let ts: Vec<i64> = value.extract()?;
                ts_map.insert(type_name, ts);
            }
            timestamps = Some(ts_map);
        }

        let plan: SamplingPlan = serde_json::from_str(&plan_json)
            .map_err(|e| DiskgraphConfigError::new_err(format!("Invalid plan JSON: {}", e)))?;

        let flags = SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: fused,
            swap_edge_direction: swap_edge_direction.unwrap_or(true),
        };

        let parsed_unit = seed_timestamp_unit
            .map(|u| u.into())
            .unwrap_or(diskgraph_core::time::TimeUnit::Nanoseconds);

        let request = BatchSamplingRequest {
            dataset_id: uuid::Uuid::nil(),
            seed_nodes: seeds,
            seed_timestamps: timestamps,
            seed_timestamp_unit: parsed_unit,
            plan,
            flags,
        };

        let seed_type = request
            .seed_nodes
            .keys()
            .next()
            .cloned()
            .unwrap_or_default();
        let seed_count = request
            .seed_nodes
            .values()
            .next()
            .map_or(0usize, |v| v.len());

        let count = self
            .sample_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count % 100 == 0 {
            info!(
                seed_type = %seed_type,
                seed_count,
                hops = request.plan.layers.len(),
                fused_feature_fetching = request.flags.fused_feature_fetching,
                swap_edge_direction = request.flags.swap_edge_direction,
                "Pybind sampling request started"
            );
        } else {
            tracing::debug!(
                seed_type = %seed_type,
                seed_count,
                hops = request.plan.layers.len(),
                fused_feature_fetching = request.flags.fused_feature_fetching,
                swap_edge_direction = request.flags.swap_edge_direction,
                "Pybind sampling request started"
            );
        }

        let (tx, rx) = bounded(1);
        let engine = Arc::clone(&self.engine);
        let mut state = self.pool.acquire();

        py.detach(move || {
            thread::spawn(move || {
                let mut handler = PyOutputHandler::new(
                    tx.clone(),
                    engine.relation_src_type.clone(),
                    engine.relation_info.clone(),
                );
                match engine.sample(&request, &mut state, &mut handler) {
                    Ok(()) => {
                        if count % 100 == 0 {
                            info!(
                                total_nodes = state.stats.total_nodes,
                                total_edges = state.stats.total_edges,
                                process_rss_mb = get_current_rss_mb(),
                                "Pybind sampling request finished"
                            );
                        } else {
                            tracing::debug!(
                                total_nodes = state.stats.total_nodes,
                                total_edges = state.stats.total_edges,
                                process_rss_mb = get_current_rss_mb(),
                                "Pybind sampling request finished"
                            );
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Pybind sampling request failed");
                        let _ = tx.send(SampleEvent::Error { err: e });
                    }
                }
            });
        });

        Ok(PySampleIterator { receiver: rx })
    }

    #[tracing::instrument(level = "debug", skip(self, py, batch))]
    pub fn fetch_features(
        &self,
        py: Python<'_>,
        batch: Bound<'_, PyList>,
    ) -> PyResult<PySampleIterator> {
        let mut req_batch = Vec::new();
        for item in batch.iter() {
            let dict: Bound<'_, PyDict> = item.extract()?;
            let node_type: String = dict
                .get_item("node_type")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'node_type' in feature request"))?
                .extract()?;
            let node_ids: Vec<u64> = dict
                .get_item("node_ids")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'node_ids' in feature request"))?
                .extract()?;
            let feature_names: Vec<String> = dict
                .get_item("feature_names")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'feature_names' in feature request"))?
                .extract()?;

            req_batch.push(FeatureRequest {
                node_type,
                node_ids,
                feature_names,
            });
        }

        let request = BatchFeatureRequest {
            dataset_id: uuid::Uuid::nil(),
            batch: req_batch,
        };

        let req_count = request.batch.len();
        let count = self.sample_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count % 100 == 0 {
            info!(requests = req_count, "Pybind feature request started");
        } else {
            tracing::debug!(requests = req_count, "Pybind feature request started");
        }

        let (tx, rx) = bounded(1);
        let engine = Arc::clone(&self.engine);

        py.detach(move || {
            thread::spawn(move || {
                let mut handler = PyOutputHandler::new(tx.clone(), Vec::new(), Vec::new());

                for feature_req in request.batch {
                    let node_type_id = match engine.node_type_to_id.get(&feature_req.node_type) {
                        Some(&id) => id,
                        None => {
                            error!(node_type = %feature_req.node_type, "Unknown node type in pybind feature request");
                            let _ = tx.send(SampleEvent::Error {
                                err: diskgraph_engine::errors::EngineError::InvalidRequest(format!("Unknown node type: {}", feature_req.node_type)),
                            });
                            return;
                        }
                    };

                    if let Err(e) = engine.fetch_features_for_type(
                        node_type_id,
                        &feature_req.node_ids,
                        &feature_req.feature_names,
                        &mut handler,
                    ) {
                        error!(error = %e, node_type = %feature_req.node_type, "Pybind feature request failed");
                        let _ = tx.send(SampleEvent::Error { err: e });
                        return;
                    }
                }
                if count % 100 == 0 {
                    info!(
                        process_rss_mb = get_current_rss_mb(),
                        "Pybind feature request finished"
                    );
                } else {
                    tracing::debug!(
                        process_rss_mb = get_current_rss_mb(),
                        "Pybind feature request finished"
                    );
                }
            });
        });

        Ok(PySampleIterator { receiver: rx })
    }

    pub fn get_type_mapping(&self) -> PyResult<String> {
        let mapping = self.engine.get_type_name_mapping();
        serde_json::to_string(&mapping).map_err(|e| DiskgraphSystemError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct PySampleIterator {
    receiver: Receiver<SampleEvent>,
}

#[pymethods]
impl PySampleIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let event = py.detach(|| self.receiver.recv());

        match event {
            Ok(SampleEvent::EdgeBatch { rel_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "edge")?;
                dict.set_item("rel_id", rel_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::NodeBatch { type_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "node")?;
                dict.set_item("type_id", type_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::FeatureBatch { type_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "feature")?;
                dict.set_item("type_id", type_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::Stats { stats }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "stats")?;
                dict.set_item("stats", stats)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::Error { err }) => Err(crate::map_engine_err(err)),
            Err(_) => Ok(None),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct FkeyConfig {
    #[pyo3(get, set)]
    pub column_name: String,
    #[pyo3(get, set)]
    pub target_table: String,
}

#[pymethods]
impl FkeyConfig {
    #[new]
    fn new(column_name: String, target_table: String) -> Self {
        Self {
            column_name,
            target_table,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ExportConfig {
    #[pyo3(get, set)]
    pub train_table_path: String,
    #[pyo3(get, set)]
    pub index_path: String,
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub row_idx_col: String,
    #[pyo3(get, set)]
    pub train_table_name: String,
    #[pyo3(get, set)]
    pub fkeys: Vec<FkeyConfig>,
}

#[pymethods]
impl ExportConfig {
    #[new]
    #[pyo3(signature = (train_table_path, index_path, output_dir, fkeys, row_idx_col="arange".to_string(), train_table_name="custom_train_table".to_string()))]
    fn new(
        train_table_path: String,
        index_path: String,
        output_dir: String,
        fkeys: Vec<FkeyConfig>,
        row_idx_col: String,
        train_table_name: String,
    ) -> Self {
        Self {
            train_table_path,
            index_path,
            output_dir,
            row_idx_col,
            train_table_name,
            fkeys,
        }
    }
}

#[pyfunction]
pub fn export_tt_artifacts(py: Python<'_>, config: &ExportConfig) -> PyResult<()> {
    info!(
        train_table_path = %config.train_table_path,
        index_path = %config.index_path,
        output_dir = %config.output_dir,
        fkeys = config.fkeys.len(),
        "Pybind export_tt_artifacts started"
    );
    let fkeys = config
        .fkeys
        .iter()
        .map(|f| diskgraph_engine::export_tt_artifacts::FkeyConfig {
            column_name: f.column_name.clone(),
            target_table: f.target_table.clone(),
        })
        .collect();

    let internal_config = diskgraph_engine::export_tt_artifacts::ExportConfig {
        train_table_path: config.train_table_path.clone(),
        index_path: config.index_path.clone(),
        output_dir: config.output_dir.clone(),
        row_idx_col: config.row_idx_col.clone(),
        train_table_name: config.train_table_name.clone(),
        fkeys,
    };

    // Release the GIL during export — this is a long-running CPU/IO operation
    // and holding the GIL would block heartbeats and other Python threads in the
    // Temporal worker.
    py.detach(move || {
        diskgraph_engine::export_tt_artifacts::process_export(&internal_config)
    })
    .map(|_| {
        info!("Pybind export_tt_artifacts finished");
    })
    .map_err(map_anyhow_err)
}

#[pyfunction]
#[pyo3(signature = (kumo_config_json, data_persist_dir, dataset_name, table_data_versions_json=None, source_paths_base=None, source_paths_json=None, source_specs_json=None))]
#[tracing::instrument(level = "info", skip_all, fields(dataset_name = %dataset_name, data_persist_dir = %data_persist_dir))]
pub fn convert_kumo_config(
    py: Python<'_>,
    kumo_config_json: &str,
    data_persist_dir: &str,
    dataset_name: &str,
    table_data_versions_json: Option<&str>,
    source_paths_base: Option<&str>,
    source_paths_json: Option<&str>,
    source_specs_json: Option<&str>,
) -> PyResult<String> {
    info!(
        has_table_data_versions = table_data_versions_json.is_some(),
        has_source_paths_base = source_paths_base.is_some(),
        has_source_paths = source_paths_json.is_some(),
        has_source_specs = source_specs_json.is_some(),
        "Pybind convert_kumo_config started"
    );
    let start_time = std::time::Instant::now();
    let kumo_config_json = kumo_config_json.to_string();
    let data_persist_dir = data_persist_dir.to_string();
    let dataset_name = dataset_name.to_string();
    let table_data_versions_json = table_data_versions_json.map(str::to_string);
    let source_paths_base = source_paths_base.map(str::to_string);
    let source_paths_json = source_paths_json.map(str::to_string);
    let source_specs_json = source_specs_json.map(str::to_string);

    let result = py.detach(move || -> PyResult<String> {
        use diskgraph_ingest::config_converter::{
            convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
        };

        let kumo_config: KumoGraphConfig = serde_json::from_str(&kumo_config_json)
            .map_err(|e| DiskgraphConfigError::new_err(format!("Invalid kumo config JSON: {e}")))?;

        let table_data_versions: Option<HashMap<String, String>> =
            match table_data_versions_json.as_deref() {
                Some(json) => Some(serde_json::from_str(json).map_err(|e| {
                    DiskgraphConfigError::new_err(format!(
                        "Invalid table_data_versions JSON: {e}"
                    ))
                })?),
                None => None,
            };

        let source_paths: Option<HashMap<String, String>> =
            match source_paths_json.as_deref() {
                Some(json) => Some(serde_json::from_str(json).map_err(|e| {
                    DiskgraphConfigError::new_err(format!("Invalid source_paths JSON: {e}"))
                })?),
                None => None,
            };

        let source_specs: Option<
            HashMap<String, diskgraph_ingest::config_converter::DirectSourceSpec>,
        > = match source_specs_json.as_deref() {
            Some(json) => Some(serde_json::from_str(json).map_err(|e| {
                DiskgraphConfigError::new_err(format!("Invalid source_specs JSON: {e}"))
            })?),
            None => None,
        };

        let options = ConvertOptions {
            data_persist_dir,
            dataset_name,
            table_data_versions,
            source_paths_base,
            source_paths,
            source_specs,
        };

        // convert_kumo_to_diskgraph handles everything: schema, timestamp detection, num_shards.
        let diskgraph_config = convert_kumo_to_diskgraph(&kumo_config, &options)
            .map_err(|e| match e {
                diskgraph_ingest::errors::ConvertError::SourcePathsIO { .. } => {
                    DiskgraphSystemError::new_err(format!("{e}"))
                }
                _ => DiskgraphDataError::new_err(format!("Config conversion failed: {e}")),
            })?;

        serde_json::to_string(&diskgraph_config)
            .map_err(|e| DiskgraphSystemError::new_err(format!("Failed to serialize config: {e}")))
    });

    result.map(|s| {
        info!(
            duration_ms = start_time.elapsed().as_millis(),
            "Pybind convert_kumo_config finished"
        );
        s
    })
}

#[pyfunction]
#[pyo3(signature = (
    config_json,
    output_dir,
    col_stats=true,
    cat_count_strategy="hybrid",
    output_dataplane_dir=None,
    success_marker_content=None,
    resume_from_phase=None,
    context_threads=None,
    encode_threads=None,
    pk_map_threads=None,
    edges_map_threads=None,
    index_threads=None,
    parent_index_path=None,
))]
#[tracing::instrument(
    level = "info",
    skip(py, config_json, success_marker_content),
    fields(
        output_dir = %output_dir,
        dataplane = output_dataplane_dir.unwrap_or("(none)"),
        resume_from_phase = ?resume_from_phase,
    )
)]
pub fn ingest(
    py: Python<'_>,
    config_json: &str,
    output_dir: &str,
    col_stats: bool,
    cat_count_strategy: &str,
    output_dataplane_dir: Option<&str>,
    success_marker_content: Option<&str>,
    resume_from_phase: Option<&str>,
    context_threads: Option<usize>,
    encode_threads: Option<usize>,
    pk_map_threads: Option<usize>,
    edges_map_threads: Option<usize>,
    index_threads: Option<usize>,
    parent_index_path: Option<&str>,
) -> PyResult<String> {
    let start_time = std::time::Instant::now();
    info!(
        output_dir = %output_dir,
        col_stats,
        cat_count_strategy = %cat_count_strategy,
        dataplane = %output_dataplane_dir.unwrap_or("(none)"),
        marker = success_marker_content.is_some(),
        resume_from_phase = ?resume_from_phase,
        "Pybind ingest started"
    );
    let source: diskgraph_ingest::GraphConfig = serde_json::from_str(config_json)
        .map_err(|e| DiskgraphConfigError::new_err(format!("Invalid config JSON: {e}")))?;

    let strategy = match cat_count_strategy {
        "hybrid" => diskgraph_ingest::col_stats::CategoryCountStrategy::Hybrid,
        "graduation" => diskgraph_ingest::col_stats::CategoryCountStrategy::Graduation,
        "none" => diskgraph_ingest::col_stats::CategoryCountStrategy::None,
        other => {
            return Err(DiskgraphConfigError::new_err(format!(
                "Unknown cat_count_strategy: {other}"
            )))
        }
    };

    let resume_phase = resume_from_phase
        .map(|s| {
            diskgraph_ingest::IngestPhase::from_name(s).ok_or_else(|| {
                DiskgraphConfigError::new_err(format!(
                    "Unknown resume_from_phase: {s} (expected: encode, pk_map, edges_map, index)"
                ))
            })
        })
        .transpose()?;

    let options = diskgraph_ingest::IngestOptions {
        profile_dir: None,
        progress: diskgraph_ingest::progress::build_progress(),
        parallel: true,
        index_mem_budget_bytes: None,
        peak_mem_stats: None,
        col_stats,
        cat_count_strategy: strategy,
        row_subsample: false,
        resume_from_phase: resume_phase,
        source_loader_threads: context_threads,
        source_encoder_threads: encode_threads,
        pk_map_threads,
        edges_map_threads,
        index_threads,
        parent_index_path: parent_index_path.map(std::path::PathBuf::from),
        resolve_columns: Vec::new(),
    };

    let output_dir = output_dir.to_string();
    let output_dataplane_dir = output_dataplane_dir.map(|s| s.to_string());
    let success_marker_content = success_marker_content.map(|s| s.to_string());

    // Release the GIL during ingest — this is a long-running CPU/IO operation
    // and holding the GIL would block heartbeats and other activities in the
    // Temporal worker.
    let result = py.detach(move || {
        diskgraph_ingest::ingest_with_stats_from_source(
            source,
            None,
            &output_dir,
            diskgraph_ingest::Dest::Local,
            options,
            output_dataplane_dir.as_deref(),
            success_marker_content.as_deref(),
        )
    });

    let (_metadata, stats) =
        result.map_err(|e| map_anyhow_err(e.context("Ingest failed")))?;

    let elapsed_ms = start_time.elapsed().as_millis();
    serde_json::to_string(&stats)
        .map(|s| {
            info!(
                total_nodes = stats.node_counts.values().copied().sum::<u64>(),
                total_edges = stats.edge_counts.values().copied().sum::<u64>(),
                peak_rss_mb = stats.perf.peak_rss_anon_bytes.unwrap_or(0) / 1024 / 1024,
                duration_ms = elapsed_ms,
                "Ingestion materialization complete"
            );
            s
        })
        .map_err(|e| DiskgraphSystemError::new_err(format!("Failed to serialize stats: {e}")))
}

// ---------------------------------------------------------------------------
// Disk Manager
// ---------------------------------------------------------------------------

#[pyclass]
struct DiskManager {
    inner: diskgraph_ingest::disk_manager::DiskManager,
}

#[pymethods]
impl DiskManager {
    #[new]
    fn new(data_root: &str) -> PyResult<Self> {
        let path = std::path::Path::new(data_root);
        if !path.exists() {
            return Err(DiskgraphConfigError::new_err(format!(
                "data_root does not exist: {data_root}"
            )));
        }
        Ok(Self {
            inner: diskgraph_ingest::disk_manager::DiskManager::new(path),
        })
    }

    /// Ensure enough disk space for an ingest, evicting LRU indexes if needed.
    /// Returns JSON EnsureSpaceResult.
    fn ensure_space(
        &self,
        py: Python<'_>,
        needed_bytes: u64,
        current_graph_key: &str,
        current_graph_version: &str,
    ) -> PyResult<String> {
        let req = diskgraph_ingest::disk_manager::types::EnsureSpaceRequest {
            needed_bytes,
            current_graph_key: current_graph_key.to_string(),
            current_graph_version: current_graph_version.to_string(),
        };
        let result = py.detach(|| self.inner.ensure_space(&req))
            .map_err(map_disk_manager_err)?;
        serde_json::to_string(&result)
            .map_err(|e| DiskgraphSystemError::new_err(format!("serialization failed: {e}")))
    }

    /// Scan all index directories under data_root. Returns JSON Vec<IndexInfo>.
    fn scan_indexes(&self, py: Python<'_>) -> PyResult<String> {
        let indexes = py.detach(|| self.inner.scan_indexes())
            .map_err(map_disk_manager_err)?;
        serde_json::to_string(&indexes)
            .map_err(|e| DiskgraphSystemError::new_err(format!("serialization failed: {e}")))
    }

    /// Get disk usage stats. Returns JSON DiskUsage.
    fn disk_usage(&self, py: Python<'_>) -> PyResult<String> {
        let usage = py.detach(|| self.inner.disk_usage())
            .map_err(map_disk_manager_err)?;
        serde_json::to_string(&usage)
            .map_err(|e| DiskgraphSystemError::new_err(format!("serialization failed: {e}")))
    }
}

/// Estimate the total source bytes for an ingest from a graph config JSON.
/// Sums file sizes for all local table paths. Returns bytes as u64.
#[pyfunction]
fn estimate_ingest_size(py: Python<'_>, config_json: &str) -> PyResult<u64> {
    let config_json = config_json.to_string();
    py.detach(move || -> PyResult<u64> {
        let config: diskgraph_ingest::GraphConfig = serde_json::from_str(&config_json)
            .map_err(|e| DiskgraphConfigError::new_err(format!("invalid config JSON: {e}")))?;
        // This estimator counts local paths from the already-converted config;
        // without a CloudTransfer, compute_total_bytes ignores cloud paths.
        let total =
            diskgraph_ingest::config_converter::inference::compute_total_bytes(&config, &None)
                .map_err(|e| {
                    DiskgraphSystemError::new_err(format!("failed to compute source bytes: {e}"))
                })?;
        if total == 0 {
            return Err(DiskgraphDataError::new_err(
                "computed source size is 0 bytes — no data files found in table paths",
            ));
        }
        Ok(total)
    })
}

/// Compute Kumo-format column statistics over one or more parquet files.
///
/// Wraps :func:`bp_col_stats::compute_parquet_col_stats` for kumo's
/// batch-prediction stats path: after a BP run writes its output parquet,
/// kumo invokes this to produce the same ``kumo_col_stats_v1`` JSON that
/// ingest emits, so downstream readers (UI, encoder inference,
/// ``PredStats.parse``) light up without further changes on the consumer
/// side.
///
/// Row groups are processed in parallel (rayon); category counts use the
/// count-capped Hybrid strategy. Scalar and category statistics are computed
/// in a first pass, then exact numerical and timestamp distributions in a
/// second pass. The full and min variants share the accumulated statistics.
///
/// Args:
///     parquet_paths: list of parquet file paths (single file or all parts
///         of a part-*.parquet directory).
///     kumo_column_specs_json: JSON list of ``{"name", "stype", "dtype"}``
///         records (optional ``"timestamp_unit"``: ``"s"``/``"ms"``/``"us"``/
///         ``"ns"`` and optional ``"timestamp_format"``). Each spec's
///         ``name`` is matched to its physical parquet
///         column by name, so the list order need not match the parquet
///         schema. Integer-physical timestamps prefer authoritative unit/format
///         metadata and retain magnitude inference for older callers. A missing
///         name, Arrow Null column, or narrow integer-array overflow raises
///         ``DiskgraphDataError``; other incompatible values are counted as
///         invalid. Semantic and logical types are mapped through the same
///         ``map_kumo_stype`` / ``map_kumo_dtype`` helpers ingest uses.
///     data_version: version string stamped into every emitted entry.
///     _category_count_limit_for_testing: keyword-only test hook that lowers
///         the retained category limit. ``None`` preserves the production
///         1,000,000-category limit; a supplied value must be greater than 0.
///
/// Returns ``(full_json, min_json)`` — the full ``kumo_col_stats_v1`` map and
/// its min (truncated-CATEGORY_COUNTS) variant, each as a JSON string,
/// suitable for direct ``fsutil.write_text(table_dir.col_stats, ...)`` /
/// ``min_col_stats``.
#[pyfunction]
#[pyo3(signature = (
    parquet_paths,
    kumo_column_specs_json,
    data_version,
    *,
    _category_count_limit_for_testing=None,
))]
#[tracing::instrument(level = "debug", skip(py, kumo_column_specs_json))]
fn compute_col_stats_for_parquet(
    py: Python<'_>,
    parquet_paths: Vec<String>,
    kumo_column_specs_json: &str,
    data_version: &str,
    _category_count_limit_for_testing: Option<u64>,
) -> PyResult<(String, String)> {
    use diskgraph_ingest::config_converter::{
        map_kumo_dtype, map_kumo_stype, map_timestamp_unit,
    };
    use diskgraph_ingest::graph_config::ColumnSpec;

    use crate::bp_col_stats::{
        compute_parquet_col_stats, compute_parquet_col_stats_with_max_categories,
    };

    if _category_count_limit_for_testing == Some(0) {
        return Err(DiskgraphConfigError::new_err(
            "_category_count_limit_for_testing must be greater than zero",
        ));
    }

    #[derive(serde::Deserialize)]
    struct KumoColumnSpec {
        name: String,
        stype: String,
        dtype: String,
        // Authoritative epoch unit for integer-physical timestamp columns
        // ("s" / "ms" / "us" / "ns"). Arrow Timestamp columns carry their
        // own unit and do not need this field; integer timestamps do.
        #[serde(default)]
        timestamp_unit: Option<String>,
        #[serde(default)]
        timestamp_format: Option<String>,
    }

    let kumo_cols: Vec<KumoColumnSpec> = serde_json::from_str(kumo_column_specs_json)
        .map_err(|e| DiskgraphConfigError::new_err(format!(
            "invalid kumo_column_specs_json: {e}"
        )))?;
    let columns: Vec<ColumnSpec> = kumo_cols
        .into_iter()
        .map(|c| {
            let ingested_dtype = map_kumo_dtype(&c.dtype).map_err(|d| {
                DiskgraphConfigError::new_err(format!(
                    "unsupported kumo dtype '{d}' for column '{}'",
                    c.name
                ))
            })?;
            let timestamp_unit = c
                .timestamp_unit
                .as_deref()
                .map(|unit| {
                    map_timestamp_unit(unit).ok_or_else(|| {
                        DiskgraphConfigError::new_err(format!("unsupported timestamp unit '{unit}'"))
                    })
                })
                .transpose()?;
            Ok(ColumnSpec {
                name: c.name,
                source_dtype: None,
                ingested_dtype,
                fixed_len: None,
                timestamp_format: c.timestamp_format,
                timestamp_unit,
                precision: None,
                scale: None,
                semantic_type: map_kumo_stype(&c.stype),
                multicategorical_delimiter: None,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let data_version = data_version.to_string();
    let paths_owned = parquet_paths;

    py.detach(|| -> anyhow::Result<(String, String)> {
        let paths: Vec<&std::path::Path> =
            paths_owned.iter().map(std::path::Path::new).collect();
        match _category_count_limit_for_testing {
            Some(limit) => compute_parquet_col_stats_with_max_categories(
                &paths,
                &columns,
                &data_version,
                limit,
            ),
            None => compute_parquet_col_stats(&paths, &columns, &data_version),
        }
    })
    // Route schema failures through the shared exception classifier rather
    // than reporting missing columns or invalid timestamp metadata as a
    // blanket system error. Unclassified failures still fall through to
    // `DiskgraphSystemError`.
    .map_err(map_anyhow_err)
}

/// Verify a local snapshot directory matches its ``_MANIFEST.json``.
///
/// Called by the Python ``DiskgraphService`` after a dataplane→local
/// copy completes (and as a defense-in-depth check on the SSD-cached
/// path) to catch lifecycle-mid-deletion races, partial copy failures,
/// and other corruption classes that would otherwise produce a silently
/// broken snapshot.
///
/// Raises [`DiskgraphCorruptionError`] (a distinct exception class so
/// callers can branch on it for alerting + fallback to fresh ingest)
/// when:
///   - ``_SUCCESS`` is missing,
///   - ``_MANIFEST.json`` is missing or unparseable,
///   - the manifest's schema version is unsupported,
///   - any manifest-listed file is missing locally, or
///   - any manifest-listed file's size does not match the manifest.
///
/// Returns ``None`` on success — no payload, since the caller already
/// has the path it just verified.
#[pyfunction]
#[tracing::instrument(level = "debug", skip(py))]
fn verify_snapshot_integrity(py: Python<'_>, output_dir: &str) -> PyResult<()> {
    let output_dir = std::path::PathBuf::from(output_dir);
    py.detach(|| diskgraph_ingest::snapshot_mirror::verify_snapshot_integrity(&output_dir))
        .map_err(|e| DiskgraphCorruptionError::new_err(e.to_string()))
}

/// CRC-32 of every `range_bytes` span of each file in the snapshot's
/// `_MANIFEST.json` and of the manifest itself, keyed by relative path.
#[pyfunction]
fn snapshot_range_checksums(
    py: Python<'_>,
    output_dir: &str,
    range_bytes: u64,
) -> PyResult<std::collections::BTreeMap<String, Vec<u32>>> {
    let output_dir = std::path::PathBuf::from(output_dir);
    py.detach(|| {
        diskgraph_ingest::snapshot_mirror::snapshot_range_checksums(&output_dir, range_bytes)
    })
    .map_err(|e| DiskgraphCorruptionError::new_err(e.to_string()))
}

#[pyfunction]
fn has_resumable_ingest(output_dir: &str) -> PyResult<bool> {
    diskgraph_ingest::initial_resume_phase(std::path::Path::new(output_dir))
        .map(|phase| phase.is_some())
        .map_err(|e| DiskgraphCorruptionError::new_err(e.to_string()))
}

/// Return sorted durable Index shard keys for a bounded Kumo resume retry.
#[pyfunction]
fn validated_index_resume_progress(output_dir: &str) -> PyResult<Option<Vec<String>>> {
    diskgraph_ingest::validated_index_resume_progress(std::path::Path::new(output_dir))
        .map_err(|e| DiskgraphCorruptionError::new_err(e.to_string()))
}

#[pymodule]
#[pyo3(name = "_c")]
fn diskgraph_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = pyo3_log::try_init();
    let _ = tracing_log::LogTracer::init();
    // Cap otherwise-global rayon users (col stats, TT-gen, pass-B) to
    // cpu.max. Ignore the error only when another owner initialized the
    // process-global pool first; dedicated ingest pools are capped separately.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(diskgraph_ingest::effective_parallelism())
        .build_global();
    m.add("DiskgraphDataError", m.py().get_type::<DiskgraphDataError>())?;
    m.add(
        "DiskgraphSourceValidationError",
        m.py().get_type::<DiskgraphSourceValidationError>(),
    )?;
    m.add(
        "DiskgraphTargetCategoryLimitError",
        m.py().get_type::<DiskgraphTargetCategoryLimitError>(),
    )?;
    m.add("DiskgraphConfigError", m.py().get_type::<DiskgraphConfigError>())?;
    m.add("DiskgraphSystemError", m.py().get_type::<DiskgraphSystemError>())?;
    m.add(
        "DiskgraphCorruptionError",
        m.py().get_type::<DiskgraphCorruptionError>(),
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("BUILD_COMMIT", env!("DISKGRAPH_BUILD_COMMIT"))?;
    m.add_function(wrap_pyfunction!(export_tt_artifacts, m)?)?;
    m.add_function(wrap_pyfunction!(convert_kumo_config, m)?)?;
    m.add_function(wrap_pyfunction!(ingest, m)?)?;
    m.add_class::<ExportConfig>()?;
    m.add_class::<FkeyConfig>()?;
    m.add_class::<PyTimeUnit>()?;
    m.add_class::<DiskManager>()?;
    m.add_function(wrap_pyfunction!(estimate_ingest_size, m)?)?;
    m.add_function(wrap_pyfunction!(index_metadata::read_index_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(success_marker::read_success_marker, m)?)?;
    m.add_function(wrap_pyfunction!(success_marker::write_success_marker, m)?)?;
    m.add_function(wrap_pyfunction!(success_marker::touch_success_marker, m)?)?;
    m.add_function(wrap_pyfunction!(verify_snapshot_integrity, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_range_checksums, m)?)?;
    m.add_function(wrap_pyfunction!(has_resumable_ingest, m)?)?;
    m.add_function(wrap_pyfunction!(validated_index_resume_progress, m)?)?;
    m.add_function(wrap_pyfunction!(compute_col_stats_for_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(col_stats_export::read_col_stats, m)?)?;
    m.add_function(wrap_pyfunction!(col_stats_export::read_edge_stats, m)?)?;
    m.add_class::<SamplerEngine>()?;
    runtime_batch_iterator::register(m)?;
    m.add_class::<PySampleIterator>()?;
    m.add_function(wrap_pyfunction!(tt_builder::build_split_spec_json, m)?)?;
    m.add_class::<tt_builder::DiskGraphTTBuilder>()?;
    m.add_class::<tt_builder::EdgeRef>()?;
    m.add_class::<tt_builder::RowFilter>()?;
    m.add_class::<tt_builder::StageFilter>()?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_plan, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_gen, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_ingest, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_bp_plan, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::drop_tt, m)?)?;
    m.add_function(wrap_pyfunction!(
        tt_executor::read_tt_parquet_batches,
        m
    )?)?;
    m.add_class::<tt_executor::TTParquetBatchIterator>()?;
    m.add_function(wrap_pyfunction!(
        tt_executor::build_prediction_table_edge_bundle_artifact,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tt_executor::persist_prediction_artifact_bundle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        custom_tt_entry::build_train_table_artifacts_for_existing_table,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        custom_tt_entry::build_prediction_table_artifacts_for_existing_table,
        m
    )?)?;
    temporal::register(m)?;
    custom_tt_contract::register(m)?;
    Ok(())
}
