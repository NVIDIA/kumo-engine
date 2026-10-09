// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom-TT per-rayon-worker function.
//!
//! Mirrors `ttgen_executor::execute_chunk` but iterates **parquet row
//! groups** of a customer-supplied parquet file instead of an entity-table
//! range. Each worker:
//!
//! 1. Sets up a `ChannelBatchWriter`, `RowProcessor<U, D>`, alias scratch
//!    vec, link-frequencies vec, and local category counts.
//! 2. Opens the customer parquet for the row groups assigned to this
//!    worker, reads each `RecordBatch`, and:
//!    a. Looks up `entity_node = plan.entity_table.node_id_from_pk(...)`
//!    for the customer's `ENTITY1` column. If the lookup fails (entity
//!    PK not present in the index), the row is silently **skipped**
//!    (documented behavior — handles stale-customer-data without
//!    failing the whole job).
//!    b. Reads the customer's `TIMESTAMP` column as an anchor.
//!    c. Calls `populate_aliases_from_record` to copy any pre-computed
//!    columns (TARGET1, autoregression, yoy) from the customer's batch
//!    into the alias array.
//!    d. Calls `execute_stages(plan, ...)` — the SAME function the PQ path
//!    uses — to fill any aliases the parquet didn't provide. The plan
//!    is responsible for containing exactly the missing ops.
//!    e. Calls `processor.observe(row)` (stats + decider) and writes the
//!    row to the writer.
//! 3. Flushes local counts to shared and finalizes the writer.
//!
//! ## Alias-mapping limitation
//!
//! Mapping customer column names → plan alias slots is hard-coded for the
//! three standard scenarios (TARGET1, autoregression, yoy). A future
//! revision will replace this with a plan-compiler-emitted column-to-alias
//! mapping.

#![allow(clippy::too_many_arguments)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, Decimal128Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, LargeListArray, ListArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::DataType;
use crossbeam_channel::Sender;
use diskgraph_core::accumulators::{
    HyperLogLog, LocalColumnCounts, SharedColumnCounts, TDigestEstimator, WelfordAccumulator,
};
use diskgraph_core::time::{Abs, TimeUnit, UnitSystem};
use diskgraph_core::tt_builder::{PredictionType, Value};
use diskgraph_index::pkmap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

use crate::custom_tt_row_errors::{RowFailureKind, RowLocation};
use crate::errors::{TTGenError, TTGenResult};
use crate::split_decider::{value_to_string_key, SplitDecider};
use crate::ttgen_accumulator::hash_value;
use crate::ttgen_executor::CustomTtSource;
use crate::ttgen_output::{ChannelBatchWriter, ExtraColumnOutput, ExtraValueRef, OutputMode};
use crate::ttgen_resolve::{
    ResolvedPlan, AUTOREGRESSION_OUTPUT_ALIAS, FUTURE_LABELS_OUTPUT_ALIAS, YOY_OUTPUT_ALIAS,
};
use crate::ttgen_stats::{ChunkStats, RowProcessor, TTRowRef};

// ─── The custom-TT column contract ──────────────────────────────────────────
//
// The names this reader looks up and the Arrow types it reads, named once so
// that a caller validating a supplied training table before it reaches here
// checks against what this reader accepts rather than against a second
// spelling of it. Exported to Python as `diskgraph.CUSTOM_TT_CONTRACT`.
//
// Every list is enforced against the reader it describes by
// `the_contract_is_what_the_readers_accept`, which walks a corpus of Arrow
// types and asserts that acceptance is *exactly* membership. An arm added to
// a reader without an entry here fails that test, and so does an entry with
// no arm behind it. Each token is a `pyarrow.types.is_<token>` predicate.

/// Entity key columns are `ENTITY1`, `ENTITY2`, ... -- one per entity primary
/// key, looked up by position. `TIMESTAMP` is the anchor.
pub const ENTITY_COLUMN_PREFIX: &str = "ENTITY";
/// Target columns are `TARGET1`, `TARGET2`, ....
pub const TARGET_COLUMN_PREFIX: &str = "TARGET";
/// The per-row anchor. Optional: a static task omits it and this reader fills
/// the entity table's default instead.
pub const TIMESTAMP_COLUMN: &str = "TIMESTAMP";
/// The weight column this reader *emits*. What it reads is named by the
/// caller, not fixed.
pub const WEIGHT_OUTPUT_COLUMN: &str = "WEIGHT";

/// The `k`-th entity key column, by the name this reader looks it up under.
/// `slot` is 0-based; the column names are 1-based.
pub fn entity_column(slot: usize) -> String {
    format!("{ENTITY_COLUMN_PREFIX}{}", slot + 1)
}

/// The `k`-th target column, by the name this reader looks it up under.
pub fn target_column(slot: usize) -> String {
    format!("{TARGET_COLUMN_PREFIX}{}", slot + 1)
}

/// What an `ENTITY{k}` column may be, for *some* graph. Narrower than
/// [`SCALAR_TYPES`] because an entity value has to survive `value_to_pk_key`
/// as well as `cell_to_value`: a `Boolean` reads fine and then resolves to no
/// key, so every row of a boolean entity column is silently skipped.
///
/// **Necessary, not sufficient.** `value_to_pk_key` matches on the *pair*
/// (column type, the graph's own primary-key `KeyType`), and this list is the
/// union over every key type -- so a consumer that checks a column against
/// this list alone has established a necessary condition, not a sufficient
/// one. Concretely: a `string` column against an `Int64`-keyed graph resolves
/// nothing, an `int64` column against a `VarBytes`-keyed one resolves
/// nothing, and against a graph keyed by `Uuid`, `Uuid256` or `Ip` *no*
/// column type resolves at all. Each of those is the same silent total row
/// drop the `Boolean` note above describes: every row skipped, the run
/// reporting success over nothing.
pub const ENTITY_TYPES: &[&str] = &["int64", "float64", "string"];

/// What `cell_to_value` reads: a scalar `TARGET{k}`, and a split column.
pub const SCALAR_TYPES: &[&str] = &["int64", "float64", "string", "boolean"];

/// Containers accepted by the link-target reader.
pub const LIST_TARGET_CONTAINERS: &[&str] = &["list", "large_list"];

/// What may be inside a list-valued `TARGET{k}`.
pub const LIST_TARGET_ITEM_TYPES: &[&str] = &["int64", "float64", "string"];

/// What a `TIMESTAMP` column may be. A typed timestamp self-describes its
/// unit; an `int64` carries none and is read in the unit the caller declares.
pub const TIMESTAMP_TYPES: &[&str] = &["timestamp", "int64"];

/// What a scalar weight column may be, per `read_weight_f64`.
pub const WEIGHT_TYPES: &[&str] = &[
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "decimal128",
];

/// Containers accepted by the per-element weight reader.
pub const WEIGHT_LIST_CONTAINERS: &[&str] = &["list", "large_list"];

/// What may be inside a per-element weight, per `read_row_weight`. Narrower
/// than [`WEIGHT_TYPES`]: there is no `decimal128` arm in the list path.
pub const WEIGHT_LIST_ITEM_TYPES: &[&str] =
    &["int8", "int16", "int32", "int64", "float32", "float64"];

/// Why a weight could not be read. `Null` and `NonFinite` are row-level
/// conditions the caller applies its `on_error` policy to; `Unsupported` is
/// a column that can never be read and always fails the run.
#[derive(Debug)]
enum WeightFault {
    Null,
    NonFinite,
    Unsupported(String),
}

impl std::fmt::Display for WeightFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "custom-TT weight is null"),
            Self::NonFinite => write!(f, "custom-TT weight is not finite"),
            Self::Unsupported(message) => write!(f, "{message}"),
        }
    }
}

impl WeightFault {
    fn into_error(self) -> TTGenError {
        TTGenError::SchemaError(self.to_string())
    }
}

/// Read a declared custom-training weight without narrowing precision.
///
/// Databricks scalar numeric columns arrive through Arrow as one of these
/// signed integer, floating-point, or Decimal128 arrays. A declared weight is
/// required to be present, non-null, finite, and scalar; silently replacing a
/// bad value with `1.0` would change training semantics.
fn read_weight_f64(arr: &dyn Array, idx: usize) -> Result<f64, WeightFault> {
    if arr.is_null(idx) {
        return Err(WeightFault::Null);
    }

    let weight = if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        a.value(idx)
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        f64::from(a.value(idx))
    } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        a.value(idx) as f64
    } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        f64::from(a.value(idx))
    } else if let Some(a) = arr.as_any().downcast_ref::<Int16Array>() {
        f64::from(a.value(idx))
    } else if let Some(a) = arr.as_any().downcast_ref::<Int8Array>() {
        f64::from(a.value(idx))
    } else if let Some(a) = arr.as_any().downcast_ref::<Decimal128Array>() {
        let DataType::Decimal128(_, scale) = arr.data_type() else {
            unreachable!("Decimal128Array must report Decimal128 data type");
        };
        (a.value(idx) as f64) / 10_f64.powi(i32::from(*scale))
    } else {
        return Err(WeightFault::Unsupported(format!(
            "custom-TT weight must be a scalar numeric value; found Arrow type {}",
            arr.data_type()
        )));
    };

    if !weight.is_finite() {
        return Err(WeightFault::NonFinite);
    }
    Ok(weight)
}

/// Determine the `OutputMode` for a custom-TT job by peeking at the
/// customer parquet's schema. For Forecasting, this picks up
/// AUTOREGRESSION_COL and (optionally) YOY_COL dtypes from the actual
/// `autoregression` / `yoy` columns; for Node, it picks up the TARGETk
/// dtype from each `TARGETk` slot the plan declares. Falls back to
/// `Float64` defaults when columns are absent.
fn output_mode_for_custom_tt<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    custom_tt_parquet: &Path,
) -> TTGenResult<OutputMode> {
    let file = std::fs::File::open(custom_tt_parquet)?;
    let probe = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let arrow_schema = probe.schema().clone();
    drop(probe);

    let lookup = |name: &str| arrow_schema.field_with_name(name).ok().cloned();
    match plan.prediction_type {
        PredictionType::PassThrough => Ok(OutputMode::PassThrough),
        PredictionType::Forecasting => {
            let ar_dtype = lookup(AUTOREGRESSION_OUTPUT_ALIAS)
                .map(|f| f.data_type().clone())
                .unwrap_or_else(|| {
                    DataType::List(Arc::new(arrow_schema::Field::new(
                        "item",
                        DataType::Float64,
                        true,
                    )))
                });
            let yoy_dtype = lookup(YOY_OUTPUT_ALIAS).map(|f| f.data_type().clone());
            let future_labels_dtype =
                lookup(FUTURE_LABELS_OUTPUT_ALIAS).map(|f| f.data_type().clone());
            Ok(OutputMode::Forecasting {
                target_dtypes: if !plan.is_prediction {
                    plan.target_dtypes.clone()
                } else {
                    Vec::new()
                },
                autoregression_dtype: ar_dtype,
                future_labels_dtype,
                yoy_dtype,
            })
        }
        PredictionType::Node => {
            // A PREDICTION bundle carries NO target column — native predict
            // emits `[ENTITY1, TIMESTAMP]` only (its plan is PassThrough).
            // Empty target_dtypes ⟹ zero TARGETk columns (same predict
            // contract as the Forecasting arm above).
            let target_dtypes: Vec<DataType> = if plan.is_prediction {
                Vec::new()
            } else {
                // Read one TARGETk dtype per declared target slot, falling back
                // to the plan-resolved target_dtype for that slot when the
                // parquet doesn't carry it.
                (0..plan.target_alias_indices.len())
                    .map(|i| {
                        let parquet_name = target_column(i);
                        lookup(&parquet_name)
                            .map(|f| f.data_type().clone())
                            .unwrap_or_else(|| {
                                plan.target_dtypes
                                    .get(i)
                                    .cloned()
                                    .unwrap_or(DataType::Float64)
                            })
                    })
                    .collect()
            };
            Ok(OutputMode::Node { target_dtypes })
        }
        PredictionType::Link { include_frequency } => {
            // Predict LP seed tables carry no TARGET1 list column → no target
            // columns in the bundle (same predict contract as Node above).
            let target_dtypes: Vec<DataType> = if plan.is_prediction {
                Vec::new()
            } else {
                (0..plan.target_alias_indices.len())
                    .map(|i| {
                        let parquet_name = target_column(i);
                        let field = lookup(&parquet_name).ok_or_else(|| {
                            TTGenError::SchemaError(format!(
                                "custom-TT LP parquet missing required list target column '{parquet_name}'"
                            ))
                        })?;
                        match field.data_type() {
                            DataType::List(inner) | DataType::LargeList(inner) => Ok(inner.data_type().clone()),
                            other => Err(TTGenError::SchemaError(format!(
                                "custom-TT LP parquet column '{parquet_name}' must be a List, got {other:?}"
                            ))),
                        }
                    })
                    .collect::<TTGenResult<Vec<_>>>()?
            };
            Ok(OutputMode::LinkArray {
                include_frequency,
                target_dtypes,
            })
        }
    }
}

fn value_to_i64_opt(value: &Value) -> Option<i64> {
    match value {
        Value::Int(v) => Some(*v),
        Value::Float(v) => Some(*v as i64),
        Value::Bool(v) => Some(if *v { 1 } else { 0 }),
        Value::Str(_) | Value::List(_) => None,
    }
}

fn min_opt_i64(current: Option<i64>, value: i64) -> Option<i64> {
    match current {
        Some(cur) => Some(cur.min(value)),
        None => Some(value),
    }
}

fn max_opt_i64(current: Option<i64>, value: i64) -> Option<i64> {
    match current {
        Some(cur) => Some(cur.max(value)),
        None => Some(value),
    }
}

/// Convert a `Value` (the executor's tag-typed scalar) to a `Key` (the PK
/// index's typed key). Mirrors the private helper inside
/// `diskgraph-index/src/graph.rs::value_to_pk_key`. Used only on the
/// custom-TT slow path (one call per parquet row), not on the PQ hot path.
pub(crate) fn value_to_pk_key(value: &Value, key_type: pkmap::KeyType) -> Option<pkmap::Key> {
    match (value, key_type) {
        (Value::Int(i), pkmap::KeyType::Int64) => Some(pkmap::Key::Int(*i as u64)),
        (Value::Str(s), pkmap::KeyType::VarBytes) => Some(pkmap::Key::Bytes(s.as_bytes().to_vec())),
        (Value::Float(f), pkmap::KeyType::Int64) => Some(pkmap::Key::Int(*f as u64)),
        _ => None,
    }
}

/// Resolve a customer-supplied entity PK value to an `entity_node` id.
/// Returns `None` if the PK is not in the index — the caller must skip
/// that row.
fn lookup_entity_node<U: UnitSystem>(plan: &ResolvedPlan<U>, pk_value: &Value) -> Option<u64> {
    let pk_map = plan.entity_table.pk_map()?;
    let key = value_to_pk_key(pk_value, pk_map.key_type())?;
    plan.entity_table.node_id_from_pk(&key).ok().flatten()
}

/// Read one cell of an Arrow array as our `Value` enum. Supports the dtypes
/// we expect in customer parquets: Int64, Float64, Utf8 (entity PKs +
/// scalar targets).
pub(crate) fn cell_to_value(arr: &dyn Array, idx: usize) -> Option<Value> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(Value::Int(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Some(Value::Float(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Some(Value::Str(a.value(idx).to_string()));
    }
    if let Some(a) = arr.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        return Some(Value::Bool(a.value(idx)));
    }
    None
}

/// Read one cell of an Arrow timestamp column as epoch seconds.
///
/// Dispatches by the column's typed Arrow `DataType` — no inference of
/// units from value magnitude. Supported:
///   * `Timestamp(Nanosecond)`  → value / 1e9 (declared ignored; array self-describes)
///   * `Timestamp(Microsecond)` → value / 1e6 (declared ignored)
///   * `Timestamp(Millisecond)` → value / 1e3 (declared ignored)
///   * `Timestamp(Second)`      → value as-is (declared ignored)
///   * `Int64`                  → divide by `declared`-derived factor
///
/// For `Int64` (no inherent unit), `declared` is the bridge-supplied
/// `CustomTtSource::input_timestamp_unit`. Kumo's rehydration declares
/// this as `Nanoseconds` because kumo's canonical TIMESTAMP is int64 ns.
/// Non-sub-second variants (Minutes, Hours, ...) are not meaningful for
/// epoch timestamps and return `None`.
fn read_timestamp_secs(arr: &dyn Array, idx: usize, declared: TimeUnit) -> Option<i64> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Some(a.value(idx) / 1_000_000_000);
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Some(a.value(idx) / 1_000_000);
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Some(a.value(idx) / 1_000);
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampSecondArray>() {
        return Some(a.value(idx));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        let v = a.value(idx);
        return match declared {
            TimeUnit::Nanoseconds => Some(v / 1_000_000_000),
            TimeUnit::Microseconds => Some(v / 1_000_000),
            TimeUnit::Milliseconds => Some(v / 1_000),
            TimeUnit::Seconds => Some(v),
            // Non-sub-second units are not used for epoch timestamps in
            // kumo's data model; surface as None so the caller's "skip
            // null timestamp" path applies.
            _ => None,
        };
    }
    None
}

/// Read a list cell as `Value::List(Vec<Value>)`. Supports inner Float64,
/// Int64, and Utf8. Returns `None` if the cell is null or the array shape is
/// unsupported.
fn read_list_cell<O: arrow_array::OffsetSizeTrait>(
    arr: &arrow_array::GenericListArray<O>,
    idx: usize,
) -> Option<Value> {
    if arr.is_null(idx) {
        return None;
    }
    let list = arr.value(idx);
    let n = list.len();
    let mut out = Vec::with_capacity(n);
    if let Some(f) = list.as_any().downcast_ref::<Float64Array>() {
        for j in 0..n {
            if f.is_null(j) {
                out.push(Value::Float(f64::NAN));
            } else {
                out.push(Value::Float(f.value(j)));
            }
        }
        return Some(Value::List(out));
    }
    if let Some(i) = list.as_any().downcast_ref::<Int64Array>() {
        for j in 0..n {
            if i.is_null(j) {
                out.push(Value::Int(0));
            } else {
                out.push(Value::Int(i.value(j)));
            }
        }
        return Some(Value::List(out));
    }
    if let Some(s) = list.as_any().downcast_ref::<StringArray>() {
        for j in 0..n {
            if s.is_null(j) {
                out.push(Value::Str(String::new()));
            } else {
                out.push(Value::Str(s.value(j).to_string()));
            }
        }
        return Some(Value::List(out));
    }
    None
}

fn read_list_values(arr: &dyn Array, idx: usize) -> Option<Vec<Value>> {
    arr.as_any()
        .downcast_ref::<ListArray>()
        .and_then(|list| match read_list_cell(list, idx) {
            Some(Value::List(values)) => Some(values),
            _ => None,
        })
}

fn read_link_targets(arr: &dyn Array, row: usize) -> TTGenResult<Vec<Option<Value>>> {
    let values = arr
        .as_any()
        .downcast_ref::<ListArray>()
        .map(|list| list.value(row))
        .or_else(|| {
            arr.as_any()
                .downcast_ref::<LargeListArray>()
                .map(|list| list.value(row))
        })
        .ok_or_else(|| {
            TTGenError::SchemaError(format!(
                "custom-TT link target must be a list; found {}",
                arr.data_type()
            ))
        })?;
    if !matches!(
        values.data_type(),
        DataType::Int64 | DataType::Float64 | DataType::Utf8
    ) {
        return Err(TTGenError::SchemaError(format!(
            "unsupported custom-TT link target item type {}",
            values.data_type()
        )));
    }
    if arr.is_null(row) {
        return Ok(Vec::new());
    }
    Ok((0..values.len())
        .map(|idx| cell_to_value(values.as_ref(), idx))
        .collect())
}

fn read_i64_list_values(arr: &dyn Array, idx: usize) -> Option<Vec<i64>> {
    let values = read_list_values(arr, idx).or_else(|| {
        arr.as_any()
            .downcast_ref::<LargeListArray>()
            .and_then(|list| match read_list_cell(list, idx) {
                Some(Value::List(values)) => Some(values),
                _ => None,
            })
    });
    values.map(|values| {
        values
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v as i64,
                Value::Bool(v) => {
                    if v {
                        1
                    } else {
                        0
                    }
                }
                Value::Str(_) | Value::List(_) => 1,
            })
            .collect()
    })
}

/// Populate per-row alias slots from a customer-supplied `RecordBatch`,
/// using the hard-coded mapping:
///   * column `TARGET1` → `plan.primary_target_alias_idx()`
///   * column `autoregression` → `plan.autoregression_alias_idx`
///   * column `future_labels` → `plan.future_labels_alias_idx`
///   * column `yoy` → `plan.yoy_alias_idx`
///
/// Columns not present in the batch are left untouched (the plan's stages
/// are responsible for filling them via execute_stages).
///
/// A future revision will replace this with a plan-compiler-emitted
/// column-to-alias map; for v0 the column names are baked in.
#[allow(dead_code)] // Reserved for non-empty stages on custom-TT path
pub(crate) fn populate_aliases_from_record<U: UnitSystem>(
    batch: &RecordBatch,
    row_idx: usize,
    aliases: &mut [Option<Value>],
    plan: &ResolvedPlan<U>,
) {
    let schema = batch.schema();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let name = field.name().as_str();
        let primary_target = target_column(0);
        let target_slot = match name {
            n if n == primary_target => Some(plan.primary_target_alias_idx()),
            n if n == AUTOREGRESSION_OUTPUT_ALIAS => plan.autoregression_alias_idx,
            n if n == FUTURE_LABELS_OUTPUT_ALIAS => plan.future_labels_alias_idx,
            n if n == YOY_OUTPUT_ALIAS => plan.yoy_alias_idx,
            _ => None,
        };
        let Some(slot) = target_slot else { continue };
        if slot >= aliases.len() {
            // Plan doesn't have this alias slot allocated — skip (e.g.,
            // forecasting plan with no yoy_alias_idx).
            continue;
        }
        let arr = batch.column(col_idx);
        let value = match arr.data_type() {
            DataType::List(_) => arr
                .as_any()
                .downcast_ref::<ListArray>()
                .and_then(|l| read_list_cell(l, row_idx)),
            _ => cell_to_value(arr.as_ref(), row_idx),
        };
        aliases[slot] = value;
    }
}

/// Per-rayon-worker function for the custom-TT pass-A path.
///
/// Generic over `D: SplitDecider<U>` so per-row dispatch into the decider
/// inlines fully (matching the PQ `execute_chunk` shape).
///
/// Returns `(rows_written, processor)` — same shape as `execute_chunk`.
pub(crate) fn custom_tt_chunk_processor<U: UnitSystem, D: SplitDecider<U> + Clone>(
    plan: &ResolvedPlan<U>,
    custom_tt_parquet: &Path,
    row_groups: &[usize],
    sender: &Sender<crate::ttgen_output::BatchMessage>,
    decider_template: &D,
    shared_target_counts: Option<Arc<SharedColumnCounts>>,
    source: &CustomTtSource,
) -> TTGenResult<(u64, RowProcessor<U, D>)> {
    // Build OutputMode from the customer parquet's *schema* (not the
    // plan's). The plan declares the prediction kind; the parquet
    // declares which forecast columns it actually carries.
    let mode = output_mode_for_custom_tt::<U>(plan, custom_tt_parquet)?;
    // Read the actual target dtype the chunk processor will emit (resolved
    // from the customer parquet schema in `output_mode_for_custom_tt`),
    // not from `plan.primary_target_dtype()` — that helper falls through
    // to `Float64` for empty-stages custom-TT plans, which would lose the
    // Boolean signal needed for JSON-bool CATEGORY_COUNTS emission.
    let actual_target_is_bool = match &mode {
        OutputMode::Node { target_dtypes } => {
            matches!(target_dtypes.first(), Some(DataType::Boolean))
        }
        OutputMode::LinkArray { target_dtypes, .. } => {
            matches!(target_dtypes.first(), Some(DataType::Boolean))
        }
        OutputMode::PassThrough | OutputMode::Forecasting { .. } => false,
    };
    let time_col_name = plan
        .entity_table_spec
        .output_time_col_name
        .as_deref()
        .unwrap_or(TIMESTAMP_COLUMN);
    let mut writer = ChannelBatchWriter::new(
        sender.clone(),
        mode,
        plan.lag_timesteps > 0,
        plan.entity_dtypes.clone(),
        time_col_name,
        plan.split_projection.as_ref().map(|p| ExtraColumnOutput {
            name: p.output_name.clone(),
            dtype: p.dtype.clone(),
        }),
        Vec::new(),
    )?;
    // Enable per-row weight collection on the writer when the resolved
    // plan declared a weight source column. The chunk processor below
    // calls `writer.push_weight` once per emitted output row (so weight
    // count == writer's emitted row count, NOT input row count).
    if plan.weight_col.is_some() {
        writer.enable_weight_buf()?;
    }

    // Per-worker scratch.
    //
    // Custom-TT plans may have an empty stage list (every output column
    // comes from the parquet); in that case `plan.num_aliases == 0` but
    // we still need at least one slot to receive parquet-supplied values
    // (TARGET1 → slot 0 in the empty-stages case; AUTOREGRESSION_COL /
    // YOY_COL → their respective `Option<usize>` slots). Allocate at
    // least three slots to cover the standard scenarios.
    let n_alias_slots = plan
        .num_aliases
        .max(plan.autoregression_alias_idx.map(|i| i + 1).unwrap_or(0))
        .max(plan.future_labels_alias_idx.map(|i| i + 1).unwrap_or(0))
        .max(plan.yoy_alias_idx.map(|i| i + 1).unwrap_or(0))
        .max(1);
    let mut aliases: Vec<Option<Value>> = vec![None; n_alias_slots];
    let link_frequencies: Vec<i64> = Vec::new();
    let _link_split_values: Vec<Value> = Vec::new();

    // Stats setup mirrors execute_chunk verbatim.
    let target_stype = plan.target_stype.as_str();
    let target_is_numerical = target_stype == "numerical";
    let target_is_categorical = matches!(
        target_stype,
        "categorical" | "boolean" | "ID" | "multicategorical"
    );
    let include_frequency = matches!(
        plan.prediction_type,
        PredictionType::Link {
            include_frequency: true,
            ..
        }
    );
    let mut stats: ChunkStats<U> = ChunkStats {
        target_is_bool_dtype: actual_target_is_bool,
        ..ChunkStats::default()
    };
    if target_is_numerical {
        stats.target_welford = Some(WelfordAccumulator::new());
        stats.target_tdigest = Some(TDigestEstimator::new(200));
        stats.target_hll = Some(HyperLogLog::new());
    }
    if target_is_categorical {
        stats.target_hll = Some(HyperLogLog::new());
    }
    if include_frequency {
        stats.frequency_welford = Some(WelfordAccumulator::new());
        stats.frequency_tdigest = Some(TDigestEstimator::new(200));
        stats.frequency_hll = Some(HyperLogLog::new());
    }
    stats.entity_hll = Some(HyperLogLog::new());
    stats.entity_welford = Some(WelfordAccumulator::new());
    stats.entity_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_welford = Some(WelfordAccumulator::new());
    stats.timestamp_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_hll = Some(HyperLogLog::new());

    let mut processor: RowProcessor<U, D> = RowProcessor::new(stats, decider_template.clone());
    // Only allocate the per-worker target map when there is a shared accumulator
    // to flush into (mirrors `execute_chunk`): LP targets have no exact-vocabulary
    // consumer (`shared_target_counts` is None), so collecting them is dead work.
    // Entity category counts are never collected (the entity column is a graph
    // FK/pkey with no consumer); only its HLL cardinality is retained.
    let mut local_target_counts: Option<LocalColumnCounts> = shared_target_counts
        .is_some()
        .then(|| LocalColumnCounts::with_capacity(5000));

    // ── Read assigned row groups ─────────────────────────────────────────
    if !row_groups.is_empty() {
        let file = std::fs::File::open(custom_tt_parquet)?;
        let probe = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = probe.parquet_schema().clone();
        // We need ENTITY1, TIMESTAMP and any of {TARGET1, autoregression, yoy}
        // present in the file. Project all columns: customer parquets are
        // narrow (5-10 cols at the wide-format spec). Keeping it simple.
        let projection = ProjectionMask::all();
        // Reopen for the actual read: ParquetRecordBatchReaderBuilder
        // consumes self when building the reader.
        let file2 = std::fs::File::open(custom_tt_parquet)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file2)?
            .with_row_groups(row_groups.to_vec())
            .with_projection(projection)
            .build()?;
        // Ensure the parquet has every ENTITYk before iterating. (Cheap
        // precondition check that yields a clearer error than a mid-row
        // panic.) TIMESTAMP is NOT required: STATIC custom tasks omit it and
        // `process_batch` synthesizes the native-static sentinel per row.
        for i in 0..plan.entity_pk_col_indices.len() {
            let name = entity_column(i);
            parquet_schema
                .columns()
                .iter()
                .position(|c| c.name() == name)
                .ok_or_else(|| {
                    TTGenError::SchemaError(format!(
                        "custom-TT parquet missing required column '{name}'"
                    ))
                })?;
        }

        let part_name = custom_tt_parquet
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| custom_tt_parquet.display().to_string());
        let mut row_base = 0u64;
        for batch_res in reader {
            let batch = batch_res.map_err(|e| TTGenError::Parquet(format!("{e}")))?;
            let batch_rows = batch.num_rows() as u64;
            process_batch(
                plan,
                &batch,
                &mut aliases,
                &link_frequencies,
                target_is_numerical,
                target_is_categorical,
                target_stype,
                &mut local_target_counts,
                &mut writer,
                &mut processor,
                source,
                &part_name,
                row_groups[0],
                row_base,
            )?;
            row_base += batch_rows;
        }
    }

    // Flush local target counts to shared.
    if let (Some(ref shared), Some(ref mut local)) =
        (&shared_target_counts, &mut local_target_counts)
    {
        shared.merge_local_batch(local);
    }

    let rows = writer.finish()?;
    processor.stats.num_total_rows = rows;
    Ok((rows, processor))
}

/// One row's declared weight: a scalar, or one per target element.
enum RowWeight {
    None,
    Scalar(f64),
    List(Vec<f64>),
}

/// Read every weight element, collecting all content conditions for the row.
fn read_row_weight(
    arr: &dyn Array,
    idx: usize,
    per_element: bool,
    allow_negative: bool,
    column: &str,
    failures: &mut Vec<(RowFailureKind, String)>,
) -> TTGenResult<RowWeight> {
    let mut record = |kind| {
        if !failures
            .iter()
            .any(|(existing, name)| *existing == kind && name == column)
        {
            failures.push((kind, column.to_string()));
        }
    };
    let mut read = |values: &dyn Array, i: usize| -> TTGenResult<f64> {
        match read_weight_f64(values, i) {
            Ok(weight) => {
                if !allow_negative && weight < 0.0 {
                    record(RowFailureKind::NegativeWeight);
                }
                Ok(weight)
            }
            Err(WeightFault::Null) => {
                record(RowFailureKind::NullWeight);
                Ok(0.0)
            }
            Err(WeightFault::NonFinite) => {
                record(RowFailureKind::NonFiniteWeight);
                Ok(0.0)
            }
            Err(fault) => Err(fault.into_error()),
        }
    };
    if !per_element || arr.is_null(idx) {
        return read(arr, idx).map(RowWeight::Scalar);
    }
    let items = arr
        .as_any()
        .downcast_ref::<ListArray>()
        .map(|l| l.value(idx))
        .or_else(|| {
            arr.as_any()
                .downcast_ref::<LargeListArray>()
                .map(|l| l.value(idx))
        })
        .ok_or_else(|| {
            TTGenError::SchemaError(format!(
                "custom-TT weight must be a list beside an array target; found Arrow type {}",
                arr.data_type()
            ))
        })?;
    if !matches!(
        items.data_type(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
    ) {
        return Err(TTGenError::SchemaError(format!(
            "custom-TT weight elements must be numeric; found Arrow type {}",
            items.data_type()
        )));
    }
    (0..items.len())
        .map(|i| read(items.as_ref(), i))
        .collect::<TTGenResult<Vec<_>>>()
        .map(RowWeight::List)
}

/// The length of the list at `idx`, or `None` for a null slot or a column that
/// is not a list.
fn list_len(arr: &dyn Array, idx: usize) -> Option<usize> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(l) = arr.as_any().downcast_ref::<ListArray>() {
        let o = l.value_offsets();
        return Some((o[idx + 1] - o[idx]) as usize);
    }
    if let Some(l) = arr.as_any().downcast_ref::<LargeListArray>() {
        let o = l.value_offsets();
        return Some((o[idx + 1] - o[idx]) as usize);
    }
    None
}

/// Check every target used by the plan. Prediction rows carry no targets.
fn target_failures<U: UnitSystem>(
    batch: &RecordBatch,
    plan: &ResolvedPlan<U>,
    target_col_indices: &[Option<usize>],
    row_idx: usize,
    failures: &mut Vec<(RowFailureKind, String)>,
) {
    if plan.is_prediction || matches!(plan.prediction_type, PredictionType::PassThrough) {
        return;
    }
    let target_count = if matches!(plan.prediction_type, PredictionType::Node) {
        target_col_indices.len()
    } else {
        1
    };
    for (slot, column) in target_col_indices.iter().take(target_count).enumerate() {
        let Some(column) = column else {
            continue;
        };
        let arr = batch.column(*column);
        if arr.is_null(row_idx) {
            failures.push((RowFailureKind::NullTarget, target_column(slot)));
        } else if matches!(plan.prediction_type, PredictionType::Link { .. }) {
            let items = arr
                .as_any()
                .downcast_ref::<ListArray>()
                .map(|l| l.value(row_idx))
                .or_else(|| {
                    arr.as_any()
                        .downcast_ref::<LargeListArray>()
                        .map(|l| l.value(row_idx))
                });
            if let Some(items) = items {
                if items.null_count() > 0 {
                    failures.push((RowFailureKind::NullTargetItem, target_column(slot)));
                }
                if items.is_empty() {
                    failures.push((RowFailureKind::EmptyTargetList, target_column(slot)));
                }
            }
        }
    }
}

/// Whether the split cell at `row_idx` is one the split decider can route:
/// non-null, and -- when the plan names its split values -- one of them, as
/// `ColumnDeciderManual` renders it. A list cell (per-link split) is checked
/// per element.
fn split_failures(
    column: &dyn Array,
    row_idx: usize,
    allowed: Option<&std::collections::HashSet<String>>,
) -> Vec<RowFailureKind> {
    if column.is_null(row_idx) {
        return vec![RowFailureKind::NullSplit];
    }
    let items = column
        .as_any()
        .downcast_ref::<ListArray>()
        .map(|l| l.value(row_idx))
        .or_else(|| {
            column
                .as_any()
                .downcast_ref::<LargeListArray>()
                .map(|l| l.value(row_idx))
        });
    let unmatched = |arr: &dyn Array, idx: usize| {
        allowed.is_some_and(|keys| match cell_to_value(arr, idx) {
            Some(value) => !keys.contains(&value_to_string_key(&value)),
            None => true,
        })
    };
    let mut failures = Vec::new();
    match items {
        Some(items) => {
            if items.null_count() > 0 {
                failures.push(RowFailureKind::NullSplit);
            }
            if (0..items.len()).any(|i| !items.is_null(i) && unmatched(items.as_ref(), i)) {
                failures.push(RowFailureKind::InvalidSplit);
            }
        }
        None => {
            if unmatched(column, row_idx) {
                failures.push(RowFailureKind::InvalidSplit);
            }
        }
    }
    failures
}

/// Process one `RecordBatch` of customer-supplied rows. Per-row body:
/// look up PK, read TARGET1/autoregression/yoy directly from the batch,
/// write via the appropriate writer entry point, observe stats.
///
/// **No `execute_stages` call here** — in v0 we restrict the supported
/// plan shapes to those with **empty stage lists** (the plan compiler
/// emits the right shape per scenario; see design doc §5.1). Plans with
/// non-empty stages on the custom-TT path are deferred to a future
/// plan-compiler revision.
fn process_batch<U: UnitSystem, D: SplitDecider<U>>(
    plan: &ResolvedPlan<U>,
    batch: &RecordBatch,
    aliases: &mut [Option<Value>],
    link_frequencies: &[i64],
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
    writer: &mut ChannelBatchWriter<U>,
    processor: &mut RowProcessor<U, D>,
    source: &CustomTtSource,
    part_name: &str,
    row_group: usize,
    row_base: u64,
) -> TTGenResult<()> {
    let input_timestamp_unit = source.input_timestamp_unit;
    let on_error = source.on_error;
    let schema = batch.schema();
    processor
        .stats
        .source_rows
        .failures
        .add_rows_read(batch.num_rows() as u64);
    // Resolve one parquet column index per ENTITYk slot the plan declares.
    // Missing columns are a hard error: the plan asked for k entities but the
    // parquet doesn't carry them.
    let entity_col_indices: Vec<usize> = (0..plan.entity_pk_col_indices.len())
        .map(|i| {
            let name = entity_column(i);
            schema.index_of(&name).map_err(|_| {
                TTGenError::SchemaError(format!(
                    "custom-TT parquet missing required column '{name}'"
                ))
            })
        })
        .collect::<TTGenResult<Vec<_>>>()?;
    // STATIC custom tasks carry no TIMESTAMP column. Rather than force kumo to
    // rewrite a large customer parquet, DG synthesizes the same sentinel anchor
    // the native TTGen path uses for non-temporal tasks (the `!has_temporal`
    // branch in `ttgen_executor.rs`): `entity_table_spec.default_time_value`,
    // or raw 0 in the plan's unit when unset. Absence of TIMESTAMP ⟹
    // synthesize, for both train and predict; a present column is used as-is
    // (temporal tasks always carry a real TIMESTAMP, enforced by kumo).
    let ts_col_idx = schema.index_of(TIMESTAMP_COLUMN).ok();
    let ts_arr = ts_col_idx.map(|idx| batch.column(idx).clone());
    let sentinel_ts: Abs<U> = plan
        .entity_table_spec
        .default_time_value
        .unwrap_or_else(|| Abs::<U>::from_raw(0));

    // Scratch reused across rows for the custom-TT LP RHS-target orphan
    // filter's temporal-validity decode (only touched when
    // `plan.custom_target_pk_table` is Some).
    let mut target_pk_scratch = diskgraph_rowpack::RowPackDecodeScratch::default();

    let sequence_feature_indices: Vec<(usize, String)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| {
            let name = field.name().as_str();
            // The bare prefixes are the legacy singular spellings; a custom
            // table naming one is a feature column, not an entity or target.
            !matches!(
                name,
                "__kumo_internal_frequency_column__" | "ENTITY_INCLUDE_BIT"
            ) && name != ENTITY_COLUMN_PREFIX
                && name != TARGET_COLUMN_PREFIX
                && !name.starts_with(ENTITY_COLUMN_PREFIX)
                && !name.starts_with(TARGET_COLUMN_PREFIX)
                && name != TIMESTAMP_COLUMN
                && name != AUTOREGRESSION_OUTPUT_ALIAS
                && name != FUTURE_LABELS_OUTPUT_ALIAS
                && name != YOY_OUTPUT_ALIAS
                && matches!(field.data_type(), arrow_schema::DataType::List(_))
        })
        .map(|(idx, field)| (idx, field.name().clone()))
        .collect();

    // Pre-resolve per-slot TARGETk / autoregression / yoy column indices for
    // direct batch reads (avoids per-row name lookups). For Node prediction
    // we read one TARGETk per declared target slot; missing TARGETk falls
    // back to NaN (matches the previous behavior for slot 1).
    let target_col_indices: Vec<Option<usize>> = (0..plan.target_alias_indices.len())
        .map(|i| schema.index_of(&target_column(i)).ok())
        .collect();
    let frequency_col_idx = schema.index_of("__kumo_internal_frequency_column__").ok();
    let autoreg_col_idx = schema.index_of(AUTOREGRESSION_OUTPUT_ALIAS).ok();
    let future_labels_col_idx = schema.index_of(FUTURE_LABELS_OUTPUT_ALIAS).ok();
    let yoy_col_idx = schema.index_of(YOY_OUTPUT_ALIAS).ok();

    // A declared weight is part of the training contract. Fail instead of
    // silently changing training semantics when the projected column is absent.
    let weight_col_idx: Option<usize> = plan
        .weight_col
        .as_deref()
        .map(|name| {
            schema.index_of(name).map_err(|_| {
                TTGenError::SchemaError(format!(
                    "custom-TT parquet declares weight column '{name}' but it is missing"
                ))
            })
        })
        .transpose()?;

    // Pre-resolve the custom-TT split column (e.g. ManualSplit on a
    // customer-provided `split_col`). When the plan declares a split projection
    // (entity_col_idx is None on the custom-TT path — the column lives in this
    // parquet, read by NAME), find that column here. Each row's value is
    // emitted as the split projection column into tt.parquet, where pass B's
    // `ColumnDeciderManual` routes by it and DROPS it from the seed output.
    // Required (hard error) when a split projection is declared but the column
    // is absent: the customer promised a split but didn't supply it.
    let split_col_idx: Option<usize> = match plan.split_projection.as_ref() {
        Some(p) => Some(schema.index_of(&p.output_name).map_err(|_| {
            TTGenError::SchemaError(format!(
                "custom-TT parquet declares split column '{}' but it is missing",
                p.output_name
            ))
        })?),
        None => None,
    };
    // Read a row's split-column value as a `Value` (used both as the projected
    // extra value and as the stats `split_column_value`).
    let read_split_value = |row_idx: usize| -> TTGenResult<Option<Value>> {
        let Some(column) = split_col_idx.map(|c| batch.column(c).as_ref()) else {
            return Ok(None);
        };
        let values = column
            .as_any()
            .downcast_ref::<ListArray>()
            .map(|list| list.value(row_idx))
            .or_else(|| {
                column
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .map(|list| list.value(row_idx))
            });
        let Some(values) = values else {
            return Ok(cell_to_value(column, row_idx));
        };
        if column.is_null(row_idx) || values.null_count() > 0 {
            return Err(TTGenError::SchemaError(format!(
                "custom-TT split list contains null at source row {row_idx}"
            )));
        }
        let values = (0..values.len())
            .map(|idx| {
                cell_to_value(values.as_ref(), idx).ok_or_else(|| {
                    TTGenError::SchemaError(format!(
                        "unsupported custom-TT split list type {:?}",
                        values.data_type()
                    ))
                })
            })
            .collect::<TTGenResult<Vec<_>>>()?;
        Ok(Some(Value::List(values)))
    };

    let per_element_weight = writer.weight_is_per_element();
    let split_name: &str = plan
        .split_projection
        .as_ref()
        .map(|p| p.output_name.as_str())
        .unwrap_or_default();
    let weight_name: &str = plan.weight_col.as_deref().unwrap_or_default();

    for row_idx in 0..batch.num_rows() {
        // Include all source rows, even those skipped later, in the reference comparison.
        if let Some(arr) = ts_arr
            .as_ref()
            .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
        {
            if !arr.is_null(row_idx) {
                processor
                    .stats
                    .source_rows
                    .observe_integer_timestamp(arr.value(row_idx));
            }
        }
        // Reset alias slots for this row (paranoia — keeps the alias array
        // in a clean state for any future plan that might run
        // execute_stages on the custom-TT path).
        aliases.iter_mut().for_each(|a| *a = None);

        // Read all ENTITYk PKs from the parquet. ENTITY1 (slot 0) is the
        // primary key used to look up the entity node in the index; ENTITY2+
        // are auxiliary join keys carried through to the output unchanged.
        let mut failures = Vec::new();
        let mut entity_pks: Vec<Value> = Vec::with_capacity(entity_col_indices.len());
        for (slot, &col) in entity_col_indices.iter().enumerate() {
            match cell_to_value(batch.column(col).as_ref(), row_idx) {
                Some(value) => entity_pks.push(value),
                None if batch.column(col).is_null(row_idx) => {
                    failures.push((RowFailureKind::NullEntity, entity_column(slot)));
                }
                None => {
                    return Err(TTGenError::SchemaError(format!(
                        "custom-TT entity column {:?} cannot be read",
                        entity_column(slot)
                    )))
                }
            }
        }
        let ts = match &ts_arr {
            Some(arr) if arr.is_null(row_idx) => {
                failures.push((RowFailureKind::NullTimestamp, TIMESTAMP_COLUMN.to_string()));
                sentinel_ts
            }
            Some(arr) => {
                let ts_secs = read_timestamp_secs(arr.as_ref(), row_idx, input_timestamp_unit)
                    .ok_or_else(|| {
                        TTGenError::SchemaError("custom-TT timestamp cannot be read".to_string())
                    })?;
                Abs::<U>::from_secs(ts_secs)
            }
            None => sentinel_ts,
        };
        target_failures(batch, plan, &target_col_indices, row_idx, &mut failures);
        let mut row_weight = RowWeight::None;
        if let Some(c) = weight_col_idx {
            row_weight = read_row_weight(
                batch.column(c).as_ref(),
                row_idx,
                matches!(plan.prediction_type, PredictionType::Link { .. }) && per_element_weight,
                source.allow_negative_weight,
                weight_name,
                &mut failures,
            )?;
            if let RowWeight::List(weights) = &row_weight {
                let targets_len = target_col_indices
                    .first()
                    .copied()
                    .flatten()
                    .and_then(|t| list_len(batch.column(t).as_ref(), row_idx));
                if targets_len.is_some_and(|len| weights.len() != len) {
                    failures.push((
                        RowFailureKind::WeightLengthMismatch,
                        weight_name.to_string(),
                    ));
                }
            }
        }
        if let Some(c) = split_col_idx {
            for kind in split_failures(
                batch.column(c).as_ref(),
                row_idx,
                plan.manual_split_keys.as_deref(),
            ) {
                failures.push((kind, split_name.to_string()));
            }
        }
        if !failures.is_empty() {
            processor.stats.source_rows.failures.reject_row(
                on_error,
                &failures,
                RowLocation {
                    part: part_name,
                    row_group,
                    row: row_base + row_idx as u64,
                },
            )?;
            continue;
        }

        // Graph lookup is a filter, applied only after content validation.
        let mut record_filter = |kind, column: String| {
            processor.stats.source_rows.failures.reject_row(
                on_error,
                &[(kind, column)],
                RowLocation {
                    part: part_name,
                    row_group,
                    row: row_base + row_idx as u64,
                },
            )
        };
        let entity_pk = entity_pks[0].clone();
        let Some(entity_node) = lookup_entity_node(plan, &entity_pk) else {
            record_filter(RowFailureKind::UnknownEntity, entity_column(0))?;
            continue;
        };
        if !plan.entity_table.is_canonical_node(entity_node, &entity_pk) {
            record_filter(RowFailureKind::NonCanonicalEntity, entity_column(0))?;
            continue;
        }
        let ts_secs = ts.to_secs();

        for (col_idx, name) in &sequence_feature_indices {
            let entry = processor
                .stats
                .sequence_feature_stats
                .entry(name.clone())
                .or_default();
            entry.observe(batch.column(*col_idx).as_ref(), row_idx);
        }

        // Per-prediction-type direct read + write.
        match &plan.prediction_type {
            PredictionType::PassThrough => {
                // Push the row's weight BEFORE write_row_pass_through_wide
                // so the writer's flush_batch (which may fire inside
                // write_row when BATCH_SIZE is reached) sees a weight
                // buffer whose length matches the about-to-be-flushed
                // batch's row count.
                if let RowWeight::Scalar(weight) = row_weight {
                    processor.stats.source_rows.observe_weight(weight);
                    writer.push_weight(weight);
                }
                let split_value = read_split_value(row_idx)?;
                writer.write_row_pass_through_wide(
                    &entity_pks,
                    ts,
                    ExtraValueRef::Scalar(split_value.as_ref()),
                    /*entity_include_bit=*/ true,
                )?;
                let row = TTRowRef {
                    entity_pk: &entity_pk,
                    ts,
                    target: &Value::Int(0),
                    frequency: None,
                    target_is_numerical,
                    target_is_categorical,
                    target_stype,
                    // Custom-TT split column (e.g. ManualSplit on `split_col`).
                    split_column_value: split_value.as_ref(),
                };
                processor.observe(&row, local_target_counts);
                processor.stats.include_true_count += 1;
                processor.stats.num_train_rows += 1;
            }
            PredictionType::Forecasting => {
                // Read autoregression list directly from the parquet. If
                // the customer didn't supply this column, we emit an
                // empty list (the writer keeps the row alive but with a
                // zero-length list; downstream may reject — caller
                // responsibility).

                let autoregression: Vec<Value> = match autoreg_col_idx {
                    Some(c) => batch
                        .column(c)
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .and_then(|l| match read_list_cell(l, row_idx) {
                            Some(Value::List(v)) => Some(v),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    None => Vec::new(),
                };

                let future_labels: Option<Vec<Value>> = match future_labels_col_idx {
                    Some(c) => batch
                        .column(c)
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .and_then(|l| match read_list_cell(l, row_idx) {
                            Some(Value::List(v)) => Some(v),
                            _ => None,
                        }),
                    None => None,
                };

                let yoy: Option<Vec<Value>> = match yoy_col_idx {
                    Some(c) => batch
                        .column(c)
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .and_then(|l| match read_list_cell(l, row_idx) {
                            Some(Value::List(v)) => Some(v),
                            _ => None,
                        }),
                    None => None,
                };
                let target_val = if !plan.is_prediction {
                    let val = target_col_indices
                        .first()
                        .copied()
                        .flatten()
                        .and_then(|c| cell_to_value(batch.column(c).as_ref(), row_idx))
                        .unwrap_or(Value::Float(f64::NAN));
                    Some(val)
                } else {
                    None
                };

                if let RowWeight::Scalar(weight) = row_weight {
                    processor.stats.source_rows.observe_weight(weight);
                    writer.push_weight(weight);
                }
                let split_value = read_split_value(row_idx)?;
                writer.write_row_forecasting_wide(
                    &entity_pks,
                    ts,
                    target_val.as_ref(),
                    &[],
                    autoregression,
                    future_labels,
                    yoy,
                    ExtraValueRef::Scalar(split_value.as_ref()),
                    true,
                )?;
                if !plan.is_prediction {
                    if let Some(tgt) = target_val.as_ref() {
                        let row = TTRowRef {
                            entity_pk: &entity_pk,
                            ts,
                            target: tgt,
                            frequency: None,
                            target_is_numerical,
                            target_is_categorical,
                            target_stype,
                            split_column_value: split_value.as_ref(),
                        };
                        processor.observe(&row, local_target_counts);
                    }
                }
                processor.stats.include_true_count += 1;
                processor.stats.num_train_rows += 1;
            }
            PredictionType::Node => {
                // A PREDICTION bundle carries NO target (native predict emits
                // `[ENTITY1, TIMESTAMP]` only). For predict, emit an empty
                // target set so `write_row_wide` writes no TARGETk columns.
                // TRAIN reads each TARGETk; missing → NaN (preserves the prior
                // single-target fallback behavior).
                let targets: Vec<Value> = if plan.is_prediction {
                    Vec::new()
                } else {
                    target_col_indices
                        .iter()
                        .map(|maybe_col| match maybe_col {
                            Some(c) => cell_to_value(batch.column(*c).as_ref(), row_idx)
                                .unwrap_or(Value::Float(f64::NAN)),
                            None => Value::Float(f64::NAN),
                        })
                        .collect()
                };

                // Custom-TT LP RHS-target orphan filter: when the plan
                // declares a `custom_target_pk_table`, TARGET1 is an FK whose
                // value was read straight from the customer parquet. Drop the
                // row if that value isn't a member of the target table's PK
                // map (or isn't temporally valid at the anchor ts). Mirrors
                // the PQL ListDistinct RHS-target filter
                // (`ttgen_executor.rs:2836-2849`) — without it, orphan RHS
                // ids silently mis-resolve at ingest (the external-parent MPH
                // does not verify). Train-only: predict carries no target.
                if let Some(pk_table) = plan.custom_target_pk_table.as_ref() {
                    if !targets.is_empty() {
                        let primary = &targets[0];
                        let keep = if plan.custom_target_temporal_create_col_idx.is_some()
                            || plan.custom_target_temporal_end_col_idx.is_some()
                        {
                            pk_table.contains_pk_value_temporal(
                                primary,
                                ts_secs,
                                plan.custom_target_temporal_create_col_idx,
                                plan.custom_target_temporal_end_col_idx,
                                &mut target_pk_scratch,
                            )
                        } else {
                            pk_table.contains_pk_value(primary)
                        };
                        if !keep {
                            processor.stats.source_rows.failures.reject_row(
                                on_error,
                                &[(RowFailureKind::UnavailableTarget, target_column(0))],
                                RowLocation {
                                    part: part_name,
                                    row_group,
                                    row: row_base + row_idx as u64,
                                },
                            )?;
                            continue;
                        }
                    }
                }

                if let RowWeight::Scalar(weight) = row_weight {
                    processor.stats.source_rows.observe_weight(weight);
                    writer.push_weight(weight);
                }
                let split_value = read_split_value(row_idx)?;
                writer.write_row_wide(
                    &entity_pks,
                    ts,
                    &targets,
                    ExtraValueRef::Scalar(split_value.as_ref()),
                    true,
                )?;
                // Stats observation keys off the primary (slot 0) target
                // (train only — predict carries no target).
                if !plan.is_prediction {
                    let primary_target = &targets[0];
                    let row = TTRowRef {
                        entity_pk: &entity_pk,
                        ts,
                        target: primary_target,
                        frequency: None,
                        target_is_numerical,
                        target_is_categorical,
                        target_stype,
                        // Custom-TT split column (e.g. ManualSplit on
                        // `split_col`): the per-row value drives split-stats.
                        split_column_value: split_value.as_ref(),
                    };
                    processor.observe(&row, local_target_counts);
                }
                processor.stats.include_true_count += 1;
                processor.stats.num_train_rows += 1;
            }
            PredictionType::Link {
                include_frequency, ..
            } => {
                // A PREDICTION seed table has NO TARGET1 column (just LHS
                // entities + TIMESTAMP), so predict writes an empty-target
                // seed row. Only TRAIN reads/requires TARGET1.
                let (targets, mut freqs): (Vec<Option<Value>>, Vec<i64>) = if plan.is_prediction {
                    (Vec::new(), Vec::new())
                } else {
                    let target_col =
                        target_col_indices
                            .first()
                            .copied()
                            .flatten()
                            .ok_or_else(|| {
                                TTGenError::SchemaError(
                                    "custom-TT LP parquet missing required column 'TARGET1'"
                                        .to_string(),
                                )
                            })?;
                    let targets = read_link_targets(batch.column(target_col).as_ref(), row_idx)?;

                    let mut freqs: Vec<i64> = if *include_frequency {
                        match frequency_col_idx {
                            Some(c) => read_i64_list_values(batch.column(c).as_ref(), row_idx)
                                .ok_or_else(|| {
                                    TTGenError::SchemaError(format!(
                                        "cannot read custom-TT frequencies at source row {row_idx}"
                                    ))
                                })?,
                            None => vec![1; targets.len()],
                        }
                    } else {
                        Vec::new()
                    };
                    if *include_frequency && freqs.len() != targets.len() {
                        freqs.resize(targets.len(), 1);
                    }
                    (targets, freqs)
                };

                let mut split_value = read_split_value(row_idx)?;
                if let Some(Value::List(values)) = &split_value {
                    if values.len() != targets.len() {
                        return Err(TTGenError::SchemaError(format!(
                            "custom-TT split list has {} entries but the target has {} at source row {row_idx}",
                            values.len(), targets.len()
                        )));
                    }
                }
                let mut weights: Vec<f64> = match &mut row_weight {
                    RowWeight::List(weights) => std::mem::take(weights),
                    _ => Vec::new(),
                };

                let split_values = match &split_value {
                    Some(Value::List(values)) => Some(values),
                    _ => None,
                };
                let mut filtered_splits = Vec::new();
                let mut filtered_targets = Vec::with_capacity(targets.len());
                let mut filtered_weights = Vec::with_capacity(weights.len());
                let mut filtered_freqs = Vec::with_capacity(freqs.len());
                for (idx, target) in targets.into_iter().enumerate() {
                    let Some(target) = target else { continue };
                    let keep = match plan.custom_target_pk_table.as_ref() {
                        Some(pk_table)
                            if plan.custom_target_temporal_create_col_idx.is_some()
                                || plan.custom_target_temporal_end_col_idx.is_some() =>
                        {
                            pk_table.contains_pk_value_temporal(
                                &target,
                                ts_secs,
                                plan.custom_target_temporal_create_col_idx,
                                plan.custom_target_temporal_end_col_idx,
                                &mut target_pk_scratch,
                            )
                        }
                        Some(pk_table) => pk_table.contains_pk_value(&target),
                        None => true,
                    };
                    if keep {
                        filtered_targets.push(target);
                        if let Some(values) = split_values {
                            filtered_splits.push(values[idx].clone());
                        }
                        if *include_frequency {
                            filtered_freqs.push(freqs.get(idx).copied().unwrap_or(1));
                        }
                        if per_element_weight {
                            filtered_weights.push(weights[idx]);
                        }
                    }
                }
                if split_values.is_some() {
                    split_value = Some(Value::List(filtered_splits));
                }
                let targets = filtered_targets;
                freqs = filtered_freqs;
                weights = filtered_weights;
                if !plan.is_prediction && targets.is_empty() {
                    processor.stats.source_rows.failures.reject_row(
                        on_error,
                        &[(RowFailureKind::UnavailableTarget, target_column(0))],
                        RowLocation {
                            part: part_name,
                            row_group,
                            row: row_base + row_idx as u64,
                        },
                    )?;
                    continue;
                }

                if weight_col_idx.is_some() {
                    if per_element_weight {
                        for &weight in &weights {
                            processor.stats.source_rows.observe_weight(weight);
                        }
                        writer.push_weight_list(weights);
                    } else if let RowWeight::Scalar(weight) = row_weight {
                        processor.stats.source_rows.observe_weight(weight);
                        writer.push_weight(weight);
                    }
                }
                writer.write_row_array(
                    entity_pk.clone(),
                    ts,
                    targets.clone(),
                    &[],
                    &freqs,
                    ExtraValueRef::Scalar(split_value.as_ref()),
                    true,
                )?;

                if let Some(first_target) = targets.first() {
                    let first_freq = if *include_frequency {
                        freqs.first().copied()
                    } else {
                        None
                    };
                    if !plan.is_prediction {
                        let row = TTRowRef {
                            entity_pk: &entity_pk,
                            ts,
                            target: first_target,
                            frequency: first_freq,
                            target_is_numerical,
                            target_is_categorical,
                            target_stype,
                            split_column_value: split_value.as_ref(),
                        };
                        processor.observe(&row, local_target_counts);
                        for (idx, tgt) in targets.iter().enumerate().skip(1) {
                            let freq = if *include_frequency {
                                freqs.get(idx).copied()
                            } else {
                                None
                            };
                            if let Some(ref mut h) = processor.stats.target_hll {
                                h.observe_i64(hash_value(tgt) as i64);
                            }
                            if let Some(v) = value_to_i64_opt(tgt) {
                                processor.stats.target_min =
                                    min_opt_i64(processor.stats.target_min, v);
                                processor.stats.target_max =
                                    max_opt_i64(processor.stats.target_max, v);
                            }
                            if let Some(freq) = freq {
                                processor.stats.frequency_min =
                                    min_opt_i64(processor.stats.frequency_min, freq);
                                processor.stats.frequency_max =
                                    max_opt_i64(processor.stats.frequency_max, freq);
                            }
                        }
                    }
                } else if !plan.is_prediction {
                    if let Some(ref mut h) = processor.stats.entity_hll {
                        h.observe_i64(hash_value(&entity_pk) as i64);
                    }
                    if let Some(ref mut h) = processor.stats.timestamp_hll {
                        h.observe_i64(ts.raw());
                    }
                    processor.stats.timestamp_min = Some(match processor.stats.timestamp_min {
                        Some(existing) => existing.min(ts),
                        None => ts,
                    });
                    processor.stats.timestamp_max = Some(match processor.stats.timestamp_max {
                        Some(existing) => existing.max(ts),
                        None => ts,
                    });
                    if let Some(v) = value_to_i64_opt(&entity_pk) {
                        processor.stats.entity_min = min_opt_i64(processor.stats.entity_min, v);
                        processor.stats.entity_max = max_opt_i64(processor.stats.entity_max, v);
                    }
                }

                processor.stats.include_true_count += 1;
                processor.stats.num_links += targets.len() as u64;
                processor.stats.num_train_rows += 1;
            }
        }
    }
    let _ = link_frequencies; // unused while custom-TT reads compact LP lists directly
    let _ = aliases; // unused while custom-TT path supports empty stages only
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use arrow_array::builder::{
        Decimal128Builder, FixedSizeListBuilder, Float32Builder, Float64Builder, Int16Builder,
        Int32Builder, Int64Builder, Int8Builder, LargeListBuilder, ListBuilder, StringBuilder,
        UInt64Builder,
    };
    use arrow_array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, DictionaryArray, LargeStringArray,
        UInt64Array,
    };
    use std::sync::Arc;

    /// One single-row array per Arrow type worth asking about, tagged with the
    /// contract token it answers to -- or `None` for a type the contract names
    /// nowhere, which every reader must therefore refuse.
    ///
    /// This corpus is what makes the constants above *derived* rather than
    /// restated: the assertions below are `accepted == listed` in both
    /// directions, so an arm added to a reader without a matching entry fails
    /// here, and so does an entry with no arm behind it.
    fn scalar_corpus() -> Vec<(Option<&'static str>, ArrayRef)> {
        let dict_keys = Int32Array::from(vec![0]);
        let dict_values = Arc::new(StringArray::from(vec!["a"])) as ArrayRef;
        vec![
            (Some("int8"), Arc::new(Int8Array::from(vec![1i8]))),
            (Some("int16"), Arc::new(Int16Array::from(vec![1i16]))),
            (Some("int32"), Arc::new(Int32Array::from(vec![1i32]))),
            (Some("int64"), Arc::new(Int64Array::from(vec![1i64]))),
            (Some("float32"), Arc::new(Float32Array::from(vec![1.0f32]))),
            (Some("float64"), Arc::new(Float64Array::from(vec![1.0f64]))),
            (Some("string"), Arc::new(StringArray::from(vec!["k"]))),
            (Some("boolean"), Arc::new(BooleanArray::from(vec![true]))),
            (
                Some("decimal128"),
                Arc::new(
                    Decimal128Array::from(vec![100i128])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ),
            (
                Some("timestamp"),
                Arc::new(TimestampNanosecondArray::from(vec![
                    1_700_000_000_000_000_000i64,
                ])),
            ),
            (
                Some("timestamp"),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    1_700_000_000_000_000i64,
                ])),
            ),
            (
                Some("timestamp"),
                Arc::new(TimestampMillisecondArray::from(vec![1_700_000_000_000i64])),
            ),
            (
                Some("timestamp"),
                Arc::new(TimestampSecondArray::from(vec![1_700_000_000i64])),
            ),
            // Time-zone-aware: the zone lives in the `DataType`, while the
            // reader downcasts on the Rust array type, so this has to behave
            // exactly as the naive one does. Asserted rather than assumed
            // because the Python side decodes this token as
            // `pa.types.is_timestamp`, which is true for tz-aware too.
            (
                Some("timestamp"),
                Arc::new(
                    TimestampNanosecondArray::from(vec![1_700_000_000_000_000_000i64])
                        .with_timezone("Etc/UTC"),
                ),
            ),
            // Named nowhere in the contract, and so refused everywhere.
            (None, Arc::new(UInt64Array::from(vec![1u64]))),
            (None, Arc::new(LargeStringArray::from(vec!["k"]))),
            (None, Arc::new(Date32Array::from(vec![1i32]))),
            (None, Arc::new(BinaryArray::from(vec![&b"k"[..]]))),
            (
                None,
                Arc::new(
                    DictionaryArray::<arrow_array::types::Int32Type>::try_new(
                        dict_keys,
                        dict_values,
                    )
                    .unwrap(),
                ),
            ),
        ]
    }

    #[test]
    fn a_scalar_target_reads_exactly_the_contract_types() {
        for (token, array) in scalar_corpus() {
            let listed = token.is_some_and(|t| SCALAR_TYPES.contains(&t));
            assert_eq!(
                cell_to_value(array.as_ref(), 0).is_some(),
                listed,
                "SCALAR_TYPES disagrees with cell_to_value for {}",
                array.data_type()
            );
        }
    }

    /// Every `KeyType` a graph's primary key can have, paired with the
    /// contract's snake-case name for it. Exhaustive on purpose: the three
    /// that resolve nothing are the reason the pairing is exported.
    fn all_key_types() -> Vec<(&'static str, pkmap::KeyType)> {
        let all = vec![
            ("int64", pkmap::KeyType::Int64),
            ("var_bytes", pkmap::KeyType::VarBytes),
            ("uuid", pkmap::KeyType::Uuid),
            ("uuid256", pkmap::KeyType::Uuid256),
            ("ip", pkmap::KeyType::Ip),
        ];
        // Compile-time exhaustiveness: adding a variant to `KeyType` without
        // adding it here fails to compile rather than silently narrowing the
        // corpus.
        for (_, kt) in &all {
            match kt {
                pkmap::KeyType::Int64
                | pkmap::KeyType::VarBytes
                | pkmap::KeyType::Uuid
                | pkmap::KeyType::Uuid256
                | pkmap::KeyType::Ip => {}
            }
        }
        all
    }

    #[test]
    fn an_entity_key_reads_exactly_the_contract_types() {
        // An entity value has to survive both reads: `cell_to_value` and then
        // `value_to_pk_key` against the index's own key type. `ENTITY_TYPES`
        // is the union over every key type, so it is asserted as such.
        for (token, array) in scalar_corpus() {
            let listed = token.is_some_and(|t| ENTITY_TYPES.contains(&t));
            let resolves = cell_to_value(array.as_ref(), 0).is_some_and(|value| {
                all_key_types()
                    .iter()
                    .any(|(_, kt)| value_to_pk_key(&value, *kt).is_some())
            });
            assert_eq!(
                resolves,
                listed,
                "ENTITY_TYPES disagrees with value_to_pk_key for {}",
                array.data_type()
            );
        }
    }

    #[test]
    fn the_entity_union_hides_which_key_types_accept_nothing() {
        // What the `ENTITY_TYPES` doc comment warns about, asserted so the
        // warning cannot go stale: acceptance is a property of the pair, and
        // three of the five key types accept no entity column at all.
        let accepting: Vec<&str> = all_key_types()
            .into_iter()
            .filter(|(_, key_type)| {
                scalar_corpus().iter().any(|(_, array)| {
                    cell_to_value(array.as_ref(), 0)
                        .is_some_and(|value| value_to_pk_key(&value, *key_type).is_some())
                })
            })
            .map(|(name, _)| name)
            .collect();
        assert_eq!(accepting, vec!["int64", "var_bytes"]);
    }

    #[test]
    fn an_anchor_reads_exactly_the_contract_types() {
        for (token, array) in scalar_corpus() {
            let listed = token.is_some_and(|t| TIMESTAMP_TYPES.contains(&t));
            assert_eq!(
                read_timestamp_secs(array.as_ref(), 0, TimeUnit::Nanoseconds).is_some(),
                listed,
                "TIMESTAMP_TYPES disagrees with read_timestamp_secs for {}",
                array.data_type()
            );
        }
    }

    #[test]
    fn a_scalar_weight_reads_exactly_the_contract_types() {
        for (token, array) in scalar_corpus() {
            let listed = token.is_some_and(|t| WEIGHT_TYPES.contains(&t));
            assert_eq!(
                read_weight_f64(array.as_ref(), 0).is_ok(),
                listed,
                "WEIGHT_TYPES disagrees with read_weight_f64 for {}",
                array.data_type()
            );
        }
    }

    /// `List` of each contract token, plus the two other list containers, so
    /// that both the item types and the container are asserted.
    fn list_corpus() -> Vec<(Option<&'static str>, &'static str, ArrayRef)> {
        let mut int64 = ListBuilder::new(Int64Builder::new());
        int64.values().append_value(1);
        int64.append(true);
        let mut float64 = ListBuilder::new(Float64Builder::new());
        float64.values().append_value(1.0);
        float64.append(true);
        let mut string = ListBuilder::new(StringBuilder::new());
        string.values().append_value("k");
        string.append(true);
        let mut int8 = ListBuilder::new(Int8Builder::new());
        int8.values().append_value(1);
        int8.append(true);
        let mut int16 = ListBuilder::new(Int16Builder::new());
        int16.values().append_value(1);
        int16.append(true);
        let mut int32 = ListBuilder::new(Int32Builder::new());
        int32.values().append_value(1);
        int32.append(true);
        let mut float32 = ListBuilder::new(Float32Builder::new());
        float32.values().append_value(1.0);
        float32.append(true);
        let mut boolean = ListBuilder::new(arrow_array::builder::BooleanBuilder::new());
        boolean.values().append_value(true);
        boolean.append(true);
        let mut decimal = ListBuilder::new(
            Decimal128Builder::new()
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        decimal.values().append_value(100);
        decimal.append(true);
        let mut unsigned = ListBuilder::new(UInt64Builder::new());
        unsigned.values().append_value(1);
        unsigned.append(true);
        let mut large_f64 = LargeListBuilder::new(Float64Builder::new());
        large_f64.values().append_value(1.0);
        large_f64.append(true);
        let mut large_str = LargeListBuilder::new(StringBuilder::new());
        large_str.values().append_value("k");
        large_str.append(true);
        let mut fixed = FixedSizeListBuilder::new(Float64Builder::new(), 1);
        fixed.values().append_value(1.0);
        fixed.append(true);
        vec![
            (Some("int64"), "list", Arc::new(int64.finish())),
            (Some("float64"), "list", Arc::new(float64.finish())),
            (Some("string"), "list", Arc::new(string.finish())),
            (Some("int8"), "list", Arc::new(int8.finish())),
            (Some("int16"), "list", Arc::new(int16.finish())),
            (Some("int32"), "list", Arc::new(int32.finish())),
            (Some("float32"), "list", Arc::new(float32.finish())),
            (Some("boolean"), "list", Arc::new(boolean.finish())),
            (Some("decimal128"), "list", Arc::new(decimal.finish())),
            (None, "list", Arc::new(unsigned.finish())),
            (Some("float64"), "large_list", Arc::new(large_f64.finish())),
            (Some("string"), "large_list", Arc::new(large_str.finish())),
            (Some("float64"), "fixed_size_list", Arc::new(fixed.finish())),
        ]
    }

    #[test]
    fn a_list_target_reads_exactly_the_contract_container_and_items() {
        for (token, container, array) in list_corpus() {
            let listed = LIST_TARGET_CONTAINERS.contains(&container)
                && token.is_some_and(|t| LIST_TARGET_ITEM_TYPES.contains(&t));
            assert_eq!(
                read_link_targets(array.as_ref(), 0).is_ok(),
                listed,
                "LIST_TARGET_* disagrees with read_link_targets for {}",
                array.data_type()
            );
        }
    }

    #[test]
    fn a_list_weight_reads_exactly_the_contract_container_and_items() {
        for (token, container, array) in list_corpus() {
            let listed = WEIGHT_LIST_CONTAINERS.contains(&container)
                && token.is_some_and(|t| WEIGHT_LIST_ITEM_TYPES.contains(&t));
            assert_eq!(
                read_row_weight(array.as_ref(), 0, true, false, "WEIGHT", &mut Vec::new()).is_ok(),
                listed,
                "WEIGHT_LIST_* disagrees with read_row_weight for {}",
                array.data_type()
            );
        }
    }

    #[test]
    fn a_scalar_column_is_not_a_list_and_the_reverse() {
        // The two families do not overlap, which is what lets a caller test
        // the container first and the item type second.
        for (_, array) in scalar_corpus() {
            assert!(read_link_targets(array.as_ref(), 0).is_err());
        }
        for (_, _, array) in list_corpus() {
            assert!(cell_to_value(array.as_ref(), 0).is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::BooleanArray;

    /// Typed `Timestamp(unit)` arrays self-describe their unit; the
    /// `declared` arg is ignored for them.
    #[test]
    fn read_timestamp_secs_typed_units_ignore_declared() {
        // Year 2024 (~1.7e9 epoch seconds) expressed in each unit.
        let secs: i64 = 1_700_000_000;
        let nanos: i64 = secs * 1_000_000_000;
        let micros: i64 = secs * 1_000_000;
        let millis: i64 = secs * 1_000;

        // Pass an intentionally-wrong `declared` to prove typed arrays
        // ignore it.
        let wrong = TimeUnit::Seconds;

        let ns_arr = TimestampNanosecondArray::from(vec![Some(nanos), None]);
        assert_eq!(read_timestamp_secs(&ns_arr, 0, wrong), Some(secs));
        assert_eq!(read_timestamp_secs(&ns_arr, 1, wrong), None);

        let us_arr = TimestampMicrosecondArray::from(vec![Some(micros)]);
        assert_eq!(read_timestamp_secs(&us_arr, 0, wrong), Some(secs));

        let ms_arr = TimestampMillisecondArray::from(vec![Some(millis)]);
        assert_eq!(read_timestamp_secs(&ms_arr, 0, wrong), Some(secs));

        let s_arr = TimestampSecondArray::from(vec![Some(secs)]);
        assert_eq!(
            read_timestamp_secs(&s_arr, 0, TimeUnit::Nanoseconds),
            Some(secs)
        );
    }

    /// `Int64` arrays carry no inherent unit; the bridge declares it via
    /// `CustomTtSource::input_timestamp_unit`. Cover all four sub-second
    /// declarations.
    #[test]
    fn read_timestamp_secs_int64_uses_declared() {
        let secs: i64 = 1_700_000_000;
        let nanos: i64 = secs * 1_000_000_000;
        let micros: i64 = secs * 1_000_000;
        let millis: i64 = secs * 1_000;

        let ns = Int64Array::from(vec![Some(nanos)]);
        assert_eq!(
            read_timestamp_secs(&ns, 0, TimeUnit::Nanoseconds),
            Some(secs)
        );

        let us = Int64Array::from(vec![Some(micros)]);
        assert_eq!(
            read_timestamp_secs(&us, 0, TimeUnit::Microseconds),
            Some(secs)
        );

        let ms = Int64Array::from(vec![Some(millis)]);
        assert_eq!(
            read_timestamp_secs(&ms, 0, TimeUnit::Milliseconds),
            Some(secs)
        );

        let s = Int64Array::from(vec![Some(secs)]);
        assert_eq!(read_timestamp_secs(&s, 0, TimeUnit::Seconds), Some(secs));
    }

    /// Non-sub-second declared units (Minutes, Hours, ...) are not
    /// meaningful for epoch timestamps — return None so the caller
    /// skips the row.
    #[test]
    fn read_timestamp_secs_int64_non_sub_second_returns_none() {
        let int_arr = Int64Array::from(vec![Some(1_700_000_000_i64)]);
        assert_eq!(read_timestamp_secs(&int_arr, 0, TimeUnit::Minutes), None);
        assert_eq!(read_timestamp_secs(&int_arr, 0, TimeUnit::Hours), None);
        assert_eq!(read_timestamp_secs(&int_arr, 0, TimeUnit::Days), None);
    }

    #[test]
    fn read_weight_preserves_float64_precision_and_numeric_types() {
        let large = Float64Array::from(vec![f64::from(f32::MAX) * 2.0]);
        assert_eq!(
            read_weight_f64(&large, 0).unwrap(),
            f64::from(f32::MAX) * 2.0
        );

        let int8 = Int8Array::from(vec![7_i8]);
        assert_eq!(read_weight_f64(&int8, 0).unwrap(), 7.0);

        let int16 = Int16Array::from(vec![32_000_i16]);
        assert_eq!(read_weight_f64(&int16, 0).unwrap(), 32_000.0);

        let decimal = Decimal128Array::from(vec![12_345_i128])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(read_weight_f64(&decimal, 0).unwrap(), 123.45);
    }

    #[test]
    fn read_weight_rejects_null_nonfinite_and_nonnumeric_values() {
        let null = Float64Array::from(vec![None]);
        assert!(read_weight_f64(&null, 0)
            .unwrap_err()
            .to_string()
            .contains("weight is null"));

        let infinite = Float64Array::from(vec![f64::INFINITY]);
        assert!(read_weight_f64(&infinite, 0)
            .unwrap_err()
            .to_string()
            .contains("weight is not finite"));

        let boolean = BooleanArray::from(vec![true]);
        assert!(read_weight_f64(&boolean, 0)
            .unwrap_err()
            .to_string()
            .contains("scalar numeric"));
    }
}
