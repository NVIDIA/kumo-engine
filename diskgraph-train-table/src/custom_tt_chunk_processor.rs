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
    Array, Decimal128Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, LargeListArray, ListArray, RecordBatch, StringArray, TimestampMicrosecondArray,
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

use crate::errors::{TTGenError, TTGenResult};
use crate::split_decider::SplitDecider;
use crate::ttgen_accumulator::hash_value;
use crate::ttgen_output::{ChannelBatchWriter, ExtraColumnOutput, ExtraValueRef, OutputMode};
use crate::ttgen_resolve::{
    ResolvedPlan, AUTOREGRESSION_OUTPUT_ALIAS, FUTURE_LABELS_OUTPUT_ALIAS, YOY_OUTPUT_ALIAS,
};
use crate::ttgen_stats::{ChunkStats, RowProcessor, TTRowRef};

/// Read a declared custom-training weight without narrowing precision.
///
/// Databricks scalar numeric columns arrive through Arrow as one of these
/// signed integer, floating-point, or Decimal128 arrays. A declared weight is
/// required to be present, non-null, finite, and scalar; silently replacing a
/// bad value with `1.0` would change training semantics.
fn read_weight_f64(arr: &dyn Array, idx: usize) -> TTGenResult<f64> {
    if arr.is_null(idx) {
        return Err(TTGenError::SchemaError(format!(
            "custom-TT weight is null at source row {idx}"
        )));
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
        return Err(TTGenError::SchemaError(format!(
            "custom-TT weight must be a scalar numeric value; found Arrow type {}",
            arr.data_type()
        )));
    };

    if !weight.is_finite() {
        return Err(TTGenError::SchemaError(format!(
            "custom-TT weight is not finite at source row {idx}"
        )));
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
                target_dtypes: if !plan.is_prediction { plan.target_dtypes.clone() } else { Vec::new() },
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
                        let parquet_name = format!("TARGET{}", i + 1);
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
                        let parquet_name = format!("TARGET{}", i + 1);
                        let field = lookup(&parquet_name).ok_or_else(|| {
                            TTGenError::SchemaError(format!(
                                "custom-TT LP parquet missing required list target column '{parquet_name}'"
                            ))
                        })?;
                        match field.data_type() {
                            DataType::List(inner) => Ok(inner.data_type().clone()),
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
fn value_to_pk_key(value: &Value, key_type: pkmap::KeyType) -> Option<pkmap::Key> {
    match (value, key_type) {
        (Value::Int(i), pkmap::KeyType::Int64) => Some(pkmap::Key::Int(*i as u64)),
        (Value::Str(s), pkmap::KeyType::VarBytes) => {
            Some(pkmap::Key::Bytes(s.as_bytes().to_vec()))
        }
        (Value::Float(f), pkmap::KeyType::Int64) => Some(pkmap::Key::Int(*f as u64)),
        _ => None,
    }
}

/// Resolve a customer-supplied entity PK value to an `entity_node` id.
/// Returns `None` if the PK is not in the index — the caller must skip
/// that row.
fn lookup_entity_node<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    pk_value: &Value,
) -> Option<u64> {
    let pk_map = plan.entity_table.pk_map()?;
    let key = value_to_pk_key(pk_value, pk_map.key_type())?;
    plan.entity_table.node_id_from_pk(&key).ok().flatten()
}

/// Read one cell of an Arrow array as our `Value` enum. Supports the dtypes
/// we expect in customer parquets: Int64, Float64, Utf8 (entity PKs +
/// scalar targets).
fn cell_to_value(arr: &dyn Array, idx: usize) -> Option<Value> {
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
fn read_list_cell(arr: &ListArray, idx: usize) -> Option<Value> {
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

fn weight_slice_f64(values: &dyn Array, start: usize, end: usize, row: usize) -> TTGenResult<Vec<f64>> {
    macro_rules! typed {
        ($ty:ty) => {
            if let Some(a) = values.as_any().downcast_ref::<$ty>() {
                return (start..end)
                    .map(|j| {
                        if a.is_null(j) {
                            return Err(TTGenError::SchemaError(format!(
                                "custom-TT weight has a null element at source row {row}"
                            )));
                        }
                        let w = a.value(j) as f64;
                        if !w.is_finite() {
                            return Err(TTGenError::SchemaError(format!(
                                "custom-TT weight is not finite at source row {row}"
                            )));
                        }
                        Ok(w)
                    })
                    .collect();
            }
        };
    }
    typed!(Float64Array);
    typed!(Float32Array);
    typed!(Int64Array);
    typed!(Int32Array);
    typed!(Int16Array);
    typed!(Int8Array);
    Err(TTGenError::SchemaError(format!(
        "custom-TT weight elements must be numeric; found Arrow type {}",
        values.data_type()
    )))
}

fn read_weight_list_f64(arr: &dyn Array, idx: usize) -> TTGenResult<Vec<f64>> {
    if arr.is_null(idx) {
        return Err(TTGenError::SchemaError(format!(
            "custom-TT weight is null at source row {idx}"
        )));
    }
    if let Some(l) = arr.as_any().downcast_ref::<ListArray>() {
        let o = l.value_offsets();
        return weight_slice_f64(l.values().as_ref(), o[idx] as usize, o[idx + 1] as usize, idx);
    }
    if let Some(l) = arr.as_any().downcast_ref::<LargeListArray>() {
        let o = l.value_offsets();
        return weight_slice_f64(l.values().as_ref(), o[idx] as usize, o[idx + 1] as usize, idx);
    }
    Err(TTGenError::SchemaError(format!(
        "custom-TT weight must be a list beside an array target; found Arrow type {}",
        arr.data_type()
    )))
}

fn read_i64_list_values(arr: &dyn Array, idx: usize) -> Option<Vec<i64>> {
    read_list_values(arr, idx).map(|values| {
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
        let target_slot = match name {
            "TARGET1" => Some(plan.primary_target_alias_idx()),
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
    shared_entity_counts: Arc<SharedColumnCounts>,
    input_timestamp_unit: TimeUnit,
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
        .unwrap_or("TIMESTAMP");
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

    let mut processor: RowProcessor<U, D> =
        RowProcessor::new(stats, decider_template.clone());
    let mut local_target_counts: Option<LocalColumnCounts> = if target_is_categorical {
        Some(LocalColumnCounts::with_capacity(5000))
    } else {
        None
    };
    // Entity CATEGORY_COUNTS has no downstream reader (the entity column is a
    // FK/pkey resolved to node ids, never encoded), so it is not collected --
    // only entity cardinality via the HLL (NUM_UNIQUE) is retained. See the
    // parallel comment in `ttgen_executor::run_pass_a`.
    let mut local_entity_counts: Option<LocalColumnCounts> = None;

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
            let name = format!("ENTITY{}", i + 1);
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

        for batch_res in reader {
            let batch =
                batch_res.map_err(|e| TTGenError::Parquet(format!("{e}")))?;
            process_batch(
                plan,
                &batch,
                &mut aliases,
                &link_frequencies,
                target_is_numerical,
                target_is_categorical,
                target_stype,
                &mut local_target_counts,
                &mut local_entity_counts,
                &mut writer,
                &mut processor,
                input_timestamp_unit,
            )?;
        }
    }


    // Flush local counts to shared.
    if let (Some(ref shared), Some(ref mut local)) =
        (&shared_target_counts, &mut local_target_counts)
    {
        shared.merge_local_batch(local);
    }
    // Entity counts are not collected (see `local_entity_counts` above), so the
    // shared accumulator stays empty and consumes none of the string budget.
    if let Some(ref mut local) = local_entity_counts {
        shared_entity_counts.merge_local_batch(local);
    }

    let rows = writer.finish()?;
    processor.stats.num_total_rows = rows;
    Ok((rows, processor))
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
    local_entity_counts: &mut Option<LocalColumnCounts>,
    writer: &mut ChannelBatchWriter<U>,
    processor: &mut RowProcessor<U, D>,
    input_timestamp_unit: TimeUnit,
) -> TTGenResult<()> {
    let schema = batch.schema();
    // Resolve one parquet column index per ENTITYk slot the plan declares.
    // Missing columns are a hard error: the plan asked for k entities but the
    // parquet doesn't carry them.
    let entity_col_indices: Vec<usize> = (0..plan.entity_pk_col_indices.len())
        .map(|i| {
            let name = format!("ENTITY{}", i + 1);
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
    let ts_col_idx = schema.index_of("TIMESTAMP").ok();
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
            !matches!(
                name,
                "ENTITY" | "TARGET" | "__kumo_internal_frequency_column__" | "ENTITY_INCLUDE_BIT"
            ) && !name.starts_with("ENTITY")
              && !name.starts_with("TARGET")
              && name != "TIMESTAMP"
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
        .map(|i| schema.index_of(&format!("TARGET{}", i + 1)).ok())
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
    let read_split_value = |row_idx: usize| -> Option<Value> {
        split_col_idx.and_then(|c| cell_to_value(batch.column(c).as_ref(), row_idx))
    };

    for row_idx in 0..batch.num_rows() {
        // Reset alias slots for this row (paranoia — keeps the alias array
        // in a clean state for any future plan that might run
        // execute_stages on the custom-TT path).
        aliases.iter_mut().for_each(|a| *a = None);

        // Read all ENTITYk PKs from the parquet. ENTITY1 (slot 0) is the
        // primary key used to look up the entity node in the index; ENTITY2+
        // are auxiliary join keys carried through to the output unchanged.
        let mut entity_pks: Vec<Value> = Vec::with_capacity(entity_col_indices.len());
        let mut all_present = true;
        for &col in &entity_col_indices {
            match cell_to_value(batch.column(col).as_ref(), row_idx) {
                Some(v) => entity_pks.push(v),
                None => {
                    all_present = false;
                    break;
                }
            }
        }
        if !all_present {
            continue;
        }
        let entity_pk = entity_pks[0].clone();

        // Resolve to entity_node via the index. Skip if unknown (documented
        // behavior — handles stale-customer-data without failing the job).
        // Multi-entity behavior: lookup uses ENTITY1 (slot 0); ENTITY2+ are
        // treated as auxiliary keys carried through unchanged. If ENTITY1
        // is unknown the row is skipped.
        let entity_node = match lookup_entity_node(plan, &entity_pk) {
            Some(n) => n,
            None => continue,
        };

        // Confirm the resolved node is canonical (paranoia: matches PQ
        // path's is_canonical_node guard).
        if !plan.entity_table.is_canonical_node(entity_node, &entity_pk) {
            continue;
        }

        for (col_idx, name) in &sequence_feature_indices {
            let entry = processor.stats.sequence_feature_stats
                .entry(name.clone())
                .or_default();
            entry.observe(batch.column(*col_idx).as_ref(), row_idx);
        }

        // Read anchor timestamp. Absent TIMESTAMP column (static task) ⟹ use
        // the synthesized native-static sentinel. A present column that is NULL
        // for this row is dropped (matches native `drop_na_rows(TIMESTAMP)`).
        let ts = match &ts_arr {
            Some(arr) => {
                let Some(ts_secs) = read_timestamp_secs(arr.as_ref(), row_idx, input_timestamp_unit)
                else {
                    continue;
                };
                Abs::<U>::from_secs(ts_secs)
            }
            None => sentinel_ts,
        };
        // Epoch seconds for the LP RHS-target temporal-validity filter below.
        let ts_secs = ts.to_secs();

        // Per-prediction-type direct read + write.
        match &plan.prediction_type {
            PredictionType::PassThrough => {
                // Push the row's weight BEFORE write_row_pass_through_wide
                // so the writer's flush_batch (which may fire inside
                // write_row when BATCH_SIZE is reached) sees a weight
                // buffer whose length matches the about-to-be-flushed
                // batch's row count.
                if let Some(c) = weight_col_idx {
                    writer.push_weight(read_weight_f64(
                        batch.column(c).as_ref(),
                        row_idx,
                    )?);
                }
                let split_value = read_split_value(row_idx);
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
                processor.observe(&row, local_target_counts, local_entity_counts);
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

                if let Some(c) = weight_col_idx {
                    writer.push_weight(read_weight_f64(
                        batch.column(c).as_ref(),
                        row_idx,
                    )?);
                }
                let split_value = read_split_value(row_idx);
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
                        processor.observe(
                            &row,
                            local_target_counts,
                            local_entity_counts,
                        );
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
                            continue;
                        }
                    }
                }

                if let Some(c) = weight_col_idx {
                    writer.push_weight(read_weight_f64(
                        batch.column(c).as_ref(),
                        row_idx,
                    )?);
                }
                let split_value = read_split_value(row_idx);
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
                    processor.observe(&row, local_target_counts, local_entity_counts);
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
                let (mut targets, mut freqs): (Vec<Value>, Vec<i64>) = if plan.is_prediction {
                    (Vec::new(), Vec::new())
                } else {
                    let target_col = target_col_indices
                        .first()
                        .copied()
                        .flatten()
                        .ok_or_else(|| {
                            TTGenError::SchemaError(
                                "custom-TT LP parquet missing required column 'TARGET1'"
                                    .to_string(),
                            )
                        })?;
                    let targets = read_list_values(batch.column(target_col).as_ref(), row_idx)
                        .unwrap_or_default();

                    let mut freqs: Vec<i64> = if *include_frequency {
                        frequency_col_idx
                            .and_then(|c| read_i64_list_values(batch.column(c).as_ref(), row_idx))
                            .unwrap_or_else(|| vec![1; targets.len()])
                    } else {
                        Vec::new()
                    };
                    if *include_frequency && freqs.len() != targets.len() {
                        freqs.resize(targets.len(), 1);
                    }
                    (targets, freqs)
                };

                let per_element_weight = writer.weight_is_per_element();
                let mut weights: Vec<f64> = match weight_col_idx {
                    Some(c) if per_element_weight => {
                        let w = read_weight_list_f64(batch.column(c).as_ref(), row_idx)?;
                        if w.len() != targets.len() {
                            return Err(TTGenError::SchemaError(format!(
                                "custom-TT weight list has {} entries but the target \
                                 has {} at source row {row_idx}",
                                w.len(),
                                targets.len()
                            )));
                        }
                        w
                    }
                    _ => Vec::new(),
                };

                if let Some(pk_table) = plan.custom_target_pk_table.as_ref() {
                    let mut filtered_targets = Vec::with_capacity(targets.len());
                    let mut filtered_weights = Vec::with_capacity(weights.len());
                    let mut filtered_freqs = if *include_frequency {
                        Vec::with_capacity(freqs.len())
                    } else {
                        Vec::new()
                    };
                    for (idx, target) in targets.into_iter().enumerate() {
                        let keep = if plan.custom_target_temporal_create_col_idx.is_some()
                            || plan.custom_target_temporal_end_col_idx.is_some()
                        {
                            pk_table.contains_pk_value_temporal(
                                &target,
                                ts_secs,
                                plan.custom_target_temporal_create_col_idx,
                                plan.custom_target_temporal_end_col_idx,
                                &mut target_pk_scratch,
                            )
                        } else {
                            pk_table.contains_pk_value(&target)
                        };
                        if keep {
                            filtered_targets.push(target);
                            if *include_frequency {
                                filtered_freqs.push(freqs.get(idx).copied().unwrap_or(1));
                            }
                            if per_element_weight {
                                filtered_weights.push(weights[idx]);
                            }
                        }
                    }
                    targets = filtered_targets;
                    if *include_frequency {
                        freqs = filtered_freqs;
                    }
                    weights = filtered_weights;
                }

                if let Some(c) = weight_col_idx {
                    if per_element_weight {
                        writer.push_weight_list(weights);
                    } else {
                        writer.push_weight(read_weight_f64(
                            batch.column(c).as_ref(),
                            row_idx,
                        )?);
                    }
                }
                let split_value = read_split_value(row_idx);
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
                        processor.observe(&row, local_target_counts, local_entity_counts);
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
        assert_eq!(read_timestamp_secs(&s_arr, 0, TimeUnit::Nanoseconds), Some(secs));
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
        assert_eq!(read_timestamp_secs(&ns, 0, TimeUnit::Nanoseconds), Some(secs));

        let us = Int64Array::from(vec![Some(micros)]);
        assert_eq!(read_timestamp_secs(&us, 0, TimeUnit::Microseconds), Some(secs));

        let ms = Int64Array::from(vec![Some(millis)]);
        assert_eq!(read_timestamp_secs(&ms, 0, TimeUnit::Milliseconds), Some(secs));

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
