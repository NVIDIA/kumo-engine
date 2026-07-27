//! Parquet shard writer for TT-gen output.
//!
//! Each thread gets its own `ParquetShardWriter` that buffers rows and
//! writes batched Arrow RecordBatches to a Parquet file.
//!
//! Output modes:
//! - **Node**: scalar f64 target (one row per entity/timestamp)
//! - **LinkArray**: one row per entity/timestamp with list columns

use std::marker::PhantomData;
use std::sync::Arc;

use crate::errors::{TTGenError, TTGenResult};
use crate::ttgen_resolve::{AUTOREGRESSION_OUTPUT_ALIAS, FUTURE_LABELS_OUTPUT_ALIAS, YOY_OUTPUT_ALIAS};
use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{
    BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use crossbeam_channel::Sender;
use diskgraph_core::time::{Abs, UnitSystem};
use diskgraph_core::tt_builder::Value;
pub use diskgraph_core::tt_builder::{PRIMARY_ENTITY_COL, TIMESTAMP_COL, PRIMARY_TARGET_COL, WEIGHT_COL};

/// Batch size: flush to Parquet every N rows.
/// 64K rows × 3-4 columns × 8 bytes = ~1.5-2MB per batch.
const BATCH_SIZE: usize = 65536;

/// Which output format the writer produces.
///
/// Per the ML team's CustomTrainTableSpec contract, all task types may carry
/// `TARGET1..TARGETn` (multi-target). The PQ path emits a single TARGET; the
/// custom-TT path may carry multiple TARGETs sourced from the customer parquet.
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// Node prediction: one TARGETk column per slot.
    Node { target_dtypes: Vec<DataType> },
    /// Link prediction: one row per entity/ts with list columns.
    LinkArray { include_frequency: bool, target_dtypes: Vec<DataType> },
    /// Emits only (ENTITY1..ENTITYn, TIMESTAMP) columns.
    PassThrough,
    /// Forecasting: emits (ENTITY1..ENTITYn, TIMESTAMP, TARGET1..TARGETn,
    /// autoregression_labels, [year_over_year_labels]).
    Forecasting {
        target_dtypes: Vec<DataType>,
        autoregression_dtype: DataType,
        future_labels_dtype: Option<DataType>,
        yoy_dtype: Option<DataType>,
    },
}

/// Extra column emitted alongside each generated training-table row.
#[derive(Debug, Clone)]
pub struct ExtraColumnOutput {
    pub name: String,
    pub dtype: DataType,
}

/// Borrowed extra column value passed from the executor into the writer.
#[derive(Debug, Clone, Copy)]
pub enum ExtraValueRef<'a> {
    Scalar(Option<&'a Value>),
    List(&'a [Value]),
}

/// Channel message: a `RecordBatch` streamed from a worker to the writer
/// thread. Per-row training weights ride inside the batch as the reserved
/// `WEIGHT` column — no separate side-channel.
pub type BatchMessage = RecordBatch;

/// Per-thread batch writer that buffers rows and sends Arrow RecordBatches
/// through a crossbeam channel for streaming consumption by Python.
pub struct ChannelBatchWriter<U: UnitSystem> {
    sender: Sender<BatchMessage>,
    schema: Arc<Schema>,
    mode: OutputMode,
    has_include_bit: bool,
    /// Per-slot ENTITY dtypes (one per ENTITYk column). Length 1 for PQ path,
    /// can be longer for custom-TT multi-entity.
    entity_dtypes: Vec<DataType>,
    extra_column: Option<ExtraColumnOutput>,
    /// User-facing feature columns (FEAT1..FEATn and LAG_FEAT1..LAG_FEATn per
    /// the ML team's CustomTrainTableSpec). Distinct from `extra_column`
    /// (which is reserved for split projection). Emitted in declaration order
    /// after the per-mode payload + extra_column.
    feat_columns: Vec<ExtraColumnOutput>,

    // Common buffers — one Vec per ENTITYk slot. Slot 0 is the primary.
    entity_bufs: Vec<Vec<Value>>,
    timestamp_buf: Vec<i64>,
    include_bit_buf: Vec<bool>,
    extra_value_buf: Vec<Value>,
    extra_list_value_buf: Vec<Vec<Value>>,
    /// Per-feat-column buffers. Scalar feat columns use the inner `Vec<Value>`
    /// (one value per row); list-typed feat columns store the list per row.
    feat_value_bufs: Vec<Vec<Value>>,
    feat_list_value_bufs: Vec<Vec<Vec<Value>>>,

    // Node mode — per-slot TARGETk buffers. Slot 0 is primary; slots 1+ are
    // populated only via the wide write entry point. For Bool/Float64 fast
    // paths we still keep slot 0 in primitive buffers; slot k>=1 always
    // goes through `target_value_bufs[k]`.
    target_f64_buf: Vec<f64>,
    target_bool_buf: Vec<bool>,
    /// Per-slot Value buffers for Node mode TARGETk slots (k = 0 used only
    /// when target dtype is non-primitive). For multi-target Node we always
    /// route slots >= 1 here, regardless of dtype.
    target_value_bufs: Vec<Vec<Value>>,

    // LinkArray mode — one Vec<Vec<Value>> per TARGET slot. Slot k's list at
    // row r is `target_value_list_bufs[k][r]`.
    target_value_list_bufs: Vec<Vec<Vec<Value>>>,
    frequency_list_buf: Vec<Vec<i64>>,

    // Forecasting mode
    autoregression_list_buf: Vec<Vec<Value>>,
    future_labels_list_buf: Vec<Vec<Value>>,
    yoy_list_buf: Vec<Vec<Value>>,

    rows_written: u64,

    /// Per-row weight buffer (custom-TT path only). `Some` when the plan
    /// declared a `weight_col` (set via `enable_weight_buf`); the chunk
    /// processor then calls `push_weight` once per emitted row. `flush_batch`
    /// drains it into the reserved `WEIGHT` column (no sidecar). PQ path leaves
    /// it `None` — invariant: `weight_col implies chunk processor`.
    weight_buf: Option<Vec<f64>>,

    _unit: PhantomData<U>,
}

impl<U: UnitSystem> ChannelBatchWriter<U> {
    /// Create a new batch writer that sends RecordBatches to `sender`.
    ///
    /// `entity_dtypes` is the Arrow data type for each ENTITYk column,
    /// matching the ingested type of the entity table's PK column. Length
    /// determines the number of ENTITYk columns emitted (ENTITY1..ENTITYn).
    /// PQ path always passes a single-element slice; custom-TT path may
    /// pass multiple slots.
    ///
    /// Schema layout:
    ///   ENTITY1, ENTITY2, ..., ENTITYn (n = entity_dtypes.len()),
    ///   TIMESTAMP,
    ///   per-mode payload columns (TARGET1..TARGETn for Node;
    ///                             TARGET for Link; autoregression_labels for Forecasting),
    ///   optional extra projection column,
    ///   optional ENTITY_INCLUDE_BIT.
    ///
    /// When `include_entity_bit` is true, an extra `ENTITY_INCLUDE_BIT` boolean
    /// column is added to the schema. This is used when `lag_timesteps > 0` to
    /// mark rows where the entity filter failed (kept for autoregressive features).
    pub fn new(
        sender: Sender<BatchMessage>,
        mode: OutputMode,
        include_entity_bit: bool,
        entity_dtypes: Vec<DataType>,
        time_col_name: &str,
        extra_column: Option<ExtraColumnOutput>,
        feat_columns: Vec<ExtraColumnOutput>,
    ) -> TTGenResult<Self> {
        if entity_dtypes.is_empty() {
            return Err(TTGenError::InvalidPlan(
                "ChannelBatchWriter requires at least one ENTITY slot".to_string(),
            ));
        }

        let mut fields: Vec<Field> = Vec::new();
        for (i, dtype) in entity_dtypes.iter().enumerate() {
            fields.push(Field::new(format!("ENTITY{}", i + 1), dtype.clone(), false));
        }
        fields.push(Field::new(
            time_col_name,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ));

        // Track per-slot target buffers up-front so they get sized. Node and
        // Forecasting both keep per-slot scalar buffers (slot 0 may use the
        // primitive fast-path bufs); LinkArray uses per-slot list buffers
        // (target_value_list_bufs).
        let target_buf_count = match &mode {
            OutputMode::Node { target_dtypes } => target_dtypes.len(),
            OutputMode::LinkArray { target_dtypes, .. } => target_dtypes.len(),
            OutputMode::Forecasting { target_dtypes, .. } => target_dtypes.len(),
            OutputMode::PassThrough => 0,
        };

        match &mode {
            OutputMode::Node { target_dtypes } => {
                // Empty target_dtypes ⟹ a PREDICTION bundle: emit no TARGETk
                // columns (`[ENTITY1..ENTITYn, TIMESTAMP]` only), matching the
                // native predict / PassThrough shape. Train always has ≥1 slot.
                for (i, dtype) in target_dtypes.iter().enumerate() {
                    fields.push(Field::new(
                        format!("TARGET{}", i + 1),
                        dtype.clone(),
                        true,
                    ));
                }
            }
            OutputMode::LinkArray { include_frequency, target_dtypes } => {
                // Empty target_dtypes ⟹ a PREDICTION LP bundle: no TARGETk (and
                // no frequency) columns — the seed table is entities + time.
                for (i, dtype) in target_dtypes.iter().enumerate() {
                    fields.push(Field::new(
                        format!("TARGET{}", i + 1),
                        DataType::List(Arc::new(Field::new("item", dtype.clone(), true))),
                        true,
                    ));
                }
                if *include_frequency && !target_dtypes.is_empty() {
                    fields.push(Field::new(
                        "__kumo_internal_frequency_column__",
                        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                        false,
                    ));
                }
            }
            OutputMode::PassThrough => {}
            OutputMode::Forecasting { target_dtypes, autoregression_dtype, future_labels_dtype, yoy_dtype } => {
                for (i, dtype) in target_dtypes.iter().enumerate() {
                    fields.push(Field::new(
                        format!("TARGET{}", i + 1),
                        dtype.clone(),
                        true,
                    ));
                }
                fields.push(Field::new(
                    AUTOREGRESSION_OUTPUT_ALIAS,
                    autoregression_dtype.clone(),
                    true,
                ));
                if let Some(future_dt) = future_labels_dtype {
                    fields.push(Field::new(
                        FUTURE_LABELS_OUTPUT_ALIAS,
                        future_dt.clone(),
                        true,
                    ));
                }
                if let Some(yoy_dt) = yoy_dtype {
                    fields.push(Field::new(
                        YOY_OUTPUT_ALIAS,
                        yoy_dt.clone(),
                        true,
                    ));
                }
            }
        }

        if let Some(extra) = &extra_column {
            fields.push(Field::new(&extra.name, extra.dtype.clone(), true));
        }

        for feat in &feat_columns {
            fields.push(Field::new(&feat.name, feat.dtype.clone(), true));
        }

        if include_entity_bit {
            fields.push(Field::new("ENTITY_INCLUDE_BIT", DataType::Boolean, false));
        }

        // `WEIGHT` is reserved (appended later by `enable_weight_buf`), so any
        // `WEIGHT`-named field here is a stray feature/output column. Reject it
        // before the engine misreads it as `task.weight` — this guards the
        // no-`weight_col` case that `enable_weight_buf`'s guard can't catch.
        if let Some(bad) = fields.iter().find(|f| f.name() == WEIGHT_COL) {
            return Err(TTGenError::InvalidPlan(format!(
                "column name `{}` is reserved for the per-row training-weight \
                 column and may not be used by a feature/output/source column \
                 (offending column: `{}`)",
                WEIGHT_COL,
                bad.name()
            )));
        }

        let schema = Arc::new(Schema::new(fields));

        let entity_bufs: Vec<Vec<Value>> = (0..entity_dtypes.len())
            .map(|_| Vec::with_capacity(BATCH_SIZE))
            .collect();
        // Always at least one scalar target buffer to keep the slot-0 path
        // happy (used by Link modes' single-row path and by Forecasting's
        // optional slot-0 TARGET).
        let target_value_buf_count = target_buf_count.max(1);
        let target_value_bufs: Vec<Vec<Value>> =
            (0..target_value_buf_count).map(|_| Vec::new()).collect();

        // Per-slot list buffers for LinkArray (one Vec<Vec<Value>> per TARGET slot).
        let target_value_list_buf_count = match &mode {
            OutputMode::LinkArray { target_dtypes, .. } => target_dtypes.len(),
            _ => 0,
        };
        let target_value_list_bufs: Vec<Vec<Vec<Value>>> =
            (0..target_value_list_buf_count).map(|_| Vec::new()).collect();

        // Per-feat-column buffers. Scalar feat columns use feat_value_bufs;
        // list feat columns use feat_list_value_bufs. The slot index is the
        // same for both; only one of the two carries data per feat col.
        let feat_value_bufs: Vec<Vec<Value>> =
            (0..feat_columns.len()).map(|_| Vec::new()).collect();
        let feat_list_value_bufs: Vec<Vec<Vec<Value>>> =
            (0..feat_columns.len()).map(|_| Vec::new()).collect();

        Ok(Self {
            sender,
            schema,
            mode,
            has_include_bit: include_entity_bit,
            entity_dtypes,
            extra_column,
            feat_columns,
            entity_bufs,
            timestamp_buf: Vec::with_capacity(BATCH_SIZE),
            include_bit_buf: Vec::with_capacity(if include_entity_bit { BATCH_SIZE } else { 0 }),
            extra_value_buf: Vec::new(),
            extra_list_value_buf: Vec::new(),
            feat_value_bufs,
            feat_list_value_bufs,
            target_f64_buf: Vec::new(),
            target_bool_buf: Vec::new(),
            target_value_bufs,
            target_value_list_bufs,
            frequency_list_buf: Vec::new(),
            autoregression_list_buf: Vec::new(),
            future_labels_list_buf: Vec::new(),
            yoy_list_buf: Vec::new(),
            rows_written: 0,
            weight_buf: None,
            _unit: PhantomData,
        })
    }

    /// Enable per-row weight collection: the chunk processor then calls
    /// `push_weight` once per emitted row, and each flushed batch carries a
    /// reserved `WEIGHT` (Float64) column holding the weights (no sidecar).
    ///
    /// MUST be called before any rows are buffered (it appends the `WEIGHT`
    /// field to the schema). Only the custom-TT path calls this. Errors on a
    /// reserved-name collision with an existing `WEIGHT` column.
    pub fn enable_weight_buf(&mut self) -> TTGenResult<()> {
        if self.weight_buf.is_some() {
            return Ok(());
        }
        if self.schema.fields().iter().any(|f| f.name() == WEIGHT_COL) {
            return Err(TTGenError::InvalidPlan(format!(
                "column name `{WEIGHT_COL}` is reserved for per-row training \
                 weights but a feature/output column already uses it"
            )));
        }
        // Append the reserved WEIGHT field to the schema so flush_batch can
        // emit it. Safe here because no rows are buffered yet.
        let mut fields: Vec<Field> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(WEIGHT_COL, DataType::Float64, false));
        self.schema = Arc::new(Schema::new(fields));
        self.weight_buf = Some(Vec::new());
        Ok(())
    }

    /// Push one weight, called per emitted output row by the chunk
    /// processor. No-op if `enable_weight_buf` was not called (PQ path).
    #[inline]
    pub fn push_weight(&mut self, weight: f64) {
        if let Some(buf) = self.weight_buf.as_mut() {
            buf.push(weight);
        }
    }

    /// Push per-row feat-column values. `values.len()` must equal the
    /// number of declared `feat_columns`. Each entry is dispatched by the
    /// declared dtype: list-typed slots take a `Value::List(_)` (or `None`
    /// → empty list); scalar slots take any other `Value`.
    pub fn push_feat_values(&mut self, values: &[Option<Value>]) -> TTGenResult<()> {
        if values.len() != self.feat_columns.len() {
            return Err(TTGenError::SystemError(format!(
                "push_feat_values: expected {} feat cols, got {}",
                self.feat_columns.len(),
                values.len()
            )));
        }
        for (slot, feat) in self.feat_columns.iter().enumerate() {
            if matches!(feat.dtype, DataType::List(_)) {
                let list = match &values[slot] {
                    Some(Value::List(items)) => items.clone(),
                    Some(other) => vec![other.clone()],
                    None => Vec::new(),
                };
                self.feat_list_value_bufs[slot].push(list);
            } else {
                let value = values[slot]
                    .clone()
                    .unwrap_or_else(|| Value::Str("None".to_string()));
                self.feat_value_bufs[slot].push(value);
            }
        }
        Ok(())
    }

    /// Buffer a single node-prediction output row (single-entity, single-target).
    ///
    /// Convenience wrapper for the PQ path which is always single-entity /
    /// single-target. The custom-TT path may use `write_row_wide` to supply
    /// multiple ENTITYs / TARGETs.
    ///
    /// The target value is dispatched by the writer's target dtype:
    /// - Boolean: stored in `target_bool_buf`
    /// - Float64: converted via `value_to_f64` into `target_f64_buf`
    /// - Utf8/Int64/other: stored as `Value` in `target_value_bufs[0]`
    #[inline]
    pub fn write_row(
        &mut self,
        entity: Value,
        timestamp: Abs<U>,
        target: &Value,
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        self.write_row_wide(
            std::slice::from_ref(&entity),
            timestamp,
            std::slice::from_ref(target),
            extra_value,
            entity_include_bit,
        )
    }

    /// Buffer a single node-prediction output row with multi-entity / multi-target
    /// support. `entities` length must equal the writer's entity-slot count;
    /// `targets` length must equal the writer's target-slot count (Node mode).
    pub fn write_row_wide(
        &mut self,
        entities: &[Value],
        timestamp: Abs<U>,
        targets: &[Value],
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        if entities.len() != self.entity_bufs.len() {
            return Err(TTGenError::SystemError(format!(
                "write_row_wide: expected {} entity slots, got {}",
                self.entity_bufs.len(),
                entities.len()
            )));
        }
        for (slot, value) in entities.iter().enumerate() {
            self.entity_bufs[slot].push(value.clone());
        }
        self.timestamp_buf.push(timestamp.raw());
        self.push_extra_value(extra_value);
        match &self.mode {
            OutputMode::Node { target_dtypes } => {
                if targets.len() != target_dtypes.len() {
                    return Err(TTGenError::SystemError(format!(
                        "write_row_wide: expected {} target slots, got {}",
                        target_dtypes.len(),
                        targets.len()
                    )));
                }
                // Empty target_dtypes ⟹ PREDICTION bundle: no target buffers to
                // push (schema has no TARGETk columns).
                if !target_dtypes.is_empty() {
                    // Slot 0 still uses the primitive fast paths (Bool/Float64)
                    // to preserve PQ-path performance characteristics.
                    let dtype0 = &target_dtypes[0];
                    let target0 = &targets[0];
                    if *dtype0 == DataType::Boolean {
                        let b = match target0 {
                            Value::Bool(b) => *b,
                            Value::Int(i) => *i != 0,
                            Value::Float(f) => *f != 0.0,
                            _ => false,
                        };
                        self.target_bool_buf.push(b);
                    } else if *dtype0 == DataType::Float64 {
                        self.target_f64_buf.push(value_to_f64(target0));
                    } else {
                        self.target_value_bufs[0].push(target0.clone());
                    }
                    // Slots >= 1 always go through the Value path.
                    for (slot, value) in targets.iter().enumerate().skip(1) {
                        self.target_value_bufs[slot].push(value.clone());
                    }
                }
            }
            _ => {
                // Non-Node modes: targets[0] is always pushed via f64 fast path
                // here. Callers that want a typed target should use the
                // mode-specific entry points (write_row_array, etc.).
                if let Some(t) = targets.first() {
                    self.target_f64_buf.push(value_to_f64(t));
                }
            }
        }
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_bufs[0].len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Buffer a single forecasting row (single-entity, single-target convenience).
    /// No allocation on the hot path — `target` is borrowed.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn write_row_forecasting(
        &mut self,
        entity: Value,
        timestamp: Abs<U>,
        target: Option<&Value>,
        autoregression: Vec<Value>,
        future_labels: Option<Vec<Value>>,
        yoy: Option<Vec<Value>>,
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        self.write_row_forecasting_wide(
            std::slice::from_ref(&entity),
            timestamp,
            target,
            &[],
            autoregression,
            future_labels,
            yoy,
            extra_value,
            entity_include_bit,
        )
    }

    /// Multi-entity / multi-target forecasting variant.
    ///
    /// Slot 0's target is `target0` (borrowed, `None` on the prediction side).
    /// Slots 1..n are `extra_targets` (borrowed slice). Total target count
    /// (1 if `target0.is_some()` else 0) + extra_targets.len() must equal
    /// the writer's declared TARGET slot count.
    ///
    /// Splitting slot 0 from the rest avoids allocating a Vec<Value> on the
    /// hot path: single-target plans pass `Some(&v)` + `&[]` with zero heap work.
    #[allow(clippy::too_many_arguments)]
    pub fn write_row_forecasting_wide(
        &mut self,
        entities: &[Value],
        timestamp: Abs<U>,
        target0: Option<&Value>,
        extra_targets: &[Value],
        autoregression: Vec<Value>,
        future_labels: Option<Vec<Value>>,
        yoy: Option<Vec<Value>>,
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        self.push_entities(entities)?;
        self.timestamp_buf.push(timestamp.raw());
        self.push_extra_value(extra_value);

        if let OutputMode::Forecasting { target_dtypes, .. } = &self.mode {
            let supplied = if target0.is_some() { 1 + extra_targets.len() } else { extra_targets.len() };
            if supplied != target_dtypes.len() {
                return Err(TTGenError::SystemError(format!(
                    "write_row_forecasting_wide: expected {} target slots, got {}",
                    target_dtypes.len(),
                    supplied
                )));
            }
            if let (Some(dtype0), Some(t0)) = (target_dtypes.first(), target0) {
                if *dtype0 == DataType::Boolean {
                    let b = match t0 {
                        Value::Bool(b) => *b,
                        Value::Int(i) => *i != 0,
                        Value::Float(f) => *f != 0.0,
                        _ => false,
                    };
                    self.target_bool_buf.push(b);
                } else if *dtype0 == DataType::Float64 {
                    self.target_f64_buf.push(crate::ttgen_output::value_to_f64(t0));
                } else {
                    self.target_value_bufs[0].push(t0.clone());
                }
            }
            for (i, value) in extra_targets.iter().enumerate() {
                self.target_value_bufs[i + 1].push(value.clone());
            }
        }

        self.autoregression_list_buf.push(autoregression);
        if let OutputMode::Forecasting { future_labels_dtype: Some(_), .. } = &self.mode {
            self.future_labels_list_buf.push(future_labels.unwrap_or_default());
        }
        if let OutputMode::Forecasting { yoy_dtype: Some(_), .. } = &self.mode {
            self.yoy_list_buf.push(yoy.unwrap_or_default());
        }
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_bufs[0].len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Buffer a single pass-through row (entity + timestamp only).
    #[inline]
    pub fn write_row_pass_through(
        &mut self,
        entity: Value,
        timestamp: Abs<U>,
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        self.write_row_pass_through_wide(
            std::slice::from_ref(&entity),
            timestamp,
            extra_value,
            entity_include_bit,
        )
    }

    /// Multi-entity pass-through variant.
    pub fn write_row_pass_through_wide(
        &mut self,
        entities: &[Value],
        timestamp: Abs<U>,
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        self.push_entities(entities)?;
        self.timestamp_buf.push(timestamp.raw());
        self.push_extra_value(extra_value);
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_bufs[0].len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Buffer a single array link-prediction row (one per entity/timestamp).
    /// Slot-0's list is `targets0` (moved); slots 1..n take their lists from
    /// `extra_targets` (borrowed; cloned into per-slot buffer). Single-target
    /// callers pass `extra_targets: &[]` — zero extra allocations vs. the
    /// pre-multi-target path.
    #[allow(clippy::too_many_arguments)]
    pub fn write_row_array(
        &mut self,
        entity: Value,
        timestamp: Abs<U>,
        targets0: Vec<Value>,
        extra_targets: &[Vec<Value>],
        frequencies: &[i64],
        extra_value: ExtraValueRef<'_>,
        entity_include_bit: bool,
    ) -> TTGenResult<()> {
        let slot_count = self.target_value_list_bufs.len();
        // Empty target slots ⟹ PREDICTION LP bundle: the schema has no TARGETk
        // (or frequency) columns, so ignore any supplied targets/frequencies
        // and write only entity + timestamp. Train always has ≥1 slot.
        if slot_count == 0 {
            self.entity_bufs[0].push(entity);
            self.timestamp_buf.push(timestamp.raw());
            self.push_extra_value(extra_value);
            if self.has_include_bit {
                self.include_bit_buf.push(entity_include_bit);
            }
            if self.entity_bufs[0].len() >= BATCH_SIZE {
                self.flush_batch()?;
            }
            return Ok(());
        }
        let supplied = 1 + extra_targets.len();
        if supplied != slot_count {
            return Err(TTGenError::SystemError(format!(
                "write_row_array: expected {} target slots, got {}",
                slot_count, supplied
            )));
        }
        self.entity_bufs[0].push(entity);
        self.timestamp_buf.push(timestamp.raw());
        self.push_extra_value(extra_value);
        self.target_value_list_bufs[0].push(targets0);
        for (i, list) in extra_targets.iter().enumerate() {
            self.target_value_list_bufs[i + 1].push(list.clone());
        }
        self.frequency_list_buf.push(frequencies.to_vec());
        if self.has_include_bit {
            self.include_bit_buf.push(entity_include_bit);
        }
        if self.entity_bufs[0].len() >= BATCH_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Push entity values into per-slot buffers, validating slot count.
    fn push_entities(&mut self, entities: &[Value]) -> TTGenResult<()> {
        if entities.len() != self.entity_bufs.len() {
            return Err(TTGenError::SystemError(format!(
                "write: expected {} entity slots, got {}",
                self.entity_bufs.len(),
                entities.len()
            )));
        }
        for (slot, value) in entities.iter().enumerate() {
            self.entity_bufs[slot].push(value.clone());
        }
        Ok(())
    }

    fn push_extra_value(&mut self, extra_value: ExtraValueRef<'_>) {
        if let Some(extra) = &self.extra_column {
            if matches!(extra.dtype, DataType::List(_)) {
                let list = match extra_value {
                    ExtraValueRef::List(values) => values.to_vec(),
                    ExtraValueRef::Scalar(Some(Value::List(values))) => values.clone(),
                    ExtraValueRef::Scalar(Some(value)) => vec![value.clone()],
                    ExtraValueRef::Scalar(None) => Vec::new(),
                };
                self.extra_list_value_buf.push(list);
            } else {
                let value = match extra_value {
                    ExtraValueRef::Scalar(Some(value)) => value.clone(),
                    ExtraValueRef::Scalar(None) => Value::Str("None".to_string()),
                    ExtraValueRef::List(values) => Value::List(values.to_vec()),
                };
                self.extra_value_buf.push(value);
            }
        }
    }

    /// Flush buffered rows as an Arrow RecordBatch and send through channel.
    fn flush_batch(&mut self) -> TTGenResult<()> {
        if self.entity_bufs[0].is_empty() {
            return Ok(());
        }

        // Per-slot ENTITYk arrays (slot 0 = primary).
        let mut columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        for (slot, dtype) in self.entity_dtypes.iter().enumerate() {
            columns.push(flush_entity_buf(&mut self.entity_bufs[slot], dtype));
        }

        // Convert epoch seconds → nanoseconds to match Kumo's datetime64[ns] convention.
        let ts_nanos: Vec<i64> = std::mem::take(&mut self.timestamp_buf)
            .into_iter()
            .map(|s| s.saturating_mul(1_000_000_000))
            .collect();
        let ts_arr = TimestampNanosecondArray::from(ts_nanos);
        columns.push(Arc::new(ts_arr));

        match &self.mode {
            OutputMode::Node { target_dtypes } => {
                for (slot, dtype) in target_dtypes.iter().enumerate() {
                    let arr = if slot == 0 && *dtype == DataType::Boolean {
                        Arc::new(BooleanArray::from(std::mem::take(
                            &mut self.target_bool_buf,
                        ))) as Arc<dyn arrow_array::Array>
                    } else if slot == 0 && *dtype == DataType::Float64 {
                        Arc::new(Float64Array::from(std::mem::take(
                            &mut self.target_f64_buf,
                        ))) as Arc<dyn arrow_array::Array>
                    } else {
                        flush_value_buf(&mut self.target_value_bufs[slot], dtype)
                    };
                    columns.push(arr);
                }
            }
            OutputMode::LinkArray { include_frequency, target_dtypes } => {
                for (slot, dtype) in target_dtypes.iter().enumerate() {
                    let lists = std::mem::take(&mut self.target_value_list_bufs[slot]);
                    columns.push(flush_value_list_buf(lists, dtype));
                }
                // No frequency column on a target-less predict LP bundle (keeps
                // the emitted columns aligned with the schema builder).
                if *include_frequency && !target_dtypes.is_empty() {
                    let freq_lists = std::mem::take(&mut self.frequency_list_buf);
                    let mut freq_builder =
                        ListBuilder::new(Int64Builder::new());
                    for list in &freq_lists {
                        for &f in list {
                            freq_builder.values().append_value(f);
                        }
                        freq_builder.append(true);
                    }
                    columns.push(Arc::new(freq_builder.finish()));
                } else {
                    self.frequency_list_buf.clear();
                }
            }
            OutputMode::PassThrough => {}
            OutputMode::Forecasting { target_dtypes, autoregression_dtype, future_labels_dtype, yoy_dtype } => {
                for (slot, dtype) in target_dtypes.iter().enumerate() {
                    let arr = if slot == 0 && *dtype == DataType::Boolean {
                        Arc::new(BooleanArray::from(std::mem::take(
                            &mut self.target_bool_buf,
                        ))) as Arc<dyn arrow_array::Array>
                    } else if slot == 0 && *dtype == DataType::Float64 {
                        Arc::new(Float64Array::from(std::mem::take(
                            &mut self.target_f64_buf,
                        ))) as Arc<dyn arrow_array::Array>
                    } else {
                        flush_value_buf(&mut self.target_value_bufs[slot], dtype)
                    };
                    columns.push(arr);
                }
                let ar_lists = std::mem::take(&mut self.autoregression_list_buf);
                if let DataType::List(inner_field) = autoregression_dtype {
                    columns.push(flush_value_list_buf_with_field(
                        ar_lists,
                        inner_field.as_ref(),
                    ));
                } else {
                    unreachable!("autoregression_dtype must be List");
                }

                if let Some(DataType::List(inner_field)) = future_labels_dtype {
                    let future_lists = std::mem::take(&mut self.future_labels_list_buf);
                    columns.push(flush_value_list_buf_with_field(
                        future_lists,
                        inner_field.as_ref(),
                    ));
                }

                if let Some(DataType::List(inner_field)) = yoy_dtype {
                    let yoy_lists = std::mem::take(&mut self.yoy_list_buf);
                    columns.push(flush_value_list_buf_with_field(
                        yoy_lists,
                        inner_field.as_ref(),
                    ));
                }
            }
        }

        if let Some(extra) = &self.extra_column {
            if let DataType::List(inner_field) = &extra.dtype {
                let lists = std::mem::take(&mut self.extra_list_value_buf);
                columns.push(flush_value_list_buf_with_field(
                    lists,
                    inner_field.as_ref(),
                ));
            } else {
                columns.push(flush_value_buf(&mut self.extra_value_buf, &extra.dtype));
            }
        }

        for (slot, feat) in self.feat_columns.iter().enumerate() {
            if let DataType::List(inner_field) = &feat.dtype {
                let lists = std::mem::take(&mut self.feat_list_value_bufs[slot]);
                columns.push(flush_value_list_buf_with_field(
                    lists,
                    inner_field.as_ref(),
                ));
            } else {
                columns.push(flush_value_buf(&mut self.feat_value_bufs[slot], &feat.dtype));
            }
        }

        if self.has_include_bit {
            let bit_arr = BooleanArray::from(std::mem::take(&mut self.include_bit_buf));
            columns.push(Arc::new(bit_arr));
        }

        // Reserved WEIGHT column: drain this batch's weights off the front of
        // `weight_buf` and emit them LAST — matching the field order
        // `enable_weight_buf` set. PQ path leaves `weight_buf == None`.
        let n_rows_for_weight = columns.first().map(|c| c.len()).unwrap_or(0);
        if let Some(buf) = self.weight_buf.as_mut() {
            let n = n_rows_for_weight;
            debug_assert!(
                buf.len() >= n,
                "weight_buf has {} weights but batch needs {}",
                buf.len(),
                n
            );
            let rest = buf.split_off(n);
            let head = std::mem::replace(buf, rest);
            columns.push(Arc::new(Float64Array::from(head)));
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| TTGenError::SystemError(format!("creating RecordBatch: {e}")))?;

        self.rows_written += batch.num_rows() as u64;

        // Send batch through channel. If the receiver is dropped, we
        // silently stop (Python side closed the iterator early).
        let _ = self.sender.send(batch);

        // Re-allocate common buffers with capacity
        for buf in self.entity_bufs.iter_mut() {
            *buf = Vec::with_capacity(BATCH_SIZE);
        }
        self.timestamp_buf = Vec::with_capacity(BATCH_SIZE);
        if self.has_include_bit {
            self.include_bit_buf = Vec::with_capacity(BATCH_SIZE);
        }

        Ok(())
    }

    /// Flush remaining rows and drop the sender (signals end of stream).
    pub fn finish(mut self) -> TTGenResult<u64> {
        self.flush_batch()?;
        // Sender is dropped here, closing the channel.
        Ok(self.rows_written)
    }

    /// Number of rows written so far (including buffered).
    pub fn rows_buffered(&self) -> u64 {
        self.rows_written + self.entity_bufs[0].len() as u64
    }
}

fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
        Value::Str(_) => 0.0,
        Value::List(v) => v.len() as f64,
    }
}

/// Convert a flat buffer of `Value` to the correct Arrow array based on dtype.
/// Used for both ENTITY and LP TARGET columns.
fn flush_value_buf(buf: &mut Vec<Value>, dtype: &DataType) -> Arc<dyn arrow_array::Array> {
    if let DataType::List(inner_field) = dtype {
        let lists = std::mem::take(buf)
            .into_iter()
            .map(|v| match v {
                Value::List(l) => l,
                _ => vec![],
            })
            .collect();
        return flush_value_list_buf(lists, inner_field.data_type());
    }
    flush_entity_buf(buf, dtype)
}

/// Convert a list-of-lists `Value` buffer to a List Arrow array based on dtype.
/// Used for LinkArray TARGET columns.
fn flush_value_list_buf(
    lists: Vec<Vec<Value>>,
    dtype: &DataType,
) -> Arc<dyn arrow_array::Array> {
    let inner_field = Field::new("item", dtype.clone(), true);
    flush_value_list_buf_with_field(lists, &inner_field)
}

fn flush_value_list_buf_with_field(
    lists: Vec<Vec<Value>>,
    inner_field: &Field,
) -> Arc<dyn arrow_array::Array> {
    match inner_field.data_type() {
        DataType::Int64 => {
            let mut builder =
                ListBuilder::new(Int64Builder::new()).with_field(inner_field.clone());
            for list in &lists {
                for v in list {
                    let i = match v {
                        Value::Int(i) => *i,
                        Value::Float(f) => *f as i64,
                        _ => 0,
                    };
                    builder.values().append_value(i);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
        DataType::Float64 => {
            let mut builder = ListBuilder::new(
                arrow_array::builder::Float64Builder::new(),
            )
            .with_field(inner_field.clone());
            for list in &lists {
                for v in list {
                    let f = match v {
                        Value::Float(f) => *f,
                        Value::Int(i) => *i as f64,
                        _ => 0.0,
                    };
                    builder.values().append_value(f);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
        _ => {
            // Utf8 and all other types: convert to string
            let mut builder =
                ListBuilder::new(StringBuilder::new()).with_field(inner_field.clone());
            for list in &lists {
                for v in list {
                    let s = match v {
                        Value::Str(s) => s.clone(),
                        Value::Int(i) => i.to_string(),
                        Value::Float(f) => f.to_string(),
                        Value::Bool(b) => b.to_string(),
                        Value::List(_) => String::new(),
                    };
                    builder.values().append_value(&s);
                }
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
    }
}

/// Convert a buffer of `Value` entities to the correct Arrow array based on dtype.
fn flush_entity_buf(buf: &mut Vec<Value>, dtype: &DataType) -> Arc<dyn arrow_array::Array> {
    let entities = std::mem::take(buf);
    match dtype {
        DataType::Int64 => {
            let values: Vec<i64> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Int(i) => i,
                    Value::Float(f) => f as i64,
                    _ => 0,
                })
                .collect();
            Arc::new(Int64Array::from(values))
        }
        DataType::Float64 => {
            let values: Vec<f64> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    _ => 0.0,
                })
                .collect();
            Arc::new(Float64Array::from(values))
        }
        // Utf8 and all other types: convert to string representation
        _ => {
            let values: Vec<String> = entities
                .into_iter()
                .map(|v| match v {
                    Value::Str(s) => s,
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::List(items) => items
                        .into_iter()
                        .map(|item| match item {
                            Value::Str(s) => s,
                            Value::Int(i) => i.to_string(),
                            Value::Float(f) => f.to_string(),
                            Value::Bool(b) => b.to_string(),
                            Value::List(_) => String::new(),
                        })
                        .collect::<Vec<_>>()
                        .join("|"),
                })
                .collect();
            Arc::new(StringArray::from(
                values.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;
    use diskgraph_core::time::Seconds;

    /// Helper: create a ChannelBatchWriter with a bounded channel and return
    /// both the writer and the receiver for collecting output batches.
    fn test_writer(
        mode: OutputMode,
        include_entity_bit: bool,
        entity_dtype: DataType,
    ) -> (ChannelBatchWriter<Seconds>, crossbeam_channel::Receiver<BatchMessage>) {
        let (sender, receiver) = bounded(64);
        let writer = ChannelBatchWriter::new(
            sender,
            mode,
            include_entity_bit,
            vec![entity_dtype],
            "TIMESTAMP",
            None,
            Vec::new(),
        )
        .unwrap();
        (writer, receiver)
    }

    fn test_writer_with_feats(
        mode: OutputMode,
        entity_dtype: DataType,
        feat_columns: Vec<ExtraColumnOutput>,
    ) -> (ChannelBatchWriter<Seconds>, crossbeam_channel::Receiver<BatchMessage>) {
        let (sender, receiver) = bounded(64);
        let writer = ChannelBatchWriter::new(
            sender,
            mode,
            false,
            vec![entity_dtype],
            "TIMESTAMP",
            None,
            feat_columns,
        )
        .unwrap();
        (writer, receiver)
    }

    /// Drain all batches from the receiver into a Vec (test-only).
    fn collect_batches(receiver: crossbeam_channel::Receiver<BatchMessage>) -> Vec<RecordBatch> {
        receiver.iter().collect()
    }

    #[test]
    fn test_write_and_receive_batches() -> TTGenResult<()> {
        let (mut writer, receiver) = test_writer(
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            false,
            DataType::Int64,
        );

        for i in 0..100i64 {
            writer.write_row(Value::Int(i), Abs::from_raw(i * 100), &Value::Float(i as f64 * 1.5), ExtraValueRef::Scalar(None), true)?;
        }
        let rows = writer.finish()?;
        assert_eq!(rows, 100);

        let batches = collect_batches(receiver);
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);

        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 3); // ENTITY1, TIMESTAMP, TARGET1

        let entities = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let timestamps = batch.column(1).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
        let targets = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

        assert_eq!(entities.value(0), 0);
        assert_eq!(timestamps.value(0), 0); // 0 seconds * 1e9 = 0 ns
        assert_eq!(targets.value(0), 0.0);

        Ok(())
    }

    #[test]
    fn test_write_with_include_bit() -> TTGenResult<()> {
        let (mut writer, receiver) = test_writer(
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            true,
            DataType::Int64,
        );

        writer.write_row(Value::Int(0), Abs::from_raw(100), &Value::Float(1.0), ExtraValueRef::Scalar(None), true)?;
        writer.write_row(Value::Int(1), Abs::from_raw(100), &Value::Float(2.0), ExtraValueRef::Scalar(None), false)?;
        writer.write_row(Value::Int(2), Abs::from_raw(100), &Value::Float(3.0), ExtraValueRef::Scalar(None), true)?;
        let rows = writer.finish()?;
        assert_eq!(rows, 3);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 4); // ENTITY1, TIMESTAMP, TARGET1, ENTITY_INCLUDE_BIT

        let bits = batch.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(bits.value(0));   // entity 0: true
        assert!(!bits.value(1));  // entity 1: false
        assert!(bits.value(2));   // entity 2: true

        Ok(())
    }

    #[test]
    fn test_batch_flush() -> TTGenResult<()> {
        let (mut writer, receiver) = test_writer(
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            false,
            DataType::Int64,
        );

        // Write more than BATCH_SIZE rows to trigger auto-flush
        for i in 0..(BATCH_SIZE as i64 + 100) {
            writer.write_row(Value::Int(i), Abs::from_raw(i), &Value::Float(i as f64), ExtraValueRef::Scalar(None), true)?;
        }
        let rows = writer.finish()?;
        assert_eq!(rows, BATCH_SIZE as u64 + 100);

        let batches = collect_batches(receiver);
        // Should have at least 2 batches (one full + one partial)
        assert!(batches.len() >= 2);
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, BATCH_SIZE + 100);

        Ok(())
    }

    #[test]
    fn test_write_array() -> TTGenResult<()> {
        use arrow_array::{Array, ListArray};

        let (mut writer, receiver) = test_writer(
            OutputMode::LinkArray { include_frequency: true, target_dtypes: vec![DataType::Utf8] },
            false,
            DataType::Int64,
        );

        writer.write_row_array(
            Value::Int(0),
            Abs::from_raw(100),
            vec![Value::Str("Nike".into()), Value::Str("Adidas".into())],
            &[],
            &[3, 1],
            ExtraValueRef::Scalar(None),
            true,
        )?;
        writer.write_row_array(
            Value::Int(1),
            Abs::from_raw(200),
            vec![Value::Str("Puma".into())],
            &[],
            &[2],
            ExtraValueRef::Scalar(None),
            true,
        )?;
        let rows = writer.finish()?;
        assert_eq!(rows, 2);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 4); // ENTITY1, TIMESTAMP, TARGET1(list), FREQUENCY(list)

        let target_list = batch.column(2).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(target_list.len(), 2); // 2 rows

        let freq_list = batch.column(3).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(freq_list.len(), 2);

        Ok(())
    }

    #[test]
    fn test_write_multi_target_link_array() -> TTGenResult<()> {
        use arrow_array::{Array, ListArray};

        let (mut writer, receiver) = test_writer(
            OutputMode::LinkArray {
                include_frequency: false,
                target_dtypes: vec![DataType::Utf8, DataType::Int64],
            },
            false,
            DataType::Int64,
        );

        writer.write_row_array(
            Value::Int(0),
            Abs::from_raw(100),
            vec![Value::Str("Nike".into()), Value::Str("Adidas".into())],
            &[vec![Value::Int(1), Value::Int(2)]],
            &[],
            ExtraValueRef::Scalar(None),
            true,
        )?;
        writer.write_row_array(
            Value::Int(1),
            Abs::from_raw(200),
            vec![Value::Str("Puma".into())],
            &[vec![Value::Int(9)]],
            &[],
            ExtraValueRef::Scalar(None),
            true,
        )?;
        let rows = writer.finish()?;
        assert_eq!(rows, 2);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        // ENTITY1, TIMESTAMP, TARGET1(list<Utf8>), TARGET2(list<Int64>)
        assert_eq!(batch.num_columns(), 4);
        let t1 = batch.column(2).as_any().downcast_ref::<ListArray>().unwrap();
        let t2 = batch.column(3).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(t1.len(), 2);
        assert_eq!(t2.len(), 2);
        Ok(())
    }

    #[test]
    fn test_write_multi_target_forecasting() -> TTGenResult<()> {
        let ar_dtype = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let (mut writer, receiver) = test_writer(
            OutputMode::Forecasting {
                target_dtypes: vec![DataType::Float64, DataType::Int64],
                autoregression_dtype: ar_dtype,
                future_labels_dtype: None,
                yoy_dtype: None,
            },
            false,
            DataType::Int64,
        );

        writer.write_row_forecasting_wide(
            &[Value::Int(0)],
            Abs::from_raw(100),
            Some(&Value::Float(1.5)),
            &[Value::Int(10)],
            vec![Value::Float(0.1), Value::Float(0.2)],
            None,
            None,
            ExtraValueRef::Scalar(None),
            true,
        )?;
        writer.write_row_forecasting_wide(
            &[Value::Int(1)],
            Abs::from_raw(200),
            Some(&Value::Float(2.5)),
            &[Value::Int(20)],
            vec![Value::Float(0.3)],
            None,
            None,
            ExtraValueRef::Scalar(None),
            true,
        )?;
        let rows = writer.finish()?;
        assert_eq!(rows, 2);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        // ENTITY1, TIMESTAMP, TARGET1, TARGET2, autoregression_labels
        assert_eq!(batch.num_columns(), 5);
        let t1 = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let t2 = batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(t1.value(0), 1.5);
        assert_eq!(t2.value(0), 10);
        Ok(())
    }

    #[test]
    fn test_write_forecasting_preserves_list_child_field_name() -> TTGenResult<()> {
        let ar_dtype =
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true)));
        let (mut writer, receiver) = test_writer(
            OutputMode::Forecasting {
                target_dtypes: vec![DataType::Int64],
                autoregression_dtype: ar_dtype,
                future_labels_dtype: None,
                yoy_dtype: None,
            },
            false,
            DataType::Int64,
        );

        writer.write_row_forecasting_wide(
            &[Value::Int(0)],
            Abs::from_raw(100),
            Some(&Value::Int(1)),
            &[],
            vec![Value::Int(10), Value::Int(11)],
            None,
            None,
            ExtraValueRef::Scalar(None),
            true,
        )?;
        let rows = writer.finish()?;
        assert_eq!(rows, 1);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        let schema = batch.schema();
        let ar_field = schema.field_with_name(AUTOREGRESSION_OUTPUT_ALIAS).unwrap();
        let DataType::List(inner_field) = ar_field.data_type() else {
            panic!("expected autoregression_labels to be a List");
        };
        assert_eq!(inner_field.name(), "element");
        assert_eq!(inner_field.data_type(), &DataType::Int64);
        Ok(())
    }

    #[test]
    fn test_write_feat_columns_vec() -> TTGenResult<()> {
        use arrow_array::{Array, ListArray};

        let lag_dtype = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let feat_cols = vec![
            ExtraColumnOutput { name: "FEAT1".to_string(), dtype: DataType::Float64 },
            ExtraColumnOutput { name: "LAG_FEAT1".to_string(), dtype: lag_dtype },
        ];
        let (mut writer, receiver) = test_writer_with_feats(
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            DataType::Int64,
            feat_cols,
        );

        for i in 0..3i64 {
            writer.write_row(
                Value::Int(i),
                Abs::from_raw(i * 100),
                &Value::Float(i as f64),
                ExtraValueRef::Scalar(None),
                true,
            )?;
            writer.push_feat_values(&[
                Some(Value::Float(i as f64 * 10.0)),
                Some(Value::List(vec![Value::Float(i as f64), Value::Float(i as f64 + 0.5)])),
            ])?;
        }
        let rows = writer.finish()?;
        assert_eq!(rows, 3);

        let batches = collect_batches(receiver);
        let batch = &batches[0];
        // ENTITY1, TIMESTAMP, TARGET1, FEAT1, LAG_FEAT1
        assert_eq!(batch.num_columns(), 5);
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["ENTITY1", "TIMESTAMP", "TARGET1", "FEAT1", "LAG_FEAT1"]);

        let feat = batch.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(feat.value(0), 0.0);
        assert_eq!(feat.value(2), 20.0);

        let lag = batch.column(4).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(lag.len(), 3);
        Ok(())
    }

    // A non-weight column named `WEIGHT` (no `weight_col`, so
    // `enable_weight_buf` is never called) must be rejected at
    // `ChannelBatchWriter::new`, not later misread as `task.weight`.

    /// A feat column named `WEIGHT` (no weight buf enabled) → hard error.
    #[test]
    fn test_feat_column_named_weight_rejected() {
        let (sender, _receiver) = bounded::<BatchMessage>(64);
        let result = ChannelBatchWriter::<Seconds>::new(
            sender,
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            false,
            vec![DataType::Int64],
            "TIMESTAMP",
            None,
            vec![ExtraColumnOutput {
                name: WEIGHT_COL.to_string(),
                dtype: DataType::Float64,
            }],
        );
        match result {
            Err(err @ TTGenError::InvalidPlan(_)) => {
                assert!(
                    format!("{err}").contains(WEIGHT_COL),
                    "error should name the reserved column: {err}"
                );
            }
            Err(other) => panic!("expected InvalidPlan, got {other:?}"),
            Ok(_) => panic!("a feature column named WEIGHT must be rejected"),
        }
    }

    /// An extra (projection) column named `WEIGHT` → hard error too.
    #[test]
    fn test_extra_column_named_weight_rejected() {
        let (sender, _receiver) = bounded::<BatchMessage>(64);
        let result = ChannelBatchWriter::<Seconds>::new(
            sender,
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            false,
            vec![DataType::Int64],
            "TIMESTAMP",
            Some(ExtraColumnOutput {
                name: WEIGHT_COL.to_string(),
                dtype: DataType::Float64,
            }),
            Vec::new(),
        );
        assert!(
            matches!(result, Err(TTGenError::InvalidPlan(_))),
            "an extra column named WEIGHT must be rejected"
        );
    }

    /// The legit case: a writer with NO column named `WEIGHT` builds fine,
    /// and preserves a Float64 weight without narrowing it through Float32.
    #[test]
    fn test_weight_buf_legit_case_still_works() -> TTGenResult<()> {
        let (mut writer, receiver) = test_writer(
            OutputMode::Node { target_dtypes: vec![DataType::Float64] },
            false,
            DataType::Int64,
        );
        writer.enable_weight_buf()?;
        assert!(writer.schema.fields().iter().any(|f| f.name() == WEIGHT_COL));
        let weight = f64::from(f32::MAX) * 2.0;
        writer.push_weight(weight);
        writer.write_row(
            Value::Int(1),
            Abs::from_raw(0),
            &Value::Float(1.0),
            ExtraValueRef::Scalar(None),
            true,
        )?;
        writer.finish()?;
        let batches = collect_batches(receiver);
        let weights = batches[0]
            .column_by_name(WEIGHT_COL)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(weights.value(0), weight);
        Ok(())
    }
}
