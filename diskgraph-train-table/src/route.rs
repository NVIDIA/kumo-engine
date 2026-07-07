//! Pure, per-batch routing kernels for the fused ttgen pass-B.
//!
//! Each kernel takes a `RecordBatch` and returns `[RecordBatch; 3]`
//! (`0`=train, `1`=val, `2`=test). Kernels are PURE: no IO, no parallelism,
//! no global state. Pass-B orchestration calls one kernel per batch and
//! streams the sub-batches to the per-stage `ArrowWriter`s.
//!
//! Two stage sources, one per kernel:
//!
//! - [`route_per_anchor`] — a [`SplitDecider`] decides each whole row's stage.
//!   (Array-LP folds in the empty-target drop: anchor rows with an empty
//!   `TARGET1` list go nowhere. The custom-TT split-column path also projects
//!   the split column out of the output sub-batches.)
//! - [`route_per_link`] — `RandomLinkSplit` / `CustomLinkSplit` over an array
//!   `TARGET1`: each link is assigned a stage, links are bucketed into ≤3
//!   sub-lists, and one output row per non-empty `(anchor, stage)` is emitted.
//!
use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::builder::*;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, ListArray, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_select::filter::filter_record_batch;
use diskgraph_core::time::UnitSystem;
use diskgraph_core::tt_builder::{Value, PRIMARY_ENTITY_COL, PRIMARY_TARGET_COL};

use crate::errors::{TTGenError, TTGenResult};
use crate::pass_b::{INCLUDE_MASK_COL, TIMESTAMP_COL};
use crate::split_decider::{splitmix64, hash_value, SplitDecider, UNUSED_STAGE};

const FREQUENCY_COL: &str = "__kumo_internal_frequency_column__";
/// STATIC LP past-visit exclude key-list column emitted by `route_per_link` on
/// every stage (empty on train; prior-stage target union on val/test). Ingest
/// resolves it into the `exclude_node_ids` rowpack column.
const EXCLUDE_TARGET_KEYS_COL: &str = "exclude_target_keys";

#[derive(Clone, Copy, Debug)]
pub(crate) struct ForecastingHorizonPolicy {
    stage_max_ts: [Option<i64>; 3],
    step_ns: i64,
    length: i64,
}

impl ForecastingHorizonPolicy {
    pub(crate) fn new(stage_max_ts: [Option<i64>; 3], step_ns: i64, length: i64) -> Option<Self> {
        if length <= 1 || step_ns <= 0 {
            return None;
        }
        Some(Self { stage_max_ts, step_ns, length })
    }

    #[inline]
    fn keep(self, stage: u8, ts_ns: i64) -> bool {
        let Some(max_ts) = self.stage_max_ts.get(stage as usize).and_then(|v| *v) else {
            return false;
        };
        let tail = self.step_ns.saturating_mul(self.length - 1);
        let adjusted_max = max_ts.saturating_sub(tail);
        if stage == 0 {
            return ts_ns <= adjusted_max;
        }
        let period = self.step_ns.saturating_mul(self.length);
        period > 0 && (ts_ns - adjusted_max).rem_euclid(period) == 0
    }
}

// ─── Kernel 1: per-anchor (decider) ─────────────────────────────────────────

/// Route each row of `batch` to a stage by `decider`, returning three
/// sub-batches `[train, val, test]`.
///
/// Per row, the stage is:
/// - `UNUSED` if `INCLUDE_MASK_COL` (when present) is `false`;
/// - `UNUSED` if `is_array_lp` and the `TARGET1` list is empty/null;
/// - otherwise `decider.decide_for_view(view, row)`.
///
/// Stages outside `{0, 1, 2}` route nowhere. `global_offset` is the global
/// row index of `batch[0]` (used by global-index deciders like `FixedDecider`).
///
/// `drop_split_column` is `Some(col)` for the custom-TT split-column path: that
/// column is the split signal and is projected OUT of every returned sub-batch
/// (the decider has already consumed it from the input batch) so it never
/// reaches the seed rowpack / features. The decider reads from `batch`, which
/// still carries the column; only the OUTPUT sub-batches have it removed.
pub(crate) fn route_per_anchor<U: UnitSystem, D: SplitDecider<U>>(
    batch: &RecordBatch,
    decider: &D,
    global_offset: u64,
    is_array_lp: bool,
    horizon_policy: Option<&ForecastingHorizonPolicy>,
    drop_split_column: Option<&str>,
) -> TTGenResult<[RecordBatch; 3]> {
    let stages = assign_per_anchor_stages(
        batch,
        decider,
        global_offset,
        is_array_lp,
        horizon_policy,
    )?;
    let mut masks: [Vec<bool>; 3] = [
        vec![false; batch.num_rows()],
        vec![false; batch.num_rows()],
        vec![false; batch.num_rows()],
    ];
    for (r, stage) in stages.into_iter().enumerate() {
        if (stage as usize) < 3 {
            masks[stage as usize][r] = true;
        }
    }
    // Drop the split column from the OUTPUT (the decider already consumed it).
    match drop_split_column {
        Some(col) => {
            let projected = project_out_column(batch, col)?;
            filter_three(&projected, &masks)
        }
        None => filter_three(batch, &masks),
    }
}

pub(crate) fn assign_per_anchor_stages<U: UnitSystem, D: SplitDecider<U>>(
    batch: &RecordBatch,
    decider: &D,
    global_offset: u64,
    is_array_lp: bool,
    horizon_policy: Option<&ForecastingHorizonPolicy>,
) -> TTGenResult<Vec<u8>> {
    let nrows = batch.num_rows();
    let view = decider.batch_view(batch, global_offset)?;

    // Optional include-mask column (downcast hoisted out of the row loop).
    let mask_arr: Option<&BooleanArray> = match batch.schema_ref().column_with_name(INCLUDE_MASK_COL)
    {
        Some((col_idx, _)) => Some(
            batch
                .column(col_idx)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    TTGenError::SystemError(format!(
                        "route_per_anchor: include mask column {INCLUDE_MASK_COL:?} must be Boolean"
                    ))
                })?,
        ),
        None => None,
    };

    // Optional TARGET1 list (only needed for the array-LP empty-target drop).
    let target_list: Option<&ListArray> = if is_array_lp {
        match batch.schema_ref().column_with_name(PRIMARY_TARGET_COL) {
            Some((col_idx, _)) => Some(
                batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| {
                        TTGenError::SystemError(
                            "route_per_anchor: TARGET1 must be ListArray for is_array_lp".to_string(),
                        )
                    })?,
            ),
            None => {
                return Err(TTGenError::SystemError(
                    "route_per_anchor: is_array_lp set but TARGET1 column missing".to_string(),
                ))
            }
        }
    } else {
        None
    };

    let ts_arr: Option<&TimestampNanosecondArray> = if horizon_policy.is_some() {
        let (idx, _) = batch.schema_ref().column_with_name(TIMESTAMP_COL).ok_or_else(|| {
            TTGenError::SystemError(
                "assign_per_anchor_stages: TIMESTAMP column missing".to_string(),
            )
        })?;
        Some(
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .ok_or_else(|| {
                    TTGenError::SystemError(
                        "assign_per_anchor_stages: TIMESTAMP must be TimestampNanosecondArray"
                            .to_string(),
                    )
                })?,
        )
    } else {
        None
    };

    let mut stages = Vec::with_capacity(nrows);

    for r in 0..nrows {
        let mut stage = if mask_arr.map(|m| !m.value(r)).unwrap_or(false) {
            UNUSED_STAGE
        } else if let Some(list) = target_list {
            // Array-LP empty-target drop: null or zero-length list → UNUSED.
            let offsets = list.value_offsets();
            let non_empty = !list.is_null(r) && (offsets[r + 1] - offsets[r]) > 0;
            if non_empty {
                decider.decide_for_view(&view, r)
            } else {
                UNUSED_STAGE
            }
        } else {
            decider.decide_for_view(&view, r)
        };
        if let (Some(policy), Some(ts)) = (horizon_policy, ts_arr) {
            if (stage as usize) < 3 && !policy.keep(stage, ts.value(r)) {
                stage = UNUSED_STAGE;
            }
        }
        stages.push(stage);
    }

    Ok(stages)
}

/// Return `batch` with the column named `col` removed. No-op clone if absent.
fn project_out_column(batch: &RecordBatch, col: &str) -> TTGenResult<RecordBatch> {
    let Some((idx, _)) = batch.schema_ref().column_with_name(col) else {
        return Ok(batch.clone());
    };
    let fields: Vec<Field> = batch
        .schema_ref()
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, f)| f.as_ref().clone())
        .collect();
    let columns: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, c)| c.clone())
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| TTGenError::SystemError(format!("route: project-out {col}: {e}")))
}

/// Filter `batch` into three sub-batches by the three boolean masks.
fn filter_three(batch: &RecordBatch, masks: &[Vec<bool>; 3]) -> TTGenResult<[RecordBatch; 3]> {
    let mut out: Vec<RecordBatch> = Vec::with_capacity(3);
    for mask in masks {
        let barr = BooleanArray::from(mask.clone());
        let filtered = filter_record_batch(batch, &barr)
            .map_err(|e| TTGenError::SystemError(format!("route: filter failed: {e}")))?;
        out.push(filtered);
    }
    let test = out.pop().unwrap();
    let val = out.pop().unwrap();
    let train = out.pop().unwrap();
    Ok([train, val, test])
}

// ─── Kernel 2: per-link (array LP RandomLink / CustomLink) ───────────────────

/// Per-link stage assigner for the array-LP `route_per_link` kernel.
///
/// - `RandomLink`: each link's stage = `stage_for_link(entity, ts_secs,
///   link_idx, seed, &cutoffs)` (content-hash bucketing, batch/position
///   independent).
/// - `CustomLink`: each link's stage = membership of the parallel
///   `split_col` link value in the train/val/test value sets (else UNUSED).
#[derive(Debug, Clone)]
pub(crate) enum LinkAssigner {
    RandomLink {
        cutoffs: Vec<f64>,
        seed: u64,
    },
    CustomLink {
        train: HashSet<String>,
        val: HashSet<String>,
        test: HashSet<String>,
        split_col: String,
    },
}

impl LinkAssigner {
    /// Convenience constructor: build a `RandomLink` assigner from a raw
    /// ratio (e.g. `[0.8, 0.1, 0.1]`) + seed.
    pub fn random_from_ratio(ratio: &[f64], seed: u64) -> Self {
        LinkAssigner::RandomLink {
            cutoffs: ratio_to_cumulative_cutoffs(ratio),
            seed,
        }
    }
}

/// Route the array `TARGET1` of each anchor row in `batch` per-link,
/// returning three sub-batches `[train, val, test]`.
///
/// For each anchor row, walk its `TARGET1` (and parallel `FREQUENCY`) list;
/// assign each link a stage (see [`LinkAssigner`]); bucket the links into
/// ≤3 sub-lists; for each non-empty `(row, stage)` emit one output row with
/// the scalar columns copied and `TARGET1` / `FREQUENCY` set to the
/// sub-list. Rows whose list is empty, or whose every link is UNUSED,
/// contribute nothing.
///
/// Output schema:
/// - `RandomLink`: identical to the input schema.
/// - `CustomLink`: input schema MINUS the `split_col` column (the split
///   projection is consumed by the assignment, not carried forward).
pub(crate) fn route_per_link(
    batch: &RecordBatch,
    assigner: &LinkAssigner,
) -> TTGenResult<[RecordBatch; 3]> {
    let exploded = collect_link_array_from_batch(batch, assigner.split_col_name())?;

    let mut train = StageBuffers::new();
    let mut val = StageBuffers::new();
    let mut test = StageBuffers::new();

    let cutoffs = match assigner {
        LinkAssigner::RandomLink { cutoffs, .. } => cutoffs.as_slice(),
        LinkAssigner::CustomLink { .. } => &[],
    };

    for row_idx in 0..exploded.row_targets.len() {
        let targets = &exploded.row_targets[row_idx];
        let freqs = &exploded.row_freqs[row_idx];
        // Per-stage sub-lists for THIS anchor row.
        let mut sub: [(Vec<ScalarCell>, Vec<i64>); 3] = [
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
        ];
        for link_idx in 0..targets.len() {
            let stage = match assigner {
                LinkAssigner::RandomLink { seed, .. } => {
                    let ts_secs = exploded.row_anchor_ts[row_idx] / 1_000_000_000;
                    let entity = &exploded.row_anchor_entity[row_idx];
                    stage_for_link(entity, ts_secs, link_idx, *seed, cutoffs)
                }
                LinkAssigner::CustomLink {
                    train, val, test, ..
                } => {
                    let values = exploded.link_split_values.as_ref().ok_or_else(|| {
                        TTGenError::SystemError(
                            "route_per_link: CustomLink requires split_col values".to_string(),
                        )
                    })?;
                    let v = values[row_idx][link_idx].as_str();
                    if train.contains(v) {
                        0
                    } else if val.contains(v) {
                        1
                    } else if test.contains(v) {
                        2
                    } else {
                        UNUSED_STAGE
                    }
                }
            };
            if (stage as usize) < 3 {
                let s = stage as usize;
                sub[s].0.push(targets[link_idx].clone());
                sub[s].1.push(freqs[link_idx]);
            }
        }
        // STATIC LP past-visit exclusion: prior-stage target unions for THIS
        // anchor. val excludes TRAIN (sub[0]); test excludes TRAIN∪VAL
        // (sub[0]∪sub[1], deduped). train excludes nothing. Disjoint from the
        // current stage's positives by construction (each link → one stage).
        let val_exclude = dedup_cells(&sub[0].0);
        let test_exclude = dedup_cells(&[&sub[0].0[..], &sub[1].0[..]].concat());
        let mut stage_excludes = [Vec::new(), val_exclude, test_exclude];
        for (stage, buf) in [&mut train, &mut val, &mut test].into_iter().enumerate() {
            if sub[stage].0.is_empty() {
                continue;
            }
            buf.push_row(
                row_idx,
                std::mem::take(&mut sub[stage].0),
                std::mem::take(&mut sub[stage].1),
                std::mem::take(&mut stage_excludes[stage]),
            );
        }
    }

    let out_schema = exploded.output_schema();
    let train_b = exploded.build_stage_batch(&out_schema, &train)?;
    let val_b = exploded.build_stage_batch(&out_schema, &val)?;
    let test_b = exploded.build_stage_batch(&out_schema, &test)?;
    Ok([train_b, val_b, test_b])
}

/// Per-link pass-B output schema for a given tt-parquet `tt_schema`: drop the
/// optional CustomLink `split_col`, then append the past-visit
/// `exclude_target_keys` column (same List dtype as TARGET1). The executor uses
/// this for empty-stage 0-row parts so every stage part shares one schema.
pub(crate) fn per_link_output_schema(
    tt_schema: &SchemaRef,
    split_col_name: Option<&str>,
) -> SchemaRef {
    let target_idx = tt_schema.index_of(PRIMARY_TARGET_COL).ok();
    let mut fields: Vec<Field> = tt_schema
        .fields()
        .iter()
        .filter(|f| Some(f.name().as_str()) != split_col_name)
        .map(|f| f.as_ref().clone())
        .collect();
    if let Some(ti) = target_idx {
        let target_dtype = tt_schema.field(ti).data_type().clone();
        fields.push(Field::new(EXCLUDE_TARGET_KEYS_COL, target_dtype, true));
    }
    Arc::new(Schema::new(fields))
}

impl LinkAssigner {
    fn split_col_name(&self) -> Option<&str> {
        match self {
            LinkAssigner::RandomLink { .. } => None,
            LinkAssigner::CustomLink { split_col, .. } => Some(split_col.as_str()),
        }
    }
}

/// Per-stage accumulation for one batch: which source row each output row
/// came from, plus its TARGET / FREQUENCY sub-lists.
struct StageBuffers {
    source_row_idxs: Vec<usize>,
    targets: Vec<Vec<ScalarCell>>,
    freqs: Vec<Vec<i64>>,
    /// STATIC LP past-visit exclusion key-list per output row (same item dtype
    /// as `targets`); see the exclude-union logic in `route_per_link`.
    excludes: Vec<Vec<ScalarCell>>,
}

impl StageBuffers {
    fn new() -> Self {
        Self {
            source_row_idxs: Vec::new(),
            targets: Vec::new(),
            freqs: Vec::new(),
            excludes: Vec::new(),
        }
    }
    fn push_row(
        &mut self,
        src: usize,
        targets: Vec<ScalarCell>,
        freqs: Vec<i64>,
        excludes: Vec<ScalarCell>,
    ) {
        self.source_row_idxs.push(src);
        self.targets.push(targets);
        self.freqs.push(freqs);
        self.excludes.push(excludes);
    }
}

/// Per-batch exploded link data (the per-batch analogue of regroup.rs's
/// `ExplodedLinkData`). Built once per batch by
/// [`collect_link_array_from_batch`], consumed by [`route_per_link`].
struct BatchExplodedLinkData {
    /// `row_targets[row_idx]` = the list of TARGET cells for that anchor row.
    row_targets: Vec<Vec<ScalarCell>>,
    /// `row_freqs[row_idx]` = the frequency values for that row (`1`s when no
    /// frequency column is present).
    row_freqs: Vec<Vec<i64>>,
    /// `row_scalar_cols[col_idx][row_idx]` = the scalar passthrough value at
    /// that input column for that row. Empty for the target/freq/split cols.
    row_scalar_cols: Vec<Vec<ScalarCell>>,
    /// `row_anchor_ts[row_idx]` — anchor timestamp in epoch NANOSECONDS.
    row_anchor_ts: Vec<i64>,
    /// `row_anchor_entity[row_idx]` — the source row's ENTITY1 value.
    row_anchor_entity: Vec<Value>,
    /// `link_split_values[row_idx][link_idx]` — per-link split value (only
    /// for `CustomLink`).
    link_split_values: Option<Vec<Vec<String>>>,
    schema: SchemaRef,
    target_idx: usize,
    freq_idx: Option<usize>,
    split_col_idx: Option<usize>,
}

impl BatchExplodedLinkData {
    /// See [`per_link_output_schema`]. Each output batch is a single stage's
    /// rows, so there is no inline stage column.
    fn output_schema(&self) -> SchemaRef {
        let split_col_name = self.split_col_idx.map(|i| self.schema.field(i).name().as_str());
        per_link_output_schema(&self.schema, split_col_name)
    }

    /// Build one stage's output `RecordBatch` from its accumulated buffers.
    fn build_stage_batch(
        &self,
        out_schema: &SchemaRef,
        buf: &StageBuffers,
    ) -> TTGenResult<RecordBatch> {
        let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(out_schema.fields().len());
        for (col_idx, field) in self.schema.fields().iter().enumerate() {
            if Some(col_idx) == self.split_col_idx {
                continue;
            }
            let col: Arc<dyn Array> = if col_idx == self.target_idx {
                build_list_array_from_cells(field.data_type(), &buf.targets, PRIMARY_TARGET_COL)?
            } else if Some(col_idx) == self.freq_idx {
                if !matches!(field.data_type(), DataType::List(_)) {
                    return Err(TTGenError::SystemError(
                        "route_per_link: frequency column must be List type".to_string(),
                    ));
                }
                let mut b = ListBuilder::new(Int64Builder::new());
                for row in &buf.freqs {
                    for f in row {
                        b.values().append_value(*f);
                    }
                    b.append(true);
                }
                Arc::new(b.finish())
            } else {
                let cells: Vec<ScalarCell> = buf
                    .source_row_idxs
                    .iter()
                    .map(|src| self.row_scalar_cols[col_idx][*src].clone())
                    .collect();
                build_scalar_array_from_cells(field.data_type(), &cells, field.name())?
            };
            columns.push(col);
        }
        // Append the past-visit `exclude_target_keys` column (same List dtype as
        // TARGET1).
        let exclude_dtype = self.schema.field(self.target_idx).data_type();
        columns.push(build_list_array_from_cells(
            exclude_dtype,
            &buf.excludes,
            EXCLUDE_TARGET_KEYS_COL,
        )?);
        RecordBatch::try_new(out_schema.clone(), columns)
            .map_err(|e| TTGenError::SystemError(format!("route_per_link: build batch: {e}")))
    }
}

/// Explode one batch's array TARGET column into per-row link data. Mirrors
/// regroup.rs's `collect_link_array_from_parquet` but operates on a single
/// in-memory `RecordBatch`.
#[allow(clippy::needless_range_loop)]
fn collect_link_array_from_batch(
    batch: &RecordBatch,
    split_col_name: Option<&str>,
) -> TTGenResult<BatchExplodedLinkData> {
    let schema = batch.schema();

    let target_idx = schema
        .index_of(PRIMARY_TARGET_COL)
        .map_err(|_| TTGenError::SystemError("route_per_link: requires TARGET1 column".to_string()))?;
    if !matches!(schema.field(target_idx).data_type(), DataType::List(_)) {
        return Err(TTGenError::SystemError(
            "route_per_link: TARGET1 must be List type".to_string(),
        ));
    }
    let freq_idx = schema.index_of(FREQUENCY_COL).ok();

    let split_col_idx = if let Some(name) = split_col_name {
        let idx = schema.index_of(name).map_err(|_| {
            TTGenError::SystemError(format!(
                "route_per_link: split projection column {name:?} not found"
            ))
        })?;
        if !matches!(schema.field(idx).data_type(), DataType::List(_)) {
            return Err(TTGenError::SystemError(format!(
                "route_per_link: split projection column {name:?} must be List type, got {:?}",
                schema.field(idx).data_type()
            )));
        }
        Some(idx)
    } else {
        None
    };

    let ts_idx = schema.index_of(TIMESTAMP_COL).map_err(|_| {
        TTGenError::SystemError("route_per_link: requires TIMESTAMP column".to_string())
    })?;
    let entity_idx = schema.index_of(PRIMARY_ENTITY_COL).map_err(|_| {
        TTGenError::SystemError("route_per_link: requires ENTITY1 column".to_string())
    })?;

    let target_list = batch
        .column(target_idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| TTGenError::SystemError("route_per_link: TARGET1 must be ListArray".to_string()))?;
    let freq_list = if let Some(fi) = freq_idx {
        Some(
            batch
                .column(fi)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| {
                    TTGenError::SystemError("route_per_link: frequency column must be ListArray".to_string())
                })?,
        )
    } else {
        None
    };
    let split_list = if let Some(sci) = split_col_idx {
        Some(
            batch
                .column(sci)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| {
                    TTGenError::SystemError(
                        "route_per_link: split projection column must be ListArray".to_string(),
                    )
                })?,
        )
    } else {
        None
    };
    let ts_arr = batch
        .column(ts_idx)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| {
            TTGenError::SystemError(
                "route_per_link: TIMESTAMP column must be TimestampNanosecondArray".to_string(),
            )
        })?;
    let entity_col = batch.column(entity_idx).as_ref();

    let nrows = batch.num_rows();
    let mut row_targets: Vec<Vec<ScalarCell>> = Vec::with_capacity(nrows);
    let mut row_freqs: Vec<Vec<i64>> = Vec::with_capacity(nrows);
    let mut row_scalar_cols: Vec<Vec<ScalarCell>> = vec![Vec::new(); schema.fields().len()];
    let mut row_anchor_ts: Vec<i64> = Vec::with_capacity(nrows);
    let mut row_anchor_entity: Vec<Value> = Vec::with_capacity(nrows);
    let mut link_split_values: Option<Vec<Vec<String>>> =
        if split_col_idx.is_some() { Some(Vec::new()) } else { None };

    for i in 0..nrows {
        let values = target_list.values();
        let start = target_list.value_offsets()[i] as usize;
        let end = target_list.value_offsets()[i + 1] as usize;
        let mut targets = Vec::with_capacity(end - start);
        for j in start..end {
            targets.push(scalar_from_array_at(values.as_ref(), j, "TARGET.item")?);
        }

        let freqs = if let Some(flist) = freq_list {
            let fvals = flist.values();
            let farr = fvals.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                TTGenError::SystemError("route_per_link: frequency list items must be Int64".to_string())
            })?;
            let fs = flist.value_offsets()[i] as usize;
            let fe = flist.value_offsets()[i + 1] as usize;
            let mut fv = Vec::with_capacity(fe - fs);
            for j in fs..fe {
                fv.push(if farr.is_null(j) { 1 } else { farr.value(j) });
            }
            if fv.len() != targets.len() {
                return Err(TTGenError::SystemError(format!(
                    "route_per_link: frequency/target length mismatch at row {i}: freq={} target={}",
                    fv.len(),
                    targets.len()
                )));
            }
            fv
        } else {
            vec![1; targets.len()]
        };

        if let (Some(slist), Some(out)) = (split_list, link_split_values.as_mut()) {
            let svals = slist.values();
            let ss = slist.value_offsets()[i] as usize;
            let se = slist.value_offsets()[i + 1] as usize;
            if se - ss != targets.len() {
                return Err(TTGenError::SystemError(format!(
                    "route_per_link: split-column length mismatch with TARGET at row {i}: split={} target={}",
                    se - ss,
                    targets.len()
                )));
            }
            let mut per_row: Vec<String> = Vec::with_capacity(se - ss);
            for j in ss..se {
                let cell = scalar_from_array_at(svals.as_ref(), j, "split_col.item")?;
                per_row.push(match cell {
                    ScalarCell::Str(s) => s,
                    ScalarCell::Int(v) => v.to_string(),
                    ScalarCell::Float(v) => v.to_string(),
                    ScalarCell::Bool(v) => v.to_string(),
                    ScalarCell::Ts(v) => v.to_string(),
                    ScalarCell::Null => "None".to_string(),
                });
            }
            out.push(per_row);
        }

        row_targets.push(targets);
        row_freqs.push(freqs);
        row_anchor_ts.push(ts_arr.value(i));
        row_anchor_entity.push(read_entity_value_at(entity_col, i)?);

        for col_idx in 0..schema.fields().len() {
            if col_idx == target_idx || Some(col_idx) == freq_idx || Some(col_idx) == split_col_idx {
                continue;
            }
            let field = schema.field(col_idx);
            if matches!(field.data_type(), DataType::List(_)) {
                return Err(TTGenError::SystemError(format!(
                    "route_per_link: unsupported passthrough list column: {}",
                    field.name()
                )));
            }
            row_scalar_cols[col_idx].push(scalar_from_array_at(
                batch.column(col_idx).as_ref(),
                i,
                field.name(),
            )?);
        }
    }

    Ok(BatchExplodedLinkData {
        row_targets,
        row_freqs,
        row_scalar_cols,
        row_anchor_ts,
        row_anchor_entity,
        link_split_values,
        schema,
        target_idx,
        freq_idx,
        split_col_idx,
    })
}

// ─── ScalarCell + array builders ───────────────────────────────────────────

/// Cell-typed clone of an Arrow scalar value at a single `(array, idx)`.
#[derive(Clone, Debug)]
enum ScalarCell {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Ts(i64),
    Null,
}

/// Order-preserving dedup of target key cells by canonical value (used for the
/// past-visit exclude set). PK key dtypes are Int64 / Utf8 in practice; other
/// scalar dtypes are rendered to a stable string for the dedup key only.
fn dedup_cells(cells: &[ScalarCell]) -> Vec<ScalarCell> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(cells.len());
    for c in cells {
        let key = match c {
            ScalarCell::Int(v) => format!("i{v}"),
            ScalarCell::Str(v) => format!("s{v}"),
            ScalarCell::Float(v) => format!("f{}", v.to_bits()),
            ScalarCell::Bool(v) => format!("b{v}"),
            ScalarCell::Ts(v) => format!("t{v}"),
            ScalarCell::Null => "n".to_string(),
        };
        if seen.insert(key) {
            out.push(c.clone());
        }
    }
    out
}

fn scalar_from_array_at(arr: &dyn Array, idx: usize, ctx: &str) -> TTGenResult<ScalarCell> {
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok(if a.is_null(idx) { ScalarCell::Null } else { ScalarCell::Int(a.value(idx)) });
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok(if a.is_null(idx) { ScalarCell::Null } else { ScalarCell::Float(a.value(idx)) });
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok(if a.is_null(idx) { ScalarCell::Null } else { ScalarCell::Str(a.value(idx).to_string()) });
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok(if a.is_null(idx) { ScalarCell::Null } else { ScalarCell::Bool(a.value(idx)) });
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(if a.is_null(idx) { ScalarCell::Null } else { ScalarCell::Ts(a.value(idx)) });
    }
    Err(TTGenError::SystemError(format!(
        "route: unsupported scalar dtype for {ctx}: {:?}",
        arr.data_type()
    )))
}

fn build_scalar_array_from_cells(
    dtype: &DataType,
    cells: &[ScalarCell],
    ctx: &str,
) -> TTGenResult<Arc<dyn Array>> {
    let typemis = |expected: &str| {
        TTGenError::SystemError(format!("route: type mismatch for {ctx}: expected {expected}"))
    };
    match dtype {
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(cells.len());
            for c in cells {
                match c {
                    ScalarCell::Int(v) => b.append_value(*v),
                    ScalarCell::Null => b.append_null(),
                    _ => return Err(typemis("Int64")),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float64 => {
            let mut b = Float64Builder::with_capacity(cells.len());
            for c in cells {
                match c {
                    ScalarCell::Float(v) => b.append_value(*v),
                    ScalarCell::Null => b.append_null(),
                    _ => return Err(typemis("Float64")),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(cells.len(), cells.len() * 8);
            for c in cells {
                match c {
                    ScalarCell::Str(v) => b.append_value(v),
                    ScalarCell::Null => b.append_null(),
                    _ => return Err(typemis("Utf8")),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(cells.len());
            for c in cells {
                match c {
                    ScalarCell::Bool(v) => b.append_value(*v),
                    ScalarCell::Null => b.append_null(),
                    _ => return Err(typemis("Boolean")),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            let mut b = TimestampNanosecondBuilder::with_capacity(cells.len());
            for c in cells {
                match c {
                    ScalarCell::Ts(v) => b.append_value(*v),
                    ScalarCell::Null => b.append_null(),
                    _ => return Err(typemis("Timestamp(ns)")),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(TTGenError::SystemError(format!(
            "route: unsupported scalar build dtype for {ctx}: {other:?}"
        ))),
    }
}

fn build_list_array_from_cells(
    list_dtype: &DataType,
    rows: &[Vec<ScalarCell>],
    ctx: &str,
) -> TTGenResult<Arc<dyn Array>> {
    let DataType::List(field) = list_dtype else {
        return Err(TTGenError::SystemError(format!(
            "route: expected List dtype for {ctx}, got {list_dtype:?}"
        )));
    };
    let item_dtype = field.data_type();
    let typemis = |expected: &str| {
        TTGenError::SystemError(format!("route: list item type mismatch for {ctx}: expected {expected}"))
    };
    match item_dtype {
        DataType::Int64 => {
            let mut b = ListBuilder::new(Int64Builder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Int(v) => b.values().append_value(*v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => return Err(typemis("List<Int64>")),
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float64 => {
            let mut b = ListBuilder::new(Float64Builder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Float(v) => b.values().append_value(*v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => return Err(typemis("List<Float64>")),
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Utf8 => {
            let mut b = ListBuilder::new(StringBuilder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Str(v) => b.values().append_value(v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => return Err(typemis("List<Utf8>")),
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Boolean => {
            let mut b = ListBuilder::new(BooleanBuilder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Bool(v) => b.values().append_value(*v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => return Err(typemis("List<Boolean>")),
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            let mut b = ListBuilder::new(TimestampNanosecondBuilder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Ts(v) => b.values().append_value(*v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => return Err(typemis("List<Timestamp(ns)>")),
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(TTGenError::SystemError(format!(
            "route: unsupported list item dtype for {ctx}: {other:?}"
        ))),
    }
}

/// Read the ENTITY1 column at `idx` into a `Value`, branching on dtype.
fn read_entity_value_at(arr: &dyn Array, idx: usize) -> TTGenResult<Value> {
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok(Value::Int(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok(Value::Str(a.value(idx).to_string()));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok(Value::Float(a.value(idx)));
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Value::Bool(a.value(idx)));
    }
    Err(TTGenError::SystemError(format!(
        "route: unsupported ENTITY1 dtype for per-link hash: {:?}",
        arr.data_type()
    )))
}

// ─── Hash + cutoff helpers ───────────────────────────────────────────────────

#[inline]
fn hash_entity_ts_link_to_unit_interval(
    entity: &Value,
    ts_secs: i64,
    link_idx: usize,
    seed: u64,
) -> f64 {
    let h0 = splitmix64(seed);
    let h1 = hash_value(entity, h0);
    let h2 = splitmix64(h1 ^ (ts_secs as u64));
    let h = splitmix64(h2 ^ (link_idx as u64));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Convert a raw ratio (e.g. `[0.8, 0.1, 0.1]`) into cumulative cutoffs
/// (length = ratio.len() - 1). Empty when total ≤ 0 (sentinel for UNUSED).
pub(crate) fn ratio_to_cumulative_cutoffs(ratio: &[f64]) -> Vec<f64> {
    let total: f64 = ratio.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut acc = 0.0;
    let mut out = Vec::with_capacity(ratio.len().saturating_sub(1));
    for r in ratio.iter().take(ratio.len().saturating_sub(1)) {
        acc += (r / total).max(0.0);
        out.push(acc);
    }
    out
}

/// Assign one link to a stage by content-hash bucketing over
/// `(entity, ts_secs, link_idx, seed)`. Returns `UNUSED_STAGE` when
/// `cutoffs` is empty.
#[inline]
pub(crate) fn stage_for_link(
    entity: &Value,
    ts_secs: i64,
    link_idx: usize,
    seed: u64,
    cutoffs: &[f64],
) -> u8 {
    if cutoffs.is_empty() {
        return UNUSED_STAGE;
    }
    let v = hash_entity_ts_link_to_unit_interval(entity, ts_secs, link_idx, seed);
    for (i, &c) in cutoffs.iter().enumerate() {
        if v < c {
            return i as u8;
        }
    }
    cutoffs.len() as u8
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
    use arrow_array::{Int64Array, StringArray, TimestampNanosecondArray, UInt8Array};
    use arrow_schema::{TimeUnit};
    use diskgraph_core::time::{Abs, Seconds};
    use crate::split_decider::{NoSplitDecider, TimeRangeDecider};
    use std::sync::Arc as StdArc;

    // ── helpers ──────────────────────────────────────────────────────────

    /// A simple 2-column (ENTITY1:Int64, TIMESTAMP:Timestamp(ns)) batch.
    fn ts_batch(ts_secs: &[i64]) -> RecordBatch {
        let entities: Vec<i64> = (0..ts_secs.len() as i64).collect();
        let ts_nanos: Vec<i64> = ts_secs.iter().map(|s| s * 1_000_000_000).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(Int64Array::from(entities)),
                StdArc::new(TimestampNanosecondArray::from(ts_nanos)),
            ],
        )
        .unwrap()
    }

    fn entities_of(batch: &RecordBatch) -> Vec<i64> {
        let idx = batch.schema().index_of("ENTITY1").unwrap();
        let arr = batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    fn target_lists_str(batch: &RecordBatch) -> Vec<Vec<String>> {
        let idx = batch.schema().index_of("TARGET1").unwrap();
        let list = batch.column(idx).as_any().downcast_ref::<ListArray>().unwrap();
        let items = list.values().as_any().downcast_ref::<StringArray>().unwrap();
        (0..list.len())
            .map(|i| {
                let s = list.value_offsets()[i] as usize;
                let e = list.value_offsets()[i + 1] as usize;
                (s..e).map(|j| items.value(j).to_string()).collect()
            })
            .collect()
    }

    /// Read the `exclude_target_keys` List<Utf8> column as per-row Vec<String>.
    fn exclude_lists_str(batch: &RecordBatch) -> Vec<Vec<String>> {
        let idx = batch.schema().index_of("exclude_target_keys").unwrap();
        let list = batch.column(idx).as_any().downcast_ref::<ListArray>().unwrap();
        let items = list.values().as_any().downcast_ref::<StringArray>().unwrap();
        (0..list.len())
            .map(|i| {
                let s = list.value_offsets()[i] as usize;
                let e = list.value_offsets()[i + 1] as usize;
                (s..e).map(|j| items.value(j).to_string()).collect()
            })
            .collect()
    }

    /// Sort a list-of-lists for set-equality comparison (exclude is a set).
    fn sorted(mut v: Vec<String>) -> Vec<String> {
        v.sort();
        v
    }

    fn freq_lists(batch: &RecordBatch) -> Vec<Vec<i64>> {
        let idx = batch.schema().index_of(FREQUENCY_COL).unwrap();
        let list = batch.column(idx).as_any().downcast_ref::<ListArray>().unwrap();
        let items = list.values().as_any().downcast_ref::<Int64Array>().unwrap();
        (0..list.len())
            .map(|i| {
                let s = list.value_offsets()[i] as usize;
                let e = list.value_offsets()[i + 1] as usize;
                (s..e).map(|j| items.value(j)).collect()
            })
            .collect()
    }

    /// Build an LP-array batch:
    /// cols = (ENTITY1:Int64, TIMESTAMP:ts(ns), TARGET1:List<Utf8>, FREQUENCY:List<Int64>).
    fn lp_array_batch(rows: &[(i64, i64, Vec<(&str, i64)>)]) -> RecordBatch {
        let mut entity = Int64Builder::new();
        let mut ts_b = TimestampNanosecondBuilder::new();
        let mut target_lb = ListBuilder::new(StringBuilder::new());
        let mut freq_lb = ListBuilder::new(Int64Builder::new());
        for (e, t, items) in rows {
            entity.append_value(*e);
            ts_b.append_value(*t * 1_000_000_000);
            for (s, f) in items {
                target_lb.values().append_value(*s);
                freq_lb.values().append_value(*f);
            }
            target_lb.append(true);
            freq_lb.append(true);
        }
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new(
                "TARGET1",
                DataType::List(StdArc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                FREQUENCY_COL,
                DataType::List(StdArc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(entity.finish()),
                StdArc::new(ts_b.finish()),
                StdArc::new(target_lb.finish()),
                StdArc::new(freq_lb.finish()),
            ],
        )
        .unwrap()
    }

    /// Build an LP-array batch with a parallel per-link split projection
    /// column `split_col` (List<Utf8>). Rows are (entity, ts, [(target, freq, split)]).
    fn lp_array_batch_with_split(
        rows: &[(i64, i64, Vec<(&str, i64, &str)>)],
        split_col: &str,
    ) -> RecordBatch {
        let mut entity = Int64Builder::new();
        let mut ts_b = TimestampNanosecondBuilder::new();
        let mut target_lb = ListBuilder::new(StringBuilder::new());
        let mut freq_lb = ListBuilder::new(Int64Builder::new());
        let mut split_lb = ListBuilder::new(StringBuilder::new());
        for (e, t, items) in rows {
            entity.append_value(*e);
            ts_b.append_value(*t * 1_000_000_000);
            for (target, freq, sval) in items {
                target_lb.values().append_value(*target);
                freq_lb.values().append_value(*freq);
                split_lb.values().append_value(*sval);
            }
            target_lb.append(true);
            freq_lb.append(true);
            split_lb.append(true);
        }
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new(
                "TARGET1",
                DataType::List(StdArc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                FREQUENCY_COL,
                DataType::List(StdArc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
            Field::new(
                split_col,
                DataType::List(StdArc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(entity.finish()),
                StdArc::new(ts_b.finish()),
                StdArc::new(target_lb.finish()),
                StdArc::new(freq_lb.finish()),
                StdArc::new(split_lb.finish()),
            ],
        )
        .unwrap()
    }

    // ── route_per_anchor ─────────────────────────────────────────────────

    #[test]
    fn test_per_anchor_timerange_routes_each_row() {
        // train=[0,100], val=[101,200], test=[201,300]; 400 → unused.
        let batch = ts_batch(&[10, 50, 150, 250, 400]);
        let d: TimeRangeDecider<Seconds> = TimeRangeDecider::new(vec![
            (Abs::from_secs(0), Abs::from_secs(100)),
            (Abs::from_secs(101), Abs::from_secs(200)),
            (Abs::from_secs(201), Abs::from_secs(300)),
        ]);
        let [train, val, test] =
            route_per_anchor::<Seconds, _>(&batch, &d, 0, false, None, None).unwrap();
        // rows: idx0(ts10)->train, idx1(ts50)->train, idx2(ts150)->val,
        //       idx3(ts250)->test, idx4(ts400)->unused(dropped).
        assert_eq!(entities_of(&train), vec![0, 1]);
        assert_eq!(entities_of(&val), vec![2]);
        assert_eq!(entities_of(&test), vec![3]);
        // Sum-of-3 invariant: 2 + 1 + 1 = 4 (one unused row dropped).
        assert_eq!(
            train.num_rows() + val.num_rows() + test.num_rows(),
            4
        );
    }

    #[test]
    fn test_per_anchor_array_lp_empty_target_dropped() {
        // Row 1 has an empty TARGET1 list → must be dropped even though its
        // ts (50) would otherwise land in train.
        let batch = lp_array_batch(&[
            (0, 10, vec![("a", 1)]),       // train
            (1, 50, vec![]),               // empty target → dropped
            (2, 150, vec![("b", 1)]),      // val
        ]);
        let d: TimeRangeDecider<Seconds> = TimeRangeDecider::new(vec![
            (Abs::from_secs(0), Abs::from_secs(100)),
            (Abs::from_secs(101), Abs::from_secs(200)),
            (Abs::from_secs(201), Abs::from_secs(300)),
        ]);
        let [train, val, test] =
            route_per_anchor::<Seconds, _>(&batch, &d, 0, true, None, None).unwrap();
        assert_eq!(entities_of(&train), vec![0]);
        assert_eq!(entities_of(&val), vec![2]);
        assert_eq!(test.num_rows(), 0);
        assert_eq!(train.num_rows() + val.num_rows() + test.num_rows(), 2);
    }

    #[test]
    fn test_per_anchor_include_mask_false_dropped() {
        // include_mask = [t, f, t]; row 1 dropped even though its ts lands.
        let entities = Int64Array::from(vec![0_i64, 1, 2]);
        let ts = TimestampNanosecondArray::from(
            vec![10_i64, 50, 90].into_iter().map(|s| s * 1_000_000_000).collect::<Vec<_>>(),
        );
        let mask = BooleanArray::from(vec![true, false, true]);
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new(INCLUDE_MASK_COL, DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![StdArc::new(entities), StdArc::new(ts), StdArc::new(mask)],
        )
        .unwrap();
        // All ts in train window.
        let d: TimeRangeDecider<Seconds> = TimeRangeDecider::new(vec![
            (Abs::from_secs(0), Abs::from_secs(100)),
            (Abs::from_secs(101), Abs::from_secs(200)),
            (Abs::from_secs(201), Abs::from_secs(300)),
        ]);
        let [train, val, test] =
            route_per_anchor::<Seconds, _>(&batch, &d, 0, false, None, None).unwrap();
        assert_eq!(entities_of(&train), vec![0, 2]);
        assert_eq!(val.num_rows(), 0);
        assert_eq!(test.num_rows(), 0);
    }

    #[test]
    fn test_per_anchor_no_split_decider_drops_all() {
        let batch = ts_batch(&[10, 50, 150]);
        let d = NoSplitDecider;
        let [train, val, test] =
            route_per_anchor::<Seconds, _>(&batch, &d, 0, false, None, None).unwrap();
        assert_eq!(train.num_rows() + val.num_rows() + test.num_rows(), 0);
    }

    // ── route_per_anchor split-column drop ───────────────────────────────

    /// Custom-TT split-column path: route by ColumnDeciderManual on a Utf8
    /// `split_col`, and assert the column is projected out of the outputs.
    #[test]
    fn test_per_anchor_drops_split_column() {
        use crate::split_decider::ColumnDeciderManual;
        let entities = Int64Array::from(vec![0_i64, 1, 2, 3]);
        let ts = TimestampNanosecondArray::from(
            vec![10_i64, 20, 30, 40]
                .into_iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        );
        let split = StringArray::from(vec!["0", "1", "2", "0"]);
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("split_col", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![StdArc::new(entities), StdArc::new(ts), StdArc::new(split)],
        )
        .unwrap();
        let d = ColumnDeciderManual::new(
            "split_col".to_string(),
            vec!["0".to_string()],
            vec!["1".to_string()],
            vec!["2".to_string()],
        );
        let [train, val, test] =
            route_per_anchor::<Seconds, _>(&batch, &d, 0, false, None, Some("split_col")).unwrap();
        assert_eq!(entities_of(&train), vec![0, 3]);
        assert_eq!(entities_of(&val), vec![1]);
        assert_eq!(entities_of(&test), vec![2]);
        for sub in [&train, &val, &test] {
            assert!(
                sub.schema().index_of("split_col").is_err(),
                "split_col must be dropped from routed sub-batch"
            );
            assert!(sub.schema().index_of("ENTITY1").is_ok());
        }
    }

    // ── route_per_link RandomLink ────────────────────────────────────────

    #[test]
    fn test_per_link_random_single_anchor_four_links() {
        // 1 anchor, 4 links. Hand-compute each link's stage via the SAME
        // stage_for_link fn, then assert the 3 sub-lists.
        let entity_id = 7_i64;
        let ts_secs = 1000_i64;
        let seed = 42_u64;
        let cutoffs = ratio_to_cumulative_cutoffs(&[0.5, 0.3, 0.2]);
        let links = ["a", "b", "c", "d"];
        let mut expected: [Vec<String>; 3] =
            [Vec::new(), Vec::new(), Vec::new()];
        for (li, name) in links.iter().enumerate() {
            let s = stage_for_link(&Value::Int(entity_id), ts_secs, li, seed, &cutoffs);
            if (s as usize) < 3 {
                expected[s as usize].push(name.to_string());
            }
        }

        let batch = lp_array_batch(&[(
            entity_id,
            ts_secs,
            vec![("a", 10), ("b", 20), ("c", 30), ("d", 40)],
        )]);
        let assigner = LinkAssigner::random_from_ratio(&[0.5, 0.3, 0.2], seed);
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();

        let outs = [&train, &val, &test];
        for (stage, out) in outs.iter().enumerate() {
            let tl = target_lists_str(out);
            if expected[stage].is_empty() {
                assert_eq!(out.num_rows(), 0, "stage {stage} expected empty");
            } else {
                assert_eq!(out.num_rows(), 1, "stage {stage} expected 1 row");
                assert_eq!(tl[0], expected[stage], "stage {stage} targets");
                // Frequency sub-list must align with target sub-list length.
                let fl = freq_lists(out);
                assert_eq!(fl[0].len(), tl[0].len());
            }
        }
        // Sum of 3 output target counts == 4 links (all assigned: ratio>0).
        let total: usize = outs.iter().map(|o| target_lists_str(o).iter().map(|l| l.len()).sum::<usize>()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_per_link_random_multi_anchor() {
        let seed = 99_u64;
        let cutoffs = ratio_to_cumulative_cutoffs(&[0.5, 0.5]);
        let rows: Vec<(i64, i64, Vec<(&str, i64)>)> = vec![
            (1, 100, vec![("p", 1), ("q", 2), ("r", 3)]),
            (2, 200, vec![("x", 4), ("y", 5)]),
        ];
        let batch = lp_array_batch(&rows);
        let assigner = LinkAssigner::random_from_ratio(&[0.5, 0.5], seed);
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();
        assert_eq!(test.num_rows(), 0); // 2-way ratio → no test stage

        // Reconstruct: total links preserved (3 + 2 = 5).
        let total: usize = [&train, &val]
            .iter()
            .map(|o| target_lists_str(o).iter().map(|l| l.len()).sum::<usize>())
            .sum();
        assert_eq!(total, 5);

        // Each non-empty output row's entity must match a source anchor and
        // its links must be exactly those that hash to that stage for that
        // anchor (hand-compute).
        for (anchor_idx, (e, ts, items)) in rows.iter().enumerate() {
            let _ = anchor_idx;
            let mut exp: [Vec<String>; 2] = [Vec::new(), Vec::new()];
            for (li, (name, _f)) in items.iter().enumerate() {
                let s = stage_for_link(&Value::Int(*e), *ts, li, seed, &cutoffs);
                if (s as usize) < 2 {
                    exp[s as usize].push(name.to_string());
                }
            }
            // Find this anchor's output row in train / val by entity match.
            for (stage, out) in [&train, &val].iter().enumerate() {
                let ents = entities_of(out);
                let tls = target_lists_str(out);
                if let Some(pos) = ents.iter().position(|x| x == e) {
                    assert_eq!(tls[pos], exp[stage], "anchor {e} stage {stage}");
                } else {
                    assert!(exp[stage].is_empty(), "anchor {e} stage {stage} should be empty");
                }
            }
        }
    }

    #[test]
    fn test_per_link_random_output_schema_unchanged() {
        let batch = lp_array_batch(&[(1, 100, vec![("a", 1)])]);
        let assigner = LinkAssigner::random_from_ratio(&[1.0, 0.0, 0.0], 0);
        let [train, _val, _test] = route_per_link(&batch, &assigner).unwrap();
        // Output schema = input schema + the appended `exclude_target_keys` col.
        assert_eq!(
            train.schema().fields().len(),
            batch.schema().fields().len() + 1
        );
        assert!(train.schema().index_of("TARGET1").is_ok());
        assert!(train.schema().index_of(FREQUENCY_COL).is_ok());
        assert!(train.schema().index_of("exclude_target_keys").is_ok());
    }

    // ── route_per_link CustomLink ────────────────────────────────────────

    #[test]
    fn test_per_link_custom_routes_by_value_set() {
        // 1 anchor with 4 links carrying split labels train/val/test/other.
        let batch = lp_array_batch_with_split(
            &[(
                1,
                100,
                vec![
                    ("a", 1, "TR"),
                    ("b", 2, "VA"),
                    ("c", 3, "TE"),
                    ("d", 4, "NONE"), // not in any set → dropped
                ],
            )],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: ["VA".to_string()].into_iter().collect(),
            test: ["TE".to_string()].into_iter().collect(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();
        assert_eq!(target_lists_str(&train), vec![vec!["a".to_string()]]);
        assert_eq!(target_lists_str(&val), vec![vec!["b".to_string()]]);
        assert_eq!(target_lists_str(&test), vec![vec!["c".to_string()]]);
        // "d" (NONE) dropped → sum of 3 == 3 of the 4 links.
        let total: usize = [&train, &val, &test]
            .iter()
            .map(|o| target_lists_str(o).iter().map(|l| l.len()).sum::<usize>())
            .sum();
        assert_eq!(total, 3);
        // Output schema drops the split_col but appends `exclude_target_keys`
        // (net: -1 +1 = same field count as input).
        assert!(train.schema().index_of("SPLIT_COL").is_err());
        assert!(train.schema().index_of("exclude_target_keys").is_ok());
        assert_eq!(train.schema().fields().len(), batch.schema().fields().len());
    }

    // ── route_per_link past-visit exclude (STATIC LP) ────────────────────────

    #[test]
    fn test_per_link_exclude_unions_custom() {
        // 1 anchor, links deterministically routed via CustomLink so the
        // per-stage target sets are known: train={a,b}, val={c}, test={d}.
        // => train exclude=∅, val exclude=train={a,b}, test exclude=train∪val={a,b,c}.
        let batch = lp_array_batch_with_split(
            &[(
                1,
                100,
                vec![
                    ("a", 1, "TR"),
                    ("b", 2, "TR"),
                    ("c", 3, "VA"),
                    ("d", 4, "TE"),
                ],
            )],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: ["VA".to_string()].into_iter().collect(),
            test: ["TE".to_string()].into_iter().collect(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();

        // train: column present but empty (no prior stage to exclude).
        assert_eq!(exclude_lists_str(&train), vec![Vec::<String>::new()]);
        // val excludes the TRAIN targets {a,b}.
        assert_eq!(
            sorted(exclude_lists_str(&val).remove(0)),
            vec!["a".to_string(), "b".to_string()]
        );
        // test excludes TRAIN∪VAL {a,b,c} (deduped union).
        assert_eq!(
            sorted(exclude_lists_str(&test).remove(0)),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_per_link_exclude_dedup_across_stages() {
        // Same key "x" appears in BOTH train and val. test exclude = train∪val
        // must contain "x" exactly ONCE (set semantics).
        let batch = lp_array_batch_with_split(
            &[(
                1,
                100,
                vec![("x", 1, "TR"), ("x", 2, "VA"), ("z", 3, "TE")],
            )],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: ["VA".to_string()].into_iter().collect(),
            test: ["TE".to_string()].into_iter().collect(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [_train, val, test] = route_per_link(&batch, &assigner).unwrap();
        assert_eq!(exclude_lists_str(&val).remove(0), vec!["x".to_string()]);
        // dedup: "x" once, not twice.
        assert_eq!(sorted(exclude_lists_str(&test).remove(0)), vec!["x".to_string()]);
    }

    #[test]
    fn test_per_link_exclude_test_only_entity() {
        // An anchor whose only links go to TEST: test exclude is ∅ (no prior).
        let batch = lp_array_batch_with_split(
            &[(1, 100, vec![("p", 1, "TE"), ("q", 2, "TE")])],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: ["VA".to_string()].into_iter().collect(),
            test: ["TE".to_string()].into_iter().collect(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();
        assert_eq!(train.num_rows() + val.num_rows(), 0);
        assert_eq!(exclude_lists_str(&test), vec![Vec::<String>::new()]);
    }

    #[test]
    fn test_per_link_exclude_disjoint_from_current_positives() {
        // Safety invariant: a stage's exclude set never overlaps its own
        // positives (each link → exactly one stage). Assert val/test excludes
        // are disjoint from that stage's TARGET list.
        let batch = lp_array_batch_with_split(
            &[(
                1,
                100,
                vec![
                    ("a", 1, "TR"),
                    ("c", 3, "VA"),
                    ("d", 4, "TE"),
                ],
            )],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: ["VA".to_string()].into_iter().collect(),
            test: ["TE".to_string()].into_iter().collect(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [_train, val, test] = route_per_link(&batch, &assigner).unwrap();
        for (out, label) in [(&val, "val"), (&test, "test")] {
            let pos: std::collections::HashSet<String> =
                target_lists_str(out)[0].iter().cloned().collect();
            for x in &exclude_lists_str(out)[0] {
                assert!(!pos.contains(x), "{label}: exclude {x} overlaps positives");
            }
        }
    }

    #[test]
    fn test_per_link_custom_all_unused_emits_nothing() {
        let batch = lp_array_batch_with_split(
            &[(1, 100, vec![("a", 1, "X"), ("b", 2, "Y")])],
            "SPLIT_COL",
        );
        let assigner = LinkAssigner::CustomLink {
            train: ["TR".to_string()].into_iter().collect(),
            val: HashSet::new(),
            test: HashSet::new(),
            split_col: "SPLIT_COL".to_string(),
        };
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();
        assert_eq!(train.num_rows() + val.num_rows() + test.num_rows(), 0);
    }

    #[test]
    fn test_per_link_empty_list_contributes_nothing() {
        let batch = lp_array_batch(&[(1, 100, vec![]), (2, 200, vec![("a", 1)])]);
        let assigner = LinkAssigner::random_from_ratio(&[1.0, 0.0, 0.0], 0);
        let [train, val, test] = route_per_link(&batch, &assigner).unwrap();
        // Only anchor 2 contributes (1 link → train).
        assert_eq!(entities_of(&train), vec![2]);
        assert_eq!(val.num_rows() + test.num_rows(), 0);
    }

    // suppress unused import warning for UInt8Array in case not used.
    #[allow(dead_code)]
    fn _touch_uint8(_: UInt8Array) {}
}
