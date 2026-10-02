// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TT-gen executor: drives the entity loop, staged execution, and threading.
//!
//! Entry point: [`execute_tt_pass`] takes a `DiskGraphTTPlan<U>`, an index
//! path, and an output directory. It runs the executor end-to-end and writes
//! a `tt.parquet` directory dataset directly to the output directory. No
//! streaming exposure; the caller reads the parts back to inspect rows.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::errors::{TTGenError, TTGenResult};
use arrow_schema::DataType;
use crossbeam_channel::{bounded, Receiver, Sender};
use diskgraph_core::accumulators::{
    for_each_split_ascii_byte, HyperLogLog, LocalColumnCounts, SharedColumnCounts,
    TDigestEstimator, WelfordAccumulator,
};
use diskgraph_core::time::{add_calendar_months, dur_from_unit, Abs, Dur, TimeUnit, UnitSystem};
use diskgraph_core::tt_builder::*;
use diskgraph_index::graph::GraphIndexReader;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::manifest::{
    sha256_file, write_manifest, write_stats_artifact, ArtifactRef, ManifestChecksums, TTManifest,
    TrainTableArtifactRef, TrainTablePartRef, SPLIT_STATS_FORMAT, STATS_FORMAT,
    TT_MANIFEST_VERSION, TT_PARQUET_FORMAT,
};
use crate::parquet_dataset::{
    first_tt_dataset_schema, ordered_tt_dataset_parts, part_name, TT_DATASET_DIR, TT_PART_MAX_ROWS,
    TT_PART_ROW_GROUP_ROWS,
};
use crate::pass_b::{run_pass_b_per_anchor, run_pass_b_per_link, StagePartsOut, STAGE_TABLE_NAMES};
use crate::route::{per_link_output_schema, LinkAssigner};
use crate::split_decider::{
    ApproxDateOffsetDecider, BacktestingDecider, ColumnDeciderManual, ColumnDeciderRatioNumeric,
    DateOffsetDecider, FixedDecider, ManualDecider, NoSplitDecider, OffsetUnit, RandomDecider,
    RandomLinkDecider, SplitDecider, TemporalDecider, TimeRangeDecider,
};
use crate::ttgen_accumulator::{hash_i64, hash_str, hash_value, Accumulator};
use crate::ttgen_filter::{
    compare_values, eval_row_filter, eval_stage_filter, is_nullish_filter_value,
    resolve_remote_node,
};
use crate::ttgen_output::{
    BatchMessage, ChannelBatchWriter, ExtraColumnOutput, ExtraValueRef, OutputMode,
};
use crate::ttgen_resolve::*;
use crate::ttgen_stats::{ChunkStats, RowProcessor, StatsMerger, TTRowRef, TrainTableStatsPayload};
use crate::ttgen_traversal::*;

/// Returned by `run_pass_a_then_pass_b`: total rows, merged chunk-stats, and
/// the fused pass-B per-stage parts (`Some` only for the per-anchor Compute
/// arms, which own the `D`-generic merged decider; deferred modes run pass-B
/// in the executor body). Per-row weights ride in the `WEIGHT` column, not here.
type PassAOutputs<U> = (u64, StatsMerger<U>, Option<[StagePartsOut; 3]>);

/// Node-task train-table targets require a complete vocabulary and have no
/// hash-encoder fallback. Link-prediction targets are graph node IDs and do
/// not use this encoder vocabulary.
const MAX_TRAIN_TABLE_TARGET_CATEGORIES: u64 = 250_000_000;

// The entity column is a graph FK/pkey resolved to node ids and is never
// encoded, so no exact entity category counts are collected — only the target
// accumulator exists, and only for node-task categorical targets (LP targets
// are graph node IDs with no exact-vocabulary consumer).
fn category_count_accumulators(
    target_is_categorical: bool,
    is_link_prediction: bool,
) -> Option<Arc<SharedColumnCounts>> {
    let collect_exact_target_categories = target_is_categorical && !is_link_prediction;
    collect_exact_target_categories.then(|| {
        Arc::new(SharedColumnCounts::new_hashed_with_max_categories(
            MAX_TRAIN_TABLE_TARGET_CATEGORIES,
        ))
    })
}

fn drain_required_target_category_counts(
    target: Option<&SharedColumnCounts>,
) -> TTGenResult<Option<HashMap<String, u64>>> {
    if target.is_some_and(SharedColumnCounts::is_aborted) {
        return Err(TTGenError::TargetCategoryLimit(format!(
            "train-table target category limit exceeded: TARGET1 requires exact \
             CATEGORY_COUNTS, but observed more than \
             {MAX_TRAIN_TABLE_TARGET_CATEGORIES} unique categories; refusing to publish \
             an incomplete train-table statistics artifact"
        )));
    }

    Ok(target.map(|counts| {
        counts
            .drain_to_map()
            .expect("non-aborted target counts must drain")
    }))
}

pub const STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION: &str =
    "diskgraph.tt_stage_artifact_inventory.v1";

fn stage_part_artifact(output_dir: &Path, relative_path: &str) -> TTGenResult<StagePartArtifact> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let path = output_dir.join(relative_path);
    let size_bytes = std::fs::metadata(&path)?.len();
    let file = File::open(&path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let row_count = u64::try_from(builder.metadata().file_metadata().num_rows()).map_err(|_| {
        TTGenError::SystemError(format!(
            "stage parquet {} has a negative footer row count",
            path.display()
        ))
    })?;
    Ok(StagePartArtifact {
        path: relative_path.to_string(),
        size_bytes,
        row_count,
    })
}

/// Build `[StageGenInfo; 3]` for train/val/test from the fused pass-B output.
fn stage_gen_infos_from_parts(
    parts: &[StagePartsOut; 3],
    output_dir: &Path,
) -> TTGenResult<Vec<StageGenInfo>> {
    const KEYS: [&str; 3] = ["train", "val", "test"];
    parts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let artifacts = p
                .parts
                .iter()
                .map(|part| stage_part_artifact(output_dir, part))
                .collect::<TTGenResult<Vec<_>>>()?;
            let artifact_rows = artifacts.iter().map(|part| part.row_count).sum::<u64>();
            if artifact_rows != p.row_count {
                return Err(TTGenError::SystemError(format!(
                    "stage {} wrote {artifact_rows} parquet rows but reported {}",
                    KEYS[i], p.row_count
                )));
            }
            Ok(StageGenInfo {
                key: KEYS[i].to_string(),
                table_name: STAGE_TABLE_NAMES[i].to_string(),
                parts: p.parts.clone(),
                row_count: p.row_count,
                seed_time: p.seed_time,
                artifact_inventory: Some(StageArtifactInventory {
                    schema_version: STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION.to_string(),
                    parts: artifacts,
                }),
            })
        })
        .collect()
}

/// Probe the arrow schema of a tt.parquet dataset (the `out_schema` for the
/// fused pass-B per-anchor / use-bin / RandomLink modes).
fn probe_tt_schema(tt_parquet: &Path) -> TTGenResult<arrow_schema::SchemaRef> {
    first_tt_dataset_schema(tt_parquet)
}

fn write_tt_dataset(
    receiver: Receiver<BatchMessage>,
    dataset_path: &Path,
    max_rows_per_part: usize,
) -> TTGenResult<u64> {
    if max_rows_per_part == 0 {
        return Err(TTGenError::InvalidPlan(
            "training table max rows per part must be positive".to_string(),
        ));
    }
    std::fs::create_dir(dataset_path)?;

    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(TT_PART_ROW_GROUP_ROWS))
        .build();
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut part_index = 0usize;
    let mut rows_in_part = 0usize;
    let mut total_rows = 0u64;

    for batch in receiver.iter() {
        let mut batch_offset = 0usize;
        while batch_offset < batch.num_rows() {
            if rows_in_part == max_rows_per_part {
                writer
                    .take()
                    .expect("non-empty part has a writer")
                    .close()?;
                part_index += 1;
                rows_in_part = 0;
            }
            if writer.is_none() {
                let file = File::create(dataset_path.join(part_name(part_index)))?;
                writer = Some(ArrowWriter::try_new(
                    file,
                    batch.schema(),
                    Some(properties.clone()),
                )?);
            }
            let rows = (batch.num_rows() - batch_offset).min(max_rows_per_part - rows_in_part);
            writer
                .as_mut()
                .expect("writer was initialized")
                .write(&batch.slice(batch_offset, rows))?;
            batch_offset += rows;
            rows_in_part += rows;
            total_rows += rows as u64;
        }
    }
    if let Some(writer) = writer {
        writer.close()?;
    }
    Ok(total_rows)
}

fn train_table_parts(
    dataset_path: &Path,
    expected_rows: u64,
) -> TTGenResult<(u64, Vec<TrainTablePartRef>)> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut size_bytes = 0u64;
    let mut row_count = 0u64;
    let mut schema = None;
    let mut parts = Vec::new();
    for path in ordered_tt_dataset_parts(dataset_path)? {
        let part_size = std::fs::metadata(&path)?.len();
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        if let Some(expected_schema) = schema.as_ref() {
            if builder.schema().as_ref() != expected_schema {
                return Err(TTGenError::SchemaError(format!(
                    "training table part schema mismatch: {}",
                    path.display()
                )));
            }
        } else {
            schema = Some(builder.schema().as_ref().clone());
        }
        let part_rows =
            u64::try_from(builder.metadata().file_metadata().num_rows()).map_err(|_| {
                TTGenError::SystemError(format!(
                    "training table part has negative row count: {}",
                    path.display()
                ))
            })?;
        size_bytes = size_bytes.checked_add(part_size).ok_or_else(|| {
            TTGenError::SystemError("training table byte count overflow".to_string())
        })?;
        row_count = row_count.checked_add(part_rows).ok_or_else(|| {
            TTGenError::SystemError("training table row count overflow".to_string())
        })?;
        parts.push(TrainTablePartRef {
            path: Path::new(TT_DATASET_DIR)
                .join(path.file_name().expect("dataset part has a filename"))
                .to_string_lossy()
                .into_owned(),
            row_count: part_rows,
            size_bytes: part_size,
        });
    }
    if row_count != expected_rows {
        return Err(TTGenError::SystemError(format!(
            "training table parts contain {row_count} rows but pass A reported {expected_rows}"
        )));
    }
    Ok((size_bytes, parts))
}

/// Return `schema` with the field named `col` removed. Used to drop the
/// custom-TT split column from the per-stage seed output schema. A no-op
/// (clone) if the column is absent.
fn schema_without_column(schema: &arrow_schema::SchemaRef, col: &str) -> arrow_schema::SchemaRef {
    let fields: Vec<arrow_schema::Field> = schema
        .fields()
        .iter()
        .filter(|f| f.name() != col)
        .map(|f| f.as_ref().clone())
        .collect();
    std::sync::Arc::new(arrow_schema::Schema::new(fields))
}

/// The pass-B mode for `run_pass_a_then_pass_b`. Only the per-anchor mode
/// runs its fused pass-B inside the (monomorphized-over-`D`) function, because
/// it needs the merged decider. Per-link / use-bin / no-split run pass A with
/// `NoSplitDecider` and let the executor body run the fused pass-B afterwards.
#[derive(Debug, Clone)]
enum PassBMode {
    /// Run the per-anchor fused pass-B with the merged decider.
    /// `is_array_lp` folds in the array-LP empty-target drop.
    /// `drop_split_column` is `Some(col)` for the custom-TT split-column path
    /// (`ManualSplit`/`ColumnSplit` on a customer-provided column): that column
    /// is the split signal and must be projected OUT of the per-stage seed
    /// parquets so it never reaches the seed rowpack / features.
    PerAnchor {
        is_array_lp: bool,
        drop_split_column: Option<String>,
    },
    /// No pass-B here (executor body handles per-link / no-split).
    Deferred,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// One per-stage seed parquet produced by gen (`<stage>_seeds.parquet`),
/// the gen→ingest hand-off record. Replaces the old split.bin/demux contract:
/// gen writes these directly and records them here; ingest reads them with no
/// demux. `seed_time` is the `(min, max)` of the stage's TIMESTAMP column in
/// int64-ns (None when the stage carried no finite timestamp).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StagePartArtifact {
    pub path: String,
    pub size_bytes: u64,
    pub row_count: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageArtifactInventory {
    pub schema_version: String,
    pub parts: Vec<StagePartArtifact>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageGenInfo {
    pub key: String,
    pub table_name: String,
    /// Relative part paths under the staging dir, e.g.
    /// `["train_seeds/part-00000.parquet", ...]` — one per bounded source
    /// row-group span that routed >0 rows to the stage. An empty stage carries
    /// a single schema-only 0-row part so ingest never hits a missing dir.
    pub parts: Vec<String>,
    pub row_count: u64,
    pub seed_time: Option<[i64; 2]>,
    /// Exact bounded-metadata contract for the stage parquets. Absent only on
    /// v1 ingest specs produced before this field existed.
    #[serde(default)]
    pub artifact_inventory: Option<StageArtifactInventory>,
}

/// Output of a single TT-gen pass: the tt.parquet dataset, per-stage seed parquets
/// (split path), and the finalized stats payload.
///
/// `stage_gen_infos` is `Some([train, val, test])` when the plan's split family
/// resolves to a [`SplitMode::Compute`] arm; `None` for the
/// no-split (prediction) path, where the single stage is `tt.parquet` itself.
#[derive(Debug)]
pub struct TTPassOutput {
    /// Total rows written to `tt_parquet_path`.
    pub rows_written: u64,
    /// Absolute path to the produced parquet dataset (`output_dir/tt.parquet`).
    pub tt_parquet_path: PathBuf,
    /// Finalized per-column stats payload (entity / timestamp / target /
    /// frequency / split). `None` only when the executor produced no rows
    /// AND no stats were observed (rare, kept for symmetry).
    pub stats: Option<TrainTableStatsPayload>,
    /// Per-stage seed parquets (`<stage>_seeds.parquet`) written into the
    /// output dir, with row counts + seed-times. `None` on the no-split
    /// (prediction) path.
    pub stage_gen_infos: Option<Vec<StageGenInfo>>,
    /// Path to `output_dir/manifest.json`. Always written.
    pub manifest_path: PathBuf,
}

/// Orchestrator-level dispatch for the split strategy.
///
/// `Compute` covers tier-1 and tier-2 split families with deciders; `None`
/// covers everything else. Custom-TT splits (customer-provided) flow through
/// `Compute` via the `ManualSplit`/`ColumnSplit` column deciders — the
/// customer's split assignment rides as a SPLIT COLUMN inside the parquet,
/// not as a `split.bin` sidecar.
#[derive(Debug, Clone)]
pub enum SplitMode {
    /// Run the fused pass-B router; emit per-stage seed parquets.
    Compute,
    /// No split; no stage parquets emitted (the single stage is `tt.parquet`).
    None,
}

/// Custom-TT source descriptor.
///
/// `parquet_path` is the customer-supplied training-table parquet. Each
/// rayon worker handles a slice of its row groups (parallelism unit:
/// parquet row groups, NOT entity ranges).
///
/// Custom-TT splits flow entirely through the plan's `SplitSpec` (e.g.
/// `ManualSplit` on a customer-provided split column) — there is no
/// `split.bin` sidecar.
///
/// `input_timestamp_unit` is the unit declaration for **untyped Int64**
/// TIMESTAMP columns in the source parquet. Required and explicit — there
/// is no default. Typed Arrow `Timestamp(unit)` arrays self-describe their
/// unit and ignore this field. Kumo's rehydration bridge sets this to
/// `Nanoseconds` because kumo's canonical TIMESTAMP representation is
/// int64 nanoseconds (Spark `LongType`).
#[derive(Debug, Clone)]
pub struct CustomTtSource {
    pub parquet_path: PathBuf,
    pub input_timestamp_unit: TimeUnit,
}

/// Determine which `SplitMode` applies to `plan` based on its split family.
///
/// Tier-1 supported: `TimeRangeSplit`, `BacktestingSplit`, and `ManualSplit`
/// **when `split_column` is `TIMESTAMP` or `PSEUDO_TIMESTAMP`** (mirrors the
/// parity check in `derive_manual_split_mask_counts`; arbitrary
/// `split_column`s use the `ColumnDecider`).
///
/// Tier-2 families (Random, RandomLink, Fixed, Temporal, DateOffset,
/// ApproxDateOffset, Column) all resolve to `SplitMode::Compute`.
///
/// Anything else (including absence of `plan.split` or ManualSplit on a
/// non-temporal column) falls through to `SplitMode::None`.
fn determine_split_mode<U: UnitSystem>(plan: &DiskGraphTTPlan<U>) -> SplitMode {
    let Some(split) = plan.split.as_ref() else {
        return SplitMode::None;
    };
    match split.kind.as_deref() {
        // Tier-1: TimeRange / Backtesting / Manual (TIMESTAMP only).
        Some("TimeRangeSplit") | Some("BacktestingSplit") => SplitMode::Compute,
        // ManualSplit on a TIMESTAMP pseudo-column uses the temporal
        // `ManualDecider`; ManualSplit on any OTHER column is the
        // customer-provided split-column path (custom-TT): the column's
        // value sets (manual_train/val/test_values) route each row via
        // `ColumnDeciderManual`, identical to `ColumnSplit` with manual
        // values. Either way → Compute.
        Some("ManualSplit") => SplitMode::Compute,
        // Tier-2: Random / Fixed / Temporal / DateOffset /
        // ApproxDateOffset / RandomLink. ColumnSplit also lifts into
        // Compute (manual-values + ratio-derived numeric variants).
        Some("RandomSplit")
        | Some("RandomLinkSplit")
        | Some("FixedSplit")
        | Some("TemporalSplit")
        | Some("DateOffsetSplit")
        | Some("ApproxDateOffsetSplit")
        | Some("ColumnSplit") => SplitMode::Compute,
        // CustomLinkSplit on the LP+array path. Pass A runs with
        // `NoSplitDecider` and emits the per-link split projection column;
        // a regroup step assigns each link to a stage by membership in
        // train/val/test value sets, then pass B reads the inline stage
        // column. The non-array CustomSplit ("ManualSplit on entity")
        // already lives elsewhere — see `ManualDecider`.
        Some("CustomLinkSplit") => SplitMode::Compute,
        _ => SplitMode::None,
    }
}

/// Execute a TT-gen plan end-to-end.
///
/// Drives the existing rayon-parallel pass A (one writer thread, one
/// `bounded(16)` channel, per-worker `RowProcessor`) and writes the produced
/// `RecordBatch`es to ordered parts under `output_dir/tt.parquet`. Returns a
/// [`TTPassOutput`] with the row count, dataset path, and finalized stats payload.
///
/// `output_dir` must exist (the caller is responsible for creation/cleanup).
/// The function does not write `manifest.json` or `split.bin` — those are
/// handled by `execute_tt_pass_with_source`.
pub fn execute_tt_pass<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &Path,
    output_dir: &Path,
) -> TTGenResult<TTPassOutput> {
    execute_tt_pass_with_source(plan, index_path, output_dir, None)
}

/// Variant of [`execute_tt_pass`] with optional custom-TT source: when
/// `custom_tt_source` is `Some`, runs the custom-TT pass-A path (parquet
/// row groups → workers) instead of the PQ entity-table walk. When
/// `None`, behaves identically to `execute_tt_pass`.
///
/// Custom-TT splits are driven by `plan.split` (e.g. `ManualSplit` on a
/// customer-provided split column) and routed through the normal
/// `SplitMode::Compute` deciders — there is no `split.bin` sidecar.
pub fn execute_tt_pass_with_source<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &Path,
    output_dir: &Path,
    custom_tt_source: Option<CustomTtSource>,
) -> TTGenResult<TTPassOutput> {
    execute_tt_pass_with_customizer(
        plan,
        index_path,
        output_dir,
        custom_tt_source,
        None::<fn(&mut ResolvedPlan<U>)>,
    )
}

/// Test-only entry point (also pub for downstream callers that need to
/// inject multi-entity / multi-target customizations not yet expressible
/// in the input `DiskGraphTTPlan` schema). The closure runs after
/// `resolve_plan` and before the executor runs, allowing tests to mutate
/// the resolved plan's `entity_pk_col_indices`, `target_alias_indices`,
/// `entity_dtypes`, or `target_dtypes` to exercise the wide-format
/// path.
pub fn execute_tt_pass_with_customizer<U: UnitSystem, F>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &Path,
    output_dir: &Path,
    custom_tt_source: Option<CustomTtSource>,
    resolved_customizer: Option<F>,
) -> TTGenResult<TTPassOutput>
where
    F: FnOnce(&mut ResolvedPlan<U>),
{
    if plan.fact_pseudo_time.is_some() && !matches!(determine_split_mode(plan), SplitMode::Compute)
    {
        return Err(TTGenError::InvalidPlan(
            "fact_pseudo_time requires stage parts from a supported split".to_string(),
        ));
    }
    // Open index
    let metadata_path = index_path.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, index_path)
        .map_err(|e| TTGenError::SystemError(format!("opening graph index: {e:#}")))?;

    // Preprocess plan (main thread, once)
    let mut resolved = resolve_plan(plan, &reader)
        .map_err(|e| TTGenError::InvalidPlan(format!("resolving plan: {e:#}")))?;
    if let Some(customizer) = resolved_customizer {
        customizer(&mut resolved);
    }
    let resolved = Arc::new(resolved);
    let mut fact_pseudo_time = plan
        .fact_pseudo_time
        .as_ref()
        .map(|spec| {
            crate::pseudo_time::FactPseudoTime::new(
                &reader,
                &plan.entity.table,
                &plan.entity.column,
                spec,
            )
        })
        .transpose()?;

    let num_entities = resolved.entity_table.node_count();
    let num_threads = rayon::current_num_threads();
    let num_chunks = (num_threads * 8).min(num_entities as usize).max(1);
    tracing::info!(
        entities = num_entities,
        threads = num_threads,
        chunks = num_chunks,
        stages = resolved.stages.len(),
        aliases = resolved.num_aliases,
        output_dir = %output_dir.display(),
        "TT-gen execution started"
    );

    // Bounded channel: buffer up to 16 batches (~1MB each) for back-pressure.
    // The writer thread (on the receiver side) terminates the channel by
    // writing batches to the parquet dataset on disk. No batches are exposed to
    // any caller — this is a fully internal pipeline.
    let (sender, receiver) = bounded::<BatchMessage>(16);

    let is_link_prediction = matches!(resolved.prediction_type, PredictionType::Link { .. });
    let has_frequency_col = matches!(
        resolved.prediction_type,
        PredictionType::Link {
            include_frequency: true,
            ..
        }
    );
    let has_include_bit_col = resolved.lag_timesteps > 0;
    let split_ratio = plan.split.as_ref().and_then(|s| s.ratio.clone());
    let split_kind = plan.split.as_ref().and_then(|s| s.kind.clone());
    let split_rnd_seed = plan.split.as_ref().and_then(|s| s.rnd_seed);
    let split_offsets: Option<Vec<Dur<U>>> = plan.split.as_ref().and_then(|s| s.offsets.clone());
    // Optional calendar-month offsets. Mutually exclusive with
    // `split_offsets` for the same SplitSpec; if both are set,
    // `month_offsets` wins (calendar arithmetic in DateOffsetDecider).
    let split_month_offsets: Option<Vec<i32>> =
        plan.split.as_ref().and_then(|s| s.month_offsets.clone());
    let date_offset_semantics = if let Some(s) = plan
        .split
        .as_ref()
        .filter(|s| s.kind.as_deref() == Some("DateOffsetSplit"))
    {
        let unit_name = s.unit.as_deref().unwrap_or("days");
        let unit = TimeUnit::from_name(unit_name).ok_or_else(|| {
            TTGenError::InvalidPlan(format!(
                "DateOffsetSplit unit '{unit_name}' is not supported"
            ))
        })?;
        if matches!(unit, TimeUnit::Months | TimeUnit::Years) {
            return Err(TTGenError::InvalidPlan(format!(
                "DateOffsetSplit unit={unit_name} is not supported for summary derivation"
            )));
        }
        Some((
            dur_from_unit::<U>(s.anchor_start_offset.unwrap_or(0), unit),
            s.timeframe.map(|v| dur_from_unit::<U>(v, unit)),
            // Absent ⇒ kumo production default (flexible_boundaries=True).
            s.flexible_boundaries.unwrap_or(true),
            s.timeframe_step.map(|v| dur_from_unit::<U>(v, unit)),
        ))
    } else {
        None
    };
    let time_range_end_shift = if let Some(s) = plan
        .split
        .as_ref()
        .filter(|s| s.kind.as_deref() == Some("TimeRangeSplit"))
    {
        if let Some(timeframe) = s.timeframe {
            let unit_name = s.unit.as_deref().unwrap_or("days");
            let unit = TimeUnit::from_name(unit_name).ok_or_else(|| {
                TTGenError::InvalidPlan(format!(
                    "TimeRangeSplit unit '{unit_name}' is not supported"
                ))
            })?;
            let shift = timeframe.saturating_add(s.anchor_start_offset.unwrap_or(0));
            let offset = match unit {
                TimeUnit::Months => OffsetUnit::Months(i32::try_from(shift).map_err(|_| {
                    TTGenError::InvalidPlan(format!(
                        "TimeRangeSplit month shift {shift} exceeds the supported range"
                    ))
                })?),
                TimeUnit::Years => OffsetUnit::Years(i32::try_from(shift).map_err(|_| {
                    TTGenError::InvalidPlan(format!(
                        "TimeRangeSplit year shift {shift} exceeds the supported range"
                    ))
                })?),
                _ => OffsetUnit::Fixed(dur_from_unit::<U>(shift, unit)),
            };
            Some(offset)
        } else {
            None
        }
    } else {
        None
    };
    let split_ranges: Option<Vec<(Abs<U>, Abs<U>)>> =
        plan.split.as_ref().and_then(|s| s.ranges.clone());
    let split_backtesting_timestamp: Option<Abs<U>> =
        plan.split.as_ref().and_then(|s| s.backtesting_timestamp);
    let split_column = plan.split.as_ref().and_then(|s| s.split_column.clone());
    let manual_train_values = plan
        .split
        .as_ref()
        .and_then(|s| s.manual_train_values.clone());
    let manual_val_values = plan
        .split
        .as_ref()
        .and_then(|s| s.manual_val_values.clone());
    let manual_test_values = plan
        .split
        .as_ref()
        .and_then(|s| s.manual_test_values.clone());

    // Determine target stype for categorical vs numerical stats
    let target_stype_str = resolved.target_stype.clone();
    let target_is_categorical = matches!(
        target_stype_str.as_str(),
        "categorical" | "boolean" | "ID" | "multicategorical"
    );

    // Node-task categorical targets need their complete encoder vocabulary.
    // Link-prediction targets are graph node IDs: their scalar/HLL stats remain
    // collected, but an exact target-ID distribution has no registered
    // consumer and can grow with the RHS entity cardinality.
    let shared_target_counts =
        category_count_accumulators(target_is_categorical, is_link_prediction);

    // ── Writer thread ────────────────────────────────────────────────────
    // Reads from the bounded(16) channel and writes the tt.parquet dataset.
    // Schema is inferred from the first batch (lazy ArrowWriter construction).
    // Runs concurrently with rayon workers; back-pressure stays via the
    // bounded channel so we don't change the concurrency model.
    let tt_parquet_path = output_dir.join(TT_DATASET_DIR);
    let writer_path = tt_parquet_path.clone();
    // Writer thread output: total_rows. Per-row training weights (custom-TT)
    // ride inside each batch as the reserved `WEIGHT` column, so they flow to
    // `tt.parquet` automatically — no separate accumulation here.
    let writer_thread: std::thread::JoinHandle<TTGenResult<u64>> =
        std::thread::spawn(move || write_tt_dataset(receiver, &writer_path, TT_PART_MAX_ROWS));

    // ── Pass A + Pass B dispatch ─────────────────────────────────────────
    // Dispatch on the resolved split family to pick the per-row
    // decider type `D` for pass A, then run pass B in `SplitMode::Compute`
    // arms. Each arm is monomorphized over D — no `dyn`, no vtable.
    //
    // When a `custom_tt_source` is supplied, the dispatch still
    // chooses a decider based on `split_kind` (so plan.split semantics
    // carry through), but pass A iterates parquet row groups instead of
    // entity ranges. Custom-TT custom splits ride as a SPLIT COLUMN inside
    // the parquet and route through the `Compute` column deciders.
    let custom_tt_source_ref: Option<&CustomTtSource> = custom_tt_source.as_ref();
    let split_mode = determine_split_mode(plan);
    // Array-LP detection. Every Link plan is array-form (LinkArray output).
    let is_array_lp = matches!(plan.prediction_type, PredictionType::Link { .. });

    // Array-form per-link splits (RandomLinkSplit / CustomLinkSplit on an LP
    // plan): run pass A with `NoSplitDecider`, then the executor body runs the
    // fused per-link pass-B. Per-anchor LP splits take the normal per-anchor path.
    let is_per_link_split = is_array_lp
        && matches!(
            split_kind.as_deref(),
            Some("RandomLinkSplit") | Some("CustomLinkSplit")
        )
        && matches!(split_mode, SplitMode::Compute);

    let pass_outputs: TTGenResult<PassAOutputs<U>> = if is_per_link_split {
        // Pass A only (NoSplitDecider). The fused per-link pass-B runs in the
        // executor body once the writer thread has drained the dataset.
        run_pass_a_then_pass_b::<U, NoSplitDecider>(
            &resolved,
            sender,
            num_chunks,
            num_entities,
            shared_target_counts.as_ref(),
            NoSplitDecider,
            &tt_parquet_path,
            writer_thread,
            PassBMode::Deferred,
            custom_tt_source_ref,
            fact_pseudo_time.as_mut(),
        )
    } else {
        match (&split_mode, split_kind.as_deref()) {
            (SplitMode::Compute, Some("TimeRangeSplit")) => {
                let ranges = split_ranges.clone().ok_or_else(|| {
                    TTGenError::InvalidPlan("TimeRangeSplit requires split.ranges".to_string())
                })?;
                let decider =
                    TimeRangeDecider::<U>::new(ranges).with_end_shift(time_range_end_shift);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("BacktestingSplit")) => {
                let test_ts = split_backtesting_timestamp.ok_or_else(|| {
                    TTGenError::InvalidPlan(
                        "BacktestingSplit requires split.backtesting_timestamp".to_string(),
                    )
                })?;
                let decider = BacktestingDecider::<U>::new(test_ts);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("ManualSplit"))
                if matches!(
                    split_column.as_deref(),
                    None | Some("TIMESTAMP") | Some("PSEUDO_TIMESTAMP")
                ) =>
            {
                // ManualSplit on the TIMESTAMP / PSEUDO_TIMESTAMP pseudo-column:
                // value-set membership over the resolved per-row Abs<U>
                // timestamp via `ManualDecider` (mirrors the parity check in
                // `derive_manual_split_mask_counts`). ManualSplit on any OTHER
                // (customer-provided) column falls through to the
                // `ColumnSplit | ManualSplit` arm below, which reads the column
                // via `ColumnDeciderManual` and drops it from the output.
                let parse_set = |vals: &Option<Vec<String>>| -> HashSet<Abs<U>> {
                    vals.as_deref()
                        .map(|xs| {
                            xs.iter()
                                .filter_map(|x| x.parse::<i64>().ok().map(Abs::<U>::from_raw))
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let train = parse_set(&manual_train_values);
                let val = parse_set(&manual_val_values);
                let test = parse_set(&manual_test_values);
                let decider = ManualDecider::<U>::new(train, val, test);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            // ── Tier-2 deciders ──────────────────────────────────────────────
            (SplitMode::Compute, Some("RandomSplit")) => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let seed = split_rnd_seed.unwrap_or(0);
                let decider = RandomDecider::<U>::new(ratio, seed);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("RandomLinkSplit")) => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let seed = split_rnd_seed.unwrap_or(0);
                let decider = RandomLinkDecider::<U>::new(ratio, seed);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("FixedSplit")) => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let decider = FixedDecider::<U>::new(ratio);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("TemporalSplit")) => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let decider = TemporalDecider::<U>::new(ratio);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("DateOffsetSplit")) => {
                // Month/year offsets stay symbolic (OffsetUnit::Months) — calendar
                // arithmetic needs the anchor, resolved later in derive_cutoffs.
                // Fixed offsets + timeframe/anchor are Dur<U>.
                let (anchor, timeframe, flexible, tf_step) =
                    date_offset_semantics.unwrap_or((Dur::<U>::zero(), None, true, None));
                let decider = if let Some(months) = split_month_offsets.clone() {
                    let units: Vec<OffsetUnit<U>> =
                        months.into_iter().map(OffsetUnit::Months).collect();
                    DateOffsetDecider::<U>::new_with_units(units, anchor, timeframe)
                } else {
                    let offsets = split_offsets.clone().unwrap_or_default();
                    DateOffsetDecider::<U>::new(offsets, anchor, timeframe)
                }
                .with_flexible_boundaries(flexible, tf_step);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            (SplitMode::Compute, Some("ApproxDateOffsetSplit")) => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let decider = ApproxDateOffsetDecider::<U>::new(ratio);
                run_pass_a_then_pass_b::<U, _>(
                    &resolved,
                    sender,
                    num_chunks,
                    num_entities,
                    shared_target_counts.as_ref(),
                    decider,
                    &tt_parquet_path,
                    writer_thread,
                    PassBMode::PerAnchor {
                        is_array_lp,
                        drop_split_column: None,
                    },
                    custom_tt_source_ref,
                    fact_pseudo_time.as_mut(),
                )
            }
            // ── ColumnSplit / ManualSplit-on-column ─────────────────────────
            //
            // Three sub-cases distinguished by what cutoffs come from:
            //   (α) manual_*_values set → `ColumnDeciderManual` (string-set
            //       membership). Stateless decider; pass A is a no-op for it.
            //   (γ) ratio set → `ColumnDeciderRatioNumeric` (numeric t-digest).
            //       Pass A accumulates a t-digest of `row.split_column_value`
            //       (numeric cast); `derive_cutoffs` quantizes at cumulative
            //       ratio thresholds.
            //   (β) explicit numeric cutoffs: NOT plumbed through SplitSpec
            //       today (`SplitSpec` carries no `cutoffs` field). Future
            //       schema extension will route here.
            //   (δ) ratio-derived on a string column: SCOPED OUT. Reservoir
            //       sampling across rayon workers needs a Reservoir merge
            //       primitive that doesn't exist yet. We return InvalidPlan
            //       for `kind=ColumnSplit + ratio` when the column dtype is
            //       known to be string at decide time — but pass-A dispatch
            //       can't peek the rowpack dtype here, so the runtime check
            //       happens via `value_to_f64` returning `None` for non-numeric
            //       Values (rows with string column values become UNUSED).
            (SplitMode::Compute, Some("ColumnSplit"))
            | (SplitMode::Compute, Some("ManualSplit")) => {
                let column_name = split_column.clone().ok_or_else(|| {
                    TTGenError::InvalidPlan(
                        "ColumnSplit / ManualSplit-on-column requires split.split_column"
                            .to_string(),
                    )
                })?;
                // The split column is read from `tt.parquet` under the split
                // projection's output name (the chunk processor / executor emit
                // it there). It is the split SIGNAL — drop it from the per-stage
                // seed parquets so it never reaches the seed rowpack / features.
                let drop_col = resolved
                    .split_projection
                    .as_ref()
                    .map(|p| p.output_name.clone())
                    .unwrap_or_else(|| column_name.clone());
                let has_manual = manual_train_values.is_some()
                    || manual_val_values.is_some()
                    || manual_test_values.is_some();
                let has_ratio = split_ratio.is_some();
                if has_manual {
                    // (α) Manual value sets on an arbitrary column. This is the
                    // customer-provided CTT split-column path (kumo emits
                    // `ManualSplit{split_column, manual_*_values}`): each row's
                    // split-column value is matched against the value sets via
                    // `ColumnDeciderManual` and the column is dropped on output.
                    let train = manual_train_values.clone().unwrap_or_default();
                    let val = manual_val_values.clone().unwrap_or_default();
                    let test = manual_test_values.clone().unwrap_or_default();
                    let decider = ColumnDeciderManual::new(drop_col.clone(), train, val, test);
                    run_pass_a_then_pass_b::<U, _>(
                        &resolved,
                        sender,
                        num_chunks,
                        num_entities,
                        shared_target_counts.as_ref(),
                        decider,
                        &tt_parquet_path,
                        writer_thread,
                        PassBMode::PerAnchor {
                            is_array_lp,
                            drop_split_column: Some(drop_col),
                        },
                        custom_tt_source_ref,
                        fact_pseudo_time.as_mut(),
                    )
                } else if has_ratio {
                    // (γ) Ratio-derived cutoffs on a numeric column. The
                    // decider's `accumulate` filters non-numeric values via
                    // `value_to_f64`; if every row is non-numeric the digest
                    // is empty and pass B will mark every row UNUSED — that's
                    // the `(δ)`-like fallback documented above.
                    let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                    let decider = ColumnDeciderRatioNumeric::new(drop_col.clone(), ratio);
                    run_pass_a_then_pass_b::<U, _>(
                        &resolved,
                        sender,
                        num_chunks,
                        num_entities,
                        shared_target_counts.as_ref(),
                        decider,
                        &tt_parquet_path,
                        writer_thread,
                        PassBMode::PerAnchor {
                            is_array_lp,
                            drop_split_column: Some(drop_col),
                        },
                        custom_tt_source_ref,
                        fact_pseudo_time.as_mut(),
                    )
                } else {
                    return Err(TTGenError::InvalidPlan(
                        "ColumnSplit requires either manual_*_values or ratio".to_string(),
                    ));
                }
            }
            // SplitMode::None or any unrecognized kind: NoSplitDecider, no pass B.
            _ => run_pass_a_then_pass_b::<U, NoSplitDecider>(
                &resolved,
                sender,
                num_chunks,
                num_entities,
                shared_target_counts.as_ref(),
                NoSplitDecider,
                &tt_parquet_path,
                writer_thread,
                PassBMode::Deferred,
                custom_tt_source_ref,
                fact_pseudo_time.as_mut(),
            ),
        }
    };

    let (mut total_rows, stats, per_anchor_parts) = pass_outputs?;
    // `total_rows` = the dataset row count pass A wrote. SPLIT plans fold the
    // array-LP empty-target drop into the route kernels (empty rows route to no
    // stage), leaving it immutable.

    // No-split (prediction) array-LP empty-target drop: the dataset IS the
    // single ingest input, so drop empty rows in place here. Gated to array LP,
    // no split of any kind, and a non-empty dataset; non-LP array tasks may
    // legitimately carry empty lists.
    // Predict LP bundles carry no TARGET1 column (entities + time only), so
    // there is nothing to drop — skip the empty-target filter for predict.
    let is_no_split = per_anchor_parts.is_none() && !is_per_link_split;
    if is_array_lp && is_no_split && total_rows > 0 && !plan.is_prediction {
        let keep = crate::regroup::drop_empty_target_rows_in_place(&tt_parquet_path)?;
        debug_assert_eq!(
            keep.len() as u64,
            total_rows,
            "empty-target drop keep-mask must be row-aligned to tt.parquet"
        );
        let kept = keep.iter().filter(|k| **k).count() as u64;
        if kept != total_rows {
            // Weight rides as a column now, so it is dropped in place with
            // the row by `drop_empty_target_rows_in_place` — no separate
            // sidecar realign needed.
            total_rows = kept;
        }
    }

    // Node-task categorical targets require exact CATEGORY_COUNTS for encoder
    // inference. LP deliberately has no accumulator, so this returns None.
    let target_category_counts =
        drain_required_target_category_counts(shared_target_counts.as_deref())?;

    // ENTITY1 exact category counts are intentionally not collected: the entity
    // column is a graph foreign key and has no category-count consumer, so no
    // accumulator is allocated and finalize receives `None`.
    let entity_category_counts = None;

    let finalized_stats = stats.finalize(
        total_rows,
        is_link_prediction,
        has_frequency_col,
        has_include_bit_col,
        target_category_counts,
        target_stype_str.as_str(),
        entity_category_counts,
    );

    // ── Fused pass-B by mode → per-stage seed parts ──────────────────────
    //
    // Per-anchor Compute arms already ran their fused pass-B in
    // `run_pass_a_then_pass_b` (including the custom-TT split-column path,
    // which routes via `ColumnDeciderManual` and drops the split column). Here
    // we run the deferred modes:
    //   - per-link: build a `LinkAssigner` and route each anchor's list per link.
    //   - no-split (prediction): no pass-B; the single stage is `tt.parquet`.
    let stage_parts: Option<[StagePartsOut; 3]> = if let Some(parts) = per_anchor_parts {
        Some(parts)
    } else if is_per_link_split {
        // CustomLink drops the split projection column from the output schema;
        // RandomLink keeps the full tt schema.
        let tt_schema = probe_tt_schema(&tt_parquet_path)?;
        let (assigner, out_schema) = match split_kind.as_deref() {
            Some("RandomLinkSplit") => {
                let ratio = split_ratio.clone().unwrap_or_else(|| vec![0.8, 0.1, 0.1]);
                let seed = split_rnd_seed.unwrap_or(0);
                // Output schema appends the past-visit `exclude_target_keys` col.
                let out_schema = per_link_output_schema(&tt_schema, None);
                (LinkAssigner::random_from_ratio(&ratio, seed), out_schema)
            }
            Some("CustomLinkSplit") => {
                let split_col_name = plan
                    .split
                    .as_ref()
                    .and_then(|s| {
                        s.split_output_column
                            .clone()
                            .or_else(|| s.split_column.clone())
                    })
                    .ok_or_else(|| {
                        TTGenError::InvalidPlan(
                            "CustomLinkSplit requires split.split_column or split_output_column"
                                .to_string(),
                        )
                    })?;
                let train = manual_train_values.clone().unwrap_or_default();
                let val = manual_val_values.clone().unwrap_or_default();
                let test = manual_test_values.clone().unwrap_or_default();
                if train.is_empty() && val.is_empty() && test.is_empty() {
                    return Err(TTGenError::InvalidPlan(
                        "CustomLinkSplit requires at least one of \
                         manual_train_values / manual_val_values / manual_test_values"
                            .to_string(),
                    ));
                }
                // Drop the split projection col + append `exclude_target_keys`.
                let out_schema = per_link_output_schema(&tt_schema, Some(&split_col_name));
                let assigner = LinkAssigner::CustomLink {
                    train: train.into_iter().collect(),
                    val: val.into_iter().collect(),
                    test: test.into_iter().collect(),
                    split_col: split_col_name,
                };
                (assigner, out_schema)
            }
            other => {
                return Err(TTGenError::InvalidPlan(format!(
                    "unexpected split kind for array-LP per-link pass-B: {other:?}"
                )));
            }
        };
        if let Some(marker) = &mut fact_pseudo_time {
            marker.initialize();
        }
        Some(run_pass_b_per_link(
            &tt_parquet_path,
            &assigner,
            output_dir,
            total_rows,
            &out_schema,
            fact_pseudo_time.as_ref(),
        )?)
    } else {
        // No split / prediction: no pass-B; ingest reads tt.parquet directly.
        None
    };

    let stage_gen_infos: Option<Vec<StageGenInfo>> = stage_parts
        .as_ref()
        .map(|parts| stage_gen_infos_from_parts(parts, output_dir))
        .transpose()?;

    // Stats sidecar (kumo `tt_stats` envelope).
    let stats_path = write_stats_artifact(output_dir, &finalized_stats)?;

    let source_mode_str = if custom_tt_source.is_some() {
        "custom_tt"
    } else {
        "pquery"
    };

    // The parquet writer and any in-place regroup/drop pass have completed.
    // Freeze the exact closed-file size without an additional full-file read.
    let (train_table_size_bytes, train_table_parts) =
        train_table_parts(&tt_parquet_path, total_rows)?;
    let stats_sha = sha256_file(&stats_path)?;

    // TT-NG carries stage row-counts + seed-times per-stage in the ingest-output
    // manifest (not split.bin/arange.bin), so the legacy flat-manifest split
    // fields are left `None` (struct retained for the custom-TT/stats path).
    let mut manifest = TTManifest {
        version: TT_MANIFEST_VERSION,
        row_count: total_rows,
        source_mode: source_mode_str.to_string(),
        split_mode: "none".to_string(),
        split_kind: split_kind.clone(),
        stage_counts: None,
        train_table: TrainTableArtifactRef::new(
            tt_parquet_path
                .file_name()
                .expect("tt.parquet path has a filename")
                .to_string_lossy()
                .into_owned(),
            TT_PARQUET_FORMAT,
            total_rows,
            train_table_size_bytes,
            train_table_parts,
        ),
        split: None,
        stats: ArtifactRef::new(
            stats_path
                .file_name()
                .expect("stats artifact path has a filename")
                .to_string_lossy()
                .into_owned(),
            STATS_FORMAT,
        ),
        split_stats: None,
        stage_seed_times: None,
        train_table_col_stats: None,
        // Edge bundle: NOT produced by `execute_tt_pass` — emitted by a
        // separate `export_tt_artifacts` call. Reader treats absent as
        // "disabled bundle".
        train_table_edges: None,
        arange_sidecar: None,
        // Weight rides as the reserved `WEIGHT` column now; sidecar field kept None.
        weight_sidecar: None,
        fact_pseudo_time: None,
        checksums: ManifestChecksums {
            split_sha256: String::new(),
            stats_sha256: stats_sha,
            split_stats_sha256: None,
            train_table_col_stats_sha256: None,
            train_table_edges_sha256: None,
            weight_sidecar_sha256: None,
            fact_pseudo_time_sha256: None,
        },
    };

    if let (Some(spec), Some(marker)) = (plan.fact_pseudo_time.as_ref(), fact_pseudo_time) {
        let parts = stage_parts.as_ref().ok_or_else(|| {
            TTGenError::InvalidPlan("fact_pseudo_time requires stage parts".to_string())
        })?;
        let nonempty = if custom_tt_source.is_some() {
            total_rows > 0
        } else {
            parts.iter().any(|stage| stage.row_count > 0)
        };
        let path = marker.finish(output_dir, nonempty, custom_tt_source.is_none())?;
        manifest.version = crate::manifest::FACT_PSEUDO_TIME_MANIFEST_VERSION;
        manifest.checksums.fact_pseudo_time_sha256 = Some(sha256_file(&path)?);
        manifest.fact_pseudo_time = Some(crate::manifest::FactPseudoTimeRef {
            path: crate::pseudo_time::FACT_PSEUDO_TIME_FILE.to_string(),
            format: crate::manifest::FACT_PSEUDO_TIME_FORMAT.to_string(),
            node_type: spec.table.clone(),
        });
    }

    if let Some(parts) = stage_parts.as_ref() {
        if let Some((path, sha)) = crate::split_stats::write_kumo_split_stats_artifact(
            output_dir,
            parts,
            &plan.prediction_type,
            target_stype_str.as_str(),
        )? {
            manifest.split_stats = Some(ArtifactRef::new(
                path.file_name()
                    .expect("split-stats artifact path has a filename")
                    .to_string_lossy()
                    .into_owned(),
                SPLIT_STATS_FORMAT,
            ));
            manifest.checksums.split_stats_sha256 = Some(sha);
        }
    }

    // Kumo-contract bridge: emits `train_table_col_stats.json` + populates
    // the matching manifest field + SHA-256 checksum.
    crate::kumo_contract_bridge::write_all(&finalized_stats, output_dir, &mut manifest)?;

    let manifest_path = write_manifest(output_dir, &manifest)?;

    tracing::info!(
        total_rows,
        parquet_path = %tt_parquet_path.display(),
        stage_gen_infos = ?stage_gen_infos,
        manifest = %manifest_path.display(),
        "TT-gen execution finished"
    );

    Ok(TTPassOutput {
        rows_written: total_rows,
        tt_parquet_path,
        stats: Some(finalized_stats),
        stage_gen_infos,
        manifest_path,
    })
}

/// Generic pass-A driver: runs the rayon scope, joins the writer thread,
/// and (when `compute_split`) runs pass B against the freshly-written
/// `tt.parquet`. Monomorphized per `D: SplitDecider<U>` so per-row
/// dispatch into the decider is statically resolved.
///
/// Takes ownership of `sender` (so we can drop it after pass A to signal
/// end-of-stream to the writer thread without UB) and `writer_thread`
/// (joined unconditionally so failures don't leak threads).
///
/// `compute_split == true` triggers pass B; otherwise the third tuple
/// element is `None`.
#[allow(clippy::too_many_arguments)]
fn run_pass_a_then_pass_b<U: UnitSystem, D: SplitDecider<U> + Clone + Sync>(
    resolved: &Arc<ResolvedPlan<U>>,
    sender: Sender<BatchMessage>,
    num_chunks: usize,
    num_entities: u64,
    shared_target_counts: Option<&Arc<SharedColumnCounts>>,
    decider_template: D,
    tt_parquet_path: &Path,
    writer_thread: std::thread::JoinHandle<TTGenResult<u64>>,
    pass_b_mode: PassBMode,
    custom_tt_source: Option<&CustomTtSource>,
    mut fact_pseudo_time: Option<&mut crate::pseudo_time::FactPseudoTime>,
) -> TTGenResult<PassAOutputs<U>> {
    let pass_a_result = if let Some(source) = custom_tt_source {
        run_pass_a_custom_tt::<U, D>(
            resolved,
            &sender,
            shared_target_counts,
            &decider_template,
            &source.parquet_path,
            source.input_timestamp_unit,
        )
    } else {
        run_pass_a::<U, D>(
            resolved,
            &sender,
            num_chunks,
            num_entities,
            shared_target_counts,
            &decider_template,
        )
    };

    // Drop the sender so the writer thread sees end-of-stream and finishes
    // (whether pass A succeeded or failed).
    drop(sender);

    // Join the writer thread regardless of pass A outcome so we don't leak
    // a worker on the error path.
    let writer_join = writer_thread.join();

    // Now combine: any pass A error wins; otherwise propagate writer errors.
    let (total_rows_from_pass_a, stats, merged_decider) = pass_a_result?;

    let writer_rows = match writer_join {
        Ok(Ok(rows)) => rows,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(TTGenError::SystemError(
                "TT-gen parquet writer thread panicked".to_string(),
            ))
        }
    };

    if writer_rows != total_rows_from_pass_a {
        // Sanity: the per-worker rows accumulator and the writer-thread
        // accumulator should agree. They can diverge only via a logic bug
        // in this file, so make the failure loud.
        return Err(TTGenError::SystemError(format!(
            "rows-written disagreement: pass A counted {}, writer counted {}",
            total_rows_from_pass_a, writer_rows
        )));
    }

    let total_rows = writer_rows;

    // ── Fused pass-B (per-anchor only) ───────────────────────────────────
    // The merged decider's `derive_cutoffs()` already ran inside `run_pass_a`,
    // so it is ready here. Only per-anchor runs pass-B here (it needs the
    // decider); deferred modes (per-link / use-bin / no-split) run it in the
    // executor body.
    let stage_parts = match pass_b_mode {
        PassBMode::PerAnchor {
            is_array_lp,
            drop_split_column,
        } => {
            // Output schema = tt.parquet schema minus the split column (when the
            // custom-TT split-column path asked to drop it). Used both for the
            // per-stage sub-batch projection and for empty-stage 0-row parts.
            let tt_schema = probe_tt_schema(tt_parquet_path)?;
            let out_schema = match drop_split_column.as_deref() {
                Some(col) => schema_without_column(&tt_schema, col),
                None => tt_schema,
            };
            let forecasting_horizon = forecasting_horizon_config(resolved)?;
            if let Some(marker) = &mut fact_pseudo_time {
                marker.initialize();
            }
            Some(run_pass_b_per_anchor::<U, D>(
                tt_parquet_path,
                &merged_decider,
                output_dir_of(tt_parquet_path)?,
                total_rows,
                is_array_lp,
                drop_split_column.as_deref(),
                &out_schema,
                forecasting_horizon,
                fact_pseudo_time.as_deref(),
            )?)
        }
        PassBMode::Deferred => None,
    };

    Ok((total_rows, stats, stage_parts))
}

/// The directory containing `tt.parquet` — the staging dir where per-stage
/// parts are written.
fn output_dir_of(tt_parquet_path: &Path) -> TTGenResult<&Path> {
    tt_parquet_path
        .parent()
        .ok_or_else(|| TTGenError::SystemError("tt_parquet_path has no parent dir".to_string()))
}

fn forecasting_horizon_config<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
) -> TTGenResult<Option<(i64, i64)>> {
    if !matches!(plan.prediction_type, PredictionType::Forecasting) || plan.is_prediction {
        return Ok(None);
    }
    let Some(offsets) = plan.future_label_offsets.as_ref() else {
        return Ok(None);
    };
    if offsets.len() <= 1 {
        return Ok(None);
    }

    let zero = Abs::<U>::from_raw(0);
    let to_ns = |dur: Dur<U>| (zero + dur).to_nanos() - zero.to_nanos();
    let first = to_ns(offsets[0]);
    if first != 0 {
        return Err(TTGenError::InvalidPlan(
            "forecasting future offsets must start at 0".to_string(),
        ));
    }
    let step = to_ns(offsets[1]) - first;
    if step <= 0 {
        return Err(TTGenError::InvalidPlan(
            "forecasting future offsets must be increasing".to_string(),
        ));
    }
    for (idx, offset) in offsets.iter().copied().enumerate() {
        let expected = first + step.saturating_mul(idx as i64);
        if to_ns(offset) != expected {
            return Err(TTGenError::InvalidPlan(
                "forecasting future offsets must be uniformly spaced".to_string(),
            ));
        }
    }
    Ok(Some((step, offsets.len() as i64)))
}

// Generic over `D: SplitDecider<U>`. Spawns the rayon parallel scope for
// pass A: each worker accumulates a `RowProcessor<U, D>` (stats + decider).
// Workers' results are then folded sequentially into a `StatsMerger<U>` and
// a single merged decider — both folds are AFTER the rayon scope, no shared
// state during workers (the bounded(16) channel + writer thread is unchanged).
//
// The merged decider is plumbed out so pass B can consume it. The
// caller passes the same decider clone-template that workers use; we fold
// each worker's accumulator state (no-op for tier-1) into a single instance
// and call `derive_cutoffs()` on it before returning, so the returned
// decider is "ready for pass B".
#[allow(clippy::too_many_arguments)]
fn run_pass_a<U: UnitSystem, D: SplitDecider<U> + Clone + Sync>(
    resolved: &Arc<ResolvedPlan<U>>,
    sender: &Sender<BatchMessage>,
    num_chunks: usize,
    num_entities: u64,
    shared_target_counts: Option<&Arc<SharedColumnCounts>>,
    decider_template: &D,
) -> TTGenResult<(u64, StatsMerger<U>, D)> {
    let results: Vec<TTGenResult<(u64, RowProcessor<U, D>)>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx as u64 * num_entities / num_chunks as u64;
            let end = (chunk_idx as u64 + 1) * num_entities / num_chunks as u64;
            execute_chunk(
                resolved,
                start..end,
                sender,
                shared_target_counts.cloned(),
                decider_template,
            )
        })
        .collect();

    let mut total_rows = 0u64;
    let mut stats: StatsMerger<U> = StatsMerger::default();
    let mut decider_merger: D = decider_template.clone();
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok((rows, processor)) => {
                total_rows += rows;
                let RowProcessor {
                    stats: chunk_stats,
                    decider,
                } = processor;
                stats.merge(chunk_stats);
                decider_merger.merge(&decider);
            }
            Err(e) => {
                return Err(e.with_context(format!("thread {i} failed")));
            }
        }
    }
    decider_merger.derive_cutoffs();
    decider_merger.validate_derived_cutoffs()?;
    Ok((total_rows, stats, decider_merger))
}

/// Enumerate the parquet part(s) backing a custom-TT source.
///
/// kumo passes the train-table **directory** (containing one or more
/// `part-*.parquet`) for the multi-part contract; legacy callers pass a single
/// `*.parquet` file. This normalizes both:
///
/// * a directory → all `*.parquet` files inside it, sorted by file name
///   (`part-00000`, `part-00001`, ... — the same lexical part order the PQL
///   multi-part stage read relies on);
/// * a file → that one path.
///
/// Schema inference happens per-part inside the chunk processor; routing is
/// per-row via the `split_col` column (Phase 1), so multi-part is safe — there
/// is no positional/external alignment across parts.
pub(crate) fn enumerate_custom_tt_parts(path: &Path) -> TTGenResult<Vec<PathBuf>> {
    if path.is_dir() {
        let mut parts: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"))
            })
            .collect();
        if parts.is_empty() {
            return Err(TTGenError::SystemError(format!(
                "custom-TT source directory {} contains no *.parquet parts",
                path.display()
            )));
        }
        // Sort by file name for deterministic part order (part-00000, ...).
        parts.sort();
        Ok(parts)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

// Pass A driver for the custom-TT path. Mirrors `run_pass_a` but
// parallelizes over the **parquet row groups** of the customer's input — which
// may be a single file OR a multi-part directory. Each rayon worker handles one
// `(part, row_group)` pair (one part for a single file), so parallelism scales
// with both the customer's part count and per-part row-group count.
//
// Same shared/merged contract: returns `(total_rows, StatsMerger<U>, D)`
// with `derive_cutoffs()` called, ready for pass B.
fn run_pass_a_custom_tt<U: UnitSystem, D: SplitDecider<U> + Clone + Sync>(
    resolved: &Arc<ResolvedPlan<U>>,
    sender: &Sender<BatchMessage>,
    shared_target_counts: Option<&Arc<SharedColumnCounts>>,
    decider_template: &D,
    custom_tt_parquet: &Path,
    input_timestamp_unit: TimeUnit,
) -> TTGenResult<(u64, StatsMerger<U>, D)> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    // Normalize file-vs-directory into the list of parts, then probe each part's
    // row-group count. Build a flat `(part_path, rg_idx)` work list so every
    // (part, row group) is its own rayon task — mirrors the PQL multi-part stage
    // read, where each `StageSpec.parquet_paths` entry contributes its own parts.
    let parts = enumerate_custom_tt_parts(custom_tt_parquet)?;
    let mut work: Vec<(PathBuf, usize)> = Vec::new();
    for part in &parts {
        let probe_file = std::fs::File::open(part)?;
        let probe = ParquetRecordBatchReaderBuilder::try_new(probe_file)?;
        let num_row_groups = probe.metadata().row_groups().len();
        drop(probe);
        for rg_idx in 0..num_row_groups {
            work.push((part.clone(), rg_idx));
        }
    }

    if work.is_empty() {
        // No row groups across any part — produce zero rows. Still merge a
        // default decider.
        let stats = StatsMerger::<U>::default();
        let mut decider = decider_template.clone();
        decider.derive_cutoffs();
        decider.validate_derived_cutoffs()?;
        return Ok((0, stats, decider));
    }

    let results: Vec<TTGenResult<(u64, RowProcessor<U, D>)>> = work
        .par_iter()
        .map(|(part_path, rg_idx)| {
            crate::custom_tt_chunk_processor::custom_tt_chunk_processor(
                resolved,
                part_path,
                &[*rg_idx],
                sender,
                decider_template,
                shared_target_counts.cloned(),
                input_timestamp_unit,
            )
        })
        .collect();

    let mut total_rows = 0u64;
    let mut stats: StatsMerger<U> = StatsMerger::default();
    let mut decider_merger: D = decider_template.clone();
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok((rows, processor)) => {
                total_rows += rows;
                let RowProcessor {
                    stats: chunk_stats,
                    decider,
                } = processor;
                stats.merge(chunk_stats);
                decider_merger.merge(&decider);
            }
            Err(e) => {
                return Err(e.with_context(format!("custom-TT row-group worker {i} failed")));
            }
        }
    }
    decider_merger.derive_cutoffs();
    decider_merger.validate_derived_cutoffs()?;
    Ok((total_rows, stats, decider_merger))
}

// ─── Per-thread execution ────────────────────────────────────────────────────

pub(crate) fn output_mode_from_prediction_type<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
) -> OutputMode {
    match &plan.prediction_type {
        PredictionType::Node => OutputMode::Node {
            target_dtypes: plan.target_dtypes.clone(),
        },
        // FK-free / array LP is the only link-prediction output mode: every
        // Link plan emits LinkArray.
        PredictionType::Link { include_frequency } => OutputMode::LinkArray {
            include_frequency: *include_frequency,
            target_dtypes: if plan.target_dtypes.is_empty() {
                vec![DataType::Float64]
            } else {
                plan.target_dtypes.clone()
            },
        },
        PredictionType::PassThrough => OutputMode::PassThrough,
        PredictionType::Forecasting => OutputMode::Forecasting {
            target_dtypes: if !plan.is_prediction && !plan.stages.is_empty() {
                plan.target_dtypes.clone()
            } else {
                Vec::new()
            },
            // Empty-stages forecasting plans (custom-TT-complete) have no
            // autoregression op in the plan, so no inferred dtype.
            // Default to List<Float64>, the standard PQ forecasting shape.
            autoregression_dtype: plan.autoregression_dtype.clone().unwrap_or_else(|| {
                DataType::List(std::sync::Arc::new(arrow_schema::Field::new(
                    "item",
                    DataType::Float64,
                    true,
                )))
            }),
            future_labels_dtype: plan.future_labels_dtype.clone(),
            yoy_dtype: plan.yoy_dtype.clone(),
        },
    }
}

fn execute_chunk<U: UnitSystem, D: SplitDecider<U> + Clone>(
    plan: &ResolvedPlan<U>,
    entity_range: std::ops::Range<u64>,
    sender: &Sender<BatchMessage>,
    shared_target_counts: Option<Arc<SharedColumnCounts>>,
    decider_template: &D,
) -> TTGenResult<(u64, RowProcessor<U, D>)> {
    let mode = output_mode_from_prediction_type(plan);
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

    // Thread-local aliases Vec, reused across entities
    let mut aliases: Vec<Option<Value>> = vec![None; plan.num_aliases];
    // Thread-local frequency counts for ListDistinct (reused across entities)
    let mut link_frequencies: Vec<i64> = Vec::new();
    let mut link_split_values: Vec<Value> = Vec::new();

    // Determine target stats mode from target_stype
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

    // Propagate the resolved target dtype into the stats accumulator so
    // bool-target CATEGORY_COUNTS emits JSON booleans (matches the
    // BooleanArray downcast in tt_executor.rs:compute_train_table_col_stats_payload
    // for the rehydration path; without this the ttgen path silently
    // emits lowercase "true"/"false" strings).
    let mut stats: ChunkStats<U> = ChunkStats {
        target_is_bool_dtype: matches!(plan.primary_target_dtype(), DataType::Boolean),
        ..ChunkStats::default()
    };
    if target_is_numerical {
        stats.target_welford = Some(WelfordAccumulator::new());
        stats.target_tdigest = Some(TDigestEstimator::new(200));
    }
    if include_frequency {
        stats.frequency_welford = Some(WelfordAccumulator::new());
        stats.frequency_tdigest = Some(TDigestEstimator::new(200));
        stats.frequency_hll = Some(HyperLogLog::new());
    }
    // HLL is always-on. It's the sole cardinality source after
    // the prior `entities` / `targets` / `timestamps` HashSets were
    // dropped, so it must be present for every stype (numerical,
    // categorical, and any future variant).
    stats.target_hll = Some(HyperLogLog::new());
    stats.entity_hll = Some(HyperLogLog::new());
    stats.entity_welford = Some(WelfordAccumulator::new());
    stats.entity_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_welford = Some(WelfordAccumulator::new());
    stats.timestamp_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_hll = Some(HyperLogLog::new());

    // Wrap stats + per-worker decider into a RowProcessor. The
    // wrapper is monomorphized over (U, D); per-row dispatch into both
    // halves is fully inlined, no vtable.
    let mut processor: RowProcessor<U, D> = RowProcessor::new(stats, decider_template.clone());

    // Thread-local categorical target counts, flushed to shared at end. Only
    // allocate when there is a shared accumulator to flush into: link-prediction
    // targets have no exact-vocabulary consumer (`shared_target_counts` is None),
    // so collecting them would be dead work.
    let mut local_target_counts: Option<LocalColumnCounts> = shared_target_counts
        .is_some()
        .then(|| LocalColumnCounts::with_capacity(5000));

    for entity_node in entity_range {
        execute_entity(
            plan,
            entity_node,
            &mut aliases,
            &mut link_frequencies,
            &mut link_split_values,
            target_is_numerical,
            target_is_categorical,
            target_stype,
            &mut local_target_counts,
            &mut writer,
            &mut processor,
        )?;
    }

    // Flush local categorical counts to shared
    if let (Some(ref shared), Some(ref mut local)) =
        (&shared_target_counts, &mut local_target_counts)
    {
        shared.merge_local_batch(local);
    }

    let rows = writer.finish()?;
    processor.stats.num_total_rows = rows;
    Ok((rows, processor))
}

// ─── Per-entity execution ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn execute_entity<U: UnitSystem, D: SplitDecider<U>>(
    plan: &ResolvedPlan<U>,
    entity_node: u64,
    aliases: &mut [Option<Value>],
    link_frequencies: &mut Vec<i64>,
    link_split_values: &mut Vec<Value>,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
    writer: &mut ChannelBatchWriter<U>,
    processor: &mut RowProcessor<U, D>,
) -> TTGenResult<()> {
    // 0. Look up the entity PK value for output (slot 0 — PQ path is single-entity)
    let entity_pk = read_column_value(
        &plan.entity_table,
        entity_node,
        plan.primary_entity_pk_col_idx(),
    )
    .unwrap_or(Value::Int(entity_node as i64));

    // 0b. Skip duplicate PK nodes. Ingest assigns separate node IDs to
    //     duplicate PKs; the MPHF stores only the canonical (first) one.
    if !plan.entity_table.is_canonical_node(entity_node, &entity_pk) {
        return Ok(());
    }

    // 1. Read entity time_col and valid_to_col
    let time_col_val: Option<Abs<U>> = plan
        .entity_time_col_idx
        .and_then(|idx| read_column_i64(&plan.entity_table, entity_node, idx))
        .map(Abs::<U>::from_secs);
    let valid_to_val: Option<Abs<U>> = plan
        .entity_valid_to_col_idx
        .and_then(|idx| read_column_i64(&plan.entity_table, entity_node, idx))
        .map(Abs::<U>::from_secs);

    // 2. Pre-resolve entity column values (for filter RHS EntityColumn)
    let entity_cols: Vec<Option<Value>> = plan
        .entity_col_indices
        .iter()
        .map(|(_, col_idx)| read_column_value(&plan.entity_table, entity_node, *col_idx))
        .collect();
    let split_value = plan
        .split_projection
        .as_ref()
        .and_then(|projection| projection.entity_col_idx)
        .map(|col_idx| {
            read_column_value_nullable(&plan.entity_table, entity_node, col_idx)
                .flatten()
                .unwrap_or_else(|| Value::Str("None".to_string()))
        });

    // 3. Entity-level filter
    //    When lag_timesteps > 0, don't skip entities that fail the filter.
    //    Instead, emit rows with entity_include_bit = false (needed for
    //    autoregressive lag features in the ML model).
    let entity_include_bit = match plan.entity_filter {
        Some(ref filter) => eval_row_filter(filter, entity_node, &plan.entity_table, &entity_cols),
        None => true,
    };
    if !entity_include_bit && plan.lag_timesteps <= 0 {
        return Ok(());
    }

    if !plan.has_temporal {
        // Non-temporal: single row per entity.
        // Match Kumo drop_na_rows(TIMESTAMP): skip if entity time_col is NULL.
        let default_ts = plan
            .entity_table_spec
            .default_time_value
            .unwrap_or(Abs::<U>::from_raw(0));
        let ts = match (plan.entity_time_col_idx, time_col_val) {
            (Some(_), None) => return Ok(()), // time_col configured but NULL → drop
            (_, Some(t)) => t,
            (None, None) => default_ts,
        };
        link_frequencies.clear();
        link_split_values.clear();
        if execute_stages(
            plan,
            entity_node,
            &entity_cols,
            None,
            aliases,
            link_frequencies,
            link_split_values,
        ) {
            write_target(
                plan,
                &entity_pk,
                ts,
                aliases,
                link_frequencies,
                link_split_values.as_slice(),
                split_value.as_ref(),
                entity_include_bit,
                target_is_numerical,
                target_is_categorical,
                target_stype,
                local_target_counts,
                writer,
                processor,
            )?;
        }
        aliases.iter_mut().for_each(|a| *a = None);
        return Ok(());
    }

    // 4. Temporal: binary search for valid timestamp range
    let timestamps = &plan.timestamps;

    // Timestamps are sorted ascending. Point anchors (epoch seconds).
    // Entity excluded if time_col > ts (strict >)
    // Entity excluded if valid_to_col < ts (strict <, i.e. LEQ boundary)
    let first_valid = match time_col_val {
        Some(t) => {
            // First ts where entity exists: ts >= time_col
            timestamps.partition_point(|ts| *ts < t)
        }
        None => 0, // No time_col → entity exists for all timestamps
    };
    let last_valid = match valid_to_val {
        Some(v) => {
            // Last ts before entity expires: ts <= valid_to (LEQ)
            timestamps.partition_point(|ts| *ts <= v)
        }
        None => timestamps.len(), // NULL valid_to → never expires
    };

    // 5. Iterate valid timestamps
    for &ts in timestamps.iter().take(last_valid).skip(first_valid) {
        // Execute all stages for this (entity, timestamp)
        link_frequencies.clear();
        link_split_values.clear();
        if execute_stages(
            plan,
            entity_node,
            &entity_cols,
            Some(ts),
            aliases,
            link_frequencies,
            link_split_values,
        ) {
            write_target(
                plan,
                &entity_pk,
                ts,
                aliases,
                link_frequencies,
                link_split_values.as_slice(),
                split_value.as_ref(),
                entity_include_bit,
                target_is_numerical,
                target_is_categorical,
                target_stype,
                local_target_counts,
                writer,
                processor,
            )?;
        }

        // Clear aliases for next timestamp
        aliases.iter_mut().for_each(|a| *a = None);
    }

    Ok(())
}

/// Write the target value(s) for one (entity, timestamp) pair.
#[allow(clippy::too_many_arguments)]
fn write_target<U: UnitSystem, D: SplitDecider<U>>(
    plan: &ResolvedPlan<U>,
    entity_pk: &Value,
    ts: Abs<U>,
    aliases: &[Option<Value>],
    link_frequencies: &[i64],
    link_split_values: &[Value],
    split_value: Option<&Value>,
    entity_include_bit: bool,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
    writer: &mut ChannelBatchWriter<U>,
    processor: &mut RowProcessor<U, D>,
) -> TTGenResult<()> {
    let extra_value = if matches!(
        plan.split_projection.as_ref().map(|p| &p.dtype),
        Some(DataType::List(_))
    ) {
        ExtraValueRef::List(link_split_values)
    } else {
        ExtraValueRef::Scalar(split_value)
    };

    match &plan.prediction_type {
        PredictionType::PassThrough => {
            writer.write_row_pass_through(
                entity_pk.clone(),
                ts,
                extra_value,
                entity_include_bit,
            )?;
        }
        PredictionType::Forecasting => {
            let mut autoregression = Vec::new();
            if let Some(idx) = plan.autoregression_alias_idx {
                if let Some(Value::List(l)) = &aliases[idx] {
                    autoregression = l.clone();
                }
            }
            let mut future_labels = None;
            if let Some(idx) = plan.future_labels_alias_idx {
                if let Some(Value::List(l)) = &aliases[idx] {
                    future_labels = Some(l.clone());
                }
            }
            let mut yoy = None;
            if let Some(idx) = plan.yoy_alias_idx {
                if let Some(Value::List(l)) = &aliases[idx] {
                    yoy = Some(l.clone());
                }
            }
            let mut target = None;
            if !plan.is_prediction {
                if let Some(val) = &aliases[plan.primary_target_alias_idx()] {
                    target = Some(val.clone());
                } else {
                    return Ok(()); // Drop row if ground-truth target is missing
                }
            }
            writer.write_row_forecasting(
                entity_pk.clone(),
                ts,
                target.as_ref(),
                autoregression,
                future_labels,
                yoy,
                extra_value,
                entity_include_bit,
            )?;

            if !plan.is_prediction {
                if let Some(t) = target {
                    let row = TTRowRef {
                        entity_pk,
                        ts,
                        target: &t,
                        frequency: None,
                        target_is_numerical,
                        target_is_categorical,
                        target_stype,
                        split_column_value: split_value,
                    };
                    processor.observe(&row, local_target_counts);
                }
            }

            if entity_include_bit {
                processor.stats.include_true_count += 1;
                processor.stats.num_train_rows += 1;
            } else {
                processor.stats.include_false_count += 1;
            }
        }
        PredictionType::Node | PredictionType::Link { .. } => {
            // PQ path: single-target; read alias slot 0.
            let target = match &aliases[plan.primary_target_alias_idx()] {
                Some(v) => v,
                None => return Ok(()),
            };

            if matches!(plan.prediction_type, PredictionType::Node)
                && *plan.primary_target_dtype() == DataType::Utf8
            {
                if let Value::Str(s) = target {
                    if s.is_empty() {
                        return Ok(());
                    }
                }
            }

            match &plan.prediction_type {
                PredictionType::Node => {
                    writer.write_row(
                        entity_pk.clone(),
                        ts,
                        target,
                        extra_value,
                        entity_include_bit,
                    )?;
                    if !plan.is_prediction {
                        let row = TTRowRef {
                            entity_pk,
                            ts,
                            target,
                            frequency: None,
                            target_is_numerical,
                            target_is_categorical,
                            target_stype,
                            // ColumnDecider observes the split-column value
                            // here. `split_value` is already projected per
                            // entity via `plan.split_projection.entity_col_idx`.
                            // For non-ColumnSplit families this stays `None`.
                            split_column_value: split_value,
                        };
                        processor.observe(&row, local_target_counts);
                    }
                    if entity_include_bit {
                        processor.stats.include_true_count += 1;
                        processor.stats.num_train_rows += 1;
                    } else {
                        processor.stats.include_false_count += 1;
                    }
                }
                PredictionType::Link {
                    include_frequency, ..
                } => {
                    if let Value::List(targets) = target {
                        let freqs: Vec<i64> = if *include_frequency {
                            link_frequencies.to_vec()
                        } else {
                            Vec::new()
                        };
                        writer.write_row_array(
                            entity_pk.clone(),
                            ts,
                            targets.clone(),
                            &[],
                            &freqs,
                            extra_value,
                            entity_include_bit,
                        )?;

                        if let Some(first_target) = targets.first() {
                            let first_freq = if *include_frequency {
                                freqs.first().copied()
                            } else {
                                None
                            };
                            if !plan.is_prediction {
                                let row = TTRowRef {
                                    entity_pk,
                                    ts,
                                    target: first_target,
                                    frequency: first_freq,
                                    target_is_numerical,
                                    target_is_categorical,
                                    target_stype,
                                    // Array-LP uses the entity-level split value
                                    // for the representative row; the regroup
                                    // path doesn't go through here.
                                    split_column_value: split_value,
                                };
                                processor.observe(&row, local_target_counts);
                                for (idx, tgt) in targets.iter().enumerate().skip(1) {
                                    let freq = if *include_frequency {
                                        freqs.get(idx).copied()
                                    } else {
                                        None
                                    };
                                    // Observe additional targets with the SAME
                                    // stype-aware basis as the first target (via
                                    // `update_exact_stats` → `observe_target_hll`).
                                    // Using an unconditional `hash_value` basis here
                                    // double-registered multicat LP targets whose
                                    // first-target observation used `hash_bytes`.
                                    if let Some(ref mut h) = processor.stats.target_hll {
                                        observe_target_hll(
                                            h,
                                            tgt,
                                            target_is_numerical,
                                            target_is_categorical,
                                            target_stype,
                                        );
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
                            // Observe the entity via the SAME canonical
                            // type-aware RAW basis used by `update_exact_stats`
                            // (`observe_entity_hll`). Using `hash_value` here
                            // double-registered ids also seen on the main path.
                            if let Some(ref mut h) = processor.stats.entity_hll {
                                observe_entity_hll(h, entity_pk);
                            }
                            if let Some(ref mut h) = processor.stats.timestamp_hll {
                                h.observe_i64(ts.raw());
                            }
                            processor.stats.timestamp_min =
                                Some(match processor.stats.timestamp_min {
                                    Some(existing) => existing.min(ts),
                                    None => ts,
                                });
                            processor.stats.timestamp_max =
                                Some(match processor.stats.timestamp_max {
                                    Some(existing) => existing.max(ts),
                                    None => ts,
                                });
                            if let Some(v) = value_to_i64_opt(entity_pk) {
                                processor.stats.entity_min =
                                    min_opt_i64(processor.stats.entity_min, v);
                                processor.stats.entity_max =
                                    max_opt_i64(processor.stats.entity_max, v);
                            }
                        }

                        if entity_include_bit {
                            processor.stats.include_true_count += 1;
                            processor.stats.num_links += targets.len() as u64;
                            processor.stats.num_train_rows += 1;
                        } else {
                            processor.stats.include_false_count += 1;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

/// Observe an entity primary key into the entity HLL using the canonical
/// type-aware RAW basis (Int→observe_i64, Str→observe_bytes, Float→observe_f64).
///
/// Both the main `update_exact_stats` path and the empty-target-row branch must
/// call this so the same logical entity id lands on the SAME HLL hash basis.
/// Observing an id via a raw value in one branch and via `hash_value(..)` in
/// the other made it register twice, over-counting `num_entities`.
#[inline]
pub(crate) fn observe_entity_hll(hll: &mut HyperLogLog, entity_pk: &Value) {
    match entity_pk {
        Value::Int(i) => hll.observe_i64(*i),
        Value::Str(s) => hll.observe_bytes(s.as_bytes()),
        Value::Float(f) => hll.observe_f64(*f),
        _ => {}
    }
}

/// Observe a target value into the target HLL replicating `update_exact_stats`'
/// per-stype dispatch: numerical→raw (`observe_i64`/`observe_f64`),
/// multicategorical→per-token `hash_bytes(token_string)`, other
/// categorical→`observe_i64(hash_value(target))`.
///
/// The first target (in `update_exact_stats`) and every additional target (the
/// `skip(1)` loop) must observe through this SAME stype-aware logic. Previously
/// the additional-targets loop unconditionally used `hash_value`, so for
/// multicategorical LP the first target (hashed via `hash_bytes(token)`) and the
/// additional targets (hashed via `hash_value`) landed on different bases and
/// the same target registered twice, over-counting `num_targets`.
#[inline]
pub(crate) fn observe_target_hll(
    hll: &mut HyperLogLog,
    target: &Value,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
) {
    if target_is_numerical {
        match target {
            Value::Int(i) => hll.observe_i64(*i),
            Value::Float(f) => hll.observe_f64(*f),
            _ => {}
        }
    } else if target_is_categorical {
        if target_stype == "multicategorical" {
            match target {
                Value::List(items) => {
                    for item in items {
                        let tok_str = value_to_string(item);
                        let tok = tok_str.trim();
                        hll.observe_hash(HyperLogLog::hash_bytes(tok.as_bytes()));
                    }
                }
                _ => {
                    let target_str = value_to_string(target);
                    for_each_split_ascii_byte(&target_str, b'|', |tok| {
                        let tok = tok.trim();
                        hll.observe_hash(HyperLogLog::hash_bytes(tok.as_bytes()));
                    });
                }
            }
        } else {
            hll.observe_i64(hash_value(target) as i64);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_exact_stats<U: UnitSystem>(
    stats: &mut ChunkStats<U>,
    entity_pk: &Value,
    ts: Abs<U>,
    target: &Value,
    frequency: Option<i64>,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
) {
    // Cardinality tracking is HLL-only. Target HLL is observed
    // in the per-stype branches below (numerical: raw value; categorical:
    // hashed identity; multicategorical: per-token). Entity / timestamp
    // HLL observations are unconditional and live below alongside their
    // Welford / TDigest siblings.

    stats.timestamp_min = Some(match stats.timestamp_min {
        Some(existing) => existing.min(ts),
        None => ts,
    });
    stats.timestamp_max = Some(match stats.timestamp_max {
        Some(existing) => existing.max(ts),
        None => ts,
    });

    if let Some(v) = value_to_i64_opt(entity_pk) {
        stats.entity_min = min_opt_i64(stats.entity_min, v);
        stats.entity_max = max_opt_i64(stats.entity_max, v);
    }

    // Observe entity value into HLL for cardinality (canonical type-aware RAW
    // basis; shared with the empty-target-row branch via `observe_entity_hll`).
    // Entity CATEGORY_COUNTS is intentionally not collected: the entity column
    // is a graph FK/pkey resolved to node ids, never encoded, so it has no
    // category-count consumer. Cardinality is tracked by the HLL alone.
    if let Some(ref mut h) = stats.entity_hll {
        observe_entity_hll(h, entity_pk);
    }

    // Observe entity value into Welford + TDigest (numeric entities only — Int values)
    if let Value::Int(i) = entity_pk {
        if let Some(ref mut w) = stats.entity_welford {
            w.observe(*i as f64);
        }
        if let Some(ref mut t) = stats.entity_tdigest {
            t.observe(*i as f64);
        }
    }

    // Observe timestamp value into Welford + TDigest + HLL
    {
        let ts_raw = ts.raw();
        if let Some(ref mut w) = stats.timestamp_welford {
            w.observe(ts_raw as f64);
        }
        if let Some(ref mut t) = stats.timestamp_tdigest {
            t.observe(ts_raw as f64);
        }
        if let Some(ref mut h) = stats.timestamp_hll {
            h.observe_i64(ts_raw);
        }
    }

    if let Some(v) = value_to_i64_opt(target) {
        stats.target_min = min_opt_i64(stats.target_min, v);
        stats.target_max = max_opt_i64(stats.target_max, v);
    }

    if let Some(freq) = frequency {
        stats.frequency_min = min_opt_i64(stats.frequency_min, freq);
        stats.frequency_max = max_opt_i64(stats.frequency_max, freq);
    }

    // Observe target value for distribution stats (numerical targets only)
    if target_is_numerical {
        if let Some(f) = match target {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        } {
            if f.is_finite() {
                if let Some(ref mut w) = stats.target_welford {
                    w.observe(f);
                }
                if let Some(ref mut t) = stats.target_tdigest {
                    t.observe(f);
                }
            } else {
                stats.target_invalid_count += 1;
            }
        }
    }

    // Observe target value into the target HLL. Shared with the
    // additional-targets `skip(1)` loop via `observe_target_hll` so the same
    // logical target lands on the SAME hash basis regardless of which path
    // observes it.
    if let Some(ref mut h) = stats.target_hll {
        observe_target_hll(
            h,
            target,
            target_is_numerical,
            target_is_categorical,
            target_stype,
        );
    }

    // Observe target value for categorical CATEGORY_COUNTS stats. HLL is handled
    // above via `observe_target_hll`; here we only accumulate local counts.
    if target_is_categorical {
        if target_stype == "multicategorical" {
            // Multicat targets arrive as `Value::List(tokens)` from per-token
            // dispatch (FIRST/LAST/Route A/Node-pred LIST_DISTINCT) or as
            // `Value::Str(pipe-joined)` from the legacy whole-cell path.
            // Dispatch on shape so the list path doesn't collapse through
            // `value_to_string` into the literal "<list>" sentinel.
            match target {
                Value::List(items) => {
                    for item in items {
                        let tok_str = value_to_string(item);
                        let tok = tok_str.trim();
                        let hash = HyperLogLog::hash_bytes(tok.as_bytes());
                        if let Some(ref mut local) = local_target_counts {
                            local.observe(hash, tok);
                        }
                    }
                }
                _ => {
                    let target_str = value_to_string(target);
                    for_each_split_ascii_byte(&target_str, b'|', |tok| {
                        let tok = tok.trim();
                        let hash = HyperLogLog::hash_bytes(tok.as_bytes());
                        if let Some(ref mut local) = local_target_counts {
                            local.observe(hash, tok);
                        }
                    });
                }
            }
        } else {
            let target_str = value_to_string(target);
            let target_hash = hash_value(target);
            if let Some(ref mut local) = local_target_counts {
                local.observe(target_hash, &target_str);
            }
        }
    }

    // Frequency distribution (if present)
    if let Some(freq) = frequency {
        if let Some(ref mut w) = stats.frequency_welford {
            w.observe(freq as f64);
        }
        if let Some(ref mut t) = stats.frequency_tdigest {
            t.observe(freq as f64);
        }
        if let Some(ref mut h) = stats.frequency_hll {
            h.observe_i64(freq);
        }
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::List(_) => "<list>".to_string(),
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

// ─── Staged execution ────────────────────────────────────────────────────────

/// Keep missing values distinguishable until the zero-offset training target
/// is extracted. Sequence outputs convert their remaining gaps to NaN later.
fn execute_target_sequence<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    entity_node: u64,
    entity_cols: &[Option<Value>],
    base_ts: Abs<U>,
    agg: &ResolvedAggregate<U>,
    offsets: &[Dur<U>],
) -> Vec<Option<Value>> {
    let mut results = Vec::with_capacity(offsets.len());
    let can_optimize =
        agg.path.len() == 1 && matches!(agg.window, Some(ResolvedWindow::Fixed { .. }));

    if !can_optimize {
        for &offset in offsets {
            let shifted_ts = base_ts + offset;
            let val_opt = execute_aggregate(
                plan,
                entity_node,
                agg,
                entity_cols,
                Some(shifted_ts),
                &mut Vec::new(),
                &mut Vec::new(),
            );
            results.push(val_opt);
        }
        return results;
    }

    let ResolvedWindow::Fixed {
        offset_lo,
        offset_hi,
        encoder,
    } = agg.window.as_ref().unwrap()
    else {
        unreachable!()
    };
    let windows: Vec<SortKeyWindow> = offsets
        .iter()
        .map(|&offset| {
            let win_ts = base_ts + offset;
            let abs_lo = offset_lo
                .map(|offset_lo| encoder.encode_upper((win_ts + offset_lo).raw()));
            let abs_hi = encoder.encode_upper((win_ts + *offset_hi).raw());
            (abs_lo, abs_hi)
        })
        .collect();
    let min_lo = windows.iter().map(|w| w.0).min().unwrap();
    let max_hi = windows.iter().map(|w| w.1).max().unwrap();
    let edge_ref = &agg.path[0];
    let edges = iter_edge(
        &edge_ref.src_table,
        edge_ref.relation_idx,
        &edge_ref.layout,
        entity_node,
        Some((min_lo, max_hi)),
    );
    let dst_table = &edge_ref.dst_table;
    let col_idx = agg.target_col_idx.unwrap_or(0);

    for (abs_lo, abs_hi) in windows {
        let start = edges.partition_point(|e| e.sort_key > abs_hi);
        let end = abs_lo
            .map(|abs_lo| {
                start + edges[start..].partition_point(|e| e.sort_key > abs_lo)
            })
            .unwrap_or(edges.len());
        let slice = &edges[start..end];

        if slice.is_empty() {
            results.push(match agg.func {
                AggFunc::Count | AggFunc::CountDistinct => Some(Value::Int(0)),
                AggFunc::Sum => Some(Value::Float(0.0)),
                _ => None,
            });
            continue;
        }

        match agg.func {
            AggFunc::First => {
                let mut oldest = None;
                for e in slice.iter().rev() {
                    if let Some(ref f) = agg.row_filter {
                        if !eval_row_filter(f, e.dst_node, dst_table, entity_cols) {
                            continue;
                        }
                    }
                    if let Some(v) = read_column_value(dst_table, e.dst_node, col_idx) {
                        oldest = Some(v);
                        break;
                    }
                }
                results.push(oldest);
            }
            AggFunc::Last => {
                let mut newest = None;
                for e in slice {
                    if let Some(ref f) = agg.row_filter {
                        if !eval_row_filter(f, e.dst_node, dst_table, entity_cols) {
                            continue;
                        }
                    }
                    if let Some(v) = read_column_value(dst_table, e.dst_node, col_idx) {
                        newest = Some(v);
                        break;
                    }
                }
                results.push(newest);
            }
            _ => {
                let mut acc = Accumulator::new(agg.func, agg.limit);
                for e in slice {
                    if let Some(ref f) = agg.row_filter {
                        if !eval_row_filter(f, e.dst_node, dst_table, entity_cols) {
                            continue;
                        }
                    }
                    match &mut acc {
                        Accumulator::Count(n) => *n += 1,
                        Accumulator::Sum(x) => {
                            if let Some(v) = read_column_f64(dst_table, e.dst_node, col_idx) {
                                *x += v;
                            }
                        }
                        Accumulator::Avg(w) => {
                            if let Some(v) = read_column_f64(dst_table, e.dst_node, col_idx) {
                                w.observe(v);
                            }
                        }
                        Accumulator::Min(x) => {
                            if let Some(v) = read_column_f64(dst_table, e.dst_node, col_idx) {
                                *x = x.min(v);
                            }
                        }
                        Accumulator::Max(x) => {
                            if let Some(v) = read_column_f64(dst_table, e.dst_node, col_idx) {
                                *x = x.max(v);
                            }
                        }
                        Accumulator::CountDistinct(set) => {
                            if let Some(v) = read_column_value(dst_table, e.dst_node, col_idx) {
                                set.insert(crate::ttgen_accumulator::hash_value(&v));
                            }
                        }
                        Accumulator::ListDistinct { set, values, limit } => {
                            if let Some(v) = read_column_value(dst_table, e.dst_node, col_idx) {
                                let h = crate::ttgen_accumulator::hash_value(&v);
                                if set.insert(h) {
                                    if let Some(l) = limit {
                                        if values.len() >= *l as usize {
                                            continue;
                                        }
                                    }
                                    values.push(v);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                results.push(acc.finalize());
            }
        }
    }
    results
}

/// Execute all stages for an (entity, timestamp) pair.
/// Returns true if all stages passed (output row should be emitted).
///
/// `observation_ts` is the current observation timestamp (epoch seconds) for
/// temporal plans. Used to compute absolute window bounds from relative offsets.
/// `None` for non-temporal plans.
///
/// `link_frequencies` is populated with occurrence counts when a `ListDistinct`
/// aggregation runs. Only the last ListDistinct's counts are kept (the target).
fn execute_stages<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    entity_node: u64,
    entity_cols: &[Option<Value>],
    observation_ts: Option<Abs<U>>,
    aliases: &mut [Option<Value>],
    link_frequencies: &mut Vec<i64>,
    link_split_values: &mut Vec<Value>,
) -> bool {
    for stage in &plan.stages {
        for op in &stage.ops {
            let result = match op {
                ResolvedOp::Aggregate(agg) => execute_aggregate(
                    plan,
                    entity_node,
                    agg,
                    entity_cols,
                    observation_ts,
                    link_frequencies,
                    link_split_values,
                ),
                ResolvedOp::HasEdges(has_edges) => {
                    execute_has_edges(plan, entity_node, has_edges, entity_cols, observation_ts)
                }
                ResolvedOp::Column(col) => {
                    match resolve_remote_node(&col.path, entity_node) {
                        Some(column_node) => {
                            // Route A direct-target read site (kumo#28400). For
                            // multicat sources, emit `Value::List(tokens)` so the
                            // downstream alias consumer (the writer) sees the row's
                            // flat token set. Non-multicat path is unchanged.
                            let is_multicat =
                                col.target_semantic_type.as_deref() == Some("multicategorical");
                            if is_multicat {
                                let mut tokens: Vec<Value> = Vec::new();
                                let outcome = dispatch_multicat_source(
                                    &col.table,
                                    column_node,
                                    col.col_idx,
                                    true,
                                    |tok| {
                                        tokens.push(match tok {
                                            MulticatToken::Str(s) => Value::Str(s.to_string()),
                                            MulticatToken::Int(n) => Value::Int(n),
                                            MulticatToken::Legacy(v) => v,
                                        });
                                        true
                                    },
                                );
                                if outcome.observed() {
                                    Some(Value::List(tokens))
                                } else {
                                    None
                                }
                            } else {
                                match read_column_value_nullable(
                                    &col.table,
                                    column_node,
                                    col.col_idx,
                                ) {
                                    Some(Some(v)) => Some(v),
                                    Some(None) => None,
                                    None => None,
                                }
                            }
                        }
                        None => None,
                    }
                }
                ResolvedOp::Compare(cmp) => {
                    let l_opt = &aliases[cmp.lhs_alias_idx];
                    match cmp.op {
                        CompareOp::IsNull => Some(Value::Bool(is_nullish_filter_value(l_opt))),
                        CompareOp::IsNotNull => Some(Value::Bool(!is_nullish_filter_value(l_opt))),
                        _ => l_opt
                            .as_ref()
                            .map(|lhs| Value::Bool(compare_values(lhs, &cmp.op, &cmp.rhs))),
                    }
                }
                ResolvedOp::LogicalOp(lop) => {
                    use diskgraph_core::tt_builder::LogicalOpKind;
                    match lop.op {
                        LogicalOpKind::Not => match &aliases[lop.operand_indices[0]] {
                            Some(Value::Bool(b)) => Some(Value::Bool(!b)),
                            None => None,
                            _ => None,
                        },
                        LogicalOpKind::And => {
                            let a = &aliases[lop.operand_indices[0]];
                            let b = &aliases[lop.operand_indices[1]];
                            match (a, b) {
                                (Some(Value::Bool(false)), _) | (_, Some(Value::Bool(false))) => {
                                    Some(Value::Bool(false))
                                }
                                (Some(Value::Bool(true)), Some(Value::Bool(true))) => {
                                    Some(Value::Bool(true))
                                }
                                _ => None,
                            }
                        }
                        LogicalOpKind::Or => {
                            let a = &aliases[lop.operand_indices[0]];
                            let b = &aliases[lop.operand_indices[1]];
                            match (a, b) {
                                (Some(Value::Bool(true)), _) | (_, Some(Value::Bool(true))) => {
                                    Some(Value::Bool(true))
                                }
                                (Some(Value::Bool(false)), Some(Value::Bool(false))) => {
                                    Some(Value::Bool(false))
                                }
                                _ => None,
                            }
                        }
                    }
                }
                ResolvedOp::Autoregression(auto) => {
                    let base_ts = observation_ts.unwrap_or(Abs::<U>::from_raw(0));
                    if let ResolvedOp::Aggregate(ref agg) = *auto.target_op {
                        let is_primary_ar = Some(auto.alias_idx) == plan.autoregression_alias_idx;
                        let pad_target = is_primary_ar && !plan.is_prediction;
                        let active_offsets = if pad_target {
                            let mut offsets = vec![Dur::<U>::from_raw(0)];
                            offsets.extend_from_slice(&auto.lag_offsets);
                            offsets
                        } else {
                            auto.lag_offsets.clone()
                        };
                        let mut results = execute_target_sequence(
                            plan,
                            entity_node,
                            entity_cols,
                            base_ts,
                            agg,
                            &active_offsets,
                        );
                        if pad_target && !results.is_empty() {
                            aliases[plan.primary_target_alias_idx()] = results.remove(0);
                        }
                        Some(Value::List(
                            results
                                .into_iter()
                                .map(|value| value.unwrap_or(Value::Float(f64::NAN)))
                                .collect(),
                        ))
                    } else {
                        None
                    }
                }
                ResolvedOp::FutureLabels(seq) => {
                    let base_ts = observation_ts.unwrap_or(Abs::<U>::from_raw(0));
                    if let ResolvedOp::Aggregate(ref agg) = *seq.target_op {
                        Some(Value::List(
                            execute_target_sequence(
                                plan,
                                entity_node,
                                entity_cols,
                                base_ts,
                                agg,
                                &seq.future_offsets,
                            )
                            .into_iter()
                            .map(|value| value.unwrap_or(Value::Float(f64::NAN)))
                            .collect(),
                        ))
                    } else {
                        None
                    }
                }
            };
            aliases[op.alias_idx()] = result;
        }

        if let Some(ref filter) = stage.filter {
            if !eval_stage_filter(
                filter,
                aliases,
                entity_cols,
                &plan.entity_table,
                entity_node,
            ) {
                return false;
            }
        }
    }
    true
}

// ─── Fast Existence Check (Null Emitting Target Logic) ───────────────────────

fn execute_has_edges<U: UnitSystem>(
    _plan: &ResolvedPlan<U>,
    entity_node: u64,
    has_edges: &ResolvedHasEdges<U>,
    entity_cols: &[Option<Value>],
    observation_ts: Option<Abs<U>>,
) -> Option<Value> {
    // SAFETY: encode_upper expects epoch-seconds (the index's internal unit).
    // This is correct when U=Seconds. If the pipeline is ever monomorphized
    // over a different unit (e.g. Nanoseconds), the encoder must be made
    // unit-aware as part of the ingest/sampling unit awareness work.
    let window = has_edges.window.as_ref().map(|w| {
        let obs = observation_ts.unwrap_or(Abs::<U>::from_raw(0));
        match w {
            ResolvedWindow::Fixed {
                offset_lo,
                offset_hi,
                encoder,
            } => {
                let lo = offset_lo
                    .map(|offset_lo| encoder.encode_upper((obs + offset_lo).raw()));
                let hi = encoder.encode_upper((obs + *offset_hi).raw());
                (lo, hi)
            }
            ResolvedWindow::Calendar {
                lo_months,
                hi_months,
                encoder,
            } => {
                let hi_abs = add_calendar_months(obs, *hi_months);
                let lo = lo_months.map(|lo_months| {
                    encoder.encode_upper(add_calendar_months(obs, lo_months).raw())
                });
                let hi = encoder.encode_upper(hi_abs.raw());
                (lo, hi)
            }
        }
    });

    let dst_table = &has_edges.path.last().unwrap().dst_table;
    let mut traversed = false;

    traverse_and_accumulate(
        entity_node,
        &has_edges.path,
        window,
        true,
        &mut |dst, _sk| {
            if let Some(ref f) = has_edges.row_filter {
                if !eval_row_filter(f, dst, dst_table, entity_cols) {
                    return true;
                }
            }
            traversed = true;
            false
        },
    );

    Some(Value::Bool(traversed))
}

// ─── Aggregate execution (match-once monomorphization) ───────────────────────

fn execute_aggregate<U: UnitSystem>(
    _plan: &ResolvedPlan<U>,
    entity_node: u64,
    agg: &ResolvedAggregate<U>,
    entity_cols: &[Option<Value>],
    observation_ts: Option<Abs<U>>,
    link_frequencies: &mut Vec<i64>,
    link_split_values: &mut Vec<Value>,
) -> Option<Value> {
    // Compute window bounds (sort key space).
    // Kumo uses half-open interval (start, end]: exclusive lower, inclusive upper.
    // encode_upper sets all weight bits to 1, giving the max sort key for that
    // timestamp bucket. Combined with the (lo, hi] search in iter_edge, this
    // ensures edges exactly at the lower boundary are excluded and edges exactly
    // at the upper boundary are included.
    let window = agg.window.as_ref().map(|w| {
        let obs = observation_ts.unwrap_or(Abs::<U>::from_raw(0));
        match w {
            ResolvedWindow::Fixed {
                offset_lo,
                offset_hi,
                encoder,
            } => {
                let lo = offset_lo
                    .map(|offset_lo| encoder.encode_upper((obs + offset_lo).raw()));
                let hi = encoder.encode_upper((obs + *offset_hi).raw());
                (lo, hi)
            }
            ResolvedWindow::Calendar {
                lo_months,
                hi_months,
                encoder,
            } => {
                let hi_abs = add_calendar_months(obs, *hi_months);
                let lo = lo_months.map(|lo_months| {
                    encoder.encode_upper(add_calendar_months(obs, lo_months).raw())
                });
                let hi = encoder.encode_upper(hi_abs.raw());
                (lo, hi)
            }
        }
    });

    // Get the destination table for column reads
    let dst_table = &agg.path.last().unwrap().dst_table;

    match agg.func {
        AggFunc::Count => {
            let mut acc = 0u64;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true; // skip but continue
                    }
                }
                acc += 1;
                true
            });
            Some(Value::Int(acc as i64))
        }
        AggFunc::Sum => {
            let mut acc = 0.0f64;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(v) = read_column_f64(dst_table, dst, col_idx) {
                        acc += v;
                    }
                }
                true
            });
            Some(Value::Float(acc))
        }
        AggFunc::Avg => {
            let mut w = WelfordAccumulator::new();
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(v) = read_column_f64(dst_table, dst, col_idx) {
                        w.observe(v);
                    }
                }
                true
            });
            Accumulator::Avg(w).finalize()
        }
        AggFunc::Min => {
            let mut acc = f64::INFINITY;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(v) = read_column_f64(dst_table, dst, col_idx) {
                        acc = acc.min(v);
                    }
                }
                true
            });
            Accumulator::Min(acc).finalize()
        }
        AggFunc::Max => {
            let mut acc = f64::NEG_INFINITY;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(v) = read_column_f64(dst_table, dst, col_idx) {
                        acc = acc.max(v);
                    }
                }
                true
            });
            Accumulator::Max(acc).finalize()
        }
        AggFunc::CountDistinct => {
            // Multicat: hash per token (not per cell). Output is the count of
            // distinct tokens across the window.
            let is_multicat = agg.target_semantic_type.as_deref() == Some("multicategorical");
            let mut set = HashSet::new();
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                let Some(col_idx) = agg.target_col_idx else {
                    return true;
                };
                if is_multicat {
                    // COUNT_DISTINCT only needs the hash; never materialize
                    // a Value (zero allocations on every observation).
                    let _ = dispatch_multicat_source(dst_table, dst, col_idx, true, |tok| {
                        let h = match tok {
                            MulticatToken::Str(s) => hash_str(s),
                            MulticatToken::Int(n) => hash_i64(n),
                            MulticatToken::Legacy(ref v) => hash_value(v),
                        };
                        set.insert(h);
                        true
                    });
                } else if let Some(val) = read_column_value(dst_table, dst, col_idx) {
                    set.insert(hash_value(&val));
                }
                true
            });
            Some(Value::Int(set.len() as i64))
        }
        AggFunc::ListDistinct => {
            let mut seen: HashMap<u64, usize> = HashMap::new();
            let mut values = Vec::new();
            let mut counts = Vec::new();
            let mut split_values = Vec::new();
            let limit = agg.limit.unwrap_or(u32::MAX) as usize;
            let pk_filter = agg.target_pk_table.as_ref();
            // Temporal validity filtering requires both an observation timestamp
            // and pre-resolved temporal column indices on the target entity table.
            // Without observation_ts (non-temporal plans), fall back to PK-only check.
            // Created by the window end, as for ranked candidates (`RHSTime`).
            let temporal_ref_ts = match observation_ts {
                Some(ts)
                    if agg.target_temporal_create_col_idx.is_some()
                        || agg.target_temporal_end_col_idx.is_some() =>
                {
                    let create_ref = match agg.window.as_ref() {
                        Some(ResolvedWindow::Fixed { offset_hi, .. }) => ts + *offset_hi,
                        Some(ResolvedWindow::Calendar { hi_months, .. }) => {
                            add_calendar_months(ts, *hi_months)
                        }
                        None => ts,
                    };
                    Some((create_ref.raw().max(ts.raw()), ts.raw()))
                }
                _ => None,
            };
            let mut rowpack_scratch = diskgraph_rowpack::RowPackDecodeScratch::default();

            let is_multicat = agg.target_semantic_type.as_deref() == Some("multicategorical");

            // PK / temporal-validity filter: only meaningful for FK-shaped LP
            // ListDistinct (`target_pk_table` is set). Multicat sources are
            // never FK columns, so the filter is unreachable on the multicat
            // branch.
            //
            // Hot-path discipline: hash the borrowed token first; only
            // materialize `Value::Str(String)` on the Vacant branch. Repeated
            // tokens hit Occupied and never allocate.

            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                let Some(col_idx) = agg.target_col_idx else {
                    return values.len() < limit;
                };

                if is_multicat {
                    let _ = dispatch_multicat_source(dst_table, dst, col_idx, true, |tok| {
                        let h = match &tok {
                            MulticatToken::Str(s) => hash_str(s),
                            MulticatToken::Int(n) => hash_i64(*n),
                            MulticatToken::Legacy(v) => hash_value(v),
                        };
                        match seen.entry(h) {
                            std::collections::hash_map::Entry::Occupied(e) => {
                                counts[*e.get()] += 1;
                            }
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(values.len());
                                values.push(match tok {
                                    MulticatToken::Str(s) => Value::Str(s.to_string()),
                                    MulticatToken::Int(n) => Value::Int(n),
                                    MulticatToken::Legacy(v) => v,
                                });
                                counts.push(1i64);
                            }
                        }
                        values.len() < limit
                    });
                    return values.len() < limit;
                }

                // Legacy whole-cell path (FK ListDistinct, scalar ListDistinct).
                if let Some(val) = read_column_value(dst_table, dst, col_idx) {
                    // Skip orphan FKs and targets invalid for the window.
                    if let Some(pk_table) = pk_filter {
                        if let Some((create_ref_ts, end_ref_ts)) = temporal_ref_ts {
                            if !pk_table.contains_pk_value_valid_between(
                                &val,
                                create_ref_ts,
                                end_ref_ts,
                                agg.target_temporal_create_col_idx,
                                agg.target_temporal_end_col_idx,
                                &mut rowpack_scratch,
                            ) {
                                return true;
                            }
                        } else if !pk_table.contains_pk_value(&val) {
                            return true;
                        }
                    }
                    let h = hash_value(&val);
                    match seen.entry(h) {
                        std::collections::hash_map::Entry::Occupied(e) => {
                            counts[*e.get()] += 1;
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(values.len());
                            if let Some(split_col_idx) = agg.link_split_col_idx {
                                split_values.push(
                                    read_column_value_nullable(dst_table, dst, split_col_idx)
                                        .flatten()
                                        .unwrap_or_else(|| Value::Str("None".to_string())),
                                );
                            }
                            values.push(val);
                            counts.push(1i64);
                        }
                    }
                }
                values.len() < limit
            });
            // Store frequency counts for link prediction output
            link_frequencies.clear();
            link_frequencies.extend_from_slice(&counts);
            link_split_values.clear();
            link_split_values.extend(split_values);
            Some(Value::List(values))
        }
        AggFunc::First => {
            // Edges iterate DESCENDING by sort key (newest first),
            // so "first" (earliest) = keep overwriting, last encountered wins.
            //
            // Multicat semantic shift: the slot stores `Value::List(tokens)`
            // (the row's full token set), not the cell value.
            let is_multicat = agg.target_semantic_type.as_deref() == Some("multicategorical");
            let mut val: Option<Value> = None;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if is_multicat {
                        let mut tokens: Vec<Value> = Vec::new();
                        let _ = dispatch_multicat_source(dst_table, dst, col_idx, true, |tok| {
                            tokens.push(match tok {
                                MulticatToken::Str(s) => Value::Str(s.to_string()),
                                MulticatToken::Int(n) => Value::Int(n),
                                MulticatToken::Legacy(v) => v,
                            });
                            true
                        });
                        val = Some(Value::List(tokens));
                    } else {
                        val = read_column_value(dst_table, dst, col_idx);
                    }
                } else {
                    val = Some(Value::Int(dst as i64));
                }
                true // keep going — earliest is last in descending order
            });
            val
        }
        AggFunc::Last => {
            // Edges iterate DESCENDING by sort key (newest first),
            // so "last" (latest) = first encountered, stop early.
            //
            // Multicat semantic shift: same as `First` — slot stores
            // `Value::List(tokens)` of the chosen row's full token set.
            let is_multicat = agg.target_semantic_type.as_deref() == Some("multicategorical");
            let mut val: Option<Value> = None;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if is_multicat {
                        let mut tokens: Vec<Value> = Vec::new();
                        let _ = dispatch_multicat_source(dst_table, dst, col_idx, true, |tok| {
                            tokens.push(match tok {
                                MulticatToken::Str(s) => Value::Str(s.to_string()),
                                MulticatToken::Int(n) => Value::Int(n),
                                MulticatToken::Legacy(v) => v,
                            });
                            true
                        });
                        val = Some(Value::List(tokens));
                    } else {
                        val = read_column_value(dst_table, dst, col_idx);
                    }
                } else {
                    val = Some(Value::Int(dst as i64));
                }
                false // stop after first match — latest in descending order
            });
            val
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_a_writer_rotates_into_ordered_parts() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{Field, Schema};

        let dir = tempfile::tempdir().unwrap();
        let dataset = dir.path().join(TT_DATASET_DIR);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ENTITY1",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 6]))],
        )
        .unwrap();
        let (sender, receiver) = bounded(1);
        sender.send(batch).unwrap();
        drop(sender);

        assert_eq!(write_tt_dataset(receiver, &dataset, 3).unwrap(), 7);
        let parts = ordered_tt_dataset_parts(&dataset).unwrap();
        assert_eq!(
            parts
                .iter()
                .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            vec![
                "part-00000.parquet",
                "part-00001.parquet",
                "part-00002.parquet"
            ]
        );
        let rows = parts
            .into_iter()
            .map(|path| {
                let builder =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        File::open(path).unwrap(),
                    )
                    .unwrap();
                builder.metadata().file_metadata().num_rows()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![3, 3, 1]);
    }
    use diskgraph_core::time::Seconds;

    #[test]
    fn stage_manifest_preserves_exact_nineteen_part_lists() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{Field, Schema};

        let output_dir = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1]))])
        .unwrap();
        let parts = std::array::from_fn(|stage| {
            let table = STAGE_TABLE_NAMES[stage];
            let table_dir = output_dir.path().join(table);
            std::fs::create_dir_all(&table_dir).unwrap();
            let part_paths = (0..19)
                .map(|part| {
                    let relative_path = format!("{table}/part-{part:05}.parquet");
                    let file = File::create(output_dir.path().join(&relative_path)).unwrap();
                    let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
                    writer.write(&batch).unwrap();
                    writer.close().unwrap();
                    relative_path
                })
                .collect();
            StagePartsOut {
                parts: part_paths,
                row_count: 19,
                seed_time: Some([stage as i64, stage as i64 + 18]),
                ..StagePartsOut::default()
            }
        });

        let infos = stage_gen_infos_from_parts(&parts, output_dir.path()).unwrap();
        assert_eq!(infos.len(), 3);
        for (stage, info) in infos.iter().enumerate() {
            assert_eq!(info.parts, parts[stage].parts);
            assert_eq!(info.parts.len(), 19);
            assert_eq!(info.table_name, STAGE_TABLE_NAMES[stage]);
            assert_eq!(info.row_count, 19);
            assert_eq!(info.seed_time, Some([stage as i64, stage as i64 + 18]));
            let inventory = info.artifact_inventory.as_ref().unwrap();
            assert_eq!(inventory.parts.len(), 19);
            assert_eq!(
                inventory
                    .parts
                    .iter()
                    .map(|part| part.row_count)
                    .sum::<u64>(),
                19
            );
        }

        let encoded = serde_json::to_value(&infos).unwrap();
        assert!(encoded
            .as_array()
            .unwrap()
            .iter()
            .all(|stage| stage["parts"].as_array().unwrap().len() == 19));
    }

    #[test]
    fn train_table_category_counts_use_target_specific_limit_without_string_budget() {
        let limit_target = category_count_accumulators(true, false)
            .expect("categorical target accumulator");
        limit_target.unique_count.store(
                MAX_TRAIN_TABLE_TARGET_CATEGORIES - 1,
                std::sync::atomic::Ordering::Relaxed,
            );
        limit_target.record_new_unique();
        assert!(!limit_target.is_aborted());
        limit_target.record_new_unique();
        assert!(limit_target.is_aborted());

        let target = category_count_accumulators(true, false)
            .expect("categorical target accumulator");

        // Forty thousand short strings all survive: target category collection
        // is bounded only by the unique-count cap, not by any string byte budget.
        const NUM_TARGETS: usize = 40_000;
        let mut target_local = LocalColumnCounts::with_capacity(NUM_TARGETS);
        for value in 0..NUM_TARGETS {
            target_local.observe(value as u64, &format!("target_{value}"));
        }
        target.merge_local_batch(&mut target_local);
        assert!(!target.is_aborted());
        assert_eq!(
            target.drain_to_map().expect("complete target counts").len(),
            NUM_TARGETS
        );
    }

    #[test]
    fn link_prediction_does_not_allocate_exact_target_category_counts() {
        // LP targets are graph node IDs with no exact-vocabulary consumer, so no
        // shared accumulator is created.
        let lp_shared = category_count_accumulators(true, true);
        assert!(lp_shared.is_none());
        // The per-worker LOCAL map is gated on the shared accumulator existing
        // (the exact predicate used in execute_chunk / custom_tt_chunk_processor),
        // so it is never allocated for LP — no dead per-worker allocation.
        let lp_local = lp_shared
            .is_some()
            .then(|| LocalColumnCounts::with_capacity(5000));
        assert!(lp_local.is_none());

        // A node-task categorical target allocates both the shared and local maps.
        let node_shared = category_count_accumulators(true, false);
        assert!(node_shared.is_some());
        let node_local = node_shared
            .is_some()
            .then(|| LocalColumnCounts::with_capacity(5000));
        assert!(node_local.is_some());
    }

    #[test]
    fn target_category_limit_is_a_generation_error() {
        let target = category_count_accumulators(true, false)
            .expect("categorical target accumulator");
        target.unique_count.store(
            MAX_TRAIN_TABLE_TARGET_CATEGORIES,
            std::sync::atomic::Ordering::Relaxed,
        );
        target.record_new_unique();

        let error = drain_required_target_category_counts(Some(target.as_ref()))
            .expect_err("an incomplete target vocabulary must fail generation");
        assert!(matches!(&error, TTGenError::TargetCategoryLimit(_)));
        let message = error.to_string();
        assert!(message.contains("TARGET1"), "{message}");
        assert!(
            message.contains(&MAX_TRAIN_TABLE_TARGET_CATEGORIES.to_string()),
            "{message}"
        );
        assert!(message.contains("refusing to publish"), "{message}");
    }

    // ─── HLL hash-basis consistency (LP train-table stat over-count fix) ─────

    /// An entity id observed through BOTH the empty-target-row path and the
    /// `update_exact_stats` path must land on the SAME HLL register, so a single
    /// logical id counts once — not twice. Previously the empty-row branch used
    /// `observe_i64(hash_value(pk))` while the main path used a type-aware RAW
    /// observation, so the same id double-registered (num_entities over-count).
    #[test]
    fn test_entity_hll_hash_basis_consistent_across_paths() {
        // Int, Str, and Float entity ids, each observed via BOTH paths.
        let ids = [
            Value::Int(42),
            Value::Str("user_7".to_string()),
            Value::Float(3.5),
        ];
        let distinct = ids.len() as f64;

        let mut hll = HyperLogLog::new();
        for id in &ids {
            // "main path" observation
            observe_entity_hll(&mut hll, id);
            // "empty-target-row path" observation of the SAME id
            observe_entity_hll(&mut hll, id);
        }

        let est = hll.estimate();
        // With a consistent basis the double observation collapses to `distinct`.
        // (Exact for tiny cardinalities on this HLL impl.)
        assert!(
            (est - distinct).abs() < 0.5,
            "expected ~{distinct} distinct entities, got {est}"
        );

        // Guard against regression to the old broken basis: `observe_i64(hash_value)`
        // is a DIFFERENT register than the canonical raw path, so mixing them
        // would inflate the estimate above `distinct`.
        let mut mixed = HyperLogLog::new();
        for id in &ids {
            observe_entity_hll(&mut mixed, id); // canonical
            mixed.observe_i64(hash_value(id) as i64); // old broken empty-row basis
        }
        assert!(
            mixed.estimate() > distinct + 0.5,
            "sanity: mixed bases MUST over-count (proves the two bases differ)"
        );
    }

    /// A multicategorical LP target observed through BOTH the first-target path
    /// and the additional-target `skip(1)` path must land on the SAME HLL
    /// register. Previously the first target used `hash_bytes(token)` and the
    /// additional targets used `hash_value`, so the same target double-registered
    /// (num_targets over-count).
    #[test]
    fn test_multicat_target_hll_hash_basis_consistent_across_paths() {
        // Multicat targets arrive as single-token lists on the per-token path.
        let targets = [
            Value::List(vec![Value::Int(101)]),
            Value::List(vec![Value::Int(202)]),
            Value::List(vec![Value::Int(303)]),
        ];
        let distinct = targets.len() as f64;

        let mut hll = HyperLogLog::new();
        for t in &targets {
            // "first-target path"
            observe_target_hll(&mut hll, t, false, true, "multicategorical");
            // "additional-target skip(1) path" for the SAME target
            observe_target_hll(&mut hll, t, false, true, "multicategorical");
        }

        let est = hll.estimate();
        assert!(
            (est - distinct).abs() < 0.5,
            "expected ~{distinct} distinct targets, got {est}"
        );

        // Regression guard: the OLD additional-target basis (`hash_value`) differs
        // from the multicat token basis (`hash_bytes`), so mixing over-counts.
        let mut mixed = HyperLogLog::new();
        for t in &targets {
            observe_target_hll(&mut mixed, t, false, true, "multicategorical"); // canonical
            mixed.observe_i64(hash_value(t) as i64); // old broken skip(1) basis
        }
        assert!(
            mixed.estimate() > distinct + 0.5,
            "sanity: mixed bases MUST over-count (proves the two bases differ)"
        );
    }

    #[test]
    fn test_add_calendar_months_basic() {
        // 2024-01-01 00:00:00 UTC
        let jan1 = Abs::<Seconds>::from_raw(1704067200);
        // +3 months = 2024-04-01 (leap year, so Jan→Apr = 91 days)
        let apr1 = add_calendar_months(jan1, 3);
        assert_eq!(apr1.raw(), 1704067200 + 91 * 86400);
    }

    #[test]
    fn test_add_calendar_months_non_leap() {
        // 2023-01-01 00:00:00 UTC
        let jan1 = Abs::<Seconds>::from_raw(1672531200);
        // +3 months = 2023-04-01 (non-leap, Jan→Apr = 90 days)
        let apr1 = add_calendar_months(jan1, 3);
        assert_eq!(apr1.raw(), 1672531200 + 90 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q2() {
        // 2023-04-01 00:00:00 UTC
        let apr1 = Abs::<Seconds>::from_raw(1680307200);
        // +3 months = 2023-07-01 (Apr→Jul = 91 days)
        let jul1 = add_calendar_months(apr1, 3);
        assert_eq!(jul1.raw(), apr1.raw() + 91 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q3() {
        // 2023-07-01 00:00:00 UTC
        let jul1 = Abs::<Seconds>::from_raw(1688169600);
        // +3 months = 2023-10-01 (Jul→Oct = 92 days)
        let oct1 = add_calendar_months(jul1, 3);
        assert_eq!(oct1.raw(), jul1.raw() + 92 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q4() {
        // 2023-10-01 00:00:00 UTC
        let oct1 = Abs::<Seconds>::from_raw(1696118400);
        // +3 months = 2024-01-01 (Oct→Jan = 92 days)
        let jan1 = add_calendar_months(oct1, 3);
        assert_eq!(jan1.raw(), oct1.raw() + 92 * 86400);
    }

    #[test]
    fn test_add_calendar_months_negative() {
        // 2023-04-01 → -3 months = 2023-01-01
        let apr1 = Abs::<Seconds>::from_raw(1680307200);
        let jan1 = add_calendar_months(apr1, -3);
        assert_eq!(jan1.raw(), 1672531200);
    }

    #[test]
    fn test_add_calendar_months_years() {
        // 2023-01-01 + 12 months = 2024-01-01
        let y2023 = Abs::<Seconds>::from_raw(1672531200);
        let y2024 = add_calendar_months(y2023, 12);
        assert_eq!(y2024.raw(), y2023.raw() + 365 * 86400);
    }
}
