//! Fused pass-B: parallel-over-row-group router that reads `tt.parquet` and
//! writes per-stage seed parts directly (`<stage_table>/part-NNNNN.parquet`).
//! Replaces the old split.bin + demux: routing and per-stage parquet write
//! are fused into one pass.
//!
//! ## Concurrency model
//!
//! `run_pass_b_routed` parallelizes across bounded groups of contiguous source
//! row groups via rayon. Each worker opens its own `File`, decodes its span,
//! routes each batch via the supplied closure → `[train, val, test]`, and
//! writes each >0-row sub-batch to
//! its own `ArrowWriter` at `<output_dir>/<stage_table>/part-<group:05>.parquet`,
//! tracking per-stage row_count + TIMESTAMP (min,max) over its OUTPUT rows.
//!
//! Workers write disjoint part files → no cross-worker sharing or sync. The main
//! thread merges per-stage results (sorted parts, summed row_count, merged
//! seed_time min/max). A stage with 0 parts gets a single schema-only 0-row part
//! so ingest never hits a missing dir. Peak RAM is bounded by one decoded
//! source row group per worker.

use std::fs::File;
use std::path::Path;

use arrow_array::{Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;

use crate::errors::{TTGenError, TTGenResult};
use crate::route::{
    assign_per_anchor_stages, route_per_anchor, route_per_link, ForecastingHorizonPolicy,
    LinkAssigner,
};
use crate::split_decider::SplitDecider;
use crate::split_stats::{observe_pass_b_stage_batch, KumoStageStats};
use diskgraph_core::time::UnitSystem;

/// TIMESTAMP column name. Always present in tt.parquet; the router uses it for
/// per-stage seed-time (min,max) over each stage's output rows.
pub(crate) const TIMESTAMP_COL: &str = "TIMESTAMP";

/// Include-mask column emitted by pass A when `lag_timesteps > 0`. The
/// per-anchor kernel forces rows whose mask byte is `false` to `UNUSED`.
pub(crate) const INCLUDE_MASK_COL: &str = "ENTITY_INCLUDE_BIT";

/// Stage subdirectory names for index 0=train / 1=val / 2=test; parts live at
/// `<stage_table>/part-NNNNN.parquet`.
pub const STAGE_TABLE_NAMES: [&str; 3] = ["train_seeds", "val_seeds", "test_seeds"];

const MIN_PASS_B_GROUPS: usize = 16;
const MAX_SOURCE_ROW_GROUPS_PER_PART: usize = 128;

/// Deterministic coalescing policy based only on immutable source shape.
///
/// For production-sized inputs this keeps at least 16 schedulable groups while
/// bounding each physical output part to 128 source row groups. It deliberately
/// does not depend on the current Rayon pool size, so filenames and bytes remain
/// stable across hosts and thread-count settings.
fn source_row_groups_per_part(num_row_groups: usize) -> usize {
    (num_row_groups / MIN_PASS_B_GROUPS).clamp(1, MAX_SOURCE_ROW_GROUPS_PER_PART)
}

/// One stage's fused-pass-B output: its relative part paths, total rows, and
/// seed-time (min,max of the TIMESTAMP column over the stage's OUTPUT rows).
#[derive(Debug, Default)]
pub struct StagePartsOut {
    /// Relative part paths, e.g. `["train_seeds/part-00000.parquet", ...]`,
    /// sorted by source-group index. Always at least one entry (a single
    /// schema-only 0-row part for an empty stage).
    pub parts: Vec<String>,
    pub row_count: u64,
    /// `(min, max)` of the stage's OUTPUT TIMESTAMP column in int64-ns, or
    /// `None` when the stage carried no finite timestamp (e.g. empty stage).
    pub seed_time: Option<[i64; 2]>,
    pub(crate) split_stats: KumoStageStats,
}

/// Per-(source-row-group span, stage) worker result.
struct WorkerStageResult {
    /// `Some(rel_path)` when this span wrote a >0-row part for the stage.
    part: Option<String>,
    row_count: u64,
    seed_time: Option<(i64, i64)>,
    split_stats: KumoStageStats,
}

/// Relative part path for a stage group, e.g.
/// `"train_seeds/part-00003.parquet"`.
fn part_rel_path(stage: usize, group_idx: usize) -> String {
    format!("{}/part-{group_idx:05}.parquet", STAGE_TABLE_NAMES[stage])
}

/// Core fused router. Parallel over row groups; each worker reads its
/// row-group (all columns), routes each batch via `route`, and writes each
/// non-empty stage sub-batch to its own part file. Returns per-stage
/// `[train, val, test]` parts + counts + seed-times.
///
/// `out_schema` is used to write schema-only 0-row parts for stages that
/// received no rows (so the ingest glob never hits a missing dir).
///
/// `route(batch, global_offset)` is the stage source. `global_offset` is the
/// global row index of `batch[0]` (used by global-index deciders such as
/// `FixedDecider` and by the use-bin slice).
fn run_pass_b_routed<F>(
    tt_parquet: &Path,
    output_dir: &Path,
    total_rows: u64,
    out_schema: &SchemaRef,
    route: F,
) -> TTGenResult<[StagePartsOut; 3]>
where
    F: Fn(&RecordBatch, u64) -> TTGenResult<[RecordBatch; 3]> + Sync,
{
    run_pass_b_routed_impl(tt_parquet, output_dir, total_rows, out_schema, None, route)
}

fn run_pass_b_routed_impl<F>(
    tt_parquet: &Path,
    output_dir: &Path,
    total_rows: u64,
    out_schema: &SchemaRef,
    source_row_groups_per_part_override: Option<usize>,
    route: F,
) -> TTGenResult<[StagePartsOut; 3]>
where
    F: Fn(&RecordBatch, u64) -> TTGenResult<[RecordBatch; 3]> + Sync,
{
    // Pre-create the three stage subdirectories so workers (and the
    // empty-stage fallback below) can write into them without a race.
    for name in STAGE_TABLE_NAMES {
        std::fs::create_dir_all(output_dir.join(name))?;
    }

    // Empty input → each stage is a single schema-only 0-row part.
    if total_rows == 0 {
        return Ok([
            write_empty_stage_part(output_dir, 0, out_schema)?,
            write_empty_stage_part(output_dir, 1, out_schema)?,
            write_empty_stage_part(output_dir, 2, out_schema)?,
        ]);
    }

    // Probe metadata once on the main thread for per-row-group offsets.
    let meta_file = File::open(tt_parquet)?;
    let probe_builder = ParquetRecordBatchReaderBuilder::try_new(meta_file)?;
    let metadata = probe_builder.metadata().clone();
    let row_groups = metadata.row_groups();
    let num_row_groups = row_groups.len();
    let source_row_groups_per_part = source_row_groups_per_part_override
        .unwrap_or_else(|| source_row_groups_per_part(num_row_groups));
    if source_row_groups_per_part == 0 {
        return Err(TTGenError::SystemError(
            "pass_b: source row groups per part must be positive".to_string(),
        ));
    }

    let mut cum_offsets: Vec<u64> = Vec::with_capacity(num_row_groups + 1);
    cum_offsets.push(0);
    for rg in row_groups {
        let prev = *cum_offsets.last().unwrap();
        cum_offsets.push(prev + rg.num_rows() as u64);
    }
    let computed_total = *cum_offsets.last().unwrap();
    if computed_total != total_rows {
        return Err(TTGenError::SystemError(format!(
            "pass_b: parquet has {computed_total} rows but pass A reported {total_rows}"
        )));
    }

    let tt_parquet = tt_parquet.to_path_buf();
    let output_dir = output_dir.to_path_buf();

    // One worker per contiguous group; each returns its per-stage result.
    let num_groups = num_row_groups.div_ceil(source_row_groups_per_part);
    let per_worker: Vec<TTGenResult<[WorkerStageResult; 3]>> = (0..num_groups)
        .into_par_iter()
        .map(|group_idx| {
            let start_rg = group_idx * source_row_groups_per_part;
            let end_rg = (start_rg + source_row_groups_per_part).min(num_row_groups);
            run_pass_b_row_group_span(
                &tt_parquet,
                &output_dir,
                group_idx,
                start_rg,
                end_rg,
                cum_offsets[start_rg],
                &route,
            )
        })
        .collect();

    // Merge: per-stage parts (sorted by rg), row_count sum, seed_time min/max.
    let mut merged: [StagePartsOut; 3] = Default::default();
    for r in per_worker {
        let stages = r?;
        for (stage, ws) in stages.into_iter().enumerate() {
            if let Some(part) = ws.part {
                merged[stage].parts.push(part);
            }
            merged[stage].row_count += ws.row_count;
            merged[stage].split_stats.merge(ws.split_stats);
            if let Some((mn, mx)) = ws.seed_time {
                merged[stage].seed_time = Some(match merged[stage].seed_time {
                    Some([emn, emx]) => [emn.min(mn), emx.max(mx)],
                    None => [mn, mx],
                });
            }
        }
    }

    // Parts were collected in rayon completion order; sort by path so the
    // recorded order is deterministic (part-00000, part-00001, ...).
    for (stage, item) in merged.iter_mut().enumerate() {
        item.parts.sort();
        // Any stage that received zero rows across all row-groups → write a
        // single schema-only 0-row part so the dir is non-empty.
        if item.parts.is_empty() {
            let empty = write_empty_stage_part(&output_dir, stage, out_schema)?;
            item.parts = empty.parts;
        }
    }

    Ok(merged)
}

/// Per-group worker: read a contiguous source row-group span (all columns),
/// route batches in source order, and write each non-empty stage to one part.
fn run_pass_b_row_group_span<F>(
    tt_parquet: &Path,
    output_dir: &Path,
    group_idx: usize,
    start_rg: usize,
    end_rg: usize,
    row_offset: u64,
    route: &F,
) -> TTGenResult<[WorkerStageResult; 3]>
where
    F: Fn(&RecordBatch, u64) -> TTGenResult<[RecordBatch; 3]>,
{
    // Lazily-created per-stage writers + accumulators for this group.
    let mut writers: [Option<ArrowWriter<File>>; 3] = [None, None, None];
    let mut counts: [u64; 3] = [0; 3];
    let mut seed_times: [Option<(i64, i64)>; 3] = [None, None, None];
    let mut split_stats: [KumoStageStats; 3] = Default::default();

    let mut local_idx: u64 = 0;
    for rg_idx in start_rg..end_rg {
        let file = File::open(tt_parquet)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_row_groups(vec![rg_idx])
            .build()?;
        for batch_res in reader {
            let batch =
                batch_res.map_err(|e| TTGenError::Parquet(format!("pass_b reader: {e}")))?;
            let nrows = batch.num_rows() as u64;
            let sub_batches = route(&batch, row_offset + local_idx)?;
            for (stage, sub) in sub_batches.into_iter().enumerate() {
                if sub.num_rows() == 0 {
                    continue;
                }
                observe_seed_time(&sub, &mut seed_times[stage])?;
                counts[stage] += sub.num_rows() as u64;
                observe_pass_b_stage_batch(&sub, &mut split_stats[stage])?;
                let writer = match writers[stage].as_mut() {
                    Some(w) => w,
                    None => {
                        let path = output_dir.join(part_rel_path(stage, group_idx));
                        let f = File::create(&path)?;
                        let w = ArrowWriter::try_new(f, sub.schema(), None)?;
                        writers[stage] = Some(w);
                        writers[stage].as_mut().unwrap()
                    }
                };
                writer.write(&sub)?;
            }
            local_idx += nrows;
        }

        // Preserve each source row-group boundary inside the coalesced file.
        // This bounds decode size and resets dictionaries at the same locality
        // boundary as the previous one-file-per-source-row-group layout.
        for writer in writers.iter_mut().flatten() {
            writer.flush()?;
        }
    }

    // Close any writers we opened; build the per-stage results.
    let mut out: [WorkerStageResult; 3] = [
        WorkerStageResult {
            part: None,
            row_count: 0,
            seed_time: None,
            split_stats: KumoStageStats::default(),
        },
        WorkerStageResult {
            part: None,
            row_count: 0,
            seed_time: None,
            split_stats: KumoStageStats::default(),
        },
        WorkerStageResult {
            part: None,
            row_count: 0,
            seed_time: None,
            split_stats: KumoStageStats::default(),
        },
    ];
    for stage in 0..3 {
        if let Some(w) = writers[stage].take() {
            w.close()?;
            out[stage] = WorkerStageResult {
                part: Some(part_rel_path(stage, group_idx)),
                row_count: counts[stage],
                seed_time: seed_times[stage],
                split_stats: std::mem::take(&mut split_stats[stage]),
            };
        }
    }
    Ok(out)
}

/// Fold the TIMESTAMP column of `batch` into `acc` as a running (min,max).
/// Null/absent timestamps are skipped. Errors if TIMESTAMP is present but not
/// `TimestampNanosecondArray` (tt.parquet always emits ns).
fn observe_seed_time(
    batch: &RecordBatch,
    acc: &mut Option<(i64, i64)>,
) -> TTGenResult<()> {
    let Some((idx, _)) = batch.schema_ref().column_with_name(TIMESTAMP_COL) else {
        return Ok(());
    };
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| {
            TTGenError::SystemError(format!(
                "pass_b: {TIMESTAMP_COL} must be TimestampNanosecondArray for seed-time"
            ))
        })?;
    for r in 0..arr.len() {
        if arr.is_null(r) {
            continue;
        }
        let v = arr.value(r);
        *acc = Some(match *acc {
            Some((mn, mx)) => (mn.min(v), mx.max(v)),
            None => (v, v),
        });
    }
    Ok(())
}

/// Write a single schema-only 0-row part for `stage` and return its
/// `StagePartsOut` (one part, 0 rows, no seed-time).
fn write_empty_stage_part(
    output_dir: &Path,
    stage: usize,
    out_schema: &SchemaRef,
) -> TTGenResult<StagePartsOut> {
    let rel = part_rel_path(stage, 0);
    let path = output_dir.join(&rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(&path)?;
    let writer = ArrowWriter::try_new(f, out_schema.clone(), None)?;
    // Closing without writing produces a valid 0-row parquet carrying the schema.
    writer.close()?;
    Ok(StagePartsOut {
        parts: vec![rel],
        row_count: 0,
        seed_time: None,
        split_stats: KumoStageStats::default(),
    })
}

// ─── Mode entry points ───────────────────────────────────────────────────────

/// Per-anchor fused pass-B: a [`SplitDecider`] decides each whole row's stage.
/// `is_array_lp` folds in the empty-target drop (anchor rows whose `TARGET1`
/// list is empty go nowhere).
///
/// `drop_split_column` is `Some(col)` for the custom-TT split-column path: that
/// column is the split signal and is projected OUT of each stage sub-batch so
/// it never reaches the seed rowpack / features. `out_schema` must already
/// reflect the drop (it is used for empty-stage 0-row parts).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pass_b_per_anchor<U: UnitSystem, D: SplitDecider<U> + Sync>(
    tt_parquet: &Path,
    decider: &D,
    output_dir: &Path,
    total_rows: u64,
    is_array_lp: bool,
    drop_split_column: Option<&str>,
    out_schema: &SchemaRef,
    forecasting_horizon: Option<(i64, i64)>,
) -> TTGenResult<[StagePartsOut; 3]> {
    let horizon_policy = if let Some((step_ns, length)) = forecasting_horizon {
        let max_ts = stage_max_timestamps::<U, D>(tt_parquet, decider, is_array_lp)?;
        ForecastingHorizonPolicy::new(max_ts, step_ns, length)
    } else {
        None
    };
    run_pass_b_routed(
        tt_parquet,
        output_dir,
        total_rows,
        out_schema,
        |batch, offset| {
            route_per_anchor::<U, D>(
                batch,
                decider,
                offset,
                is_array_lp,
                horizon_policy.as_ref(),
                drop_split_column,
            )
        },
    )
}

fn stage_max_timestamps<U: UnitSystem, D: SplitDecider<U>>(
    tt_parquet: &Path,
    decider: &D,
    is_array_lp: bool,
) -> TTGenResult<[Option<i64>; 3]> {
    let file = File::open(tt_parquet)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut max_ts: [Option<i64>; 3] = [None, None, None];
    let mut offset = 0u64;
    for batch_res in reader {
        let batch =
            batch_res.map_err(|e| TTGenError::Parquet(format!("pass_b reader: {e}")))?;
        let stages = assign_per_anchor_stages::<U, D>(&batch, decider, offset, is_array_lp, None)?;
        let ts = timestamp_ns_array(&batch)?;
        for (row, stage) in stages.into_iter().enumerate() {
            if (stage as usize) < 3 {
                let value = ts.value(row);
                let slot = &mut max_ts[stage as usize];
                *slot = Some(slot.map_or(value, |cur| cur.max(value)));
            }
        }
        offset += batch.num_rows() as u64;
    }
    Ok(max_ts)
}

fn timestamp_ns_array(batch: &RecordBatch) -> TTGenResult<&TimestampNanosecondArray> {
    let (idx, _) = batch.schema_ref().column_with_name(TIMESTAMP_COL).ok_or_else(|| {
        TTGenError::SystemError("pass_b: TIMESTAMP column missing".to_string())
    })?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| {
            TTGenError::SystemError(
                "pass_b: TIMESTAMP must be TimestampNanosecondArray".to_string(),
            )
        })
}

/// Per-link fused pass-B: array-LP `RandomLinkSplit` / `CustomLinkSplit`. Each
/// link in an anchor's `TARGET1` list is assigned a stage; the links are
/// bucketed into ≤3 sub-lists and one output row per non-empty `(anchor,
/// stage)` is emitted. (`global_offset` is unused for per-link.)
pub(crate) fn run_pass_b_per_link(
    tt_parquet: &Path,
    assigner: &LinkAssigner,
    output_dir: &Path,
    total_rows: u64,
    out_schema: &SchemaRef,
) -> TTGenResult<[StagePartsOut; 3]> {
    run_pass_b_routed(
        tt_parquet,
        output_dir,
        total_rows,
        out_schema,
        |batch, _offset| route_per_link(batch, assigner),
    )
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::type_complexity, clippy::len_zero)]
mod tests {
    use super::*;
    use crate::split_decider::{NoSplitDecider, TimeRangeDecider};
    use crate::split_stats::write_kumo_split_stats_artifact;
    use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder, TimestampNanosecondBuilder};
    use arrow_array::{Int64Array, RecordBatch, TimestampNanosecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use diskgraph_core::time::{Abs, Seconds};
    use diskgraph_core::tt_builder::PredictionType;
    use std::path::PathBuf;
    use std::sync::Arc as StdArc;

    const FREQUENCY_COL: &str = "__kumo_internal_frequency_column__";

    /// Write a tiny tt-parquet (ENTITY1:Int64, TIMESTAMP:ts(ns)) with optional
    /// row-group size and return its schema.
    fn write_tt_parquet(path: &Path, ts_secs: &[i64], max_rg_rows: Option<usize>) -> SchemaRef {
        let entities: Vec<i64> = (0..ts_secs.len() as i64).collect();
        let ts_nanos: Vec<i64> = ts_secs.iter().map(|s| s * 1_000_000_000).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(Int64Array::from(entities)),
                StdArc::new(TimestampNanosecondArray::from(ts_nanos)),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut props = parquet::file::properties::WriterProperties::builder();
        if let Some(n) = max_rg_rows {
            props = props.set_max_row_group_row_count(Some(n));
        }
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props.build())).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        schema
    }

    fn write_numeric_tt_parquet(path: &Path, ts_secs: &[i64], max_rg_rows: usize) -> SchemaRef {
        let entities = (0..ts_secs.len() as i64).collect::<Vec<_>>();
        let ts_nanos = ts_secs
            .iter()
            .map(|seconds| seconds * 1_000_000_000)
            .collect::<Vec<_>>();
        let targets = (0..ts_secs.len())
            .map(|row| (row % 11) as f64 * 0.25)
            .collect::<Vec<_>>();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("TARGET1", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(Int64Array::from(entities)),
                StdArc::new(TimestampNanosecondArray::from(ts_nanos)),
                StdArc::new(arrow_array::Float64Array::from(targets)),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(max_rg_rows))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        schema
    }

    /// Write an array-LP tt-parquet (ENTITY1, TIMESTAMP, TARGET1:List<Utf8>,
    /// FREQUENCY:List<Int64>) with optional row-group size; return its schema.
    fn write_lp_parquet(
        path: &Path,
        rows: &[(i64, i64, Vec<(&str, i64)>)],
        max_rg_rows: Option<usize>,
    ) -> SchemaRef {
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
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(entity.finish()),
                StdArc::new(ts_b.finish()),
                StdArc::new(target_lb.finish()),
                StdArc::new(freq_lb.finish()),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut props = parquet::file::properties::WriterProperties::builder();
        if let Some(n) = max_rg_rows {
            props = props.set_max_row_group_row_count(Some(n));
        }
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props.build())).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        schema
    }

    /// Read all ENTITY1 values across a stage's parts (in `output_dir`).
    fn stage_entities(output_dir: &Path, parts: &[String]) -> Vec<i64> {
        let mut out = Vec::new();
        for p in parts {
            let file = File::open(output_dir.join(p)).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                let idx = batch.schema().index_of("ENTITY1").unwrap();
                let arr = batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend((0..arr.len()).map(|i| arr.value(i)));
            }
        }
        out
    }

    fn part_row_count(output_dir: &Path, part: &str) -> u64 {
        let file = File::open(output_dir.join(part)).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        reader.map(|b| b.unwrap().num_rows() as u64).sum()
    }

    fn time_range_decider() -> TimeRangeDecider<Seconds> {
        TimeRangeDecider::new(vec![
            (Abs::from_secs(0), Abs::from_secs(100)),
            (Abs::from_secs(101), Abs::from_secs(200)),
            (Abs::from_secs(201), Abs::from_secs(300)),
        ])
    }

    fn run_per_anchor_with_group_size<D: SplitDecider<Seconds> + Sync>(
        tt: &Path,
        decider: &D,
        output: &Path,
        total_rows: u64,
        schema: &SchemaRef,
        group_size: usize,
    ) -> TTGenResult<[StagePartsOut; 3]> {
        run_pass_b_routed_impl(
            tt,
            output,
            total_rows,
            schema,
            Some(group_size),
            |batch, offset| {
                route_per_anchor::<Seconds, D>(batch, decider, offset, false, None, None)
            },
        )
    }

    fn write_split_stats(output: &Path, parts: &[StagePartsOut; 3]) -> Vec<u8> {
        let (path, _) =
            write_kumo_split_stats_artifact(output, parts, &PredictionType::Node, "numerical")
                .unwrap()
                .unwrap();
        std::fs::read(path).unwrap()
    }

    #[test]
    fn source_row_group_policy_boundaries() {
        for (num_row_groups, expected) in [
            (0, 1),
            (1, 1),
            (15, 1),
            (16, 1),
            (31, 1),
            (32, 2),
            (255, 15),
            (256, 16),
            (2_047, 127),
            (2_048, 128),
            (2_049, 128),
            (usize::MAX, 128),
        ] {
            assert_eq!(source_row_groups_per_part(num_row_groups), expected);
        }

        for num_row_groups in 16..10_000 {
            let group_size = source_row_groups_per_part(num_row_groups);
            assert!(group_size <= MAX_SOURCE_ROW_GROUPS_PER_PART);
            assert!(num_row_groups.div_ceil(group_size) >= MIN_PASS_B_GROUPS);
        }
    }

    #[test]
    fn grouping_preserves_order_semantics_and_split_stats() {
        let source = tempfile::tempdir().unwrap();
        let tt = source.path().join("tt.parquet");
        let timestamps = (0..64)
            .map(|row| [10_i64, 110, 210, 310][row % 4])
            .collect::<Vec<_>>();
        let schema = write_numeric_tt_parquet(&tt, &timestamps, 1);
        let decider = time_range_decider();

        let one_part_per_row_group = tempfile::tempdir().unwrap();
        let baseline = run_per_anchor_with_group_size(
            &tt,
            &decider,
            one_part_per_row_group.path(),
            timestamps.len() as u64,
            &schema,
            1,
        )
        .unwrap();
        let grouped_dir = tempfile::tempdir().unwrap();
        let grouped = run_per_anchor_with_group_size(
            &tt,
            &decider,
            grouped_dir.path(),
            timestamps.len() as u64,
            &schema,
            4,
        )
        .unwrap();

        for stage in 0..3 {
            assert_eq!(grouped[stage].row_count, baseline[stage].row_count);
            assert_eq!(grouped[stage].seed_time, baseline[stage].seed_time);
            assert_eq!(
                stage_entities(grouped_dir.path(), &grouped[stage].parts),
                stage_entities(one_part_per_row_group.path(), &baseline[stage].parts,),
            );
            assert_eq!(grouped[stage].parts.len(), 16);
            assert_eq!(
                grouped[stage].parts,
                (0..16)
                    .map(|part| part_rel_path(stage, part))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            write_split_stats(grouped_dir.path(), &grouped),
            write_split_stats(one_part_per_row_group.path(), &baseline),
        );
    }

    #[test]
    fn production_grouping_is_thread_count_and_byte_invariant() {
        let source = tempfile::tempdir().unwrap();
        let tt = source.path().join("tt.parquet");
        let timestamps = (0..64)
            .map(|row| [10_i64, 110, 210, 310][row % 4])
            .collect::<Vec<_>>();
        let schema = write_tt_parquet(&tt, &timestamps, Some(1));
        let decider = time_range_decider();

        let run = |threads| {
            let output = tempfile::tempdir().unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let parts = pool
                .install(|| {
                    run_pass_b_per_anchor::<Seconds, _>(
                        &tt,
                        &decider,
                        output.path(),
                        timestamps.len() as u64,
                        false,
                        None,
                        &schema,
                        None,
                    )
                })
                .unwrap();
            let files = parts.each_ref().map(|stage| {
                stage
                    .parts
                    .iter()
                    .map(|part| {
                        (
                            part.clone(),
                            std::fs::read(output.path().join(part)).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            let stats = write_split_stats(output.path(), &parts);
            (parts.map(|stage| stage.parts), files, stats)
        };

        assert_eq!(run(1), run(7));
    }

    #[test]
    fn per_anchor_routes_and_writes_parts() {
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet");
        let ts_secs = vec![10_i64, 50, 150, 250, 400]; // train,train,val,test,unused
        let schema = write_tt_parquet(&tt, &ts_secs, None);
        let d = time_range_decider();
        let out = run_pass_b_per_anchor::<Seconds, _>(
            &tt, &d, dir.path(), ts_secs.len() as u64, false, None, &schema, None,
        )
        .unwrap();
        assert_eq!(out[0].row_count, 2);
        assert_eq!(out[1].row_count, 1);
        assert_eq!(out[2].row_count, 1);
        assert_eq!(stage_entities(dir.path(), &out[0].parts), vec![0, 1]);
        assert_eq!(stage_entities(dir.path(), &out[1].parts), vec![2]);
        assert_eq!(stage_entities(dir.path(), &out[2].parts), vec![3]);
        // Every stage has at least one part on disk.
        for s in &out {
            assert!(!s.parts.is_empty());
            for p in &s.parts {
                assert!(dir.path().join(p).exists());
            }
        }
        // Seed-time of train = (min,max) over ts 10,50 (in ns).
        assert_eq!(out[0].seed_time, Some([10 * 1_000_000_000, 50 * 1_000_000_000]));
    }

    #[test]
    fn per_anchor_empty_stage_gets_schema_only_part() {
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet");
        // All rows land in train; val/test empty.
        let ts_secs = vec![10_i64, 20, 30];
        let schema = write_tt_parquet(&tt, &ts_secs, None);
        let d = time_range_decider();
        let out = run_pass_b_per_anchor::<Seconds, _>(
            &tt, &d, dir.path(), ts_secs.len() as u64, false, None, &schema, None,
        )
        .unwrap();
        assert_eq!(out[0].row_count, 3);
        for stage in 1..3 {
            assert_eq!(out[stage].row_count, 0);
            assert_eq!(out[stage].parts.len(), 1, "empty stage → single schema-only part");
            assert_eq!(part_row_count(dir.path(), &out[stage].parts[0]), 0);
            assert!(out[stage].seed_time.is_none());
        }
    }

    #[test]
    fn per_anchor_zero_rows_all_schema_only() {
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet"); // never created
        let schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        let d = NoSplitDecider;
        let out = run_pass_b_per_anchor::<Seconds, NoSplitDecider>(
            &tt, &d, dir.path(), 0, false, None, &schema, None,
        )
        .unwrap();
        for s in &out {
            assert_eq!(s.row_count, 0);
            assert_eq!(s.parts.len(), 1);
            assert_eq!(part_row_count(dir.path(), &s.parts[0]), 0);
        }
    }

    #[test]
    fn multi_rowgroup_yields_multiple_parts_summing_to_count() {
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet");
        // 10 rows all in train, 3 rows/group → 4 row-groups → 4 train parts.
        let ts_secs: Vec<i64> = (0..10).map(|i| i as i64 + 1).collect();
        let schema = write_tt_parquet(&tt, &ts_secs, Some(3));
        let d = time_range_decider();
        let out = run_pass_b_per_anchor::<Seconds, _>(
            &tt, &d, dir.path(), ts_secs.len() as u64, false, None, &schema, None,
        )
        .unwrap();
        assert_eq!(out[0].row_count, 10);
        assert!(out[0].parts.len() > 1, "multi-rowgroup → >1 part for non-empty stage");
        let sum: u64 = out[0].parts.iter().map(|p| part_row_count(dir.path(), p)).sum();
        assert_eq!(sum, out[0].row_count);
        // Parts are sorted (part-00000, part-00001, ...).
        let mut sorted = out[0].parts.clone();
        sorted.sort();
        assert_eq!(out[0].parts, sorted);
    }

    #[test]
    fn per_anchor_array_lp_drops_empty_target() {
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet");
        let rows: Vec<(i64, i64, Vec<(&str, i64)>)> = vec![
            (0, 10, vec![("a", 1)]),  // train
            (1, 50, vec![]),          // empty target → dropped
            (2, 150, vec![("b", 1)]), // val
        ];
        let schema = write_lp_parquet(&tt, &rows, None);
        let d = time_range_decider();
        let out = run_pass_b_per_anchor::<Seconds, _>(
            &tt, &d, dir.path(), rows.len() as u64, true, None, &schema, None,
        )
        .unwrap();
        assert_eq!(out[0].row_count, 1);
        assert_eq!(out[1].row_count, 1);
        assert_eq!(out[0].row_count + out[1].row_count + out[2].row_count, 2);
        // Empty-target row (ts=50) does NOT contribute to any seed-time.
        assert_eq!(out[0].seed_time, Some([10 * 1_000_000_000, 10 * 1_000_000_000]));
    }

    /// Custom-TT split-column drop: the configured split column is projected
    /// out of every stage's seed parquet, while routing still works.
    #[test]
    fn per_anchor_drops_split_column() {
        use arrow_array::StringArray;
        let dir = tempfile::tempdir().unwrap();
        let tt = dir.path().join("tt.parquet");
        // ENTITY1, TIMESTAMP, split_col(Utf8). Route by ColumnDeciderManual.
        let entities = Int64Array::from(vec![0_i64, 1, 2, 3]);
        let ts = TimestampNanosecondArray::from(
            vec![10_i64, 20, 30, 40]
                .into_iter()
                .map(|s| s * 1_000_000_000)
                .collect::<Vec<_>>(),
        );
        let split = StringArray::from(vec!["0", "1", "2", "0"]);
        let in_schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("split_col", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            in_schema.clone(),
            vec![StdArc::new(entities), StdArc::new(ts), StdArc::new(split)],
        )
        .unwrap();
        let file = File::create(&tt).unwrap();
        let mut w = ArrowWriter::try_new(file, in_schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        // out_schema with split_col dropped (the caller's responsibility).
        let out_schema = StdArc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        let d = crate::split_decider::ColumnDeciderManual::new(
            "split_col".to_string(),
            vec!["0".to_string()],
            vec!["1".to_string()],
            vec!["2".to_string()],
        );
        let out = run_pass_b_per_anchor::<Seconds, _>(
            &tt,
            &d,
            dir.path(),
            4,
            false,
            Some("split_col"),
            &out_schema,
            None,
        )
        .unwrap();
        // Routing: 0,3 → train; 1 → val; 2 → test.
        assert_eq!(stage_entities(dir.path(), &out[0].parts), vec![0, 3]);
        assert_eq!(stage_entities(dir.path(), &out[1].parts), vec![1]);
        assert_eq!(stage_entities(dir.path(), &out[2].parts), vec![2]);
        // split_col is ABSENT from every stage part's schema.
        for s in &out {
            for p in &s.parts {
                let f = File::open(dir.path().join(p)).unwrap();
                let schema = ParquetRecordBatchReaderBuilder::try_new(f)
                    .unwrap()
                    .schema()
                    .clone();
                assert!(
                    schema.index_of("split_col").is_err(),
                    "split_col must be dropped from stage part {p}"
                );
                assert!(schema.index_of("ENTITY1").is_ok());
            }
        }
    }

    // ── per-link batch/rowgroup invariance ───────────────────────────────

    /// Collect each stage's TARGET lists across parts (sorted by entity for
    /// stable comparison). Returns Vec of (entity, sorted-target-list).
    fn stage_membership(output_dir: &Path, parts: &[String]) -> Vec<(i64, Vec<String>)> {
        let mut out: Vec<(i64, Vec<String>)> = Vec::new();
        for p in parts {
            let file = File::open(output_dir.join(p)).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                let eidx = batch.schema().index_of("ENTITY1").unwrap();
                let earr = batch.column(eidx).as_any().downcast_ref::<Int64Array>().unwrap();
                let tidx = batch.schema().index_of("TARGET1").unwrap();
                let tlist = batch
                    .column(tidx)
                    .as_any()
                    .downcast_ref::<arrow_array::ListArray>()
                    .unwrap();
                let items = tlist
                    .values()
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap();
                for r in 0..batch.num_rows() {
                    let s = tlist.value_offsets()[r] as usize;
                    let e = tlist.value_offsets()[r + 1] as usize;
                    let mut lst: Vec<String> = (s..e).map(|j| items.value(j).to_string()).collect();
                    lst.sort();
                    out.push((earr.value(r), lst));
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn per_link_grouping_preserves_semantics() {
        let rows: Vec<(i64, i64, Vec<(&str, i64)>)> = (0..64)
            .map(|i| {
                let items: Vec<(&str, i64)> = vec![("a", 1), ("b", 2), ("c", 3)];
                (i as i64, 1000 + i as i64, items)
            })
            .collect();
        let seed = 7_u64;
        let assigner = LinkAssigner::random_from_ratio(&[0.5, 0.3, 0.2], seed);

        // Single row-group run.
        let dir1 = tempfile::tempdir().unwrap();
        let tt1: PathBuf = dir1.path().join("tt.parquet");
        let schema1 = write_lp_parquet(&tt1, &rows, None);
        let out1 =
            run_pass_b_per_link(&tt1, &assigner, dir1.path(), rows.len() as u64, &schema1).unwrap();

        // One source row per row group selects the production group size of 4.
        let dir2 = tempfile::tempdir().unwrap();
        let tt2: PathBuf = dir2.path().join("tt.parquet");
        let schema2 = write_lp_parquet(&tt2, &rows, Some(1));
        let out2 =
            run_pass_b_per_link(&tt2, &assigner, dir2.path(), rows.len() as u64, &schema2).unwrap();

        for stage in 0..3 {
            assert_eq!(
                out1[stage].row_count, out2[stage].row_count,
                "stage {stage} row_count differs across row-group sizes"
            );
            let m1 = stage_membership(dir1.path(), &out1[stage].parts);
            let m2 = stage_membership(dir2.path(), &out2[stage].parts);
            assert_eq!(m1, m2, "stage {stage} membership differs across row-group sizes");
        }
        // The shared production entry bounds all physical paths to 16 groups;
        // summing their rows still yields the exact routed stage count.
        for stage in 0..3 {
            if out2[stage].row_count > 0 {
                let expected_paths = (0..16)
                    .map(|part| part_rel_path(stage, part))
                    .collect::<Vec<_>>();
                assert!(out2[stage]
                    .parts
                    .iter()
                    .all(|part| expected_paths.contains(part)));
                assert!(out2[stage].parts.len() <= 16);
                let sum: u64 = out2[stage]
                    .parts
                    .iter()
                    .map(|p| part_row_count(dir2.path(), p))
                    .sum();
                assert_eq!(sum, out2[stage].row_count);
            }
        }
    }
}
