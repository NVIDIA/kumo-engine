//! TT-gen executor: drives the entity loop, staged execution, and threading.
//!
//! Entry point: [`execute`] takes a `DiskGraphTTPlan` and an index path,
//! returns output Parquet shard paths.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_schema::DataType;
use diskgraph_core::accumulators::WelfordAccumulator;
use diskgraph_core::tt_builder::*;
use diskgraph_index::graph::GraphIndexReader;
use rayon::prelude::*;

use chrono::{DateTime, Months};

use crate::ttgen_accumulator::{hash_value, Accumulator};
use crate::ttgen_filter::{compare_values, eval_row_filter, eval_stage_filter};
use crate::ttgen_output::{OutputMode, ParquetShardWriter};
use crate::ttgen_resolve::*;

/// Add calendar months to an epoch-seconds timestamp.
/// Matches Kumo's `pd.DateOffset(months=N)` behavior.
fn add_calendar_months(epoch_secs: i64, months: i32) -> i64 {
    let dt = DateTime::from_timestamp(epoch_secs, 0)
        .expect("invalid epoch timestamp for calendar month computation");
    let result = if months >= 0 {
        dt.checked_add_months(Months::new(months as u32))
            .expect("calendar month addition overflow")
    } else {
        dt.checked_sub_months(Months::new((-months) as u32))
            .expect("calendar month subtraction overflow")
    };
    result.timestamp()
}
use crate::ttgen_traversal::*;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Execute a TT-gen plan against an index, producing output Parquet shards.
///
/// Returns the paths of the output shard files.
pub fn execute(
    plan: &DiskGraphTTPlan,
    index_path: &Path,
    output_dir: &Path,
) -> Result<Vec<PathBuf>> {
    // Open index
    let metadata_path = index_path.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, index_path)
        .context("opening graph index")?;

    // Preprocess plan (main thread, once)
    let resolved = resolve_plan(plan, &reader)
        .context("resolving plan")?;
    let resolved = Arc::new(resolved);

    let num_entities = resolved.entity_table.node_count();
    let num_threads = rayon::current_num_threads();

    // Use 8x more chunks than threads for better load balancing via
    // Rayon's work-stealing scheduler. Entities have highly uneven degree
    // distributions, so static per-thread partitioning leaves threads idle.
    let num_chunks = (num_threads * 8).min(num_entities as usize).max(1);

    eprintln!(
        "TT-gen: {} entities, {} threads, {} chunks, {} stages, {} aliases",
        num_entities,
        num_threads,
        num_chunks,
        resolved.stages.len(),
        resolved.num_aliases,
    );

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    // Partition entities into many small chunks for work-stealing
    let shard_paths: Vec<PathBuf> = (0..num_chunks)
        .map(|i| output_dir.join(format!("shard_{}.parquet", i)))
        .collect();

    // Parallel execution with fine-grained chunks
    let results: Vec<Result<u64>> = shard_paths
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, path)| {
            let start = chunk_idx as u64 * num_entities / num_chunks as u64;
            let end = (chunk_idx as u64 + 1) * num_entities / num_chunks as u64;
            execute_chunk(&resolved, start..end, path)
        })
        .collect();

    // Check for errors and report
    let mut total_rows = 0u64;
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(rows) => {
                total_rows += rows;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("thread {} failed: {}", i, e));
            }
        }
    }

    eprintln!("TT-gen: {} total output rows", total_rows);

    Ok(shard_paths)
}

// ─── Per-thread execution ────────────────────────────────────────────────────

fn output_mode_from_prediction_type(pt: &PredictionType, target_dtype: DataType) -> OutputMode {
    match pt {
        PredictionType::Node => OutputMode::Node { target_dtype },
        PredictionType::Link { explode: true, include_frequency } => {
            OutputMode::LinkExploded {
                include_frequency: *include_frequency,
                target_dtype,
            }
        }
        PredictionType::Link { explode: false, include_frequency } => {
            OutputMode::LinkArray {
                include_frequency: *include_frequency,
                target_dtype,
            }
        }
    }
}

fn execute_chunk(
    plan: &ResolvedPlan,
    entity_range: std::ops::Range<u64>,
    output_path: &Path,
) -> Result<u64> {
    let mode = output_mode_from_prediction_type(&plan.prediction_type, plan.target_dtype.clone());
    let mut writer = ParquetShardWriter::new(
        output_path,
        mode,
        plan.lag_timesteps > 0,
        plan.entity_dtype.clone(),
    )?;

    // Thread-local aliases Vec, reused across entities
    let mut aliases: Vec<Option<Value>> = vec![None; plan.num_aliases];
    // Thread-local frequency counts for ListDistinct (reused across entities)
    let mut link_frequencies: Vec<i64> = Vec::new();

    for entity_node in entity_range {
        execute_entity(plan, entity_node, &mut aliases, &mut link_frequencies, &mut writer)?;
    }

    writer.finish()
}

// ─── Per-entity execution ────────────────────────────────────────────────────

fn execute_entity(
    plan: &ResolvedPlan,
    entity_node: u64,
    aliases: &mut Vec<Option<Value>>,
    link_frequencies: &mut Vec<i64>,
    writer: &mut ParquetShardWriter,
) -> Result<()> {
    // 0. Look up the entity PK value for output
    let entity_pk = read_column_value(&plan.entity_table, entity_node, plan.entity_pk_col_idx)
        .unwrap_or(Value::Int(entity_node as i64));

    // 0b. Skip duplicate PK nodes. Ingest assigns separate node IDs to
    //     duplicate PKs; the MPHF stores only the canonical (first) one.
    if !plan.entity_table.is_canonical_node(entity_node, &entity_pk) {
        return Ok(());
    }

    // 1. Read entity time_col and valid_to_col
    let time_col_val: Option<i64> = plan
        .entity_time_col_idx
        .and_then(|idx| read_column_i64(&plan.entity_table, entity_node, idx));
    let valid_to_val: Option<i64> = plan
        .entity_valid_to_col_idx
        .and_then(|idx| read_column_i64(&plan.entity_table, entity_node, idx));

    // 2. Pre-resolve entity column values (for filter RHS EntityColumn)
    let entity_cols: Vec<Option<Value>> = plan
        .entity_col_indices
        .iter()
        .map(|(_, col_idx)| read_column_value(&plan.entity_table, entity_node, *col_idx))
        .collect();

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
        // Non-temporal: single row per entity
        link_frequencies.clear();
        if execute_stages(plan, entity_node, &entity_cols, None, aliases, link_frequencies) {
            let ts = time_col_val.unwrap_or(0);
            write_target(plan, &entity_pk, ts, aliases, link_frequencies, entity_include_bit, writer)?;
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
    for ts_idx in first_valid..last_valid {
        let ts = timestamps[ts_idx];

        // Execute all stages for this (entity, timestamp)
        link_frequencies.clear();
        if execute_stages(plan, entity_node, &entity_cols, Some(ts), aliases, link_frequencies) {
            write_target(plan, &entity_pk, ts, aliases, link_frequencies, entity_include_bit, writer)?;
        }

        // Clear aliases for next timestamp
        aliases.iter_mut().for_each(|a| *a = None);
    }

    Ok(())
}

/// Write the target value(s) for one (entity, timestamp) pair.
fn write_target(
    plan: &ResolvedPlan,
    entity_pk: &Value,
    ts: i64,
    aliases: &[Option<Value>],
    link_frequencies: &[i64],
    entity_include_bit: bool,
    writer: &mut ParquetShardWriter,
) -> Result<()> {
    let target = match &aliases[plan.target_alias_idx] {
        Some(v) => v,
        None => return Ok(()),
    };

    match &plan.prediction_type {
        PredictionType::Node => {
            writer.write_row(entity_pk.clone(), ts, target, entity_include_bit)?;
        }
        PredictionType::Link { explode: true, include_frequency } => {
            if let Value::List(targets) = target {
                for (i, val) in targets.iter().enumerate() {
                    let freq = if *include_frequency {
                        link_frequencies.get(i).copied().unwrap_or(1)
                    } else {
                        0
                    };
                    writer.write_row_exploded(
                        entity_pk.clone(), ts, val.clone(), freq, entity_include_bit,
                    )?;
                }
            }
        }
        PredictionType::Link { explode: false, include_frequency } => {
            if let Value::List(targets) = target {
                let freqs: Vec<i64> = if *include_frequency {
                    link_frequencies.to_vec()
                } else {
                    Vec::new()
                };
                writer.write_row_array(
                    entity_pk.clone(), ts, targets.clone(), &freqs, entity_include_bit,
                )?;
            }
        }
    }
    Ok(())
}

// ─── Staged execution ────────────────────────────────────────────────────────

/// Execute all stages for an (entity, timestamp) pair.
/// Returns true if all stages passed (output row should be emitted).
///
/// `observation_ts` is the current observation timestamp (epoch seconds) for
/// temporal plans. Used to compute absolute window bounds from relative offsets.
/// `None` for non-temporal plans.
///
/// `link_frequencies` is populated with occurrence counts when a `ListDistinct`
/// aggregation runs. Only the last ListDistinct's counts are kept (the target).
fn execute_stages(
    plan: &ResolvedPlan,
    entity_node: u64,
    entity_cols: &[Option<Value>],
    observation_ts: Option<i64>,
    aliases: &mut [Option<Value>],
    link_frequencies: &mut Vec<i64>,
) -> bool {
    for stage in &plan.stages {
        // Execute all ops in this stage
        for op in &stage.ops {
            let result = match op {
                ResolvedOp::Aggregate(agg) => {
                    execute_aggregate(
                        plan, entity_node, agg, entity_cols, observation_ts,
                        link_frequencies,
                    )
                }
                ResolvedOp::Column(col) => {
                    Some(read_column_value(&col.table, entity_node, col.col_idx)
                        .unwrap_or(Value::Int(0)))
                }
                ResolvedOp::Compare(cmp) => {
                    let lhs = aliases[cmp.lhs_alias_idx]
                        .as_ref()
                        .unwrap_or(&Value::Int(0));
                    Some(Value::Bool(compare_values(lhs, &cmp.op, &cmp.rhs)))
                }
            };
            aliases[op.alias_idx()] = result;
        }

        // Stage filter: short-circuit on failure
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

// ─── Aggregate execution (match-once monomorphization) ───────────────────────

fn execute_aggregate(
    _plan: &ResolvedPlan,
    entity_node: u64,
    agg: &ResolvedAggregate,
    entity_cols: &[Option<Value>],
    observation_ts: Option<i64>,
    link_frequencies: &mut Vec<i64>,
) -> Option<Value> {
    // Compute window bounds (sort key space).
    // Kumo uses half-open interval (start, end]: exclusive lower, inclusive upper.
    // encode_upper sets all weight bits to 1, giving the max sort key for that
    // timestamp bucket. Combined with the (lo, hi] search in iter_edge, this
    // ensures edges exactly at the lower boundary are excluded and edges exactly
    // at the upper boundary are included.
    let window = agg.window.as_ref().map(|w| {
        let obs = observation_ts.unwrap_or(0);
        match w {
            ResolvedWindow::Fixed { offset_lo_secs, offset_hi_secs, encoder } => {
                let lo = encoder.encode_upper(obs + offset_lo_secs);
                let hi = encoder.encode_upper(obs + offset_hi_secs);
                (lo, hi)
            }
            ResolvedWindow::Calendar { lo_months, hi_months, encoder } => {
                let lo_secs = add_calendar_months(obs, *lo_months);
                let hi_secs = add_calendar_months(obs, *hi_months);
                let lo = encoder.encode_upper(lo_secs);
                let hi = encoder.encode_upper(hi_secs);
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
                    acc += read_column_f64(dst_table, dst, col_idx);
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
                    w.observe(read_column_f64(dst_table, dst, col_idx));
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
                    acc = acc.min(read_column_f64(dst_table, dst, col_idx));
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
                    acc = acc.max(read_column_f64(dst_table, dst, col_idx));
                }
                true
            });
            Accumulator::Max(acc).finalize()
        }
        AggFunc::CountDistinct => {
            let mut set = HashSet::new();
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(val) = read_column_value(dst_table, dst, col_idx) {
                        set.insert(hash_value(&val));
                    }
                }
                true
            });
            Some(Value::Int(set.len() as i64))
        }
        AggFunc::ListDistinct => {
            let mut seen: HashMap<u64, usize> = HashMap::new();
            let mut values = Vec::new();
            let mut counts = Vec::new();
            let limit = agg.limit.unwrap_or(u32::MAX) as usize;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    if let Some(val) = read_column_value(dst_table, dst, col_idx) {
                        let h = hash_value(&val);
                        match seen.entry(h) {
                            std::collections::hash_map::Entry::Occupied(e) => {
                                counts[*e.get()] += 1;
                            }
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(values.len());
                                values.push(val);
                                counts.push(1i64);
                            }
                        }
                    }
                }
                values.len() < limit // early exit when limit reached
            });
            // Store frequency counts for link prediction output
            link_frequencies.clear();
            link_frequencies.extend_from_slice(&counts);
            Some(Value::List(values))
        }
        AggFunc::First => {
            // Edges iterate DESCENDING by sort key (newest first),
            // so "first" (earliest) = keep overwriting, last encountered wins.
            let mut val = None;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    val = read_column_value(dst_table, dst, col_idx);
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
            let mut val = None;
            traverse_and_accumulate(entity_node, &agg.path, window, true, &mut |dst, _sk| {
                if let Some(ref f) = agg.row_filter {
                    if !eval_row_filter(f, dst, dst_table, entity_cols) {
                        return true;
                    }
                }
                if let Some(col_idx) = agg.target_col_idx {
                    val = read_column_value(dst_table, dst, col_idx);
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
    fn test_add_calendar_months_basic() {
        // 2024-01-01 00:00:00 UTC
        let jan1 = 1704067200i64;
        // +3 months = 2024-04-01 (leap year, so Jan→Apr = 91 days)
        let apr1 = add_calendar_months(jan1, 3);
        assert_eq!(apr1, 1704067200 + 91 * 86400);
    }

    #[test]
    fn test_add_calendar_months_non_leap() {
        // 2023-01-01 00:00:00 UTC
        let jan1 = 1672531200i64;
        // +3 months = 2023-04-01 (non-leap, Jan→Apr = 90 days)
        let apr1 = add_calendar_months(jan1, 3);
        assert_eq!(apr1, 1672531200 + 90 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q2() {
        // 2023-04-01 00:00:00 UTC
        let apr1 = 1680307200i64;
        // +3 months = 2023-07-01 (Apr→Jul = 91 days)
        let jul1 = add_calendar_months(apr1, 3);
        assert_eq!(jul1, apr1 + 91 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q3() {
        // 2023-07-01 00:00:00 UTC
        let jul1 = 1688169600i64;
        // +3 months = 2023-10-01 (Jul→Oct = 92 days)
        let oct1 = add_calendar_months(jul1, 3);
        assert_eq!(oct1, jul1 + 92 * 86400);
    }

    #[test]
    fn test_add_calendar_months_q4() {
        // 2023-10-01 00:00:00 UTC
        let oct1 = 1696118400i64;
        // +3 months = 2024-01-01 (Oct→Jan = 92 days)
        let jan1 = add_calendar_months(oct1, 3);
        assert_eq!(jan1, oct1 + 92 * 86400);
    }

    #[test]
    fn test_add_calendar_months_negative() {
        // 2023-04-01 → -3 months = 2023-01-01
        let apr1 = 1680307200i64;
        let jan1 = add_calendar_months(apr1, -3);
        assert_eq!(jan1, 1672531200);
    }

    #[test]
    fn test_add_calendar_months_years() {
        // 2023-01-01 + 12 months = 2024-01-01
        let y2023 = 1672531200i64;
        let y2024 = add_calendar_months(y2023, 12);
        assert_eq!(y2024, y2023 + 365 * 86400);
    }
}
