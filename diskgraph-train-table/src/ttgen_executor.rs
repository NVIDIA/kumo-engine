//! TT-gen executor: drives the entity loop, staged execution, and threading.
//!
//! Entry point: [`execute`] takes a `DiskGraphTTPlan<U>` and an index path,
//! returns output Parquet shard paths.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use crate::errors::{TTGenError, TTGenResult};
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use crossbeam_channel::{bounded, Receiver, Sender};
use diskgraph_core::accumulators::{HyperLogLog, LocalColumnCounts, SharedColumnCounts, TDigestEstimator, WelfordAccumulator, for_each_split_ascii_byte};
use diskgraph_core::time::{Abs, Dur, UnitSystem, add_calendar_months, dur_from_unit, TimeUnit};
use diskgraph_core::tt_builder::*;
use diskgraph_index::graph::GraphIndexReader;
use rayon::prelude::*;

use crate::ttgen_accumulator::{hash_i64, hash_str, hash_value, Accumulator};
use crate::ttgen_filter::{
    compare_values, eval_row_filter, eval_stage_filter, is_nullish_filter_value,
};
use crate::ttgen_output::{ChannelBatchWriter, ExtraColumnOutput, ExtraValueRef, OutputMode};
use crate::ttgen_resolve::*;
use crate::ttgen_stats::{ChunkStats, DateOffsetSplitSemantics, StatsMerger, TTExecutionSummary};
use crate::ttgen_traversal::{*, MulticatToken};

// ─── Public API ──────────────────────────────────────────────────────────────

/// Execute a TT-gen plan against an index, streaming RecordBatches through
/// a crossbeam channel.
///
/// Returns a `Receiver<RecordBatch>` that yields batches as rayon workers
/// produce them. The caller should iterate until the receiver is exhausted.
/// A background thread is spawned to drive the rayon parallel executor;
/// it is joined via the returned `JoinHandle`.
pub fn execute<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &Path,
) -> TTGenResult<(
    Receiver<RecordBatch>,
    std::thread::JoinHandle<TTGenResult<u64>>,
)> {
    let (receiver, summary_handle) = execute_with_summary(plan, index_path)?;
    let handle = std::thread::spawn(move || {
        let summary = summary_handle
            .join()
            .map_err(|_| TTGenError::SystemError("TT-gen summary thread panicked".to_string()))??;
        Ok(summary.rows_written)
    });
    Ok((receiver, handle))
}

/// Execute a TT-gen plan and return a rich summary payload.
pub fn execute_with_summary<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    index_path: &Path,
) -> TTGenResult<(
    Receiver<RecordBatch>,
    std::thread::JoinHandle<TTGenResult<TTExecutionSummary>>,
)> {
    // Open index
    let metadata_path = index_path.join("index_metadata.json");
    let reader = GraphIndexReader::open(&metadata_path, index_path)
        .map_err(|e| TTGenError::SystemError(format!("opening graph index: {e:#}")))?;

    // Preprocess plan (main thread, once)
    let resolved = resolve_plan(plan, &reader)
        .map_err(|e| TTGenError::InvalidPlan(format!("resolving plan: {e:#}")))?;
    let resolved = Arc::new(resolved);

    let num_entities = resolved.entity_table.node_count();
    let num_threads = rayon::current_num_threads();
    let num_chunks = (num_threads * 8).min(num_entities as usize).max(1);
    tracing::info!(
        entities = num_entities,
        threads = num_threads,
        chunks = num_chunks,
        stages = resolved.stages.len(),
        aliases = resolved.num_aliases,
        "TT-gen execution started"
    );

    // Bounded channel: buffer up to 16 batches (~1MB each) for back-pressure.
    let (sender, receiver) = bounded::<RecordBatch>(16);
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
    let split_is_temporal = plan
        .split
        .as_ref()
        .and_then(|s| s.is_temporal)
        .unwrap_or(false);
    let split_offsets: Option<Vec<Dur<U>>> = plan
        .split
        .as_ref()
        .and_then(|s| s.offsets.clone());
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
        Some(DateOffsetSplitSemantics {
            timeframe: s.timeframe.map(|v| dur_from_unit::<U>(v, unit)),
            anchor_start_offset: dur_from_unit::<U>(s.anchor_start_offset.unwrap_or(0), unit),
        })
    } else {
        None
    };
    let split_ranges: Option<Vec<(Abs<U>, Abs<U>)>> = plan
        .split
        .as_ref()
        .and_then(|s| s.ranges.clone());
    let split_backtesting_timestamp: Option<Abs<U>> = plan
        .split
        .as_ref()
        .and_then(|s| s.backtesting_timestamp);
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

    // Shared concurrent category counts for categorical targets
    let shared_target_counts: Option<Arc<SharedColumnCounts>> = if target_is_categorical {
        Some(Arc::new(SharedColumnCounts::new_hashed()))
    } else {
        None
    };

    // Shared concurrent category counts for ENTITY (always needed for CATEGORY_COUNTS)
    let shared_entity_counts: Arc<SharedColumnCounts> = Arc::new(SharedColumnCounts::new_hashed());

    let handle = std::thread::spawn(move || {
        let shared_tc = shared_target_counts.clone();
        let shared_ec = shared_entity_counts.clone();
        // Parallel execution with fine-grained chunks
        let results: Vec<TTGenResult<(u64, ChunkStats<U>)>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx as u64 * num_entities / num_chunks as u64;
                let end = (chunk_idx as u64 + 1) * num_entities / num_chunks as u64;
                execute_chunk(&resolved, start..end, &sender, shared_tc.as_ref().map(Arc::clone), Arc::clone(&shared_ec))
            })
            .collect();

        // Drop the sender so the receiver sees end-of-stream.
        drop(sender);

        // Check for errors and report
        let mut total_rows = 0u64;
        let mut stats: StatsMerger<U> = StatsMerger::default();
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok((rows, chunk_stats)) => {
                    total_rows += rows;
                    stats.merge(chunk_stats);
                }
                Err(e) => {
                    return Err(TTGenError::SystemError(format!(
                        "thread {} failed: {}",
                        i, e
                    )));
                }
            }
        }

        // Drain shared categorical target counts after all chunks are done
        let target_category_counts = shared_target_counts
            .as_ref()
            .and_then(|s| s.drain_to_map());

        // Drain shared entity category counts after all chunks are done
        let entity_category_counts = shared_entity_counts.drain_to_map();

        tracing::info!(total_rows, "TT-gen execution finished");
        Ok(TTExecutionSummary {
            rows_written: total_rows,
            stats: Some(stats.finalize_with_split_semantics(
                total_rows,
                is_link_prediction,
                has_frequency_col,
                has_include_bit_col,
                split_ratio.as_deref(),
                split_kind.as_deref(),
                split_is_temporal,
                split_offsets.as_deref(),
                date_offset_semantics,
                split_ranges.as_deref(),
                split_backtesting_timestamp,
                split_column.as_deref(),
                manual_train_values.as_deref(),
                manual_val_values.as_deref(),
                manual_test_values.as_deref(),
                target_category_counts,
                target_stype_str.as_str(),
                entity_category_counts,
            )),
        })
    });

    Ok((receiver, handle))
}

// ─── Per-thread execution ────────────────────────────────────────────────────

fn output_mode_from_prediction_type(
    pt: &PredictionType,
    target_dtype: DataType,
    autoregression_dtype: Option<DataType>,
    yoy_dtype: Option<DataType>,
) -> OutputMode {
    match pt {
        PredictionType::Node => OutputMode::Node { target_dtype },
        PredictionType::Link {
            explode: true,
            include_frequency,
        } => OutputMode::LinkExploded {
            include_frequency: *include_frequency,
            target_dtype,
        },
        PredictionType::Link {
            explode: false,
            include_frequency,
        } => OutputMode::LinkArray {
            include_frequency: *include_frequency,
            target_dtype,
        },
        PredictionType::PassThrough => OutputMode::PassThrough,
        PredictionType::Forecasting => OutputMode::Forecasting {
            autoregression_dtype: autoregression_dtype.expect("forecasting missing ar_dtype"),
            yoy_dtype,
        },
    }
}

fn execute_chunk<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    entity_range: std::ops::Range<u64>,
    sender: &Sender<RecordBatch>,
    shared_target_counts: Option<Arc<SharedColumnCounts>>,
    shared_entity_counts: Arc<SharedColumnCounts>,
) -> TTGenResult<(u64, ChunkStats<U>)> {
    let mode = output_mode_from_prediction_type(
        &plan.prediction_type,
        plan.target_dtype.clone(),
        plan.autoregression_dtype.clone(),
        plan.yoy_dtype.clone(),
    );
    let time_col_name = plan
        .entity_table_spec
        .output_time_col_name
        .as_deref()
        .unwrap_or("TIMESTAMP");
    let mut writer = ChannelBatchWriter::new(
        sender.clone(),
        mode,
        plan.lag_timesteps > 0,
        plan.entity_dtype.clone(),
        time_col_name,
        plan.split_projection.as_ref().map(|p| ExtraColumnOutput {
            name: p.output_name.clone(),
            dtype: p.dtype.clone(),
        }),
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
        PredictionType::Link { include_frequency: true, .. }
    );

    // Propagate the resolved target dtype into the stats accumulator so
    // bool-target CATEGORY_COUNTS emits JSON booleans (matches the
    // BooleanArray downcast in tt_executor.rs:compute_train_table_col_stats_payload
    // for the rehydration path; without this the ttgen path silently
    // emits lowercase "true"/"false" strings).
    let mut stats: ChunkStats<U> = ChunkStats {
        target_is_bool_dtype: matches!(plan.target_dtype, DataType::Boolean),
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
    // Always create entity and timestamp accumulators
    stats.entity_hll = Some(HyperLogLog::new());
    stats.entity_welford = Some(WelfordAccumulator::new());
    stats.entity_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_welford = Some(WelfordAccumulator::new());
    stats.timestamp_tdigest = Some(TDigestEstimator::new(200));
    stats.timestamp_hll = Some(HyperLogLog::new());

    // Thread-local categorical target counts (flushed to shared at end)
    let mut local_target_counts: Option<LocalColumnCounts> = if target_is_categorical {
        Some(LocalColumnCounts::with_capacity(5000))
    } else {
        None
    };

    // Thread-local entity category counts (flushed to shared at end)
    let mut local_entity_counts = LocalColumnCounts::with_capacity(5000);

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
            &mut local_entity_counts,
            &mut writer,
            &mut stats,
        )?;
    }

    // Flush local categorical counts to shared
    if let (Some(ref shared), Some(ref mut local)) = (&shared_target_counts, &mut local_target_counts) {
        shared.merge_local_batch(local);
    }
    // Flush local entity counts to shared
    shared_entity_counts.merge_local_batch(&mut local_entity_counts);

    let rows = writer.finish()?;
    stats.num_total_rows = rows;
    Ok((rows, stats))
}

// ─── Per-entity execution ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn execute_entity<U: UnitSystem>(
    plan: &ResolvedPlan<U>,
    entity_node: u64,
    aliases: &mut [Option<Value>],
    link_frequencies: &mut Vec<i64>,
    link_split_values: &mut Vec<Value>,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
    local_entity_counts: &mut LocalColumnCounts,
    writer: &mut ChannelBatchWriter<U>,
    stats: &mut ChunkStats<U>,
) -> TTGenResult<()> {
    // 0. Look up the entity PK value for output
    let entity_pk = read_column_value(&plan.entity_table, entity_node, plan.entity_pk_col_idx)
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
        let default_ts = plan.entity_table_spec.default_time_value.unwrap_or(Abs::<U>::from_raw(0));
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
                local_entity_counts,
                writer,
                stats,
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
                local_entity_counts,
                writer,
                stats,
            )?;
        }

        // Clear aliases for next timestamp
        aliases.iter_mut().for_each(|a| *a = None);
    }

    Ok(())
}

/// Write the target value(s) for one (entity, timestamp) pair.
#[allow(clippy::too_many_arguments)]
fn write_target<U: UnitSystem>(
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
    local_entity_counts: &mut LocalColumnCounts,
    writer: &mut ChannelBatchWriter<U>,
    stats: &mut ChunkStats<U>,
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
            writer.write_row_pass_through(entity_pk.clone(), ts, extra_value, entity_include_bit)?;
        }
        PredictionType::Forecasting => {
            let mut autoregression = Vec::new();
            if let Some(idx) = plan.autoregression_alias_idx {
                if let Some(Value::List(l)) = &aliases[idx] {
                    autoregression = l.clone();
                }
            }
            let mut yoy = None;
            if let Some(idx) = plan.yoy_alias_idx {
                if let Some(Value::List(l)) = &aliases[idx] {
                    yoy = Some(l.clone());
                }
            }
            writer.write_row_forecasting(
                entity_pk.clone(),
                ts,
                autoregression,
                yoy,
                extra_value,
                entity_include_bit,
            )?;
        }
        PredictionType::Node | PredictionType::Link { .. } => {
            let target = match &aliases[plan.target_alias_idx] {
                Some(v) => v,
                None => return Ok(()),
            };

            if matches!(plan.prediction_type, PredictionType::Node)
                && plan.target_dtype == DataType::Utf8
            {
                if let Value::Str(s) = target {
                    if s.is_empty() {
                        return Ok(());
                    }
                }
            }

            match &plan.prediction_type {
                PredictionType::Node => {
                    writer.write_row(entity_pk.clone(), ts, target, extra_value, entity_include_bit)?;
                    update_exact_stats(
                        stats,
                        entity_pk,
                        ts,
                        target,
                        None,
                        target_is_numerical,
                        target_is_categorical,
                        target_stype,
                        local_target_counts,
                        local_entity_counts,
                    );
                    if entity_include_bit {
                        stats.include_true_count += 1;
                        stats.num_train_rows += 1;
                    } else {
                        stats.include_false_count += 1;
                    }
                }
                PredictionType::Link {
                    explode: true,
                    include_frequency,
                } => {
                    if let Value::List(targets) = target {
                        for (i, val) in targets.iter().enumerate() {
                            let freq = if *include_frequency {
                                link_frequencies.get(i).copied().unwrap_or(1)
                            } else {
                                0
                            };
                            writer.write_row_exploded(
                                entity_pk.clone(),
                                ts,
                                val.clone(),
                                freq,
                                extra_value,
                                entity_include_bit,
                            )?;
                            update_exact_stats(stats, entity_pk, ts, val, Some(freq), target_is_numerical, target_is_categorical, target_stype, local_target_counts, local_entity_counts);
                            if entity_include_bit {
                                stats.include_true_count += 1;
                                stats.num_links += 1;
                                stats.num_train_rows += 1;
                            } else {
                                stats.include_false_count += 1;
                            }
                        }
                    }
                }
                PredictionType::Link {
                    explode: false,
                    include_frequency,
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
                            // Index 0: full row + target observation.
                            update_exact_stats(stats, entity_pk, ts, first_target, first_freq, target_is_numerical, target_is_categorical, target_stype, local_target_counts, local_entity_counts);
                            // Index 1+: target-only observation. Without this,
                            // multicat `MULTI_CATEGORY_COUNTS` and `target_hll`
                            // miss every token after the first under the
                            // array-output writer arm (kumo#28400 codex review).
                            for (idx, tgt) in targets.iter().enumerate().skip(1) {
                                let freq = if *include_frequency {
                                    freqs.get(idx).copied()
                                } else {
                                    None
                                };
                                observe_target_value(
                                    stats,
                                    tgt,
                                    freq,
                                    target_is_numerical,
                                    target_is_categorical,
                                    target_stype,
                                    local_target_counts,
                                );
                            }
                        } else {
                            stats.entities.insert(hash_value(entity_pk));
                            stats.timestamps.insert(ts);
                            *stats.timestamp_counts.entry(ts).or_insert(0) += 1;
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
                        }

                        if entity_include_bit {
                            stats.include_true_count += 1;
                            stats.num_links += targets.len() as u64;
                            stats.num_train_rows += 1;
                        } else {
                            stats.include_false_count += 1;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_exact_stats<U: UnitSystem>(
    stats: &mut ChunkStats<U>,
    entity_pk: &Value,
    ts: Abs<U>,
    target: &Value,
    frequency: Option<i64>,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
    local_entity_counts: &mut LocalColumnCounts,
) {
    stats.entities.insert(hash_value(entity_pk));
    stats.timestamps.insert(ts);
    *stats.timestamp_counts.entry(ts).or_insert(0) += 1;

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

    // Observe entity value into HLL for cardinality
    if let Some(ref mut h) = stats.entity_hll {
        match entity_pk {
            Value::Int(i) => h.observe_i64(*i),
            Value::Str(s) => h.observe_bytes(s.as_bytes()),
            Value::Float(f) => h.observe_f64(*f),
            _ => {}
        }
    }

    // Observe entity value into local category counts (for CATEGORY_COUNTS stat)
    {
        let entity_str = value_to_string(entity_pk);
        let entity_hash = hash_value(entity_pk);
        local_entity_counts.observe(entity_hash, &entity_str);
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

    observe_target_value(
        stats,
        target,
        frequency,
        target_is_numerical,
        target_is_categorical,
        target_stype,
        local_target_counts,
    );

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

/// Observe one target value (and its frequency, if any) into the per-target
/// stat aggregates: targets-set hash, target_min/max, frequency_min/max,
/// numerical Welford/tdigest/HLL, categorical HLL + local category counts.
///
/// Called once per target by `update_exact_stats` for the row's primary
/// target, and additionally per-element by the link-pred array-output writer
/// arm so `MULTI_CATEGORY_COUNTS` and `target_hll` see every distinct token
/// in the array, not just index 0.
///
/// Multicat shapes after the per-token observe fix (kumo#28400):
///   - `Value::List(tokens)` — node-pred multilabel-classification with a
///     multicat source (FIRST/LAST/Route A/Node-pred LIST_DISTINCT). Iterate
///     items directly; do NOT call `value_to_string` on the list (it
///     yields the literal `"<list>"`, collapsing every row into one
///     synthetic category).
///   - `Value::Str(tok)` — link-pred-explode arm: each row is one token.
///     The legacy `'|'`-split path is preserved for compatibility with
///     pre-fix callers that may still emit a pipe-joined whole-cell string;
///     for single-token Value::Str the split is a no-op.
fn observe_target_value<U: UnitSystem>(
    stats: &mut ChunkStats<U>,
    target: &Value,
    frequency: Option<i64>,
    target_is_numerical: bool,
    target_is_categorical: bool,
    target_stype: &str,
    local_target_counts: &mut Option<LocalColumnCounts>,
) {
    stats.targets.insert(hash_value(target));

    if let Some(v) = value_to_i64_opt(target) {
        stats.target_min = min_opt_i64(stats.target_min, v);
        stats.target_max = max_opt_i64(stats.target_max, v);
    }

    if let Some(freq) = frequency {
        stats.frequency_min = min_opt_i64(stats.frequency_min, freq);
        stats.frequency_max = max_opt_i64(stats.frequency_max, freq);
    }

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
        if let Some(ref mut h) = stats.target_hll {
            match target {
                Value::Int(i) => h.observe_i64(*i),
                Value::Float(f) => h.observe_f64(*f),
                _ => {}
            }
        }
    }

    if target_is_categorical {
        if target_stype == "multicategorical" {
            match target {
                Value::List(items) => {
                    for item in items {
                        let tok = value_to_string(item);
                        let hash = HyperLogLog::hash_bytes(tok.as_bytes());
                        if let Some(ref mut h) = stats.target_hll {
                            h.observe_hash(hash);
                        }
                        if let Some(ref mut local) = local_target_counts {
                            local.observe(hash, &tok);
                        }
                    }
                }
                _ => {
                    let target_str = value_to_string(target);
                    for_each_split_ascii_byte(&target_str, b'|', |tok| {
                        let tok = tok.trim();
                        let hash = HyperLogLog::hash_bytes(tok.as_bytes());
                        if let Some(ref mut h) = stats.target_hll {
                            h.observe_hash(hash);
                        }
                        if let Some(ref mut local) = local_target_counts {
                            local.observe(hash, tok);
                        }
                    });
                }
            }
        } else {
            let target_hash = hash_value(target);
            let target_str = value_to_string(target);
            if let Some(ref mut local) = local_target_counts {
                local.observe(target_hash, &target_str);
            }
            if let Some(ref mut h) = stats.target_hll {
                h.observe_i64(target_hash as i64);
            }
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
                    // Route A direct-target read site (kumo#28400). For
                    // multicat sources, emit `Value::List(tokens)` so the
                    // downstream alias consumer (the writer) sees the row's
                    // flat token set. Non-multicat path is unchanged.
                    //
                    // NULL handling: distinguish "source NULL" (return None,
                    // skip the row) from "source present, list of zero
                    // tokens" (return Some(Value::List(vec![]))). The dispatch
                    // outcome reports SourceNull when the row's source data
                    // is missing; without this check, NULL rows mis-render as
                    // empty-target training rows and skew downstream stats.
                    let is_multicat = col.target_semantic_type.as_deref()
                        == Some("multicategorical");
                    if is_multicat {
                        let mut tokens: Vec<Value> = Vec::new();
                        let outcome = dispatch_multicat_source(
                            &col.table,
                            entity_node,
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
                            entity_node,
                            col.col_idx,
                        ) {
                            Some(Some(v)) => Some(v),
                            Some(None) => None,
                            None => None,
                        }
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
                    let mut results = Vec::with_capacity(auto.lag_offsets.len());
                    let base_ts = observation_ts.unwrap_or(Abs::<U>::from_raw(0));

                    if let ResolvedOp::Aggregate(ref agg) = *auto.target_op {
                        for &offset in &auto.lag_offsets {
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
                            match val_opt {
                                Some(v) => results.push(v),
                                None => results.push(Value::Float(f64::NAN)),
                            }
                        }
                        Some(Value::List(results))
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
                let lo = encoder.encode_upper((obs + *offset_lo).raw());
                let hi = encoder.encode_upper((obs + *offset_hi).raw());
                (lo, hi)
            }
            ResolvedWindow::Calendar {
                lo_months,
                hi_months,
                encoder,
            } => {
                let lo_abs = add_calendar_months(obs, *lo_months);
                let hi_abs = add_calendar_months(obs, *hi_months);
                let lo = encoder.encode_upper(lo_abs.raw());
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
                let lo = encoder.encode_upper((obs + *offset_lo).raw());
                let hi = encoder.encode_upper((obs + *offset_hi).raw());
                (lo, hi)
            }
            ResolvedWindow::Calendar {
                lo_months,
                hi_months,
                encoder,
            } => {
                let lo_abs = add_calendar_months(obs, *lo_months);
                let hi_abs = add_calendar_months(obs, *hi_months);
                let lo = encoder.encode_upper(lo_abs.raw());
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
                    let _ = dispatch_multicat_source(
                        dst_table,
                        dst,
                        col_idx,
                        true,
                        |tok| {
                            let h = match tok {
                                MulticatToken::Str(s) => hash_str(s),
                                MulticatToken::Int(n) => hash_i64(n),
                                MulticatToken::Legacy(ref v) => hash_value(v),
                            };
                            set.insert(h);
                            true
                        },
                    );
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
            let temporal_obs_ts = match observation_ts {
                Some(ts) if agg.target_temporal_create_col_idx.is_some()
                    || agg.target_temporal_end_col_idx.is_some() => Some(ts.raw()),
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
                    let _ = dispatch_multicat_source(
                        dst_table,
                        dst,
                        col_idx,
                        true,
                        |tok| {
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
                        },
                    );
                    return values.len() < limit;
                }

                // Legacy whole-cell path (FK ListDistinct, scalar ListDistinct).
                if let Some(val) = read_column_value(dst_table, dst, col_idx) {
                    // Skip target values whose FK doesn't exist in the
                    // destination table, or whose target entity is not
                    // temporally valid at the observation timestamp
                    // (e.g., FIDO added to catalog after anchor time).
                    if let Some(pk_table) = pk_filter {
                        if let Some(obs_ts) = temporal_obs_ts {
                            if !pk_table.contains_pk_value_temporal(
                                &val,
                                obs_ts,
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
                        let _ = dispatch_multicat_source(
                            dst_table,
                            dst,
                            col_idx,
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
                        let _ = dispatch_multicat_source(
                            dst_table,
                            dst,
                            col_idx,
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
    use diskgraph_core::time::Seconds;

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
