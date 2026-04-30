//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` that returns a
//! `TTPlanIterator` yielding `pyarrow.RecordBatch` objects one at a time.
#![allow(
    clippy::inconsistent_digit_grouping,
    clippy::manual_contains,
    clippy::needless_late_init,
    clippy::needless_question_mark,
    clippy::needless_range_loop,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::while_let_on_iterator
)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;
use tracing::info;

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

use crate::split_assignments;
use arrow::compute::take;
use arrow::pyarrow::ToPyArrow;
use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
    TimestampNanosecondBuilder,
};
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, ListArray, RecordBatch, StringArray,
    TimestampNanosecondArray, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use crossbeam_channel::Receiver;
use diskgraph_core::time::{abs_from_unit, Nanoseconds, Seconds, TimeUnit};
use diskgraph_core::tt_builder::DiskGraphTTPlan;
use diskgraph_train_table::errors::TTGenResult;
use diskgraph_train_table::ttgen_stats::TTExecutionSummary;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3::Py;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};

#[pyclass]
pub struct TTPlanIterator {
    receiver: Option<Receiver<RecordBatch>>,
    handle: Option<JoinHandle<TTGenResult<TTExecutionSummary>>>,
    summary: Option<TTExecutionSummary>,
}

impl TTPlanIterator {
    fn finish_execution(&mut self) -> TTGenResult<()> {
        if self.summary.is_some() {
            return Ok(());
        }

        self.receiver = None;
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(Ok(summary)) => {
                    self.summary = Some(summary);
                    Ok(())
                }
                Ok(Err(e)) => Err(e),
                Err(_) => Err(diskgraph_train_table::errors::TTGenError::SystemError(
                    "TT-gen executor thread panicked".to_string(),
                )),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for TTPlanIterator {
    fn drop(&mut self) {
        // Drop the receiver first to unblock any workers stuck on send().
        self.receiver = None;
        // Join the handle with GIL released to avoid deadlock with pyo3-log.
        // The background thread uses tracing::info! which routes through
        // pyo3-log and needs the GIL. Drop is called with GIL held (Python GC),
        // so we must release it before joining.
        if let Some(handle) = self.handle.take() {
            let _ = pyo3::Python::attach(|py| py.detach(|| handle.join()));
        }
    }
}

#[pymethods]
impl TTPlanIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let receiver = match &self.receiver {
            Some(r) => r.clone(),
            None => return Err(PyStopIteration::new_err(())),
        };

        let result = py.detach(|| receiver.recv());

        match result {
            Ok(batch) => batch.to_pyarrow(py).map(|b| b.unbind()),
            Err(_) => {
                self.receiver = None;

                // Join the background thread with GIL released.
                // The bg thread uses tracing::info! which routes through
                // pyo3-log and needs the GIL. If we hold the GIL while
                // joining, the bg thread can't log and we deadlock.
                if self.summary.is_none() {
                    if let Some(handle) = self.handle.take() {
                        let join_result = py.detach(|| handle.join());
                        match join_result {
                            Ok(Ok(summary)) => { self.summary = Some(summary); }
                            Ok(Err(e)) => {
                                return Err(crate::map_ttgen_err(e));
                            }
                            Err(_) => {
                                return Err(crate::DiskgraphSystemError::new_err(
                                    "TT-gen executor thread panicked"));
                            }
                        }
                    }
                }

                Err(PyStopIteration::new_err(()))
            }
        }
    }

    /// Returns the final execution summary.
    ///
    /// If called before stream exhaustion, this forces finalization and blocks
    /// until all workers complete.
    fn execution_summary(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.finish_execution().map_err(crate::map_ttgen_err)?;

        let out = PyDict::new(py);
        if let Some(summary) = &self.summary {
            out.set_item("rows_written", summary.rows_written)?;
            // Backward-compatible top-level aliases expected by Kumo.
            out.set_item("output_rows", summary.rows_written)?;
            if let Some(stats) = &summary.stats {
                out.set_item("pre_target_drop_rows", stats.num_total_rows)?;
                let stats_dict = PyDict::new(py);
                stats_dict.set_item("version", stats.version)?;
                stats_dict.set_item("num_total_rows", stats.num_total_rows)?;
                stats_dict.set_item("num_train_rows", stats.num_train_rows)?;
                stats_dict.set_item("num_entities", stats.num_entities)?;
                stats_dict.set_item("num_targets", stats.num_targets)?;
                stats_dict.set_item("num_links", stats.num_links)?;

                let exact_dict = PyDict::new(py);
                let entity = PyDict::new(py);
                entity.set_item("count", stats.exact_stats.entity.count)?;
                entity.set_item("null_count", stats.exact_stats.entity.null_count)?;
                entity.set_item("num_unique", stats.exact_stats.entity.num_unique)?;
                entity.set_item("min", stats.exact_stats.entity.min)?;
                entity.set_item("max", stats.exact_stats.entity.max)?;
                exact_dict.set_item("entity", entity)?;

                let timestamp = PyDict::new(py);
                timestamp.set_item("count", stats.exact_stats.timestamp.count)?;
                timestamp.set_item("null_count", stats.exact_stats.timestamp.null_count)?;
                timestamp.set_item("num_unique", stats.exact_stats.timestamp.num_unique)?;
                timestamp.set_item("min", stats.exact_stats.timestamp.min)?;
                timestamp.set_item("max", stats.exact_stats.timestamp.max)?;
                exact_dict.set_item("timestamp", timestamp)?;

                let target = PyDict::new(py);
                target.set_item("count", stats.exact_stats.target.count)?;
                target.set_item("null_count", stats.exact_stats.target.null_count)?;
                target.set_item("num_unique", stats.exact_stats.target.num_unique)?;
                target.set_item("min", stats.exact_stats.target.min)?;
                target.set_item("max", stats.exact_stats.target.max)?;
                exact_dict.set_item("target", target)?;

                if let Some(freq) = &stats.exact_stats.frequency {
                    let frequency = PyDict::new(py);
                    frequency.set_item("count", freq.count)?;
                    frequency.set_item("null_count", freq.null_count)?;
                    frequency.set_item("num_unique", freq.num_unique)?;
                    frequency.set_item("min", freq.min)?;
                    frequency.set_item("max", freq.max)?;
                    exact_dict.set_item("frequency", frequency)?;
                } else {
                    exact_dict.set_item("frequency", py.None())?;
                }

                if let Some(bit) = &stats.exact_stats.entity_include_bit {
                    let include = PyDict::new(py);
                    include.set_item("count", bit.count)?;
                    include.set_item("null_count", bit.null_count)?;
                    include.set_item("true_count", bit.true_count)?;
                    include.set_item("false_count", bit.false_count)?;
                    exact_dict.set_item("entity_include_bit", include)?;
                } else {
                    exact_dict.set_item("entity_include_bit", py.None())?;
                }

                stats_dict.set_item("exact_stats", exact_dict)?;

                if let Some(split) = &stats.split_payload {
                    let split_dict = PyDict::new(py);
                    split_dict.set_item("version", split.version)?;
                    split_dict.set_item("status", split.status.as_str())?;
                    split_dict
                        .set_item("supports_native_execution", split.supports_native_execution)?;
                    split_dict.set_item("split_kind", split.split_kind.as_deref())?;
                    split_dict.set_item("derivation_method", split.derivation_method.as_deref())?;
                    split_dict.set_item("num_total_rows", split.num_total_rows)?;
                    if let Some(art) = &split.artifacts_v3 {
                        let art_dict = PyDict::new(py);
                        art_dict.set_item("strategy", art.strategy.as_str())?;
                        art_dict.set_item("split_kind", art.split_kind.as_deref())?;
                        art_dict.set_item("ratio", art.ratio.clone())?;
                        art_dict.set_item("offsets", art.offsets.clone())?;
                        art_dict.set_item("ranges", art.ranges.clone())?;
                        art_dict.set_item("backtesting_timestamp", art.backtesting_timestamp)?;
                        art_dict.set_item("split_column", art.split_column.as_deref())?;
                        art_dict
                            .set_item("manual_train_values", art.manual_train_values.clone())?;
                        art_dict.set_item("manual_val_values", art.manual_val_values.clone())?;
                        art_dict.set_item("manual_test_values", art.manual_test_values.clone())?;
                        art_dict.set_item(
                            "authoritative_supported_family",
                            art.authoritative_supported_family,
                        )?;
                        if let Some(assignments) = &art.timestamp_stage_assignments {
                            let py_assignments = pyo3::types::PyList::empty(py);
                            for a in assignments {
                                let item = PyDict::new(py);
                                item.set_item("timestamp", a.timestamp)?;
                                item.set_item("stage", a.stage.as_str())?;
                                py_assignments.append(item)?;
                            }
                            art_dict.set_item("timestamp_stage_assignments", py_assignments)?;
                        } else {
                            art_dict.set_item("timestamp_stage_assignments", py.None())?;
                        }
                        if let Some(sizes) = &art.index_stage_sizes {
                            let sizes_dict = PyDict::new(py);
                            sizes_dict.set_item("train", sizes.train)?;
                            sizes_dict.set_item("val", sizes.val)?;
                            sizes_dict.set_item("test", sizes.test)?;
                            art_dict.set_item("index_stage_sizes", sizes_dict)?;
                        } else {
                            art_dict.set_item("index_stage_sizes", py.None())?;
                        }
                        split_dict.set_item("artifacts", art_dict.clone())?;
                        split_dict.set_item("artifacts_v3", art_dict)?;
                    } else {
                        split_dict.set_item("artifacts", py.None())?;
                        split_dict.set_item("artifacts_v3", py.None())?;
                    }
                    if let Some(mask_counts) = &split.mask_counts {
                        let counts = PyDict::new(py);
                        counts.set_item("train", mask_counts.train)?;
                        counts.set_item("val", mask_counts.val)?;
                        counts.set_item("test", mask_counts.test)?;
                        split_dict.set_item("mask_counts", counts)?;
                    } else {
                        split_dict.set_item("mask_counts", py.None())?;
                    }
                    stats_dict.set_item("split_payload", split_dict)?;
                } else {
                    stats_dict.set_item("split_payload", py.None())?;
                }

                // Per-timestamp row counts for accurate stats_per_timeframe
                if !stats.timestamp_counts.is_empty() {
                    let tc_dict = PyDict::new(py);
                    for (ts, count) in &stats.timestamp_counts {
                        tc_dict.set_item(ts.to_string(), *count)?;
                    }
                    stats_dict.set_item("timestamp_counts", tc_dict)?;
                }

                // Inline col stats for encoder inference (computed during execution)
                if let Some(col_stats) = &stats.parsed_train_table_col_stats {
                    let col_stats_json = serde_json::to_string(col_stats)
                        .map_err(|e| crate::DiskgraphSystemError::new_err(e.to_string()))?;
                    stats_dict.set_item("parsed_train_table_col_stats_json", col_stats_json)?;
                }

                out.set_item("stats", stats_dict)?;
            } else {
                return Err(crate::DiskgraphSystemError::new_err(
                    "TT-gen summary missing stats payload; cannot produce pre_target_drop_rows",
                ));
            }
        } else {
            return Err(crate::DiskgraphSystemError::new_err(
                "TT-gen execution summary is unavailable",
            ));
        }

        Ok(out.into())
    }
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all, fields(index_path = %index_path))]
pub fn execute_tt_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
) -> PyResult<TTPlanIterator> {
    info!("Legacy execute_tt_plan started");
    let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(plan_json)
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();

    let (receiver, handle) = py
        .detach(|| {
            diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))
        })
        .map_err(crate::map_ttgen_err)?;

    Ok(TTPlanIterator {
        receiver: Some(receiver),
        handle: Some(handle),
        summary: None,
    })
}

fn append_arange_column(batch: RecordBatch, row_offset: i64) -> anyhow::Result<RecordBatch> {
    let num_rows = batch.num_rows() as i64;
    let arange = Int64Array::from_iter_values(row_offset..(row_offset + num_rows));

    // Drop any existing arange column so the caller's fresh arange replaces
    // it (e.g., when reading from the scratch parquet which already carries
    // an arange from the consumer-loop write, we need to re-index post-DS).
    let (fields, columns): (Vec<_>, Vec<_>) = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns().iter())
        .filter(|(f, _)| f.name() != "arange")
        .map(|(f, c)| (f.as_ref().clone(), c.clone()))
        .unzip();

    let mut fields = fields;
    fields.push(Field::new("arange", DataType::Int64, false));
    let mut columns = columns;
    columns.push(Arc::new(arange));

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| anyhow::anyhow!("failed to append arange column: {e}"))
}

/// Distribute `total` items into `buckets` using largest-remainder method.
/// Guarantees the result sums to exactly `total`.
#[allow(dead_code)]
fn largest_remainder_distribute(total: u64, buckets: usize) -> Vec<u64> {
    if buckets == 0 {
        return Vec::new();
    }
    if total == 0 {
        return vec![0; buckets];
    }
    let ideal: Vec<f64> = (0..buckets)
        .map(|_| total as f64 / buckets as f64)
        .collect();
    let mut sizes: Vec<u64> = ideal.iter().map(|v| v.floor() as u64).collect();
    let assigned: u64 = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);

    let mut order: Vec<usize> = (0..buckets).collect();
    order.sort_by(|&a, &b| {
        let ra = ideal[a] - sizes[a] as f64;
        let rb = ideal[b] - sizes[b] as f64;
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut k = 0usize;
    while remaining > 0 {
        sizes[order[k]] += 1;
        remaining -= 1;
        k += 1;
        if k == order.len() {
            k = 0;
        }
    }
    sizes
}

fn take_batch_rows(batch: &RecordBatch, local_indices: &[u64]) -> anyhow::Result<RecordBatch> {
    if local_indices.is_empty() {
        return Ok(batch.slice(0, 0));
    }
    let idx_arr = Int64Array::from_iter_values(local_indices.iter().map(|v| *v as i64));
    let mut cols = Vec::with_capacity(batch.num_columns());
    for c in batch.columns() {
        cols.push(take(c.as_ref(), &idx_arr, None)?);
    }
    RecordBatch::try_new(batch.schema(), cols)
        .map_err(|e| anyhow::anyhow!("failed to take batch rows: {e}"))
}

fn write_batches_with_arange(
    path: &str,
    batches: &[RecordBatch],
    selected_global_indices: Option<&[usize]>,
) -> anyhow::Result<u64> {
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer: Option<ArrowWriter<File>> = None;
    let mut row_offset: i64 = 0;
    let mut global_base: usize = 0;

    for batch in batches {
        let to_write = if let Some(selected) = selected_global_indices {
            let end = global_base + batch.num_rows();
            let local = selected
                .iter()
                .filter_map(|g| {
                    if *g >= global_base && *g < end {
                        Some((*g - global_base) as u64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            global_base = end;
            if local.is_empty() {
                continue;
            }
            take_batch_rows(batch, &local)?
        } else {
            global_base += batch.num_rows();
            batch.clone()
        };

        let with_idx = append_arange_column(to_write, row_offset)?;
        row_offset += with_idx.num_rows() as i64;

        if writer.is_none() {
            writer = Some(ArrowWriter::try_new(
                file.try_clone()?,
                with_idx.schema(),
                Some(props.clone()),
            )?);
        }
        if let Some(w) = writer.as_mut() {
            w.write(&with_idx)?;
        }
    }

    if row_offset == 0 {
        return Err(anyhow::anyhow!("zero rows written"));
    }
    if let Some(w) = writer.take() {
        w.close()?;
    }
    Ok(row_offset as u64)
}

#[allow(dead_code)]
fn materialize_selected_batches(
    batches: &[RecordBatch],
    selected_global_indices: Option<&[usize]>,
) -> anyhow::Result<Vec<RecordBatch>> {
    if selected_global_indices.is_none() {
        return Ok(batches.to_vec());
    }

    let selected = selected_global_indices.unwrap();
    let mut out = Vec::new();
    let mut global_base = 0usize;
    for batch in batches {
        let end = global_base + batch.num_rows();
        let local = selected
            .iter()
            .filter_map(|g| {
                if *g >= global_base && *g < end {
                    Some((*g - global_base) as u64)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        global_base = end;
        if local.is_empty() {
            continue;
        }
        out.push(take_batch_rows(batch, &local)?);
    }
    Ok(out)
}

fn row_count_of_batches(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Maximum TT rows allowed without downsampling. Beyond this the
/// in-memory entity/target string vectors become prohibitively large.
/// This is a transitional safety valve — once the streaming refactor
/// (step 2+3) lands, this limit can be raised or removed.
const MAX_TT_ROWS_WITHOUT_DOWNSAMPLING: u64 = 200_000_000;

// ─── Scratch parquet writer/reader ──────────────────────────────────────

/// Write RecordBatches to a scratch parquet file incrementally.
/// Each batch becomes one row group for efficient sequential read-back.
struct ScratchParquetWriter {
    writer: Option<ArrowWriter<File>>,
    path: String,
    props: WriterProperties,
    rows_written: u64,
}

impl ScratchParquetWriter {
    fn new(path: &str) -> Self {
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        Self {
            writer: None,
            path: path.to_owned(),
            props,
            rows_written: 0,
        }
    }

    /// Write a batch with an arange column appended. Lazily creates the
    /// writer on the first batch (schema inferred from data, not upfront).
    fn write_batch(&mut self, batch: &RecordBatch) -> anyhow::Result<()> {
        let with_arange = append_arange_column(batch.clone(), self.rows_written as i64)?;
        if self.writer.is_none() {
            let file = File::create(&self.path)?;
            self.writer = Some(ArrowWriter::try_new(
                file,
                with_arange.schema(),
                Some(self.props.clone()),
            )?);
        }
        self.rows_written += batch.num_rows() as u64;
        self.writer.as_mut().unwrap().write(&with_arange)?;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<u64> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        Ok(self.rows_written)
    }
}

/// Guard that deletes a scratch file on drop unless disarmed.
/// Ensures cleanup on all error paths (panics, early returns via `?`).
struct ScratchFileGuard {
    path: String,
    disarmed: bool,
}

impl ScratchFileGuard {
    fn new(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            disarmed: false,
        }
    }

    /// Disarm the guard (e.g., after rename to pre_downsample artifact).
    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl Drop for ScratchFileGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Read batches back from a scratch parquet, optionally filtering rows
/// by entity membership and/or capping at max_rows.
///
/// When `sampled_entities` is Some, only rows whose ENTITY column value
/// is in the set are kept. When None, all rows pass through.
/// Reading stops once `max_rows` kept rows have been collected.
fn read_scratch_filtered(
    path: &str,
    sampled_entities: Option<&std::collections::HashSet<String>>,
    max_rows: usize,
) -> anyhow::Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut out = Vec::new();
    let mut total_kept = 0usize;

    for batch_result in reader {
        let batch = batch_result?;

        let to_push = if let Some(sampled) = sampled_entities {
            // Filter rows by entity membership (O(1) HashSet lookup per row)
            let entities = extract_entity_strings(&batch)?;
            let mask: BooleanArray = entities
                .iter()
                .map(|e| Some(sampled.contains(e)))
                .collect();
            arrow::compute::filter_record_batch(&batch, &mask)?
        } else {
            batch
        };

        let rows = to_push.num_rows();
        if rows > 0 {
            let remaining = max_rows.saturating_sub(total_kept);
            if rows <= remaining {
                total_kept += rows;
                out.push(to_push);
            } else {
                // Truncate final batch to not exceed max_rows
                out.push(to_push.slice(0, remaining));
                break;
            }
        }
    }
    Ok(out)
}

#[derive(Clone)]
enum ScalarCell {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Ts(i64),
    Null,
}

fn scalar_from_array_at(arr: &dyn Array, idx: usize, ctx: &str) -> anyhow::Result<ScalarCell> {
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok(if a.is_null(idx) {
            ScalarCell::Null
        } else {
            ScalarCell::Int(a.value(idx))
        });
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok(if a.is_null(idx) {
            ScalarCell::Null
        } else {
            ScalarCell::Float(a.value(idx))
        });
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok(if a.is_null(idx) {
            ScalarCell::Null
        } else {
            ScalarCell::Str(a.value(idx).to_string())
        });
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok(if a.is_null(idx) {
            ScalarCell::Null
        } else {
            ScalarCell::Bool(a.value(idx))
        });
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(if a.is_null(idx) {
            ScalarCell::Null
        } else {
            ScalarCell::Ts(a.value(idx))
        });
    }
    Err(anyhow::anyhow!(
        "RandomLinkSplit unsupported scalar dtype for {ctx}: {:?}",
        arr.data_type()
    ))
}

fn build_scalar_array_from_cells(
    dtype: &DataType,
    cells: &[ScalarCell],
    ctx: &str,
) -> anyhow::Result<Arc<dyn Array>> {
    match dtype {
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(cells.len());
            for c in cells {
                match c {
                    ScalarCell::Int(v) => b.append_value(*v),
                    ScalarCell::Null => b.append_null(),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected Int64"
                        ))
                    }
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
                    _ => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected Float64"
                        ))
                    }
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
                    _ => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected Utf8"
                        ))
                    }
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
                    _ => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected Boolean"
                        ))
                    }
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
                    _ => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected Timestamp(ns)"
                        ))
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(anyhow::anyhow!(
            "RandomLinkSplit unsupported scalar build dtype for {ctx}: {other:?}"
        )),
    }
}

fn build_list_array_from_cells(
    list_dtype: &DataType,
    rows: &[Vec<ScalarCell>],
    ctx: &str,
) -> anyhow::Result<Arc<dyn Array>> {
    let DataType::List(field) = list_dtype else {
        return Err(anyhow::anyhow!(
            "RandomLinkSplit expected List dtype for {ctx}"
        ));
    };
    let item_dtype = field.data_type();

    match item_dtype {
        DataType::Int64 => {
            let mut b = ListBuilder::new(Int64Builder::new());
            for row in rows {
                for c in row {
                    match c {
                        ScalarCell::Int(v) => b.values().append_value(*v),
                        ScalarCell::Null => b.values().append_null(),
                        _ => {
                            return Err(anyhow::anyhow!(
                                "RandomLinkSplit type mismatch for {ctx}: expected List<Int64>"
                            ))
                        }
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
                        _ => {
                            return Err(anyhow::anyhow!(
                                "RandomLinkSplit type mismatch for {ctx}: expected List<Float64>"
                            ))
                        }
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
                        _ => {
                            return Err(anyhow::anyhow!(
                                "RandomLinkSplit type mismatch for {ctx}: expected List<Utf8>"
                            ))
                        }
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
                        _ => {
                            return Err(anyhow::anyhow!(
                                "RandomLinkSplit type mismatch for {ctx}: expected List<Boolean>"
                            ))
                        }
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
                        _ => {
                            return Err(anyhow::anyhow!(
                            "RandomLinkSplit type mismatch for {ctx}: expected List<Timestamp(ns)>"
                        ))
                        }
                    }
                }
                b.append(true);
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(anyhow::anyhow!(
            "RandomLinkSplit unsupported target list item dtype for {ctx}: {other:?}"
        )),
    }
}

fn is_link_array_batches(batches: &[RecordBatch]) -> bool {
    let Some(first) = batches.first() else {
        return false;
    };
    if let Ok(idx) = first.schema().index_of("TARGET") {
        matches!(first.schema().field(idx).data_type(), DataType::List(_))
    } else {
        false
    }
}

const STAGE_ID_COL_NAME: &str = "__internal_kumo_split_id_col__";

struct ExplodedLinkData {
    row_targets: Vec<Vec<ScalarCell>>,
    row_freqs: Vec<Vec<i64>>,
    row_scalar_cols: Vec<Vec<ScalarCell>>,
    exploded_refs: Vec<(usize, usize)>,
    split_col_values: Vec<String>,
    schema: arrow_schema::SchemaRef,
    target_idx: usize,
    freq_idx: Option<usize>,
    split_col_idx: Option<usize>,
}

fn collect_link_array_batches(
    batches: &[RecordBatch],
    split_col: Option<&str>,
    kind_label: &str,
) -> anyhow::Result<ExplodedLinkData> {
    let first = batches.first().ok_or_else(|| {
        anyhow::anyhow!("{kind_label} cannot transform empty LinkArray batches")
    })?;
    let schema = first.schema();

    let target_idx = schema
        .index_of("TARGET")
        .map_err(|_| anyhow::anyhow!("{kind_label} requires TARGET column"))?;
    let freq_idx = schema.index_of("__kumo_internal_frequency_column__").ok();
    if !matches!(schema.field(target_idx).data_type(), DataType::List(_)) {
        return Err(anyhow::anyhow!(
            "{kind_label} LinkArray transform requires TARGET as List type"
        ));
    }

    let split_col_idx = if let Some(name) = split_col {
        let idx = schema
            .index_of(name)
            .map_err(|_| anyhow::anyhow!("{kind_label} requires split column {name:?}"))?;
        if !matches!(schema.field(idx).data_type(), DataType::List(_)) {
            return Err(anyhow::anyhow!(
                "{kind_label} requires split column {name:?} to be List type"
            ));
        }
        Some(idx)
    } else {
        None
    };

    let mut row_targets: Vec<Vec<ScalarCell>> = Vec::new();
    let mut row_freqs: Vec<Vec<i64>> = Vec::new();
    let mut row_scalar_cols: Vec<Vec<ScalarCell>> = vec![Vec::new(); schema.fields().len()];
    let mut exploded_refs: Vec<(usize, usize)> = Vec::new();
    let mut split_col_values: Vec<String> = Vec::new();

    for batch in batches {
        for i in 0..batch.num_rows() {
            let list = batch
                .column(target_idx)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow::anyhow!("{kind_label} TARGET must be ListArray"))?;
            let values = list.values();
            let start = list.value_offsets()[i] as usize;
            let end = list.value_offsets()[i + 1] as usize;

            let mut targets = Vec::new();
            for j in start..end {
                targets.push(scalar_from_array_at(values.as_ref(), j, "TARGET.item")?);
            }

            let freqs = if let Some(fi) = freq_idx {
                let flist = batch
                    .column(fi)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| anyhow::anyhow!("frequency column must be ListArray"))?;
                let fvals = flist.values();
                let farr = fvals
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow::anyhow!("frequency list items must be Int64"))?;
                let fs = flist.value_offsets()[i] as usize;
                let fe = flist.value_offsets()[i + 1] as usize;
                let mut fv = Vec::new();
                for j in fs..fe {
                    fv.push(if farr.is_null(j) { 1 } else { farr.value(j) });
                }
                if fv.len() != targets.len() {
                    return Err(anyhow::anyhow!(
                        "frequency/target list length mismatch for {kind_label}"
                    ));
                }
                fv
            } else {
                vec![1; targets.len()]
            };

            if let Some(sci) = split_col_idx {
                let slist = batch
                    .column(sci)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| anyhow::anyhow!("{kind_label} split column must be ListArray"))?;
                let ss = slist.value_offsets()[i] as usize;
                let se = slist.value_offsets()[i + 1] as usize;
                if se - ss != targets.len() {
                    return Err(anyhow::anyhow!(
                        "{kind_label} split-column list length mismatch with TARGET at row {i}: split={} target={}",
                        se - ss,
                        targets.len(),
                    ));
                }
                let svals = slist.values();
                for j in ss..se {
                    let cell = scalar_from_array_at(svals.as_ref(), j, "split_col.item")?;
                    split_col_values.push(match cell {
                        ScalarCell::Str(s) => s,
                        ScalarCell::Int(v) => v.to_string(),
                        ScalarCell::Float(v) => v.to_string(),
                        ScalarCell::Bool(v) => v.to_string(),
                        ScalarCell::Ts(v) => v.to_string(),
                        ScalarCell::Null => "None".to_string(),
                    });
                }
            }

            let row_idx = row_targets.len();
            for link_idx in 0..targets.len() {
                exploded_refs.push((row_idx, link_idx));
            }

            row_targets.push(targets);
            row_freqs.push(freqs);

            for col_idx in 0..schema.fields().len() {
                if col_idx == target_idx
                    || Some(col_idx) == freq_idx
                    || Some(col_idx) == split_col_idx
                {
                    continue;
                }
                let field = schema.field(col_idx);
                if matches!(field.data_type(), DataType::List(_)) {
                    return Err(anyhow::anyhow!(
                        "{kind_label} unsupported passthrough list column: {}",
                        field.name()
                    ));
                }
                row_scalar_cols[col_idx].push(scalar_from_array_at(
                    batch.column(col_idx).as_ref(),
                    i,
                    field.name(),
                )?);
            }
        }
    }

    Ok(ExplodedLinkData {
        row_targets,
        row_freqs,
        row_scalar_cols,
        exploded_refs,
        split_col_values,
        schema,
        target_idx,
        freq_idx,
        split_col_idx,
    })
}

fn regroup_output_schema(
    input_schema: &arrow_schema::SchemaRef,
    split_col_idx: Option<usize>,
) -> arrow_schema::SchemaRef {
    let mut fields: Vec<Field> = Vec::with_capacity(input_schema.fields().len() + 1);
    for (col_idx, field) in input_schema.fields().iter().enumerate() {
        if Some(col_idx) == split_col_idx {
            continue;
        }
        fields.push(field.as_ref().clone());
    }
    fields.push(Field::new(STAGE_ID_COL_NAME, DataType::UInt8, false));
    Arc::new(Schema::new(fields))
}

fn regroup_link_array_by_stage(
    data: ExplodedLinkData,
    link_stages: Vec<u8>,
    kind_label: &str,
) -> anyhow::Result<(Vec<RecordBatch>, Vec<u8>)> {
    let ExplodedLinkData {
        row_targets,
        row_freqs,
        row_scalar_cols,
        exploded_refs,
        schema,
        target_idx,
        freq_idx,
        split_col_idx,
        ..
    } = data;
    let out_schema = regroup_output_schema(&schema, split_col_idx);

    if exploded_refs.is_empty() {
        let empty_cols: Vec<Arc<dyn Array>> = out_schema
            .fields()
            .iter()
            .map(|f| arrow_array::new_empty_array(f.data_type()))
            .collect();
        let empty = RecordBatch::try_new(out_schema.clone(), empty_cols)
            .map_err(|e| anyhow::anyhow!("failed to build empty {kind_label} batch: {e}"))?;
        return Ok((vec![empty], Vec::new()));
    }
    if link_stages.len() != exploded_refs.len() {
        return Err(anyhow::anyhow!(
            "{kind_label} stage/link count mismatch: stages={} links={}",
            link_stages.len(),
            exploded_refs.len(),
        ));
    }

    let mut grouped_targets: Vec<[Vec<ScalarCell>; 3]> = (0..row_targets.len())
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let mut grouped_freqs: Vec<[Vec<i64>; 3]> = (0..row_targets.len())
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();

    for (eid, (row_idx, link_idx)) in exploded_refs.iter().enumerate() {
        let stage = link_stages[eid] as usize;
        if stage > 2 {
            continue;
        }
        grouped_targets[*row_idx][stage].push(row_targets[*row_idx][*link_idx].clone());
        grouped_freqs[*row_idx][stage].push(row_freqs[*row_idx][*link_idx]);
    }

    let mut out_source_row_idxs: Vec<usize> = Vec::new();
    let mut out_targets: Vec<Vec<ScalarCell>> = Vec::new();
    let mut out_freqs: Vec<Vec<i64>> = Vec::new();
    let mut out_assignments: Vec<u8> = Vec::new();
    for row_idx in 0..row_targets.len() {
        for stage in 0..3usize {
            if grouped_targets[row_idx][stage].is_empty() {
                continue;
            }
            out_source_row_idxs.push(row_idx);
            out_targets.push(grouped_targets[row_idx][stage].clone());
            out_freqs.push(grouped_freqs[row_idx][stage].clone());
            out_assignments.push(stage as u8);
        }
    }

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(out_schema.fields().len());
    for (col_idx, field) in schema.fields().iter().enumerate() {
        if Some(col_idx) == split_col_idx {
            continue;
        }
        let col = if col_idx == target_idx {
            build_list_array_from_cells(field.data_type(), &out_targets, "TARGET")?
        } else if Some(col_idx) == freq_idx {
            if !matches!(field.data_type(), DataType::List(_)) {
                return Err(anyhow::anyhow!(
                    "{kind_label} frequency column must be List type"
                ));
            }
            let mut b = ListBuilder::new(Int64Builder::new());
            for row in &out_freqs {
                for f in row {
                    b.values().append_value(*f);
                }
                b.append(true);
            }
            Arc::new(b.finish())
        } else {
            let cells = out_source_row_idxs
                .iter()
                .map(|src| row_scalar_cols[col_idx][*src].clone())
                .collect::<Vec<_>>();
            build_scalar_array_from_cells(field.data_type(), &cells, field.name())?
        };
        columns.push(col);
    }
    columns.push(Arc::new(UInt8Array::from(out_assignments.clone())));

    let out_batch = RecordBatch::try_new(out_schema.clone(), columns)
        .map_err(|e| anyhow::anyhow!("failed to build {kind_label} transformed batch: {e}"))?;
    Ok((vec![out_batch], out_assignments))
}

fn transform_random_link_split_link_array(
    batches: &[RecordBatch],
    ratio: Vec<f64>,
    seed: Option<u64>,
) -> anyhow::Result<(Vec<RecordBatch>, Vec<u8>)> {
    let data = collect_link_array_batches(batches, None, "RandomLinkSplit")?;
    let link_stages = if data.exploded_refs.is_empty() {
        Vec::new()
    } else {
        let mut perm: Vec<usize> = (0..data.exploded_refs.len()).collect();
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
        use rand::seq::SliceRandom;
        perm.shuffle(&mut rng);
        split_assignments::assign_by_ratio_on_perm(data.exploded_refs.len(), ratio, perm)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };
    regroup_link_array_by_stage(data, link_stages, "RandomLinkSplit")
}

fn transform_custom_link_split_link_array(
    batches: &[RecordBatch],
    split_col: &str,
    manual_train_values: Vec<String>,
    manual_val_values: Vec<String>,
    manual_test_values: Vec<String>,
) -> anyhow::Result<(Vec<RecordBatch>, Vec<u8>)> {
    let data = collect_link_array_batches(batches, Some(split_col), "CustomLinkSplit")?;
    let link_stages = if data.exploded_refs.is_empty() {
        Vec::new()
    } else {
        split_assignments::assign_manual_split(
            data.split_col_values.clone(),
            manual_train_values,
            manual_val_values,
            manual_test_values,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };
    regroup_link_array_by_stage(data, link_stages, "CustomLinkSplit")
}

use crate::artifact_helpers::{
    artifact_ref, ensure_artifacts_dir, merge_prefixed_checksums, sha256_hex_bytes, sha256_hex_path,
    write_artifact_file, write_manifest_json,
};

fn build_table_edges_bundle_value(
    output_dir: &Path,
    table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
    artifact_kind: &str,
    artifact_format: &str,
    artifact_label: &str,
) -> anyhow::Result<Value> {
    let fkey_path = output_dir.join(format!("{}_fkey_cols.parquet", table_name));
    if !fkey_path.exists() {
        return Err(anyhow::anyhow!(
            "fkey_cols artifact missing for {artifact_label} bundle: {}",
            fkey_path.display()
        ));
    }

    let mut reverse_edges = serde_json::Map::new();
    let mut reverse_checksums = serde_json::Map::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.contains("__rev_") || !name.ends_with(&format!("__{table_name}")) {
            continue;
        }
        let edge_path = path.join("edge.parquet");
        if !edge_path.exists() {
            continue;
        }
        reverse_edges.insert(
            name.to_string(),
            json!({"path": edge_path.to_string_lossy(), "format": "edge_parquet"}),
        );
        reverse_checksums.insert(name.to_string(), json!(sha256_hex_path(&edge_path)?));
    }
    if reverse_edges.is_empty() {
        return Err(anyhow::anyhow!(
            "no reverse edge artifacts discovered for {artifact_label} bundle in {}",
            output_dir.display()
        ));
    }

    let mask_value = if mask_present {
        let mask_path = output_dir.join(format!("{}_mask.parquet", table_name));
        if !mask_path.exists() {
            return Err(anyhow::anyhow!(
                "mask artifact missing for {artifact_label} bundle: {}",
                mask_path.display()
            ));
        }
        json!({
            "present": true,
            "path": mask_path.to_string_lossy(),
            "format": "mask_parquet",
        })
    } else {
        json!({
            "present": false,
            "reason": mask_absent_reason.unwrap_or("mask_not_required"),
        })
    };

    let mut checksums = serde_json::Map::new();
    checksums.insert(
        "fkey_cols_sha256".to_string(),
        json!(sha256_hex_path(&fkey_path)?),
    );
    for (k, v) in &reverse_checksums {
        checksums.insert(format!("{}_sha256", k), v.clone());
    }
    if mask_present {
        let mask_path = output_dir.join(format!("{}_mask.parquet", table_name));
        checksums.insert(
            "mask_sha256".to_string(),
            json!(sha256_hex_path(&mask_path)?),
        );
    }

    let edge_attrs_value = if let Some(path) = edge_attrs_path {
        let p = Path::new(path);
        if !p.exists() {
            return Err(anyhow::anyhow!(
                "edge_attrs artifact missing for {artifact_label} bundle: {}",
                p.display()
            ));
        }
        checksums.insert("edge_attrs_sha256".to_string(), json!(sha256_hex_path(p)?));
        Some(json!({"path": path, "format": "edge_attrs_json"}))
    } else {
        None
    };

    let mut root = serde_json::Map::new();
    root.insert("version".to_string(), json!(1));
    root.insert("kind".to_string(), json!(artifact_kind));
    root.insert("format".to_string(), json!(artifact_format));
    root.insert("reverse_edges".to_string(), Value::Object(reverse_edges));
    root.insert(
        "fkey_cols".to_string(),
        json!({"path": fkey_path.to_string_lossy(), "format": "fkey_cols_parquet"}),
    );
    root.insert("mask".to_string(), mask_value);
    let mut provided_fields = vec!["reverse_edges", "fkey_cols", "mask"];
    if let Some(v) = edge_attrs_value {
        root.insert("edge_attrs".to_string(), v);
        provided_fields.push("edge_attrs");
    }
    root.insert("checksums".to_string(), Value::Object(checksums));
    root.insert("provided_fields".to_string(), json!(provided_fields));
    Ok(Value::Object(root))
}

fn build_train_table_edges_bundle_value(
    output_dir: &Path,
    train_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> anyhow::Result<Value> {
    build_table_edges_bundle_value(
        output_dir,
        train_table_name,
        mask_present,
        mask_absent_reason,
        edge_attrs_path,
        "tt_train_table_edges",
        "tt_edge_bundle_v1",
        "train_table_edges",
    )
}

fn build_prediction_table_edges_bundle_value(
    output_dir: &Path,
    prediction_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> anyhow::Result<Value> {
    build_table_edges_bundle_value(
        output_dir,
        prediction_table_name,
        mask_present,
        mask_absent_reason,
        edge_attrs_path,
        "tt_prediction_table_edges",
        "tt_pred_edge_bundle_v1",
        "prediction_table_edges",
    )
}

fn split_family_from_config(
    split_family_hint: Option<&str>,
    split_cfg: Option<&Value>,
) -> Option<String> {
    if let Some(h) = split_family_hint {
        if !h.is_empty() {
            return Some(h.to_string());
        }
    }
    if let Some(Value::Object(map)) = split_cfg {
        if let Some(Value::String(target)) = map.get("_target_") {
            return target.rsplit('.').next().map(|s| s.to_string());
        }
    }
    None
}

fn parse_ratio(split_cfg: &Value) -> anyhow::Result<Vec<f64>> {
    let ratio = split_cfg
        .get("ratio")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("split ratio is required"))?;
    let out = ratio
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| anyhow::anyhow!("split ratio must be numeric"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(out)
}

fn parse_i64_list(split_cfg: &Value, key: &str) -> anyhow::Result<Vec<i64>> {
    let arr = split_cfg
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("split {key} is required"))?;
    arr.iter()
        .map(|v| {
            v.as_i64()
                .or_else(|| v.as_f64().map(|x| x as i64))
                .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))
                .ok_or_else(|| anyhow::anyhow!("split {key} entries must be numeric"))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

fn parse_time_ranges_ns(split_cfg: &Value, timestamp_unit: &str) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    let ranges = split_cfg
        .get("ranges")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("split ranges are required"))?;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for pair in ranges.iter().take(3) {
        let p = pair
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("split ranges entries must be [start,end]"))?;
        if p.len() != 2 {
            return Err(anyhow::anyhow!("split ranges entries must be length-2"));
        }
        let raw_start = parse_timestamp_ns_json(&p[0])?;
        let raw_end = parse_timestamp_ns_json(&p[1])?;
        let ns_start = to_nanoseconds(raw_start, timestamp_unit)?;
        let ns_end = to_nanoseconds(raw_end, timestamp_unit)?;
        starts.push(ns_start);
        ends.push(ns_end);
    }
    Ok((starts, ends))
}

#[derive(Clone, Debug)]
enum SortValue {
    Null,
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Ts(i64),
}

impl SortValue {
    fn rank(&self) -> u8 {
        match self {
            SortValue::Int(_) => 0,
            SortValue::Float(_) => 1,
            SortValue::Str(_) => 2,
            SortValue::Bool(_) => 3,
            SortValue::Ts(_) => 4,
            SortValue::Null => 5,
        }
    }
}

impl PartialEq for SortValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for SortValue {}

impl PartialOrd for SortValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let r = self.rank().cmp(&other.rank());
        if r != std::cmp::Ordering::Equal {
            return r;
        }
        match (self, other) {
            (SortValue::Null, SortValue::Null) => std::cmp::Ordering::Equal,
            (SortValue::Int(a), SortValue::Int(b)) => a.cmp(b),
            (SortValue::Float(a), SortValue::Float(b)) => a.total_cmp(b),
            (SortValue::Str(a), SortValue::Str(b)) => a.cmp(b),
            (SortValue::Bool(a), SortValue::Bool(b)) => a.cmp(b),
            (SortValue::Ts(a), SortValue::Ts(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "None".to_string(),
        _ => v.to_string(),
    }
}

fn parse_optional_string_list(split_cfg: &Value, key: &str) -> Vec<String> {
    match split_cfg.get(key) {
        Some(Value::Array(arr)) => arr.iter().map(value_to_string).collect::<Vec<_>>(),
        Some(v) => vec![value_to_string(v)],
        None => Vec::new(),
    }
}

fn parse_split_column_name(split_cfg: &Value) -> Option<String> {
    for key in ["split_output_column", "split_projection"] {
        if let Some(s) = split_cfg.get(key).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    match split_cfg.get("col") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Object(m)) => m
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => split_cfg
            .get("split_column")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
    }
}

#[derive(Clone, Debug, Default)]
struct DownsamplingConfigNative {
    enabled: bool,
    max_rows: usize,
    preserve_pre_downsample_tt: bool,
    is_link_prediction: bool,
    list_distinct_target_is_array: bool,
    uses_arl_labels: bool,
}

fn parse_downsampling_config(raw: Option<&Value>) -> anyhow::Result<DownsamplingConfigNative> {
    let mut cfg = DownsamplingConfigNative {
        enabled: false,
        max_rows: i32::MAX as usize,
        preserve_pre_downsample_tt: true,
        is_link_prediction: false,
        list_distinct_target_is_array: false,
        uses_arl_labels: false,
    };

    let Some(Value::Object(m)) = raw else {
        return Ok(cfg);
    };

    cfg.enabled = m.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    cfg.max_rows = m
        .get("max_rows")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(i32::MAX as usize);
    cfg.preserve_pre_downsample_tt = m
        .get("preserve_pre_downsample_tt")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    cfg.is_link_prediction = m
        .get("is_link_prediction")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    cfg.list_distinct_target_is_array = m
        .get("list_distinct_target_is_array")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    cfg.uses_arl_labels = m
        .get("uses_arl_labels")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if cfg.enabled && cfg.max_rows == 0 {
        return Err(anyhow::anyhow!("downsampling max_rows must be positive"));
    }

    Ok(cfg)
}

fn extract_entity_strings(batch: &RecordBatch) -> anyhow::Result<Vec<String>> {
    extract_string_values(batch, "ENTITY")
}

#[allow(dead_code)]
fn downsample_indices_legacy(
    cfg: &DownsamplingConfigNative,
    entities: &[String],
    row_count: usize,
) -> anyhow::Result<(Vec<usize>, String)> {
    if row_count <= cfg.max_rows {
        return Ok(((0..row_count).collect(), "none".to_string()));
    }
    if cfg.uses_arl_labels {
        return Err(anyhow::anyhow!(
            "downsampling refuses ARL labels parity path"
        ));
    }

    if cfg.is_link_prediction && !cfg.list_distinct_target_is_array {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for e in entities {
            if seen.insert(e.clone()) {
                unique.push(e.clone());
            }
        }
        let num_ids = unique.len();
        let id_sample_size = ((num_ids * cfg.max_rows) / row_count).max(1);
        let sampled: HashSet<String> = unique.into_iter().take(id_sample_size).collect();
        let mut idx = Vec::new();
        for (i, e) in entities.iter().enumerate() {
            if sampled.contains(e) {
                idx.push(i);
            }
        }
        if idx.len() > cfg.max_rows {
            idx.truncate(cfg.max_rows);
        }
        return Ok((idx, "link_by_entity".to_string()));
    }

    Ok(((0..cfg.max_rows).collect(), "limit".to_string()))
}

fn patch_summary_for_final_rows(
    summary_json: String,
    final_row_count: usize,
    entity_values: Option<&[String]>,
    target_values: Option<&[String]>,
    cfg: &DownsamplingConfigNative,
) -> anyhow::Result<String> {
    let mut value: Value = serde_json::from_str(&summary_json)?;
    let root = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("summary root must be object"))?;
    let stats_val = root
        .entry("stats")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let stats = stats_val
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("summary.stats must be object"))?;

    stats.insert(
        "num_total_rows".to_string(),
        Value::from(final_row_count as u64),
    );
    stats.insert(
        "num_train_rows".to_string(),
        Value::from(final_row_count as u64),
    );

    if let Some(entities) = entity_values {
        let uniq = entities.iter().collect::<HashSet<_>>().len();
        stats.insert("num_entities".to_string(), Value::from(uniq as u64));
    }
    if let Some(targets) = target_values {
        let uniq = targets.iter().collect::<HashSet<_>>().len();
        stats.insert("num_targets".to_string(), Value::from(uniq as u64));
    }
    if cfg.is_link_prediction {
        stats.insert("num_links".to_string(), Value::from(final_row_count as u64));
    }

    serde_json::to_string(&value).map_err(Into::into)
}

fn build_tt_stats_artifact_value(
    summary: Option<&Value>,
    row_count: u64,
    plan: Option<&Value>,
) -> Value {
    let stats = summary
        .and_then(|s| s.get("stats"))
        .and_then(|s| s.as_object())
        .cloned()
        .unwrap_or_default();

    let num_total_rows = stats
        .get("num_total_rows")
        .and_then(|v| v.as_u64())
        .unwrap_or(row_count);
    let num_train_rows = stats
        .get("num_train_rows")
        .and_then(|v| v.as_u64())
        .unwrap_or(row_count);
    let num_entities = stats
        .get("num_entities")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let num_targets = stats
        .get("num_targets")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let num_links = stats.get("num_links").and_then(|v| v.as_u64());

    let exact_stats = stats.get("exact_stats").cloned().unwrap_or(Value::Null);
    let split_payload = stats.get("split_payload").cloned().unwrap_or(Value::Null);

    let timestamp_unique = exact_stats
        .get("timestamp")
        .and_then(|v| v.get("num_unique"))
        .and_then(|v| v.as_u64());

    // Use actual produced timestamps from split_payload (only timestamps with
    // data), not the full input timestamp list from the plan.  The plan's
    // temporal.timestamps includes ALL candidate timestamps, but some may
    // produce zero rows (e.g. ASSUMING clause filters all entities).  GE's
    // Spark path naturally excludes empty timeframes; DG must do the same.
    let mut produced_timestamps: Vec<i64> = split_payload
        .as_object()
        .and_then(|sp| sp.get("artifacts_v3"))
        .and_then(|a| a.get("timestamp_stage_assignments"))
        .and_then(|arr| arr.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("timestamp").and_then(|t| t.as_i64()))
                .collect()
        })
        .unwrap_or_default();
    // Sort newest-first: stats_per_timeframe must be in reverse chronological
    // order (Python's update_train_table_stats asserts this).
    produced_timestamps.sort_unstable_by(|a, b| b.cmp(a));

    let is_temporal = !produced_timestamps.is_empty()
        || plan
            .and_then(|p| p.get("temporal"))
            .and_then(|t| t.get("timestamps"))
            .and_then(|arr| arr.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

    let num_timeframes = if !produced_timestamps.is_empty() {
        produced_timestamps.len() as u64
    } else {
        timestamp_unique.unwrap_or(0)
    };
    let num_timeframes_post_filter = num_timeframes;
    let avg_entities_per_timeframe = if num_timeframes_post_filter > 0 {
        (num_train_rows as f64 / num_timeframes_post_filter as f64).round() as u64
    } else {
        0
    };
    let avg_targets_per_timeframe = if num_timeframes_post_filter > 0 {
        (num_targets as f64 / num_timeframes_post_filter as f64).round() as u64
    } else {
        0
    };

    let timestamp_max = exact_stats
        .get("timestamp")
        .and_then(|v| v.get("max"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let is_link_prediction = plan
        .and_then(|p| p.get("prediction_type"))
        .and_then(|v| v.as_object())
        .map(|m| m.contains_key("link"))
        .unwrap_or(false);

    // Extract per-timestamp row counts from the Rust TT gen stats.
    // These are the real counts (like GE's Spark groupby on TIME_BUCKET_COL).
    let ts_counts: HashMap<i64, u64> = stats
        .get("timestamp_counts")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    let ts = k.parse::<i64>().ok()?;
                    let count = v.as_u64()?;
                    Some((ts, count))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut stats_per_timeframe: Vec<Value> = Vec::new();
    if is_temporal {
        let mut ts_values = produced_timestamps.clone();
        if ts_values.is_empty() {
            ts_values = vec![timestamp_max, timestamp_max];
        } else if ts_values.len() == 1 {
            ts_values.push(ts_values[0]);
        }
        for (idx, ts) in ts_values.iter().enumerate() {
            let rows_for_tf = ts_counts.get(ts).copied().unwrap_or(0);
            let dt = Utc
                .timestamp_opt(*ts, 0)
                .single()
                .map(|d| d.to_rfc3339())
                .unwrap_or_else(|| ts.to_string());
            if is_link_prediction {
                stats_per_timeframe.push(json!({
                    "min_date": dt,
                    "max_date": dt,
                    "anchor_date": dt,
                    "num_train_rows": rows_for_tf,
                    "num_entities": num_entities,
                    "num_targets": num_targets,
                    "num_links": num_links,
                    "index": idx,
                }));
            } else {
                stats_per_timeframe.push(json!({
                    "min_date": dt,
                    "max_date": dt,
                    "anchor_date": dt,
                    "num_train_rows": rows_for_tf,
                    "num_entities": num_entities,
                    "num_targets": num_targets,
                    "target_distribution": json!({}),
                    "index": idx,
                }));
            }
        }
    }

    let mut payload = json!({
        "version": 2,
        "kind": "tt_stats",
        "num_total_rows": num_total_rows,
        "num_train_rows": num_train_rows,
        "num_entities": num_entities,
        "num_targets": num_targets,
        "num_links": num_links,
        "num_target_rows": num_train_rows,
        "num_timeframes": num_timeframes,
        "num_timeframes_post_filter": num_timeframes_post_filter,
        "avg_entities_per_timeframe": avg_entities_per_timeframe,
        "avg_targets_per_timeframe": avg_targets_per_timeframe,
        "node_degree_stats": Value::Null,
        "stats_per_timeframe": stats_per_timeframe,
        "target_distribution": json!({}),
        "scoring_table_column_stats": Value::Null,
        "exact_stats": exact_stats,
        "split_payload": split_payload,
    });

    let mut provided_fields = vec![
        "num_total_rows",
        "num_train_rows",
        "num_entities",
        "num_targets",
        "num_target_rows",
        "num_timeframes",
        "num_timeframes_post_filter",
        "avg_entities_per_timeframe",
        "avg_targets_per_timeframe",
        "node_degree_stats",
        "stats_per_timeframe",
        "target_distribution",
        "scoring_table_column_stats",
    ];
    if num_links.is_some() {
        provided_fields.push("num_links");
    }
    if !payload
        .get("exact_stats")
        .map(|v| v.is_null())
        .unwrap_or(true)
    {
        provided_fields.push("exact_stats");
    }
    if !payload
        .get("split_payload")
        .map(|v| v.is_null())
        .unwrap_or(true)
    {
        provided_fields.push("split_payload");
    }
    payload["provided_fields"] = Value::Array(
        provided_fields
            .into_iter()
            .map(|s| Value::String(s.to_string()))
            .collect(),
    );
    payload
}

fn array_to_opt_strings(arr: &dyn Array) -> anyhow::Result<Vec<Option<String>>> {
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    Err(anyhow::anyhow!("unsupported column dtype for split stats"))
}

fn collect_column_opt_strings(
    path: &str,
    col_name: &str,
) -> anyhow::Result<Option<Vec<Option<String>>>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let Ok(col_idx) = schema.index_of(col_name) else {
        return Ok(None);
    };
    let mut reader = builder.build()?;
    let mut out: Vec<Option<String>> = Vec::new();
    while let Some(batch_res) = reader.next() {
        let batch = batch_res?;
        let mut vals = array_to_opt_strings(batch.column(col_idx).as_ref())?;
        out.append(&mut vals);
    }
    Ok(Some(out))
}






fn build_stage_stats_from_columns(
    assignments: &[u8],
    stage: u8,
    entity_vals: Option<&[Option<String>]>,
    target_vals: Option<&[Option<String>]>,
    time_vals: Option<&[Option<String>]>,
    task_type: Option<&str>,
) -> Value {
    let idx = assignments
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if *s == stage { Some(i) } else { None })
        .collect::<Vec<_>>();
    if idx.is_empty() {
        return Value::Null;
    }

    let num_rows = idx.len() as u64;
    let num_entities = if let Some(e) = entity_vals {
        idx.iter()
            .filter_map(|i| e.get(*i).and_then(|v| v.clone()))
            .collect::<HashSet<_>>()
            .len() as u64
    } else {
        num_rows
    };

    let mut num_targets: Option<u64> = None;
    let mut target_distribution = Value::Null;
    if let Some(t) = target_vals {
        let vals = idx
            .iter()
            .filter_map(|i| t.get(*i).and_then(|v| v.clone()))
            .collect::<Vec<_>>();
        if !vals.is_empty() {
            num_targets = Some(vals.iter().collect::<HashSet<_>>().len() as u64);
            if let Some(tt) = task_type {
                if tt == "binary_classification" || tt == "multiclass_classification" {
                    let mut counts = std::collections::BTreeMap::<String, u64>::new();
                    for v in vals {
                        *counts.entry(v).or_insert(0) += 1;
                    }
                    let total: u64 = counts.values().sum();
                    if total > 0 {
                        let count_json = counts
                            .iter()
                            .map(|(k, v)| (k.clone(), Value::from(*v as f64)))
                            .collect::<serde_json::Map<String, Value>>();
                        let pct_json = counts
                            .iter()
                            .map(|(k, v)| {
                                (k.clone(), Value::from((*v as f64) * 100.0 / (total as f64)))
                            })
                            .collect::<serde_json::Map<String, Value>>();
                        target_distribution = json!({"count": count_json, "percentage": pct_json});
                    }
                }
            }
        }
    }

    let (min_date, max_date, anchor_date) = if let Some(t) = time_vals {
        let vals = idx
            .iter()
            .filter_map(|i| t.get(*i).and_then(|v| v.clone()))
            .collect::<Vec<_>>();
        if vals.is_empty() {
            (Value::Null, Value::Null, Value::Null)
        } else if vals.iter().all(|v| v.parse::<f64>().is_ok()) {
            let nums = vals
                .iter()
                .filter_map(|v| v.parse::<f64>().ok())
                .collect::<Vec<_>>();
            let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_dt = Utc
                .timestamp_opt(min as i64, 0)
                .single()
                .map(|d| d.to_rfc3339())
                .unwrap_or_else(|| (min as i64).to_string());
            let max_dt = Utc
                .timestamp_opt(max as i64, 0)
                .single()
                .map(|d| d.to_rfc3339())
                .unwrap_or_else(|| (max as i64).to_string());
            (json!(min_dt), json!(max_dt), json!(min_dt))
        } else {
            let min = vals.iter().min().cloned().unwrap_or_default();
            let max = vals.iter().max().cloned().unwrap_or_default();
            (json!(min.clone()), json!(max), json!(min))
        }
    } else {
        (Value::Null, Value::Null, Value::Null)
    };

    json!({
        "num_train_rows": num_rows,
        "num_entities": num_entities,
        "num_targets": num_targets,
        "target_distribution": target_distribution,
        "min_date": min_date,
        "max_date": max_date,
        "anchor_date": anchor_date,
    })
}

fn parse_timestamp_ns_json(v: &Value) -> anyhow::Result<i64> {
    if let Some(i) = v.as_i64() {
        return Ok(i);
    }
    if let Some(f) = v.as_f64() {
        return Ok(f as i64);
    }
    if let Some(s) = v.as_str() {
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Ok(dt
                .with_timezone(&Utc)
                .timestamp_nanos_opt()
                .ok_or_else(|| anyhow::anyhow!("timestamp ns overflow"))?);
        }
        if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let dt = d
                .and_hms_opt(0, 0, 0)
                .ok_or_else(|| anyhow::anyhow!("invalid date"))?
                .and_utc();
            return Ok(dt
                .timestamp_nanos_opt()
                .ok_or_else(|| anyhow::anyhow!("timestamp ns overflow"))?);
        }
    }
    Err(anyhow::anyhow!(
        "timestamp must be numeric epoch-ns or parseable date string"
    ))
}

/// Convert a timestamp value to nanoseconds based on the declared unit.
fn to_nanoseconds(value: i64, timestamp_unit: &str) -> anyhow::Result<i64> {
    let unit = TimeUnit::from_name(timestamp_unit)
        .ok_or_else(|| anyhow::anyhow!("unsupported timestamp_unit: {}", timestamp_unit))?;
    Ok(abs_from_unit::<Nanoseconds>(value, unit).raw())
}

/// Read `timestamp_unit` from split config, defaulting to "seconds".
fn get_timestamp_unit(split_cfg: &Value) -> &str {
    split_cfg
        .get("timestamp_unit")
        .and_then(|v| v.as_str())
        .unwrap_or("seconds")
}

fn extract_sort_values(batch: &RecordBatch, col_name: &str) -> anyhow::Result<Vec<SortValue>> {
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| anyhow::anyhow!("ColumnSplit missing split column: {col_name}"))?;
    let arr = batch.column(idx);

    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    SortValue::Null
                } else {
                    SortValue::Int(a.value(i))
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    SortValue::Null
                } else {
                    SortValue::Float(a.value(i))
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    SortValue::Null
                } else {
                    SortValue::Str(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    SortValue::Null
                } else {
                    SortValue::Bool(a.value(i))
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    SortValue::Null
                } else {
                    SortValue::Ts(a.value(i))
                }
            })
            .collect());
    }

    Err(anyhow::anyhow!(
        "ColumnSplit unsupported split column dtype for {col_name}"
    ))
}

fn extract_string_values(batch: &RecordBatch, col_name: &str) -> anyhow::Result<Vec<String>> {
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| anyhow::anyhow!("ManualSplit missing split column: {col_name}"))?;
    let arr = batch.column(idx);

    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    "None".to_string()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    "None".to_string()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    "None".to_string()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    "None".to_string()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    "None".to_string()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect());
    }

    Err(anyhow::anyhow!(
        "ManualSplit unsupported split column dtype for {col_name}"
    ))
}

fn extend_timestamps_ns_from_batch(batch: &RecordBatch, out: &mut Vec<i64>) {
    if let Ok(idx) = batch.schema().index_of("TIMESTAMP") {
        if let Some(ts_arr) = batch
            .column(idx)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
        {
            for i in 0..ts_arr.len() {
                out.push(if ts_arr.is_null(i) {
                    0
                } else {
                    ts_arr.value(i)
                });
            }
        }
    }
}

fn extend_include_mask_from_batch(batch: &RecordBatch, include_mask: &mut Option<Vec<bool>>) {
    if let Ok(idx) = batch.schema().index_of("ENTITY_INCLUDE_BIT") {
        if let Some(mask_arr) = batch.column(idx).as_any().downcast_ref::<BooleanArray>() {
            if include_mask.is_none() {
                *include_mask = Some(Vec::new());
            }
            if let Some(mask) = include_mask.as_mut() {
                for i in 0..mask_arr.len() {
                    mask.push(!mask_arr.is_null(i) && mask_arr.value(i));
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature=(plan_json, index_path, output_path, split_config_json=None, split_family_hint=None, downsampling_config_json=None))]
#[tracing::instrument(level = "info", skip(py, plan_json, split_config_json, downsampling_config_json), fields(index_path = %index_path, output_path = %output_path))]
pub fn execute_tt_plan_materialize_with_split(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_path: &str,
    split_config_json: Option<&str>,
    split_family_hint: Option<&str>,
    downsampling_config_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let overall_start = std::time::Instant::now();
    let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(plan_json)
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("invalid plan JSON: {e}")))?;

    let split_cfg: Option<Value> = match split_config_json {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("invalid split config JSON: {e}"))
        })?),
        _ => None,
    };

    let family = split_family_from_config(split_family_hint, split_cfg.as_ref())
        .unwrap_or_else(|| "UnknownSplit".to_string());

    let split_column_name = split_cfg.as_ref().and_then(parse_split_column_name);
    let downsampling_cfg: Option<Value> = match downsampling_config_json {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("invalid downsampling config JSON: {e}"))
        })?),
        _ => None,
    };
    let downsampling_cfg = parse_downsampling_config(downsampling_cfg.as_ref()).map_err(|e| {
        crate::DiskgraphConfigError::new_err(format!("invalid downsampling config: {e}"))
    })?;

    let manual_train_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "manual_train_values"))
        .unwrap_or_default();
    let manual_val_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "manual_val_values"))
        .unwrap_or_default();
    let manual_test_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "manual_test_values"))
        .unwrap_or_default();

    let idx = index_path.to_owned();
    let out = output_path.to_owned();

    let (
        rows_written,
        summary_json,
        split_bytes,
        split_authoritative,
        split_reason,
        downsampling_handled,
        downsampling_applied,
        pre_downsample_row_count,
        train_table_col_stats_json,
    ) = py
        .detach(|| -> anyhow::Result<(u64, String, Option<Vec<u8>>, bool, Option<String>, bool, bool, u64, Option<String>)> {
            let traversal_start = std::time::Instant::now();
            let (receiver, handle) =
                diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))?;

            let mut row_offset = 0i64;
            let mut total_bytes_processed = 0usize;

            // For downsampling (link_by_entity mode): track unique entities
            // in encounter order. O(unique_entities) ≈ ~1.8 GB for Fetch.
            // All other per-row data (timestamps, masks, entity/target
            // strings) is extracted from post-DS final_batches after
            // read-back — not accumulated during the consumer loop.
            let mut entity_unique_ordered: Vec<String> = Vec::new();
            let mut entity_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Write raw batches to a scratch parquet file instead of
            // accumulating in memory. The scratch file serves as:
            //   1. The pre-downsample artifact (.pre_downsample.parquet)
            //   2. The read-back source for post-downsample filtering
            let scratch_path = format!("{}.scratch.parquet", out);
            let mut scratch_guard = ScratchFileGuard::new(&scratch_path);
            let mut scratch_writer = ScratchParquetWriter::new(&scratch_path);
            let mut batch_count = 0u64;

            let mut last_logged_gb = 0usize;
            for batch in receiver.iter() {
                let batch_memory = batch.get_array_memory_size();
                total_bytes_processed += batch_memory;
                batch_count += 1;

                let current_gb = total_bytes_processed / (1024 * 1024 * 1024);
                if current_gb > last_logged_gb {
                    info!(
                        batch_count = batch_count,
                        bytes_processed_mb = total_bytes_processed / 1024 / 1024,
                        process_rss_mb = get_current_rss_mb(),
                        "TT-gen materialization progress"
                    );
                    last_logged_gb = current_gb;
                }

                // Build unique entity set only when link_by_entity downsampling
                // may be needed (LP with non-array targets and DS enabled).
                if downsampling_cfg.enabled
                    && downsampling_cfg.is_link_prediction
                    && !downsampling_cfg.list_distinct_target_is_array
                {
                    let entities = extract_entity_strings(&batch)?;
                    for e in entities {
                        if entity_seen.insert(e.clone()) {
                            entity_unique_ordered.push(e);
                        }
                    }
                }

                row_offset += batch.num_rows() as i64;
                scratch_writer.write_batch(&batch)?;
            }
            scratch_writer.finish()?;

            if row_offset == 0 {
                return Err(anyhow::anyhow!("diskgraph execute_tt_plan returned zero rows"));
            }

            let pre_downsample_row_count = row_offset as usize;

            // Safety valve: if row count exceeds the threshold and
            // downsampling is disabled, fail fast instead of OOMing
            // on the entity/target string vectors (which are still
            // accumulated in memory until streaming step 2 lands).
            if row_offset as u64 > MAX_TT_ROWS_WITHOUT_DOWNSAMPLING
                && !downsampling_cfg.enabled
            {
                // scratch_guard handles cleanup on return
                return Err(anyhow::anyhow!(
                    "TT-gen produced {} rows, exceeding the {} row limit. \
                     Enable downsampling via downsampling_config_json \
                     or reduce the entity/anchor space.",
                    row_offset,
                    MAX_TT_ROWS_WITHOUT_DOWNSAMPLING,
                ));
            }

            info!(
                rows_generated = pre_downsample_row_count,
                bytes_processed_mb = total_bytes_processed / 1024 / 1024,
                process_rss_mb = get_current_rss_mb(),
                duration_ms = traversal_start.elapsed().as_millis(),
                "Parallel traversal complete (batches written to scratch)"
            );

            // Free the seen set — no longer needed after consumer loop.
            drop(entity_seen);

            // Downsampling decision: build entity filter HashSet if needed.
            // The HashSet is passed to read_scratch_filtered for O(1) per-row
            // filtering during a single scratch read pass.
            let mut sampled_entities: Option<std::collections::HashSet<String>> = None;
            let mut downsampling_applied = false;
            let mut downsampling_handled = false;
            let mut ds_max_rows = usize::MAX;

            if downsampling_cfg.enabled {
                let downsample_start = std::time::Instant::now();
                downsampling_handled = true;

                if pre_downsample_row_count <= downsampling_cfg.max_rows {
                    info!(
                        rows = pre_downsample_row_count,
                        max_rows = downsampling_cfg.max_rows,
                        "Downsampling: under limit, no action"
                    );
                } else if downsampling_cfg.uses_arl_labels {
                    return Err(anyhow::anyhow!(
                        "downsampling refuses ARL labels parity path"
                    ));
                } else if downsampling_cfg.is_link_prediction
                    && !downsampling_cfg.list_distinct_target_is_array
                {
                    // link_by_entity: sample first N unique entities
                    let num_unique = entity_unique_ordered.len();
                    let id_sample_size =
                        ((num_unique as u64 * downsampling_cfg.max_rows as u64)
                            / pre_downsample_row_count as u64)
                            .max(1) as usize;
                    let sampled: std::collections::HashSet<String> = entity_unique_ordered
                        [..id_sample_size.min(num_unique)]
                        .iter()
                        .cloned()
                        .collect();
                    info!(
                        unique_entities = num_unique,
                        sampled_entities = sampled.len(),
                        "Downsampling: link_by_entity entity sampling"
                    );
                    sampled_entities = Some(sampled);
                    ds_max_rows = downsampling_cfg.max_rows;
                    downsampling_applied = true;
                } else {
                    // Simple limit: cap at max_rows (no entity filter)
                    ds_max_rows = downsampling_cfg.max_rows;
                    downsampling_applied = true;
                }

                info!(
                    downsampling_applied = downsampling_applied,
                    ds_max_rows = ds_max_rows,
                    duration_ms = downsample_start.elapsed().as_millis(),
                    "Downsampling decision complete"
                );
            }
            drop(entity_unique_ordered);

            // Rename scratch to pre-downsample artifact path if preserving,
            // otherwise read from scratch in place (guard handles cleanup).
            let readback_path;
            if downsampling_cfg.enabled && downsampling_cfg.preserve_pre_downsample_tt {
                let pre_path = format!("{}.pre_downsample.parquet", out);
                std::fs::rename(&scratch_path, &pre_path).map_err(|e| {
                    anyhow::anyhow!("failed to rename scratch to pre_downsample artifact: {e}")
                })?;
                scratch_guard.disarm(); // file renamed, guard should not delete
                readback_path = pre_path;
            } else {
                readback_path = scratch_path.clone();
            }

            // Single-pass read-back with entity filter + row cap.
            // Post-DS row count is bounded (≤max_rows) so final_batches fits in memory.
            let readback_start = std::time::Instant::now();
            let mut final_batches = read_scratch_filtered(
                &readback_path,
                sampled_entities.as_ref(),
                ds_max_rows,
            )?;
            // Disarm guard after successful read-back; delete scratch if not preserved.
            scratch_guard.disarm();
            if readback_path == scratch_path {
                let _ = std::fs::remove_file(&scratch_path);
            }
            let mut row_offset = row_count_of_batches(&final_batches) as i64;
            info!(
                rows_after_readback = row_offset,
                duration_ms = readback_start.elapsed().as_millis(),
                "Scratch read-back complete"
            );

            // Extract ALL per-row data from post-DS final_batches.
            // These are bounded (≤200M rows) so they fit in memory.
            let mut timestamps_ns: Vec<i64> = Vec::new();
            let mut include_mask: Option<Vec<bool>> = None;
            let mut column_sort_values: Vec<SortValue> = Vec::new();
            let mut manual_values: Vec<String> = Vec::new();
            let mut entity_values: Vec<String> = Vec::new();
            let mut target_values: Vec<String> = Vec::new();
            for batch in &final_batches {
                extend_timestamps_ns_from_batch(batch, &mut timestamps_ns);
                extend_include_mask_from_batch(batch, &mut include_mask);
                if family == "ColumnSplit" {
                    let col_name = split_column_name
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("ColumnSplit requires split.col name"))?;
                    column_sort_values.append(&mut extract_sort_values(batch, col_name)?);
                } else if family == "ManualSplit" || family == "CustomSplit" {
                    let col_name = split_column_name
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("{family} requires split.col name"))?;
                    manual_values.append(&mut extract_string_values(batch, col_name)?);
                }
                if let Ok(mut e) = extract_entity_strings(batch) {
                    entity_values.append(&mut e);
                }
                if let Ok(mut t) = extract_string_values(batch, "TARGET") {
                    target_values.append(&mut t);
                }
            }

            let summary = handle
                .join()
                .map_err(|_| anyhow::anyhow!("TT-gen summary thread panicked"))??;
            let raw_summary_json = serde_json::to_string(&summary)?;

            let split_authoritative = true;
            let split_reason: Option<String> = None;
            let split_assignments = if let Some(split_cfg) = split_cfg.as_ref() {
                let mut out = match family.as_str() {
                    "FixedSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        split_assignments::compute_dense_split_assignments_core(
                            "FixedSplit",
                            row_offset as usize,
                            ratio,
                            None,
                            None,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "RandomSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        let seed = split_cfg.get("rnd_seed").and_then(|v| v.as_u64());
                        split_assignments::compute_dense_split_assignments_core(
                            "RandomSplit",
                            row_offset as usize,
                            ratio,
                            seed,
                            None,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "TemporalSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        split_assignments::compute_dense_split_assignments_core(
                            "TemporalSplit",
                            row_offset as usize,
                            ratio,
                            None,
                            Some(timestamps_ns.clone()),
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "ApproxDateOffsetSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        split_assignments::assign_approx_date_offset_split(
                            timestamps_ns.clone(),
                            ratio,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "DateOffsetSplit" => {
                        let offsets = parse_i64_list(split_cfg, "offsets")?;
                        let offset_unit = split_cfg
                            .get("offset_unit")
                            .and_then(|v| v.as_str())
                            .unwrap_or("days");
                        let unit = split_cfg
                            .get("unit")
                            .and_then(|v| v.as_str())
                            .unwrap_or("days");
                        let timeframe = split_cfg.get("timeframe").and_then(|v| v.as_i64());
                        let anchor_start_offset = split_cfg
                            .get("anchor_start_offset")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        split_assignments::assign_date_offset_split(
                            timestamps_ns.clone(),
                            offsets,
                            offset_unit,
                            unit,
                            timeframe,
                            anchor_start_offset,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "TimeRangeSplit" => {
                        let ts_unit = get_timestamp_unit(split_cfg);
                        let (starts, ends) = parse_time_ranges_ns(split_cfg, ts_unit)?;
                        let timeframe = split_cfg.get("timeframe").and_then(|v| v.as_i64());
                        let unit = split_cfg
                            .get("unit")
                            .and_then(|v| v.as_str())
                            .unwrap_or("days");
                        let anchor_start_offset = split_cfg
                            .get("anchor_start_offset")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        split_assignments::assign_time_range_split(
                            timestamps_ns.clone(),
                            starts,
                            ends,
                            timeframe,
                            unit,
                            anchor_start_offset,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "BacktestingSplit" => {
                        let ts_unit = get_timestamp_unit(split_cfg);
                        let ts_val = split_cfg
                            .get("backtesting_timestamp")
                            .ok_or_else(|| anyhow::anyhow!("BacktestingSplit requires backtesting_timestamp"))?;
                        let ts = to_nanoseconds(parse_timestamp_ns_json(ts_val)?, ts_unit)?;
                        split_assignments::assign_backtesting_split(timestamps_ns.clone(), ts)
                    }
                    "ColumnSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        if column_sort_values.len() != row_offset as usize {
                            return Err(anyhow::anyhow!(
                                "ColumnSplit collected values mismatch rows (values={}, rows={})",
                                column_sort_values.len(),
                                row_offset
                            ));
                        }
                        let mut keyed: Vec<(usize, SortValue)> = column_sort_values
                            .iter()
                            .cloned()
                            .enumerate()
                            .collect();
                        keyed.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                        let perm = keyed.into_iter().map(|(idx, _)| idx).collect::<Vec<_>>();
                        split_assignments::assign_by_ratio_on_perm(
                            row_offset as usize,
                            ratio,
                            perm,
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "ManualSplit" | "CustomSplit" => {
                        if manual_values.len() != row_offset as usize {
                            return Err(anyhow::anyhow!(
                                "{family} collected values mismatch rows (values={}, rows={})",
                                manual_values.len(),
                                row_offset
                            ));
                        }
                        split_assignments::assign_manual_split(
                            manual_values.clone(),
                            manual_train_values.clone(),
                            manual_val_values.clone(),
                            manual_test_values.clone(),
                        )
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    }
                    "CustomLinkSplit" => {
                        if !is_link_array_batches(&final_batches) {
                            return Err(anyhow::anyhow!(
                                "CustomLinkSplit requires array-target link-prediction batches"
                            ));
                        }
                        let col_name = split_column_name
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!("CustomLinkSplit requires split.col name"))?;
                        let (transformed_batches, transformed_assignments) =
                            transform_custom_link_split_link_array(
                                &final_batches,
                                col_name,
                                manual_train_values.clone(),
                                manual_val_values.clone(),
                                manual_test_values.clone(),
                            )?;
                        final_batches = transformed_batches;
                        row_offset = row_count_of_batches(&final_batches) as i64;

                        if let Ok(mut entities) = final_batches
                            .iter()
                            .map(extract_entity_strings)
                            .collect::<anyhow::Result<Vec<_>>>()
                            .map(|chunks| chunks.into_iter().flatten().collect::<Vec<_>>())
                        {
                            entity_values = std::mem::take(&mut entities);
                        }
                        timestamps_ns.clear();
                        include_mask = None;
                        for batch in &final_batches {
                            extend_timestamps_ns_from_batch(batch, &mut timestamps_ns);
                            extend_include_mask_from_batch(batch, &mut include_mask);
                        }

                        transformed_assignments
                    }
                    "RandomLinkSplit" => {
                        let ratio = parse_ratio(split_cfg)?;
                        let seed = split_cfg.get("rnd_seed").and_then(|v| v.as_u64());
                        if is_link_array_batches(&final_batches) {
                            let (transformed_batches, transformed_assignments) =
                                transform_random_link_split_link_array(&final_batches, ratio, seed)?;
                            final_batches = transformed_batches;
                            row_offset = row_count_of_batches(&final_batches) as i64;

                            if let Ok(mut entities) = final_batches
                                .iter()
                                .map(extract_entity_strings)
                                .collect::<anyhow::Result<Vec<_>>>()
                                .map(|chunks| chunks.into_iter().flatten().collect::<Vec<_>>())
                            {
                                entity_values = std::mem::take(&mut entities);
                            }
                            timestamps_ns.clear();
                            include_mask = None;
                            for batch in &final_batches {
                                extend_timestamps_ns_from_batch(batch, &mut timestamps_ns);
                                extend_include_mask_from_batch(batch, &mut include_mask);
                            }

                            transformed_assignments
                        } else {
                            split_assignments::compute_dense_split_assignments_core(
                                "RandomLinkSplit",
                                row_offset as usize,
                                ratio,
                                seed,
                                None,
                            )
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?
                        }
                    }
                    other => {
                        return Err(anyhow::anyhow!(
                            "unsupported split family for diskgraph dense split assignment: {other}"
                        ));
                    }
                };

                if split_authoritative {
                    out = if let Some(mask) = include_mask {
                        split_assignments::apply_entity_include_mask(out, mask)
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    } else {
                        out
                    };
                    Some(out)
                } else {
                    None
                }
            } else {
                None
            };

            // Extract inline col stats from summary (produced by StatsMerger::finalize)
            let train_table_col_stats_json = {
                let summary_val: serde_json::Value = serde_json::from_str(&raw_summary_json)?;
                let inline_stats = summary_val
                    .get("stats")
                    .and_then(|s| s.get("parsed_train_table_col_stats"));
                match inline_stats {
                    Some(stats) if !stats.is_null() => Some(serde_json::to_string(stats)?),
                    _ => None,
                }
            };

            let materialization_start = std::time::Instant::now();
            let rows_written = write_batches_with_arange(&out, &final_batches, None)?;
            if rows_written as usize != row_offset as usize {
                return Err(anyhow::anyhow!(
                    "materialized row count mismatch after split processing (written={}, expected={})",
                    rows_written,
                    row_offset
                ));
            }
            info!(
                final_rows = rows_written,
                duration_ms = materialization_start.elapsed().as_millis(),
                "Final materialization complete"
            );

            // If downsampling actually ran, the inline col stats produced
            // during the executor pass reflect pre-downsample rows, including
            // ones that were dropped. Recompute over the final parquet so the
            // stats artifact matches the rows that survived downsampling.
            let train_table_col_stats_json = if downsampling_applied {
                let split_bytes_ref = split_assignments.as_deref().unwrap_or(&[]);
                if split_bytes_ref.len() == rows_written as usize {
                    let target_stype = plan
                        .target_stype
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!(
                            "post-downsample col stats recompute requires target_stype on plan"
                        ))?;
                    let recompute_start = std::time::Instant::now();
                    let payload = compute_train_table_col_stats_payload(
                        &out,
                        split_bytes_ref,
                        "v1",
                        target_stype,
                    )?;
                    info!(
                        rows = rows_written,
                        duration_ms = recompute_start.elapsed().as_millis(),
                        "Recomputed train_table col stats post-downsample"
                    );
                    Some(serde_json::to_string(&payload)?)
                } else {
                    // Length mismatch: keep inline stats. Should not happen.
                    train_table_col_stats_json
                }
            } else {
                // Downsampling didn't run, inline stats are correct.
                train_table_col_stats_json
            };

            let summary_json = patch_summary_for_final_rows(
                raw_summary_json,
                row_offset as usize,
                if entity_values.len() == row_offset as usize {
                    Some(entity_values.as_slice())
                } else {
                    None
                },
                if target_values.len() == row_offset as usize {
                    Some(target_values.as_slice())
                } else {
                    None
                },
                &downsampling_cfg,
            )?;

            Ok((
                rows_written,
                summary_json,
                split_assignments,
                split_authoritative,
                split_reason,
                downsampling_handled,
                downsampling_applied,
                pre_downsample_row_count as u64,
                train_table_col_stats_json,
            ))
        })
        .map_err(|e| crate::map_anyhow_err(e))?;

    info!(
        total_rows = rows_written,
        overall_duration_ms = overall_start.elapsed().as_millis(),
        "execute_tt_plan_materialize_with_split finished"
    );

    let out = PyDict::new(py);
    out.set_item("rows_written", rows_written)?;
    out.set_item("summary_json", summary_json)?;
    out.set_item("split_family", family)?;
    out.set_item("split_authoritative", split_authoritative)?;
    out.set_item("split_reason", split_reason)?;
    out.set_item("downsampling_handled", downsampling_handled)?;
    out.set_item("downsampling_applied", downsampling_applied)?;
    out.set_item("pre_downsample_row_count", pre_downsample_row_count)?;
    if let Some(bytes) = split_bytes {
        out.set_item("split_assignments", PyBytes::new(py, &bytes))?;
    } else {
        out.set_item("split_assignments", py.None())?;
    }
    if let Some(train_col_stats_json) = train_table_col_stats_json {
        out.set_item("train_table_col_stats_json", train_col_stats_json)?;
    } else {
        out.set_item("train_table_col_stats_json", py.None())?;
    }
    Ok(out.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(row_count, summary_json=None, plan_json=None))]
pub fn build_tt_stats_artifact(
    py: Python<'_>,
    row_count: u64,
    summary_json: Option<&str>,
    plan_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let summary_value = summary_json.and_then(|s| serde_json::from_str::<Value>(s).ok());
    let plan_value = plan_json.and_then(|s| serde_json::from_str::<Value>(s).ok());
    let payload =
        build_tt_stats_artifact_value(summary_value.as_ref(), row_count, plan_value.as_ref());

    let out = PyDict::new(py);
    out.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_path, split_assignments, split_stats_config_json=None))]
pub fn build_split_stats_artifact(
    py: Python<'_>,
    output_path: &str,
    split_assignments: &[u8],
    split_stats_config_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out = output_path.to_owned();
    let assignments = split_assignments.to_vec();
    let cfg_value = split_stats_config_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or_else(|| json!({}));
    let entity_col = cfg_value
        .get("entity_col")
        .and_then(|v| v.as_str())
        .unwrap_or("ENTITY")
        .to_string();
    let target_col = cfg_value
        .get("target_col")
        .and_then(|v| v.as_str())
        .unwrap_or("TARGET")
        .to_string();
    let time_col = cfg_value
        .get("time_col")
        .and_then(|v| v.as_str())
        .unwrap_or("TIMESTAMP")
        .to_string();
    let task_type = cfg_value
        .get("task_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let payload = py
        .detach(|| -> anyhow::Result<Value> {
            let row_count = parquet::file::serialized_reader::SerializedFileReader::new(File::open(&out)?)?
                .metadata()
                .file_metadata()
                .num_rows() as usize;
            if row_count != assignments.len() {
                return Err(anyhow::anyhow!(
                    "split stats row count mismatch while building producer artifact (table_rows={}, assignment_rows={})",
                    row_count,
                    assignments.len()
                ));
            }

            let entity_vals = collect_column_opt_strings(&out, &entity_col)?;
            let target_vals = collect_column_opt_strings(&out, &target_col)?;
            let time_vals = collect_column_opt_strings(&out, &time_col)?;

            let train = build_stage_stats_from_columns(
                &assignments,
                0,
                entity_vals.as_deref(),
                target_vals.as_deref(),
                time_vals.as_deref(),
                task_type.as_deref(),
            );
            if train.is_null() {
                return Err(anyhow::anyhow!(
                    "split stats artifact must include non-empty train_table_stats"
                ));
            }

            Ok(json!({
                "version": 1,
                "kind": "tt_split_stats",
                "train_table_stats": train,
                "val_table_stats": build_stage_stats_from_columns(
                    &assignments,
                    1,
                    entity_vals.as_deref(),
                    target_vals.as_deref(),
                    time_vals.as_deref(),
                    task_type.as_deref(),
                ),
                "test_table_stats": build_stage_stats_from_columns(
                    &assignments,
                    2,
                    entity_vals.as_deref(),
                    target_vals.as_deref(),
                    time_vals.as_deref(),
                    task_type.as_deref(),
                ),
                "provided_fields": ["train_table_stats", "val_table_stats", "test_table_stats"],
            }))
        })
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("build_split_stats_artifact failed: {e:#}")))?;

    let out_dict = PyDict::new(py);
    out_dict.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out_dict.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_dir, train_table_name="custom_train_table", mask_present=true, mask_absent_reason=None, edge_attrs_path=None))]
pub fn build_train_table_edge_bundle_artifact(
    py: Python<'_>,
    output_dir: &str,
    train_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out_dir = output_dir.to_owned();
    let train_table_name = train_table_name.to_owned();
    let edge_attrs_path = edge_attrs_path.map(|s| s.to_owned());
    let mask_absent_reason = mask_absent_reason.map(|s| s.to_owned());

    let payload = py
        .detach(|| -> anyhow::Result<Value> {
            build_train_table_edges_bundle_value(
                Path::new(&out_dir),
                &train_table_name,
                mask_present,
                mask_absent_reason.as_deref(),
                edge_attrs_path.as_deref(),
            )
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "build_train_table_edge_bundle_artifact failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out_dict.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_dir, prediction_table_name="custom_prediction_table", mask_present=true, mask_absent_reason=None, edge_attrs_path=None))]
pub fn build_prediction_table_edge_bundle_artifact(
    py: Python<'_>,
    output_dir: &str,
    prediction_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out_dir = output_dir.to_owned();
    let prediction_table_name = prediction_table_name.to_owned();
    let edge_attrs_path = edge_attrs_path.map(|s| s.to_owned());
    let mask_absent_reason = mask_absent_reason.map(|s| s.to_owned());

    let payload = py
        .detach(|| -> anyhow::Result<Value> {
            build_prediction_table_edges_bundle_value(
                Path::new(&out_dir),
                &prediction_table_name,
                mask_present,
                mask_absent_reason.as_deref(),
                edge_attrs_path.as_deref(),
            )
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "build_prediction_table_edge_bundle_artifact failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out_dict.into())
}

/// Pure Rust core of [`build_train_table_col_stats_artifact`].
///
/// Opens the parquet at `parquet_path`, observes each non-downsampled row
/// (assignment != 255), and returns the inline col-stats payload as a
/// `serde_json::Value`. Safe to call from any thread; no Python interaction.
fn compute_train_table_col_stats_payload(
    parquet_path: &str,
    split_assignments: &[u8],
    data_version: &str,
    target_stype: &str,
) -> anyhow::Result<Value> {
    use diskgraph_core::accumulators::{
        for_each_split_ascii_byte, HyperLogLog, TDigestEstimator, WelfordAccumulator,
    };
    use diskgraph_train_table::ttgen_stats::build_inline_col_stats;

    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // Verify row count
    let file2 = File::open(parquet_path)?;
    let row_count = parquet::file::serialized_reader::SerializedFileReader::new(file2)?
        .metadata()
        .file_metadata()
        .num_rows() as usize;
    if row_count != split_assignments.len() {
        return Err(anyhow::anyhow!(
            "col stats row count mismatch (table_rows={}, assignment_rows={})",
            row_count,
            split_assignments.len()
        ));
    }

    // Find column indices
    let target_idx = schema
        .index_of("TARGET")
        .map_err(|_| anyhow::anyhow!("missing TARGET column"))?;
    let entity_idx = schema
        .index_of("ENTITY")
        .map_err(|_| anyhow::anyhow!("missing ENTITY column"))?;
    let timestamp_idx = schema
        .index_of("TIMESTAMP")
        .map_err(|_| anyhow::anyhow!("missing TIMESTAMP column"))?;
    let frequency_idx = schema.index_of("__kumo_internal_frequency_column__").ok();
    let include_bit_idx = schema.index_of("ENTITY_INCLUDE_BIT").ok();

    let target_is_categorical = matches!(
        target_stype,
        "categorical" | "boolean" | "ID" | "multicategorical"
    );
    let target_is_numerical = !target_is_categorical;
    let is_multicat = target_stype == "multicategorical";

    // Initialize accumulators
    let mut target_welford = if target_is_numerical {
        Some(WelfordAccumulator::new())
    } else {
        None
    };
    let mut target_tdigest = if target_is_numerical {
        Some(TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA))
    } else {
        None
    };
    let mut target_hll = Some(HyperLogLog::new());
    let mut target_invalid_count: u64 = 0;
    let mut target_category_counts: Option<HashMap<String, u64>> = if target_is_categorical {
        Some(HashMap::new())
    } else {
        None
    };

    let mut entity_hll = Some(HyperLogLog::new());
    let mut entity_welford = Some(WelfordAccumulator::new());
    let mut entity_tdigest = Some(TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA));
    let mut entity_category_counts: HashMap<String, u64> = HashMap::new();

    let mut timestamp_welford = Some(WelfordAccumulator::new());
    let mut timestamp_tdigest = Some(TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA));
    let mut timestamp_hll = Some(HyperLogLog::new());

    let mut frequency_welford: Option<WelfordAccumulator> = if frequency_idx.is_some() {
        Some(WelfordAccumulator::new())
    } else {
        None
    };
    let mut frequency_tdigest: Option<TDigestEstimator> = if frequency_idx.is_some() {
        Some(TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA))
    } else {
        None
    };
    let mut frequency_hll: Option<HyperLogLog> = if frequency_idx.is_some() {
        Some(HyperLogLog::new())
    } else {
        None
    };
    let mut frequency_invalid_count: u64 = 0;

    let mut include_true_count: u64 = 0;
    let mut include_false_count: u64 = 0;

    let mut num_total_rows: u64 = 0;
    let mut global_row: usize = 0;
    // Detected once from the first batch's TARGET column. Required for boolean
    // CATEGORY_COUNTS to round-trip as JSON booleans (matching GE/Spark).
    let mut target_is_bool_dtype: bool = false;

    let mut reader = builder.build()?;
    while let Some(batch_res) = reader.next() {
        let batch = batch_res?;
        let n = batch.num_rows();

        let target_col = batch.column(target_idx);
        let entity_col = batch.column(entity_idx);
        let timestamp_col = batch.column(timestamp_idx);
        let frequency_col = frequency_idx.map(|i| batch.column(i));
        let include_bit_col = include_bit_idx.map(|i| batch.column(i));
        if !target_is_bool_dtype
            && target_col.as_any().downcast_ref::<BooleanArray>().is_some()
        {
            target_is_bool_dtype = true;
        }

        for row in 0..n {
            let assign = split_assignments[global_row];
            global_row += 1;

            // Skip downsampled-out rows (assignment == 255).
            // Compute stats on all selected rows (train + val + test),
            // matching GE/Spark behavior.
            if assign == 255 {
                continue;
            }
            num_total_rows += 1;

            // --- ENTITY ---
            let entity_str = if let Some(a) = entity_col.as_any().downcast_ref::<StringArray>() {
                if a.is_null(row) {
                    String::new()
                } else {
                    a.value(row).to_string()
                }
            } else if let Some(a) = entity_col.as_any().downcast_ref::<Int64Array>() {
                if a.is_null(row) {
                    String::new()
                } else {
                    a.value(row).to_string()
                }
            } else if let Some(a) = entity_col.as_any().downcast_ref::<Float64Array>() {
                if a.is_null(row) {
                    String::new()
                } else {
                    a.value(row).to_string()
                }
            } else {
                String::new()
            };

            // Entity HLL
            if let Some(ref mut h) = entity_hll {
                h.observe_bytes(entity_str.as_bytes());
            }

            // Entity category counts
            *entity_category_counts.entry(entity_str.clone()).or_insert(0) += 1;

            // Entity Welford + TDigest (for integer entities)
            if let Some(a) = entity_col.as_any().downcast_ref::<Int64Array>() {
                if !a.is_null(row) {
                    let v = a.value(row) as f64;
                    if let Some(ref mut w) = entity_welford {
                        w.observe(v);
                    }
                    if let Some(ref mut t) = entity_tdigest {
                        t.observe(v);
                    }
                }
            }

            // --- TIMESTAMP ---
            if let Some(a) = timestamp_col.as_any().downcast_ref::<Int64Array>() {
                if !a.is_null(row) {
                    let ts = a.value(row) as f64;
                    if let Some(ref mut w) = timestamp_welford {
                        w.observe(ts);
                    }
                    if let Some(ref mut t) = timestamp_tdigest {
                        t.observe(ts);
                    }
                    if let Some(ref mut h) = timestamp_hll {
                        h.observe_i64(a.value(row));
                    }
                }
            } else if let Some(a) = timestamp_col.as_any().downcast_ref::<Float64Array>() {
                if !a.is_null(row) {
                    let ts = a.value(row);
                    if let Some(ref mut w) = timestamp_welford {
                        w.observe(ts);
                    }
                    if let Some(ref mut t) = timestamp_tdigest {
                        t.observe(ts);
                    }
                    if let Some(ref mut h) = timestamp_hll {
                        h.observe_f64(ts);
                    }
                }
            }

            // --- TARGET ---
            if target_is_numerical {
                let fval = if let Some(a) = target_col.as_any().downcast_ref::<Float64Array>() {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row))
                    }
                } else if let Some(a) = target_col.as_any().downcast_ref::<Int64Array>() {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row) as f64)
                    }
                } else {
                    None
                };
                if let Some(f) = fval {
                    if f.is_finite() {
                        if let Some(ref mut w) = target_welford {
                            w.observe(f);
                        }
                        if let Some(ref mut t) = target_tdigest {
                            t.observe(f);
                        }
                    } else {
                        target_invalid_count += 1;
                    }
                    // HLL for numerical target
                    if let Some(ref mut h) = target_hll {
                        if let Some(a) = target_col.as_any().downcast_ref::<Int64Array>() {
                            h.observe_i64(a.value(row));
                        } else {
                            h.observe_f64(f);
                        }
                    }
                }
            } else {
                // Categorical / multicategorical / boolean target
                let target_str = if let Some(a) =
                    target_col.as_any().downcast_ref::<StringArray>()
                {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                } else if let Some(a) = target_col.as_any().downcast_ref::<Int64Array>() {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                } else if let Some(a) = target_col.as_any().downcast_ref::<Float64Array>() {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                } else if let Some(a) = target_col.as_any().downcast_ref::<BooleanArray>() {
                    if a.is_null(row) {
                        None
                    } else {
                        Some(a.value(row).to_string())
                    }
                } else {
                    None
                };
                if let Some(ts) = target_str {
                    if is_multicat {
                        for_each_split_ascii_byte(&ts, b'|', |tok| {
                            let tok = tok.trim();
                            let hash = HyperLogLog::hash_bytes(tok.as_bytes());
                            if let Some(ref mut h) = target_hll {
                                h.observe_hash(hash);
                            }
                            if let Some(ref mut counts) = target_category_counts {
                                *counts.entry(tok.to_string()).or_insert(0) += 1;
                            }
                        });
                    } else {
                        if let Some(ref mut counts) = target_category_counts {
                            *counts.entry(ts.clone()).or_insert(0) += 1;
                        }
                        if let Some(ref mut h) = target_hll {
                            h.observe_bytes(ts.as_bytes());
                        }
                    }
                }
            }

            // --- FREQUENCY ---
            if let Some(ref freq_col) = frequency_col {
                if let Some(a) = freq_col.as_any().downcast_ref::<Int64Array>() {
                    if !a.is_null(row) {
                        let freq = a.value(row);
                        if let Some(ref mut w) = frequency_welford {
                            w.observe(freq as f64);
                        }
                        if let Some(ref mut t) = frequency_tdigest {
                            t.observe(freq as f64);
                        }
                        if let Some(ref mut h) = frequency_hll {
                            h.observe_i64(freq);
                        }
                    }
                } else if let Some(a) = freq_col.as_any().downcast_ref::<Float64Array>() {
                    if !a.is_null(row) {
                        let freq = a.value(row);
                        if freq.is_finite() {
                            if let Some(ref mut w) = frequency_welford {
                                w.observe(freq);
                            }
                            if let Some(ref mut t) = frequency_tdigest {
                                t.observe(freq);
                            }
                        } else {
                            frequency_invalid_count += 1;
                        }
                        if let Some(ref mut h) = frequency_hll {
                            h.observe_f64(freq);
                        }
                    }
                }
            }

            // --- ENTITY_INCLUDE_BIT ---
            if let Some(ref bit_col) = include_bit_col {
                if let Some(a) = bit_col.as_any().downcast_ref::<BooleanArray>() {
                    if !a.is_null(row) {
                        if a.value(row) {
                            include_true_count += 1;
                        } else {
                            include_false_count += 1;
                        }
                    }
                }
            }
        }
    }

    // If entity_welford has no observations (entity was not Int64), clear it
    if entity_welford.as_ref().is_none_or(|w| w.count == 0) {
        entity_welford = None;
        entity_tdigest = None;
    }

    let result = build_inline_col_stats(
        &target_welford,
        &mut target_tdigest,
        &target_hll,
        target_invalid_count,
        &frequency_welford,
        &mut frequency_tdigest,
        &frequency_hll,
        frequency_invalid_count,
        include_true_count,
        include_false_count,
        num_total_rows,
        data_version,
        target_category_counts,
        target_stype,
        target_is_bool_dtype,
        &entity_hll,
        &entity_welford,
        &mut entity_tdigest,
        Some(entity_category_counts),
        &timestamp_welford,
        &mut timestamp_tdigest,
        &timestamp_hll,
    );

    Ok(result.unwrap_or_else(|| {
        json!({
            "version": 1,
            "kind": "tt_train_table_col_stats",
            "stats_format": "kumo_col_stats_v1",
            "row_count": num_total_rows,
            "stats": {},
            "provided_fields": ["stats"]
        })
    }))
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_path, split_assignments, train_table_col_stats_config_json=None))]
pub fn build_train_table_col_stats_artifact(
    py: Python<'_>,
    output_path: &str,
    split_assignments: &[u8],
    train_table_col_stats_config_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out = output_path.to_owned();
    let assignments = split_assignments.to_vec();
    let cfg_value = train_table_col_stats_config_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or_else(|| json!({}));
    let data_version = cfg_value
        .get("data_version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::DiskgraphConfigError::new_err(
                "build_train_table_col_stats_artifact: data_version is required in \
                 train_table_col_stats_config (no fallback)",
            )
        })?
        .to_string();
    let target_stype = cfg_value
        .get("target_stype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::DiskgraphConfigError::new_err(
                "build_train_table_col_stats_artifact: target_stype is required in \
                 train_table_col_stats_config (no fallback)",
            )
        })?
        .to_string();

    let payload = py
        .detach(|| -> anyhow::Result<Value> {
            compute_train_table_col_stats_payload(
                &out,
                &assignments,
                &data_version,
                &target_stype,
            )
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "build_train_table_col_stats_artifact failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out_dict.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_path, row_count, prediction_table_edges_json, prediction_table_col_stats_json=None, producer_summary_json=None))]
pub fn persist_prediction_artifact_bundle(
    py: Python<'_>,
    output_path: &str,
    row_count: u64,
    prediction_table_edges_json: &str,
    prediction_table_col_stats_json: Option<&str>,
    producer_summary_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out = output_path.to_owned();
    let prediction_table_edges_payload = prediction_table_edges_json.to_string();
    let prediction_table_edges_value: Value =
        serde_json::from_str::<Value>(&prediction_table_edges_payload).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "invalid prediction_table_edges_json: {e}"
            ))
        })?;

    let kind_ok = prediction_table_edges_value
        .get("kind")
        .and_then(|v| v.as_str())
        == Some("tt_prediction_table_edges");
    if !kind_ok {
        return Err(crate::DiskgraphConfigError::new_err(
            "prediction_table_edges_json.kind must be tt_prediction_table_edges",
        ));
    }

    let format_ok = prediction_table_edges_value
        .get("format")
        .and_then(|v| v.as_str())
        == Some("tt_pred_edge_bundle_v1");
    if !format_ok {
        return Err(crate::DiskgraphConfigError::new_err(
            "prediction_table_edges_json.format must be tt_pred_edge_bundle_v1",
        ));
    }

    let prediction_col_stats_payload = prediction_table_col_stats_json.map(|s| s.to_string());
    let prediction_col_stats_value: Option<Value> = match prediction_col_stats_payload.as_ref() {
        Some(s) => Some(serde_json::from_str::<Value>(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "invalid prediction_table_col_stats_json: {e}"
            ))
        })?),
        None => None,
    };
    if let Some(v) = prediction_col_stats_value.as_ref() {
        let kind_ok =
            v.get("kind").and_then(|x| x.as_str()) == Some("tt_prediction_table_col_stats");
        if !kind_ok {
            return Err(crate::DiskgraphConfigError::new_err(
                "prediction_table_col_stats_json.kind must be tt_prediction_table_col_stats",
            ));
        }
        let format_ok = v.get("stats_format").and_then(|x| x.as_str()) == Some("kumo_col_stats_v1");
        if !format_ok {
            return Err(crate::DiskgraphConfigError::new_err(
                "prediction_table_col_stats_json.stats_format must be kumo_col_stats_v1",
            ));
        }
    }

    let producer_summary_value: Value = producer_summary_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or(Value::Null);

    let (manifest_path, artifacts_dir) = py
        .detach(|| -> anyhow::Result<(String, String)> {
            let artifacts_dir = format!("{}.artifacts", out);
            ensure_artifacts_dir(&artifacts_dir)?;
            let manifest_path = format!("{}.prediction.manifest.json", out);
            let edge_bundle_path = format!("{}/prediction_table_edges.bundle.json", artifacts_dir);
            let col_stats_path = format!("{}/stats.prediction_table_col_stats.json", artifacts_dir);

            write_artifact_file(&edge_bundle_path, prediction_table_edges_payload.as_bytes())?;
            if let Some(raw) = prediction_col_stats_payload.as_ref() {
                write_artifact_file(&col_stats_path, raw.as_bytes())?;
            }

            let mut checksums = serde_json::Map::new();
            checksums.insert(
                "prediction_table_sha256".to_string(),
                json!(sha256_hex_path(Path::new(&out))?),
            );
            checksums.insert(
                "prediction_table_edges_sha256".to_string(),
                json!(sha256_hex_bytes(prediction_table_edges_payload.as_bytes())),
            );
            merge_prefixed_checksums(
                &mut checksums,
                &prediction_table_edges_value,
                "prediction_table_edges",
            );
            if let Some(raw) = prediction_col_stats_payload.as_ref() {
                checksums.insert(
                    "prediction_table_col_stats_sha256".to_string(),
                    json!(sha256_hex_bytes(raw.as_bytes())),
                );
                if let Some(stats_value) = prediction_col_stats_value.as_ref() {
                    merge_prefixed_checksums(
                        &mut checksums,
                        stats_value,
                        "prediction_table_col_stats",
                    );
                }
            }

            let mut manifest = serde_json::Map::new();
            manifest.insert("version".to_string(), json!(1));
            manifest.insert("row_count".to_string(), json!(row_count));
            manifest.insert(
                "prediction_table".to_string(),
                artifact_ref(&out, "parquet_with_arange"),
            );
            manifest.insert(
                "prediction_table_edges".to_string(),
                prediction_table_edges_value.clone(),
            );
            if let Some(stats_value) = prediction_col_stats_value.as_ref() {
                manifest.insert(
                    "prediction_table_col_stats".to_string(),
                    stats_value.clone(),
                );
            }

            let mut prediction_artifacts = serde_json::Map::new();
            prediction_artifacts.insert(
                "prediction_table_edges".to_string(),
                artifact_ref(&edge_bundle_path, "tt_prediction_table_edges_json"),
            );
            if prediction_col_stats_payload.is_some() {
                prediction_artifacts.insert(
                    "prediction_table_col_stats".to_string(),
                    artifact_ref(&col_stats_path, "tt_prediction_table_col_stats_json"),
                );
            }
            manifest.insert(
                "prediction_artifacts".to_string(),
                Value::Object(prediction_artifacts),
            );
            manifest.insert("checksums".to_string(), Value::Object(checksums));
            if !producer_summary_value.is_null() {
                manifest.insert("producer_summary".to_string(), producer_summary_value);
            }

            let manifest = Value::Object(manifest);
            write_manifest_json(&manifest_path, &manifest)?;
            Ok((manifest_path, artifacts_dir))
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "persist_prediction_artifact_bundle failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item("manifest_path", manifest_path)?;
    out_dict.set_item("artifacts_dir", artifacts_dir)?;
    Ok(out_dict.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_path, row_count, stats_json, split_stats_json, split_bytes, train_table_col_stats_json, split_family=None, split_authoritative=false, split_reason=None, downsampling_json=None, producer_summary_json=None, train_table_edges_json=None))]
pub fn persist_tt_artifact_bundle(
    py: Python<'_>,
    output_path: &str,
    row_count: u64,
    stats_json: &str,
    split_stats_json: &str,
    split_bytes: &[u8],
    train_table_col_stats_json: &str,
    split_family: Option<&str>,
    split_authoritative: bool,
    split_reason: Option<&str>,
    downsampling_json: Option<&str>,
    producer_summary_json: Option<&str>,
    train_table_edges_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out = output_path.to_owned();
    let split_family_s = split_family.unwrap_or("UnknownSplit").to_string();
    let split_reason_s = split_reason.map(|s| s.to_string());
    let downsampling_value: Value = downsampling_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or_else(|| json!({}));
    let producer_summary_value: Value = producer_summary_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or(Value::Null);
    let train_table_col_stats_payload = train_table_col_stats_json.to_string();
    let train_table_edges_payload = train_table_edges_json.map(|s| s.to_string());
    let train_table_edges_value: Option<Value> = match train_table_edges_payload.as_ref() {
        Some(s) => Some(serde_json::from_str::<Value>(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("invalid train_table_edges_json: {e}"))
        })?),
        None => None,
    };

    let (manifest_path, artifacts_dir) = py
        .detach(|| -> anyhow::Result<(String, String)> {
            let artifacts_dir = format!("{}.artifacts", out);
            ensure_artifacts_dir(&artifacts_dir)?;

            let stats_path = format!("{}/stats.tt.json", artifacts_dir);
            let split_path = format!("{}/split_assignments.row_stage.bin", artifacts_dir);
            let split_stats_path = format!("{}/stats.split.json", artifacts_dir);
            let train_table_col_stats_path =
                format!("{}/stats.train_table_col_stats.json", artifacts_dir);
            let manifest_path = format!("{}.manifest.json", out);

            write_artifact_file(&stats_path, stats_json.as_bytes())?;
            write_artifact_file(&split_path, split_bytes)?;
            write_artifact_file(&split_stats_path, split_stats_json.as_bytes())?;
            write_artifact_file(
                &train_table_col_stats_path,
                train_table_col_stats_payload.as_bytes(),
            )?;

            let mut checksums = serde_json::Map::new();
            checksums.insert(
                "train_table_sha256".to_string(),
                json!(sha256_hex_path(Path::new(&out))?),
            );
            checksums.insert(
                "split_sha256".to_string(),
                json!(sha256_hex_bytes(split_bytes)),
            );
            checksums.insert(
                "stats_sha256".to_string(),
                json!(sha256_hex_bytes(stats_json.as_bytes())),
            );
            checksums.insert(
                "split_stats_sha256".to_string(),
                json!(sha256_hex_bytes(split_stats_json.as_bytes())),
            );
            checksums.insert(
                "train_table_col_stats_sha256".to_string(),
                json!(sha256_hex_bytes(train_table_col_stats_payload.as_bytes())),
            );
            if let Some(edge_bundle) = train_table_edges_value.as_ref() {
                let raw = train_table_edges_payload
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing train_table_edges_payload"))?;
                checksums.insert(
                    "train_table_edges_sha256".to_string(),
                    json!(sha256_hex_bytes(raw.as_bytes())),
                );
                merge_prefixed_checksums(&mut checksums, edge_bundle, "train_table_edges");
            }

            let mut manifest = serde_json::Map::new();
            manifest.insert("version".to_string(), json!(1));
            manifest.insert("row_count".to_string(), json!(row_count));
            manifest.insert(
                "train_table".to_string(),
                artifact_ref(&out, "parquet_with_arange"),
            );
            manifest.insert(
                "split".to_string(),
                json!({
                    "path": split_path,
                    "format": "split_assignment_row_stage_bin",
                    "authoritative": split_authoritative,
                    "family": split_family_s,
                    "reason": split_reason_s,
                }),
            );
            manifest.insert(
                "stats".to_string(),
                artifact_ref(&stats_path, "tt_stats_json"),
            );
            manifest.insert(
                "split_stats".to_string(),
                artifact_ref(&split_stats_path, "tt_split_stats_json"),
            );
            manifest.insert(
                "train_table_col_stats".to_string(),
                artifact_ref(&train_table_col_stats_path, "tt_train_table_col_stats_json"),
            );
            if let Some(edge_bundle) = train_table_edges_value.as_ref() {
                manifest.insert("train_table_edges".to_string(), edge_bundle.clone());
            }
            manifest.insert("checksums".to_string(), Value::Object(checksums));
            manifest.insert("downsampling".to_string(), downsampling_value);
            manifest.insert("producer_summary".to_string(), producer_summary_value);
            let manifest = Value::Object(manifest);

            write_manifest_json(&manifest_path, &manifest)?;

            Ok((manifest_path, artifacts_dir))
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "persist_tt_artifact_bundle failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item("manifest_path", manifest_path)?;
    out_dict.set_item("artifacts_dir", artifacts_dir)?;
    Ok(out_dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_random_link_split_link_array_basic() {
        let target_values = {
            let mut b = ListBuilder::new(StringBuilder::new());
            b.values().append_value("a");
            b.values().append_value("b");
            b.values().append_value("c");
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };

        let freq_values = {
            let mut b = ListBuilder::new(Int64Builder::new());
            b.values().append_value(1);
            b.values().append_value(1);
            b.values().append_value(1);
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY", DataType::Int64, true),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new(
                "TARGET",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "__kumo_internal_frequency_column__",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
            Field::new("ENTITY_INCLUDE_BIT", DataType::Boolean, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1)])),
                Arc::new(TimestampNanosecondArray::from(vec![Some(100)])),
                target_values,
                freq_values,
                Arc::new(BooleanArray::from(vec![Some(true)])),
            ],
        )
        .unwrap();

        let (out_batches, assignments) =
            transform_random_link_split_link_array(&[batch], vec![0.5, 0.25, 0.25], Some(0))
                .unwrap();

        assert_eq!(row_count_of_batches(&out_batches), assignments.len());
        assert_eq!(assignments.iter().filter(|x| **x == 0).count(), 1);
        assert_eq!(assignments.iter().filter(|x| **x == 1).count(), 1);
        assert_eq!(assignments.iter().filter(|x| **x == 2).count(), 1);
    }

    #[test]
    fn test_transform_random_link_split_link_array_preserves_weight_column() {
        let target_values = {
            let mut b = ListBuilder::new(StringBuilder::new());
            b.values().append_value("a");
            b.values().append_value("b");
            b.values().append_value("c");
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };

        let freq_values = {
            let mut b = ListBuilder::new(Int64Builder::new());
            b.values().append_value(1);
            b.values().append_value(1);
            b.values().append_value(1);
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY", DataType::Int64, true),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new(
                "TARGET",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "__kumo_internal_frequency_column__",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
            Field::new("WEIGHT", DataType::Float64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1)])),
                Arc::new(TimestampNanosecondArray::from(vec![Some(100)])),
                target_values,
                freq_values,
                Arc::new(Float64Array::from(vec![Some(2.5)])),
            ],
        )
        .unwrap();

        let (out_batches, assignments) =
            transform_random_link_split_link_array(&[batch], vec![0.5, 0.25, 0.25], Some(0))
                .unwrap();

        assert_eq!(row_count_of_batches(&out_batches), assignments.len());
        let out = &out_batches[0];
        let widx = out.schema().index_of("WEIGHT").unwrap();
        let warr = out
            .column(widx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(warr.len(), 3);
        for i in 0..warr.len() {
            assert!(!warr.is_null(i));
            assert!((warr.value(i) - 2.5).abs() < 1e-9);
        }
    }

    fn make_custom_link_split_batch(targets: &[&str], split_labels: &[&str]) -> RecordBatch {
        let target_values = {
            let mut b = ListBuilder::new(StringBuilder::new());
            for target in targets {
                b.values().append_value(target);
            }
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };
        let freq_values = {
            let mut b = ListBuilder::new(Int64Builder::new());
            for _ in targets {
                b.values().append_value(1);
            }
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };
        let split_values = {
            let mut b = ListBuilder::new(StringBuilder::new());
            for label in split_labels {
                b.values().append_value(label);
            }
            b.append(true);
            Arc::new(b.finish()) as Arc<dyn Array>
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY", DataType::Int64, true),
            Field::new(
                "TIMESTAMP",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new(
                "TARGET",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "__kumo_internal_frequency_column__",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
            Field::new(
                "__kumo_internal_split_column__",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1)])),
                Arc::new(TimestampNanosecondArray::from(vec![Some(100)])),
                target_values,
                freq_values,
                split_values,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_transform_custom_link_split_link_array_basic() {
        let batch = make_custom_link_split_batch(&["a", "b", "c"], &["train", "val", "train"]);
        let (out_batches, assignments) = transform_custom_link_split_link_array(
            &[batch],
            "__kumo_internal_split_column__",
            vec!["train".to_string()],
            vec!["val".to_string()],
            vec!["test".to_string()],
        )
        .unwrap();

        assert_eq!(row_count_of_batches(&out_batches), assignments.len());
        assert_eq!(assignments, vec![0u8, 1]);
        let out = &out_batches[0];
        assert!(out.schema().index_of("__kumo_internal_split_column__").is_err());
        let stage_idx = out.schema().index_of("__internal_kumo_split_id_col__").unwrap();
        let stage_arr = out
            .column(stage_idx)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        assert_eq!(stage_arr.values(), &[0u8, 1]);
    }

    #[test]
    fn test_transform_custom_link_split_rejects_misaligned_arrays() {
        let batch = make_custom_link_split_batch(&["a", "b", "c"], &["train", "val"]);
        let err = transform_custom_link_split_link_array(
            &[batch],
            "__kumo_internal_split_column__",
            vec!["train".to_string()],
            vec!["val".to_string()],
            vec!["test".to_string()],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("split-column list length mismatch with TARGET"));
    }

    #[test]
    fn test_build_train_table_edges_bundle_value_basic() {
        let root = std::env::temp_dir().join(format!("dg_tt_edges_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();

        let fkey = root.join("custom_train_table_fkey_cols.parquet");
        std::fs::create_dir_all(&fkey).unwrap();
        std::fs::write(fkey.join("part-0000.parquet"), b"fkey").unwrap();

        let mask = root.join("custom_train_table_mask.parquet");
        std::fs::create_dir_all(&mask).unwrap();
        std::fs::write(mask.join("part-0000.parquet"), b"mask").unwrap();

        let edge = root
            .join("users__rev_user_id__custom_train_table")
            .join("edge.parquet");
        std::fs::create_dir_all(&edge).unwrap();
        std::fs::write(edge.join("part-0000.parquet"), b"edge").unwrap();

        let payload =
            build_train_table_edges_bundle_value(&root, "custom_train_table", true, None, None)
                .unwrap();

        assert_eq!(
            payload.get("format").and_then(|v| v.as_str()),
            Some("tt_edge_bundle_v1")
        );
        assert_eq!(
            payload.get("kind").and_then(|v| v.as_str()),
            Some("tt_train_table_edges")
        );
        assert!(payload
            .get("reverse_edges")
            .and_then(|v| v.as_object())
            .map(|m| m.contains_key("users__rev_user_id__custom_train_table"))
            .unwrap_or(false));
        assert!(payload
            .get("checksums")
            .and_then(|v| v.as_object())
            .map(|m| m.contains_key("fkey_cols_sha256") && m.contains_key("mask_sha256"))
            .unwrap_or(false));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_build_prediction_table_edges_bundle_value_basic() {
        let root = std::env::temp_dir().join(format!("dg_pred_edges_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();

        let fkey = root.join("custom_prediction_table_fkey_cols.parquet");
        std::fs::create_dir_all(&fkey).unwrap();
        std::fs::write(fkey.join("part-0000.parquet"), b"fkey").unwrap();

        let edge = root
            .join("users__rev_user_id__custom_prediction_table")
            .join("edge.parquet");
        std::fs::create_dir_all(&edge).unwrap();
        std::fs::write(edge.join("part-0000.parquet"), b"edge").unwrap();

        let payload = build_prediction_table_edges_bundle_value(
            &root,
            "custom_prediction_table",
            false,
            Some("mask_not_required"),
            None,
        )
        .unwrap();

        assert_eq!(
            payload.get("format").and_then(|v| v.as_str()),
            Some("tt_pred_edge_bundle_v1")
        );
        assert_eq!(
            payload.get("kind").and_then(|v| v.as_str()),
            Some("tt_prediction_table_edges")
        );
        assert_eq!(
            payload
                .get("mask")
                .and_then(|v| v.get("present"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_to_nanoseconds_seconds() {
        assert_eq!(to_nanoseconds(1, "seconds").unwrap(), 1_000_000_000);
        assert_eq!(to_nanoseconds(0, "seconds").unwrap(), 0);
        assert_eq!(
            to_nanoseconds(1735689600, "seconds").unwrap(),
            1735689600_000_000_000
        );
    }

    #[test]
    fn test_to_nanoseconds_milliseconds() {
        assert_eq!(to_nanoseconds(1, "milliseconds").unwrap(), 1_000_000);
        assert_eq!(to_nanoseconds(1000, "milliseconds").unwrap(), 1_000_000_000);
    }

    #[test]
    fn test_to_nanoseconds_microseconds() {
        assert_eq!(to_nanoseconds(1, "microseconds").unwrap(), 1_000);
        assert_eq!(
            to_nanoseconds(1_000_000, "microseconds").unwrap(),
            1_000_000_000
        );
    }

    #[test]
    fn test_to_nanoseconds_passthrough() {
        assert_eq!(to_nanoseconds(42, "nanoseconds").unwrap(), 42);
    }

    #[test]
    fn test_to_nanoseconds_unknown_unit() {
        assert!(to_nanoseconds(1, "fortnights").is_err());
    }

    #[test]
    fn test_get_timestamp_unit_present() {
        let cfg: Value = serde_json::json!({"timestamp_unit": "milliseconds"});
        assert_eq!(get_timestamp_unit(&cfg), "milliseconds");
    }

    #[test]
    fn test_get_timestamp_unit_missing_defaults_to_seconds() {
        let cfg: Value = serde_json::json!({});
        assert_eq!(get_timestamp_unit(&cfg), "seconds");
    }

    #[test]
    fn test_parse_time_ranges_ns_converts_seconds() {
        // Ranges in epoch seconds, timestamp_unit = "seconds"
        let cfg: Value = serde_json::json!({
            "ranges": [[1000, 2000], [2000, 3000]],
            "timestamp_unit": "seconds"
        });
        let (starts, ends) = parse_time_ranges_ns(&cfg, "seconds").unwrap();
        assert_eq!(starts, vec![1000_000_000_000i64, 2000_000_000_000i64]);
        assert_eq!(ends, vec![2000_000_000_000i64, 3000_000_000_000i64]);
    }

    #[test]
    fn test_parse_time_ranges_ns_nanoseconds_passthrough() {
        let cfg: Value = serde_json::json!({
            "ranges": [[1_000_000_000i64, 2_000_000_000i64], [2_000_000_000i64, 3_000_000_000i64]],
            "timestamp_unit": "nanoseconds"
        });
        let (starts, ends) = parse_time_ranges_ns(&cfg, "nanoseconds").unwrap();
        assert_eq!(starts, vec![1_000_000_000i64, 2_000_000_000i64]);
        assert_eq!(ends, vec![2_000_000_000i64, 3_000_000_000i64]);
    }
}
