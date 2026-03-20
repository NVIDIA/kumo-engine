//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` that returns a
//! `TTPlanIterator` yielding `pyarrow.RecordBatch` objects one at a time.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

use arrow::compute::take;
use arrow::pyarrow::ToPyArrow;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use crate::split_assignments;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::reader::FileReader;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use crossbeam_channel::Receiver;
use diskgraph_core::tt_builder::DiskGraphTTPlan;
use diskgraph_train_table::ttgen_stats::TTExecutionSummary;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use pyo3::Py;

#[pyclass]
pub struct TTPlanIterator {
    receiver: Option<Receiver<RecordBatch>>,
    handle: Option<JoinHandle<anyhow::Result<TTExecutionSummary>>>,
    summary: Option<TTExecutionSummary>,
}

impl TTPlanIterator {
    fn finish_execution(&mut self) -> Result<(), String> {
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
                Ok(Err(e)) => Err(format!("TT-gen executor failed: {e}")),
                Err(_) => Err("TT-gen executor thread panicked".to_string()),
            }
        } else {
            Ok(())
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
                self.finish_execution().map_err(PyRuntimeError::new_err)?;
                Err(PyStopIteration::new_err(()))
            }
        }
    }

    /// Returns the final execution summary.
    ///
    /// If called before stream exhaustion, this forces finalization and blocks
    /// until all workers complete.
    fn execution_summary(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.finish_execution().map_err(PyRuntimeError::new_err)?;

        let out = PyDict::new(py);
        if let Some(summary) = &self.summary {
            out.set_item("rows_written", summary.rows_written)?;
            if let Some(stats) = &summary.stats {
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
                    split_dict.set_item(
                        "supports_native_execution",
                        split.supports_native_execution,
                    )?;
                    split_dict.set_item("split_kind", split.split_kind.as_deref())?;
                    split_dict.set_item(
                        "derivation_method",
                        split.derivation_method.as_deref(),
                    )?;
                    split_dict.set_item("num_total_rows", split.num_total_rows)?;
                    if let Some(art) = &split.artifacts_v3 {
                        let art_dict = PyDict::new(py);
                        art_dict.set_item("strategy", art.strategy.as_str())?;
                        art_dict.set_item("split_kind", art.split_kind.as_deref())?;
                        art_dict.set_item("ratio", art.ratio.clone())?;
                        art_dict.set_item("offsets", art.offsets.clone())?;
                        art_dict.set_item("ranges", art.ranges.clone())?;
                        art_dict.set_item(
                            "backtesting_timestamp",
                            art.backtesting_timestamp,
                        )?;
                        art_dict.set_item("split_column", art.split_column.as_deref())?;
                        art_dict.set_item(
                            "manual_train_values",
                            art.manual_train_values.clone(),
                        )?;
                        art_dict.set_item("manual_val_values", art.manual_val_values.clone())?;
                        art_dict.set_item(
                            "manual_test_values",
                            art.manual_test_values.clone(),
                        )?;
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
                            art_dict.set_item(
                                "timestamp_stage_assignments",
                                py_assignments,
                            )?;
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

                out.set_item("stats", stats_dict)?;
            } else {
                out.set_item("stats", py.None())?;
            }
        } else {
            out.set_item("rows_written", 0u64)?;
            out.set_item("stats", py.None())?;
        }

        Ok(out.into())
    }

}

#[pyfunction]
pub fn execute_tt_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
) -> PyResult<TTPlanIterator> {
    let plan: DiskGraphTTPlan = serde_json::from_str(plan_json)
        .map_err(|e| PyValueError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();

    let (receiver, handle) = py
        .detach(|| {
            diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))
        })
        .map_err(|e| PyValueError::new_err(format!("execute failed: {e:#}")))?;

    Ok(TTPlanIterator {
        receiver: Some(receiver),
        handle: Some(handle),
        summary: None,
    })
}

fn append_arange_column(batch: RecordBatch, row_offset: i64) -> anyhow::Result<RecordBatch> {
    let num_rows = batch.num_rows() as i64;
    let arange = Int64Array::from_iter_values(row_offset..(row_offset + num_rows));

    let mut fields = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("arange", DataType::Int64, false));

    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(arange));

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| anyhow::anyhow!("failed to append arange column: {e}"))
}

/// Distribute `total` items into `buckets` using largest-remainder method.
/// Guarantees the result sums to exactly `total`.
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
            writer = Some(ArrowWriter::try_new(file.try_clone()?, with_idx.schema(), Some(props.clone()))?);
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

fn sha256_hex_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sha256_hex_file(path: &str) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut f = File::open(path)?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{:02x}", b)).collect())
}

fn split_family_from_config(split_family_hint: Option<&str>, split_cfg: Option<&Value>) -> Option<String> {
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
        .map(|v| v.as_f64().ok_or_else(|| anyhow::anyhow!("split ratio must be numeric")))
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

fn parse_time_ranges_ns(split_cfg: &Value) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
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
        starts.push(parse_timestamp_ns_json(&p[0])?);
        ends.push(parse_timestamp_ns_json(&p[1])?);
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
    match split_cfg.get("col") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Object(m)) => m
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
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

    stats.insert("num_total_rows".to_string(), Value::from(final_row_count as u64));
    stats.insert("num_train_rows".to_string(), Value::from(final_row_count as u64));

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

    let temporal_timestamps = plan
        .and_then(|p| p.get("temporal"))
        .and_then(|t| t.get("timestamps"))
        .and_then(|arr| arr.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|v| v.as_i64())
        .collect::<Vec<_>>();
    let is_temporal = !temporal_timestamps.is_empty();
    let num_timeframes = if is_temporal {
        temporal_timestamps.len() as u64
    } else {
        timestamp_unique.unwrap_or(0)
    };
    let num_timeframes_post_filter = num_timeframes;
    let avg_entities_per_timeframe = if num_timeframes_post_filter > 0 {
        (num_train_rows as f64 / num_timeframes_post_filter as f64).round() as u64
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

    let mut stats_per_timeframe: Vec<Value> = Vec::new();
    if is_temporal {
        let mut ts_values = temporal_timestamps.clone();
        if ts_values.is_empty() {
            ts_values = vec![timestamp_max, timestamp_max];
        } else if ts_values.len() == 1 {
            ts_values.push(ts_values[0]);
        }
        let per_timeframe_rows = largest_remainder_distribute(num_train_rows, ts_values.len());
        for (idx, ts) in ts_values.iter().enumerate() {
            let rows_for_tf = per_timeframe_rows.get(idx).copied().unwrap_or(0);
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
        "node_degree_stats",
        "stats_per_timeframe",
        "target_distribution",
        "scoring_table_column_stats",
    ];
    if num_links.is_some() {
        provided_fields.push("num_links");
    }
    if !payload.get("exact_stats").map(|v| v.is_null()).unwrap_or(true) {
        provided_fields.push("exact_stats");
    }
    if !payload.get("split_payload").map(|v| v.is_null()).unwrap_or(true) {
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
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i).to_string()) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i).to_string()) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i).to_string()) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i).to_string()) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i).to_string()) })
            .collect());
    }
    Err(anyhow::anyhow!("unsupported column dtype for split stats"))
}

fn collect_column_opt_strings(path: &str, col_name: &str) -> anyhow::Result<Option<Vec<Option<String>>>> {
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
            return Ok(dt.with_timezone(&Utc).timestamp_nanos_opt().ok_or_else(|| anyhow::anyhow!("timestamp ns overflow"))?);
        }
        if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let dt = d
                .and_hms_opt(0, 0, 0)
                .ok_or_else(|| anyhow::anyhow!("invalid date"))?
                .and_utc();
            return Ok(dt.timestamp_nanos_opt().ok_or_else(|| anyhow::anyhow!("timestamp ns overflow"))?);
        }
    }
    Err(anyhow::anyhow!("timestamp must be numeric epoch-ns or parseable date string"))
}

fn extract_sort_values(batch: &RecordBatch, col_name: &str) -> anyhow::Result<Vec<SortValue>> {
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| anyhow::anyhow!("ColumnSplit missing split column: {col_name}"))?;
    let arr = batch.column(idx);

    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { SortValue::Null } else { SortValue::Int(a.value(i)) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { SortValue::Null } else { SortValue::Float(a.value(i)) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { SortValue::Null } else { SortValue::Str(a.value(i).to_string()) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { SortValue::Null } else { SortValue::Bool(a.value(i)) })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { SortValue::Null } else { SortValue::Ts(a.value(i)) })
            .collect());
    }

    Err(anyhow::anyhow!("ColumnSplit unsupported split column dtype for {col_name}"))
}

fn extract_string_values(batch: &RecordBatch, col_name: &str) -> anyhow::Result<Vec<String>> {
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| anyhow::anyhow!("ManualSplit missing split column: {col_name}"))?;
    let arr = batch.column(idx);

    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { "None".to_string() } else { a.value(i).to_string() })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { "None".to_string() } else { a.value(i).to_string() })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { "None".to_string() } else { a.value(i).to_string() })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { "None".to_string() } else { a.value(i).to_string() })
            .collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..a.len())
            .map(|i| if a.is_null(i) { "None".to_string() } else { a.value(i).to_string() })
            .collect());
    }

    Err(anyhow::anyhow!("ManualSplit unsupported split column dtype for {col_name}"))
}

#[pyfunction]
#[pyo3(signature=(plan_json, index_path, output_path, split_config_json=None, split_family_hint=None, downsampling_config_json=None))]
pub fn execute_tt_plan_materialize_with_split(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_path: &str,
    split_config_json: Option<&str>,
    split_family_hint: Option<&str>,
    downsampling_config_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let plan: DiskGraphTTPlan = serde_json::from_str(plan_json)
        .map_err(|e| PyValueError::new_err(format!("invalid plan JSON: {e}")))?;

    let split_cfg: Option<Value> = match split_config_json {
        Some(s) if !s.is_empty() => Some(
            serde_json::from_str(s)
                .map_err(|e| PyValueError::new_err(format!("invalid split config JSON: {e}")))?,
        ),
        _ => None,
    };

    let family = split_family_from_config(split_family_hint, split_cfg.as_ref())
        .unwrap_or_else(|| "UnknownSplit".to_string());

    let split_column_name = split_cfg.as_ref().and_then(parse_split_column_name);
    let downsampling_cfg: Option<Value> = match downsampling_config_json {
        Some(s) if !s.is_empty() => Some(
            serde_json::from_str(s)
                .map_err(|e| PyValueError::new_err(format!("invalid downsampling config JSON: {e}")))?,
        ),
        _ => None,
    };
    let downsampling_cfg = parse_downsampling_config(downsampling_cfg.as_ref())
        .map_err(|e| PyValueError::new_err(format!("invalid downsampling config: {e}")))?;

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
    ) = py
        .detach(|| -> anyhow::Result<(u64, String, Option<Vec<u8>>, bool, Option<String>, bool, bool, u64)> {
            let (receiver, handle) =
                diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))?;

            let mut row_offset = 0i64;
            let mut timestamps_ns: Vec<i64> = Vec::new();
            let mut include_mask: Option<Vec<bool>> = None;
            let mut column_sort_values: Vec<SortValue> = Vec::new();
            let mut manual_values: Vec<String> = Vec::new();
            let mut entity_values: Vec<String> = Vec::new();
            let mut target_values: Vec<String> = Vec::new();
            let mut raw_batches: Vec<RecordBatch> = Vec::new();

            for batch in receiver.iter() {
                if let Some(ts_arr) = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                {
                    timestamps_ns.extend(ts_arr.values().iter().copied());
                }

                if include_mask.is_none() {
                    if let Ok(idx) = batch.schema().index_of("ENTITY_INCLUDE_BIT") {
                        include_mask = Some(Vec::new());
                        if let Some(mask_arr) = batch
                            .column(idx)
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                        {
                            if let Some(mask) = include_mask.as_mut() {
                                mask.extend(mask_arr.values().iter());
                            }
                        }
                    }
                } else if let Some(mask) = include_mask.as_mut() {
                    if let Ok(idx) = batch.schema().index_of("ENTITY_INCLUDE_BIT") {
                        if let Some(mask_arr) = batch
                            .column(idx)
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                        {
                            mask.extend(mask_arr.values().iter());
                        }
                    }
                }

                if family == "ColumnSplit" {
                    let col_name = split_column_name
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("ColumnSplit requires split.col name"))?;
                    let mut vals = extract_sort_values(&batch, col_name)?;
                    column_sort_values.append(&mut vals);
                } else if family == "ManualSplit" {
                    let col_name = split_column_name
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("ManualSplit requires split.col name"))?;
                    let mut vals = extract_string_values(&batch, col_name)?;
                    manual_values.append(&mut vals);
                }

                if let Ok(mut entities) = extract_entity_strings(&batch) {
                    entity_values.append(&mut entities);
                }
                if let Ok(mut targets) = extract_string_values(&batch, "TARGET") {
                    target_values.append(&mut targets);
                }

                row_offset += batch.num_rows() as i64;
                raw_batches.push(batch);
            }

            if row_offset == 0 {
                return Err(anyhow::anyhow!("diskgraph execute_tt_plan returned zero rows"));
            }

            let pre_downsample_row_count = row_offset as usize;
            let mut selected_indices: Option<Vec<usize>> = None;
            let mut downsampling_applied = false;
            let mut downsampling_handled = false;

            if downsampling_cfg.enabled {
                downsampling_handled = true;
                if downsampling_cfg.is_link_prediction
                    && !downsampling_cfg.list_distinct_target_is_array
                    && entity_values.len() != pre_downsample_row_count
                {
                    return Err(anyhow::anyhow!(
                        "downsampling requires ENTITY column for link_by_entity path"
                    ));
                }
                let (idxs, _path) = downsample_indices_legacy(
                    &downsampling_cfg,
                    &entity_values,
                    pre_downsample_row_count,
                )?;
                if idxs.len() != pre_downsample_row_count {
                    downsampling_applied = true;
                    selected_indices = Some(idxs);
                }
            }

            if downsampling_cfg.enabled && downsampling_cfg.preserve_pre_downsample_tt {
                let pre_path = format!("{}.pre_downsample.parquet", out);
                let _ = write_batches_with_arange(&pre_path, &raw_batches, None)?;
            }

            let final_rows = if let Some(idxs) = selected_indices.as_ref() {
                write_batches_with_arange(&out, &raw_batches, Some(idxs))?
            } else {
                write_batches_with_arange(&out, &raw_batches, None)?
            } as usize;

            if let Some(idxs) = selected_indices.as_ref() {
                timestamps_ns = idxs.iter().map(|i| timestamps_ns[*i]).collect();
                if let Some(mask) = include_mask.as_ref() {
                    include_mask = Some(idxs.iter().map(|i| mask[*i]).collect());
                }
                if !column_sort_values.is_empty() {
                    column_sort_values = idxs.iter().map(|i| column_sort_values[*i].clone()).collect();
                }
                if !manual_values.is_empty() {
                    manual_values = idxs.iter().map(|i| manual_values[*i].clone()).collect();
                }
                if entity_values.len() == pre_downsample_row_count {
                    entity_values = idxs.iter().map(|i| entity_values[*i].clone()).collect();
                }
                if target_values.len() == pre_downsample_row_count {
                    target_values = idxs.iter().map(|i| target_values[*i].clone()).collect();
                }
                row_offset = final_rows as i64;
            }

            let summary = handle
                .join()
                .map_err(|_| anyhow::anyhow!("TT-gen summary thread panicked"))??;
            let mut summary_json = serde_json::to_string(&summary)?;
            summary_json = patch_summary_for_final_rows(
                summary_json,
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

            let mut split_authoritative = true;
            let mut split_reason: Option<String> = None;
            let split_assignments = if let Some(split_cfg) = split_cfg.as_ref() {
                let out = match family.as_str() {
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
                        let seed = split_cfg.get("seed").and_then(|v| v.as_u64());
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
                        let (starts, ends) = parse_time_ranges_ns(split_cfg)?;
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
                        let ts_val = split_cfg
                            .get("backtesting_timestamp")
                            .ok_or_else(|| anyhow::anyhow!("BacktestingSplit requires backtesting_timestamp"))?;
                        let ts = parse_timestamp_ns_json(ts_val)?;
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
                    "ManualSplit" => {
                        if manual_values.len() != row_offset as usize {
                            return Err(anyhow::anyhow!(
                                "ManualSplit collected values mismatch rows (values={}, rows={})",
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
                    "RandomLinkSplit" => {
                        return Err(anyhow::anyhow!(
                            "RandomLinkSplit is not supported for diskgraph dense split assignment"
                        ));
                    }
                    other => {
                        return Err(anyhow::anyhow!(
                            "unsupported split family for diskgraph dense split assignment: {other}"
                        ));
                    }
                };

                if split_authoritative {
                    let out = if let Some(mask) = include_mask {
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

            Ok((
                row_offset as u64,
                summary_json,
                split_assignments,
                split_authoritative,
                split_reason,
                downsampling_handled,
                downsampling_applied,
                pre_downsample_row_count as u64,
            ))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("materialize_with_split failed: {e:#}")))?;

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
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature=(row_count, summary_json=None, plan_json=None))]
pub fn build_tt_stats_artifact(
    py: Python<'_>,
    row_count: u64,
    summary_json: Option<&str>,
    plan_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let summary_value = summary_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok());
    let plan_value = plan_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok());
    let payload = build_tt_stats_artifact_value(
        summary_value.as_ref(),
        row_count,
        plan_value.as_ref(),
    );

    let out = PyDict::new(py);
    out.set_item("payload_json", serde_json::to_string(&payload).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)?;
    Ok(out.into())
}

#[pyfunction]
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
        .map_err(|e| PyRuntimeError::new_err(format!("build_split_stats_artifact failed: {e:#}")))?;

    let out_dict = PyDict::new(py);
    out_dict.set_item("payload_json", serde_json::to_string(&payload).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)?;
    Ok(out_dict.into())
}

#[pyfunction]
#[pyo3(signature=(output_path, row_count, stats_json, split_stats_json, split_bytes, split_family=None, split_authoritative=false, split_reason=None, downsampling_json=None, producer_summary_json=None))]
pub fn persist_tt_artifact_bundle(
    py: Python<'_>,
    output_path: &str,
    row_count: u64,
    stats_json: &str,
    split_stats_json: &str,
    split_bytes: &[u8],
    split_family: Option<&str>,
    split_authoritative: bool,
    split_reason: Option<&str>,
    downsampling_json: Option<&str>,
    producer_summary_json: Option<&str>,
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

    let (manifest_path, artifacts_dir) = py
        .detach(|| -> anyhow::Result<(String, String)> {
            let artifacts_dir = format!("{}.artifacts", out);
            std::fs::create_dir_all(&artifacts_dir)?;

            let stats_path = format!("{}/stats.tt.json", artifacts_dir);
            let split_path = format!("{}/split_assignments.row_stage.bin", artifacts_dir);
            let split_stats_path = format!("{}/stats.split.json", artifacts_dir);
            let manifest_path = format!("{}.manifest.json", out);

            {
                let mut f = File::create(&stats_path)?;
                f.write_all(stats_json.as_bytes())?;
            }
            {
                let mut f = File::create(&split_path)?;
                f.write_all(split_bytes)?;
            }
            {
                let mut f = File::create(&split_stats_path)?;
                f.write_all(split_stats_json.as_bytes())?;
            }

            let manifest = json!({
                "version": 1,
                "row_count": row_count,
                "train_table": {"path": out, "format": "parquet_with_arange"},
                "split": {
                    "path": split_path,
                    "format": "split_assignment_row_stage_bin",
                    "authoritative": split_authoritative,
                    "family": split_family_s,
                    "reason": split_reason_s,
                },
                "stats": {"path": stats_path, "format": "tt_stats_json"},
                "split_stats": {"path": split_stats_path, "format": "tt_split_stats_json"},
                "checksums": {
                    "train_table_sha256": sha256_hex_file(&out)?,
                    "split_sha256": sha256_hex_bytes(split_bytes),
                    "stats_sha256": sha256_hex_bytes(stats_json.as_bytes()),
                    "split_stats_sha256": sha256_hex_bytes(split_stats_json.as_bytes()),
                },
                "downsampling": downsampling_value,
                "producer_summary": producer_summary_value,
            });

            let mut f = File::create(&manifest_path)?;
            f.write_all(serde_json::to_string(&manifest)?.as_bytes())?;

            Ok((manifest_path, artifacts_dir))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("persist_tt_artifact_bundle failed: {e:#}")))?;

    let out_dict = PyDict::new(py);
    out_dict.set_item("manifest_path", manifest_path)?;
    out_dict.set_item("artifacts_dir", artifacts_dir)?;
    Ok(out_dict.into())
}
