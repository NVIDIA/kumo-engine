//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` that returns a
//! `TTPlanIterator` yielding `pyarrow.RecordBatch` objects one at a time.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

use arrow::pyarrow::ToPyArrow;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use crate::split_assignments;
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use crossbeam_channel::Receiver;
use diskgraph_core::tt_builder::DiskGraphTTPlan;
use diskgraph_train_table::ttgen_stats::TTExecutionSummary;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;
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

    /// Backward-compat alias for older callers.
    fn get_execution_stats(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.execution_summary(py)
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
pub fn execute_tt_plan_materialize(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_path: &str,
) -> PyResult<Py<PyAny>> {
    let plan: DiskGraphTTPlan = serde_json::from_str(plan_json)
        .map_err(|e| PyValueError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();
    let out = output_path.to_owned();

    let (rows_written, summary_json) = py
        .detach(|| -> anyhow::Result<(u64, String)> {
            let (receiver, handle) =
                diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))?;

            let file = File::create(&out)?;
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();

            let mut writer: Option<ArrowWriter<File>> = None;
            let mut file_opt = Some(file);
            let mut row_offset = 0i64;

            for batch in receiver.iter() {
                let batch = append_arange_column(batch, row_offset)?;
                row_offset += batch.num_rows() as i64;

                if writer.is_none() {
                    let f = file_opt
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing output file handle"))?;
                    writer = Some(ArrowWriter::try_new(
                        f,
                        batch.schema(),
                        Some(props.clone()),
                    )?);
                }

                if let Some(w) = writer.as_mut() {
                    w.write(&batch)?;
                }
            }

            if row_offset == 0 {
                return Err(anyhow::anyhow!("diskgraph execute_tt_plan returned zero rows"));
            }

            if let Some(w) = writer.take() {
                w.close()?;
            }

            let summary = handle
                .join()
                .map_err(|_| anyhow::anyhow!("TT-gen summary thread panicked"))??;

            let summary_json = serde_json::to_string(&summary)?;
            Ok((row_offset as u64, summary_json))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("materialize failed: {e:#}")))?;

    let out = PyDict::new(py);
    out.set_item("rows_written", rows_written)?;
    out.set_item("summary_json", summary_json)?;
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature=(plan_json, index_path, output_path, split_config_json=None, split_family_hint=None))]
pub fn execute_tt_plan_materialize_with_split(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_path: &str,
    split_config_json: Option<&str>,
    split_family_hint: Option<&str>,
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
    let manual_train_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "train_value"))
        .unwrap_or_default();
    let manual_val_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "val_value"))
        .unwrap_or_default();
    let manual_test_values = split_cfg
        .as_ref()
        .map(|cfg| parse_optional_string_list(cfg, "test_value"))
        .unwrap_or_default();

    let idx = index_path.to_owned();
    let out = output_path.to_owned();

    let (rows_written, summary_json, split_bytes, split_authoritative, split_reason) = py
        .detach(|| -> anyhow::Result<(u64, String, Option<Vec<u8>>, bool, Option<String>)> {
            let (receiver, handle) =
                diskgraph_train_table::ttgen_executor::execute_with_summary(&plan, Path::new(&idx))?;

            let file = File::create(&out)?;
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();

            let mut writer: Option<ArrowWriter<File>> = None;
            let mut file_opt = Some(file);
            let mut row_offset = 0i64;
            let mut timestamps_ns: Vec<i64> = Vec::new();
            let mut include_mask: Option<Vec<bool>> = None;
            let mut column_sort_values: Vec<SortValue> = Vec::new();
            let mut manual_values: Vec<String> = Vec::new();

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

                let batch = append_arange_column(batch, row_offset)?;
                row_offset += batch.num_rows() as i64;

                if writer.is_none() {
                    let f = file_opt
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing output file handle"))?;
                    writer = Some(ArrowWriter::try_new(
                        f,
                        batch.schema(),
                        Some(props.clone()),
                    )?);
                }

                if let Some(w) = writer.as_mut() {
                    w.write(&batch)?;
                }
            }

            if row_offset == 0 {
                return Err(anyhow::anyhow!("diskgraph execute_tt_plan returned zero rows"));
            }

            if let Some(w) = writer.take() {
                w.close()?;
            }

            let summary = handle
                .join()
                .map_err(|_| anyhow::anyhow!("TT-gen summary thread panicked"))??;
            let summary_json = serde_json::to_string(&summary)?;

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
                            .get("timestamp")
                            .ok_or_else(|| anyhow::anyhow!("BacktestingSplit requires timestamp"))?;
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
                        split_authoritative = false;
                        split_reason = Some(
                            "RandomLinkSplit is not implemented for diskgraph dense split assignment"
                                .to_string(),
                        );
                        Vec::new()
                    }
                    other => {
                        split_authoritative = false;
                        split_reason = Some(format!(
                            "native materialize split unsupported for family: {other}"
                        ));
                        Vec::new()
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
            ))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("materialize_with_split failed: {e:#}")))?;

    let out = PyDict::new(py);
    out.set_item("rows_written", rows_written)?;
    out.set_item("summary_json", summary_json)?;
    out.set_item("split_family", family)?;
    out.set_item("split_authoritative", split_authoritative)?;
    out.set_item("split_reason", split_reason)?;
    if let Some(bytes) = split_bytes {
        out.set_item("split_assignments", PyBytes::new(py, &bytes))?;
    } else {
        out.set_item("split_assignments", py.None())?;
    }
    Ok(out.into())
}
