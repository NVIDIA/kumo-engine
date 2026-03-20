//! PyO3 binding for the TT-gen executor.
//!
//! Exposes [`execute_tt_plan`] as a `#[pyfunction]` that returns a
//! `TTPlanIterator` yielding `pyarrow.RecordBatch` objects one at a time.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

use arrow::pyarrow::ToPyArrow;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use crossbeam_channel::Receiver;
use diskgraph_core::tt_builder::DiskGraphTTPlan;
use diskgraph_train_table::ttgen_stats::TTExecutionSummary;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
