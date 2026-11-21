// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python conversion for the shared native row-query and aggregation executor.
use crate::DiskgraphConfigError;
use arrow::array::{Array, Int64Array, UInt64Array};
use arrow::pyarrow::ToPyArrow;
pub(crate) use diskgraph_train_table::batch_aggregate::{
    execute_aggregate_plan, execute_lengths_parallel, execute_parallel, LengthResult, QueryResult,
};
use diskgraph_train_table::batch_aggregate::{scalar_values_to_array, ScalarValue};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) trait QueryPython {
    fn into_python(self, py: Python<'_>, model_format: bool) -> PyResult<Py<PyDict>>;
    fn into_runtime_python(self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}
pub(crate) trait LengthPython {
    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyDict>>;
}
fn append_scalar_to(value: &ScalarValue, list: &Bound<'_, PyList>) -> PyResult<()> {
    match value {
        ScalarValue::Null => list.append(list.py().None()),
        ScalarValue::Bool(v) => list.append(*v),
        ScalarValue::I64(v) => list.append(*v),
        ScalarValue::U64(v) => list.append(*v),
        ScalarValue::F64(v) => list.append(*v),
        ScalarValue::String(v) => list.append(v),
    }
}

impl QueryPython for QueryResult {
    fn into_python(self, py: Python<'_>, model_format: bool) -> PyResult<Py<PyDict>> {
        Ok(query_to_python(self, py, model_format, false)?
            .into_bound(py)
            .cast_into::<PyDict>()?
            .unbind())
    }

    fn into_runtime_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        query_to_python(self, py, true, true)
    }
}

fn query_record<'py>(
    types: Option<&Bound<'py, PyModule>>,
    name: &str,
    fields: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    match types {
        Some(types) => types.getattr(name)?.call((), Some(&fields)),
        None => Ok(fields.into_any()),
    }
}

fn query_to_python(
    result: QueryResult,
    py: Python<'_>,
    model_format: bool,
    records: bool,
) -> PyResult<Py<PyAny>> {
    let types = if records {
        Some(py.import("diskgraph.runtime_batch")?)
    } else {
        None
    };
    let output = PyDict::new(py);
    let sequences = PyList::empty(py);
    for mut sequence in result.sequences {
        sequence.timestamps = sequence
            .timestamps
            .into_iter()
            .map(|v| {
                v.map(|t| {
                    diskgraph_core::time::convert_trunc(t, sequence.clock, result.timestamp_unit)
                })
            })
            .collect();
        let item = PyDict::new(py);
        item.set_item("name", sequence.name)?;
        if result.timestamp_unit != diskgraph_core::time::TimeUnit::Seconds {
            item.set_item(
                "timestamp_unit",
                serde_json::to_value(result.timestamp_unit)
                    .unwrap()
                    .as_str()
                    .unwrap(),
            )?;
        }
        if model_format {
            let offsets = UInt64Array::from(
                sequence
                    .offsets
                    .into_iter()
                    .map(|value| value as u64)
                    .collect::<Vec<_>>(),
            );
            item.set_item("offsets", offsets.to_data().to_pyarrow(py)?)?;
            item.set_item(
                "node_ids",
                UInt64Array::from(sequence.node_ids)
                    .to_data()
                    .to_pyarrow(py)?,
            )?;
            item.set_item(
                "timestamps",
                Int64Array::from(sequence.timestamps)
                    .to_data()
                    .to_pyarrow(py)?,
            )?;
        } else {
            item.set_item("offsets", sequence.offsets)?;
            item.set_item("node_ids", sequence.node_ids)?;
            item.set_item("timestamps", sequence.timestamps)?;
        }
        item.set_item("batch", sequence.batch.to_pyarrow(py)?)?;
        sequences.append(query_record(
            types.as_ref(),
            "RuntimeHistorySequence",
            item,
        )?)?;
    }
    output.set_item("sequences", sequences)?;

    let aggregates = PyList::empty(py);
    for aggregate in result.aggregates {
        let item = PyDict::new(py);
        item.set_item("name", aggregate.name)?;
        if model_format {
            let values = scalar_values_to_array(&aggregate.values, aggregate.kind)
                .map_err(DiskgraphConfigError::new_err)?;
            item.set_item("values", values.to_data().to_pyarrow(py)?)?;
        } else {
            let values = PyList::empty(py);
            for value in aggregate.values {
                append_scalar_to(&value, &values)?;
            }
            item.set_item("values", values)?;
        }
        aggregates.append(query_record(
            types.as_ref(),
            "RuntimeHistoryAggregate",
            item,
        )?)?;
    }
    output.set_item("aggregates", aggregates)?;

    let timings = PyDict::new(py);
    timings.set_item("traversal", result.timings.traversal)?;
    timings.set_item("decode_and_select", result.timings.decode_and_select)?;
    timings.set_item("sequence_assembly", result.timings.sequence_assembly)?;
    timings.set_item("aggregation", result.timings.aggregation)?;
    timings.set_item("total", result.timings.total)?;
    output.set_item(
        "timings_ns",
        query_record(types.as_ref(), "RuntimeHistoryTimings", timings)?,
    )?;

    let stats = PyDict::new(py);
    stats.set_item("frames_materialized", result.stats.frames_materialized)?;
    stats.set_item("rows_materialized", result.stats.rows_materialized)?;
    stats.set_item("rows_decoded", result.stats.rows_decoded)?;
    stats.set_item("values_decoded", result.stats.values_decoded)?;
    stats.set_item("edges_visited", result.stats.edges_visited)?;
    output.set_item(
        "stats",
        query_record(types.as_ref(), "RuntimeHistoryStats", stats)?,
    )?;
    Ok(query_record(types.as_ref(), "RuntimeHistory", output)?.unbind())
}

impl LengthPython for LengthResult {
    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let sequences = PyList::empty(py);
        for sequence in self.sequences {
            let item = PyDict::new(py);
            item.set_item("name", sequence.name)?;
            let lengths = UInt64Array::from(
                sequence
                    .lengths
                    .into_iter()
                    .map(|value| value as u64)
                    .collect::<Vec<_>>(),
            );
            item.set_item("lengths", lengths.to_data().to_pyarrow(py)?)?;
            sequences.append(item)?;
        }
        output.set_item("sequences", sequences)?;

        let timings = PyDict::new(py);
        timings.set_item("traversal", self.timings.traversal)?;
        timings.set_item("time_filter", self.timings.time_filter)?;
        timings.set_item("total", self.timings.total)?;
        output.set_item("timings_ns", timings)?;

        let stats = PyDict::new(py);
        stats.set_item("frames_counted", self.stats.frames_counted)?;
        stats.set_item("rows_considered", self.stats.rows_considered)?;
        stats.set_item("edges_visited", self.stats.edges_visited)?;
        stats.set_item("sort_key_timestamps", self.stats.sort_key_timestamps)?;
        stats.set_item("rowpack_timestamps", self.stats.rowpack_timestamps)?;
        output.set_item("stats", stats)?;
        Ok(output.unbind())
    }
}

pub(crate) fn config_error(error: String) -> PyErr {
    DiskgraphConfigError::new_err(error)
}
