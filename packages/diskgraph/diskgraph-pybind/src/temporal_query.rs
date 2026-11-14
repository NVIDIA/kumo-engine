// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python conversion for the shared native row-query and aggregation executor.
use arrow::array::{Array, Int64Array, UInt64Array};
use arrow::pyarrow::ToPyArrow;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::DiskgraphConfigError;
use serde::Deserialize;
use diskgraph_train_table::batch_aggregate::{scalar_values_to_array, ScalarValue};
pub(crate) use diskgraph_train_table::batch_aggregate::{
    execute_parallel, execute_aggregate_plan, execute_lengths_parallel, QueryResult, LengthResult,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EventMerge {
    pub name: String,
    pub sequences: Vec<String>,
    pub ascending: Vec<bool>,
    maximum: Option<usize>,
    keep_last: bool,
    descending: bool,
}

pub(crate) trait QueryPython {
    #[allow(clippy::type_complexity)]
    fn merged_event_orders(&self, plans: &[EventMerge]) -> Result<Vec<(String, Vec<i64>, Vec<i64>)>, String>;
    fn into_python(self, py: Python<'_>, model_format: bool) -> PyResult<Py<PyDict>>;
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
    /// Plan component ordering while histories are still native batch buffers.
    fn merged_event_orders(
        &self,
        plans: &[EventMerge],
    ) -> Result<Vec<(String, Vec<i64>, Vec<i64>)>, String> {
        plans
            .iter()
            .map(|plan| {
                if plan.sequences.len() != plan.ascending.len() {
                    return Err("merge sequence/order counts differ".into());
                }
                let parts = plan
                    .sequences
                    .iter()
                    .zip(&plan.ascending)
                    .map(|(name, ascending)| {
                        let sequence = self
                            .sequences
                            .iter()
                            .find(|s| &s.name == name)
                            .ok_or_else(|| format!("unknown merge sequence {name:?}"))?;
                        Ok((
                            Int64Array::from(
                                sequence
                                    .offsets
                                    .iter()
                                    .map(|&v| v as i64)
                                    .collect::<Vec<_>>(),
                            ),
                            Int64Array::from(sequence.timestamps.clone()),
                            *ascending,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let (bounds, order) = crate::token_stream::event_stream_order(
                    &parts,
                    plan.maximum,
                    plan.keep_last,
                    plan.descending,
                )?;
                Ok((plan.name.clone(), bounds, order))
            })
            .collect()
    }

    fn into_python(self, py: Python<'_>, model_format: bool) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let sequences = PyList::empty(py);
        for mut sequence in self.sequences {
            sequence.timestamps = sequence.timestamps.into_iter().map(|v| v.map(|t|
                diskgraph_core::time::convert_trunc(t,sequence.clock,diskgraph_core::time::TimeUnit::Seconds))).collect();
            let item = PyDict::new(py);
            item.set_item("name", sequence.name)?;
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
            sequences.append(item)?;
        }
        output.set_item("sequences", sequences)?;

        let aggregates = PyList::empty(py);
        for aggregate in self.aggregates {
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
            aggregates.append(item)?;
        }
        output.set_item("aggregates", aggregates)?;

        let timings = PyDict::new(py);
        timings.set_item("traversal", self.timings.traversal)?;
        timings.set_item("decode_and_select", self.timings.decode_and_select)?;
        timings.set_item("sequence_assembly", self.timings.sequence_assembly)?;
        timings.set_item("aggregation", self.timings.aggregation)?;
        timings.set_item("total", self.timings.total)?;
        output.set_item("timings_ns", timings)?;

        let stats = PyDict::new(py);
        stats.set_item("frames_materialized", self.stats.frames_materialized)?;
        stats.set_item("rows_materialized", self.stats.rows_materialized)?;
        stats.set_item("rows_decoded", self.stats.rows_decoded)?;
        stats.set_item("values_decoded", self.stats.values_decoded)?;
        stats.set_item("edges_visited", self.stats.edges_visited)?;
        output.set_item("stats", stats)?;
        Ok(output.unbind())
    }
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
