// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Arrow C Data export for local and remote minibatches.
use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
use diskgraph_runtime::{resolve_reference, ArrowBatch, ArrowValue};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use serde_json::Value;

/// Restore the minibatch structure with PyArrow arrays and record batches.
///
/// Both local and remote iterators use this exporter. Arrow buffers cross the
/// language boundary through the Arrow C Data Interface without IPC encoding.
pub(crate) fn export_batch<'py>(
    py: Python<'py>,
    batch: &ArrowBatch,
) -> PyResult<Bound<'py, PyAny>> {
    export_layout(py, &batch.layout, &batch.components)
}

fn export_layout<'py>(
    py: Python<'py>,
    value: &Value,
    components: &[RecordBatch],
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(value) = resolve_reference(value, components)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))?
    {
        return match value {
            ArrowValue::Array(array) => array.to_data().to_pyarrow(py),
            ArrowValue::RecordBatch(batch) => batch.to_pyarrow(py),
        };
    }
    match value {
        Value::Object(values) => {
            let result = PyDict::new(py);
            for (key, value) in values {
                result.set_item(key, export_layout(py, value, components)?)?;
            }
            Ok(result.into_any())
        }
        Value::Array(values) => {
            let result = PyList::empty(py);
            for value in values {
                result.append(export_layout(py, value, components)?)?;
            }
            Ok(result.into_any())
        }
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any()),
        Value::String(value) => Ok(value.into_pyobject(py)?.into_any()),
        Value::Number(value) => {
            if let Some(value) = value.as_u64() {
                return Ok(value.into_pyobject(py)?.into_any());
            }
            if let Some(value) = value.as_i64() {
                return Ok(value.into_pyobject(py)?.into_any());
            }
            Ok(value.as_f64().unwrap().into_pyobject(py)?.into_any())
        }
    }
}
