// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Import batch columns without iterating over Python scalars. The engine owns
//! Vec inputs, so this removes Python boxing, not the native buffer copy.

use arrow::array::{make_array, Array, ArrayData, ArrayRef, Int64Array, UInt64Array};
use arrow::datatypes::DataType;
use arrow::pyarrow::FromPyArrow;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::{DiskgraphConfigError, DiskgraphDataError};

fn array_input(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<ArrayRef>> {
    if value.hasattr("__arrow_c_array__")? || value.hasattr("_export_to_c")? {
        let array = make_array(ArrayData::from_pyarrow_bound(value)?);
        if array.null_count() != 0 {
            return Err(DiskgraphDataError::new_err(format!(
                "{name} must not contain nulls"
            )));
        }
        Ok(Some(array))
    } else if value.hasattr("__arrow_c_stream__")? {
        Err(PyTypeError::new_err(format!(
            "{name} requires one Arrow Array; combine chunks explicitly"
        )))
    } else {
        Ok(None)
    }
}

pub(crate) fn node_ids(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    let Some(array) = array_input(value, name)? else {
        return value.extract();
    };
    match array.data_type() {
        DataType::UInt64 => Ok(array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("checked uint64")
            .values()
            .to_vec()),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("checked int64")
            .values()
            .iter()
            .enumerate()
            .map(|(offset, &id)| {
                u64::try_from(id).map_err(|_| {
                    DiskgraphDataError::new_err(format!(
                        "{name} has negative node ID at batch offset {offset}"
                    ))
                })
            })
            .collect(),
        dtype => Err(PyTypeError::new_err(format!(
            "{name} requires Arrow uint64 or int64, got {dtype}"
        ))),
    }
}

pub(crate) fn timestamps(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let Some(array) = array_input(value, name)? else {
        return value.extract();
    };
    let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{name} requires Arrow int64 in the API's declared time unit, got {}",
            array.data_type()
        ))
    })?;
    Ok(values.values().to_vec())
}

pub(crate) fn timestamp_map(
    values: Option<Bound<'_, PyDict>>,
    seeds: &HashMap<String, Vec<u64>>,
) -> PyResult<Option<HashMap<String, Vec<i64>>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let mut result = HashMap::new();
    for (key, value) in values.iter() {
        let node_type: String = key.extract()?;
        let ids = seeds.get(&node_type).ok_or_else(|| {
            DiskgraphConfigError::new_err(format!(
                "seed_timestamps has no matching seed type {node_type:?}"
            ))
        })?;
        let times = timestamps(&value, "seed_timestamps")?;
        if times.len() != ids.len() {
            return Err(DiskgraphConfigError::new_err(format!(
                "seed_timestamps for {node_type:?} must have equal length to seeds: {} != {}",
                times.len(),
                ids.len()
            )));
        }
        result.insert(node_type, times);
    }
    Ok(Some(result))
}
