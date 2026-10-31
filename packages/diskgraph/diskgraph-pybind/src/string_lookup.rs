// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reuse the string-to-position table across Arrow batches; callers own token IDs.

use arrow::array::{make_array, Array, ArrayData, Int32Array, StringArray};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use gxhash::{HashMap, HashMapExt};
use pyo3::prelude::*;

use crate::DiskgraphConfigError;

fn strings(value: &Bound<'_, PyAny>) -> PyResult<StringArray> {
    make_array(ArrayData::from_pyarrow_bound(value)?)
        .as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| DiskgraphConfigError::new_err("expected an Arrow string array"))
}

#[pyclass(frozen, module = "diskgraph._c")]
pub struct StringLookup {
    positions: HashMap<String, i32>,
    null_position: Option<i32>,
}

#[pymethods]
impl StringLookup {
    #[new]
    fn new(keys: &Bound<'_, PyAny>) -> PyResult<Self> {
        let keys = strings(keys)?;
        let mut positions = HashMap::with_capacity(keys.len());
        let mut null_position = None;
        for (index, key) in keys.iter().enumerate() {
            let index = i32::try_from(index)
                .map_err(|_| DiskgraphConfigError::new_err("too many string lookup keys"))?;
            match key {
                Some(key) => {
                    positions.entry(key.to_owned()).or_insert(index);
                }
                None => {
                    null_position.get_or_insert(index);
                }
            }
        }
        positions.shrink_to_fit();
        Ok(Self {
            positions,
            null_position,
        })
    }

    fn lookup(&self, py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let values = strings(values)?;
        let positions = py.detach(|| {
            Int32Array::from_iter(values.iter().map(|value| match value {
                Some(value) => self.positions.get(value).copied(),
                None => self.null_position,
            }))
        });
        Ok(positions.to_data().to_pyarrow(py)?.unbind())
    }
}
