// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// Copy fixed-width rows from two C-contiguous byte matrices into one owned buffer.
/// Nonnegative indices select cached rows; bitwise complements select incoming rows.
#[pyfunction]
pub(crate) fn _gather_byte_rows<'py>(
    py: Python<'py>,
    cached: PyBuffer<u8>,
    incoming: PyBuffer<u8>,
    indices: PyBuffer<i64>,
) -> PyResult<Bound<'py, PyByteArray>> {
    if cached.dimensions() != 2 || incoming.dimensions() != 2 || indices.dimensions() != 1 {
        return Err(PyValueError::new_err(
            "expected two matrices and a vector of row indices",
        ));
    }
    let width = cached.shape()[1];
    if width == 0 || incoming.shape()[1] != width {
        return Err(PyValueError::new_err(
            "row widths must be equal and nonzero",
        ));
    }
    let strides = || PyValueError::new_err("row gather requires C-contiguous buffers");
    let cached = cached.as_slice(py).ok_or_else(strides)?;
    let incoming = incoming.as_slice(py).ok_or_else(strides)?;
    let indices = indices.as_slice(py).ok_or_else(strides)?;
    let len = indices
        .len()
        .checked_mul(width)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| PyValueError::new_err("row gather size overflow"))?;
    // Keep the GIL while accessing borrowed Python buffers. The result has no aliases.
    PyByteArray::new_with(py, len, |out| {
        for (dest, index) in out.chunks_exact_mut(width).zip(indices) {
            let index = index.get();
            let (source, row) = if index >= 0 {
                (cached, index)
            } else {
                (incoming, !index)
            };
            let row = usize::try_from(row)
                .ok()
                .filter(|&r| r < source.len() / width)
                .ok_or_else(|| PyValueError::new_err("row index out of bounds"))?;
            for (dest, src) in dest.iter_mut().zip(&source[row * width..][..width]) {
                *dest = src.get();
            }
        }
        Ok(())
    })
}
