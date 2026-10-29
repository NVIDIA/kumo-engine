// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interleave encoded column buffers into bounded, per-sample token streams.
//! Vocabulary fitting/column encoding remain caller policy. No Python row or
//! token objects cross this interface; the full packing pass releases the GIL.

use arrow::array::{make_array, Array, ArrayData, Int64Array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::prelude::*;

use crate::DiskgraphConfigError;

fn pack(
    offsets: &[i64],
    parts: &[(Int64Array, Int64Array)],
    maximum: usize,
    keep_last: bool,
    event_boundary: bool,
    separator: Option<i64>,
    between: bool,
    bos: Option<i64>,
    eos: Option<i64>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let reserved = usize::from(bos.is_some()) + usize::from(eos.is_some());
    if maximum <= reserved
        || offsets.len() < 2
        || offsets[0] != 0
        || offsets.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(
            "expected nonempty samples, increasing event offsets and a positive token budget"
                .into(),
        );
    }
    let n = *offsets.last().unwrap() as usize;
    let mut lengths = vec![usize::from(separator.is_some()); n];
    if separator.is_some() && between {
        for &end in &offsets[1..] {
            lengths[end as usize - 1] = 0;
        }
    }
    for (events, ids) in parts {
        if events.len() != ids.len() || events.null_count() != 0 || ids.null_count() != 0 {
            return Err(
                "each token part requires aligned, non-null event IDs and token IDs".into(),
            );
        }
        for event in events.values() {
            if *event < 0 || *event as usize >= n {
                return Err("token event ID out of range".into());
            }
            lengths[*event as usize] += 1;
        }
    }
    let mut starts = vec![0usize];
    for length in lengths {
        starts.push(starts.last().unwrap() + length);
    }
    let mut cursor = starts[..n].to_vec();
    let mut tokens = vec![0i64; starts[n]];
    for (events, ids) in parts {
        for (&event, &id) in events.values().iter().zip(ids.values()) {
            let pos = &mut cursor[event as usize];
            tokens[*pos] = id;
            *pos += 1;
        }
    }
    if let Some(id) = separator {
        for event in 0..n {
            if cursor[event] < starts[event + 1] {
                tokens[cursor[event]] = id;
            }
        }
    }
    let budget = maximum - reserved;
    let mut result = Vec::new();
    let mut boundaries = vec![0];
    for sample in offsets.windows(2) {
        let range = &starts[sample[0] as usize..=sample[1] as usize];
        let (mut lo, mut hi) = (range[0], *range.last().unwrap());
        if hi - lo > budget {
            if keep_last {
                let limit = hi - budget;
                lo = if event_boundary {
                    let candidate = range[range.partition_point(|v| *v < limit)];
                    if candidate == hi {
                        limit
                    } else {
                        candidate
                    }
                } else {
                    limit
                };
            } else {
                let limit = lo + budget;
                hi = if event_boundary {
                    let candidate = range[range.partition_point(|v| *v <= limit) - 1];
                    if candidate == lo {
                        limit
                    } else {
                        candidate
                    }
                } else {
                    limit
                };
            }
        }
        result.extend(bos);
        result.extend_from_slice(&tokens[lo..hi]);
        result.extend(eos);
        boundaries.push(result.len() as i64);
    }
    Ok((boundaries, result))
}

#[pyfunction]
pub fn pack_token_stream(
    py: Python<'_>,
    offsets: &Bound<'_, PyAny>,
    parts: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    maximum: usize,
    keep_last: bool,
    event_boundary: bool,
    separator: Option<i64>,
    between: bool,
    bos: Option<i64>,
    eos: Option<i64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    fn column(value: &Bound<'_, PyAny>) -> PyResult<Int64Array> {
        make_array(ArrayData::from_pyarrow_bound(value)?)
            .as_any()
            .downcast_ref::<Int64Array>()
            .cloned()
            .ok_or_else(|| DiskgraphConfigError::new_err("expected an Arrow int64 array"))
    }
    let offsets = column(offsets)?;
    if offsets.null_count() != 0 {
        return Err(DiskgraphConfigError::new_err("null event offset"));
    }
    let parts = parts
        .iter()
        .map(|(a, b)| Ok((column(a)?, column(b)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let (boundaries, tokens) = py
        .detach(|| {
            pack(
                offsets.values(),
                &parts,
                maximum,
                keep_last,
                event_boundary,
                separator,
                between,
                bos,
                eos,
            )
        })
        .map_err(DiskgraphConfigError::new_err)?;
    Ok((
        Int64Array::from(boundaries)
            .to_data()
            .to_pyarrow(py)?
            .unbind(),
        Int64Array::from(tokens).to_data().to_pyarrow(py)?.unbind(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_token_stream, m)?)
}
