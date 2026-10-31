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
    let mut entries = Vec::with_capacity(parts.iter().map(|(events, _)| events.len()).sum());
    for (events, ids) in parts {
        if events.len() != ids.len() || events.null_count() != 0 || ids.null_count() != 0 {
            return Err(
                "each token part requires aligned, non-null event IDs and token IDs".into(),
            );
        }
        for (&event, &id) in events.values().iter().zip(ids.values()) {
            if event < 0 || event as usize >= n {
                return Err("token event ID out of range".into());
            }
            entries.push((event as usize, id));
        }
    }
    // Stable: tokens of one event keep part order, then value order.
    entries.sort_by_key(|entry| entry.0);
    let before = |event: usize| entries.partition_point(|entry| entry.0 < event);
    let budget = maximum - reserved;
    let mut result = Vec::new();
    let mut boundaries = vec![0];
    for sample in offsets.windows(2) {
        let (first, end) = (sample[0] as usize, sample[1] as usize);
        let base = before(first);
        let has_separator = |event: usize| separator.is_some() && !(between && event == end - 1);
        // Event start relative to the sample; `start(end)` is the sample length.
        let start = |event: usize| {
            let separators = if separator.is_some() {
                event - first - usize::from(between && event == end)
            } else {
                0
            };
            before(event) - base + separators
        };
        // `partition_point` over the sample's event starts `start(first..=end)`.
        let partition = |predicate: &dyn Fn(usize) -> bool| {
            let (mut low, mut high) = (first, end + 1);
            while low < high {
                let middle = low + (high - low) / 2;
                if predicate(start(middle)) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            low
        };
        let (mut lo, mut hi) = (0, start(end));
        if hi - lo > budget {
            if keep_last {
                let limit = hi - budget;
                lo = if event_boundary {
                    let candidate = start(partition(&|v| v < limit));
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
                    let candidate = start(partition(&|v| v <= limit) - 1);
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
        let mut position = lo;
        let mut event = partition(&|v| v <= lo) - 1;
        while position < hi {
            let (begin, offset) = (start(event), before(event));
            let count = before(event + 1) - offset;
            let stop = (count + usize::from(has_separator(event))).min(hi - begin);
            for index in position - begin..stop {
                result.push(if index < count {
                    entries[offset + index].1
                } else {
                    separator.unwrap()
                });
            }
            position = begin + stop;
            // Without separators, jump straight to the next event that has tokens.
            event = match separator {
                Some(_) => event + 1,
                None => entries.get(offset + count).map_or(end, |entry| entry.0),
            };
        }
        result.extend(eos);
        boundaries.push(result.len() as i64);
    }
    Ok((boundaries, result))
}

#[pyfunction]
pub fn pack_token_stream(
    py: Python<'_>,
    event_offsets: &Bound<'_, PyAny>,
    parts: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    max_tokens: usize,
    keep_last: bool,
    event_boundary: bool,
    separator_id: Option<i64>,
    separator_between: bool,
    bos_id: Option<i64>,
    eos_id: Option<i64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    fn column(value: &Bound<'_, PyAny>) -> PyResult<Int64Array> {
        make_array(ArrayData::from_pyarrow_bound(value)?)
            .as_any()
            .downcast_ref::<Int64Array>()
            .cloned()
            .ok_or_else(|| DiskgraphConfigError::new_err("expected an Arrow int64 array"))
    }
    let offsets = column(event_offsets)?;
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
                max_tokens,
                keep_last,
                event_boundary,
                separator_id,
                separator_between,
                bos_id,
                eos_id,
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
