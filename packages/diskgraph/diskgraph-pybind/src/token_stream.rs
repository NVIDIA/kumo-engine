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
use diskgraph_train_table::event_stream::event_stream_order;

/// Same clipping policy for planning and token delivery, without token IDs.
fn clipped_range(
    first: usize,
    end: usize,
    budget: usize,
    keep_last: bool,
    boundary: bool,
    start: &impl Fn(usize) -> usize,
) -> (usize, usize) {
    let total = start(end);
    if total <= budget {
        return (0, total);
    }
    // Binary search logical event starts without allocating one entry per event.
    // Token packing can represent billions of empty logical events sparsely.
    let partition = |predicate: &dyn Fn(usize) -> bool| {
        let (mut lo, mut hi) = (first, end + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if predicate(start(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    };
    if keep_last {
        let limit = total - budget;
        let candidate = start(partition(&|v| v < limit));
        (
            if boundary && candidate != total {
                candidate
            } else {
                limit
            },
            total,
        )
    } else {
        let candidate = start(partition(&|v| v <= budget) - 1);
        (
            0,
            if boundary && candidate != 0 {
                candidate
            } else {
                budget
            },
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn packed_length(
    events: &[usize],
    maximum: usize,
    keep_last: bool,
    boundary: bool,
    separator: bool,
    between: bool,
    before: bool,
    after: bool,
) -> Result<usize, String> {
    let reserved = usize::from(before) + usize::from(after);
    if maximum <= reserved {
        return Err("token budget must leave room beyond markers".into());
    }
    let mut starts = Vec::with_capacity(events.len() + 1);
    starts.push(0usize);
    for (i, &tokens) in events.iter().enumerate() {
        let extra = usize::from(separator && !(between && i + 1 == events.len()));
        starts.push(
            starts
                .last()
                .unwrap()
                .checked_add(tokens)
                .and_then(|v| v.checked_add(extra))
                .ok_or("token length overflow")?,
        );
    }
    let (lo, hi) = clipped_range(
        0,
        events.len(),
        maximum - reserved,
        keep_last,
        boundary,
        &|i| starts[i],
    );
    Ok(hi - lo + reserved)
}

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
        let (lo, hi) = clipped_range(first, end, budget, keep_last, event_boundary, &start);
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

#[pyfunction(signature = (parts, max_events, keep_last, descending))]
pub fn merge_event_streams(
    py: Python<'_>,
    parts: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>, bool)>,
    max_events: Option<usize>,
    keep_last: bool,
    descending: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    fn column(value: &Bound<'_, PyAny>) -> PyResult<Int64Array> {
        let array = make_array(ArrayData::from_pyarrow_bound(value)?);
        if !matches!(
            array.data_type(),
            arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::UInt64
        ) {
            return Err(DiskgraphConfigError::new_err(
                "expected an Arrow integer array",
            ));
        }
        let array = arrow::compute::cast(array.as_ref(), &arrow::datatypes::DataType::Int64)
            .map_err(|error| DiskgraphConfigError::new_err(error.to_string()))?;
        Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast int64")
            .clone())
    }
    let parts = parts
        .iter()
        .map(|(offsets, times, ascending)| Ok((column(offsets)?, column(times)?, *ascending)))
        .collect::<PyResult<Vec<_>>>()?;
    let (boundaries, order) = py
        .detach(|| event_stream_order(&parts, max_events, keep_last, descending))
        .map_err(DiskgraphConfigError::new_err)?;
    Ok((
        Int64Array::from(boundaries)
            .to_data()
            .to_pyarrow(py)?
            .unbind(),
        Int64Array::from(order).to_data().to_pyarrow(py)?.unbind(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_token_stream, m)?)?;
    m.add_function(wrap_pyfunction!(merge_event_streams, m)?)
}
