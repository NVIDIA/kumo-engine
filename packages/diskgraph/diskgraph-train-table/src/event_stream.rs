// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native event ordering shared by temporal queries and runtime history.

use arrow_array::{Array, Int64Array};
use serde::Deserialize;

use crate::batch_aggregate::QueryResult;

/// Named per-anchor offsets and indices into concatenated sequence rows.
pub type EventOrders = Vec<(String, Vec<i64>, Vec<i64>)>;

/// Named temporal sequences to merge into a per-anchor event stream.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventMerge {
    /// Name of the merged output stream.
    pub name: String,
    /// Input sequence names in concatenation and tie-break order.
    pub sequences: Vec<String>,
    /// Whether each input sequence is sorted in ascending timestamp order.
    pub ascending: Vec<bool>,
    maximum: Option<usize>,
    keep_last: bool,
    descending: bool,
}

impl QueryResult {
    /// Compute named merge offsets and indices against native history buffers.
    ///
    /// # Errors
    /// Returns an error for unknown sequences, mismatched order counts, or
    /// invalid event offsets/timestamps.
    pub fn merged_event_orders(&self, plans: &[EventMerge]) -> Result<EventOrders, String> {
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
                let (bounds, order) =
                    event_stream_order(&parts, plan.maximum, plan.keep_last, plan.descending)?;
                Ok((plan.name.clone(), bounds, order))
            })
            .collect()
    }
}

/// Stable multi-component event merge. Returned indices address the concatenated
/// input components, so the caller can reuse its existing Arrow field buffers.
/// `maximum` bounds events per anchor; `keep_last` selects latest rather than
/// earliest events, and `descending` controls the output order.
///
/// # Errors
/// Returns an error for unaligned/null offsets or timestamps, invalid bounds,
/// or timestamps that disagree with their declared sort order.
pub fn event_stream_order(
    parts: &[(Int64Array, Int64Array, bool)],
    maximum: Option<usize>,
    keep_last: bool,
    descending: bool,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let samples = parts
        .first()
        .map_or(0, |(offsets, _, _)| offsets.len().saturating_sub(1));
    let mut bases = Vec::with_capacity(parts.len());
    let mut total = 0;
    for (offsets, times, ascending) in parts {
        if offsets.len() != samples + 1
            || offsets.null_count() != 0
            || times.null_count() != 0
            || offsets.value(0) != 0
            || offsets.value(samples) != times.len() as i64
            || offsets.values().windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("event streams require aligned non-null offsets and timestamps".into());
        }
        for bounds in offsets.values().windows(2) {
            let rows = &times.values()[bounds[0] as usize..bounds[1] as usize];
            if rows.windows(2).any(|pair| {
                if *ascending {
                    pair[0] > pair[1]
                } else {
                    pair[0] < pair[1]
                }
            }) {
                return Err("event stream timestamps do not match their declared order".into());
            }
        }
        bases.push(total);
        total += times.len();
    }
    let mut boundaries = vec![0];
    let mut selected = Vec::new();
    for anchor in 0..samples {
        let mut heap = BinaryHeap::new();
        let sign = if keep_last { -1_i128 } else { 1 };
        let key = |component: usize, row: usize| {
            let (offsets, times, ascending) = &parts[component];
            let ordinal = if *ascending {
                row - offsets.value(anchor) as usize
            } else {
                offsets.value(anchor + 1) as usize - 1 - row
            };
            Reverse((
                sign * i128::from(times.value(row)),
                sign * component as i128,
                sign * ordinal as i128,
                component,
                row,
            ))
        };
        let mut available = 0;
        for (component, (offsets, _, ascending)) in parts.iter().enumerate() {
            let (start, stop) = (
                offsets.value(anchor) as usize,
                offsets.value(anchor + 1) as usize,
            );
            available += stop - start;
            if start != stop {
                heap.push(key(
                    component,
                    if *ascending != keep_last {
                        start
                    } else {
                        stop - 1
                    },
                ));
            }
        }
        let begin = selected.len();
        let count = maximum.map_or(available, |maximum| available.min(maximum));
        while selected.len() - begin < count {
            let Reverse((_, _, _, component, row)) = heap.pop().expect("remaining event");
            selected.push((bases[component] + row) as i64);
            let (offsets, _, ascending) = &parts[component];
            let next = if *ascending != keep_last {
                row.checked_add(1)
            } else {
                row.checked_sub(1)
            };
            if let Some(next) = next.filter(|&next| {
                next >= offsets.value(anchor) as usize && next < offsets.value(anchor + 1) as usize
            }) {
                heap.push(key(component, next));
            }
        }
        if keep_last != descending {
            selected[begin..].reverse();
        }
        boundaries.push(selected.len() as i64);
    }
    Ok((boundaries, selected))
}
