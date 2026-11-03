// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact `np.quantile` of a parquet column's finite values, by radix selection.

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::bp_col_stats::{float64s, row_group_batches};

const DIGIT: u32 = 8;
const BUCKETS: usize = 1 << DIGIT;
/// Most values sorted in memory instead of refined further.
pub(crate) const GATHER: u64 = 1 << 22;

pub(crate) struct Column<'a> {
    pub paths: &'a [&'a Path],
    pub work: &'a [(usize, usize)],
    /// Parquet column index per path.
    pub indices: Vec<usize>,
}

impl Column<'_> {
    fn fold<S: Send>(
        &self,
        init: impl Fn() -> S + Sync,
        add: impl Fn(&mut S, u64) + Sync,
        merge: impl Fn(&mut S, S) + Sync,
    ) -> Result<S> {
        let parts = self
            .work
            .par_iter()
            .map(|&(pi, rg)| {
                let mut state = init();
                for batch in row_group_batches(self.paths[pi], rg, &[self.indices[pi]], &[])? {
                    let values = float64s(batch?.column(0))?;
                    for value in values.iter().flatten().filter(|v| v.is_finite()) {
                        add(&mut state, key(value));
                    }
                }
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut total = init();
        for part in parts {
            merge(&mut total, part);
        }
        Ok(total)
    }
}

/// Order-preserving; -0.0 and 0.0 map to one key.
fn key(value: f64) -> u64 {
    let bits = (value + 0.0).to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

fn value(key: u64) -> f64 {
    f64::from_bits(if key >> 63 == 1 {
        key & !(1 << 63)
    } else {
        !key
    })
}

/// A rank among the values whose keys start with `prefix`.
#[derive(Clone, Copy)]
struct Target {
    prefix: u64,
    rank: u64,
    population: u64,
}

fn select(column: &Column<'_>, count: u64, ranks: &[u64], gather: u64) -> Result<Vec<f64>> {
    let mut targets: Vec<Target> = ranks
        .iter()
        .map(|&rank| Target {
            prefix: 0,
            rank,
            population: count,
        })
        .collect();
    let mut found = vec![None; ranks.len()];
    let mut known = 0;
    while found.iter().any(Option::is_none) {
        let mut groups: Vec<(u64, bool)> = targets
            .iter()
            .zip(&found)
            .filter(|(_, done)| done.is_none())
            .map(|(target, _)| (target.prefix, target.population <= gather))
            .collect();
        groups.sort_unstable();
        groups.dedup();
        let prefix = |key: u64| if known == 0 { 0 } else { key >> (64 - known) };
        let (counts, mut gathered) = column.fold(
            || {
                (
                    vec![[0u64; BUCKETS]; groups.len()],
                    vec![Vec::new(); groups.len()],
                )
            },
            |(counts, gathered), key| {
                if let Ok(at) = groups.binary_search_by_key(&prefix(key), |(p, _)| *p) {
                    if groups[at].1 {
                        gathered[at].push(key);
                    } else {
                        counts[at][(key >> (64 - known - DIGIT)) as usize & (BUCKETS - 1)] += 1;
                    }
                }
            },
            |(counts, gathered), (more_counts, more_gathered)| {
                for (total, part) in counts.iter_mut().zip(more_counts) {
                    for (total, count) in total.iter_mut().zip(part) {
                        *total += count;
                    }
                }
                for (total, part) in gathered.iter_mut().zip(more_gathered) {
                    total.extend(part);
                }
            },
        )?;
        for keys in &mut gathered {
            keys.sort_unstable();
        }
        for (target, slot) in targets.iter_mut().zip(found.iter_mut()) {
            if slot.is_some() {
                continue;
            }
            let at = groups
                .binary_search_by_key(&target.prefix, |(p, _)| *p)
                .ok()
                .context("quantile group")?;
            if groups[at].1 {
                *slot = Some(value(gathered[at][target.rank as usize]));
                continue;
            }
            let mut below = 0;
            for (bucket, &bucket_count) in counts[at].iter().enumerate() {
                if target.rank < below + bucket_count {
                    target.prefix = target.prefix << DIGIT | bucket as u64;
                    target.rank -= below;
                    target.population = bucket_count;
                    break;
                }
                below += bucket_count;
            }
            if known + DIGIT == 64 {
                *slot = Some(value(target.prefix));
            }
        }
        known += DIGIT;
    }
    Ok(found
        .into_iter()
        .map(|value| value.expect("selected"))
        .collect())
}

/// The count of finite values and their quantiles.
pub(crate) fn exact_quantiles(
    column: &Column<'_>,
    quantiles: &[f64],
    gather: u64,
) -> Result<(u64, Vec<f64>)> {
    let count = column.fold(
        || 0u64,
        |count, _| *count += 1,
        |count, more| *count += more,
    )?;
    if count == 0 {
        return Ok((0, Vec::new()));
    }
    let last = (count - 1) as f64;
    let points: Vec<(u64, u64, f64)> = quantiles
        .iter()
        .map(|&q| {
            let at = last * q;
            if at >= last {
                (count - 1, count - 1, at + 1.0)
            } else {
                let below = at.floor();
                (below as u64, below as u64 + 1, at - below)
            }
        })
        .collect();
    let mut ranks: Vec<u64> = points.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let values = select(column, count, &ranks, gather)?;
    let at = |rank: u64| values[ranks.binary_search(&rank).expect("selected rank")];
    Ok((
        count,
        points
            .into_iter()
            .map(|(lo, hi, t)| {
                let (a, b) = (at(lo), at(hi));
                let diff = b - a;
                if t >= 0.5 {
                    b - diff * (1.0 - t)
                } else {
                    a + diff * t
                }
            })
            .collect(),
    ))
}
