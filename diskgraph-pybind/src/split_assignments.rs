use diskgraph_core::time::{add_offset, Abs, Nanoseconds, TimeUnit};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};

/// Nanosecond-unit absolute timestamp. All split functions work in this unit
/// because Arrow provides `TimestampNanosecondArray` and the Python caller
/// converts boundaries to nanoseconds via `to_nanoseconds()`.
type AbsNs = Abs<Nanoseconds>;

const UNUSED_STAGE_ID: u8 = 255;

fn stage_assignments_by_ratio_on_perm(
    row_count: usize,
    ratio: &[f64],
    perm: &[usize],
) -> PyResult<Vec<u8>> {
    if perm.len() != row_count {
        return Err(crate::DiskgraphConfigError::new_err(format!(
            "perm length mismatch: expected {} got {}",
            row_count,
            perm.len()
        )));
    }

    let mut assignments = vec![UNUSED_STAGE_ID; row_count];

    let mut ratio_vals: Vec<f64> = ratio.iter().take(3).map(|x| x.max(0.0)).collect();
    if ratio_vals.is_empty() {
        return Ok(assignments);
    }

    let total: f64 = ratio_vals.iter().sum();
    if total <= 0.0 {
        return Ok(assignments);
    }
    for x in &mut ratio_vals {
        *x /= total;
    }

    let ideal: Vec<f64> = ratio_vals.iter().map(|x| x * row_count as f64).collect();
    let mut sizes: Vec<usize> = ideal.iter().map(|v| v.floor() as usize).collect();

    let assigned_floor: usize = sizes.iter().sum();
    let mut remaining = row_count.saturating_sub(assigned_floor);

    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = ideal[a] - (sizes[a] as f64);
        let rb = ideal[b] - (sizes[b] as f64);
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut k = 0usize;
    while remaining > 0 && k < order.len() {
        sizes[order[k]] += 1;
        remaining -= 1;
        k += 1;
        if k == order.len() {
            k = 0;
        }
    }

    let mut i = 0usize;
    for (split_idx, size) in sizes.iter().enumerate() {
        let j = (i + *size).min(row_count);
        let stage_id = match split_idx {
            0 => 0u8,
            1 => 1u8,
            _ => 2u8,
        };
        for &p in &perm[i..j] {
            assignments[p] = stage_id;
        }
        i = j;
    }

    if i < row_count {
        let last_stage = match sizes.len() {
            0 | 1 => 0u8,
            2 => 1u8,
            _ => 2u8,
        };
        for &p in &perm[i..row_count] {
            assignments[p] = last_stage;
        }
    }

    Ok(assignments)
}

/// Parse a unit name string and add an offset to a typed nanosecond timestamp.
///
/// Delegates to [`add_offset`] which handles both fixed-size units (seconds
/// through weeks) via pure arithmetic and calendar units (months/years) via
/// chrono-based calendar arithmetic.
fn add_offset_ns(ts: AbsNs, unit_str: &str, value: i64) -> PyResult<AbsNs> {
    let unit = TimeUnit::from_name(unit_str).ok_or_else(|| {
        crate::DiskgraphConfigError::new_err(format!("unsupported time unit: {}", unit_str))
    })?;
    Ok(add_offset(ts, unit, value))
}

#[pyfunction]
pub fn assign_by_ratio_on_perm(
    row_count: usize,
    ratio: Vec<f64>,
    perm: Vec<usize>,
) -> PyResult<Vec<u8>> {
    stage_assignments_by_ratio_on_perm(row_count, &ratio, &perm)
}

#[pyfunction]
#[pyo3(signature=(family, row_count, ratio, seed=None, timestamps=None))]
pub fn compute_dense_split_assignments_core(
    family: &str,
    row_count: usize,
    ratio: Vec<f64>,
    seed: Option<u64>,
    timestamps: Option<Vec<i64>>,
) -> PyResult<Vec<u8>> {
    let perm: Vec<usize> = match family {
        "FixedSplit" => (0..row_count).collect(),
        "RandomSplit" | "RandomLinkSplit" => {
            let mut p: Vec<usize> = (0..row_count).collect();
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            p.shuffle(&mut rng);
            p
        }
        "TemporalSplit" => {
            let ts = timestamps.ok_or_else(|| {
                crate::DiskgraphConfigError::new_err("timestamps are required for TemporalSplit")
            })?;
            if ts.len() != row_count {
                return Err(crate::DiskgraphConfigError::new_err(format!(
                    "timestamps length mismatch: expected {} got {}",
                    row_count,
                    ts.len()
                )));
            }
            let mut p: Vec<usize> = (0..row_count).collect();
            p.sort_by(|&a, &b| ts[a].cmp(&ts[b]).then_with(|| a.cmp(&b)));
            p
        }
        other => {
            return Err(crate::DiskgraphConfigError::new_err(format!(
                "unsupported family for Rust split core: {}",
                other
            )))
        }
    };

    stage_assignments_by_ratio_on_perm(row_count, &ratio, &perm)
}

#[pyfunction]
pub fn apply_entity_include_mask(
    assignments: Vec<u8>,
    include_mask: Vec<bool>,
) -> PyResult<Vec<u8>> {
    if assignments.len() != include_mask.len() {
        return Err(crate::DiskgraphConfigError::new_err(format!(
            "include mask length mismatch: assignments={} include_mask={}",
            assignments.len(),
            include_mask.len()
        )));
    }
    let mut out = assignments;
    for (i, keep) in include_mask.iter().enumerate() {
        if !*keep {
            out[i] = UNUSED_STAGE_ID;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn assign_manual_split(
    values: Vec<String>,
    train_values: Vec<String>,
    val_values: Vec<String>,
    test_values: Vec<String>,
) -> PyResult<Vec<u8>> {
    let train: HashSet<String> = train_values.into_iter().collect();
    let val: HashSet<String> = val_values.into_iter().collect();
    let test: HashSet<String> = test_values.into_iter().collect();

    let mut valid: HashSet<String> = HashSet::new();
    valid.extend(train.iter().cloned());
    valid.extend(val.iter().cloned());
    valid.extend(test.iter().cloned());

    let mut out = vec![UNUSED_STAGE_ID; values.len()];
    for (i, v) in values.iter().enumerate() {
        if train.contains(v) {
            out[i] = 0;
        } else if val.contains(v) {
            out[i] = 1;
        } else if test.contains(v) {
            out[i] = 2;
        }

        if !valid.contains(v) {
            return Err(crate::DiskgraphConfigError::new_err(format!(
                "Manual split value '{}' is outside configured split values",
                v
            )));
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn assign_backtesting_split(timestamps_ns: Vec<i64>, target_timestamp_ns: i64) -> Vec<u8> {
    let target = AbsNs::from_raw(target_timestamp_ns);
    let mut out = vec![UNUSED_STAGE_ID; timestamps_ns.len()];
    for (i, raw) in timestamps_ns.iter().enumerate() {
        if AbsNs::from_raw(*raw) == target {
            out[i] = 2;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature=(timestamps_ns, ratio))]
pub fn assign_approx_date_offset_split(
    timestamps_ns: Vec<i64>,
    ratio: Vec<f64>,
) -> PyResult<Vec<u8>> {
    if timestamps_ns.is_empty() {
        return Ok(Vec::new());
    }
    if ratio.len() < 2 || ratio.len() > 3 {
        return Err(crate::DiskgraphConfigError::new_err(
            "ApproxDateOffsetSplit requires 2 or 3 ratios",
        ));
    }

    // Convert to typed timestamps for grouping/sorting
    let typed: Vec<AbsNs> = timestamps_ns.iter().map(|&v| AbsNs::from_raw(v)).collect();

    let mut counts: HashMap<AbsNs, usize> = HashMap::new();
    for ts in &typed {
        *counts.entry(*ts).or_insert(0usize) += 1;
    }

    let mut groups: Vec<(AbsNs, usize)> = counts.into_iter().collect();
    groups.sort_by_key(|a| a.0);

    let total_rows: usize = groups.iter().map(|(_, c)| *c).sum();
    if total_rows == 0 {
        return Ok(vec![0u8; timestamps_ns.len()]);
    }

    let mut cum: Vec<f64> = Vec::with_capacity(groups.len());
    let mut run = 0usize;
    for (_, c) in &groups {
        run += *c;
        cum.push(run as f64 / total_rows as f64);
    }

    let num_split = ratio.len();
    let n = groups.len();

    let mut train_val_idx = cum
        .iter()
        .position(|v| *v >= ratio[0])
        .unwrap_or(n.saturating_sub(1));
    train_val_idx = train_val_idx.min(n.saturating_sub(num_split));

    let mut val_test_idx: Option<usize> = None;
    if num_split == 3 {
        let mut idx = cum
            .iter()
            .position(|v| *v >= (ratio[0] + ratio[1]))
            .unwrap_or(n.saturating_sub(1));
        idx = idx.min(n.saturating_sub(2));
        idx = idx.max(train_val_idx.saturating_add(1));
        val_test_idx = Some(idx);
    }

    let mut stage_by_ts: HashMap<AbsNs, u8> = HashMap::new();
    for (i, (ts, _)) in groups.iter().enumerate() {
        let stage = if i <= train_val_idx {
            0u8
        } else if num_split == 2 {
            1u8
        } else {
            let vt = val_test_idx.unwrap();
            if i <= vt {
                1u8
            } else {
                2u8
            }
        };
        stage_by_ts.insert(*ts, stage);
    }

    Ok(typed
        .iter()
        .map(|ts| *stage_by_ts.get(ts).unwrap_or(&UNUSED_STAGE_ID))
        .collect())
}

#[pyfunction]
#[pyo3(signature=(timestamps_ns, offsets, offset_unit="days", unit="days", timeframe=None, anchor_start_offset=0))]
pub fn assign_date_offset_split(
    timestamps_ns: Vec<i64>,
    offsets: Vec<i64>,
    offset_unit: &str,
    unit: &str,
    timeframe: Option<i64>,
    anchor_start_offset: i64,
) -> PyResult<Vec<u8>> {
    if offsets.is_empty() {
        return Err(crate::DiskgraphConfigError::new_err(
            "DateOffsetSplit requires offsets",
        ));
    }
    if timestamps_ns.is_empty() {
        return Ok(Vec::new());
    }

    // Convert to typed timestamps at the PyO3 boundary
    let typed: Vec<AbsNs> = timestamps_ns.iter().map(|&v| AbsNs::from_raw(v)).collect();
    let min_ts = *typed.iter().min().unwrap();
    let max_ts = *typed.iter().max().unwrap();

    let tf = match timeframe {
        Some(v) => v,
        None => {
            if offsets.len() == 2 {
                offsets[1] - offsets[0]
            } else {
                -offsets[0]
            }
        }
    };

    let mut boundaries: Vec<AbsNs> = vec![min_ts];
    for off in &offsets {
        boundaries.push(add_offset_ns(max_ts, offset_unit, *off)?);
    }
    boundaries.push(add_offset_ns(
        max_ts,
        unit,
        tf.saturating_add(anchor_start_offset),
    )?);

    let mut out = vec![UNUSED_STAGE_ID; typed.len()];
    for (i, ts) in typed.iter().enumerate() {
        let shifted = add_offset_ns(add_offset_ns(*ts, unit, tf)?, unit, anchor_start_offset)?;
        for (split_idx, (start, end)) in
            boundaries.iter().zip(boundaries.iter().skip(1)).enumerate()
        {
            let stage_id = if split_idx == 0 {
                0u8
            } else if split_idx == 1 {
                1u8
            } else {
                2u8
            };
            if *ts >= *start && shifted <= *end {
                out[i] = stage_id;
            }
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature=(timestamps_ns, ranges_start_ns, ranges_end_ns, timeframe=None, unit="days", anchor_start_offset=0))]
pub fn assign_time_range_split(
    timestamps_ns: Vec<i64>,
    ranges_start_ns: Vec<i64>,
    ranges_end_ns: Vec<i64>,
    timeframe: Option<i64>,
    unit: &str,
    anchor_start_offset: i64,
) -> PyResult<Vec<u8>> {
    if ranges_start_ns.len() < 2 || ranges_start_ns.len() != ranges_end_ns.len() {
        return Err(crate::DiskgraphConfigError::new_err(
            "TimeRangeSplit requires at least two ranges with matching starts/ends",
        ));
    }

    // Convert to typed timestamps at the PyO3 boundary
    let typed: Vec<AbsNs> = timestamps_ns.iter().map(|&v| AbsNs::from_raw(v)).collect();
    let starts: Vec<AbsNs> = ranges_start_ns.iter().map(|&v| AbsNs::from_raw(v)).collect();
    let ends: Vec<AbsNs> = ranges_end_ns.iter().map(|&v| AbsNs::from_raw(v)).collect();

    let mut out = vec![UNUSED_STAGE_ID; typed.len()];
    for (i, ts) in typed.iter().enumerate() {
        for split_idx in 0..starts.len().min(3) {
            let stage_id = if split_idx == 0 {
                0u8
            } else if split_idx == 1 {
                1u8
            } else {
                2u8
            };
            let start = starts[split_idx];
            let end = ends[split_idx];
            let cond = if let Some(tf) = timeframe {
                let shifted = add_offset_ns(add_offset_ns(*ts, unit, tf)?, unit, anchor_start_offset)?;
                *ts >= start && shifted <= end
            } else {
                *ts > start && *ts <= end
            };
            if cond {
                out[i] = stage_id;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_dense_split_assignments_core_random_link_seeded() {
        let a = compute_dense_split_assignments_core(
            "RandomLinkSplit",
            20,
            vec![0.8, 0.1, 0.1],
            Some(7),
            None,
        )
        .unwrap();
        let b = compute_dense_split_assignments_core(
            "RandomLinkSplit",
            20,
            vec![0.8, 0.1, 0.1],
            Some(7),
            None,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.iter().filter(|x| **x == 0).count(), 16);
        assert_eq!(a.iter().filter(|x| **x == 1).count(), 2);
        assert_eq!(a.iter().filter(|x| **x == 2).count(), 2);
    }

    #[test]
    fn test_add_offset_ns_days() {
        let ts = AbsNs::from_raw(1_000_000_000_000i64); // 1000 seconds in nanos
        let result = add_offset_ns(ts, "days", 1).unwrap();
        assert_eq!(result.raw(), 1_000_000_000_000 + 86_400_000_000_000);
    }

    #[test]
    fn test_add_offset_ns_seconds() {
        let ts = AbsNs::from_raw(0);
        let result = add_offset_ns(ts, "seconds", 5).unwrap();
        assert_eq!(result.raw(), 5_000_000_000);
    }

    #[test]
    fn test_add_offset_ns_invalid_unit() {
        let ts = AbsNs::from_raw(0);
        assert!(add_offset_ns(ts, "invalid_unit", 1).is_err());
    }

    #[test]
    fn test_backtesting_split_typed() {
        // Timestamps in nanoseconds
        let timestamps_ns = vec![
            1_000_000_000_000i64,
            2_000_000_000_000,
            3_000_000_000_000,
            2_000_000_000_000,
        ];
        let result = assign_backtesting_split(timestamps_ns, 2_000_000_000_000);
        assert_eq!(result, vec![255, 2, 255, 2]);
    }

    #[test]
    fn test_time_range_split_typed() {
        // Two ranges: [1000s, 2000s] and [2000s, 3000s] in nanoseconds
        let timestamps_ns = vec![
            1_500_000_000_000i64,
            2_500_000_000_000,
            500_000_000_000,
        ];
        let starts = vec![1_000_000_000_000i64, 2_000_000_000_000];
        let ends = vec![2_000_000_000_000i64, 3_000_000_000_000];
        let result = assign_time_range_split(timestamps_ns, starts, ends, None, "days", 0).unwrap();
        // ts > start && ts <= end: 1500 > 1000 && 1500 <= 2000 -> stage 0
        // 2500 > 2000 && 2500 <= 3000 -> stage 1
        // 500 > 1000 is false -> UNUSED
        assert_eq!(result, vec![0, 1, 255]);
    }

    #[test]
    fn test_approx_date_offset_split_typed() {
        // 5 timestamps, ratio 0.6/0.4
        let timestamps_ns = vec![
            1_000_000_000_000i64,
            2_000_000_000_000,
            3_000_000_000_000,
            4_000_000_000_000,
            5_000_000_000_000,
        ];
        let result = assign_approx_date_offset_split(timestamps_ns, vec![0.6, 0.4]).unwrap();
        // 60% of 5 = 3 rows in train, 2 in val
        assert_eq!(result.iter().filter(|&&x| x == 0).count(), 3);
        assert_eq!(result.iter().filter(|&&x| x == 1).count(), 2);
    }
}
