use chrono::{DateTime, Months, TimeZone, Utc};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};

const UNUSED_STAGE_ID: u8 = 255;
const NS_PER_SEC: i64 = 1_000_000_000;

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

fn ns_to_dt(ts_ns: i64) -> PyResult<DateTime<Utc>> {
    let secs = ts_ns.div_euclid(NS_PER_SEC);
    let nanos = ts_ns.rem_euclid(NS_PER_SEC) as u32;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .ok_or_else(|| crate::DiskgraphConfigError::new_err(format!("invalid timestamp ns: {}", ts_ns)))
}

fn dt_to_ns(dt: DateTime<Utc>) -> PyResult<i64> {
    dt.timestamp_nanos_opt()
        .ok_or_else(|| crate::DiskgraphConfigError::new_err("timestamp nanos overflow"))
}

fn add_unit(ts_ns: i64, unit: &str, value: i64) -> PyResult<i64> {
    match unit {
        "seconds" => Ok(ts_ns.saturating_add(value.saturating_mul(NS_PER_SEC))),
        "minutes" => Ok(ts_ns.saturating_add(value.saturating_mul(60 * NS_PER_SEC))),
        "hours" => Ok(ts_ns.saturating_add(value.saturating_mul(3600 * NS_PER_SEC))),
        "days" => Ok(ts_ns.saturating_add(value.saturating_mul(86_400 * NS_PER_SEC))),
        "weeks" => Ok(ts_ns.saturating_add(value.saturating_mul(7 * 86_400 * NS_PER_SEC))),
        "milliseconds" => Ok(ts_ns.saturating_add(value.saturating_mul(1_000_000))),
        "microseconds" => Ok(ts_ns.saturating_add(value.saturating_mul(1_000))),
        "nanoseconds" => Ok(ts_ns.saturating_add(value)),
        "months" => {
            let dt = ns_to_dt(ts_ns)?;
            let months = value.unsigned_abs() as u32;
            let out = if value >= 0 {
                dt.checked_add_months(Months::new(months))
            } else {
                dt.checked_sub_months(Months::new(months))
            }
            .ok_or_else(|| crate::DiskgraphConfigError::new_err("month offset overflow"))?;
            dt_to_ns(out)
        }
        "years" => {
            let dt = ns_to_dt(ts_ns)?;
            let months = value.saturating_mul(12).unsigned_abs() as u32;
            let out = if value >= 0 {
                dt.checked_add_months(Months::new(months))
            } else {
                dt.checked_sub_months(Months::new(months))
            }
            .ok_or_else(|| crate::DiskgraphConfigError::new_err("year offset overflow"))?;
            dt_to_ns(out)
        }
        other => Err(crate::DiskgraphConfigError::new_err(format!("unsupported time unit: {}", other))),
    }
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
        "RandomSplit" => {
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
pub fn apply_entity_include_mask(assignments: Vec<u8>, include_mask: Vec<bool>) -> PyResult<Vec<u8>> {
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
pub fn assign_manual_split(values: Vec<String>, train_values: Vec<String>, val_values: Vec<String>, test_values: Vec<String>) -> PyResult<Vec<u8>> {
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
    let mut out = vec![UNUSED_STAGE_ID; timestamps_ns.len()];
    for (i, ts) in timestamps_ns.iter().enumerate() {
        if *ts == target_timestamp_ns {
            out[i] = 2;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature=(timestamps_ns, ratio))]
pub fn assign_approx_date_offset_split(timestamps_ns: Vec<i64>, ratio: Vec<f64>) -> PyResult<Vec<u8>> {
    if timestamps_ns.is_empty() {
        return Ok(Vec::new());
    }
    if ratio.len() < 2 || ratio.len() > 3 {
        return Err(crate::DiskgraphConfigError::new_err("ApproxDateOffsetSplit requires 2 or 3 ratios"));
    }

    let mut counts: HashMap<i64, usize> = HashMap::new();
    for ts in &timestamps_ns {
        *counts.entry(*ts).or_insert(0usize) += 1;
    }

    let mut groups: Vec<(i64, usize)> = counts.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0));

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

    let mut stage_by_ts: HashMap<i64, u8> = HashMap::new();
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

    Ok(timestamps_ns
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
        return Err(crate::DiskgraphConfigError::new_err("DateOffsetSplit requires offsets"));
    }
    if timestamps_ns.is_empty() {
        return Ok(Vec::new());
    }

    let min_ts = *timestamps_ns.iter().min().unwrap();
    let max_ts = *timestamps_ns.iter().max().unwrap();

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

    let mut boundaries: Vec<i64> = vec![min_ts];
    for off in &offsets {
        boundaries.push(add_unit(max_ts, offset_unit, *off)?);
    }
    boundaries.push(add_unit(max_ts, unit, tf.saturating_add(anchor_start_offset))?);

    let mut out = vec![UNUSED_STAGE_ID; timestamps_ns.len()];
    for (i, ts) in timestamps_ns.iter().enumerate() {
        let shifted = add_unit(add_unit(*ts, unit, tf)?, unit, anchor_start_offset)?;
        for (split_idx, (start, end)) in boundaries.iter().zip(boundaries.iter().skip(1)).enumerate()
        {
            let stage_id = if split_idx == 0 { 0u8 } else if split_idx == 1 { 1u8 } else { 2u8 };
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

    let mut out = vec![UNUSED_STAGE_ID; timestamps_ns.len()];
    for (i, ts) in timestamps_ns.iter().enumerate() {
        for split_idx in 0..ranges_start_ns.len().min(3) {
            let stage_id = if split_idx == 0 { 0u8 } else if split_idx == 1 { 1u8 } else { 2u8 };
            let start = ranges_start_ns[split_idx];
            let end = ranges_end_ns[split_idx];
            let cond = if let Some(tf) = timeframe {
                let shifted = add_unit(add_unit(*ts, unit, tf)?, unit, anchor_start_offset)?;
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
