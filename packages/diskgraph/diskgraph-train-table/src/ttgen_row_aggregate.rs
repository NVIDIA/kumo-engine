// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed scalar/group reductions used below both frontend execution policies.
//! Callers retain physical Arrow decoding and row selection; this module owns
//! reduction semantics, nested grouping and checked integer SUM, with no PyO3.
use crate::ttgen_accumulator::ValueBuildHasher;
use diskgraph_core::tt_builder::{
    AggregateGroup as GroupPlan, AggregateSumType as SumType, ScalarAggregate,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

pub trait AggregateScalar: Clone {
    fn null() -> Self;
    fn from_i64(value: i64) -> Self;
    fn from_u64(value: u64) -> Self;
    fn from_f64(value: f64) -> Self;
    fn is_null(&self) -> bool;
    /// A floating scalar, without converting typed integer extrema to f64.
    fn floating(&self) -> Option<f64> {
        None
    }
    fn integer(&self, unsigned: bool) -> Result<i128, String>;
    fn numeric(&self) -> Result<Option<f64>, String>;
    fn epoch_seconds(&self) -> Result<Option<i64>, String>;
    fn equality_key(&self) -> String;
    fn stable_sort_key(&self) -> (String, String);
    fn compare(&self, other: &Self) -> Result<Ordering, String>;
}

pub trait AggregateRows<V: AggregateScalar> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn values(&self, column: &str) -> Result<Vec<V>, String>;
}

#[derive(Clone)]
pub struct AggregateRow<V> {
    pub values: BTreeMap<String, V>,
}
impl<V: AggregateScalar> AggregateRows<V> for &[AggregateRow<V>] {
    fn len(&self) -> usize {
        <[AggregateRow<V>]>::len(self)
    }
    fn values(&self, column: &str) -> Result<Vec<V>, String> {
        self.iter()
            .map(|row| {
                row.values
                    .get(column)
                    .cloned()
                    .ok_or_else(|| format!("grouped row has no column {column:?}"))
            })
            .collect()
    }
}

pub fn validate_scalar_reduction(operation: &str, column: Option<&str>) -> Result<(), String> {
    const OPERATIONS: &[&str] = &[
        "count",
        "count_distinct",
        "sum",
        "mean",
        "min",
        "max",
        "std_population",
        "std_sample",
        "var_population",
        "var_sample",
        "recency",
        "mode",
    ];
    if !OPERATIONS.contains(&operation) {
        return Err(format!("unsupported scalar reduction {operation:?}"));
    }
    if operation != "count" && column.is_none() {
        return Err(format!("{operation} requires a column"));
    }
    Ok(())
}

pub fn reduce_rows<V: AggregateScalar>(
    rows: &impl AggregateRows<V>,
    plan: &ScalarAggregate,
    cutoff: i64,
) -> Result<V, String> {
    let operation = plan.operation.as_str();
    let column = plan.column.as_deref();
    let sum_type = plan.sum_type;
    validate_scalar_reduction(operation, column)?;
    if sum_type.is_some() && operation != "sum" {
        return Err("sum_type is only valid for sum".into());
    }
    if operation == "count" {
        return i64::try_from(rows.len())
            .map(V::from_i64)
            .map_err(|_| "count exceeds int64 range".to_string());
    }
    let values = rows
        .values(column.expect("validated reduction column"))?
        .into_iter()
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    if operation == "count_distinct" {
        return i64::try_from(
            values
                .iter()
                .map(AggregateScalar::equality_key)
                .collect::<HashSet<_, ValueBuildHasher>>()
                .len(),
        )
        .map(V::from_i64)
        .map_err(|_| "count_distinct exceeds int64 range".to_string());
    }
    if operation == "sum"
        && (rows.len() == 0 || values.is_empty() && plan.semantics.all_null_sum_is_zero)
    {
        return Ok(match sum_type {
            Some(SumType::Int64) => V::from_i64(0),
            Some(SumType::Uint64) => V::from_u64(0),
            None => V::from_f64(0.0),
        });
    }
    if values.is_empty() {
        return Ok(V::null());
    }
    if operation == "sum" {
        if let Some(sum_type) = sum_type {
            let total = values.iter().try_fold(0i128, |total, value| {
                let value = value.integer(matches!(sum_type, SumType::Uint64))?;
                total
                    .checked_add(value)
                    .ok_or_else(|| "integer sum exceeds accumulator range".to_string())
            })?;
            return match sum_type {
                SumType::Int64 => i64::try_from(total)
                    .map(V::from_i64)
                    .map_err(|_| "integer sum exceeds int64 range".to_string()),
                SumType::Uint64 => u64::try_from(total)
                    .map(V::from_u64)
                    .map_err(|_| "integer sum exceeds uint64 range".to_string()),
            };
        }
    }
    if operation == "recency" {
        let latest = values
            .iter()
            .map(AggregateScalar::epoch_seconds)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .max()
            .expect("non-null timestamp values");
        return Ok(V::from_f64(cutoff.saturating_sub(latest) as f64));
    }
    if operation == "mode" {
        let mut counts = HashMap::<String, (usize, V), ValueBuildHasher>::default();
        for value in values {
            let entry = counts
                .entry(value.equality_key())
                .or_insert_with(|| (0, value.clone()));
            entry.0 += 1;
        }
        return Ok(counts
            .into_values()
            .max_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.stable_sort_key().cmp(&left.1.stable_sort_key()))
            })
            .expect("non-empty mode")
            .1);
    }
    if (operation == "min" || operation == "max")
        && plan.semantics.keep_first_equal
        && values
            .iter()
            .all(|value| value.floating().is_some_and(f64::is_finite))
    {
        use crate::ttgen_accumulator::Accumulator;
        use diskgraph_core::tt_builder::{AggFunc, Value};
        let mut state = Accumulator::new(
            if operation == "min" {
                AggFunc::Min
            } else {
                AggFunc::Max
            },
            None,
        );
        state.accumulate_numeric(values.iter().map(AggregateScalar::floating), true, false);
        return match state.finalize() {
            Some(Value::Float(value)) => Ok(V::from_f64(value)),
            _ => unreachable!("nonempty finite extrema"),
        };
    }
    if operation == "min" || operation == "max" {
        let mut best = values[0].clone();
        for value in &values[1..] {
            let ordering = value.compare(&best)?;
            if (operation == "min" && ordering == Ordering::Less)
                || (operation == "max" && ordering == Ordering::Greater)
                || (!plan.semantics.keep_first_equal && ordering == Ordering::Equal)
            {
                best = value.clone();
            }
        }
        return Ok(best);
    }

    let numbers = values
        .iter()
        .map(AggregateScalar::numeric)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if operation == "sum" || operation == "mean" {
        use crate::ttgen_accumulator::Accumulator;
        use diskgraph_core::tt_builder::{AggFunc, Value};
        let mut state = Accumulator::new(
            if operation == "sum" {
                AggFunc::Sum
            } else {
                AggFunc::Avg
            },
            None,
        );
        state.accumulate_numeric(numbers.into_iter().map(Some), true, false);
        return match state.finalize() {
            Some(Value::Float(value)) => Ok(V::from_f64(value)),
            None => Ok(V::null()),
            _ => unreachable!("sum/mean produce float scalars"),
        };
    }
    use diskgraph_core::accumulators::{update_running_mean, WelfordAccumulator};
    let mut moments = WelfordAccumulator::new();
    for value in numbers {
        let (_, delta, _) = update_running_mean(&mut moments.count, &mut moments.m1, value);
        // Preserve row-aggregate rounding, including nonfinite observations.
        moments.m2 += delta * (value - moments.m1);
    }
    let count = moments.count;
    let m2 = moments.m2;
    Ok(match operation {
        "var_population" => V::from_f64(m2 / count as f64),
        "std_population" => V::from_f64((m2 / count as f64).sqrt()),
        "var_sample" if count >= 2 => V::from_f64(m2 / (count - 1) as f64),
        "std_sample" if count >= 2 => V::from_f64((m2 / (count - 1) as f64).sqrt()),
        "var_sample" | "std_sample" => V::null(),
        _ => return Err(format!("unsupported numeric reduction {operation:?}")),
    })
}

pub fn group_rows<V: AggregateScalar>(
    rows: &impl AggregateRows<V>,
    plan: &GroupPlan,
    cutoff: i64,
) -> Result<Vec<AggregateRow<V>>, String> {
    if plan.keys.is_empty() || plan.values.is_empty() {
        return Err("group_by requires keys and values".into());
    }
    let key_columns = plan
        .keys
        .iter()
        .map(|key| rows.values(key))
        .collect::<Result<Vec<_>, _>>()?;
    let mut groups = BTreeMap::<Vec<(String, String)>, Vec<usize>>::new();
    for row in 0..rows.len() {
        let key = key_columns
            .iter()
            .map(|values| values[row].stable_sort_key())
            .collect::<Vec<_>>();
        groups.entry(key).or_default().push(row);
    }
    let value_columns = plan
        .values
        .iter()
        .map(|value| {
            validate_scalar_reduction(&value.operation, value.column.as_deref())?;
            if value.sum_type.is_some() && value.operation != "sum" {
                return Err("sum_type is only valid for sum".to_string());
            }
            Ok(match &value.column {
                Some(column) => Some(rows.values(column)?),
                None => None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut output = Vec::with_capacity(groups.len());
    for indices in groups.into_values() {
        let mut values = BTreeMap::new();
        for (key_index, key) in plan.keys.iter().enumerate() {
            values.insert(key.clone(), key_columns[key_index][indices[0]].clone());
        }
        for (reduction_index, reduction) in plan.values.iter().enumerate() {
            let selected = value_columns[reduction_index]
                .as_ref()
                .map(|column| {
                    indices
                        .iter()
                        .map(|index| AggregateRow {
                            values: BTreeMap::from([(
                                reduction.column.clone().expect("column present"),
                                column[*index].clone(),
                            )]),
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| {
                    vec![
                        AggregateRow {
                            values: BTreeMap::new()
                        };
                        indices.len()
                    ]
                });
            values.insert(
                reduction.name.clone(),
                reduce_rows(
                    &selected.as_slice(),
                    &ScalarAggregate {
                        name: reduction.name.clone(),
                        frame: String::new(),
                        groups: Vec::new(),
                        operation: reduction.operation.clone(),
                        column: reduction.column.clone(),
                        sum_type: reduction.sum_type,
                        semantics: Default::default(),
                        row_filter: None,
                        limit: None,
                        target_entity_table: None,
                    },
                    cutoff,
                )?,
            );
        }
        output.push(AggregateRow { values });
    }
    Ok(output)
}

/// Union of overlapping half-open windows. None is an unbounded endpoint;
/// endpoint equality alone is not overlap for (lo,hi] windows.
pub fn merge_window_bounds<T: Copy + Ord>(
    left: (Option<T>, Option<T>),
    right: (Option<T>, Option<T>),
) -> Option<(Option<T>, Option<T>)> {
    if left.1.zip(right.0).is_some_and(|(hi, lo)| hi <= lo)
        || right.1.zip(left.0).is_some_and(|(hi, lo)| hi <= lo)
    {
        return None;
    }
    let lower = match (left.0, right.0) {
        (Some(a), Some(b)) => Some(a.min(b)),
        _ => None,
    };
    let upper = match (left.1, right.1) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };
    Some((lower, upper))
}
