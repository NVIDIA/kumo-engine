// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TT-gen accumulators — one per `AggFunc` variant.
//!
//! Each accumulator is created fresh per (entity, timestamp, op), used during
//! traversal, and finalized into a [`Value`] for the aliases Vec.

use std::collections::HashSet;

use crate::ttgen_resolve::ResolvedAggregate;
use arrow_array::{ArrayRef, Float32Array, Float64Array};
use diskgraph_core::accumulators::WelfordAccumulator;
use diskgraph_core::time::UnitSystem;
use diskgraph_core::tt_builder::Value;

/// Update compatible PQL-resolved reductions from caller-ordered projected
/// columns. Both frontends keep their selected row order and empty-SUM policy.
/// False means the caller must use its existing general evaluator; no result
/// from these partially updated states may then be published.
pub fn accumulate_numeric_batch<U: UnitSystem>(
    operations: &[&ResolvedAggregate<U>],
    states: &mut [Accumulator],
    observed: &mut [usize],
    columns: &[(usize, ArrayRef)],
    rows: std::ops::Range<usize>,
    keep_first_equal: bool,
) -> bool {
    // Eligibility is invariant for the whole batch, including empty batches.
    if operations.iter().any(|operation| {
        !matches!(
            operation.func,
            diskgraph_core::tt_builder::AggFunc::Count
                | diskgraph_core::tt_builder::AggFunc::Sum
                | diskgraph_core::tt_builder::AggFunc::Avg
                | diskgraph_core::tt_builder::AggFunc::Min
                | diskgraph_core::tt_builder::AggFunc::Max
        )
    }) {
        return false;
    }
    // Resolve physical arrays once; check finiteness once per projected
    // column, even when several reductions share it.
    let arrays = columns
        .iter()
        .map(|(slot, array)| {
            if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
                let array = array.slice(rows.start, rows.len());
                if array.iter().flatten().any(|v| !v.is_finite()) {
                    return None;
                }
                Some((*slot, Some(array), None))
            } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
                let array = array.slice(rows.start, rows.len());
                if array.iter().flatten().any(|v| !v.is_finite()) {
                    return None;
                }
                Some((*slot, None, Some(array)))
            } else {
                // Convert once per projected integer array. Preserve PQL's
                // floating output without adding an Arrow compute dependency.
                macro_rules! integer_array {
                    ($($ty:ty),*) => { $(
                        if let Some(values) = array.as_any().downcast_ref::<$ty>() {
                            let values = values.slice(rows.start, rows.len());
                            let floats = Float64Array::from_iter(values.iter().map(|v| v.map(|v| v as f64)));
                            return Some((*slot, None, Some(floats)));
                        }
                    )* };
                }
                integer_array!(
                    arrow_array::Int8Array, arrow_array::Int16Array,
                    arrow_array::Int32Array, arrow_array::Int64Array,
                    arrow_array::UInt8Array, arrow_array::UInt16Array,
                    arrow_array::UInt32Array, arrow_array::UInt64Array
                );
                None
            }
        })
        .collect::<Option<Vec<_>>>();
    let Some(arrays) = arrays else {
        return false;
    };
    // Dispatch once per operation/column, not once per row. Rows remain in
    // caller order: vectorized SUM/mean would change floating cancellation.
    for ((operation, state), count) in operations.iter().zip(states).zip(observed) {
        if operation.func == diskgraph_core::tt_builder::AggFunc::Count {
            let Accumulator::Count(total) = state else {
                return false;
            };
            *total += rows.len() as u64;
            continue;
        }
        let Some((_, f32s, f64s)) = operation
            .target_col_idx
            .and_then(|column| arrays.iter().find(|(index, _, _)| *index == column))
        else {
            return false;
        };
        *count += if let Some(array) = f32s {
            state.accumulate_numeric(
                array.iter().map(|v| v.map(f64::from)),
                keep_first_equal,
                true,
            )
        } else {
            state.accumulate_numeric(f64s.as_ref().unwrap().iter(), keep_first_equal, true)
        };
    }
    true
}

/// Runtime accumulator for a single aggregation op.
pub enum Accumulator {
    Count(u64),
    Sum(f64),
    Avg(WelfordAccumulator),
    Min(f64),
    Max(f64),
    CountDistinct(HashSet<u64>),
    ListDistinct {
        set: HashSet<u64>,
        /// Maps hash → original value for output reconstruction.
        values: Vec<Value>,
        limit: Option<u32>,
    },
    First(Option<Value>),
    Last(Option<Value>),
}

impl Accumulator {
    /// Observe one selected row for scalar numeric reductions. A NULL value
    /// still counts for COUNT(*); the other reductions skip it.
    #[inline]
    pub fn observe_numeric_row(&mut self, value: Option<f64>, keep_first_equal: bool) {
        self.accumulate_numeric(std::iter::once(value), keep_first_equal, true);
    }

    /// Ordered scalar reduction shared by projected Arrow batches and general
    /// frames (including grouped/multi-hop inputs). Only AVG's nonfinite policy
    /// differs: PQL skips nonfinite values, row aggregates propagate them.
    #[inline]
    pub fn accumulate_numeric(
        &mut self,
        values: impl Iterator<Item = Option<f64>>,
        keep_first_equal: bool,
        finite_mean_only: bool,
    ) -> usize {
        let mut observed = 0;
        match self {
            Self::Count(count) => {
                for value in values {
                    *count += 1;
                    observed += usize::from(value.is_some());
                }
            }
            Self::Sum(sum) => {
                for value in values.flatten() {
                    *sum += value;
                    observed += 1;
                }
            }
            Self::Avg(state) => {
                for value in values.flatten() {
                    observed += 1;
                    if !finite_mean_only || value.is_finite() {
                        diskgraph_core::accumulators::update_running_mean(
                            &mut state.count,
                            &mut state.m1,
                            value,
                        );
                    }
                }
            }
            Self::Min(minimum) => {
                for value in values.flatten() {
                    if !keep_first_equal || value < *minimum {
                        *minimum = minimum.min(value);
                    }
                    observed += 1;
                }
            }
            Self::Max(maximum) => {
                for value in values.flatten() {
                    if !keep_first_equal || value > *maximum {
                        *maximum = maximum.max(value);
                    }
                    observed += 1;
                }
            }
            _ => unreachable!("numeric rows require a scalar numeric accumulator"),
        }
        observed
    }

    /// Create a fresh accumulator for the given function.
    pub fn new(func: diskgraph_core::tt_builder::AggFunc, limit: Option<u32>) -> Self {
        use diskgraph_core::tt_builder::AggFunc;
        match func {
            AggFunc::Count => Accumulator::Count(0),
            AggFunc::Sum => Accumulator::Sum(0.0),
            AggFunc::Avg => Accumulator::Avg(WelfordAccumulator::new()),
            AggFunc::Min => Accumulator::Min(f64::INFINITY),
            AggFunc::Max => Accumulator::Max(f64::NEG_INFINITY),
            AggFunc::CountDistinct => Accumulator::CountDistinct(HashSet::new()),
            AggFunc::ListDistinct => Accumulator::ListDistinct {
                set: HashSet::new(),
                values: Vec::new(),
                limit,
            },
            AggFunc::First => Accumulator::First(None),
            AggFunc::Last => Accumulator::Last(None),
        }
    }

    /// Convert the accumulated state into a final output value.
    ///
    /// Returns `None` for accumulators with no observed data (AVG, MIN, MAX,
    /// FIRST, LAST with empty traversals). This matches Kumo's PySpark
    /// behavior where an INNER temporal join naturally drops entity/timestamp
    /// pairs with no matching rows.
    pub fn finalize(self) -> Option<Value> {
        match self {
            Accumulator::Count(n) => Some(Value::Int(n as i64)),
            Accumulator::Sum(x) => Some(Value::Float(x)),
            Accumulator::Avg(w) => {
                if w.count == 0 {
                    None
                } else {
                    Some(Value::Float(w.mean()))
                }
            }
            Accumulator::Min(x) => {
                if x == f64::INFINITY {
                    None // no values observed
                } else {
                    Some(Value::Float(x))
                }
            }
            Accumulator::Max(x) => {
                if x == f64::NEG_INFINITY {
                    None // no values observed
                } else {
                    Some(Value::Float(x))
                }
            }
            Accumulator::CountDistinct(set) => Some(Value::Int(set.len() as i64)),
            Accumulator::ListDistinct { values, .. } => Some(Value::List(values)),
            Accumulator::First(v) | Accumulator::Last(v) => v,
        }
    }
}

/// Hash a Value for use in CountDistinct / ListDistinct sets.
pub fn hash_value(v: &Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match v {
        Value::Int(i) => i.hash(&mut hasher),
        Value::Float(f) => f.to_bits().hash(&mut hasher),
        Value::Str(s) => s.hash(&mut hasher),
        Value::Bool(b) => b.hash(&mut hasher),
        Value::List(vs) => {
            vs.len().hash(&mut hasher);
            for item in vs {
                hash_value(item).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Hash a borrowed string slice without first wrapping it in a
/// `Value::Str(String)`. Must agree with `hash_value(&Value::Str(s.into()))`
/// so multicat-aware seen-maps stay coherent with the legacy hash path.
#[inline]
pub fn hash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Hash an `i64` token. Must agree with `hash_value(&Value::Int(n))`.
#[inline]
pub fn hash_i64(n: i64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    n.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_numeric_reduction_policies() {
        use diskgraph_core::tt_builder::AggFunc;
        let values = [Some(1e16), None, Some(-1e16), Some(1.0)];
        let mut sum = Accumulator::new(AggFunc::Sum, None);
        assert_eq!(sum.accumulate_numeric(values.into_iter(), true, false), 3);
        assert_eq!(sum.finalize(), Some(Value::Float(1.0)));
        for finite_only in [false, true] {
            let mut mean = Accumulator::new(AggFunc::Avg, None);
            mean.accumulate_numeric(
                [Some(f64::INFINITY), Some(2.0)].into_iter(),
                true,
                finite_only,
            );
            let Some(Value::Float(value)) = mean.finalize() else {
                panic!("nonempty mean")
            };
            if finite_only {
                assert_eq!(value, 2.0);
            } else {
                assert!(value.is_nan());
            }
        }
        for function in [AggFunc::Min, AggFunc::Max] {
            let mut state = Accumulator::new(function, None);
            state.accumulate_numeric([Some(-0.0), Some(0.0)].into_iter(), true, false);
            let Some(Value::Float(value)) = state.finalize() else {
                panic!("nonempty extremum")
            };
            assert_eq!(value.to_bits(), (-0.0f64).to_bits());
        }
    }

    #[test]
    fn test_count() {
        let acc = Accumulator::Count(42);
        assert_eq!(acc.finalize(), Some(Value::Int(42)));
    }

    #[test]
    fn test_sum() {
        let acc = Accumulator::Sum(std::f64::consts::PI);
        assert_eq!(acc.finalize(), Some(Value::Float(std::f64::consts::PI)));
    }

    #[test]
    fn test_avg() {
        let mut w = WelfordAccumulator::new();
        w.observe(1.0);
        w.observe(2.0);
        w.observe(3.0);
        let acc = Accumulator::Avg(w);
        match acc.finalize() {
            Some(Value::Float(f)) => assert!((f - 2.0).abs() < 1e-10),
            other => panic!("expected Some(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_avg_empty() {
        let w = WelfordAccumulator::new();
        let acc = Accumulator::Avg(w);
        assert_eq!(acc.finalize(), None);
    }

    #[test]
    fn test_min_max() {
        let acc_min = Accumulator::Min(1.5);
        assert_eq!(acc_min.finalize(), Some(Value::Float(1.5)));

        let acc_max = Accumulator::Max(99.9);
        assert_eq!(acc_max.finalize(), Some(Value::Float(99.9)));
    }

    #[test]
    fn test_min_max_empty() {
        let acc_min = Accumulator::Min(f64::INFINITY);
        assert_eq!(acc_min.finalize(), None);

        let acc_max = Accumulator::Max(f64::NEG_INFINITY);
        assert_eq!(acc_max.finalize(), None);
    }

    #[test]
    fn test_count_distinct() {
        let mut set = HashSet::new();
        set.insert(1u64);
        set.insert(2);
        set.insert(3);
        set.insert(1); // duplicate
        let acc = Accumulator::CountDistinct(set);
        assert_eq!(acc.finalize(), Some(Value::Int(3)));
    }

    #[test]
    fn test_first_last() {
        let acc_first = Accumulator::First(Some(Value::Int(10)));
        assert_eq!(acc_first.finalize(), Some(Value::Int(10)));

        let acc_last = Accumulator::Last(Some(Value::Str("hello".into())));
        assert_eq!(acc_last.finalize(), Some(Value::Str("hello".into())));

        let acc_none = Accumulator::First(None);
        assert_eq!(acc_none.finalize(), None);
    }

    #[test]
    fn test_first_empty_is_none() {
        let acc = Accumulator::First(None);
        assert_eq!(acc.finalize(), None);
    }

    #[test]
    fn test_last_empty_is_none() {
        let acc = Accumulator::Last(None);
        assert_eq!(acc.finalize(), None);
    }

    #[test]
    fn test_count_empty() {
        let acc = Accumulator::Count(0);
        assert_eq!(acc.finalize(), Some(Value::Int(0)));
    }

    #[test]
    fn test_sum_empty() {
        let acc = Accumulator::Sum(0.0);
        assert_eq!(acc.finalize(), Some(Value::Float(0.0)));
    }

    #[test]
    fn test_count_distinct_empty() {
        let acc = Accumulator::CountDistinct(HashSet::new());
        assert_eq!(acc.finalize(), Some(Value::Int(0)));
    }

    #[test]
    fn test_list_distinct_empty() {
        let acc = Accumulator::ListDistinct {
            set: HashSet::new(),
            values: vec![],
            limit: None,
        };
        assert_eq!(acc.finalize(), Some(Value::List(vec![])));
    }

    #[test]
    fn test_list_distinct_with_values() {
        let mut set = HashSet::new();
        let v1 = Value::Str("a".into());
        let v2 = Value::Str("b".into());
        set.insert(hash_value(&v1));
        set.insert(hash_value(&v2));
        let acc = Accumulator::ListDistinct {
            set,
            values: vec![v1.clone(), v2.clone()],
            limit: None,
        };
        match acc.finalize() {
            Some(Value::List(v)) => assert_eq!(v.len(), 2),
            other => panic!("expected Some(List), got {:?}", other),
        }
    }

    #[test]
    fn test_list_distinct_with_limit() {
        let mut set = HashSet::new();
        let vals: Vec<Value> = (0..5).map(Value::Int).collect();
        for v in &vals {
            set.insert(hash_value(v));
        }
        // Limit doesn't affect finalize — it's enforced during accumulation.
        // finalize returns whatever values are stored.
        let acc = Accumulator::ListDistinct {
            set,
            values: vals.clone(),
            limit: Some(3),
        };
        match acc.finalize() {
            Some(Value::List(v)) => assert_eq!(v.len(), 5), // all values stored
            other => panic!("expected Some(List), got {:?}", other),
        }
    }

    #[test]
    fn test_first_with_float() {
        let acc = Accumulator::First(Some(Value::Float(std::f64::consts::PI)));
        assert_eq!(acc.finalize(), Some(Value::Float(std::f64::consts::PI)));
    }

    #[test]
    fn test_last_with_float() {
        let acc = Accumulator::Last(Some(Value::Float(2.71)));
        assert_eq!(acc.finalize(), Some(Value::Float(2.71)));
    }

    #[test]
    fn test_sum_negative() {
        let acc = Accumulator::Sum(-100.5);
        assert_eq!(acc.finalize(), Some(Value::Float(-100.5)));
    }

    #[test]
    fn test_hash_value() {
        let h1 = hash_value(&Value::Int(42));
        let h2 = hash_value(&Value::Int(42));
        assert_eq!(h1, h2);

        let h3 = hash_value(&Value::Int(43));
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_value_different_types() {
        let h_int = hash_value(&Value::Int(5));
        let h_float = hash_value(&Value::Float(5.0));
        let h_str = hash_value(&Value::Str("5".into()));
        // Different types should produce different hashes
        assert_ne!(h_int, h_float);
        assert_ne!(h_int, h_str);
    }

    #[test]
    fn test_hash_value_bool() {
        let h_true = hash_value(&Value::Bool(true));
        let h_false = hash_value(&Value::Bool(false));
        assert_ne!(h_true, h_false);
    }

    #[test]
    fn test_hash_value_list() {
        let h1 = hash_value(&Value::List(vec![Value::Int(1), Value::Int(2)]));
        let h2 = hash_value(&Value::List(vec![Value::Int(1), Value::Int(2)]));
        let h3 = hash_value(&Value::List(vec![Value::Int(2), Value::Int(1)]));
        assert_eq!(h1, h2);
        assert_ne!(h1, h3); // order matters
    }
}
