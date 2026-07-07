//! TT-gen accumulators — one per `AggFunc` variant.
//!
//! Each accumulator is created fresh per (entity, timestamp, op), used during
//! traversal, and finalized into a [`Value`] for the aliases Vec.

use std::collections::HashSet;

use diskgraph_core::accumulators::WelfordAccumulator;
use diskgraph_core::tt_builder::Value;

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
