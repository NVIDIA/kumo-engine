//! Filter evaluation for row-level and stage-level filters.

use diskgraph_core::tt_builder::{CompareOp, Value};
use std::sync::Arc;

use diskgraph_index::graph::TableReader;

use crate::ttgen_resolve::{
    ResolvedRowFilter, ResolvedRowFilterLhs, ResolvedRowFilterRhs,
    ResolvedStageFilter, ResolvedStageFilterRhs, ResolvedEdge,
};
use crate::ttgen_traversal::{
    dispatch_multicat_source, read_column_value, MulticatDispatchOutcome, MulticatToken,
};

pub(crate) fn is_nullish_filter_value(value: &Option<Value>) -> bool {
    match value {
        None => true,
        Some(Value::Float(v)) => v.is_nan(),
        Some(_) => false,
    }
}

// ─── Row filter evaluation (per-destination, during traversal) ───────────────

/// Evaluate a row filter on a destination node during traversal.
///
/// `entity_cols` are pre-resolved entity column values (indexed by position
/// in the entity_col_indices array).
pub fn eval_row_filter(
    filter: &ResolvedRowFilter,
    current_node: u64,
    current_table: &Arc<TableReader>,
    entity_cols: &[Option<Value>],
) -> bool {
    match filter {
        ResolvedRowFilter::Compare { lhs, op, rhs } => {
            // Membership (CONTAINS / NOT CONTAINS) is element-wise over a
            // multicategorical (native-list) column. It cannot go through the
            // scalar LHS read (`value_ref_to_value` collapses arrays to Int(0))
            // nor scalar `compare_values`; route it to per-token decode.
            if matches!(op, CompareOp::Contains | CompareOp::NotContains) {
                let r = match resolve_row_rhs(rhs, current_node, current_table, entity_cols) {
                    Some(v) => v,
                    None => return false,
                };
                return eval_row_membership(lhs, op, &r, current_node, current_table);
            }

            let l_opt = resolve_row_lhs(lhs, current_node, current_table);

            match op {
                CompareOp::IsNull => return is_nullish_filter_value(&l_opt),
                CompareOp::IsNotNull => return !is_nullish_filter_value(&l_opt),
                _ => {}
            }

            let l = match l_opt {
                Some(v) => v,
                None => return false, // SQL tri-state: op with NULL evaluates to NULL (fails filter)
            };
            let r = match resolve_row_rhs(rhs, current_node, current_table, entity_cols) {
                Some(v) => v,
                None => return false,
            };
            compare_values(&l, op, &r)
        }
        ResolvedRowFilter::And(filters) => filters
            .iter()
            .all(|f| eval_row_filter(f, current_node, current_table, entity_cols)),
        ResolvedRowFilter::Or(filters) => filters
            .iter()
            .any(|f| eval_row_filter(f, current_node, current_table, entity_cols)),
        ResolvedRowFilter::Not(f) => !eval_row_filter(f, current_node, current_table, entity_cols),
    }
}

fn resolve_row_lhs(
    lhs: &ResolvedRowFilterLhs,
    current_node: u64,
    current_table: &Arc<TableReader>,
) -> Option<Value> {
    match lhs {
        ResolvedRowFilterLhs::Column { col_idx, .. } => {
            read_column_value(current_table, current_node, *col_idx)
        }
        ResolvedRowFilterLhs::Remote { path, col_idx } => {
            resolve_remote_value(path, current_node, *col_idx)
        }
    }
}

/// Evaluate `CONTAINS` / `NOT CONTAINS` on the current row's LHS column.
///
/// `Contains`  → true iff the source column has a token equal to `rhs`.
/// `NotContains` → true iff the source is non-null and has *no* matching token
///                  (SQL tri-state: a NULL source yields NULL, i.e. fails).
///
/// Only a multicategorical (native-list) LHS column reaches the per-token
/// path. A scalar (non-multicat) LHS with a membership op degrades to a single
/// whole-cell equality test, so `genres = 'Comedy'` written as a CONTAINS still
/// behaves sanely rather than silently matching nothing.
fn eval_row_membership(
    lhs: &ResolvedRowFilterLhs,
    op: &CompareOp,
    rhs: &Value,
    current_node: u64,
    current_table: &Arc<TableReader>,
) -> bool {
    let (col_idx, is_multicat) = match lhs {
        ResolvedRowFilterLhs::Column {
            col_idx,
            semantic_type,
        } => (
            *col_idx,
            semantic_type.as_deref() == Some("multicategorical"),
        ),
        // Remote (sub-traversal) membership is not produced by the compiler.
        // A scalar `Contains`/`NotContains` here is a `LIKE '%x%'` substring
        // test, so route the whole-cell value through `compare_values` (which
        // does `a.contains(b)` for strings) rather than equality.
        ResolvedRowFilterLhs::Remote { path, col_idx } => {
            let value = match resolve_remote_value(path, current_node, *col_idx) {
                Some(v) => v,
                // NULL source: SQL tri-state — both Contains and NotContains
                // evaluate to NULL, i.e. fail the filter.
                None => return false,
            };
            return compare_values(&value, op, rhs);
        }
    };

    // Scalar (non-multicat) LHS: `Contains`/`NotContains` is a `LIKE '%x%'`
    // substring test on a string column, not per-token membership. Read the
    // whole cell and delegate to `compare_values` (`compare_str` does the
    // substring match), matching the behavior of every other scalar op.
    if !is_multicat {
        let value = match read_column_value(current_table, current_node, col_idx) {
            Some(v) => v,
            // NULL source fails the filter (tri-state) for both ops.
            None => return false,
        };
        return compare_values(&value, op, rhs);
    }

    let mut any_match = false;
    let outcome = dispatch_multicat_source(
        current_table,
        current_node,
        col_idx,
        is_multicat,
        |tok| {
            if token_matches(&tok, rhs) {
                any_match = true;
                false // early-exit: one match settles Contains
            } else {
                true
            }
        },
    );

    match op {
        CompareOp::Contains => any_match,
        // NOT CONTAINS: NULL source → filter fails (tri-state); present source
        // with no matching token → passes.
        CompareOp::NotContains => {
            outcome != MulticatDispatchOutcome::SourceNull && !any_match
        }
        _ => unreachable!("eval_row_membership only handles Contains/NotContains"),
    }
}

/// One decoded multicat token equals the RHS constant.
fn token_matches(tok: &MulticatToken<'_>, rhs: &Value) -> bool {
    match (tok, rhs) {
        (MulticatToken::Str(s), Value::Str(r)) => *s == r.as_str(),
        (MulticatToken::Int(n), Value::Int(r)) => n == r,
        (MulticatToken::Int(n), Value::Float(r)) => (*n as f64) == *r,
        // Legacy whole-cell (non-multicat source decoded as one value).
        (MulticatToken::Legacy(v), _) => values_equal(v, rhs),
        _ => false,
    }
}

/// Scalar equality used by the membership fallback paths.
fn values_equal(lhs: &Value, rhs: &Value) -> bool {
    compare_values(lhs, &CompareOp::Eq, rhs)
}

fn resolve_row_rhs(
    rhs: &ResolvedRowFilterRhs,
    current_node: u64,
    _current_table: &Arc<TableReader>,
    entity_cols: &[Option<Value>],
) -> Option<Value> {
    match rhs {
        ResolvedRowFilterRhs::Constant(v) => Some(v.clone()),
        ResolvedRowFilterRhs::EntityColumn(idx) => {
            entity_cols.get(*idx).and_then(|v| v.clone())
        }
        ResolvedRowFilterRhs::Remote { path, col_idx } => {
            resolve_remote_value(path, current_node, *col_idx)
        }
    }
}

/// Follow a sub-traversal path from a node, taking the first destination
/// at each hop (FK direction = many-to-one = exactly one destination).
fn resolve_remote_value(
    path: &[ResolvedEdge],
    start_node: u64,
    col_idx: usize,
) -> Option<Value> {
    let mut current = start_node;
    for edge in path {
        // Iterate edges and take first destination (FK = 1:1)
        let view = edge.src_table.edges_for_node(current).ok()?;
        let segments = view.relation_segments().ok()?;
        let segment = segments.get(edge.relation_idx)?;

        if segment.count == 0 {
            return None;
        }

        // Decode first edge entry
        let entry_bytes = segment.layout.entry_bytes;
        let chunk = &segment.data[..entry_bytes];
        let mut buf = [0u8; 16];
        buf[..entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        current = (val & segment.layout.dst_mask as u128) as u64;
    }

    // Read column from final destination, preserving NULL as None.
    let dst_table = &path.last()?.dst_table;
    read_column_value(dst_table, current, col_idx)
}

// ─── Stage filter evaluation (per entity-timestamp, between stages) ──────────

/// Evaluate a stage filter using computed aliases.
pub fn eval_stage_filter(
    filter: &ResolvedStageFilter,
    aliases: &[Option<Value>],
    entity_cols: &[Option<Value>],
    entity_table: &Arc<TableReader>,
    entity_node: u64,
) -> bool {
    match filter {
        ResolvedStageFilter::Compare { alias_idx, op, rhs } => {
            let l_opt = &aliases[*alias_idx];

            match op {
                CompareOp::IsNull => return is_nullish_filter_value(l_opt),
                CompareOp::IsNotNull => return !is_nullish_filter_value(l_opt),
                _ => {}
            }

            let l = match l_opt {
                Some(v) => v,
                None => return false, // alias not computed = filter fails (unless IsNull)
            };
            let r = match resolve_stage_rhs(rhs, aliases, entity_cols, entity_table, entity_node) {
                Some(v) => v,
                None => return false, // SQL tri-state: op with NULL evaluates to NULL (fails filter)
            };
            compare_values(l, op, &r)
        }
        ResolvedStageFilter::NonEmptyList { alias_idx } => {
            matches!(&aliases[*alias_idx], Some(Value::List(values)) if !values.is_empty())
        }
        ResolvedStageFilter::And(filters) => filters
            .iter()
            .all(|f| eval_stage_filter(f, aliases, entity_cols, entity_table, entity_node)),
        ResolvedStageFilter::Or(filters) => filters
            .iter()
            .any(|f| eval_stage_filter(f, aliases, entity_cols, entity_table, entity_node)),
        ResolvedStageFilter::Not(f) => {
            !eval_stage_filter(f, aliases, entity_cols, entity_table, entity_node)
        }
    }
}

fn resolve_stage_rhs(
    rhs: &ResolvedStageFilterRhs,
    aliases: &[Option<Value>],
    entity_cols: &[Option<Value>],
    _entity_table: &Arc<TableReader>,
    entity_node: u64,
) -> Option<Value> {
    match rhs {
        ResolvedStageFilterRhs::Constant(v) => Some(v.clone()),
        ResolvedStageFilterRhs::EntityColumn(idx) => {
            entity_cols.get(*idx).and_then(|v| v.clone())
        }
        ResolvedStageFilterRhs::Remote { path, col_idx } => {
            resolve_remote_value(path, entity_node, *col_idx)
        }
        ResolvedStageFilterRhs::Alias(idx) => {
            aliases.get(*idx).and_then(|v| v.clone())
        }
    }
}

// ─── Value comparison ────────────────────────────────────────────────────────

/// Compare two values with a comparison operator.
pub fn compare_values(lhs: &Value, op: &CompareOp, rhs: &Value) -> bool {
    if let CompareOp::In = op {
        return match (lhs, rhs) {
            (Value::List(_), _) => false,
            // Match scalar equality semantics, including Int/Float coercion.
            // Multicategorical left-hand sides use the separate CONTAINS op.
            (_, Value::List(values)) => values.iter().any(|value| values_equal(lhs, value)),
            _ => false,
        };
    }

    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => compare_ord(a, b, op),
        (Value::Float(a), Value::Float(b)) => compare_f64(*a, *b, op),
        (Value::Int(a), Value::Float(b)) => compare_f64(*a as f64, *b, op),
        (Value::Float(a), Value::Int(b)) => compare_f64(*a, *b as f64, op),
        (Value::Str(a), Value::Str(b)) => compare_str(a, b, op),
        (Value::Bool(a), Value::Bool(b)) => compare_ord(a, b, op),
        (Value::List(v), b) => compare_list(v, b, op),
        (Value::Int(a), Value::List(v)) => compare_ord(a, &(v.len() as i64), op),
        (Value::Float(a), Value::List(v)) => compare_f64(*a, v.len() as f64, op),
        // Type mismatch — always false
        _ => false,
    }
}

fn compare_list(values: &[Value], rhs: &Value, op: &CompareOp) -> bool {
    match op {
        CompareOp::Contains => values.iter().any(|value| value == rhs),
        CompareOp::NotContains => values.iter().all(|value| value != rhs),
        _ => match rhs {
            Value::Int(b) => compare_ord(&(values.len() as i64), b, op),
            Value::Float(b) => compare_f64(values.len() as f64, *b, op),
            Value::List(b) => compare_ord(&values.len(), &b.len(), op),
            _ => false,
        },
    }
}

fn compare_str(a: &str, b: &str, op: &CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Neq => a != b,
        CompareOp::Gt => a > b,
        CompareOp::Gte => a >= b,
        CompareOp::Lt => a < b,
        CompareOp::Lte => a <= b,
        CompareOp::StartsWith => a.starts_with(b),
        CompareOp::EndsWith => a.ends_with(b),
        CompareOp::Contains => a.contains(b),
        CompareOp::NotContains => !a.contains(b),
        CompareOp::In => false,
        CompareOp::IsNull | CompareOp::IsNotNull => unreachable!("null checks handled in eval_filter"),
    }
}

fn compare_ord<T: Ord>(a: &T, b: &T, op: &CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Neq => a != b,
        CompareOp::Gt => a > b,
        CompareOp::Gte => a >= b,
        CompareOp::Lt => a < b,
        CompareOp::Lte => a <= b,
        CompareOp::StartsWith
        | CompareOp::EndsWith
        | CompareOp::Contains
        | CompareOp::NotContains
        | CompareOp::In => false,
        CompareOp::IsNull | CompareOp::IsNotNull => unreachable!("null checks handled in eval_filter"),
    }
}

fn compare_f64(a: f64, b: f64, op: &CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Neq => a != b,
        CompareOp::Gt => a > b,
        CompareOp::Gte => a >= b,
        CompareOp::Lt => a < b,
        CompareOp::Lte => a <= b,
        CompareOp::StartsWith
        | CompareOp::EndsWith
        | CompareOp::Contains
        | CompareOp::NotContains
        | CompareOp::In => false,
        CompareOp::IsNull | CompareOp::IsNotNull => unreachable!("null checks handled in eval_filter"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_int() {
        assert!(compare_values(&Value::Int(5), &CompareOp::Gt, &Value::Int(3)));
        assert!(compare_values(&Value::Int(3), &CompareOp::Lt, &Value::Int(5)));
        assert!(compare_values(&Value::Int(5), &CompareOp::Eq, &Value::Int(5)));
        assert!(compare_values(&Value::Int(5), &CompareOp::Neq, &Value::Int(3)));
        assert!(compare_values(&Value::Int(5), &CompareOp::Gte, &Value::Int(5)));
        assert!(compare_values(&Value::Int(5), &CompareOp::Lte, &Value::Int(5)));
    }

    #[test]
    fn test_compare_float() {
        assert!(compare_values(
            &Value::Float(std::f64::consts::PI),
            &CompareOp::Gt,
            &Value::Float(std::f64::consts::E)
        ));
        assert!(compare_values(
            &Value::Float(std::f64::consts::E),
            &CompareOp::Lt,
            &Value::Float(std::f64::consts::PI)
        ));
    }

    #[test]
    fn test_compare_mixed_numeric() {
        assert!(compare_values(&Value::Int(5), &CompareOp::Eq, &Value::Float(5.0)));
        assert!(compare_values(&Value::Float(5.0), &CompareOp::Eq, &Value::Int(5)));
        assert!(compare_values(&Value::Int(5), &CompareOp::Lt, &Value::Float(5.5)));
    }

    #[test]
    fn test_compare_string() {
        assert!(compare_values(
            &Value::Str("apple".into()),
            &CompareOp::Lt,
            &Value::Str("banana".into())
        ));
    }

    #[test]
    fn test_compare_type_mismatch() {
        // Type mismatch returns false
        assert!(!compare_values(&Value::Int(5), &CompareOp::Eq, &Value::Str("5".into())));
    }

    // ─── Bool comparisons ────────────────────────────────────────────────

    #[test]
    fn test_compare_bool_eq() {
        assert!(compare_values(&Value::Bool(true), &CompareOp::Eq, &Value::Bool(true)));
        assert!(compare_values(&Value::Bool(false), &CompareOp::Eq, &Value::Bool(false)));
        assert!(!compare_values(&Value::Bool(true), &CompareOp::Eq, &Value::Bool(false)));
    }

    #[test]
    fn test_compare_bool_neq() {
        assert!(compare_values(&Value::Bool(true), &CompareOp::Neq, &Value::Bool(false)));
        assert!(!compare_values(&Value::Bool(true), &CompareOp::Neq, &Value::Bool(true)));
    }

    #[test]
    fn test_compare_bool_ordering() {
        // false < true in Rust's Ord
        assert!(compare_values(&Value::Bool(false), &CompareOp::Lt, &Value::Bool(true)));
        assert!(compare_values(&Value::Bool(true), &CompareOp::Gt, &Value::Bool(false)));
        assert!(compare_values(&Value::Bool(true), &CompareOp::Gte, &Value::Bool(true)));
        assert!(compare_values(&Value::Bool(false), &CompareOp::Lte, &Value::Bool(false)));
    }

    #[test]
    fn test_compare_bool_int_mismatch() {
        // Bool vs Int is type mismatch → false
        assert!(!compare_values(&Value::Bool(true), &CompareOp::Eq, &Value::Int(1)));
        assert!(!compare_values(&Value::Bool(false), &CompareOp::Eq, &Value::Int(0)));
    }

    // ─── String ordering ─────────────────────────────────────────────────

    #[test]
    fn test_compare_string_ordering() {
        assert!(compare_values(&Value::Str("a".into()), &CompareOp::Lt, &Value::Str("b".into())));
        assert!(compare_values(&Value::Str("z".into()), &CompareOp::Gt, &Value::Str("a".into())));
        assert!(compare_values(&Value::Str("abc".into()), &CompareOp::Lt, &Value::Str("abd".into())));
    }

    #[test]
    fn test_compare_string_eq() {
        assert!(compare_values(&Value::Str("hello".into()), &CompareOp::Eq, &Value::Str("hello".into())));
        assert!(!compare_values(&Value::Str("hello".into()), &CompareOp::Eq, &Value::Str("world".into())));
    }

    #[test]
    fn test_compare_string_empty() {
        assert!(compare_values(&Value::Str("".into()), &CompareOp::Lt, &Value::Str("a".into())));
        assert!(compare_values(&Value::Str("".into()), &CompareOp::Eq, &Value::Str("".into())));
    }

    // ─── Float edge cases ────────────────────────────────────────────────

    #[test]
    fn test_compare_float_nan() {
        // NaN comparisons: NaN != NaN, NaN is not >/</>=/<= anything
        let nan = Value::Float(f64::NAN);
        let five = Value::Float(5.0);
        assert!(!compare_values(&nan, &CompareOp::Eq, &nan));
        assert!(compare_values(&nan, &CompareOp::Neq, &nan)); // NaN != NaN is true in IEEE754
        assert!(!compare_values(&nan, &CompareOp::Gt, &five));
        assert!(!compare_values(&nan, &CompareOp::Lt, &five));
        assert!(!compare_values(&five, &CompareOp::Gt, &nan));
    }

    #[test]
    fn test_nan_is_nullish_for_null_predicates() {
        assert!(is_nullish_filter_value(&None));
        assert!(is_nullish_filter_value(&Some(Value::Float(f64::NAN))));
        assert!(!is_nullish_filter_value(&Some(Value::Float(f64::INFINITY))));
        assert!(!is_nullish_filter_value(&Some(Value::Float(0.0))));
    }

    #[test]
    fn test_compare_float_infinity() {
        let inf = Value::Float(f64::INFINITY);
        let neg_inf = Value::Float(f64::NEG_INFINITY);
        let zero = Value::Float(0.0);
        assert!(compare_values(&inf, &CompareOp::Gt, &zero));
        assert!(compare_values(&neg_inf, &CompareOp::Lt, &zero));
        assert!(compare_values(&inf, &CompareOp::Gt, &neg_inf));
    }

    #[test]
    fn test_compare_float_negative_zero() {
        // -0.0 == 0.0 in IEEE754
        assert!(compare_values(&Value::Float(-0.0), &CompareOp::Eq, &Value::Float(0.0)));
    }

    // ─── List comparisons ────────────────────────────────────────────────

    #[test]
    fn test_compare_list_vs_int() {
        // List vs Int compares list length
        let list = Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert!(compare_values(&list, &CompareOp::Eq, &Value::Int(3)));
        assert!(compare_values(&list, &CompareOp::Gt, &Value::Int(2)));
        assert!(compare_values(&list, &CompareOp::Lt, &Value::Int(4)));
    }

    #[test]
    fn test_compare_list_vs_float() {
        let list = Value::List(vec![Value::Int(1), Value::Int(2)]);
        assert!(compare_values(&list, &CompareOp::Eq, &Value::Float(2.0)));
        assert!(compare_values(&list, &CompareOp::Lt, &Value::Float(2.5)));
    }

    #[test]
    fn test_compare_list_vs_list() {
        let l1 = Value::List(vec![Value::Int(1), Value::Int(2)]);
        let l2 = Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert!(compare_values(&l1, &CompareOp::Lt, &l2)); // 2 < 3
    }

    #[test]
    fn test_compare_list_contains_membership() {
        let list = Value::List(vec![
            Value::Int(1),
            Value::Str("open".into()),
            Value::Bool(true),
        ]);
        assert!(compare_values(
            &list,
            &CompareOp::Contains,
            &Value::Str("open".into())
        ));
        assert!(compare_values(&list, &CompareOp::Contains, &Value::Int(1)));
        assert!(compare_values(
            &list,
            &CompareOp::NotContains,
            &Value::Str("closed".into())
        ));
        assert!(!compare_values(
            &list,
            &CompareOp::NotContains,
            &Value::Bool(true)
        ));
    }

    #[test]
    fn test_compare_scalar_in_list() {
        let ints = Value::List(vec![Value::Int(1), Value::Int(2)]);
        assert!(compare_values(&Value::Int(2), &CompareOp::In, &ints));
        assert!(compare_values(&Value::Float(2.0), &CompareOp::In, &ints));
        assert!(!compare_values(&Value::Int(3), &CompareOp::In, &ints));

        let strings = Value::List(vec![Value::Str("open".into()), Value::Str("closed".into())]);
        assert!(compare_values(
            &Value::Str("open".into()),
            &CompareOp::In,
            &strings
        ));
        assert!(!compare_values(
            &Value::Str("pending".into()),
            &CompareOp::In,
            &strings
        ));

        assert!(!compare_values(
            &Value::Int(1),
            &CompareOp::In,
            &Value::List(vec![])
        ));
        assert!(!compare_values(&ints, &CompareOp::In, &ints));
        assert!(!compare_values(
            &Value::Int(1),
            &CompareOp::In,
            &Value::Int(1)
        ));
    }

    #[test]
    fn test_compare_empty_list() {
        let empty = Value::List(vec![]);
        assert!(compare_values(&empty, &CompareOp::Eq, &Value::Int(0)));
        assert!(compare_values(&empty, &CompareOp::Lt, &Value::Int(1)));
    }

    // ─── Cross-type mismatches ───────────────────────────────────────────

    #[test]
    fn test_compare_int_vs_str_mismatch() {
        assert!(!compare_values(&Value::Int(5), &CompareOp::Eq, &Value::Str("5".into())));
        assert!(!compare_values(&Value::Int(5), &CompareOp::Gt, &Value::Str("3".into())));
    }

    #[test]
    fn test_compare_bool_vs_str_mismatch() {
        assert!(!compare_values(&Value::Bool(true), &CompareOp::Eq, &Value::Str("true".into())));
    }

    #[test]
    fn test_compare_str_vs_float_mismatch() {
        assert!(!compare_values(&Value::Str("5.0".into()), &CompareOp::Eq, &Value::Float(5.0)));
    }

    #[test]
    fn test_compare_string_predicates() {
        let value = Value::Str("investment_account".into());
        assert!(compare_values(
            &value,
            &CompareOp::StartsWith,
            &Value::Str("investment".into())
        ));
        assert!(!compare_values(
            &value,
            &CompareOp::StartsWith,
            &Value::Str("checking".into())
        ));
        assert!(compare_values(
            &value,
            &CompareOp::EndsWith,
            &Value::Str("account".into())
        ));
        assert!(compare_values(
            &value,
            &CompareOp::Contains,
            &Value::Str("ment_acc".into())
        ));
        assert!(compare_values(
            &value,
            &CompareOp::NotContains,
            &Value::Str("credit".into())
        ));
        assert!(!compare_values(
            &value,
            &CompareOp::NotContains,
            &Value::Str("account".into())
        ));
    }

    #[test]
    fn test_string_predicates_require_strings() {
        assert!(!compare_values(
            &Value::Int(5),
            &CompareOp::StartsWith,
            &Value::Str("5".into())
        ));
        assert!(!compare_values(
            &Value::Str("5".into()),
            &CompareOp::Contains,
            &Value::Int(5)
        ));
    }

    // ─── All CompareOp variants ──────────────────────────────────────────

    #[test]
    fn test_all_ops_int() {
        let a = Value::Int(5);
        let b = Value::Int(10);
        assert!(!compare_values(&a, &CompareOp::Eq, &b));
        assert!(compare_values(&a, &CompareOp::Neq, &b));
        assert!(compare_values(&a, &CompareOp::Lt, &b));
        assert!(compare_values(&a, &CompareOp::Lte, &b));
        assert!(!compare_values(&a, &CompareOp::Gt, &b));
        assert!(!compare_values(&a, &CompareOp::Gte, &b));
    }

    #[test]
    fn test_gte_lte_boundary() {
        let five = Value::Int(5);
        assert!(compare_values(&five, &CompareOp::Gte, &Value::Int(5)));
        assert!(compare_values(&five, &CompareOp::Lte, &Value::Int(5)));
        assert!(compare_values(&five, &CompareOp::Gte, &Value::Int(4)));
        assert!(compare_values(&five, &CompareOp::Lte, &Value::Int(6)));
    }
}
