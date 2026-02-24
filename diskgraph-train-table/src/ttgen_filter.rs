//! Filter evaluation for row-level and stage-level filters.

use diskgraph_core::tt_builder::{CompareOp, Value};
use std::sync::Arc;

use diskgraph_index::graph::TableReader;

use crate::ttgen_resolve::{
    ResolvedRowFilter, ResolvedRowFilterLhs, ResolvedRowFilterRhs,
    ResolvedStageFilter, ResolvedStageFilterRhs, ResolvedEdge,
};
use crate::ttgen_traversal::read_column_value;

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
            let l = resolve_row_lhs(lhs, current_node, current_table);
            let r = resolve_row_rhs(rhs, current_node, current_table, entity_cols);
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
) -> Value {
    match lhs {
        ResolvedRowFilterLhs::Column(col_idx) => {
            read_column_value(current_table, current_node, *col_idx)
                .unwrap_or(Value::Int(0))
        }
        ResolvedRowFilterLhs::Remote { path, col_idx } => {
            resolve_remote_value(path, current_node, *col_idx)
                .unwrap_or(Value::Int(0))
        }
    }
}

fn resolve_row_rhs(
    rhs: &ResolvedRowFilterRhs,
    current_node: u64,
    _current_table: &Arc<TableReader>,
    entity_cols: &[Option<Value>],
) -> Value {
    match rhs {
        ResolvedRowFilterRhs::Constant(v) => v.clone(),
        ResolvedRowFilterRhs::EntityColumn(idx) => {
            entity_cols
                .get(*idx)
                .and_then(|v| v.clone())
                .unwrap_or(Value::Int(0))
        }
        ResolvedRowFilterRhs::Remote { path, col_idx } => {
            resolve_remote_value(path, current_node, *col_idx)
                .unwrap_or(Value::Int(0))
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

    // Read column from final destination
    let dst_table = &path.last()?.dst_table;
    Some(read_column_value(dst_table, current, col_idx).unwrap_or(Value::Int(0)))
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
            let l = match &aliases[*alias_idx] {
                Some(v) => v,
                None => return false, // alias not computed = filter fails
            };
            let r = resolve_stage_rhs(rhs, aliases, entity_cols, entity_table, entity_node);
            compare_values(l, op, &r)
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
) -> Value {
    match rhs {
        ResolvedStageFilterRhs::Constant(v) => v.clone(),
        ResolvedStageFilterRhs::EntityColumn(idx) => {
            entity_cols
                .get(*idx)
                .and_then(|v| v.clone())
                .unwrap_or(Value::Int(0))
        }
        ResolvedStageFilterRhs::Remote { path, col_idx } => {
            resolve_remote_value(path, entity_node, *col_idx)
                .unwrap_or(Value::Int(0))
        }
        ResolvedStageFilterRhs::Alias(idx) => {
            aliases
                .get(*idx)
                .and_then(|v| v.clone())
                .unwrap_or(Value::Int(0))
        }
    }
}

// ─── Value comparison ────────────────────────────────────────────────────────

/// Compare two values with a comparison operator.
pub fn compare_values(lhs: &Value, op: &CompareOp, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => compare_ord(a, b, op),
        (Value::Float(a), Value::Float(b)) => compare_f64(*a, *b, op),
        (Value::Int(a), Value::Float(b)) => compare_f64(*a as f64, *b, op),
        (Value::Float(a), Value::Int(b)) => compare_f64(*a, *b as f64, op),
        (Value::Str(a), Value::Str(b)) => compare_ord(a, b, op),
        (Value::Bool(a), Value::Bool(b)) => compare_ord(a, b, op),
        // List vs numeric: compare list length
        (Value::List(v), Value::Int(b)) => compare_ord(&(v.len() as i64), b, op),
        (Value::Int(a), Value::List(v)) => compare_ord(a, &(v.len() as i64), op),
        (Value::List(v), Value::Float(b)) => compare_f64(v.len() as f64, *b, op),
        (Value::Float(a), Value::List(v)) => compare_f64(*a, v.len() as f64, op),
        (Value::List(a), Value::List(b)) => compare_ord(&a.len(), &b.len(), op),
        // Type mismatch — always false
        _ => false,
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
        assert!(compare_values(&Value::Float(3.14), &CompareOp::Gt, &Value::Float(2.71)));
        assert!(compare_values(&Value::Float(2.71), &CompareOp::Lt, &Value::Float(3.14)));
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
