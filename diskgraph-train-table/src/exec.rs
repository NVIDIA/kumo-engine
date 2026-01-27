use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use diskgraph_rowpack::{IngestedType, RowPackDecodeScratch, RowPackDecoder, ValueRef};

use crate::ast::{AggFunc, AggTarget, AggregationExpr, PredictExpr, Query};
use crate::index_reader::IndexReader;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i64),
    UInt(u64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetValue {
    Number(f64),
    List(Vec<Value>),
    Count(u64),
    Scalar(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRow {
    pub entity: Value,
    pub target: TargetValue,
    pub timestamp: i64,
}

pub fn execute_simple(query: &Query, reader: &IndexReader) -> Result<Vec<TrainRow>> {
    let entity_table = &query.for_each.table;
    let agg = match &query.predict {
        PredictExpr::Aggregation(agg) => agg,
        PredictExpr::Column(col) => {
            return execute_simple_column(reader, entity_table, col, query);
        }
        PredictExpr::Comparison(pred) => {
            return execute_simple_comparison(reader, entity_table, pred, query);
        }
    };
    let target_table = match &agg.target {
        AggTarget::Column(col) => &col.table,
        AggTarget::Star(table) => &table.table,
    };

    if entity_table != target_table {
        return Err(anyhow!("cross-table aggregation not supported in simple executor"));
    }

    let ctx = reader
        .node_context(entity_table)
        .ok_or_else(|| anyhow!("missing rowpack context for table {entity_table}"))?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let col_index = column_index_map(ctx);

    let entity_col = &query.for_each.column;
    let entity_idx = *col_index
        .get(entity_col)
        .ok_or_else(|| anyhow!("missing entity column {entity_col}"))?;
    let timestamp_idx = find_timestamp_index(ctx, &col_index);

    let target_idx = match &agg.target {
        AggTarget::Column(col) => Some(*col_index
            .get(&col.column)
            .ok_or_else(|| anyhow!("missing target column {}", col.column))?),
        AggTarget::Star(_) => None,
    };

    let mut scratch = RowPackDecodeScratch::default();
    let mut iter = reader.iter_node_rows(entity_table, 0)?;

    let allowed_entities = if let Some(assuming) = &query.assuming {
        Some(evaluate_assuming(
            assuming,
            agg,
            reader,
            entity_table,
            entity_idx,
            &decoder,
            &col_index,
        )?)
    } else {
        None
    };

    match agg.func {
        AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
            let target_idx = target_idx.ok_or_else(|| anyhow!("numeric agg requires column"))?;
            let mut sums: HashMap<Value, (f64, u64, f64, f64)> = HashMap::new();
            let mut timestamps: HashMap<Value, i64> = HashMap::new();
            while let Some(row) = iter.next_row()? {
                if let Some(filter) = &agg.filter {
                    if !eval_predicate_row(filter, &decoder, &row, &col_index, &mut scratch)? {
                        continue;
                    }
                }
                let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
                    continue;
                };
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
                }
                let Some(value) = decode_numeric_opt(&decoder, &row, target_idx, &mut scratch)? else {
                    continue;
                };
                let entry = sums.entry(entity).or_insert((0.0, 0, value, value));
                entry.0 += value;
                entry.1 += 1;
                if value < entry.2 {
                    entry.2 = value;
                }
                if value > entry.3 {
                    entry.3 = value;
                }
                if let Some(ts_idx) = timestamp_idx {
                    if let Some(ts) = decode_timestamp(&decoder, &row, ts_idx, &mut scratch)? {
                        update_max_timestamp(&mut timestamps, &entity_key, ts);
                    }
                }
            }
            Ok(sums
                .into_iter()
                .map(|(entity, (sum, count, min, max))| {
                    let value = match agg.func {
                        AggFunc::Sum => sum,
                        AggFunc::Avg => sum / count as f64,
                        AggFunc::Min => min,
                        AggFunc::Max => max,
                        _ => sum,
                    };
                    TrainRow {
                        timestamp: timestamps.get(&entity).copied().unwrap_or(0),
                        entity,
                        target: TargetValue::Number(value),
                    }
                })
                .collect())
        }
        AggFunc::Count => {
            let mut counts: HashMap<Value, u64> = HashMap::new();
            let mut timestamps: HashMap<Value, i64> = HashMap::new();
            while let Some(row) = iter.next_row()? {
                if let Some(filter) = &agg.filter {
                    if !eval_predicate_row(filter, &decoder, &row, &col_index, &mut scratch)? {
                        continue;
                    }
                }
                let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
                    continue;
                };
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
                }
                if let Some(target_idx) = target_idx {
                    let Some(_value) = decode_value_opt(&decoder, &row, target_idx, &mut scratch)? else {
                        continue;
                    };
                }
                let entry = counts.entry(entity).or_insert(0);
                *entry += 1;
                if let Some(ts_idx) = timestamp_idx {
                    if let Some(ts) = decode_timestamp(&decoder, &row, ts_idx, &mut scratch)? {
                        update_max_timestamp(&mut timestamps, &entity_key, ts);
                    }
                }
            }
            Ok(counts
                .into_iter()
                .map(|(entity, count)| TrainRow {
                    timestamp: timestamps.get(&entity).copied().unwrap_or(0),
                    entity,
                    target: TargetValue::Count(count),
                })
                .collect())
        }
        AggFunc::ListDistinct => {
            let target_idx = target_idx.ok_or_else(|| anyhow!("LIST_DISTINCT requires column"))?;
            let mut values: HashMap<Value, HashSet<Value>> = HashMap::new();
            let mut timestamps: HashMap<Value, i64> = HashMap::new();
            while let Some(row) = iter.next_row()? {
                if let Some(filter) = &agg.filter {
                    if !eval_predicate_row(filter, &decoder, &row, &col_index, &mut scratch)? {
                        continue;
                    }
                }
                let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
                    continue;
                };
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
                }
                let Some(target) = decode_value_opt(&decoder, &row, target_idx, &mut scratch)? else {
                    continue;
                };
                values
                    .entry(entity)
                    .or_insert_with(HashSet::new)
                    .insert(target);
                if let Some(ts_idx) = timestamp_idx {
                    if let Some(ts) = decode_timestamp(&decoder, &row, ts_idx, &mut scratch)? {
                        update_max_timestamp(&mut timestamps, &entity_key, ts);
                    }
                }
            }
            Ok(values
                .into_iter()
                .map(|(entity, set)| TrainRow {
                    timestamp: timestamps.get(&entity).copied().unwrap_or(0),
                    entity,
                    target: TargetValue::List(set.into_iter().collect()),
                })
                .collect())
        }
    }
}

fn execute_simple_comparison(
    reader: &IndexReader,
    entity_table: &str,
    pred: &crate::ast::PredicateExpr,
    query: &Query,
) -> Result<Vec<TrainRow>> {
    if comparison_has_agg(pred) {
        return execute_simple_comparison_agg(reader, entity_table, pred, query);
    }
    execute_simple_predicate(reader, entity_table, pred, query)
}

fn execute_simple_comparison_agg(
    reader: &IndexReader,
    entity_table: &str,
    pred: &crate::ast::PredicateExpr,
    query: &Query,
) -> Result<Vec<TrainRow>> {
    let (agg, op, literal) = extract_comparison_agg(pred)?;

    let target_table = match &agg.target {
        AggTarget::Column(col) => &col.table,
        AggTarget::Star(tbl) => &tbl.table,
    };
    if entity_table != target_table {
        return Err(anyhow!("cross-table aggregation not supported in simple executor"));
    }

    let ctx = reader
        .node_context(entity_table)
        .ok_or_else(|| anyhow!("missing rowpack context for table {entity_table}"))?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let col_index = column_index_map(ctx);

    let entity_col = &query.for_each.column;
    let entity_idx = *col_index
        .get(entity_col)
        .ok_or_else(|| anyhow!("missing entity column {entity_col}"))?;
    let timestamp_idx = find_timestamp_index(ctx, &col_index);
    let target_idx = match &agg.target {
        AggTarget::Column(col) => Some(*col_index
            .get(&col.column)
            .ok_or_else(|| anyhow!("missing target column {}", col.column))?),
        AggTarget::Star(_) => None,
    };

    let mut scratch = RowPackDecodeScratch::default();
    let mut iter = reader.iter_node_rows(entity_table, 0)?;

    let allowed_entities = if let Some(assuming) = &query.assuming {
        Some(evaluate_assuming(
            assuming,
            &agg,
            reader,
            entity_table,
            entity_idx,
            &decoder,
            &col_index,
        )?)
    } else {
        None
    };

    let mut timestamps: HashMap<Value, i64> = HashMap::new();
    let mut sum_map: HashMap<Value, (f64, u64, f64, f64)> = HashMap::new();
    let mut count_map: HashMap<Value, u64> = HashMap::new();

    while let Some(row) = iter.next_row()? {
        if let Some(filter) = &agg.filter {
            if !eval_predicate_row(filter, &decoder, &row, &col_index, &mut scratch)? {
                continue;
            }
        }
        let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
            continue;
        };
        let entity_key = entity.clone();
        if let Some(allowed) = &allowed_entities {
            if !allowed.contains(&entity) {
                continue;
            }
        }

        match agg.func {
            AggFunc::Count => {
                if let Some(target_idx) = target_idx {
                    let Some(_value) = decode_value_opt(&decoder, &row, target_idx, &mut scratch)? else {
                        continue;
                    };
                }
                let entry = count_map.entry(entity).or_insert(0);
                *entry += 1;
            }
            AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
                let target_idx = target_idx.ok_or_else(|| anyhow!("numeric agg requires column"))?;
                let Some(value) = decode_numeric_opt(&decoder, &row, target_idx, &mut scratch)? else {
                    continue;
                };
                let entry = sum_map.entry(entity).or_insert((0.0, 0, value, value));
                entry.0 += value;
                entry.1 += 1;
                if value < entry.2 {
                    entry.2 = value;
                }
                if value > entry.3 {
                    entry.3 = value;
                }
            }
            AggFunc::ListDistinct => return Err(anyhow!("LIST_DISTINCT not supported in comparison")),
        }

        if let Some(ts_idx) = timestamp_idx {
            if let Some(ts) = decode_timestamp(&decoder, &row, ts_idx, &mut scratch)? {
                update_max_timestamp(&mut timestamps, &entity_key, ts);
            }
        }
    }

    let mut rows = Vec::new();
    match agg.func {
        AggFunc::Count => {
            for (entity, count) in count_map {
                let value = count as f64;
                let result = compare_numeric(value, literal, op);
                let ts = timestamps.get(&entity).copied().unwrap_or(0);
                rows.push(TrainRow {
                    entity,
                    target: TargetValue::Scalar(Value::Bool(result)),
                    timestamp: ts,
                });
            }
        }
        AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
            for (entity, (sum, count, min, max)) in sum_map {
                let value = match agg.func {
                    AggFunc::Sum => sum,
                    AggFunc::Avg => sum / count as f64,
                    AggFunc::Min => min,
                    AggFunc::Max => max,
                    _ => sum,
                };
                let result = compare_numeric(value, literal, op);
                let ts = timestamps.get(&entity).copied().unwrap_or(0);
                rows.push(TrainRow {
                    entity,
                    target: TargetValue::Scalar(Value::Bool(result)),
                    timestamp: ts,
                });
            }
        }
        AggFunc::ListDistinct => {}
    }

    Ok(rows)
}

fn execute_simple_predicate(
    reader: &IndexReader,
    entity_table: &str,
    pred: &crate::ast::PredicateExpr,
    query: &Query,
) -> Result<Vec<TrainRow>> {
    let ctx = reader
        .node_context(entity_table)
        .ok_or_else(|| anyhow!("missing rowpack context for table {entity_table}"))?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let col_index = column_index_map(ctx);

    let entity_col = &query.for_each.column;
    let entity_idx = *col_index
        .get(entity_col)
        .ok_or_else(|| anyhow!("missing entity column {entity_col}"))?;
    let timestamp_idx = find_timestamp_index(ctx, &col_index);
    let pred_cols = collect_predicate_columns(pred, &col_index)?;

    let mut scratch = RowPackDecodeScratch::default();
    let mut iter = reader.iter_node_rows(entity_table, 0)?;
    let mut rows = Vec::new();
    while let Some(row) = iter.next_row()? {
        let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
            continue;
        };
        if predicate_has_null(&decoder, &row, &pred_cols, &mut scratch)? {
            continue;
        }
        let result = eval_predicate_row(pred, &decoder, &row, &col_index, &mut scratch)?;
        let ts = if let Some(ts_idx) = timestamp_idx {
            decode_timestamp(&decoder, &row, ts_idx, &mut scratch)?.unwrap_or(0)
        } else {
            0
        };
        rows.push(TrainRow {
            entity,
            target: TargetValue::Scalar(Value::Bool(result)),
            timestamp: ts,
        });
    }
    Ok(rows)
}

fn collect_predicate_columns(
    pred: &crate::ast::PredicateExpr,
    col_index: &HashMap<String, usize>,
) -> Result<Vec<usize>> {
    let mut cols = Vec::new();
    collect_predicate_columns_inner(pred, &mut cols, col_index)?;
    Ok(cols)
}

fn collect_predicate_columns_inner(
    pred: &crate::ast::PredicateExpr,
    out: &mut Vec<usize>,
    col_index: &HashMap<String, usize>,
) -> Result<()> {
    match pred {
        crate::ast::PredicateExpr::Comparison { left, right, .. } => {
            collect_predicate_columns_from_expr(left, out, col_index)?;
            collect_predicate_columns_from_expr(right, out, col_index)?;
        }
        crate::ast::PredicateExpr::And(items) | crate::ast::PredicateExpr::Or(items) => {
            for item in items {
                collect_predicate_columns_inner(item, out, col_index)?;
            }
        }
    }
    Ok(())
}

fn collect_predicate_columns_from_expr(
    expr: &crate::ast::Expr,
    out: &mut Vec<usize>,
    col_index: &HashMap<String, usize>,
) -> Result<()> {
    match expr {
        crate::ast::Expr::Column(col) => {
            let idx = *col_index
                .get(&col.column)
                .ok_or_else(|| anyhow!("missing column {}", col.column))?;
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
        crate::ast::Expr::Agg(_) | crate::ast::Expr::Number(_) | crate::ast::Expr::String(_) => {}
    }
    Ok(())
}

fn predicate_has_null(
    decoder: &RowPackDecoder,
    row: &[u8],
    cols: &[usize],
    scratch: &mut RowPackDecodeScratch,
) -> Result<bool> {
    for idx in cols {
        let value = decoder.decode_column_value(row, *idx, scratch)?;
        if matches!(value, ValueRef::Null) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn extract_comparison_agg(
    pred: &crate::ast::PredicateExpr,
) -> Result<(AggregationExpr, crate::ast::CompareOp, f64)> {
    let crate::ast::PredicateExpr::Comparison { left, op, right } = pred else {
        return Err(anyhow!("predict comparison must be a single comparison"));
    };
    match (left.as_ref(), right.as_ref()) {
        (crate::ast::Expr::Agg(agg), crate::ast::Expr::Number(n)) => {
            Ok(((**agg).clone(), *op, *n))
        }
        (crate::ast::Expr::Number(n), crate::ast::Expr::Agg(agg)) => {
            Ok(((**agg).clone(), *op, *n))
        }
        _ => Err(anyhow!("predict comparison must compare aggregation to a number")),
    }
}

fn comparison_has_agg(pred: &crate::ast::PredicateExpr) -> bool {
    match pred {
        crate::ast::PredicateExpr::Comparison { left, right, .. } => {
            expr_has_agg(left) || expr_has_agg(right)
        }
        crate::ast::PredicateExpr::And(items) | crate::ast::PredicateExpr::Or(items) => {
            items.iter().any(comparison_has_agg)
        }
    }
}

fn expr_has_agg(expr: &crate::ast::Expr) -> bool {
    matches!(expr, crate::ast::Expr::Agg(_))
}

fn execute_simple_column(reader: &IndexReader, entity_table: &str, col: &crate::ast::ColumnRef, query: &Query) -> Result<Vec<TrainRow>> {
    if entity_table != col.table {
        return Err(anyhow!("predict column must come from entity table"));
    }
    if query.assuming.is_some() {
        return Err(anyhow!("ASSUMING not supported for non-aggregation predict"));
    }
    let ctx = reader
        .node_context(entity_table)
        .ok_or_else(|| anyhow!("missing rowpack context for table {entity_table}"))?;
    let decoder = RowPackDecoder::new(ctx.clone())?;
    let col_index = column_index_map(ctx);

    let entity_col = &query.for_each.column;
    let entity_idx = *col_index
        .get(entity_col)
        .ok_or_else(|| anyhow!("missing entity column {entity_col}"))?;
    let target_idx = *col_index
        .get(&col.column)
        .ok_or_else(|| anyhow!("missing target column {}", col.column))?;
    let timestamp_idx = find_timestamp_index(ctx, &col_index);

    let mut scratch = RowPackDecodeScratch::default();
    let mut iter = reader.iter_node_rows(entity_table, 0)?;
    let mut rows = Vec::new();
    while let Some(row) = iter.next_row()? {
        let Some(entity) = decode_value_opt(&decoder, &row, entity_idx, &mut scratch)? else {
            continue;
        };
        let value = decoder.decode_column_value(&row, target_idx, &mut scratch)?;
        let target = match value {
            ValueRef::Null => continue,
            ValueRef::Float64(v) => TargetValue::Number(v),
            ValueRef::Float32(v) => TargetValue::Number(v as f64),
            ValueRef::Int64(v) => TargetValue::Scalar(Value::Int(v)),
            ValueRef::Int32(v) => TargetValue::Scalar(Value::Int(v as i64)),
            ValueRef::UInt64(v) => TargetValue::Scalar(Value::UInt(v)),
            ValueRef::UInt32(v) => TargetValue::Scalar(Value::UInt(v as u64)),
            ValueRef::Bool(v) => TargetValue::Scalar(Value::Bool(v)),
            ValueRef::String(v) => TargetValue::Scalar(Value::String(v.to_string())),
            ValueRef::Bytes(v) => TargetValue::Scalar(Value::Bytes(v.to_vec())),
            ValueRef::Uuid(bytes) => TargetValue::Scalar(Value::String(hex_encode(&bytes))),
            ValueRef::Uuid256(bytes) => TargetValue::Scalar(Value::String(hex_encode(&bytes))),
            ValueRef::Ip(bytes) => TargetValue::Scalar(Value::String(hex_encode(&bytes))),
            other => return Err(anyhow!("unsupported value type: {other:?}")),
        };
        let ts = if let Some(ts_idx) = timestamp_idx {
            decode_timestamp(&decoder, &row, ts_idx, &mut scratch)?.unwrap_or(0)
        } else {
            0
        };
        rows.push(TrainRow {
            entity,
            target,
            timestamp: ts,
        });
    }
    Ok(rows)
}

fn column_index_map(ctx: &diskgraph_rowpack::RowPackContext) -> HashMap<String, usize> {
    ctx.columns
        .iter()
        .enumerate()
        .map(|(idx, col)| (col.name.clone(), idx))
        .collect()
}

fn decode_value_opt(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<Option<Value>> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(Some(match value {
        ValueRef::Null => return Ok(None),
        ValueRef::Int64(v) => Value::Int(v),
        ValueRef::Int32(v) => Value::Int(v as i64),
        ValueRef::UInt64(v) => Value::UInt(v),
        ValueRef::UInt32(v) => Value::UInt(v as u64),
        ValueRef::Bool(v) => Value::Bool(v),
        ValueRef::String(v) => Value::String(v.to_string()),
        ValueRef::Bytes(v) => Value::Bytes(v.to_vec()),
        ValueRef::Uuid(bytes) => Value::String(hex_encode(&bytes)),
        ValueRef::Uuid256(bytes) => Value::String(hex_encode(&bytes)),
        ValueRef::Ip(bytes) => Value::String(hex_encode(&bytes)),
        other => return Err(anyhow!("unsupported value type: {other:?}")),
    }))
}

fn decode_numeric_opt(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<Option<f64>> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(Some(match value {
        ValueRef::Null => return Ok(None),
        ValueRef::Int64(v) => v as f64,
        ValueRef::Int32(v) => v as f64,
        ValueRef::UInt64(v) => v as f64,
        ValueRef::UInt32(v) => v as f64,
        ValueRef::Float64(v) => v,
        ValueRef::Float32(v) => v as f64,
        other => return Err(anyhow!("unsupported numeric type: {other:?}")),
    }))
}

fn decode_timestamp(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<Option<i64>> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(match value {
        ValueRef::Timestamp { secs, .. } => Some(secs),
        ValueRef::Date(days) => Some(days as i64 * 86_400),
        ValueRef::Int64(v) => Some(v),
        ValueRef::Int32(v) => Some(v as i64),
        ValueRef::UInt64(v) => Some(v as i64),
        ValueRef::UInt32(v) => Some(v as i64),
        _ => None,
    })
}

fn find_timestamp_index(
    ctx: &diskgraph_rowpack::RowPackContext,
    col_index: &HashMap<String, usize>,
) -> Option<usize> {
    for (idx, col) in ctx.columns.iter().enumerate() {
        if matches!(col.logical_type, IngestedType::Timestamp | IngestedType::Date) {
            return Some(idx);
        }
    }
    for name in ["timestamp", "time", "ts"] {
        if let Some(idx) = col_index.get(name) {
            return Some(*idx);
        }
    }
    None
}

fn update_max_timestamp(map: &mut HashMap<Value, i64>, entity: &Value, ts: i64) {
    let entry = map.entry(entity.clone()).or_insert(ts);
    if ts > *entry {
        *entry = ts;
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn evaluate_assuming(
    predicate: &crate::ast::PredicateExpr,
    base_agg: &AggregationExpr,
    reader: &IndexReader,
    table: &str,
    entity_idx: usize,
    decoder: &RowPackDecoder,
    col_index: &HashMap<String, usize>,
) -> Result<HashSet<Value>> {
    let mut allowed = HashSet::new();

    match predicate {
        crate::ast::PredicateExpr::Comparison { left, op, right } => {
            let (agg_expr, literal) = match (left.as_ref(), right.as_ref()) {
                (crate::ast::Expr::Agg(agg), crate::ast::Expr::Number(n)) => {
                    (agg.as_ref(), *n as f64)
                }
                (crate::ast::Expr::Number(n), crate::ast::Expr::Agg(agg)) => {
                    (agg.as_ref(), *n as f64)
                }
                _ => return Err(anyhow!("unsupported ASSUMING predicate")),
            };

            let agg_table = match &agg_expr.target {
                AggTarget::Column(col) => &col.table,
                AggTarget::Star(tbl) => &tbl.table,
            };
            if table != agg_table {
                return Err(anyhow!("ASSUMING cross-table not supported in simple executor"));
            }

            let map = compute_numeric_agg(
                agg_expr,
                reader,
                table,
                entity_idx,
                decoder,
                col_index,
            )?;

            for (entity, value) in map {
                if compare_numeric(value, literal, *op) {
                    allowed.insert(entity);
                }
            }
        }
        _ => return Err(anyhow!("unsupported ASSUMING predicate")),
    }

    let _ = base_agg;
    Ok(allowed)
}

fn compute_numeric_agg(
    agg: &AggregationExpr,
    reader: &IndexReader,
    table: &str,
    entity_idx: usize,
    decoder: &RowPackDecoder,
    col_index: &HashMap<String, usize>,
) -> Result<HashMap<Value, f64>> {
    let target_idx = match &agg.target {
        AggTarget::Column(col) => Some(*col_index
            .get(&col.column)
            .ok_or_else(|| anyhow!("missing target column {}", col.column))?),
        AggTarget::Star(_) => None,
    };

    let mut iter = reader.iter_node_rows(table, 0)?;
    let mut scratch = RowPackDecodeScratch::default();
    let mut sums: HashMap<Value, (f64, u64, f64, f64)> = HashMap::new();
    let mut counts: HashMap<Value, u64> = HashMap::new();

    while let Some(row) = iter.next_row()? {
        if let Some(filter) = &agg.filter {
            if !eval_predicate_row(filter, decoder, &row, col_index, &mut scratch)? {
                continue;
            }
        }
        let Some(entity) = decode_value_opt(decoder, &row, entity_idx, &mut scratch)? else {
            continue;
        };
        match agg.func {
            AggFunc::Count => {
                if let Some(target_idx) = target_idx {
                    let Some(_value) = decode_value_opt(decoder, &row, target_idx, &mut scratch)? else {
                        continue;
                    };
                }
                let entry = counts.entry(entity).or_insert(0);
                *entry += 1;
            }
            AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
                let target_idx = target_idx.ok_or_else(|| anyhow!("numeric agg requires column"))?;
                let Some(value) = decode_numeric_opt(decoder, &row, target_idx, &mut scratch)? else {
                    continue;
                };
                let entry = sums.entry(entity).or_insert((0.0, 0, value, value));
                entry.0 += value;
                entry.1 += 1;
                if value < entry.2 {
                    entry.2 = value;
                }
                if value > entry.3 {
                    entry.3 = value;
                }
            }
            AggFunc::ListDistinct => {
                return Err(anyhow!("unsupported numeric agg"));
            }
        }
    }

    let mut out = HashMap::new();
    match agg.func {
        AggFunc::Count => {
            for (entity, count) in counts {
                out.insert(entity, count as f64);
            }
        }
        AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
            for (entity, (sum, count, min, max)) in sums {
                let v = match agg.func {
                    AggFunc::Sum => sum,
                    AggFunc::Avg => sum / count as f64,
                    AggFunc::Min => min,
                    AggFunc::Max => max,
                    _ => sum,
                };
                out.insert(entity, v);
            }
        }
        AggFunc::ListDistinct => {}
    }
    Ok(out)
}

fn compare_numeric(left: f64, right: f64, op: crate::ast::CompareOp) -> bool {
    match op {
        crate::ast::CompareOp::Eq => left == right,
        crate::ast::CompareOp::Neq => left != right,
        crate::ast::CompareOp::Gt => left > right,
        crate::ast::CompareOp::Gte => left >= right,
        crate::ast::CompareOp::Lt => left < right,
        crate::ast::CompareOp::Lte => left <= right,
    }
}

fn eval_predicate_row(
    pred: &crate::ast::PredicateExpr,
    decoder: &RowPackDecoder,
    row: &[u8],
    col_index: &HashMap<String, usize>,
    scratch: &mut RowPackDecodeScratch,
) -> Result<bool> {
    match pred {
        crate::ast::PredicateExpr::Comparison { left, op, right } => {
            let left_val = eval_expr_value(left, decoder, row, col_index, scratch)?;
            let right_val = eval_expr_value(right, decoder, row, col_index, scratch)?;
            Ok(compare_values(&left_val, &right_val, *op))
        }
        crate::ast::PredicateExpr::And(items) => {
            for item in items {
                if !eval_predicate_row(item, decoder, row, col_index, scratch)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        crate::ast::PredicateExpr::Or(items) => {
            for item in items {
                if eval_predicate_row(item, decoder, row, col_index, scratch)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

#[derive(Debug, Clone)]
enum SimpleValue {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
}

fn eval_expr_value(
    expr: &crate::ast::Expr,
    decoder: &RowPackDecoder,
    row: &[u8],
    col_index: &HashMap<String, usize>,
    scratch: &mut RowPackDecodeScratch,
) -> Result<SimpleValue> {
    match expr {
        crate::ast::Expr::Number(n) => Ok(SimpleValue::Number(*n)),
        crate::ast::Expr::String(s) => Ok(SimpleValue::String(s.clone())),
        crate::ast::Expr::Column(col) => {
            let idx = *col_index
                .get(&col.column)
                .ok_or_else(|| anyhow!("missing column {}", col.column))?;
            let value = decoder.decode_column_value(row, idx, scratch)?;
            Ok(match value {
                ValueRef::Null => SimpleValue::Null,
                ValueRef::Int64(v) => SimpleValue::Number(v as f64),
                ValueRef::Int32(v) => SimpleValue::Number(v as f64),
                ValueRef::UInt64(v) => SimpleValue::Number(v as f64),
                ValueRef::UInt32(v) => SimpleValue::Number(v as f64),
                ValueRef::Float64(v) => SimpleValue::Number(v),
                ValueRef::Float32(v) => SimpleValue::Number(v as f64),
                ValueRef::String(v) => SimpleValue::String(v.to_string()),
                ValueRef::Bool(v) => SimpleValue::Bool(v),
                _ => return Err(anyhow!("unsupported predicate value")),
            })
        }
        crate::ast::Expr::Agg(_) => Err(anyhow!("nested agg in predicate not supported")),
    }
}

fn compare_values(left: &SimpleValue, right: &SimpleValue, op: crate::ast::CompareOp) -> bool {
    if matches!(left, SimpleValue::Null) || matches!(right, SimpleValue::Null) {
        return false;
    }
    match (left, right) {
        (SimpleValue::Number(l), SimpleValue::Number(r)) => compare_numeric(*l, *r, op),
        (SimpleValue::String(l), SimpleValue::String(r)) => match op {
            crate::ast::CompareOp::Eq => l == r,
            crate::ast::CompareOp::Neq => l != r,
            _ => false,
        },
        (SimpleValue::Bool(l), SimpleValue::Bool(r)) => match op {
            crate::ast::CompareOp::Eq => l == r,
            crate::ast::CompareOp::Neq => l != r,
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_query;
    use diskgraph_core::metadata::{DatasetMetadata, NodeDataCompression, NodeShardMetadata, NodeTypeMetadata};
    use diskgraph_rowpack::{
        ColumnContext, ColumnEncoding, IngestedType, RowPackContext,
        RowPackEncoder, RowPackScratch, ValueRef,
    };
    use std::io::Write;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("diskgraph_train_table_exec_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn build_context() -> RowPackContext {
        RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "article_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "price".to_string(),
                    logical_type: IngestedType::Float64,
                    encoding: ColumnEncoding::Float {
                        storage: diskgraph_rowpack::FloatStorage::F64,
                    },
                },
            ],
            estimated_row_bytes: 32,
        }
    }

    fn write_rowpack_ctx(path: &std::path::Path, ctx: &RowPackContext) {
        let bytes = ctx.to_bytes().expect("ctx bytes");
        std::fs::create_dir_all(path.parent().unwrap()).expect("ctx dir");
        std::fs::write(path, bytes).expect("write ctx");
    }

    fn write_node_data(path: &std::path::Path, rows: Vec<Vec<u8>>) {
        std::fs::create_dir_all(path.parent().unwrap()).expect("node data dir");
        let mut file = std::fs::File::create(path).expect("node data");
        for row in rows {
            let len = row.len() as u32;
            file.write_all(&len.to_le_bytes()).expect("write len");
            file.write_all(&row).expect("write row");
        }
    }

    #[test]
    fn execute_sum_same_table() {
        let dir = temp_dir("sum");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(11), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(12), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT SUM(transactions.price) FOR EACH transactions.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entity, Value::Int(1));
        assert_eq!(rows[0].target, TargetValue::Number(4.0));
    }

    #[test]
    fn execute_avg_min_max_same_table() {
        let dir = temp_dir("avg_min_max");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(10), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(12), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");

        let avg_query = parse_query("PREDICT AVG(transactions.price) FOR EACH transactions.customer_id")
            .expect("parse");
        let avg_rows = execute_simple(&avg_query, &reader).expect("execute");
        assert_eq!(avg_rows[0].target, TargetValue::Number(2.0));

        let min_query = parse_query("PREDICT MIN(transactions.price) FOR EACH transactions.customer_id")
            .expect("parse");
        let min_rows = execute_simple(&min_query, &reader).expect("execute");
        assert_eq!(min_rows[0].target, TargetValue::Number(1.5));

        let max_query = parse_query("PREDICT MAX(transactions.price) FOR EACH transactions.customer_id")
            .expect("parse");
        let max_rows = execute_simple(&max_query, &reader).expect("execute");
        assert_eq!(max_rows[0].target, TargetValue::Number(2.5));
    }

    #[test]
    fn execute_count_with_filter() {
        let dir = temp_dir("count_filter");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(10), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(12), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query(
            "PREDICT COUNT(transactions.* WHERE transactions.article_id > 10) FOR EACH transactions.customer_id",
        )
        .expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows[0].target, TargetValue::Count(1));
    }

    #[test]
    fn execute_list_distinct_same_table() {
        let dir = temp_dir("list");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(10), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(11), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT LIST_DISTINCT(transactions.article_id) FOR EACH transactions.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        match &rows[0].target {
            TargetValue::List(values) => {
                let mut ids: Vec<i64> = values
                    .iter()
                    .filter_map(|v| if let Value::Int(id) = v { Some(*id) } else { None })
                    .collect();
                ids.sort();
                assert_eq!(ids, vec![10, 11]);
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn execute_predict_comparison() {
        let dir = temp_dir("predict_cmp");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(10), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(11), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query(
            "PREDICT COUNT(transactions.*, 0, 7, days) = 2 FOR EACH transactions.customer_id",
        )
        .expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].target, TargetValue::Scalar(Value::Bool(true)));
    }

    #[test]
    fn execute_assuming_filter() {
        let dir = temp_dir("assuming");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/transactions/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 3,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 3,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());

        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(10), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Int64(11), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");
        let mut row3 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Int64(12), ValueRef::Float64(3.0)],
                &mut row3,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/transactions/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2, row3]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query(
            "PREDICT SUM(transactions.price) FOR EACH transactions.customer_id ASSUMING COUNT(transactions.article_id) > 1",
        )
        .expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entity, Value::Int(1));
    }

    #[test]
    fn execute_predict_column() {
        let dir = temp_dir("predict_col");
        let ctx = RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: IngestedType::Bool,
                    encoding: ColumnEncoding::Bool,
                },
            ],
            estimated_row_bytes: 16,
        };
        let ctx_path = dir.join("nodes/customers/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "customers".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/customers/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Bool(true)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Bool(false)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/customers/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT customers.active FOR EACH customers.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].entity, Value::Int(1));
        assert_eq!(rows[0].target, TargetValue::Scalar(Value::Bool(true)));
    }

    #[test]
    fn execute_predict_column_ignores_null() {
        let dir = temp_dir("predict_col_null");
        let ctx = RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: IngestedType::Bool,
                    encoding: ColumnEncoding::Bool,
                },
            ],
            estimated_row_bytes: 16,
        };
        let ctx_path = dir.join("nodes/customers/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "customers".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/customers/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Bool(true)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Null],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/customers/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT customers.active FOR EACH customers.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entity, Value::Int(1));
        assert_eq!(rows[0].target, TargetValue::Scalar(Value::Bool(true)));
    }

    #[test]
    fn execute_predict_column_float() {
        let dir = temp_dir("predict_col_float");
        let ctx = RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "score".to_string(),
                    logical_type: IngestedType::Float64,
                    encoding: ColumnEncoding::Float {
                        storage: diskgraph_rowpack::FloatStorage::F64,
                    },
                },
            ],
            estimated_row_bytes: 16,
        };
        let ctx_path = dir.join("nodes/customers/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "customers".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/customers/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Float64(1.5)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Float64(2.5)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/customers/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT customers.score FOR EACH customers.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].target, TargetValue::Number(1.5));
    }

    #[test]
    fn execute_predict_comparison_column() {
        let dir = temp_dir("predict_cmp_col");
        let ctx = RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: IngestedType::Float64,
                    encoding: ColumnEncoding::Float {
                        storage: diskgraph_rowpack::FloatStorage::F64,
                    },
                },
            ],
            estimated_row_bytes: 16,
        };
        let ctx_path = dir.join("nodes/customers/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "customers".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/customers/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Float64(1.0)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Float64(0.0)],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/customers/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT customers.active = 1 FOR EACH customers.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].target, TargetValue::Scalar(Value::Bool(true)));
        assert_eq!(rows[1].target, TargetValue::Scalar(Value::Bool(false)));
    }

    #[test]
    fn execute_predict_comparison_column_ignores_null() {
        let dir = temp_dir("predict_cmp_col_null");
        let ctx = RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: IngestedType::Float64,
                    encoding: ColumnEncoding::Float {
                        storage: diskgraph_rowpack::FloatStorage::F64,
                    },
                },
            ],
            estimated_row_bytes: 16,
        };
        let ctx_path = dir.join("nodes/customers/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "customers".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/customers/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Float64(1.0)],
                &mut row1,
                &mut scratch,
            )
            .expect("encode row");
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(2), ValueRef::Null],
                &mut row2,
                &mut scratch,
            )
            .expect("encode row");

        let node_data_path = dir.join("nodes/customers/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row1, row2]);

        let reader = IndexReader::open(&dir).expect("reader");
        let query = parse_query("PREDICT customers.active = 1 FOR EACH customers.customer_id").expect("parse");
        let rows = execute_simple(&query, &reader).expect("execute");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entity, Value::Int(1));
        assert_eq!(rows[0].target, TargetValue::Scalar(Value::Bool(true)));
    }
}
