use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use diskgraph_rowpack::{IngestedType, RowPackDecodeScratch, RowPackDecoder, ValueRef};

use crate::ast::{AggFunc, AggTarget, AggregationExpr, Query};
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRow {
    pub entity: Value,
    pub target: TargetValue,
    pub timestamp: i64,
}

pub fn execute_simple(query: &Query, reader: &IndexReader) -> Result<Vec<TrainRow>> {
    let entity_table = &query.for_each.table;
    let agg = &query.predict.aggregation;
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
        AggFunc::Sum => {
            let target_idx = target_idx.ok_or_else(|| anyhow!("SUM requires column"))?;
            let mut sums: HashMap<Value, f64> = HashMap::new();
            let mut timestamps: HashMap<Value, i64> = HashMap::new();
            while let Some(row) = iter.next_row()? {
                let entity = decode_value(&decoder, &row, entity_idx, &mut scratch)?;
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
                }
                let value = decode_numeric(&decoder, &row, target_idx, &mut scratch)?;
                let entry = sums.entry(entity).or_insert(0.0);
                *entry += value;
                if let Some(ts_idx) = timestamp_idx {
                    if let Some(ts) = decode_timestamp(&decoder, &row, ts_idx, &mut scratch)? {
                        update_max_timestamp(&mut timestamps, &entity_key, ts);
                    }
                }
            }
            Ok(sums
                .into_iter()
                .map(|(entity, sum)| TrainRow {
                    timestamp: timestamps.get(&entity).copied().unwrap_or(0),
                    entity,
                    target: TargetValue::Number(sum),
                })
                .collect())
        }
        AggFunc::Count => {
            let mut counts: HashMap<Value, u64> = HashMap::new();
            let mut timestamps: HashMap<Value, i64> = HashMap::new();
            while let Some(row) = iter.next_row()? {
                let entity = decode_value(&decoder, &row, entity_idx, &mut scratch)?;
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
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
                let entity = decode_value(&decoder, &row, entity_idx, &mut scratch)?;
                let entity_key = entity.clone();
                if let Some(allowed) = &allowed_entities {
                    if !allowed.contains(&entity) {
                        continue;
                    }
                }
                let target = decode_value(&decoder, &row, target_idx, &mut scratch)?;
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
        _ => Err(anyhow!("aggregation not supported in simple executor")),
    }
}

fn column_index_map(ctx: &diskgraph_rowpack::RowPackContext) -> HashMap<String, usize> {
    ctx.columns
        .iter()
        .enumerate()
        .map(|(idx, col)| (col.name.clone(), idx))
        .collect()
}

fn decode_value(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<Value> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(match value {
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
    })
}

fn decode_numeric(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<f64> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(match value {
        ValueRef::Int64(v) => v as f64,
        ValueRef::Int32(v) => v as f64,
        ValueRef::UInt64(v) => v as f64,
        ValueRef::UInt32(v) => v as f64,
        ValueRef::Float64(v) => v,
        ValueRef::Float32(v) => v as f64,
        other => return Err(anyhow!("unsupported numeric type: {other:?}")),
    })
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
        AggTarget::Column(col) => *col_index
            .get(&col.column)
            .ok_or_else(|| anyhow!("missing target column {}", col.column))?,
        AggTarget::Star(_) => return Err(anyhow!("numeric agg requires column")),
    };

    let mut iter = reader.iter_node_rows(table, 0)?;
    let mut scratch = RowPackDecodeScratch::default();
    let mut sums: HashMap<Value, (f64, u64)> = HashMap::new();

    while let Some(row) = iter.next_row()? {
        if let Some(filter) = &agg.filter {
            if !eval_predicate_row(filter, decoder, &row, col_index, &mut scratch)? {
                continue;
            }
        }
        let entity = decode_value(decoder, &row, entity_idx, &mut scratch)?;
        let value = decode_numeric(decoder, &row, target_idx, &mut scratch)?;
        let entry = sums.entry(entity).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    let mut out = HashMap::new();
    for (entity, (sum, count)) in sums {
        let v = match agg.func {
            AggFunc::Sum => sum,
            AggFunc::Count => count as f64,
            AggFunc::Avg => sum / count as f64,
            AggFunc::Min | AggFunc::Max | AggFunc::ListDistinct => {
                return Err(anyhow!("unsupported numeric agg"));
            }
        };
        out.insert(entity, v);
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
}

fn eval_expr_value(
    expr: &crate::ast::Expr,
    decoder: &RowPackDecoder,
    row: &[u8],
    col_index: &HashMap<String, usize>,
    scratch: &mut RowPackDecodeScratch,
) -> Result<SimpleValue> {
    match expr {
        crate::ast::Expr::Number(n) => Ok(SimpleValue::Number(*n as f64)),
        crate::ast::Expr::String(s) => Ok(SimpleValue::String(s.clone())),
        crate::ast::Expr::Column(col) => {
            let idx = *col_index
                .get(&col.column)
                .ok_or_else(|| anyhow!("missing column {}", col.column))?;
            let value = decoder.decode_column_value(row, idx, scratch)?;
            Ok(match value {
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
}
