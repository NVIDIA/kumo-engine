use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::Result;
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_rowpack::{ColumnContext, ColumnEncoding, IngestedType, RowPackContext};

use crate::ast::*;
use crate::report::ValidationMessage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    Numeric,
    Bool,
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Array,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TableSchema {
    pub columns: HashMap<String, ColumnType>,
}

#[derive(Debug, Clone)]
pub struct IndexSchema {
    pub metadata: DatasetMetadata,
    pub tables: HashMap<String, TableSchema>,
}

#[derive(Debug, Default, Clone)]
pub struct ValidationOutcome {
    pub errors: Vec<ValidationMessage>,
    pub warnings: Vec<ValidationMessage>,
}

impl ValidationOutcome {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn error(&mut self, code: &str, message: impl Into<String>) {
        self.errors.push(ValidationMessage {
            code: code.to_string(),
            message: message.into(),
        });
    }

    fn warn(&mut self, code: &str, message: impl Into<String>) {
        self.warnings.push(ValidationMessage {
            code: code.to_string(),
            message: message.into(),
        });
    }
}

impl IndexSchema {
    pub fn from_index_path(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref();
        let metadata_path = index_path.join("index_metadata.json");
        let metadata_bytes = fs::read(&metadata_path)?;
        let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;

        let mut tables = HashMap::new();
        for node in &metadata.node_types {
            let schema = if let Some(node_data) = &node.node_data_compression {
                let ctx_path = index_path.join(&node_data.context_path);
                if ctx_path.exists() {
                    let bytes = fs::read(ctx_path)?;
                    let ctx = RowPackContext::from_bytes(&bytes)?;
                    Some(build_table_schema(&ctx.columns))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(schema) = schema {
                tables.insert(node.name.clone(), schema);
            }
        }

        Ok(Self { metadata, tables })
    }

    pub fn has_table(&self, name: &str) -> bool {
        self.metadata.node_types.iter().any(|t| t.name == name)
    }

    pub fn has_relation(&self, name: &str) -> bool {
        self.metadata.edge_relations.iter().any(|r| r.name == name)
    }

    pub fn get_column_type(&self, table: &str, column: &str) -> Option<ColumnType> {
        self.tables
            .get(table)
            .and_then(|t| t.columns.get(column))
            .cloned()
    }

    pub fn has_schema(&self, table: &str) -> bool {
        self.tables.contains_key(table)
    }
}

fn build_table_schema(columns: &[ColumnContext]) -> TableSchema {
    let mut map = HashMap::new();
    for col in columns {
        map.insert(col.name.clone(), column_type(&col.encoding, col.logical_type));
    }
    TableSchema { columns: map }
}

fn column_type(encoding: &ColumnEncoding, logical: IngestedType) -> ColumnType {
    match encoding {
        ColumnEncoding::Int { .. } => ColumnType::Numeric,
        ColumnEncoding::Float { .. } => ColumnType::Numeric,
        ColumnEncoding::Decimal { .. } => ColumnType::Numeric,
        ColumnEncoding::Bool => ColumnType::Bool,
        ColumnEncoding::String { .. } => ColumnType::String,
        ColumnEncoding::Bytes { .. } => ColumnType::Bytes,
        ColumnEncoding::Uuid => ColumnType::Uuid,
        ColumnEncoding::Uuid256 => ColumnType::Uuid256,
        ColumnEncoding::Ip => ColumnType::Ip,
        ColumnEncoding::Date { .. } => ColumnType::Date,
        ColumnEncoding::Timestamp => ColumnType::Timestamp,
        ColumnEncoding::FloatArray { .. } => ColumnType::Array,
    }
    .or_else(|| match logical {
        IngestedType::Date => ColumnType::Date,
        IngestedType::Timestamp => ColumnType::Timestamp,
        IngestedType::Uuid => ColumnType::Uuid,
        IngestedType::Uuid256 => ColumnType::Uuid256,
        IngestedType::Ip => ColumnType::Ip,
        IngestedType::Float16Array | IngestedType::Float32Array | IngestedType::Float64Array => {
            ColumnType::Array
        }
        _ => ColumnType::Unknown,
    })
}

trait ColumnTypeExt {
    fn or_else(self, fallback: impl FnOnce() -> ColumnType) -> ColumnType;
}

impl ColumnTypeExt for ColumnType {
    fn or_else(self, fallback: impl FnOnce() -> ColumnType) -> ColumnType {
        match self {
            ColumnType::Unknown => fallback(),
            other => other,
        }
    }
}

pub fn validate_query(query: &Query, schema: &IndexSchema) -> ValidationOutcome {
    let mut outcome = ValidationOutcome::default();

    let mut table_refs = HashSet::new();
    collect_table_refs(query, &mut table_refs);

    for table in &table_refs {
        if !schema.has_table(table) && !schema.has_relation(table) {
            outcome.error("UNKNOWN_TABLE", format!("unknown table: {table}"));
        } else if schema.has_table(table) && !schema.has_schema(table) {
            outcome.warn("MISSING_SCHEMA", format!("no node schema for table: {table}"));
        }
    }

    let mut col_refs = Vec::new();
    collect_column_refs(query, &mut col_refs);
    for col in &col_refs {
        if !schema.has_table(&col.table) {
            continue;
        }
        if schema.has_schema(&col.table) && schema.get_column_type(&col.table, &col.column).is_none() {
            outcome.error(
                "UNKNOWN_COLUMN",
                format!("unknown column: {}.{}", col.table, col.column),
            );
        }
    }

    match &query.predict {
        PredictExpr::Aggregation(agg) => {
            validate_aggregation(agg, schema, &mut outcome);
        }
        PredictExpr::Column(col) => {
            if col.table != query.for_each.table {
                outcome.error(
                    "PREDICT_COLUMN_TABLE_MISMATCH",
                    "predict column must come from entity table",
                );
            }
        }
        PredictExpr::Comparison(pred) => {
            match extract_predict_comparison(pred) {
                Ok(agg) => validate_aggregation(&agg, schema, &mut outcome),
                Err(msg) => outcome.error("PREDICT_COMPARISON_INVALID", msg),
            }
        }
    }
    if let Some(assuming) = &query.assuming {
        validate_predicate(assuming, schema, &mut outcome);
        if matches!(query.predict, PredictExpr::Column(_)) && has_agg_in_pred(assuming) {
            outcome.error(
                "PREDICT_COLUMN_ASSUMING_AGG",
                "ASSUMING aggregations not supported for scalar predict",
            );
        }
    }

    if matches!(query.predict, PredictExpr::Aggregation(_) | PredictExpr::Comparison(_)) {
        validate_temporal_query(query, schema, &mut outcome);
    }

    outcome
}

fn validate_temporal_query(query: &Query, schema: &IndexSchema, outcome: &mut ValidationOutcome) {
    let mut aggs = Vec::new();
    match &query.predict {
        PredictExpr::Aggregation(agg) => aggs.push(agg.clone()),
        PredictExpr::Comparison(pred) => collect_aggs_from_pred(pred, &mut aggs),
        PredictExpr::Column(_) => return,
    }
    if let Some(assuming) = &query.assuming {
        collect_aggs_from_pred(assuming, &mut aggs);
    }
    for agg in aggs {
        if agg.window.is_none() {
            continue;
        }
        let entity_table = &query.for_each.table;
        match &agg.target {
            AggTarget::Star(tbl) => {
                if agg.filter.is_some() {
                    outcome.error(
                        "temporal_filter_invalid",
                        "filters on edge relations are not supported",
                    );
                }
                let rel = schema
                    .metadata
                    .edge_relations
                    .iter()
                    .find(|rel| rel.name == tbl.table);
                match rel {
                    Some(rel) => {
                        let has_ts = rel
                            .dest_sort_key
                            .as_ref()
                            .and_then(|k| k.timestamp.as_ref())
                            .is_some();
                        if !has_ts {
                            outcome.error(
                                "temporal_missing_timestamp",
                                format!("relation {} missing timestamped edge list", tbl.table),
                            );
                        }
                    }
                    None => outcome.error(
                        "temporal_missing_relation",
                        format!("temporal aggregation requires edge relation {}", tbl.table),
                    ),
                }
            }
            AggTarget::Column(col) => {
                if let Some(filter) = &agg.filter {
                    let mut cols = Vec::new();
                    collect_columns_from_pred(filter, &mut cols);
                    for c in cols {
                        if c.table != col.table {
                            outcome.error(
                                "temporal_filter_invalid",
                                "temporal filters must reference dst node table only",
                            );
                            break;
                        }
                    }
                }
                let rels: Vec<_> = schema
                    .metadata
                    .edge_relations
                    .iter()
                    .filter(|rel| rel.src_type == *entity_table && rel.dst_type == col.table)
                    .collect();
                if rels.is_empty() {
                    outcome.error(
                        "temporal_missing_relation",
                        format!("no relation from {} to {}", entity_table, col.table),
                    );
                } else if rels.len() > 1 {
                    outcome.error(
                        "temporal_ambiguous_relation",
                        format!("multiple relations from {} to {}", entity_table, col.table),
                    );
                } else {
                    let rel = rels[0];
                    let has_ts = rel
                        .dest_sort_key
                        .as_ref()
                        .and_then(|k| k.timestamp.as_ref())
                        .is_some();
                    if !has_ts {
                        outcome.error(
                            "temporal_missing_timestamp",
                            format!("relation {} missing timestamped edge list", rel.name),
                        );
                    }
                }
            }
        }
    }
}

fn collect_table_refs(query: &Query, out: &mut HashSet<String>) {
    out.insert(query.for_each.table.clone());
    match &query.predict {
        PredictExpr::Aggregation(agg) => collect_tables_from_agg(agg, out),
        PredictExpr::Column(col) => {
            out.insert(col.table.clone());
        }
        PredictExpr::Comparison(pred) => collect_tables_from_pred(pred, out),
    }
    if let Some(assuming) = &query.assuming {
        collect_tables_from_pred(assuming, out);
    }
}

fn collect_tables_from_agg(agg: &AggregationExpr, out: &mut HashSet<String>) {
    match &agg.target {
        AggTarget::Column(col) => {
            out.insert(col.table.clone());
        }
        AggTarget::Star(tbl) => {
            out.insert(tbl.table.clone());
        }
    }
    if let Some(filter) = &agg.filter {
        collect_tables_from_pred(filter, out);
    }
}

fn collect_tables_from_pred(pred: &PredicateExpr, out: &mut HashSet<String>) {
    match pred {
        PredicateExpr::Comparison { left, right, .. } => {
            collect_tables_from_expr(left, out);
            collect_tables_from_expr(right, out);
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                collect_tables_from_pred(item, out);
            }
        }
    }
}

fn collect_tables_from_expr(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Column(col) => {
            out.insert(col.table.clone());
        }
        Expr::Agg(agg) => collect_tables_from_agg(agg, out),
        Expr::Number(_) | Expr::String(_) => {}
    }
}

fn collect_column_refs(query: &Query, out: &mut Vec<ColumnRef>) {
    out.push(ColumnRef {
        table: query.for_each.table.clone(),
        column: query.for_each.column.clone(),
    });
    match &query.predict {
        PredictExpr::Aggregation(agg) => collect_columns_from_agg(agg, out),
        PredictExpr::Column(col) => out.push(col.clone()),
        PredictExpr::Comparison(pred) => collect_columns_from_pred(pred, out),
    }
    if let Some(assuming) = &query.assuming {
        collect_columns_from_pred(assuming, out);
    }
}

fn has_agg_in_pred(pred: &PredicateExpr) -> bool {
    match pred {
        PredicateExpr::Comparison { left, right, .. } => {
            expr_has_agg(left) || expr_has_agg(right)
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            items.iter().any(has_agg_in_pred)
        }
    }
}

fn expr_has_agg(expr: &crate::ast::Expr) -> bool {
    match expr {
        crate::ast::Expr::Agg(_) => true,
        _ => false,
    }
}

fn collect_columns_from_agg(agg: &AggregationExpr, out: &mut Vec<ColumnRef>) {
    if let AggTarget::Column(col) = &agg.target {
        out.push(col.clone());
    }
    if let Some(filter) = &agg.filter {
        collect_columns_from_pred(filter, out);
    }
}

fn collect_columns_from_pred(pred: &PredicateExpr, out: &mut Vec<ColumnRef>) {
    match pred {
        PredicateExpr::Comparison { left, right, .. } => {
            collect_columns_from_expr(left, out);
            collect_columns_from_expr(right, out);
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                collect_columns_from_pred(item, out);
            }
        }
    }
}

fn collect_columns_from_expr(expr: &Expr, out: &mut Vec<ColumnRef>) {
    match expr {
        Expr::Column(col) => out.push(col.clone()),
        Expr::Agg(agg) => collect_columns_from_agg(agg, out),
        Expr::Number(_) | Expr::String(_) => {}
    }
}

fn collect_aggs_from_pred(pred: &PredicateExpr, out: &mut Vec<AggregationExpr>) {
    match pred {
        PredicateExpr::Comparison { left, right, .. } => {
            collect_aggs_from_expr(left, out);
            collect_aggs_from_expr(right, out);
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                collect_aggs_from_pred(item, out);
            }
        }
    }
}

fn extract_predict_comparison(pred: &PredicateExpr) -> Result<AggregationExpr, String> {
    let PredicateExpr::Comparison { left, right, .. } = pred else {
        return Err("predict comparison must be a single comparison".to_string());
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Agg(agg), Expr::Number(_)) | (Expr::Number(_), Expr::Agg(agg)) => {
            if matches!(agg.func, AggFunc::ListDistinct) {
                return Err("LIST_DISTINCT is not supported in predict comparisons".to_string());
            }
            Ok((**agg).clone())
        }
        _ => Err("predict comparison must compare aggregation to a number".to_string()),
    }
}

fn collect_aggs_from_expr(expr: &Expr, out: &mut Vec<AggregationExpr>) {
    match expr {
        Expr::Agg(agg) => out.push((**agg).clone()),
        Expr::Column(_) | Expr::Number(_) | Expr::String(_) => {}
    }
}

fn validate_aggregation(agg: &AggregationExpr, schema: &IndexSchema, outcome: &mut ValidationOutcome) {
    let target_type = match &agg.target {
        AggTarget::Column(col) => schema.get_column_type(&col.table, &col.column),
        AggTarget::Star(_) => None,
    };

    match agg.func {
        AggFunc::Sum | AggFunc::Avg => {
            if let Some(ty) = target_type {
                if ty != ColumnType::Numeric {
                    outcome.error("INVALID_TYPE", "SUM/AVG require numeric column");
                }
            }
        }
        AggFunc::Min | AggFunc::Max => {
            if let Some(ty) = target_type {
                if !(ty == ColumnType::Numeric || ty == ColumnType::Date || ty == ColumnType::Timestamp) {
                    outcome.error("INVALID_TYPE", "MIN/MAX require numeric or time column");
                }
            }
        }
        AggFunc::Count | AggFunc::ListDistinct => {}
    }

    if let Some(filter) = &agg.filter {
        validate_predicate(filter, schema, outcome);
    }

}

fn validate_predicate(pred: &PredicateExpr, schema: &IndexSchema, outcome: &mut ValidationOutcome) {
    match pred {
        PredicateExpr::Comparison { left, op, right } => {
            let left_type = expr_type(left, schema);
            let right_type = expr_type(right, schema);
            if let (Some(l), Some(r)) = (left_type, right_type) {
                if !types_comparable(&l, &r, *op) {
                    outcome.error("INVALID_TYPE", "incompatible predicate types");
                }
            }
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                validate_predicate(item, schema, outcome);
            }
        }
    }
}

fn expr_type(expr: &Expr, schema: &IndexSchema) -> Option<ColumnType> {
    match expr {
        Expr::Column(col) => schema.get_column_type(&col.table, &col.column),
        Expr::Number(_) => Some(ColumnType::Numeric),
        Expr::String(_) => Some(ColumnType::String),
        Expr::Agg(agg) => match agg.func {
            AggFunc::Sum | AggFunc::Avg | AggFunc::Count => Some(ColumnType::Numeric),
            AggFunc::Min | AggFunc::Max => Some(ColumnType::Numeric),
            AggFunc::ListDistinct => None,
        },
    }
}

fn types_comparable(left: &ColumnType, right: &ColumnType, op: CompareOp) -> bool {
    match (left, right) {
        (ColumnType::Numeric, ColumnType::Numeric) => true,
        (ColumnType::String, ColumnType::String)
        | (ColumnType::Bytes, ColumnType::Bytes)
        | (ColumnType::Uuid, ColumnType::Uuid)
        | (ColumnType::Uuid256, ColumnType::Uuid256)
        | (ColumnType::Ip, ColumnType::Ip) => true,
        (ColumnType::Bool, ColumnType::Bool) => matches!(op, CompareOp::Eq | CompareOp::Neq),
        (ColumnType::Date, ColumnType::Date) | (ColumnType::Timestamp, ColumnType::Timestamp) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_query;
    use diskgraph_core::metadata::{DatasetMetadata, NodeDataCompression, NodeShardMetadata, NodeTypeMetadata};
    use diskgraph_rowpack::{RowPackConfig, RowPackContextBuilder, RowPackColumn};
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("diskgraph_train_table_test_{}", name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_context(path: &Path, columns: Vec<RowPackColumn>) {
        let builder = RowPackContextBuilder::new(columns, RowPackConfig::default());
        let ctx = builder.finish().expect("ctx");
        let bytes = ctx.to_bytes().expect("ctx bytes");
        fs::write(path, bytes).expect("write ctx");
    }

    fn build_metadata(index_path: &Path) -> DatasetMetadata {
        let user_ctx = index_path.join("nodes/users/node_data.ctx");
        let txn_ctx = index_path.join("nodes/transactions/node_data.ctx");
        fs::create_dir_all(user_ctx.parent().unwrap()).expect("user ctx dir");
        fs::create_dir_all(txn_ctx.parent().unwrap()).expect("txn ctx dir");

        write_context(
            &user_ctx,
            vec![RowPackColumn { name: "user_id".to_string(), logical_type: IngestedType::Int64, fixed_len: None, precision: None, scale: None }],
        );
        write_context(
            &txn_ctx,
            vec![
                RowPackColumn { name: "price".to_string(), logical_type: IngestedType::Float64, fixed_len: None, precision: None, scale: None },
                RowPackColumn { name: "article_id".to_string(), logical_type: IngestedType::Uuid, fixed_len: None, precision: None, scale: None },
            ],
        );

        DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![
                NodeTypeMetadata {
                    name: "users".to_string(),
                    max_id: 1,
                    has_node_data: true,
                    node_data_compression: Some(NodeDataCompression { codec: "rowpack".to_string(), context_path: "nodes/users/node_data.ctx".to_string() }),
                    primary_key_map: None,
                    shards: vec![NodeShardMetadata { min_id: 0, max_id: 0, node_data_paths: None }],
                },
                NodeTypeMetadata {
                    name: "transactions".to_string(),
                    max_id: 1,
                    has_node_data: true,
                    node_data_compression: Some(NodeDataCompression { codec: "rowpack".to_string(), context_path: "nodes/transactions/node_data.ctx".to_string() }),
                    primary_key_map: None,
                    shards: vec![NodeShardMetadata { min_id: 0, max_id: 0, node_data_paths: None }],
                },
            ],
            edge_relations: vec![diskgraph_core::metadata::EdgeRelationMetadata {
                name: "transactions".to_string(),
                src_type: "users".to_string(),
                dst_type: "transactions".to_string(),
                shards: vec![],
                dest_sort_key_bits: Some(16),
                dest_sort_key: Some(diskgraph_core::metadata::EdgeSortKeyMetadata {
                    width_bits: 16,
                    timestamp: Some(diskgraph_core::metadata::EdgeSortKeyTimestampMetadata {
                        table: "transactions".to_string(),
                        column: "timestamp".to_string(),
                        encoding: diskgraph_core::metadata::EdgeSortKeyTimestampEncoding {
                            kind: diskgraph_core::metadata::EdgeSortKeyTimestampKind::Int16,
                            anchor: 0,
                            step: 1,
                        },
                    }),
                    weight: None,
                }),
                format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
            }],
        }
    }

    fn write_metadata(index_path: &Path, metadata: &DatasetMetadata) {
        let bytes = serde_json::to_vec_pretty(metadata).expect("metadata json");
        fs::write(index_path.join("index_metadata.json"), bytes).expect("write metadata");
    }

    #[test]
    fn validate_valid_query() {
        let dir = temp_dir("valid");
        let metadata = build_metadata(&dir);
        write_metadata(&dir, &metadata);

        let schema = IndexSchema::from_index_path(&dir).expect("schema");
        let query = parse_query("PREDICT SUM(transactions.price, 0, 7, days) FOR EACH users.user_id").expect("parse");
        let outcome = validate_query(&query, &schema);
        assert!(outcome.errors.is_empty());
    }

    #[test]
    fn validate_missing_column() {
        let dir = temp_dir("missing_col");
        let metadata = build_metadata(&dir);
        write_metadata(&dir, &metadata);

        let schema = IndexSchema::from_index_path(&dir).expect("schema");
        let query = parse_query("PREDICT SUM(transactions.missing, 0, 7, days) FOR EACH users.user_id").expect("parse");
        let outcome = validate_query(&query, &schema);
        assert!(outcome.errors.iter().any(|e| e.code == "UNKNOWN_COLUMN"));
    }

    #[test]
    fn validate_invalid_type() {
        let dir = temp_dir("invalid_type");
        let metadata = build_metadata(&dir);
        write_metadata(&dir, &metadata);

        let schema = IndexSchema::from_index_path(&dir).expect("schema");
        let query = parse_query("PREDICT SUM(transactions.article_id, 0, 7, days) FOR EACH users.user_id").expect("parse");
        let outcome = validate_query(&query, &schema);
        assert!(outcome.errors.iter().any(|e| e.code == "INVALID_TYPE"));
    }

    #[test]
    fn validate_unknown_table_in_assuming() {
        let dir = temp_dir("unknown_table");
        let metadata = build_metadata(&dir);
        write_metadata(&dir, &metadata);

        let schema = IndexSchema::from_index_path(&dir).expect("schema");
        let query = parse_query("PREDICT SUM(transactions.price, 0, 7, days) FOR EACH users.user_id ASSUMING COUNT(promotion.* WHERE promotion.TYPE = 1, 0, 7, days) > 1").expect("parse");
        let outcome = validate_query(&query, &schema);
        assert!(outcome.errors.iter().any(|e| e.code == "UNKNOWN_TABLE"));
    }

    #[test]
    fn validate_predict_comparison() {
        let dir = temp_dir("predict_comparison");
        let metadata = build_metadata(&dir);
        write_metadata(&dir, &metadata);

        let schema = IndexSchema::from_index_path(&dir).expect("schema");
        let query = parse_query(
            "PREDICT COUNT(transactions.*, 0, 7, days) = 0 FOR EACH users.user_id",
        )
        .expect("parse");
        let outcome = validate_query(&query, &schema);
        assert!(outcome.errors.is_empty());
    }
}
