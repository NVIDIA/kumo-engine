use anyhow::{anyhow, Result};
use diskgraph_core::metadata::EdgeStorageFormat;
use diskgraph_rowpack::RowPackConfig;
use fasteval::{Evaler, Parser, Slab};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

fn source_dtype_is_decimal(dtype: Option<&str>) -> bool {
    let Some(dtype) = dtype else {
        return false;
    };
    let dtype = dtype.trim_start();
    dtype.starts_with("Decimal128") || dtype.starts_with("Decimal256") || dtype.starts_with("Decimal")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub dataset_name: String,
    pub tables: Vec<TableSpec>,
    #[serde(default)]
    pub relationships: Vec<RelationshipSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_sort_keys: Option<EdgeSortKeysSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_shards: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_format: Option<EdgeStorageFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSpec {
    pub name: String,
    pub format: TableFormat,
    pub paths: Vec<String>,
    pub columns: Vec<ColumnSpec>,
    #[serde(default)]
    pub primary_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint_bits: Option<u8>,
    #[serde(default)]
    pub create_time: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub edge_table: Option<EdgeTableSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<String>,
    pub ingested_dtype: IngestedType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_len: Option<usize>,
    // For timestamp/date columns inferred from strings, this captures the parse format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_format: Option<String>,
    // For integer timestamps, indicates the epoch unit to interpret the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_unit: Option<TimestampUnit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestedType {
    Int8,
    Int16,
    Int64,
    Int32,
    UInt8,
    UInt16,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Bool,
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float32Array,
    Float64Array,
    Float16Array,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampUnit {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableFormat {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTableSpec {
    pub src_fk: ForeignKeySpec,
    pub dst_fk: ForeignKeySpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeySpec {
    pub column: String,
    pub ref_table: String,
    pub ref_column: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipSpec {
    #[serde(default)]
    pub name: Option<String>,
    pub src_table: String,
    pub src_column: String,
    pub dst_table: String,
    pub dst_column: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeSortKeysSpec {
    #[serde(default)]
    pub relationships: HashMap<String, RelationshipSortKeysSpec>,
    #[serde(default)]
    pub edge_tables: HashMap<String, EdgeTableSortKeysSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelationshipSortKeysSpec {
    #[serde(default)]
    pub forward: Option<SortKeySpec>,
    #[serde(default)]
    pub reverse: Option<SortKeySpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeTableSortKeysSpec {
    #[serde(default)]
    pub src_to_dst: Option<SortKeySpec>,
    #[serde(default)]
    pub dst_to_src: Option<SortKeySpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SortKeySpec {
    #[serde(default)]
    pub timestamp: Option<SortKeyTimestampSpec>,
    #[serde(default)]
    pub weight: Option<SortKeyWeightSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortKeyTimestampSpec {
    pub table: String,
    pub column: String,
    pub encoding: TimestampEncodingSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampEncodingSpec {
    pub kind: TimestampEncodingKind,
    pub anchor: i64,
    pub step: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampEncodingKind {
    Int8,
    Int16,
    Int32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortKeyWeightSpec {
    pub table: String,
    pub expr: String,
    pub encoding: WeightEncodingSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEncodingSpec {
    pub dtype: WeightEncodingKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeightEncodingKind {
    Byte,
    UInt16,
    UInt32,
    Float16,
    Float32,
}

pub fn relationship_name(rel: &RelationshipSpec) -> String {
    rel.name.clone().unwrap_or_else(|| {
        format!(
            "{}_{}_to_{}_{}",
            rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
        )
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CloudSource {
    S3 { bucket: String },
    Gcs { bucket: String },
    Adls { account: String, container: String },
}

fn parse_cloud_source(path: &str) -> Result<Option<CloudSource>> {
    if let Some(rest) = path.strip_prefix("s3://") {
        let mut parts = rest.splitn(2, '/');
        let bucket = parts
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if bucket.is_empty() {
            return Err(anyhow!("invalid s3 path {path}"));
        }
        return Ok(Some(CloudSource::S3 { bucket }));
    }
    if let Some(rest) = path.strip_prefix("gs://") {
        let mut parts = rest.splitn(2, '/');
        let bucket = parts
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if bucket.is_empty() {
            return Err(anyhow!("invalid gs path {path}"));
        }
        return Ok(Some(CloudSource::Gcs { bucket }));
    }
    if let Some(rest) = path.strip_prefix("az://") {
        let mut parts = rest.splitn(3, '/');
        let account = parts.next().unwrap_or("").trim().to_string();
        let container = parts.next().unwrap_or("").trim().to_string();
        if account.is_empty() || container.is_empty() {
            return Err(anyhow!("invalid az path {path}"));
        }
        return Ok(Some(CloudSource::Adls { account, container }));
    }
    Ok(None)
}

impl SourceMetadata {
    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        if self.dataset_name.trim().is_empty() {
            errors.push("dataset_name is required".to_string());
        }
        if self.tables.is_empty() {
            errors.push("at least one table is required".to_string());
        }
        if let Some(shards) = self.num_shards {
            if shards == 0 {
                errors.push("num_shards must be >= 1".to_string());
            }
        }
        if let Some(format) = self.edge_format {
            if matches!(format, EdgeStorageFormat::Parquet) {
                errors.push("parquet output format is no longer supported".to_string());
            }
        }
        let mut table_names = HashSet::new();
        let mut table_map: HashMap<&str, &TableSpec> = HashMap::new();
        let mut rel_keys = HashSet::new();
        let mut cloud_source: Option<CloudSource> = None;
        let mut has_local_paths = false;
        for table in &self.tables {
            if table.name.trim().is_empty() {
                errors.push("table name cannot be empty".to_string());
                continue;
            }
            if !table_names.insert(table.name.as_str()) {
                errors.push(format!("duplicate table name {}", table.name));
            }
            if table.paths.is_empty() {
                errors.push(format!("table {} has no paths", table.name));
            }
            if table.columns.is_empty() {
                errors.push(format!("table {} has no columns", table.name));
            }
            let mut col_names = HashSet::new();
            for path in &table.paths {
                match parse_cloud_source(path) {
                    Ok(Some(source)) => {
                        if has_local_paths {
                            errors.push(format!(
                                "table {} mixes local and cloud paths",
                                table.name
                            ));
                        }
                        if let Some(existing) = &cloud_source {
                            if existing != &source {
                                errors.push(
                                    "all cloud paths must share the same bucket/container"
                                        .to_string(),
                                );
                            }
                        } else {
                            cloud_source = Some(source);
                        }
                    }
                    Ok(None) => {
                        if cloud_source.is_some() {
                            errors.push(format!(
                                "table {} mixes local and cloud paths",
                                table.name
                            ));
                        }
                        has_local_paths = true;
                    }
                    Err(err) => errors.push(err.to_string()),
                }
            }
            for col in &table.columns {
                if col.name.trim().is_empty() {
                    errors.push(format!("table {} has empty column name", table.name));
                    continue;
                }
                if !col_names.insert(col.name.as_str()) {
                    errors.push(format!("table {} has duplicate column {}", table.name, col.name));
                }
                if col.fixed_len.is_some()
                    && !matches!(col.ingested_dtype, IngestedType::String | IngestedType::Bytes)
                {
                    errors.push(format!(
                        "table {} column {} fixed_len only valid for string/bytes",
                        table.name, col.name
                    ));
                }
                let is_decimal = matches!(
                    col.ingested_dtype,
                    IngestedType::Decimal128 | IngestedType::Decimal256
                );
                if is_decimal {
                    let precision = match col.precision {
                        Some(v) => v,
                        None => {
                            errors.push(format!(
                                "table {} decimal column {} missing precision",
                                table.name, col.name
                            ));
                            continue;
                        }
                    };
                    let scale = match col.scale {
                        Some(v) => v,
                        None => {
                            errors.push(format!(
                                "table {} decimal column {} missing scale",
                                table.name, col.name
                            ));
                            continue;
                        }
                    };
                    let max_precision = match col.ingested_dtype {
                        IngestedType::Decimal128 => 38,
                        IngestedType::Decimal256 => 76,
                        _ => 0,
                    };
                    if precision == 0 || precision > max_precision {
                        errors.push(format!(
                            "table {} decimal column {} precision out of range",
                            table.name, col.name
                        ));
                    }
                    if scale.unsigned_abs() > precision {
                        errors.push(format!(
                            "table {} decimal column {} scale out of range",
                            table.name, col.name
                        ));
                    }
                    if scale < i8::MIN as i32 || scale > i8::MAX as i32 {
                        errors.push(format!(
                            "table {} decimal column {} scale out of range",
                            table.name, col.name
                        ));
                    }
                } else if (col.precision.is_some() || col.scale.is_some())
                    && !source_dtype_is_decimal(col.source_dtype.as_deref())
                {
                    errors.push(format!(
                        "table {} column {} precision/scale only valid for decimal",
                        table.name, col.name
                    ));
                }
            }
            if let Some(pk) = &table.primary_key {
                if !col_names.contains(pk.as_str()) {
                    errors.push(format!("table {} primary_key {} not found", table.name, pk));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *pk) {
                if !matches!(
                    col.ingested_dtype,
                    IngestedType::Int8
                        | IngestedType::Int16
                        | IngestedType::Int64
                        | IngestedType::Int32
                        | IngestedType::UInt8
                        | IngestedType::UInt16
                        | IngestedType::UInt64
                        | IngestedType::UInt32
                        | IngestedType::Float32
                        | IngestedType::Float64
                        | IngestedType::Decimal128
                        | IngestedType::Decimal256
                        | IngestedType::String
                        | IngestedType::Bytes
                        | IngestedType::Uuid
                        | IngestedType::Uuid256
                        | IngestedType::Ip
                ) {
                    errors.push(format!(
                        "table {} primary_key {} must be numeric or string/bytes/uuid/uuid256/ip",
                        table.name, pk
                    ));
                }
                }
                if let Some(bits) = table.fingerprint_bits {
                    if !matches!(bits, 64 | 32 | 16 | 8 | 0) {
                        errors.push(format!(
                            "table {} fingerprint_bits must be one of 64, 32, 16, 8, 0",
                            table.name
                        ));
                    }
                }
            } else if table.fingerprint_bits.is_some() {
                errors.push(format!(
                    "table {} fingerprint_bits requires a primary_key",
                    table.name
                ));
            }
            if let Some(weight) = &table.weight {
                if !col_names.contains(weight.as_str()) {
                    errors.push(format!("table {} weight column {} not found", table.name, weight));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *weight) {
                    if !matches!(
                        col.ingested_dtype,
                        IngestedType::Int8
                            | IngestedType::Int16
                            | IngestedType::Int64
                            | IngestedType::Int32
                            | IngestedType::UInt8
                            | IngestedType::UInt16
                            | IngestedType::UInt64
                            | IngestedType::UInt32
                            | IngestedType::Float64
                            | IngestedType::Float32
                    ) {
                        errors.push(format!(
                            "table {} weight column {} must be numeric",
                            table.name, weight
                        ));
                    }
                }
            }
            if let Some(ts) = &table.create_time {
                if !col_names.contains(ts.as_str()) {
                    errors.push(format!(
                        "table {} create_time column {} not found",
                        table.name, ts
                    ));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *ts) {
                    if !matches!(col.ingested_dtype, IngestedType::Date | IngestedType::Timestamp) {
                        errors.push(format!(
                            "table {} create_time column {} must be date/timestamp",
                            table.name, ts
                        ));
                    }
                }
            }
            if let Some(edge) = &table.edge_table {
                for fk in [&edge.src_fk, &edge.dst_fk] {
                    if !col_names.contains(fk.column.as_str()) {
                        errors.push(format!(
                            "table {} edge fk column {} not found",
                            table.name, fk.column
                        ));
                    }
                }
            }
            table_map.insert(table.name.as_str(), table);
        }

        for rel in &self.relationships {
            let src = match table_map.get(rel.src_table.as_str()) {
                Some(table) => *table,
                None => {
                    errors.push(format!("relationship src table {} not found", rel.src_table));
                    continue;
                }
            };
            let dst = match table_map.get(rel.dst_table.as_str()) {
                Some(table) => *table,
                None => {
                    errors.push(format!("relationship dst table {} not found", rel.dst_table));
                    continue;
                }
            };
            let src_col = src
                .columns
                .iter()
                .find(|c| c.name == rel.src_column);
            let dst_col = dst
                .columns
                .iter()
                .find(|c| c.name == rel.dst_column);
            if src_col.is_none() {
                errors.push(format!(
                    "relationship src column {} not found in {}",
                    rel.src_column, rel.src_table
                ));
            }
            if dst_col.is_none() {
                errors.push(format!(
                    "relationship dst column {} not found in {}",
                    rel.dst_column, rel.dst_table
                ));
            }
            if let (Some(src_col), Some(dst_col)) = (src_col, dst_col) {
                let src_is_pk = src.primary_key.as_deref() == Some(&src_col.name);
                let dst_is_pk = dst.primary_key.as_deref() == Some(&dst_col.name);
                if !src_is_pk && !dst_is_pk {
                    errors.push(format!(
                        "relationship {} -> {} must reference at least one primary key",
                        rel.src_table, rel.dst_table
                    ));
                }
                if let Err(msg) = compatible_key_columns(src_col, dst_col) {
                    errors.push(format!(
                        "relationship {}.{} and {}.{} {}",
                        rel.src_table,
                        rel.src_column,
                        rel.dst_table,
                        rel.dst_column,
                        msg
                    ));
                }
                let key = format!(
                    "{}:{}->{}:{}",
                    rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
                );
                if !rel_keys.insert(key.clone()) {
                    errors.push(format!(
                        "relationship {}.{} -> {}.{} is duplicated",
                        rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
                    ));
                }
                if src_is_pk && dst_is_pk {
                    let reverse = format!(
                        "{}:{}->{}:{}",
                        rel.dst_table, rel.dst_column, rel.src_table, rel.src_column
                    );
                    if rel_keys.contains(&reverse) {
                        errors.push(format!(
                            "relationship {}.{} -> {}.{} has reverse already declared",
                            rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
                        ));
                    }
                }
            }
        }

        for table in &self.tables {
            if let Some(edge) = &table.edge_table {
                for fk in [&edge.src_fk, &edge.dst_fk] {
                    let ref_table = match table_map.get(fk.ref_table.as_str()) {
                        Some(table) => *table,
                        None => {
                            errors.push(format!(
                                "edge table {} references missing table {}",
                                table.name, fk.ref_table
                            ));
                            continue;
                        }
                    };
                    if ref_table.primary_key.as_deref() != Some(fk.ref_column.as_str()) {
                        errors.push(format!(
                            "edge table {} fk {} must reference primary key {}.{}",
                            table.name, fk.column, fk.ref_table, fk.ref_column
                        ));
                        continue;
                    }
                    if let Some(fk_col) = table.columns.iter().find(|c| c.name == fk.column) {
                        if let Some(pk_col) = ref_table.columns.iter().find(|c| c.name == fk.ref_column) {
                            if let Err(msg) = compatible_key_columns(fk_col, pk_col) {
                                errors.push(format!(
                                    "edge table {} fk {} and {}.{} {}",
                                    table.name,
                                    fk.column,
                                    fk.ref_table,
                                    fk.ref_column,
                                    msg
                                ));
                            }
                        }
                    }
                }
            }
        }

        if let Some(sort_keys) = &self.edge_sort_keys {
            for (rel_name, cfg) in &sort_keys.relationships {
                let rel = match self.relationships.iter().find(|r| relationship_name(r) == *rel_name) {
                    Some(rel) => rel,
                    None => {
                        errors.push(format!("sort key relationship {} not found", rel_name));
                        continue;
                    }
                };
                if let Some(spec) = &cfg.forward {
                    validate_sort_key_spec(
                        spec,
                        rel_name,
                        &rel.dst_table,
                        None,
                        &table_map,
                        &mut errors,
                    );
                }
                if let Some(spec) = &cfg.reverse {
                    validate_sort_key_spec(
                        spec,
                        rel_name,
                        &rel.src_table,
                        None,
                        &table_map,
                        &mut errors,
                    );
                }
            }
            for (table_name, cfg) in &sort_keys.edge_tables {
                let edge_table = match table_map.get(table_name.as_str()) {
                    Some(table) => *table,
                    None => {
                        errors.push(format!("sort key edge table {} not found", table_name));
                        continue;
                    }
                };
                let edge = match &edge_table.edge_table {
                    Some(edge) => edge,
                    None => {
                        errors.push(format!(
                            "sort key edge table {} missing edge_table spec",
                            table_name
                        ));
                        continue;
                    }
                };
                if let Some(spec) = &cfg.src_to_dst {
                    validate_sort_key_spec(
                        spec,
                        table_name,
                        &edge.dst_fk.ref_table,
                        Some(table_name.as_str()),
                        &table_map,
                        &mut errors,
                    );
                }
                if let Some(spec) = &cfg.dst_to_src {
                    validate_sort_key_spec(
                        spec,
                        table_name,
                        &edge.src_fk.ref_table,
                        Some(table_name.as_str()),
                        &table_map,
                        &mut errors,
                    );
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(errors.join("\n")))
        }
    }
}

fn compatible_key_columns(a: &ColumnSpec, b: &ColumnSpec) -> Result<(), String> {
    use IngestedType::*;
    match (&a.ingested_dtype, &b.ingested_dtype) {
        (String, String) | (Bytes, Bytes) | (String, Bytes) | (Bytes, String) => Ok(()),
        (Uuid, Uuid) | (Uuid256, Uuid256) | (Ip, Ip) => Ok(()),
        (a_type, b_type) if is_numeric(a_type) && is_numeric(b_type) => Ok(()),
        (a_type, b_type) if is_numeric(a_type) && matches!(b_type, String | Bytes) => Ok(()),
        (a_type, b_type) if is_numeric(b_type) && matches!(a_type, String | Bytes) => Ok(()),
        _ => Err("have incompatible types".to_string()),
    }
}

fn column_spec<'a>(table: &'a TableSpec, name: &str) -> Result<&'a ColumnSpec> {
    table
        .columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| anyhow!("column {} not found in {}", name, table.name))
}

fn validate_sort_key_spec(
    spec: &SortKeySpec,
    rel_name: &str,
    dst_table: &str,
    edge_table: Option<&str>,
    table_map: &HashMap<&str, &TableSpec>,
    errors: &mut Vec<String>,
) {
    if spec.timestamp.is_none() && spec.weight.is_none() {
        errors.push(format!(
            "sort key {} requires at least one of timestamp or weight",
            rel_name
        ));
        return;
    }
    let mut width_bits = 0u16;
    if let Some(ts) = &spec.timestamp {
        let table = match table_map.get(ts.table.as_str()) {
            Some(table) => *table,
            None => {
                errors.push(format!(
                    "sort key {} timestamp table {} not found",
                    rel_name, ts.table
                ));
                return;
            }
        };
        if !matches!(edge_table, Some(name) if name == ts.table) && ts.table != dst_table {
            errors.push(format!(
                "sort key {} timestamp table {} invalid for destination {}",
                rel_name, ts.table, dst_table
            ));
        }
        match column_spec(table, &ts.column) {
            Ok(col) => {
                if !matches!(col.ingested_dtype, IngestedType::Date | IngestedType::Timestamp) {
                    errors.push(format!(
                        "sort key {} timestamp column {} must be date/timestamp",
                        rel_name, ts.column
                    ));
                }
            }
            Err(_) => {
                errors.push(format!(
                    "sort key {} timestamp column {} not found in {}",
                    rel_name, ts.column, ts.table
                ));
            }
        }
        if ts.encoding.step <= 0 {
            errors.push(format!(
                "sort key {} timestamp step must be > 0",
                rel_name
            ));
        }
        width_bits += match ts.encoding.kind {
            TimestampEncodingKind::Int8 => 8,
            TimestampEncodingKind::Int16 => 16,
            TimestampEncodingKind::Int32 => 32,
        };
    }
    if let Some(weight) = &spec.weight {
        let table = match table_map.get(weight.table.as_str()) {
            Some(table) => *table,
            None => {
                errors.push(format!(
                    "sort key {} weight table {} not found",
                    rel_name, weight.table
                ));
                return;
            }
        };
        if !matches!(edge_table, Some(name) if name == weight.table) && weight.table != dst_table {
            errors.push(format!(
                "sort key {} weight table {} invalid for destination {}",
                rel_name, weight.table, dst_table
            ));
        }
        if let Err(err) = validate_weight_expr(&weight.expr, table) {
            errors.push(format!(
                "sort key {} weight expr invalid: {}",
                rel_name, err
            ));
        }
        width_bits += match weight.encoding.dtype {
            WeightEncodingKind::Byte => 8,
            WeightEncodingKind::UInt16 => 16,
            WeightEncodingKind::UInt32 => 32,
            WeightEncodingKind::Float16 => 16,
            WeightEncodingKind::Float32 => 32,
        };
    }
    if width_bits > 64 {
        errors.push(format!(
            "sort key {} width {} exceeds 64 bits",
            rel_name, width_bits
        ));
    } else if width_bits > 0 && width_bits % 8 != 0 {
        errors.push(format!(
            "sort key {} width {} must be multiple of 8",
            rel_name, width_bits
        ));
    }
}

fn validate_weight_expr(expr: &str, table: &TableSpec) -> Result<()> {
    let (rewritten, var_map) = rewrite_weight_expr(expr)?;
    let parser = Parser::new();
    let mut slab = Slab::new();
    let expr_i = parser
        .parse(&rewritten, &mut slab.ps)
        .map_err(|e| anyhow!("parse error: {e}"))?;
    let expr_ref = expr_i.from(&slab.ps);
    let vars = expr_ref.var_names(&slab);
    if vars.is_empty() {
        return Err(anyhow!("expression has no column references"));
    }
    for var in vars {
        let column = var_map.get(&var).cloned().unwrap_or_else(|| var.clone());
        let col = column_spec(table, &column)?;
        if !is_numeric(&col.ingested_dtype) {
            return Err(anyhow!("column {} is not numeric", column));
        }
    }
    Ok(())
}

fn rewrite_weight_expr(expr: &str) -> Result<(String, HashMap<String, String>)> {
    let mut out = String::new();
    let mut map: HashMap<String, String> = HashMap::new();
    let mut chars = expr.chars().peekable();
    let mut idx = 0usize;
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut name = String::new();
            let mut escaped = false;
            while let Some(c) = chars.next() {
                if escaped {
                    match c {
                        'n' => name.push('\n'),
                        'r' => name.push('\r'),
                        't' => name.push('\t'),
                        '\\' => name.push('\\'),
                        '`' => name.push('`'),
                        other => name.push(other),
                    }
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '`' {
                    break;
                }
                name.push(c);
            }
            if escaped {
                return Err(anyhow!("unterminated escape in `{}`", expr));
            }
            if name.is_empty() {
                return Err(anyhow!("empty backtick identifier"));
            }
            let var = format!("__c{idx}");
            idx += 1;
            out.push_str(&var);
            map.insert(var, name);
        } else {
            out.push(ch);
        }
    }
    Ok((out, map))
}

fn is_signed_int(t: &IngestedType) -> bool {
    matches!(
        t,
        IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64
    )
}

fn is_unsigned_int(t: &IngestedType) -> bool {
    matches!(
        t,
        IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64
    )
}

fn is_numeric(t: &IngestedType) -> bool {
    is_signed_int(t)
        || is_unsigned_int(t)
        || matches!(
            t,
            IngestedType::Float32
                | IngestedType::Float64
                | IngestedType::Decimal128
                | IngestedType::Decimal256
        )
}
