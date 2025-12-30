use anyhow::{anyhow, Result};
use diskgraph_rowpack::RowPackConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub dataset_name: String,
    pub tables: Vec<TableSpec>,
    #[serde(default)]
    pub relationships: Vec<RelationshipSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSpec {
    pub name: String,
    pub format: TableFormat,
    pub paths: Vec<String>,
    pub columns: Vec<ColumnSpec>,
    #[serde(default)]
    pub primary_key: Option<String>,
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
    pub dtype: ColumnType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnType {
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
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float32Array,
    Float64Array,
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

impl SourceMetadata {
    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        if self.dataset_name.trim().is_empty() {
            errors.push("dataset_name is required".to_string());
        }
        if self.tables.is_empty() {
            errors.push("at least one table is required".to_string());
        }
        let mut table_names = HashSet::new();
        let mut table_map: HashMap<&str, &TableSpec> = HashMap::new();
        let mut rel_keys = HashSet::new();
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
            for col in &table.columns {
                if col.name.trim().is_empty() {
                    errors.push(format!("table {} has empty column name", table.name));
                    continue;
                }
                if !col_names.insert(col.name.as_str()) {
                    errors.push(format!("table {} has duplicate column {}", table.name, col.name));
                }
                if matches!(col.dtype, ColumnType::String | ColumnType::Bytes) {
                    if col.fixed_len.is_none()
                        && Some(col.name.as_str()) == table.primary_key.as_deref()
                    {
                        errors.push(format!(
                            "table {} primary key {} requires fixed_len",
                            table.name, col.name
                        ));
                    }
                }
                let is_decimal = matches!(col.dtype, ColumnType::Decimal128 | ColumnType::Decimal256);
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
                    let max_precision = match col.dtype {
                        ColumnType::Decimal128 => 38,
                        ColumnType::Decimal256 => 76,
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
                } else if col.precision.is_some() || col.scale.is_some() {
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
                    col.dtype,
                    ColumnType::Int8
                        | ColumnType::Int16
                        | ColumnType::Int64
                        | ColumnType::Int32
                        | ColumnType::UInt8
                        | ColumnType::UInt16
                        | ColumnType::UInt64
                        | ColumnType::UInt32
                        | ColumnType::String
                        | ColumnType::Bytes
                ) {
                        errors.push(format!(
                            "table {} primary_key {} must be int or string/bytes",
                            table.name, pk
                        ));
                    }
                }
            }
            if let Some(weight) = &table.weight {
                if !col_names.contains(weight.as_str()) {
                    errors.push(format!("table {} weight column {} not found", table.name, weight));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *weight) {
                    if !matches!(
                        col.dtype,
                        ColumnType::Int8
                            | ColumnType::Int16
                            | ColumnType::Int64
                            | ColumnType::Int32
                            | ColumnType::UInt8
                            | ColumnType::UInt16
                            | ColumnType::UInt64
                            | ColumnType::UInt32
                            | ColumnType::Float64
                            | ColumnType::Float32
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
                    if !matches!(col.dtype, ColumnType::Date | ColumnType::Timestamp) {
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
                if !compatible_key_types(&src_col.dtype, &dst_col.dtype) {
                    errors.push(format!(
                        "relationship {}.{} and {}.{} have incompatible types",
                        rel.src_table,
                        rel.src_column,
                        rel.dst_table,
                        rel.dst_column
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
                    }
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

fn compatible_key_types(a: &ColumnType, b: &ColumnType) -> bool {
    use ColumnType::*;
    match (a, b) {
        (String, String) | (Bytes, Bytes) => true,
        (a, b) if is_signed_int(a) && is_signed_int(b) => true,
        (a, b) if is_unsigned_int(a) && is_unsigned_int(b) => true,
        _ => false,
    }
}

fn is_signed_int(t: &ColumnType) -> bool {
    matches!(t, ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64)
}

fn is_unsigned_int(t: &ColumnType) -> bool {
    matches!(t, ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64)
}
