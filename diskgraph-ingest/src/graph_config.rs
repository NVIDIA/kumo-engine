use anyhow::{anyhow, Result};
use diskgraph_core::metadata::EdgeStorageFormat;
use diskgraph_rowpack::RowPackConfig;
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
pub struct GraphConfig {
    pub dataset_name: String,
    pub tables: Vec<TableSpec>,
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
    /// Table format (csv or parquet) - used with `paths` field.
    pub format: TableFormat,
    /// File paths - used with `format` field. Prefer using `source` instead.
    pub paths: Vec<String>,
    pub columns: Vec<ColumnSpec>,
    #[serde(default)]
    pub primary_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    /// End time column (for temporal graphs with validity windows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub edge_table: Option<EdgeTableSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
    /// Data source configuration (new - preferred over format+paths).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<DataSource>,
    /// Foreign key relationships (new - edges embedded in table).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fkeys: Vec<FKeySpec>,
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

/// Data source configuration - abstracts where table data comes from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSource {
    /// Parquet files (local or cloud storage).
    Parquet {
        /// List of file paths (local, s3://, gs://, az://).
        paths: Vec<String>,
    },
    /// Snowflake data warehouse.
    Snowflake {
        /// Connection identifier referencing a connection config.
        connection_id: String,
        /// Table or view name in Snowflake.
        table_or_view: String,
    },
    /// Databricks.
    Databricks {
        /// Connection identifier.
        connection_id: String,
        /// Catalog name.
        catalog: String,
        /// Schema name.
        schema: String,
        /// Table name.
        table: String,
    },
    /// Google BigQuery.
    BigQuery {
        /// Connection identifier referencing a connection config.
        connection_id: String,
        /// GCP project ID.
        project: String,
        /// BigQuery dataset name.
        dataset: String,
        /// Table name.
        table: String,
    },
}

/// Time-based sort key specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeSpec {
    /// The table containing the timestamp column.
    pub table: String,
    /// The timestamp column name.
    pub column: String,
    /// Bit width for encoding: 8, 16, or 32. Null for auto-detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitwidth: Option<u8>,
    /// Anchor time in epoch seconds. Null for auto-detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_time: Option<i64>,
    /// Time unit: "s", "ms", "day", "hour", etc.
    pub unit: String,
}

/// Weight-based sort key specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightSpec {
    /// The table containing columns referenced in the expression.
    pub table: String,
    /// Weight expression - column name or math expression.
    pub expr: String,
    /// Encoding dtype: "byte", "uint16", "uint32", "float16", "float32"
    pub dtype: String,
}

/// Direction-specific configuration for edge traversal.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DirectionSpec {
    /// Time-based sort key for this direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<TimeSpec>,
    /// Weight-based sort key for this direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<WeightSpec>,
}

/// Foreign key specification defining a 1:N relationship.
/// Defines an edge from this table to another table's primary key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FKeySpec {
    /// Column on this table that references the destination table.
    pub src_column: String,
    /// Destination table name. Must have a primary key.
    pub dst_table: String,
    /// Forward direction config (src -> dst).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<DirectionSpec>,
    /// Reverse direction config (dst -> src).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse: Option<DirectionSpec>,
}

/// Endpoint specification for an edge table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeEndpoint {
    /// Column in the edge table.
    pub column: String,
    /// Referenced node table name. Must have a primary key.
    pub table: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTableSpec {
    /// Source foreign key.
    pub src_fk: ForeignKeySpec,
    /// Destination foreign key.
    pub dst_fk: ForeignKeySpec,
    /// Forward direction config (src -> dst).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<DirectionSpec>,
    /// Reverse direction config (dst -> src).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse: Option<DirectionSpec>,
}

/// Foreign key specification (old format - for backward compatibility).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForeignKeySpec {
    pub column: String,
    pub ref_table: String,
    pub ref_column: String,
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

impl GraphConfig {
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
