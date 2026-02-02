//! Kumo GraphConfig JSON structures for deserialization.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level Kumo graph configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoGraphConfig {
    /// Graph name (optional).
    pub name: Option<String>,
    /// Data source configurations (connector configs).
    #[serde(default)]
    pub data_source_dict: HashMap<String, serde_json::Value>,
    /// List of tables in the graph.
    #[serde(default)]
    pub tables: Vec<KumoTable>,
}

/// Kumo table configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoTable {
    /// Table name.
    pub table_name: String,
    /// Data source/connector name.
    pub data_source: Option<String>,
    /// Source name in the connector (may differ from table_name).
    pub source_name: Option<String>,
    /// Data version identifier.
    pub data_version: Option<String>,
    /// Primary key configuration.
    pub pkey: Option<KumoPrimaryKey>,
    /// Foreign key configurations.
    #[serde(default)]
    pub fkeys: Vec<KumoForeignKey>,
    /// Regular column configurations.
    #[serde(default)]
    pub cols: Vec<KumoColumn>,
    /// Time column (for temporal ordering).
    pub time_col: Option<KumoTimeCol>,
    /// End time column (for validity windows).
    pub end_time_col: Option<KumoTimeCol>,
}

/// Kumo primary key configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoPrimaryKey {
    /// Column name.
    pub name: String,
    /// Data type (e.g., "int64", "string").
    pub dtype: Option<String>,
    /// Semantic type.
    pub stype: Option<String>,
    /// Whether to drop duplicates.
    #[serde(default)]
    pub drop_duplicates: bool,
}

/// Kumo foreign key configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoForeignKey {
    /// Column name in this table.
    pub name: String,
    /// Data type.
    pub dtype: Option<String>,
    /// Semantic type.
    pub stype: Option<String>,
    /// Destination table name (may include data_source prefix).
    pub dst_table: String,
    /// Destination key column name.
    pub dst_key: String,
    /// Destination data source (optional).
    pub dst_data_source: Option<String>,
}

/// Kumo column configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoColumn {
    /// Column name.
    pub name: String,
    /// Data type.
    pub dtype: Option<String>,
    /// Semantic type.
    pub stype: Option<String>,
    /// Timestamp unit for epoch-based timestamps (e.g., "s", "ms", "us", "ns").
    pub timestamp_unit: Option<String>,
}

/// Kumo time column configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KumoTimeCol {
    /// Column name.
    pub name: String,
    /// Data type.
    pub dtype: Option<String>,
    /// Semantic type.
    pub stype: Option<String>,
    /// Timestamp unit.
    pub timestamp_unit: Option<String>,
    /// Whether this is a pseudo time column.
    #[serde(default)]
    pub is_pseudo: bool,
}
