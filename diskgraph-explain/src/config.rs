use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdKind {
    Int64,
    Utf8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdLookup {
    NodeId,
    PrimaryKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampKind {
    Int64,
    TimestampMillis,
    TimestampMicros,
    TimestampNanos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainConfig {
    pub input_parquet: PathBuf,
    pub output_explanations_parquet: PathBuf,
    pub output_global_trends_parquet: PathBuf,
    pub index_path: PathBuf,
    pub root_table: String,
    #[serde(default = "default_id_column")]
    pub id_column: String,
    #[serde(default = "default_timestamp_column")]
    pub timestamp_column: String,
    #[serde(default = "default_prediction_column")]
    pub prediction_column: String,
    #[serde(default = "default_embedding_column")]
    pub embedding_column: String,
    #[serde(default = "default_id_kind")]
    pub id_kind: IdKind,
    #[serde(default = "default_id_lookup")]
    pub id_lookup: IdLookup,
    #[serde(default = "default_timestamp_kind")]
    pub timestamp_kind: TimestampKind,
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
    #[serde(default = "default_kmeans_passes")]
    pub kmeans_passes: usize,
    #[serde(default = "default_time_bucket_secs")]
    pub time_bucket_secs: i64,
    #[serde(default = "default_max_time_buckets")]
    pub max_time_buckets: usize,
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
    #[serde(default = "default_relation_fanout")]
    pub relation_fanout: usize,
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    #[serde(default = "default_skip_unbounded_relations")]
    pub skip_unbounded_relations: bool,
    #[serde(default = "default_degree_buckets")]
    pub degree_buckets: Vec<u32>,
    #[serde(default = "default_feature_top_k")]
    pub feature_top_k: usize,
    #[serde(default = "default_event_top_k")]
    pub event_top_k: usize,
    #[serde(default = "default_smoothing_alpha")]
    pub smoothing_alpha: f32,
    #[serde(default = "default_min_support")]
    pub min_support: u32,
    #[serde(default)]
    pub attributes: Vec<TableAttributeConfig>,
}

fn default_id_column() -> String {
    "id".to_string()
}

fn default_timestamp_column() -> String {
    "timestamp".to_string()
}

fn default_prediction_column() -> String {
    "prediction".to_string()
}

fn default_embedding_column() -> String {
    "embedding".to_string()
}

fn default_id_kind() -> IdKind {
    IdKind::Int64
}

fn default_id_lookup() -> IdLookup {
    IdLookup::PrimaryKey
}

fn default_timestamp_kind() -> TimestampKind {
    TimestampKind::Int64
}

fn default_num_clusters() -> usize {
    64
}

fn default_kmeans_passes() -> usize {
    2
}

fn default_time_bucket_secs() -> i64 {
    86_400
}

fn default_max_time_buckets() -> usize {
    30
}

fn default_max_hops() -> usize {
    2
}

fn default_relation_fanout() -> usize {
    100
}

fn default_max_nodes() -> usize {
    2000
}

fn default_skip_unbounded_relations() -> bool {
    true
}

fn default_degree_buckets() -> Vec<u32> {
    vec![0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
}

fn default_feature_top_k() -> usize {
    8
}

fn default_event_top_k() -> usize {
    5
}

fn default_smoothing_alpha() -> f32 {
    1.0
}

fn default_min_support() -> u32 {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableAttributeConfig {
    pub table: String,
    pub columns: Vec<ColumnSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(default = "default_attr_kind")]
    pub kind: AttributeKind,
    #[serde(default)]
    pub buckets: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeKind {
    Categorical,
    Numeric,
    Boolean,
}

fn default_attr_kind() -> AttributeKind {
    AttributeKind::Categorical
}
