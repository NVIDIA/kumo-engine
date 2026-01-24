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
    #[serde(default = "default_timestamp_kind")]
    pub timestamp_kind: TimestampKind,
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
    #[serde(default = "default_kmeans_passes")]
    pub kmeans_passes: usize,
    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,
    #[serde(default = "default_knn_k")]
    pub knn_k: usize,
    #[serde(default = "default_cluster_sample_size")]
    pub cluster_sample_size: usize,
    #[serde(default = "default_knn_sample_size")]
    pub knn_sample_size: usize,
    #[serde(default = "default_ridge_lambda")]
    pub ridge_lambda: f64,
    #[serde(default = "default_top_factors")]
    pub top_factors: usize,
    #[serde(default = "default_time_bucket_secs")]
    pub time_bucket_secs: i64,
    #[serde(default = "default_max_time_buckets")]
    pub max_time_buckets: usize,
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

fn default_timestamp_kind() -> TimestampKind {
    TimestampKind::Int64
}

fn default_num_clusters() -> usize {
    64
}

fn default_kmeans_passes() -> usize {
    2
}

fn default_minibatch_size() -> usize {
    2048
}

fn default_knn_k() -> usize {
    16
}

fn default_cluster_sample_size() -> usize {
    1024
}

fn default_knn_sample_size() -> usize {
    2048
}

fn default_ridge_lambda() -> f64 {
    1e-3
}

fn default_top_factors() -> usize {
    5
}

fn default_time_bucket_secs() -> i64 {
    86_400
}

fn default_max_time_buckets() -> usize {
    30
}
