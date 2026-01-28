//! Configuration for the explanation pipeline.
//!
//! All configuration is task-agnostic and works for classification, regression, etc.

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

/// Configuration for LLM prompt generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Description of the prediction task (e.g., "probability of no purchase in next 90 days")
    pub task_description: String,

    /// Graph schema description (e.g., "customers -> transactions -> articles")
    /// If empty, will be auto-inferred from relation metadata
    #[serde(default)]
    pub graph_description: String,

    /// Domain context (e.g., "retail fashion", "financial services")
    #[serde(default = "default_domain")]
    pub domain: String,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            task_description: "prediction target".to_string(),
            graph_description: String::new(),
            domain: "general".to_string(),
        }
    }
}

fn default_domain() -> String {
    "general".to_string()
}

/// Main configuration for the explanation pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainConfig {
    // ========== Input/Output Paths ==========
    pub input_parquet: PathBuf,
    pub output_parquet: PathBuf,
    pub index_path: PathBuf,

    // ========== Schema Configuration ==========
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

    // ========== Subgraph Extraction ==========
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,

    #[serde(default = "default_fanout_per_hop")]
    pub fanout_per_hop: Vec<usize>,

    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,

    #[serde(default = "default_skip_unbounded_relations")]
    pub skip_unbounded_relations: bool,

    // ========== Motif Configuration ==========
    #[serde(default)]
    pub motif_config: MotifConfig,

    // ========== Clustering ==========
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,

    #[serde(default = "default_kmeans_iterations")]
    pub kmeans_iterations: usize,

    #[serde(default = "default_min_cluster_fit")]
    pub min_cluster_fit: f32,

    // ========== Population Statistics ==========
    #[serde(default = "default_num_deciles")]
    pub num_deciles: usize,

    // ========== Output Control ==========
    #[serde(default = "default_top_drivers_k")]
    pub top_drivers_k: usize,

    #[serde(default = "default_include_cluster_context")]
    pub include_cluster_context: bool,

    #[serde(default = "default_include_homophily")]
    pub include_homophily: bool,

    // ========== Narrative Relevance Filtering ==========
    /// Z-score threshold for global outlier (|z| > threshold)
    #[serde(default = "default_z_threshold")]
    pub z_threshold: f64,

    /// Conditional rate ratio threshold for predictive signal (ratio > threshold or < 1/threshold)
    #[serde(default = "default_rate_ratio_threshold")]
    pub rate_ratio_threshold: f64,

    /// Number of entities to sample per cluster for archetype profiling
    #[serde(default = "default_archetype_sample_size")]
    pub archetype_sample_size: usize,

    // ========== Prompt Generation ==========
    /// Configuration for LLM prompt generation (optional)
    #[serde(default)]
    pub prompt_config: Option<PromptConfig>,
}

/// Configuration for motif computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifConfig {
    // ========== Structural Motifs ==========
    #[serde(default = "default_true")]
    pub compute_edge_counts: bool,

    #[serde(default = "default_true")]
    pub compute_unique_destinations: bool,

    // ========== Temporal Motifs ==========
    #[serde(default = "default_true")]
    pub compute_recency: bool,

    #[serde(default = "default_true")]
    pub compute_frequency: bool,

    #[serde(default = "default_true")]
    pub compute_trend: bool,

    #[serde(default = "default_true")]
    pub compute_inter_arrival: bool,

    #[serde(default = "default_min_edges_for_trend")]
    pub min_edges_for_trend: usize,

    // ========== Attribute Motifs ==========
    #[serde(default = "default_numeric_aggregations")]
    pub numeric_aggregations: Vec<NumericAggregation>,

    #[serde(default = "default_categorical_aggregations")]
    pub categorical_aggregations: Vec<CategoricalAggregation>,

    #[serde(default = "default_max_categorical_cardinality")]
    pub max_categorical_cardinality: usize,

    // ========== Homophily ==========
    #[serde(default = "default_homophily_hop")]
    pub homophily_hop: usize,

    // ========== Column Selection ==========
    /// Relations to include (empty = all)
    #[serde(default)]
    pub include_relations: Vec<String>,

    /// Relations to exclude
    #[serde(default)]
    pub exclude_relations: Vec<String>,

    /// Columns to include (empty = auto-detect)
    #[serde(default)]
    pub include_columns: Vec<ColumnPath>,

    /// Columns to exclude
    #[serde(default)]
    pub exclude_columns: Vec<ColumnPath>,

    /// Columns to force as categorical (even if stored as numeric)
    /// Use for ID columns like "colour_group_code", "department_no", etc.
    #[serde(default)]
    pub force_categorical: Vec<String>,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            compute_edge_counts: true,
            compute_unique_destinations: true,
            compute_recency: true,
            compute_frequency: true,
            compute_trend: true,
            compute_inter_arrival: true,
            min_edges_for_trend: 3,
            numeric_aggregations: default_numeric_aggregations(),
            categorical_aggregations: default_categorical_aggregations(),
            max_categorical_cardinality: 100,
            homophily_hop: 2,
            include_relations: Vec::new(),
            exclude_relations: Vec::new(),
            include_columns: Vec::new(),
            exclude_columns: Vec::new(),
            force_categorical: Vec::new(),
        }
    }
}

/// Numeric aggregation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NumericAggregation {
    Sum,
    Mean,
    Std,
    Min,
    Max,
    Count,
}

/// Categorical aggregation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CategoricalAggregation {
    Mode,
    Nunique,
}

/// Path to a column (table.column)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ColumnPath {
    pub table: String,
    pub column: String,
}

// ========== Default Functions ==========

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
    IdKind::Utf8
}

fn default_id_lookup() -> IdLookup {
    IdLookup::PrimaryKey
}

fn default_timestamp_kind() -> TimestampKind {
    TimestampKind::Int64
}

fn default_max_hops() -> usize {
    2
}

fn default_fanout_per_hop() -> Vec<usize> {
    vec![50, 20]
}

fn default_max_nodes() -> usize {
    1000
}

fn default_skip_unbounded_relations() -> bool {
    true
}

fn default_num_clusters() -> usize {
    32
}

fn default_kmeans_iterations() -> usize {
    10
}

fn default_min_cluster_fit() -> f32 {
    0.2
}

fn default_num_deciles() -> usize {
    10
}

fn default_top_drivers_k() -> usize {
    10
}

fn default_include_cluster_context() -> bool {
    true
}

fn default_include_homophily() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_min_edges_for_trend() -> usize {
    3
}

fn default_numeric_aggregations() -> Vec<NumericAggregation> {
    vec![
        NumericAggregation::Sum,
        NumericAggregation::Mean,
        NumericAggregation::Std,
        NumericAggregation::Min,
        NumericAggregation::Max,
    ]
}

fn default_categorical_aggregations() -> Vec<CategoricalAggregation> {
    vec![
        CategoricalAggregation::Mode,
        CategoricalAggregation::Nunique,
    ]
}

fn default_max_categorical_cardinality() -> usize {
    100
}

fn default_homophily_hop() -> usize {
    2
}

fn default_z_threshold() -> f64 {
    2.0
}

fn default_rate_ratio_threshold() -> f64 {
    2.0
}

fn default_archetype_sample_size() -> usize {
    100
}
