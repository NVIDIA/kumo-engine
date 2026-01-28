//! Output structures for entity explanations.
//!
//! This is the final format that gets serialized to parquet/JSON for LLM consumption.

use crate::motifs::EntityMotifs;
use crate::population::FeatureContext;
use serde::{Deserialize, Serialize};

/// Complete explanation for a single entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExplanation {
    /// Entity identifier
    pub entity_id: String,

    /// Model prediction value
    pub prediction: f64,

    /// Raw motifs computed from subgraph
    pub motifs: EntityMotifs,

    /// Feature contexts with population statistics
    pub feature_contexts: Vec<FeatureContext>,

    /// Top drivers (features most associated with prediction)
    pub top_drivers: Vec<FeatureContext>,

    /// Cluster context (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster: Option<ClusterContext>,

    /// Population summary
    pub population: PopulationSummary,
}

/// Cluster context for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterContext {
    /// Cluster ID
    pub cluster_id: usize,

    /// Number of entities in cluster
    pub cluster_size: usize,

    /// Mean prediction in cluster
    pub cluster_mean_prediction: f64,

    /// Std of predictions in cluster
    pub cluster_std_prediction: f64,

    /// How well entity fits cluster profile (0-1)
    pub fit_score: f32,

    /// Cluster profile: features that characterize this cluster
    pub profile: Vec<ClusterProfileFeature>,

    /// Features where entity matches cluster profile
    pub entity_matches: Vec<String>,

    /// Features where entity differs from cluster profile
    pub entity_differs: Vec<String>,
}

/// Feature in cluster profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterProfileFeature {
    pub feature: String,
    /// Mean value of this feature for cluster archetype (sampled from nearest to centroid)
    pub cluster_mean: f64,
    /// Std of this feature for cluster archetype
    pub cluster_std: f64,
    /// Z-score of cluster mean vs population (how distinctive is this cluster on this feature)
    pub cluster_z_score: f64,
    /// Conditional rate for this feature's typical decile in cluster
    pub conditional_rate: f64,
}

/// Population summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSummary {
    pub total_entities: usize,
    pub mean_prediction: f64,
    pub std_prediction: f64,
    pub min_prediction: f64,
    pub max_prediction: f64,
}

impl EntityExplanation {
    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Convert to pretty JSON string
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
