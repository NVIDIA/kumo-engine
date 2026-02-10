use crate::dataset::DatasetId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRequest {
    pub node_type: String,
    pub node_ids: Vec<u64>,
    pub feature_names: Vec<String>,
}

/// Request to fetch features for a batch of nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFeatureRequest {
    pub dataset_id: DatasetId,
    pub batch: Vec<FeatureRequest>,
}