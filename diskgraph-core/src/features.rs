use crate::dataset::DatasetId;
use serde::{Deserialize, Serialize};

/// Request to fetch features for a batch of nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFeatureRequest {
    /// The dataset identifier.
    pub dataset_id: DatasetId,
    /// The node type (table name) to fetch features for.
    pub node_type: String,
    /// The batch of global node IDs to fetch.
    pub node_ids: Vec<u64>,
    /// Optional: List of column names to fetch. If None, fetch all available features.
    pub feature_projection: Option<Vec<String>>,
}