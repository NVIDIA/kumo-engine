use crate::dataset::DatasetId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request to fetch features for a batch of nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFeatureRequest {
    pub dataset_id: DatasetId,
    // Map node_type -> list of global node IDs
    pub nodes: HashMap<String, Vec<u64>>,
    // Map node_type -> list of feature names (or empty/missing for all)
    pub features: HashMap<String, Vec<String>>,
}