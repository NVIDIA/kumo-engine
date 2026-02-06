use crate::dataset::DatasetId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration flags for the sampling process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingFlags {
    /// If true, allows selecting the same neighbor multiple times to satisfy fanout.
    /// Default: false
    pub replace: bool,
    /// If true, follows edge direction. If false, treats graph as undirected.
    /// Default: true
    pub directed: bool,
    /// If true, ensures nodes sampled at hop K are treated as distinct seeds for hop K+1.
    /// Required for correct temporal sampling.
    /// Default: true
    pub disjoint: bool,
    /// If true, output includes edge identifiers (required for edge feature fetching).
    /// Default: false
    pub return_edge_id: bool,
    /// If true, enables dynamic fanout compensation (Metapath-aware/Adaptive).
    /// Default: false
    pub adaptive: bool,
}

/// A request to sample a neighborhood subgraph starting from a batch of seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSamplingRequest {
    /// Target dataset identifier.
    pub dataset_id: DatasetId,
    /// Batch of seed node IDs to start the random walk.
    pub seed_nodes: Vec<u64>,
    /// Optional batch of timestamps (1:1 with seed_nodes).
    /// If present, acts as the anchor time for temporal filtering.
    pub seed_timestamps: Option<Vec<i64>>,
    /// The recursive sampling strategy.
    pub plan: SamplingPlan,
    /// Global behavioral modifiers.
    pub flags: SamplingFlags,
}

/// The definition of a sampling traversal, organized by layers (hops).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Ordered list of layers. Index 0 corresponds to the traversal from Seed -> Hop 1.
    pub layers: Vec<LayerSpec>,
}

/// Configuration for a single hop in the traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Fanout configuration per edge type (e.g., "User__Click__Item").
    pub fanouts: HashMap<String, FanoutSpec>,
}

/// Fanout and selection strategy for a specific edge type at a specific layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanoutSpec {
    /// Maximum number of neighbors to sample.
    pub fanout: u32,
    /// Strategy for selecting neighbors from the candidate list.
    pub mode: SamplingMode,
    /// Policy used when the mode requires a key (weight/time) that is absent for a relation.
    pub absent_key_policy: AbsentKeyPolicy,
}

/// Policy used when a plan references keys (time/weight) but the relation lacks them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbsentKeyPolicy {
    /// Fail the request if the required keys are missing.
    RejectPlan,
    /// Fallback to uniform random sampling if keys are missing.
    UniformFallback,
}

/// Strategy for selecting neighbors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    /// Random uniform selection.
    Uniform,
    /// Deterministic selection of the most recent edges.
    TemporalMostRecent,
    /// Probabilistic selection based on edge weight.
    Weighted { weighting: Weighting },
    /// Probabilistic selection based on time-decayed weight.
    TemporalWeighted { weighting: Weighting },
}

/// Configurable weighting functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    /// w = exp(-lambda * delta_seconds)
    ExponentialDecay { lambda: f64 },
    /// w = max(0, 1 - delta_seconds / half_life_seconds)
    LinearDecay { half_life_seconds: u64 },
    /// Placeholder for arbitrary caller-provided expression.
    CustomExpression { id: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetId;
    use serde_json::json;

    #[test]
    fn test_sampling_request_serialization() {
        let plan = SamplingPlan {
            layers: vec![
                LayerSpec {
                    fanouts: [
                        ("user__click__item".to_string(), FanoutSpec {
                            fanout: 10,
                            mode: SamplingMode::TemporalMostRecent,
                            absent_key_policy: AbsentKeyPolicy::RejectPlan,
                        }),
                    ].into(),
                },
                LayerSpec {
                    fanouts: [
                        ("item__click__user".to_string(), FanoutSpec {
                            fanout: 5,
                            mode: SamplingMode::Weighted {
                                weighting: Weighting::LinearDecay { half_life_seconds: 3600 }
                            },
                            absent_key_policy: AbsentKeyPolicy::UniformFallback,
                        }),
                    ].into(),
                },
            ],
        };

        let request = BatchSamplingRequest {
            dataset_id: DatasetId::nil(),
            seed_nodes: vec![1, 2, 3],
            seed_timestamps: Some(vec![100, 200, 300]),
            plan,
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: true,
                return_edge_id: false,
                adaptive: false,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        
        // check top level fields
        assert_eq!(json["seed_nodes"], json!([1, 2, 3]));
        
        // check enum serialization (snake_case)
        let layer0 = &json["plan"]["layers"][0]["fanouts"]["user__click__item"];
        assert_eq!(layer0["mode"], "temporal_most_recent");
        
        let layer1 = &json["plan"]["layers"][1]["fanouts"]["item__click__user"];
        assert_eq!(layer1["mode"]["weighted"]["weighting"]["linear_decay"]["half_life_seconds"], 3600);
    }
}