use crate::dataset::DatasetId;
use crate::time::TimeUnit;
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
    /// If true, returns tickets for side-channel feature streams.
    /// Default: false
    #[serde(default)]
    pub fused_feature_fetching: bool,
    /// If true, swaps edge direction (src↔dst) and maps to reverse relation.
    /// Required for PyG's CSC convention. Default: true.
    #[serde(default = "default_swap_edge_direction")]
    pub swap_edge_direction: bool,
}

fn default_swap_edge_direction() -> bool {
    true
}

/// A request to sample a neighborhood subgraph starting from a batch of seeds.
///
/// # Why `seed_timestamps` is `Vec<i64>` rather than `Vec<Abs<U>>`
///
/// This struct is a **wire-format / API contract**: callers (Python via PyO3,
/// JSON over FlightSQL, bincode) specify the timestamp unit dynamically via
/// `seed_timestamp_unit`. The concrete `UnitSystem` is not known at compile
/// time at these API boundaries, so making the struct generic over
/// `U: UnitSystem` would cascade into the gateway, pybind, and storage-trait
/// layers (~20 files) for no safety gain. Instead, the `SamplingEngine`
/// converts to the index's internal unit in a single pass at the start of
/// `sample()` (see `engine.rs` lines ~527-544).
///
/// If the engine is ever made generic over `U`, this struct should grow a type
/// parameter and the conversion would move to the API-boundary adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSamplingRequest {
    /// Target dataset identifier.
    pub dataset_id: DatasetId,
    /// Batch of seed node IDs to start the random walk, grouped by node type.
    pub seed_nodes: HashMap<String, Vec<u64>>,
    /// Optional batch of timestamps (1:1 with seed_nodes), grouped by node type.
    /// If present, acts as the anchor time for temporal filtering.
    /// These are raw i64 values in the unit specified by `seed_timestamp_unit`;
    /// see the struct-level doc comment for why this is not `Vec<Abs<U>>`.
    pub seed_timestamps: Option<HashMap<String, Vec<i64>>>,
    /// The unit of the provided seed_timestamps. Becomes redundant if this
    /// struct is ever made generic over `UnitSystem` (the unit would be
    /// encoded in the type), but is required for the current dynamic-unit API.
    #[serde(default)]
    pub seed_timestamp_unit: TimeUnit,
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
    /// Default fanout used if an edge type is not present in the fanouts map.
    pub default_fanout: Option<FanoutSpec>,
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
    /// Uniform selection with a deterministic internal seed.
    DeterministicUniform,
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
                    fanouts: [(
                        "user__click__item".to_string(),
                        FanoutSpec {
                            fanout: 10,
                            mode: SamplingMode::TemporalMostRecent,
                            absent_key_policy: AbsentKeyPolicy::RejectPlan,
                        },
                    )]
                    .into(),
                    default_fanout: None,
                },
                LayerSpec {
                    fanouts: [(
                        "item__click__user".to_string(),
                        FanoutSpec {
                            fanout: 5,
                            mode: SamplingMode::Weighted {
                                weighting: Weighting::LinearDecay {
                                    half_life_seconds: 3600,
                                },
                            },
                            absent_key_policy: AbsentKeyPolicy::UniformFallback,
                        },
                    )]
                    .into(),
                    default_fanout: None,
                },
            ],
        };

        let request = BatchSamplingRequest {
            dataset_id: DatasetId::nil(),
            seed_nodes: [("User".to_string(), vec![1, 2, 3])].into(),
            seed_timestamps: Some([("User".to_string(), vec![100, 200, 300])].into()),
            seed_timestamp_unit: crate::time::TimeUnit::Seconds,
            plan,
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: true,
                return_edge_id: false,
                adaptive: false,
                fused_feature_fetching: false,
                swap_edge_direction: true,
            },
        };

        let json = serde_json::to_value(&request).unwrap();

        // check top level fields
        assert_eq!(json["seed_nodes"]["User"], json!([1, 2, 3]));
        assert_eq!(json["seed_timestamps"]["User"], json!([100, 200, 300]));

        // check enum serialization (snake_case)
        let layer0 = &json["plan"]["layers"][0]["fanouts"]["user__click__item"];
        assert_eq!(layer0["mode"], "temporal_most_recent");

        let layer1 = &json["plan"]["layers"][1]["fanouts"]["item__click__user"];
        assert_eq!(
            layer1["mode"]["weighted"]["weighting"]["linear_decay"]["half_life_seconds"],
            3600
        );
    }

    #[test]
    fn test_sampling_request_bincode() {
        let request = BatchSamplingRequest {
            dataset_id: DatasetId::nil(),
            seed_nodes: [("User".to_string(), vec![1, 2, 3])].into(),
            seed_timestamps: None,
            seed_timestamp_unit: crate::time::TimeUnit::Seconds,
            plan: SamplingPlan {
                layers: vec![LayerSpec {
                    fanouts: [(
                        "user__click__item".to_string(),
                        FanoutSpec {
                            fanout: 10,
                            mode: SamplingMode::TemporalMostRecent,
                            absent_key_policy: AbsentKeyPolicy::RejectPlan,
                        },
                    )]
                    .into(),
                    default_fanout: None,
                }],
            },
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: true,
                return_edge_id: false,
                adaptive: false,
                fused_feature_fetching: false,
                swap_edge_direction: true,
            },
        };

        let encoded = bincode::serialize(&request).unwrap();
        let decoded: BatchSamplingRequest = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.seed_nodes["User"], vec![1, 2, 3]);
        assert_eq!(decoded.plan.layers.len(), 1);
    }

    #[test]
    fn test_deterministic_uniform_mode_serialization() {
        let mode = SamplingMode::DeterministicUniform;

        let json = serde_json::to_value(&mode).unwrap();
        assert_eq!(json, json!("deterministic_uniform"));

        let decoded: SamplingMode =
            serde_json::from_value(json!("deterministic_uniform")).unwrap();
        assert!(matches!(decoded, SamplingMode::DeterministicUniform));
    }
}
