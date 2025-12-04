use crate::dataset::DatasetId;
use crate::errors::{DiskgraphError, Result};
use crate::storage::GraphStorage;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// How to interpret dest_sort_key when present on a relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DestSortKeyMode {
    /// Choose up to N most recent edges where dest_sort_key <= anchor_time.
    TemporalMostRecent,
    /// Weighted sampling with recency bias. The plan supplies a weighting
    /// function to convert (anchor_time - dest_sort_key) into weights.
    TemporalWeighted { weighting: Weighting },
    /// Weighted sampling without any temporal filter; dest_sort_key is treated
    /// purely as a weight or score.
    WeightedOnly { weighting: Weighting },
}

/// Policy used when a plan references dest_sort_key but the relation lacks it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbsentKeyPolicy {
    RejectPlan,
    UniformWithoutWeights,
}

/// Configurable weighting functions. Only the configuration is stored here; the
/// actual math is implemented by the sampler runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    /// w = exp(-lambda * delta_seconds)
    ExponentialDecay { lambda: f64 },
    /// w = max(0, 1 - delta_seconds / half_life_seconds)
    LinearDecay { half_life_seconds: u64 },
    /// Placeholder for arbitrary caller-provided expression. The runtime is
    /// expected to implement a resolver that can interpret this identifier.
    CustomExpression { id: String },
}

/// A single hop in the metapath tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopPlan {
    pub relation: String,
    pub fanout: u32,
    pub dest_sort_key_mode: Option<DestSortKeyMode>,
    pub absent_key_policy: AbsentKeyPolicy,
}

/// Top-level sampling plan rooted at an input node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPlan {
    pub root_type: String,
    pub hops: Vec<HopPlan>,
    /// If true, edge identifiers will be included in the response when present.
    pub include_edge_ids: bool,
}

/// Request issued by HTTP or Python bindings to sample a neighborhood.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequest {
    pub dataset_id: DatasetId,
    pub root_id: u64,
    pub anchor_time: Option<DateTime<Utc>>,
    pub plan: SamplingPlan,
}

/// Minimal header used in responses to trace a sampled neighborhood back to its
/// originating request without echoing the full plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequestHeader {
    pub dataset_id: DatasetId,
    pub root_id: u64,
    pub plan_root_type: String,
}

/// A sampled node reference in the output tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledNodeRef {
    pub node_type: String,
    pub node_id: u64,
    /// Hop depth where this node was sampled.
    pub hop: usize,
}

/// A sampled edge connecting two node refs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledEdge {
    pub src: SampledNodeRef,
    pub dst: SampledNodeRef,
    pub dest_sort_key: Option<u64>,
    pub edge_id: Option<u64>,
}

/// The output neighborhood structure. Implementations are free to embed
/// additional annotations for downstream ML consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledNeighborhood {
    pub request: SamplingRequestHeader,
    pub nodes: Vec<SampledNodeRef>,
    pub edges: Vec<SampledEdge>,
}

/// Trait implemented by sampling engines. Implementations must obey the
/// semantics documented on DestSortKeyMode and AbsentKeyPolicy.
#[async_trait]
pub trait Sampler: Send + Sync {
    async fn sample(&self, request: SamplingRequest) -> Result<SampledNeighborhood>;
}

/// A thin sampler that delegates storage access to a GraphStorage and leaves
/// the actual sampling math to the storage layer. This keeps the HTTP layer
/// decoupled from the storage backend selection.
pub struct RoutingSampler {
    storage: Arc<dyn GraphStorage>,
}

impl RoutingSampler {
    pub fn new(storage: Arc<dyn GraphStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Sampler for RoutingSampler {
    async fn sample(&self, request: SamplingRequest) -> Result<SampledNeighborhood> {
        // The storage implementation is expected to respect the sampling plan.
        // At this layer we only perform shallow validation.
        if request.plan.hops.is_empty() {
            return Err(DiskgraphError::InvalidMetadata("sampling plan has no hops".into()));
        }
        self.storage.sample(request).await
    }
}
