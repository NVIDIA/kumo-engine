//! DiskGraph Explanation Engine
//!
//! Generates explanations for GNN predictions by:
//! 1. Extracting entity subgraphs from the graph index
//! 2. Computing motifs (structural, temporal, attribute aggregations)
//! 3. Contextualizing with population statistics (deciles, conditional rates, z-scores)
//! 4. Clustering embeddings to identify behavioral archetypes
//!
//! All computations are task-agnostic and work for classification, regression, etc.

mod clustering;
mod config;
mod motifs;
mod output;
mod parquet_io;
mod pipeline;
mod population;
mod prompt;
mod subgraph;

pub use config::{ExplainConfig, IdKind, MotifConfig, PromptConfig, TimestampKind};
pub use motifs::{AttributeMotif, EntityMotifs, HomophilyMotifs, RelationMotifs, TemporalMotifs};
pub use output::{ClusterContext, EntityExplanation, PopulationSummary};
pub use pipeline::run;
pub use population::{FeatureContext, PopulationStats};
pub use prompt::{generate_prompt, infer_graph_description};
pub use subgraph::{ExtractedSubgraph, SubgraphExtractor};
