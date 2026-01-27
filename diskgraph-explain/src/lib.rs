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

pub use config::{ExplainConfig, IdKind, TimestampKind, MotifConfig, PromptConfig};
pub use motifs::{EntityMotifs, RelationMotifs, TemporalMotifs, AttributeMotif, HomophilyMotifs};
pub use output::{EntityExplanation, ClusterContext, PopulationSummary};
pub use pipeline::run;
pub use population::{PopulationStats, FeatureContext};
pub use prompt::{generate_prompt, infer_graph_description};
pub use subgraph::{SubgraphExtractor, ExtractedSubgraph};
