// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Main explanation pipeline.
//!
//! Orchestrates:
//! 1. Load predictions + embeddings
//! 2. Extract subgraphs for all entities
//! 3. Compute motifs
//! 4. Compute population statistics
//! 5. Cluster embeddings
//! 6. Generate explanations with context
//! 7. Generate LLM prompts (if prompt_config provided)

use crate::clustering::{build_cluster_context, find_archetype_samples, run_kmeans};
use crate::config::{ExplainConfig, PromptConfig};
use crate::motifs::{compute_entity_motifs, EntityMotifs};
use crate::output::{EntityExplanation, PopulationSummary};
use crate::parquet_io::ParquetStream;
use crate::population::PopulationStatsBuilder;
use crate::prompt::{generate_prompt, infer_graph_description};
use crate::subgraph::SubgraphExtractor;
use anyhow::Result;
use std::collections::HashMap;

/// Run the full explanation pipeline
pub fn run(config: ExplainConfig) -> Result<()> {
    eprintln!("Starting explanation pipeline...");

    // Phase 1: Load graph index
    eprintln!("Loading graph index from {:?}...", config.index_path);
    let reader = diskgraph_index::graph::GraphIndexReader::open(
        &config.index_path.join("index_metadata.json"),
        &config.index_path,
    )?;
    let extractor = SubgraphExtractor::new(reader, &config)?;

    // Phase 2: First pass - collect all predictions and embeddings
    eprintln!("Pass 1: Reading predictions and embeddings...");
    let mut all_entities: Vec<EntityData> = Vec::new();
    let mut node_id_to_prediction: HashMap<u64, f64> = HashMap::new();

    {
        let mut stream = ParquetStream::open(
            &config.input_parquet,
            &config.id_column,
            &config.timestamp_column,
            &config.prediction_column,
            &config.embedding_column,
            config.id_kind.clone(),
            config.timestamp_kind.clone(),
        )?;

        while let Some(batch) = stream.next_batch() {
            let batch = batch?;
            for i in 0..batch.len() {
                let id = batch.id_as_string(i)?;
                let timestamp = batch.timestamp_as_i64(i)?;
                let prediction = batch.prediction_as_f64(i)?;
                let embedding = batch.embedding_as_f32(i)?;

                // Store node_id -> prediction mapping for homophily lookup
                if let Ok(node_id) = extractor.lookup_node_id(&id) {
                    node_id_to_prediction.insert(node_id, prediction);
                }

                all_entities.push(EntityData {
                    id,
                    timestamp,
                    prediction,
                    embedding,
                });
            }
        }
    }

    eprintln!("  Loaded {} entities", all_entities.len());

    // Phase 3: Extract subgraphs and compute motifs
    eprintln!("Pass 2: Extracting subgraphs and computing motifs...");
    let mut all_motifs: Vec<EntityMotifs> = Vec::new();
    let mut motifs_with_predictions: Vec<(EntityMotifs, f64)> = Vec::new();

    for (idx, entity) in all_entities.iter().enumerate() {
        if (idx + 1) % 1000 == 0 {
            eprintln!("  Processed {}/{} entities...", idx + 1, all_entities.len());
        }

        let subgraph = match extractor.extract_subgraph(&entity.id, entity.timestamp) {
            Ok(sg) => sg,
            Err(e) => {
                eprintln!(
                    "  Warning: Failed to extract subgraph for {}: {}",
                    entity.id, e
                );
                continue;
            }
        };

        let motifs = compute_entity_motifs(&subgraph, &node_id_to_prediction, &config.motif_config);
        all_motifs.push(motifs.clone());
        motifs_with_predictions.push((motifs, entity.prediction));
    }

    eprintln!("  Computed motifs for {} entities", all_motifs.len());

    // Phase 4: Compute population statistics
    eprintln!("Pass 3: Computing population statistics...");
    let mut stats_builder = PopulationStatsBuilder::new();
    for (motifs, prediction) in &motifs_with_predictions {
        stats_builder.add_entity(motifs, *prediction);
    }
    let population_stats = stats_builder.build(config.num_deciles);
    eprintln!(
        "  Population: {} entities, mean prediction = {:.4}",
        population_stats.total_entities, population_stats.prediction_mean
    );

    // Phase 5: Cluster embeddings
    eprintln!(
        "Pass 4: Clustering embeddings (k={})...",
        config.num_clusters
    );
    let embeddings: Vec<Vec<f32>> = all_entities.iter().map(|e| e.embedding.clone()).collect();
    let predictions: Vec<f64> = all_entities.iter().map(|e| e.prediction).collect();
    let clustering = run_kmeans(
        &embeddings,
        &predictions,
        config.num_clusters,
        config.kmeans_iterations,
    );

    // Find archetype samples (N nearest entities to each centroid)
    eprintln!(
        "  Finding archetype samples ({} per cluster)...",
        config.archetype_sample_size
    );
    let archetype_samples = find_archetype_samples(&clustering, config.archetype_sample_size);

    // Add feature values to cluster stats
    let mut cluster_stats = clustering.cluster_stats.clone();

    // Add all member features (for fallback)
    for (idx, motifs) in all_motifs.iter().enumerate() {
        let cluster_id = clustering.assignments[idx];
        let contexts = population_stats.contextualize_motifs(motifs);
        for ctx in &contexts {
            cluster_stats[cluster_id].add_feature(&ctx.feature, ctx.value);
        }
    }

    // Add archetype features (from sampled nearest entities)
    for (cluster_id, sample_indices) in archetype_samples.iter().enumerate() {
        for &entity_idx in sample_indices {
            if entity_idx < all_motifs.len() {
                let contexts = population_stats.contextualize_motifs(&all_motifs[entity_idx]);
                for ctx in &contexts {
                    cluster_stats[cluster_id].add_archetype_feature(&ctx.feature, ctx.value);
                }
            }
        }
    }
    eprintln!(
        "  Built cluster profiles for {} clusters",
        cluster_stats.len()
    );

    // Phase 6: Generate explanations
    eprintln!("Pass 5: Generating explanations...");
    let mut explanations: Vec<EntityExplanation> = Vec::new();

    let population_summary = PopulationSummary {
        total_entities: population_stats.total_entities,
        mean_prediction: population_stats.prediction_mean,
        std_prediction: population_stats.prediction_std,
        min_prediction: population_stats.prediction_min,
        max_prediction: population_stats.prediction_max,
    };

    for (idx, (entity, motifs)) in all_entities.iter().zip(all_motifs.iter()).enumerate() {
        if (idx + 1) % 1000 == 0 {
            eprintln!(
                "  Generated {}/{} explanations...",
                idx + 1,
                all_entities.len()
            );
        }

        let all_feature_contexts = population_stats.contextualize_motifs(motifs);

        // Filter for narrative relevance (|z| > threshold or rate_ratio outside bounds)
        let feature_contexts = population_stats.filter_for_narrative_relevance(
            &all_feature_contexts,
            config.z_threshold,
            config.rate_ratio_threshold,
        );

        // Get top drivers from filtered features
        let top_drivers = population_stats.get_top_drivers(&feature_contexts, config.top_drivers_k);

        // Cluster context - always include if enabled (let LLM interpret fit score)
        let cluster_id = clustering.assignments[idx];
        let cluster_context = if config.include_cluster_context {
            Some(build_cluster_context(
                cluster_id,
                &cluster_stats[cluster_id],
                &all_feature_contexts, // Use all contexts for cluster matching
                &population_stats,
                config.top_drivers_k,
            ))
        } else {
            None
        };

        explanations.push(EntityExplanation {
            entity_id: entity.id.clone(),
            prediction: entity.prediction,
            motifs: motifs.clone(),
            feature_contexts,
            top_drivers,
            cluster: cluster_context,
            population: population_summary.clone(),
        });
    }

    // Phase 7: Generate prompts (if configured)
    let prompts: Option<Vec<String>> = if let Some(ref prompt_config) = config.prompt_config {
        eprintln!("Pass 6: Generating LLM prompts...");

        // Auto-infer graph description if not provided
        let effective_config = if prompt_config.graph_description.is_empty() {
            // Use first entity's motifs to infer graph structure
            let graph_desc = if let Some(first_motifs) = all_motifs.first() {
                infer_graph_description(&first_motifs.relation_motifs, &config.root_table)
            } else {
                config.root_table.clone()
            };
            PromptConfig {
                task_description: prompt_config.task_description.clone(),
                graph_description: graph_desc,
                domain: prompt_config.domain.clone(),
            }
        } else {
            prompt_config.clone()
        };

        let prompts: Vec<String> = explanations
            .iter()
            .map(|exp| generate_prompt(exp, &effective_config))
            .collect();

        eprintln!(
            "  Generated {} prompts (avg {} chars)",
            prompts.len(),
            prompts.iter().map(|p| p.len()).sum::<usize>() / prompts.len().max(1)
        );
        Some(prompts)
    } else {
        None
    };

    // Phase 8: Write output
    eprintln!(
        "Writing {} explanations to {:?}...",
        explanations.len(),
        config.output_parquet
    );
    write_explanations(&config.output_parquet, &explanations, prompts.as_deref())?;

    eprintln!("Done!");
    Ok(())
}

/// Entity data loaded from parquet
struct EntityData {
    id: String,
    timestamp: i64,
    prediction: f64,
    embedding: Vec<f32>,
}

/// Write explanations to parquet
fn write_explanations(
    path: &std::path::Path,
    explanations: &[EntityExplanation],
    prompts: Option<&[String]>,
) -> Result<()> {
    use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let mut fields = vec![
        Field::new("entity_id", DataType::Utf8, false),
        Field::new("prediction", DataType::Float64, false),
        Field::new("explanation_json", DataType::Utf8, false),
    ];

    if prompts.is_some() {
        fields.push(Field::new("prompt", DataType::Utf8, false));
    }

    let schema = Schema::new(fields);

    let mut ids = Vec::with_capacity(explanations.len());
    let mut preds = Vec::with_capacity(explanations.len());
    let mut jsons = Vec::with_capacity(explanations.len());

    for exp in explanations {
        ids.push(exp.entity_id.clone());
        preds.push(exp.prediction);
        jsons.push(serde_json::to_string(exp)?);
    }

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids)),
        Arc::new(Float64Array::from(preds)),
        Arc::new(StringArray::from(jsons)),
    ];

    if let Some(prompts) = prompts {
        columns.push(Arc::new(StringArray::from(prompts.to_vec())));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;

    crate::parquet_io::write_parquet(path, schema, vec![batch])?;
    Ok(())
}
