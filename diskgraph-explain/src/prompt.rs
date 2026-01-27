//! LLM prompt generation from entity explanations.
//!
//! Generates comprehensive, task-agnostic prompts that contain:
//! - Context (task, graph schema, domain)
//! - How to read the data (explanation of metrics)
//! - Entity data (subgraph, features)
//! - Comparison context (vs population, vs cluster)
//! - Task instructions

use crate::config::PromptConfig;
use crate::motifs::{AttributeMotif, AttributeMotifType, EntityMotifs, RelationMotifs};
use crate::output::{ClusterContext, ClusterProfileFeature, EntityExplanation, PopulationSummary};
use crate::population::FeatureContext;
use std::collections::HashMap;
use std::fmt::Write;

/// Generate a complete LLM prompt from an entity explanation
pub fn generate_prompt(explanation: &EntityExplanation, config: &PromptConfig) -> String {
    let mut prompt = String::with_capacity(8000);

    // === CONTEXT ===
    write_context_section(&mut prompt, &explanation.population, config);

    // === HOW TO READ THE DATA ===
    write_how_to_read_section(&mut prompt, explanation.population.mean_prediction);

    // === ENTITY ===
    write_entity_section(&mut prompt, explanation);

    // === SUBGRAPH DATA ===
    write_subgraph_section(&mut prompt, &explanation.motifs);

    // === IMPORTANT FEATURES (vs Population) ===
    write_population_features_section(&mut prompt, &explanation.top_drivers);

    // === IMPORTANT FEATURES (vs Cluster) ===
    if let Some(cluster) = &explanation.cluster {
        let entity_values = extract_entity_values(&explanation.motifs);
        write_cluster_section(&mut prompt, cluster, &entity_values);
    }

    // === TASK ===
    write_task_section(&mut prompt);

    prompt
}

fn write_context_section(prompt: &mut String, pop: &PopulationSummary, config: &PromptConfig) {
    writeln!(prompt, "=== CONTEXT ===").unwrap();
    writeln!(prompt, "Task: {}", config.task_description).unwrap();
    writeln!(prompt, "Graph: {}", config.graph_description).unwrap();
    writeln!(prompt, "Domain: {}", config.domain).unwrap();
    writeln!(prompt).unwrap();
    writeln!(
        prompt,
        "Population (all {} entities):",
        pop.total_entities
    )
    .unwrap();
    writeln!(
        prompt,
        "  mean_prediction: {:.3} (average {})",
        pop.mean_prediction, config.task_description
    )
    .unwrap();
    writeln!(prompt, "  std_prediction: {:.3}", pop.std_prediction).unwrap();
    writeln!(prompt).unwrap();
}

fn write_how_to_read_section(prompt: &mut String, pop_rate: f64) {
    writeln!(prompt, "=== HOW TO READ THE DATA ===").unwrap();
    writeln!(prompt, "For each feature:").unwrap();
    writeln!(prompt, "- value: the actual value for this entity").unwrap();
    writeln!(
        prompt,
        "- z_score: (entity_value - population_mean) / population_std"
    )
    .unwrap();
    writeln!(
        prompt,
        "  Measures how unusual this value is. |z| > 2 means very unusual."
    )
    .unwrap();
    writeln!(
        prompt,
        "- decile: which 10% bucket (1=bottom 10%, 10=top 10%)"
    )
    .unwrap();
    writeln!(
        prompt,
        "- conditional_rate: prediction rate among ALL entities with similar values of this feature"
    )
    .unwrap();
    writeln!(
        prompt,
        "- population_rate: overall prediction rate ({:.3})",
        pop_rate
    )
    .unwrap();
    writeln!(
        prompt,
        "  If conditional_rate >> population_rate, this feature is predictive."
    )
    .unwrap();
    writeln!(prompt).unwrap();
    writeln!(prompt, "For cluster matches:").unwrap();
    writeln!(
        prompt,
        "- These are features where entity's pattern matches the cluster's pattern"
    )
    .unwrap();
    writeln!(
        prompt,
        "- They explain WHY this entity belongs to this cluster"
    )
    .unwrap();
    writeln!(prompt).unwrap();
}

fn write_entity_section(prompt: &mut String, explanation: &EntityExplanation) {
    writeln!(prompt, "=== ENTITY ===").unwrap();
    writeln!(prompt, "ID: {}", explanation.entity_id).unwrap();
    writeln!(prompt, "Prediction: {:.3}", explanation.prediction).unwrap();
    writeln!(prompt).unwrap();
}

fn write_subgraph_section(prompt: &mut String, motifs: &EntityMotifs) {
    writeln!(prompt, "=== SUBGRAPH DATA ===").unwrap();
    writeln!(prompt, "Local neighborhood extracted for this entity:").unwrap();
    writeln!(prompt).unwrap();

    // Node counts from relation motifs
    let node_counts = count_subgraph_nodes(&motifs.relation_motifs);
    if !node_counts.is_empty() {
        let counts_str: Vec<String> = node_counts
            .iter()
            .map(|(t, c)| format!("{}: {}", t, c))
            .collect();
        writeln!(prompt, "Node counts: {}", counts_str.join(", ")).unwrap();
    } else {
        writeln!(prompt, "Node counts: (unknown)").unwrap();
    }
    writeln!(prompt).unwrap();

    // Relation motifs
    write_relation_motifs(prompt, &motifs.relation_motifs);
    writeln!(prompt).unwrap();

    // Attribute motifs
    write_attribute_motifs(prompt, &motifs.attribute_motifs);
    writeln!(prompt).unwrap();
}

fn write_relation_motifs(prompt: &mut String, motifs: &HashMap<String, RelationMotifs>) {
    if motifs.is_empty() {
        writeln!(prompt, "  (no relation data)").unwrap();
        return;
    }

    for (rel_name, rel_data) in motifs {
        writeln!(
            prompt,
            "  Relation: {} (-> {})",
            rel_name, rel_data.dst_table
        )
        .unwrap();
        writeln!(prompt, "    edge_count: {}", rel_data.edge_count).unwrap();
        writeln!(
            prompt,
            "    unique_destinations: {}",
            rel_data.unique_destinations
        )
        .unwrap();

        if let Some(temporal) = &rel_data.temporal {
            let recency_days = temporal.recency as f64 / 86400.0;
            writeln!(
                prompt,
                "    recency_seconds: {} ({:.1} days)",
                temporal.recency, recency_days
            )
            .unwrap();
            let span_days = temporal.span as f64 / 86400.0;
            writeln!(
                prompt,
                "    span_seconds: {} ({:.1} days)",
                temporal.span, span_days
            )
            .unwrap();
            if let Some(trend) = temporal.trend_slope {
                writeln!(prompt, "    trend_slope: {:.3}", trend).unwrap();
            }
        }
    }
}

fn write_attribute_motifs(prompt: &mut String, motifs: &HashMap<String, AttributeMotif>) {
    if motifs.is_empty() {
        writeln!(prompt, "  (no attribute data)").unwrap();
        return;
    }

    // Group by table
    let mut by_table: HashMap<String, Vec<(&String, &AttributeMotif)>> = HashMap::new();
    for (path, motif) in motifs {
        let table = path
            .split('.')
            .next()
            .unwrap_or("unknown")
            .to_string();
        by_table.entry(table).or_default().push((path, motif));
    }

    for (table, attrs) in by_table.iter() {
        // Get node count from first attribute
        let node_count = attrs
            .first()
            .map(|(_, m)| match &m.motif_type {
                AttributeMotifType::Numeric(n) => n.count,
                AttributeMotifType::Categorical(c) => c.count,
            })
            .unwrap_or(0);

        writeln!(
            prompt,
            "  Attributes from {} (aggregated over {} nodes):",
            table, node_count
        )
        .unwrap();

        for (path, motif) in attrs {
            let col = path.split('.').last().unwrap_or(path);
            match &motif.motif_type {
                AttributeMotifType::Numeric(n) => {
                    let parts = vec![
                        format!("count={}", n.count),
                        format_number("sum", n.sum),
                        format_number("mean", n.mean),
                        format_number("std", n.std),
                        format_number("min", n.min),
                        format_number("max", n.max),
                    ];
                    writeln!(prompt, "    {}: {}", col, parts.join(", ")).unwrap();
                }
                AttributeMotifType::Categorical(c) => {
                    let parts = vec![
                        format!("count={}", c.count),
                        format!("nunique={}", c.nunique),
                        format!("mode={}", c.mode),
                        format!("mode_count={}", c.mode_count),
                    ];
                    writeln!(prompt, "    {}: {}", col, parts.join(", ")).unwrap();
                }
            }
        }
    }
}

fn format_number(name: &str, val: f64) -> String {
    if val.abs() > 1000.0 {
        format!("{}={:.2e}", name, val)
    } else {
        format!("{}={:.4}", name, val)
    }
}

fn write_population_features_section(prompt: &mut String, top_drivers: &[FeatureContext]) {
    writeln!(prompt, "=== IMPORTANT FEATURES (vs Population) ===").unwrap();
    if top_drivers.is_empty() {
        writeln!(prompt, "  (no features stand out vs population)").unwrap();
    } else {
        for ctx in top_drivers {
            writeln!(prompt, "  {}", ctx.feature).unwrap();
            writeln!(
                prompt,
                "    value: {:.4}, z_score: {:+.2}, decile: {}",
                ctx.value, ctx.z_score, ctx.decile
            )
            .unwrap();
            writeln!(
                prompt,
                "    conditional_rate: {:.3}, population_rate: {:.3}",
                ctx.conditional_rate, ctx.population_rate
            )
            .unwrap();
        }
    }
    writeln!(prompt).unwrap();
}

fn write_cluster_section(
    prompt: &mut String,
    cluster: &ClusterContext,
    entity_values: &HashMap<String, f64>,
) {
    writeln!(prompt, "=== IMPORTANT FEATURES (vs Cluster) ===").unwrap();
    writeln!(
        prompt,
        "  Cluster: {} entities, mean_prediction: {:.3}, fit: {:.0}%",
        cluster.cluster_size,
        cluster.cluster_mean_prediction,
        cluster.fit_score * 100.0
    )
    .unwrap();
    writeln!(prompt).unwrap();

    // Build cluster profile lookup
    let profile_lookup: HashMap<&str, &ClusterProfileFeature> = cluster
        .profile
        .iter()
        .map(|p| (p.feature.as_str(), p))
        .collect();

    // Matches
    writeln!(prompt, "  Matches (entity fits cluster pattern):").unwrap();
    if cluster.entity_matches.is_empty() {
        writeln!(prompt, "    (none)").unwrap();
    } else {
        for m in cluster.entity_matches.iter().take(8) {
            if let Some(cp) = profile_lookup.get(m.as_str()) {
                writeln!(prompt, "    - {}:", m).unwrap();

                // Cluster stats
                let cluster_desc = if cp.cluster_z_score > 1.0 {
                    "cluster is high vs global"
                } else if cp.cluster_z_score < -1.0 {
                    "cluster is low vs global"
                } else {
                    "cluster is typical vs global"
                };
                writeln!(
                    prompt,
                    "        cluster: mean={:.4}, std={:.4}, cluster_z_vs_global={:+.2} ({})",
                    cp.cluster_mean, cp.cluster_std, cp.cluster_z_score, cluster_desc
                )
                .unwrap();

                // Entity stats
                if let Some(entity_val) = entity_values.get(m) {
                    let entity_z_in_cluster = if cp.cluster_std > 0.0 {
                        (entity_val - cp.cluster_mean) / cp.cluster_std
                    } else {
                        0.0
                    };
                    writeln!(
                        prompt,
                        "        entity: value={:.4}, entity_z_in_cluster={:+.2}",
                        entity_val, entity_z_in_cluster
                    )
                    .unwrap();
                } else {
                    writeln!(prompt, "        entity: (matches cluster)").unwrap();
                }
            }
        }
    }
    writeln!(prompt).unwrap();

    // Differs
    writeln!(prompt, "  Differs (entity deviates from cluster):").unwrap();
    if cluster.entity_differs.is_empty() {
        writeln!(prompt, "    (none)").unwrap();
    } else {
        for d in cluster.entity_differs.iter().take(5) {
            if let Some(cp) = profile_lookup.get(d.as_str()) {
                writeln!(prompt, "    - {}:", d).unwrap();
                writeln!(
                    prompt,
                    "        cluster: mean={:.4}, std={:.4}, cluster_z_vs_global={:+.2}",
                    cp.cluster_mean, cp.cluster_std, cp.cluster_z_score
                )
                .unwrap();

                if let Some(entity_val) = entity_values.get(d) {
                    let entity_z_in_cluster = if cp.cluster_std > 0.0 {
                        (entity_val - cp.cluster_mean) / cp.cluster_std
                    } else {
                        0.0
                    };
                    writeln!(
                        prompt,
                        "        entity: value={:.4}, entity_z_in_cluster={:+.2} (deviates)",
                        entity_val, entity_z_in_cluster
                    )
                    .unwrap();
                } else {
                    writeln!(prompt, "        entity: (deviates from cluster)").unwrap();
                }
            }
        }
    }
    writeln!(prompt).unwrap();
}

fn write_task_section(prompt: &mut String) {
    writeln!(prompt, "=== TASK ===").unwrap();
    writeln!(
        prompt,
        "Write a 3-4 sentence explanation for why this entity has this prediction."
    )
    .unwrap();
    writeln!(prompt).unwrap();
    writeln!(prompt, "Guidelines:").unwrap();
    writeln!(
        prompt,
        "- Tell a convincing story grounded in the data above"
    )
    .unwrap();
    writeln!(
        prompt,
        "- IMPORTANT: Use only the data that directly supports your story - do not try to use all the data"
    )
    .unwrap();
    writeln!(prompt, "- Use specific numbers to support your claims").unwrap();
    writeln!(
        prompt,
        "- Focus on what makes this entity different from the population or typical of its cluster"
    )
    .unwrap();
    writeln!(
        prompt,
        "- REQUIRED: Include at least 1 sentence about the cluster and how this entity relates to it"
    )
    .unwrap();
    writeln!(prompt).unwrap();
    writeln!(prompt, "Note on clusters:").unwrap();
    writeln!(
        prompt,
        "- Clusters are formed using the actual ML model embeddings, so cluster membership is high signal"
    )
    .unwrap();
    writeln!(
        prompt,
        "- If an entity matches cluster features (low entity_z_in_cluster), this strongly supports the prediction"
    )
    .unwrap();
    writeln!(
        prompt,
        "- cluster_z_vs_global: how the cluster differs from the overall population"
    )
    .unwrap();
    writeln!(
        prompt,
        "- entity_z_in_cluster: how the entity differs from its cluster (|z| < 1 = matches, |z| > 2 = deviates)"
    )
    .unwrap();
}

/// Count nodes per table from relation motifs
fn count_subgraph_nodes(motifs: &HashMap<String, RelationMotifs>) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for motif in motifs.values() {
        let dst = &motif.dst_table;
        let count = motif.unique_destinations as usize;
        let entry = counts.entry(dst.clone()).or_insert(0);
        *entry = (*entry).max(count);
    }
    counts
}

/// Extract entity feature values from motifs for cluster comparison
fn extract_entity_values(motifs: &EntityMotifs) -> HashMap<String, f64> {
    let mut values = HashMap::new();

    // From relation motifs
    for (rel_name, rel_data) in &motifs.relation_motifs {
        let prefix = format!("relation:{}", rel_name);
        values.insert(format!("{}:edge_count", prefix), rel_data.edge_count as f64);
        values.insert(
            format!("{}:unique_destinations", prefix),
            rel_data.unique_destinations as f64,
        );

        if let Some(temporal) = &rel_data.temporal {
            values.insert(format!("{}:recency", prefix), temporal.recency as f64);
            values.insert(format!("{}:span", prefix), temporal.span as f64);
            if let Some(v) = temporal.trend_slope {
                values.insert(format!("{}:trend_slope", prefix), v);
            }
        }
    }

    // From attribute motifs
    for (path, motif) in &motifs.attribute_motifs {
        match &motif.motif_type {
            AttributeMotifType::Numeric(n) => {
                values.insert(format!("{}:count", path), n.count as f64);
                values.insert(format!("{}:sum", path), n.sum);
                values.insert(format!("{}:mean", path), n.mean);
                values.insert(format!("{}:std", path), n.std);
                values.insert(format!("{}:min", path), n.min);
                values.insert(format!("{}:max", path), n.max);
            }
            AttributeMotifType::Categorical(c) => {
                values.insert(format!("{}:count", path), c.count as f64);
                values.insert(format!("{}:nunique", path), c.nunique as f64);
            }
        }
    }

    values
}

/// Infer graph description from relation motifs
pub fn infer_graph_description(relation_motifs: &HashMap<String, RelationMotifs>, root_table: &str) -> String {
    // Collect all unique destination tables from all relations
    let mut tables: Vec<String> = Vec::new();
    for motif in relation_motifs.values() {
        // Include all destination tables (don't skip reverse - they lead to important nodes)
        if !tables.contains(&motif.dst_table) && motif.dst_table != root_table {
            tables.push(motif.dst_table.clone());
        }
    }

    if tables.is_empty() {
        return root_table.to_string();
    }

    // Sort for consistent output
    tables.sort();

    // Chain representation: root -> table1 -> table2 -> ...
    let mut parts = vec![root_table.to_string()];
    parts.extend(tables.into_iter().take(3));
    parts.join(" -> ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number("sum", 1234567.0), "sum=1.23e6");
        assert_eq!(format_number("mean", 12.345), "mean=12.3450");
    }
}
