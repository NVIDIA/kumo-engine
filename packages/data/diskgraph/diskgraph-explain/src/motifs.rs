// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Motif computation from extracted subgraph data.
//!
//! Motifs are aggregations computed from an entity's subgraph:
//! - Structural: edge counts, unique destinations
//! - Temporal: recency, frequency, trend, inter-arrival statistics
//! - Attribute: numeric (sum, mean, std, min, max) and categorical (mode, nunique)
//! - Homophily: statistics about same-type nodes reached at hop k

use crate::config::MotifConfig;
use crate::subgraph::{AttributeValue, ExtractedEdge, ExtractedNode, ExtractedSubgraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete motif output for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMotifs {
    pub entity_id: String,

    /// Structural + temporal motifs per relation
    pub relation_motifs: HashMap<String, RelationMotifs>,

    /// Attribute motifs per (table.column) path
    pub attribute_motifs: HashMap<String, AttributeMotif>,

    /// Homophily motifs (if same-type nodes reached)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homophily: Option<HomophilyMotifs>,
}

/// Motifs for a single relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationMotifs {
    pub relation_name: String,
    pub dst_table: String,

    // Structural
    pub edge_count: u32,
    pub unique_destinations: u32,

    // Temporal (only if relation has timestamps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalMotifs>,
}

/// Temporal motifs computed from edge timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMotifs {
    /// Time since most recent edge (seconds)
    pub recency: i64,

    /// Time span from oldest to newest edge (seconds)
    pub span: i64,

    /// Number of edges
    pub count: u32,

    /// Average time between consecutive edges (seconds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inter_arrival_mean: Option<f64>,

    /// Standard deviation of inter-arrival times (seconds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inter_arrival_std: Option<f64>,

    /// Trend slope: positive = increasing activity, negative = decreasing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trend_slope: Option<f64>,

    /// Raw timestamp bounds
    pub timestamp_min: i64,
    pub timestamp_max: i64,
}

/// Attribute motif - either numeric or categorical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeMotif {
    pub path: String,
    pub motif_type: AttributeMotifType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AttributeMotifType {
    Numeric(NumericMotifs),
    Categorical(CategoricalMotifs),
}

/// Numeric attribute motifs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericMotifs {
    pub count: u32,
    pub sum: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl NumericMotifs {
    pub fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len() as u32;
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance = if count > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Some(Self {
            count,
            sum,
            mean,
            std,
            min,
            max,
        })
    }
}

/// Categorical attribute motifs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalMotifs {
    pub count: u32,
    pub nunique: u32,
    pub mode: String,
    pub mode_count: u32,
}

impl CategoricalMotifs {
    pub fn from_values(values: &[String]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len() as u32;
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for v in values {
            *counts.entry(v.as_str()).or_default() += 1;
        }

        let nunique = counts.len() as u32;
        let (mode, mode_count) = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, v)| (k.to_string(), *v))
            .unwrap_or_default();

        Some(Self {
            count,
            nunique,
            mode,
            mode_count,
        })
    }
}

/// Homophily motifs - statistics about same-type nodes reached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomophilyMotifs {
    /// At which hop we found same-type nodes
    pub hop: u8,

    /// Number of same-type nodes reached
    pub neighbor_count: u32,

    /// Mean prediction of neighbor nodes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_mean: Option<f64>,

    /// Std of predictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_std: Option<f64>,

    /// Min/max predictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_max: Option<f64>,
}

/// Compute all motifs from an extracted subgraph
pub fn compute_entity_motifs(
    subgraph: &ExtractedSubgraph,
    neighbor_predictions: &HashMap<u64, f64>,
    config: &MotifConfig,
) -> EntityMotifs {
    let mut relation_motifs = HashMap::new();
    let mut attribute_motifs = HashMap::new();

    // Compute relation motifs (structural + temporal)
    for (rel_name, edges) in &subgraph.edges_by_relation {
        let rel_motifs = compute_relation_motifs(rel_name, edges, subgraph.anchor_time, config);
        relation_motifs.insert(rel_name.clone(), rel_motifs);
    }

    // Compute attribute motifs
    for (table, nodes) in &subgraph.nodes_by_table {
        compute_attribute_motifs_for_table(table, nodes, config, &mut attribute_motifs);
    }

    // Compute homophily motifs
    let homophily = if !subgraph.same_type_nodes.is_empty() {
        compute_homophily_motifs(
            config.homophily_hop as u8,
            &subgraph.same_type_nodes,
            neighbor_predictions,
        )
    } else {
        None
    };

    EntityMotifs {
        entity_id: subgraph.root_id.clone(),
        relation_motifs,
        attribute_motifs,
        homophily,
    }
}

/// Compute structural and temporal motifs for a relation
fn compute_relation_motifs(
    rel_name: &str,
    edges: &[ExtractedEdge],
    anchor_time: i64,
    config: &MotifConfig,
) -> RelationMotifs {
    let edge_count = edges.len() as u32;

    // Unique destinations
    let unique_destinations = edges
        .iter()
        .map(|e| e.dst_node_id)
        .collect::<std::collections::HashSet<_>>()
        .len() as u32;

    // Get destination table (assume all edges in a relation go to same table)
    let dst_table = edges
        .first()
        .map(|e| e.dst_table.clone())
        .unwrap_or_default();

    // Temporal motifs
    let timestamps: Vec<i64> = edges.iter().filter_map(|e| e.timestamp).collect();
    let temporal = compute_temporal_motifs(&timestamps, anchor_time, config.min_edges_for_trend);

    RelationMotifs {
        relation_name: rel_name.to_string(),
        dst_table,
        edge_count,
        unique_destinations,
        temporal,
    }
}

/// Compute temporal motifs from timestamps
pub fn compute_temporal_motifs(
    timestamps: &[i64],
    anchor_time: i64,
    min_points_for_trend: usize,
) -> Option<TemporalMotifs> {
    if timestamps.is_empty() {
        return None;
    }

    let mut sorted = timestamps.to_vec();
    sorted.sort();

    let count = sorted.len() as u32;
    let timestamp_min = sorted[0];
    let timestamp_max = sorted[sorted.len() - 1];

    let recency = anchor_time - timestamp_max;
    let span = timestamp_max - timestamp_min;

    // Inter-arrival times
    let inter_arrivals: Vec<f64> = sorted.windows(2).map(|w| (w[1] - w[0]) as f64).collect();

    let inter_arrival_mean = if !inter_arrivals.is_empty() {
        Some(inter_arrivals.iter().sum::<f64>() / inter_arrivals.len() as f64)
    } else {
        None
    };

    let inter_arrival_std = if inter_arrivals.len() > 1 {
        let mean = inter_arrival_mean.unwrap();
        let variance = inter_arrivals
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (inter_arrivals.len() - 1) as f64;
        Some(variance.sqrt())
    } else {
        None
    };

    // Trend slope via simple comparison of first half vs second half
    let trend_slope = if sorted.len() >= min_points_for_trend {
        Some(compute_trend_slope(&sorted))
    } else {
        None
    };

    Some(TemporalMotifs {
        recency,
        span,
        count,
        inter_arrival_mean,
        inter_arrival_std,
        trend_slope,
        timestamp_min,
        timestamp_max,
    })
}

/// Compute trend slope: positive = accelerating, negative = decelerating
fn compute_trend_slope(timestamps: &[i64]) -> f64 {
    let n = timestamps.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let t_min = timestamps[0] as f64;
    let t_max = timestamps[timestamps.len() - 1] as f64;
    let t_range = (t_max - t_min).max(1.0);

    // Divide into two halves and compare activity
    let midpoint = t_min + t_range / 2.0;
    let first_half = timestamps
        .iter()
        .filter(|&&t| (t as f64) < midpoint)
        .count() as f64;
    let second_half = timestamps
        .iter()
        .filter(|&&t| (t as f64) >= midpoint)
        .count() as f64;

    // Return relative change
    if first_half > 0.0 {
        (second_half - first_half) / first_half
    } else if second_half > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Compute attribute motifs for all nodes in a table
fn compute_attribute_motifs_for_table(
    table: &str,
    nodes: &[ExtractedNode],
    config: &MotifConfig,
    output: &mut HashMap<String, AttributeMotif>,
) {
    // Check if a column should be forced to categorical
    let is_force_categorical = |col: &str| -> bool {
        config.force_categorical.iter().any(|fc| {
            // Match "column" or "table.column"
            fc == col || fc == &format!("{}.{}", table, col)
        })
    };

    // Collect all values per column
    let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
    let mut categorical_values: HashMap<String, Vec<String>> = HashMap::new();

    for node in nodes {
        for (col_name, value) in &node.attributes {
            // Check if this column should be forced to categorical
            if is_force_categorical(col_name) {
                // Convert any value to string for categorical treatment
                let str_val = match value {
                    AttributeValue::Numeric(v) => {
                        // Format as integer if it's a whole number (likely an ID)
                        if v.fract() == 0.0 {
                            format!("{}", *v as i64)
                        } else {
                            format!("{}", v)
                        }
                    }
                    AttributeValue::Categorical(v) => v.clone(),
                    AttributeValue::Boolean(b) => b.to_string(),
                    AttributeValue::Missing => continue,
                };
                categorical_values
                    .entry(col_name.clone())
                    .or_default()
                    .push(str_val);
                continue;
            }

            match value {
                AttributeValue::Numeric(v) => {
                    numeric_values.entry(col_name.clone()).or_default().push(*v);
                }
                AttributeValue::Categorical(v) => {
                    categorical_values
                        .entry(col_name.clone())
                        .or_default()
                        .push(v.clone());
                }
                AttributeValue::Boolean(b) => {
                    // Treat boolean as categorical
                    categorical_values
                        .entry(col_name.clone())
                        .or_default()
                        .push(b.to_string());
                }
                AttributeValue::Missing => {}
            }
        }
    }

    // Compute numeric motifs
    for (col_name, values) in numeric_values {
        if let Some(motifs) = NumericMotifs::from_values(&values) {
            let path = format!("{}.{}", table, col_name);
            output.insert(
                path.clone(),
                AttributeMotif {
                    path,
                    motif_type: AttributeMotifType::Numeric(motifs),
                },
            );
        }
    }

    // Compute categorical motifs
    for (col_name, values) in categorical_values {
        // Check cardinality limit
        let unique_count = values
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if unique_count > config.max_categorical_cardinality {
            continue;
        }

        if let Some(motifs) = CategoricalMotifs::from_values(&values) {
            let path = format!("{}.{}", table, col_name);
            output.insert(
                path.clone(),
                AttributeMotif {
                    path,
                    motif_type: AttributeMotifType::Categorical(motifs),
                },
            );
        }
    }
}

/// Compute homophily motifs from neighbor predictions
pub fn compute_homophily_motifs(
    hop: u8,
    neighbor_ids: &[u64],
    predictions: &HashMap<u64, f64>,
) -> Option<HomophilyMotifs> {
    let neighbor_preds: Vec<f64> = neighbor_ids
        .iter()
        .filter_map(|id| predictions.get(id).copied())
        .collect();

    if neighbor_preds.is_empty() {
        return Some(HomophilyMotifs {
            hop,
            neighbor_count: neighbor_ids.len() as u32,
            prediction_mean: None,
            prediction_std: None,
            prediction_min: None,
            prediction_max: None,
        });
    }

    let neighbor_count = neighbor_preds.len() as u32;
    let prediction_mean = neighbor_preds.iter().sum::<f64>() / neighbor_count as f64;

    let prediction_std = if neighbor_preds.len() > 1 {
        let variance = neighbor_preds
            .iter()
            .map(|p| (p - prediction_mean).powi(2))
            .sum::<f64>()
            / (neighbor_preds.len() - 1) as f64;
        Some(variance.sqrt())
    } else {
        None
    };

    let prediction_min = neighbor_preds.iter().cloned().fold(f64::INFINITY, f64::min);
    let prediction_max = neighbor_preds
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    Some(HomophilyMotifs {
        hop,
        neighbor_count,
        prediction_mean: Some(prediction_mean),
        prediction_std,
        prediction_min: Some(prediction_min),
        prediction_max: Some(prediction_max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_motifs() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let motifs = NumericMotifs::from_values(&values).unwrap();
        assert_eq!(motifs.count, 5);
        assert!((motifs.mean - 3.0).abs() < 0.001);
        assert!((motifs.sum - 15.0).abs() < 0.001);
        assert!((motifs.min - 1.0).abs() < 0.001);
        assert!((motifs.max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_categorical_motifs() {
        let values = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "a".to_string(),
        ];
        let motifs = CategoricalMotifs::from_values(&values).unwrap();
        assert_eq!(motifs.count, 4);
        assert_eq!(motifs.nunique, 2);
        assert_eq!(motifs.mode, "a");
        assert_eq!(motifs.mode_count, 3);
    }

    #[test]
    fn test_temporal_motifs() {
        let timestamps = vec![100, 200, 300, 400, 500];
        let anchor_time = 600;
        let motifs = compute_temporal_motifs(&timestamps, anchor_time, 3).unwrap();

        assert_eq!(motifs.count, 5);
        assert_eq!(motifs.recency, 100);
        assert_eq!(motifs.span, 400);
        assert!((motifs.inter_arrival_mean.unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_trend_slope_increasing() {
        // More activity in second half
        let timestamps = vec![100, 400, 450, 480, 490];
        let slope = compute_trend_slope(&timestamps);
        assert!(slope > 0.0); // Increasing activity
    }

    #[test]
    fn test_trend_slope_decreasing() {
        // More activity in first half
        let timestamps = vec![100, 110, 120, 400, 500];
        let slope = compute_trend_slope(&timestamps);
        assert!(slope < 0.0); // Decreasing activity
    }
}
