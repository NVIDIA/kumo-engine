//! Population statistics for contextualizing entity motifs.
//!
//! Computes:
//! - Decile boundaries for each motif
//! - Conditional rates (mean prediction per decile)
//! - Z-scores for comparing entity values to population

use crate::motifs::{AttributeMotifType, EntityMotifs};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Population statistics for all motifs
#[derive(Debug, Clone, Default)]
pub struct PopulationStats {
    /// Stats per feature key (e.g., "relation:customer_transactions:edge_count")
    pub feature_stats: HashMap<String, FeatureStats>,

    /// Global prediction statistics
    pub prediction_mean: f64,
    pub prediction_std: f64,
    pub prediction_min: f64,
    pub prediction_max: f64,
    pub total_entities: usize,
}

/// Statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub feature_name: String,
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    /// Decile boundaries (9 values for 10 deciles)
    pub decile_boundaries: Vec<f64>,
    /// Conditional rate (mean prediction) per decile
    pub decile_conditional_rates: Vec<f64>,
}

/// Context for a single feature value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContext {
    pub feature: String,
    pub value: f64,
    pub z_score: f64,
    pub decile: usize,
    pub decile_min: f64,
    pub decile_max: f64,
    pub conditional_rate: f64,
    pub population_rate: f64,
    pub population_mean: f64,
    pub population_std: f64,
    pub population_min: f64,
    pub population_max: f64,
}

/// Builder for collecting population statistics
#[derive(Default)]
pub struct PopulationStatsBuilder {
    /// (feature_name, value, prediction) tuples
    feature_values: HashMap<String, Vec<(f64, f64)>>,
    predictions: Vec<f64>,
}

impl PopulationStatsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity's motifs and prediction to the builder
    pub fn add_entity(&mut self, motifs: &EntityMotifs, prediction: f64) {
        self.predictions.push(prediction);

        // Extract numeric values from relation motifs
        for (rel_name, rel_motifs) in &motifs.relation_motifs {
            let edge_count_key = format!("relation:{}:edge_count", rel_name);
            self.feature_values
                .entry(edge_count_key)
                .or_default()
                .push((rel_motifs.edge_count as f64, prediction));

            let unique_dst_key = format!("relation:{}:unique_destinations", rel_name);
            self.feature_values
                .entry(unique_dst_key)
                .or_default()
                .push((rel_motifs.unique_destinations as f64, prediction));

            // Temporal motifs
            if let Some(temporal) = &rel_motifs.temporal {
                let recency_key = format!("relation:{}:recency", rel_name);
                self.feature_values
                    .entry(recency_key)
                    .or_default()
                    .push((temporal.recency as f64, prediction));

                let span_key = format!("relation:{}:span", rel_name);
                self.feature_values
                    .entry(span_key)
                    .or_default()
                    .push((temporal.span as f64, prediction));

                if let Some(inter_arrival_mean) = temporal.inter_arrival_mean {
                    let key = format!("relation:{}:inter_arrival_mean", rel_name);
                    self.feature_values
                        .entry(key)
                        .or_default()
                        .push((inter_arrival_mean, prediction));
                }

                if let Some(inter_arrival_std) = temporal.inter_arrival_std {
                    let key = format!("relation:{}:inter_arrival_std", rel_name);
                    self.feature_values
                        .entry(key)
                        .or_default()
                        .push((inter_arrival_std, prediction));
                }

                if let Some(trend_slope) = temporal.trend_slope {
                    let key = format!("relation:{}:trend_slope", rel_name);
                    self.feature_values
                        .entry(key)
                        .or_default()
                        .push((trend_slope, prediction));
                }
            }
        }

        // Extract numeric values from attribute motifs
        for (path, attr_motif) in &motifs.attribute_motifs {
            match &attr_motif.motif_type {
                AttributeMotifType::Numeric(num) => {
                    let sum_key = format!("{}:sum", path);
                    self.feature_values
                        .entry(sum_key)
                        .or_default()
                        .push((num.sum, prediction));

                    let mean_key = format!("{}:mean", path);
                    self.feature_values
                        .entry(mean_key)
                        .or_default()
                        .push((num.mean, prediction));

                    let std_key = format!("{}:std", path);
                    self.feature_values
                        .entry(std_key)
                        .or_default()
                        .push((num.std, prediction));

                    let min_key = format!("{}:min", path);
                    self.feature_values
                        .entry(min_key)
                        .or_default()
                        .push((num.min, prediction));

                    let max_key = format!("{}:max", path);
                    self.feature_values
                        .entry(max_key)
                        .or_default()
                        .push((num.max, prediction));
                }
                AttributeMotifType::Categorical(cat) => {
                    let nunique_key = format!("{}:nunique", path);
                    self.feature_values
                        .entry(nunique_key)
                        .or_default()
                        .push((cat.nunique as f64, prediction));

                    let count_key = format!("{}:count", path);
                    self.feature_values
                        .entry(count_key)
                        .or_default()
                        .push((cat.count as f64, prediction));
                }
            }
        }

        // Homophily motifs
        if let Some(homophily) = &motifs.homophily {
            let key = "homophily:neighbor_count".to_string();
            self.feature_values
                .entry(key)
                .or_default()
                .push((homophily.neighbor_count as f64, prediction));

            if let Some(pred_mean) = homophily.prediction_mean {
                let key = "homophily:prediction_mean".to_string();
                self.feature_values
                    .entry(key)
                    .or_default()
                    .push((pred_mean, prediction));
            }
        }
    }

    /// Build the final population statistics
    pub fn build(self, num_deciles: usize) -> PopulationStats {
        let mut feature_stats = HashMap::new();

        for (feature_name, values) in self.feature_values {
            if values.len() < num_deciles {
                continue; // Not enough data for deciles
            }

            let stats = compute_feature_stats(&feature_name, &values, num_deciles);
            feature_stats.insert(feature_name, stats);
        }

        // Compute global prediction stats
        let (pred_mean, pred_std, pred_min, pred_max) = if !self.predictions.is_empty() {
            let mean = self.predictions.iter().sum::<f64>() / self.predictions.len() as f64;
            let variance = self
                .predictions
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / (self.predictions.len() - 1).max(1) as f64;
            let std = variance.sqrt();
            let min = self
                .predictions
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = self
                .predictions
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            (mean, std, min, max)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        PopulationStats {
            feature_stats,
            prediction_mean: pred_mean,
            prediction_std: pred_std,
            prediction_min: pred_min,
            prediction_max: pred_max,
            total_entities: self.predictions.len(),
        }
    }
}

/// Compute statistics for a single feature
fn compute_feature_stats(
    feature_name: &str,
    values: &[(f64, f64)],
    num_deciles: usize,
) -> FeatureStats {
    let n = values.len();

    // Extract just the feature values for basic stats
    let feature_values: Vec<f64> = values.iter().map(|(v, _)| *v).collect();

    let mean = feature_values.iter().sum::<f64>() / n as f64;
    let variance = feature_values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;
    let std = variance.sqrt();
    let min = feature_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = feature_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Sort by feature value to compute deciles
    let mut sorted: Vec<(f64, f64)> = values.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute decile boundaries
    let mut decile_boundaries = Vec::with_capacity(num_deciles - 1);
    for i in 1..num_deciles {
        let idx = (i * n) / num_deciles;
        decile_boundaries.push(sorted[idx.min(n - 1)].0);
    }

    // Compute conditional rates per decile
    let mut decile_conditional_rates = Vec::with_capacity(num_deciles);
    let decile_size = n / num_deciles;

    for i in 0..num_deciles {
        let start = i * decile_size;
        let end = if i == num_deciles - 1 {
            n
        } else {
            (i + 1) * decile_size
        };

        let decile_predictions: Vec<f64> = sorted[start..end].iter().map(|(_, p)| *p).collect();
        let decile_mean = if !decile_predictions.is_empty() {
            decile_predictions.iter().sum::<f64>() / decile_predictions.len() as f64
        } else {
            0.0
        };
        decile_conditional_rates.push(decile_mean);
    }

    FeatureStats {
        feature_name: feature_name.to_string(),
        count: n,
        mean,
        std,
        min,
        max,
        decile_boundaries,
        decile_conditional_rates,
    }
}

impl PopulationStats {
    /// Get context for a feature value
    pub fn get_feature_context(&self, feature_name: &str, value: f64) -> Option<FeatureContext> {
        let stats = self.feature_stats.get(feature_name)?;

        // Compute z-score
        let z_score = if stats.std > 0.0 {
            (value - stats.mean) / stats.std
        } else {
            0.0
        };

        // Find decile
        let mut decile = 1;
        for (i, &boundary) in stats.decile_boundaries.iter().enumerate() {
            if value > boundary {
                decile = i + 2;
            } else {
                break;
            }
        }
        decile = decile.min(stats.decile_conditional_rates.len());

        // Get decile boundaries
        let decile_min = if decile == 1 {
            stats.min
        } else {
            stats.decile_boundaries[decile - 2]
        };
        let decile_max = if decile > stats.decile_boundaries.len() {
            stats.max
        } else {
            stats.decile_boundaries[decile - 1]
        };

        // Get conditional rate for this decile
        let conditional_rate = stats
            .decile_conditional_rates
            .get(decile - 1)
            .copied()
            .unwrap_or(0.0);

        Some(FeatureContext {
            feature: feature_name.to_string(),
            value,
            z_score,
            decile,
            decile_min,
            decile_max,
            conditional_rate,
            population_rate: self.prediction_mean,
            population_mean: stats.mean,
            population_std: stats.std,
            population_min: stats.min,
            population_max: stats.max,
        })
    }

    /// Get all feature contexts for an entity's motifs
    pub fn contextualize_motifs(&self, motifs: &EntityMotifs) -> Vec<FeatureContext> {
        let mut contexts = Vec::new();

        // Relation motifs
        for (rel_name, rel_motifs) in &motifs.relation_motifs {
            // Edge count
            let key = format!("relation:{}:edge_count", rel_name);
            if let Some(ctx) = self.get_feature_context(&key, rel_motifs.edge_count as f64) {
                contexts.push(ctx);
            }

            // Unique destinations
            let key = format!("relation:{}:unique_destinations", rel_name);
            if let Some(ctx) = self.get_feature_context(&key, rel_motifs.unique_destinations as f64)
            {
                contexts.push(ctx);
            }

            // Temporal motifs
            if let Some(temporal) = &rel_motifs.temporal {
                let key = format!("relation:{}:recency", rel_name);
                if let Some(ctx) = self.get_feature_context(&key, temporal.recency as f64) {
                    contexts.push(ctx);
                }

                let key = format!("relation:{}:span", rel_name);
                if let Some(ctx) = self.get_feature_context(&key, temporal.span as f64) {
                    contexts.push(ctx);
                }

                if let Some(v) = temporal.inter_arrival_mean {
                    let key = format!("relation:{}:inter_arrival_mean", rel_name);
                    if let Some(ctx) = self.get_feature_context(&key, v) {
                        contexts.push(ctx);
                    }
                }

                if let Some(v) = temporal.inter_arrival_std {
                    let key = format!("relation:{}:inter_arrival_std", rel_name);
                    if let Some(ctx) = self.get_feature_context(&key, v) {
                        contexts.push(ctx);
                    }
                }

                if let Some(v) = temporal.trend_slope {
                    let key = format!("relation:{}:trend_slope", rel_name);
                    if let Some(ctx) = self.get_feature_context(&key, v) {
                        contexts.push(ctx);
                    }
                }
            }
        }

        // Attribute motifs
        for (path, attr_motif) in &motifs.attribute_motifs {
            match &attr_motif.motif_type {
                AttributeMotifType::Numeric(num) => {
                    for (suffix, value) in [
                        ("sum", num.sum),
                        ("mean", num.mean),
                        ("std", num.std),
                        ("min", num.min),
                        ("max", num.max),
                    ] {
                        let key = format!("{}:{}", path, suffix);
                        if let Some(ctx) = self.get_feature_context(&key, value) {
                            contexts.push(ctx);
                        }
                    }
                }
                AttributeMotifType::Categorical(cat) => {
                    let key = format!("{}:nunique", path);
                    if let Some(ctx) = self.get_feature_context(&key, cat.nunique as f64) {
                        contexts.push(ctx);
                    }

                    let key = format!("{}:count", path);
                    if let Some(ctx) = self.get_feature_context(&key, cat.count as f64) {
                        contexts.push(ctx);
                    }
                }
            }
        }

        // Homophily
        if let Some(homophily) = &motifs.homophily {
            let key = "homophily:neighbor_count".to_string();
            if let Some(ctx) = self.get_feature_context(&key, homophily.neighbor_count as f64) {
                contexts.push(ctx);
            }

            if let Some(pred_mean) = homophily.prediction_mean {
                let key = "homophily:prediction_mean".to_string();
                if let Some(ctx) = self.get_feature_context(&key, pred_mean) {
                    contexts.push(ctx);
                }
            }
        }

        contexts
    }

    /// Get top drivers sorted by deviation from population rate
    pub fn get_top_drivers(&self, contexts: &[FeatureContext], k: usize) -> Vec<FeatureContext> {
        let mut sorted = contexts.to_vec();
        sorted.sort_by(|a, b| {
            let a_dev = (a.conditional_rate - a.population_rate).abs();
            let b_dev = (b.conditional_rate - b.population_rate).abs();
            b_dev
                .partial_cmp(&a_dev)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }

    /// Filter features for narrative relevance based on z-scores and conditional rate ratio
    ///
    /// A feature is narratively relevant if:
    /// - |z_score| > z_threshold: Global outlier (significantly above/below population)
    /// - conditional_rate / population_rate > ratio_threshold: Strong predictive signal
    /// - conditional_rate / population_rate < 1/ratio_threshold: Strong inverse signal
    pub fn filter_for_narrative_relevance(
        &self,
        contexts: &[FeatureContext],
        z_threshold: f64,
        rate_ratio_threshold: f64,
    ) -> Vec<FeatureContext> {
        contexts
            .iter()
            .filter(|ctx| {
                // Global outlier check
                let is_global_outlier = ctx.z_score.abs() > z_threshold;

                // Predictive signal check (conditional rate ratio)
                let rate_ratio = if ctx.population_rate > 0.0 {
                    ctx.conditional_rate / ctx.population_rate
                } else {
                    1.0
                };
                let is_predictive =
                    rate_ratio > rate_ratio_threshold || rate_ratio < (1.0 / rate_ratio_threshold);

                is_global_outlier || is_predictive
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decile_computation() {
        let values: Vec<(f64, f64)> = (0..100)
            .map(|i| (i as f64, if i < 50 { 0.3 } else { 0.7 }))
            .collect();

        let stats = compute_feature_stats("test", &values, 10);

        assert_eq!(stats.decile_boundaries.len(), 9);
        assert_eq!(stats.decile_conditional_rates.len(), 10);

        // First 5 deciles should have lower conditional rate
        for rate in &stats.decile_conditional_rates[..5] {
            assert!(*rate < 0.5);
        }
        // Last 5 deciles should have higher conditional rate
        for rate in &stats.decile_conditional_rates[5..] {
            assert!(*rate > 0.5);
        }
    }

    #[test]
    fn test_z_score() {
        let values: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, 0.5)).collect();

        let stats = compute_feature_stats("test", &values, 10);

        // Mean should be ~49.5, std should be ~29
        assert!((stats.mean - 49.5).abs() < 0.1);

        let pop_stats = PopulationStats {
            feature_stats: [("test".to_string(), stats)].into_iter().collect(),
            prediction_mean: 0.5,
            prediction_std: 0.1,
            prediction_min: 0.0,
            prediction_max: 1.0,
            total_entities: 100,
        };

        // Value at mean should have z_score ~0
        let ctx = pop_stats.get_feature_context("test", 49.5).unwrap();
        assert!(ctx.z_score.abs() < 0.1);

        // Value well above mean should have positive z_score
        let ctx = pop_stats.get_feature_context("test", 90.0).unwrap();
        assert!(ctx.z_score > 1.0);
    }
}
