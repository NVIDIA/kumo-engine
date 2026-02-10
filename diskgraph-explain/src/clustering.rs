//! K-means clustering on entity embeddings.

use crate::output::{ClusterContext, ClusterProfileFeature};
use crate::population::{FeatureContext, PopulationStats};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

/// K-means clustering result
pub struct ClusteringResult {
    pub centroids: Vec<Vec<f32>>,
    pub cluster_stats: Vec<ClusterStats>,
    /// Entity assignments: entity index -> cluster id
    pub assignments: Vec<usize>,
    /// Distance from each entity to its assigned centroid
    pub distances: Vec<f32>,
}

/// Statistics for a single cluster
#[derive(Clone, Default)]
pub struct ClusterStats {
    pub count: usize,
    pub prediction_sum: f64,
    pub prediction_sum_sq: f64,
    /// Feature values for cluster profile: feature -> (sum, count)
    pub feature_sums: HashMap<String, (f64, usize)>,
    /// Feature values from archetype sample (N nearest to centroid): feature -> values
    pub archetype_values: HashMap<String, Vec<f64>>,
}

impl ClusterStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, prediction: f64) {
        self.count += 1;
        self.prediction_sum += prediction;
        self.prediction_sum_sq += prediction * prediction;
    }

    pub fn add_feature(&mut self, feature: &str, value: f64) {
        let entry = self
            .feature_sums
            .entry(feature.to_string())
            .or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    pub fn add_archetype_feature(&mut self, feature: &str, value: f64) {
        self.archetype_values
            .entry(feature.to_string())
            .or_default()
            .push(value);
    }

    /// Get archetype mean for a feature (from sampled nearest entities)
    pub fn archetype_mean(&self, feature: &str) -> Option<f64> {
        self.archetype_values.get(feature).and_then(|values| {
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
        })
    }

    /// Get archetype std for a feature
    pub fn archetype_std(&self, feature: &str) -> Option<f64> {
        self.archetype_values.get(feature).and_then(|values| {
            if values.len() < 2 {
                None
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                Some(variance.sqrt())
            }
        })
    }

    pub fn mean_prediction(&self) -> f64 {
        if self.count > 0 {
            self.prediction_sum / self.count as f64
        } else {
            0.0
        }
    }

    pub fn std_prediction(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean_prediction();
            let variance = (self.prediction_sum_sq / self.count as f64) - mean * mean;
            variance.max(0.0).sqrt()
        } else {
            0.0
        }
    }
}

/// Run K-means clustering on embeddings
pub fn run_kmeans(
    embeddings: &[Vec<f32>],
    predictions: &[f64],
    k: usize,
    iterations: usize,
) -> ClusteringResult {
    if embeddings.is_empty() || k == 0 {
        return ClusteringResult {
            centroids: Vec::new(),
            cluster_stats: Vec::new(),
            assignments: Vec::new(),
            distances: Vec::new(),
        };
    }

    let dim = embeddings[0].len();

    // Initialize centroids from random samples
    let mut centroids = initialize_centroids(embeddings, k);

    // Run Lloyd's algorithm
    for _ in 0..iterations {
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for emb in embeddings {
            let (cluster, _) = nearest_centroid(emb, &centroids);
            counts[cluster] += 1;
            for (j, v) in emb.iter().enumerate() {
                new_centroids[cluster][j] += v;
            }
        }

        // Update centroids
        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for v in centroid.iter_mut() {
                    *v /= counts[i] as f32;
                }
                centroids[i] = centroid.clone();
            }
        }
    }

    // Compute final assignments and distances
    let mut assignments = Vec::with_capacity(embeddings.len());
    let mut distances = Vec::with_capacity(embeddings.len());

    for emb in embeddings {
        let (cluster, dist) = nearest_centroid(emb, &centroids);
        assignments.push(cluster);
        distances.push(dist);
    }

    // Compute cluster statistics
    let mut cluster_stats: Vec<ClusterStats> = (0..k).map(|_| ClusterStats::new()).collect();

    for (&cluster, &pred) in assignments.iter().zip(predictions.iter()) {
        cluster_stats[cluster].add(pred);
    }

    ClusteringResult {
        centroids,
        cluster_stats,
        assignments,
        distances,
    }
}

/// Find N nearest entities to each centroid for archetype profiling
pub fn find_archetype_samples(
    clustering: &ClusteringResult,
    n_per_cluster: usize,
) -> Vec<Vec<usize>> {
    let k = clustering.centroids.len();
    let mut result = vec![Vec::new(); k];

    // Group entities by cluster with their distances
    let mut cluster_entities: Vec<Vec<(usize, f32)>> = vec![Vec::new(); k];
    for (idx, (&cluster, &dist)) in clustering
        .assignments
        .iter()
        .zip(clustering.distances.iter())
        .enumerate()
    {
        cluster_entities[cluster].push((idx, dist));
    }

    // Sort each cluster by distance and take top N
    for (cluster_id, entities) in cluster_entities.iter_mut().enumerate() {
        entities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result[cluster_id] = entities
            .iter()
            .take(n_per_cluster)
            .map(|(idx, _)| *idx)
            .collect();
    }

    result
}

/// Initialize centroids using k-means++ like approach
fn initialize_centroids(embeddings: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    let dim = embeddings[0].len();

    if embeddings.len() <= k {
        let mut centroids = embeddings.to_vec();
        while centroids.len() < k {
            centroids.push(vec![0.0; dim]);
        }
        return centroids;
    }

    // Simple approach: shuffle and take first k
    let mut indices: Vec<usize> = (0..embeddings.len()).collect();
    indices.shuffle(&mut rng);

    indices
        .iter()
        .take(k)
        .map(|&i| embeddings[i].clone())
        .collect()
}

/// Find nearest centroid to a vector
pub fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_dist = f32::MAX;

    for (idx, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance(vec, centroid);
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }

    (best, best_dist)
}

/// Compute L2 distance between two vectors
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Build cluster context for an entity using z-score based matching
pub fn build_cluster_context(
    cluster_id: usize,
    cluster_stats: &ClusterStats,
    entity_contexts: &[FeatureContext],
    population_stats: &PopulationStats,
    top_k: usize,
) -> ClusterContext {
    // Build cluster profile from archetype samples (or fall back to all members)
    let mut profile_features: Vec<ClusterProfileFeature> = Vec::new();
    let mut entity_matches = Vec::new();
    let mut entity_differs = Vec::new();

    // Collect features from archetype values (preferred) or feature_sums (fallback)
    let feature_names: std::collections::HashSet<_> = cluster_stats
        .archetype_values
        .keys()
        .chain(cluster_stats.feature_sums.keys())
        .collect();

    for feature in feature_names {
        // Get cluster mean/std from archetype if available, else from all members
        let (cluster_mean, cluster_std) = if let (Some(mean), std) = (
            cluster_stats.archetype_mean(feature),
            cluster_stats.archetype_std(feature),
        ) {
            (mean, std.unwrap_or(0.0))
        } else if let Some(&(sum, count)) = cluster_stats.feature_sums.get(feature) {
            if count > 0 {
                (sum / count as f64, 0.0)
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Get population stats for this feature
        let pop_stats = match population_stats.feature_stats.get(feature) {
            Some(s) => s,
            None => continue,
        };

        // Compute cluster z-score vs population
        let cluster_z_score = if pop_stats.std > 0.0 {
            (cluster_mean - pop_stats.mean) / pop_stats.std
        } else {
            0.0
        };

        // Only include features where cluster is distinctive (|z| > 0.5)
        if cluster_z_score.abs() < 0.5 {
            continue;
        }

        // Get conditional rate for cluster's typical decile
        let mut decile = 0;
        for (i, &boundary) in pop_stats.decile_boundaries.iter().enumerate() {
            if cluster_mean > boundary {
                decile = i + 1;
            }
        }
        let conditional_rate = pop_stats
            .decile_conditional_rates
            .get(decile)
            .copied()
            .unwrap_or(population_stats.prediction_mean);

        profile_features.push(ClusterProfileFeature {
            feature: feature.clone(),
            cluster_mean,
            cluster_std,
            cluster_z_score,
            conditional_rate,
        });
    }

    // Sort by absolute cluster z-score (most distinctive features first)
    profile_features.sort_by(|a, b| {
        b.cluster_z_score
            .abs()
            .partial_cmp(&a.cluster_z_score.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    profile_features.truncate(top_k);

    // Check which profile features the entity matches/differs using entity_z_in_cluster
    // entity_z_in_cluster = (entity_value - cluster_mean) / cluster_std
    let profile_feature_map: HashMap<&str, &ClusterProfileFeature> = profile_features
        .iter()
        .map(|f| (f.feature.as_str(), f))
        .collect();

    for ctx in entity_contexts {
        if let Some(cluster_feat) = profile_feature_map.get(ctx.feature.as_str()) {
            // Compute entity's z-score within the cluster
            let entity_z_in_cluster = if cluster_feat.cluster_std > 0.0 {
                (ctx.value - cluster_feat.cluster_mean) / cluster_feat.cluster_std
            } else {
                0.0 // If no variance, entity matches by default
            };

            if entity_z_in_cluster.abs() < 1.0 {
                // Entity is within 1 std of cluster mean -> matches
                entity_matches.push(ctx.feature.clone());
            } else if entity_z_in_cluster.abs() > 2.0 {
                // Entity is more than 2 std from cluster mean -> deviates
                entity_differs.push(ctx.feature.clone());
            }
            // Features with 1.0 <= |z| <= 2.0 are neither match nor differ
        }
    }

    // Compute fit score based on z-score matching
    let total_profile = profile_features.len();
    let fit_score = if total_profile > 0 {
        entity_matches.len() as f32 / total_profile as f32
    } else {
        1.0
    };

    ClusterContext {
        cluster_id,
        cluster_size: cluster_stats.count,
        cluster_mean_prediction: cluster_stats.mean_prediction(),
        cluster_std_prediction: cluster_stats.std_prediction(),
        fit_score,
        profile: profile_features,
        entity_matches,
        entity_differs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let predictions = vec![0.2, 0.3, 0.8, 0.9];

        let result = run_kmeans(&embeddings, &predictions, 2, 10);

        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.cluster_stats.len(), 2);

        // One cluster should have low predictions, one should have high
        let means: Vec<f64> = result
            .cluster_stats
            .iter()
            .map(|s| s.mean_prediction())
            .collect();
        assert!(means[0] < 0.5 || means[1] < 0.5);
        assert!(means[0] > 0.5 || means[1] > 0.5);
    }

    #[test]
    fn test_nearest_centroid() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        let (idx, _) = nearest_centroid(&[1.0, 1.0], &centroids);
        assert_eq!(idx, 0);

        let (idx, _) = nearest_centroid(&[9.0, 9.0], &centroids);
        assert_eq!(idx, 1);
    }
}
