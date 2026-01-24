use crate::config::ExplainConfig;
use crate::parquet_io::{
    explanation_schema, trends_schema, ParquetStream, write_parquet,
};
use anyhow::{anyhow, Result};
use arrow_array::{
    ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub fn run(config: ExplainConfig) -> Result<()> {
    let (dim, initial_samples) = sample_embeddings(&config.input_parquet, &config, config.num_clusters)?;
    if dim == 0 {
        return Err(anyhow!("embedding dimension is 0"));
    }
    let mut centroids = initialize_centroids(dim, config.num_clusters, initial_samples);
    let mut counts = vec![0u64; config.num_clusters];

    for _ in 0..config.kmeans_passes {
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
                let emb = batch.embedding_as_f32(i)?;
                let (best, _) = nearest_centroid(&emb, &centroids);
                counts[best] += 1;
                let lr = 1.0f32 / counts[best].max(1) as f32;
                let c = &mut centroids[best];
                for (j, v) in emb.iter().enumerate() {
                    c[j] += (v - c[j]) * lr;
                }
            }
        }
    }

    let mut ridge = RidgeProbe::new(dim, config.ridge_lambda as f32);
    let mut global_pred = RunningStats::new();
    let mut cluster_stats = Vec::with_capacity(config.num_clusters);
    for _ in 0..config.num_clusters {
        cluster_stats.push(ClusterStats::new(config.cluster_sample_size));
    }

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
                let emb = batch.embedding_as_f32(i)?;
                let pred = batch.prediction_as_f64(i)? as f32;
                global_pred.add(pred);
                ridge.update(&emb, pred);
                let (cluster, dist2) = nearest_centroid(&emb, &centroids);
                cluster_stats[cluster].add(pred, dist2.sqrt(), batch.timestamp_as_i64(i)?, config.time_bucket_secs);
            }
        }
    }

    let ridge_weights = ridge.weights();

    let mut explanations_batches: Vec<RecordBatch> = Vec::new();
    let mut trends_rows: Vec<TrendRow> = Vec::new();

    let mut knn_reservoirs: Vec<Reservoir> = (0..config.num_clusters)
        .map(|_| Reservoir::new(config.knn_sample_size))
        .collect();

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
        let mut ids: Vec<String> = Vec::with_capacity(batch.len());
        let mut ts: Vec<i64> = Vec::with_capacity(batch.len());
        let mut preds: Vec<f32> = Vec::with_capacity(batch.len());
        let mut cluster_ids: Vec<i32> = Vec::with_capacity(batch.len());
        let mut cluster_mean: Vec<f32> = Vec::with_capacity(batch.len());
        let mut cluster_std: Vec<f32> = Vec::with_capacity(batch.len());
        let mut cluster_rank: Vec<f32> = Vec::with_capacity(batch.len());
        let mut knn_mean: Vec<f32> = Vec::with_capacity(batch.len());
        let mut knn_std: Vec<f32> = Vec::with_capacity(batch.len());
        let mut knn_p90: Vec<f32> = Vec::with_capacity(batch.len());
        let mut local_lift: Vec<f32> = Vec::with_capacity(batch.len());
        let mut global_lift: Vec<f32> = Vec::with_capacity(batch.len());
        let mut emb_outlier: Vec<f32> = Vec::with_capacity(batch.len());
        let mut pred_outlier: Vec<f32> = Vec::with_capacity(batch.len());
        let mut top_factors: Vec<String> = Vec::with_capacity(batch.len());
        let mut lookalikes: Vec<String> = Vec::with_capacity(batch.len());

        for i in 0..batch.len() {
            let id = batch.id_as_string(i)?;
            let timestamp = batch.timestamp_as_i64(i)?;
            let pred = batch.prediction_as_f64(i)? as f32;
            let emb = batch.embedding_as_f32(i)?;
            let (cluster, dist2) = nearest_centroid(&emb, &centroids);
            let stats = &cluster_stats[cluster];

            let knn = knn_reservoirs[cluster].query_knn(&emb, config.knn_k);
            let knn_stats = RunningStats::from_values(&knn.iter().map(|n| n.pred).collect::<Vec<_>>());
            let knn_p90_val = percentile(&knn.iter().map(|n| n.pred).collect::<Vec<_>>(), 0.9);
            let local = pred - knn_stats.mean;
            let global = pred - global_pred.mean;
            let rank = stats.rank(pred);

            let factor_str = format_top_factors(&emb, &ridge_weights, config.top_factors);
            let lookalike_str = knn
                .iter()
                .take(5)
                .map(|n| n.id.clone())
                .collect::<Vec<_>>()
                .join(",");

            ids.push(id.clone());
            ts.push(timestamp);
            preds.push(pred);
            cluster_ids.push(cluster as i32);
            cluster_mean.push(stats.pred_stats.mean);
            cluster_std.push(stats.pred_stats.stddev());
            cluster_rank.push(rank);
            knn_mean.push(knn_stats.mean);
            knn_std.push(knn_stats.stddev());
            knn_p90.push(knn_p90_val);
            local_lift.push(local);
            global_lift.push(global);
            emb_outlier.push(stats.distance_outlier(dist2.sqrt()));
            pred_outlier.push(stats.pred_outlier(pred));
            top_factors.push(factor_str);
            lookalikes.push(lookalike_str);

            knn_reservoirs[cluster].insert(id, pred, emb);
        }

        let batch = RecordBatch::try_new(
            Arc::new(explanation_schema()),
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(Int64Array::from(ts)),
                Arc::new(Float32Array::from(preds)),
                Arc::new(Int32Array::from(cluster_ids)),
                Arc::new(Float32Array::from(cluster_mean)),
                Arc::new(Float32Array::from(cluster_std)),
                Arc::new(Float32Array::from(cluster_rank)),
                Arc::new(Float32Array::from(knn_mean)),
                Arc::new(Float32Array::from(knn_std)),
                Arc::new(Float32Array::from(knn_p90)),
                Arc::new(Float32Array::from(local_lift)),
                Arc::new(Float32Array::from(global_lift)),
                Arc::new(Float32Array::from(emb_outlier)),
                Arc::new(Float32Array::from(pred_outlier)),
                Arc::new(StringArray::from(top_factors)),
                Arc::new(StringArray::from(lookalikes)),
            ],
        )?;
        explanations_batches.push(batch);
    }

    for (idx, stats) in cluster_stats.iter().enumerate() {
        trends_rows.push(TrendRow::cohort(idx as i32, stats));
        trends_rows.extend(stats.temporal_rows(idx as i32, config.time_bucket_secs, config.max_time_buckets));
    }

    for (dim, weight) in ridge_weights.iter().enumerate() {
        trends_rows.push(TrendRow::feature_importance(dim as i32, *weight, ridge.contribution(dim, *weight)));
    }

    for (idx, stats) in cluster_stats.iter().enumerate() {
        if stats.outlier_score() > 2.0 {
            trends_rows.push(TrendRow::outlier(idx as i32, stats));
        }
    }

    let trends_batches = vec![TrendRow::to_batch(&trends_rows)?];

    write_parquet(&config.output_explanations_parquet, explanation_schema(), explanations_batches)?;
    write_parquet(&config.output_global_trends_parquet, trends_schema(), trends_batches)?;

    Ok(())
}

fn sample_embeddings(path: &Path, config: &ExplainConfig, count: usize) -> Result<(usize, Vec<Vec<f32>>)> {
    let mut stream = ParquetStream::open(
        path,
        &config.id_column,
        &config.timestamp_column,
        &config.prediction_column,
        &config.embedding_column,
        config.id_kind.clone(),
        config.timestamp_kind.clone(),
    )?;
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(count);
    let mut dim = 0usize;
    while let Some(batch) = stream.next_batch() {
        let batch = batch?;
        for i in 0..batch.len() {
            let emb = batch.embedding_as_f32(i)?;
            if dim == 0 {
                dim = emb.len();
            }
            if emb.len() != dim {
                return Err(anyhow!("inconsistent embedding length"));
            }
            if samples.len() < count {
                samples.push(emb);
            }
        }
        if samples.len() >= count {
            break;
        }
    }
    Ok((dim, samples))
}

fn initialize_centroids(dim: usize, k: usize, mut samples: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return vec![vec![0.0; dim]; k];
    }
    let mut rng = thread_rng();
    samples.shuffle(&mut rng);
    let mut centroids = Vec::with_capacity(k);
    for i in 0..k {
        let idx = i % samples.len();
        centroids.push(samples[idx].clone());
    }
    centroids
}

fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_dist = f32::MAX;
    for (idx, c) in centroids.iter().enumerate() {
        let dist = l2(vec, c);
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    (best, best_dist)
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

struct RunningStats {
    pub count: u64,
    pub mean: f32,
    pub m2: f32,
}

impl RunningStats {
    fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    fn add(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn stddev(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count as f32 - 1.0)).sqrt()
        }
    }

    fn from_values(values: &[f32]) -> Self {
        let mut stats = RunningStats::new();
        for v in values {
            stats.add(*v);
        }
        stats
    }
}

struct RidgeProbe {
    sum_x2: Vec<f32>,
    sum_xy: Vec<f32>,
    lambda: f32,
    sum_abs_x: Vec<f32>,
    count: u64,
}

impl RidgeProbe {
    fn new(dim: usize, lambda: f32) -> Self {
        Self {
            sum_x2: vec![0.0; dim],
            sum_xy: vec![0.0; dim],
            lambda,
            sum_abs_x: vec![0.0; dim],
            count: 0,
        }
    }

    fn update(&mut self, x: &[f32], y: f32) {
        self.count += 1;
        for (i, v) in x.iter().enumerate() {
            let x2 = v * v;
            self.sum_x2[i] += x2;
            self.sum_xy[i] += v * y;
            self.sum_abs_x[i] += v.abs();
        }
    }

    fn weights(&self) -> Vec<f32> {
        self.sum_xy
            .iter()
            .zip(self.sum_x2.iter())
            .map(|(xy, x2)| xy / (x2 + self.lambda))
            .collect()
    }

    fn contribution(&self, dim: usize, weight: f32) -> f32 {
        let abs_x = self.sum_abs_x.get(dim).copied().unwrap_or(0.0);
        let mean_abs = if self.count == 0 { 0.0 } else { abs_x / self.count as f32 };
        mean_abs * weight.abs()
    }
}

struct ClusterStats {
    pred_stats: RunningStats,
    dist_stats: RunningStats,
    sample: Vec<f32>,
    time_buckets: HashMap<i64, RunningStats>,
    sample_limit: usize,
}

impl ClusterStats {
    fn new(sample_limit: usize) -> Self {
        Self {
            pred_stats: RunningStats::new(),
            dist_stats: RunningStats::new(),
            sample: Vec::with_capacity(sample_limit),
            time_buckets: HashMap::new(),
            sample_limit,
        }
    }

    fn add(&mut self, pred: f32, distance: f32, timestamp: i64, bucket_secs: i64) {
        self.pred_stats.add(pred);
        self.dist_stats.add(distance);
        if self.sample.len() < self.sample_limit {
            self.sample.push(pred);
        }
        let bucket = if bucket_secs > 0 { timestamp / bucket_secs } else { 0 };
        let entry = self.time_buckets.entry(bucket).or_insert_with(RunningStats::new);
        entry.add(pred);
    }

    fn rank(&self, pred: f32) -> f32 {
        if self.sample.is_empty() {
            return 0.5;
        }
        let mut below = 0u32;
        for v in &self.sample {
            if pred >= *v {
                below += 1;
            }
        }
        below as f32 / self.sample.len() as f32
    }

    fn distance_outlier(&self, distance: f32) -> f32 {
        let std = self.dist_stats.stddev();
        if std == 0.0 {
            0.0
        } else {
            (distance - self.dist_stats.mean) / std
        }
    }

    fn pred_outlier(&self, pred: f32) -> f32 {
        let std = self.pred_stats.stddev();
        if std == 0.0 {
            0.0
        } else {
            (pred - self.pred_stats.mean) / std
        }
    }

    fn outlier_score(&self) -> f32 {
        self.pred_stats.stddev() + self.dist_stats.stddev()
    }

    fn temporal_rows(&self, cluster_id: i32, bucket_secs: i64, max_buckets: usize) -> Vec<TrendRow> {
        let mut rows = Vec::new();
        if bucket_secs <= 0 {
            return rows;
        }
        let mut keys: Vec<i64> = self.time_buckets.keys().copied().collect();
        keys.sort();
        for key in keys.into_iter().take(max_buckets) {
            if let Some(stats) = self.time_buckets.get(&key) {
                rows.push(TrendRow::temporal(cluster_id, key, stats));
            }
        }
        rows
    }
}

struct Reservoir {
    samples: Vec<Neighbor>,
    limit: usize,
    seen: u64,
}

impl Reservoir {
    fn new(limit: usize) -> Self {
        Self { samples: Vec::with_capacity(limit), limit, seen: 0 }
    }

    fn insert(&mut self, id: String, pred: f32, emb: Vec<f32>) {
        self.seen += 1;
        if self.samples.len() < self.limit {
            self.samples.push(Neighbor { id, pred, emb });
            return;
        }
        if self.limit == 0 {
            return;
        }
        let mut rng = thread_rng();
        let idx = (rng.gen::<u64>() % self.seen) as usize;
        if idx < self.limit {
            self.samples[idx] = Neighbor { id, pred, emb };
        }
    }

    fn query_knn(&self, emb: &[f32], k: usize) -> Vec<Neighbor> {
        let mut scored: Vec<(f32, &Neighbor)> = self.samples.iter().map(|n| (l2(emb, &n.emb), n)).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(_, n)| n.clone()).collect()
    }
}

#[derive(Clone)]
struct Neighbor {
    id: String,
    pred: f32,
    emb: Vec<f32>,
}

fn percentile(values: &Vec<f32>, p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

fn format_top_factors(emb: &[f32], weights: &[f32], k: usize) -> String {
    let mut contribs: Vec<(usize, f32)> = emb
        .iter()
        .zip(weights.iter())
        .enumerate()
        .map(|(i, (x, w))| (i, x * w))
        .collect();
    contribs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
    contribs
        .into_iter()
        .take(k)
        .map(|(i, v)| format!("{}:{:.4}", i, v))
        .collect::<Vec<_>>()
        .join("|")
}

struct TrendRow {
    artifact_type: String,
    cluster_id: Option<i32>,
    dim: Option<i32>,
    time_bucket: Option<i32>,
    count: Option<i64>,
    mean_pred: Option<f32>,
    std_pred: Option<f32>,
    p10: Option<f32>,
    p50: Option<f32>,
    p90: Option<f32>,
    weight: Option<f32>,
    abs_weight: Option<f32>,
    global_contribution: Option<f32>,
    outlier_score: Option<f32>,
    notes: Option<String>,
}

impl TrendRow {
    fn cohort(cluster_id: i32, stats: &ClusterStats) -> Self {
        let (p10, p50, p90) = percentile_triplet(&stats.sample);
        Self {
            artifact_type: "cohort_summary".to_string(),
            cluster_id: Some(cluster_id),
            dim: None,
            time_bucket: None,
            count: Some(stats.pred_stats.count as i64),
            mean_pred: Some(stats.pred_stats.mean),
            std_pred: Some(stats.pred_stats.stddev()),
            p10: Some(p10),
            p50: Some(p50),
            p90: Some(p90),
            weight: None,
            abs_weight: None,
            global_contribution: None,
            outlier_score: None,
            notes: None,
        }
    }

    fn feature_importance(dim: i32, weight: f32, global_contribution: f32) -> Self {
        Self {
            artifact_type: "feature_importance".to_string(),
            cluster_id: None,
            dim: Some(dim),
            time_bucket: None,
            count: None,
            mean_pred: None,
            std_pred: None,
            p10: None,
            p50: None,
            p90: None,
            weight: Some(weight),
            abs_weight: Some(weight.abs()),
            global_contribution: Some(global_contribution),
            outlier_score: None,
            notes: None,
        }
    }

    fn outlier(cluster_id: i32, stats: &ClusterStats) -> Self {
        Self {
            artifact_type: "outlier_cluster".to_string(),
            cluster_id: Some(cluster_id),
            dim: None,
            time_bucket: None,
            count: Some(stats.pred_stats.count as i64),
            mean_pred: Some(stats.pred_stats.mean),
            std_pred: Some(stats.pred_stats.stddev()),
            p10: None,
            p50: None,
            p90: None,
            weight: None,
            abs_weight: None,
            global_contribution: None,
            outlier_score: Some(stats.outlier_score()),
            notes: Some("cluster stddev high".to_string()),
        }
    }

    fn temporal(cluster_id: i32, bucket: i64, stats: &RunningStats) -> Self {
        Self {
            artifact_type: "temporal_trend".to_string(),
            cluster_id: Some(cluster_id),
            dim: None,
            time_bucket: Some(bucket as i32),
            count: Some(stats.count as i64),
            mean_pred: Some(stats.mean),
            std_pred: Some(stats.stddev()),
            p10: None,
            p50: None,
            p90: None,
            weight: None,
            abs_weight: None,
            global_contribution: None,
            outlier_score: None,
            notes: None,
        }
    }

    fn to_batch(rows: &[TrendRow]) -> Result<RecordBatch> {
        let mut artifact_type = Vec::with_capacity(rows.len());
        let mut cluster_id = Vec::with_capacity(rows.len());
        let mut dim = Vec::with_capacity(rows.len());
        let mut time_bucket = Vec::with_capacity(rows.len());
        let mut count = Vec::with_capacity(rows.len());
        let mut mean_pred = Vec::with_capacity(rows.len());
        let mut std_pred = Vec::with_capacity(rows.len());
        let mut p10 = Vec::with_capacity(rows.len());
        let mut p50 = Vec::with_capacity(rows.len());
        let mut p90 = Vec::with_capacity(rows.len());
        let mut weight = Vec::with_capacity(rows.len());
        let mut abs_weight = Vec::with_capacity(rows.len());
        let mut global_contribution = Vec::with_capacity(rows.len());
        let mut outlier_score = Vec::with_capacity(rows.len());
        let mut notes = Vec::with_capacity(rows.len());

        for row in rows {
            artifact_type.push(row.artifact_type.clone());
            cluster_id.push(row.cluster_id);
            dim.push(row.dim);
            time_bucket.push(row.time_bucket);
            count.push(row.count);
            mean_pred.push(row.mean_pred);
            std_pred.push(row.std_pred);
            p10.push(row.p10);
            p50.push(row.p50);
            p90.push(row.p90);
            weight.push(row.weight);
            abs_weight.push(row.abs_weight);
            global_contribution.push(row.global_contribution);
            outlier_score.push(row.outlier_score);
            notes.push(row.notes.clone());
        }

        let batch = RecordBatch::try_new(
            Arc::new(trends_schema()),
            vec![
                Arc::new(StringArray::from(artifact_type)) as ArrayRef,
                Arc::new(Int32Array::from(cluster_id)),
                Arc::new(Int32Array::from(dim)),
                Arc::new(Int32Array::from(time_bucket)),
                Arc::new(Int64Array::from(count)),
                Arc::new(Float32Array::from(mean_pred)),
                Arc::new(Float32Array::from(std_pred)),
                Arc::new(Float32Array::from(p10)),
                Arc::new(Float32Array::from(p50)),
                Arc::new(Float32Array::from(p90)),
                Arc::new(Float32Array::from(weight)),
                Arc::new(Float32Array::from(abs_weight)),
                Arc::new(Float32Array::from(global_contribution)),
                Arc::new(Float32Array::from(outlier_score)),
                Arc::new(StringArray::from(notes)),
            ],
        )?;
        Ok(batch)
    }
}

fn percentile_triplet(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p10 = v[((v.len() - 1) as f64 * 0.1) as usize];
    let p50 = v[((v.len() - 1) as f64 * 0.5) as usize];
    let p90 = v[((v.len() - 1) as f64 * 0.9) as usize];
    (p10, p50, p90)
}
