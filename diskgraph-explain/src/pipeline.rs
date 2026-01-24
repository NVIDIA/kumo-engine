use crate::config::ExplainConfig;
use crate::graph_features::{FeatureKey, GraphExtractor, RowFeatures, SchemaIndex, ValueToken};
use crate::parquet_io::{explanation_schema, trends_schema, ParquetStream, write_parquet};
use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, Float32Array, Int32Array, Int64Array, ListArray, RecordBatch, StringArray};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Serialize;
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

    let mut cohort_stats = Vec::with_capacity(config.num_clusters);
    let mut cohort_time: Vec<HashMap<i64, RunningStats>> = Vec::with_capacity(config.num_clusters);
    for _ in 0..config.num_clusters {
        cohort_stats.push(RunningStats::new());
        cohort_time.push(HashMap::new());
    }
    let mut global_stats = RunningStats::new();

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
                let ts = batch.timestamp_as_i64(i)?;
                let (cluster, _) = nearest_centroid(&emb, &centroids);
                cohort_stats[cluster].add(pred);
                global_stats.add(pred);
                if config.time_bucket_secs > 0 {
                    let bucket = ts / config.time_bucket_secs;
                    let entry = cohort_time[cluster].entry(bucket).or_insert_with(RunningStats::new);
                    entry.add(pred);
                }
            }
        }
    }

    let contrast_cohorts = compute_contrast_cohorts(&centroids, &cohort_stats, global_stats.mean);

    let reader = diskgraph_index::graph::GraphIndexReader::open(
        &config.index_path.join("index_metadata.json"),
        &config.index_path,
    )?;
    let extractor = GraphExtractor::new(reader, &config)?;

    let mut global_counts: HashMap<FeatureKey, u32> = HashMap::new();
    let mut cohort_counts: Vec<HashMap<FeatureKey, u32>> = (0..config.num_clusters)
        .map(|_| HashMap::new())
        .collect();

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
                let emb = batch.embedding_as_f32(i)?;
                let (cluster, _) = nearest_centroid(&emb, &centroids);
                let row_features = extractor.extract_row_features(&id, timestamp)?;
                update_feature_counts(&row_features, &mut global_counts, &mut cohort_counts[cluster]);
            }
        }
    }

    let mut trends_rows: Vec<TrendRow> = Vec::new();
    for (cohort_id, stats) in cohort_stats.iter().enumerate() {
        trends_rows.push(TrendRow::cohort_summary(cohort_id as i32, stats));
    }
    for (cohort_id, buckets) in cohort_time.iter().enumerate() {
        let mut keys: Vec<i64> = buckets.keys().copied().collect();
        keys.sort();
        for key in keys.into_iter().take(config.max_time_buckets) {
            if let Some(stats) = buckets.get(&key) {
                trends_rows.push(TrendRow::temporal(cohort_id as i32, key as i32, stats));
            }
        }
    }

    let lift_table = LiftTable::new(
        cohort_counts,
        global_counts,
        cohort_stats.iter().map(|s| s.count).collect(),
        global_stats.count,
        config.smoothing_alpha,
        config.min_support,
    );

    for cohort_id in 0..config.num_clusters {
        let mut top = lift_table.top_features(cohort_id, config.feature_top_k, &extractor.schema);
        for row in top.drain(..) {
            trends_rows.push(row);
        }
    }

    let mut explanations_batches: Vec<RecordBatch> = Vec::new();
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
        let mut ids = Vec::with_capacity(batch.len());
        let mut timestamps = Vec::with_capacity(batch.len());
        let mut preds = Vec::with_capacity(batch.len());
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        let mut cohort_ids = Vec::with_capacity(batch.len());
        let mut cohort_means = Vec::with_capacity(batch.len());
        let mut cohort_stds = Vec::with_capacity(batch.len());
        let mut global_means = Vec::with_capacity(batch.len());
        let mut local_lifts = Vec::with_capacity(batch.len());
        let mut contrast_ids = Vec::with_capacity(batch.len());
        let mut top_features = Vec::with_capacity(batch.len());
        let mut top_events = Vec::with_capacity(batch.len());

        for i in 0..batch.len() {
            let id = batch.id_as_string(i)?;
            let timestamp = batch.timestamp_as_i64(i)?;
            let pred = batch.prediction_as_f64(i)? as f32;
            let emb = batch.embedding_as_f32(i)?;
            let (cluster, _) = nearest_centroid(&emb, &centroids);
            let contrast = contrast_cohorts[cluster];
            let row_features = extractor.extract_row_features(&id, timestamp)?;

            let feature_explanations = lift_table.explain_features(
                cluster,
                contrast,
                &row_features,
                config.feature_top_k,
                &extractor.schema,
            );
            let event_explanations = lift_table.explain_events(
                cluster,
                contrast,
                &row_features,
                config.event_top_k,
                &extractor.schema,
            );

            ids.push(id);
            timestamps.push(timestamp);
            preds.push(pred);
            embeddings.push(emb);
            cohort_ids.push(cluster as i32);
            cohort_means.push(cohort_stats[cluster].mean);
            cohort_stds.push(cohort_stats[cluster].stddev());
            global_means.push(global_stats.mean);
            local_lifts.push(pred - cohort_stats[cluster].mean);
            contrast_ids.push(contrast as i32);
            top_features.push(serde_json::to_string(&feature_explanations)?);
            top_events.push(serde_json::to_string(&event_explanations)?);
        }

        let embedding_array = build_embedding_array(&embeddings);
        let batch = RecordBatch::try_new(
            Arc::new(explanation_schema()),
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(Float32Array::from(preds)),
                embedding_array,
                Arc::new(Int32Array::from(cohort_ids)),
                Arc::new(Float32Array::from(cohort_means)),
                Arc::new(Float32Array::from(cohort_stds)),
                Arc::new(Float32Array::from(global_means)),
                Arc::new(Float32Array::from(local_lifts)),
                Arc::new(Int32Array::from(contrast_ids)),
                Arc::new(StringArray::from(top_features)),
                Arc::new(StringArray::from(top_events)),
            ],
        )?;
        explanations_batches.push(batch);
    }

    let trends_batch = TrendRow::to_batch(&trends_rows)?;
    write_parquet(&config.output_explanations_parquet, explanation_schema(), explanations_batches)?;
    write_parquet(&config.output_global_trends_parquet, trends_schema(), vec![trends_batch])?;

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

fn compute_contrast_cohorts(centroids: &[Vec<f32>], stats: &[RunningStats], global_mean: f32) -> Vec<usize> {
    let mut out = vec![0usize; centroids.len()];
    for (i, c) in centroids.iter().enumerate() {
        let is_high = stats[i].mean >= global_mean;
        let mut best = i;
        let mut best_dist = f32::MAX;
        for (j, other) in centroids.iter().enumerate() {
            if i == j {
                continue;
            }
            let other_high = stats[j].mean >= global_mean;
            if is_high == other_high {
                continue;
            }
            let dist = l2(c, other);
            if dist < best_dist {
                best_dist = dist;
                best = j;
            }
        }
        out[i] = best;
    }
    out
}

#[derive(Clone, Default)]
struct RunningStats {
    count: u64,
    mean: f32,
    m2: f32,
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
}

fn update_feature_counts(
    row: &RowFeatures,
    global: &mut HashMap<FeatureKey, u32>,
    cohort: &mut HashMap<FeatureKey, u32>,
) {
    for feature in &row.feature_set {
        *global.entry(feature.clone()).or_insert(0) += 1;
        *cohort.entry(feature.clone()).or_insert(0) += 1;
    }
}

struct LiftTable {
    cohort_counts: Vec<HashMap<FeatureKey, u32>>,
    global_counts: HashMap<FeatureKey, u32>,
    cohort_sizes: Vec<u64>,
    global_size: u64,
    alpha: f32,
    min_support: u32,
}

impl LiftTable {
    fn new(
        cohort_counts: Vec<HashMap<FeatureKey, u32>>,
        global_counts: HashMap<FeatureKey, u32>,
        cohort_sizes: Vec<u64>,
        global_size: u64,
        alpha: f32,
        min_support: u32,
    ) -> Self {
        Self {
            cohort_counts,
            global_counts,
            cohort_sizes,
            global_size,
            alpha,
            min_support,
        }
    }

    fn lift(&self, cohort_id: usize, feature: &FeatureKey) -> Option<LiftScore> {
        let cohort_count = *self.cohort_counts[cohort_id].get(feature).unwrap_or(&0);
        let global_count = *self.global_counts.get(feature).unwrap_or(&0);
        if cohort_count < self.min_support || global_count < self.min_support {
            return None;
        }
        let cohort_rate = (cohort_count as f32 + self.alpha)
            / (self.cohort_sizes[cohort_id] as f32 + 2.0 * self.alpha);
        let global_rate = (global_count as f32 + self.alpha)
            / (self.global_size as f32 + 2.0 * self.alpha);
        let lift = if global_rate > 0.0 { cohort_rate / global_rate } else { 0.0 };
        Some(LiftScore { lift, support: cohort_count as u64 })
    }

    fn explain_features(
        &self,
        cohort_id: usize,
        contrast_id: usize,
        row: &RowFeatures,
        k: usize,
        schema: &SchemaIndex,
    ) -> Vec<FeatureExplanation> {
        let mut scored = Vec::new();
        for feature in &row.feature_set {
            let lift = match self.lift(cohort_id, feature) {
                Some(lift) => lift,
                None => continue,
            };
            let contrast = self.lift(contrast_id, feature);
            let contrast_lift = contrast.map(|c| c.lift).unwrap_or(1.0);
            let delta = lift.lift - contrast_lift;
            scored.push((delta, feature.clone(), lift));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(delta, feature, lift)| FeatureExplanation {
                feature: feature_to_string(&feature, schema),
                lift: lift.lift,
                delta,
                support: lift.support,
            })
            .collect()
    }

    fn explain_events(
        &self,
        cohort_id: usize,
        contrast_id: usize,
        row: &RowFeatures,
        k: usize,
        schema: &SchemaIndex,
    ) -> Vec<EventExplanation> {
        let mut scored = Vec::new();
        for node in &row.node_features {
            let mut best = None;
            for feature in &node.features {
                let lift = match self.lift(cohort_id, feature) {
                    Some(lift) => lift,
                    None => continue,
                };
                let contrast = self.lift(contrast_id, feature);
                let contrast_lift = contrast.map(|c| c.lift).unwrap_or(1.0);
                let delta = lift.lift - contrast_lift;
                if best.as_ref().map(|b: &EventScore| delta > b.delta).unwrap_or(true) {
                    best = Some(EventScore { feature: feature.clone(), lift, delta });
                }
            }
            if let Some(best) = best {
                scored.push((best.delta, EventExplanation {
                    relation: node.rel_name.clone(),
                    table: node.dst_table.clone(),
                    node_id: node.node_id,
                    pk: node.pk_value.clone(),
                    feature: feature_to_string(&best.feature, schema),
                    lift: best.lift.lift,
                    delta: best.delta,
                    support: best.lift.support,
                }));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(_, e)| e).collect()
    }

    fn top_features(&self, cohort_id: usize, k: usize, schema: &SchemaIndex) -> Vec<TrendRow> {
        let mut scored = Vec::new();
        for (feature, _) in &self.cohort_counts[cohort_id] {
            if let Some(lift) = self.lift(cohort_id, feature) {
                scored.push((lift.lift, feature.clone(), lift.support));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(lift, feature, support)| TrendRow::feature_lift(cohort_id as i32, feature_to_string(&feature, schema), lift, support))
            .collect()
    }
}

#[derive(Clone, Copy)]
struct LiftScore {
    lift: f32,
    support: u64,
}

#[derive(Serialize)]
struct FeatureExplanation {
    feature: String,
    lift: f32,
    delta: f32,
    support: u64,
}

#[derive(Serialize)]
struct EventExplanation {
    relation: String,
    table: String,
    node_id: u64,
    pk: Option<String>,
    feature: String,
    lift: f32,
    delta: f32,
    support: u64,
}

struct EventScore {
    feature: FeatureKey,
    lift: LiftScore,
    delta: f32,
}

fn feature_to_string(feature: &FeatureKey, schema: &SchemaIndex) -> String {
    match feature {
        FeatureKey::RelationCount { rel_id, bucket } => {
            let rel = schema.strings.get(*rel_id).unwrap_or_else(|| "?".to_string());
            format!("relation:{}:bucket:{}", rel, bucket)
        }
        FeatureKey::NodeAttr { table_id, column_id, value } => {
            let table = schema.strings.get(*table_id).unwrap_or_else(|| "?".to_string());
            let column = schema.strings.get(*column_id).unwrap_or_else(|| "?".to_string());
            let value = value_to_string(value, schema);
            format!("{}.{}={}", table, column, value)
        }
    }
}

fn value_to_string(value: &ValueToken, schema: &SchemaIndex) -> String {
    match value {
        ValueToken::Str(id) => schema.strings.get(*id).unwrap_or_else(|| "?".to_string()),
        ValueToken::Bool(v) => v.to_string(),
        ValueToken::FloatBucket(b) => format!("float_bucket_{b}"),
        ValueToken::Missing => "missing".to_string(),
    }
}

#[derive(Clone)]
struct TrendRow {
    artifact_type: String,
    cohort_id: Option<i32>,
    time_bucket: Option<i32>,
    count: Option<i64>,
    mean_pred: Option<f32>,
    std_pred: Option<f32>,
    p10: Option<f32>,
    p50: Option<f32>,
    p90: Option<f32>,
    feature_key: Option<String>,
    lift: Option<f32>,
    support: Option<i64>,
    notes: Option<String>,
}

impl TrendRow {
    fn cohort_summary(cohort_id: i32, stats: &RunningStats) -> Self {
        Self {
            artifact_type: "cohort_summary".to_string(),
            cohort_id: Some(cohort_id),
            time_bucket: None,
            count: Some(stats.count as i64),
            mean_pred: Some(stats.mean),
            std_pred: Some(stats.stddev()),
            p10: None,
            p50: None,
            p90: None,
            feature_key: None,
            lift: None,
            support: None,
            notes: None,
        }
    }

    fn feature_lift(cohort_id: i32, feature_key: String, lift: f32, support: u64) -> Self {
        Self {
            artifact_type: "cohort_feature_lift".to_string(),
            cohort_id: Some(cohort_id),
            time_bucket: None,
            count: None,
            mean_pred: None,
            std_pred: None,
            p10: None,
            p50: None,
            p90: None,
            feature_key: Some(feature_key),
            lift: Some(lift),
            support: Some(support as i64),
            notes: None,
        }
    }

    fn temporal(cohort_id: i32, bucket: i32, stats: &RunningStats) -> Self {
        Self {
            artifact_type: "temporal_trend".to_string(),
            cohort_id: Some(cohort_id),
            time_bucket: Some(bucket),
            count: Some(stats.count as i64),
            mean_pred: Some(stats.mean),
            std_pred: Some(stats.stddev()),
            p10: None,
            p50: None,
            p90: None,
            feature_key: None,
            lift: None,
            support: None,
            notes: None,
        }
    }

    fn to_batch(rows: &[TrendRow]) -> Result<RecordBatch> {
        let mut artifact_type = Vec::with_capacity(rows.len());
        let mut cohort_id = Vec::with_capacity(rows.len());
        let mut time_bucket = Vec::with_capacity(rows.len());
        let mut count = Vec::with_capacity(rows.len());
        let mut mean_pred = Vec::with_capacity(rows.len());
        let mut std_pred = Vec::with_capacity(rows.len());
        let mut p10 = Vec::with_capacity(rows.len());
        let mut p50 = Vec::with_capacity(rows.len());
        let mut p90 = Vec::with_capacity(rows.len());
        let mut feature_key = Vec::with_capacity(rows.len());
        let mut lift = Vec::with_capacity(rows.len());
        let mut support = Vec::with_capacity(rows.len());
        let mut notes = Vec::with_capacity(rows.len());

        for row in rows {
            artifact_type.push(row.artifact_type.clone());
            cohort_id.push(row.cohort_id);
            time_bucket.push(row.time_bucket);
            count.push(row.count);
            mean_pred.push(row.mean_pred);
            std_pred.push(row.std_pred);
            p10.push(row.p10);
            p50.push(row.p50);
            p90.push(row.p90);
            feature_key.push(row.feature_key.clone());
            lift.push(row.lift);
            support.push(row.support);
            notes.push(row.notes.clone());
        }

        let batch = RecordBatch::try_new(
            Arc::new(trends_schema()),
            vec![
                Arc::new(StringArray::from(artifact_type)) as ArrayRef,
                Arc::new(Int32Array::from(cohort_id)),
                Arc::new(Int32Array::from(time_bucket)),
                Arc::new(Int64Array::from(count)),
                Arc::new(Float32Array::from(mean_pred)),
                Arc::new(Float32Array::from(std_pred)),
                Arc::new(Float32Array::from(p10)),
                Arc::new(Float32Array::from(p50)),
                Arc::new(Float32Array::from(p90)),
                Arc::new(StringArray::from(feature_key)),
                Arc::new(Float32Array::from(lift)),
                Arc::new(Int64Array::from(support)),
                Arc::new(StringArray::from(notes)),
            ],
        )?;
        Ok(batch)
    }
}

fn build_embedding_array(values: &[Vec<f32>]) -> ArrayRef {
    let mut offsets = Vec::with_capacity(values.len() + 1);
    offsets.push(0i32);
    let mut flat = Vec::new();
    for vec in values {
        flat.extend_from_slice(vec);
        offsets.push(flat.len() as i32);
    }
    let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
    let offsets = arrow_buffer::OffsetBuffer::new(offsets.into());
    Arc::new(ListArray::new(
        Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, false)),
        offsets,
        values,
        None,
    ))
}
