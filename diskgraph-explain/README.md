# diskgraph-explain

One-shot explainability pipeline for embedding-based predictions.

## What it does
- Reads a parquet with `(id, timestamp, prediction, embedding)`.
- Builds embedding cohorts (MiniBatch k-means).
- Computes per-row explanations (cluster context, kNN lift, ridge factors).
- Writes two parquet outputs:
  - `explanations.parquet` (per-row explanations)
  - `global_trends.parquet` (cohort summaries + feature importance + trends)

## Usage

```
./diskgraph-explain/scripts/explain_local.sh /path/to/config.json
```

Example config: `diskgraph-explain/config.example.json`.

## Output schemas

### explanations.parquet
- `id` (utf8)
- `timestamp` (int64 seconds)
- `prediction` (float32)
- `cluster_id` (int32)
- `cluster_mean_pred` (float32)
- `cluster_std_pred` (float32)
- `cluster_rank` (float32)
- `knn_mean_pred` (float32)
- `knn_std_pred` (float32)
- `knn_p90_pred` (float32)
- `local_lift` (float32)
- `global_lift` (float32)
- `embedding_outlier_score` (float32)
- `prediction_outlier_score` (float32)
- `top_factors` (utf8, `dim:contribution` pipe-separated)
- `lookalike_ids` (utf8, comma-separated)

### global_trends.parquet
Rows are tagged by `artifact_type`:
- `cohort_summary`: per cluster stats
- `feature_importance`: per embedding dimension
- `outlier_cluster`: clusters with unusually high variability
- `temporal_trend`: cluster mean by time bucket

Shared columns:
- `artifact_type` (utf8)
- `cluster_id` (int32, nullable)
- `dim` (int32, nullable)
- `time_bucket` (int32, nullable)
- `count` (int64, nullable)
- `mean_pred` (float32, nullable)
- `std_pred` (float32, nullable)
- `p10`, `p50`, `p90` (float32, nullable)
- `weight`, `abs_weight`, `global_contribution` (float32, nullable)
- `outlier_score` (float32, nullable)
- `notes` (utf8, nullable)

## Notes
- kNN uses a bounded reservoir per cluster for memory control.
- Ridge probe is diagonal for speed and memory efficiency.
