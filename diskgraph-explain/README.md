# diskgraph-explain

Embedding-only explainability pipeline with raw event/feature attribution via DiskGraph.

## What it does
- Reads parquet with `(id, timestamp, prediction, embedding)`.
- Clusters embeddings to form cohorts.
- For each cohort, samples neighborhoods from DiskGraph and computes feature enrichment.
- Produces per-row explanations grounded in **raw node attributes** and **edges**.

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
- `embedding` (list<float32>)
- `cohort_id` (int32)
- `cohort_mean_pred` (float32)
- `cohort_std_pred` (float32)
- `global_mean_pred` (float32)
- `local_lift` (float32)
- `contrast_cohort_id` (int32)
- `top_features` (utf8, JSON list of `{feature,lift,delta,support}`)
- `top_events` (utf8, JSON list of `{relation,table,node_id,pk,feature,lift,delta,support}`)

### global_trends.parquet
Rows keyed by `artifact_type`:
- `cohort_summary`
- `cohort_feature_lift`

Columns:
- `artifact_type` (utf8)
- `cohort_id` (int32, nullable)
- `time_bucket` (int32, nullable)
- `count` (int64, nullable)
- `mean_pred` (float32, nullable)
- `std_pred` (float32, nullable)
- `p10/p50/p90` (float32, nullable; reserved)
- `feature_key` (utf8, nullable)
- `lift` (float32, nullable)
- `support` (int64, nullable)
- `notes` (utf8, nullable)

## Notes
- Explanations are statistical: features/events enriched in the user’s embedding cohort.
- Requires DiskGraph index and node data for configured tables.
