# diskgraph-explain

Embedding-only explainability pipeline with raw event/feature attribution via DiskGraph.

## What it does
- Reads parquet with `(id, timestamp, prediction, embedding)`.
- Clusters embeddings to form cohorts.
- For each cohort, samples neighborhoods from DiskGraph and computes feature enrichment.
- Produces per-row explanations grounded in **raw node attributes** and **edges**.

## Usage

1) Build or obtain a DiskGraph index for the dataset you want to explain.

```
# Example ingest (local files)
./diskgraph-ingest/scripts/ingest_local.sh /path/to/source_dir_or_diskgraph_config.json
```

2) Create a config JSON for the explain pipeline.

```
cp diskgraph-explain/config.example.json /path/to/config.json
# edit paths + table names
```

Required fields:
- `input_parquet`: parquet with `(id, timestamp, prediction, embedding)`
- `index_path`: DiskGraph index root containing `index_metadata.json`
- `root_table`: table name for the `id` column
- `output_explanations_parquet`, `output_global_trends_parquet`

3) Run the explain pipeline.

```
./diskgraph-explain/scripts/explain_local.sh /path/to/config.json
```

Or run directly with cargo:

```
cargo run -p diskgraph-explain --bin explain --release -- /path/to/config.json
```

## Interactive inspector

After generating the outputs, you can inspect explanations interactively:

```
./diskgraph-explain/scripts/explain_inspect_local.sh /path/to/explanations.parquet /path/to/global_trends.parquet
```

### Useful commands
- `help` — list commands
- `stats` — show row/cohort counts
- `show <id>` — show raw row summary
- `reason <id>` — compelling reasons for positive/negative prediction
- `similar <id> [k]` — nearest neighbors in embedding space
- `cohort <id>` — cohort summary + top lifted features
- `trend <cohort_id>` — temporal trend of cohort predictions
- `list <cohort_id> [n]` — list sample ids in cohort
- `exit` / `quit`

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
