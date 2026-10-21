<p align="center">
  <img height="180" src="https://raw.githubusercontent.com/NVIDIA/kumo-engine/main/ui/src/images/kumo_logo_green.svg" alt="Kumo" />
</p>

______________________________________________________________________

`kumo.engine` runs relational deep learning in-process: define tables and graphs,
materialize a snapshot, and train or predict with a GNN.

## Object model

Pass connectors, tables, graphs, and trainers directly as Python objects.
A `Table` holds its source and schema, and a `Graph` holds tables and edges.
Import these types with `from kumo.engine import Column, Edge, Graph, Table`.
`Column` and `Edge` come from `kumoapi`. `Table.from_source_table()` reads metadata
and may sample rows to infer column types; `Graph.snapshot()` ingests the graph.

`Graph.snapshot()`, `Trainer.fit()`, and `Trainer.predict()` are synchronous.
Training and prediction require `snapshot=`; reuse table rows with another
snapshot as described under
[local storage and results](#local-storage-and-results).

Classification targets must have between two and the configured maximum number
of classes, including custom training tables. Temporal training requires
`train_end_offset < train_start_offset` when both are supplied. Prediction
anchors may be in the future, but cannot precede the query's source-data range.

Temporal PQL windows support calendar months, for example
`PREDICT COUNT(events.*, 0, 1, months) FOR EACH users.id`. Month windows and
integer time options respect month-end and leap-year boundaries.

The package uses `kumoapi` model plans and connector plugin contracts.
[Output configurations](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/kumo/engine/trainer/output_config.py)
take connector objects directly.

## Installation

Requires Linux x86_64, CPython 3.11 or newer, and glibc 2.34 or newer.
For ARM64, see the
[DGX Spark installation guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/dgx-spark.md).

```
pip install kumo-engine
```

Install connector dependencies with the `snowflake`, `s3`, `gcs`, `bigquery`,
or `databricks` extra:

```
pip install "kumo-engine[snowflake]"
```

## Text columns

Text columns use word vectors. Install the default model, `glove.6B-50d`, with:

```
pip install "kumo-engine[text]"
```

Register other models in a store and pass it to the trainer:

```python
from kumo.engine import EmbeddingModelStore, GloVeVectors, Trainer
from kumoapi.encoder import TextEmbedding

store = EmbeddingModelStore()
store.add("desc", GloVeVectors("~/Downloads/glove.42B.300d.zip"))
plan.column_processing.encoder_overrides = {
    "articles.detail_desc": TextEmbedding(model="desc"),
}
trainer = Trainer(plan, embedding_models=store)
```

`GloVeVectors` accepts an [official Stanford ZIP](https://nlp.stanford.edu/projects/glove/)
or converted directory. ZIPs are converted on first use; pass `dim=` for archives
with several sizes. Reload with `Trainer.load(bundle, embedding_models=store)`;
vectors are matched by content. Custom models implement `TokenEmbeddingModel`.
Kumo does not download vectors automatically.

## Local files

Read CSV or Parquet below a filesystem root with `FileConnector`; no credentials
or connector extra are required. For `restaurant-data/users.csv` with an `id`
column:

```python
from kumo.engine import FileConnector, Graph, Table, configure

configure(scratch_dir="/mnt/nvme/kumo-scratch")
files = FileConnector(root_dir="/mnt/nvme/restaurant-data")
users = Table.from_source_table(files["users"], primary_key="id")
snapshot = Graph(tables=[users], edges=[]).snapshot()
```

Identifier names such as `user_id` give integer and string columns the `ID`
semantic type, even for non-primary keys. Quantity names such as `num_items`,
`price`, and `score` give integer and floating-point columns the numerical type,
even with few distinct values. Other floating-point columns use sampled values.
Source types and `with_column()` overrides take precedence over inference.

Foreign keys in supported `Edge` relationships receive the `ID` type in the graph;
the source `Table` keeps its original types. Omitted self-relations leave types
unchanged.

Use a table mapping when graph names differ from source names, such as
`Graph(tables={"users": users})` for a table read from `users.parquet`.
Edges and PQL use the mapping keys; connector reads keep the original source names.
Table names are relative paths, optionally with a `.csv`/`.parquet` suffix,
or directories of parts in one format. Absolute paths, URIs, traversal and
globs are unsupported. For CSV timestamps, set `dtype=Dtype.time` (from
`kumoapi.typing`) and `timestamp_format` with `Table.with_column()`.

The root and scratch directory must not overlap. Directory symlinks and links
escaping the root are rejected; file symlinks within the root are allowed.
Keep sources and links unchanged during reads. Versions use paths, sizes and
modification times, not content hashes; do not restore timestamps after edits.
Source identities use canonical paths and are not portable across machines.

Use a separate output root for artifacts. Single-file outputs overwrite; use
an empty directory for partitioned outputs. Append is unsupported. Model
publication copies bundles to an empty destination and rejects symlinks and
overlap. Reload a published bundle with
`Trainer.load_from_connector(connector, path=...)`.

## Local storage and results

Snapshots, generated tables, model bundles, and unexported predictions live under
`~/.cache/kumo-engine`. Call `configure(scratch_dir=...)` before creating graphs
or trainers to choose another local disk. `Graph.snapshot()` warns about
network-backed storage such as EBS. Each `fit()` and `predict()` uses a separate
run directory.

Snapshot indexes are reused by graph definition and source version; changed
versions get separate directories. Builds for the same graph use a filesystem
lock. See the [implementation](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/kumo/engine/graph.py).

To build an index on one machine and train on another, export it through an
S3, GCS or file connector and load it there:

```python
location = graph.snapshot().export(S3Connector(), "s3://bucket/snapshots")
# On the training machine:
snapshot = GraphSnapshot.load_from_connector(S3Connector(), path=location)
```

`export` writes `<path>/<graph id>/<data version>/` and returns it. It first
removes `kumo_snapshot.json` and anything else at that location, uploads every
file, and writes `kumo_snapshot.json` (with per-range CRC32s) last, so an
interrupted export is never loadable and re-running starts over. A load either
completes or leaves nothing; a load that overlaps a re-export of the same
location can fail and should be retried. To keep several builds of one
snapshot (for example from two engine versions), export them to different
paths.
Snowflake, Databricks and BigQuery connectors cannot hold an export: use a
`FileConnector` on a mounted volume (e.g. `/Volumes/<catalog>/<schema>/<volume>`)
or the S3/GCS connector on the bucket behind it. Load only from paths whose
writers you trust: checksums detect corruption, not tampering.

`snapshot.source_changes(graph)` reports, per table, whether the source moved
on since the snapshot was built, from metadata only:

```python
{"users":  {"status": "unchanged", "recorded": "1792539971", "current": "1792539971"},
 "events": {"status": "changed",   "recorded": "1792539971", "current": "1792539974"}}
```

The status is `unknown` when there is nothing reliable to compare: no recorded
version, a different connector or location, Custom SQL, or a version whose
`version_kind` is `hour_bucket` or `unknown` (views, undated sources). `load_from_connector(..., graph=graph)` warns when tables `changed`.

The return values expose the retained files directly:

```python
from kumo.engine import configure

configure(scratch_dir="/mnt/nvme/kumo-scratch")

fit_result = trainer.fit(training_table, snapshot=snapshot)
print(fit_result.bundle_dir)       # pass this path to Trainer.load(...)

prediction_result = trainer.predict(prediction_table, snapshot=snapshot)
print(prediction_result.files)     # selected Parquet files
predictions = prediction_result.frame  # collect them into a pandas DataFrame
```

### Progress events

`fit(callbacks=[...])` calls each callback with every progress `Event`:
`fit_started`, then per search trial `trial_started`, `epoch_ended` and
`trial_ended`, then `fit_ended`. Ended events carry `params["status"]` and, on
failure, a short `params["error"]`. A callback that raises is logged and not
called again; it never fails the fit. For JSON lines on stdout:

```python
import json

trainer.fit(training_table, snapshot=snapshot,
            callbacks=[lambda e: print(json.dumps(e.to_dict()))])
```

### Holdout predictions

`fit_result.holdout_df()` reads test predictions from `holdout_path` into a pandas
DataFrame. Holdouts are excluded from published bundles. Link-prediction tasks
and loaded trainers have no holdout; `holdout_df()` raises `FileNotFoundError`.
Holdouts are capped at 1 GiB in memory. `holdout_truncated` is `True` if rows were
dropped, or `None` if the size record is unreadable; training still succeeds.

To reuse table rows and labels with updated graph features, pass a new snapshot:

```python
updated_snapshot = graph.snapshot()
fit_result = trainer.fit(training_table, snapshot=updated_snapshot)
```

Each fit uses the selected snapshot and saved table artifacts, preserving the
table's rows, labels, splits, and generation settings.

### Warm-start training

Pass a fitted or restored trainer to start from its selected checkpoint:

```python
previous = Trainer.load("/models/previous/published-model")
# Or: Trainer.load_from_connector(registry, location=location)
trainer = Trainer(model_plan)
result = trainer.fit(training_table, snapshot=snapshot, warm_start=previous)
```

Warm start reuses weights, optimizer state, fitted encoders, architecture, and
neighbor sampling. The destination plan controls training and refits; its
`model_plan` stays unchanged. The effective plan is saved in a new bundle, leaving
the source intact. Checkpoint priority is full refit, train/validation refit, then
original fit, depending on which completed. The saved plan takes precedence over
later edits to the source trainer's plan. Task types must match; forecasting warm
starts are unsupported.

If source training or refitting included the destination's held-out rows,
validation and test metrics may be optimistic. Use independent held-out data
for evaluation.

Older bundles may contain an AutoML search space instead of the fitted
architecture. They support prediction but are unreliable warm-start sources.

### Caller-supplied training and prediction rows

A connector table can supply labeled training rows directly. The table uses
the standard `ENTITY1`, `TARGET1` and, for temporal queries, `TIMESTAMP`
columns. A named split column contains `0` for train, `1` for validation and
`2` for test; a weight column carries per-row loss weights:

```python
from kumo.engine import TrainingTable
from kumoapi.train import TrainingTableSpec

training_table = TrainingTable.from_source_table(
    files["curated_training.parquet"],
    TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT"),
    pq=pquery,
    snapshot=snapshot,
)
result = trainer.fit(training_table, snapshot=snapshot)
```

Without a split column, rows use the task's inferred split. Override it with
`TrainingTableGenerationPlan(split=...)`.

### Exporting a generated training table

To train with per-row weights, export a generated table, add a weight column,
and load it back:

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from kumo.engine import FileConnector, TrainingTable, TrainingTableOutputConfig
from kumoapi.train import TrainingTableSpec

training_table = pquery.generate_training_table(snapshot)
exports = FileConnector(root_dir="/data/exports")
training_table.export(TrainingTableOutputConfig(exports, "training_table"))

weighted_dir = Path("/data/exports/weighted")
weighted_dir.mkdir()
for part in sorted(Path("/data/exports/training_table").glob("*.parquet")):
    source = pq.ParquetFile(part)
    schema = source.schema_arrow.append(pa.field("WEIGHT", pa.float64()))
    with pq.ParquetWriter(weighted_dir / part.name, schema) as writer:
        for batch in source.iter_batches():
            weight = pa.repeat(pa.scalar(1.0), batch.num_rows)
            writer.write_batch(batch.append_column("WEIGHT", weight))

weighted = TrainingTable.from_source_table(
    exports["weighted"],
    TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT"),
    pq=pquery,
    snapshot=snapshot,
)
trainer.fit(weighted, snapshot=snapshot)
```

Replace `1.0` with per-row weights. The example processes one batch at a time.
For Snowflake or Databricks exports, add `WEIGHT` in the warehouse and reload.

Exports preserve generated train, validation, and test rows, including entity,
target, timestamp, task-specific columns, and `SPLIT`. Reload with
`split_col="SPLIT"` to preserve each stage. File, S3, and GCS exports use Parquet
directories: the same table can replace its export; a different table needs an
empty directory. Warehouse exports use `write_mode` for overwrite or append.

- For link prediction, `WEIGHT` must be a list with one weight per element of
  `TARGET1`.
- Forecasting tables are loaded back without `split_col`; their split is
  recomputed from the timestamps.
- A static link-prediction table that was split per link loses the rule that
  excludes a user's training links from validation and test ranking, so its
  validation metrics can differ from the original.
- Snowflake stores list columns as JSON strings, so export link-prediction
  and forecasting tables to files.

A custom prediction table supplies `ENTITY1` and, for temporal queries,
`TIMESTAMP`, with no target column:

```python
from kumo.engine import PredictionTable

prediction_table = PredictionTable.from_source_table(
    files["curated_population.parquet"],
    pq=pquery,
    snapshot=snapshot,
    plan=trainer.training_table_generation_plan,
)
predictions = trainer.predict(prediction_table, snapshot=snapshot)
```

Both constructors stage and validate rows immediately, retaining local paths and
query/generation metadata. Select graph data at `fit` or `predict` with `snapshot=`.

Keep the scratch directory while those paths are needed. Use
`PredictionOutputConfig` with `Trainer.predict(..., output_config=...)` or
`result.export(...)` to export predictions. Use `ModelOutputConfig` with
`Trainer.fit(..., export_config=...)` to publish the fitted model bundle.

Custom forecasting rows must include precomputed `future_labels` for training,
`autoregression_labels` for lag features, and `year_over_year_labels` for
year-over-year features when enabled. Constructors preserve these sequences;
they do not derive them from entity, target, and timestamp columns.

## Experiment tracking

`fit(tracking=...)` records a fit in an experiment tracker: one run per fit,
one run per hyperparameter-search trial attempt. It is opt-in; tracking errors
are logged as warnings and never fail the fit. Only global rank 0 tracks.

```python
from kumo.engine.integrations.mlflow import MlflowTracking  # kumo-engine[mlflow]

result = trainer.fit(training_table, snapshot=snapshot,
                     tracking=MlflowTracking(experiment_name="users-spend"))
```

MLflow runs go to `MLFLOW_TRACKING_URI`: on Databricks this is automatic; for
SageMaker managed MLflow set it to the tracking server ARN with the
`sagemaker-mlflow` plugin installed. With nothing configured MLflow writes a
local store in the working directory.

Pass a list to record to several trackers; each is isolated from the
others. Any other system plugs in through the `kumo.engine.Tracking` protocol:

- `start(context)`: a `FitContext` once the model plan is resolved;
- `on_event(event)`: every progress `Event` (trial events carry
  `params["attempt"]`);
- `finish(result, error)`: once, when the fit returns or raises;
- optional `trial_logger(trial, attempt)`: a Lightning logger for one trial
  attempt's step metrics, or `None`;
- optional `capabilities`: a `Capabilities` stating what the adapter records
  (nested runs, system metrics, step metrics, artifacts, model links).

Build what to log with `run_record(context, result, fit_ended)` and
`trial_record(context, trial_events)`, so every adapter uses the same
parameter, metric and tag names:

```python
import json
from kumo.engine import run_record, trial_record

class JsonTracking:
    def start(self, context):
        self.context, self.trials, self.ended = context, {}, None
        print(json.dumps({"run": run_record(context).tags}))

    def on_event(self, event):
        if event.trial is not None:
            key = (event.trial, event.params["attempt"])
            self.trials.setdefault(key, []).append(event)
            if event.name == "trial_ended":
                record = trial_record(self.context, self.trials.pop(key))
                print(json.dumps({"trial": record.name, "params": record.params,
                                  "metrics": record.metrics}))
        elif event.name == "fit_ended":
            self.ended = event

    def finish(self, result, error):
        if result is not None:
            record = run_record(self.context, result, self.ended)
            print(json.dumps({"metrics": record.metrics}))

trainer.fit(training_table, snapshot=snapshot, tracking=JsonTracking())
```

## BigQuery staged ingestion

BigQuery streams through the Storage Read API by default. For large tables,
configure an owned GCS prefix to export Parquet and download it locally:

```python
from kumo.engine import BigQueryConnector

events = BigQueryConnector(
    project_id="data-project",
    dataset_id="analytics",
    staging_uri="gs://data-staging/kumo/run-123",
    require_staged=True,
)
```

`require_staged=True` fails if staging is unavailable. With only `staging_uri`,
reads try staging first and can fall back to streaming.

## Databricks staged ingestion

Databricks streams through CloudFetch by default. To unload Parquet through
an existing Unity Catalog volume and download it locally:

```python
from kumo.engine import DatabricksConnector, DatabricksIdentity

identity = DatabricksIdentity(
    host="https://workspace.cloud.databricks.com",
    client_id="service-principal-id",
    client_secret="...",
)
events = DatabricksConnector(
    catalog="main",
    schema_name="analytics",
    warehouse_id="warehouse-id",
    identity=identity,
    staging_volume_path="/Volumes/main/kumo/staging",
    require_staged=True,
)
```

`require_staged=True` disables streaming fallback. With only
`staging_volume_path`, reads try staging first and retain streaming fallback.
The identity needs access to both the volume and source table.

## Custom SQL tables

Snowflake, Databricks, and BigQuery connectors accept read-only `SELECT` queries.
Results use the same ingestion and staging options as named tables:

```python
from kumo.engine import Edge, Graph, Table

active_users = Table.from_sql(
    events,
    "SELECT user_id, country, signup_ts FROM users WHERE active",
    primary_key="user_id",
    time_column="signup_ts",
)
graph = Graph({"active_users": active_users, "orders": orders},
              edges=[Edge(src_table="orders", fkey="user_id",
                          dst_table="active_users")])
snapshot = graph.snapshot()
```

Alternatively, pass `events.sql(query)` to `Table.from_source_table`. Queries must
be one statement with named output columns, restricted to the connector's
database/catalog/project and schema/dataset. Inference samples query results;
BigQuery skips sampling queries estimated to scan more than 10 GiB. To skip
inference, pass `Table(events, name, definition=kumoapi.table.TableDefinition(
cols=..., source_table=events.sql(query).to_source_table(), ...))`.

## Model-plan suggestions

Given `pquery = PredictiveQuery(graph, query)` and `snapshot = graph.snapshot()`,
suggest and customize a plan before training. A scoring job without `Graph`
code loads the snapshot the training job exported
(`GraphSnapshot.load_from_connector(store, path=location)`, where `location` is
what `graph.snapshot().export(store, 'snapshots')` returned) and scores it with
`trainer.predict_snapshot(snapshot)`, which reuses the trained query and
table-generation plan (`entities=` scores given rows instead;
`prediction_time=`, `output_config=` and the thresholds are as in `predict`;
it warns when the snapshot was taken of a different graph than the training
snapshot).
`PredictiveQuery.from_snapshot(snapshot, trainer.training_query)` binds the
query directly. Its graph check only means the snapshot belongs to the query's
graph, not that it is compatible with the trained model.

```python
from kumo.engine import Trainer
from kumoapi.model_plan import RunMode

plan = pquery.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
print(plan.training_job.metrics)
plan.optimization.max_epochs = 5
trainer = Trainer(plan)
```

Suggestions use graph/query metadata. Inferred settings resolve during training.

## Status

The pipeline supports graph ingestion, query validation, plan suggestions,
training/prediction table generation, GNN training, batch prediction, and model
reload from local files, S3, GCS, or Snowflake.

Limitations:

- `PredictiveQuery.suggest_prediction_table_plan`/`suggest_distilled_model_plan`
  are deferred.
- BigQuery cannot publish or reload model bundles.
- Scoped per-table ingest credentials, and a table staged across multiple
  directories, both raise `NotImplementedError` in `Graph.snapshot()`.

### Reloading a local model

Every successful fit writes a reloadable bundle to `result.bundle_dir`:

```python
result = trainer.fit(training_table, snapshot=snapshot)
# In a fresh Python process:
restored = Trainer.load(result.bundle_dir)  # pass the recorded path
```

For an exported model, use
`Trainer.load_from_connector(connector, path="model/version")`, or
`location=ModelRegistryLocation(...)` for Snowflake Model Registry. A scoring
job with `Graph` code rebuilds the live graph and snapshot; one without loads an
exported snapshot with `GraphSnapshot.load_from_connector(...)` and calls
`trainer.predict_snapshot(snapshot)`. Bundles contain trained artifacts, not
source data or graph storage.

With `export_config`, `result.published_model` records the assigned name, version,
and URI; otherwise it is `None`. Snowflake also records training metrics and
metadata for model discovery and comparison. Retain the assigned version for reload:

```python
result = trainer.fit(
    training_table,
    snapshot=snapshot,
    export_config=ModelOutputConfig(
        snowflake, location=ModelRegistryLocation("CHURN_GNN")),
)
published = result.published_model
restored = Trainer.load_from_connector(snowflake, location=ModelRegistryLocation(
    published.name, version=published.version))
```

S3, GCS, and file connectors expose the published path or URI in the same result.

## Prediction and embedding destinations

Request both artifacts with separate destinations. Mapping keys select which
artifacts to generate; paths receive no artifact-name suffix. Use a trailing
`/` for a dataset directory, regardless of the number of output parts.

```python
from kumo.engine import PredictionOutputConfig, S3Connector

outputs = PredictionOutputConfig(
    connector=S3Connector(),
    output_paths={
        "predictions": "s3://my-bucket/predictions/project/",
        "embeddings": "s3://my-bucket/embeddings/project/",
    },
)
result = trainer.predict(prediction_table, snapshot=snapshot,
                         output_config=outputs)
```

S3, GCS, and file connectors support `output_paths` using configured or ambient
credentials. Rooted connectors accept relative paths. Do not combine
`output_paths` with `artifact_type` or `table_name`. ADLS is unsupported.

`result.files`, `result.frame`, and `result.num_rows` select predictions when both
artifacts are requested, or embeddings for embeddings-only requests.
`result.export(outputs)` copies existing artifacts without rerunning inference.
Choose `extra_fields` during `predict`; `export()` rejects them. Directory retries
can reuse their own output; different exports need an empty destination. Explicit
`.parquet`/`.csv` paths retain single-file/multipart behavior.

Destinations export independently; failures may leave completed outputs and partial
directories. Retry `result.export()` with the same destinations. Directory retries
delete `_SUCCESS` before copying and restore it on completion, so delete permission
is required. Readers should wait for `_SUCCESS`. Exports are not transactional
across destinations, atomic for single files, or isolated from concurrent access.

## BigQuery prediction output

A `BigQueryConnector` writes predictions or embeddings to a table in its own
dataset, either during `predict` or later through `result.export`:

```python
from kumo.engine import (
    BigQueryConnector,
    BigQueryIdentity,
    PredictionOutputConfig,
)
from kumoapi.jobs import PredictionArtifactType, WriteMode

output = BigQueryConnector(
    identity=BigQueryIdentity(),  # application default credentials
    project_id="data-project",
    dataset_id="predictions",
    staging_uri="gs://data-staging/kumo",
)
result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    binary_classification_threshold=0.5,
    output_config=PredictionOutputConfig(
        connector=output,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name="user_scores",
    ),
)
# Later, without rerunning inference:
result.export(PredictionOutputConfig(
    connector=output,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name="user_scores_history",
    write_mode=WriteMode.APPEND,
))
```

- `table_name` is an unqualified table ID. Each configuration writes one
  artifact; `output_paths` is for file and object-store connectors.
- `OVERWRITE` replaces rows and schema. `APPEND` requires matching destination
  columns and types; integers also fit `FLOAT`/`NUMERIC`, and all-null columns fit
  any type. Schema mismatches fail before writing.
- Rows commit together. Retrying `APPEND` duplicates rows; appending to a missing
  table creates it.
- With a usable `staging_uri`, files are uploaded under
  `<staging_uri>/_kumo.export/<id>/`, loaded in one job, then deleted. Otherwise,
  the Storage Write API commits pending streams together. Streamed `OVERWRITE`
  uses an expiring table and a `WRITE_TRUNCATE` query. `require_staged=True`
  disables streaming fallback; staging requires a colocated bucket.
- Prefer staging for large exports. Streaming and the overwrite query can incur
  charges. Streamed rows are queryable immediately, but table metadata can lag
  and immediate copy jobs may miss rows.
- `OVERWRITE` preserves partitioning, clustering, and column descriptions and
  policy tags for unchanged names/types. Streamed overwrites restore column
  metadata after replacing rows.
- Embeddings land as `REPEATED FLOAT` columns, timestamps as `TIMESTAMP`.
- Before scoring, `predict` checks dataset location, write access, and required
  staging. Grant `roles/bigquery.dataEditor` on the dataset,
  `roles/bigquery.jobUser` on the billing project, and
  `roles/storage.objectAdmin` on the staging bucket when used.
