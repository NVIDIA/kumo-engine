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

### Scoring defaults

A bundle can carry the prediction arguments `predict` uses when a call omits them:

```python
from kumo.engine import ModelOutputConfig, ModelRegistryLocation, ScoringDefaults

tuned = trainer.with_scoring_defaults(
    ScoringDefaults(binary_classification_threshold=0.37))
tuned.predict(prediction_table, snapshot=snapshot)  # thresholds at 0.37
tuned.export(ModelOutputConfig(registry, location=ModelRegistryLocation("CHURN_GNN")))
```

`with_scoring_defaults` returns a trainer on a copy with its own `bundle_id`; the copy's
`provenance.parents.derived_from` names the original. The threshold applies to
binary and multilabel classification, `num_classes_to_return` to multiclass, multilabel
and link prediction. Explicit `predict` arguments and a query's `TOP k` win;
`ScoringDefaults()` removes the defaults. `Trainer.export(config)` publishes a fitted or
loaded bundle like `fit(export_config=...)`.

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
with each hyperparameter-search trial attempt as a child run (MLflow) or
inside the fit run (Snowflake). It is opt-in; tracking errors are logged as warnings and never fail the fit. Only global rank 0 tracks.
Tracker calls run on a background thread, so a slow or unreachable tracking
server does not hold up training: at the end of the fit, `fit` waits at most
30 s for tracking to finish (the rest continues in the background). During
the fit, MLflow HTTP calls use 2 retries and a 30 s timeout unless
`MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` are set.

```python
from kumo.engine.integrations.mlflow import MlflowTracking  # kumo-engine[mlflow]

result = trainer.fit(training_table, snapshot=snapshot,
                     tracking=MlflowTracking(experiment_name="users-spend"))
```

MLflow runs go to `MLFLOW_TRACKING_URI`: on Databricks this is automatic; for
SageMaker managed MLflow set it to the tracking server ARN with the
`sagemaker-mlflow` plugin installed. With nothing configured MLflow writes a
local store in the working directory. Per-epoch metrics use step = epoch.
After the fit, `tracking.run_id` is the fit run's MLflow run id.

The fit's source tables (from the snapshot manifest, plus a custom training
table's source) are logged as MLflow run inputs: Databricks tables as Delta
tables at the snapshot's Delta version, so Unity Catalog links each table to a
model registered from the run in a Databricks notebook or job; file and
object-store sources by URI. Other warehouse tables and Custom SQL sources are
not logged as inputs; they stay in the bundle's `provenance.json`.

Snowflake records the fit in Snowflake Experiment Tracking (Snowsight: AI & ML
» Experiments), using a Snowpark session:

```python
from kumo.engine.integrations.snowflake import SnowflakeTracking  # kumo-engine[snowflake]

result = trainer.fit(training_table, snapshot=snapshot,
                     tracking=SnowflakeTracking(session, experiment_name="users_spend"))
```

Snowflake has no child runs or tags, so trial attempts are logged in the fit
run as `trial-1-attempt-1/<name>` params and metrics, and tags are params of
the same name. `native_model_lineage=True` links a model the fit publishes
through Snowflake lineage. Names beyond Snowflake's 200 metrics and 1000 params
per run are counted in `kumo.dropped_metrics` / `kumo.dropped_params`. If a
Snowflake run is already active in the process, the fit is not tracked there;
an existing run is never resumed (a taken `run_name` gets a suffix). Snowflake
Experiment Tracking has no dataset inputs, so source tables are not recorded
there. The Snowpark session's parameters are left as they are. After the fit,
`tracking.run_id` is the fit run's name.

Other systems plug in through the `kumo.engine.tracking.Tracking` protocol,
which receives ready-built `RunRecord`s so every tracker records the same
names: `start(record)` with the fit run, `on_trial(record, event)` with a trial
attempt's run after each of its events, and `finish(record, error)` with the
fit run's final record (`None` when tracking stopped early). A tracker that
raises is not called again for that fit and its run is ended as failed.
Params are the resolved model plan (fit) or searched hyperparameters and
`attempt` (trial). Runs carry pointers, not lineage: `kumo.kind` and
`kumo.run_id` (plus `kumo.parent_run`, `kumo.status` and `kumo.error` for
outcomes); lineage stays in the model bundle, whose `provenance.json` is logged
on the fit run when the bundle has one, with `query.pql` and `model_plan.json`.
`RunRecord.inputs` maps each source table to its `connector`, `location`,
`version` and `version_kind`. Your `tags=` are added as given, except `kumo.*`
names and param names.

The fit run also carries `kumo.bundle_id` once the fit has a bundle, and
under a scheduler `kumo.scheduler`, `kumo.job_id` and `kumo.job_run_id`. In the
other direction, a tracked fit's bundle records `"tracking": {"run_id": ...}`
in `provenance.json`, the `kumo.run_id` every tracker tags its fit run with.

`predict(tracking=...)` (and `predict_snapshot(tracking=...)`) take the same
tracker and record each prediction run as one more run named by its kumo run
id: scoring options as params; row counts and score statistics (for example
`predictions.True_PROB.q50`) as metrics; the pointers `kumo.kind`
(`prediction`), `kumo.run_id`, `kumo.bundle_id` (which joins it to the fit run)
and the scheduler's `kumo.scheduler`, `kumo.job_id` and `kumo.job_run_id`; and
`prediction-manifest.json` as an artifact, which holds the run's lineage.
If an export raises, the run is still recorded, ended as failed with a
`kumo.error` tag, and `predict` raises the export's error.
Custom trackers opt in with a `log_prediction_run(record)` hook (a record with
a `kumo.error` tag is a failed run). The call runs
on the tracking thread with the fit's time limit, and tracking errors never fail
the prediction.

```python
scores = trainer.predict(prediction_table, snapshot=snapshot,
                         binary_classification_threshold=0.5,
                         tracking=MlflowTracking(experiment_name="users-churn"))
```

MLflow models: `log_model(trainer, snapshot=...)` logs a bundle as a pyfunc
model, with the bundle's source tables as inputs of the run it is logged in,
that scores where `kumo-engine[s3,gcs]` of the same version is installed (its
pip requirement); `load_model(uri)` restores the `Trainer`.
`MlflowTracking(registered_model_name=..., aliases=...)` logs, registers and
aliases the fit's model on the fit run (a failed fit logs none). `fit()` waits
for the upload and registration before it returns, for up to an hour, with a
600 s MLflow HTTP timeout during the upload (unless
`MLFLOW_HTTP_REQUEST_TIMEOUT` is set); if they fail, the tracking thread stops
or the hour passes, `fit()` warns, still returns its result, the run ends
`FAILED`, and a model that finishes uploading later is not registered. The model's metadata is `kumo_bundle_id`; its `provenance.json`
artifact is the bundle's. A tracker can do the same through the optional
`log_model(result, snapshot)` protocol member.

```python
from kumo.engine.integrations.mlflow import MlflowTracking

result = trainer.fit(training_table, snapshot=snapshot, tracking=MlflowTracking(
    experiment_name="/Shared/users_spend",
    registered_model_name="main.ml.users_spend", aliases=["champion"]))
```

```python
model = mlflow.pyfunc.load_model("models:/users-spend@champion")
scores = model.predict(entities, params={"snapshot": "s3://<bucket>/snapshots/<id>"})
```

The pyfunc model loads a snapshot on its first `predict` with that URI and
keeps it for the life of the process: each serving replica, and each Spark
executor of `mlflow.pyfunc.spark_udf`, downloads the snapshot once and keeps
every snapshot it has loaded (no eviction), so its disk must hold them. With
`serving_snapshot=<exported snapshot URI>` (on `log_model` or
`MlflowTracking`), each replica downloads that snapshot when it starts, and
`predict` uses it when no `snapshot` param is given.
`params={"population": True}` scores every entity and returns the whole
frame in memory; for large populations use `Trainer.predict` with a
`PredictionOutputConfig`. Each `predict` call works in its own scratch
directory and removes it before returning.

The `mlflow` extra needs an MLflow 3.13 or later client: DiskGraph pins
pyarrow 24, which older MLflow releases exclude. SageMaker managed MLflow
servers run MLflow 3.0; a 3.17 client against a 3.0 server handles logging,
registration, aliases and `load_model`.

On Databricks, the parent folder of an MLflow experiment path must exist.
Registering to Unity Catalog from outside Databricks also needs `boto3`, which
the `mlflow` extra does not install.

#### Which registry

There are two ways to register a model; pick one per registered model, since
the two store different formats. Both record only the `kumo_bundle_id`
pointer; lineage stays in the bundle's `provenance.json`, which an MLflow
model also carries as its `provenance.json` artifact. An MLflow model logs
each source table (and a custom training table) as a run input, Unity
Catalog tables as Delta datasets. Both accept the same aliases
(`kumoapi.plugin.check_model_alias`).

| | Connector registry | MLflow model |
|---|---|---|
| Publish | `fit(export_config=ModelOutputConfig(connector, location=ModelRegistryLocation(name, alias=...)))` or `Trainer.export` | `MlflowTracking(registered_model_name=..., aliases=[...])` or `log_model(...)` |
| Load | `Trainer.load_from_connector(connector, location=...)` | `mlflow.pyfunc.load_model(uri)` or `load_model(uri)` |
| Promote | `set_model_alias(connector, location, alias)` | MLflow's `set_registered_model_alias` |
| Databricks | `DatabricksConnector` (Unity Catalog) | Unity Catalog through MLflow |
| SageMaker | `S3Connector` path | SageMaker managed MLflow (`MLFLOW_TRACKING_URI` = tracking server ARN) |
| Snowflake | `SnowflakeConnector` (Snowflake Model Registry) | not available |
| Plain VM | `FileConnector`, `S3Connector`, `GCSConnector` path (no aliases) | any MLflow tracking server |

## Scheduled jobs

Wrap a job script's body in `kumo.job()` so a scheduler gets one JSON summary
line on stdout and an exit code it can act on:

```python
import kumo.engine as kumo

with kumo.job() as run:
    snapshot = graph.snapshot()
    result = kumo.Trainer(plan).fit(training_table, snapshot=snapshot)
    run.record(bundle_id=result.bundle_id, snapshot_id=snapshot.id)
```

The summary carries `status`, `exit_code`, the scheduler context, the recorded
ids and outputs, the first 20 distinct warnings other than deprecations (300
characters each), and `error` on failure. The line is printed
however the body ends, including Ctrl-C and `sys.exit()`. The job exits 0 on
success, 1 on failure and 143 on SIGTERM; `error.kind` says whether a failure
is `transient` (a source, connection or timeout error worth retrying),
`permanent` (a source refused the request) or `unknown`. Scheduler ids come
from `job(job_id=..., job_run_id=...)`, `KUMO_JOB_ID`/`KUMO_JOB_RUN_ID`, or the
detected platform, and are recorded in every manifest and tracking run the job
writes. Logs go to stderr.

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
`location=ModelRegistryLocation(...)` for Snowflake or Databricks. A scoring
job with `Graph` code rebuilds the live graph and snapshot; one without loads an
exported snapshot with `GraphSnapshot.load_from_connector(...)` and calls
`trainer.predict_snapshot(snapshot)`. Bundles contain trained artifacts, not
source data or graph storage.

With `export_config`, `result.published_model` records the assigned name, version,
and URI; otherwise it is `None`. Retain the assigned version for reload:

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

### Publish and promote

Snowflake Model Registry and Databricks Unity Catalog connectors support aliases:

```python
from kumo.engine import ModelOutputConfig, ModelRegistryLocation, set_model_alias

result = trainer.fit(training_table, snapshot=snapshot, export_config=ModelOutputConfig(
    registry, location=ModelRegistryLocation("CHURN_GNN", alias="challenger")))
published = result.published_model

# Promote: the alias moves from any version that held it.
set_model_alias(registry, ModelRegistryLocation(
    published.name, version=published.version), "prod")

# Scoring jobs load whichever version "prod" points at.
trainer = Trainer.load_from_connector(registry, location=ModelRegistryLocation(
    "CHURN_GNN", alias="prod"))
```

An alias is a letter or underscore followed by letters, digits or underscores;
`DEFAULT`, `FIRST`, `LAST`, `LATEST` and `v<digits>` are reserved, and `fit()`
rejects an invalid export alias before training. The alias is set after
publishing; if that fails, `fit()` warns and still returns the published model.
Snowflake allows one alias per version; if moving an alias fails, it is put
back on the version that held it. Snowflake's `DEFAULT`, `FIRST` and `LAST` can
be loaded but not set. File, S3 and GCS connectors have no aliases.

Each published version records only pointers: the `kumo_engine` marker and
`kumo_bundle_id` (Snowflake version metrics, Databricks version tags). Training
metrics stay in the bundle's `metrics.json` and on the tracked fit run. The bundle's own `provenance.json` holds its
lineage. `list_models` and `fetch_model` return `kumo_bundle_id` in
`PublishedModel.metadata`; `load_from_connector`
warns if the fetched bundle differs from the recorded `kumo_bundle_id`.

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

`PredictionOutputConfig(..., lineage_columns=True)` passed to `predict` writes
constant `KUMO_RUN_ID`, `KUMO_BUNDLE_ID` and `KUMO_SNAPSHOT_ID` string columns
into every output row (`result.files`, `result.frame` and every export); unknown
ids are omitted. It changes the destination's column set, so enable it on a new
table or directory: on a table created without it, Snowflake `OVERWRITE`,
BigQuery `APPEND` and Databricks `APPEND` fail, and Snowflake `APPEND` silently
drops the new columns. `export()` copies whatever columns the result has.

Destinations export independently; failures may leave completed outputs and partial
directories. Retry `result.export()` with the same destinations. Directory retries
delete `_SUCCESS` before copying and restore it on completion, so delete permission
is required. Readers should wait for `_SUCCESS`. Exports are not transactional
across destinations, atomic for single files, or isolated from concurrent access.

## Monitoring prediction runs

Every `predict()` writes `prediction-manifest.json` beside the run's scratch
outputs (`result.manifest`): parents (`bundle`, `snapshot`, `prediction_table`),
the snapshot's `source_tables`, anchors, scoring options, `outputs` (one record
per export), timings and output score statistics. Parquet directory exports
carry a copy as `_kumo_manifest.json`. A row's lineage columns lead back to its
sources: `KUMO_RUN_ID` names the manifest, whose `source_tables` give each
table's connector, location and version.

`trainer.evaluate(result, snapshot=newer_snapshot)` regenerates the query's
labels at the run's anchors on a newer snapshot, joins them with the run's
predictions on `ENTITY` (and `ANCHOR_TIMESTAMP`), and returns an
`EvaluationResult` with `metrics`, `labelled_fraction` and `actuals`, a Parquet
directory of the joined labels (`ENTITY`, `ANCHOR_TIMESTAMP`, `TARGET`) next to
the run. DiskGraph streams both sides from Parquet and hash-partitions the join,
so memory follows the partition size rather than the run size. It never
refuses; flags say how far to trust it: `matured` is `False` while the label
window ends after the earliest latest-event time of the query's target tables,
and `bundle_matches` compares the run's model with the trainer's. Static
queries read labels at the snapshot, so `anchors` is empty and `matured` is
`None`. A failed step leaves `metrics` empty with a `reasons` entry.

Metrics use the fit-time `test_*` definitions: binary `auroc`/`ap` (exact,
scores compared at float32 precision); regression `mae`/`rmse`; multiclass `acc`
of the top-scored `CLASS`; link prediction `map@k`/`precision@k`, with `k` the
classes returned per entity and anchor, over predictions with a target. Link
prediction labels exist only for entities with a target in the window, so
`labelled_fraction` is the share with any future link.

`trainer.score_drift(result, previous=None, snapshot=None,
training_snapshot=None)` returns a `DriftReport`: PSI of the run's scores
against the bundle's holdout and against a previous run, and per-table row and
per-edge count changes between the training and scoring snapshots. PSI is taken
over the baseline's deciles, reading each side's share per bin off its recorded
percentile curve (linear between min, q01..q99 and max). Missing records leave
a field `None` with a `reasons` entry.

To monitor in a later job, export the run to a Parquet directory and load it
back:

```python
from kumo.engine import PredictionResult, Trainer

trainer = Trainer.load_from_connector(registry, path="models/churn/")
run = PredictionResult.load_from_connector(connector, "predictions/2024-06-01/")
report = trainer.evaluate(run, snapshot=newer_snapshot)
drift = trainer.score_drift(run)
```

`load_from_connector` requires the export's `_SUCCESS` marker and fetches only
the run's `_kumo_manifest.json`; `score_drift` needs nothing else, and the
Parquet parts are downloaded in parallel when `evaluate` (or `files`) first
reads them, into a scratch directory removed with the result. A run exported by an older engine has no manifest: temporal queries then need
the `ANCHOR_TIMESTAMP` column (request it with `extra_fields`), and
`score_drift` reports `psi` as `None`.

For warehouse model monitors (Snowflake ML Observability, Databricks inference
profiles), predict with `extra_fields=[MetadataField.ANCHOR_TIMESTAMP]` and
`lineage_columns=True`, and add actuals from `evaluate(...).actuals` joined on
`ENTITY` (and `ANCHOR_TIMESTAMP`). Use `ANCHOR_TIMESTAMP` as the timestamp,
`True_PROB` or `TARGET_PRED` as the prediction, `TARGET` as the label and
`KUMO_BUNDLE_ID` as the model id. Neither supports link prediction or ranking
outputs, and Kumo does not yet set `task=` on Snowflake model versions, so a
Snowflake monitor cannot attach to a Kumo-published version today.

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
