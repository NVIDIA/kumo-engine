# MLOps with Kumo Engine

## Turning MLOps on

`kumo.MLOps` says where artifacts go and what records them. Use it as a block, which notebooks and scheduled scripts alike can do, or pass it as `mlops=` on a single call:

```python
import kumo.engine as kumo

cfg = kumo.MLOps(artifact_root="s3://bucket/kumo")   # or (connector, path), e.g. a volume or stage
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
```

- **`artifact_root`** is required. It is a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach.
- **`export`** names the artifact kinds stored under the root when they are produced: `snapshot`, `training_table`, `model` and `prediction` (default: all).
- **`tracking`** and **`registry`** record fits and prediction runs and register models.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has. Plain exports (`export(config)`, `export_config=`, `output_config=`) never write to the artifact root.

## Lineage

Every artifact carries a `_provenance.json` manifest, metadata only. The leading underscore keeps Spark and Hive Parquet readers from treating it as data.

A graph snapshot's manifest, next to its index, records each source table's connector, location and version, plus the engine build that wrote it. `graph.snapshot(record_sources=False)` keeps only connector and table names, and keeps Custom SQL only as a hash.

### Training tables

A generated or caller-supplied training table carries `_provenance.json` in its staging directory:
- the query, the effective generation plan and the task type;
- the split seed;
- the snapshot it was generated from (that snapshot's manifest, embedded);
- for `TrainingTable.from_source_table`, the source table's connector, location and version.

`TrainingTable.id` is deterministic: the same snapshot, query, plan, split seed and engine build give the same id. A caller-supplied source without a version gets a random id. An export to files (`TrainingTable.export`) writes the manifest next to the Parquet parts. `TrainingTable.load_from_connector(connector, path=..., pq=..., snapshot=...)` materializes an export again.

With MLOps on, each table is stored once under `<root>/training_tables/<id>/`.

### Models

Every fitted bundle carries `_provenance.json` (`ModelManifest`):
- its `id`, the bundle id (a content hash that leaves the manifest file out);
- when it was fitted and with which fit mode;
- its parents: the training table's manifest, embedded with that table's snapshot, and the warm-start model's manifest, if any.
- `holdout`: statistics of the scores the published model gave its holdout rows (row count, truncation, and per score column the count, mean, min, max and percentiles q01..q99; the label's positive rate for binary tasks). Ranked outputs are summarised by each prediction's top score. These are the baseline `score_drift` compares prediction runs against; they are left out where the model is embedded in another manifest.

An embedded model keeps only the ids of its own model parents, so a long chain of warm-start retrains does not grow each manifest. `fit` returns a new `TrainingResult` per call (`.id`, `.manifest`); `Trainer.result` and `Trainer.manifest` describe the latest fit or the loaded bundle.

With MLOps on, each bundle is stored under `<root>/models/<bundle id>/`. A complete copy that is already there is kept; a partial one is replaced.

### Scoring defaults

A model can carry the prediction arguments `predict` uses when a call omits them:

```python
from kumo.engine import ModelOutputConfig, ScoringDefaults

tuned = result.with_scoring_defaults(
    ScoringDefaults(binary_classification_threshold=0.37))
tuned.predict(snapshot)  # thresholds at 0.37
tuned.scoring_defaults   # ScoringDefaults(binary_classification_threshold=0.37, ...)
```

`with_scoring_defaults` returns a `TrainingResult` for a copy of the bundle with its own id. The copy's manifest embeds the original's in `parents.derived_from`. The threshold applies to binary and multilabel classification, `num_classes_to_return` to multiclass, multilabel and link prediction. Explicit `predict` arguments and a query's `TOP k` win; `ScoringDefaults()` removes the defaults. `Trainer.export(config)` publishes a fitted or loaded bundle like `fit(export_config=...)`.

### Registering models

With `registry=ModelOutputConfig(connector, location=...)`, every fit in the block registers its model in the connector's registry (Snowflake Model Registry, or Unity Catalog through a `DatabricksConnector`) after it is stored and tracked:

```python
from kumo.engine import ModelOutputConfig, ModelRegistryLocation

cfg = kumo.MLOps("s3://bucket/kumo", registry=ModelOutputConfig(
    registry, location=ModelRegistryLocation("CHURN_GNN", alias="challenger")))
with kumo.mlops(cfg):
    result = trainer.fit(training_table, snapshot=snapshot)
result.registered_model   # PublishedModel(name='CHURN_GNN', version='V3', uri=...)

# Promote: the alias moves from any version that held it.
kumo.set_model_alias(registry, ModelRegistryLocation("CHURN_GNN", version="V3"), "prod")

# Scoring jobs load whichever version "prod" points at.
trainer = kumo.Trainer.load_from_connector(registry, location=ModelRegistryLocation(
    "CHURN_GNN", alias="prod"))
```

- Registration is independent of `export`. If it fails, `fit` warns and returns the result with `registered_model=None`; an invalid alias or a connector that cannot publish models raises before training.
- A plain `fit(export_config=ModelOutputConfig(connector, location=...))` or `Trainer.export(...)` publishes as before, with or without a block, and also accepts an `alias`; its version is `result.published_model`.
- An alias is a letter or underscore followed by letters, digits or underscores; `DEFAULT`, `FIRST`, `LAST`, `LATEST` and `v<digits>` are reserved. The alias is set after publishing; if that fails, the publish still stands, with a warning. Snowflake allows one alias per version; if moving an alias fails, it is put back on the version that held it. Snowflake's `DEFAULT`, `FIRST` and `LAST` can be loaded but not set. File, S3 and GCS connectors have no aliases.
- Each registered version records only pointers: the `kumo_engine` marker and `kumo_bundle_id` (Snowflake version metrics, Databricks version tags). The bundle's own `_provenance.json` holds its lineage. `list_models` and `fetch_model` return `kumo_bundle_id` in `PublishedModel.metadata`; `load_from_connector` warns if the fetched bundle differs from it.

### Prediction runs

Every `predict` call records the run in `_provenance.json` (`PredictionManifest`, `result.manifest`) next to its outputs:
- its `id` (`result.id`);
- its parents: the model's manifest (without its holdout statistics) and the scoring snapshot's manifest, both embedded; `result.bundle_id` and `result.snapshot_id` read their ids;
- the prediction table: a hash of the query, whether it was generated or supplied, its anchor times and row count;
- the scoring options and where each came from (`argument`, `bundle`, the query or a default);
- `outputs`: the destinations of `predict(output_config=...)`, recorded even when that export fails part-way;
- `status`, timings, and per output the row count and score statistics (min, q01..q99, max), which `score_drift` reads.

A parent without a manifest, such as a bundle saved before this release, is recorded by its id alone. Exports to files carry the manifest next to the Parquet parts (not CSV). A later `result.export(...)` copies the files and the manifest as they are and records nothing new.

With MLOps on, each run is stored after its `output_config` export, under `<root>/predictions/<id>/predictions/` (and `embeddings/`), with the manifest, including those `outputs`.

### Moving snapshots between machines

To build an index on one machine and train or score on another, export it to a file destination and load it there:

```python
from kumo.engine import GraphSnapshot, S3Connector, SnapshotOutputConfig

location = graph.snapshot().export(SnapshotOutputConfig(S3Connector(), "s3://bucket/snapshots"))
# On the training machine:
snapshot = GraphSnapshot.load_from_connector(S3Connector(), path=location)
```

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, the two halves of `snapshot.id`, and returns that location.
- **Transfer:** files move in parallel with a CRC32 per range. An interrupted export resumes, and a load either completes or leaves nothing.
- **Reuse:** loading again reuses the local copy, unless that copy fails DiskGraph's integrity check; then it is downloaded again.
- **Manifest:** the snapshot's `_provenance.json` travels with the export.
- **Trust:** load only from paths whose writers you trust. Checksums detect corruption, not tampering.
- **Destinations:** any destination whose sink holds files at the path works: S3, GCS, local files, or a volume or stage once those sinks are installed. A warehouse connector raises.

With MLOps on (`graph.snapshot(mlops=...)` or a `kumo.mlops` block), each snapshot is exported under `<root>/snapshots/`, unless a complete export of it is already there.

### Scoring without the graph

A scoring job does not need the training job's `Graph` code. It loads the snapshot the training job exported and the model, then scores:

```python
snapshot = GraphSnapshot.load_from_connector(store, path=location)
trainer = Trainer.load_from_connector(store, path="models/<bundle id>")
result = trainer.predict(snapshot, binary_classification_threshold=0.5)
```

- `predict(snapshot)` reuses the trained query and table-generation plan, and scores its population at `prediction_time`.
- `entities=` scores given rows instead; `table=` scores a prediction table generated beforehand.
- It warns when the snapshot was taken of a different graph than the one the model was trained on.
- `predict(prediction_table, snapshot=...)` keeps working as before.
- `fit` results score too: `result.predict(snapshot, ...)`.
- `PredictiveQuery.from_snapshot(snapshot, query)` binds a query to a snapshot without its `Graph`. Its graph check only means the snapshot belongs to the query's graph, not that it is compatible with a trained model.

## Progress events

`fit(callbacks=[...])` calls each callback with every progress
`kumo.engine.tracking.Event`: `fit_started`, then per search trial
`trial_started`, `epoch_ended` and `trial_ended`, then `fit_ended`. Ended events
carry `params["status"]` and, on failure, a short `params["error"]`. A callback
that raises is logged and not called again; it never fails the fit. For JSON
lines on stdout:

```python
import json

trainer.fit(training_table, snapshot=snapshot,
            callbacks=[lambda e: print(json.dumps(e.to_dict()))])
```

## Experiment tracking

`kumo.MLOps(tracking=...)` records each fit in an experiment tracker: one run
per fit, with each hyperparameter-search trial attempt as a child run (MLflow)
or inside the fit run (Snowflake). Tracking is on only
inside a `kumo.mlops(cfg)` block or for a `fit` given `mlops=cfg`.

```python
from kumo.engine.integrations.mlflow import MlflowTracking  # kumo-engine[mlflow]

cfg = kumo.MLOps(artifact_root="s3://bucket/kumo",
                 tracking=MlflowTracking(experiment_name="users-spend"))
with kumo.mlops(cfg):
    result = trainer.fit(training_table, snapshot=snapshot)
```

- Tracking errors are logged as warnings and never fail the fit. Only global rank 0 tracks.
- Tracker calls run on a background thread, so a slow or unreachable tracking server does not hold up training. At the end of the fit, `fit` waits at most 30 s for tracking to finish; the rest continues in the background.
- During the fit, MLflow HTTP calls use 2 retries and a 30 s timeout unless `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` are set.
- MLflow runs go to `MLFLOW_TRACKING_URI`: on Databricks this is automatic; for SageMaker managed MLflow set it to the tracking server ARN with the `sagemaker-mlflow` plugin installed. With nothing configured MLflow writes a local store in the working directory.
- Per-epoch metrics use step = epoch. After the fit, `tracking.run_id` is the fit run's MLflow run id.
- The fit's source tables are logged as MLflow run inputs, read from the training table's embedded snapshot manifest (plus a custom training table's source). Databricks tables are logged as Delta tables at the snapshot's Delta version, so Unity Catalog links each table to a model registered from the run; file and object-store sources by URI. Other warehouse tables and Custom SQL sources are not logged as inputs; they stay in the model's `_provenance.json`.

Snowflake records the fit in Snowflake Experiment Tracking (Snowsight: AI & ML
» Experiments), using a Snowpark session:

```python
from kumo.engine.integrations.snowflake import SnowflakeTracking  # kumo-engine[snowflake]

cfg = kumo.MLOps(artifact_root="s3://bucket/kumo",
                 tracking=SnowflakeTracking(session, experiment_name="users_spend"))
with kumo.mlops(cfg):
    result = trainer.fit(training_table, snapshot=snapshot)
```

- Snowflake has no child runs or tags, so trial attempts are logged in the fit run as `trial-1-attempt-1/<name>` params and metrics, and tags are params of the same name.
- `native_model_lineage=True` links a model the fit publishes through Snowflake lineage.
- Names beyond Snowflake's 200 metrics and 1000 params per run are counted in `kumo.dropped_metrics` / `kumo.dropped_params`.
- If a Snowflake run is already active in the process, the fit is not tracked there; an existing run is never resumed (a taken `run_name` gets a suffix).
- Snowflake Experiment Tracking has no dataset inputs, so source tables are not recorded there.
- The Snowpark session's parameters are left as they are. After the fit, `tracking.run_id` is the fit run's name.

Runs carry pointers, not lineage:

| tag | on | value |
|---|---|---|
| `kumo.kind` | every run | `fit`, `trial`, `prediction` or `monitoring` |
| `kumo.run_id` | every run | the kumo run id (trials: `<fit run id>-trial-<n>-attempt-<m>`; predictions: the prediction id) |
| `kumo.bundle_id` | a finished fit run, prediction and monitoring runs | the model's bundle id (`models/<bundle id>/` under the artifact root) |
| `kumo.prediction_run` | monitoring runs | the prediction id it monitors |
| `kumo.parent_run`, `kumo.status`, `kumo.error` | trial runs; `kumo.error` also on a failed fit or prediction | outcome |

The model's `_provenance.json` records the tracked fit as `tracking.run_id`
(the `kumo.run_id` tag) and is logged on the fit run with `query.pql` and
`model_plan.json`.

With tracking on, `predict` also records each prediction run as one run named
by its prediction id, after its `output_config` export and its artifact-root
copy: scoring options as params; row counts and score statistics (for example
`predictions.True_PROB.q50`) as metrics; the scoring snapshot's source tables as
inputs; and its `_provenance.json` as an artifact. If the export raises, the run
is still recorded, ended as failed with `kumo.error`, and `predict` raises the
export's error. It runs on a tracking thread for up to 30 s; tracking errors
never change the prediction. MLflow nests it under an active run, like a fit.

Other systems plug in through the `kumo.engine.tracking.Tracking` protocol,
which receives ready-built `RunRecord`s so every tracker records the same
names: `start(record)` with the fit run, `on_trial(record, event)` with a trial
attempt's run after each of its events, and `finish(record, error)` with the
fit run's final record (`None` when tracking stopped early). A tracker that
raises is not called again for that fit and its run is ended as failed.
Optional members: `prepare()`, `tags`, `log_model(result, snapshot, registry)` (see MLflow models below), and the best-effort hooks
`log_prediction_run(record)` and `log_monitoring(record)` (a record tagged
`kumo.error` is a failed run). Params are the resolved model plan
(fit) or searched hyperparameters and `attempt` (trial). `RunRecord.inputs`
maps each source table to its `connector`, `location`, `version` and
`version_kind`. A tracker's `tags=` are added as given, except `kumo.*` names
and param names.

### MLflow models

With `registry=ModelRegistryLocation(name, alias=...)` and an `MlflowTracking` tracker, every fit in the block logs its model on the fit run as an MLflow pyfunc model and registers it as a new version of `name` (Unity Catalog with `MLFLOW_REGISTRY_URI=databricks-uc`, SageMaker managed MLflow, or any MLflow registry). One alias is set on the new version. A bare `ModelRegistryLocation` without such a tracker, a fixed `version`, or an invalid alias raises when `kumo.MLOps` is built.

```python
from kumo.engine import ModelRegistryLocation
from kumo.engine.integrations.mlflow import MlflowTracking, load_model

cfg = kumo.MLOps("s3://bucket/kumo",
                 tracking=MlflowTracking(experiment_name="/Shared/users_spend"),
                 registry=ModelRegistryLocation("main.ml.users_spend", alias="champion"))
with kumo.mlops(cfg):
    result = trainer.fit(training_table, snapshot=snapshot)
result.registered_model   # PublishedModel(name='main.ml.users_spend', version='4', uri='models:/main.ml.users_spend/4')

trainer = load_model("models:/main.ml.users_spend@champion")   # the Trainer back
model = mlflow.pyfunc.load_model("models:/main.ml.users_spend@champion")
scores = model.predict(entities, params={"snapshot": "s3://bucket/kumo/snapshots/<graph key>/<version>"})
```

- `fit` waits for the upload and registration before it returns, for up to an hour, with a 600 s MLflow HTTP timeout during the upload (unless `MLFLOW_HTTP_REQUEST_TIMEOUT` is set). If they fail, the tracking thread stops or the hour passes, `fit` warns and still returns its result with `registered_model=None`, the run ends `FAILED`, and a model that finishes uploading later is not registered.
- `MlflowTracking(log_model=True)` logs the model without registering it. `log_model(trainer, snapshot=...)` logs a fitted or loaded model in any run, with the bundle's source tables as run inputs.
- The model's metadata is `kumo_bundle_id`; its `provenance` artifact is the bundle's `_provenance.json`. Its pip requirement is `kumo-engine[s3,gcs]` of the same version.
- The pyfunc model loads a snapshot on its first `predict` with that URI and keeps it for the life of the process: each serving replica, and each Spark executor of `mlflow.pyfunc.spark_udf`, downloads the snapshot once and keeps every snapshot it has loaded (no eviction), so its disk must hold them. With `serving_snapshot=<snapshot URI>` (on `log_model` or `MlflowTracking`), each replica downloads that snapshot when it starts, and `predict` uses it when no `snapshot` param is given. `params={"population": True}` scores every entity and returns the whole frame in memory; for large populations use `Trainer.predict` with a `PredictionOutputConfig`. Each `predict` call works in its own scratch directory and removes it before returning.
- The `mlflow` extra needs an MLflow 3.13 or later client: DiskGraph pins pyarrow 24, which older MLflow releases exclude. SageMaker managed MLflow servers run MLflow 3.0; a 3.17 client against a 3.0 server handles logging, registration, aliases and `load_model`.
- On Databricks, the parent folder of an MLflow experiment path must exist. Registering to Unity Catalog from outside Databricks also needs `boto3`, which the `mlflow` extra does not install.

#### Which registry

Pick one per registered model, since the two store different formats. Both record only the `kumo_bundle_id` pointer; lineage stays in the bundle's `_provenance.json`. Both accept the same aliases (`kumoapi.plugin.check_model_alias`).

| | Connector registry | MLflow model |
|---|---|---|
| Register | `kumo.MLOps(registry=ModelOutputConfig(connector, location=ModelRegistryLocation(name, alias=...)))` | `kumo.MLOps(tracking=MlflowTracking(), registry=ModelRegistryLocation(name, alias=...))` |
| Load | `Trainer.load_from_connector(connector, location=...)` | `mlflow.pyfunc.load_model(uri)` or `load_model(uri)` |
| Promote | `set_model_alias(connector, location, alias)` | MLflow's `set_registered_model_alias` |
| Databricks | `DatabricksConnector` (Unity Catalog) | Unity Catalog through MLflow |
| SageMaker | not available | SageMaker managed MLflow (`MLFLOW_TRACKING_URI` = tracking server ARN) |
| Snowflake | `SnowflakeConnector` (Snowflake Model Registry) | not available |
| Plain VM | not available (store under the artifact root instead) | any MLflow tracking server |

## Monitoring prediction runs

### Realized performance

`trainer.evaluate(result, snapshot=newer_snapshot)` regenerates the query's labels at the run's anchors on a newer snapshot, joins them with the run's predictions on `ENTITY` (and `ANCHOR_TIMESTAMP`), and returns an `EvaluationResult`:
- `metrics`, with the fit-time `test_*` definitions: binary `auroc`/`ap` (exact, scores compared at float32 precision); regression `mae`/`rmse`; multiclass `acc` of the top-scored `CLASS`; link prediction `map@k`/`precision@k`, with `k` the classes returned per entity and anchor;
- `rows`, `labelled` and `labelled_fraction` (for link prediction, the share of entities with any future link);
- `matured`: `False` while the label window ends after the latest event of the query's target tables; `None` for static queries, which read labels at the snapshot;
- `bundle_matches`, comparing the run's model with the trainer's;
- `actuals`, a Parquet directory of the evaluated rows next to the run: `ENTITY`, `ANCHOR_TIMESTAMP` (the run's prediction time for static queries), the prediction (`True_PROB` and the predicted label `TARGET_PRED`; `TARGET_PRED`; or the top `CLASS` and its `SCORE`) and the label `TARGET`.

`evaluation.export(PredictionOutputConfig(...))` copies the evaluated rows to a table or directory. The join runs in DiskGraph: both sides stream from Parquet and are hash-partitioned, so memory follows the partition size rather than the run size. `evaluate` never refuses; a failed step leaves `metrics` empty with a `reasons` entry.

### Score drift

`trainer.score_drift(result, previous=None, snapshot=None, training_snapshot=None)` returns a `DriftReport` comparing the run's scores with the model's holdout (`psi`, `ks`, `wasserstein`) and with a previous run (`previous_*`), plus per-table row and per-edge count changes between the training and scoring snapshots. The statistics are read off the recorded percentile curves (min, q01..q99, max; linear in between), with no pass over the data:

| statistic | definition | error from the curves |
|---|---|---|
| `psi` | population stability over the baseline's deciles | — |
| `ks` | largest gap between the two CDFs | ≤ 0.02 + 2× the sketch's rank error |
| `wasserstein` | mean absolute gap between the two quantile functions, in score units | ≤ 0.01 × (sum of both score ranges) + sketch error |

`wasserstein` is comparable over time for probabilities and scale-dependent for regression. Missing records leave a field `None` with a `reasons` entry.

### Tracking

With MLOps tracking on (`kumo.MLOps(tracking=...)`), each `evaluate` and `score_drift` call logs a new run in the tracker's experiment, tagged `kumo.kind=monitoring`, `kumo.prediction_run=<prediction id>` and `kumo.bundle_id`, with the metrics, the drift statistics and the input count changes. MLflow also tags the prediction run with `kumo.monitored=<monitoring run id>`.

### Monitoring in a later job

Export the run to a Parquet directory and load it back:

```python
from kumo.engine import PredictionResult, Trainer

trainer = Trainer.load_from_connector(store, path="models/<bundle id>")
run = PredictionResult.load_from_connector(store, path="predictions/2024-06-01/")
report = trainer.evaluate(run, snapshot=newer_snapshot)
drift = trainer.score_drift(run)
```

`load_from_connector` requires the export's `_SUCCESS` marker and fetches only its `_provenance.json`; `score_drift` needs nothing else. The Parquet parts are downloaded in parallel when `evaluate` (or `files`) first reads them, into a scratch directory removed with the result.

### Warehouse model monitors

Both recipes monitor the evaluated rows, exported to a warehouse table. Neither platform monitors link prediction or ranking outputs.

**Databricks (Data Quality Monitoring, inference log; databricks-sdk ≥ 0.68).** Export to a Unity Catalog Delta table (change data feed recommended), add the model id, and create the monitor:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import (
    AggregationGranularity, DataProfilingConfig, InferenceLogConfig,
    InferenceProblemType, Monitor, Refresh)

evaluation.export(PredictionOutputConfig(dbx, PredictionArtifactType.PREDICTIONS, "c.s.churn_evaluated"))
spark.sql("ALTER TABLE c.s.churn_evaluated ADD COLUMN model_id STRING")
spark.sql(f"UPDATE c.s.churn_evaluated SET model_id = '{evaluation.bundle_id}'")

w = WorkspaceClient()
table_id = w.tables.get("c.s.churn_evaluated").table_id
w.data_quality.create_monitor(monitor=Monitor(
    object_type="table", object_id=table_id,
    data_profiling_config=DataProfilingConfig(
        output_schema_id=w.schemas.get("c.s").schema_id,
        assets_dir="/Workspace/Users/<you>/monitoring",
        inference_log=InferenceLogConfig(
            problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
            prediction_column="TARGET_PRED", label_column="TARGET",
            model_id_column="model_id", timestamp_column="ANCHOR_TIMESTAMP",
            granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY]))))
w.data_quality.create_refresh(object_type="table", object_id=table_id, refresh=Refresh())
```

For regression use `INFERENCE_PROBLEM_TYPE_REGRESSION`; for multiclass, `prediction_column="CLASS"`. The inference log has no probability column, so `True_PROB` is profiled as an ordinary column.

**Snowflake (Model Monitor).** The model must be a Snowflake Model Registry version, so fit with `kumo.MLOps(registry=ModelOutputConfig(sf, location=...))`. The baseline is the holdout's evaluated rows, `training_result.holdout_df()` (written with `session.write_pandas`); the source is each run's evaluated rows:

```sql
CREATE MODEL MONITOR churn_monitor WITH
  MODEL=churn VERSION=v3 FUNCTION=predict
  SOURCE=churn_evaluated BASELINE=churn_holdout_evaluated
  WAREHOUSE=wh REFRESH_INTERVAL='1 day' AGGREGATION_WINDOW='1 day'
  TIMESTAMP_COLUMN=ANCHOR_TIMESTAMP ID_COLUMNS=(ENTITY)
  PREDICTION_SCORE_COLUMNS=(True_PROB) ACTUAL_CLASS_COLUMNS=(TARGET);
```

- Binary classification: `PREDICTION_SCORE_COLUMNS` (in [0, 1]) with `ACTUAL_CLASS_COLUMNS`; multiclass: `PREDICTION_CLASS_COLUMNS=(CLASS)` with `ACTUAL_CLASS_COLUMNS`; regression: `PREDICTION_SCORE_COLUMNS=(TARGET_PRED)` with `ACTUAL_SCORE_COLUMNS`.
- The timestamp must be `TIMESTAMP_NTZ`, the columns free of nulls, NaN and infinities, and each column used once. Without `BASELINE` there is no drift.
