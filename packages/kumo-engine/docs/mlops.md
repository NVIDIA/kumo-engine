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
| `kumo.kind` | every run | `fit` or `trial` |
| `kumo.run_id` | every run | the kumo fit run id (trials: `<fit run id>-trial-<n>-attempt-<m>`) |
| `kumo.bundle_id` | a finished fit run | the model's bundle id (`models/<bundle id>/` under the artifact root) |
| `kumo.parent_run`, `kumo.status`, `kumo.error` | trial runs; `kumo.error` also on a failed fit | outcome |

The model's `_provenance.json` records the tracked fit as `tracking.run_id`
(the `kumo.run_id` tag) and is logged on the fit run with `query.pql` and
`model_plan.json`.

Other systems plug in through the `kumo.engine.tracking.Tracking` protocol,
which receives ready-built `RunRecord`s so every tracker records the same
names: `start(record)` with the fit run, `on_trial(record, event)` with a trial
attempt's run after each of its events, and `finish(record, error)` with the
fit run's final record (`None` when tracking stopped early). A tracker that
raises is not called again for that fit and its run is ended as failed.
Optional members: `prepare()`, `tags`, and the best-effort hooks `log_model`,
`log_prediction_run` and `log_monitoring`. Params are the resolved model plan
(fit) or searched hyperparameters and `attempt` (trial). `RunRecord.inputs`
maps each source table to its `connector`, `location`, `version` and
`version_kind`. A tracker's `tags=` are added as given, except `kumo.*` names
and param names.
