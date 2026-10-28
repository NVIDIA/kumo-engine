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
- **Volumes and stages:** `(DatabricksConnector(...), "/Volumes/<catalog>/<schema>/<volume>/kumo")` and `(SnowflakeConnector(...), "@DB.SCHEMA.STAGE/kumo")` work as roots once the installed kumo-databricks or kumo-snowflake sink holds files at those paths; with an older plugin, building `kumo.MLOps` raises.
- **`export`** names the artifact kinds stored under the root when they are produced: `snapshot`, `training_table` and `model` (default: all).
- **`tracking`** and **`registry`** record fits and register models.
- **`on_error`** says what a tracking or registration failure in `fit` does. `"warn"` (default): the call warns, naming the error, and returns as usual. `"raise"`, for scheduled jobs: the call finishes and stores its result first, then raises a `RuntimeError` whose `__cause__` is the failure; after `fit`, `trainer.result` holds the fitted model.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has. Plain exports (`export(config)`, `export_config=`, `output_config=`) never write to the artifact root. A path export that finds a complete copy of the same artifact at its destination reuses it instead of uploading again (logged at info): a snapshot with the same bytes, a training table or prediction directory with `_SUCCESS` and the same `_KUMO_EXPORT` identity, or a model bundle with the same files and `_provenance.json`.

## Lineage

Every artifact carries a `_provenance.json` manifest, metadata only. The leading underscore keeps Spark and Hive Parquet readers from treating it as data.

A graph snapshot's manifest, next to its index, records each source table's connector, location and version, plus the engine build that wrote it. `graph.snapshot(record_sources=False)` keeps only connector and table names, plus a SHA-256 of each full location so a source moved elsewhere is still noticed, and keeps Custom SQL only as a hash.

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

An embedded model keeps only the ids of its own model parents, so a long chain of warm-start retrains does not grow each manifest. `fit` returns a new `TrainingResult` per call (`.id`, `.manifest`); `Trainer.result` is the latest fit's result or the loaded bundle's (`trainer.result.id`, `trainer.result.manifest`).

With MLOps on, each bundle is stored under `<root>/models/<bundle id>/`. A complete copy that is already there is kept; a partial one is replaced.

The bundle also carries the published model's holdout predictions (`holdout.parquet`, with its truncation flag), so `result.holdout_df()` works on a model loaded with `Trainer.load` or `Trainer.load_from_connector` (a path or a registry), not only on the result of `fit`. Bundles fitted before this have no copy, and `holdout_df()` raises `FileNotFoundError` for them.

### Artifacts, bundles and dependencies

- An **artifact** is anything with its own id and `_provenance.json`: a snapshot, a training table, a model, a prediction run, or an asset.
- A model's **bundle** holds only the model's own state: weights, encoders, the training spec and the holdout.
- Its manifest's **`parents`** are lineage: the training table and the warm-start model it came from.
- Its **`dependencies`** are what `predict` needs besides the bundle and a snapshot. The snapshot is an input to each call, never a dependency. Each entry has a `kind`:
  - `predictor`: the entry point that scores the bundle;
  - `package`: the `kumo-engine` version and its extras, with `[text]` only when the model encodes a text column;
  - `asset`: a content-addressed file set, by `asset_kind`, `name`, `version` and `sha256`.
- Bundles fitted before dependencies were recorded list none. They load as before, with the built-in predictor.
- **Assets** are not copied into bundles. With MLOps on, each asset is stored once under `<root>/assets/<asset_kind>/<sha256>/` with its own `_provenance.json`, and every model that uses it shares that copy. The dependency records where it was stored.
- **Resolving assets:** `Trainer.load`, the MLflow pyfunc and scoring jobs look for each asset at the location the dependency records, then under the artifact root of the enclosing `kumo.mlops` block, then in the embedding-model store. They check its sha256 and cache non-local copies under `~/.cache/kumo/assets/<sha256>/`. If an asset is stored but cannot be found, loading fails with `MissingDependencyError`, which names the asset kind, name and sha256. A bundle without stored assets resolves its models from the store, as before.
- **Text encoders** are the first assets (`asset_kind="text_encoder"`): the word vectors of each text column's embedding model. Word-vector models (`GloVeVectors`) are stored. Other `TokenEmbeddingModel`s are recorded by fingerprint and must be registered wherever the model is loaded.

To train with your own word vectors and serve the model without registering them again, register them under a name, fit with MLOps on, and load the stored model anywhere that can reach the artifact root:

```python
store = kumo.EmbeddingModelStore()
store.add("colors", kumo.GloVeVectors("/data/vectors"))   # a .zip release or a directory of converted files
plan.column_processing.encoder_overrides = {"users.bio": TextEmbedding("colors")}
with kumo.mlops(kumo.MLOps(artifact_root="s3://bucket/kumo")):
    result = kumo.Trainer(plan, embedding_models=store).fit(table, snapshot=snapshot)
# Elsewhere, with no store: the vectors come from <root>/assets/text_encoder/<sha256>/.
trainer = kumo.Trainer.load_from_connector(kumo.S3Connector(root_dir="s3://bucket/kumo"),
                                           path=f"models/{result.id}")
```

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

# Scoring jobs load whichever version an alias points at.
trainer = kumo.Trainer.load_from_connector(registry, location=ModelRegistryLocation(
    "CHURN_GNN", alias="challenger"))
```

- Registration is independent of `export`. If it fails, `fit` warns with the error and returns the result with `registered_model=None` (with `on_error="raise"`, it raises instead), and a tracked fit run ends failed with `kumo.error`; an invalid alias or a connector that cannot publish models raises before training, and an invalid alias or model name raises when `kumo.MLOps` is built. Snowflake model names are Snowflake identifiers (a letter or underscore, then letters, digits, underscores or `$`, e.g. `USERS_CHURN`; or a double-quoted name); Unity Catalog model names are at most 255 characters with no space, `/`, `.` or control character, optionally qualified with the connector's own catalog and schema.
- `result.export(ModelOutputConfig(connector, location=...))` on a `TrainingResult`, or `trainer.export(...)` for its latest result, publishes the model with or without a block and also accepts an `alias`; `Trainer.load_from_connector` loads it back. `fit(export_config=...)` is shorthand for fitting then exporting, and its version is `result.published_model`.
- An alias is a letter or underscore followed by letters, digits or underscores; `DEFAULT`, `FIRST`, `LAST`, `LATEST` and `v<digits>` are reserved. The alias is set on the new version after publishing, moving it from any version that held it; if that fails, the publish still stands, with a warning. Snowflake allows one alias per version; if moving an alias fails, it is put back on the version that held it. Snowflake's `DEFAULT`, `FIRST` and `LAST` can be loaded but not set. File, S3 and GCS connectors have no aliases.
- Each registered version records only pointers: the `kumo_engine` marker and `kumo_bundle_id`, as version metrics in the Snowflake Model Registry and as version tags in Unity Catalog through a `DatabricksConnector`. (A version registered through MLflow, below, gets a `kumo_bundle_id` version tag only.) The bundle's own `_provenance.json` holds its lineage. `list_models` and `fetch_model` return `kumo_bundle_id` in `PublishedModel.metadata`; `load_from_connector` warns if the fetched bundle differs from it. Through a `DatabricksConnector`, publishing a bundle that a ready version of the model already holds (same `kumo_bundle_id`) returns that version, logged at info, instead of adding another; an alias then moves to it.

### Promote a Kumo model

To register every fit as a candidate and promote only a better one, register under a candidate alias (or none) and move the serving alias yourself with the registry's own API. Compare the new model's test metrics (`result.metrics`, also on its tracked fit run) with the current champion's (`load_from_connector(...).result.metrics`):

```python
champion = kumo.Trainer.load_from_connector(
    registry, location=ModelRegistryLocation("CHURN_GNN", alias="champion"))
better = result.metrics["test_auroc"] > champion.result.metrics["test_auroc"]
version = result.registered_model.version
```

Unity Catalog (registered through MLflow or a `DatabricksConnector`) and SageMaker managed MLflow:

```python
from mlflow import MlflowClient

if better:
    MlflowClient().set_registered_model_alias("main.ml.churn_gnn", "champion", version)
```

`set_registered_model_alias` moves the alias from the version that held it. For models registered through MLflow, the champion's metrics are also on its run: `client.get_run(client.get_model_version_by_alias(name, "champion").run_id).data.metrics`.

Snowflake Model Registry, where a version holds at most one alias, so the alias is unset first:

```sql
ALTER MODEL DB.SCHEMA.CHURN_GNN VERSION champion UNSET ALIAS;
ALTER MODEL DB.SCHEMA.CHURN_GNN VERSION V3 SET ALIAS = champion;
```

Scoring jobs that load `alias="champion"` (or `models:/<name>@champion`) pick up the new version on their next run.

### Moving snapshots between machines

To build an index on one machine and train or score on another, export it to a file destination and load it there:

```python
from kumo.engine import GraphSnapshot, S3Connector, SnapshotOutputConfig

location = graph.snapshot().export(SnapshotOutputConfig(S3Connector(), "s3://bucket/snapshots"))
# On the training machine:
snapshot = GraphSnapshot.load_from_connector(S3Connector(), path=location)
```

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, the two halves of `snapshot.id`, and returns that location. It holds `kumo_snapshot.json` and one directory per export generation (`index/` and `_provenance.json`).
- **Transfer:** files move in parallel with a CRC32 per range. Each export uploads into a new generation and commits by writing `kumo_snapshot.json` last, naming that generation. Until then the previous export stays loadable, and concurrent exports never mix files: the last commit wins whole. Superseded generations, and abandoned ones over a day old, are removed best-effort. A load either completes or leaves nothing; if a newer export commits while it downloads, it switches to that one.
- **Reuse:** loading again reuses the local copy only if it holds this export's bytes (the same range checksums) and passes DiskGraph's integrity check; otherwise it is downloaded again. Exporting again skips the upload when the destination already holds a complete export (its `kumo_snapshot.json` and `_DONE`) with the local index's bytes, and logs that it reused it.
- **Staleness:** `snapshot.source_changes(graph)` reports, per table of `graph`, whether its source is `unchanged`, `changed` or `unknown` since the snapshot was built. `load_from_connector(..., graph=graph)` warns when the snapshot was taken of another graph, naming the schema changes (tables, columns, types, keys, edges), or when any of its tables changed. A graph id also covers each table's connector and location, so a graph that differs only there also warns, saying the schema is the same.
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
- `entities=` scores given rows instead.
- It warns when the snapshot was taken of a different graph than the one the model was trained on, naming the schema changes, for example `Schema changes: events edges.` A graph that differs only in table connectors or locations (a dev graph and a prod graph of the same schema) warns that the schema is the same. The model records its training graph from this release on; for an older model the warning cannot say what differs.
- `predict(prediction_table, snapshot=...)` scores a prediction table generated beforehand, as before.
- `fit` results score too: `result.predict(snapshot, ...)`.

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

- A tracking or registration failure does not fail the fit: it is logged, and reported once as a `UserWarning` naming the full error, so both notebooks and job logs show it. With `on_error="raise"` the call raises once instead, after storing its result, with the error as the cause. Only global rank 0 tracks.
- Tracker calls run on a background thread, so a slow or unreachable tracking server does not hold up training. At the end of the fit, `fit` waits at most 30 s for tracking to finish; the rest continues in the background.
- During the fit, MLflow HTTP calls use 2 retries and a 30 s timeout unless `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` are set.
- `MlflowTracking()` records where MLflow's own configuration points when it is built (`MLFLOW_TRACKING_URI`, or `mlflow.set_tracking_uri`): on Databricks this is automatic, and with nothing configured MLflow writes a local store in the working directory. `MlflowTracking(target=...)` names the server and the credentials instead; see [Running from a laptop](#running-from-a-laptop).
- `kumo.MLOps(...)` checks the tracker when it is built: one experiment lookup, plus one registry listing and (Unity Catalog) the model's catalog and schema when it registers. An unreachable server, refused credentials or a missing catalog or schema raise there, before any work.
- The `View run ... at` links MLflow prints do not open on SageMaker; after a tracked fit kumo-engine warns how to open the UI: from SageMaker Studio, or at the URL `aws sagemaker create-presigned-mlflow-tracking-server-url --tracking-server-name <name>` prints.
- MLflow rejects `@` in metric names, so link-prediction metrics such as `test_map@10` are logged as `test_map_at_10`.
- Per-epoch metrics use step = epoch. After the fit, `tracking.run_id` is the fit run's MLflow run id.
- The fit run is named `fit-<UTC start time>` (for example `fit-20261027T061838Z`) unless `run_name` is given; trial runs are `trial-<n>-attempt-<m>`.
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
| `kumo.run_id` | every run | the kumo run id (trials: `<fit run id>-trial-<n>-attempt-<m>`) |
| `kumo.bundle_id` | a finished fit run | the model's bundle id (`models/<bundle id>/` under the artifact root) |
| `kumo.parent_run`, `kumo.status`, `kumo.error` | trial runs; `kumo.error` also on a failed fit (including a failed model logging or registration) | outcome |

The model's `_provenance.json` records the tracked fit as `tracking.run_id`
(the `kumo.run_id` tag) and is logged on the fit run with `query.pql` and
`model_plan.json`.

Which id is which:

| id | what it names | where to read it |
|---|---|---|
| kumo run id | one `fit` call | the `kumo.run_id` tag; `tracking.run_id` in the model's `_provenance.json` |
| bundle id | the fitted model's content | `result.id`; the `kumo.bundle_id` tag; `kumo_bundle_id` in the registry; `models/<bundle id>/` under the artifact root |
| tracker run id | the run in the tracker | the tracker's `run_id` attribute (`tracking.run_id` on the `MlflowTracking` object: the MLflow run id; Snowflake: the run name) |
| registry version | the registered model version | `result.registered_model.version` |

To find a bundle's MLflow fit run, search for its tag: ``mlflow.search_runs(filter_string="tags.`kumo.bundle_id` = '<bundle id>' AND tags.`kumo.kind` = 'fit'", search_all_experiments=True)``.

Other systems plug in through the `kumo.engine.tracking.Tracking` protocol,
which receives ready-built `RunRecord`s so every tracker records the same
names: `start(record)` with the fit run, `on_trial(record, event)` with a trial
attempt's run after each of its events, and `finish(record, error)` with the
fit run's final record (`None` when tracking stopped early). A tracker that
raises is not called again for that fit and its run is ended as failed.
Optional members: `prepare()`, `tags`, `check(registry)` (called when `kumo.MLOps` is built) and `log_model(result, snapshot, registry)` (see MLflow models below). Params are the resolved model plan
(fit) or searched hyperparameters and `attempt` (trial). `RunRecord.inputs`
maps each source table to its `connector`, `location`, `version` and
`version_kind`. A tracker's `tags=` are added as given, except `kumo.*` names
and param names.

### MLflow models

With `registry=ModelRegistryLocation(name, alias=...)` and an `MlflowTracking` tracker, every fit in the block logs its model on the fit run as an MLflow pyfunc model and registers it as a new version of `name` (Unity Catalog with `MLFLOW_REGISTRY_URI=databricks-uc`, SageMaker managed MLflow, or any MLflow registry). One alias is set on the new version. A bundle the registry already holds under `name` (same `kumo_bundle_id` version tag) is not registered again: its version is returned, the alias moves to it, and an INFO log says it was reused. A bare `ModelRegistryLocation` without such a tracker, a fixed `version`, an invalid alias, or a name the registry (`mlflow.get_registry_uri()`) rejects raises when `kumo.MLOps` is built:

- SageMaker managed MLflow (a tracking server or MLflow app ARN): letters, digits and hyphens, starting and ending with a letter or digit, at most 57 letters and digits (`^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,56}$`), so no `_` or `.`: `users-churn`.
- Unity Catalog (`databricks-uc`): `catalog.schema.model`, each part at most 255 characters with no space, `/` or control character: `main.ml.users_spend`. On a cluster a shorter name takes the current catalog and schema.
- Other MLflow registries: no extra rule.

The rules conflict, so no one name fits every registry: SageMaker forbids the `.` Unity Catalog needs between its parts and the `_` Snowflake needs instead of `-`. The examples use `users-churn` for SageMaker and other MLflow registries, `<catalog>.<schema>.users_churn` for Unity Catalog and `USERS_CHURN` for Snowflake.

```python
import mlflow
import pandas as pd
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
scores = model.predict(pd.DataFrame({"ENTITY": [1, 2, 3]}), params={
    "snapshot": "s3://bucket/kumo/snapshots/<graph key>/<version>",
    "prediction_time": "2024-06-01",              # temporal queries
    "binary_classification_threshold": 0.5,       # binary classification only
})
```

- `fit` waits for the upload and registration before it returns, for up to an hour, with a 600 s MLflow HTTP timeout during the upload (unless `MLFLOW_HTTP_REQUEST_TIMEOUT` is set). If they fail, the tracking thread stops or the hour passes, `fit` warns once (`Could not log the model: ...` or `Could not register the model: ...`, with the full error) and still returns its result with `registered_model=None` (with `on_error="raise"`, it raises instead), the run ends `FAILED` with the error in `kumo.error`, and a model that finishes uploading later is not registered.
- `MlflowTracking(log_model=True)` logs the model without registering it. `log_model(trainer, snapshot=...)` logs a fitted or loaded model in any run, with the bundle's source tables as run inputs.
- The model's metadata, the registered version's tags and `result.registered_model.metadata` hold `kumo_bundle_id`. The model's `provenance` artifact is the bundle's `_provenance.json`. Its pip requirement is `kumo-engine[s3,gcs]` of the same version, plus `[text]` when the model encodes a text column. The pyfunc scores with the predictor the bundle's manifest names. It resolves the model's assets, such as non-default word vectors, as described in [Artifacts, bundles and dependencies](#artifacts-bundles-and-dependencies).
- The pyfunc model loads a snapshot on its first `predict` with that URI and keeps it for the life of the process: each serving replica, and each Spark executor of `mlflow.pyfunc.spark_udf`, downloads the snapshot once and keeps every snapshot it has loaded (no eviction), so its disk must hold them. With `serving_snapshot=<snapshot URI>` (on `log_model` or `MlflowTracking`), each replica downloads that snapshot when it starts, and `predict` uses it when no `snapshot` param is given. `params={"population": True}` scores every entity and returns the whole frame in memory; for large populations use `Trainer.predict` with a `PredictionOutputConfig`. Each `predict` call works in its own scratch directory and removes it before returning.
- Input rows have an `ENTITY` column and, optionally, a `TIMESTAMP` column. Scores come back in the input's row order, one per input row (repeated rows included); rows whose entity is not in the snapshot are left out. A temporal query scores entities at a time: pass `params={"prediction_time": ...}` or a `TIMESTAMP` column; without either, `predict` fails with `temporal query: pass params={'prediction_time': ...} or a TIMESTAMP column`. Binary classification needs `params={"binary_classification_threshold": ...}` unless the model's `scoring_defaults` set one.
- Serving needs an image with kumo-engine already installed; MLflow's stock serving containers cannot install it. Serve with `mlflow models serve -m <model dir> --env-manager local` (on SageMaker endpoints, `-m /opt/ml/model -p 8080`). SageMaker batch transform takes JSON input (`dataframe_split` plus `params`), not CSV, which carries no params.
- GPU real-time endpoints on SageMaker need an inference image whose NVIDIA driver supports CUDA 12.6. The default GPU hosting image ships driver 470, so set a newer `InferenceAmiVersion` on the production variant, for example `al2023-ami-sagemaker-inference-gpu-4-1`. Not yet tested live.
- The `mlflow` extra installs `mlflow-skinny` (3.0.1 or later), the client without the MLflow server's dependencies. SageMaker managed MLflow servers run MLflow 3.0; a 3.17 client against a 3.0 server handles logging, registration, aliases and `load_model`.
- On Databricks, use Databricks Runtime 17.3 LTS ML or later: its torch already satisfies kumo-core, so installing kumo-engine keeps the runtime's torch, torchvision and MLflow. kumo-engine accepts pyarrow 16 through 24, so a runtime's pyarrow in that range (19 on 17.3 LTS ML) is kept too; a custom CA bundle for S3 needs pyarrow 21 or later. On 16.4 LTS ML pip replaces torch 2.6 and breaks the preinstalled torchvision.
- On Databricks, the parent folder of an MLflow experiment path must exist. Registering to Unity Catalog from outside Databricks also needs `boto3`, which the `mlflow` extra does not install.

#### Which registry

Pick one per registered model, since the two store different formats. On Databricks both work; use the MLflow model unless only kumo-engine jobs load it, since it also serves through `mlflow.pyfunc` (Model Serving, `spark_udf`). Both record only the `kumo_bundle_id` pointer; lineage stays in the bundle's `_provenance.json`. Both accept the same aliases (`kumoapi.plugin.check_model_alias`).

| | Connector registry | MLflow model |
|---|---|---|
| Register | `kumo.MLOps(registry=ModelOutputConfig(connector, location=ModelRegistryLocation(name, alias=...)))` | `kumo.MLOps(tracking=MlflowTracking(), registry=ModelRegistryLocation(name, alias=...))` |
| Load | `Trainer.load_from_connector(connector, location=...)` | `mlflow.pyfunc.load_model(uri)` or `load_model(uri)` |
| Promote | the registry's own API (see [Promote a Kumo model](#promote-a-kumo-model)) | MLflow's `set_registered_model_alias` |
| Databricks | `DatabricksConnector` (Unity Catalog) | Unity Catalog through MLflow |
| SageMaker | not available | SageMaker managed MLflow (`MlflowTracking(target=SageMakerMLflow(arn), ...)`) |
| Snowflake | `SnowflakeConnector` (Snowflake Model Registry) | not available |
| Plain VM | not available (store under the artifact root instead) | any MLflow tracking server |

#### SageMaker managed MLflow

1. Create a tracking server once and reuse it; creation takes about 15-25 minutes:
   `aws sagemaker create-mlflow-tracking-server --tracking-server-name <name> --artifact-store-uri s3://<bucket>/mlflow --role-arn <server role ARN>`
   (the server role needs read and write on that S3 prefix).
2. Give the identity that runs the job (notebook, VM role or pipeline execution role) `sagemaker-mlflow:*` on the tracking server ARN, and S3 read and write on the artifact store prefix and on the kumo artifact root.
3. Install `pip install "kumo-engine[mlflow,sagemaker]"` (the MLflow client, S3 and the `sagemaker-mlflow` plugin) and pass `MlflowTracking(target=SageMakerMLflow("arn:aws:sagemaker:<region>:<account>:mlflow-tracking-server/<name>"), experiment_name=...)`; the registry is the same server. Without an identity, the job's AWS identity (a role or `aws configure`) signs; `MLFLOW_TRACKING_URI` need not be set. The plugin signs and routes with the process-wide tracking URI, so each MLflow call kumo-engine makes sets it to the ARN and restores it afterwards: other threads that call MLflow during a fit see the SageMaker URI.
4. Model names use letters, digits and hyphens only (`users-churn`); `kumo.MLOps` checks this when it is built.
5. The run links MLflow prints do not open on SageMaker. Open the UI from SageMaker Studio or at the URL `aws sagemaker create-presigned-mlflow-tracking-server-url --tracking-server-name <name>` prints; a tracked fit warns with this command.

On S3 the artifact root, source tables and prediction outputs are plain `s3://` locations read and written with the job's AWS identity (instance profile, `AWS_PROFILE` or the job role):

```python
from kumo.engine import PredictionOutputConfig, S3Connector

data = S3Connector(root_dir="s3://<bucket>/data")              # data["orders.parquet"] reads s3://<bucket>/data/orders.parquet; data["orders"] a directory of files
cfg = kumo.MLOps("s3://<bucket>/kumo",
                 tracking=MlflowTracking(target=SageMakerMLflow("<tracking server ARN>"), experiment_name="users-churn"),
                 registry=ModelRegistryLocation("users-churn", alias="candidate"))
output = PredictionOutputConfig(S3Connector(), output_paths={"predictions": "s3://<bucket>/scores/"})
store = S3Connector()                                           # monitoring: load with absolute s3:// paths
```

### Running from a laptop

`MlflowTracking(target=...)` and `SnowflakeTracking(connector=...)` take the
same identities as the connectors, so a fit on a laptop or any VM can record
to Databricks, SageMaker or Snowflake. An identity with no credentials uses
the platform's ambient chain (a notebook, job or instance role, `aws
configure`, `databricks auth login`); one with credentials uses exactly
those.

```python
import kumo.engine as kumo
from kumo.engine import ModelRegistryLocation
from kumo.engine.connector import DatabricksConnector, DatabricksIdentity, S3Identity, SnowflakeConnector, SnowflakeIdentity
from kumo.engine.integrations.mlflow import MlflowTracking, SageMakerMLflow
from kumo.engine.integrations.snowflake import SnowflakeTracking

# Databricks MLflow and Unity Catalog, as a service principal (kumo-engine[mlflow,databricks]).
workspace = DatabricksConnector(catalog="main", schema_name="ml", warehouse_id="<warehouse id>",
                                identity=DatabricksIdentity(host="https://<workspace>", client_id="<id>", client_secret="<secret>"))
cfg = kumo.MLOps("s3://<bucket>/kumo",
                 tracking=MlflowTracking(target=workspace, experiment_name="/Users/<you>/users_churn"),
                 registry=ModelRegistryLocation("main.ml.users_churn", alias="candidate"))

# SageMaker managed MLflow, with an access key or a role to assume (kumo-engine[mlflow,sagemaker]).
cfg = kumo.MLOps("s3://<bucket>/kumo",
                 tracking=MlflowTracking(target=SageMakerMLflow("<tracking server ARN>", identity=S3Identity(role_arn="<role ARN>")),
                                         experiment_name="users-churn"),
                 registry=ModelRegistryLocation("users-churn", alias="candidate"))

# Snowflake Experiment Tracking, as a key-pair user (kumo-engine[snowflake]).
snowflake = SnowflakeConnector(warehouse="<warehouse>", database="ML", schema_name="RUNS",
                               identity=SnowflakeIdentity(account="<account>", username="<user>", private_key="<PEM>"))
cfg = kumo.MLOps("s3://<bucket>/kumo", tracking=SnowflakeTracking(connector=snowflake, experiment_name="USERS_CHURN"))
```

- Each target is checked when `kumo.MLOps` is built, so wrong credentials or a missing catalog, schema or experiment folder parent fail before the fit starts.
- MLflow has no per-client credentials: its Databricks store reads `DATABRICKS_*` variables or a profile, and the SageMaker plugin signs with the process-wide tracking URI. So with explicit credentials, each MLflow call kumo-engine makes sets `DATABRICKS_HOST`, a short-lived `DATABRICKS_TOKEN` (never the client secret) and the process-wide tracking URI `databricks` (Unity Catalog registration downloads the logged model through it), or the process-wide tracking URI and the AWS session, and restores them when the call returns. With explicit credentials, don't call MLflow or Databricks yourself on other threads during a fit. Ambient identities change no credentials; a SageMaker target still sets the process-wide tracking URI for each call, so with a SageMaker target, other threads calling MLflow during a fit see the SageMaker URI.
- SageMaker model artifacts are uploaded and downloaded with the identity's credentials. `load_model(uri, tracking=MlflowTracking(target=..., experiment_name=...))` and `log_model(..., tracking=...)` take the same target.
- `examples/mlops/smoke.py databricks|sagemaker|snowflake` is a manual live check per platform: a small fit recorded and registered with an explicit identity, then loaded back, in a few minutes on a CPU. It is not run in CI.
