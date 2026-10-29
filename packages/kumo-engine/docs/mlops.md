# MLOps with Kumo Engine

Turn MLOps on with one `kumo.MLOps` and the normal `kumo.engine` calls store their snapshots, training tables and models under an artifact root, record fits in your experiment tracker, and register models in its registry.

Install with `pip install "kumo-engine[mlflow]"` (add `s3`, `gcs`, `databricks`, `sagemaker` or `snowflake` as needed; Snowflake needs only `kumo-engine[snowflake]`). GPU, CUDA and Databricks Runtime notes are under [Install](#install).

## Quickstart

Each block assumes a `graph` and a `pquery` (see the README). It trains, stores and registers a model, then scores with it.

### Laptop or VM

```python
import kumo.engine as kumo
from kumo.engine.integrations.mlflow import MlflowTracking   # MLflow's own tracking URI, else ./mlruns

cfg = kumo.MLOps("artifacts", tracking=MlflowTracking("churn"),
                 registry=kumo.ModelRegistryLocation("users-churn", alias="candidate"))
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=0.5))
    scores = result.predict(snapshot).frame
    print(result.registered_model.version, result.tracking.run_id, len(scores))
```

### Databricks

In a notebook or job on Databricks Runtime 17.3 LTS ML or later, after `%pip install "<wheel dir>/<kumo_engine wheel>[databricks,mlflow,text]"` and `dbutils.library.restartPython()`:

```python
import kumo.engine as kumo
from kumo.engine.integrations.mlflow import MlflowTracking

cfg = kumo.MLOps("/Volumes/main/ml/kumo", tracking=MlflowTracking("/Users/<you>/churn"),
                 registry=kumo.ModelRegistryLocation("main.ml.users_churn", alias="candidate"))
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=0.5))
    print(result.registered_model.version, result.tracking.url)
```

### SageMaker

With the managed MLflow tracking server's ARN (setup: [SageMaker managed MLflow](#sagemaker-managed-mlflow)):

```python
import kumo.engine as kumo
from kumo.engine.integrations.mlflow import MlflowTracking, SageMakerMLflow

cfg = kumo.MLOps("s3://<bucket>/kumo", tracking=MlflowTracking("users-churn", target=SageMakerMLflow("<tracking server ARN>")),
                 registry=kumo.ModelRegistryLocation("users-churn", alias="candidate"))
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=0.5))
    print(result.registered_model.version, result.tracking.url)
```

Tables on S3 are read through an `S3Connector`:

```python
s3 = kumo.S3Connector(root_dir="s3://<bucket>/data")
users = kumo.Table.from_source_table(s3["users.parquet"], primary_key="USER_ID").infer_metadata()
orders = kumo.Table.from_source_table(s3["orders.parquet"], primary_key="ORDER_ID", time_column="ORDER_TIME").infer_metadata()
graph = kumo.Graph(tables={"users": users, "orders": orders.with_column("USER_ID", stype="ID")},
                   edges=[kumo.Edge(src_table="orders", fkey="USER_ID", dst_table="users")])
```

### Snowflake

Tracking and the Snowflake Model Registry through one connector (`kumo-engine[snowflake,s3]`):

```python
import kumo.engine as kumo
from kumo.engine.integrations.snowflake import SnowflakeTracking

sf = kumo.SnowflakeConnector(warehouse="WH", database="ML", schema_name="MODELS")
cfg = kumo.MLOps("s3://<bucket>/kumo", tracking=SnowflakeTracking(sf, experiment_name="CHURN"),
                 registry=kumo.ModelRegistryLocation("CHURN_GNN", alias="candidate"))
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=0.5))
    print(result.registered_model.version, result.tracking.url)
```

To serve only a better model, move its alias yourself: [Promote a model](#promote-a-model).

## How it fits together

- The **artifact root** holds every artifact a block produces, each in its own directory with a `_provenance.json` manifest that embeds its parents' manifests.
- The **tracker** records each fit as a run of its own; the **registry** is the tracker's: MLflow (Unity Catalog, SageMaker MLflow, any MLflow server) or the Snowflake Model Registry.

| id | names | where to find it |
|---|---|---|
| `snapshot.id` | a graph snapshot | `<root>/snapshots/<graph id>/<data version>/`; `result.snapshot_id` |
| `table.id` | a training table | `<root>/training_tables/<id>/` |
| `result.id` | a model (its bundle id) | `<root>/models/<id>/`; the `kumo.bundle_id` run tag; `kumo_bundle_id` on the registered version |
| `result.registered_model.version` | the registered version | the registry |
| `result.tracking` | the tracker's own run of the fit: `platform`, `run_id`, `url` | the tracker's UI |

Find runs with the platform's UI, or by tag: `kumo.kind` (`fit`, `trial`) and `kumo.bundle_id`.

## Turning MLOps on

`kumo.MLOps` says where artifacts go and what records them. Use it as a block, or pass it as `mlops=` on a single call (`fit`, `graph.snapshot`, `generate_training_table`):

```python
cfg = kumo.MLOps("s3://bucket/kumo", tracking=..., registry=..., on_error="raise")
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
trainer.fit(table, snapshot=snapshot, mlops=cfg)
```

- **`artifact_root`** (required): a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach: a volume through a `DatabricksConnector` (`"/Volumes/<catalog>/<schema>/<volume>/kumo"`) or a named stage through a `SnowflakeConnector` (`"@DB.SCHEMA.STAGE/kumo"`) once the installed sink holds files there.
- **`tracking`**: `MlflowTracking` or `SnowflakeTracking` (below), or your own [`Tracking`](#custom-trackers).
- **`registry`**: a `ModelRegistryLocation(name, alias=...)`; every fit registers its model there, in the tracker's registry.
- **`store`**: the artifact kinds stored when produced: `snapshot`, `training_table` and `model` (default: all).
- **`on_error`**: `"warn"` (default) warns about a failed store, tracking call or registration and returns; `"raise"`, for scheduled jobs, raises once the call has finished, with the failure as the `__cause__`. After `fit`, `trainer.result` still holds the model. A model whose bundle or assets could not be stored under the root is never registered.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has; plain exports (`export(config)`, `export_config=`) never write to the artifact root.
- `kumo.MLOps(...)` checks the tracker and registry when it is built (one experiment lookup, one registry listing, the model name's rules), so wrong credentials or a bad name fail before any work.

### Reading artifacts back

Pass the `kumo.MLOps` where a connector goes, with paths relative to the artifact root:

```python
snapshot = kumo.GraphSnapshot.load_from_connector(cfg, id=result.snapshot_id)
trainer = kumo.Trainer.load_from_connector(cfg, path=f"models/{result.id}")
champion = kumo.Trainer.load_from_connector(cfg, location=kumo.ModelRegistryLocation("users-churn", alias="champion"))
table = kumo.TrainingTable.load_from_connector(cfg, path=f"training_tables/{table_id}", pq=pquery, snapshot=snapshot)
```

`location=` loads the version an alias (or `version=`) names from the tracker's registry, one way on every platform. A loaded model is a `Trainer`; its `result` is the `TrainingResult` (`.id`, `.manifest`, `.metrics`, `.scoring_defaults`, `.holdout_df()`).

## Train and register

`fit` inside the block stores the bundle under `models/<id>/`, records the fit as one run (a child run per hyperparameter-search trial attempt on MLflow, `trial-<n>-attempt-<m>/` names inside the run on Snowflake), and registers the model as a new version of the registry location, moving its alias to it.

- `result.registered_model` is the version `MLOps(registry=)` registered; `result.published_model` the one `fit(export_config=...)` published.
- `result.tracking` is the tracker's run: `TrackingRef(platform="databricks", run_id="...", url="https://.../ml/experiments/.../runs/...")`. The model's `_provenance.json` records the same.
- `fit(scoring_defaults=kumo.ScoringDefaults(...))` stores prediction defaults in the bundle before it is stored and registered: `binary_classification_threshold` (binary and multilabel classification) and `num_classes_to_return` (multiclass, multilabel, link prediction). `predict` and the MLflow pyfunc apply them when a call omits them; explicit arguments and a query's `TOP k` win. `result.with_scoring_defaults(...)` retunes a fitted model as a new bundle whose manifest names the original in `parents.derived_from`.
- `result.export(cfg)` (or `trainer.export(cfg)` for a loaded model) registers a model at `cfg.registry` through its tracker. On every platform, registering a bundle the registry already holds under that name returns its version instead of adding one.
- If registration fails, `fit` warns with the full error and returns with `registered_model=None` (with `on_error="raise"`, it raises), and the fit run ends failed with `kumo.error`. Registering a bundle the registry already holds under the name returns that version (logged at info) and moves the alias to it.

**Model names** follow the registry's rules, checked when `kumo.MLOps` is built:

| registry | rule | example |
|---|---|---|
| Unity Catalog | `catalog.schema.model`, each part ≤ 255 characters without space, `/` or control characters | `main.ml.users_churn` |
| SageMaker MLflow | letters, digits and hyphens, ≤ 57, starting and ending with a letter or digit | `users-churn` |
| Snowflake Model Registry | a Snowflake identifier | `USERS_CHURN` |

An alias is a letter or underscore followed by letters, digits or underscores; `DEFAULT`, `FIRST`, `LAST`, `LATEST` and `v<digits>` are reserved.

### Promote a model

Register every fit under a candidate alias and move the serving alias yourself once the new model is better, comparing `result.metrics` with the champion's:

```python
with cfg.tracking.client() as client:          # the tracker's MLflow server, as its identity
    first = "champion" not in client.get_registered_model(name).aliases
    if first or result.metrics["test_auroc"] > kumo.Trainer.load_from_connector(
            cfg, location=kumo.ModelRegistryLocation(name, alias="champion")).result.metrics["test_auroc"]:
        client.set_registered_model_alias(name, "champion", result.registered_model.version)
```

In the Snowflake Model Registry a version holds at most one alias: take `champion` off the old version (there is none on the first promotion) and `candidate` off the new one, then set it:

```sql
ALTER MODEL ML.MODELS.CHURN_GNN VERSION champion UNSET ALIAS;   -- skip on the first promotion
ALTER MODEL ML.MODELS.CHURN_GNN VERSION V3 UNSET ALIAS;         -- V3 holds candidate
ALTER MODEL ML.MODELS.CHURN_GNN VERSION V3 SET ALIAS = champion;
```

Jobs that load `alias="champion"` pick up the new version on their next run.

### Trackers

`MlflowTracking(experiment_name=None, run_name=None, log_system_metrics=True, tags=None, log_model=False, *, target=None)`:
- With no `target`, it records where MLflow's own configuration points (`MLFLOW_TRACKING_URI`, `mlflow.set_tracking_uri`; automatic on Databricks). `target` names another server with its credentials: a `DatabricksConnector`, a `SageMakerMLflow(arn)`, or a tracking URI. With a `target`, `experiment_name` is required to record runs.
- `log_model=True` logs the model on the fit run without registering it; `MLOps(registry=)` implies it.

`SnowflakeTracking(source, *, experiment_name, database=None, schema_name=None, run_name=None, tags=None, native_model_lineage=False)`:
- `source` is a `SnowflakeConnector`, whose identity opens the session and whose Snowflake Model Registry `MLOps(registry=)` registers in, or a Snowpark session, which tracks only.
- `database` and `schema_name` default to the session's (`PUBLIC` when only `database` is given). The fit is in Snowsight under AI & ML » Experiments.
- The registered version carries the bundle; load it with `Trainer.load_from_connector`.

## Score

`predict(snapshot)` scores the trained query's population at its prediction time; `entities=` scores given rows; `predict(prediction_table, snapshot=...)` scores a table generated beforehand. A scoring job needs no `Graph` code: `GraphSnapshot.load_from_connector(cfg, id=...)` and `Trainer.load_from_connector(cfg, ...)`, then `predict`. It warns when the snapshot was taken of a different graph than the model's, naming the schema changes.

### MLflow pyfunc

A model registered through `MlflowTracking` is an MLflow pyfunc: it serves with `mlflow.pyfunc.load_model(uri)`, `spark_udf` and Model Serving.

```python
model = mlflow.pyfunc.load_model("models:/main.ml.users_churn@champion")
scores = model.predict(pd.DataFrame({"ENTITY": [1, 2, 3]}), params={
    "snapshot": cfg.snapshot_uri(snapshot),        # "/Volumes/main/ml/kumo/snapshots/<graph id>/<data version>"
    "prediction_time": "2024-06-01"})              # temporal queries
```

- Input rows have an `ENTITY` column and optionally a `TIMESTAMP`; scores come back in input order, one per scored row. Binary classification needs `params={"binary_classification_threshold": ...}` unless the model carries scoring defaults.
- Each replica loads a snapshot on its first `predict` with that URI and keeps it. `log_model(trainer, snapshot=..., serving_snapshot=...)` logs a model whose replicas load that snapshot at startup and use it when no `snapshot` param is given; `serving_snapshot` takes a URI, or a `GraphSnapshot` or snapshot id stored under the block's artifact root.
- Serving needs an image with kumo-engine installed: `mlflow models serve -m <model dir> --env-manager local`. Its pip requirement is `kumo-engine[s3,gcs]` of the same version (plus `[text]` for text columns).

## Credentials and running off-platform

`MlflowTracking(target=...)` and `SnowflakeTracking(connector)` take the same identities as the connectors, so a fit on a laptop or any VM can record to Databricks, SageMaker or Snowflake. An identity with no credentials (or no identity) uses the platform's ambient chain (a notebook, job or instance role, `aws configure`, `databricks auth login`); one with credentials uses exactly those.

```python
from kumo.engine import DatabricksConnector, DatabricksIdentity, S3Identity, SnowflakeConnector, SnowflakeIdentity

workspace = DatabricksConnector(catalog="main", schema_name="ml", warehouse_id="<id>",
                                identity=DatabricksIdentity(host="https://<workspace>", client_id="<id>", client_secret="<secret>"))
databricks = MlflowTracking("/Users/<you>/users_churn", target=workspace)
sagemaker = MlflowTracking("users-churn", target=SageMakerMLflow("<ARN>", identity=S3Identity(role_arn="<role ARN>")))
snowflake = SnowflakeTracking(SnowflakeConnector(warehouse="<wh>", database="ML", schema_name="RUNS", role="<role>",
                              identity=SnowflakeIdentity(account="<account>", username="<user>", private_key="<PEM>")),
                              experiment_name="USERS_CHURN")
```

`load_model(uri, target=...)` and `log_model(..., target=...)` take the same targets. A `DatabricksConnector` without an identity uses a notebook, a job or a `~/.databrickscfg` profile with no environment variables to set. MLflow has no per-client credentials, so with explicit credentials each MLflow call kumo-engine makes sets `DATABRICKS_HOST` and a short-lived `DATABRICKS_TOKEN` (never the secret), or the SageMaker tracking URI and AWS session, and restores them afterwards: don't call MLflow or Databricks yourself on other threads during a fit. For your own MLflow calls as that identity, use `with tracking.client() as client:`, an `MlflowClient` for the tracker's server with the same state set for the block (`mlflow.search_runs` inside it reaches that server too). A `SnowflakeConnector` without `role=` uses the user's default role. `examples/mlops/smoke.py databricks|sagemaker|snowflake` is a manual live check per platform.

### SageMaker managed MLflow

1. Create a tracking server once (15-25 minutes): `aws sagemaker create-mlflow-tracking-server --tracking-server-name <name> --artifact-store-uri s3://<bucket>/mlflow --role-arn <server role ARN>`; the server role needs read and write on that prefix. A stopped server starts with `aws sagemaker start-mlflow-tracking-server --tracking-server-name <name>` (about 20 minutes); `kumo.MLOps` fails until `aws sagemaker describe-mlflow-tracking-server --tracking-server-name <name>` reports `Started`.
2. Give the job's identity `sagemaker-mlflow:*` on the server ARN and S3 read and write on the artifact store and the kumo artifact root. Rights scoped to the root's prefix (`s3:GetObject`, `s3:PutObject` and `s3:DeleteObject` on `<bucket>/<prefix>/*`, `s3:ListBucket` limited to that prefix) suffice: kumo-engine makes no bucket-level calls.
3. Install `kumo-engine[mlflow,sagemaker]` and pass `MlflowTracking(..., target=SageMakerMLflow(arn))`; the registry is the same server.
4. MLflow's run links do not open on SageMaker: open the UI from SageMaker Studio, or at the URL `aws sagemaker create-presigned-mlflow-tracking-server-url --tracking-server-name <name>` prints. A tracked fit warns with this command, and `result.tracking.url` is `None`.

## Reference

### Install

- A bare `kumo-engine[mlflow]` with `--find-links <wheel dir>` can resolve to an unrelated `kumo-engine` on PyPI; install the engine wheel by path: `pip install --find-links <wheel dir> "<wheel dir>/<kumo_engine wheel>[mlflow]"`.
- **GPU:** on a plain GPU VM or SageMaker image, install a torch build that matches the NVIDIA driver first, e.g. `pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.12.*" torchvision`. PyPI's default torch is built for a recent CUDA (`torch.version.cuda`; CUDA 13 needs driver ≥ 580); cu126 runs on ≥ 525.60. With `accelerator="auto"`, a torch that cannot use the driver runs on CPU: check `torch.cuda.is_available()`.
- **Databricks:** Runtime 17.3 LTS ML or later keeps its torch, torchvision and pyarrow (kumo-engine accepts pyarrow 16 through 24; a custom CA bundle for S3 needs 21 or later). On 16.4 LTS ML pip replaces torch and breaks torchvision. The parent folder of an experiment path must exist. Registering to Unity Catalog from outside Databricks also needs `boto3`.
- **MLflow:** the `mlflow` extra installs `mlflow-skinny` 3.5 or later (Databricks Runtime 17.3 ships 3.0.1, which `%pip` upgrades); a 3.x client works against SageMaker's MLflow 3.0 servers.
- **SageMaker serving:** MLflow's stock serving containers cannot install kumo-engine; on SageMaker endpoints serve with `-m /opt/ml/model -p 8080`. GPU endpoints need a driver for CUDA 12.6 (set `InferenceAmiVersion`, e.g. `al2023-ami-sagemaker-inference-gpu-4-1`); batch transform takes JSON (`dataframe_split` plus `params`).

### Exporting and loading artifacts

Every artifact has one way out and one way back. A `kumo.MLOps` stands in for the connector to read the artifact root.

| Artifact | Export | Load |
|---|---|---|
| Snapshot | `snapshot.export(SnapshotOutputConfig(connector, path))`, returns the location | `GraphSnapshot.load_from_connector(connector, path=location)`, `(cfg, id=...)` |
| Training table | `table.export(TrainingTableOutputConfig(connector, name))` | `TrainingTable.load_from_connector(connector, path=name, pq=pq, snapshot=snapshot)` |
| Model | `result.export(ModelOutputConfig(connector, path=... or location=...))`, or `fit(export_config=...)` | `Trainer.load_from_connector(connector, path=... or location=...)` |

Loading needs a file destination; a table export is for consumers outside kumo-engine. Exporting again to a destination that holds a complete copy of the same artifact reuses it (logged at info). `ModelOutputConfig(connector, location=...)` publishes to a connector's registry outside a block.

### Lineage and manifests

Every artifact carries `_provenance.json`, metadata only (the underscore keeps Spark and Hive readers off it). Parents are embedded, minus their statistics and mutable fields; a parent without a manifest (a bundle from an older release) is an id stub.

- **Snapshot:** each source table's connector, location and version, and the engine build. `graph.snapshot(record_sources=False)` keeps only connector and table names, a SHA-256 of each location, and Custom SQL as a hash.
- **Training table:** the query, generation plan, task type and split seed, its snapshot, and for `TrainingTable.from_source_table` the source table. `table.id` is deterministic over snapshot, query, plan, seed and engine build; a source without a version gets a random id.
- **Model** (`ModelManifest`, `result.manifest`): the bundle id (a content hash excluding the manifest), fit times and fit mode, the training table and warm-start model (an embedded model keeps only ids of its own model parents), `tracking` (the tracker's run) and `dependencies`. The bundle carries the holdout predictions (`holdout.parquet`), so `holdout_df()` works on a loaded model.

### Artifacts, bundles and dependencies

- A model's **bundle** holds only its own state: weights, encoders, the training spec and the holdout. Its manifest's **`parents`** are lineage; its **`dependencies`** are what `predict` needs besides the bundle and a snapshot: the `predictor` entry point, the `kumo-engine` `package` (with `[text]` when a text column is encoded) and content-addressed `asset`s.
- **Assets** are stored once under `<root>/assets/<asset_kind>/<sha256>/` and shared by every model that uses them. `Trainer.load` and the pyfunc look for each at the location recorded, then under the enclosing block's root, then in the embedding-model store, check its sha256 and cache non-local copies under `~/.cache/kumo/assets/`. A stored asset that cannot be found raises `MissingDependencyError`.
- **Text encoders** are the first assets: word-vector models (`GloVeVectors`) are stored; other `TokenEmbeddingModel`s are recorded by fingerprint and must be registered wherever the model loads:

```python
store = kumo.EmbeddingModelStore()
store.add("glove100", kumo.GloVeVectors("glove.6B.zip", dim=100))   # the official GloVe release .zip, unchanged
plan.column_processing.encoder_overrides = {"users.bio": kumo.TextEmbedding("glove100")}
with kumo.mlops(cfg):
    result = kumo.Trainer(plan, embedding_models=store).fit(table, snapshot=snapshot)
trainer = kumo.Trainer.load_from_connector(cfg, path=f"models/{result.id}")   # anywhere, no store
```

A text column `infer_metadata()` reads as categorical (short values from a small vocabulary) takes a text encoder once its stype is text: `users.with_column("bio", stype="text")`.

### Moving snapshots between machines

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, holding `kumo_snapshot.json` and one directory per export generation (`index/` and `_provenance.json`).
- **Transfer:** parallel, with a CRC32 per range. Each export commits by writing `kumo_snapshot.json` last, so the previous export stays loadable and concurrent exports never mix: the last commit wins. Superseded generations are removed best-effort. A load completes or leaves nothing, and follows a newer export that commits meanwhile.
- **Reuse:** a load reuses the local copy only if it holds the export's bytes and passes DiskGraph's integrity check; an export skips the upload when the destination holds a complete export with the same bytes.
- **Staleness:** `snapshot.source_changes(graph)` reports each table as `unchanged`, `changed` or `unknown`; `load_from_connector(..., graph=graph)` warns about another graph's schema or changed tables.
- **Trust:** load only from paths whose writers you trust; checksums detect corruption, not tampering. A warehouse connector cannot hold a snapshot.

### Custom trackers

Other systems plug in through `kumo.engine.tracking.Tracking`, which receives ready-built `RunRecord`s so every tracker records the same names: `start(record)` with the fit run, `on_trial(record, event)` with a trial attempt after each of its events, and `finish(record, error)` with the final record (`None` when tracking stopped early). Optional members: `prepare()`, `tags`, `check(registry)` (when `kumo.MLOps` is built), `log_model(bundle_dir, manifest, snapshot, registry)` (logs and registers, returning the version) or `connector` (a connector whose model registry `registry` registers in), `load_model(location)` and `register(result, location)` (for loading and `result.export(cfg)`), `platform`/`run_id`/`run_url` (recorded as `result.tracking`). A tracker that raises is not called again for that fit and its run ends failed. `RunRecord.inputs` maps each source table to its `connector`, `location`, `version` and `version_kind`; a tracker's `tags` are added except `kumo.*` and param names.

### Progress events

`fit(callbacks=[...])` calls each callback with every `kumo.engine.tracking.Event`: `fit_started`, per trial `trial_started`, `epoch_ended` and `trial_ended`, then `fit_ended`. Ended events carry `params["status"]` and, on failure, `params["error"]`; a callback that raises is logged and not called again. For JSON lines: `callbacks=[lambda e: print(json.dumps(e.to_dict()))]`.

### Limits

- Tracker calls run on a background thread; `fit` waits at most 30 s for tracking to finish, and up to an hour for model logging and registration (600 s MLflow HTTP timeout during the upload). During a fit MLflow HTTP calls use 2 retries and a 30 s timeout unless `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` are set. Only global rank 0 tracks.
- MLflow rejects `@` in metric names: `test_map@10` is logged as `test_map_at_10`. Params MLflow rejects are counted in `kumo.dropped_params`.
- Snowflake keeps 200 metrics and 1000 params per run; the rest are counted in `kumo.dropped_metrics` / `kumo.dropped_params`. Snowflake has no tags (they are params of the same name), no child runs and no dataset inputs. A taken `run_name` gets a random suffix; an active run elsewhere in the process is never resumed.
- The fit's source tables are MLflow run inputs: Databricks tables as Delta tables at the snapshot's version (so Unity Catalog links them to the registered model), file sources by URI.
