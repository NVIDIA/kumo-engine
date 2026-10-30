# MLOps with Kumo Engine

Turn MLOps on with one `kumo.MLOps` and the normal `kumo.engine` calls store their snapshots, training tables, models and prediction runs under an artifact root, record fits and prediction runs in your experiment tracker, and register models in its registry.

Install the engine wheel by path, `pip install "<wheel dir>/<kumo_engine wheel>[mlflow]"` (add `s3`, `gcs`, `databricks`, `sagemaker` or `snowflake` as needed; Snowflake needs only `[snowflake]`): PyPI's `kumo-engine` 1.0.0 predates this API (see [Install](#install)). GPU, CUDA and Databricks Runtime notes are under [Install](#install).

## Quickstart

Each block assumes a `graph` and a `pquery` (see the README). It trains, stores and registers a model, scores with it, and checks score drift.

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
    run = result.predict(snapshot, output_config=kumo.PredictionOutputConfig(kumo.FileConnector(root_dir="out"), path="scores/"))
    print(result.registered_model.version, run.scored_rows, result.score_drift(run).psi)
```

### Databricks

In a notebook or job on Databricks Runtime 17.3 LTS ML or later, after `%pip install "<wheel dir>/<kumo_engine wheel>[databricks,mlflow,text]"` and `dbutils.library.restartPython()`:

```python
import kumo.engine as kumo
from kumo.engine.integrations.mlflow import MlflowTracking

cfg = kumo.MLOps("/Volumes/main/ml/kumo", tracking=MlflowTracking("/Users/<you>/churn"),
                 registry=kumo.ModelRegistryLocation("main.ml.users_churn", alias="candidate"))
dbx = kumo.DatabricksConnector(catalog="main", schema_name="ml", warehouse_id="<warehouse id>")
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=0.5))
    run = result.predict(snapshot, output_config=kumo.PredictionOutputConfig(dbx, path="churn_scores"))
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
    run = result.predict(snapshot, output_config=kumo.PredictionOutputConfig(kumo.S3Connector(), path="s3://<bucket>/scores/"))
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
    run = result.predict(snapshot, output_config=kumo.PredictionOutputConfig(sf, path="CHURN_SCORES"))
```

To serve only a better model, move its alias yourself: [Promote a model](#promote-a-model).

## How it fits together

- The **artifact root** holds every artifact a block produces, each in its own directory with a `_provenance.json` manifest that embeds its parents' manifests.
- The **tracker** records each fit and prediction run as a run of its own; the **registry** is the tracker's: MLflow (Unity Catalog, SageMaker MLflow, any MLflow server) or the Snowflake Model Registry.

| id | names | where to find it |
|---|---|---|
| `snapshot.id` | a graph snapshot | `<root>/snapshots/<graph id>/<data version>/`; `result.snapshot_id` |
| `table.id` | a training table | `<root>/training_tables/<id>/` |
| `result.id` | a model (its bundle id) | `<root>/models/<id>/`; the `kumo.bundle_id` run tag; `kumo_bundle_id` on the registered version |
| `run.id` | a prediction run | `<root>/predictions/<id>/`; the `kumo.prediction_id` run tag; the `PREDICTION_ID` column of a table output |
| `result.registered_model.version` | the registered version | the registry |
| `result.tracking`, `run.tracking` | the tracker's run of the fit or prediction: `platform`, `experiment`, `run_id`, `run_name`, `url` | the tracker's UI |

Find runs with the platform's UI, or by tag: `kumo.kind` (`fit`, `prediction`), `kumo.bundle_id`, `kumo.prediction_id`. [What you'll see in MLflow and Snowsight](#what-youll-see-in-mlflow-and-snowsight) lists each run's names, metrics, tables and tags.

## Turning MLOps on

`kumo.MLOps` says where artifacts go and what records them. Use it as a block, or pass it as `mlops=` on a single call (`fit`, `predict`, `evaluate`, `score_drift`, `graph.snapshot`, `generate_training_table`):

```python
cfg = kumo.MLOps("s3://bucket/kumo", tracking=..., registry=..., on_error="raise")
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
trainer.fit(table, snapshot=snapshot, mlops=cfg)
```

- **`artifact_root`** (required): a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach: a volume through a `DatabricksConnector` (`"/Volumes/<catalog>/<schema>/<volume>/kumo"`) or a named stage through a `SnowflakeConnector` (`"@DB.SCHEMA.STAGE/kumo"`) once the installed sink holds files there.
- **`tracking`**: `MlflowTracking` or `SnowflakeTracking` (below), or your own [`Tracking`](#custom-trackers).
- **`registry`**: a `ModelRegistryLocation(name, alias=...)`; every fit registers its model there, in the tracker's registry.
- **`store`**: the artifact kinds stored when produced: `snapshot`, `training_table`, `model` and `prediction` (default: all).
- **`on_error`**: `"warn"` (default) warns about a failed store, tracking call or registration and returns; `"raise"`, for scheduled jobs, raises once the call has finished, with the failure as the `__cause__`. After `fit`, `trainer.result` still holds the model. A model whose bundle or assets could not be stored under the root is never registered. Under `"raise"`, `predict` also refuses a snapshot whose graph schema differs from the training graph's, and a run that could not be stored or tracked ends failed with `kumo.error`.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has; plain exports (`export(config)`, `export_config=`, `output_config=`) never write to the artifact root.
- `kumo.MLOps(...)` checks the tracker and registry when it is built (one experiment lookup, one registry listing, the model name's rules), so wrong credentials or a bad name fail before any work.

### Reading artifacts back

Pass the `kumo.MLOps` where a connector goes, with paths relative to the artifact root:

```python
snapshot = kumo.GraphSnapshot.load_from_connector(cfg, id=result.snapshot_id)
trainer = kumo.Trainer.load_from_connector(cfg, path=f"models/{result.id}")
champion = kumo.Trainer.load_from_connector(cfg, location=kumo.ModelRegistryLocation("users-churn", alias="champion"))
run = kumo.PredictionResult.load_from_connector(cfg, path=f"predictions/{run_id}")
table = kumo.TrainingTable.load_from_connector(cfg, path=f"training_tables/{table_id}", pquery=pquery, snapshot=snapshot)
```

`location=` loads the version an alias (or `version=`) names from the tracker's registry, one way on every platform. A loaded model is a `Trainer`; its `result` is the `TrainingResult` (`.id`, `.manifest`, `.metrics`, `.scoring_defaults`, `.holdout_df()`, `.holdout_evaluation()`).

## Train and register

`fit` inside the block stores the bundle under `models/<id>/`, records the fit as one run (a child run per hyperparameter-search trial attempt on MLflow, `trial-<n>-attempt-<m>/` names inside the run on Snowflake), and registers the model as a new version of the registry location, moving its alias to it.

- `result.registered_model` is the version `MLOps(registry=)` registered; `result.published_model` the one `fit(export_config=...)` published.
- `result.tracking` is the tracker's run: `TrackingRef(platform="databricks", run_id="...", url="https://.../ml/experiments/.../runs/...")`. The model's `_provenance.json` records the same.
- `fit(scoring_defaults=kumo.ScoringDefaults(...))` stores prediction defaults in the bundle before it is stored and registered: `binary_classification_threshold` (binary and multilabel classification) and `num_classes_to_return` (multiclass, multilabel, link prediction). `predict`, the MLflow pyfunc and the scoring jobs apply them when a call omits them; explicit arguments and a query's `TOP k` win. A binary model always returns `True_PROB`; with no threshold passed or stored, its `TARGET_PRED` is null. `result.with_scoring_defaults(...)` retunes a fitted model as a new bundle whose manifest names the original in `parents.derived_from`.
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
SHOW VERSIONS IN MODEL ML.MODELS.CHURN_GNN;                         -- or result.registered_model.version
ALTER MODEL ML.MODELS.CHURN_GNN VERSION champion UNSET ALIAS;        -- skip on the first promotion
ALTER MODEL ML.MODELS.CHURN_GNN VERSION <version> UNSET ALIAS;       -- it holds candidate
ALTER MODEL ML.MODELS.CHURN_GNN VERSION <version> SET ALIAS = champion;
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

### What you'll see in MLflow and Snowsight

Every value comes from exact counts over all rows, with no sampling. kumo-engine draws no figures of its own: numbers you compare across runs are metrics, curves and grids are tables, and the rest is in the run's card. Every artifact's location and how to load it is in the run's `_provenance.json`.

**Run kinds and names.** `kumo.kind` is `fit` or `prediction`.
- A fit is `fit-<YYYYMMDD-HHMMSS>-<model>` while it runs, and becomes `v<version>-<model>` when it registers a version. `<model>` is the registered name (the last part of a Unity Catalog name), else `<entity table>-<task type>`. A search's trials nest under it as `trial-<i>-<fit name>`; a single-trial fit has no trial run.
- A prediction is `<as of>-v<version>-<model>`, or `<as of>-<model>-<bundle id prefix>` when the model is not registered. `<as of>` is the date the rows were scored at (tag `kumo.as_of`; `<first>..<last>` for several). A `-2`, `-3` suffix marks a repeated run of the same day and model; it is not part of the model name.
- `score_drift` and `evaluate` write into the prediction run they check, so one run holds one day of scoring.
- Snowflake names are upper-case identifiers: `<MODEL>_FIT_<YYYYMMDD_HHMMSS>` and `AS_OF_<YYYYMMDD>_V<version>_<MODEL>` (`_2` for a repeat). Finished Snowflake runs cannot change, so `score_drift` and `evaluate` add `<prediction>_DRIFT` and `<prediction>_EVAL` runs.

**The card.** Each run's description (the Overview on MLflow, `run_card.md` on Snowflake) starts with a card.
- Line 1 is the run's status and watch numbers. For a fit: the tuned holdout metric, data problems, the model version, the number of trials and the task. For a prediction: realized quality, `psi` against training, the entities scored and their date, when their labels mature (tag `kumo.labels_mature`, until `evaluate` logs realized quality), skipped share and the model version.
- Then the model and the fit run that trained it, data problems, small tables, where things are, and the run's files.
- Write your own notes below the card's last line ("Notes below this line are kept"). Editing above it stops kumo-engine updating that card; the run is then tagged `kumo.card_frozen`.

**Metrics.**
- Fit: the holdout quality unprefixed (`auroc`, `ap`, `acc`, `mae`, `map_at_10`, `holdout_loss` for the training objective, and `logloss`, `brier`, `f1`, `r2` and the like from exact counts). They fill MLflow's "Model metrics" section, the logged model's metrics (Models tab) and the registered version's description. After a refit they describe the refit model the bundle ships; the model before the refit is `before_refit/<metric>`. Per-epoch `train/<metric>` and `val/<metric>`, step = epoch.
- Prediction: `pred/rows`, `pred/skipped_share`, `pred/positive_rate` (binary, with a threshold), `pred/item_coverage` and `pred/top_1pct_items_share` (ranking and link prediction); `drift/psi`, `drift/ks` and `drift/wasserstein` against the training holdout. After `score_drift`: `drift/psi_prev`, `drift/ks_prev`, `drift/wasserstein_prev` against the previous run. After `evaluate`: `realized/<metric>` and `realized/labelled_share`. A prediction run shows "Model metrics (0)": the model's own quality is on its fit run.
- System metrics: `system/*`, step = seconds since the start, every second for 10 minutes and then every 10 seconds. A run with fewer than 10 readings logs none.
- A series is logged once it has two points, so a one-epoch fit or trial logs no epoch curves; the fit's card lists that epoch's values instead.
- Read one run's numbers in its Overview's Metrics table, below Details. The Model metrics tab is for the epoch and system curves; MLflow draws every other metric there as a single bar.

**Tables** (Artifacts tab, "View as table"; JSON files on Snowflake), one per grid whose x is not time:

| Run | Tables |
|---|---|
| fit | `holdout/by_threshold` (binary: threshold, share at or above, tp, fp, fn, tn, precision, recall, fpr), `holdout/calibration`, `holdout/confusion` (multiclass), `holdout/by_class` or `holdout/by_label`, `holdout/by_k`, `holdout/by_label_count` (multilabel), `holdout/by_horizon` (forecast), `holdout/histogram` and `holdout/residuals` (regression, forecast) |
| fit, prediction | `data/health` (each table and link: rows, dropped rows, duplicate keys, matched rows, failed lookups), `data/columns` (column statistics; `≈` marks approximate ones) |
| prediction | `pred/histogram`, `pred/by_class` (multiclass, multilabel), `pred/by_label_count` (multilabel), `pred/item_frequency` (ranking, link prediction); after `score_drift`, `drift/features` (every column the model encodes, sorted by PSI) |

A histogram table has a row per bin with `from` and `to` (the end bins are open), then a count and a cumulative share per column.

**Tags and datasets.**
- Every run: `kumo.kind`, `kumo.engine_version`.
- Fit: `kumo.task_type`, `kumo.tune_metric`, `kumo.snapshot_id`, `kumo.bundle_id`, `kumo.bundle_uri`, `kumo.model_name`, `kumo.model_version`, `kumo.best_trial`, and `kumo.data_warnings`, the number of data problems the card lists.
- Prediction: `kumo.prediction_id`, `kumo.as_of`, `kumo.entity`, `kumo.tune_metric`, `kumo.labels_mature` (temporal queries), `kumo.bundle_id`, `kumo.fit_run_id`, `kumo.model_name`, `kumo.model_version`, and `kumo.output_uri` or `kumo.output_table`. After `score_drift`: `kumo.drift_previous`; after `evaluate`: `kumo.labels_snapshot_id`.
- Datasets: the snapshot, then each source table, at their real locations; after `evaluate`, the labels snapshot (context `labels`). A prediction of a model its fit logged lists that model under "Logged models", Type Input.

**Finding runs.** Keys with a `/` or `.` need backticks:
- a model's fits: ``tags.`kumo.kind` = 'fit' and tags.`kumo.model_name` = 'churn'``;
- its scoring history, one row per day: ``tags.`kumo.kind` = 'prediction' and tags.`kumo.model_name` = 'churn'``, sorted by Run Name, with the columns `pred/rows`, `drift/psi`, `drift/psi_prev` and `realized/auroc`;
- drifted days: ``metrics.`drift/psi` > 0.2``;
- everything one model version scored: ``tags.`kumo.bundle_id` = '<bundle id>'``.

In a Chart view of fits only, the empty `drift/`, `pred/` and `realized/` sections go away with the chart settings' "Hide charts with no data". If the Models tab shows no metric columns, add them with its Columns selector. Compare registered versions with the registry's Compare versions view. On MLflow 3.10 and later, including SageMaker MLflow Apps, the UI opens on GenAI pages: choose "Model training" in the sidebar, or add `?workflowType=machine_learning` to the URL. A new, empty experiment gets a description with the two run kinds and their filters.

**Per platform.**
- Databricks: the Unity Catalog version page shows the logged model's metrics; the version comment names the fit run and the bundle.
- SageMaker Tracking Server (MLflow 3.0): the version page shows no metrics, so the version description carries the headline metrics. MLflow's printed run links do not open; kumo-engine prints how to open the UI instead.
- Snowflake: metrics against step only; no tags (they are params), datasets or table views (tables are JSON artifacts); 200 metric names per run and 500 runs per experiment. kumo-engine warns at 450 runs and stops tracking at 500, so use one experiment per model.

MLflow has no 2-D heatmap, threshold slider, confusion picture, reference line, bar histogram or labelled category chart; those views are tables here.

## Score

`predict(snapshot)` scores the trained query's population at its prediction time; `entities=` scores given rows; `predict(prediction_table, snapshot=...)` scores a table generated beforehand.

```python
run = champion.predict(snapshot, output_config=kumo.PredictionOutputConfig(connector, path="..."))
run.id, run.bundle_id, run.snapshot_id, run.scored_rows, run.skipped_rows
```

- **`PredictionOutputConfig(connector, path=...)`**: a directory or file on a file destination (local, S3, GCS, a volume or stage), else a table. `output_paths={"predictions": ..., "embeddings": ...}` writes embeddings too.
- **Tables** append and get two columns, `PREDICTION_ID` (`run.id`) and `BUNDLE_ID`, so each row names its run and model. **Files** go to a directory per run, `<path>/<prediction id>/` (or `<dir>/<prediction id>/scores.csv` for a file name), with the run's `_provenance.json` next to the Parquet parts. `write_mode=WriteMode.OVERWRITE` replaces the table or exactly `path`.
- With MLOps on, each run is stored under `<root>/predictions/<id>/` and recorded as a tracker run named after its id: scoring options as params, row counts and score percentiles as metrics, the scoring snapshot's source tables as inputs. A failed export still records the run, ended failed with `kumo.error`.
- Entities the snapshot does not hold are skipped and counted (`run.skipped_rows`, the `data/skipped_fraction` metric), never refused.
- A scoring job needs no `Graph` code: `GraphSnapshot.load_from_connector(cfg, id=...)` and `Trainer.load_from_connector(cfg, ...)`, then `predict`. It warns when the snapshot was taken of a different graph than the model's, naming the schema changes (tables, columns, types, keys, edges); under `on_error="raise"` it raises instead, before scoring.

**Scheduled jobs:** [`examples/mlops`](../../../examples/mlops/README.md) has a fit job and a score job with Databricks Jobs, SageMaker Pipelines and Snowflake Tasks wiring.

### MLflow pyfunc

A model registered through `MlflowTracking` is an MLflow pyfunc: it serves with `mlflow.pyfunc.load_model(uri)`, `spark_udf` and Model Serving.

```python
model = mlflow.pyfunc.load_model("models:/main.ml.users_churn@champion")
scores = model.predict(pd.DataFrame({"ENTITY": [1, 2, 3]}), params={
    "snapshot": cfg.snapshot_uri(snapshot),        # "/Volumes/main/ml/kumo/snapshots/<graph id>/<data version>"
    "prediction_time": "2024-06-01"})              # temporal queries
```

- Input rows have an `ENTITY` column and optionally a `TIMESTAMP`; scores come back in input order, one per scored row. Binary classification returns `True_PROB`; `TARGET_PRED` needs `params={"binary_classification_threshold": ...}` or a stored threshold, and is null without one.
- Each replica loads a snapshot on its first `predict` with that URI and keeps it. `log_model(trainer, snapshot=..., serving_snapshot=...)` logs a model whose replicas load that snapshot at startup and use it when no `snapshot` param is given; `serving_snapshot` takes a URI, or a `GraphSnapshot` or snapshot id stored under the block's artifact root.
- Serving needs an image with kumo-engine installed: `mlflow models serve -m <model dir> --env-manager local`. Its pip requirement is `kumo-engine[s3,gcs]` of the same version (plus `[text]` for text columns).

## Monitor

`score_drift` compares a run's scores with the model's holdout and a previous run; `evaluate` scores a run once its outcomes are in a newer snapshot. With tracking on, both write into the prediction run they check: its `drift/*` and `realized/*` metrics, `drift/features`, tags and card (on Snowflake, a `<prediction>_DRIFT` or `<prediction>_EVAL` run instead). `drift.tracking` and `evaluation.tracking` name that run.

```python
with kumo.mlops(cfg):
    drift = champion.score_drift(run, previous=earlier_run)          # psi, ks, wasserstein, features, input_changes
    evaluation = champion.evaluate(run, labels_snapshot=newer)       # metrics, matured, labelled_fraction
```

- Inside a block, the training and scoring snapshots come from the manifests and the artifact root (only their column statistics are fetched); outside one, pass `snapshot=` and `training_snapshot=`. Missing inputs leave a field `None` with a `reasons` entry.
- `evaluate` returns the metrics the fit reports (`auroc`/`ap`, `mae`/`rmse`, `acc`, `map@k`), `matured=False` while the label window is still open, and the evaluated rows: `evaluation.frame` as a DataFrame, `evaluation.actuals` as a local Parquet directory. `evaluation.export(PredictionOutputConfig(...))` writes them to a table or directory for a warehouse monitor.

A later job finds the champion's runs by tag and reads them from the root:

```python
champion = kumo.Trainer.load_from_connector(cfg, location=kumo.ModelRegistryLocation("users-churn", alias="champion"))
with cfg.tracking.client():                       # the tracker's MLflow server, as its identity
    runs = mlflow.search_runs(experiment_names=["churn"], order_by=["attributes.start_time DESC"], max_results=2,
                              filter_string=f"tags.`kumo.kind` = 'prediction' AND tags.`kumo.bundle_id` = '{champion.result.id}' "
                                            "AND attributes.status = 'FINISHED'")
latest, *earlier = [kumo.PredictionResult.load_from_connector(cfg, path=f"predictions/{prediction_id}")
                    for prediction_id in runs["tags.kumo.prediction_id"]]
with kumo.mlops(cfg):
    drift = champion.score_drift(latest, previous=earlier[0] if earlier else None)
```

Snowflake runs have no tags, so read the ids from a table the runs wrote instead:

```python
ids = [row[0] for row in sf.session().sql(
    f"SELECT DISTINCT PREDICTION_ID FROM CHURN_SCORES WHERE BUNDLE_ID = '{champion.result.id}'").collect()]
latest, *earlier = sorted((kumo.PredictionResult.load_from_connector(cfg, path=f"predictions/{prediction_id}")
                           for prediction_id in ids), key=lambda run: run.manifest.created_at, reverse=True)
```

`latest` is the run started or written last; when a backfill scores older data after a daily run, order by `run.manifest.prediction_table.anchors` instead.

`load_from_connector` needs the run's `_SUCCESS` marker and fetches only its `_provenance.json`; the Parquet parts download when `evaluate` (or `files`) first reads them.

### Warehouse model monitors

Both platforms monitor the evaluated rows exported to a table; neither monitors link prediction or ranking outputs.

**Databricks (Data Quality Monitoring, inference log; the `databricks` extra installs a databricks-sdk with this API).** The classification monitor reads `TARGET_PRED`, so the run needs a threshold (passed or stored):

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import (AggregationGranularity, DataProfilingConfig, InferenceLogConfig,
                                                InferenceProblemType, Monitor)

evaluation.export(kumo.PredictionOutputConfig(dbx, path="churn_evaluated"))
w = WorkspaceClient()
table_id = w.tables.get("main.ml.churn_evaluated").table_id
w.data_quality.create_monitor(monitor=Monitor(object_type="table", object_id=table_id,
    data_profiling_config=DataProfilingConfig(output_schema_id=w.schemas.get("main.ml").schema_id,
        assets_dir="/Workspace/Users/<you>/monitoring", inference_log=InferenceLogConfig(
            problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
            prediction_column="TARGET_PRED", label_column="TARGET", model_id_column="BUNDLE_ID",
            timestamp_column="EVALUATED_AT", granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY]))))
```

**Snowflake (Model Monitor)**, on a version `MLOps(registry=)` registered. `evaluation.export` to a Snowflake table writes `TARGET` as a number and `EVALUATED_AT` as `TIMESTAMP_NTZ`, as the monitor needs. The baseline is the model's holdout as evaluated rows, `result.holdout_evaluation()`, exported the same way, so its columns and types are the source's. Pass the threshold the monitored runs were scored with, so both sides' `TARGET_PRED` match (null for runs without one):

```python
evaluation.export(kumo.PredictionOutputConfig(sf, path="churn_evaluated"))
threshold = latest.manifest.scoring.binary_classification_threshold      # None if the runs had none
champion.result.holdout_evaluation(binary_classification_threshold=threshold).export(
    kumo.PredictionOutputConfig(sf, path="churn_holdout_evaluated"))
```

```sql
CREATE MODEL MONITOR churn_monitor WITH MODEL=churn_gnn VERSION=<version> FUNCTION=predict
  SOURCE=churn_evaluated BASELINE=churn_holdout_evaluated WAREHOUSE=wh REFRESH_INTERVAL='1 day'
  AGGREGATION_WINDOW='1 day' TIMESTAMP_COLUMN=EVALUATED_AT ID_COLUMNS=('ENTITY')
  PREDICTION_SCORE_COLUMNS=('True_PROB') ACTUAL_CLASS_COLUMNS=('TARGET');
```

Snowflake allows one model monitor per model version: drop the old monitor (`DROP MODEL MONITOR churn_monitor`) before creating another on that version.

Details per task type: [Warehouse monitor columns](#warehouse-monitor-columns).

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
2. Give the job's identity `sagemaker-mlflow:*` on the server ARN (an MLflow App instead needs `sagemaker:CallMlflowAppApi` on the app ARN; the sample policy on AWS's MLflow Apps setup page lists the Tracking Server actions, which don't apply) and S3 read and write on the artifact store and the kumo artifact root. Rights scoped to the root's prefix (`s3:GetObject`, `s3:PutObject` and `s3:DeleteObject` on `<bucket>/<prefix>/*`, `s3:ListBucket` limited to that prefix) suffice: kumo-engine makes no bucket-level calls.
3. Install `kumo-engine[mlflow,sagemaker]` and pass `MlflowTracking(..., target=SageMakerMLflow(arn))`; the registry is the same server.
4. MLflow's run links do not open on SageMaker. A tracked fit warns with the command below, and `result.tracking.url` is `None`. To open the UI:
   - **SageMaker AI Studio** (not Studio Classic): Applications » MLflow, then Open MLflow on the server's menu. Studio acts as the domain's execution role, which needs `sagemaker-mlflow:AccessUI` on the server ARN (and `sagemaker:ListMlflowTrackingServers` to list it).
   - **SageMaker Unified Studio** lists only servers connected to the project: on the project's MLflow page, Connect Tracking Server with its ARN.
   - **Anywhere**, with your own credentials: `aws sagemaker create-presigned-mlflow-tracking-server-url --tracking-server-name <name>` prints a one-time URL valid for 5 minutes; it needs `sagemaker:CreatePresignedMlflowTrackingServerUrl` and `sagemaker-mlflow:AccessUI`.
5. In a SageMaker notebook, use a Python 3.11+ kernel with the wheel installed by path (`pip install --find-links <wheel dir> "<wheel dir>/<kumo_engine wheel>[mlflow,sagemaker,s3,text]" ipykernel ipywidgets`, then `python -m ipykernel install --user --name kumo-engine`); `SageMakerMLflow(arn)` uses the notebook's role. Notebook instances need the AL2023 platform (`notebook-al2023-v1`): the wheel needs glibc 2.34.

## Reference

### Install

- Install the engine wheel by path, `pip install "<wheel dir>/<kumo_engine wheel>[mlflow]"`. PyPI's `kumo-engine` is the 1.0.0 release, which predates the MLOps API, and a bare `kumo-engine[mlflow]` resolves to it over a dev wheel; with it installed, `kumo.MLOps` raises `AttributeError: module 'kumo.engine' has no attribute 'MLOps'` and `pip show kumo-engine` says 1.0.0. The wheel `scripts/build_wheel.sh` builds bundles kumo-core, kumo-api, kumo-pql, DiskGraph and the connector packages; with separately built wheels, add `--find-links <wheel dir>`.
- **GPU:** on a plain GPU VM or SageMaker image, install a torch build that matches the NVIDIA driver first, e.g. `pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.12.*" torchvision`. PyPI's default torch is built for a recent CUDA (`torch.version.cuda`; CUDA 13 needs driver ≥ 580); cu126 runs on ≥ 525.60. With `accelerator="auto"`, a torch that cannot use the driver runs on CPU: check `torch.cuda.is_available()`.
- **Databricks:** Runtime 17.3 LTS ML or later keeps its torch, torchvision and pyarrow (kumo-engine accepts pyarrow 16 through 24; a custom CA bundle for S3 needs 21 or later). On 16.4 LTS ML pip replaces torch and breaks torchvision. The parent folder of an experiment path must exist. Registering to Unity Catalog from outside Databricks also needs `boto3`.
- **Disk:** `kumo.configure(scratch_dir=...)` holds snapshots, tables and bundles; Python's own temporary files (multiprocessing) follow `TMPDIR`, so point both at a large local disk.
- **MLflow:** the `mlflow` extra installs `mlflow-skinny` 3.5 or later (Databricks Runtime 17.3 ships 3.0.1, which `%pip` upgrades); a 3.x client works against SageMaker's MLflow 3.0 servers.
- **SageMaker serving:** MLflow's stock serving containers cannot install kumo-engine; on SageMaker endpoints serve with `-m /opt/ml/model -p 8080`. GPU endpoints need a driver for CUDA 12.6 (set `InferenceAmiVersion`, e.g. `al2023-ami-sagemaker-inference-gpu-4-1`); batch transform takes JSON (`dataframe_split` plus `params`).

### Exporting and loading artifacts

Every artifact has one way out and one way back. A `kumo.MLOps` stands in for the connector to read the artifact root.

| Artifact | Export | Load |
|---|---|---|
| Snapshot | `snapshot.export(SnapshotOutputConfig(connector, path))`, returns the location | `GraphSnapshot.load_from_connector(connector, path=location)`, `(cfg, id=...)` |
| Training table | `table.export(TrainingTableOutputConfig(connector, name))` | `TrainingTable.load_from_connector(connector, path=name, pquery=pquery, snapshot=snapshot)` |
| Model | `result.export(ModelOutputConfig(connector, path=... or location=...))`, or `fit(export_config=...)` | `Trainer.load_from_connector(connector, path=... or location=...)` |
| Prediction run | `run.export(PredictionOutputConfig(connector, path=...))`, or `predict(output_config=...)` | `PredictionResult.load_from_connector(connector, path=...)` |
| Evaluation | `evaluation.export(PredictionOutputConfig(connector, path=...))` | `EvaluationResult.load_from_connector(connector, path=...)` |

Loading needs a file destination; a table export is for consumers outside kumo-engine. Exporting again to a destination that holds a complete copy of the same artifact reuses it (logged at info). `ModelOutputConfig(connector, location=...)` publishes to a connector's registry outside a block.

### Lineage and manifests

Every artifact carries `_provenance.json`, metadata only (the underscore keeps Spark and Hive readers off it). Parents are embedded, minus their statistics and mutable fields; a parent without a manifest (a bundle from an older release) is an id stub.

- **Snapshot:** each source table's connector, location and version, and the engine build. `graph.snapshot(record_sources=False)` keeps only connector and table names, a SHA-256 of each location, and Custom SQL as a hash.
- **Training table:** the query, generation plan, task type and split seed, its snapshot, and for `TrainingTable.from_source_table` the source table. `table.id` is deterministic over snapshot, query, plan, seed and engine build; a source without a version gets a random id.
- **Model** (`ModelManifest`, `result.manifest`): the bundle id (a content hash excluding the manifest), fit times and fit mode, the training table and warm-start model (an embedded model keeps only ids of its own model parents), `tracking` (the tracker's run), `dependencies` and `holdout` (score statistics of the holdout rows: count, mean, min, max, q01..q99; the positive rate for binary tasks), the baseline `score_drift` compares against. The bundle carries the holdout predictions (`holdout.parquet`), so `holdout_df()` works on a loaded model.
- **Prediction run** (`PredictionManifest`, `run.manifest`): its id; the model and the scoring snapshot; the prediction table (query hash, anchors, `requested_rows`, `scored_rows`, `skipped_rows` and up to 5 skipped ids per reason); the scoring options and where each came from; `outputs`, the destinations `output_config` wrote (even after a partial failure) and, with MLOps on, the copies stored under the artifact root; status, timings and score statistics per output; with MLOps on, also the scores' exact counts in fixed bins (`stats['predictions'].counts`): probabilities in hundredths, or the training target's bins.

### Artifacts, bundles and dependencies

- A model's **bundle** holds only its own state: weights, encoders, the training spec and the holdout. Its manifest's **`parents`** are lineage; its **`dependencies`** are what `predict` needs besides the bundle and a snapshot: the `predictor` entry point, the `kumo-engine` `package` (with `[text]` when a text column is encoded) and content-addressed `asset`s.
- **Assets** are stored once under `<root>/assets/<asset_kind>/<sha256>/` and shared by every model that uses them. `Trainer.load`, the pyfunc and the jobs look for each at the location recorded, then under the enclosing block's root, then in the embedding-model store, check its sha256 and cache non-local copies under `~/.cache/kumo/assets/`. A stored asset that cannot be found raises `MissingDependencyError`.
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

### Prediction output contract

| Column | Task types | Meaning |
|---|---|---|
| `ENTITY` | all | the entity's primary key |
| `ANCHOR_TIMESTAMP` | temporal queries | the prediction time, only with `PredictionOutputConfig(..., extra_fields=[MetadataField.ANCHOR_TIMESTAMP])` (`from kumoapi.jobs import MetadataField`); otherwise it is `run.manifest.prediction_table.anchors` |
| `TARGET_PRED` | binary classification, regression, forecasting | the predicted label or value; null for a binary run without a threshold |
| `False_PROB`, `True_PROB` | binary classification | class probabilities; `True_PROB` is summarized |
| `CLASS`, `SCORE` | multiclass, multilabel, ranking | one row per returned class; the top `SCORE` is summarized |

`predict` checks each destination first: a directory holding another export raises `ArtifactDestinationConflict`. A table written without the id columns takes one `OVERWRITE` before it can append. Object stores replace a directory by deleting, then writing, so a reader in between finds no `_SUCCESS`; `APPEND` to files is not supported.

### Moving snapshots between machines

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, holding `kumo_snapshot.json` and one directory per export generation (`index/` and `_provenance.json`).
- **Transfer:** parallel, with a CRC32 per range. Each export commits by writing `kumo_snapshot.json` last, so the previous export stays loadable and concurrent exports never mix: the last commit wins. Superseded generations are removed best-effort. A load completes or leaves nothing, and follows a newer export that commits meanwhile.
- **Reuse:** a load reuses the local copy only if it holds the export's bytes and passes DiskGraph's integrity check; an export skips the upload when the destination holds a complete export with the same bytes.
- **Staleness:** `snapshot.source_changes(graph)` reports each table as `unchanged`, `changed` or `unknown`; `load_from_connector(..., graph=graph)` warns about another graph's schema or changed tables.
- **Trust:** load only from paths whose writers you trust; checksums detect corruption, not tampering. A warehouse connector cannot hold a snapshot.

### Drift statistics

| statistic | definition |
|---|---|
| `psi` | population stability over up to 10 groups of bins, cut at the reference side's deciles (the holdout for `drift/psi`, the previous run for `drift/psi_prev`); shares are floored at 1e-6 |
| `ks` | the largest gap between the two CDFs at bin edges |
| `wasserstein` | the mean absolute gap over bin midpoints, in score units; the end bins are closed at each side's recorded minimum and maximum |

Each is an exact function of the score counts both sides stored on the same bins: probability hundredths, the training target's bins, or 101 quantile edges of the holdout's top scores (ranking and link prediction). It is absent, with a note, when the two sides counted on different bins, as for a previous run of another model version. `DriftReport.features` compares, per column the model encodes (by its stype in the model plan), the statistics DiskGraph wrote when both snapshots were built: numeric deltas of mean, std, min, max, quartiles, skewness, kurtosis, null rate and distinct count plus `standardized_mean_shift`; categorical null rate, distinct count, PSI and Jensen-Shannon divergence (`category_counts_capped` flags truncated counts); timestamp min and max shift; text null rate, distinct count, average and maximum tokens per value and `glove_overlap`, the share of tokens the default GloVe vocabulary covers (new words lower it). Other columns are not compared and are named in `reasons["features"]`. Distinct counts are DiskGraph's approximate counts (within a few percent). These are whole-snapshot statistics, so a slow change in a large event table is diluted. A tracked `score_drift` writes the full comparison into the prediction run as `drift/features`, sorted by PSI, and names the three most moved columns in its card.

### Warehouse monitor columns

- Databricks keys the monitor on `EVALUATED_AT` (anchors are older than its lookback by the label window). For regression use `INFERENCE_PROBLEM_TYPE_REGRESSION`; for multiclass `prediction_column="CLASS"`. `True_PROB` is profiled as an ordinary column.
- Snowflake: binary classification uses `PREDICTION_SCORE_COLUMNS` (in [0, 1]) with `ACTUAL_CLASS_COLUMNS`; multiclass `PREDICTION_CLASS_COLUMNS=(CLASS)`; regression `PREDICTION_SCORE_COLUMNS=(TARGET_PRED)` with `ACTUAL_SCORE_COLUMNS`. The timestamp is `TIMESTAMP_NTZ`, columns free of nulls, NaN and infinities, each used once; columns are listed as string literals in their stored case, `('True_PROB')` (`'"True_PROB"'` is not found). Without `BASELINE` there is no drift. Registration records the version's task type, which Model Monitor requires.

### Custom trackers

Other systems plug in through `kumo.engine.tracking.Tracking`, which receives ready-built `RunRecord`s so every tracker records the same names: `start(record)` with the fit run, `on_trial(record, event)` with a trial attempt after each of its events, and `finish(record, error)` with the final record (`None` when tracking stopped early). Optional members: `prepare()`, `tags`, `check(registry)` (when `kumo.MLOps` is built), `log_model(bundle_dir, manifest, snapshot, registry)` (logs and registers, returning the version) or `connector` (a connector whose model registry `registry` registers in), `load_model(location)` and `register(result, location)` (for loading and `result.export(cfg)`), `platform`/`run_id`/`run_url` (recorded as `result.tracking`), and the best-effort `log_prediction_run(record)` and `log_monitoring(record)`. A tracker that raises is not called again for that fit and its run ends failed. `RunRecord.inputs` maps each source table to its `connector`, `location`, `version` and `version_kind`; a tracker's `tags` are added except `kumo.*` and param names.

### Progress events

`fit(callbacks=[...])` calls each callback with every `kumo.engine.tracking.Event`: `fit_started`, per trial `trial_started`, `epoch_ended` and `trial_ended`, then `fit_ended`. Ended events carry `params["status"]` and, on failure, `params["error"]`; a callback that raises is logged and not called again. For JSON lines: `callbacks=[lambda e: print(json.dumps(e.to_dict()))]`. `graph.snapshot(callbacks=[...])` reports `snapshot_started`, `table_staged` per table and `snapshot_ended` (with `snapshot_id` and `reused`), each with `params["seconds"]`, the time since the snapshot started, and logs each at INFO. `callbacks=[print]` prints each as its name and fields, e.g. `table_staged table=users seconds=12.0`.

### Limits

- Tracker calls run on a background thread; `fit` waits at most 30 s for tracking to finish, and up to an hour for model logging and registration (600 s MLflow HTTP timeout during the upload). During a fit MLflow HTTP calls use 2 retries and a 30 s timeout unless `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` are set. Only global rank 0 tracks.
- MLflow rejects `@` in metric names: `test_map@10` is logged as `test_map_at_10`. Params MLflow rejects are counted in `kumo.dropped_params`.
- Snowflake keeps 200 metrics and 1000 params per run; the rest are counted in `kumo.dropped_metrics` / `kumo.dropped_params`. Snowflake has no tags (they are params of the same name), no child runs and no dataset inputs. A taken `run_name` gets a random suffix; an active run elsewhere in the process is never resumed.
- The fit's source tables are MLflow run inputs: Databricks tables as Delta tables at the snapshot's version (so Unity Catalog links them to the registered model), file sources by URI.
