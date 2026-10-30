# Scheduled kumo-engine jobs

Two plain Python job scripts that run the normal `kumo.engine` API inside
`with kumo.mlops(kumo.MLOps(...)):`, and the wiring that runs them on a
schedule. Both take their settings as command-line flags.

- `fit_job.py` builds a snapshot, generates the users-churn training table and
  trains the model. Inside the block each artifact is stored under the
  artifact root (`snapshots/…`, `training_tables/<id>/`, `models/<bundle id>/`).
  The model carries its `--threshold` (0.5 by default) as scoring defaults.
  With a registry, the model is registered and the `--alias` alias
  (`champion` by default) moves to it. It prints `model: models/<bundle id>`
  and, with `--model-path-file`, writes it to that file.
- `score_job.py` loads the version `--alias` points at (or `--model-path`)
  through `Trainer.load_from_connector(cfg, ...)`, scores a fresh snapshot
  (or the stored one `--snapshot` names) and appends to `--predictions-table`
  when given, with `PREDICTION_ID` and `BUNDLE_ID` columns. Inside the block
  the run is stored under `predictions/<id>/` with its `_provenance.json`.
- `jobs.py` holds what both share: the flags, the graph (users, items, orders),
  the connectors and the `kumo.MLOps` settings. Replace it with your own module;
  list text columns in `TEXT_COLUMNS` (e.g. `('users.BIO',)`), since inference
  reads short text as categorical. With `--on-error raise`, a score job whose
  graph differs from the one the model was trained on fails before scoring.

| Directory | Platform | Registry | Schedule |
|---|---|---|---|
| [databricks](databricks/README.md) | Databricks Jobs (Asset Bundle) | Unity Catalog through MLflow | train on demand, score daily |
| [sagemaker](sagemaker/README.md) | SageMaker Pipelines | SageMaker managed MLflow | fit step, then score step |
| [snowflake](snowflake/README.md) (experimental, preview APIs) | Snowflake ML Jobs + Tasks | Snowflake Model Registry, Snowflake Experiment Tracking | train weekly, score daily |

## Flags

| Flag | Meaning |
|---|---|
| `--artifact-root` | `s3://`, `gs://`, `file://`, a local or `/Volumes` path, or with `--source snowflake` an `@DB.SCHEMA.STAGE/...` path; required |
| `--source` | `databricks`, `snowflake`, or a directory holding `users`, `items` and `orders` Parquet files |
| `--catalog`, `--schema`, `--warehouse` | Databricks catalog, schema and SQL warehouse id, or Snowflake database, schema and warehouse |
| `--table-prefix` | prefix of the `users`, `items` and `orders` table or file names, e.g. `demo_` for `demo_users`; default none |
| `--registry` | `mlflow` (`MLOps(tracking=MlflowTracking(), registry=ModelRegistryLocation(...))`) or `snowflake` (`MLOps(tracking=SnowflakeTracking(connector, ...), registry=ModelRegistryLocation(...))`); omit for no registry |
| `--model-name` | registered model name, with `--registry`: `users-churn` on SageMaker (letters, digits and hyphens only), `<catalog>.<schema>.users_churn` in Unity Catalog, `USERS_CHURN` in Snowflake |
| `--alias` | with `--registry`: the alias the fit moves to its new version and the score job loads; default `champion`. With MLflow, register under `candidate` and promote yourself to gate scoring |
| `--model-path` | score only: `models/<bundle id>` under the artifact root, loaded instead of `--alias` |
| `--model-path-file` | fit: a file to write `models/<bundle id>` to; score: a file to read `--model-path` from, so a scoring step scores its own fit even after a later fit moved the alias |
| `--snapshot` | score only: the id of a snapshot stored under the artifact root |
| `--predictions-table` | score only: a table of the source connector to append to; with a directory `--source`, a path under it that each run replaces |
| `--query`, `--num-experiments`, `--threshold` | fit only: the predictive query, the number of experiments and the classification threshold the model scores with |
| `--scratch-dir` | local disk for snapshots, tables and bundles |
| `--on-error` | `raise` (default): a failed tracking call or registration fails the job after its result is stored (`kumo.MLOps(on_error="raise")`); `warn`: the job warns and exits 0 |

## Run locally

With `users.parquet`, `items.parquet` and `orders.parquet` in a directory:

```bash
cd examples/mlops
python fit_job.py --source <data dir> --artifact-root <artifact dir>
python score_job.py --source <data dir> --artifact-root <artifact dir> \
  --model-path models/<bundle id>
```

The default query, `PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH
users.USER_ID`, needs a few months of order history; a 30-day window needs
about 6 months. For a quick run on a small sample pass `--num-experiments 1`.

## What every scheduler sees

The scripts are ordinary Python: they exit 0 on success and non-zero with a
traceback on failure, and log to stderr. A failed registration or tracking
call is a failure too (`--on-error raise`), so a fit that stored its model but
did not register it does not leave the score job on the previous version
unnoticed. Databricks Jobs, SageMaker Pipelines
and Snowflake Tasks keep their own run ids, status and logs.
