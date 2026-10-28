# Scheduled kumo-engine jobs

Two plain Python job scripts that run the normal `kumo.engine` API inside
`with kumo.mlops(kumo.MLOps(...)):`, and the wiring that runs them on a
schedule. Both take their settings as command-line flags.

- `fit_job.py` builds a snapshot, generates the users-churn training table and
  trains the model. Inside the block each artifact is stored under the
  artifact root (`snapshots/…`, `training_tables/<id>/`, `models/<bundle id>/`).
  With a registry, the model is registered and the `--alias` alias
  (`champion` by default) moves to it. It prints `model: models/<bundle id>`.
- `score_job.py` loads the version `--alias` points at (or `--model-path`), scores a fresh snapshot
  (or the stored one `--snapshot` names) and writes `--predictions-table` when
  given. Inside the block the run is stored under `predictions/<id>/` with its
  `_provenance.json`. It passes a 0.5 threshold unless the model carries
  scoring defaults.
- `jobs.py` holds what both share: the flags, the graph (users, items, orders),
  the connectors and the `kumo.MLOps` settings. Replace it with your own module.

| Directory | Platform | Registry | Schedule |
|---|---|---|---|
| [databricks](databricks/README.md) | Databricks Jobs (Asset Bundle) | Unity Catalog through MLflow | train on demand, score daily |
| [sagemaker](sagemaker/README.md) | SageMaker Pipelines | SageMaker managed MLflow | fit step, then score step |
| [snowflake](snowflake/README.md) (experimental, preview APIs) | Snowflake ML Jobs + Tasks | Snowflake Model Registry, Snowflake Experiment Tracking | train weekly, score daily |

## Flags

| Flag | Meaning |
|---|---|
| `--artifact-root` | `s3://`, `gs://`, `file://` or a local or `/Volumes` path; required |
| `--source` | `databricks`, `snowflake`, or a directory holding `users`, `items` and `orders` Parquet files |
| `--catalog`, `--schema`, `--warehouse` | Databricks catalog, schema and SQL warehouse id, or Snowflake database, schema and warehouse |
| `--table-prefix` | prefix of the `users`, `items` and `orders` table or file names, e.g. `demo_` for `demo_users`; default none |
| `--registry` | `mlflow` (`MLOps(tracking=MlflowTracking(), registry=ModelRegistryLocation(...))`) or `snowflake` (`MLOps(tracking=SnowflakeTracking(...), registry=ModelOutputConfig(...))`); omit for no registry |
| `--model-name` | registered model name, with `--registry`: `users-churn` on SageMaker (letters, digits and hyphens only), `<catalog>.<schema>.users_churn` in Unity Catalog, `USERS_CHURN` in Snowflake |
| `--alias` | with `--registry`: the alias the fit moves to its new version and the score job loads; default `champion`. With MLflow, register under `candidate` and promote yourself to gate scoring |
| `--model-path` | score only, without a registry: `models/<bundle id>` under the artifact root |
| `--snapshot` | score only: a stored snapshot under the artifact root |
| `--predictions-table` | score only: a table of the source connector to write |
| `--query`, `--num-experiments` | fit only: the predictive query and the number of experiments |
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

The query needs several 30-day windows of order history; for a small sample
pass a shorter `--query` and `--num-experiments 1`.

## What every scheduler sees

The scripts are ordinary Python: they exit 0 on success and non-zero with a
traceback on failure, and log to stderr. A failed registration or tracking
call is a failure too (`--on-error raise`), so a fit that stored its model but
did not register it does not leave the score job on the previous version
unnoticed. Databricks Jobs, SageMaker Pipelines
and Snowflake Tasks keep their own run ids, status and logs.
