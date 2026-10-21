# Scheduled kumo-engine jobs

Two plain Python job scripts that use the normal `kumo.engine` API inside
`kumo.job()`, and the wiring that runs them on a schedule:

- `fit_job.py` builds a snapshot (exported under `KUMO_ARTIFACT_ROOT` when
  set), trains the users-churn model, publishes it and moves the `champion`
  alias to it.
- `score_job.py` loads `champion`, scores a fresh snapshot (or the exported
  one named by `KUMO_SNAPSHOT`) with `ANCHOR_TIMESTAMP` on every row, and
  writes the predictions. It passes a 0.5 threshold unless the model carries
  scoring defaults.
- `jobs.py` holds what both share: the graph (users, items, orders), the
  connectors and the registry choice, all read from environment variables
  (listed in its docstring). Replace it with your own module.

| Directory | Platform | Registry | Schedule |
|---|---|---|---|
| [databricks](databricks/README.md) | Databricks Jobs (Asset Bundle) | Unity Catalog through MLflow | train on demand, score daily |
| [sagemaker](sagemaker/README.md) | SageMaker Pipelines | SageMaker managed MLflow | fit step, then score step |
| [snowflake](snowflake/README.md) (experimental, preview APIs) | Snowflake ML Jobs + Tasks | Snowflake Model Registry, Snowflake Experiment Tracking | train weekly, score daily |

## Run locally

With `users.parquet`, `items.parquet` and `orders.parquet` in a directory:

```bash
cd examples/mlops
export KUMO_SOURCE=file KUMO_DATA_ROOT=<data dir> KUMO_ARTIFACT_ROOT=<artifact dir>
python fit_job.py --job-id local --job-run-id run-1
KUMO_MODEL_PATH=runs/run-1/model python score_job.py --job-id local --job-run-id run-2
```

Without `KUMO_REGISTRY` the fit publishes the model to
`<artifact dir>/runs/<fit run>/model` (also in its summary's `outputs.model`),
and the score job loads the path `KUMO_MODEL_PATH` names under
`KUMO_ARTIFACT_ROOT`; scheduled score runs have their own run ids, so set it to
the fit run you want to score with. The query needs
several 30-day windows of order history; for a small sample set `KUMO_QUERY`
to a shorter window and `KUMO_NUM_EXPERIMENTS=1`.

## What every scheduler sees

- **Exit codes.** `0` ok, `1` failure, `143` SIGTERM (timeout or cancel).
  The summary's `error.kind` says whether a failure is `transient` (worth a
  retry), `permanent` or `unknown`.
- **Output.** Each script prints one JSON line on stdout when it ends:
  `status`, `exit_code`, `scheduler`, `bundle_id`, `snapshot_id`, `run_id`,
  `outputs` (the exported snapshot and published model, or the prediction
  destinations), `warnings` and `error`. Logs go to stderr.
- **Job ids.** `--job-id`/`--job-run-id` (or `KUMO_JOB_ID`/`KUMO_JOB_RUN_ID`)
  are recorded as `scheduler: {platform, job_id, job_run_id}` in the snapshot
  and training-table manifests and the bundle's `provenance.json`, and as
  `kumo.scheduler`/`kumo.job_id`/`kumo.job_run_id` on a tracked fit run.
  Without them the engine detects SageMaker and Snowflake ML Jobs itself.
