# Databricks Jobs

`databricks.yml` is an Asset Bundle with two jobs on a single-node GPU job
cluster. Each task is a `spark_python_task` running a job script with
`--job-id {{job.id}} --job-run-id {{job.run_id}}`; the cluster's
`spark_env_vars` configure `jobs.py`.

- **`kumo_train`** runs `fit_job.py` on the `users`, `items` and `orders`
  tables in `<catalog>.<schema>`. `KUMO_REGISTRY=mlflow` logs the fit to
  `MLFLOW_EXPERIMENT_NAME`, registers the model as a new version of
  `<catalog>.<schema>.users_churn` and moves the `champion` alias to it. The
  snapshot is exported to the volume.
- **`kumo_score`** runs `score_job.py` daily at 03:00 UTC (created paused).
  It loads `models:/<catalog>.<schema>.users_churn@champion`, builds a fresh
  snapshot and writes `users_churn_predictions` in the connector's schema. Set
  `KUMO_SNAPSHOT=snapshots/<id>` to score an exported snapshot instead.

## Before deploying

1. Make the kumo-engine wheel and its dependencies (kumo-core, kumo-api,
   kumo-pql, diskgraph, kumo-databricks, mlflow 3) available to the cluster;
   set `kumo_engine_wheel` or add further `libraries` entries. On DBR 16.4
   install with `pip install --prefer-binary` (for example from an init
   script): its setuptools cannot build the SQLAlchemy sdist MLflow pulls.
2. Replace the `<...>` placeholders in `databricks.yml` (or pass
   `--var catalog=...`). The parent folder of the MLflow experiment
   path must already exist in the workspace.
3. Add a `targets` mapping for your workspace, then `databricks bundle
   validate` and `databricks bundle deploy -t <target>`. The bundle syncs this
   whole `examples/mlops` directory (`sync.paths: [..]`).

## Privileges for the run-as principal

- `USE CATALOG`, `USE SCHEMA` and `SELECT` on the three tables; `CREATE TABLE`
  and `MODIFY` on the schema for predictions.
- `READ VOLUME`/`WRITE VOLUME` on `<volume>`, and `CREATE VOLUME` on the
  schema (or a pre-created `<catalog>.<schema>._kumo_staging` volume) for
  staged table writes.
- `CAN USE` on the SQL warehouse; edit permission on the MLflow experiment.
- `CREATE MODEL` on the schema and `EXECUTE` on the model; moving an alias
  needs rights on the model itself.

## Notes

- **Script, not `python_wheel_task`.** Databricks overrides a custom
  `PYTHONPATH` at driver startup, so a wheel task cannot import `jobs.py`; the
  scripts put their own directory on `sys.path`. Databricks sets no job-id
  environment variable, so ids come from `{{job.id}}`/`{{job.run_id}}`.
- **Dedicated access mode** (`SINGLE_USER`): standard compute runs as a
  low-privilege user and has no GPUs. `KUMO_SCRATCH_DIR` is
  `/local_disk0/kumo`.
- **MLflow.** `MLFLOW_TRACKING_URI=databricks` and
  `MLFLOW_REGISTRY_URI=databricks-uc` are set explicitly;
  `MLFLOW_EXPERIMENT_NAME` is required in jobs.
- **One registry format.** A `DatabricksConnector` publish and an MLflow
  pyfunc model are different formats; use one per registered model.
- **Job clusters.** Both jobs repeat the cluster block; YAML anchors would
  remove the copy but are untested with `databricks bundle validate`.

## Unverified

The bundle has not been deployed or validated (only parsed), and the
workspace layout of files synced from outside the bundle root is unverified.

## References

- Task types and parameters:
  https://docs.databricks.com/aws/en/dev-tools/bundles/job-task-types
- Custom `PYTHONPATH` overridden:
  https://kb.databricks.com/en_US/clusters/cannot-set-a-custom-pythonpath
- Dynamic value references:
  https://docs.databricks.com/aws/en/jobs/dynamic-value-references
- Standard compute limitations:
  https://docs.databricks.com/aws/en/compute/standard-limitations
- Unity Catalog models:
  https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/
