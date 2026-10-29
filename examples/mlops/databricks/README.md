# Databricks Jobs

`databricks.yml` is an Asset Bundle with two jobs on a single-node job
cluster, a GPU node with the GPU ML runtime by default. Each task is a `spark_python_task` running a job script with its
flags as `parameters`; the cluster's `spark_env_vars` hold only MLflow's own
settings.

- **`kumo_train`** runs `fit_job.py` on the `users`, `items` and `orders`
  tables (each prefixed with `table_prefix`) in `<catalog>.<schema>` inside
  `kumo.mlops(...)` with the artifact root
  `/Volumes/<catalog>/<schema>/<volume>/kumo`: the snapshot, training table
  and model are stored under it. `--registry mlflow` logs the fit to
  `MLFLOW_EXPERIMENT_NAME`, registers the model as a new version of
  `<catalog>.<schema>.<model_name>` (default `users_churn`) and moves the
  `alias` variable's alias (default `candidate`) to it; move `champion` to it
  once it is better (see
  [Promote a model](../../../packages/kumo-engine/docs/mlops.md#promote-a-model)).
  Set `alias` to `champion` to serve every fit without a gate.
- **`kumo_score`** runs `score_job.py` daily at 03:00 UTC (created paused).
  It loads the version `score_alias` (default `champion`) points at in
  `<catalog>.<schema>.<model_name>`, builds a fresh snapshot and appends to
  `<predictions_table>` (default `users_churn_predictions`) in the
  connector's schema. Add `--snapshot <snapshot id>` to score a stored
  snapshot instead.

## Before deploying

1. Put the kumo-engine wheel and a `requirements.txt` for its dependencies
   (kumo-core, kumo-api, kumo-pql, diskgraph, kumo-databricks, mlflow-skinny)
   on a volume and set `kumo_engine_wheel` and `requirements`. Each task
   installs the requirements, then the wheel. Start the requirements file with
   `--prefer-binary`, so pip installs wheels rather than building sdists.
   - **Runtime:** Databricks Runtime 17.3 LTS ML or later: the GPU flavour
     (`17.3.x-gpu-ml-scala2.13`) on a GPU node such as `g5.xlarge`, or the
     CPU flavour (`17.3.x-cpu-ml-scala2.13`) on a node type with local NVMe
     disks such as `m5d.xlarge`. These runtimes already carry a torch that
     kumo-core accepts, so pip keeps the runtime's torch, torchvision,
     MLflow and cryptography; leave torch out of the requirements file.
     kumo-engine accepts pyarrow 16 through 24, so the runtime's pyarrow
     (19 on 17.3) is kept too. On 16.4 LTS ML, whose torch 2.6 is
     older than kumo-core needs, pip replaces torch and breaks the
     preinstalled torchvision.
2. Replace the `<...>` placeholders in `databricks.yml` (or pass
   `--var catalog=...`). `model_name`, `predictions_table`, `table_prefix`,
   `experiment` and `on_error` have defaults; set them to run beside existing
   objects in the same schema. The parent folder of the MLflow experiment
   path must already exist in the workspace.
3. Adjust the `dev` target (workspace `root_path`, under your user folder by
   default) or add your own, then
   `databricks bundle validate` and `databricks bundle deploy -t <target>`. The bundle syncs this
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

- **Strict mode.** Both jobs pass `--on-error raise` (variable `on_error`):
  if MLflow tracking or Unity Catalog registration fails, the task fails after
  the model or predictions are stored, and `champion` stays where it was.

- **Script, not `python_wheel_task`.** Databricks overrides a custom
  `PYTHONPATH` at driver startup, so a wheel task cannot import `jobs.py`; the
  scripts put their own directory on `sys.path` (from `sys.argv[0]`: a
  `spark_python_task` runs the file without `__file__`).
- **Dedicated access mode** (`SINGLE_USER`): standard access mode runs as a
  low-privilege user and has no GPUs. `--scratch-dir` is
  `/local_disk0/kumo`.
- **MLflow.** `MLFLOW_TRACKING_URI=databricks` and
  `MLFLOW_REGISTRY_URI=databricks-uc` are set explicitly;
  `MLFLOW_EXPERIMENT_NAME` is required in jobs.
- **Job clusters.** Both jobs repeat the cluster block; YAML anchors would
  remove the copy but are untested with `databricks bundle validate`.

## Verified

This bundle was deployed and both jobs were run on a single-node cluster
against synthetic `users`, `items` and `orders` tables, on a GPU ML runtime
on a `g5.xlarge` (A10G; the fit trained on the GPU): the fit registered a
new `users_churn` version and moved `champion`, and the score job loaded it and
wrote the predictions table. `test_job_templates.py` checks the scripts
locally.

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
