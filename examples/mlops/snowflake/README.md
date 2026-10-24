# Snowflake ML Jobs and Tasks (experimental)

**Experimental.** This template is built on preview Snowflake APIs
(`MLJobDefinition` is in preview and the `SYSTEM$EXECUTE_ML_JOB` statement it
generates is undocumented) and has not been run end to end; see
[Unverified](#unverified). They may change without notice.

`register_jobs.py` registers two ML Job definitions whose payload is this
`examples/mlops` directory, and one Task per definition that runs it.

- **`KUMO_TRAIN`** (`KUMO_TRAIN_TASK`, Sundays 02:00 UTC) runs `fit_job.py`
  on `USERS`, `ITEMS` and `ORDERS` in `<database>.<schema>`, publishes a new
  version of model `USERS_CHURN` in that schema's Model Registry and moves the
  `champion` alias to it (`--registry snowflake`, which builds
  `kumo.MLOps(registry=ModelOutputConfig(connector, location=...))`). The
  snapshot, training table and model are stored under `--artifact-root`. The
  fit is recorded in
  Snowflake Experiment Tracking through the connector's own Snowpark session
  (`SnowflakeConnector.session()`, the container's OAuth token).
- **`KUMO_SCORE`** (`KUMO_SCORE_TASK`, daily 03:00 UTC) runs `score_job.py`,
  loading the version `champion` points at and writing
  `USERS_CHURN_PREDICTIONS`.

To review a model before it is scored, drop the alias from the registry
location in `jobs.mlops` and move `champion` yourself with
`kumo.engine.set_model_alias`.
If moving the alias fails, the fit still publishes, warns, and scoring keeps
the previous `champion`.

Each Task runs its job synchronously with the statement
`MLJobDefinition.to_sql(job_args=<flags>, use_async=False)` generates. `USER_TASK_TIMEOUT_MS` is 24 h for train and 6 h for score
(default 1 h); a job that outlives it, or the warehouse's
`STATEMENT_TIMEOUT_IN_SECONDS`, is cancelled.

## Before running

1. Make `kumo-engine[snowflake]` and its dependency wheels installable in the
   job container (package index via an external access integration, or a
   custom image).
2. Create the GPU compute pool `<compute_pool>` and the stage `<stage>`. The
   artifact root must outlive the job container: an `s3://` or `gs://` URI
   the external access integration reaches.
3. Run:
   ```bash
   python examples/mlops/snowflake/register_jobs.py --connection <connection> \
     --compute-pool <compute_pool> --stage <stage> --database <database> \
     --schema <schema> --warehouse <warehouse> \
     --artifact-root s3://<bucket>/kumo/ \
     --requirement 'kumo-engine[snowflake]' --external-access-integration <eai>
   ```
4. Tasks are created suspended: `ALTER TASK <name> RESUME`, or
   `EXECUTE TASK <name>` to run one now.

## Privileges for the role that owns the tasks

- `USAGE` on the database, schema, warehouse and compute pool;
  `CREATE SERVICE`, `CREATE TASK`, `CREATE TABLE` and `CREATE MODEL` on the
  schema; `EXECUTE TASK` on the account.
- Read and write on the stage; `USAGE` on the external access integration (or
  `SNOWFLAKE.PYPI_REPOSITORY_USER`).
- `SELECT` on the source tables, `INSERT` for predictions, and ownership of
  the model to move its alias.

## Notes

- The ML Jobs launcher runs the job script from the payload directory, so
  `jobs.py` imports; a non-zero exit fails the job (its `mljob_result` is
  still `None`, so judge a run by its status).
- The Task's synchronous `CALL` holds the task warehouse while the job runs;
  a serverless task avoids that.
- `--scratch-dir` is `/tmp/kumo`; add a volume through `spec_overrides` if the
  graph needs more disk. Tasks retry only via `TASK_AUTO_RETRY_ATTEMPTS`.

## Unverified

Nothing has been registered or run. `SYSTEM$EXECUTE_ML_JOB` (what `to_sql`
generates) is not a documented SQL API and `MLJobDefinition` is in preview.
Unverified: reaching the reader's presigned URLs without an external access
integration, a failed job failing the Task, and creating experiments from a job's OAuth session.

## References

- `MLJobDefinition`:
  https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/jobs/snowflake.ml.jobs.MLJobDefinition
- ML Jobs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/overview
- `CREATE TASK`: https://docs.snowflake.com/en/sql-reference/sql/create-task
- Model versions:
  https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/model-management
