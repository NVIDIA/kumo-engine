# MLOps platform validation

Use the [shared lifecycle](mlops-validation.md) with live platform storage,
tracking and registries. Each attempt creates a unique model, experiment and
artifact prefix. Existing models and aliases are not selected. Resources remain
available for inspection after the run.

The input is small synthetic Parquet data generated on the executor. Warehouse
table ingestion has separate connector tests.

## Snowflake

Start with the [Snowflake notebook E2E acceptance plan](snowflake-notebook-e2e.md).
The [branch-scoped Actions trial](snowflake-notebook-actions.md) submits a
headless Container Runtime notebook and collects its receipts. The interactive
notebook below remains available. Separate-job recovery and browser assertions
are not implemented.

Prepare an existing test database, schema, warehouse and internal stage using
`SNOWFLAKE_SSE` encryption. The executing role needs access to the stage,
experiment tracking and model registry in the selected namespaces.

Import [validation.ipynb](../../examples/mlops/snowflake/validation.ipynb) into
an existing Snowflake Workspace notebook using Container Runtime. Snowflake
execution is notebook-only; the CLI supports planning but rejects `--execute`
for this profile. Warehouse Runtime, client connections and standalone ML Jobs
are not execution paths for this test.

Upload the bundled wheel, the three helpers listed below and
[snowflake.json](../../packages/kumo-engine/test/mlops_platforms/snowflake.json)
into one stage prefix. The notebook downloads them, installs the candidate,
lets you configure existing test resources, previews the plan and starts the
lifecycle. Restart its kernel after installation before starting validation.
Dependency downloads need approved external access or preinstalled packages.

The launcher requires Snowflake's notebook IPython extension, records the active
session and optional workspace identifier, and launches
fresh subprocesses. Each worker requires the notebook launch context and mounted
runtime OAuth token, account and host. It checks its account, user and role
against the notebook session before constructing MLOps or writing artifacts.
Missing runtime authentication or a different identity blocks execution; there
is no client-credential fallback. Unknown notebook runtime variants are blocked.
The launch context contains identifiers, not credentials.

The mounted token is an internal authentication mechanism for notebook workers,
not an alternative standalone SPCS execution profile. See
[Container Runtime notebooks](https://docs.snowflake.com/en/developer-guide/snowflake-ml/notebooks-on-spcs)
and [runtime SQL authentication](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/spcs-execute-sql).
The exact notebook authentication path still needs live account verification.

Snowflake readback uses `SHOW RUNS`, `SHOW RUN PARAMETERS` and `SHOW RUN METRICS`
for the exact experiment. The test checks finished status, epoch history,
prediction/model identities and monitoring metrics. Model versions are read
through the registry, and the test moves `champion` between its two versions.
Drift and evaluation can have separate tracking runs. See the
[Snowflake run commands](https://docs.snowflake.com/en/sql-reference/sql/show-run-in-experiment).

## Databricks

Edit [databricks.json](../../packages/kumo-engine/test/mlops_platforms/databricks.json)
to select an existing test catalog, schema, volume and workspace experiment
parent. Run in an existing Databricks runtime with the candidate wheel installed.
The profile uses Unity Catalog for model versions and the Files API for volume
artifacts.

`identity: "ambient"` uses runtime authentication. Every worker must be able to
authenticate independently. For a service principal, select
`identity: "service_principal"` and provide `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` in the runtime environment.
Do not put credential values in the profile.

## SageMaker

Edit [sagemaker.json](../../packages/kumo-engine/test/mlops_platforms/sagemaker.json)
with an existing managed MLflow server/app ARN and S3 test prefix. Run inside an
existing training or processing job. The job role needs the corresponding
MLflow and S3 permissions. An optional `role_arn` uses an explicitly selected
assumed identity for tracking and artifact access.

Runtime evidence uses the training job environment or the processing job's
configuration file. The latter is provided by
[SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html).
The test does not create a pipeline, endpoint, job or image repository.

## Candidate and execution

Use the finished bundled candidate wheel, including its DiskGraph extension and
ingest binary. Separately installed component wheels are not admitted as that
candidate. Install the platform extra: `snowflake`, `databricks`, or `sagemaker`.

`environment: "venv"` requires a clean virtual environment. `"managed"` permits
the platform's Python layout while checking the wheel checksum/version and
comparing imported Engine, Core, DiskGraph and native files with the selected
wheel. Both modes require `pip check` to pass.

Stage these three files together on the executor:

```sh
cp packages/kumo-engine/test/wheel_mlops_lifecycle.py \
  packages/kumo-engine/test/wheel_mlops_platform.py \
  packages/kumo-engine/test/wheel_gpu_smoke.py /your/job/payload/
```

Inspect a profile without importing provider clients or contacting services:

```sh
python /your/job/payload/wheel_mlops_platform.py \
  --config /your/job/snowflake.json
```

For Databricks or SageMaker, run in the selected existing runtime:

```sh
python -I /your/job/payload/wheel_mlops_platform.py \
  --config /your/job/databricks.json \
  --wheel /your/job/kumo_engine-CANDIDATE.whl \
  --expected-version CANDIDATE \
  --output-dir /your/job/evidence/attempt-001 \
  --execute
```

Snowflake starts the live lifecycle only through the notebook cell. For the
other platforms, `--execute` starts it. Use a new output directory for each attempt. The initial profiles use
CPU; a GPU profile requires one visible CUDA device and records real CUDA work
using the shared task runner's callbacks.

To test provider permission denial, add `denied_artifact_root` pointing to a test
prefix the executing identity cannot write. Missing configuration or an
unverifiable denial remains blocked. Transport errors and missing stages do not
count as permission-denial proof. Remote corruption and active-writer recovery
remain separate from this permission check.

## Evidence

Retain `summary.json`, phase receipts, logs, `platform.json`, wheel identity and
test hashes with the notebook or platform job reference. Receipts contain the observed run,
metric, model-version and lineage records. Failed attempts remain separate.

Inspect the platform UI against those exact run and model identifiers, including
displayed epoch/monitoring metrics and both registered versions. API readback does
not verify the UI. Returned run links are recorded where available; links are not
invented for Snowflake or SageMaker.

UI review remains a required blocked item in the automatic report. Full epoch
history in exported bundles remains not implemented. Neither a configuration
plan nor local harness tests count as live platform qualification.

Run the platform harness checks without accounts or cloud access:

```sh
.venv/bin/python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_mlops_platform.py \
  test/scripts/test_engine_mlops_lifecycle.py test/scripts/test_engine_gpu_smoke.py
```
