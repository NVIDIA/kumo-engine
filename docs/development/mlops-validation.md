# Local MLOps lifecycle validation

Run a small Linux CPU lifecycle against an installed candidate wheel using a
local SQLite MLflow tracker and registry. The fixture has 60 users and 1,500
events; it trains two versions for two epochs each, with two steps per epoch.
Each worker sets both ambient MLflow URIs to the attempt's SQLite database.

## Scenario

The [runner](../../packages/kumo-engine/test/wheel_mlops_lifecycle.py) executes
four separate processes:

1. **Fit:** export the training snapshot/table, train and register two versions,
   verify multi-epoch tracking, and point `champion` to the first version.
2. **Score:** restore models, snapshots and the training table from durable
   artifacts. Predict by each explicit version and by alias before and after
   promotion. Verify resolved model identity, tracking, complete prediction IDs,
   distinct training/scoring lineage and the ordinary prediction schema without
   MLOps. Check explicit accounting for an unknown prediction ID.
3. **Evaluate:** restore saved predictions after the alias has moved, ingest
   later events, and evaluate the original predictions. Verify drift identities,
   label coverage, an independent event-window label join and recomputed AUROC.
4. **Negative cases:** reject an incomplete model, an unwritable artifact root
   and a prediction export without completion markers.

Successful producer scratch and original source files are removed before
recovery. Failed attempts retain their scratch, logs and receipts. Each attempt
needs a new output directory. Worker timeouts terminate the process group.

## Reused coverage

| Existing test code | Use in this scenario |
| --- | --- |
| `wheel_gpu_smoke.py::make_plan('binary')` | The existing binary model recipe, with smaller epoch/step/batch budgets and two seeds. |
| `test_mlflow_model.py`, `test_mlflow_target.py` | Extend registry/version coverage across process restarts and alias promotion. |
| `test_snapshot_export.py`, `test_training_table_export.py` | Extend export/reload coverage after removing original source and scratch files. |
| `test_monitoring.py` | Reuse the temporal fixture pattern and independent label/AUROC assertions on durable predictions. |
| `test_query_from_snapshot.py`, `test_mlops.py` | Reuse unknown-ID accounting and local permission-failure scenarios. |

Numerical reload consistency remains covered by the eight-task runner. This
scenario checks which model was resolved and recorded, rather than comparing
prediction values from explicit-version and alias loads.

## Run

Prerequisites: a candidate wheel containing the MLOps APIs, a supported Linux
Python, a non-root user, and enough local disk for the environment and artifacts.
Use a fresh environment. Full MLflow supplies the local SQL tracking dependencies.
Build the wheel using the [release guide](releasing-kumo-engine.md).

```sh
python3 -m venv /tmp/mlops-venv
/tmp/mlops-venv/bin/python -m pip --isolated install \
  '/absolute/path/kumo_engine-CANDIDATE.whl[mlflow]' 'mlflow>=3.5,<4'
mlops_tools_dir="$(mktemp -d /tmp/mlops-tools-XXXXXX)"
cp packages/kumo-engine/test/wheel_mlops_lifecycle.py \
  packages/kumo-engine/test/wheel_gpu_smoke.py "$mlops_tools_dir/"
/tmp/mlops-venv/bin/python -I "$mlops_tools_dir/wheel_mlops_lifecycle.py" \
  --wheel /absolute/path/kumo_engine-CANDIDATE.whl \
  --expected-version CANDIDATE \
  --output-dir /tmp/mlops-attempt-001
```

The runner verifies the installed wheel checksum, version, import locations,
environment isolation and `pip check`. Each phase writes a log and JSON receipt.
`summary.json` records the wheel identity, command, runner/helper hashes, statuses
and runtime. Worker receipts record CPU threads and Linux process and terminated
child RSS maxima separately, not simultaneous process-tree memory.

Full epoch history is currently unavailable in model exports. Tracker curves
alone do not satisfy that artifact requirement: the summary records it as
`not implemented`, leaves overall qualification `blocked` and exits nonzero.
Installation/access prerequisites are `blocked`; incorrect behavior is `failed`.
Unreached phases remain `blocked`.

Removing completion markers checks incomplete-artifact rejection. Active-writer
interruption/retry remains covered by the existing snapshot-export tests.
Local execution does not qualify Databricks, Snowflake or SageMaker.

Run the harness checks and existing shared-runner checks without training:

```sh
.venv/bin/python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_mlops_lifecycle.py test/scripts/test_engine_gpu_smoke.py
```

## ADLS wheel lifecycle

The manual [ADLS workflow](../../.github/workflows/kumo-engine-adls-smoke.yml)
reuses [the ADLS lifecycle tests](../../packages/kumo-engine/test/test_adls_mlops.py).
Select a successful run of `Kumo Engine - Build and publish wheel` and its full
candidate commit SHA. The workflow downloads that run's Linux x86_64 wheel,
checks its build identity and verifies the installed wheel with the existing
lifecycle helper. The test harness comes from the selected dispatch ref.

The job uses the `gha-kumo-workflow-tests` environment's Azure identity, storage
account and container. Set `KUMO_MLOPS_LIVE_ADLS_ROOT` there to use another
existing test prefix. Otherwise, the job uses `engine-lifecycle` in the configured
container. The identity needs read, write and delete access to the test prefix.
The job supplies one projected OIDC token file to Python and native Arrow, then
removes it without saving it in the evidence. Each test creates a unique child
prefix and removes it during test teardown.

The first Python process ingests the Azure fixture, trains, saves the model and
snapshot, and exports predictions. After that process exits, the second process
uses new scratch and cache directories, reloads from Azure, and predicts again.
The tests compare saved output contents and predictions before and after reload.
The job rejects skipped tests and retains logs, wheel identity, dependency
versions and worker receipts as workflow artifacts.

This workflow has no automatic trigger and does not add a release gate. It uses
ambient Azure credentials. Explicit per-connector identities, denied-access
cases, long-running token renewal and large datasets need separate live checks.
