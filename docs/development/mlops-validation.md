# Local MLOps lifecycle validation

Run a small Linux CPU lifecycle against an installed candidate wheel using a
local SQLite MLflow tracker and registry. The fixture has 60 users and 1,500
events; it trains two versions for two epochs each, with two steps per epoch.
An untracked fit also tries two candidates and records two epochs each. It
refits the selected candidate on train and validation data and generates
training explanations.
Tracked phases set both ambient MLflow URIs to the attempt's SQLite database.

## Scenario

The [runner](../../packages/kumo-engine/test/wheel_mlops_lifecycle.py) executes
five separate processes:

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
5. **Explain:** reopen a copy of the untracked training run with the public
   explanation API. Verify that the explanations describe the published refit
   model and not the pre-refit model. Render the gradient and cohort plots and
   inspect every sampled graph. Verify the diagnostics for an unknown sample, a
   model bundle, and an unreadable or missing gradient file.

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
| `test_explanation_artifacts.py`, `test_explanation_plots.py`, `test_training_explanations.py` | Extend fixture and mocked explanation coverage to a real retained run. Failed explanation generation remains covered by the mocked fit tests. |

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
  '/absolute/path/kumo_engine-CANDIDATE.whl[mlflow]' 'mlflow>=3.5,<4' \
  --extra-index-url https://download.pytorch.org/whl/cpu
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

The bundle retains recorded search-trial epochs in `metric-history.json`.
The negative-case phase loads and re-exports history without importing MLflow.
Older bundles remain readable and return `None` for `metric_history`.
The required CPU job reuses the candidate-wheel build.
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
