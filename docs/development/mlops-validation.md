# Local MLOps lifecycle test

Run a small Linux CPU lifecycle against an installed candidate wheel, using
SQLite MLflow tracking and registry. For Snowflake execution and UI checks, see
[Snowflake notebook tests](snowflake-notebook-actions.md).

## What it tests

The [runner](../../packages/kumo-engine/test/wheel_mlops_lifecycle.py) uses four
separate processes:

1. **Fit:** export the training snapshot/table, train/register two versions,
   verify multi-epoch tracking and set `champion` to the first version.
2. **Score:** restore durable artifacts; predict by version and alias before
   and after promotion; verify model identity, tracking, lineage, complete
   prediction IDs and the ordinary schema without MLOps. Account for unknown IDs.
3. **Evaluate:** reload saved predictions after alias movement, add later labels,
   and verify drift identity, label coverage and independently recomputed AUROC.
4. **Negative cases:** reject incomplete models, unwritable artifact storage
   and prediction exports without completion markers.

The fixture has 60 users and 1,500 events. It reuses
`wheel_gpu_smoke.py::make_plan('binary')`, with two fits of two epochs and two
steps per epoch. Successful producer scratch and source files are removed
before recovery. Prediction-value consistency is covered by the existing task
runner; this test checks which model was selected and recorded.

## Setup and run

Use a supported Linux Python, a non-root user, a fresh environment and an
identified candidate wheel containing the MLOps APIs. Build the wheel using the
[release guide](releasing-kumo-engine.md). Run from the repository root:

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

| Option | Purpose |
| --- | --- |
| `--wheel FILE` | Required candidate wheel; installed files are checked against it. |
| `--expected-version VERSION` | Required installed version. |
| `--output-dir DIR` | Required new directory for results and artifacts. |
| `--phase-timeout SECONDS` | Deadline per worker; default 600 seconds. |

## Results

`summary.json`, phase receipts and logs record assertions, wheel/helper hashes,
runtime and process memory measurements. Failed attempts retain their scratch.
Use a new output directory for each attempt.

Full epoch history in model exports remains unimplemented: overall qualification
is blocked and the command exits nonzero even if the implemented phases pass.
Drift checks include identity and zero self-comparison, not independent nonzero
sensitivity. Incomplete-artifact checks do not simulate an active writer's
interruption. Local execution does not qualify a cloud platform or its UI.

To check the harness without training or cloud access:

```sh
.venv/bin/python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_mlops_lifecycle.py test/scripts/test_engine_gpu_smoke.py
```
