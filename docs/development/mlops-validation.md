# Local MLOps lifecycle validation

Run a small Linux CPU lifecycle against an installed candidate wheel using a
local SQLite MLflow tracker and registry. For managed execution and UI checks,
see [Snowflake notebook tests](snowflake-notebook-actions.md). The fixture has
60 users and 1,500 events; it trains two versions for two epochs each, with two
steps per epoch. Local execution also trains an untracked model with two
candidates and two epochs each, refits the selected candidate on train and
validation data, and generates training explanations. It verifies exported
epoch history after a fresh-process reload and re-export. Managed profiles
retain the two tracked fits and their platform-specific negative checks.
Tracked local phases set both ambient MLflow URIs to the attempt's SQLite database.

## Scenario

The [runner](../../packages/kumo-engine/test/wheel_mlops_lifecycle.py) executes
five separate processes locally (managed profiles use the first four):

1. **Fit:** export the training snapshot/table, train and register two versions,
   verify multi-epoch tracking, and point `champion` to the first version. Local
   execution also exports an untracked model with both candidates’ epoch history.
2. **Score:** restore models, snapshots and the training table from durable
   artifacts. Predict by each explicit version and by alias before and after
   promotion. Verify resolved model identity, tracking, complete prediction IDs,
   distinct training/scoring lineage and the ordinary prediction schema without
   MLOps. Check explicit accounting for an unknown prediction ID.
3. **Evaluate:** restore saved predictions after the alias has moved, ingest
   later events, and evaluate the original predictions. Verify drift identities,
   label coverage, an independent event-window label join and recomputed AUROC.
4. **Negative cases:** reject an incomplete model, an unwritable artifact root
   and a prediction export without completion markers. Verify the untracked model’s
   history survives reload/re-export without importing MLflow.
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

Drift checks cover identity and zero self-comparison, not independently nonzero
sensitivity. Removing completion markers checks incomplete-artifact rejection;
this scenario does not interrupt an active remote writer. Existing snapshot-export
tests cover writer interruption separately. Local execution does not qualify
Databricks, Snowflake, SageMaker, or their UI.

Run the harness checks and existing shared-runner checks without training:

```sh
.venv/bin/python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_mlops_lifecycle.py test/scripts/test_engine_gpu_smoke.py
```

## Customer models and kumo-pytorch

Pass `--workload /absolute/path/customer.py` to use a customer-owned lifecycle.
Without it, the runner executes the Kumo scenario above. A workload is ordinary
Python code with this interface:

```python
PHASES = ('train', 'recover')


def run_phase(name, root, profile):
    # root is the attempt's pathlib.Path. profile is None locally, or the
    # selected backend with config, model, experiment, and artifact_root.
    # Train/load your model here and raise on invalid results.
    # Return JSON-serializable evidence for the phase receipt.
    ...
```

The workload owns its model, training loop, metrics, prediction schema, registry
serialization and loader. No `Trainer`, AUROC, binary prediction schema, or
explanation API is required. Declare only the phases your scenario implements;
a successful custom scenario establishes only the checks that workload performs.
Phase names must be unique lowercase filename components. `summary`,
`platform`, and `platform-prerequisite` are reserved for the harness.

The runner retains candidate Engine wheel checks, `pip check`, separate `-I`
workers, timeouts, logs, receipts, and failure handling. It records the workload
path and SHA-256. Put disposable producer files under `root / 'scratch-PHASE'`;
the runner removes that directory after each successful phase. Persist everything
needed by later phases in durable artifacts. Dependencies and helper modules
must be installed in the environment; isolated workers do not add the workload's
directory to Python's import path. Keep imports and execution inside `run_phase`
where possible so missing dependencies produce a blocked worker receipt.

The [regression example](../../packages/kumo-engine/test/wheel_mlops_pytorch.py)
uses experimental `kumo-pytorch` batches with a user-owned `torch.nn.Linear`
model and optimizer. It derives a log feature, verifies input normalization
was fitted on the training split, and logs
`train/mse` and `validation/rmse` for two epochs, and registers two native
MLflow PyTorch versions. A fresh process downloads the preprocessing artifact
and weights, checks their fingerprints with `Artifact.load` and `load_weights`,
and compares complete predictions from both the custom loader and the promoted
registry model. It uses the small in-memory engine; it does not test DiskGraph
execution or model quality.

Install a compatible `kumo-pytorch` wheel alongside the candidate Engine wheel
and MLflow, following its [installation guide](../../packages/experimental/kumo-pytorch/docs/installation.md).
Unlike the Engine candidate, the additional wheel is not checksum-qualified by
the harness; record its build identity separately when qualifying a release.
Then run:

```sh
/tmp/mlops-venv/bin/python -I \
  /absolute/path/wheel_mlops_lifecycle.py \
  --wheel /absolute/path/kumo_engine-CANDIDATE.whl \
  --expected-version CANDIDATE \
  --workload /absolute/path/wheel_mlops_pytorch.py \
  --output-dir /tmp/customer-mlops-attempt-001
```

For authorized managed execution, pass `--workload` with the existing
`--platform-config` and `--execute-platform` arguments in the selected runtime.
The Snowflake `run_notebook` helper also accepts `workload=...` for a full
invocation. Copy the workload into that runtime before launching. Existing
notebook payload builders do not automatically package custom workload files or
install their dependencies. A separate `platform-prerequisite` phase runs the
profile's real denied-artifact probe before any customer phase. UI review and
native runtime checks remain required; the Kumo-specific exported-history gap
applies only to the default scenario.

The example uses the configured MLflow tracking/registry APIs for Databricks
and SageMaker. Those paths require platform validation; a local pass does not
qualify either service. Snowflake's native registry needs a customer workload
using its native model API; this MLflow example reports that prerequisite as
blocked. The generic workload interface accepts such a workload without routing
its model through Engine's registry serializer.

Run the optional real regression lifecycle with `kumo-pytorch` and MLflow
installed (it skips when either is absent):

```sh
python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_mlops_pytorch.py
```


## CPU and GPU platform matrix

The [shared platform test](../../packages/kumo-engine/test/byom_platform_test.py)
uses one workload contract across Databricks, SageMaker and Snowflake. It installs
the complete Engine wheel, its matching DiskGraph constituent wheel and
`kumo-pytorch`. Engine supplies platform connectors and MLOps configuration;
`kumo-pytorch` supplies batches and persisted preprocessing; the customer owns
the PyTorch model and optimizer. This example uses the in-memory batch engine.

Each CPU case registers two model versions. Each GPU case registers two CUDA
versions. Each case uses its selected device throughout training and recovery. A fresh worker restores preprocessing
and weights, loads each version through the native registry, checks its device,
compares predictions and promotes the alias. The common verifier requires
matching workload hashes, distinct versions/runs, successful worker exits and
cleanup. Missing CUDA or CPU-only evidence fails a requested GPU case.

GPU environments select `torch==2.9.1+cu128` from the
[official CUDA 12.8 index](https://download.pytorch.org/whl/cu128/torch/).
The distribution version and CUDA runtime must agree. Snowflake GPU versions
retain `cuda_version=12.8` and the CUDA-tagged dependency; CPU versions use
`cuda_version=None`. Native reload validation stays enabled. Snowflake documents
these options under [PyTorch registration](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/built-in-models/pytorch).

| Platform | CPU fixture | GPU fixture |
| --- | --- | --- |
| Databricks | Single-node `KUMO_TEST_` cluster, `i3.xlarge` or `i3.2xlarge` | Single-node `KUMO_TEST_` cluster, `g5.xlarge` or `g4dn.xlarge`, GPU ML runtime |
| SageMaker | Pinned Distribution CPU image, `ml.m5.2xlarge` | Pinned Distribution GPU image, `ml.g5.xlarge`; dedicated test policies/quota must permit it |
| Snowflake | `KUMO_TEST_POOL`, CPU runtime | `KUMO_TEST_GPU_POOL`, single-node `GPU_NV_S`, GPU runtime |

An authorized Snowflake administrator can provision the separate GPU fixture:

```sql
CREATE COMPUTE POOL KUMO_TEST_GPU_POOL
  MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 60 INITIALLY_SUSPENDED = TRUE;
GRANT USAGE ON COMPUTE POOL KUMO_TEST_GPU_POOL TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
```

For one Actions run, open **SageMaker notebook platform checks**, select the
reviewed branch, set `platform_matrix=true`, `platform=all` and
`accelerator=all`, and supply both Databricks cluster IDs. This existing entrypoint
calls the [customer-model matrix](../../.github/workflows/kumo-engine-byom-platforms.yml),
which builds both packages once and executes all six platform/device cases.
`platform` or `accelerator` can select a smaller diagnostic run. Manual dispatch,
exact branch admission and environment-specific identities remain enforced.
The three test environments admit `main` and `egu/byom-mlops-lifecycle`, have no
required deployment reviewers, and disable administrator bypass. Remove the
exact development-branch exception after testing. Older main-only/reviewer-gated
guard revisions reject this policy until the update is merged.

The same test can run locally with already configured platform credentials:

```sh
python packages/kumo-engine/test/byom_platform_test.py --execute \
  --platform all --accelerator all \
  --wheel /absolute/path/kumo_engine-CANDIDATE.whl \
  --workload-wheel /absolute/path/diskgraph-MATCHING.whl \
  --workload-wheel /absolute/path/kumo_pytorch-CANDIDATE.whl \
  --cpu-cluster-id CPU_CLUSTER_ID --gpu-cluster-id GPU_CLUSTER_ID \
  --output-dir /tmp/byom-platforms-001
```

Use Linux x86_64 Python 3.12 and the owning platform's submission-client
requirements. Omit `--execute` to inspect all six command plans without cloud
access. The public summary contains only fixed platform/device/phase statuses;
raw logs stay in private attempt directories. Native UI, GPU serving endpoints,
DiskGraph execution and model quality are separate checks.
