# Engine PR check

The `Kumo Engine PR` workflow builds the current candidate wheel on a CPU
runner. The GPU job downloads that wheel and installs it once in a fresh
Python 3.12 environment. It runs on the
`nv-gpu-amd64-rtxpro6000-1gpu` group with the
`linux-amd64-gpu-rtxpro6000-latest-1` label.

## Workflow selection

The workflow runs on trusted `pull-request/<number>` branches created by the
copy bot. A manual run accepts a PR number. The build and GPU jobs require
the checked-out commit to match the current PR head.

The `changes` job reads all changed files in the PR, including previous names
of renamed files. It skips GPU work only when every file matches the documented
exemptions for docs, UI, Helm, tools, or benchmarks. A docs-only follow-up push
still checks the full PR. A manual run always selects the GPU check.

The `kumo-engine-pr-ci` check always runs. It requires either a successful
exemption decision or successful build and GPU jobs. Failed selection, build,
and GPU jobs fail this check.

## Eight task checks

Each run fits eight models with seed zero. The tasks are binary, multiclass,
multilabel, multilabel ranking, regression, forecasting, static link prediction,
and temporal link prediction. Each task fits, predicts, saves, loads, and
predicts again. Prediction repeats use the same fitted model.

The runner requires actual nonempty CUDA training and prediction work. Outputs
must contain all expected IDs and coordinates, with finite values. Forecasts
must contain all twelve horizons at the expected time anchor. Classification
and link outputs must contain all requested classes or catalog items.

Repeated and reloaded predictions must match within an absolute tolerance of
`0.0001`, with zero relative tolerance. This limit applies to every task,
including regression and forecasting. Chosen labels and published ranks must
match exactly. A comparison outside the tolerance fails the job.

Training metrics report separately. They do not set a learning-quality gate.
The shared `engine-gpu-smoke` action accepts one wheel path and an optional
`reload-atol` value. Every caller enforces its chosen absolute tolerance for
all eight tasks, with zero relative tolerance. The PR workflow sets `0.0001`
for RTX. The shared action defaults to `0.00001`. There is no report-only
comparison mode.

## Fixture download

The action installs dependencies before requesting AWS credentials. Only the
fixture download step receives those credentials. Installation and GPU training
do not receive them through the action environment.

The fixture manifest records a shared S3 object and its SHA-256. The existing
bucket enables versioning. The action checks the archive hash before
extraction. The tables remain outside Git. See the
[fixture record](../../packages/kumo-engine/test/pr172/README.md).

## Local feedback

Run the pure output and workflow checks in the existing development environment:

```sh
.venv/bin/python -m pytest -q -c /dev/null --confcutdir=test/scripts \
  test/scripts/test_engine_gpu_smoke.py test/scripts/test_engine_pr_workflow.py
```

These tests need pytest, NumPy, pandas, and PyYAML. They require no GPU or cloud
access. The existing OSS Readiness pytest job also runs them.

The CPU build and GPU jobs have separate time limits. Build time no longer
uses the GPU job's budget. GitHub reports queue, build, and GPU job durations.
The workflow retains the task summary as an artifact. Source and package
test workflows keep their existing checks.
