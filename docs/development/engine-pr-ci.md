# Engine PR check

The `Kumo Engine PR` workflow builds the current candidate wheel on a CPU
runner. The GPU job downloads that wheel and installs it once in a fresh
Python 3.12 environment. It runs on the
`nv-gpu-amd64-rtxpro6000-1gpu` group with the
`linux-amd64-gpu-rtxpro6000-latest-1` label.

The `kumo-engine-pr-cpu-lifecycle` job installs the same wheel in a clean CPU
environment and runs the small MLflow lifecycle. A failed lifecycle fails
`kumo-engine-pr-ci` and blocks the merge.

## Workflow selection

The workflow runs on trusted `pull-request/<number>` branches created by the
copy bot. For a manual run, select the PR branch in **Run workflow**, then enter
its PR number. The selected branch commit must match the current PR head.
Selecting `main` for an open PR fails that check.

The `changes` job reads all changed files in the PR, including previous names
of renamed files. It skips GPU work only when every file matches the documented
exemptions for docs, UI, Helm, tools, or benchmarks. A docs-only follow-up push
still checks the full PR. A manual run always selects the GPU check.

The `kumo-engine-pr-ci` check always runs. It requires either a successful
exemption decision or successful build, CPU lifecycle, and GPU jobs. Failed
selection, build, CPU lifecycle, and GPU jobs fail this check. Pushes to `main` build the wheel and seed caches
without selecting the GPU or PR-required jobs.

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
`0.005`, with zero relative tolerance. This limit applies to every task,
including regression and forecasting. A comparison outside the tolerance fails.

Labels can change only when the scores explain the change within that same
tolerance. Binary and multilabel decisions must follow the `0.5` threshold in
each output. Multiclass decisions must choose a highest-scoring class. If the
chosen class changes, the two classes must differ by at most `0.005` in both
outputs. A rank reversal is allowed only when each affected pair differs by at
most `0.005` in both outputs. A chain of adjacent ties cannot justify reversing
two scores farther apart. Rank coordinates remain complete and unique.

Training metrics report separately. They do not set a learning-quality gate.
The shared `engine-gpu-smoke` action requires one wheel path and an explicit
`reload-atol` value. Its Python runner and comparison functions also require
the tolerance. The PR workflow supplies `0.005` for RTX. Every caller enforces
its chosen absolute tolerance for all eight tasks, with zero relative tolerance.
There is no report-only comparison mode.

## Build and installer caches

Both Rust build paths inherit one `CARGO_TARGET_DIR` at
`packages/diskgraph/target`. The Rust cache restores that directory using a
shared key and the action's toolchain and dependency hashes. It excludes
workspace crates and Cargo binaries. Only `main` saves this cache; PR branches
restore compatible entries from `main`.

The build pins `ziglang`, `cargo-zigbuild` and pip to exact versions. A pip build
constraint also pins Zig inside isolated Python builds.

Stable Zig compiler wrappers give both Rust paths the same linker identity and
glibc 2.17 target. Without them, the command-line builder and maturin select
different wrapper paths, including paths from temporary isolated environments.
Cargo then invalidates dependencies despite a successful cache restore.

A CPU-only job on `main` seeds the same uv installer cache used on the GPU
runner. It installs the candidate wheel without importing models or training.
It uses Ubuntu 24.04 to match the RTX runner's platform in the uv cache key.
PR jobs restore this cache without saving branch-specific copies. Before saving,
uv prunes the cache for CI. Cache hits alone do not establish a build speedup.
Measure complete builds with and without restored caches on the same runner.

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

The CPU build, CPU lifecycle, and GPU jobs have separate time limits. The
wheel job has a 60-minute limit, the CPU lifecycle job has a 30-minute limit,
and the GPU job has a 22-minute limit. GitHub reports queue,
build, and GPU job durations.
The workflow retains the task summary as an artifact. Source and package
test workflows keep their existing checks.
