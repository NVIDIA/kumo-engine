# Local validation and benchmark evidence

Use a prepared **Python 3.11+** environment. These author tools do not install
packages, build environments, provision machines, dispatch CI, or access cloud
accounts. Their children run as your user; these commands are not a sandbox.
Check each profile's declared effects before running it.

## Find and run an existing check

```bash
.venv/bin/python scripts/author_checks.py list
.venv/bin/python scripts/author_checks.py show readiness
.venv/bin/python scripts/author_checks.py doctor readiness
.venv/bin/python scripts/author_checks.py run readiness --dry-run
.venv/bin/python scripts/author_checks.py run readiness --reports /tmp/readiness-attempt-1
```

Each run needs a new report directory **outside the checkout**. Select another
prepared interpreter with `--python /absolute/path/to/venv/bin/python`.
`doctor` checks executable/module presence and Python version; it does not
establish dependency compatibility, matching native builds, GPU functionality,
Docker daemon access, or enough RAM. Missing prerequisites are reported, never
installed automatically. See [setup](setup.md) and the owning profile's guide.

`results.json` records the profile, resolved argv, selected interpreter probe,
source revision/dirty state, host CPU/memory information, outcome, and raw-log
location. Nonzero exits and timeouts fail the command. A timeout terminates the
process group. Existing report directories are never reused. Logs and commands
can contain paths or workload output; inspect them before posting to a PR.
Credentials, private data, and tokens must not be command arguments or reports.

The catalog delegates to existing harnesses. `readiness` is small pure Python;
`public-connectors` uses the [existing public Docker runner](public-tests.md);
`diskgraph-rust` compiles native tests. None replaces the full test matrix.
The full [eight-task GPU CI](engine-pr-ci.md) still uses a pinned private fixture;
the author tool does not claim to reproduce it without access to that fixture.

## Generate and verify a public smoke dataset

Use a separate environment prepared with
[`benchmark/commerce/generator/requirements.txt`](../../benchmark/commerce/generator/requirements.txt).
All paths below are local; choose new output paths and a suitable SSD.

```bash
.venv-generator/bin/python scripts/author_checks.py run commerce-generate-smoke \
  --param data_root=/local-ssd/commerce-smoke \
  --disk-budget-gib 2 --reports /tmp/commerce-generate-1
.venv-generator/bin/python scripts/author_checks.py run commerce-verify-smoke \
  --param data_root=/local-ssd/commerce-smoke --reports /tmp/commerce-verify-1
```

The budget checks available disk space on output filesystems; **it is not a
quota or an estimate that the workload must fit**. Choose a budget based on your
inputs and keep monitoring space. The generator uses one worker and the checked-in
small configuration. Generation and verification are preparation, not timed
engine work. Hidden `_ground_truth` files must not become model inputs.

With Engine and a matching in-tree DiskGraph installed in your runtime environment:

```bash
.venv/bin/python scripts/author_checks.py run commerce-snapshot-smoke \
  --param data=/local-ssd/commerce-smoke/commerce/smoke \
  --param scratch=/local-ssd/commerce-scratch-1 \
  --param results=/local-ssd/commerce-snapshot-1 \
  --disk-budget-gib 4 --reports /tmp/commerce-snapshot-receipt-1
```

`--reports` holds the outer command receipt; `--param results` holds the existing
harness's artifacts. Snapshot-only success establishes neither model quality nor
inference speed. For training and prediction, use the small explicit budgets in
the [commerce workflow guide](../../benchmark/commerce/README.md).

## Scale on your own machine

The existing generator also supports `100M`, `1B`, and `50B` rungs. These are
large workloads, not smoke defaults. Decide your disk/RAM/time budget first;
start with one generator worker and the smallest useful rung. The public
placeholder distribution differs from the archived 50B reference data: this
creates a new experiment, not a reproduction of its historical rates.

From `benchmark/commerce/generator/`, with its prepared environment:

```bash
python -m scalebench.generate run --graph commerce --rung 100M \
  --config config.json --root /local-ssd/commerce-scale --workers 1
python verify.py --root /local-ssd/commerce-scale --rung 100M
```

Then follow the [context harness](../../packages/diskgraph/scripts/benchmarks/commerce/README.md)
to prepare seed IDs, build a fresh index, and map IDs. For a bounded first sampling
matrix (substitute your prepared index and seed map):

```bash
.venv/bin/python packages/diskgraph/scripts/benchmarks/commerce/run_matrix.py \
  --rung 100M --index /local-ssd/index --seedmap /local-ssd/seedmap.parquet \
  --vcpus 4 --dg-threads 1 --hops 2,3 --scale 0.01 --cap-s 60 \
  --out /local-ssd/sampling-attempt-1.jsonl
```

This uses four processes with one native thread each. Declare that configuration,
cache procedure, completed work, input manifest, and dataset hashes alongside the
result. Size worker counts to RAM and CPU availability. The harness controls its
own timers and marks incomplete cells; interpret them using its guide. Do not
add `--drop-all` on a shared machine. The old continuous ingest service can
format/mount disks and update its checkout; it is intentionally absent from the
author catalog.

## Compare baseline and candidate

Start with the runnable, dependency-free **self-comparison control**:

```bash
.venv/bin/python scripts/benchmark_pair.py benchmark/author-example/plan.json \
  --output /tmp/paired-control-1
```

This checks the runner, not engine performance. For a real change, copy its JSON
plan outside the checkout and edit both variants. Paths resolve relative to the
plan file. Use separate worktrees and prepared interpreters. Rebuild native
artifacts separately with matching compilers/flags; do not reuse an old native
module under new Python code. `{runner_python}` selects the tool's interpreter;
`{python}` expands to each variant's selected interpreter in argv. A venv's
executable path is preserved, including when it is a symlink.

The plan declares:

- `workload_id` and `scope`: the exact behavior and timing claim.
- `context`: seed, worker count, cache state, and device. Add shape, thread count,
  output order/tolerance, build flags, and known environmental differences.
- `inputs`: dataset/configuration **files** and optional expected SHA-256 values.
  For large datasets, identify a verified manifest; the tool hashes that file,
  not every file it references. Have the owning harness verify the full data.
- `baseline` and `candidate`: `cwd`, `python`, timed `argv`, checker
  `correctness_argv`, optional explicit `env` overrides, and `artifacts` files to
  hash (native libraries, wheels, scripts, locks). No shell is inserted.
- `correctness_expect`: a nonempty expected JSON object. Each checker receives
  a fresh `{result}` filename and must write that exact object after substantive
  checks; exit zero alone is insufficient. The control checks known integer
  answers. Real checkers should establish source-derived answers, valid output
  completeness/semantics, and allowed tolerances for the same workload.
- `timeout_seconds`: applies to every child, including checkers and warmups;
  `pairs` (default 4) and `warmups` (default 1 per side).

Checkers run **before timing**. This does not prove correctness of each measured
iteration; timed harnesses should validate their own outputs where needed. No
successful timing ratio is published for a failed/incomplete comparison;
`summary.md` still records the failure.
Raw partial attempts remain for diagnosis. The runner refuses existing output
paths and outputs anywhere inside either measured Git worktree.

Each sample is a fresh process. Pair order alternates AB then BA. `summary.md`
and `results.json` retain every time, per-pair ratio, logs, checker output,
interpreter probes, declared environment overrides, Git state, and file hashes.
The ratio is baseline time divided by candidate time; its paired median is not
the ratio of independently aggregated medians. There is no hard regression gate.

**These times include complete command startup, execution and shutdown.** They
are not kernel timers, CUDA event measurements, or memory measurements. Device
work must be synchronized by the harness. Short commands are especially affected
by startup and process-wait granularity. Existing specialized benchmarks such as
[temporal ranges](../../packages/diskgraph/scripts/benchmarks/shared_temporal_ranges.md)
retain their more precise operation timers, oracles, native hashes, and controls;
use `temporal-ranges-smoke` to discover/run its small profile.

A Git revision cannot identify uncommitted, untracked, or ignored build output.
Dirty state is recorded, but untracked/ignored contents are not fingerprinted.
List relevant files as artifacts. Inherited environment, full installed package
versions, thermal state, competing workloads, and device clocks are not captured.
Use matched isolated environments and retain dependency/build receipts. These
limitations are shown in the report rather than hidden behind a green result.
