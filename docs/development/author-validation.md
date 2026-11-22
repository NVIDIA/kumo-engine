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
The [eight-task GPU CI](engine-pr-ci.md) reuses the generated S3 fixture listed
below. Download it once using authorized AWS access, then reuse it locally.

## Existing generated datasets: download once, reuse

The data is already generated in S3. Download the required fixture or rung to
local SSD once, then use that same cache for baseline and candidate. Keep its
manifest, validation stamp and evaluation files. Do not regenerate inputs
between variants; use separate scratch/index and output locations.

AWS access below uses your configured credentials with read permission for the
selected objects. These are data downloads, not provisioning commands.

### Engine CI: the small eight-task fixture

The [CI action](../../.github/actions/engine-gpu-smoke/action.yml) downloads this
**`pr172-v1`** archive, pinned by
[`inputs.json`](../../packages/kumo-engine/test/pr172/inputs.json):

```text
s3://kumo-unit-test/diskgraph-profiles/engine-pr/pr172-fixtures/v1/a20a1e1412c963bf732e6323c3aeac3e96e4b2c253d25a3c45c7f5acf37d6c9a/pr172-inputs-v1.tar.gz
```

Region: `us-west-2`. Archive SHA-256:

```text
a20a1e1412c963bf732e6323c3aeac3e96e4b2c253d25a3c45c7f5acf37d6c9a
```

The [fixture contents](../../packages/kumo-engine/test/pr172/README.md) are:

| Extracted path | Data |
| --- | --- |
| `adtech/pr/` | 556,127 table rows |
| `delivery/pr/` | 1,442,968 table rows |
| `commerce/pr/` | 74,416 table rows |
| `graphs.json` | Graph/table/column/key/time/edge metadata used by the runner |

With a prepared Python 3.12+ environment containing `boto3`, download and verify
using the same manifest as CI. Choose a new cache directory; skip this block
when you already have the verified fixture:

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
import tarfile
from pathlib import Path

import boto3

spec = json.loads(Path('packages/kumo-engine/test/pr172/inputs.json').read_text())
root = Path('/local-ssd/engine-ci-inputs') / spec['version']
root.mkdir(parents=True, exist_ok=False)
archive = root / 'inputs.tar.gz'
boto3.client('s3', region_name=spec['region']).download_file(
    spec['bucket'], spec['key'], str(archive))
with archive.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
        raise SystemExit('CI fixture checksum differs; refusing extraction')
with tarfile.open(archive) as source:
    source.extractall(root, filter='data')
print(root)
PY
```

The existing [GPU runner](../../packages/kumo-engine/test/wheel_gpu_smoke.py)
accepts that extracted root as `--data-dir` and its `graphs.json` as
`--graph-metadata`. The CI action gives the installed-wheel, single-CUDA-GPU
invocation and `0.005` reload tolerance. Reuse these files across runs; no
replacement dataset or generator environment is needed. The local `readiness`
profile runs pure output/workflow checks, not these eight GPU tasks.

### Scale datasets: seven graphs, 100M through 50B

The existing generated scale data is stored at:

```text
s3://kumo-test-dataplane-bucket/scale-bench/datasets/<graph>/<rung>/
```

| Graph | Prefix under `s3://kumo-test-dataplane-bucket/scale-bench/datasets/` |
| --- | --- |
| Commerce | `commerce/` |
| Streaming | `streaming/` |
| Community | `community/` |
| Delivery | `delivery/` |
| Adtech | `adtech/` |
| Payments | `payments/` |
| Fraud | `fraud/` |

The documented rungs are `100M`, `1B`, `10B`, `25B`, and `50B`. For example:

```text
s3://kumo-test-dataplane-bucket/scale-bench/datasets/commerce/100M/
s3://kumo-test-dataplane-bucket/scale-bench/datasets/commerce/1B/
s3://kumo-test-dataplane-bucket/scale-bench/datasets/commerce/50B/
```

Each rung has table Parquet files, `manifest.json`, and `_VALIDATED.json`;
retain `_eval/` when present for the corresponding test seeds. Select a rung
that fits your SSD/RAM/time budget. The small CI archive above is a separate
fixture, not a 100M or 50B rung.

For example, download the existing commerce 100M rung into a new cache:

```bash
aws s3 sync \
  s3://kumo-test-dataplane-bucket/scale-bench/datasets/commerce/100M/ \
  /local-ssd/scale-bench/datasets/commerce/100M/ --only-show-errors
```

Before use, require the passing validation stamp to identify the downloaded
manifest, then verify table checksums/schemas/row counts using the existing
commerce verifier:

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

rung = Path('/local-ssd/scale-bench/datasets/commerce/100M')
manifest_sha = hashlib.sha256((rung / 'manifest.json').read_bytes()).hexdigest()
stamp = json.loads((rung / '_VALIDATED.json').read_text())
if stamp.get('ok') is not True or stamp.get('manifest_sha256') != manifest_sha:
    raise SystemExit('Missing passing validation for this exact manifest')
print('Validated manifest:', manifest_sha)
PY
.venv-generator/bin/python benchmark/commerce/generator/verify.py \
  --root /local-ssd/scale-bench/datasets --rung 100M
```

The verifier needs its [declared dependencies](../../benchmark/commerce/generator/requirements.txt);
it reads the existing files and does not generate data. The manifest identifies
the generator/configuration and records per-file hashes. Retain it and the
validation stamp in your experiment evidence. Do not update the cache with
another S3 sync during a comparison.

Ground-truth task labels, when needed by a task-specific evaluator, are separate:

```text
s3://kumo-test-dataplane-bucket/scale-bench/datasets/_ground_truth/<graph>/<rung>/labels/<task>/
```

Use them only for evaluation, never as model input. Keep the rung's original
evaluation population and prediction times when comparing published workloads.
The public placeholder generator can produce different bytes/distributions;
it is not a substitute for these existing measured inputs.

## Run on the cached scale data

Use the [context harness](../../packages/diskgraph/scripts/benchmarks/commerce/README.md)
to prepare seed IDs if absent, build a fresh index, and map IDs from the cached
commerce rung. For a bounded first sampling matrix, substitute your prepared
index and seed map:

```bash
.venv/bin/python packages/diskgraph/scripts/benchmarks/commerce/run_matrix.py \
  --rung 100M --index /local-ssd/index --seedmap /local-ssd/seedmap.parquet \
  --vcpus 4 --dg-threads 1 --hops 2,3 --scale 0.01 --cap-s 60 \
  --out /local-ssd/sampling-attempt-1.jsonl
```

This uses four processes with one native thread each. Record the configuration,
cache procedure, completed work and dataset identity. Size worker counts to RAM
and CPU availability. Use separate index/scratch/results per variant when
native formats differ. The harness owns its timers and incomplete-cell markers;
a small run does not establish the larger rung's performance. For training and
prediction budgets, use the [commerce workflow guide](../../benchmark/commerce/README.md).
Do not add `--drop-all` on a shared machine. The old continuous ingest service
can format/mount disks and update its checkout; it is absent from this catalog.

## Optional: generate a new smoke dataset

Only use generation for a new experiment or when no suitable existing dataset
is accessible. Prepare the
[generator environment](../../benchmark/commerce/generator/requirements.txt),
choose new local output paths, and run:

```bash
.venv-generator/bin/python scripts/author_checks.py run commerce-generate-smoke \
  --param data_root=/local-ssd/commerce-smoke \
  --disk-budget-gib 2 --reports /tmp/commerce-generate-1
.venv-generator/bin/python scripts/author_checks.py run commerce-verify-smoke \
  --param data_root=/local-ssd/commerce-smoke --reports /tmp/commerce-verify-1
```

The disk budget is a free-space preflight, not a quota or a fit estimate.
Generation and verification stay outside engine benchmark timers. Keep the
result for subsequent runs; do not regenerate it between variants.

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
