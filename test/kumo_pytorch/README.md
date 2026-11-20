# PyTorch full testing suite

Repository-owned tests for `kumo.pytorch`, outside the installed package.
The suite combines existing package tests with independent source/target checks,
model training, lifecycle, full-population delivery and DiskGraph benchmarks.
Arena supplies provenance and some pinned data, with no runtime code dependency.

[The roadmap](PLAN.md) records scope and outstanding acceptance;
[the coverage map](docs/coverage-audit.md) maps PDF requirements to assertion
owners. Neither a catalog entry nor a passing tiny fixture establishes real-data
acceptance. Review reports from the intended package revision.

## Organization and adding bindings

**PyTorch — Full Suite** uses one plan for relevant direct PR events, main pushes,
scheduled runs and manual runs. The final `kumo-pytorch-gate` accounts for every
required group. Package, example and H&M workflows are reusable workers, avoiding
independent duplicate runs. Unrelated PRs have an explicit changed-file exemption.

| Group | Owner | Execution unit |
| --- | --- | --- |
| Setup | `nightly plan`, shared DiskGraph build/setup actions | One plan and candidate wheel |
| Shared checks | Package, conformance, example and fixture tests | One assertion owner; three conformance shards |
| Difficult Behavior | [Adversarial cases](docs/difficult-behavior.md) | Required fixture group, sharing the candidate wheel and gate |
| Correctness | `nightly run`, independent source/feature checks | Binding and declared smoke/history campaign |
| Training | Shared [training acceptance](docs/training-acceptance.md) | Binding; two deterministic 100-update CPU repetitions |
| Lifecycle | Binding runner, HI Small downstream-task probes and saved H&M runs | Reload, inference, serving, frozen-input finetuning and resume |
| Population | [Native AnchorTable checks](docs/population.md) | Every eligible anchor in a binding |
| Benchmarks | [Profiler](docs/benchmarking.md), [fitting memory](docs/fitting-memory.md) | Binding, experiment and repetition |
| Results | `nightly summary`, `automation record` | Binding overview and complete job gate |

The [binding catalog](workloads/bindings.json) owns dataset definitions, anchors,
input profiles, tasks, models, display names and enabled campaigns. Add one
binding and its independent feature assertions, then enable campaigns explicitly.
Reuse acquisition, source preparation, projection, model heads and execution;
specialize only for a new capability. New bindings normally need no workflow.
See the [feature-to-assertion map](docs/binding-coverage.md).

The default plan contains 40 binding campaigns: 12 correctness, 11 training,
6 lifecycle and 11 full-population runs. Three additional grouped HI Small jobs
retain source/target/loss checks, history sweeps and downstream-task lifecycle
checks. The latter replaces pretraining heads and uses separate forecast anchors;
it is not interchangeable with same-task finetuning. These grouped jobs use
`nightly group --mode smoke|histories|lifecycle`, sharing input verification and
source conversion with the binding runner. H&M Small/Medium train/save
once and reuse artifacts across their correctness, training and lifecycle checks.
Only L01 Combined runs for AML LI Medium; L02/L03/L04 fixture tests remain.

Timing campaigns run only for L01: training profiling, index build/two reopens,
DiskGraph loader timing and native length-planning/cache reuse. HI Small keeps
15 fitting-memory repetitions (five bindings, three repetitions) but no timing
campaigns. CUDA model training and GPU collectors remain; Polars/GPU reference
engines are removed. Polars still prepares Streaming 25B anchors.

```sh
python -m test.kumo_pytorch.nightly plan --seed-start 240 --output /tmp/suite-plan.json
```

Streaming 25B / T01 GraphSAGE real-data campaigns are **on demand only**, including
for manual full-suite runs. Its package fixture checks remain required.
Use **PyTorch — On-Demand Bindings** with `all`, `smoke`, `training`, `lifecycle`
or `full`. It reuses the catalog, wheel, worker and reports. The declared runner
and storage preflight still apply.

```sh
gh workflow run kumo-pytorch-on-demand.yml --ref YOUR_BRANCH -f campaign=all
python -m test.kumo_pytorch.nightly matrix --on-demand --campaign all
```

Direct PR access to the RTX runner is an open infrastructure prerequisite. The
user chose to keep direct PR events and resolve admission separately; unavailable
GPU jobs cannot yield a passing gate. Required-check activation and post-merge
schedule verification are tracked in [automation](docs/required-coverage.md).
The approximately 20-minute runtime goal does not permit reducing populations,
repetitions, fitting/cache conditions or assertions to make runs finish sooner.

## Results and acceptance

The `kumo-pytorch-campaign-results` artifact groups outcomes by dataset/binding,
then links individual reports. Per-job `kumo-pytorch-dataset-<mode>-<binding>`
artifacts retain contracts, reports and child diagnostics. Missing required
evidence fails; disabled campaigns say **Not enabled**. Grouped HI Small reports
retain separate artifacts; their jobs also contribute to the full-suite gate.

Actions logs show elapsed time, totals and concise failure explanations. Test
purpose, expected outcome, parameter variant and observed mismatch appear in
summaries; exact test IDs/evidence are collapsible. Full stdout/stderr, assertions,
tracebacks and JSON/JUnit stay in `kumo-pytorch-diagnostics-*` artifacts. Long
steps emit progress every 30 seconds. Summaries show up to 12 failure details and
state the remainder. Reporting preserves command exit codes.

Use a test docstring's first paragraph for its plain-language purpose, followed
by `Expected:` and optional `Evidence:` paragraphs. The reporter reads source
without importing tests. It does not invent expectations or infer root causes.
Third-party action failures retain their own logs; a job that never starts
cannot publish a summary.

[Required coverage](docs/required-coverage.md) distinguishes executed passes,
accepted explicit preflight refusals, failures and skips. Wrong results, late
errors, silent fallbacks, missing cases and unexpected skips remain failures.
Expected answers come from declared semantics and independent source calculations,
not another engine. Existing internal serving dependencies are allowed.

Campaign stage timings cover verification, conversion, projection, ingestion,
materialization, planning, retrieval and assertions, with separate training,
lifecycle and history spans. Nonoverlapping spans exclude uninstrumented work;
first-batch latency is part of retrieval, not additive. Interrupted stages remain
visible. These diagnostics are separate from the [benchmark metric contract](PLAN.md#complete-metric-checklist).

## Data

Use Python 3.12+ and `boto3>=1.34,<2`. From the repository root:

```sh
python test/kumo_pytorch/data.py list
python test/kumo_pytorch/data.py fetch --dataset hi-small \
  --report test/kumo_pytorch/.runs/hi-small-download.json
python test/kumo_pytorch/data.py verify --dataset hi-small
```

The [dataset catalog](datasets/inputs.json) pins one S3 inventory by SHA-256.
That inventory retains every source, identity and anchor object's size and hash.
Fetching a dataset first downloads the small inventory into the same data directory;
verification reuses it offline. `data list` reads only the local catalog. Local
mirrors must include the inventory at its pinned key. Keep download reports outside
the input directories.
Only the selected dataset is downloaded. Use `--profile NAME` for an existing
AWS profile or the normal
credential chain; `--source-dir DIR` reads a local mirror. See [AWS access](datasets/ACCESS.md).

Payloads retain S3 keys under ignored `.data/`; `--data-dir` selects another disk.
Reports live in ignored `.runs/`. Bulk data, indexes and checkpoints stay out of
Git and package artifacts. The downloader verifies size/SHA-256 before publishing
or reusing a file; corrupt files fail without replacement. Handled failures remove
partial files, retaining verified downloads. It performs no S3 writes. Acquisition
reports include transfer/hash time, bytes, cache hits and failures. Four workers
run by default (`--jobs`); per-file elapsed times overlap.

Dataset provenance and licenses remain in the inventory and
[aml-source.json](datasets/aml-source.json). AML laundering flags/patterns are
labels or evaluation data, not input features. [Anchor rules](anchors/README.md)
retain HI Small v2 clocks for GraphSAGE/Combined and v1 for sequence-only models.

## Tests and workload selections

With the repository's PyTorch environment and a matching DiskGraph build:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/model_checks -q
```

`--confcutdir` avoids parent service fixtures. Tiny fixtures exercise source
mapping, targets/losses, training, lifecycle and boundary sweeps without downloads.
The [HI Small task guide](docs/aml-prototype.md) describes its independent oracles
and sweep dimensions; [model provenance](models/README.md) identifies copied
consumers. Existing package tests retain generic semantics. The [Arena golden
audit](docs/arena-golden-audit.md) records selective ports without duplicate owners.

### Dataset bindings

Training and population use the same `nightly run` entry point for every binding;
the former AML-only training/population CLIs have been removed. For example:

```sh
python -m test.kumo_pytorch.data fetch --dataset li-medium --data-dir /data/inputs
python -m test.kumo_pytorch.nightly run --binding L01 --mode smoke \
  --data-dir /data/inputs --output /results/smoke/L01 --batch-size 32
python -m test.kumo_pytorch.nightly run --binding L01 --mode training \
  --data-dir /data/inputs --output /results/training/L01 --batch-size 32
python -m test.kumo_pytorch.nightly run --binding L01 --mode full \
  --data-dir /data/inputs --output /results/full/L01 --batch-size 32
```

Use a new output directory. Training selects 32 anchors per split by default;
smoke selects four. Full mode accepts no anchor limit. Native AnchorTable owns
populations, planning, shuffling and iteration. Reports identify each feature
assertion and checked-anchor counts by split/phase. L01 additionally enables
`lifecycle` and `histories`; the catalog declares modes for other bindings.
