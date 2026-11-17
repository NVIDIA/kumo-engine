# PyTorch nightly suite

Standalone repository tests for `kumo.pytorch`, outside its installed package.
Current implementation: AML acquisition, verified source/anchor preparation,
[model consumers](models/README.md), task checks and bounded HI-small training
with persistence, inference, frozen-input finetuning and fresh-process resume
checks, plus two controlled learning checks on tiny fixtures.
The [general benchmark interface](docs/benchmarking.md) now supports independent
dataset/backend/model adapters, complete-epoch timing and readable/structured
reports, AML bindings, fresh-process fitting comparisons and explicit Linux
file-cache experiments. Measurement coverage remains incomplete.
Full-population coverage remains pending; see [the plan](PLAN.md).
The [population acceptance runner](docs/population.md) now provides smoke and
full modes for all five AML combinations. Fixture validation does not establish
full HI-small acceptance; required graph cases currently expose decimal failures.
The [multi-source history follow-up](PLAN.md#pr-353-follow-up-multi-source-histories)
adds required filter/cutoff checks and a bounded real-AML run. Unsupported
required behavior fails normally; current row-filter rejection is not a skip
or expected success.

The existing nightly workflow runs the package's regression tests in place,
alongside its conformance and retail lifecycle jobs and these AML additions.
Generic package behavior stays in the existing tests; this directory owns AML
source, anchor, task and model integration plus the general benchmark runner.
Linux memory/cache and GPU collector validations have dedicated nightly jobs.
The workflow also configures all five bounded AML CUDA benchmark campaigns on
Engine's existing GPU runner, with per-worker cgroups and fresh/reused fitting.
CI verified bucket access and executed all five campaigns. With the decimal
fix deliberately removed, graph workloads fail. All three sequence campaigns
now pass measurement coverage with longer GPU windows and explicitly labelled
sampled host peaks. See the [latest benchmark validation](PLAN.md#gpu-and-sampled-memory-validation),
[collective CI review](PLAN.md#collective-ci-review) and
[benchmark execution and reports](docs/benchmarking.md#suite-execution-and-acceptance).
See [coverage ownership](PLAN.md#coverage-ownership).
The [coverage audit](docs/coverage-audit.md) maps the remaining requirements to
existing tests and identifies the next AML additions.
Its [PDF reconciliation](docs/coverage-audit.md#pdf-reconciliation) distinguishes
validated prototype coverage from the outstanding full nightly requirements.

The [real-AML training acceptance job](docs/training-acceptance.md) adds
100-update CPU checks for a001/Fusion and a017/PRAGMA, with repeatability,
per-parameter updates and retained loss series. Missing behavior fails normally.

The [fitting-memory experiment](docs/fitting-memory.md) adds fresh-process
25k/400k comparisons for all five AML contracts, using the general cgroup
measurement runner. Nightly reports retain all repetitions and enforce the
configurable proposed growth bound of less than 2x. Real-data validation is pending.

## Data

Use Python 3.12+ and `boto3>=1.34,<2`. From the repository root:

```sh
python test/kumo_pytorch/data.py list
python test/kumo_pytorch/data.py fetch --dataset hi-small \
  --report test/kumo_pytorch/.runs/hi-small-download.json
python test/kumo_pytorch/data.py verify --dataset hi-small
```

The [inventory](datasets/inputs.json) pins sources, identity bindings and both
anchor versions for six AML v8 variants. Only the selected variant downloads.
Files come from the Arena S3 bucket; no Arena code or checkout is needed.
Use `--profile NAME` for an existing AWS profile, or the normal credential chain.
`--source-dir DIR` reads a local mirror instead. See [AWS access](datasets/ACCESS.md).

Payloads retain their S3 keys under this directory's ignored `.data/` folder;
`--data-dir` can select another disk. Reports live in ignored `.runs/`. Bulk
payloads, indexes and checkpoints do not belong in Git or package artifacts.
AML patterns and laundering flags are labels/evaluation data, not input features.

The downloader verifies size and SHA-256 before publishing a file and before
reusing a cached copy. Corrupt existing files fail without replacement. Partial
files are removed on handled transfer failures; verified files remain reusable.
No command writes to S3. Reports record transfer/hash times, bytes, cache hits,
failures and wall time. Four workers run by default (`--jobs`); per-file time
sums overlap. These measure acquisition, not model or dataloader performance.

## Tests and workload selections

For the downloaded dataset, follow the [real AML prototype guide](docs/aml-prototype.md).
The runner prepares complete histories for selected accounts and exercises all
five workload/model combinations with training, validation and test anchors.

With `pytest>=8,<10`, run:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/checks -q
```

These checks need no AWS or native runtime; `--confcutdir` prevents loading the
parent service fixtures. The existing PyTorch nightly workflow runs them, and
its PR trigger includes this directory. Passing tooling tests is not AML runtime
acceptance.

With the repository's PyTorch test environment and a DiskGraph build matching
this checkout, run the model checks separately:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/model_checks -q
```

The existing nightly checks job runs this command after the shared PyTorch setup.
It checks all five AML workload/model combinations on tiny emitted batches,
including raw-file preparation, training updates, masking and v2 graph clocks. It downloads no data
and does not count as full AML, GPU or performance validation. The raw-file
checks also verify persistence, inference, frozen-input finetuning and optimizer resume.
Combined trains for 50 steps and PRAGMA for 100 on fixed tiny batches, with
explicit learning gates. The nightly retains their JSON reports.
The [history sweeps](docs/aml-prototype.md#history-coverage-sweeps) check
16/128/1,024-event inputs, explicit caps/windows and merged physical tables.
The [batching sweeps](docs/aml-prototype.md#batching-coverage-sweeps) check
32/128/512 samples, partial tails and selected token budgets in 23 cases.
The [graph sweeps](docs/aml-prototype.md#graph-coverage-sweeps) add 36 package
cases for fan-outs 2/32/128 and one-to-three-hop degree profiles, plus 12 AML
GraphSAGE/combined input cases. Edge-attribute coverage remains unresolved.
The [feature/encoding sweeps](docs/aml-prototype.md#feature-and-encoding-coverage-sweeps)
add 24 package cases for 8/32/128 output columns, 16/256/4,096 categories per
source field and 16/128/1,024-token texts, reusing existing fixtures.
The [edge-case checks](docs/aml-prototype.md#empty-histories-and-wide-source-rows)
add 20 AML empty/mixed-history training cases and 12 physical wide-row cases.
They caught a copied nuFormer empty-history output mismatch, now fixed locally.
The [source/anchor matrix](docs/aml-prototype.md#independent-source-and-fitting-anchor-sweeps)
adds 18 cases crossing 64/1,024/16,384 event rows with 8/64/512 fitting anchors.
Larger-than-RAM and performance validation remain pending.

Local prototype validation on base `08577147` with the decimal transport fix:
352 AML suite checks passed, including all three tasks, lifecycle, controlled
learning, 60 history, 23 batching, 12 graph and 20 empty-history cases; no failures
or skips. All 12 physical wide-row cases passed separately. Earlier runs passed
36 package graph sweeps, 24 feature/encoding cases and 63 focused engine checks.
The subsequent 18 source/anchor cases passed, zero failures/skips; unchanged AML
checks were not rerun in that increment.
All five bounded real AML cases previously passed
source/input, target/loss and lifecycle checks through finetuning and resume.
See the [involvement limits](docs/aml-prototype.md#account-involvement-checks),
[PRAGMA scope](docs/aml-prototype.md#pragma-masked-value-checks) and
[next-token scope](docs/aml-prototype.md#next-token-checks) and
[inference scope](docs/aml-prototype.md#saveload-and-inference) and
[finetuning/resume scope](docs/aml-prototype.md#finetuning-and-resume) and
[controlled learning scope](docs/aml-prototype.md#controlled-learning-checks).
DiskGraph was built from that base. The checks used
Python 3.12.14, torch 2.13.0 and torch-geometric 2.8.1 on CPU. GitHub Actions
execution has not yet been verified.

[Workload selections](workloads/aml.json) retain split/target meanings and mark
bounded prototype implementation. [Anchor selection](anchors/README.md) uses v2 for GraphSAGE
and combined, v1 for sequence-only models. Do not change labels or splits to
make a check pass.

Dataset records were copied from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`; the inventory retains provenance.
[aml-source.json](datasets/aml-source.json) retains upstream identities, raw
checksums and CDLA-Sharing-1.0 licensing. Historical records are not current
execution results.
