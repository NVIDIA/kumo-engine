# PyTorch full testing suite

Standalone repository tests for `kumo.pytorch`, outside its installed package.
Current bindings cover AML HI Small, AML LI Medium, Synthetic Medium S01/S02/S04 and MBD M04
through shared acquisition, verified source/anchor preparation,
[model consumers](models/README.md), task checks, training and lifecycle runners.
Fixture validation and real-data acceptance are tracked separately in the
[binding expansion plan](docs/binding-coverage.md).
The [general benchmark interface](docs/benchmarking.md) now supports independent
dataset/backend/model adapters, complete-epoch timing and readable/structured
reports, AML bindings, fresh-process fitting comparisons and explicit Linux
file-cache experiments. Measurement coverage remains incomplete.
Full-population coverage remains pending; see [the plan](PLAN.md).
The [population acceptance runner](docs/population.md) now provides smoke and
full modes for all five AML combinations. Fixture validation does not establish
full HI-small acceptance. The upstream decimal fix is integrated; remaining
capability refusals and execution failures are tracked in the plan.
The [multi-source history follow-up](PLAN.md#pr-353-follow-up-multi-source-histories)
adds required filter/cutoff checks and a bounded real-AML run. Under the
[acceptance policy](docs/required-coverage.md#capability-refusals), an explicit
declared refusal before execution passes as `refused`. Accepted capabilities
must still produce the expected answers; execution errors remain failures.

**PyTorch — Full Suite** uses one execution plan for direct PR events, main
pushes, scheduled runs and manual runs. The final `kumo-pytorch-gate` accounts
for every required group. Package, example and H&M workflows are reusable workers;
they no longer launch overlapping runs independently. Unrelated PRs retain an
explicit changed-file exemption; relevant PRs run the full suite.

Actions logs show step status, elapsed time, test totals and concise failure
reasons. Long steps emit a progress line every 30 seconds. Each job summary links
to its `kumo-pytorch-diagnostics-*` artifact containing full stdout/stderr,
tracebacks and copied JSON/JUnit reports; original campaign artifacts remain.
Summaries show up to 12 failure details, with the remaining count stated.
Skipped tests and accepted capability refusals are reported separately. Reporting
preserves command exit codes and does not change coverage or acceptance rules.
Third-party action failures (for example checkout or runner setup) retain their
own Actions logs; a job that never starts cannot publish a summary.

Direct PR access to the RTX runner remains an **open infrastructure prerequisite**.
The documented runner currently rejects `pull_request` jobs. The user chose to
keep that trigger and resolve access separately. GPU checks stay required; a
blocked, skipped or unavailable job cannot produce a passing full gate. No
runner permissions, IAM policies or required-check settings changed here.

Generic package behavior stays in its existing tests; this directory owns
source, anchor, task and model integration plus the general benchmark runner.
Timing/profiling campaigns now run only for LI Medium / Combined. HI Small's
five workload combinations retain their fitting-memory, correctness, training,
population and lifecycle checks; their model profiling, index-reuse timing and
DiskGraph/Polars/GPU loader-timing campaigns are disabled. Earlier HI Small
campaigns verified bucket access and exposed the decimal defect; upstream fix
#348 is now integrated. Historical results do not establish current acceptance.
See the [historical benchmark validation](PLAN.md#gpu-and-sampled-memory-validation),
[collective CI review](PLAN.md#collective-ci-review) and
[benchmark execution and reports](docs/benchmarking.md#suite-execution-and-acceptance).
See [coverage ownership](PLAN.md#coverage-ownership) and
[required CI coverage](docs/required-coverage.md).
The [Arena golden audit](docs/arena-golden-audit.md) maps existing assertion
owners and records selective boundary-test ports without an Arena dependency.
The [binding expansion plan](docs/binding-coverage.md) tracks the approved LI
Medium, Synthetic Medium and MBD additions. S02 adds customer-history next-token
checks; S01 adds account-history masked reconstruction with BPE; S04 adds Combined
LTV prediction with graph, history and grouped aggregates. M04 adds graph-based
transaction-count prediction with an independently checked generated count. All use original
source rows for feature assertions. Their campaigns use the existing RTX runner
with CPU training and a recorded storage preflight; full real-data acceptance
is pending. The remaining selected bindings are added one at a time.
The [coverage audit](docs/coverage-audit.md) maps the remaining requirements to
existing tests and identifies the next AML additions.
Its [PDF reconciliation](docs/coverage-audit.md#pdf-reconciliation) distinguishes
validated prototype coverage from the outstanding full nightly requirements.

The [real-AML training acceptance job](docs/training-acceptance.md) adds
100-update CPU checks for all five AML workload/model combinations: GraphSAGE,
nuFormer and Fusion involvement, PRAGMA masked-value pretraining and nuFormer
next-token pretraining. Each repeats twice, with per-parameter updates, loss
decrease and retained loss series. Missing behavior fails normally.

The [fitting-memory experiment](docs/fitting-memory.md) adds fresh-process
25k/400k comparisons for all five AML contracts, using the general cgroup
measurement runner. Nightly reports retain all repetitions and growth ratios
without a pass/fail growth limit; invalid or incomplete measurements still fail.

The [parallel nightly layout](docs/benchmarking.md#parallel-nightly-execution)
separates independent campaigns, fitting repetitions and population bindings.
It targets roughly 20 minutes without removing coverage or cutting off valid runs;
actual duration remains subject to CI validation and runner availability.

Development is consolidated in [PR #353](https://github.com/NVIDIA/kumo-engine/pull/353),
including the lifecycle import guards, fresh-process special-ID cases and fitting
chunk-boundary cases from #361/#362/#363. See the [current requirements status](PLAN.md#12-requirements-status-after-testing-pr-consolidation).

## Organization and adding bindings

| Group | Owner | Unit of execution |
| --- | --- | --- |
| Setup | `nightly plan`, shared DiskGraph wheel action | One plan and candidate wheel |
| Shared checks | Existing package, conformance, example and fixture tests | One owner per check; three conformance shards |
| Correctness | `nightly run`, source feature/history checks | Binding and declared smoke/history campaign |
| Training | Shared training acceptance runner | Binding, retaining both 100-update repetitions |
| Lifecycle | Shared binding runner and saved H&M runs | Reload, inference, serving, finetuning and resume |
| Population | Native AnchorTable population runner | Complete binding population |
| Benchmarks | Existing benchmark and fitting-memory runners | Binding, experiment and repetition |
| Results | `nightly summary`, `automation record` | Binding overview and complete job gate |

The [binding catalog](workloads/bindings.json) owns dataset definitions, anchor
and input profiles, task targets, models and enabled campaigns. Dataset display
names live with those definitions. Add one binding and its independently
calculated feature assertions, then explicitly enable its campaigns. Reuse
existing preparation, projection, model heads and execution functions; add a
specialized component only for a new capability. A new binding normally needs
no workflow or separate dataset runner.

The planner partitions the 40 binding campaigns into correctness (12),
training (11), lifecycle (6) and full population (11). The Lifecycle matrix also
contains one grouped HI Small entry, preserving its original five model/workload
checks. HI Small's source/target oracles and history sweeps run in separate
Correctness matrix entries; the source/target pass makes no optimizer updates.
H&M Small/Medium train and save their runs once, then reuse those artifacts in
six checks jobs: Correctness, Training/evaluation and Lifecycle for each scale. Only L01 / Combined is enabled for AML Medium; L02/L03/L04 real-data
campaigns are disabled while their fixture tests remain. The plan
also retains 15 HI Small fitting repetitions and four L01 benchmark campaigns.
The same resolved seed range feeds all conformance shards. Every test has one
owner; a seed's properties stay together, and each worker retains its required
coverage guard. The default unsharded local conformance command is unchanged.

```sh
python -m test.kumo_pytorch.nightly plan --seed-start 240 --output /tmp/suite-plan.json
```

`kumo-pytorch-campaign-results/report.md` groups results by dataset and binding,
then links individual experiments. Refused capabilities remain distinguished
from executed passes. Disabled campaigns are labelled **Not enabled**; missing
required evidence fails. The final gate publishes all required job outcomes.

Correctness reports now record explicit source verification/read/conversion,
anchor projection, ingestion, materialization, population setup/planning,
batch retrieval and assertion times. Training/lifecycle/history stages retain
separate elapsed times. Spans do not overlap; campaign wall time also includes
uninstrumented work. First-batch latency is a subset of retrieval time and is
not additive. Interrupted long stages retain `active_timing_stage`. These are
diagnostics, separate from the detailed benchmark metric contract.

Thread counts, batch sizes, populations, repetitions and cold/fresh-fitting
conditions are unchanged. Speed improvements here remove duplicate execution
and partition conformance. Resource tuning and compatible prepared-data reuse
require measurements from the new stage reports; a 20-minute total is not yet
established.

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
It checks all five AML workload/model combinations on tiny DiskGraph-emitted batches,
including raw-file preparation, training updates, masking and v2 graph clocks. It downloads no data
and does not count as full AML, GPU or performance validation. The raw-file
checks also verify persistence, inference, frozen-input finetuning and optimizer resume.
Combined trains for 50 steps and PRAGMA for 100 on fixed tiny batches, with
explicit learning gates. The nightly retains their JSON reports.
The [history sweeps](docs/aml-prototype.md#history-coverage-sweeps) check
16/128/1,024-event inputs, explicit caps/windows and merged physical tables.
The [batching sweeps](docs/aml-prototype.md#batching-coverage-sweeps) check
32/128/512 samples, partial tails and selected token budgets in 23 cases.
The [graph sweeps](docs/aml-prototype.md#graph-coverage-sweeps) add 18 DiskGraph package
cases for fan-outs 2/32/128 and one-to-three-hop degree profiles, plus 12 AML
GraphSAGE/combined input cases. Edge-attribute coverage remains unresolved.
The [feature/encoding sweeps](docs/aml-prototype.md#feature-and-encoding-coverage-sweeps)
add 12 DiskGraph package cases for 8/32/128 output columns, 16/256/4,096 categories per
source field and 16/128/1,024-token texts, reusing existing fixtures.
The [edge-case checks](docs/aml-prototype.md#empty-histories-and-wide-source-rows)
add 10 AML empty/mixed-history training cases and 6 physical wide-row cases.
They caught a copied nuFormer empty-history output mismatch, now fixed locally.
The [source/anchor matrix](docs/aml-prototype.md#independent-source-and-fitting-anchor-sweeps)
adds 9 DiskGraph cases crossing 64/1,024/16,384 event rows with 8/64/512 fitting anchors.
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

[Workload selections](workloads/bindings.json) retain split/target meanings and mark
bounded prototype implementation. [Anchor selection](anchors/README.md) uses v2 for GraphSAGE
and combined, v1 for sequence-only models. Do not change labels or splits to
make a check pass.

Dataset records were copied from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`; the inventory retains provenance.
[aml-source.json](datasets/aml-source.json) retains upstream identities, raw
checksums and CDLA-Sharing-1.0 licensing. Historical records are not current
execution results.

### Dataset bindings

The exact pinned-object grant is recorded in [datasets/ACCESS.md](datasets/ACCESS.md).
HI Small and LI Medium use the same catalog and `nightly run` command.
For example:

```bash
python -m test.kumo_pytorch.data fetch --dataset li-medium --data-dir /data/aml
python -m test.kumo_pytorch.nightly run --binding L01 --mode smoke \
  --data-dir /data/aml --output /results/smoke/L01 --batch-size 32
python -m test.kumo_pytorch.nightly run --binding L01 --mode training \
  --data-dir /data/aml --output /results/training/L01 --batch-size 32
python -m test.kumo_pytorch.nightly run --binding L01 --mode full \
  --data-dir /data/aml --output /results/full/L01 --batch-size 32
```

Training runs two deterministic 100-update CPU repetitions on 32 anchors per
split. L01 smoke checks four anchors per split against the source feature/target
oracles. Full mode checks every forecast-split anchor and rejects limits.
L01's reports identify graph, history, aggregate and target assertions separately,
with checked-anchor counts by split and execution phase. Expected answers come
from original source rows and declared semantics; missing evidence fails summary
validation. See the [feature-to-check map](docs/binding-coverage.md#l01-feature-coverage)
for scope and references.
The `kumo-pytorch-campaign-results` artifact and combined job summary show all
26 correctness/training/lifecycle/history outcomes plus nine benchmark/fitting reports;
per-job artifacts retain partial results and child logs on failure.
See [binding coverage](docs/binding-coverage.md) for heads, loss definitions and
enabled L01 campaigns and remaining package gaps.


L01 additionally runs `--mode lifecycle` and `--mode histories` on four anchors
per split. The benchmark matrix enables detailed training, index build/two reopens,
DiskGraph-only loader timing, and a separate native length-planning/cache probe.
Polars/GPU reference timing and 25k/400k fitting remain disabled for L01.
The same history, graph, batching and empty-history fixture sweeps now exercise
L01's amount/currency task. Required failures remain visible; enabling a campaign
does not establish that the package supports every case.
