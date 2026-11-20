# Required dataset/model coverage

CI uses the reviewed node IDs in `../required-tests.json`: 668 CPU cases and
82 separate difficult-behavior cases. The Polars and GPU reference backends and
their 11 required cases have been removed. The separate GPU job retains native
CUDA package regressions and GPU collector validation, with skips rejected.

The LI Medium increments add 43 CPU checks for native target delivery, train-only
fitting, masks, invalid source/binding rejection, stored-target heads, strict
training replay and source-to-runner execution. They use tiny fixtures; real
LI Medium campaign execution remains pending in the
[binding expansion plan](binding-coverage.md).

`--required-coverage=cpu` or
`--required-coverage=difficult` requires the corresponding inventory. The last
profile belongs to the required [Difficult Behavior group](difficult-behavior.md).
Missing/deleted cases, unexpected additions, collection-time skips,
missing imports, deselection of required cases, runtime skips and xfails fail
execution. A required case must execute or verify a declared preflight refusal
as described below. Other setup failures remain failures. Expected output
assertions are unchanged. Local focused runs may omit the option.

S02 adds 28 required native fixture checks for source ownership, mixed-event
encoding, chronological ties, corruption detection, training and lifecycle. Its
real-data campaigns use the same required-report accounting as AML.

S01 adds 42 required account-history/BPE fixture checks, including ownership,
complete piece sequences, corruption/masking, fitting cutoffs and all four shared
campaign paths. S01 is one extended binding, not separate base/BPE campaigns.

S04 adds 38 required fixture cases for its four campaigns, source-derived graph,
history, aggregate and threshold-label checks, corruption detection and window
boundaries. One unrelated multi-table graph training case verifies reuse of the
same model/runner. Its checkpoint-resume failure remains a required failure.

M04 adds 30 required graph/count forecast cases covering original-row windows,
identities, integer precision, graph semantics, corruption and all four campaigns.
Its existing package checkpoint-resume failure remains a required failure.

## Capability refusals

Per the user's 2026-11-17 instruction, a clean unsupported-capability refusal
before execution passes. It must come from public `prepare()` preflight and
contain only engine-declared `check()` refusal codes, with paths, reasons and
suggestions. Generic exceptions, invalid contracts, crashes, chained failures,
wrong results, silent fallback and errors after execution starts still fail.
Tests retain their independent expected answers for accepted capabilities.

The user also explicitly accepts DiskGraph's intentional rejection of zero-row
base tables. The harness recognizes only its exact native `DiskgraphSystemError`
from base-index ingestion through `prepare()` and `DiskGraphEngine.open()`,
with the rejected table present in the dataset/schema and its bound Parquet
metadata independently confirming zero source rows. Reports label this
`stage=ingestion`, `execution=stopped_before_materialization` and the test-owned
code `DG-EMPTY-BASE-TABLE`; it is not presented as a public preflight check.
Fabricated messages, other ingestion errors and later iteration errors fail.

Pytest reports accepted refusals in the terminal summary and the
`capability_refusals` JUnit property. The AML, history, training and population
CLI reports label them `status=passed, outcome=refused` and retain the evidence.
The combined nightly summary preserves that label for refused population shards
without claiming full delivery; incomplete execution still fails.
A fixture's verified refusal can satisfy a case without calling its dependent
body; teardown errors still fail. Required inventory and skip enforcement remain
active, and generated/corpus loops continue checking subsequent cases.

Documentation subprocesses and notebook kernels classify refusals before their
exceptions lose the original traceback. Parent reports retain that evidence;
matching error text alone never qualifies. A refused notebook does not claim
generated-file reproduction, and regeneration commands still fail on refusals.
Earlier assertion/output failures remain failures. Negative-input tests check
native contract support before asserting a more specific rejection message.

A refusal pass proves rejection, not execution. It does not claim training
updates, completed populations or benchmark measurements. The benchmark metric
completeness requirements are unchanged. Late lifecycle refusals cannot turn an
already executing AML workload into a pass.

The expanded training-acceptance module contributes 22 required CPU cases:
two fixture paths for all five combinations, plus harness and model-regression
checks. All 22 pass after removing unused final non-root branches from the
standalone GraphSAGE test model; see [training acceptance](training-acceptance.md).

When intentionally adding or renaming cases, collect from the repository root:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/model_checks --collect-only -q
```

Review the collected IDs against `required-tests.json` and update the inventory
in the same change as the tests. Keep IDs starting with `model_checks/` (or `difficult_behavior/` for that
profile), sorted within each owner. Never remove a required case merely to pass enforcement.
Collection alone does not establish execution acceptance.

## DiskGraph-only scope

Per the user's 2026-11-17 instruction, 87 InMemory variants were removed from
the standalone inventory. The cutoff-alignment and fresh-process benchmark-state
checks now use DiskGraph instead of InMemory, retaining their assertions.
Polars/GPU timing-reference cases were subsequently removed at the user's request.
The user's later instruction
also migrates inherited package tests and runnable examples to DiskGraph while
retaining their required behavior assertions.

Local execution before expanding training acceptance from two combinations to
five (326 CPU cases): **325 passed, one failed, zero
skipped**, with five GPU cases explicitly delegated. The failure remains
`test_raw_aml_multisource_history[filters-and-windows]`: `DG-ROW-FILTER` rejects
the required operation. It remains a normal failure. The converted checks passed.
Separately, all **105** focused package checks passed, including the 29 new
DiskGraph golden boundary checks. These runs use the existing compatible local
native extension; branch-native CI validation remains separate.

## Native AnchorTable migration

Benchmarking, AML training/evaluation, history checks, inference, finetuning and
checkpoint continuation now materialize Parquet AnchorTables and iterate the
owned population without an explicit request argument. Bounded correctness
probes retain expected records only for independent assertions and table
projection. The benchmark reuses the streaming population projection directly.
Three added required cases check native-path rejection, cache-reset rejection,
and complete per-anchor length reuse across a partial final batch.

This migration exposes a Fusion fresh-process checkpoint failure: aggregate
materialization changes the internal anchor fingerprint on reload, so
`TaskBatches.load_state_dict()` rejects the same original population/index as a
different task. The two Fusion checkpoint cases and the combined raw-AML case
remain ordinary failures. `restore-fingerprints.json` records both identities;
the package's check is not bypassed. The pre-existing `DG-ROW-FILTER` failure
also remains. These are local fixture findings, not full AML/GPU CI results.

## GitHub merge enforcement

PR #353 owns `kumo-pytorch-gate` in the **PyTorch — Full Suite** workflow.
Direct PR, main, scheduled and manual executions use the same complete plan,
including Difficult Behavior, real-data training, full populations and GPU
benchmarks. Existing package/example/H&M workflows are reusable workers, not independent triggers.
A single candidate DiskGraph wheel feeds all native checks. Unrelated PRs pass
through explicit changed-file detection; API errors run the suite instead of
assuming an exemption. Missing jobs, failures, cancellations and unexpected
skips fail the gate.

The RTX runner's documented refusal of direct `pull_request` jobs remains an
open infrastructure issue. The user requested keeping direct PR events while
resolving runner access separately. GPU jobs remain required. No runner/IAM
configuration or branch protection is changed by the suite reorganization.

Required-check activation and post-merge schedule verification remain deferred.
The last permissions audit found maintain access without admin access and a
main ruleset requiring `kumo-core-gpu-ci`, not the new PyTorch gate. After the gate is merged
and has reported on both a relevant and an unrelated PR, an administrator can
create the main-only rule in [merge-gate-ruleset.json](merge-gate-ruleset.json):

```sh
gh api --method POST repos/NVIDIA/kumo-engine/rulesets \
  --input test/kumo_pytorch/docs/merge-gate-ruleset.json
```

This adds `kumo-pytorch-gate` from the GitHub Actions integration; it does not
replace existing reviews, code-owner requirements, checks or release-branch
rules. Re-read the active rules before applying it, and avoid adding a duplicate
if another administrator has already enabled the gate. Known package failures
must be resolved before the required gate can pass; do not exempt those tests.

## Scheduled-run verification after merge

The nightly's final `Results / Complete Nightly` job accounts for all required
jobs, even when earlier jobs fail. It writes `kumo-pytorch-nightly-status` with
the run ID, attempt, commit, trigger and every job result. Missing, skipped,
cancelled or failed required jobs fail this final check. Matrix campaign reports
continue to account for each binding and repetition.

Actual schedule verification remains **pending until after merge**. It cannot be
established by a manual run or unit tests. After the first scheduled main run,
use its numeric run ID in place of RUN_ID below:

```sh
gh api repos/NVIDIA/kumo-engine/actions/runs/RUN_ID > scheduled-run.json
gh run download RUN_ID --repo NVIDIA/kumo-engine \
  --name kumo-pytorch-nightly-status --dir scheduled-evidence
python -m test.kumo_pytorch.automation verify-schedule \
  --run scheduled-run.json --evidence scheduled-evidence/status.json
```

Run this from the merged checkout. Confirm the run's commit contains the merged
change (a later main commit is allowed), and inspect the test/training/benchmark
artifacts. The verifier rejects manual/PR runs, wrong branches/workflows,
incomplete runs and mismatched revision/run/attempt evidence. It distinguishes
an automatic trigger from test success: a failed test is reported as failed,
while missing or skipped required jobs leave execution verification incomplete.
Keep downloaded evidence outside Git. No scheduled-run acceptance has yet been
claimed for these changes.

Historical validation before the DiskGraph-only scope change: all 413 CPU
cases executed; 412 passed, one required multi-source filter case failed with
`DG-ROW-FILTER`, and none skipped. All six Polars cases passed. The five GPU
cases were explicitly delegated, not locally validated. Seven subprocess
fault-injection checks passed. These results use the existing compatible native
extension; CI must rebuild and validate the final native revision.


## Native conformance scope

Correctness / Conformance invokes `python -m conformance --engine diskgraph
--no-in-memory --seeds 12 --seed-start K`. All native behavior obligations remain
required. Aggregate answers, fitted statistics, graph expansion and sequence
selection use fixture-derived expectations, with no reference engine. The flag
excludes harness self-tests only; package/DiskGraph CI runs those on native
execution too. Required coverage still fails on missing, skipped or xfailed cases.

Package fixtures and examples use Parquet/DiskGraph. Runtime guards reject
InMemory as a test backend or reference oracle. Per the user's 2026-11-17 scope
decision, calls inside the package's existing `ServingAssembler` are allowed;
changing that implementation is outside this testing PR. Serving still executes
its assertions, and dependency errors remain failures. The exception ends when
the serving call returns or raises and also applies in doc/notebook subprocesses.
The migration is implemented, with independent expected answers and no new
skip/xfail/native-gap exemptions. Package acceptance is still failing on native
support and correctness gaps. See the [remaining acceptance work](../PLAN.md#remaining-acceptance-work);
historical validation records remain in Git history. Neither full nightly
acceptance nor a runtime improvement has been established for this revision.

The [native AnchorTable scope](../PLAN.md#agreed-scope) covers #353's AML checks and added/extended package tests. The benchmark's
required fault cases include a missing native history plan; first-epoch length
planning and second-epoch cache reuse remain enforced. This follow-up does not
claim repository-wide removal of request-based loading. Native package failures
remain required failures.


## Shared dataset checks

The required inventory includes 19 dataset-general checks: unrelated table/ID/clock
schemas through all four models and the shared profiler, all five existing HI
bindings through the common population runner, multi-family/alternate-join
history bounds, graph degrees and incompatible native objective/model pairs.
The current reviewed inventory is listed at the top of this document.
Dataset registration alone does not add execution coverage; enabled campaigns
and required test IDs remain explicit.
