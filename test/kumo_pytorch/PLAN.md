# Nightly prototype implementation plan

The supplied testing strategy specifies nightly coverage. Weekly and on-demand
suite development are outside this task. Implement a simple working prototype
incrementally; optimize runtime and calibrate performance gates afterwards.

The [PDF reconciliation](docs/coverage-audit.md#pdf-reconciliation) maps all six
document sections to code and actual evidence, including user-directed deferrals.
The bounded correctness prototype passed before the decimal fix was withdrawn.
The [collective CI review](#collective-ci-review) records the current failures
and measurement gaps; the complete nightly specification is not yet satisfied.

## Agreed scope

- Keep all new test models, workload definitions, fixtures, dataset records and
  tooling under `test/kumo_pytorch/`, outside the installed package.
- Keep bulk sources and anchors inside the suite's ignored `.data/` directory.
  Fetch exact files from the Arena S3 bucket; no Arena Python or checkout
  dependency. User authorization explicitly permits this shared storage.
- Start workload execution with AML HI-small v8/account-daily anchors: v2 for
  GraphSAGE and combined, v1 for sequence-only models. Preserve
  configuration support for the other AML variants.
- Only AML is in the active prototype scope. Synthetic and MBD requirements
  remain recorded in the original roadmap but receive no current implementation
  or acquisition work. The user cancelled the synthetic download.
- Use the existing PyTorch nightly workflow and build/setup conventions. Keep
  existing retail lifecycle and conformance coverage.
- Run existing package regression tests in place in that workflow. Reuse their
  coverage rather than copying or reimplementing it under this directory.
- The Polars comparison was deferred during correctness work. The user's
  subsequent requirement for every benchmark measurement now includes it in
  the revised benchmarking plan below. Match workload semantics without
  requiring identical batches.

## Coverage ownership

The nightly combines existing package tests, existing conformance and retail
lifecycle jobs, and AML workload checks. Each test has one execution owner in
the nightly; the existing PR/main package workflow remains independent.

| Owner | Existing coverage to retain | AML addition |
| --- | --- | --- |
| Package regressions: `packages/experimental/kumo-pytorch/tests/` | Generic time boundaries, target layouts, masking, padding, encoders, persistence and API regressions | No duplicate generic cases; package defects such as decimal transport get regressions here |
| Existing nightly conformance job | Goldens, generated properties, dtype coverage and backend comparisons on both engines | AML source/task expectations only; do not recreate the conformance harness |
| Existing nightly retail lifecycle jobs | Training, persistence, fresh-process inference, serving, finetuning and leakage checks | AML persistence, inference and frozen-input finetuning use the same artifact/weight APIs, followed by optimizer resume; no copied retail checks or thresholds |
| `test/kumo_pytorch/` | AML acquisition, source mapping, anchors and copied model consumers | Next-24-hour involvement labels, eligible windows, v1/v2 clock translation, sender/receiver/self-transfer histories and loss consumption across the three AML models |

The package-regressions job excludes `tests/conformance/` and
`tests/e2e/test_lifecycle.py`, which their existing nightly jobs execute.
Notebook execution retains its existing separate workflow. Shared setup and the
same commit's DiskGraph wheel serve all native jobs. Downloaded AML execution
now runs in CI; see the [collective review](#collective-ci-review) for evidence
and remaining runner requirements.

Similar assertions can cover distinct integration boundaries: a generic time
window test does not prove that AML chose the correct clock and input fields.
Keep that AML contract coverage, but consolidate repeated assertions for the
same model/task. The earlier supervised loss smoke checks have been replaced by
passing independent involvement checks, including zero-eligible backward coverage.
PRAGMA's earlier loss smoke checks are likewise replaced by independent source,
target-selection and smoothed-loss checks. Generic masking algorithms stay in
the existing package tests.
The next-token loss smoke checks are now replaced too; generic shifted-target
layout coverage stays in the package, while AML checks own source stream
alignment and the copied nuFormer consumer's causal behavior and objective.

## Implementation increments

| Increment | Deliverable | Status |
| --- | --- | --- |
| 1. Data foundation | Pinned AML inventories, local/S3 fetch, checksum verification, download timing, focused tests | Implemented; live timing results stay in ignored run reports |
| 2. AML input preparation | Source schema mapping, identity bindings, anchor validation, train/valid/test selection and temporal rules | Implemented for bounded HI-small selections with complete selected-root histories |
| 3. AML starter workloads | Account involvement with GraphSAGE, nuFormer and combined inputs; PRAGMA masked-value pretraining; nuFormer next-token pretraining | Execution validated for all five bounded real AML cases; complete task correctness and population coverage pending |
| 4. Task correctness | Explicit task contracts and independent target/input/mask/loss/evaluation goldens | Bounded prototype checks pass for all three tasks/five combinations; full-population acceptance remains pending |
| 5. Lifecycle and training | Fresh-process persistence, inference, frozen-state finetuning/resume and two controlled 50–100-step training checks | Validated for the bounded CPU prototype, including combined/PRAGMA learning checks; full-population and GPU execution remain pending |
| 6. Measurements and coverage | Required ingestion/fitting/batching counters, memory, latency, Polars performance comparison and AML coverage sweeps; GPU diagnostics | Selected CPU sweeps and bounded source/anchor matrix pass; edge attributes, larger-than-RAM/full-data scale and measurements pending; complete measurement checklist and Polars comparison now planned below |
| 7. Nightly execution | Run the implemented workloads with suitable storage/hardware, retain complete reports and fail missing required coverage | Pending; tooling and tiny model checks are wired initially |
| 8. AML expansion | Repeat validated workloads across the remaining AML variants | Pending |

Each increment needs meaningful validation and a report of passed, failed and
unavailable checks. Model-only or tiny-fixture success is not full-data evidence.
Performance measurements remain required even before regression thresholds exist.

## First workload contracts

| Workload | Models | Anchor selection |
| --- | --- | --- |
| account-involvement-next24h | GraphSAGE, combined graph/history/aggregates | account-daily v2; graph_as_of_time for execution; original as_of_time for prediction; forecast_split; involvement label |
| account-involvement-next24h | nuFormer | account-daily v1; forecast_split; involvement label |
| masked-value-pretrain | PRAGMA | account-daily v1; split; generated masked-value targets |
| next-token-pretrain | nuFormer | account-daily v1; split; generated next-token targets |

Input/evaluation definitions copied from Arena are not all finalized. Validate
retained-token eligibility and freeze encoding/history configurations when
implementing them. GraphSAGE and combined must use the published v2 counterpart anchors, as
confirmed with the user. They shift only the execution clock to preserve strict
history. Retain original prediction clocks for labels and ages; validate the
combined sequence and aggregate clock translation against the original rules.

## Task correctness

Account involvement now has independent source-derived labels, graph membership
and relationships, sequence grouping, combined count windows, masks, BCE and
evaluation counts across GraphSAGE, nuFormer and combined. Tiny fixtures cover
positive/negative labels, maturity boundaries, corrupted outputs, future/label
perturbations and excluded gradients. All 182 suite checks pass after removing
six superseded supervised loss smoke cases. The bounded real HI-small run passes
all five execution cases and the three involvement checks; its 12 involvement
anchors are all negative. Positive labels are validated by fixtures, not claimed
for that real-data selection. See [scope and limits](docs/aml-prototype.md#account-involvement-checks).

PRAGMA now checks original encoded targets against source values and saved
fitted vocabularies, historical field/event alignment, calendar/age features,
corruption versus supervision, reserved-ID remapping, selected head positions,
smoothed CE and gradients. Missing values are omitted; original UNK values remain
eligible, while unknown replacements are unsupervised under this prototype's
policy. Padding, ignored/invalid targets and zero-target backward passes have
explicit coverage. The 25 new checks passed on both engines before two older
PRAGMA smoke cases were removed. That increment passed 205 suite checks without
failures or skips. The bounded real PRAGMA run independently verified 132/229/187
supervised positions in train/valid/test. These are current prototype choices;
draft task policies and full-population acceptance remain unresolved.

Next-token checks now reconstruct the complete historical AML stream from source
values and fitted encoder state, then compare emitted inputs with all but its
last token and targets with all but its first. Separators, original unknowns,
single-event histories, terminal positions, padding, ignored targets, CE and
gradients are explicit. Suffix perturbations preserve prefix predictions; a
deliberately bidirectional model fails that check. Future/evaluation source fields
and target values cannot affect model inputs/predictions. The declared missing-
field error remains explicit rather than silently dropping tokens.

All 31 new checks passed on both engines before the two old next-token loss
smoke cases were removed. Current AML suite: 352 passed, zero failed/skipped.
All five bounded real HI-small cases pass input and target/loss checks; next-token
evaluation covered 850/2082/1477 eligible positions in train/valid/test.

The [coverage audit](docs/coverage-audit.md) now maps the remaining requirements
to existing owners and explicit gaps. History coverage now passes at
16/128/1,024 events, and batching covers 32/128/512 samples and selected token
budgets. Graph degree/fan-out sweeps now pass selected package and AML cases.
Feature/encoding sweeps now cover selected output widths, cardinalities and text
lengths. Selected empty-history/wide-source checks are implemented.
The bounded source-row/fit-anchor matrix also passes. The PDF is now reconciled:
next correctness/integration work is retained-owner execution, required-coverage
checks, the edge-attribute decision and AML expansion; measurement work stays deferred.
Polars comparison
is deferred to performance work;
the unfinished backend copy was removed and no comparison checks were added.
The three task checks do not complete the full nightly specification.

There are three task definitions across five model/workload combinations.
Columns such as next amount and next recipient do not establish task coverage
merely by existing in the downloaded anchors. Passing a training step establishes
execution, not complete task correctness.

| Task | Models | Independent checks |
| --- | --- | --- |
| Binary account involvement | GraphSAGE, nuFormer, combined | Expected involvement labels under the task definition; null/valid masks; binary cross-entropy; eligible-example evaluation counts |
| Masked-value recovery | PRAGMA | Original values at selected positions; corruption versus supervision masks; reserved/missing/unknown-value eligibility; loss including declared smoothing; eligible-position evaluation counts |
| Next-token prediction | nuFormer | Input/target alignment with exactly one shift; terminal/padding masks; causal attention; cross-entropy and evaluation over eligible target positions |

For each task, record the target definition, allowed inputs, split selection,
loss and evaluation rules before asserting correctness. Independently derive
expected values from source rows or explicit goldens; do not use the production
target-selection or loss helpers to compute their own expected answers.

Acceptance requires:

- Verify emitted targets and masks against the independent expected values.
- Verify permitted historical inputs and strict cutoff boundaries, including
  v2 graph clock translation, combined branch alignment and training-only fitting.
  Target columns and evaluation-only fields must not leak into inputs.
- Compare loss and evaluation numerators/denominators after applying the exact
  eligibility rules. Cover padding, invalid targets and zero-eligible cases.
- Include negative cases where an incorrect target, mask or cutoff is detected.
- Apply these checks to all five combinations and report task correctness
  separately from execution, with explicit unsupported or unresolved outcomes.

## Lifecycle progress

All five AML cases now save their trained model through Kumo's `save_weights`
and their fitted task through `to_artifact().save`. A fresh Python process loads
both, reopens the same DiskGraph index with `ingest="never"`, and infers on
validation/test requests with no targets. Any attempted fitting path fails.
The complete fitted-state fingerprint, input hashes and model outputs must
remain unchanged under the same CPU/thread/batch configuration.

PRAGMA infers on unmasked fields and checks both user representations and its
trained MLM head over all fields; these are not masked-objective losses.
nuFormer next-token inference restores full history without the training shift.
The existing raw-file pipeline tests now exercise this lifecycle, including a
held-out category and a deliberately changed checkpoint. Generic malformed or
incompatible artifact tests remain owned by the package suite.

Validation: 234 suite checks passed, zero failed/skipped. All five real HI-small
cases passed save/load and fresh-process inference on eight selected validation/
test anchors each. See [lifecycle details](docs/aml-prototype.md#saveload-and-inference).

Frozen-input finetuning and resume now pass for all five bounded AML paths.
PRAGMA and next-token nuFormer keep their trained backbone and gain a binary
involvement head; supervised models continue their existing task. All use
forecast training anchors, preserve input state and provenance, and update
backbone parameters. A fresh process restores the finetuned artifact, model,
AdamW moments/steps/hyperparameters and CPU RNG. Its next probe update must match
uninterrupted training, including inputs, loss, model and optimizer state.

The existing pipeline fixture now separates pretraining from forecast training
anchors and includes a category first seen at finetuning; it must remain unknown.
A reset optimizer fails the continuation comparison. Validation remains 234
passed, zero failed/skipped, plus all five real HI-small paths. These are bounded
two-update lifecycle probes, not complete epochs or learning evidence. See
[finetuning and resume](docs/aml-prototype.md#finetuning-and-resume).

## Controlled learning progress

The existing tiny AML fixture now drives two DiskGraph learning checks: combined
for 50 optimizer steps and PRAGMA for 100. They reuse task/loss oracles and the
model construction helper, with an explicit training-only fitting cutoff. Fixed
initialization, sample order and masking isolate learning on one cached batch.
Checks require finite loss/gradients/parameters, first-step gradients and updates
through every named model branch, unchanged inputs/state, answer recovery and
an 80% reduction in loss above the objective's theoretical floor. An unchanged
copy of the initialized model provides the no-training reference.

Combined recovered both labels; PRAGMA recovered 16/17 masked values. These are
fixture overfitting results, not generalization or throughput measurements.
The learning run exposed float32 cancellation in confident BCE predictions;
the independent loss comparison now uses a float32-spacing error bound. A
regression checks float64 agreement and still rejects incorrect losses, even
with a huge ignored logit. No package or copied model implementation changed.

Validation: 237 passed, zero failed/skipped; all five bounded real HI-small paths
still pass. The existing nightly/PR test command includes the new checks and
retains learning JSON reports. CI execution itself remains unverified. See
[controlled learning](docs/aml-prototype.md#controlled-learning-checks).

## Coverage audit

The [requirement matrix](docs/coverage-audit.md) identifies existing package,
conformance, retail and AML ownership. Basic reductions, BPE, masking/layouts,
PK-less tables, clock precision and explicit special IDs should be reused in
place. Missing requested sizes, graph/scale variants, edge-attribute support,
measurement and execution evidence remain explicit. Expected refusals and known
gap replays do not count as successful feature coverage.

The original audit changed only documentation. Subsequent implementation and
validation are recorded below and in the matrix.

## History sweep progress

Sixty DiskGraph cases now cover 16/128/1,024-event AML-shaped histories across
nuFormer involvement, nuFormer next-token, PRAGMA masked values and combined
inputs. Five declared profiles exercise full histories, a count cap, a time
window, two interleaved physical source tables, and a merged/windowed/capped
history. Unequal account lengths exercise padding; rows at/after the cutoff
must be excluded. Canonical full-history workload definitions are unchanged.

Existing independent helpers verify encoded values/order, event/field offsets,
targets and eligibility. Combined graph membership and all twelve count features
still use complete history when the sequence is limited. PRAGMA's reference now
preserves source-qualified field IDs; swapping source provenance fails. Model
input validators accept every size, without executing long-sequence attention.

Validation: 297 passed, zero failed/skipped; all five bounded real HI-small paths
still pass. The existing nightly/PR test command collects all 60 new cases.
No engine/package or copied model changes were required. See
[history sweep details](docs/aml-prototype.md#history-coverage-sweeps).

## Batching sweep progress

Twenty-three cases cover all five AML workload/model combinations at
32/128/512 samples and eight selected token-budget configurations. Materialized
Parquet anchors exercise chunked delivery, five-sample tails, split filtering
and unique sample ownership over repeated roots. Existing independent task
oracles and copied input validators check every batch, including combined graph,
history and count alignment. Token limits check source-derived and actual tensor
costs, with bucketing/shuffling in the larger-budget cases.

Only the test suite and documentation changed; the existing nightly/PR command
collects the new file. All 23 new cases and all 320 suite checks passed, with
zero failures or skips. This establishes selected input correctness, not neural
training at these sizes or a full history × batching cross-product. See
[batching details](docs/aml-prototype.md#batching-coverage-sweeps).

## Graph sweep progress

Thirty-six package cases reuse existing graph conformance oracles for fan-outs
2/32/128, one-to-three-hop paths and uniform/skewed root degrees on both engines.
Twelve AML cases cover GraphSAGE/combined with those fan-outs, even-degree and
hot-account fixtures, original v2 cutoffs and exact source-derived membership.
Combined sequences/counts remain complete when first-hop graph fan-outs are
limited. The shared input oracle accepts a separate graph reference for these
explicit variants; its default full-history expectation is unchanged.

The existing package-regression and AML jobs collect the new modules. No package
runtime, copied model, workflow or published workload declarations changed.
These are input-correctness checks, not large-graph neural training. Edge fields
are absent from the current declaration, so the edge-attribute requirement
remains unresolved. See [graph details](docs/aml-prototype.md#graph-coverage-sweeps).
Validation: 332 AML checks and 36 package graph sweeps passed, zero failed/skipped.
The complete package/conformance/retail nightly and downloaded AML were not rerun.

## Feature and encoding sweep progress

Twenty-four package cases cover 8/32/128 mixed output columns, 16/256/4,096
categories per source field with shared/separate state, and 16/128/1,024-token
texts. They reuse the existing reduction fixture and controlled BPE golden;
checks include exact values/mappings, cutoff exclusion, missing/unknown handling,
piece offsets and unchanged fitted state. Both engines are checked independently
using their supported output paths. See
[feature/encoding details](docs/aml-prototype.md#feature-and-encoding-coverage-sweeps).

All 24 new cases passed, zero failures/skips. Only a package test module and
documentation changed; the existing nightly/PR package-regression command
collects them. The unchanged AML suite was not rerun. This is selected CPU
correctness evidence, not physical wide-row, BPE training-scale or performance
validation.

## Empty-history and wide-source progress

Twenty AML checks cover all-empty/mixed histories across the five workload/model
combinations and both existing engines. Independent input/target checks precede
finite forward/backward checks: supervised labels remain eligible, while empty
pretraining objectives have zero weight/loss/gradients. Two tests exposed a
copied nuFormer bug: an internal padding position leaked into zero-width
next-token logits. A small local `forward_kumo` alignment fix preserves both
cases as ordinary regressions; no installed-package or Arena changes were needed.

Twelve package checks cover 32/128/512 physical data columns on both engines,
with dense and null-heavy projected inputs, all-null columns, category unknowns
and partial batch tails. Existing generic null/duplicate/cutoff owners and
the earlier selected hot-root graph cases remain in place. See
[edge-case details](docs/aml-prototype.md#empty-histories-and-wide-source-rows).
Validation: all 352 AML checks and 12 new wide-source cases passed, zero failed
or skipped. Downloaded AML and the full package/conformance/retail nightly were
not rerun; existing workflow jobs collect both new modules.

## Source and fitting-anchor sweep progress

Eighteen package cases cross 64/1,024/16,384 event rows with 8/64/512 fitting
anchors independently on both engines. Eight roots and five held-out anchors
are additional to those counts. The existing materialized-feature contract and
anchor-table binding are reused, with independent exact count/sum expectations,
training-only mean/population scale and vocabulary membership, repeated-context
weighting, split delivery and partial batch tails.

All 18 cases passed, zero failures/skips. Only a test module and documentation
changed; the existing nightly/PR package-regression job collects it. The unchanged
AML suite was not rerun. See [matrix details](docs/aml-prototype.md#independent-source-and-fitting-anchor-sweeps).
This is bounded correctness evidence, not a larger-than-RAM or performance run.
Next: reconcile the document's remaining requirements and integrated nightly
acceptance gaps before larger-data/measurement work.

Engine defects must remain visible as test failures. Keep the decimal fix for
the prototype, retain its regression checks, and treat engine fixes separately
from test expectations. Do not hide failures through skips, data conversions or
relaxed assertions.

## Access and execution decisions still needed

- Verify the CI role can read the allowed dataset/anchor prefixes. Local SSO
  access does not establish CI access. Do not change IAM during implementation.
- Select storage/hardware for AML data and native execution, accounting for
  source, anchor and index sizes. No scheduled full-data job is activated yet.
- Expand the validated [bounded AML prototype](docs/aml-prototype.md) into the
  remaining correctness, lifecycle and complete-population nightly requirements.

## Proposed benchmarking increment

Planning review: 2026-11-16. Benchmarking is now the requested next workstream;
the earlier measurement deferral is superseded for this plan. No benchmark
implementation or workload execution was performed during this review. Other
open requirements in the [coverage audit](docs/coverage-audit.md) remain open.

Reference: Arena revision `220aa187921ddb1d846985bec8f8edff1d3581c9`, detailed
profiler CLI, measurement design/selection/reporting guides, and counting,
backend-observation, resource and execution-report implementations. Arena's
detailed mode selects 17 probe groups; its 22-metric inventory is a declaration,
not a guarantee that every selected field is available. Its defaults are three
repetitions, one selected-population pass and no additional warmup. Preparation
is reused across repetitions. It does not implement automatic cache eviction.

### Placement and reuse

Add `test/kumo_pytorch/benchmark.py` as the suite entry point, with a small
`benchmarking/` helper directory if needed for measurement, resource collection
and reporting. Keep configuration under `workloads/` and results in ignored
`.runs/`. Use the existing dataset records, source mapping, contracts and model
consumers. Copy and adapt the relevant Arena counting/timing/report logic and
focused tests, recording its source revision and preserving applicable notices.
Do not import Arena or port its complete campaign/distributed/publication system.

The current `run_aml.py` checks source/target/loss correctness inside iteration;
it is not the benchmark loop. Validate the selected configuration before timing,
then execute an independent timed loop without reference checks in the window.
Counting and probe overhead remain inside enclosing execution time and are
reported. Helpers must not alter batch contents, seeds or workload selection.

### Complete metric checklist

Revised after the user's explicit requirement to measure every metric in the
provided excerpt. This checklist includes the continuation at the bottom of the
PDF: rows examined/decoded, memory scope and probe settings. Every item below
is required coverage, not an optional Arena capability.

**Stage measurements.** For every workload, record enclosing elapsed time and
peak host memory separately for ingestion, materialization and batch production.
Also instrument the named operations below explicitly. Preserve parent/child
intervals and operation identity instead of summing nested time. The design
will retain stage-memory attribution for these operations where separable;
fused work must identify its shared scope rather than invent exclusive costs.

| Stage | Every named operation requiring an explicit elapsed-time measurement |
| --- | --- |
| Ingestion | Source reading; source conversion; new DiskGraph index construction; equivalent reference-engine loading; index reopening separately from construction |
| Materialization | Anchor preparation; input fitting; target fitting; vocabulary fitting; numerical-statistics fitting; BPE fitting; task setup; saved-state loading/reuse separately from fresh fitting |
| Native batch production | Enclosing batch production; DiskGraph/Rust sampling; retrieval; aggregate computation |
| Python batch production | Planning; post-processing, including merge, encoding, tokenization, label production, mask production and tensor assembly |
| Reference batch production | Equivalent loading, fitting, planning and batch-production operations for the reference engine, retaining its actual implementation boundaries |

**Counts, sizes, rates, memory and latency.** Counts underlying rates are retained
alongside durations, never inferred from configured batch limits.

| Metric | Required definition |
| --- | --- |
| Source rows | Actual source/input and converted-output row counts, with table identity |
| Source bytes | Source/input and converted-output logical bytes, with scope; not mislabelled as physical device I/O |
| Index size | Measured index bytes after construction and on reuse |
| Rows examined | Actual rows considered by the observed operation, including filtered-out work; count at the relevant source/native boundary |
| Rows decoded | Actual decoded rows at the relevant boundary; distinguish repeated decoding from unique source rows |
| Real example count and samples/s | Emitted example occurrences excluding distributed request padding; primary rate for every workload |
| Input token count and tokens/s | Non-padding input sequence positions: encoded values, BPE pieces and markers, each counted once; exclude targets and parallel field-ID arrays; numerical columns are not tokens |
| Emitted node count and nodes/s | Actual graph node occurrences, preserving output multiplicity |
| Emitted edge count and edges/s | Actual graph edge occurrences, preserving output multiplicity |
| Actual event counts | Emitted event occurrences with representation-specific provenance; no estimate from token count or history cap |
| Actual output counts | Produced batches and output elements by representation; retain partial tails and separate terminal requests |
| Batch sizes | Actual examples and applicable tokens/nodes/edges in each emitted batch, including padding information |
| Output sizes | Actual emitted tensor shapes, element counts and bytes, with a stated convention for shared storage |
| Peak host memory | Separate ingestion/materialization/batch-production peaks including native allocations and workload children; memory scope and peak-measurement method retained |
| Peak allocated GPU memory | Framework allocated-memory high-water counter for each measured window |
| Peak reserved GPU memory | Framework reserved-memory high-water counter for each measured window, separate from allocated memory |
| First-batch latency | Startup-through-first-produced-batch elapsed time, plus separate iterator-construction and first-request timings |
| p50 batch latency | Median of individual produced-request elapsed times, with population/sample count stated |
| p95 batch latency | 95th percentile of produced-request elapsed times, with percentile method and sample count stated |
| Training steps/s | Completed training iterations divided by completed execution-window time; retain step count and optimizer-update count separately |
| Batch-wait time | Explicit host time spent awaiting loader results; terminal requests identified separately |
| Transfer time | Explicit host-to-device transfer host spans and device-event elapsed time with clock/stream scope |
| Sampled GPU utilization | Device-identified utilization observations and summary, with interval, timestamps, gaps and measured steady-state coverage |

**Required experimental separations and records.** These are not additional
throughput counters; they are mandatory conditions for interpreting the metrics.

- First-batch versus measured steady-state execution, with a defined warmup and
  observation window; do not automatically call all later batches steady state.
- First epoch versus later epochs, with separate time/count/memory records.
- Controlled cold caches versus warm caches, declaring OS page-cache, index,
  application and device cache scope and verifying the performed procedure.
- Fresh input/target fitting versus loaded/reused fitted state.
- Repeated independent runs; retain every repetition and observed variation.
- Code revision/source identity and exact dependency versions/pins.
- Hardware identity/configuration, including CPU, memory, storage and GPU.
- Actual batch/output sizes, memory scope and rows examined/decoded as above.
- Probe settings: enabled instrumentation, sampling cadence, clocks, warmup,
  repetition/epoch settings and measured instrumentation overhead.
- Dataset/anchor identities, selected population, splits, model, precision,
  threads, seed and fitted-state identity so each result can be reproduced.

### Instrumentation and acceptance rules

All checklist items must have an execution owner, measurement source, unit,
scope, counting/timing definition and independent validation before this work is
complete. Arena provides reusable implementations, not the acceptance boundary.
Its missing fields become implementation work in the engine branch, including
Python, native/Rust or binding hooks where necessary. In particular, implement
native graph breakdowns, granular Python assembly/planning timings, actual event
provenance and rows-examined/decoded counters if existing APIs do not expose them.

Retain explicit unavailable/error results on failed or intermediate runs, but
**an unavailable required metric cannot pass completeness or close its checklist
item**. Missing GPU hardware, cache control, counters or samples are incomplete
coverage, not successful skips. A genuinely absent operation is not applicable
for that workload only: graph-only inputs have no sequence tokens, for example.
Build the coverage matrix so every required operation is exercised somewhere.
Keep the existing controlled BPE fixture for BPE fitting/tokenization metrics
because AML has no text input; this does not reintroduce the deferred synthetic
dataset. Genuine zero work is recorded as zero with its operation/count evidence.

Rates use completed execution-window time, not summed request durations.
Nested/native/device timings can overlap and must not be added into a fabricated
total. Shared planning is counted once per invocation. Unattributed time is never
called Python time; instrument the actual Python operations.

Implement stage peak-memory collection on the chosen benchmark host. Prefer
resettable OS workload-group high-water accounting where it provides the stated
stage scope; validate reset/read semantics and included processes. Retain
timestamped RSS samples to attribute operations and expose native/child usage.
Arena's sampled RSS maxima alone must not be represented as exact stage peaks;
a host lacking the required peak mechanism leaves that requirement incomplete.
Never sum independent per-process maxima, and document shared-page accounting.
Keep GPU allocator peaks distinct from device-wide utilization. Complete device
work at window boundaries rather than synchronizing every batch.

### Execution protocol and implementation order

1. **Checklist, coverage matrix and collectors.** Map every checklist entry to
   workloads/operations and the measurement implementation. Implement host
   spans, counts, resource/peak collection and report arithmetic with independent
   fixtures, including tails, empty inputs, overlap and failures. Add missing
   Python/Rust hooks and validate native counters and known operation boundaries.
   Missing required readings must fail completeness tests. Reuse Arena helpers
   and focused tests rather than porting the complete profiler.
2. **AML loader benchmark.** Exercise all five existing combinations on real
   HI-small with v2 graph/combined and v1 sequence anchors. Begin with an explicit
   reproducible bounded selection for instrument validation, then complete
   passes over a declared benchmark population using full mapped HI-small
   sources. Preserve required histories. Report exact coverage; the existing
   17-root correctness view is not a full-data benchmark. Stream anchor
   selection/batching rather than loading all anchors as Python objects.
3. **Setup/reuse/cache experiments and selected sweeps.** Separate new mapping/index/
   fit, reopened index with fresh fit, and reopened index with loaded fitted
   state in fresh processes. Propose three repetitions and two separately
   timed epochs for each selected case, restoring model/optimizer/RNG when
   applicable. Repeat setup in separate invocations when measuring its
   variability. Implement and verify cold/warm-cache experiments on the chosen
   host, separately from new/reopened indexes and fresh/reused fitted state.
   Define measured warmup and steady-state coverage. Include equivalent Polars
   reference-engine measurements using matching workload semantics, sources,
   thread/memory budgets and cache conditions, without requiring identical
   batches. Use selected configurations, not a full cross product. Prove
   larger-than-RAM claims from index size and effective memory budget.
4. **Training and GPU comparison.** Add loader-plus-training and prebuilt
   device-resident model execution (forward, loss, backward, optimizer) for the
   same declared batch population, initial state, model and precision. Prepare
   resident inputs outside the model timing window; report capacity failures
   without silently shrinking coverage. Warmup is separate and recorded.
   CPU execution does not satisfy the GPU requirement. Actual GPU measurements,
   including utilization coverage and both allocator peaks, are necessary to
   complete this increment; absent hardware leaves it incomplete.
5. **Nightly reporting and regression policy.** Add a benchmark job to the
   existing workflow using its build/setup conventions; keep the PR trigger.
   Retain config, per-request observations, resource samples, coverage and
   JSON/Markdown summaries, including partial reports on failure. Exclude bulk
   data/index/model payloads from benchmark report artifacts. Validate required
   cases and every applicable checklist metric, and distinguish correctness, collection, capacity,
   memory-regression and timing-regression failures. Collect variation on fixed
   hardware before setting numerical performance gates. Compare only matching
   dataset/configuration/hardware/measurement identities.

Fresh process/index/fitted state does not imply cold OS caches. Cold-cache and
steady-state procedures are required deliverables of this increment, not later
optional experiments. Intermediate uncontrolled runs remain labelled as such
and cannot satisfy the corresponding coverage obligations.
Measure the overhead of the chosen detailed probes on matched runs; do not
subtract an assumed constant from reported execution time.

The user's request for every measurement in the excerpt brings equivalent
reference-engine measurements into this revised benchmarking plan, superseding
the earlier Polars deferral for this workstream. Synthetic/MBD and weekly/on-demand
expansion remain deferred. CI read access and stable CPU/GPU hardware with the
required memory/cache controls must be resolved before claiming nightly
acceptance.

### General interface implementation status

The benchmark runner is dataset/backend/model independent. AML is one intended
adapter, not a condition in the core runner. Arbitrary workloads supply source
preparation, epoch iteration, semantic batch counts and an optional model step.
The public Kumo adapter accepts any compatible contract, dataset, engine, anchor
selection and consumer factory. It does not select one of four model families.

The [first increment](docs/benchmarking.md) implements that interface, explicit
operation spans, complete-epoch timing, actual counts/rates, first/p50/p95 request
latency, training steps, structured/raw reports and per-epoch metric completeness.
Every PDF metric is listed even if its collector is not implemented. Missing
required metrics and experiments produce an incomplete result and nonzero CLI
exit. These helpers and native adapter checks use the existing tooling/model-check
jobs; no separate data/GPU benchmark job has been enabled.

Next: implement the missing source/native/Python probes and resource/peak
collectors, register real dataset adapters, and implement the required fresh-process,
reuse/cache/GPU experiments. Current repetitions create fresh adapters in one
process and cannot satisfy independent-process or cold-cache coverage. This
increment does not complete the full measurement checklist.

Validation for this increment: 29 generic runner/report checks and four native
adapter cases passed, with zero failures/skips in the final focused runs. The
native cases cover a non-AML product table and linear consumer, both loader-only
and training, on InMemory and DiskGraph. The existing 25 acquisition checks also
passed alongside the initial 28 runner checks; one final setup-scope regression
was added and the 29 runner checks rerun. Scoped Ruff and 45 documentation links
passed. No full nightly, real-data throughput or GPU run is claimed.

### Internal instrumentation and peak-memory increment

The general Kumo adapter now installs scoped observers for index creation/reopen,
input/target and scalar-encoder fitting, Python planning/merge/encoding/token
construction/masking/labels/assembly, and available native source profiles.
Temporal-query traversal, decode/select, aggregate timings and decoded-row counts
are retained with their source observation IDs. No native graph-only timing or
rows-examined value is inferred from those fields. Functions are restored on
close/failure; tests compare emitted inputs with uninstrumented execution.

The optional Linux cgroup-v2 collector records separate stage/loader-interval
high-water memory through reset/read on the same `memory.peak` descriptor. It
requires a caller-provisioned dedicated group and makes no membership or limit
changes. Its scope includes memory-controller file-cache/kernel charges. This
implementation now passes real Linux validation of transient parent/child
allocations and independent stage resets in an isolated Linux 7.0.14 container.
At this increment, CI cgroup provisioning remained outstanding. Unsupported collection remains
incomplete, never sampled RSS or zero.
See [measurement details](docs/benchmarking.md#linux-stage-peak-memory).

Prior increment: 61 tooling/runner checks and nine native adapter/observer cases passed,
zero failures/skips. Native evidence includes actual DiskGraph build/reopen and
combined temporal/graph execution, raw-profile ownership, unchanged emitted
inputs, stale-profile rejection and restoration after failure. At that point, next work included
native graph/rows-examined gaps, all real-data adapters and the
fresh-process/cache/reuse experiments. CUDA allocator peaks, deferred transfer
events and UUID-matched NVML utilization collection are implemented. The nightly
now configures a real GPU collector validation on the existing engine RTX PRO
6000 runner group with the locked CUDA environment. Actual GPU CI execution is
pending; fake-device checks do not establish GPU acceptance. No developer-device
pin or Arena runtime dependency is introduced. This remains incomplete benchmark
coverage, not full GPU/performance validation.

Current increment validation: 81 tooling/collector checks and nine native
adapter/observer checks passed, with zero failures or skips. The real Linux
peak-memory validator passed again against the final collector sources. Workflow
YAML, lint and changed-document local links passed. Live GPU results await the
configured nightly CI job.

### AML benchmark bindings and experiment increment

- The general runner now has an AML binding for all five selected combinations,
  retaining v2 graph cutoffs and v1 sequence anchors. It records verified source,
  anchor and prepared-view identities; exact emitted counts; and existing model
  training. The example selects runner-visible CUDA. Its bounded train population
  and full selected-root histories are explicit; this is not full-AML acceptance.
- Fresh-process campaigns retain every worker report/log and compare persisted
  fitting state with fresh fitting. State, contract and fitting-anchor identities
  must agree; loading and enclosing materialization durations remain separate.
  A non-AML product fixture verifies this across real subprocesses.
- Explicit Linux file-page-cache control verifies selected-file residency before
  cold/warm epochs. Application/device caches remain uncontrolled. No global cache
  clearing or claim of isolated causal speedup is made.
- The nightly provisions an isolated private container cgroup for memory/cache
  collector validation. Real Linux validation passed for transient allocations,
  stage resets, exited children and warm/cold/warm file residency. CI execution
  still awaits the workflow; full benchmark workers need separate cgroups.
- Missing graph-native timers and rows-examined counters now emit explicit
  unavailable observations, including fused-runtime graph calls. Temporal-query
  measurements cannot hide these missing graph measurements.

Remaining: native graph timers/rows-examined instrumentation, remaining operation
boundaries, equivalent reference-engine benchmarks, steady-state characterization,
full real-data/GPU execution and scheduling, and memory-cgroup assignment for full
benchmark workers. Benchmark results remain incomplete. No actual GPU/full-AML
performance result is claimed by fixture or collector acceptance.

Validation for this increment: 89 tooling/collector checks and 48 native/model
checks passed, zero failures/skips. The latter include all five AML bindings on
tiny mapped sources, unchanged model regressions, saved-state/index reuse,
independent-process non-AML comparisons and missing native graph evidence.
Real Linux memory and file-cache validators passed. Scoped lint, workflow YAML,
new job shell syntax and 54 local documentation links/fragments passed. No
workflow dispatch, full-data benchmark, upload or commit was performed.

### Native instrumentation and scheduled campaign implementation

Implemented after the preceding increment:

- DiskGraph exposes completed native graph sampling/retrieval profiles, including
  asynchronous and fused-runtime execution. Temporal queries count candidate
  frame rows before selection. Counts retain their declared scope; fused timing
  overlaps are explicit and are never summed into exclusive elapsed time.
- All five AML bindings now cover their applicable native/Python operations,
  fresh index construction and a separate reopen probe. Contract-based
  inapplicability is explicit (for example, no BPE in AML and no graph in a
  sequence-only workload). Missing measurements still fail coverage.
- Generic campaigns can provision a dedicated Linux memory cgroup per worker,
  execute as the runner UID/GID, and clean up descendants on timeout. Actual
  Linux validation passed for unprivileged workers, complete stages, two epochs,
  transient child allocations and cgroup cleanup.
- Explicit post-warmup windows retain counts/rates, latency percentiles and
  variation, plus in-window GPU samples. Campaign acceptance requires applicable
  metric coverage, epochs, verified cold/warm file-page conditions, separate
  interpreter repetitions and matching fresh/reused fitted-state evidence.
- A dataset/model-independent suite command runs declared adapter/config cases.
  The nightly constructs the five AML cases from the workload catalog, downloads
  pinned S3 inputs with Engine's existing test OIDC role and schedules CUDA runs
  on its existing RTX runner group. Each case selects 64 anchors per split with
  full histories, three fresh/reused pairs and two epochs. This remains bounded
  population coverage, not a full-AML population benchmark. CPU PR jobs remain.
- Actions retains the suite/campaign/worker reports, coverage and raw observations;
  artifact filters exclude datasets, indexes and fitted-state payloads.

Validation: 142 focused Python tests passed, with no failures/skips; 58 Engine
and 288 train-table Rust library tests passed. Native bindings rebuilt and the
real adapter checks used that rebuilt extension. Linux worker isolation passed
on kernel 7.0.14-linuxkit. Scoped lint, workflow YAML/shell syntax and documentation
links passed. These checks are implementation evidence, not live GPU results.
The final Linux cache validator also passed with a live read-only mapping:
eviction verified zero resident pages, subsequent reads preserved contents, and
writable mappings were rejected. This handles DiskGraph's retained file mappings
without changing any machine-wide cache setting.

Remaining execution boundary: the candidate branch has not been committed or
published, so GitHub cannot run this code yet. Publish the reviewed branch, run
`kumo-pytorch-t2b.yml` on that branch, and inspect the real GPU collector and all
AML campaign artifacts. Runner sudo/controller/kernel support, the CI role's
AML bucket reads and live cache/GPU measurement coverage must be verified by
that run. No IAM changes or S3 publication are part of this implementation.
Reference-engine performance comparison remains deferred per the earlier user
instruction; native reports do not claim that comparison. Synthetic and weekly
coverage also remain deferred. Benchmark completion is not claimed before CI.

### CI fault-detection run

The user authorized publishing this branch and dispatching the nightly, then
requested removal of the decimal transport fix to verify failure detection.
The fix is now removed while the decimal regression expectations remain. Tests
that called the removed helper now exercise feature retrieval instead, so their
failures concern decimal behavior rather than a missing private method. Local
validation without the fix: nine decimal regressions failed and three unchanged
array cases passed. These are intentional fault-detection results, not passes.

AML correctness/model checks now run independently of lifecycle jobs. Tooling,
package regressions, AML checks and lifecycle checks retain JUnit results alongside
existing model-learning and benchmark reports. Review the complete run jointly;
missing or skipped jobs are coverage gaps, not successes. Earlier passing evidence
in this plan predates withdrawal of the decimal fix.

### Collective CI review

Reviewed on 2026-11-16. The decimal transport fix remains removed. The
[published candidate run](https://github.com/NVIDIA/kumo-engine/actions/runs/43857600971)
executes commit `f27cce6f`; results below distinguish failed execution from missing
measurements. Reports and JUnit files are retained as Actions artifacts, not
committed payloads. The earlier implementation notes above are historical.

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Dataset/tooling checks | 95 passed; no failures/skips | Harness checks, not full-data acceptance |
| Existing package regressions | 1,883 passed, 9 failed, 3 CPU-only skips | All nine failures are decimal transport regressions; the three skipped CUDA tests passed separately on the GPU runner |
| AML correctness/model checks | 338 passed, 7 failed; no skips | Two graph benchmark integration checks and all five raw-file lifecycle checks detect the decimal defect |
| CUDA collector and existing CUDA package tests | Collector passed; 3 package tests passed, no skips | 512 optimizer updates/transfers, real NVML samples and allocated/reserved peaks; collector fixture is not complete workload coverage |
| Existing conformance | 1,271 passed; no failures/skips | Both engines and the required plus daily seed batches; 25m19s |
| Existing lifecycle training and checks | Four training jobs and 22 checks passed; no failed/skipped checks | Saved/reloaded predictions matched exactly for all four runs |
| Linux memory/cache/isolation validation | Passed on the dedicated Linux validation host | Does not establish kernel capabilities of the separate GPU host |
| AML acquisition using Engine OIDC role | All 11 pinned objects downloaded and verified | 375,509,265 bytes in 2.84 seconds; only the approved exact-object GetObject grant was added |
| Real AML CUDA campaigns | 12 failed workers; 18 completed training with incomplete measurements | All 30 independent worker reports retained; see details below |

The candidate's retail test AUROCs were 0.83405/0.83413 for GNN and
0.83483/0.83501 for PRAGMA, against the retained recency baseline of 0.81025.
All four saved/reloaded prediction pairs were independently inspected and
matched exactly across 200,000 rows each. The lifecycle checks passed their
declared reference and cutoff tolerances. These are the existing retail tasks,
not AML generalization results.

The [earlier complete run](https://github.com/NVIDIA/kumo-engine/actions/runs/43854428777)
at `76a27dae`, also without the decimal fix, independently passed all 1,271
conformance tests. The later candidate completed with 11 successful jobs and
three failed jobs (package regressions, AML checks and AML benchmarks).
Conformance passing does not supersede the decimal regression failures.

The candidate's tiny controlled-learning checks completed 50 combined-model
updates (final loss 2.93e-9, accuracy 1.0) and 100 PRAGMA updates (final loss
1.00237, accuracy 16/17). Their frozen controls and eligibility checks passed.
This establishes learning on their fixed fixtures, not real-data model quality.

Real-data profiler review:

- All five cases used verified HI-small inputs, 64 anchors per split with full
  selected-root histories, three fresh/reused pairs and two requested epochs.
  Source inspection counted 5,078,345 transactions and 515,088 account-binding
  rows; prepared indexes contain the declared bounded population.
- GraphSAGE failed on Arrow string values where decimal graph features were
  required, before any training update. Combined failed during fitting on a
  decimal event value. The raw-file correctness tests also exercise inspectable
  decimal reads; sequence training completing on the model path does not
  invalidate those failures. Its reused-state workers then failed because fitting
  had not produced an artifact. These downstream errors are retained, not
  independent evidence of a persistence defect.
- nuFormer involvement, PRAGMA masked-value and nuFormer next-token completed
  all 18 workers and 36 epochs: 576 optimizer updates, 64 emitted examples per
  epoch, with actual token/event/output counts, native temporal counters,
  Python operation timers, latencies, transfer timings and GPU measurements.
- Those sequence workers each met 70/73 metric obligations, counting explicit
  contract-based inapplicability. The three missing obligations were ingestion,
  materialization and batch-production host-memory peaks. Each report preserves
  the reset PermissionError rather than reporting zero or a sampled estimate.
- The GPU host reports Linux 6.8, whose memory.peak counter is read-only.
  Separate stage peaks require a compatible per-descriptor reset interface;
  see [runner requirements](docs/benchmarking.md#linux-stage-peak-memory).
  The successful Linux validation job ran on a different host/kernel.
- All sequence workers verified cold then warm selected-file residency and all
  nine fresh/reused fitting comparisons matched saved-state identities. PRAGMA's
  post-warmup GPU windows had sufficient samples. All 12 later-epoch nuFormer
  windows had only one in-window NVML sample at the configured 0.1-second
  interval, so their steady-window experiment evidence remains incomplete.
- Graph-native successful execution and full graph measurement coverage cannot
  be accepted while the intentionally exposed decimal defect stops those paths.
  No benchmark campaign passed the overall required-evidence gate.

Indicative profiler values from the three fresh runs per sequence case (medians,
with full histories and all observers enabled):

| Workload / model | First / later epoch samples/s | Fresh / reused materialization ms | Later-epoch p95 batch latency ms | Later-epoch allocated GPU peak MiB |
| --- | --- | --- | --- | --- |
| Involvement / nuFormer | 82.64 / 539.54 | 58.09 / 18.27 | 4.52 | 68.35 |
| Masked value / PRAGMA | 49.52 / 114.65 | 45.43 / 13.69 | 5.24 | 67.63 |
| Next token / nuFormer | 82.72 / 565.61 | 52.48 / 17.12 | 4.14 | 87.00 |

Samples/s includes the training execution window. Reused materialization uses
its paired three reused workers. These short, instrumented runs do not establish
steady equilibrium, causal cache speedup or performance regression thresholds.
Missing host peaks and graph failures remain absent from numeric comparisons.
Each artifact worker report retains all 73 metric obligations and raw evidence.

CI integration fixes made while collecting this evidence were limited to the
measurement/test harness: cgroup placement for Linux validation, CUDA UUID
normalization, keeping native availability metadata in the report instead of
changing the package profile keys, the existing CUDA test's field_ids key,
independent evidence jobs, JUnit retention and candidate-run concurrency. The
approved dataset grant is recorded in [the access guide](datasets/ACCESS.md).
No Arena runtime dependency, public dataset copy, IAM role policy change or
production decimal workaround was introduced.

Next: provision a GPU runner with resettable memory.peak, extend the nuFormer
measurement windows or choose and validate an adequate sampling interval, then
rerun the campaigns. Keep the decimal failures visible until an explicit package
fix is accepted. Full-population/larger-than-RAM measurements and performance
threshold calibration remain outstanding. Synthetic and weekly remain deferred.
Reference-engine performance comparison has no execution evidence; native-only
inapplicability declarations do not establish that comparison. This review does not declare all PDF
requirements complete.

### GPU sampling window follow-up

The nightly benchmark selection increases from 64 to 512 anchors per split
for all five AML cases, retaining full histories and batch size four. This
extends each epoch from 16 to 128 real batches, including 126 after warmup.
The general runner, 0.1-second NVML polling interval and two-sample coverage
requirement remain unchanged. No delay or synthetic samples are added.
The earlier 64-anchor results remain historical evidence. Live validation of
the larger selection is pending; the host-kernel peak limitation and exposed
decimal defect remain separate failures.
